"""Export a parsed structure (CIF or SHELX) as the fractional-coordinate
JSON contract consumed by the JavaScript renderer (see ``js/README.md``).

Structure **parsing** stays in Python (via ``gemmi``/``shelxfile``, same as
:mod:`fastmolwidget.loader`); growing the asymmetric unit to whole molecules
and packing unit cells now happen in the browser (``js/sdm.js``), so this
module only needs to ship the asymmetric unit in fractional coordinates plus
the symmetry operations and cell.

:func:`export_density` additionally ships a residual (Fo−Fc) map, which the
browser contours itself (``js/density.js``).  It is deliberately **opt-in**: a
page that does not ask for density carries no map payload at all.

Usage::

    from fastmolwidget.web_export import export_cif, export_shelx
    import json

    data = export_cif('structure.cif')
    print(json.dumps(data))
"""

from __future__ import annotations

import base64
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from fastmolwidget.tools import to_float

__all__ = [
    'DEFAULT_WEB_GRID_SPACING',
    'export_cif',
    'export_density',
    'export_shelx',
    'export_to_json',
]

#: Grid spacing in Å used for web exports.  Coarser than the desktop default
#: (:data:`fastmolwidget.density.DEFAULT_GRID_SPACING`, 0.15 Å) because the map
#: has to travel inside the HTML page: 0.20 Å is about a tenth of the payload
#: and still resolves the features a report-sized view can show.
DEFAULT_WEB_GRID_SPACING: float = 0.20

#: Extra padding, in grid steps, added to the masking radius.  Grid points just
#: outside the displayed envelope still take part in the interpolation of the
#: contour that runs *inside* it, so they must survive the mask.
_MASK_SLACK_STEPS: int = 2


def export_cif(path: str | Path) -> dict[str, Any]:
    """Parse a CIF file and return the fractional-coordinate JSON contract."""
    from fastmolwidget.cif.cif_file_io import CifReader

    cif = CifReader(Path(path))
    adp_by_label: dict[str, tuple] = {
        dp.label: (
            to_float(dp.U11), to_float(dp.U22), to_float(dp.U33),
            to_float(dp.U23), to_float(dp.U13), to_float(dp.U12),
        )
        for dp in cif.displacement_parameters()
    }
    atoms = [
        {
            'label': label,
            'type': type_,
            'x': x, 'y': y, 'z': z,
            'part': part or 0,
            'adp': adp_by_label.get(label),
        }
        for label, type_, x, y, z, part, _occ, _u_iso in cif.atoms_fract
    ]
    return {
        'cell': list(cif.cell[:6]),
        'centric': bool(cif.is_centrosymm),
        'symmops': list(cif.symmops),
        'atoms': atoms,
    }


def export_shelx(path: str | Path) -> dict[str, Any]:
    """Parse a SHELX ``.res``/``.ins`` file and return the fractional-coordinate
    JSON contract."""
    from shelxfile import Shelxfile

    shx = Shelxfile()
    shx.read_file(Path(path))

    cell_params = [
        shx.cell.a, shx.cell.b, shx.cell.c,
        shx.cell.alpha, shx.cell.beta, shx.cell.gamma,
    ]
    symmops = [s.to_shelxl() for s in shx.symmcards]
    centric = bool(shx.latt.centric) if shx.latt else False

    adp_by_lp: dict[tuple, tuple] = {}
    atoms = []
    for at in shx.atoms:
        if at.qpeak:
            continue
        x, y, z = at.frac_coords
        label = at.fullname_short  # unique across residues (e.g. "C1_1")
        part = at.part.n
        if not at.is_isotropic:
            u11, u22, u33, u23, u13, u12 = at.uvals
            adp_by_lp[(label, part)] = (u11, u22, u33, u23, u13, u12)
        atoms.append({
            'label': label,
            'type': at.element,
            'x': x, 'y': y, 'z': z,
            'part': part,
            'adp': adp_by_lp.get((label, part)),
        })

    return {
        'cell': cell_params,
        'centric': centric,
        'symmops': symmops,
        'atoms': atoms,
    }


# ---------------------------------------------------------------------------
# Residual (Fo-Fc) density
# ---------------------------------------------------------------------------

def _coverage_atoms(data: dict[str, Any], coverage: str) -> np.ndarray:
    """Cartesian positions of every atom the browser might display.

    The map is masked around these, so whatever the viewer grows or packs to
    still has density around it.

    :param data: The structure dict from :func:`export_cif` / :func:`export_shelx`.
    :param coverage: ``'asu'``, ``'grow'`` or ``'cell'``.
    :returns: An ``(N, 3)`` array of Cartesian coordinates in Å.
    """
    from fastmolwidget.sdm import SDM

    cell = tuple(data['cell'])
    fract = [
        [atom['label'], atom['type'], atom['x'], atom['y'], atom['z'],
         atom['part'], 1.0, 0.05]
        for atom in data['atoms']
    ]
    symmops = list(data['symmops'])
    centric = bool(data['centric'])

    if coverage == 'asu':
        from fastmolwidget.dsrmath import frac_to_cart

        return np.array(
            [frac_to_cart([a['x'], a['y'], a['z']], cell) for a in data['atoms']],
            dtype=float,
        ).reshape(-1, 3)
    if coverage == 'grow':
        sdm = SDM([list(a) for a in fract], symmops, cell, centric=centric)
        atoms = sdm.packer(sdm, sdm.calc_sdm())
    elif coverage == 'cell':
        sdm = SDM([list(a) for a in fract], symmops, cell, centric=centric)
        atoms = sdm.pack_unit_cell()
    else:
        raise ValueError(
            f"coverage must be 'asu', 'grow' or 'cell', not {coverage!r}")
    return np.array([[a.x, a.y, a.z] for a in atoms], dtype=float).reshape(-1, 3)


def _envelope_mask(
    shape: tuple[int, int, int],
    orth: np.ndarray,
    atoms: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Boolean grid marking the points within *radius* of any atom.

    Only the neighbourhood of each atom is examined — a few thousand points per
    atom — instead of testing the whole grid against every atom, which for a
    packed cell would be hundreds of millions of distances.

    Grid indices wrap, because the map covers one unit cell periodically and
    the atoms may lie outside it.

    :param shape: Grid dimensions of one unit cell.
    :param orth: ``3x3`` fractional-to-Cartesian matrix.
    :param atoms: ``(N, 3)`` Cartesian atom positions.
    :param radius: Masking radius in Å.
    """
    dims = np.asarray(shape)
    mask = np.zeros(tuple(dims), dtype=bool)
    if len(atoms) == 0:
        return mask

    inverse = np.linalg.inv(orth)
    # Width of the radius along each axis in fractional units, then in grid
    # steps.  The row norms of the inverse keep this correct for oblique cells.
    steps = np.ceil(radius * np.linalg.norm(inverse, axis=1) * dims).astype(int)
    ranges = [np.arange(-s, s + 1) for s in steps]
    block = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1)
    limit = radius * radius

    for position in atoms:
        centre = np.rint(position @ inverse.T * dims).astype(int)
        indices = block + centre
        cartesian = (indices / dims) @ orth.T
        close = ((cartesian - position) ** 2).sum(axis=-1) <= limit
        if not close.any():
            continue
        wrapped = np.mod(indices[close], dims)
        mask[wrapped[:, 0], wrapped[:, 1], wrapped[:, 2]] = True
    return mask


def export_density(
    model: str | Path,
    hkl: str | Path | None = None,
    *,
    level: float | None = None,
    grid_spacing: float = DEFAULT_WEB_GRID_SPACING,
    margin: float = 1.5,
    coverage: str = 'asu',
    compress: bool = True,
) -> dict[str, Any]:
    """Compute a residual (Fo−Fc) map and pack it for the JavaScript renderer.

    The map is shipped as the **whole unit cell**, quantised to one byte per
    grid point, so the browser can contour it at any level and wrap it around
    grown or packed molecules exactly as
    :meth:`~fastmolwidget.density.ResidualDensityMap._region` does.

    Density far from any atom is never drawn (the isosurface is clipped to
    *margin* around the displayed atoms), so it is zeroed here before
    compression.  That costs nothing visually and shrinks the payload three- to
    fourfold, which matters because it travels inside the HTML page.

    :param model: The refined model — a CIF, or a SHELX ``.res``/``.ins``.
    :param hkl: Reflections.  ``None`` looks them up automatically.
    :param level: Contour level in e/Å³ the viewer should start at.  ``None``
        uses :data:`~fastmolwidget.density.DEFAULT_SIGMA` times the map's RMS.
    :param grid_spacing: FFT grid spacing in Å.  The default is coarser than
        the desktop one to keep the page small.
    :param margin: Radius in Å around the covered atoms that is kept.
    :param coverage: Which atoms to keep density around — ``'asu'`` (the
        asymmetric unit, matching the viewer's default view), ``'grow'`` (whole
        molecules) or ``'cell'`` (a packed unit cell).  Use the widest mode the
        page's controls allow, since the browser cannot recover what was
        masked away.
    :param compress: gzip the payload before base64-encoding it.  The browser
        inflates it with ``DecompressionStream``.
    :returns: The ``density`` object of the JSON contract.
    :raises ValueError: If *coverage* is not one of the three modes.
    """
    from fastmolwidget.density import calculate_residual_density

    model = Path(model)
    density_map = calculate_residual_density(model, hkl, grid_spacing=grid_spacing)

    if model.suffix.lower() == '.cif':
        structure = export_cif(model)
    else:
        structure = export_shelx(model)
    atoms = _coverage_atoms(structure, coverage)

    values = np.array(density_map.array, dtype=np.float64)
    orth = density_map.orth_matrix
    spacings = np.linalg.norm(orth, axis=0) / np.asarray(values.shape)
    mask = _envelope_mask(
        values.shape, orth, atoms,
        margin + _MASK_SLACK_STEPS * float(spacings.max()),
    )
    values[~mask] = 0.0

    # Full-range quantisation - no clipping, so even the strongest peak keeps
    # its shape when the level is raised.
    scale = float(np.abs(values).max())
    if scale <= 0.0:
        raise ValueError(f'The residual density of {model} is empty')
    quantised = np.rint(values / scale * 127.0).astype(np.int8)

    payload = quantised.tobytes()
    if compress:
        payload = gzip.compress(payload, 9)

    cell = density_map.cell
    return {
        'mode': 'grid',
        'size': [int(n) for n in values.shape],
        'cell': [float(v) for v in cell],
        'rms': float(density_map.rms),
        'max': float(density_map.max),
        'min': float(density_map.min),
        'level': float(density_map.sigma_level() if level is None else abs(level)),
        'scale': scale / 127.0,
        'margin': float(margin),
        'coverage': coverage,
        'encoding': 'gzip+base64' if compress else 'base64',
        'data': base64.b64encode(payload).decode('ascii'),
    }


def export_to_json(path: str | Path, out_path: str | Path | None = None) -> str:
    """Export a CIF or SHELX file to the JSON contract, returning the JSON
    string (and optionally writing it to *out_path*)."""
    path = Path(path)
    if path.suffix.lower() == '.cif':
        data = export_cif(path)
    elif path.suffix.lower() in ('.res', '.ins'):
        data = export_shelx(path)
    else:
        raise ValueError(f'Unsupported file type: {path.suffix}')
    text = json.dumps(data)
    if out_path is not None:
        Path(out_path).write_text(text)
    return text


if __name__ == '__main__':
    import sys

    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix('.json')
    export_to_json(src, dst)
    print(f'Wrote {dst}')
