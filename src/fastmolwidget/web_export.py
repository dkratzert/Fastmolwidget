"""Export structures and optional density for the web renderer."""

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

#: Web density grid spacing in Å. Coarser than desktop to keep payloads small.
DEFAULT_WEB_GRID_SPACING: float = 0.20

#: Extra mask padding in grid steps so contour interpolation stays intact.
_MASK_SLACK_STEPS: int = 2


def export_cif(path: str | Path) -> dict[str, Any]:
    """Parse a CIF file into the web JSON contract."""
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
    """Parse a SHELX ``.res``/``.ins`` file into the web JSON contract."""
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
    """Return Cartesian atom positions for the chosen coverage mode."""
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
    """Return grid points within *radius* of any atom.

    Indices wrap because the map is one periodic unit cell.
    """
    dims = np.asarray(shape)
    mask = np.zeros(tuple(dims), dtype=bool)
    if len(atoms) == 0:
        return mask

    inverse = np.linalg.inv(orth)
    # Convert the radius to per-axis grid steps for oblique cells too.
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
    """Compute a residual-density payload for the web renderer.

    Exports one whole unit cell so the browser can wrap it around grown or
    packed views. Density outside the display margin is zeroed before
    compression to shrink the payload.
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

    # Full-range quantization: no clipping.
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
    """Export a CIF or SHELX file to JSON."""
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
