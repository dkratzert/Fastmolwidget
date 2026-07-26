"""Export a parsed structure (CIF or SHELX) as the fractional-coordinate
JSON contract consumed by the JavaScript renderer (see ``js/README.md``).

Structure **parsing** stays in Python (via ``gemmi``/``shelxfile``, same as
:mod:`fastmolwidget.loader`); growing the asymmetric unit to whole molecules
and packing unit cells now happen in the browser (``js/sdm.js``), so this
module only needs to ship the asymmetric unit in fractional coordinates plus
the symmetry operations and cell.

Usage::

    from fastmolwidget.web_export import export_cif, export_shelx
    import json

    data = export_cif('structure.cif')
    print(json.dumps(data))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastmolwidget.tools import to_float

__all__ = ['export_cif', 'export_shelx', 'export_to_json']


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
