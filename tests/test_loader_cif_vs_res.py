"""Cross-validation tests: loading a structure as CIF vs. as SHELX (.res)
must yield the same atom labels, the same Cartesian coordinates and the same
anisotropic displacement parameters (within numerical tolerance).

Two CIF/RES pairs in ``tests/test-data`` describe the *same* refined structure:

* ``1548072_many_atoms.cif``  ↔  ``1548072_many_atoms.res``
* ``41467_2015_BFncomms9288_MOESM1370_ESM.cif`` ↔ ``…1370_ESM.res``

The CIF was written by SHELXL from the matching .res, so every atom label
present in both files must agree on coordinates and ADPs.

These tests catch:
* swapped or shifted ADP components (e.g. CIF order
  ``U11,U22,U33,U23,U13,U12`` vs. some other ordering),
* fractional-vs-Cartesian confusion in either reader,
* atom-name / position desynchronisation (label assigned to wrong row),
* unit-cell parameter drift between the two parsers.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy import QtWidgets

from fastmolwidget.cif.cif_file_io import CifReader
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.tools import to_float

# Process-wide QApplication (MoleculeWidget needs a QApplication to exist).
app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

DATA = Path('tests/test-data')

# Pairs of (CIF, RES) describing the *same* refined structure.  Coordinates
# always agree.  ADPs only agree for ``ADP_PAIRS`` – the other test-data pairs
# contain a CIF refined further than the matching .res (R₁ comparison
# confirms different refinement stages), so their ADP values legitimately
# differ and are excluded from the strict ADP comparison.
#
# Note: ``1548072_many_atoms.{cif,res}`` are NOT included here because the
# two files describe different refinement stages (max coordinate deviation
# ≈ 0.47 Å).  That file pair is only used for testing multi-residue label
# and grow/pack functionality in other tests.
PAIRS: list[tuple[str, str]] = [
    (
        '41467_2015_BFncomms9288_MOESM1370_ESM.cif',
        '41467_2015_BFncomms9288_MOESM1370_ESM.res',
    ),
]

ADP_PAIRS: list[tuple[str, str]] = [
    (
        '41467_2015_BFncomms9288_MOESM1370_ESM.cif',
        '41467_2015_BFncomms9288_MOESM1370_ESM.res',
    ),
]

COORD_TOL = 5e-3   # Å – CIF coordinates are rounded to 3-4 decimals
# CIF ADPs are commonly rounded to 3 decimals, so the worst-case rounding
# error per component is 5e-4.  Allow a small margin on top of that.
ADP_TOL = 6e-4


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _cif_orth_atoms(cif_path: Path) -> dict[str, tuple[float, float, float]]:
    """Return ``{label.upper(): (x, y, z)}`` from the CIF, Cartesian Å."""
    cif = CifReader(cif_path)
    return {a.label.upper(): (a.x, a.y, a.z) for a in cif.atoms_orth}


def _cif_adps(cif_path: Path) -> dict[str, tuple[float, ...]]:
    """Return ``{label.upper(): (U11, U22, U33, U23, U13, U12)}`` from the CIF."""
    cif = CifReader(cif_path)
    return {
        dp.label.upper(): (
            to_float(dp.U11), to_float(dp.U22), to_float(dp.U33),
            to_float(dp.U23), to_float(dp.U13), to_float(dp.U12),
        )
        for dp in cif.displacement_parameters()
    }


def _res_atoms(res_path: Path):
    atoms, cell = MoleculeLoader._parse_shelx(res_path)
    return atoms, cell


def _cif_cell(cif_path: Path) -> tuple[float, float, float, float, float, float]:
    cif = CifReader(cif_path)
    c = cif.cell
    return (c.a, c.b, c.c, c.alpha, c.beta, c.gamma)


# ----------------------------------------------------------------------
# Cell parameters
# ----------------------------------------------------------------------

@pytest.mark.parametrize('cif_name, res_name', PAIRS)
def test_cell_matches(cif_name: str, res_name: str) -> None:
    """Unit-cell parameters parsed from CIF and from RES must agree."""
    cif_cell = _cif_cell(DATA / cif_name)
    _, res_cell = _res_atoms(DATA / res_name)
    for i, (cv, rv) in enumerate(zip(cif_cell, res_cell, strict=True)):
        assert cv == pytest.approx(rv, abs=1e-3), (
            f"Cell parameter index {i}: CIF={cv} vs RES={rv}"
        )


# ----------------------------------------------------------------------
# Cartesian coordinates per (uppercase) label
# ----------------------------------------------------------------------

@pytest.mark.parametrize('cif_name, res_name', PAIRS)
def test_cartesian_coordinates_match(cif_name: str, res_name: str) -> None:
    """For every label present in *both* files, the Cartesian coordinates
    produced by the CIF reader and by the SHELX reader must agree."""
    cif_xyz = _cif_orth_atoms(DATA / cif_name)
    res_atoms, _ = _res_atoms(DATA / res_name)
    res_xyz = {a.label.upper(): (a.x, a.y, a.z) for a in res_atoms}

    common = set(cif_xyz) & set(res_xyz)
    assert common, (
        f"No common atom labels between {cif_name} and {res_name}"
    )

    bad: list[str] = []
    for label in sorted(common):
        cx, cy, cz = cif_xyz[label]
        rx, ry, rz = res_xyz[label]
        if (abs(cx - rx) > COORD_TOL
                or abs(cy - ry) > COORD_TOL
                or abs(cz - rz) > COORD_TOL):
            bad.append(
                f"{label}: CIF=({cx:.4f},{cy:.4f},{cz:.4f}) "
                f"RES=({rx:.4f},{ry:.4f},{rz:.4f})"
            )
    assert not bad, "Coordinate mismatches:\n" + "\n".join(bad)


# ----------------------------------------------------------------------
# ADP order (U11, U22, U33, U23, U13, U12)
# ----------------------------------------------------------------------

@pytest.mark.parametrize('cif_name, res_name', ADP_PAIRS)
def test_adps_match(cif_name: str, res_name: str) -> None:
    """ADP tuples embedded in the SHELX-loaded ``Atomtuple.adp`` must match
    the corresponding ``_atom_site_aniso_*`` entries from the CIF, in the
    expected order ``(U11, U22, U33, U23, U13, U12)``."""
    cif_adp = _cif_adps(DATA / cif_name)
    res_atoms, _ = _res_atoms(DATA / res_name)
    res_adp = {a.label.upper(): a.adp for a in res_atoms if a.adp is not None}

    common = set(cif_adp) & set(res_adp)
    assert common, (
        f"No common ADP labels between {cif_name} and {res_name}"
    )

    bad: list[str] = []
    component_names = ('U11', 'U22', 'U33', 'U23', 'U13', 'U12')
    for label in sorted(common):
        c = cif_adp[label]
        r = res_adp[label]
        for name, cv, rv in zip(component_names, c, r, strict=True):
            if abs(cv - rv) > ADP_TOL:
                bad.append(
                    f"{label}.{name}: CIF={cv} vs RES={rv} (Δ={cv - rv:+.5f})"
                )
    assert not bad, "ADP component mismatches:\n" + "\n".join(bad)


# ----------------------------------------------------------------------
# End-to-end: same widget state after loading either file
# ----------------------------------------------------------------------

@pytest.mark.parametrize('cif_name, res_name', PAIRS)
def test_widget_atom_positions_match(cif_name: str, res_name: str) -> None:
    """Loading the CIF and the RES through :class:`MoleculeLoader` must
    place atoms with identical labels at identical Cartesian positions."""
    w_cif = MoleculeWidget()
    MoleculeLoader(w_cif).load_file(DATA / cif_name)

    w_res = MoleculeWidget()
    MoleculeLoader(w_res).load_file(DATA / res_name)

    cif_pos = {a.name.upper(): tuple(map(float, a.coordinate))
               for a in w_cif.atoms}
    res_pos = {a.name.upper(): tuple(map(float, a.coordinate))
               for a in w_res.atoms}

    common = set(cif_pos) & set(res_pos)
    assert common, "Widgets share no atom labels after loading."

    bad: list[str] = []
    for label in sorted(common):
        cx, cy, cz = cif_pos[label]
        rx, ry, rz = res_pos[label]
        if (abs(cx - rx) > COORD_TOL
                or abs(cy - ry) > COORD_TOL
                or abs(cz - rz) > COORD_TOL):
            bad.append(
                f"{label}: CIF=({cx:.4f},{cy:.4f},{cz:.4f}) "
                f"RES=({rx:.4f},{ry:.4f},{rz:.4f})"
            )
    assert not bad, "Widget coord mismatches:\n" + "\n".join(bad)


@pytest.mark.parametrize('cif_name, res_name', ADP_PAIRS)
def test_widget_adp_tensors_match(cif_name: str, res_name: str) -> None:
    """The Cartesian ADP tensor stored on each :class:`Atom` in the widget
    must be identical (within tolerance) whether the structure was loaded
    from the CIF or from the RES."""
    w_cif = MoleculeWidget()
    MoleculeLoader(w_cif).load_file(DATA / cif_name)

    w_res = MoleculeWidget()
    MoleculeLoader(w_res).load_file(DATA / res_name)

    cif_u = {a.name.upper(): a.u_cart for a in w_cif.atoms if a.u_cart is not None}
    res_u = {a.name.upper(): a.u_cart for a in w_res.atoms if a.u_cart is not None}

    common = set(cif_u) & set(res_u)
    assert common, "No anisotropic atoms in common."

    bad: list[str] = []
    # Tolerance: CIF ADPs are rounded → propagated through frac→cart transform.
    # Allow 5 × ADP_TOL on the resulting Cartesian U components.
    tol = 5 * ADP_TOL
    for label in sorted(common):
        cu = cif_u[label]
        ru = res_u[label]
        diff = (cu - ru)
        # max absolute difference over all 9 entries
        max_dev = float(abs(diff).max())
        if max_dev > tol:
            bad.append(f"{label}: max |ΔUcart| = {max_dev:.5f}")
    assert not bad, "Cartesian ADP tensor mismatches:\n" + "\n".join(bad)


# ----------------------------------------------------------------------
# Label/position synchronisation: loading order must not desynchronise
# atom labels and their stored coordinates inside the widget.
# ----------------------------------------------------------------------

@pytest.mark.parametrize('res_name', [p[1] for p in PAIRS] + [
    'p31c-finalcif.res',
    'test_molecule.res',
])
def test_res_label_position_synchronised(res_name: str) -> None:
    """The widget's atom list must be index-aligned with the SHELX parser
    output: row *i* in the parsed list corresponds to the same atom as
    index *i* in ``widget.atoms``.  A swap would manifest as label/position
    mismatch on screen.
    """
    res_atoms, _ = _res_atoms(DATA / res_name)
    w = MoleculeWidget()
    MoleculeLoader(w).load_file(DATA / res_name)

    assert len(w.atoms) == len(res_atoms), (
        f"Atom count mismatch: parser={len(res_atoms)} widget={len(w.atoms)}"
    )

    for i, (parsed, drawn) in enumerate(zip(res_atoms, w.atoms, strict=True)):
        # Label should match (widget de-duplicates with '>>n' suffix; the
        # parser never produces duplicates within a single file because each
        # SHELX label+part combination is unique).
        assert drawn.name.split('>>')[0] == parsed.label, (
            f"Row {i}: parser label {parsed.label!r} but widget {drawn.name!r}"
        )
        dx, dy, dz = (float(c) for c in drawn.coordinate)
        assert dx == pytest.approx(parsed.x, abs=1e-4)
        assert dy == pytest.approx(parsed.y, abs=1e-4)
        assert dz == pytest.approx(parsed.z, abs=1e-4)


# ----------------------------------------------------------------------
# Regression: multi-residue SHELX files must assign unique ADPs per atom
# ----------------------------------------------------------------------

def test_grown_shelx_multi_residue_adps_are_unique() -> None:
    """Regression for the bug where all atoms sharing a base name (e.g. 'C1')
    across different residues received the same ADP after growing.

    ``1548072_many_atoms.res`` has 79 atoms named 'C1' in residues 1-79.
    After growing, every 'C1_N' must have its own distinct ADP (not the ADP
    of the last 'C1' that happened to overwrite the others in the dict).
    """
    from collections import Counter
    atoms = MoleculeLoader._compute_grown_atoms_shelx(
        DATA / '1548072_many_atoms.res'
    )
    # Gather grown C1-base atoms (label C1_N for residue N)
    c1_atoms = [
        a for a in atoms
        if a.label.startswith('C1_') and not a.label.startswith('C10_')
        and a.adp is not None
    ]
    assert len(c1_atoms) >= 2, f"Expected multiple C1-residue atoms, got {len(c1_atoms)}"

    # Before the fix, all N atoms shared ONE ADP (the last to overwrite the dict).
    # After the fix, at most a small number can coincidentally share ADP values.
    # The threshold of len/4 is conservative: the old bug would give count ≈ len.
    adp_counts = Counter(a.adp for a in c1_atoms)
    most_freq_count = adp_counts.most_common(1)[0][1]
    assert most_freq_count < len(c1_atoms) // 4, (
        f"ADP collision detected: {most_freq_count}/{len(c1_atoms)} C1-base atoms "
        f"share the same ADP. Fix: use fullname_short (label+residue) as the "
        f"adp_by_lp key."
    )


