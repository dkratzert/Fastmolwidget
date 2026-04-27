"""Tests for the optional sdm_cpp C++ extension module.

Skipped automatically when the module has not been compiled.

Run after building:
    pip install -e . --no-build-isolation
    uv run pytest tests/test_sdm_cpp.py -v
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

sdm_cpp = pytest.importorskip("sdm_cpp", reason="sdm_cpp C++ extension not built — skipping")

from fastmolwidget.cif.cif_file_io import CifReader
from fastmolwidget.atoms import get_radius_from_element
import fastmolwidget.sdm as sdm_module
from fastmolwidget.sdm import SDM

DATA = Path("tests/test-data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_sdm_inputs(cif_path: Path) -> tuple:
    """Return (fract_atoms, symmops, cell, centric) from a CIF file."""
    cif = CifReader(cif_path)
    fract_atoms = list(cif.atoms_fract)
    return fract_atoms, cif.symmops, cif.cell, cif.is_centrosymm


def _run_python_path(fract_atoms, symmops, cell, centric) -> list:
    """Run SDM entirely via the Python fallback (HAS_CPP forced to False)."""
    original = sdm_module.HAS_CPP
    sdm_module.HAS_CPP = False
    try:
        atoms_copy = [list(a) for a in fract_atoms]
        s = SDM(atoms_copy, symmops, cell, centric=centric)
        s.calc_sdm()
        return s.sdm_list
    finally:
        sdm_module.HAS_CPP = original


def _run_cpp_path(fract_atoms, symmops, cell, centric) -> list:
    """Run SDM via the C++ path (HAS_CPP forced to True)."""
    original = sdm_module.HAS_CPP
    sdm_module.HAS_CPP = True
    try:
        atoms_copy = [list(a) for a in fract_atoms]
        s = SDM(atoms_copy, symmops, cell, centric=centric)
        s.calc_sdm()
        return s.sdm_list
    finally:
        sdm_module.HAS_CPP = original


# ---------------------------------------------------------------------------
# Basic module checks
# ---------------------------------------------------------------------------

def test_module_has_calc_sdm_cpp():
    assert hasattr(sdm_cpp, "calc_sdm_cpp")
    assert callable(sdm_cpp.calc_sdm_cpp)


def test_module_exposes_has_openmp():
    assert hasattr(sdm_cpp, "has_openmp")
    assert isinstance(sdm_cpp.has_openmp, bool)


# ---------------------------------------------------------------------------
# Direct function call — minimal geometry (identity symmetry only)
# ---------------------------------------------------------------------------

def test_calc_sdm_cpp_returns_list():
    # 2 atoms, 1 symmetry op (identity), simple cubic cell (a=10)
    coords   = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2]]
    symm_m   = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    symm_t   = [[0.0, 0.0, 0.0]]
    a = 10.0
    aga = bbe = cal = 0.0    # orthorhombic
    asq = bsq = csq = a * a

    radii = [0.77, 0.66]  # C, O
    is_h  = [False, False]
    parts = [0.0, 0.0]

    result = sdm_cpp.calc_sdm_cpp(
        coords, symm_m, symm_t,
        aga, bbe, cal, asq, bsq, csq,
        radii, is_h, parts,
    )
    assert isinstance(result, list)
    # At least the (0→1) pair should be found
    assert len(result) >= 1


def test_calc_sdm_cpp_tuple_fields():
    coords   = [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]
    symm_m   = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    symm_t   = [[0.0, 0.0, 0.0]]
    a = 5.0
    asq = bsq = csq = a * a
    aga = bbe = cal = 0.0

    result = sdm_cpp.calc_sdm_cpp(
        coords, symm_m, symm_t,
        aga, bbe, cal, asq, bsq, csq,
        [0.77, 0.77], [False, False], [0.0, 0.0],
    )
    for tup in result:
        i, j, best_n, mind, dddd, covalent = tup
        assert isinstance(i, int)
        assert isinstance(j, int)
        assert isinstance(best_n, int)
        assert isinstance(mind, float)
        assert isinstance(dddd, float)
        assert isinstance(covalent, bool)
        assert mind > 0.0
        assert best_n >= 0


# ---------------------------------------------------------------------------
# Agreement between C++ and Python paths on real CIF data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cif_name", [
    "1979688_small.cif",
    "p21c.cif",
    "p31c.cif",
])
def test_cpp_matches_python_sdm_list_length(cif_name):
    """C++ and Python paths must produce the same number of SDM items."""
    fract_atoms, symmops, cell, centric = _build_sdm_inputs(DATA / cif_name)

    py_list  = _run_python_path([list(a) for a in fract_atoms], symmops, cell, centric)
    cpp_list = _run_cpp_path([list(a) for a in fract_atoms], symmops, cell, centric)

    assert len(cpp_list) == len(py_list), (
        f"{cif_name}: C++ produced {len(cpp_list)} items, Python produced {len(py_list)}"
    )


@pytest.mark.parametrize("cif_name", [
    "1979688_small.cif",
    "p21c.cif",
])
def test_cpp_matches_python_sorted_distances(cif_name):
    """Sorted distance vectors must agree to 5 decimal places."""
    fract_atoms, symmops, cell, centric = _build_sdm_inputs(DATA / cif_name)

    py_dists  = sorted(item.dist for item in
                       _run_python_path([list(a) for a in fract_atoms], symmops, cell, centric))
    cpp_dists = sorted(item.dist for item in
                       _run_cpp_path([list(a) for a in fract_atoms], symmops, cell, centric))

    assert len(py_dists) == len(cpp_dists)
    for py_d, cpp_d in zip(py_dists, cpp_dists):
        assert cpp_d == pytest.approx(py_d, rel=1e-5), (
            f"{cif_name}: distance mismatch {cpp_d!r} vs {py_d!r}"
        )


@pytest.mark.parametrize("cif_name", [
    "1979688_small.cif",
    "p21c.cif",
])
def test_cpp_matches_python_covalent_flags(cif_name):
    """Covalent flags must be identical for every SDM item."""
    fract_atoms, symmops, cell, centric = _build_sdm_inputs(DATA / cif_name)

    py_list  = sorted(_run_python_path([list(a) for a in fract_atoms], symmops, cell, centric),
                      key=lambda x: x.dist)
    cpp_list = sorted(_run_cpp_path([list(a) for a in fract_atoms], symmops, cell, centric),
                      key=lambda x: x.dist)

    flags_py  = [item.covalent for item in py_list]
    flags_cpp = [item.covalent for item in cpp_list]
    assert flags_cpp == flags_py, f"{cif_name}: covalent flag mismatch"


# ---------------------------------------------------------------------------
# Integration test: full grow pipeline via C++ path
# ---------------------------------------------------------------------------

def test_full_grow_cpp_path():
    """MoleculeLoader._compute_grown_atoms must work when C++ SDM is active."""
    from fastmolwidget.loader import MoleculeLoader

    original = sdm_module.HAS_CPP
    sdm_module.HAS_CPP = True
    try:
        cif = CifReader(DATA / "1979688_small.cif")
        result = MoleculeLoader._compute_grown_atoms(cif)
        assert len(result) > 0
    finally:
        sdm_module.HAS_CPP = original


