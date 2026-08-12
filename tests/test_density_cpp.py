"""Smoke tests for the optional density_cpp marching-cubes extension.

Skipped automatically when the module has not been compiled.

Run after building:
    uv pip install pybind11
    uv pip install -e . --no-build-isolation
    uv run pytest tests/test_density_cpp.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

density_cpp = pytest.importorskip("fastmolwidget.density_cpp",
                                  reason="density_cpp C++ extension not built — skipping")


def _sphere_grid(size: int = 32, radius: float = 10.0, center: tuple[float, float, float] = (16.0, 16.0, 16.0)) -> np.ndarray:
    x, y, z = np.indices((size, size, size), dtype=np.float64)
    cx, cy, cz = center
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
    return radius - distance


def test_marching_cubes_sphere_smoke():
    grid = _sphere_grid()

    vertices, edges = density_cpp.marching_cubes(grid, 0.0)

    assert isinstance(vertices, np.ndarray)
    assert isinstance(edges, np.ndarray)
    assert vertices.dtype == np.float64
    assert edges.dtype == np.int64
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert vertices.shape[0] > 0
    assert edges.shape[0] > 0
    assert np.all(edges >= 0)
    assert np.all(edges < vertices.shape[0])

    center = np.array([16.0, 16.0, 16.0], dtype=np.float64)
    radii = np.linalg.norm(vertices - center, axis=1)
    assert radii.mean() == pytest.approx(10.0, rel=0.1)


# ---------------------------------------------------------------------------
# structure_factors
# ---------------------------------------------------------------------------

def _single_atom_arguments(hkl: np.ndarray, fract: tuple[float, float, float]):
    """One isotropic point scatterer of unit form factor in P1."""
    n = len(hkl)
    return {
        "hkl": np.asarray(hkl, dtype=np.int32),
        "stol2": np.zeros(n),
        "rotations": np.eye(3).reshape(1, 3, 3),
        "translations": np.zeros((1, 3)),
        "fract": np.array([fract], dtype=float),
        "occupancies": np.ones(1),
        "u_iso": np.zeros(1),
        "aniso": np.zeros((1, 6)),
        "form_index": np.zeros(1, dtype=np.int32),
        "form_factors": np.ones((1, n)),
        "reciprocal": (0.1, 0.1, 0.1),
    }


def test_structure_factors_is_the_phase_factor_for_one_atom():
    hkl = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [-3, 1, 4]], dtype=np.int32)
    position = (0.11, 0.23, 0.37)

    result = density_cpp.structure_factors(**_single_atom_arguments(hkl, position))

    expected = np.exp(2j * np.pi * (hkl @ np.array(position)))
    assert result.dtype == np.complex128
    assert np.allclose(result, expected)


def test_structure_factors_applies_the_isotropic_debye_waller_factor():
    hkl = np.array([[1, 2, 3]], dtype=np.int32)
    arguments = _single_atom_arguments(hkl, (0.0, 0.0, 0.0))
    arguments["stol2"] = np.array([0.25])
    arguments["u_iso"] = np.array([0.05])

    result = density_cpp.structure_factors(**arguments)

    assert result[0].real == pytest.approx(np.exp(-8 * np.pi**2 * 0.25 * 0.05))


def test_structure_factors_sums_over_symmetry_images():
    hkl = np.array([[1, 0, 0], [2, 1, 0]], dtype=np.int32)
    arguments = _single_atom_arguments(hkl, (0.1, 0.2, 0.3))
    # Add an inversion centre: the sum must become real.
    arguments["rotations"] = np.stack([np.eye(3), -np.eye(3)])
    arguments["translations"] = np.zeros((2, 3))

    result = density_cpp.structure_factors(**arguments)

    expected = 2 * np.cos(2 * np.pi * (hkl @ np.array([0.1, 0.2, 0.3])))
    assert np.allclose(result.imag, 0.0)
    assert np.allclose(result.real, expected)


def test_structure_factors_rejects_mismatched_shapes():
    hkl = np.array([[1, 0, 0]], dtype=np.int32)
    arguments = _single_atom_arguments(hkl, (0.0, 0.0, 0.0))
    arguments["occupancies"] = np.ones(2)

    with pytest.raises(ValueError):
        density_cpp.structure_factors(**arguments)
