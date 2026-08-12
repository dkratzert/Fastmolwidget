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
