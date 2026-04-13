"""Tests for :mod:`fastmolwidget.density` and :mod:`fastmolwidget.isosurface`."""

from pathlib import Path

import numpy as np
import pytest

from fastmolwidget.density import (
    compute_difference_density,
    read_hkl_data,
    _merge_reflections,
)
from fastmolwidget.isosurface import marching_cubes_wireframe

test_data_dir = Path('tests/test-data')


# ------------------------------------------------------------------
# HKL reading
# ------------------------------------------------------------------

class TestReadHklData:
    def test_read_from_file(self):
        reflections = read_hkl_data(test_data_dir / 'p31c-finalcif.hkl')
        assert len(reflections) > 1000
        h, k, l, fo2, sig = reflections[0]
        assert isinstance(h, int)
        assert isinstance(fo2, float)

    def test_read_from_string(self):
        text = "   1   0   0  12.92  0.58\n   0   1   0   8.50  0.42\n   0   0   0   0.00  0.00\n"
        reflections = read_hkl_data(text)
        assert len(reflections) == 2
        assert reflections[0] == (1, 0, 0, 12.92, 0.58)

    def test_stops_at_sentinel(self):
        text = "   1   0   0  10.00  1.00\n   0   0   0   0.00  0.00\n   2   0   0  20.00  2.00\n"
        reflections = read_hkl_data(text)
        assert len(reflections) == 1

    def test_empty_input(self):
        reflections = read_hkl_data("")
        assert reflections == []


class TestMergeReflections:
    def test_merges_duplicates(self):
        data_list = [
            (1, 0, 0, 10.0, 1.0),
            (1, 0, 0, 12.0, 1.0),
            (0, 1, 0, 5.0, 0.5),
        ]
        merged = _merge_reflections(data_list)
        assert len(merged) == 2
        assert merged[(1, 0, 0)][0] == pytest.approx(11.0)
        assert merged[(0, 1, 0)][0] == pytest.approx(5.0)


# ------------------------------------------------------------------
# Difference density computation
# ------------------------------------------------------------------

class TestComputeDifferenceDensity:
    def test_p21c_cif_embedded_hkl(self):
        arr, cell, sg = compute_difference_density(test_data_dir / 'p21c.cif')
        assert arr.ndim == 3
        assert arr.shape[0] > 10
        assert arr.shape[1] > 10
        assert arr.shape[2] > 10
        rms = np.sqrt(np.mean(arr ** 2))
        assert rms > 0
        # RMS should be small for a well-refined structure
        assert rms < 1.0

    def test_p31c_cif_external_hkl(self):
        arr, cell, sg = compute_difference_density(
            test_data_dir / 'p31c.cif',
            hkl_path=test_data_dir / 'p31c-finalcif.hkl',
        )
        assert arr.ndim == 3
        rms = np.sqrt(np.mean(arr ** 2))
        assert rms > 0

    def test_p31c_cif_embedded_hkl(self):
        arr, cell, sg = compute_difference_density(test_data_dir / 'p31c.cif')
        assert arr.ndim == 3
        rms = np.sqrt(np.mean(arr ** 2))
        assert rms > 0

    def test_returns_unit_cell_and_spacegroup(self):
        import gemmi
        arr, cell, sg = compute_difference_density(test_data_dir / 'p21c.cif')
        assert isinstance(cell, gemmi.UnitCell)
        assert isinstance(sg, gemmi.SpaceGroup)
        assert cell.a > 0
        assert cell.b > 0
        assert cell.c > 0


# ------------------------------------------------------------------
# Isosurface extraction
# ------------------------------------------------------------------

class TestMarchingCubesWireframe:
    def test_positive_isosurface(self):
        arr, cell, sg = compute_difference_density(test_data_dir / 'p21c.cif')
        rms = np.sqrt(np.mean(arr ** 2))
        segments = marching_cubes_wireframe(arr, 3.0 * rms, cell)
        assert segments.ndim == 2
        assert segments.shape[1] == 6
        assert len(segments) > 0

    def test_negative_isosurface(self):
        arr, cell, sg = compute_difference_density(test_data_dir / 'p21c.cif')
        rms = np.sqrt(np.mean(arr ** 2))
        segments = marching_cubes_wireframe(arr, -3.0 * rms, cell)
        assert segments.ndim == 2
        assert segments.shape[1] == 6
        assert len(segments) > 0

    def test_no_isosurface_at_extreme_level(self):
        arr, cell, sg = compute_difference_density(test_data_dir / 'p21c.cif')
        # Very high level - should produce no segments
        segments = marching_cubes_wireframe(arr, 1000.0, cell)
        assert len(segments) == 0

    def test_segments_are_cartesian(self):
        arr, cell, sg = compute_difference_density(test_data_dir / 'p21c.cif')
        rms = np.sqrt(np.mean(arr ** 2))
        segments = marching_cubes_wireframe(arr, 3.0 * rms, cell)
        # Segments should be within the unit cell bounds (approximately)
        # Max coordinate should be less than max cell dimension + some margin
        max_dim = max(cell.a, cell.b, cell.c)
        assert np.all(segments < max_dim * 1.5)
        assert np.all(segments > -max_dim * 0.5)
