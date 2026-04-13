"""Tests for :mod:`fastmolwidget.diff_map` and the density-mesh integration."""

from pathlib import Path

import numpy as np
import pytest
from qtpy import QtWidgets

from fastmolwidget.diff_map import compute_diff_map, get_mesh_segments, DiffMapResult
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

data = Path('tests/test-data')
CIF = data / 'p31c.cif'
HKL = data / 'p31c-finalcif.hkl'


# ---------------------------------------------------------------------------
# compute_diff_map
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def result():
    """Compute the diff map once for all tests in this module."""
    return compute_diff_map(CIF, HKL)


def test_result_type(result):
    assert isinstance(result, DiffMapResult)


def test_grid_shape(result):
    nu, nv, nw = result.nu, result.nv, result.nw
    assert result.grid.shape == (nu, nv, nw)
    assert nu > 0 and nv > 0 and nw > 0


def test_sigma_positive(result):
    assert result.sigma > 0.0


def test_grid_has_positive_and_negative_values(result):
    """A real difference map should contain both positive and negative features."""
    assert result.grid.max() > 0.0
    assert result.grid.min() < 0.0


def test_embedded_hkl_gives_same_result(result):
    """compute_diff_map without explicit HKL reads the embedded _shelx_hkl_file."""
    result2 = compute_diff_map(CIF)
    np.testing.assert_allclose(result.sigma, result2.sigma, rtol=1e-4)


def test_missing_hkl_raises():
    """Raises ValueError when the CIF has no HKL data and no file is given."""
    # Use a CIF without embedded HKL and provide no external path
    with pytest.raises(ValueError):
        compute_diff_map(data / 'smallcif.cif')


# ---------------------------------------------------------------------------
# get_mesh_segments
# ---------------------------------------------------------------------------

def test_segments_shape_positive(result):
    pos, neg = get_mesh_segments(result, level_sigma=3.0)
    assert pos.ndim == 3
    assert pos.shape[1] == 2
    assert pos.shape[2] == 3


def test_segments_shape_negative(result):
    pos, neg = get_mesh_segments(result, level_sigma=3.0)
    assert neg.ndim == 3
    assert neg.shape[1] == 2
    assert neg.shape[2] == 3


def test_segments_are_float32(result):
    pos, neg = get_mesh_segments(result, level_sigma=3.0)
    assert pos.dtype == np.float32
    assert neg.dtype == np.float32


def test_high_level_gives_fewer_segments(result):
    pos3, neg3 = get_mesh_segments(result, level_sigma=3.0)
    pos10, neg10 = get_mesh_segments(result, level_sigma=10.0)
    # A stricter threshold should produce fewer (or equal) segments
    assert len(pos10) <= len(pos3)
    assert len(neg10) <= len(neg3)


def test_very_high_level_can_be_empty(result):
    """At an extremely high sigma the isosurface may vanish."""
    pos, neg = get_mesh_segments(result, level_sigma=100.0)
    # Should gracefully return empty arrays, not raise
    assert pos.ndim == 3
    assert neg.ndim == 3


# ---------------------------------------------------------------------------
# MoleculeWidget density mesh API
# ---------------------------------------------------------------------------

@pytest.fixture()
def widget():
    w = MoleculeWidget()
    w.resize(400, 300)
    return w


def test_set_density_mesh(widget, result):
    pos, neg = get_mesh_segments(result, level_sigma=3.0)
    widget.set_density_mesh(pos, neg)
    assert widget._density_pos is not None
    assert widget._density_neg is not None
    assert len(widget._density_pos) == len(pos)
    assert len(widget._density_neg) == len(neg)


def test_clear_density_mesh(widget, result):
    pos, neg = get_mesh_segments(result, level_sigma=3.0)
    widget.set_density_mesh(pos, neg)
    widget.clear_density_mesh()
    assert widget._density_pos is None
    assert widget._density_neg is None


def test_show_density_toggle(widget, result):
    pos, neg = get_mesh_segments(result, level_sigma=3.0)
    widget.set_density_mesh(pos, neg)
    widget.show_density(False)
    assert not widget._show_density
    widget.show_density(True)
    assert widget._show_density


# ---------------------------------------------------------------------------
# MoleculeLoader.load_diff_map
# ---------------------------------------------------------------------------

@pytest.fixture()
def loaded_widget():
    w = MoleculeWidget()
    w.resize(400, 300)
    loader = MoleculeLoader(w)
    loader.load_file(CIF)
    return w, loader


def test_loader_load_diff_map_with_hkl(loaded_widget):
    widget, loader = loaded_widget
    loader.load_diff_map(hkl_path=HKL, level_sigma=3.0)
    assert widget._density_pos is not None
    assert widget._density_neg is not None
    assert len(widget._density_pos) > 0 or len(widget._density_neg) > 0


def test_loader_load_diff_map_auto_hkl(loaded_widget):
    """Auto-detection: CIF's embedded _shelx_hkl_file should be used."""
    widget, loader = loaded_widget
    loader.load_diff_map(level_sigma=3.0)
    assert widget._density_pos is not None


def test_loader_load_diff_map_without_cif_raises():
    """Should raise ValueError when no CIF has been loaded yet."""
    w = MoleculeWidget()
    loader = MoleculeLoader(w)
    with pytest.raises(ValueError, match="CIF file must be loaded"):
        loader.load_diff_map()


def test_density_persists_after_grow(loaded_widget):
    """Growing the structure should not destroy the density mesh."""
    widget, loader = loaded_widget
    loader.load_diff_map(hkl_path=HKL)
    loader.set_grow(True)
    assert widget._density_pos is not None


def test_density_renders_without_crash(loaded_widget):
    widget, loader = loaded_widget
    loader.load_diff_map(hkl_path=HKL)
    widget.show()
    app.processEvents()
    pixmap = widget.grab()
    assert not pixmap.isNull()
