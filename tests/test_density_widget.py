"""Widget-level tests for residual (Fo−Fc) density display.

These exercise the public API of :class:`MoleculeWidget3D` and
:class:`MoleculeViewer3DWidget` — map computation, geometry building and
tear-down.  The actual OpenGL draw calls are not exercised here (they need a
real GPU context), only the CPU-side state that feeds them.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy import QtWidgets

from fastmolwidget.density import HAS_DENSITY_CPP
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.molecule3D import MoleculeWidget3D
from fastmolwidget.viewer_widget3D import MoleculeViewer3DWidget

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

DATA = Path('tests/test-data')
RES = DATA / 'p31c-finalcif.res'
HKL = DATA / 'p31c-finalcif.hkl'

needs_cpp = pytest.mark.skipif(
    not HAS_DENSITY_CPP,
    reason='density_cpp C++ extension not built',
)


@pytest.fixture
def widget():
    w = MoleculeWidget3D()
    MoleculeLoader(w).load_file(RES)
    return w


# ------------------------------------------------------------------
# Model tracking
# ------------------------------------------------------------------

def test_loader_records_model_path(widget):
    """The loader tells the widget which file it is showing."""
    assert widget._model_path == RES


def test_density_absent_by_default(widget):
    assert widget.residual_density_map is None


def test_show_without_model_raises():
    """A widget that never loaded a file cannot compute Fc."""
    bare = MoleculeWidget3D()

    with pytest.raises(RuntimeError, match='No structure model'):
        bare.show_residual_density(HKL)


# ------------------------------------------------------------------
# Automatic reflection-data discovery
# ------------------------------------------------------------------

@needs_cpp
def test_hkl_is_found_automatically(widget):
    """No argument needed - the sibling .hkl is located by itself."""
    widget.show_residual_density(level=0.15)

    assert widget.residual_density_map is not None
    assert widget._density_pos_count > 0
    assert widget._density_neg_count > 0


@needs_cpp
def test_self_contained_cif_needs_no_arguments():
    """p21c.cif embeds its own reflections."""
    view = MoleculeWidget3D()
    MoleculeLoader(view).load_file(DATA / 'p21c.cif')

    view.show_residual_density()

    assert view.residual_density_map is not None


def test_missing_reflection_data_raises(tmp_path):
    """Without any data the caller gets a clear error, not a silent no-op."""
    import shutil

    lonely = tmp_path / 'lonely.res'
    shutil.copy(RES, lonely)

    view = MoleculeWidget3D()
    MoleculeLoader(view).load_file(lonely)

    with pytest.raises(FileNotFoundError, match='No reflection data'):
        view.show_residual_density()


@needs_cpp
def test_default_level_is_0_3(widget):
    widget.show_residual_density()

    assert widget._density_level == pytest.approx(0.30)


@needs_cpp
def test_viewer_uses_embedded_reflections_without_a_dialog():
    """A self-contained CIF must not pop up a file dialog."""
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(DATA / 'p21c.cif')

    assert viewer._auto_reflection_file() == DATA / 'p21c.cif'
    assert viewer._density_level_spinbox.value() == pytest.approx(0.30)


def test_viewer_asks_when_a_separate_hkl_is_needed():
    """A .res with a sibling .hkl must still go through the dialog.

    Silently picking up a neighbouring file would hide which dataset is
    actually being used.
    """
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(RES)

    assert HKL.exists()                       # the sibling really is there
    assert viewer._auto_reflection_file() is None


def test_dialog_preselects_the_sibling_hkl():
    """The user should not have to hunt for the obvious file."""
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(RES)

    assert viewer._residual_density_start_path() == str(HKL)


def test_viewer_auto_lookup_is_none_without_a_model():
    viewer = MoleculeViewer3DWidget()

    assert viewer._auto_reflection_file() is None


# ------------------------------------------------------------------
# Computing and clearing
# ------------------------------------------------------------------

@needs_cpp
def test_show_residual_density_builds_geometry(widget):
    widget.show_residual_density(HKL, level=0.25)

    assert widget.residual_density_map is not None
    assert widget._density_pos_count > 0
    assert widget._density_neg_count > 0
    assert widget._density_verts.size > 0
    assert widget._density_idx.size == (
        widget._density_pos_count + widget._density_neg_count
    )


@needs_cpp
def test_indices_stay_inside_the_vertex_buffer(widget):
    """The two lobes share one buffer, so the negative lobe must be offset."""
    widget.show_residual_density(HKL, level=0.25)

    vertex_count = widget._density_verts.size // 3
    assert widget._density_idx.max() < vertex_count


@needs_cpp
def test_explicit_model_path_overrides_loaded_file():
    bare = MoleculeWidget3D()

    bare.show_residual_density(HKL, level=0.3, model_path=RES)

    assert bare.residual_density_map is not None


@needs_cpp
def test_set_level_recontours_without_recomputing(widget):
    widget.show_residual_density(HKL, level=0.2)
    original_map = widget.residual_density_map
    low_count = widget._density_pos_count

    widget.set_residual_density_level(0.35)

    assert widget.residual_density_map is original_map  # not recomputed
    assert widget._density_pos_count < low_count


def test_set_level_without_map_is_a_no_op(widget):
    widget.set_residual_density_level(0.3)

    assert widget.residual_density_map is None
    assert widget._density_pos_count == 0


# ------------------------------------------------------------------
# Only around the visible atoms
# ------------------------------------------------------------------

@needs_cpp
def test_density_only_near_visible_atoms(widget):
    """Every vertex must sit within DENSITY_MARGIN of a *visible* atom."""
    import numpy as np

    from fastmolwidget.molecule3D import DENSITY_MARGIN

    widget.show_residual_density(HKL, level=0.2)
    vertices = widget._density_verts.reshape(-1, 3)
    assert len(vertices) > 0

    visible = widget._visible_atom_positions()
    distances = np.linalg.norm(
        vertices[:, None, :] - visible[None, :, :], axis=2).min(axis=1)
    assert distances.max() <= DENSITY_MARGIN + 1e-3


@needs_cpp
def test_hiding_hydrogens_recontours_the_density(widget):
    """Hidden atoms must stop pulling density into the view."""
    widget.show_residual_density(HKL, level=0.15)
    with_hydrogens = widget._density_verts.size

    widget.show_hydrogens(False)

    assert widget._density_verts.size <= with_hydrogens
    assert widget.residual_density_map is not None  # not recomputed


@needs_cpp
def test_part_filter_recontours_the_density(widget):
    widget.show_residual_density(HKL, level=0.15)
    all_parts = widget._density_verts.size

    widget.set_visible_parts({0})

    assert widget._density_verts.size <= all_parts


def test_visible_positions_follow_the_filters(widget):
    everything = len(widget._visible_atom_positions())

    widget.show_hydrogens(False)
    without_h = len(widget._visible_atom_positions())

    assert without_h < everything


@needs_cpp
def test_clear_removes_everything(widget):
    widget.show_residual_density(HKL, level=0.25)

    widget.clear_residual_density()

    assert widget.residual_density_map is None
    assert widget._density_pos_count == 0
    assert widget._density_neg_count == 0
    assert widget._density_verts.size == 0


def test_clear_without_map_is_safe(widget):
    widget.clear_residual_density()

    assert widget.residual_density_map is None


# ------------------------------------------------------------------
# 2-D renderer stubs
# ------------------------------------------------------------------

def test_2d_widget_stubs_are_no_ops():
    """The 2-D renderer accepts the calls but draws nothing."""
    flat = MoleculeWidget()

    assert flat.show_residual_density(HKL, 0.1) is None
    assert flat.clear_residual_density() is None


# ------------------------------------------------------------------
# Viewer integration
# ------------------------------------------------------------------

@needs_cpp
def test_viewer_forwards_to_render_widget():
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(RES)

    viewer.show_residual_density(HKL, 0.25)
    assert viewer.render_widget.residual_density_map is not None

    viewer.clear_residual_density()
    assert viewer.render_widget.residual_density_map is None
