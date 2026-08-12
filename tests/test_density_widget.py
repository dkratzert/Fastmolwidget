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
def test_default_level_is_three_sigma(widget):
    """A fixed absolute level cannot suit every dataset; 3σ adapts."""
    widget.show_residual_density()

    density = widget.residual_density_map
    assert widget._density_level == pytest.approx(3.0 * density.rms, abs=0.005)


@needs_cpp
def test_explicit_level_overrides_the_sigma_default(widget):
    widget.show_residual_density(level=0.42)

    assert widget._density_level == pytest.approx(0.42)


@needs_cpp
def test_level_adapts_to_the_structure():
    """Two structures with different residual scales get different levels."""
    first = MoleculeWidget3D()
    MoleculeLoader(first).load_file(RES)
    first.show_residual_density()

    second = MoleculeWidget3D()
    MoleculeLoader(second).load_file(DATA / 'p21c.cif')
    second.show_residual_density()

    assert first.residual_density_map.rms != pytest.approx(
        second.residual_density_map.rms, rel=0.1)
    assert first._density_level != pytest.approx(second._density_level, rel=0.1)


@needs_cpp
def test_viewer_spinbox_shows_the_level_actually_used():
    """The spin box must not keep claiming a stale default."""
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(DATA / 'p21c.cif')

    viewer.show_residual_density()

    assert viewer._density_level_spinbox.value() == pytest.approx(
        viewer.render_widget._density_level)


@needs_cpp
def test_viewer_uses_embedded_reflections_without_a_dialog():
    """A self-contained CIF must not pop up a file dialog."""
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(DATA / 'p21c.cif')

    assert viewer._auto_reflection_file() == DATA / 'p21c.cif'


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
# 2-D renderer
# ------------------------------------------------------------------

def test_2d_widget_without_a_model_raises():
    """Same contract as the 3-D renderer: no model, no map."""
    flat = MoleculeWidget()

    with pytest.raises(RuntimeError, match='No structure model'):
        flat.show_residual_density(HKL, 0.1)


def test_2d_clear_without_a_map_is_safe():
    flat = MoleculeWidget()

    flat.clear_residual_density()

    assert flat.residual_density_map is None
    assert len(flat._density_pos_lines) == 0


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


# ------------------------------------------------------------------
# The button shows the on/off state
# ------------------------------------------------------------------

def test_density_button_is_a_toggle():
    viewer = MoleculeViewer3DWidget()

    assert viewer._residual_density_button.isCheckable()
    assert not viewer._residual_density_button.isChecked()
    # nothing to contour yet, so the level control is inert
    assert not viewer._density_level_spinbox.isEnabled()


@needs_cpp
def test_button_reflects_the_density_state():
    """Clicking toggles the density on and off, and the button shows it."""
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(DATA / 'p21c.cif')      # reflections embedded, no dialog
    button = viewer._residual_density_button

    button.click()
    assert button.isChecked()
    assert viewer.render_widget.residual_density_map is not None
    assert viewer._density_level_spinbox.isEnabled()

    button.click()
    assert not button.isChecked()
    assert viewer.render_widget.residual_density_map is None
    assert not viewer._density_level_spinbox.isEnabled()


@needs_cpp
def test_api_calls_keep_the_button_in_sync():
    """Showing/clearing from code must not leave a stale button state."""
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(DATA / 'p21c.cif')

    viewer.show_residual_density()
    assert viewer._residual_density_button.isChecked()

    viewer.clear_residual_density()
    assert not viewer._residual_density_button.isChecked()


def test_button_pops_back_out_when_the_dialog_is_cancelled(tmp_path):
    """A cancelled file dialog must not leave the button looking active."""
    import shutil

    lonely = tmp_path / 'lonely.res'
    shutil.copy(RES, lonely)

    viewer = MoleculeViewer3DWidget()
    viewer.load_file(lonely)
    viewer._ask_for_reflection_file = lambda: None   # user cancels

    viewer._residual_density_button.click()

    assert not viewer._residual_density_button.isChecked()
    assert not viewer._density_level_spinbox.isEnabled()


def test_button_pops_back_out_on_failure(monkeypatch, tmp_path):
    """A failed computation must not leave the button looking active."""
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(RES)
    viewer._ask_for_reflection_file = lambda: tmp_path / 'missing.hkl'
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning',
                        staticmethod(lambda *a, **k: None))

    viewer._residual_density_button.click()

    assert not viewer._residual_density_button.isChecked()
    assert viewer.render_widget.residual_density_map is None


def test_checked_button_has_a_distinct_appearance():
    """Relief alone is easy to miss - the checked state is also coloured."""
    viewer = MoleculeViewer3DWidget()
    style = viewer._residual_density_button.styleSheet()

    assert 'checked' in style
    assert 'background-color' in style


# ------------------------------------------------------------------
# Loading another structure resets the density
# ------------------------------------------------------------------

@needs_cpp
def test_loading_another_file_clears_the_density(widget):
    """The map belongs to the previous model and must not survive."""
    widget.show_residual_density(HKL, level=0.15)
    assert widget.residual_density_map is not None

    MoleculeLoader(widget).load_file(DATA / 'p21c.cif')

    assert widget.residual_density_map is None
    assert widget._density_pos_count == 0
    assert widget._density_neg_count == 0
    assert widget._density_verts.size == 0


@needs_cpp
def test_reloading_the_same_file_keeps_the_density():
    """Grow and pack reload the same path - the map is still valid."""
    view = MoleculeWidget3D()
    loader = MoleculeLoader(view)
    loader.load_file(RES)
    view.show_residual_density(HKL, level=0.15)

    loader.load_file(RES, keep_view=True)

    assert view.residual_density_map is not None


@needs_cpp
def test_growing_reclips_the_density():
    """More atoms on screen must mean more density around them."""
    view = MoleculeWidget3D()
    loader = MoleculeLoader(view)
    loader.load_file(RES)
    view.show_residual_density(HKL, level=0.15)
    before = view._density_verts.size
    atoms_before = len(view.atoms)

    loader.set_grow(True)

    assert len(view.atoms) > atoms_before
    assert view.residual_density_map is not None   # not recomputed
    assert view._density_verts.size > before


@needs_cpp
def test_viewer_button_resets_when_a_new_file_is_loaded():
    """The control bar must not claim density is shown after a reload."""
    viewer = MoleculeViewer3DWidget()
    viewer.load_file(DATA / 'p21c.cif')
    viewer._residual_density_button.click()
    assert viewer._residual_density_button.isChecked()

    viewer.load_file(DATA / 'p31c.cif')

    assert not viewer._residual_density_button.isChecked()
    assert not viewer._density_level_spinbox.isEnabled()
    assert viewer.render_widget.residual_density_map is None


def test_loading_a_file_without_density_is_harmless(widget):
    """Clearing on load must work even when nothing was ever shown."""
    MoleculeLoader(widget).load_file(DATA / 'p21c.cif')

    assert widget.residual_density_map is None


# ------------------------------------------------------------------
# Ctrl + mouse wheel changes the contour level (3-D)
# ------------------------------------------------------------------

def _wheel_3d(widget, notches: int, *, ctrl: bool):
    """Send a wheel event and report whether the widget claimed it."""
    from qtpy import QtGui
    from qtpy.QtCore import QPoint, QPointF, Qt

    event = QtGui.QWheelEvent(
        QPointF(50.0, 50.0),
        QPointF(50.0, 50.0),
        QPoint(0, 0),
        QPoint(0, 120 * notches),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.ControlModifier if ctrl else Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    widget.wheelEvent(event)
    return event.isAccepted()


@needs_cpp
def test_3d_ctrl_wheel_changes_the_level(widget):
    from fastmolwidget.molecule_base import DENSITY_LEVEL_STEP

    widget.show_residual_density(HKL, 0.25)
    font_size = widget.fontsize

    _wheel_3d(widget, 1, ctrl=True)

    assert widget.residual_density_level == pytest.approx(0.25 + DENSITY_LEVEL_STEP)
    assert widget.fontsize == font_size


@needs_cpp
def test_3d_ctrl_wheel_recontours(widget):
    widget.show_residual_density(HKL, 0.2)
    dense = widget._density_pos_count

    _wheel_3d(widget, 1, ctrl=True)

    assert widget._density_pos_count < dense


def test_3d_plain_wheel_still_resizes_the_labels(widget):
    font_size = widget.fontsize

    _wheel_3d(widget, -1, ctrl=False)

    assert widget.fontsize == font_size - 2


def test_3d_ctrl_wheel_without_a_map_is_ignored(widget):
    font_size = widget.fontsize

    accepted = _wheel_3d(widget, 1, ctrl=True)

    assert not accepted
    assert widget.fontsize == font_size


@needs_cpp
def test_3d_ctrl_wheel_updates_the_viewer_spinbox():
    from fastmolwidget.molecule_base import DENSITY_LEVEL_STEP

    viewer = MoleculeViewer3DWidget()
    viewer.load_file(RES)
    viewer.show_residual_density(HKL, 0.25)

    _wheel_3d(viewer.render_widget, 1, ctrl=True)

    assert viewer._density_level_spinbox.value() == pytest.approx(
        0.25 + DENSITY_LEVEL_STEP)
