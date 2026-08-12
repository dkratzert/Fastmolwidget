"""Tests for residual (Fo−Fc) density display in the 2-D QPainter renderer.

The map computation itself is covered by ``test_density.py``; what matters here
is that the wireframe really reaches the canvas, that it follows the molecule
through rotations, and that it tracks the same visibility filters as the atoms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qtpy import QtGui, QtWidgets
from qtpy.QtCore import QPoint, QPointF, Qt

from fastmolwidget.density import HAS_DENSITY_CPP
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.molecule_base import (
    DENSITY_LEVEL_MAX,
    DENSITY_LEVEL_MIN,
    DENSITY_LEVEL_STEP,
)
from fastmolwidget.molecule_painter import DENSITY_NEG_COLOR, DENSITY_POS_COLOR
from fastmolwidget.viewer_widget import MoleculeViewerWidget

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

DATA = Path('tests/test-data')
RES = DATA / 'p31c-finalcif.res'
HKL = DATA / 'p31c-finalcif.hkl'
P21C = DATA / 'p21c.cif'

needs_cpp = pytest.mark.skipif(
    not HAS_DENSITY_CPP,
    reason='density_cpp C++ extension not built',
)

WIDTH, HEIGHT = 500, 400


@pytest.fixture
def widget():
    w = MoleculeWidget()
    w.resize(WIDTH, HEIGHT)
    MoleculeLoader(w).load_file(RES)
    return w


def render(widget: MoleculeWidget) -> np.ndarray:
    """Render *widget* offscreen and return the pixels as ``(H, W, 4)`` BGRA."""
    image = QtGui.QImage(WIDTH, HEIGHT, QtGui.QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.white)
    painter = QtGui.QPainter(image)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
    widget._painter = painter
    try:
        widget.draw()
    finally:
        widget._painter = None
        painter.end()
    return np.frombuffer(image.constBits(), dtype=np.uint8).reshape(
        HEIGHT, WIDTH, 4).copy()


def count_color(pixels: np.ndarray, color: QtGui.QColor, tolerance: int = 60) -> int:
    """Number of pixels close to *color* (the wireframe is antialiased)."""
    target = np.array([color.blue(), color.green(), color.red()], dtype=np.int16)
    delta = np.abs(pixels[:, :, :3].astype(np.int16) - target)
    return int((delta.max(axis=2) <= tolerance).sum())


# ------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------

@needs_cpp
def test_show_builds_both_lobes(widget):
    widget.show_residual_density(HKL)

    assert widget.residual_density_map is not None
    assert len(widget._density_pos_lines) > 0
    assert len(widget._density_neg_lines) > 0
    # (K, 2, 3) line segments
    assert widget._density_pos_lines.shape[1:] == (2, 3)


@needs_cpp
def test_segments_stay_near_the_atoms(widget):
    """Contouring is clipped to a margin around the visible atoms."""
    from fastmolwidget.molecule_painter import DENSITY_MARGIN

    widget.show_residual_density(HKL)
    vertices = widget._density_pos_lines.reshape(-1, 3)
    atoms = widget._model_coords_array

    distances = np.linalg.norm(vertices[:, None, :] - atoms[None, :, :], axis=2)
    assert distances.min(axis=1).max() <= DENSITY_MARGIN + 1e-6


@needs_cpp
def test_set_level_recontours_without_recomputing(widget):
    widget.show_residual_density(HKL, 0.2)
    density_map = widget.residual_density_map
    dense = len(widget._density_pos_lines)

    widget.set_residual_density_level(0.6)

    assert widget.residual_density_map is density_map  # not recomputed
    assert len(widget._density_pos_lines) < dense


@needs_cpp
def test_clear_removes_the_geometry(widget):
    widget.show_residual_density(HKL)
    assert len(widget._density_pos_lines) > 0

    widget.clear_residual_density()

    assert widget.residual_density_map is None
    assert len(widget._density_pos_lines) == 0
    assert len(widget._density_neg_lines) == 0


@needs_cpp
def test_hiding_hydrogens_reclips_the_density(widget, monkeypatch):
    """The surface must follow the visibility filters, not the loaded atoms."""
    widget.show_residual_density(HKL)
    with_h = len(widget._visible_model_positions())
    rebuilds = []
    monkeypatch.setattr(
        type(widget), '_build_density_geometry',
        lambda self: rebuilds.append(None),
    )

    widget.show_hydrogens(False)

    assert rebuilds, 'hiding hydrogens did not re-contour'
    assert len(widget._visible_model_positions()) < with_h


@needs_cpp
def test_part_filter_recontours():
    w = MoleculeWidget()
    w.resize(WIDTH, HEIGHT)
    MoleculeLoader(w).load_file(P21C)
    w.show_residual_density()
    before = len(w._density_pos_lines)

    w.set_visible_parts({0})

    assert len(w._density_pos_lines) != before


# ------------------------------------------------------------------
# Drawing
# ------------------------------------------------------------------

@needs_cpp
def test_both_lobes_reach_the_canvas(widget):
    plain = render(widget)
    widget.show_residual_density(HKL)
    with_density = render(widget)

    assert (count_color(with_density, DENSITY_POS_COLOR)
            > count_color(plain, DENSITY_POS_COLOR) + 200)
    assert (count_color(with_density, DENSITY_NEG_COLOR)
            > count_color(plain, DENSITY_NEG_COLOR) + 200)


@needs_cpp
def test_clearing_removes_the_wireframe_from_the_canvas(widget):
    plain = render(widget)
    widget.show_residual_density(HKL)
    widget.clear_residual_density()

    assert np.array_equal(render(widget), plain)


@needs_cpp
def test_drawing_does_not_disturb_the_painter_state(widget):
    """The lobes save/restore, so the axis indicator after them is unaffected."""
    widget.show_residual_density(HKL)
    image = QtGui.QImage(WIDTH, HEIGHT, QtGui.QImage.Format.Format_RGB32)
    painter = QtGui.QPainter(image)
    pen_before = QtGui.QPen(QtGui.QColor('#123456'), 4.0)
    painter.setPen(pen_before)
    widget._painter = painter
    widget._draw_residual_density()
    pen_after = painter.pen()
    widget._painter = None
    painter.end()

    assert pen_after.color().name() == pen_before.color().name()
    assert pen_after.widthF() == pytest.approx(pen_before.widthF())


# ------------------------------------------------------------------
# The wireframe follows the molecule
# ------------------------------------------------------------------

@needs_cpp
def test_segments_are_kept_in_the_model_frame(widget):
    """Rotating must not re-contour, only re-project."""
    widget.show_residual_density(HKL)
    before = widget._density_pos_lines.copy()

    widget._align_to_reciprocal_axis(0)

    assert np.array_equal(widget._density_pos_lines, before)


@needs_cpp
@pytest.mark.parametrize('rotate', [
    lambda w: w._align_to_reciprocal_axis(0),
    lambda w: w._align_to_reciprocal_axis(2),
    lambda w: w.align_best_view(),
])
def test_view_transform_matches_the_atoms(widget, rotate):
    """The density and the atoms must undergo exactly the same motion."""
    widget.show_residual_density(HKL)
    rotate(widget)

    expected = widget._to_view_frame(widget._model_coords_array)

    assert np.allclose(expected, widget._coords_array, atol=1e-4)


@needs_cpp
def test_view_transform_survives_a_pan_between_rotations(widget):
    """Panning moves the pivot, so the mapping is not a rotation about one point."""
    widget.show_residual_density(HKL)
    widget._align_to_reciprocal_axis(1)
    widget.molecule_center[0] += 3.0   # what pan_molecule() does
    widget.molecule_center[1] -= 2.0
    widget._align_to_reciprocal_axis(2)

    expected = widget._to_view_frame(widget._model_coords_array)

    assert np.allclose(expected, widget._coords_array, atol=1e-4)


@needs_cpp
def test_rotating_moves_the_wireframe_on_screen(widget):
    widget.show_residual_density(HKL)
    widget._align_to_reciprocal_axis(0)
    first = render(widget)
    widget._align_to_reciprocal_axis(2)
    second = render(widget)

    assert not np.array_equal(first, second)


@needs_cpp
def test_reloading_the_same_file_reclips_the_density():
    """Pack reloads in place; the surface has to follow the new atoms."""
    w = MoleculeWidget()
    w.resize(WIDTH, HEIGHT)
    loader = MoleculeLoader(w)
    loader.load_file(P21C)
    w.show_residual_density()
    before = len(w._density_pos_lines)

    loader.set_pack(True)

    assert w.residual_density_map is not None
    assert len(w._density_pos_lines) > before


# ------------------------------------------------------------------
# Viewer integration
# ------------------------------------------------------------------

def test_viewer_has_density_controls():
    viewer = MoleculeViewerWidget()

    assert viewer._residual_density_button.isCheckable()
    assert not viewer._residual_density_button.isChecked()
    assert not viewer._density_level_spinbox.isEnabled()


@needs_cpp
def test_viewer_forwards_to_the_render_widget():
    viewer = MoleculeViewerWidget()
    viewer.load_file(RES)

    viewer.show_residual_density(HKL, 0.25)
    assert viewer.render_widget.residual_density_map is not None
    assert viewer._residual_density_button.isChecked()
    assert viewer._density_level_spinbox.value() == pytest.approx(0.25)

    viewer.clear_residual_density()
    assert viewer.render_widget.residual_density_map is None
    assert not viewer._residual_density_button.isChecked()


@needs_cpp
def test_viewer_spinbox_recontours():
    viewer = MoleculeViewerWidget()
    viewer.load_file(RES)
    viewer.show_residual_density(HKL, 0.2)
    before = len(viewer.render_widget._density_pos_lines)

    viewer._density_level_spinbox.setValue(0.6)

    assert len(viewer.render_widget._density_pos_lines) < before


@needs_cpp
def test_viewer_button_uses_embedded_reflections():
    viewer = MoleculeViewerWidget()
    viewer.load_file(P21C)

    viewer._residual_density_button.click()

    assert viewer._residual_density_button.isChecked()
    assert viewer.render_widget.residual_density_map is not None
    assert viewer._density_level_spinbox.isEnabled()


@needs_cpp
def test_viewer_button_resets_when_another_file_is_loaded():
    viewer = MoleculeViewerWidget()
    viewer.load_file(P21C)
    viewer._residual_density_button.click()
    assert viewer._residual_density_button.isChecked()

    viewer.load_file(DATA / 'test_molecule.res')

    assert viewer.render_widget.residual_density_map is None
    assert not viewer._residual_density_button.isChecked()


def test_viewer_button_pops_back_out_when_the_dialog_is_cancelled(tmp_path, monkeypatch):
    if not HAS_DENSITY_CPP:
        pytest.skip('density_cpp C++ extension not built')
    plain = tmp_path / 'plain.res'
    plain.write_text((DATA / 'test_molecule.res').read_text())
    viewer = MoleculeViewerWidget()
    viewer.load_file(plain)
    # The user cancels the "pick a reflection file" dialog.
    monkeypatch.setattr(viewer, '_ask_for_reflection_file', lambda: None)

    viewer._residual_density_button.click()

    assert not viewer._residual_density_button.isChecked()
    assert not viewer._density_level_spinbox.isEnabled()


# ------------------------------------------------------------------
# Qt Quick renderer shares the implementation
# ------------------------------------------------------------------

@needs_cpp
def test_quick_item_supports_density():
    quick = pytest.importorskip('fastmolwidget.molecule_quick')
    if not quick._HAS_QTQUICK:
        pytest.skip('Qt Quick not available')
    item = quick.MoleculeQuickItem()
    MoleculeLoader(item).load_file(RES)

    item.show_residual_density(HKL)

    assert item.residual_density_map is not None
    assert len(item._density_pos_lines) > 0


# ------------------------------------------------------------------
# Ctrl + mouse wheel changes the contour level
# ------------------------------------------------------------------

def _wheel(widget, notches: int, *, ctrl: bool):
    """Send a wheel event and report whether the widget claimed it."""
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
def test_ctrl_wheel_raises_and_lowers_the_level(widget):
    widget.show_residual_density(HKL)
    start = widget.residual_density_level

    _wheel(widget, 1, ctrl=True)
    assert widget.residual_density_level == pytest.approx(start + DENSITY_LEVEL_STEP)

    _wheel(widget, -1, ctrl=True)
    _wheel(widget, -1, ctrl=True)
    assert widget.residual_density_level == pytest.approx(start - DENSITY_LEVEL_STEP)


@needs_cpp
def test_ctrl_wheel_recontours_the_surface(widget):
    widget.show_residual_density(HKL)
    dense = len(widget._density_pos_lines)

    _wheel(widget, 1, ctrl=True)

    assert len(widget._density_pos_lines) < dense


@needs_cpp
def test_ctrl_wheel_leaves_the_label_font_alone(widget):
    widget.show_residual_density(HKL)
    font_size = widget.fontsize

    _wheel(widget, 1, ctrl=True)

    assert widget.fontsize == font_size


def test_plain_wheel_still_resizes_the_labels(widget):
    font_size = widget.fontsize

    _wheel(widget, 1, ctrl=False)

    assert widget.fontsize == font_size + 2


def test_ctrl_wheel_without_a_map_is_ignored(widget):
    """No map to re-contour, so the event is left to the parent widget."""
    font_size = widget.fontsize

    accepted = _wheel(widget, 1, ctrl=True)

    assert not accepted
    assert widget.fontsize == font_size


@needs_cpp
def test_ctrl_wheel_is_clamped_to_the_control_range(widget):
    widget.show_residual_density(HKL)

    for _ in range(500):
        _wheel(widget, -1, ctrl=True)
    assert widget.residual_density_level == pytest.approx(DENSITY_LEVEL_MIN)

    for _ in range(800):
        _wheel(widget, 1, ctrl=True)
    assert widget.residual_density_level == pytest.approx(DENSITY_LEVEL_MAX)


@needs_cpp
def test_level_change_is_signalled(widget):
    widget.show_residual_density(HKL)
    seen: list[float] = []
    widget.densityLevelChanged.connect(seen.append)

    _wheel(widget, 1, ctrl=True)

    assert seen == [pytest.approx(widget.residual_density_level)]


@needs_cpp
def test_setting_the_same_level_does_not_signal(widget):
    widget.show_residual_density(HKL, 0.25)
    seen: list[float] = []
    widget.densityLevelChanged.connect(seen.append)

    widget.set_residual_density_level(0.25)

    assert seen == []


@needs_cpp
def test_ctrl_wheel_updates_the_viewer_spinbox():
    viewer = MoleculeViewerWidget()
    viewer.resize(WIDTH, HEIGHT)
    viewer.load_file(RES)
    viewer.show_residual_density(HKL, 0.25)

    _wheel(viewer.render_widget, 1, ctrl=True)
    _wheel(viewer.render_widget, 1, ctrl=True)

    assert viewer.render_widget.residual_density_level == pytest.approx(
        0.25 + 2 * DENSITY_LEVEL_STEP)
    assert viewer._density_level_spinbox.value() == pytest.approx(
        viewer.render_widget.residual_density_level)


@needs_cpp
def test_one_wheel_event_is_one_step(widget):
    """Matches the label-font behaviour: the delta magnitude is not scaled."""
    widget.show_residual_density(HKL, 0.25)

    _wheel(widget, 3, ctrl=True)

    assert widget.residual_density_level == pytest.approx(0.25 + DENSITY_LEVEL_STEP)


@needs_cpp
def test_spinbox_still_drives_the_renderer():
    """The signal round-trip must not break the other direction."""
    viewer = MoleculeViewerWidget()
    viewer.resize(WIDTH, HEIGHT)
    viewer.load_file(RES)
    viewer.show_residual_density(HKL, 0.25)

    viewer._density_level_spinbox.setValue(0.35)

    assert viewer.render_widget.residual_density_level == pytest.approx(0.35)
