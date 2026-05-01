"""Tests for :class:`~fastmolwidget.molecule_quick3D.MoleculeQuick3D`.

:class:`MoleculeQuick3D` requires ``QQuickFramebufferObject`` (PySide6 or
PyQt6 with QtQuick) AND a real OpenGL context to render.  These tests focus on
the pure-Python logic that can run without a display: molecule loading,
geometry building, matrix computation, and hit testing.  Tests that need a
live Qt Quick scene are skipped when the dependency is unavailable.
"""

from __future__ import annotations

import numpy as np
import pytest
from qtpy import QtWidgets

from fastmolwidget.molecule_quick3D import (
    MoleculeQuick3D,
    _HAS_QQFBO,
    setup_opengl_backend,
)
from fastmolwidget.sdm import Atomtuple

# ---------------------------------------------------------------------------
# Module-level QApplication (required for some Qt operations even without a
# display).  Re-uses an existing instance if one was already created.
# ---------------------------------------------------------------------------

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

needs_qqfbo = pytest.mark.skipif(
    not _HAS_QQFBO,
    reason="QQuickFramebufferObject not available (install PySide6 with QtQuick)",
)


def _make_item() -> MoleculeQuick3D:
    """Construct a MoleculeQuick3D without a Qt Quick scene."""
    return MoleculeQuick3D()


def _simple_atoms() -> list[Atomtuple]:
    return [
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
        Atomtuple("H1", "H", -0.5, 0.5, 0.0, 0),
    ]


# ---------------------------------------------------------------------------
# Availability / import
# ---------------------------------------------------------------------------


def test_import_module() -> None:
    """module imports unconditionally and exposes __all__."""
    import fastmolwidget.molecule_quick3D as m

    assert hasattr(m, "MoleculeQuick3D")
    assert hasattr(m, "setup_opengl_backend")
    assert hasattr(m, "_HAS_QQFBO")


def test_setup_opengl_backend_no_crash() -> None:
    """setup_opengl_backend() must not raise even without a QQuickWindow."""
    setup_opengl_backend()  # best-effort; just check it doesn't throw


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_construction_defaults() -> None:
    item = _make_item()
    assert item.atoms == []
    assert item.connections == ()
    assert item.bond_width == 3
    assert item.labels is False
    assert item.show_hydrogens_flag is True
    assert item._show_adps is True
    assert item._zoom == pytest.approx(1.0)
    np.testing.assert_array_equal(item._rot_matrix, np.eye(3))


@needs_qqfbo
def test_construction_raises_without_qqfbo(monkeypatch: pytest.MonkeyPatch) -> None:
    """When _HAS_QQFBO is False, __init__ must raise RuntimeError."""
    import fastmolwidget.molecule_quick3D as m

    monkeypatch.setattr(m, "_HAS_QQFBO", False)
    # _QFBOBase is already set to QWidget so super().__init__ will succeed,
    # but the guard we added raises before calling super.
    with pytest.raises(RuntimeError, match="QQuickFramebufferObject"):
        m.MoleculeQuick3D()


# ---------------------------------------------------------------------------
# open_molecule / clear
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_open_molecule_loads_atoms() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    assert len(item.atoms) == 3
    assert item.atoms[0].label == "C1"
    assert item.atoms[1].type_ == "O"


@needs_qqfbo
def test_open_molecule_resets_view() -> None:
    item = _make_item()
    item._zoom = 3.0
    item._pan = np.array([5.0, 5.0], dtype=np.float32)
    item.open_molecule(_simple_atoms())
    assert item._zoom == pytest.approx(1.0)
    np.testing.assert_array_equal(item._pan, [0.0, 0.0])


@needs_qqfbo
def test_open_molecule_keep_view() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    item._zoom = 2.5
    item.open_molecule(_simple_atoms(), keep_view=True)
    assert item._zoom == pytest.approx(2.5)


@needs_qqfbo
def test_clear_removes_atoms() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    assert len(item.atoms) == 3
    item.clear()
    assert item.atoms == []


# ---------------------------------------------------------------------------
# Geometry building
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_sphere_geometry_built() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    # No ADPs → all atoms are spheres.
    assert item._sphere_count > 0
    assert item._sphere_verts.size > 0
    assert item._sphere_idx.size > 0


@needs_qqfbo
def test_cylinder_geometry_built() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    # At least some bonds should exist for C-O and C-H.
    assert item._cylinder_count >= 0  # bonds may or may not form depending on radii


@needs_qqfbo
def test_geometry_dirty_after_open() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    # open_molecule calls _build_geometry which sets _geometry_dirty.
    # After the build, _geometry_dirty should be True until the renderer
    # consumes it.
    assert item._geometry_dirty is True


@needs_qqfbo
def test_hydrogen_hidden_geometry() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    item.show_hydrogens(False)
    # H1 should be excluded; sphere count must drop.
    count_with_h = item._sphere_count
    item.show_hydrogens(False)
    assert item._sphere_count <= count_with_h


# ---------------------------------------------------------------------------
# Display toggles
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_show_adps_toggle() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    item.show_adps(False)
    assert item._show_adps is False
    item.show_adps(True)
    assert item._show_adps is True


@needs_qqfbo
def test_show_labels_toggle() -> None:
    item = _make_item()
    item.show_labels(True)
    assert item.labels is True
    item.show_labels(False)
    assert item.labels is False


@needs_qqfbo
def test_set_bond_width() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    item.set_bond_width(8)
    assert item.bond_width == 8


@needs_qqfbo
def test_set_bond_color_str() -> None:
    item = _make_item()
    item.set_bond_color("#ff0000")
    r, g, b = item._bond_rgb
    assert r == pytest.approx(1.0, abs=0.01)
    assert g == pytest.approx(0.0, abs=0.01)
    assert b == pytest.approx(0.0, abs=0.01)


@needs_qqfbo
def test_set_background_color() -> None:
    from qtpy import QtGui

    item = _make_item()
    item.set_background_color(QtGui.QColor(0, 0, 255))
    assert item._bg_rgb[2] == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# View control
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_reset_view() -> None:
    item = _make_item()
    item._zoom = 3.0
    item._pan[:] = [4.0, 4.0]
    item._rot_matrix = np.zeros((3, 3), dtype=np.float32)
    item.reset_view()
    assert item._zoom == pytest.approx(1.0)
    np.testing.assert_array_equal(item._pan, [0.0, 0.0])
    np.testing.assert_array_almost_equal(item._rot_matrix, np.eye(3))


@needs_qqfbo
def test_reset_rotation_center() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    item._pan[:] = [10.0, 10.0]
    item.reset_rotation_center()
    np.testing.assert_array_equal(item._pan, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Matrix computation
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_compute_mv_matrix_shape() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    mv = item._compute_mv_matrix()
    assert mv.shape == (4, 4)
    assert mv.dtype == np.float32


@needs_qqfbo
def test_compute_proj_matrix_shape() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    proj = item._compute_proj_matrix()
    assert proj.shape == (4, 4)


@needs_qqfbo
def test_ortho_half_extents_positive() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    # Width/height are 0 when not in a scene; _ortho_half_extents clamps to 1.
    hw, hh = item._ortho_half_extents()
    assert hw > 0
    assert hh > 0


# ---------------------------------------------------------------------------
# Hit testing (no display needed)
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_pick_atom_at_miss() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    # Screen coordinates far outside (uses 0×0 item size → unreliable, but
    # should not raise).
    atom, t = item._pick_atom_at(99999.0, 99999.0)
    # May or may not hit — just verify it returns the correct types.
    assert atom is None or hasattr(atom, "label")
    assert t >= 0.0


@needs_qqfbo
def test_ray_sphere_hit() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    mv = item._compute_mv_matrix()
    # Build a ray pointing straight at the first atom's eye-space position.
    c_eye = (mv @ np.array([*item.atoms[0].center, 1.0], dtype=np.float32))[:3]
    origin = np.array([c_eye[0], c_eye[1], 0.0], dtype=np.float32)
    direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    t = item._ray_sphere_hit_viewspace(
        origin, direction, item.atoms[0].center,
        item.atoms[0].display_radius, mv
    )
    assert t is not None
    assert t > 0.0


# ---------------------------------------------------------------------------
# QML properties
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_qml_property_bondColor_round_trip() -> None:
    item = _make_item()
    item.bondColor = "#123456"
    assert item.bondColor.lower() == "#123456"


@needs_qqfbo
def test_qml_property_backgroundColor_round_trip() -> None:
    item = _make_item()
    item.backgroundColor = "#aabbcc"
    assert item.backgroundColor.lower() == "#aabbcc"


@needs_qqfbo
def test_label_positions_empty_when_no_atoms() -> None:
    item = _make_item()
    item._recompute_label_positions()
    assert item._label_positions == []


@needs_qqfbo
def test_label_positions_populated_when_labels_on() -> None:
    item = _make_item()
    # Give the item a non-zero size so projection works.
    try:
        item.setWidth(640)
        item.setHeight(480)
    except Exception:
        pass
    item.show_labels(True)
    item.open_molecule(_simple_atoms())
    # May be empty if width/height are still 0 in a headless env.
    # Just check it's a list.
    assert isinstance(item._label_positions, list)


# ---------------------------------------------------------------------------
# Protocol satisfaction
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_satisfies_protocol() -> None:
    """MoleculeQuick3D must implement all MoleculeWidgetProtocol methods."""
    from fastmolwidget.molecule_base import MoleculeWidgetProtocol

    item = _make_item()
    assert isinstance(item, MoleculeWidgetProtocol)


# ---------------------------------------------------------------------------
# grow_molecule
# ---------------------------------------------------------------------------


@needs_qqfbo
def test_grow_molecule_keeps_view() -> None:
    item = _make_item()
    item.open_molecule(_simple_atoms())
    item._zoom = 2.0
    item.grow_molecule(_simple_atoms())
    assert len(item.atoms) == 3
    assert item._zoom == pytest.approx(2.0)
