from pathlib import Path

import numpy as np
import pytest
from qtpy import QtGui, QtWidgets

from fastmolwidget.cif.cif_file_io import CifReader
from fastmolwidget.molecule2D import calc_volume, RenderItem, MoleculeWidget
from fastmolwidget.sdm import Atomtuple

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])
data = Path('tests/test-data')


def test_calc_volume():
    # Test with orthogonal cell (e.g., cubic 10, 10, 10, 90, 90, 90)
    vol = calc_volume(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    assert vol == pytest.approx(1000.0, rel=1e-5)

    # Test with monoclinic cell
    vol = calc_volume(10.0, 10.0, 10.0, 90.0, 120.0, 90.0)
    assert vol == pytest.approx(866.0254, rel=1e-5)


def test_render_item():
    item = RenderItem(is_bond=True, z_order=1.5)
    assert item.is_bond is True
    assert item.z_order == 1.5
    assert item.atom1 is None


def test_molecule_widget_creation():
    widget = MoleculeWidget()
    assert widget.atoms_size == 70.0
    assert widget.fontsize == 13
    assert widget.bond_width == 3
    assert widget.labels is True
    assert widget._show_adps is True


def test_adp_intersection_line_width_scales_with_zoom():
    widget = MoleculeWidget()
    widget.zoom = 0.2
    thin = widget._adp_intersection_line_width()
    widget.zoom = 1.0
    thick = widget._adp_intersection_line_width()
    assert thick > thin


def test_molecule_widget_with_cif():
    from fastmolwidget.loader import MoleculeLoader
    from fastmolwidget.molecule2D import MoleculeWidget as _MW

    widget = _MW()
    widget.resize(800, 600)
    loader = MoleculeLoader(widget)
    loader.load_file(data / '1979688_small.cif')
    widget.show()

    assert len(widget.atoms) == 94

    clicked_atom = widget.atoms[7]
    clicked_atom.screenx = 80
    clicked_atom.screeny = 222
    assert widget.is_point_inside_atom(clicked_atom, 80.0, 222) == True

    # Ensure grabbing the widget content as pixmap (invoking paintEvent) does not crash
    pixmap = widget.grab()
    assert not pixmap.isNull()

    # Test setting parameters and re-drawing
    widget.labels = False
    widget.show_adps = False
    widget.zoom = 15 / 70  # equivalent to atoms_size = 15
    widget.bond_width = 4
    widget.repaint()

    # Test grabbing again to ensure settings do not crash rendering
    pixmap_updated = widget.grab()
    assert not pixmap_updated.isNull()

    # Test interaction (zooming, reset)
    widget.reset_view()
    widget.zoom = 1.2

    pixmap_transformed = widget.grab()
    assert not pixmap_transformed.isNull()


def test_molecule_widget_toggles():
    widget = MoleculeWidget()

    # Test setting label visibility
    widget.set_labels_visible(False)
    assert widget.labels is False
    widget.show_labels(True)
    assert widget.labels is True

    # Test hydrogen visibility
    widget.show_hydrogens(False)
    assert widget.show_hydrogens_flag is False

    # Test ADP visibility
    widget.show_adps(False)
    assert widget._show_adps is False


    # Test label font setting
    widget.setLabelFont(20)
    assert widget.fontsize == 20
    widget.setLabelFont(-5)
    assert widget.fontsize == 1

    # Test set background color
    from qtpy.QtGui import QColor, QPalette
    from qtpy import QtCore
    widget.set_background_color(QColor(QtCore.Qt.GlobalColor.black))
    assert widget.palette().color(QPalette.ColorRole.Window).name() == QColor(QtCore.Qt.GlobalColor.black).name()


def test_molecule_widget_clear():
    widget = MoleculeWidget()

    # create dummy atoms
    dummy_atom = Atomtuple('C1', 'C', 0.0, 0.0, 0.0, 0)
    widget.open_molecule([dummy_atom])
    assert len(widget.atoms) == 1

    assert widget.is_point_inside_atom(widget.atoms[0], 0, 0) == True
    assert widget.is_point_inside_atom(widget.atoms[0], 100, 100) == False

    widget.clear()
    assert len(widget.atoms) == 0


def test_molecule_widget_rotation_matrices():
    widget = MoleculeWidget()
    widget.x_angle = 3.14159 / 2  # 90 degrees approx
    widget.y_angle = 3.14159 / 2

    rx = widget.rotate_x()
    ry = widget.rotate_y()

    import numpy as np
    assert rx.shape == (3, 3)
    assert ry.shape == (3, 3)
    # just checking that they run and return a matrix
    assert isinstance(rx, np.ndarray)


def test_mouse_events_record_position():
    widget = MoleculeWidget()
    widget.resize(200, 200)
    widget.show()

    from qtpy.QtCore import QPointF, QPoint
    from qtpy import QtCore
    from qtpy.QtTest import QTest

    QTest.mousePress(widget, QtCore.Qt.MouseButton.LeftButton,
                     QtCore.Qt.KeyboardModifier.NoModifier, QPoint(10, 20))

    assert widget._lastPos == QPointF(10.0, 20.0)
    assert widget._pressPos == QPointF(10.0, 20.0)


# ------------------------------------------------------------------
# Bond color control
# ------------------------------------------------------------------

def test_set_bond_color_with_qcolor():
    """Test set_bond_color with QColor input."""
    widget = MoleculeWidget()
    widget.set_bond_color(QtGui.QColor("#6b5d4f"))
    assert widget.bond_color == QtGui.QColor("#6b5d4f")


def test_set_bond_color_with_hex_string():
    """Test set_bond_color with hex string input."""
    widget = MoleculeWidget()
    widget.set_bond_color("#5f5348")
    assert widget.bond_color == QtGui.QColor("#5f5348")


def test_set_bond_color_with_integer_tuple():
    """Test set_bond_color with integer RGB tuple (0..255)."""
    widget = MoleculeWidget()
    widget.set_bond_color((120, 110, 100))
    expected = QtGui.QColor(120, 110, 100)
    assert widget.bond_color == expected


def test_set_bond_color_with_float_tuple():
    """Test set_bond_color with float RGB tuple (0..1)."""
    widget = MoleculeWidget()
    widget.set_bond_color((0.5, 0.4, 0.3))
    expected = QtGui.QColor(int(0.5 * 255), int(0.4 * 255), int(0.3 * 255))
    assert widget.bond_color == expected


def test_set_bond_color_updates_bond_brush():
    """bond_brush must be rebuilt when set_bond_color is called, so that
    rounded-bond rendering actually uses the new colour."""
    widget = MoleculeWidget()
    old_brush = widget.bond_brush

    widget.set_bond_color(QtGui.QColor("#ff0000"))  # bright red

    # brush object must be replaced (not the same instance)
    assert widget.bond_brush is not old_brush

    # The gradient inside the new brush must contain colours derived from red.
    # We sample the gradient at the 'light' stop (t=0.2) and check that the
    # red channel dominates over green and blue.
    new_gradient = widget.bond_brush.gradient()
    stops = new_gradient.stops()
    # stops is a list of (position, QColor) tuples
    colors = [c for (_, c) in stops]
    # At least one stop should have a significantly higher red channel
    assert any(c.red() > c.blue() + 20 for c in colors), (
        "After set_bond_color('#ff0000') the bond_brush gradient should "
        "contain reddish colours, but got: " + str([(c.red(), c.green(), c.blue()) for c in colors])
    )


def test_set_bond_color_visible_in_rounded_mode():
    """Rendered pixels must differ between the default grey and a vivid new
    bond colour when round-bond mode is active (the default)."""
    import numpy as np
    from fastmolwidget.sdm import Atomtuple

    # Two atoms close enough to be bonded
    atoms = [
        Atomtuple('C1', 'C', 0.0, 0.0, 0.0, 0),
        Atomtuple('C2', 'C', 1.5, 0.0, 0.0, 0),
    ]

    widget = MoleculeWidget()
    widget.resize(400, 300)
    widget.show()
    widget.open_molecule(atoms)

    # Flush pending paint events so the widget has actually drawn
    app.processEvents()

    # Capture with the default grey bond colour
    pixmap_grey = widget.grab()
    img_grey = pixmap_grey.toImage()

    # Change to a vivid blue and flush again
    widget.set_bond_color(QtGui.QColor("#0000ff"))
    app.processEvents()
    pixmap_blue = widget.grab()
    img_blue = pixmap_blue.toImage()

    # Convert to numpy arrays for easy pixel comparison
    def img_to_array(img):
        import ctypes
        img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        w, h = img.width(), img.height()
        n_bytes = h * w * 4
        ptr = img.bits()
        if isinstance(ptr, (bytes, memoryview)):
            # PySide6: bits() returns a memoryview or bytes directly
            arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape((h, w, 4))
        elif hasattr(ptr, 'setsize'):
            # PyQt5 / early PyQt6: sip.voidptr with setsize()
            ptr.setsize(n_bytes)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
        else:
            # PyQt6: sip.voidptr without setsize(); use ctypes via raw address
            cbuf = (ctypes.c_uint8 * n_bytes).from_address(int(ptr))
            arr = np.frombuffer(cbuf, dtype=np.uint8).reshape((h, w, 4))
        return arr[:, :, :3].copy()

    arr_grey = img_to_array(img_grey)
    arr_blue = img_to_array(img_blue)

    # Sanity: at least some pixels should be non-white (i.e., the bond was drawn)
    assert arr_grey.min() < 255, "No bond pixels rendered in grey mode – widget may not have painted."

    diff = np.abs(arr_grey.astype(int) - arr_blue.astype(int))
    assert diff.max() > 10, (
        "Rendered bond pixels did not change after set_bond_color('#0000ff'); "
        "max pixel diff = " + str(diff.max())
    )


# ------------------------------------------------------------------
# Hover labels (atom name + bond distance)
# ------------------------------------------------------------------

from qtpy import QtCore  # noqa: E402  (used only by the hover tests below)


def _make_two_atom_widget(label1: str = "C1", elem1: str = "C",
                          label2: str = "O1", elem2: str = "O",
                          dx: float = 1.5) -> MoleculeWidget:
    """Build a paint-ready widget with two atoms ``dx`` Å apart on the X axis.

    The widget is resized and shown so that ``draw()`` has populated
    ``screenx`` / ``screeny`` on every atom — a precondition for the 2-D hit
    tests that drive the hover state.
    """
    widget = MoleculeWidget()
    widget.resize(800, 600)
    widget.show()
    widget.open_molecule([
        Atomtuple(label1, elem1, 0.0, 0.0, 0.0, 0),
        Atomtuple(label2, elem2, dx,  0.0, 0.0, 0),
    ])
    app.processEvents()
    widget.grab()  # force a paint pass → screenx/screeny populated
    return widget


def test_hover_atom_sets_hovered_atom_name():
    widget = _make_two_atom_widget()
    ax = widget.atoms[0].screenx
    ay = widget.atoms[0].screeny

    widget._update_hover(ax, ay)
    assert widget.hovered_atom == "C1"
    # When an atom is hovered, no bond hover state must be active.
    assert widget.hovered_bond is None
    assert widget._hovered_bond_distance is None
    assert widget._hover_cursor is None


def test_hover_bond_records_distance_in_angstrom():
    widget = _make_two_atom_widget(dx=1.5)
    a, b = widget.atoms[0], widget.atoms[1]
    mx = (a.screenx + b.screenx) / 2.0
    my = (a.screeny + b.screeny) / 2.0

    widget._update_hover(mx, my)
    assert widget.hovered_bond == ("C1", "O1")
    assert widget._hovered_bond_distance == pytest.approx(1.5, abs=1e-3)
    # Cursor position must be tracked so the rounded label can anchor to it.
    assert widget._hover_cursor is not None
    assert widget._hover_cursor.x() == pytest.approx(mx)
    assert widget._hover_cursor.y() == pytest.approx(my)
    # Atom hover must not be set when the cursor is between atoms.
    assert widget.hovered_atom is None


def test_hover_bond_distance_label_renders_in_paint():
    """The rounded distance label must actually appear in the painted output
    when a bond is hovered, even with ``Show Labels`` off."""
    import numpy as np

    widget = _make_two_atom_widget(dx=1.5)
    widget.show_labels(False)
    a, b = widget.atoms[0], widget.atoms[1]
    mx = (a.screenx + b.screenx) / 2.0
    my = (a.screeny + b.screeny) / 2.0

    def grab_array() -> np.ndarray:
        import ctypes
        app.processEvents()
        img = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
        w, h = img.width(), img.height()
        n_bytes = h * w * 4
        ptr = img.bits()
        if isinstance(ptr, (bytes, memoryview)):
            arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape((h, w, 4))
        elif hasattr(ptr, 'setsize'):
            ptr.setsize(n_bytes)
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
        else:
            cbuf = (ctypes.c_uint8 * n_bytes).from_address(int(ptr))
            arr = np.frombuffer(cbuf, dtype=np.uint8).reshape((h, w, 4))
        return arr[:, :, :3].copy()

    # Baseline: no hover state at all.
    widget.hovered_atom = None
    widget.hovered_bond = None
    widget._hovered_bond_distance = None
    widget._hover_cursor = None
    widget.update()
    arr_off = grab_array()

    # Activate bond hover and grab again.
    widget._update_hover(mx, my)
    assert widget.hovered_bond is not None  # precondition
    widget.update()
    arr_on = grab_array()

    # Only the rounded distance label changes between the two grabs (the
    # molecule itself is identical), so any non-trivial pixel diff over the
    # full image is attributable to the hover label.  We compare the full
    # image to avoid making assumptions about HiDPI scaling factors applied
    # by ``QWidget.grab()``.
    if arr_on.shape != arr_off.shape:
        # Different image sizes ⇒ different code paths; treat as changed.
        return
    diff = np.abs(arr_on.astype(int) - arr_off.astype(int)).sum(axis=2)
    changed = int((diff > 20).sum())
    assert changed > 100, (
        f"Only {changed} pixels changed between hover-off and hover-on grabs; "
        f"the rounded distance label is probably not being drawn."
    )


def test_hover_atom_priority_over_bond_at_atom_center():
    """If the cursor is over the atom centre, atom hover wins over bond hover."""
    widget = _make_two_atom_widget(dx=1.5)
    ax = widget.atoms[0].screenx
    ay = widget.atoms[0].screeny

    widget._update_hover(ax, ay)
    assert widget.hovered_atom == "C1"
    assert widget.hovered_bond is None


def test_hover_excludes_hidden_hydrogens_2d():
    """Hidden hydrogens must never produce a hover label, neither as atoms
    nor as the endpoints of a bond."""
    widget = MoleculeWidget()
    widget.resize(800, 600)
    widget.show()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("H1", "H", 1.0, 0.0, 0.0, 0),
    ])
    widget.show_hydrogens(False)
    app.processEvents()
    widget.grab()

    h_atom = widget.atoms[1]
    widget._update_hover(h_atom.screenx, h_atom.screeny)
    assert widget.hovered_atom is None
    assert widget.hovered_bond is None


def test_hover_shows_hydrogen_atom_label_when_visible():
    """Hydrogens are displayed but never receive an atom-name hover label."""
    widget = MoleculeWidget()
    widget.resize(800, 600)
    widget.show()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("H1", "H", 1.0, 0.0, 0.0, 0),
    ])
    app.processEvents()
    widget.grab()

    h_atom = widget.atoms[1]
    widget._update_hover(h_atom.screenx, h_atom.screeny)
    assert widget.hovered_atom == 'H1'


def test_leave_event_clears_hover_state_2d():
    widget = _make_two_atom_widget(dx=1.5)
    a, b = widget.atoms[0], widget.atoms[1]
    mx = (a.screenx + b.screenx) / 2.0
    my = (a.screeny + b.screeny) / 2.0
    widget._update_hover(mx, my)
    assert widget.hovered_bond is not None

    widget.leaveEvent(QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert widget.hovered_atom is None
    assert widget.hovered_bond is None
    assert widget._hovered_bond_distance is None
    assert widget._hover_cursor is None


def test_hover_excludes_hidden_part_atom_2d():
    """Atoms whose disorder-part is not in *_visible_parts* must produce no hover label."""
    widget = MoleculeWidget()
    widget.resize(800, 600)
    widget.show()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),   # part 0 – always visible
        Atomtuple("C2", "C", 3.0, 0.0, 0.0, 1),   # part 1 – will be hidden
    ])
    app.processEvents()
    widget.grab()  # force paint pass → screenx/screeny populated

    c2 = widget.atoms[1]

    # Baseline: all parts visible → C2 is hovered at its screen position.
    widget.set_visible_parts(None)
    widget._update_hover(c2.screenx, c2.screeny)
    assert widget.hovered_atom == "C2"

    # Hide part 1 → C2 must not produce a hover label.
    widget.set_visible_parts({0})
    widget._update_hover(c2.screenx, c2.screeny)
    assert widget.hovered_atom is None
    assert widget.hovered_bond is None


def test_hover_excludes_hidden_part_bond_2d():
    """Bonds whose endpoints are in a hidden disorder-part must not show a
    distance hover label."""
    widget = MoleculeWidget()
    widget.resize(800, 600)
    widget.show()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 1),   # part 1
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 1),   # part 1
    ])
    app.processEvents()
    widget.grab()

    a, b = widget.atoms[0], widget.atoms[1]
    mx = (a.screenx + b.screenx) / 2.0
    my = (a.screeny + b.screeny) / 2.0

    # Baseline: all parts visible → bond is hovered at its midpoint.
    widget.set_visible_parts(None)
    widget._update_hover(mx, my)
    assert widget.hovered_bond is not None

    # Hide part 1 → bond must not produce a hover distance label.
    widget.set_visible_parts({0})
    widget._update_hover(mx, my)
    assert widget.hovered_bond is None
    assert widget.hovered_atom is None


def test_drag_clears_hover_state_2d():
    """While the user is rotating / panning / zooming the molecule, hover
    labels must be suppressed."""
    widget = _make_two_atom_widget(dx=1.5)
    a, b = widget.atoms[0], widget.atoms[1]
    mx = (a.screenx + b.screenx) / 2.0
    my = (a.screeny + b.screeny) / 2.0
    widget._update_hover(mx, my)
    assert widget.hovered_bond is not None

    widget._clear_hover_state()
    assert widget.hovered_atom is None
    assert widget.hovered_bond is None
    assert widget._hovered_bond_distance is None
    assert widget._hover_cursor is None


# ------------------------------------------------------------------
# align_best_view – 2D renderer
# ------------------------------------------------------------------

def test_align_best_view_is_rotation_matrix_2d():
    """align_best_view() must produce a valid rotation matrix (det≈1, R@Rᵀ≈I)."""
    widget = MoleculeWidget()
    atoms = [
        Atomtuple("C1", "C",  0.0,  0.0, 0.0, 0),
        Atomtuple("C2", "C",  3.0,  0.0, 0.0, 0),
        Atomtuple("C3", "C",  1.5,  2.0, 0.0, 0),
        Atomtuple("C4", "C",  1.5,  1.0, 1.0, 0),
        Atomtuple("O1", "O", -1.0, -1.0, 2.0, 0),
    ]
    widget.open_molecule(atoms)
    widget.align_best_view()

    R = widget.cumulative_R.astype(np.float64)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_align_best_view_planar_atoms_z_is_thin_direction_2d():
    """For atoms in the XY plane the camera Z axis should align with original Z."""
    widget = MoleculeWidget()
    atoms = [
        Atomtuple("C1", "C", -5.0,  0.0, 0.0, 0),
        Atomtuple("C2", "C",  5.0,  0.0, 0.0, 0),
        Atomtuple("C3", "C",  0.0,  5.0, 0.0, 0),
        Atomtuple("C4", "C",  0.0, -5.0, 0.0, 0),
    ]
    widget.open_molecule(atoms)
    widget.align_best_view()

    # Third row of cumulative_R is the camera Z direction;
    # it must be roughly ±[0, 0, 1] because Z was already the thin axis.
    z_camera = widget.cumulative_R[2]
    assert abs(abs(z_camera[2]) - 1.0) < 1e-4


def test_align_best_view_noop_on_empty_2d():
    """align_best_view() must not crash or alter rotation on an empty widget."""
    widget = MoleculeWidget()
    widget.align_best_view()
    np.testing.assert_array_equal(widget.cumulative_R, np.eye(3, dtype=np.float32))


def test_align_best_view_noop_on_single_atom_2d():
    """align_best_view() must leave rotation unchanged with only one atom."""
    widget = MoleculeWidget()
    widget.open_molecule([Atomtuple("C1", "C", 1.0, 2.0, 3.0, 0)])
    widget.align_best_view()
    np.testing.assert_array_equal(widget.cumulative_R, np.eye(3, dtype=np.float32))


def test_align_best_view_hydrogen_filter_2d():
    """When hydrogens are hidden, H atoms must not influence the PCA."""
    widget = MoleculeWidget()
    widget.show_hydrogens(False)

    # Heavy atoms: flat in XY (no Z spread)
    # H atoms: far out in Z – would distort PCA if included
    atoms = [
        Atomtuple("C1", "C", -5.0,  0.0,  0.0, 0),
        Atomtuple("C2", "C",  5.0,  0.0,  0.0, 0),
        Atomtuple("C3", "C",  0.0,  3.0,  0.0, 0),
        Atomtuple("C4", "C",  0.0, -3.0,  0.0, 0),
        Atomtuple("H1", "H",  0.0,  0.0, 20.0, 0),
        Atomtuple("H2", "H",  0.0,  0.0,-20.0, 0),
    ]
    widget.open_molecule(atoms)
    widget.align_best_view()

    # Z camera axis should match original Z (thin for heavy atoms only)
    z_camera = widget.cumulative_R[2]
    assert abs(abs(z_camera[2]) - 1.0) < 1e-4


def test_align_best_view_coords_updated_2d():
    """After align_best_view() the atom coordinates must reflect the new rotation."""
    widget = MoleculeWidget()
    atoms = [
        Atomtuple("C1", "C",  0.0,  0.0, 0.0, 0),
        Atomtuple("C2", "C",  4.0,  0.0, 0.0, 0),
        Atomtuple("C3", "C",  2.0,  3.0, 0.0, 0),
        Atomtuple("C4", "C",  2.0,  1.5, 2.0, 0),
    ]
    widget.open_molecule(atoms)
    coords_before = widget._coords_array.copy()
    widget.align_best_view()
    # Coordinates must have been rotated (i.e. are different now)
    assert not np.allclose(widget._coords_array, coords_before)


# ------------------------------------------------------------------
# set_visible_parts / partsChanged (MoleculeWidget 2D)
# ------------------------------------------------------------------

class TestVisibleParts2D:
    """Tests for the disorder-part filter in MoleculeWidget (2D)."""

    def _make_atoms(self):
        return [
            Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
            Atomtuple("C2", "C", 1.5, 0.0, 0.0, 1),
            Atomtuple("C3", "C", 3.0, 0.0, 0.0, 2),
        ]

    def test_available_parts_after_open_molecule(self):
        widget = MoleculeWidget()
        widget.open_molecule(self._make_atoms())
        assert widget.available_parts == frozenset({0, 1, 2})

    def test_visible_parts_default_is_none(self):
        widget = MoleculeWidget()
        widget.open_molecule(self._make_atoms())
        assert widget._visible_parts is None

    def test_parts_changed_signal_emitted(self):
        widget = MoleculeWidget()
        received: list[frozenset] = []
        widget.partsChanged.connect(received.append)
        widget.open_molecule(self._make_atoms())
        assert len(received) == 1
        assert received[0] == frozenset({0, 1, 2})

    def test_set_visible_parts_stores_value(self):
        widget = MoleculeWidget()
        widget.open_molecule(self._make_atoms())
        widget.set_visible_parts({0, 1})
        assert widget._visible_parts == {0, 1}

    def test_set_visible_parts_none_shows_all(self):
        widget = MoleculeWidget()
        widget.open_molecule(self._make_atoms())
        widget.set_visible_parts({0})
        widget.set_visible_parts(None)
        assert widget._visible_parts is None

    def test_set_visible_parts_empty_hides_all(self):
        widget = MoleculeWidget()
        widget.open_molecule(self._make_atoms())
        widget.set_visible_parts(set())
        assert widget._visible_parts == set()

    def test_parts_reset_on_new_open_molecule(self):
        widget = MoleculeWidget()
        widget.open_molecule(self._make_atoms())
        widget.set_visible_parts({0})
        widget.open_molecule([Atomtuple("N1", "N", 0.0, 0.0, 0.0, 0)])
        assert widget._visible_parts is None
        assert widget.available_parts == frozenset({0})

    def test_single_part_structure_has_no_disorder(self):
        widget = MoleculeWidget()
        widget.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
        assert widget.available_parts == frozenset({0})


# ------------------------------------------------------------------
# Part-filter viewer controls (2D viewer widget)
# ------------------------------------------------------------------

def test_2d_viewer_part_container_hidden_when_no_disorder():
    from fastmolwidget.viewer_widget import MoleculeViewerWidget
    viewer = MoleculeViewerWidget()
    viewer._render_widget.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    assert viewer._part_widget.isHidden()


def test_2d_viewer_part_container_shown_for_disordered_structure():
    from fastmolwidget.viewer_widget import MoleculeViewerWidget
    viewer = MoleculeViewerWidget()
    viewer._render_widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C", 1.5, 0.0, 0.0, 1),
        Atomtuple("C3", "C", 3.0, 0.0, 0.0, 2),
    ])
    assert not viewer._part_widget.isHidden()


def test_2d_viewer_combo_all_parts_checked_by_default():
    from fastmolwidget.viewer_widget import MoleculeViewerWidget
    viewer = MoleculeViewerWidget()
    viewer._render_widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C", 1.5, 0.0, 0.0, 1),
    ])
    assert viewer._part_widget.checked_values() == [0, 1]


def test_2d_viewer_all_checked_passes_none_to_renderer():
    from fastmolwidget.viewer_widget import MoleculeViewerWidget
    viewer = MoleculeViewerWidget()
    viewer._render_widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C", 1.5, 0.0, 0.0, 1),
    ])
    assert viewer._render_widget._visible_parts is None


# ------------------------------------------------------------------
# NPD (non-positive-definite ADP) placeholder cube
# ------------------------------------------------------------------

NPD_CIF = data / 'p21c.cif'


def _rotation_z(angle_deg: float) -> np.ndarray:
    a = np.radians(angle_deg)
    return np.array([
        [np.cos(a), -np.sin(a), 0.0],
        [np.sin(a), np.cos(a), 0.0],
        [0.0, 0.0, 1.0],
    ])


def _grab_rgb(widget) -> np.ndarray:
    """Return the widget's painted content as an (h, w, 3) uint8 array."""
    import ctypes
    app.processEvents()
    img = widget.grab().toImage().convertToFormat(QtGui.QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    n_bytes = h * w * 4
    ptr = img.bits()
    if isinstance(ptr, (bytes, memoryview)):
        arr = np.frombuffer(bytes(ptr), dtype=np.uint8).reshape((h, w, 4))
    elif hasattr(ptr, 'setsize'):
        ptr.setsize(n_bytes)
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
    else:
        cbuf = (ctypes.c_uint8 * n_bytes).from_address(int(ptr))
        arr = np.frombuffer(cbuf, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, :3].copy()


def _npd_widget() -> MoleculeWidget:
    """A widget showing p21c.cif, which contains exactly one NPD atom (Al1)."""
    from fastmolwidget.loader import MoleculeLoader
    widget = MoleculeWidget()
    widget.resize(600, 500)
    MoleculeLoader(widget).load_file(NPD_CIF)
    return widget


def _single_npd_atom_widget() -> MoleculeWidget:
    """A widget showing *only* Al1 of p21c.cif.

    With a single atom the rotation pivot is the atom itself, so its screen
    position is invariant under rotation and any pixel change between two
    views is caused by the placeholder cube alone.
    """
    from fastmolwidget.tools import to_float

    cif = CifReader(NPD_CIF)
    adp = next(dp for dp in cif.displacement_parameters() if dp.label == 'Al1')
    at = next(a for a in cif.atoms_orth if a.label == 'Al1')
    atom = Atomtuple(
        label=at.label, type=at.type, x=at.x, y=at.y, z=at.z, part=at.part,
        adp=(to_float(adp.U11), to_float(adp.U22), to_float(adp.U33),
             to_float(adp.U23), to_float(adp.U13), to_float(adp.U12)),
    )
    widget = MoleculeWidget()
    widget.resize(400, 400)
    widget.show_labels(False)
    widget.open_molecule([atom], cell=cif.cell[:6])
    return widget


def test_p21c_has_exactly_one_npd_atom():
    """Al1 in p21c.cif has U33 = -0.0137 -> its tensor is not positive definite."""
    widget = _npd_widget()
    invalid = [at.name for at in widget.atoms
               if at.u_cart is not None and not at.adp_valid]
    assert invalid == ['Al1']


def test_npd_cube_faces_geometry():
    from fastmolwidget.molecule_painter import npd_cube_faces

    half = 10.0
    faces = npd_cube_faces(np.eye(3), half)
    assert len(faces) == 6

    # Outward normals of an axis-aligned cube are the six unit axes.
    normals = sorted(tuple(np.round(n, 9)) for _p, _z, n in faces)
    expected = sorted(
        tuple(np.round(s * np.eye(3)[i], 9)) for i in range(3) for s in (-1.0, 1.0)
    )
    assert normals == expected

    # Faces come back sorted back-to-front (descending depth: smaller z is
    # nearer the viewer in this renderer).
    depths = [z for _p, z, _n in faces]
    assert depths == sorted(depths, reverse=True)

    # Every corner is a +/-half combination.
    for pts, _z, _n in faces:
        assert pts.shape == (4, 2)
        assert np.allclose(np.abs(pts), half)


def test_npd_cube_faces_follow_the_view_rotation():
    """The cube must be rotated by the view matrix, not frozen in screen space."""
    from fastmolwidget.molecule_painter import npd_cube_faces

    half = 7.5
    R = _rotation_z(37.0) @ np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(0.4), -np.sin(0.4)],
        [0.0, np.sin(0.4), np.cos(0.4)],
    ])

    def corner_set(faces):
        return sorted(tuple(np.round(p, 6)) for pts, _z, _n in faces for p in pts)

    rotated = npd_cube_faces(R, half)
    expected_3d = [
        half * (si * R[:, 0] + sj * R[:, 1] + sk * R[:, 2])
        for si in (-1.0, 1.0) for sj in (-1.0, 1.0) for sk in (-1.0, 1.0)
    ]
    expected = sorted(tuple(np.round(p[:2], 6)) for p in expected_3d for _ in range(3))
    assert corner_set(rotated) == expected

    # ... and the result actually differs from the unrotated cube.
    assert corner_set(rotated) != corner_set(npd_cube_faces(np.eye(3), half))

    # Normals stay orthonormal after rotation.
    for _pts, _z, n in rotated:
        assert float(np.linalg.norm(n)) == pytest.approx(1.0)


def _drag_rotate(widget, dx: float, dy: float) -> None:
    """Rotate the view like a real left-button drag would."""
    from qtpy import QtCore
    widget._lastPos = QtCore.QPointF(100.0, 100.0)
    event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(100.0 + dx, 100.0 + dy),
        QtCore.QPointF(100.0 + dx, 100.0 + dy),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    widget.rotate_molecule(event)


def test_npd_cube_is_repainted_when_the_structure_is_rotated():
    """Regression: the NPD cube used fixed screen offsets and never rotated.

    Only the NPD atom is displayed and it sits on the rotation pivot, so its
    screen position cannot change — every differing pixel comes from the cube.
    """
    widget = _single_npd_atom_widget()
    widget.show()
    assert not widget.atoms[0].adp_valid  # precondition: it really is NPD

    before = _grab_rgb(widget)
    screenx, screeny = widget.atoms[0].screenx, widget.atoms[0].screeny
    _drag_rotate(widget, 47.0, 23.0)
    widget.update()
    after = _grab_rgb(widget)

    assert not np.allclose(widget.cumulative_R, np.eye(3))
    assert widget.atoms[0].screenx == pytest.approx(screenx)
    assert widget.atoms[0].screeny == pytest.approx(screeny)
    if before.shape != after.shape:
        return
    changed = (np.abs(before.astype(int) - after.astype(int)).sum(axis=2) > 20).sum()
    assert changed > 0


def test_npd_cube_orientation_changes_with_the_view():
    """The cube's projected corners must differ between two view rotations."""
    from fastmolwidget.molecule_painter import NPD_CUBE_HALF_FACTOR, npd_cube_faces

    widget = _npd_widget()
    half = widget.atoms_size * NPD_CUBE_HALF_FACTOR

    def corners(w):
        faces = npd_cube_faces(w.cumulative_R, half)
        return sorted(tuple(np.round(p, 6)) for pts, _z, _n in faces for p in pts)

    before = corners(widget)
    _drag_rotate(widget, 47.0, 23.0)
    assert corners(widget) != before


def test_npd_cube_is_drawn_with_adps_switched_off():
    """NPD atoms keep their cube in isotropic mode so they stay recognisable."""
    from fastmolwidget.molecule_painter import NPD_CUBE_BOUND_FACTOR

    widget = _npd_widget()
    widget.show_adps(False)
    widget.show()
    widget.grab()  # force a paint pass

    npd = next(at for at in widget.atoms if at.u_cart is not None and not at.adp_valid)
    bound = widget.atoms_size * NPD_CUBE_BOUND_FACTOR
    # Hit-testing uses the cube's bounding circle, not the isotropic sphere.
    assert widget.is_point_inside_atom(npd, npd.screenx + bound * 0.9, npd.screeny)
    assert not widget.is_point_inside_atom(npd, npd.screenx + bound * 1.2, npd.screeny)


def test_npd_atom_hit_test_does_not_raise():
    """u_iso of an NPD atom can be negative -> sqrt() used to blow up here."""
    widget = _npd_widget()
    widget.show()
    widget.grab()
    npd = next(at for at in widget.atoms if at.u_cart is not None and not at.adp_valid)
    assert widget.is_point_inside_atom(npd, npd.screenx, npd.screeny)
    assert widget.get_spherical_radius(npd) > 0.0


def test_npd_cube_front_face_follows_the_rotation():
    """A 90 deg turn about y must bring a different cube face to the front."""
    from fastmolwidget.molecule_painter import npd_cube_faces

    # Faces come back back-to-front, so the last one faces the viewer
    # (smallest z).  Unrotated, that is the -w face.
    front = npd_cube_faces(np.eye(3), 5.0)[-1]
    assert np.allclose(front[2], [0.0, 0.0, -1.0])

    # After a quarter turn about y the +u face has taken its place.
    Ry = np.array([
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ])
    front = npd_cube_faces(Ry, 5.0)[-1]
    assert np.allclose(front[2], [0.0, 0.0, -1.0])          # still faces us ...
    assert np.allclose(Ry.T @ front[2], [1.0, 0.0, 0.0])    # ... but it is +u now


def test_npd_cube_front_face_colour_changes_with_rotation():
    """Each face is shaded from its own normal, so the pixel at the atom
    centre tells us *which* face is currently on top.  Rotating the structure
    must swap in a differently shaded face."""
    from fastmolwidget.molecule_painter import (
        NPD_CUBE_HALF_FACTOR,
        npd_cube_faces,
        npd_face_shade,
    )

    widget = _single_npd_atom_widget()
    widget.show()
    atom = widget.atoms[0]

    def centre_colour() -> tuple[int, int, int]:
        arr = _grab_rgb(widget)
        h, w = arr.shape[:2]
        # grab() may apply a device-pixel ratio; scale the logical position.
        sx = round(atom.screenx * w / widget.width())
        sy = round(atom.screeny * h / widget.height())
        b, g, r = arr[sy, sx]  # Format_RGB32 is BGRA in memory
        return int(r), int(g), int(b)

    def expected_front_colour() -> tuple[int, int, int]:
        half = widget.atoms_size * NPD_CUBE_HALF_FACTOR
        _pts, _z, normal = npd_cube_faces(widget.cumulative_R, half)[-1]
        shade = npd_face_shade(normal)
        col = atom.color.lighter(max(1, round(shade * 100)))
        return col.red(), col.green(), col.blue()

    widget.grab()  # populate screenx/screeny
    before_expected = expected_front_colour()
    assert centre_colour() == before_expected

    _drag_rotate(widget, 90.0, 40.0)
    widget.update()
    after_expected = expected_front_colour()
    assert centre_colour() == after_expected
    # The face on top really did change.
    assert after_expected != before_expected




# ------------------------------------------------------------------
# Anisotropic hydrogen atoms
# ------------------------------------------------------------------

def _load(path: Path) -> MoleculeWidget:
    """Load *path* into a sized, shown MoleculeWidget."""
    from fastmolwidget.loader import MoleculeLoader

    widget = MoleculeWidget()
    widget.resize(800, 600)
    MoleculeLoader(widget).load_file(path)
    return widget


def _hydrogens(widget: MoleculeWidget) -> list:
    return [a for a in widget.atoms if a.type_ in ('H', 'D')]


def test_anisotropic_hydrogen_keeps_its_tensor():
    """nospera2.cif refines both its hydrogens anisotropically."""
    widget = _load(data / 'nospera2.cif')
    hydrogens = _hydrogens(widget)
    assert len(hydrogens) == 2
    for atom in hydrogens:
        assert atom.u_cart is not None
        assert atom.adp_valid
        assert atom.u_inv is not None


def test_anisotropic_hydrogen_draws_ellipsoid():
    """With ADPs on, an anisotropic H is a real ellipsoid, not a fixed sphere."""
    from fastmolwidget.atoms import HYDROGEN_DISPLAY_RADIUS

    widget = _load(data / 'nospera2.cif')
    widget.show_adps(True)
    atom = _hydrogens(widget)[0]

    assert widget._draws_adp_ellipsoid(atom)

    rx = widget.get_directional_radius(atom, np.array([1.0, 0.0, 0.0]))
    ry = widget.get_directional_radius(atom, np.array([0.0, 1.0, 0.0]))
    rz = widget.get_directional_radius(atom, np.array([0.0, 0.0, 1.0]))

    # Anisotropic: the radius depends on the direction ...
    assert len({round(r, 6) for r in (rx, ry, rz)}) > 1
    # ... and none of them is the fixed hydrogen radius.
    for r in (rx, ry, rz):
        assert r != pytest.approx(HYDROGEN_DISPLAY_RADIUS)


def test_anisotropic_hydrogen_falls_back_to_fixed_radius():
    """With ADPs off, the same H returns to the fixed hydrogen radius."""
    from fastmolwidget.atoms import HYDROGEN_DISPLAY_RADIUS

    widget = _load(data / 'nospera2.cif')
    widget.show_adps(False)
    atom = _hydrogens(widget)[0]

    assert not widget._draws_adp_ellipsoid(atom)
    assert widget.get_spherical_radius(atom) == pytest.approx(HYDROGEN_DISPLAY_RADIUS)
    for direction in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
        assert widget.get_directional_radius(atom, direction) == pytest.approx(
            HYDROGEN_DISPLAY_RADIUS
        )


def test_riding_hydrogen_keeps_fixed_radius_in_both_modes():
    """A hydrogen without an anisotropic tensor never changes size."""
    from fastmolwidget.atoms import HYDROGEN_DISPLAY_RADIUS

    widget = _load(data / '1979688_small.cif')
    atom = next(a for a in _hydrogens(widget) if a.u_cart is None)

    for show in (True, False):
        widget.show_adps(show)
        assert not widget._draws_adp_ellipsoid(atom)
        assert widget.get_spherical_radius(atom) == pytest.approx(HYDROGEN_DISPLAY_RADIUS)
        assert widget.get_directional_radius(
            atom, np.array([1.0, 0.0, 0.0])
        ) == pytest.approx(HYDROGEN_DISPLAY_RADIUS)


def test_anisotropic_hydrogen_paints():
    """The new hydrogen-ellipsoid path must actually render."""
    widget = _load(data / 'nospera2.cif')
    widget.show()
    for show in (True, False):
        widget.show_adps(show)
        assert not widget.grab().isNull()


def test_npd_hydrogen_draws_cube(monkeypatch):
    """A hydrogen with a broken tensor shows the NPD cube, like any element."""
    from fastmolwidget.molecule_painter import NPD_CUBE_BOUND_FACTOR

    widget = _load(data / 'nospera2.cif')
    atom = _hydrogens(widget)[0]
    atom.adp_valid = False

    for show in (True, False):
        widget.show_adps(show)
        assert not widget._draws_adp_ellipsoid(atom)
        # Sized as the cube's bounding circle, not the fixed hydrogen sphere.
        expected = widget.atoms_size * NPD_CUBE_BOUND_FACTOR / widget.scale
        assert widget.get_spherical_radius(atom) == pytest.approx(expected)

        called: list = []
        monkeypatch.setattr(
            type(widget), '_draw_invalid_adp',
            lambda self, at, _sink=called: _sink.append(at), raising=True,
        )
        widget.draw_atom(atom)
        assert called == [atom]
        monkeypatch.undo()
