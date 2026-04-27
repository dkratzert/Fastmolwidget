from pathlib import Path

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
    assert widget.atoms_size == 12
    assert widget.fontsize == 13
    assert widget.bond_width == 3
    assert widget.labels is True
    assert widget._show_adps is True


def test_adp_intersection_line_width_scales_with_zoom():
    widget = MoleculeWidget()
    widget._factor = 0.2
    thin = widget._adp_intersection_line_width()
    widget._factor = 1.0
    thick = widget._adp_intersection_line_width()
    assert thick > thin


def test_molecule_widget_with_cif():
    cif = CifReader(data / '1979688_small.cif')
    adp_dict = {dp.label: (dp.U11, dp.U22, dp.U33, dp.U23, dp.U13, dp.U12) for dp in cif.displacement_parameters()}

    widget = MoleculeWidget()
    widget.resize(800, 600)
    widget.open_molecule(list(cif.atoms_orth), cif.cell[:6], adp_dict)
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
    widget.atoms_size = 15
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
        img = img.convertToFormat(QtGui.QImage.Format.Format_RGB32)
        w, h = img.width(), img.height()
        ptr = img.bits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
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


