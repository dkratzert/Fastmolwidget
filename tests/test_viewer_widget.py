"""Tests for :class:`~fastmolwidget.viewer_widget.MoleculeViewerWidget`."""

from pathlib import Path

import pytest
from qtpy import QtGui, QtWidgets

from fastmolwidget.viewer_widget import MoleculeViewerWidget

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

data = Path('tests/test-data')


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_construction_no_args():
    w = MoleculeViewerWidget()
    assert w is not None
    assert w.render_widget is not None


# ------------------------------------------------------------------
# load_file
# ------------------------------------------------------------------

def test_load_cif():
    w = MoleculeViewerWidget()
    w.load_file(data / '1979688_small.cif')
    assert len(w.render_widget.atoms) == 94


def test_load_xyz():
    w = MoleculeViewerWidget()
    w.load_file(data / 'test_molecule.xyz')
    assert len(w.render_widget.atoms) == 5


def test_load_shelx():
    w = MoleculeViewerWidget()
    w.load_file(data / 'test_molecule.res')
    assert len(w.render_widget.atoms) == 5


def test_load_unsupported_format():
    w = MoleculeViewerWidget()
    with pytest.raises(ValueError, match='Unsupported file format'):
        w.load_file(data / 'fake.pdb')


def test_load_missing_file():
    w = MoleculeViewerWidget()
    with pytest.raises(FileNotFoundError):
        w.load_file('nonexistent.cif')


# ------------------------------------------------------------------
# Rendering sanity check
# ------------------------------------------------------------------

def test_viewer_renders():
    w = MoleculeViewerWidget()
    w.load_file(data / '1979688_small.cif')
    # app.processEvents()
    assert w.render_widget.width() > 0
    assert w.render_widget.height() > 0
    pixmap = w.grab()
    assert not pixmap.isNull()


# ------------------------------------------------------------------
# Bond color control
# ------------------------------------------------------------------

def test_set_bond_color_with_qcolor():
    """Test set_bond_color with QColor input."""
    widget = MoleculeViewerWidget()
    widget.set_bond_color(QtGui.QColor("#6b5d4f"))
    assert widget.render_widget.bond_color == QtGui.QColor("#6b5d4f")


def test_set_bond_color_with_hex_string():
    """Test set_bond_color with hex string input."""
    widget = MoleculeViewerWidget()
    widget.set_bond_color("#5f5348")
    assert widget.render_widget.bond_color == QtGui.QColor("#5f5348")


def test_set_bond_color_with_integer_tuple():
    """Test set_bond_color with integer RGB tuple (0..255)."""
    widget = MoleculeViewerWidget()
    widget.set_bond_color((120, 110, 100))
    expected = QtGui.QColor(120, 110, 100)
    assert widget.render_widget.bond_color == expected


def test_set_bond_color_with_float_tuple():
    """Test set_bond_color with float RGB tuple (0..1)."""
    widget = MoleculeViewerWidget()
    widget.set_bond_color((0.5, 0.4, 0.3))
    expected = QtGui.QColor(int(0.5 * 255), int(0.4 * 255), int(0.3 * 255))
    assert widget.render_widget.bond_color == expected


# ------------------------------------------------------------------
# Save-image button
# ------------------------------------------------------------------

def test_save_image_button_exists():
    """MoleculeViewerWidget must have a Save Image button."""
    w = MoleculeViewerWidget()
    assert hasattr(w, '_save_image_button')
    assert isinstance(w._save_image_button, QtWidgets.QPushButton)


def test_save_image_preserves_labels_off(tmp_path, monkeypatch):
    """Labels must remain off after saving when they were off before."""
    w = MoleculeViewerWidget()
    w.load_file(data / 'test_molecule.xyz')
    w.render_widget.show_labels(False)

    labels_during: list[bool] = []

    def _mock_save(path, **kwargs):
        labels_during.append(w.render_widget.labels)

    monkeypatch.setattr(w.render_widget, 'save_image', _mock_save)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **kw: (str(tmp_path / 'out.png'), 'PNG Image (*.png)')),
    )

    w._save_image_button.click()

    assert labels_during == [False], "Labels must stay False during save_image()"
    assert w.render_widget.labels is False, "Labels must still be False after save"


def test_save_image_preserves_labels_on(tmp_path, monkeypatch):
    """Labels must remain on after saving when they were on before."""
    w = MoleculeViewerWidget()
    w.load_file(data / 'test_molecule.xyz')
    w.render_widget.show_labels(True)

    labels_during: list[bool] = []

    def _mock_save(path, **kwargs):
        labels_during.append(w.render_widget.labels)

    monkeypatch.setattr(w.render_widget, 'save_image', _mock_save)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **kw: (str(tmp_path / 'out.png'), 'PNG Image (*.png)')),
    )

    w._save_image_button.click()

    assert labels_during == [True], "Labels must stay True during save_image()"
    assert w.render_widget.labels is True, "Labels must still be True after save"


def test_save_image_cancelled_does_not_call_save(monkeypatch):
    """Cancelling the file dialog must not trigger save_image at all."""
    w = MoleculeViewerWidget()
    called = []

    monkeypatch.setattr(w.render_widget, 'save_image', lambda *a, **kw: called.append(1))
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **kw: ('', '')),  # user cancelled
    )

    w._save_image_button.click()

    assert called == [], "save_image must not be called when dialog is cancelled"


def test_save_image_2d_labels_appear_in_file(tmp_path):
    """Labels must be visible in the saved PNG when labels are enabled (2D).

    Uses :class:`MoleculeWidget` directly (no viewer wrapper) so the widget
    widget size stays stable between the two saves.  Saves the same scene
    twice (labels off / on), then counts pixels that differ between the two
    images.  Text glyphs produce a visible difference that must exceed a
    small noise floor.
    """
    import numpy as np
    from PIL import Image
    from fastmolwidget.molecule2D import MoleculeWidget
    from fastmolwidget.loader import MoleculeLoader

    widget = MoleculeWidget()
    widget.resize(400, 300)
    loader = MoleculeLoader(widget)
    loader.load_file(data / 'test_molecule.xyz')

    path_off = tmp_path / 'labels_off_2d.png'
    path_on  = tmp_path / 'labels_on_2d.png'

    widget.show_labels(False)
    widget.save_image(path_off, image_scale=1.0)

    widget.show_labels(True)
    widget.save_image(path_on,  image_scale=1.0)

    arr_off = np.array(Image.open(path_off).convert('RGB'))
    arr_on  = np.array(Image.open(path_on).convert('RGB'))

    assert arr_off.shape == arr_on.shape, (
        f"Image sizes differ: {arr_off.shape} vs {arr_on.shape}"
    )

    diff = np.abs(arr_on.astype(int) - arr_off.astype(int)).sum(axis=2)
    n_changed = int((diff > 5).sum())

    assert n_changed > 50, (
        f"Expected significant pixel differences when labels are ON "
        f"(only {n_changed} pixels changed — labels may not be rendered)."
    )


