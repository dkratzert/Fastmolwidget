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

