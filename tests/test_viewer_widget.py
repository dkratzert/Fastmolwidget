"""Tests for :class:`~fastmolwidget.viewer_widget.MoleculeViewerWidget`."""

from pathlib import Path

import pytest
from qtpy import QtWidgets

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
