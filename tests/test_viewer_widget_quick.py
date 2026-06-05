"""Tests for :class:`~fastmolwidget.viewer_widget_quick.MoleculeViewerQuickWidget`
and :class:`~fastmolwidget.viewer_widget_quick.MoleculeViewerBackend`.

The Quick viewer embeds a ``QQuickWidget`` whose QML scene creates the
``MoleculeQuickItem`` render item.  Because ``Component.onCompleted`` fires
asynchronously, the backend is also tested in isolation by manually calling
``registerRenderItem`` with a standalone ``MoleculeQuickItem``.

Skipped when Qt Quick is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from fastmolwidget.sdm import Atomtuple

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

data = Path('tests/test-data')

# Skip the entire module when Qt Quick is not installed.
try:
    from fastmolwidget.molecule_quick import MoleculeQuickItem, _HAS_QTQUICK
    from fastmolwidget.viewer_widget_quick import (
        MoleculeViewerBackend,
        MoleculeViewerQuickWidget,
    )
except ImportError:
    _HAS_QTQUICK = False

pytestmark = pytest.mark.skipif(not _HAS_QTQUICK, reason="Qt Quick unavailable")


# ------------------------------------------------------------------
# Helper: backend with a manually registered render item
# ------------------------------------------------------------------

def _make_backend() -> tuple[MoleculeViewerBackend, MoleculeQuickItem]:
    """Create a backend + render item pair without a QML engine."""
    backend = MoleculeViewerBackend()
    item = MoleculeQuickItem()
    backend.registerRenderItem(item)
    return backend, item


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_construction_no_args():
    w = MoleculeViewerQuickWidget()
    assert w is not None
    assert w._backend is not None


def test_backend_construction():
    backend = MoleculeViewerBackend()
    assert backend._render_item is None
    assert backend._loader is None
    assert backend._grow_active is False
    assert backend._pack_active is False
    assert backend._show_adps is True
    assert backend._show_labels is False
    assert backend._hide_hydrogens is False


def test_register_render_item():
    backend, item = _make_backend()
    assert backend._render_item is item
    assert backend._loader is not None


# ------------------------------------------------------------------
# File I/O through backend
# ------------------------------------------------------------------

def test_load_cif():
    backend, item = _make_backend()
    backend.load_file(data / '1979688_small.cif')
    assert len(item.atoms) == 94


def test_load_xyz():
    backend, item = _make_backend()
    backend.load_file(data / 'test_molecule.xyz')
    assert len(item.atoms) == 5


def test_load_shelx():
    backend, item = _make_backend()
    backend.load_file(data / 'test_molecule.res')
    assert len(item.atoms) == 5


def test_load_unsupported_format():
    backend, item = _make_backend()
    with pytest.raises(ValueError, match='Unsupported file format'):
        backend.load_file(data / 'fake.pdb')


def test_load_missing_file():
    backend, item = _make_backend()
    with pytest.raises(FileNotFoundError):
        backend.load_file('nonexistent.cif')


# ------------------------------------------------------------------
# Display toggle slots
# ------------------------------------------------------------------

def test_setShowAdps():
    backend, item = _make_backend()
    backend.setShowAdps(False)
    assert backend._show_adps is False
    assert item._show_adps is False
    backend.setShowAdps(True)
    assert item._show_adps is True


def test_setShowLabels():
    backend, item = _make_backend()
    backend.setShowLabels(True)
    assert backend._show_labels is True
    assert item.labels is True
    backend.setShowLabels(False)
    assert item.labels is False


def test_setHideHydrogens():
    backend, item = _make_backend()
    backend.setHideHydrogens(True)
    assert backend._hide_hydrogens is True
    assert item.show_hydrogens_flag is False
    backend.setHideHydrogens(False)
    assert item.show_hydrogens_flag is True


def test_setBondWidth():
    backend, item = _make_backend()
    backend.setBondWidth(7)
    assert item.bond_width == 7


# ------------------------------------------------------------------
# Grow / Pack mutual exclusion
# ------------------------------------------------------------------

def test_setGrow():
    backend, item = _make_backend()
    backend.setGrow(True)
    assert backend._grow_active is True


def test_setPack():
    backend, item = _make_backend()
    backend.setPack(True)
    assert backend._pack_active is True


def test_grow_deactivates_pack():
    backend, item = _make_backend()
    backend.setPack(True)
    assert backend._pack_active is True

    backend.setGrow(True)
    assert backend._grow_active is True
    assert backend._pack_active is False


def test_pack_deactivates_grow():
    backend, item = _make_backend()
    backend.setGrow(True)
    assert backend._grow_active is True

    backend.setPack(True)
    assert backend._pack_active is True
    assert backend._grow_active is False


# ------------------------------------------------------------------
# Bond color
# ------------------------------------------------------------------

def test_set_bond_color_with_qcolor():
    backend, item = _make_backend()
    backend.set_bond_color(QtGui.QColor("#6b5d4f"))
    assert item.bond_color == QtGui.QColor("#6b5d4f")


def test_set_bond_color_with_hex_string():
    backend, item = _make_backend()
    backend.set_bond_color("#5f5348")
    assert item.bond_color == QtGui.QColor("#5f5348")


def test_set_bond_color_with_integer_tuple():
    backend, item = _make_backend()
    backend.set_bond_color((120, 110, 100))
    expected = QtGui.QColor(120, 110, 100)
    assert item.bond_color == expected


def test_set_bond_color_with_float_tuple():
    backend, item = _make_backend()
    backend.set_bond_color((0.5, 0.4, 0.3))
    expected = QtGui.QColor(int(0.5 * 255), int(0.4 * 255), int(0.3 * 255))
    assert item.bond_color == expected


# ------------------------------------------------------------------
# View slots
# ------------------------------------------------------------------

def test_resetCenter():
    backend, item = _make_backend()
    item.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 2.0, 0.0, 0.0, 0),
    ])
    # Should not raise
    backend.resetCenter()


def test_bestView():
    backend, item = _make_backend()
    item.open_molecule([
        Atomtuple("C1", "C",  0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C",  3.0, 0.0, 0.0, 0),
        Atomtuple("C3", "C",  1.5, 2.0, 0.0, 0),
    ])
    backend.bestView()
    # After bestView, cumulative_R may differ from identity (for non-planar)
    # but at minimum it should still be a valid rotation matrix.
    import numpy as np
    R = item.cumulative_R.astype(np.float64)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)


# ------------------------------------------------------------------
# Parts filter (backend)
# ------------------------------------------------------------------

def _disordered_atoms():
    return [
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C", 1.5, 0.0, 0.0, 1),
        Atomtuple("C3", "C", 3.0, 0.0, 0.0, 2),
    ]


def test_parts_model_populated_on_disorder():
    backend, item = _make_backend()
    item.open_molecule(_disordered_atoms())
    assert backend._parts_model == [0, 1, 2]
    assert backend.hasParts is True


def test_parts_model_empty_on_single_part():
    backend, item = _make_backend()
    item.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    assert backend._parts_model == []
    assert backend.hasParts is False


def test_togglePart_hides_part():
    backend, item = _make_backend()
    item.open_molecule(_disordered_atoms())
    # Untick part 2
    backend.togglePart(2, False)
    assert item._visible_parts is not None
    assert 2 not in item._visible_parts


def test_togglePart_recheck_shows_all():
    backend, item = _make_backend()
    item.open_molecule(_disordered_atoms())
    backend.togglePart(2, False)
    assert item._visible_parts is not None
    # Re-check part 2
    backend.togglePart(2, True)
    # All checked → renderer receives None
    assert item._visible_parts is None


def test_all_parts_checked_passes_none_to_renderer():
    backend, item = _make_backend()
    item.open_molecule(_disordered_atoms())
    # All parts should be ticked by default
    assert item._visible_parts is None


def test_parts_reset_on_new_load():
    backend, item = _make_backend()
    item.open_molecule(_disordered_atoms())
    backend.togglePart(2, False)
    assert item._visible_parts is not None
    # Loading a new molecule resets parts
    backend.load_file(data / 'test_molecule.xyz')
    assert backend._manual_parts is None
    assert item._visible_parts is None


def test_parts_model_hidden_after_single_part_reload():
    backend, item = _make_backend()
    item.open_molecule(_disordered_atoms())
    assert backend.hasParts is True
    item.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    assert backend.hasParts is False


# ------------------------------------------------------------------
# Notify signals
# ------------------------------------------------------------------

def test_growActiveChanged_signal():
    backend, _ = _make_backend()
    received = []
    backend.growActiveChanged.connect(received.append)
    backend.setGrow(True)
    assert received == [True]


def test_packActiveChanged_signal():
    backend, _ = _make_backend()
    received = []
    backend.packActiveChanged.connect(received.append)
    backend.setPack(True)
    assert received == [True]


def test_showAdpsChanged_signal():
    backend, _ = _make_backend()
    received = []
    backend.showAdpsChanged.connect(received.append)
    backend.setShowAdps(False)
    assert received == [False]


def test_showLabelsChanged_signal():
    backend, _ = _make_backend()
    received = []
    backend.showLabelsChanged.connect(received.append)
    backend.setShowLabels(True)
    assert received == [True]


def test_hideHydrogensChanged_signal():
    backend, _ = _make_backend()
    received = []
    backend.hideHydrogensChanged.connect(received.append)
    backend.setHideHydrogens(True)
    assert received == [True]


def test_partsModelChanged_signal():
    backend, item = _make_backend()
    received = []
    backend.partsModelChanged.connect(received.append)
    item.open_molecule(_disordered_atoms())
    assert len(received) == 1
    assert received[0] == [0, 1, 2]


def test_hasPartsChanged_signal():
    backend, item = _make_backend()
    received = []
    backend.hasPartsChanged.connect(received.append)
    # First load with disorder → hasParts changes from False to True
    item.open_molecule(_disordered_atoms())
    assert True in received


# ------------------------------------------------------------------
# Viewer wrapper public API
# ------------------------------------------------------------------

def test_viewer_set_bond_color():
    w = MoleculeViewerQuickWidget()
    # Without a render item (QML hasn't fired yet), this should not crash
    w.set_bond_color("#ff0000")


def test_viewer_load_file_without_qml():
    """load_file before QML Component.onCompleted should not crash.

    The backend stores the loader as None until registerRenderItem is called,
    so an early load_file is a no-op rather than a crash.
    """
    w = MoleculeViewerQuickWidget()
    # render_widget is None until QML initialises
    # load_file should gracefully no-op
    w.load_file(data / 'test_molecule.xyz')
