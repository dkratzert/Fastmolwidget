"""Tests for :class:`~fastmolwidget.molecule_quick.MoleculeQuickItem`.

``MoleculeQuickItem`` shares all rendering logic with
:class:`~fastmolwidget.molecule2D.MoleculeWidget` via
:class:`~fastmolwidget.molecule_painter.MoleculeRendererMixin`.  These tests
mirror :mod:`test_molecule2D` but instantiate the Quick item directly (no QML
engine needed for data-model / toggle / signal tests).

Skipped when Qt Quick is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
except ImportError:
    _HAS_QTQUICK = False

pytestmark = pytest.mark.skipif(not _HAS_QTQUICK, reason="Qt Quick unavailable")


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_construction_defaults():
    item = MoleculeQuickItem()
    assert item.atoms_size == 70.0       # zoom=1.0 → 1.0 * 70
    assert item.fontsize == 13
    assert item.bond_width == 3
    assert item.labels is True
    assert item._show_adps is True
    assert item.show_hydrogens_flag is True


# ------------------------------------------------------------------
# open_molecule / clear
# ------------------------------------------------------------------

def test_open_molecule_atoms():
    item = MoleculeQuickItem()
    atoms = [
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ]
    item.open_molecule(atoms)
    assert len(item.atoms) == 2
    assert item.atoms[0].name == "C1"
    assert item.atoms[1].type_ == "O"


def test_clear():
    item = MoleculeQuickItem()
    item.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    assert len(item.atoms) == 1

    item.clear()
    assert len(item.atoms) == 0


def test_connections_built():
    """Two bonded atoms should produce one connection."""
    item = MoleculeQuickItem()
    item.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C", 1.5, 0.0, 0.0, 0),
    ])
    assert len(item.connections) == 1


def test_no_connections_far_apart():
    item = MoleculeQuickItem()
    item.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C", 10.0, 0.0, 0.0, 0),
    ])
    assert len(item.connections) == 0


# ------------------------------------------------------------------
# Display toggles
# ------------------------------------------------------------------

def test_show_adps_toggle():
    item = MoleculeQuickItem()
    item.show_adps(False)
    assert item._show_adps is False
    item.show_adps(True)
    assert item._show_adps is True


def test_show_labels_toggle():
    item = MoleculeQuickItem()
    item.show_labels(True)
    assert item.labels is True
    item.show_labels(False)
    assert item.labels is False


def test_show_labels_via_set_labels_visible():
    item = MoleculeQuickItem()
    item.set_labels_visible(False)
    assert item.labels is False


def test_show_hydrogens_toggle():
    item = MoleculeQuickItem()
    item.show_hydrogens(False)
    assert item.show_hydrogens_flag is False
    item.show_hydrogens(True)
    assert item.show_hydrogens_flag is True


def test_set_bond_width():
    item = MoleculeQuickItem()
    item.set_bond_width(7)
    assert item.bond_width == 7


def test_set_label_font():
    item = MoleculeQuickItem()
    item.setLabelFont(20)
    assert item.fontsize == 20
    # Must clamp to minimum of 1
    item.setLabelFont(-5)
    assert item.fontsize == 1


def test_set_background_color():
    item = MoleculeQuickItem()
    item.set_background_color(QtGui.QColor(0, 0, 0))
    assert item._bg_color == QtGui.QColor(0, 0, 0)


# ------------------------------------------------------------------
# Bond color control
# ------------------------------------------------------------------

def test_set_bond_color_with_qcolor():
    item = MoleculeQuickItem()
    item.set_bond_color(QtGui.QColor("#6b5d4f"))
    assert item.bond_color == QtGui.QColor("#6b5d4f")


def test_set_bond_color_with_hex_string():
    item = MoleculeQuickItem()
    item.set_bond_color("#5f5348")
    assert item.bond_color == QtGui.QColor("#5f5348")


def test_set_bond_color_with_integer_tuple():
    item = MoleculeQuickItem()
    item.set_bond_color((120, 110, 100))
    expected = QtGui.QColor(120, 110, 100)
    assert item.bond_color == expected


def test_set_bond_color_with_float_tuple():
    item = MoleculeQuickItem()
    item.set_bond_color((0.5, 0.4, 0.3))
    expected = QtGui.QColor(int(0.5 * 255), int(0.4 * 255), int(0.3 * 255))
    assert item.bond_color == expected


def test_set_bond_color_updates_bond_brush():
    """bond_brush must be rebuilt when set_bond_color is called."""
    item = MoleculeQuickItem()
    old_brush = item.bond_brush

    item.set_bond_color(QtGui.QColor("#ff0000"))

    assert item.bond_brush is not old_brush
    new_gradient = item.bond_brush.gradient()
    stops = new_gradient.stops()
    colors = [c for (_, c) in stops]
    assert any(c.red() > c.blue() + 20 for c in colors)


# ------------------------------------------------------------------
# Rotation matrices / align_best_view
# ------------------------------------------------------------------

def test_rotation_matrices():
    item = MoleculeQuickItem()
    item.x_angle = 3.14159 / 2
    item.y_angle = 3.14159 / 2

    rx = item.rotate_x()
    ry = item.rotate_y()
    assert rx.shape == (3, 3)
    assert ry.shape == (3, 3)
    assert isinstance(rx, np.ndarray)


def test_align_best_view_is_rotation_matrix():
    item = MoleculeQuickItem()
    atoms = [
        Atomtuple("C1", "C",  0.0,  0.0, 0.0, 0),
        Atomtuple("C2", "C",  3.0,  0.0, 0.0, 0),
        Atomtuple("C3", "C",  1.5,  2.0, 0.0, 0),
        Atomtuple("C4", "C",  1.5,  1.0, 1.0, 0),
        Atomtuple("O1", "O", -1.0, -1.0, 2.0, 0),
    ]
    item.open_molecule(atoms)
    item.align_best_view()

    R = item.cumulative_R.astype(np.float64)
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-5)
    np.testing.assert_allclose(np.linalg.det(R), 1.0, atol=1e-5)


def test_align_best_view_planar_atoms_z_is_thin_direction():
    item = MoleculeQuickItem()
    atoms = [
        Atomtuple("C1", "C", -5.0,  0.0, 0.0, 0),
        Atomtuple("C2", "C",  5.0,  0.0, 0.0, 0),
        Atomtuple("C3", "C",  0.0,  5.0, 0.0, 0),
        Atomtuple("C4", "C",  0.0, -5.0, 0.0, 0),
    ]
    item.open_molecule(atoms)
    item.align_best_view()

    z_camera = item.cumulative_R[2]
    assert abs(abs(z_camera[2]) - 1.0) < 1e-4


def test_align_best_view_noop_on_empty():
    item = MoleculeQuickItem()
    item.align_best_view()
    np.testing.assert_array_equal(item.cumulative_R, np.eye(3, dtype=np.float32))


def test_align_best_view_noop_on_single_atom():
    item = MoleculeQuickItem()
    item.open_molecule([Atomtuple("C1", "C", 1.0, 2.0, 3.0, 0)])
    item.align_best_view()
    np.testing.assert_array_equal(item.cumulative_R, np.eye(3, dtype=np.float32))


def test_align_best_view_hydrogen_filter():
    """When hydrogens are hidden, H atoms must not influence PCA."""
    item = MoleculeQuickItem()
    item.show_hydrogens(False)

    atoms = [
        Atomtuple("C1", "C", -5.0,  0.0,  0.0, 0),
        Atomtuple("C2", "C",  5.0,  0.0,  0.0, 0),
        Atomtuple("C3", "C",  0.0,  3.0,  0.0, 0),
        Atomtuple("C4", "C",  0.0, -3.0,  0.0, 0),
        Atomtuple("H1", "H",  0.0,  0.0, 20.0, 0),
        Atomtuple("H2", "H",  0.0,  0.0,-20.0, 0),
    ]
    item.open_molecule(atoms)
    item.align_best_view()

    z_camera = item.cumulative_R[2]
    assert abs(abs(z_camera[2]) - 1.0) < 1e-4


def test_align_best_view_coords_updated():
    item = MoleculeQuickItem()
    atoms = [
        Atomtuple("C1", "C",  0.0,  0.0, 0.0, 0),
        Atomtuple("C2", "C",  4.0,  0.0, 0.0, 0),
        Atomtuple("C3", "C",  2.0,  3.0, 0.0, 0),
        Atomtuple("C4", "C",  2.0,  1.5, 2.0, 0),
    ]
    item.open_molecule(atoms)
    coords_before = item._coords_array.copy()
    item.align_best_view()
    assert not np.allclose(item._coords_array, coords_before)


# ------------------------------------------------------------------
# set_visible_parts / partsChanged
# ------------------------------------------------------------------

class TestVisiblePartsQuick:
    """Tests for the disorder-part filter in MoleculeQuickItem."""

    def _make_atoms(self):
        return [
            Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
            Atomtuple("C2", "C", 1.5, 0.0, 0.0, 1),
            Atomtuple("C3", "C", 3.0, 0.0, 0.0, 2),
        ]

    def test_available_parts_after_open_molecule(self):
        item = MoleculeQuickItem()
        item.open_molecule(self._make_atoms())
        assert item.available_parts == frozenset({0, 1, 2})

    def test_visible_parts_default_is_none(self):
        item = MoleculeQuickItem()
        item.open_molecule(self._make_atoms())
        assert item._visible_parts is None

    def test_parts_changed_signal_emitted(self):
        item = MoleculeQuickItem()
        received: list[frozenset] = []
        item.partsChanged.connect(received.append)
        item.open_molecule(self._make_atoms())
        assert len(received) == 1
        assert received[0] == frozenset({0, 1, 2})

    def test_set_visible_parts_stores_value(self):
        item = MoleculeQuickItem()
        item.open_molecule(self._make_atoms())
        item.set_visible_parts({0, 1})
        assert item._visible_parts == {0, 1}

    def test_set_visible_parts_none_shows_all(self):
        item = MoleculeQuickItem()
        item.open_molecule(self._make_atoms())
        item.set_visible_parts({0})
        item.set_visible_parts(None)
        assert item._visible_parts is None

    def test_set_visible_parts_empty_hides_all(self):
        item = MoleculeQuickItem()
        item.open_molecule(self._make_atoms())
        item.set_visible_parts(set())
        assert item._visible_parts == set()

    def test_parts_reset_on_new_open_molecule(self):
        item = MoleculeQuickItem()
        item.open_molecule(self._make_atoms())
        item.set_visible_parts({0})
        item.open_molecule([Atomtuple("N1", "N", 0.0, 0.0, 0.0, 0)])
        assert item._visible_parts is None
        assert item.available_parts == frozenset({0})

    def test_single_part_structure_has_no_disorder(self):
        item = MoleculeQuickItem()
        item.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
        assert item.available_parts == frozenset({0})


# ------------------------------------------------------------------
# CIF loading (via MoleculeLoader)
# ------------------------------------------------------------------

def test_load_cif_via_loader():
    from fastmolwidget.loader import MoleculeLoader

    item = MoleculeQuickItem()
    loader = MoleculeLoader(item)
    loader.load_file(data / '1979688_small.cif')
    assert len(item.atoms) == 94


def test_load_xyz_via_loader():
    from fastmolwidget.loader import MoleculeLoader

    item = MoleculeQuickItem()
    loader = MoleculeLoader(item)
    loader.load_file(data / 'test_molecule.xyz')
    assert len(item.atoms) == 5


def test_load_shelx_via_loader():
    from fastmolwidget.loader import MoleculeLoader

    item = MoleculeQuickItem()
    loader = MoleculeLoader(item)
    loader.load_file(data / 'test_molecule.res')
    assert len(item.atoms) == 5


# ------------------------------------------------------------------
# Image export
# ------------------------------------------------------------------

def test_save_image(tmp_path):
    """save_image must produce a non-empty PNG file."""
    item = MoleculeQuickItem()
    # Give the item a size so save_image doesn't bail out.
    item.setWidth(400)
    item.setHeight(300)
    item.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ])

    path = tmp_path / "test_quick.png"
    item.save_image(path, image_scale=1.0)
    assert path.exists()
    assert path.stat().st_size > 0
