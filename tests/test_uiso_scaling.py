"""Isotropic atoms are scaled by their U value, not by the element radius.

An atom that was refined isotropically carries ``Atomtuple.u_iso`` (Å²) and is
drawn as a sphere of radius ``1.5382 · √U_iso`` — the same 50 % probability
factor the ADP ellipsoids use. Atoms without a usable U keep the fixed element
display radius.
"""

from __future__ import annotations

from math import sqrt
from pathlib import Path

import numpy as np
import pytest
from qtpy import QtWidgets

from fastmolwidget.atoms import display_radius_for_element
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.molecule3D import _ADP_SCALE, MoleculeWidget3D
from fastmolwidget.sdm import Atomtuple
from fastmolwidget.tools import display_u_iso

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

data = Path('tests/test-data')

CELL = (10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
U_ISO = 0.04
DIRECTION = np.array([1.0, 0.0, 0.0])


def _widget2d(atoms: list[Atomtuple]) -> MoleculeWidget:
    widget = MoleculeWidget()
    widget.open_molecule(atoms=atoms, cell=CELL)
    return widget


# ------------------------------------------------------------------
# Atomtuple
# ------------------------------------------------------------------

def test_atomtuple_u_iso_defaults_to_none():
    atom = Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0)
    assert atom.u_iso is None
    assert atom.adp is None


# ------------------------------------------------------------------
# display_u_iso filter
# ------------------------------------------------------------------

@pytest.mark.parametrize(('element', 'u_iso', 'expected'), [
    ('C', 0.04, 0.04),
    ('C', None, None),
    # Hydrogen keeps its fixed small sphere.
    ('H', 0.05, None),
    ('D', 0.05, None),
    # An unresolved SHELX riding code or free variable is not a U value.
    ('C', -1.5, None),
    ('C', 0.0, None),
    ('C', 10.05, None),
    ('C', 'x', None),
])
def test_display_u_iso(element, u_iso, expected):
    assert display_u_iso(element, u_iso) == expected


# ------------------------------------------------------------------
# 2-D / Qt Quick renderer
# ------------------------------------------------------------------

def test_2d_isotropic_atom_is_scaled_by_u_iso():
    widget = _widget2d([
        Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0, u_iso=U_ISO),
    ])
    atom = widget.atoms[0]
    assert atom.u_cart is None
    assert atom.u_iso == pytest.approx(U_ISO)
    expected = sqrt(U_ISO) * widget.adp_scale
    assert widget.get_directional_radius(atom, DIRECTION) == pytest.approx(expected)
    assert widget.get_spherical_radius(atom) == pytest.approx(expected)


def test_2d_atom_without_u_iso_uses_the_display_radius():
    widget = _widget2d([
        Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0),
    ])
    atom = widget.atoms[0]
    assert atom.u_iso is None
    assert widget.get_directional_radius(atom, DIRECTION) == atom.display_radius
    assert widget.get_spherical_radius(atom) == atom.display_radius


def test_2d_hydrogen_keeps_its_fixed_sphere():
    widget = _widget2d([
        Atomtuple(label='H1', type='H', x=0.0, y=0.0, z=0.0, part=0, u_iso=0.06),
    ])
    atom = widget.atoms[0]
    assert widget.get_directional_radius(atom, DIRECTION) == atom.display_radius
    assert widget.get_spherical_radius(atom) == atom.display_radius


def test_2d_u_iso_is_ignored_while_adps_are_off():
    widget = _widget2d([
        Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0, u_iso=U_ISO),
    ])
    widget.show_adps(False)
    atom = widget.atoms[0]
    assert widget.get_directional_radius(atom, DIRECTION) == atom.display_radius


def test_2d_anisotropic_adp_wins_over_u_iso():
    """``u_iso`` is only a fallback; a real tensor still defines the ellipsoid."""
    widget = _widget2d([
        Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0,
                  adp=(0.02, 0.02, 0.02, 0.0, 0.0, 0.0), u_iso=1.0),
    ])
    atom = widget.atoms[0]
    assert atom.u_cart is not None
    assert atom.u_iso == pytest.approx(0.02)


# ------------------------------------------------------------------
# set_isotropic_u_scaling
# ------------------------------------------------------------------

def test_2d_isotropic_u_scaling_can_be_switched_off():
    widget = _widget2d([
        Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0, u_iso=U_ISO),
    ])
    atom = widget.atoms[0]
    assert widget._scale_isotropic_u is True

    widget.set_isotropic_u_scaling(False)
    assert widget.get_directional_radius(atom, DIRECTION) == atom.display_radius
    assert widget.get_spherical_radius(atom) == atom.display_radius

    widget.set_isotropic_u_scaling(True)
    expected = sqrt(U_ISO) * widget.adp_scale
    assert widget.get_directional_radius(atom, DIRECTION) == pytest.approx(expected)


def test_2d_isotropic_u_scaling_leaves_anisotropic_atoms_alone():
    widget = _widget2d([
        Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0,
                  adp=(0.02, 0.02, 0.02, 0.0, 0.0, 0.0)),
    ])
    atom = widget.atoms[0]
    before = widget.get_directional_radius(atom, DIRECTION)
    widget.set_isotropic_u_scaling(False)
    assert widget.get_directional_radius(atom, DIRECTION) == pytest.approx(before)
    assert before != atom.display_radius


def test_3d_isotropic_u_scaling_can_be_switched_off():
    widget = MoleculeWidget3D()
    widget.open_molecule(
        atoms=[
            Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0, u_iso=U_ISO),
            Atomtuple(label='C2', type='C', x=2.0, y=0.0, z=0.0, part=0,
                      adp=(0.02, 0.02, 0.02, 0.0, 0.0, 0.0)),
        ],
        cell=CELL,
    )
    isotropic, anisotropic = widget.atoms
    assert widget._scale_isotropic_u is True
    assert widget._sphere_radius(isotropic) == pytest.approx(sqrt(U_ISO) * _ADP_SCALE)

    widget.set_isotropic_u_scaling(False)
    assert widget._sphere_radius(isotropic) == isotropic.display_radius
    # The ellipsoid atom keeps its U-based size.
    assert widget._sphere_radius(anisotropic) == pytest.approx(
        sqrt(anisotropic.u_iso) * _ADP_SCALE)


def test_isotropic_u_scaling_is_part_of_the_protocol():
    from fastmolwidget.molecule_base import MoleculeWidgetProtocol

    assert hasattr(MoleculeWidgetProtocol, 'set_isotropic_u_scaling')
    for widget in (MoleculeWidget(), MoleculeWidget3D()):
        assert isinstance(widget, MoleculeWidgetProtocol)


# ------------------------------------------------------------------
# 3-D renderer
# ------------------------------------------------------------------

def test_3d_isotropic_atom_is_scaled_by_u_iso():
    widget = MoleculeWidget3D()
    widget.open_molecule(
        atoms=[
            Atomtuple(label='C1', type='C', x=0.0, y=0.0, z=0.0, part=0, u_iso=U_ISO),
            Atomtuple(label='C2', type='C', x=1.5, y=0.0, z=0.0, part=0),
            Atomtuple(label='H1', type='H', x=0.0, y=1.0, z=0.0, part=0, u_iso=0.06),
        ],
        cell=CELL,
    )
    scaled, plain, hydrogen = widget.atoms
    assert scaled.u_iso == pytest.approx(U_ISO)
    assert sqrt(scaled.u_iso) * _ADP_SCALE == pytest.approx(sqrt(U_ISO) * 1.5382)
    # No U value and hydrogens both keep the element display radius.
    assert plain.u_iso is None
    assert hydrogen.u_iso is None
    assert hydrogen.display_radius == display_radius_for_element('H')


# ------------------------------------------------------------------
# Loading real files
# ------------------------------------------------------------------

def test_cif_isotropic_atoms_carry_u_iso():
    widget = MoleculeWidget()
    MoleculeLoader(widget).load_file(data / 'p21c.cif')
    isotropic = [a for a in widget.atoms if a.u_cart is None and a.type_ not in ('H', 'D')]
    assert isotropic, 'p21c.cif should contain isotropically refined non-H atoms'
    assert all(a.u_iso is not None and a.u_iso > 0 for a in isotropic)
    # Hydrogens keep their fixed sphere.
    assert all(a.u_iso is None for a in widget.atoms if a.type_ == 'H')


def test_u_iso_survives_grow_and_pack():
    widget = MoleculeWidget()
    loader = MoleculeLoader(widget)
    loader.load_file(data / 'p21c.cif')

    def u_iso_of(name: str) -> float | None:
        for atom in widget.atoms:
            if atom.name.split('>>')[0] == name:
                return atom.u_iso
        return None

    plain = u_iso_of('Ga1')
    assert plain is not None
    loader.set_grow(True)
    assert u_iso_of('Ga1') == pytest.approx(plain)
    loader.set_grow(False)
    loader.set_pack(True)
    assert u_iso_of('Ga1') == pytest.approx(plain)


def test_shelx_riding_u_values_are_resolved():
    """A riding U reaches the widget resolved, never as its ``-T`` code."""
    widget = MoleculeWidget()
    MoleculeLoader(widget).load_file(data / '1548072_many_atoms.res')
    values = [a.u_iso for a in widget.atoms if a.u_iso is not None]
    assert values
    assert all(0.0 < value < 5.0 for value in values)


def test_web_export_contains_u_iso():
    from fastmolwidget.web_export import export_cif

    exported = export_cif(data / 'p21c.cif')
    by_label = {atom['label']: atom for atom in exported['atoms']}
    assert any(atom['u_iso'] for atom in exported['atoms'])
    assert all(atom['u_iso'] is None
               for atom in exported['atoms'] if atom['type'] in ('H', 'D'))
    assert by_label['Ga1']['u_iso'] == pytest.approx(0.02486, abs=1e-5)
