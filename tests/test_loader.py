"""Tests for :mod:`fastmolwidget.loader`."""

from pathlib import Path

import pytest
from qtpy import QtWidgets

from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

data = Path('tests/test-data')


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture()
def widget():
    w = MoleculeWidget()
    w.resize(400, 300)
    return w


@pytest.fixture()
def loader(widget):
    return MoleculeLoader(widget)


# ------------------------------------------------------------------
# Basic construction
# ------------------------------------------------------------------

def test_loader_holds_widget(widget):
    loader = MoleculeLoader(widget)
    assert loader.widget is widget


# ------------------------------------------------------------------
# CIF loading
# ------------------------------------------------------------------

def test_load_cif(loader, widget):
    loader.load_file(data / '1979688_small.cif')
    assert len(widget.atoms) == 94


def test_load_cif_str_path(loader, widget):
    loader.load_file(str(data / '1979688_small.cif'))
    assert len(widget.atoms) == 94


# ------------------------------------------------------------------
# SHELX loading
# ------------------------------------------------------------------

def test_load_shelx_res(loader, widget):
    loader.load_file(data / 'test_molecule.res')
    assert len(widget.atoms) == 5
    labels = [a.name for a in widget.atoms]
    assert 'C1' in labels
    assert 'O1' in labels


def test_parse_shelx_cell(loader):
    atoms, cell, adps = MoleculeLoader._parse_shelx(data / 'test_molecule.res')
    assert len(cell) == 6
    assert cell[0] == pytest.approx(6.0)
    assert cell[3] == pytest.approx(90.0)


def test_parse_shelx_types(loader):
    atoms, _, _ = MoleculeLoader._parse_shelx(data / 'test_molecule.res')
    types = [a.type for a in atoms]
    assert 'C' in types
    assert 'O' in types
    assert 'H' in types


def test_load_shelx_res_disordered(loader, widget):
    """Test loading a .res file with disorder (multiple PART instructions)."""
    loader.load_file(data / 'p31c-finalcif.res')
    # Should only include PART 0 and PART 1 atoms, not PART 2
    assert len(widget.atoms) > 0
    # Verify no duplicate labels (would happen if both PART 1 and PART 2 were included)
    labels = [a.name for a in widget.atoms]
    assert len(labels) == len(set(labels)), "Duplicate atom labels found - disorder parts not filtered"
    # Should have around 50-60 atoms (PART 0 and PART 1 only, no PART 2)
    assert len(widget.atoms) < 100


def test_parse_shelx_excludes_q_peaks():
    """Test that Q-peaks (residual electron density) are not included as atoms."""
    atoms, _, _ = MoleculeLoader._parse_shelx(data / 'p31c-finalcif.res')
    q_labels = [a.label for a in atoms if a.label.startswith('Q')]
    assert len(q_labels) == 0, f"Q-peaks should be excluded, found: {q_labels}"


def test_parse_shelx_returns_adps():
    """Test that anisotropic displacement parameters are extracted from .res files."""
    atoms, cell, adps = MoleculeLoader._parse_shelx(data / 'p31c-finalcif.res')
    # Should have ADPs for anisotropic atoms (non-hydrogen, non-isotropic)
    assert len(adps) > 0, "Should have ADPs for anisotropic atoms"
    # CL1 should have ADPs
    assert 'CL1' in adps, "CL1 should have ADP values"
    u11, u22, u33, u23, u13, u12 = adps['CL1']
    assert u11 == pytest.approx(0.01547)
    assert u22 == pytest.approx(0.01791)
    assert u33 == pytest.approx(0.02428)
    # Hydrogen atoms should NOT have ADPs (they are isotropic/riding)
    h_in_adps = [k for k in adps if k.startswith('H')]
    assert len(h_in_adps) == 0, f"Hydrogen atoms should not have ADPs, found: {h_in_adps}"


# ------------------------------------------------------------------
# XYZ loading
# ------------------------------------------------------------------

def test_load_xyz(loader, widget):
    loader.load_file(data / 'test_molecule.xyz')
    assert len(widget.atoms) == 5


def test_parse_xyz_labels():
    atoms = MoleculeLoader._parse_xyz(data / 'test_molecule.xyz')
    # Labels should be Element + 1-based index
    assert atoms[0].label == 'O1'
    assert atoms[3].label == 'C4'


def test_parse_xyz_coords():
    atoms = MoleculeLoader._parse_xyz(data / 'test_molecule.xyz')
    assert atoms[0].x == pytest.approx(0.0)
    assert atoms[0].z == pytest.approx(0.1173)


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_unsupported_format(loader):
    with pytest.raises(ValueError, match='Unsupported file format'):
        loader.load_file(data / 'fake.pdb')


def test_file_not_found(loader):
    with pytest.raises(FileNotFoundError):
        loader.load_file('nonexistent.cif')


def test_xyz_bad_atom_count(tmp_path):
    bad_xyz = tmp_path / 'bad.xyz'
    bad_xyz.write_text('10\ncomment\nC 0 0 0\n')
    with pytest.raises(ValueError, match='declares 10 atoms but 1'):
        MoleculeLoader._parse_xyz(bad_xyz)


def test_xyz_too_short(tmp_path):
    bad_xyz = tmp_path / 'short.xyz'
    bad_xyz.write_text('1\n')
    with pytest.raises(ValueError, match='too short'):
        MoleculeLoader._parse_xyz(bad_xyz)


def test_shelx_no_cell(tmp_path):
    bad_res = tmp_path / 'nocell.res'
    bad_res.write_text('TITL test\nSFAC C\nC1 1 0.1 0.2 0.3 11.0 0.02\nEND\n')
    with pytest.raises(ValueError, match='No CELL instruction'):
        MoleculeLoader._parse_shelx(bad_res)


# ------------------------------------------------------------------
# Keep view
# ------------------------------------------------------------------

def test_load_keep_view(loader, widget):
    loader.load_file(data / 'test_molecule.xyz')
    first_count = len(widget.atoms)
    assert first_count == 5
    # Load again with keep_view
    loader.load_file(data / 'test_molecule.xyz', keep_view=True)
    assert len(widget.atoms) == 5


# ------------------------------------------------------------------
# Rendering sanity check
# ------------------------------------------------------------------

def test_loaded_molecule_renders(loader, widget):
    loader.load_file(data / '1979688_small.cif')
    widget.show()
    pixmap = widget.grab()
    assert not pixmap.isNull()


# ------------------------------------------------------------------
# Grow (SDM)
# ------------------------------------------------------------------

def test_grow_produces_more_atoms_than_asym_unit(widget):
    loader = MoleculeLoader(widget)
    loader.load_file(data / '1979688_small.cif')
    asym_count = len(widget.atoms)
    loader.set_grow(True)
    grown_count = len(widget.atoms)
    assert grown_count >= asym_count, "Grown structure should have at least as many atoms"


def test_grow_toggle_restores_asym_unit(widget):
    loader = MoleculeLoader(widget)
    loader.load_file(data / '1979688_small.cif')
    asym_count = len(widget.atoms)
    loader.set_grow(True)
    loader.set_grow(False)
    assert len(widget.atoms) == asym_count


def test_grow_has_no_effect_on_xyz(widget):
    loader = MoleculeLoader(widget)
    loader.load_file(data / 'test_molecule.xyz')
    count_before = len(widget.atoms)
    loader.set_grow(True)  # should silently do nothing for non-CIF
    assert len(widget.atoms) == count_before


def test_grow_shelx_produces_more_atoms(widget):
    """Grow should expand the asymmetric unit for .res files too."""
    loader = MoleculeLoader(widget)
    loader.load_file(data / 'p31c-finalcif.res')
    asym_count = len(widget.atoms)
    loader.set_grow(True)
    grown_count = len(widget.atoms)
    assert grown_count > asym_count, "Grown .res structure should have more atoms"


def test_grow_shelx_toggle_restores_asym_unit(widget):
    """Toggling grow off should restore the original atom count for .res files."""
    loader = MoleculeLoader(widget)
    loader.load_file(data / 'p31c-finalcif.res')
    asym_count = len(widget.atoms)
    loader.set_grow(True)
    loader.set_grow(False)
    assert len(widget.atoms) == asym_count


