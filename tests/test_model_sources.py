"""Tests for in-memory model sources and the host-friendly density controls.

These cover what a host application needs when it feeds the widget its own
atoms instead of letting :class:`~fastmolwidget.loader.MoleculeLoader` open a
file: declaring the model with ``set_model_source``, asking whether there is
any reflection data at all, and a control bar that never opens a file dialog.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest
from qtpy import QtWidgets

from fastmolwidget.density import (
    HAS_DENSITY_CPP,
    calculate_residual_density,
    small_structure_from_cif,
)
from fastmolwidget.density_controls import ResidualDensityControls
from fastmolwidget.hkl_io import (
    block_has_reflections,
    block_shelx_parameters,
    has_reflections,
    read_reflections,
    read_shelx_parameters,
)
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

DATA = Path('tests/test-data')
#: A self-contained SHELXL CIF: model and reflections in one file.
CIF_WITH_HKL = DATA / '1979688.cif'
#: A deposited CIF without any reflection data.
CIF_WITHOUT_HKL = DATA / '1000006.cif'

needs_cpp = pytest.mark.skipif(
    not HAS_DENSITY_CPP,
    reason='density_cpp C++ extension not built',
)


@pytest.fixture
def document():
    return gemmi.cif.read(str(CIF_WITH_HKL))


@pytest.fixture
def block(document):
    return document[0]


# ------------------------------------------------------------------
# hkl_io: block-level readers
# ------------------------------------------------------------------

def test_block_has_reflections(block):
    assert block_has_reflections(block)


def test_block_without_reflections():
    doc = gemmi.cif.read(str(CIF_WITHOUT_HKL))
    assert not block_has_reflections(doc[0])


def test_has_reflections_accepts_document_and_block(document, block):
    assert has_reflections(document)
    assert has_reflections(block)


def test_read_reflections_from_block_matches_file(block):
    from_file = read_reflections(CIF_WITH_HKL)
    from_block = read_reflections(block)

    assert len(from_block) == len(from_file)
    assert np.array_equal(from_block.hkl, from_file.hkl)


def test_read_reflections_returns_given_data(block):
    """Already read reflections are passed through unchanged."""
    data = read_reflections(block)

    assert read_reflections(data) is data


def test_shelx_parameters_from_block(block):
    from_block = block_shelx_parameters(block)

    assert from_block is not None
    assert from_block.osf == read_shelx_parameters(CIF_WITH_HKL).osf


def test_read_reflections_without_data_raises():
    doc = gemmi.cif.read(str(CIF_WITHOUT_HKL))

    with pytest.raises(ValueError, match='No reflections'):
        read_reflections(doc[0])


# ------------------------------------------------------------------
# density: in-memory models
# ------------------------------------------------------------------

@needs_cpp
def test_map_from_block_equals_map_from_file(block):
    from_file = calculate_residual_density(CIF_WITH_HKL)
    from_block = calculate_residual_density(block)

    assert np.allclose(from_block.array, from_file.array)


@needs_cpp
def test_map_from_document_equals_map_from_file(document):
    from_file = calculate_residual_density(CIF_WITH_HKL)

    assert np.allclose(calculate_residual_density(document).array, from_file.array)


def test_small_structure_needs_explicit_reflections(block):
    """A bare structure has no file to look next to."""
    structure = small_structure_from_cif(block)

    with pytest.raises(FileNotFoundError, match='in-memory model'):
        calculate_residual_density(structure)


@needs_cpp
def test_small_structure_with_reflections(block):
    structure = small_structure_from_cif(block)

    assert calculate_residual_density(structure, block).rms > 0.0


# ------------------------------------------------------------------
# Renderer: set_model_source
# ------------------------------------------------------------------

def test_no_data_without_a_source():
    assert not MoleculeWidget().has_residual_density_data


def test_block_source_provides_data(block):
    widget = MoleculeWidget()
    widget.set_model_source(block)

    assert widget.has_residual_density_data


def test_model_without_reflections_has_no_data():
    widget = MoleculeWidget()
    widget.set_model_source(gemmi.cif.read(str(CIF_WITHOUT_HKL))[0])

    assert not widget.has_residual_density_data


@needs_cpp
def test_show_density_from_block_source(block):
    widget = MoleculeWidget()
    widget.set_model_source(block, reflections=block)
    widget.show_residual_density()

    assert widget.residual_density_map is not None


@needs_cpp
def test_changing_the_source_drops_the_map(block):
    widget = MoleculeWidget()
    widget.set_model_source(block, reflections=block)
    widget.show_residual_density()

    widget.set_model_source(gemmi.cif.read(str(CIF_WITHOUT_HKL))[0])

    assert widget.residual_density_map is None


@needs_cpp
def test_reloading_the_same_file_keeps_the_map():
    """Grow and pack reload the same path; the map must survive that."""
    widget = MoleculeWidget()
    loader = MoleculeLoader(widget)
    loader.load_file(CIF_WITH_HKL)
    widget.show_residual_density()

    loader.load_file(CIF_WITH_HKL)

    assert widget.residual_density_map is not None


@needs_cpp
def test_refresh_keeps_the_map(block):
    widget = MoleculeWidget()
    widget.set_model_source(block, reflections=block)
    widget.show_residual_density()
    density_map = widget.residual_density_map

    widget.refresh_residual_density()

    assert widget.residual_density_map is density_map


# ------------------------------------------------------------------
# ResidualDensityControls
# ------------------------------------------------------------------

@pytest.fixture
def controls():
    widget = MoleculeWidget()
    bar = ResidualDensityControls(render_widget=widget,
                                  allow_reflection_dialog=False)
    return bar, widget


def test_controls_disabled_without_data(controls):
    bar, _ = controls
    bar.update_density_availability()

    assert not bar.button.isEnabled()
    assert 'No usable reflection data' in bar.button.toolTip()


def test_controls_enabled_with_data(controls, block):
    bar, widget = controls
    widget.set_model_source(block, reflections=block)
    bar.update_density_availability()

    assert bar.button.isEnabled()


def test_no_dialog_when_it_is_switched_off(controls, monkeypatch):
    """Without data and without a dialog the button just pops back out."""
    bar, _ = controls
    monkeypatch.setattr(
        QtWidgets.QFileDialog, 'getOpenFileName',
        lambda *a, **k: pytest.fail('a file dialog was opened'),
    )

    bar.button.setChecked(True)

    assert not bar.button.isChecked()


@needs_cpp
def test_button_shows_and_hides_density(controls, block):
    bar, widget = controls
    widget.set_model_source(block, reflections=block)
    bar.update_density_availability()

    bar.button.setChecked(True)
    assert widget.residual_density_map is not None
    assert bar.level_spinbox.isEnabled()

    bar.button.setChecked(False)
    assert widget.residual_density_map is None
    assert not bar.level_spinbox.isEnabled()
