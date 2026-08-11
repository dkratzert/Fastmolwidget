"""Tests for the SHELX ``.hkl`` / CIF reflection readers in
:mod:`fastmolwidget.hkl_io`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastmolwidget.hkl_io import (
    embedded_shelx_res,
    read_cif_reflections,
    read_reflections,
    read_shelx_hkl,
    read_shelx_parameters,
)

DATA = Path('tests/test-data')
HKL = DATA / 'p31c-finalcif.hkl'
RES = DATA / 'p31c-finalcif.res'
CIF = DATA / 'p31c.cif'


def test_read_shelx_hkl_basic():
    data = read_shelx_hkl(HKL)

    assert len(data) == 35969
    assert data.hkl.shape == (35969, 3)
    assert data.hkl.dtype == np.int32
    assert data.f_sq_meas.shape == (35969,)
    assert data.sigma.shape == (35969,)
    assert not data.has_f_calc


def test_read_shelx_hkl_first_and_last_record():
    data = read_shelx_hkl(HKL)

    assert tuple(data.hkl[0]) == (1, 0, 0)
    assert data.f_sq_meas[0] == pytest.approx(12.9175)
    assert data.sigma[0] == pytest.approx(0.5785)
    # The 0 0 0 terminator must not be part of the data.
    assert not np.all(data.hkl == 0, axis=1).any()


def test_read_shelx_hkl_free_format(tmp_path):
    """Whitespace-separated files are accepted as a fallback."""
    path = tmp_path / 'free.hkl'
    path.write_text('1 2 3 45.5 1.25\n-1 -2 -3 22.0 0.75\n0 0 0 0 0\n')

    data = read_shelx_hkl(path)

    assert len(data) == 2
    assert tuple(data.hkl[0]) == (1, 2, 3)
    assert data.f_sq_meas[1] == pytest.approx(22.0)


def test_read_shelx_hkl_empty_file_raises(tmp_path):
    path = tmp_path / 'empty.hkl'
    path.write_text('   0   0   0    0.00    0.00\n')

    with pytest.raises(ValueError, match='No reflections'):
        read_shelx_hkl(path)


def test_read_reflections_dispatches_on_suffix():
    data = read_reflections(HKL)

    assert len(data) == 35969


def test_read_cif_reflections_returns_none_without_loop():
    """p31c.cif carries no _refln loop, so there is nothing to read."""
    assert read_cif_reflections(CIF) is None


def test_read_shelx_parameters_from_res():
    params = read_shelx_parameters(RES)

    assert params is not None
    assert params.osf == pytest.approx(0.22604)
    assert params.wght_a == pytest.approx(0.0346)
    assert params.wght_b == pytest.approx(0.6436)
    assert params.exti == pytest.approx(0.0)
    assert params.wavelength == pytest.approx(0.71073)
    assert params.free_variables == pytest.approx([0.22604, 0.7606, 0.85131])


def test_read_shelx_parameters_finds_sibling_res(tmp_path):
    """A CIF without SHELX data picks up the .res of the same basename."""
    (tmp_path / 'x.cif').write_text('data_x\n_cell_length_a 10\n')
    (tmp_path / 'x.res').write_text(
        'CELL 0.71073 10 10 10 90 90 90\nWGHT 0.05 0.1\n'
        'EXTI 0.0012\nFVAR 0.5\nHKLF 4\n'
    )

    params = read_shelx_parameters(tmp_path / 'x.cif')

    assert params is not None
    assert params.osf == pytest.approx(0.5)
    assert params.exti == pytest.approx(0.0012)


def test_read_shelx_parameters_from_embedded_cif_block(tmp_path):
    """A SHELX .res embedded in a CIF is used when no sibling file exists."""
    res_text = (
        'TITL embedded\n'
        'CELL 0.71073 10 11 12 90 95 90\n'
        'WGHT 0.0400 0.9000\n'
        'FVAR 0.31234\n'
        'HKLF 4\n'
    )
    cif = tmp_path / 'embedded.cif'
    cif.write_text(f"data_x\n_shelx_res_file\n;\n{res_text};\n")

    assert embedded_shelx_res(cif) is not None
    params = read_shelx_parameters(cif)

    assert params is not None
    assert params.osf == pytest.approx(0.31234)
    assert params.wght_a == pytest.approx(0.04)
    assert params.wght_b == pytest.approx(0.9)


def test_embedded_shelx_res_found_in_real_cif():
    """p31c.cif carries the full final .res in ``_shelx_res_file``."""
    text = embedded_shelx_res(CIF)

    assert text is not None
    assert 'FVAR' in text
    assert 'CELL' in text


def test_read_shelx_parameters_from_real_cif_alone(tmp_path):
    """A deposited CIF alone yields the exact refined parameters.

    The CIF is copied on its own so that no sibling ``.res`` can be found and
    the embedded ``_shelx_res_file`` block is really what gets parsed.
    """
    import shutil

    lonely = tmp_path / 'only.cif'
    shutil.copy(CIF, lonely)

    params = read_shelx_parameters(lonely)

    assert params is not None
    assert params.osf == pytest.approx(0.22604)
    assert params.wght_a == pytest.approx(0.0346)
    assert params.wght_b == pytest.approx(0.6436)
    assert params.wavelength == pytest.approx(0.71073)


def test_read_shelx_parameters_missing_returns_none(tmp_path):
    lonely = tmp_path / 'nothing.cif'
    lonely.write_text('data_x\n_cell_length_a 10\n')

    assert read_shelx_parameters(lonely) is None
