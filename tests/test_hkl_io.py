"""Tests for the SHELX ``.hkl`` / CIF reflection readers in
:mod:`fastmolwidget.hkl_io`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fastmolwidget.hkl_io import (
    embedded_shelx_res,
    find_reflection_file,
    has_reflections,
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


# ---------------------------------------------------------------------------
# Automatic discovery of the reflection data
# ---------------------------------------------------------------------------

P21C = DATA / 'p21c.cif'


def test_has_reflections_detects_each_source():
    assert has_reflections(HKL)          # plain SHELX .hkl
    assert has_reflections(P21C)         # embedded _shelx_hkl_file
    assert not has_reflections(DATA / 'does-not-exist.hkl')


def test_has_reflections_false_for_a_plain_cif(tmp_path):
    plain = tmp_path / 'plain.cif'
    plain.write_text('data_x\n_cell_length_a 10\n')

    assert not has_reflections(plain)


def test_find_reflection_file_prefers_the_model_itself():
    """A self-contained CIF needs nothing else."""
    assert find_reflection_file(P21C) == P21C


def test_find_reflection_file_finds_the_sibling_hkl():
    assert find_reflection_file(RES) == HKL


def test_find_reflection_file_returns_none_when_nothing_is_there(tmp_path):
    lonely = tmp_path / 'lonely.res'
    lonely.write_text('CELL 0.71073 10 10 10 90 90 90\nFVAR 0.5\nHKLF 4\n')

    assert find_reflection_file(lonely) is None


def test_find_reflection_file_ignores_an_empty_sibling(tmp_path):
    """A sibling of the right name but without reflections must be rejected."""
    import shutil

    model = tmp_path / 'x.res'
    shutil.copy(RES, model)
    (tmp_path / 'x.cif').write_text('data_x\n_cell_length_a 10\n')

    assert find_reflection_file(model) is None


def test_find_reflection_file_picks_the_sibling_cif_when_it_has_data(tmp_path):
    import shutil

    model = tmp_path / 'x.res'
    shutil.copy(RES, model)
    shutil.copy(P21C, tmp_path / 'x.cif')

    assert find_reflection_file(model) == tmp_path / 'x.cif'


# ---------------------------------------------------------------------------
# The vectorised fixed-format fast path
# ---------------------------------------------------------------------------

def _reference_parse(text: str):
    """Parse with the record-by-record fallback only."""
    from fastmolwidget.hkl_io import ReflectionData, _parse_hkl_line

    hkl, f_sq, sigma, batch = [], [], [], []
    for raw in text.splitlines():
        line = raw.rstrip('\r')
        if not line.strip():
            continue
        parsed = _parse_hkl_line(line)
        if parsed is None:
            continue
        h, k, l, fsq, s, n = parsed
        if h == 0 and k == 0 and l == 0:
            break
        hkl.append((h, k, l))
        f_sq.append(fsq)
        sigma.append(s)
        batch.append(n)
    return ReflectionData(
        hkl=np.array(hkl, dtype=np.int32),
        f_sq_meas=np.array(f_sq, dtype=float),
        sigma=np.array(sigma, dtype=float),
        batch=np.array(batch, dtype=np.int32),
    )


@pytest.mark.parametrize('text, uses_fast_path', [
    ('   1   2   3   12.34    1.23\n   0   0   0    0.00    0.00\n', True),
    ('   1   2   3   12.34    1.23\r\n   1   2   4    5.00    0.50   2\r\n', True),
    ('   1   2   3   12.34    1.23\n\n   1   2   4    5.00    0.50   2\n', True),
    ('  -1  -2  -3   12.34    1.23\n 100  99 -99  5.00     0.50\n', True),
    ('1 2 3 12.34 1.23\n1 2 4 5.0 0.5 2\n', False),           # free format
    # SADABS appends its scaling report after the terminator.  Everything from
    # the 0 0 0 record on is ignored, so this is an ordinary file.
    (('   1   2   3   12.34    1.23\n   0   0   0    0.00    0.00\n'
      '_exptl_absorpt_correction_type multi-scan\n'
      '_exptl_absorpt_process_details\n;\n'
      ' SADABS 2016/2: Krause, L., Herbst-Irmer, R. et al.\n;\n'), True),
    # A stray record with no terminator to hide behind is dropped and the rest
    # still goes down the fast path - one such line must not cost a whole
    # 350 000 record file its vectorised parse.
    ('   1   2   3   12.34    1.23\nrubbish, and plenty of it here\n', True),
    ('   1   2   3   12.34    1.23\n! a comment\n   1   2   4    5.00    0.50\n',
     True),
])
def test_fixed_format_fast_path_matches_the_fallback(text, uses_fast_path):
    from fastmolwidget.hkl_io import _parse_fixed_format, parse_shelx_hkl

    assert (_parse_fixed_format(text.splitlines()) is not None) is uses_fast_path

    fast = parse_shelx_hkl(text)
    slow = _reference_parse(text)

    assert np.array_equal(fast.hkl, slow.hkl)
    assert np.array_equal(fast.f_sq_meas, slow.f_sq_meas)
    assert np.array_equal(fast.sigma, slow.sigma)
    assert np.array_equal(np.asarray(fast.batch), np.asarray(slow.batch))


def test_fast_path_matches_the_fallback_on_a_real_file():
    from fastmolwidget.hkl_io import parse_shelx_hkl

    text = HKL.read_text(errors='replace')

    fast = parse_shelx_hkl(text)
    slow = _reference_parse(text)

    assert len(fast) == len(slow) > 30000
    assert np.array_equal(fast.hkl, slow.hkl)
    assert np.array_equal(fast.f_sq_meas, slow.f_sq_meas)
    assert np.array_equal(fast.sigma, slow.sigma)
    assert np.array_equal(np.asarray(fast.batch), np.asarray(slow.batch))


def test_sadabs_trailer_needs_no_repair_pass():
    """The scaling report after the terminator is cut, not repaired.

    Everything from the ``0 0 0`` record on is ignored outright, so a file
    with a SADABS trailer must go through the plain conversion and never
    reach the drop-what-cannot-be-a-record retry.
    """
    from fastmolwidget import hkl_io as module

    text = (
        '   1   2   3   12.34    1.23\n'
        '   0   0   0    0.00    0.00\n'
        '_exptl_absorpt_correction_type multi-scan\n'
        '_exptl_absorpt_process_details\n;\n'
        ' SADABS 2016/2: Krause, L., Herbst-Irmer, R. et al.\n;\n'
    )

    calls = []
    original = module._fixed_format_mask
    module._fixed_format_mask = lambda codes: calls.append(1) or original(codes)
    try:
        data = module._parse_fixed_format(text.splitlines())
    finally:
        module._fixed_format_mask = original

    assert data is not None
    assert calls == []  # the retry was not needed
    assert np.array_equal(data.hkl, np.array([[1, 2, 3]], dtype=np.int32))


def test_records_after_the_terminator_are_ignored():
    """A trailer cannot add reflections, whatever it looks like."""
    from fastmolwidget.hkl_io import parse_shelx_hkl

    text = (
        '   1   2   3   12.34    1.23\n'
        '   0   0   0    0.00    0.00\n'
        '   9   9   9   99.99    9.99\n'
    )

    data = parse_shelx_hkl(text)

    assert np.array_equal(data.hkl, np.array([[1, 2, 3]], dtype=np.int32))


# ---------------------------------------------------------------------------
# Raw _diffrn_refln_* measurements
# ---------------------------------------------------------------------------

def _raw_diffrn_cif(reflections, *, sigma_tag='_diffrn_refln_intensity_u',
                    with_batch=True) -> str:
    """A minimal CIF carrying *reflections* as a raw diffraction loop."""
    tags = [
        '_diffrn_refln_index_h',
        '_diffrn_refln_index_k',
        '_diffrn_refln_index_l',
        '_diffrn_refln_intensity_net',
        sigma_tag,
    ]
    if with_batch:
        tags.append('_diffrn_refln_scale_group_code')

    rows = []
    for (h, k, l), f_sq, sigma, batch in zip(
        reflections.hkl, reflections.f_sq_meas, reflections.sigma,
        np.asarray(reflections.batch),
    ):
        values = [str(h), str(k), str(l), repr(float(f_sq)), repr(float(sigma))]
        if with_batch:
            values.append(str(int(batch)))
        rows.append(' '.join(values))

    return ('data_raw\n_cell_length_a 10\nloop_\n'
            + ''.join(f'  {tag}\n' for tag in tags)
            + '\n'.join(rows) + '\n')


def test_raw_diffrn_loop_matches_the_hkl_it_came_from(tmp_path):
    """``_diffrn_refln_*`` is a ``.hkl`` in CIF clothing and must read as one.

    FinalCif and Olex2 write the measured data into the final CIF for
    checkCIF; round-tripping a real ``.hkl`` through that loop has to give
    back exactly the same reflections.
    """
    from fastmolwidget.hkl_io import read_reflections, read_shelx_hkl

    original = read_shelx_hkl(HKL)
    cif = tmp_path / 'raw.cif'
    cif.write_text(_raw_diffrn_cif(original))

    data = read_reflections(cif)

    assert len(data.hkl) == len(original.hkl) > 30000
    assert np.array_equal(data.hkl, original.hkl)
    assert np.allclose(data.f_sq_meas, original.f_sq_meas)
    assert np.allclose(data.sigma, original.sigma)
    assert np.array_equal(np.asarray(data.batch), np.asarray(original.batch))
    assert data.sigma_known


def test_raw_diffrn_loop_makes_a_cif_a_reflection_source(tmp_path):
    """A CIF with only raw measurements still counts as having data."""
    from fastmolwidget.hkl_io import (
        find_reflection_file,
        has_reflections,
        read_shelx_hkl,
    )

    cif = tmp_path / 'raw.cif'
    cif.write_text(_raw_diffrn_cif(read_shelx_hkl(HKL)))

    assert has_reflections(cif)
    assert find_reflection_file(cif) == cif


def test_raw_diffrn_loop_accepts_the_older_sigma_tag(tmp_path):
    """``_diffrn_refln_intensity_sigma`` is the older name for the same thing."""
    from fastmolwidget.hkl_io import read_reflections, read_shelx_hkl

    original = read_shelx_hkl(HKL)
    cif = tmp_path / 'raw.cif'
    cif.write_text(_raw_diffrn_cif(
        original, sigma_tag='_diffrn_refln_intensity_sigma'))

    data = read_reflections(cif)

    assert np.allclose(data.sigma, original.sigma)
    assert data.sigma_known


def test_raw_diffrn_loop_without_a_batch_column(tmp_path):
    """A missing scale-group column means batch 1, as for a plain .hkl."""
    from fastmolwidget.hkl_io import read_reflections, read_shelx_hkl

    cif = tmp_path / 'raw.cif'
    cif.write_text(_raw_diffrn_cif(read_shelx_hkl(HKL), with_batch=False))

    data = read_reflections(cif)

    assert np.all(np.asarray(data.batch) == 1)


def test_processed_reflections_win_over_the_raw_ones(tmp_path):
    """The refinement's own data is preferred to the raw measurements.

    Raw ``_diffrn_refln_*`` data is unmerged and unscaled, so it is the last
    resort: a CIF that also carries the ``.hkl`` the refinement used must
    yield that instead.
    """
    from fastmolwidget.hkl_io import read_reflections

    raw = _raw_diffrn_cif(_ReflectionsStub())
    embedded = ('data_both\n_shelx_hkl_file\n;\n'
                '   1   2   3   12.34    1.23\n'
                '   0   0   0    0.00    0.00\n;\n')
    cif = tmp_path / 'both.cif'
    cif.write_text(raw + embedded)

    data = read_reflections(cif)

    # The embedded .hkl holds one reflection, the raw loop two.
    assert np.array_equal(data.hkl, np.array([[1, 2, 3]], dtype=np.int32))


class _ReflectionsStub:
    """Two made-up reflections, enough to tell the two sources apart."""

    hkl = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.int32)
    f_sq_meas = np.array([10.0, 20.0])
    sigma = np.array([1.0, 2.0])
    batch = np.array([1, 1], dtype=np.int32)


# ---------------------------------------------------------------------------
# CIF document cache
# ---------------------------------------------------------------------------

def test_cif_document_cache_returns_the_same_document():
    from fastmolwidget.hkl_io import read_cif_document

    assert read_cif_document(CIF) is read_cif_document(CIF)


def test_cif_document_cache_notices_an_edited_file(tmp_path):
    """The cache is keyed on mtime and size, so a rewrite must be picked up."""
    import os

    from fastmolwidget.hkl_io import read_cif_document

    path = tmp_path / 'edited.cif'
    path.write_text('data_one\n_cell_length_a 10.0\n')
    first = read_cif_document(path)
    assert first.sole_block().find_value('_cell_length_a') == '10.0'

    path.write_text('data_one\n_cell_length_a 20.0\n_cell_length_b 5.0\n')
    os.utime(path, (0, 0))  # force a different mtime even on a coarse clock

    second = read_cif_document(path)

    assert second.sole_block().find_value('_cell_length_a') == '20.0'


def test_clear_cif_cache_forgets_everything():
    from fastmolwidget.hkl_io import clear_cif_cache, read_cif_document

    first = read_cif_document(CIF)
    clear_cif_cache()

    assert read_cif_document(CIF) is not first
