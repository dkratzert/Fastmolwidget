"""Parsing SHELXL's console output and list file into :class:`RefinementSummary`."""

from __future__ import annotations

import pytest

from fastmolwidget.shelxl_summary import RefinementSummary

#: Real SHELXL 2025/1 console output (tests/test-data/p31c-finalcif), shortened.
CONSOLE = """\
 Read instructions and data
 Data:    5352 unique,      0 suppressed   R(int) = 0.0592   R(sigma) = 0.0390
 wR2 =  0.0727 before cycle   1 for    5352 data and    287 /    287 parameters
 GooF = S =     1.044;     Restrained GooF =      1.022 for     357 restraints
 Mean shift/esd =   0.002  Maximum =     0.013 for  U13 P2         at 15:42:10
 Max. shift = 0.000 A for H2^a        Max. dU = 0.000 for N1^a
 wR2 =  0.0729 before cycle   2 for    5352 data and    287 /    287 parameters
 GooF = S =     1.045;     Restrained GooF =      1.021 for     357 restraints
 Mean shift/esd =   0.001  Maximum =    -0.004 for   z  H1^a       at 15:42:10
 wR2 =  0.0727 before cycle   3 for    5352 data and      0 /    287 parameters
 GooF = S =     1.044;     Restrained GooF =      1.022 for     357 restraints
 wR2 = 0.0727, GooF = S = 1.044, Restrained GooF = 1.022 for all data
 R1 = 0.0308 for 4999 Fo > 4sig(Fo) and 0.0343 for all 5352 data
 Flack x =   -0.018(65) by classical fit to all intensities
 Flack x =    0.007(31) from 2300 selected quotients (Parsons' method)
     0  atoms may be split and     0  atoms NPD
 R1 =  0.0278 for   2684 unique reflections after merging for Fourier
 Highest peak    0.22  at  0.7005  0.1079  0.6530  [  0.83 A from C5 ]
 Deepest hole   -0.25  at  0.9788  0.3238  0.8100  [  0.41 A from P2 ]
"""

#: The tail of the matching .lst file, where the layout differs.
LIST_TAIL = """\
 wR2 =  0.0727 before cycle   2 for    5352 data and    287 /    287 parameters
 Mean shift/esd =   0.001  Maximum =    -0.004 for   z  H1^a
 Final Structure Factor Calculation for  p-31c-neu in P-31c
 wR2 =  0.0727 before cycle   3 for    5352 data and      0 /    287 parameters
 GooF = S =     1.044;     Restrained GooF =      1.022 for     357 restraints
 wR2 = 0.0727, GooF = S = 1.044, Restrained GooF = 1.022 for all data
 R1 = 0.0308 for 4999 Fo > 4sig(Fo) and 0.0343 for all 5352 data
 Flack x =   -0.018(65) by classical fit to all intensities
              0.007(31) from 2300 selected quotients (Parsons' method)
 Recommended weighting scheme:  WGHT    0.0348    0.6275
 Highest peak    0.22  at  0.7005  0.1079  0.6530  [  0.83 A from C5 ]
 Deepest hole   -0.25  at  0.9788  0.3238  0.8100  [  0.41 A from P2 ]
"""


def test_the_console_output_gives_every_headline_value():
    summary = RefinementSummary(cycles=2)
    for line in CONSOLE.splitlines():
        summary.feed(line)
    assert summary.cycle == 2 and summary.final
    assert (summary.data, summary.restraints, summary.parameters) == (5352, 357, 287)
    assert summary.wr2 == 0.0727
    assert (summary.goof, summary.restrained_goof) == (1.044, 1.022)
    assert (summary.r1_obs, summary.n_obs) == (0.0308, 4999)
    assert (summary.r1_all, summary.n_all) == (0.0343, 5352)
    assert (summary.mean_shift, summary.max_shift) == (0.001, -0.004)
    assert summary.max_shift_parameter == 'z H1^a'
    assert (summary.flack, summary.flack_parsons) == ('-0.018(65)', '0.007(31)')
    assert summary.peak.height == 0.22 and summary.peak.atom == 'C5'
    assert summary.peak.distance == 0.83
    assert summary.hole.height == -0.25 and summary.hole.atom == 'P2'
    assert summary.wght is None  # only in the .lst file


def test_values_follow_the_cycles_while_shelxl_runs():
    summary = RefinementSummary(cycles=2)
    lines = iter(CONSOLE.splitlines())
    for line in lines:
        summary.feed(line)
        if 'Maximum =' in line:
            break
    assert summary.cycle == 1 and not summary.final
    assert summary.max_shift == 0.013 and summary.max_shift_parameter == 'U13 P2'
    assert summary.r1_obs is None


def test_unrelated_lines_change_nothing():
    summary = RefinementSummary()
    assert not summary.feed(' R1 =  0.0278 for   2684 unique reflections after merging for Fourier')
    assert not summary.feed(' Read instructions and data')
    assert summary == RefinementSummary()


def test_the_list_file_adds_the_weighting_scheme(tmp_path):
    lst = tmp_path / 'p31c.lst'
    lst.write_text('lots of earlier output\n' * 1000 + LIST_TAIL)
    summary = RefinementSummary.from_list_file(lst, cycles=2)
    assert summary is not None
    assert summary.wght == (0.0348, 0.6275)
    assert summary.cycle == 2 and summary.final
    assert summary.max_shift == -0.004
    assert summary.flack_parsons == '0.007(31)'
    assert summary.r1_obs == 0.0308


def test_a_missing_list_file_gives_none(tmp_path):
    assert RefinementSummary.from_list_file(tmp_path / 'nothing.lst') is None


@pytest.mark.parametrize(('name', 'expected'), [
    ('p31c-finalcif.res', 'P31c'),  # LATT -1; the TITL wrongly says P-31c
    ('BB_LJ45_a.res', 'P21/c'),
])
def test_the_space_group_comes_from_latt_and_symm(name, expected):
    from shelxfile import Shelxfile

    from fastmolwidget.shelxl_summary import space_group_of

    shx = Shelxfile()
    shx.read_file(f'tests/test-data/{name}')
    assert space_group_of(shx)[0] == expected


def test_the_title_is_the_fallback_space_group():
    from types import SimpleNamespace

    from fastmolwidget.shelxl_summary import space_group_of

    broken = SimpleNamespace(symmcards=None, latt=None, space_group='P2(1)/c')
    name, tooltip = space_group_of(broken)
    assert name == 'P2(1)/c' and 'title' in tooltip


def test_a_real_list_file(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path

    from shelxfile.refine.refine import find_shelxl_exe

    exe = find_shelxl_exe()
    if not exe:
        pytest.skip('SHELXL not found')
    source = Path('tests/test-data/p31c-finalcif.res')
    shutil.copy(source, tmp_path / 'p31c.ins')
    shutil.copy(source.with_suffix('.hkl'), tmp_path / 'p31c.hkl')
    subprocess.run([exe, 'p31c'], cwd=tmp_path, capture_output=True, check=False)
    summary = RefinementSummary.from_list_file(tmp_path / 'p31c.lst')
    assert summary is not None and summary.final
    assert summary.r1_obs is not None and summary.wght is not None
    assert summary.peak is not None and summary.mean_shift is not None
