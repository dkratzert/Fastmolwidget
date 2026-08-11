"""Tests for :mod:`fastmolwidget.density` — residual (Fo−Fc) maps.

The reference values come from ``tests/test-data/p31c.cif``, which reports the
results of the original SHELXL refinement::

    _refine_ls_R_factor_all      0.0343
    _refine_ls_R_factor_gt       0.0308
    _refine_diff_density_max     0.224
    _refine_diff_density_min    -0.252
    _refine_diff_density_rms     0.053

Our independent implementation is not expected to reproduce SHELXL's numbers
exactly (SHELXL merges Friedel pairs, neglects *f″* and contours on its own
grid), so the map tests assert the right order of magnitude and sign rather
than equality.  The *R*\\ :sub:`1` test is the strict one — it verifies that the
structure-factor calculation itself is correct.
"""

from __future__ import annotations

from pathlib import Path

import gemmi
import numpy as np
import pytest

from fastmolwidget.density import (
    HAS_DENSITY_CPP,
    ResidualDensityMap,
    calculate_residual_density,
    small_structure_from_cif,
    small_structure_from_shelx,
)

DATA = Path('tests/test-data')
HKL = DATA / 'p31c-finalcif.hkl'
RES = DATA / 'p31c-finalcif.res'
CIF = DATA / 'p31c.cif'

needs_cpp = pytest.mark.skipif(
    not HAS_DENSITY_CPP,
    reason='density_cpp C++ extension not built',
)


@pytest.fixture(scope='module')
def shelx_structure():
    from shelxfile import Shelxfile

    shx = Shelxfile()
    shx.read_file(str(RES))
    return small_structure_from_shelx(shx)


@pytest.fixture(scope='module')
def cif_structure():
    return small_structure_from_cif(CIF)


@pytest.fixture(scope='module')
def density_map():
    return calculate_residual_density(RES, HKL)


# ---------------------------------------------------------------------------
# Building the model
# ---------------------------------------------------------------------------

def test_shelx_structure_basics(shelx_structure):
    assert len(shelx_structure.sites) == 88
    assert shelx_structure.spacegroup.xhm() == 'P 3 1 c'
    assert shelx_structure.cell.a == pytest.approx(12.5067)
    assert shelx_structure.cell.gamma == pytest.approx(120.0)


def test_shelx_structure_sets_cell_images(shelx_structure):
    """gemmi reads the symmetry for SF summation from the UnitCell."""
    assert len(shelx_structure.cell.images) == 5


def test_shelx_structure_sets_scattering_elements(shelx_structure):
    """``type_symbol`` is only a label - ``element`` drives the form factors."""
    elements = {str(site.element) for site in shelx_structure.sites}

    assert 'X' not in elements
    by_label = {s.label.upper(): s for s in shelx_structure.sites}
    assert by_label['CL1'].element == gemmi.Element('Cl')
    assert by_label['N3'].element == gemmi.Element('N')


def test_shelx_riding_hydrogen_uiso_resolved(shelx_structure):
    """``U = -1.5`` means 1.5 x Ueq of the atom the hydrogen rides on."""
    by_label = {s.label.upper(): s for s in shelx_structure.sites}

    assert by_label['H23A'].u_iso == pytest.approx(0.0786, abs=1e-3)
    assert all(site.u_iso >= 0.0 for site in shelx_structure.sites)


def test_shelx_occupancies_decode_negative_sof(shelx_structure):
    """``-30.33333`` is ``(1 - FVAR3) * 0.33333``, not ``1 - FVAR3 * 0.33333``.

    Guards the workaround for the shelxfile occupancy bug.
    """
    by_label = {s.label.upper(): s for s in shelx_structure.sites}

    assert by_label["C1"].occ == pytest.approx(0.2838, abs=1e-3)
    assert by_label["C1'"].occ == pytest.approx(0.0496, abs=1e-3)
    assert by_label["C12'"].occ == pytest.approx(0.0798, abs=1e-3)


def test_shelx_and_cif_structures_agree(shelx_structure, cif_structure):
    """The SHELX and CIF routes must build the same model."""
    assert len(shelx_structure.sites) == len(cif_structure.sites)

    reference = {s.label.upper(): s for s in cif_structure.sites}
    for site in shelx_structure.sites:
        other = reference[site.label.upper()]
        assert site.occ == pytest.approx(other.occ, abs=2e-3)
        assert site.u_iso == pytest.approx(other.u_iso, abs=2e-3)
        assert site.fract.x == pytest.approx(other.fract.x, abs=2e-3)
        assert site.fract.z == pytest.approx(other.fract.z, abs=2e-3)


# ---------------------------------------------------------------------------
# Structure factors
# ---------------------------------------------------------------------------

def test_r1_against_published_value(shelx_structure):
    """|Fo| vs |Fc| must reproduce the R1 reported by SHELXL (0.0343).

    This is the real correctness check on the structure-factor summation:
    a wrong scattering factor, ADP convention, occupancy or symmetry setting
    pushes R1 far above 0.05.
    """
    from fastmolwidget.hkl_io import read_shelx_hkl

    data = read_shelx_hkl(HKL)
    ops = shelx_structure.spacegroup.operations()
    asu = gemmi.ReciprocalAsu(shelx_structure.spacegroup)

    merged: dict[tuple[int, int, int], list[float]] = {}
    for (h, k, l), f_sq, sigma in zip(data.hkl, data.f_sq_meas, data.sigma):
        index, _ = asu.to_asu([int(h), int(k), int(l)], ops)
        weight = 1.0 / max(float(sigma), 1e-6) ** 2
        entry = merged.setdefault(tuple(index), [0.0, 0.0])
        entry[0] += weight * float(f_sq)
        entry[1] += weight

    calculator = gemmi.StructureFactorCalculatorX(shelx_structure.cell)
    calculator.addends.add_cl_fprime(gemmi.hc / 0.71073)

    f_obs, f_calc = [], []
    for index, (weighted, total) in merged.items():
        f_obs.append(np.sqrt(max(weighted / total, 0.0)))
        f_calc.append(abs(calculator.calculate_sf_from_small_structure(
            shelx_structure, list(index))))
    f_obs = np.array(f_obs)
    f_calc = np.array(f_calc)

    scale = np.sum(f_obs * f_calc) / np.sum(f_calc ** 2)
    r1 = np.sum(np.abs(f_obs - scale * f_calc)) / np.sum(f_obs)

    assert r1 < 0.05, f'R1 = {r1:.4f}, expected close to the published 0.0343'
    # The least-squares scale must land on SHELXL's refined OSF.
    assert scale == pytest.approx(0.22604, abs=0.01)


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------

def test_map_uses_shelxl_scale_factor(density_map):
    """The refined OSF from FVAR is used, not a re-derived estimate."""
    assert density_map.scale == pytest.approx(0.22604)


def test_map_resolution_and_grid(density_map):
    assert density_map.d_min == pytest.approx(0.76, abs=0.02)
    assert density_map.array.ndim == 3
    assert min(density_map.array.shape) > 8


def test_map_statistics_are_plausible(density_map):
    """Peak, hole and rms must be in the range SHELXL reported."""
    assert 0.1 < density_map.max < 0.6      # SHELXL: +0.224
    assert -0.6 < density_map.min < -0.1    # SHELXL: -0.252
    assert 0.02 < density_map.rms < 0.12    # SHELXL:  0.053


def test_map_is_centred_on_zero(density_map):
    """A difference map has no bulk offset."""
    assert abs(density_map.array.mean()) < 0.1 * density_map.rms


def test_cif_and_shelx_models_give_the_same_map(density_map):
    from_cif = calculate_residual_density(CIF, HKL)

    assert from_cif.max == pytest.approx(density_map.max, abs=0.05)
    assert from_cif.min == pytest.approx(density_map.min, abs=0.05)
    assert from_cif.rms == pytest.approx(density_map.rms, abs=0.01)


def test_map_is_independent_of_grid_sampling():
    coarse = calculate_residual_density(RES, HKL, sample_rate=1.5)
    fine = calculate_residual_density(RES, HKL, sample_rate=4.0)

    assert coarse.rms == pytest.approx(fine.rms, abs=0.01)


def test_resolution_cutoff_reduces_detail():
    truncated = calculate_residual_density(RES, HKL, d_min=1.0)

    assert truncated.d_min >= 1.0


def test_orthogonalisation_matrix_matches_cell_volume(density_map):
    a, b, c, _, _, gamma = density_map.cell
    matrix = density_map.orth_matrix

    assert np.linalg.det(matrix) == pytest.approx(
        a * b * c * np.sin(np.radians(gamma)), rel=1e-6)


# ---------------------------------------------------------------------------
# Isosurfaces
# ---------------------------------------------------------------------------

@needs_cpp
def test_isosurface_returns_wireframe(density_map):
    vertices, edges = density_map.isosurface(0.25)

    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert edges.ndim == 2 and edges.shape[1] == 2
    assert len(vertices) > 0
    assert edges.max() < len(vertices)


@needs_cpp
def test_isosurface_positive_and_negative_lobes_differ(density_map):
    positive, _ = density_map.isosurface(0.25)
    negative, _ = density_map.isosurface(-0.25)

    assert len(positive) > 0
    assert len(negative) > 0
    assert not np.array_equal(positive, negative)


@needs_cpp
def test_isosurface_shrinks_as_level_rises(density_map):
    low, _ = density_map.isosurface(0.2)
    high, _ = density_map.isosurface(0.3)

    assert len(high) < len(low)


@needs_cpp
def test_isosurface_above_maximum_is_empty(density_map):
    vertices, edges = density_map.isosurface(density_map.max + 1.0)

    assert len(vertices) == 0
    assert len(edges) == 0


@needs_cpp
def test_isosurface_restricted_to_atoms(density_map):
    """Passing atoms clips the surface to their neighbourhood."""
    from shelxfile import Shelxfile

    shx = Shelxfile()
    shx.read_file(str(RES))
    positions = np.array([a.cart_coords for a in shx.atoms if not a.qpeak])

    full, _ = density_map.isosurface(0.25)
    local, _ = density_map.isosurface(0.25, atoms=positions, margin=1.0)

    assert len(local) > 0
    assert len(local) <= len(full) * 3  # a clipped region, not an explosion


@needs_cpp
def test_isosurface_vertices_are_cartesian(density_map):
    """Vertices must be transformed out of fractional space.

    The test cell is hexagonal (gamma = 120 deg), so a fractional-space
    surface would stay inside a 0..1 box instead of spanning the cell in A.
    """
    vertices, _ = density_map.isosurface(0.25)

    extent = vertices.max(axis=0) - vertices.min(axis=0)
    assert extent.max() > 5.0


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_unreadable_reflection_file_raises(tmp_path):
    bad = tmp_path / 'bad.hkl'
    bad.write_text('not a reflection file\n')

    with pytest.raises(ValueError):
        calculate_residual_density(RES, bad)


def test_map_dataclass_is_constructible():
    array = np.zeros((4, 4, 4), dtype=np.float32)
    density = ResidualDensityMap(array=array, cell=(10, 10, 10, 90, 90, 90),
                                 d_min=1.0, scale=1.0)

    assert density.rms == 0.0
    assert density.max == 0.0
