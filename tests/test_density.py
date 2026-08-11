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
    DEFAULT_GRID_SPACING,
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
    coarse = calculate_residual_density(RES, HKL, grid_spacing=0.6)
    fine = calculate_residual_density(RES, HKL, grid_spacing=0.25)

    assert coarse.array.size < fine.array.size
    assert coarse.rms == pytest.approx(fine.rms, abs=0.015)


def test_grid_size_follows_the_cell_not_the_resolution():
    """Adding a resolution cut-off must not change the grid dimensions."""
    full = calculate_residual_density(RES, HKL)
    truncated = calculate_residual_density(RES, HKL, d_min=1.5)

    assert full.array.shape == truncated.array.shape


def test_grid_spacing_is_about_the_requested_length():
    density = calculate_residual_density(RES, HKL, grid_spacing=0.4)
    a, b, c = density.cell[:3]

    for length, points in zip((a, b, c), density.array.shape):
        assert length / points <= 0.4 + 1e-6


def test_default_grid_uses_the_default_spacing():
    """The default map must honour DEFAULT_GRID_SPACING on every axis."""
    density = calculate_residual_density(RES, HKL)
    a, b, c = density.cell[:3]

    for length, points in zip((a, b, c), density.array.shape):
        assert length / points <= DEFAULT_GRID_SPACING + 1e-6
        # ...and not be wastefully finer than one FFT size step
        assert length / points > DEFAULT_GRID_SPACING / 2.0


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
def test_isosurface_never_exceeds_the_margin(density_map):
    """Every vertex must lie within *margin* of some atom."""
    from shelxfile import Shelxfile

    shx = Shelxfile()
    shx.read_file(str(RES))
    positions = np.array([a.cart_coords for a in shx.atoms if not a.qpeak])
    margin = 1.5

    vertices, _ = density_map.isosurface(0.2, atoms=positions, margin=margin)

    assert len(vertices) > 0
    distances = np.linalg.norm(
        vertices[:, None, :] - positions[None, :, :], axis=2).min(axis=1)
    assert distances.max() <= margin + 1e-4


@needs_cpp
def test_smaller_margin_keeps_fewer_vertices(density_map):
    """Shrinking the margin can only ever remove vertices, never add them.

    Equality is expected over a range of margins, because for a well-refined
    structure essentially all residual density already sits within ~1 Å of an
    atom.
    """
    from shelxfile import Shelxfile

    shx = Shelxfile()
    shx.read_file(str(RES))
    positions = np.array([a.cart_coords for a in shx.atoms if not a.qpeak])

    counts = [
        len(density_map.isosurface(0.2, atoms=positions, margin=m)[0])
        for m in (0.8, 1.5, 4.0)
    ]

    assert counts == sorted(counts)
    assert counts[0] < counts[-1]


@needs_cpp
def test_isosurface_edges_stay_valid_after_clipping(density_map):
    from shelxfile import Shelxfile

    shx = Shelxfile()
    shx.read_file(str(RES))
    positions = np.array([a.cart_coords for a in shx.atoms if not a.qpeak])

    vertices, edges = density_map.isosurface(0.2, atoms=positions, margin=1.5)

    assert edges.min() >= 0
    assert edges.max() < len(vertices)


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


# ---------------------------------------------------------------------------
# Self-contained CIFs, global blocks and defective ADPs
# ---------------------------------------------------------------------------

P21C = DATA / 'p21c.cif'


def test_self_contained_cif_needs_no_separate_hkl():
    """p21c.cif carries its own _shelx_hkl_file, so the CIF alone suffices."""
    from fastmolwidget.hkl_io import embedded_shelx_hkl, read_reflections

    assert embedded_shelx_hkl(P21C) is not None
    data = read_reflections(P21C)

    assert len(data) > 40000


def test_p21c_map_from_the_cif_alone():
    density = calculate_residual_density(P21C)

    assert density.scale == pytest.approx(0.08684)
    # SHELXL reports +0.406 / -0.689 / 0.073 for this refinement.  Al1's
    # anisotropic ADP is corrupt in the deposited CIF and is downgraded to
    # isotropic, which legitimately leaves extra residual density behind.
    assert 0.2 < density.max < 3.0
    assert -3.0 < density.min < -0.2
    assert 0.03 < density.rms < 0.3


def test_p21c_calculation_is_fast():
    """A ~130-atom structure with 43k reflections must stay interactive."""
    import time

    start = time.perf_counter()
    calculate_residual_density(P21C)
    elapsed = time.perf_counter() - start

    assert elapsed < 4.0, f'took {elapsed:.1f} s'


def test_defective_adps_are_downgraded_to_isotropic():
    """A non-positive-definite ADP would blow up Fc at high angle."""
    with pytest.warns(RuntimeWarning, match='non-positive-definite'):
        structure = small_structure_from_cif(P21C)

    by_label = {s.label.upper(): s for s in structure.sites}
    al1 = by_label['AL1']
    assert al1.aniso.u33 == 0.0        # replaced
    assert al1.u_iso > 0.0             # kept a sensible Ueq

    for site in structure.sites:
        adp = site.aniso
        if adp.u11 or adp.u22 or adp.u33:
            matrix = np.array([[adp.u11, adp.u12, adp.u13],
                               [adp.u12, adp.u22, adp.u23],
                               [adp.u13, adp.u23, adp.u33]])
            assert np.all(np.linalg.eigvalsh(matrix) > 0)


def test_good_adps_are_left_alone(cif_structure):
    """p31c has clean ADPs, so nothing should be touched."""
    aniso = [s for s in cif_structure.sites if s.aniso.u11 != 0.0]

    assert len(aniso) > 30


def test_global_block_is_ignored(tmp_path):
    """A leading ``global_`` block must not be mistaken for the structure."""
    original = P21C.read_text(errors='replace')
    prefixed = tmp_path / 'with_global.cif'
    prefixed.write_text('global_\n_audit_creation_method fastmolwidget\n\n'
                        + original)

    structure = small_structure_from_cif(prefixed)

    assert len(structure.sites) == len(small_structure_from_cif(P21C).sites)


def test_systematically_absent_reflections_are_dropped():
    """Their Fc is zero, so their noise would enter the map amplified by 1/k."""
    from fastmolwidget.density import _merge_to_asu
    from fastmolwidget.hkl_io import read_reflections

    structure = small_structure_from_cif(P21C)
    data = read_reflections(P21C)
    hkl, _ = _merge_to_asu(data, structure.spacegroup, None, structure.cell)

    ops = structure.spacegroup.operations()
    assert not any(ops.is_systematically_absent(list(h)) for h in hkl)
    # p21c.cif reports _reflns_number_total 10786
    assert len(hkl) == 10786
