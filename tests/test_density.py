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
    CUBE_MASK_BLOCK,
    DEFAULT_GRID_SPACING,
    DEFAULT_SIGMA,
    DEFAULT_WEAK_WEIGHT,
    HAS_DENSITY_CPP,
    ResidualDensityMap,
    calculate_residual_density,
    force_isotropic_adps,
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

def test_extinction_matches_the_per_reflection_formula():
    """The vectorised correction reproduces SHELXL's formula exactly.

    ``_apply_extinction`` evaluates the whole reflection array at once; this
    pins it against a literal, one-reflection-at-a-time transcription of

        Fc* = Fc (1 + 0.001 x Fc² λ³ / sin 2θ)^(-1/4)
    """
    from math import sqrt

    from fastmolwidget.density import _apply_extinction
    from fastmolwidget.hkl_io import ShelxParameters

    cell = gemmi.UnitCell(10.1, 12.3, 14.7, 90.0, 101.2, 90.0)
    hkl = np.asarray(
        gemmi.make_miller_array(cell, gemmi.SpaceGroup('P 21/c'), 0.9),
        dtype=np.int32)
    rng = np.random.default_rng(0)
    f_calc = (rng.normal(size=len(hkl)) + 1j * rng.normal(size=len(hkl))) * 50.0
    params = ShelxParameters(exti=0.0038, wavelength=0.71073)

    expected = np.empty_like(f_calc)
    for i, index in enumerate(hkl):
        d = cell.calculate_d(list(index))
        sin_theta = min(params.wavelength / (2.0 * d), 1.0)
        sin_2theta = max(
            2.0 * sin_theta * sqrt(max(1.0 - sin_theta ** 2, 0.0)), 1e-6)
        expected[i] = f_calc[i] * (
            1.0 + 0.001 * params.exti * abs(f_calc[i]) ** 2
            * params.wavelength ** 3 / sin_2theta) ** -0.25

    assert np.allclose(_apply_extinction(f_calc, hkl, cell, params), expected,
                       rtol=1e-12, atol=1e-12)


def test_extinction_is_skipped_without_a_refined_exti():
    """No EXTI card means the amplitudes are handed back untouched."""
    from fastmolwidget.density import _apply_extinction
    from fastmolwidget.hkl_io import ShelxParameters

    cell = gemmi.UnitCell(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    hkl = np.array([[1, 0, 0], [0, 2, 1]], dtype=np.int32)
    f_calc = np.array([3.0 + 1.0j, -2.0 + 0.5j])

    result = _apply_extinction(f_calc, hkl, cell, ShelxParameters())

    assert result is f_calc


@needs_cpp
def test_cube_mask_does_not_change_the_surface(density_map, shelx_structure):
    """Confining marching cubes to the occupied blocks is exactly lossless.

    The mask only decides which cubes are visited; every cube that could carry
    a vertex within ``margin`` of an atom must still be visited, so the
    clipped surface has to come out exactly as it does without a mask.
    """
    from fastmolwidget import density_cpp
    from fastmolwidget.density import _clip_to_atoms

    atoms = np.array([
        [p.x, p.y, p.z] for p in
        (shelx_structure.cell.orthogonalize(s.fract)
         for s in shelx_structure.sites)
    ])
    margin = 1.5
    level = density_map.sigma_level()
    sub, origin, step = density_map._region(atoms, margin)
    mask = density_map._cube_mask(sub.shape, origin, step, atoms, margin)
    assert mask is not None
    assert not mask.all()  # otherwise the test proves nothing

    arguments = (sub, float(level), tuple(map(float, origin)),
                 tuple(map(float, step)))
    plain = density_cpp.marching_cubes(*arguments)
    masked = density_cpp.marching_cubes(*arguments, mask=mask,
                                        block=CUBE_MASK_BLOCK)
    assert len(masked[0]) < len(plain[0])  # empty space really is skipped

    expected = _clip_to_atoms(plain[0] @ density_map.orth_matrix.T, plain[1],
                              atoms, margin)
    actual = _clip_to_atoms(masked[0] @ density_map.orth_matrix.T, masked[1],
                            atoms, margin)
    assert np.array_equal(expected[0], actual[0])
    assert np.array_equal(expected[1], actual[1])


@needs_cpp
def test_isosurfaces_matches_separate_isosurface_calls(density_map):
    """Asking for both lobes at once gives what two separate calls give."""
    level = density_map.sigma_level()

    both = density_map.isosurfaces((level, -level))
    separate = [density_map.isosurface(level), density_map.isosurface(-level)]

    for (verts, edges), (ref_verts, ref_edges) in zip(both, separate):
        assert np.array_equal(verts, ref_verts)
        assert np.array_equal(edges, ref_edges)


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


# ---------------------------------------------------------------------------
# Down-weighting of weak data (the map's only smoothing)
# ---------------------------------------------------------------------------

def test_weak_data_damping_formula():
    """The factor is ``1 / (1 + w·(σ/|Fc|)³)``."""
    from fastmolwidget.density import WEAK_DATA_EXPONENT, _weak_data_damping

    sigma = np.array([0.0, 1.0, 2.0, 5.0])
    f_calc = np.array([10.0, 10.0, 10.0, 10.0])
    factors = _weak_data_damping(sigma, f_calc, 1.0)

    assert WEAK_DATA_EXPONENT == 3.0
    assert factors == pytest.approx(
        1.0 / (1.0 + (sigma / f_calc) ** WEAK_DATA_EXPONENT))
    assert factors[0] == 1.0
    assert np.all(np.diff(factors) < 0)      # weaker data is damped more


def test_weak_data_damping_ignores_vanishing_fcalc():
    """A zero |Fc| makes the ratio meaningless - leave the term alone."""
    from fastmolwidget.density import _weak_data_damping

    factors = _weak_data_damping(np.array([3.0, 3.0]),
                                 np.array([0.0, 1.0]), 1.0)

    assert factors[0] == 1.0
    assert factors[1] < 0.05


def test_weak_data_damping_can_be_switched_off():
    from fastmolwidget.density import _weak_data_damping

    factors = _weak_data_damping(np.array([5.0, 50.0]),
                                 np.array([1.0, 1.0]), 0.0)

    assert factors == pytest.approx(1.0)


def test_weak_weight_damps_the_map():
    """Down-weighting weak data may only ever remove density, never add it."""
    undamped = calculate_residual_density(RES, HKL, weak_weight=0.0)
    damped = calculate_residual_density(RES, HKL, weak_weight=8.0)

    assert damped.rms < undamped.rms
    assert damped.max <= undamped.max + 1e-9
    assert damped.array.shape == undamped.array.shape


def test_default_map_is_damped(density_map):
    """The default really is DEFAULT_WEAK_WEIGHT, not 'no filter'."""
    explicit = calculate_residual_density(RES, HKL,
                                          weak_weight=DEFAULT_WEAK_WEIGHT)

    assert explicit.rms == pytest.approx(density_map.rms, abs=1e-9)


def test_damping_is_skipped_without_standard_uncertainties(monkeypatch):
    """Placeholder σ values must not be mistaken for real ones."""
    from fastmolwidget import density as density_module
    from fastmolwidget.hkl_io import read_reflections

    def without_sigma(path):
        data = read_reflections(path)
        data.sigma_known = False
        return data

    monkeypatch.setattr(density_module, 'read_reflections', without_sigma)
    unfiltered = calculate_residual_density(RES, HKL, weak_weight=8.0)

    assert unfiltered.rms == pytest.approx(
        calculate_residual_density(RES, HKL, weak_weight=0.0).rms, abs=1e-9)


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


def test_sigma_level_scales_with_the_map(density_map):
    assert density_map.sigma_level(3.0) == pytest.approx(
        round(3.0 * density_map.rms, 2))
    assert density_map.sigma_level(1.0) < density_map.sigma_level(3.0)


def test_sigma_level_is_never_zero():
    """A featureless map must still give the spin box a usable value."""
    flat = ResidualDensityMap(array=np.zeros((4, 4, 4), dtype=np.float32),
                              cell=(10, 10, 10, 90, 90, 90),
                              d_min=1.0, scale=1.0)

    assert flat.sigma_level() == 0.01


def test_default_sigma_is_the_crystallographic_three():
    assert DEFAULT_SIGMA == 3.0


# ---------------------------------------------------------------------------
# SHELX lattice centring
# ---------------------------------------------------------------------------

def test_latt_centring_is_applied(tmp_path):
    """``LATT`` adds centring translations that ``SYMM`` alone does not.

    Dropping them silently yields a primitive subgroup — C2/c would become
    P2/c — which halves the symmetry mates and makes every Fc wrong.
    """
    from shelxfile import Shelxfile

    res = tmp_path / 'c2c.res'
    res.write_text(
        'TITL c2c\n'
        'CELL 0.71073 28.8539 11.3652 9.0813 90 93.83 90\n'
        'ZERR 8 0 0 0 0 0 0\n'
        'LATT 7\n'                     # C-centred, centrosymmetric
        'SYMM -X, Y, 1/2-Z\n'
        'SFAC C\n'
        'UNIT 8\n'
        'FVAR 0.3\n'
        'C1 1 0.1 0.2 0.3 11.0 0.05\n'
        'HKLF 4\n'
    )
    shx = Shelxfile()
    shx.read_file(str(res))

    structure = small_structure_from_shelx(shx)

    assert structure.spacegroup is not None
    assert structure.spacegroup.xhm() == 'C 1 2/c 1'
    assert len(structure.spacegroup.operations()) == 8


def test_primitive_latt_is_unchanged():
    """LATT -1 has no centring and no inversion; P31c must stay P31c."""
    from shelxfile import Shelxfile

    shx = Shelxfile()
    shx.read_file(str(RES))

    structure = small_structure_from_shelx(shx)

    assert structure.spacegroup.xhm() == 'P 3 1 c'


# ---------------------------------------------------------------------------
# Twinning
# ---------------------------------------------------------------------------

def _twinned_copy(tmp_path, *, basf: float, hklf: int = 4,
                  matrix: str = '0 1 0 1 0 0 0 0 -1') -> Path:
    """Return a copy of the p31c model declared as a twin."""
    import shutil

    text = RES.read_text(errors='replace')
    text = text.replace('HKLF 4', f'TWIN {matrix} 2\nBASF {basf}\nHKLF {hklf}')
    model = tmp_path / 'twin.res'
    model.write_text(text)
    shutil.copy(HKL, tmp_path / 'twin.hkl')
    return model


def test_twin_law_is_read(tmp_path):
    from fastmolwidget.hkl_io import read_shelx_parameters

    params = read_shelx_parameters(_twinned_copy(tmp_path, basf=0.25))

    assert params.is_twinned
    assert params.twin_components == 2
    assert params.basf == pytest.approx([0.25])
    assert params.twin_fractions() == pytest.approx([0.75, 0.25])


def test_bare_twin_card_means_racemic(tmp_path):
    """``TWIN`` without a matrix is SHELXL's inversion twin."""
    from fastmolwidget.hkl_io import _DEFAULT_TWIN_MATRIX, read_shelx_parameters

    text = RES.read_text(errors='replace').replace(
        'HKLF 4', 'TWIN\nBASF 0.4\nHKLF 4')
    model = tmp_path / 'racemic.res'
    model.write_text(text)

    params = read_shelx_parameters(model)

    assert params.twin_matrix == _DEFAULT_TWIN_MATRIX


def test_zero_basf_detwinning_changes_nothing(tmp_path):
    """A twin whose second domain has zero volume is the untwinned case."""
    model = _twinned_copy(tmp_path, basf=0.0)

    plain = calculate_residual_density(RES, HKL)
    twinned = calculate_residual_density(model, tmp_path / 'twin.hkl')

    assert twinned.rms == pytest.approx(plain.rms, abs=1e-3)
    assert twinned.max == pytest.approx(plain.max, abs=5e-3)


def test_detwinning_changes_the_map(tmp_path):
    """A real twin fraction must actually redistribute the intensities."""
    plain = calculate_residual_density(RES, HKL)
    twinned = calculate_residual_density(_twinned_copy(tmp_path, basf=0.35),
                                         tmp_path / 'twin.hkl')

    assert twinned.rms != pytest.approx(plain.rms, rel=0.02)


def test_twin_domain_indices_follow_the_law():
    """``TWIN`` uses the same convention as ``HKLF``: ``h' = M h``.

    The matrix here is deliberately **asymmetric** — a symmetric twin law
    gives the same answer under either convention, so it would not catch a
    transposed matrix.
    """
    from fastmolwidget.density import _twin_domain_indices

    # h' = h + 2k, k' = k, l' = l   (determinant +1, not symmetric)
    law = (1, 2, 0,
           0, 1, 0,
           0, 0, 1)
    hkl = np.array([[1, 2, 3]], dtype=np.int32)

    domains = _twin_domain_indices(hkl, law, 2)

    assert len(domains) == 2
    assert list(domains[0][0]) == [1, 2, 3]
    # M h = (1 + 2*2, 2, 3); the transposed convention would give (1, 4, 3)
    assert list(domains[1][0]) == [5, 2, 3]


def test_twin_and_hklf_use_the_same_convention():
    """Both cards must transform an index the same way."""
    from fastmolwidget.density import _apply_hklf_transform, _twin_domain_indices
    from fastmolwidget.hkl_io import ReflectionData, ShelxParameters

    law = (1, 2, 0, 0, 1, 0, 0, 0, 1)
    hkl = np.array([[1, 2, 3]], dtype=np.int32)

    via_twin = _twin_domain_indices(hkl, law, 2)[1][0]
    via_hklf = _apply_hklf_transform(
        ReflectionData(hkl=hkl, f_sq_meas=np.array([1.0]),
                       sigma=np.array([1.0])),
        ShelxParameters(hklf_matrix=law),
    ).hkl[0]

    assert list(via_twin) == list(via_hklf)


def test_negative_component_count_is_racemic(tmp_path):
    """A negative TWIN count means general *and* racemic twinning.

    ``|n|`` components in total: the matrix generates ``1…m`` and components
    ``m+1…2m`` are their Friedel opposites.
    """
    from fastmolwidget.hkl_io import read_shelx_parameters

    text = RES.read_text(errors='replace').replace(
        'HKLF 4', 'TWIN 0 1 0 1 0 0 0 0 -1 -4\nBASF 0.2 0.1 0.05\nHKLF 4')
    model = tmp_path / 'negtwin.res'
    model.write_text(text)
    import shutil
    shutil.copy(HKL, tmp_path / 'negtwin.hkl')

    params = read_shelx_parameters(model)
    assert params.twin_racemic
    assert params.twin_components == 4
    assert params.twin_fractions() == pytest.approx([0.65, 0.2, 0.1, 0.05])

    # must compute without warning about an unsupported ordering
    density = calculate_residual_density(model, tmp_path / 'negtwin.hkl')
    assert density.rms > 0


def test_racemic_components_are_friedel_opposites():
    from fastmolwidget.density import _twin_domain_indices

    law = (0, 1, 0, 1, 0, 0, 0, 0, -1)
    hkl = np.array([[1, 2, 3]], dtype=np.int32)

    domains = _twin_domain_indices(hkl, law, 4, racemic=True)

    assert len(domains) == 4
    assert list(domains[0][0]) == [1, 2, 3]
    assert list(domains[1][0]) == [2, 1, -3]
    assert list(domains[2][0]) == [-1, -2, -3]     # Friedel of component 1
    assert list(domains[3][0]) == [-2, -1, 3]      # Friedel of component 2


def test_missing_basf_means_perfect_twinning(tmp_path):
    """Without BASF all components share the volume equally."""
    from fastmolwidget.hkl_io import read_shelx_parameters

    text = RES.read_text(errors='replace').replace(
        'HKLF 4', 'TWIN 0 1 0 1 0 0 0 0 -1 3\nHKLF 4')
    model = tmp_path / 'perfect.res'
    model.write_text(text)

    params = read_shelx_parameters(model)

    assert params.twin_fractions() == pytest.approx([1 / 3, 1 / 3, 1 / 3])


def test_negative_batch_in_hklf4_is_not_an_overlap_group(tmp_path):
    """In HKLF 4 a negative batch number flags an R-free reflection.

    Treating it as an HKLF 5 overlap marker would silently merge unrelated
    reflections into one observation.
    """
    import shutil

    from fastmolwidget.hkl_io import read_shelx_hkl

    data = read_shelx_hkl(HKL)
    lines = []
    for position, ((h, k, l), f_sq, sigma) in enumerate(
            zip(data.hkl, data.f_sq_meas, data.sigma)):
        batch = -1 if position % 3 == 0 else 1     # every third is R-free
        lines.append(f'{h:4d}{k:4d}{l:4d}{f_sq:8.2f}{sigma:8.2f}{batch:4d}')
    lines.append(f'{0:4d}{0:4d}{0:4d}{0.0:8.2f}{0.0:8.2f}{0:4d}')
    (tmp_path / 'rfree.hkl').write_text('\n'.join(lines))
    shutil.copy(RES, tmp_path / 'rfree.res')

    flagged = read_shelx_hkl(tmp_path / 'rfree.hkl')
    assert flagged.has_overlap_groups          # negative numbers are present

    plain = calculate_residual_density(RES, HKL)
    with_flags = calculate_residual_density(tmp_path / 'rfree.res',
                                            tmp_path / 'rfree.hkl')

    assert with_flags.rms == pytest.approx(plain.rms, abs=1e-3)


# ---------------------------------------------------------------------------
# HKLF index transformation
# ---------------------------------------------------------------------------

def test_hklf_matrix_reindexes_the_data(tmp_path):
    """``HKLF 4 1 r11..r33`` transforms h before anything else uses it."""
    from fastmolwidget.hkl_io import read_shelx_hkl, read_shelx_parameters

    # Swap h and k in the file, and declare the inverse swap on the HKLF card
    # so the data lands back on the model's setting.
    data = read_shelx_hkl(HKL)
    lines = []
    for (h, k, l), f_sq, sigma in zip(data.hkl, data.f_sq_meas, data.sigma):
        lines.append(f'{k:4d}{h:4d}{-l:4d}{f_sq:8.2f}{sigma:8.2f}{1:4d}')
    lines.append(f'{0:4d}{0:4d}{0:4d}{0.0:8.2f}{0.0:8.2f}{0:4d}')
    (tmp_path / 'swap.hkl').write_text('\n'.join(lines))

    text = RES.read_text(errors='replace').replace(
        'HKLF 4', 'HKLF 4 1 0 1 0 1 0 0 0 0 -1')
    (tmp_path / 'swap.res').write_text(text)

    params = read_shelx_parameters(tmp_path / 'swap.res')
    assert params.hklf_matrix == (0, 1, 0, 1, 0, 0, 0, 0, -1)

    plain = calculate_residual_density(RES, HKL)
    reindexed = calculate_residual_density(tmp_path / 'swap.res',
                                           tmp_path / 'swap.hkl')

    assert reindexed.rms == pytest.approx(plain.rms, abs=1e-3)
    assert reindexed.max == pytest.approx(plain.max, abs=0.01)


def test_hklf_transform_follows_the_row_convention():
    """``h' = r11·h + r12·k + r13·l``, as the SHELXL manual states."""
    from fastmolwidget.density import _apply_hklf_transform
    from fastmolwidget.hkl_io import ReflectionData, ShelxParameters

    data = ReflectionData(hkl=np.array([[1, 2, 3]], dtype=np.int32),
                          f_sq_meas=np.array([10.0]),
                          sigma=np.array([1.0]))
    # swaps h and k; determinant is +1 with the l row negated twice over
    params = ShelxParameters(hklf_matrix=(0, 1, 0, 1, 0, 0, 0, 0, -1))

    out = _apply_hklf_transform(data, params)

    assert list(out.hkl[0]) == [2, 1, -3]


def test_hklf_scale_factors_are_applied():
    from fastmolwidget.density import _apply_hklf_transform
    from fastmolwidget.hkl_io import ReflectionData, ShelxParameters

    data = ReflectionData(hkl=np.array([[1, 0, 0]], dtype=np.int32),
                          f_sq_meas=np.array([10.0]),
                          sigma=np.array([2.0]))
    params = ShelxParameters(hklf_scale=3.0, hklf_sigma_scale=5.0)

    out = _apply_hklf_transform(data, params)

    assert out.f_sq_meas[0] == pytest.approx(30.0)
    assert out.sigma[0] == pytest.approx(2.0 * 3.0 * 5.0)


def test_hklf_matrix_must_have_positive_determinant():
    from fastmolwidget.density import _apply_hklf_transform
    from fastmolwidget.hkl_io import ReflectionData, ShelxParameters

    data = ReflectionData(hkl=np.array([[1, 0, 0]], dtype=np.int32),
                          f_sq_meas=np.array([10.0]),
                          sigma=np.array([1.0]))
    params = ShelxParameters(hklf_matrix=(-1, 0, 0, 0, 1, 0, 0, 0, 1))

    with pytest.raises(ValueError, match='positive determinant'):
        _apply_hklf_transform(data, params)


def test_default_hklf_card_changes_nothing():
    from fastmolwidget.hkl_io import read_shelx_parameters

    params = read_shelx_parameters(RES)

    assert params.hklf == 4
    assert params.hklf_matrix is None
    assert params.hklf_scale == 1.0


def test_hklf5_groups_are_parsed(tmp_path):
    """Consecutive records ending in a positive batch form one observation."""
    from fastmolwidget.density import _hklf5_groups
    from fastmolwidget.hkl_io import parse_shelx_hkl

    text = (
        '   1   2   3   10.00    1.00  -2\n'
        '   1   2   3   10.00    1.00   1\n'
        '   4   5   6   20.00    2.00   1\n'
        '   0   0   0    0.00    0.00   0\n'
    )
    data = parse_shelx_hkl(text)
    assert data.has_overlap_groups

    groups = list(_hklf5_groups(data))

    assert len(groups) == 2
    primary, members, f_sq, _sigma = groups[0]
    assert primary == (1, 2, 3)          # the domain-1 record
    assert [c for c, _ in members] == [1, 0]
    assert f_sq == pytest.approx(10.0)
    assert groups[1][0] == (4, 5, 6)


def test_hklf5_reproduces_the_untwinned_map(tmp_path):
    """An HKLF 5 file whose second domain has zero weight is a round trip."""
    from fastmolwidget.hkl_io import read_shelx_hkl

    text = RES.read_text(errors='replace').replace(
        'HKLF 4', 'BASF 0.0\nHKLF 5')
    model = tmp_path / 'twin5.res'
    model.write_text(text)

    data = read_shelx_hkl(HKL)
    lines = []
    for (h, k, l), f_sq, sigma in zip(data.hkl, data.f_sq_meas, data.sigma):
        lines.append(f'{h:4d}{k:4d}{l:4d}{f_sq:8.2f}{sigma:8.2f}{-2:4d}')
        lines.append(f'{h:4d}{k:4d}{l:4d}{f_sq:8.2f}{sigma:8.2f}{1:4d}')
    lines.append(f'{0:4d}{0:4d}{0:4d}{0.0:8.2f}{0.0:8.2f}{0:4d}')
    (tmp_path / 'twin5.hkl').write_text('\n'.join(lines))

    plain = calculate_residual_density(RES, HKL)
    from_hklf5 = calculate_residual_density(model, tmp_path / 'twin5.hkl')

    assert from_hklf5.array.shape == plain.array.shape
    assert np.abs(from_hklf5.array - plain.array).max() < 0.01


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


def test_force_isotropic_adps_flattens_every_site(shelx_structure):
    flat = force_isotropic_adps(shelx_structure, 0.045)
    assert len(flat.sites) == len(shelx_structure.sites)
    for site in flat.sites:
        assert site.aniso.u11 == 0.0 and site.aniso.u22 == 0.0 and site.aniso.u33 == 0.0
        assert site.u_iso == pytest.approx(0.045)
    # The original structure is untouched.
    assert any(s.aniso.u11 != 0.0 for s in shelx_structure.sites)


def test_force_isotropic_adps_keeps_positions_and_elements(shelx_structure):
    flat = force_isotropic_adps(shelx_structure, 0.05)
    for original, flattened in zip(shelx_structure.sites, flat.sites):
        assert flattened.label == original.label
        assert flattened.element.name == original.element.name
        assert flattened.fract.x == pytest.approx(original.fract.x)
        assert flattened.fract.y == pytest.approx(original.fract.y)
        assert flattened.fract.z == pytest.approx(original.fract.z)
        assert flattened.occ == pytest.approx(original.occ)


def test_calculate_residual_density_accepts_iso_u_override():
    """A dedicated map for disorder-fitting: same reflections, flattened ADPs."""
    normal = calculate_residual_density(RES, HKL)
    flattened = calculate_residual_density(RES, HKL, iso_u_override=0.045)
    assert flattened.array.shape == normal.array.shape
    # Flattening the ADPs changes Fc and therefore the map - it should not be
    # byte-identical to the normal map.
    assert not np.allclose(flattened.array, normal.array)


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
    hkl, _, _ = _merge_to_asu(data, structure.spacegroup, None, structure.cell)

    ops = structure.spacegroup.operations()
    assert not any(ops.is_systematically_absent(list(h)) for h in hkl)
    # p21c.cif reports _reflns_number_total 10786
    assert len(hkl) == 10786


# ---------------------------------------------------------------------------
# Performance-critical rewrites: the fast paths must equal the reference ones
# ---------------------------------------------------------------------------

@needs_cpp
@pytest.mark.parametrize('model, reflections', [(RES, HKL), (P21C, P21C)])
def test_fast_structure_factors_match_gemmi(model, reflections):
    """The C++ summation must reproduce gemmi's reference implementation."""
    from fastmolwidget.density import (
        _load_model,
        _merge_to_asu,
        _summed_structure_factors,
    )
    from fastmolwidget.hkl_io import read_reflections

    structure, params = _load_model(Path(model))
    data = read_reflections(reflections)
    hkl, _, _ = _merge_to_asu(data, structure.spacegroup, None, structure.cell)

    calculator = gemmi.StructureFactorCalculatorX(structure.cell)
    if params.wavelength:
        calculator.addends.add_cl_fprime(gemmi.hc / params.wavelength)
    expected = np.array([
        calculator.calculate_sf_from_small_structure(structure, list(h))
        for h in hkl
    ])

    result = _summed_structure_factors(structure, hkl, calculator)

    # gemmi stores the IT92 coefficients as float32, so the two differ by
    # single-precision round-off and no more.
    assert result.shape == expected.shape
    assert np.abs(result - expected).max() < 1e-6 * np.abs(expected).max()


@needs_cpp
def test_structure_factors_handle_an_isotropic_only_model():
    """The anisotropic branch must not be needed for an all-isotropic model."""
    from fastmolwidget.density import _merge_to_asu, _summed_structure_factors
    from fastmolwidget.hkl_io import read_reflections

    structure = small_structure_from_cif(CIF)
    for site in structure.sites:
        site.aniso = gemmi.SMat33d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        site.u_iso = 0.03
    data = read_reflections(HKL)
    hkl, _, _ = _merge_to_asu(data, structure.spacegroup, None, structure.cell)

    calculator = gemmi.StructureFactorCalculatorX(structure.cell)
    expected = np.array([
        calculator.calculate_sf_from_small_structure(structure, list(h))
        for h in hkl
    ])

    result = _summed_structure_factors(structure, hkl, calculator)

    assert np.abs(result - expected).max() < 1e-6 * np.abs(expected).max()


def test_merged_reflections_carry_the_weighted_mean():
    """Merging must average symmetry equivalents with 1/sigma^2 weights."""
    from fastmolwidget.density import _merge_to_asu
    from fastmolwidget.hkl_io import ReflectionData

    structure = small_structure_from_cif(CIF)
    # (1, 2, 3) and its Friedel mate are one unique reflection.
    data = ReflectionData(
        hkl=np.array([[1, 2, 3], [-1, -2, -3]], dtype=np.int32),
        f_sq_meas=np.array([100.0, 200.0]),
        sigma=np.array([1.0, 2.0]),
    )

    hkl, f_obs, _ = _merge_to_asu(data, structure.spacegroup, None, structure.cell)

    weights = np.array([1.0, 0.25])
    expected = np.sqrt((weights * [100.0, 200.0]).sum() / weights.sum())
    assert len(hkl) == 1
    assert f_obs[0] == pytest.approx(expected)


def test_equivalence_classes_agree_with_gemmis_asu():
    """Every class must be exactly one reciprocal-asymmetric-unit reflection."""
    from fastmolwidget.density import _equivalence_classes
    from fastmolwidget.hkl_io import read_reflections

    structure = small_structure_from_cif(P21C)
    ops = structure.spacegroup.operations()
    asu = gemmi.ReciprocalAsu(structure.spacegroup)
    hkl = read_reflections(P21C).hkl[:2000]

    labels, representatives = _equivalence_classes(hkl, ops)

    reference = np.array([asu.to_asu([int(h), int(k), int(l)], ops)[0]
                          for h, k, l in hkl])
    # Two observations share a label exactly when they share an ASU index.
    by_label = np.array([asu.to_asu([int(h), int(k), int(l)], ops)[0]
                         for h, k, l in representatives])
    assert np.array_equal(by_label[labels], reference)
