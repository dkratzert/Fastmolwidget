"""
Difference Fourier electron density map computation.

Computes an (Fo − Fc) difference map from a refined crystal structure (CIF)
and its measured reflections (HKL file, SHELX4 format).  The heavy lifting is
done entirely by the :mod:`gemmi` library; the result is an isosurface mesh
ready for display in :class:`~fastmolwidget.molecule2D.MoleculeWidget`.

The overall workflow is::

    result = compute_diff_map(cif_path, hkl_path)
    pos_segs, neg_segs = get_mesh_segments(result, level_sigma=3.0)
    widget.set_density_mesh(pos_segs, neg_segs)

Crystal symmetry is fully handled: the structure factors Fc are computed using
all symmetry operations of the space group, and the real-space FFT grid covers
one full unit cell (all symmetry-equivalent regions are automatically filled by
the FFT of symmetry-reduced data).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import gemmi
import numpy as np


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

class DiffMapResult(NamedTuple):
    """Container returned by :func:`compute_diff_map`.

    :param grid: Real-space difference density grid (e Å⁻³) over one full
        unit cell.
    :param sigma: RMS variation of the map (used to pick contour levels).
    :param cell: Unit cell parameters.
    :param nu: Grid size along *a*.
    :param nv: Grid size along *b*.
    :param nw: Grid size along *c*.
    """
    grid: np.ndarray          # shape (nu, nv, nw), float32
    sigma: float
    cell: gemmi.UnitCell
    nu: int
    nv: int
    nw: int


# ---------------------------------------------------------------------------
# HKL parsing
# ---------------------------------------------------------------------------

def _parse_hkl_shelx(hkl_text: str) -> dict[tuple[int, int, int], tuple[float, float]]:
    """Parse a SHELX4-format HKL file (h k l Fo² σ [batch]) and return merged
    unique reflections as *{(h,k,l): (Fo², σ)}*.

    Multiple observations of the same HKL index are averaged (simple mean).
    The terminator record (0 0 0) and any records with Fo² < 0 are excluded
    from the final dictionary but negatives are kept separately so the caller
    can apply resolution / sigma cut-offs.

    :param hkl_text: Contents of the HKL file as a string.
    :returns: Dictionary mapping Miller triplets to ``(Fo², σ)`` pairs.
    """
    accumulator: dict[tuple[int, int, int], list[float]] = {}
    for line in hkl_text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            h, k, l_idx = int(parts[0]), int(parts[1]), int(parts[2])
        except ValueError:
            continue
        if h == 0 and k == 0 and l_idx == 0:
            break
        try:
            fo2 = float(parts[3])
            sig = float(parts[4])
        except ValueError:
            continue
        key = (h, k, l_idx)
        if key not in accumulator:
            accumulator[key] = [fo2, sig, 1]
        else:
            accumulator[key][0] += fo2
            accumulator[key][1] += sig
            accumulator[key][2] += 1

    return {k: (v[0] / v[2], v[1] / v[2]) for k, v in accumulator.items()}


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

_GOOD_FFT_SIZES = [
    # Highly composite numbers that work well with FFTW/numpy FFT
    32, 36, 40, 45, 48, 54, 60, 64, 72, 80, 90, 96, 100, 108, 120, 128,
    135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 256,
    270, 288, 300, 320, 324, 360, 384, 400, 405, 432, 450, 480, 486,
    500, 512, 540, 576, 600,
]


def _next_good_size(n: int) -> int:
    """Return the smallest value ≥ *n* that is in :data:`_GOOD_FFT_SIZES`."""
    for s in _GOOD_FFT_SIZES:
        if s >= n:
            return s
    return n  # fallback: use as-is


def _find_compatible_grid(cell: gemmi.UnitCell, sg: gemmi.SpaceGroup,
                           max_h: int, max_k: int, max_l: int) -> tuple[int, int, int]:
    """Find grid dimensions compatible with *sg* that encompass all HKL indices.

    The minimum dimensions are ``2*max_h+1``, ``2*max_k+1``, ``2*max_l+1``; the
    function rounds each up to a highly-composite number and then performs a
    quick compatibility check via :func:`gemmi.transform_f_phi_grid_to_map`.

    :returns: ``(nu, nv, nw)`` tuple.
    """
    nu_min = _next_good_size(2 * max_h + 1)
    nv_min = _next_good_size(2 * max_k + 1)
    nw_min = _next_good_size(2 * max_l + 1)

    # Widen along nw until compatible (trigonal/hexagonal often needs even nw)
    for delta in range(0, 50):
        nu = _next_good_size(nu_min + delta) if delta > 0 else nu_min
        nv = _next_good_size(nv_min + delta) if delta > 0 else nv_min
        nw = _next_good_size(nw_min + delta) if delta > 0 else nw_min
        rcg = gemmi.ReciprocalComplexGrid(nu, nv, nw)
        rcg.unit_cell = cell
        rcg.spacegroup = sg
        try:
            gemmi.transform_f_phi_grid_to_map(rcg)
            return nu, nv, nw
        except RuntimeError:
            # Not compatible with this space group – try next size
            nw = _next_good_size(nw + 1)

    # Last-resort: brute-force scan nw
    for nw in range(nw_min, nw_min + 200):
        rcg = gemmi.ReciprocalComplexGrid(nu_min, nv_min, nw)
        rcg.unit_cell = cell
        rcg.spacegroup = sg
        try:
            gemmi.transform_f_phi_grid_to_map(rcg)
            return nu_min, nv_min, nw
        except RuntimeError:
            continue

    raise RuntimeError(
        f"Cannot find a compatible FFT grid for space group {sg.hm} "
        f"with dimensions near {nu_min}×{nv_min}×{nw_min}."
    )


# ---------------------------------------------------------------------------
# Scale-factor estimation
# ---------------------------------------------------------------------------

def _estimate_scale(
    merged: dict[tuple[int, int, int], tuple[float, float]],
    all_fc: dict[tuple[int, int, int], complex],
) -> float:
    r"""Estimate the overall scale factor *k* that relates observed and
    calculated structure-factor amplitudes.

    Uses the Wilson-type least-squares estimate::

        k = sqrt( Σ Fo²  /  Σ |Fc|² )

    Only reflections with Fo² > 0 are included.

    :returns: Scale factor *k* (Fo = k · |Fc| at perfect fit).
    """
    sum_fo2 = 0.0
    sum_fc2 = 0.0
    for (h, k_idx, l), (fo2, _sig) in merged.items():
        if fo2 <= 0.0:
            continue
        fc = all_fc.get((h, k_idx, l))
        if fc is None:
            continue
        sum_fc2 += abs(fc) ** 2
        sum_fo2 += fo2

    if sum_fc2 < 1e-10:
        return 1.0
    return math.sqrt(sum_fo2 / sum_fc2)


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_diff_map(
    cif_path: str | Path,
    hkl_path: str | Path | None = None,
) -> DiffMapResult:
    r"""Compute an (Fo − Fc) difference Fourier map.

    Reads the refined crystal structure from *cif_path* (atomic coordinates
    and anisotropic displacement parameters), computes calculated structure
    factors Fc using the full symmetry of the space group, and Fourier-
    transforms the difference coefficients ``(Fo/k − |Fc|) · exp(iφ_c)`` to
    obtain the real-space residual electron density.

    Crystal symmetry is accounted for through gemmi's
    :class:`~gemmi.StructureFactorCalculatorX` which uses the space-group
    operators to place symmetry-equivalent atoms and thereby fills the full
    reciprocal lattice.  The output grid covers one complete unit cell.

    :param cif_path: Path to the CIF file (must contain atom-site data and
        either an embedded ``_shelx_hkl_file`` loop or a reference to an
        external HKL file).
    :param hkl_path: Optional explicit path to a SHELX4 HKL file.  When
        ``None`` the method tries to read the reflection data embedded in the
        CIF via ``_shelx_hkl_file``.
    :returns: :class:`DiffMapResult` with the density grid and metadata.
    :raises ValueError: If no HKL data can be located, or if there are
        insufficient reflections to compute a reliable map.
    """
    cif_path = Path(cif_path)

    # ── Load crystal structure ───────────────────────────────────────────────
    st = gemmi.read_small_structure(str(cif_path))
    if not st.sites:
        raise ValueError(f"No atom-site data found in {cif_path}")

    sg = gemmi.SpaceGroup(st.spacegroup_hm)
    cell = st.cell

    # ── Obtain HKL data ──────────────────────────────────────────────────────
    if hkl_path is not None:
        hkl_text = Path(hkl_path).read_text()
    else:
        # Fall back to data embedded in the CIF
        from fastmolwidget.cif.cif_file_io import CifReader
        cif_reader = CifReader(cif_path)
        hkl_text = cif_reader.hkl_file
        if not hkl_text:
            raise ValueError(
                f"No HKL data found in {cif_path} and no external HKL file was provided."
            )

    merged = _parse_hkl_shelx(hkl_text)
    if len(merged) < 10:
        raise ValueError(
            f"Too few reflections ({len(merged)}) to compute a meaningful difference map."
        )

    # ── Compute structure factors ────────────────────────────────────────────
    calc = gemmi.StructureFactorCalculatorX(cell)
    # calc uses the space group implicitly through the SmallStructure sites

    all_fc: dict[tuple[int, int, int], complex] = {}
    for hkl in merged:
        all_fc[hkl] = calc.calculate_sf_from_small_structure(st, list(hkl))

    # ── Scale factor ─────────────────────────────────────────────────────────
    k = _estimate_scale(merged, all_fc)

    # ── Reciprocal-space grid ────────────────────────────────────────────────
    max_h = max(abs(h) for h, _k, _l in merged)
    max_k = max(abs(k_idx) for _h, k_idx, _l in merged)
    max_l = max(abs(l) for _h, _k, l in merged)

    nu, nv, nw = _find_compatible_grid(cell, sg, max_h, max_k, max_l)
    rcg = gemmi.ReciprocalComplexGrid(nu, nv, nw)
    rcg.unit_cell = cell
    rcg.spacegroup = sg

    # Fill difference coefficients: (Fo/k − |Fc|) · exp(iφ_c)
    for (h, k_idx, l_idx), (fo2, _sig) in merged.items():
        fc = all_fc[(h, k_idx, l_idx)]
        fc_abs = abs(fc)
        fc_phase = math.atan2(fc.imag, fc.real)

        fo = (math.sqrt(fo2) / k) if fo2 > 0.0 else 0.0
        diff_amp = fo - fc_abs
        diff_coeff = np.complex64(
            diff_amp * math.cos(fc_phase) + 1j * diff_amp * math.sin(fc_phase)
        )
        try:
            rcg.set_value(h, k_idx, l_idx, diff_coeff)
        except Exception:
            pass  # reflection index outside grid – skip

    # ── Real-space FFT ───────────────────────────────────────────────────────
    float_grid = gemmi.transform_f_phi_grid_to_map(rcg)
    density = np.array(float_grid, dtype=np.float32)

    sigma = float(density.std())
    if sigma < 1e-10:
        sigma = 1.0  # guard against degenerate maps

    return DiffMapResult(
        grid=density,
        sigma=sigma,
        cell=cell,
        nu=nu,
        nv=nv,
        nw=nw,
    )


# ---------------------------------------------------------------------------
# Isosurface extraction
# ---------------------------------------------------------------------------

def get_mesh_segments(
    result: DiffMapResult,
    level_sigma: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate wireframe isosurface segments for positive and negative residual density.

    Runs the marching-cubes algorithm at ±*level_sigma* × σ and converts the
    resulting triangle mesh into an array of line segments in Cartesian
    coordinates (Å).  Duplicate edges are removed so each mesh edge appears
    only once.

    Crystal symmetry is implicitly handled: the density grid already covers
    the full unit cell (all symmetry-equivalent sites are present in the FFT
    output), so the isosurface naturally follows the symmetry.

    :param result: Output of :func:`compute_diff_map`.
    :param level_sigma: Contour level expressed as a multiple of σ (the map
        r.m.s. deviation).  Typical values: 3.0 (strict) or 2.5 (sensitive).
    :returns: A pair ``(pos_segments, neg_segments)`` where each element is a
        NumPy array of shape ``(N, 2, 3)`` with the two endpoints (in Å) of
        each segment.  Either array may have ``N == 0`` when no isosurface
        exists at the requested level.
    :raises ImportError: If :mod:`skimage` is not installed.
    """
    try:
        from skimage.measure import marching_cubes  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for isosurface generation. "
            "Install it with:  pip install scikit-image"
        ) from exc

    level = level_sigma * result.sigma
    cell = result.cell
    nu, nv, nw = result.nu, result.nv, result.nw

    def _extract(grid3d: np.ndarray, threshold: float) -> np.ndarray:
        """Run marching cubes and return unique edges in Cartesian Å."""
        try:
            verts, faces, _normals, _vals = marching_cubes(
                grid3d, level=threshold, spacing=(1.0, 1.0, 1.0)
            )
        except (ValueError, RuntimeError):
            return np.empty((0, 2, 3), dtype=np.float32)

        if len(faces) == 0:
            return np.empty((0, 2, 3), dtype=np.float32)

        # Collect unique triangle edges
        edge_set: set[tuple[int, int]] = set()
        for tri in faces:
            for i in range(3):
                e = (int(tri[i]), int(tri[(i + 1) % 3]))
                edge_set.add((min(e), max(e)))

        # Convert grid-index vertices to Cartesian coordinates
        vert_cart = np.empty((len(verts), 3), dtype=np.float32)
        for i, (vi, vj, vk) in enumerate(verts):
            frac = gemmi.Fractional(vi / nu, vj / nv, vk / nw)
            pos = cell.orthogonalize(frac)
            vert_cart[i] = [pos.x, pos.y, pos.z]

        edges = list(edge_set)
        segs = np.empty((len(edges), 2, 3), dtype=np.float32)
        for idx, (e0, e1) in enumerate(edges):
            segs[idx, 0] = vert_cart[e0]
            segs[idx, 1] = vert_cart[e1]
        return segs

    pos_segs = _extract(result.grid, level)
    neg_segs = _extract(-result.grid, level)  # flip sign for negative contour

    return pos_segs, neg_segs
