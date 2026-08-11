"""
Residual (Fo−Fc) electron-density maps computed directly from reflection data.

The map is calculated from a raw reflection file (SHELX ``.hkl`` or an
fcf-style CIF reflection loop) together with the refined atomic model, using
`gemmi <https://gemmi.readthedocs.io>`_ for the structure-factor summation and
the FFT.  No pre-computed map file (``.fcf``, ``.map``) is required.

Method
------
1. Reflections are read with :mod:`fastmolwidget.hkl_io` and merged into the
   reciprocal-space asymmetric unit (σ-weighted mean of F²).
2. *F*\\ :sub:`c` is obtained either from the reflection file itself (fcf-style
   CIFs already contain it) or by direct summation with
   :class:`gemmi.StructureFactorCalculatorX`, including the real part of the
   anomalous dispersion *f′* for the radiation wavelength.
3. SHELXL's refined overall scale factor (the first ``FVAR``) puts the two on a
   common scale — SHELX refines such that ``|Fo| ≈ OSF · |Fc|``, so the map is
   computed on the *F*\\ :sub:`c` (electron) scale using ``|Fo| / OSF``.
4. The difference coefficients are SHELXL's own, **unweighted**, convention::

       ΔF(hkl) = (|Fo| / OSF − |Fc|) · exp(i·φc)

   SHELXL applies the ``WGHT`` scheme only to the least-squares objective, not
   to Fourier map coefficients, so it is deliberately *not* used here.
5. :meth:`gemmi.ComplexAsuData.transform_f_phi_to_map` expands the coefficients
   over the space group and runs the FFT, giving ρ in e/Å³ over one unit cell.

Isosurfaces are extracted with the optional :mod:`density_cpp` marching-cubes
extension.  When it is not compiled in, :attr:`HAS_DENSITY_CPP` is ``False``
and :meth:`ResidualDensityMap.isosurface` raises a clear error instead of
crashing the host application.

This module is Qt-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, sqrt
from pathlib import Path

import gemmi
import numpy as np

from fastmolwidget.hkl_io import (
    ReflectionData,
    ShelxParameters,
    read_reflections,
    read_shelx_parameters,
)

try:
    import density_cpp

    HAS_DENSITY_CPP: bool = True
except ImportError:  # pragma: no cover - depends on the compiled extension
    HAS_DENSITY_CPP = False

__all__ = [
    'HAS_DENSITY_CPP',
    'ResidualDensityMap',
    'calculate_residual_density',
    'small_structure_from_cif',
    'small_structure_from_shelx',
]


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------

@dataclass
class ResidualDensityMap:
    """A residual (Fo−Fc) density map covering one unit cell.

    The map is periodic, so it can be sampled outside ``[0, 1)`` in fractional
    coordinates simply by wrapping the grid indices — which is what
    :meth:`isosurface` does to cover grown or packed molecules.

    :param array: ``(nu, nv, nw)`` grid of ρ in e/Å³, indexed along the *a*,
        *b* and *c* axes.
    :param cell: Unit-cell parameters ``(a, b, c, α, β, γ)``.
    :param d_min: Resolution limit of the data used, in Å.
    :param scale: The scale factor applied to ``|Fo|`` (SHELXL's OSF).
    """

    array: np.ndarray
    cell: tuple[float, float, float, float, float, float]
    d_min: float
    scale: float

    @property
    def rms(self) -> float:
        """Root-mean-square density of the map in e/Å³."""
        return float(self.array.std())

    @property
    def max(self) -> float:
        """Highest (most positive) density in the map, in e/Å³."""
        return float(self.array.max())

    @property
    def min(self) -> float:
        """Lowest (most negative) density in the map, in e/Å³."""
        return float(self.array.min())

    @property
    def orth_matrix(self) -> np.ndarray:
        """``3×3`` matrix converting fractional to Cartesian coordinates."""
        return _orthogonalisation_matrix(self.cell)

    def isosurface(
        self,
        level: float,
        *,
        atoms: np.ndarray | None = None,
        margin: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a wireframe isosurface at *level* in Cartesian coordinates.

        Marching cubes runs in fractional-coordinate space and the resulting
        vertices are transformed to Cartesian afterwards, so triclinic and
        monoclinic cells are handled correctly.

        When *atoms* is given, only the region of the (periodic) map around
        those atoms is contoured.  This keeps the surface aligned with grown
        or packed molecules that extend beyond a single unit cell, and avoids
        drawing density where nothing is displayed.

        :param level: Contour level in e/Å³.  Use a negative value for the
            negative-density surface.
        :param atoms: Optional ``(N, 3)`` array of Cartesian atom positions
            used to restrict the contoured region.  ``None`` contours exactly
            one unit cell.
        :param margin: Padding around *atoms* in Å.
        :returns: ``(vertices, edges)`` — an ``(M, 3)`` float array of
            Cartesian vertex positions and a ``(K, 2)`` integer array of
            deduplicated line segments, ready for ``GL_LINES``.
        :raises RuntimeError: If the :mod:`density_cpp` extension is missing.
        """
        if not HAS_DENSITY_CPP:
            raise RuntimeError(
                'Residual-density isosurfaces need the compiled "density_cpp" '
                'extension. Build it with:  '
                'pip install -e . --no-build-isolation'
            )

        sub, origin_frac, step_frac = self._region(atoms, margin)
        verts, edges = density_cpp.marching_cubes(
            sub, float(level),
            tuple(float(v) for v in origin_frac),
            tuple(float(v) for v in step_frac),
        )
        if len(verts):
            verts = verts @ self.orth_matrix.T
        return np.ascontiguousarray(verts, dtype=np.float32), edges

    def _region(
        self,
        atoms: np.ndarray | None,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cut the periodic map down to the region around *atoms*.

        :returns: ``(sub_grid, origin_fractional, step_fractional)`` where the
            sub-grid is a contiguous copy sampled with wrapped (periodic)
            indices.
        """
        shape = np.array(self.array.shape)
        step = 1.0 / shape

        if atoms is None or len(atoms) == 0:
            # One unit cell, plus one duplicated layer so the surface closes
            # across the periodic boundary.
            sub = np.empty(shape + 1, dtype=np.float32)
            idx = [np.arange(n + 1) % n for n in shape]
            sub[:] = self.array[np.ix_(*idx)]
            return sub, np.zeros(3), step

        frac = np.asarray(atoms, dtype=float) @ np.linalg.inv(self.orth_matrix).T
        pad = margin / np.array(self.cell[:3])
        lo = np.floor((frac.min(axis=0) - pad) * shape).astype(int)
        hi = np.ceil((frac.max(axis=0) + pad) * shape).astype(int) + 1

        idx = [np.arange(a, b) % n for a, b, n in zip(lo, hi, shape)]
        sub = np.ascontiguousarray(self.array[np.ix_(*idx)], dtype=np.float32)
        return sub, lo * step, step


def _orthogonalisation_matrix(
    cell: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Return the ``3×3`` fractional-to-Cartesian matrix for *cell*.

    Uses the standard crystallographic convention with *a* along *x* and *b*
    in the *xy* plane, matching :func:`fastmolwidget.dsrmath.frac_to_cart`.
    """
    a, b, c, alpha, beta, gamma = cell
    ca, cb, cg = cos(radians(alpha)), cos(radians(beta)), cos(radians(gamma))
    sg = sin(radians(gamma))
    volume_term = sqrt(max(1.0 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 1e-12))
    return np.array([
        [a, b * cg, c * cb],
        [0.0, b * sg, c * (ca - cb * cg) / sg],
        [0.0, 0.0, c * volume_term / sg],
    ])


# ---------------------------------------------------------------------------
# Building a gemmi SmallStructure from the refined model
# ---------------------------------------------------------------------------

def small_structure_from_cif(path: str | Path) -> gemmi.SmallStructure:
    """Read a CIF into a :class:`gemmi.SmallStructure` ready for SF calculation.

    ``change_occupancies_to_crystallographic()`` is required by gemmi before
    the structure factors are summed, so that atoms on special positions
    contribute with the correct multiplicity.

    :param path: Path to the CIF file.
    """
    structure = gemmi.read_small_structure(str(path))
    structure.change_occupancies_to_crystallographic()
    return structure


def small_structure_from_shelx(shx) -> gemmi.SmallStructure:
    """Build a :class:`gemmi.SmallStructure` from a parsed SHELX model.

    Two SHELX conventions have to be resolved explicitly:

    * **Negative U_iso** — a riding hydrogen with ``U = -1.5`` means
      *1.5 × U_eq of the atom it rides on*.
    * **Occupancy codes** — the SOF column encodes a free variable and a site
      occupancy factor (``-30.33333`` is ``(1 − FVAR₃) × 0.33333``).

    .. note::
       The occupancies are decoded here from ``atom.sof`` instead of using
       ``atom.occupancy``, because *shelxfile* (as of v24) returns wrong values
       in two cases: ``Atom._get_negative_occupancy`` drops the site-occupancy
       factor for negative SOFs (correct only when it is exactly 1), and an
       ``AFIX`` card without an explicit SOF overwrites a riding atom's own SOF
       with ``11.0``.  Remove :func:`_shelx_occupancies` once *shelxfile*
       resolves both.

    :param shx: A :class:`shelxfile.Shelxfile` that has already read a file.
    :returns: The structure, with the space group and cell filled in.
    :raises ValueError: If the SHELX model has no ``CELL`` instruction.
    """
    if shx.cell is None:
        raise ValueError('SHELX model has no CELL instruction')

    cell = gemmi.UnitCell(shx.cell.a, shx.cell.b, shx.cell.c,
                          shx.cell.alpha, shx.cell.beta, shx.cell.gamma)
    ops = [gemmi.Op(s.to_shelxl().replace(' ', '')) for s in shx.symmcards]
    if shx.latt is not None and shx.latt.centric:
        group = gemmi.GroupOps(ops)
        group.add_inversion()
    else:
        group = gemmi.GroupOps(ops)
    spacegroup = gemmi.find_spacegroup_by_ops(group)

    structure = gemmi.SmallStructure()
    structure.cell = cell
    if spacegroup is not None:
        structure.spacegroup = spacegroup
        structure.spacegroup_hm = spacegroup.xhm()

    occupancies = _shelx_occupancies(shx)
    last_ueq = 0.05
    for atom in shx.atoms:
        if atom.qpeak:
            continue
        site = gemmi.SmallStructure.Site()
        site.label = atom.fullname_short
        site.type_symbol = atom.element.capitalize()
        # type_symbol is only a label - the scattering factors are looked up
        # from .element, which must be set explicitly.
        site.element = gemmi.Element(atom.element.capitalize())
        site.fract = gemmi.Fractional(atom.x, atom.y, atom.z)
        site.occ = occupancies.get(id(atom), 1.0)

        if atom.is_isotropic:
            u_iso = atom.uvals[0] if atom.uvals else 0.05
            if u_iso < 0:  # SHELX riding-U convention
                u_iso = abs(u_iso) * last_ueq
            site.u_iso = u_iso
        else:
            u11, u22, u33, u23, u13, u12 = atom.uvals
            site.aniso = gemmi.SMat33d(u11, u22, u33, u12, u13, u23)
            site.u_iso = atom.ueq
            last_ueq = atom.ueq
        structure.add_site(site)

    # gemmi's structure-factor calculator reads the symmetry from the UnitCell,
    # not from the space group, so the cell images must be generated here.
    structure.setup_cell_images()
    return structure


def _shelx_occupancies(shx) -> dict[int, float]:
    """Decode SHELX SOF codes into crystallographic occupancies.

    A SOF value encodes a free-variable reference *n* and a site occupancy
    factor *p* as ``10·n + p``:

    * ``n = 1``   → ``occ = p`` (fixed, not tied to a free variable)
    * ``n > 1``   → ``occ = FVARₙ · p``
    * ``n < -1``  → ``occ = (1 − FVAR|ₙ|) · p``

    Riding atoms inside an ``AFIX`` group whose SOF was lost by *shelxfile*
    (reported as ``11.0``) inherit the occupancy of the atom they ride on,
    which is what SHELX does.

    :param shx: A parsed :class:`shelxfile.Shelxfile`.
    :returns: Mapping of ``id(atom)`` to occupancy.
    """
    from shelxfile.misc.misc import split_fvar_and_parameter

    free_vars = [f.fvar_value for f in shx.fvars.fvars]
    result: dict[int, float] = {}
    last_heavy = 1.0

    for atom in shx.atoms:
        if atom.qpeak:
            continue
        index, factor = split_fvar_and_parameter(atom.sof)
        n = abs(index)
        p = abs(factor)
        if n <= 1:
            occ = p
        elif n <= len(free_vars):
            value = free_vars[n - 1]
            occ = value * p if index > 0 else (1.0 - value) * p
        else:
            occ = p

        if atom.is_hydrogen and atom.afix and abs(atom.sof - 11.0) < 1e-9:
            # shelxfile lost the riding atom's own SOF - inherit the pivot's.
            occ = last_heavy
        elif not atom.is_hydrogen:
            last_heavy = occ

        result[id(atom)] = occ
    return result


# ---------------------------------------------------------------------------
# The calculation
# ---------------------------------------------------------------------------

def calculate_residual_density(
    model_path: str | Path,
    hkl_path: str | Path,
    *,
    sample_rate: float = 3.0,
    d_min: float | None = None,
) -> ResidualDensityMap:
    """Compute a residual (Fo−Fc) density map from a model and reflection file.

    :param model_path: The refined model — a CIF, or a SHELX ``.res``/``.ins``.
    :param hkl_path: Reflections — a SHELX ``.hkl`` or an fcf-style CIF loop.
    :param sample_rate: FFT grid oversampling; ``3.0`` gives roughly
        ``d_min / 3`` grid spacing, which is fine for smooth isosurfaces.
    :param d_min: Optional resolution cut-off in Å.  ``None`` uses all data.
    :returns: The computed map.
    :raises ValueError: If the model cannot be interpreted or the reflection
        file contains no usable data.
    """
    model_path = Path(model_path)
    structure, params = _load_model(model_path)
    if structure.spacegroup is None:
        raise ValueError(f'Could not determine the space group of {model_path}')

    reflections = read_reflections(hkl_path)
    hkl, f_obs = _merge_to_asu(reflections, structure.spacegroup, d_min,
                               structure.cell)
    if len(hkl) == 0:
        raise ValueError(f'No usable reflections in {hkl_path}')

    f_calc = _calculated_structure_factors(structure, hkl, params, reflections)
    scale = _scale_factor(params, f_obs, np.abs(f_calc))

    delta = (f_obs / scale - np.abs(f_calc)) * np.exp(1j * np.angle(f_calc))
    asu = gemmi.ComplexAsuData(structure.cell, structure.spacegroup,
                               hkl, delta.astype(np.complex64))
    grid = asu.transform_f_phi_to_map(sample_rate=sample_rate)

    resolution = min(structure.cell.calculate_d(list(h)) for h in hkl)
    cell = structure.cell
    return ResidualDensityMap(
        array=np.array(grid, copy=True),
        cell=(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma),
        d_min=resolution,
        scale=scale,
    )


def _load_model(path: Path) -> tuple[gemmi.SmallStructure, ShelxParameters]:
    """Read the refined model and its SHELX refinement parameters.

    SHELX files are parsed with :mod:`shelxfile`; anything else is read as a
    CIF.  The refined ``FVAR`` / ``WGHT`` / ``EXTI`` values are taken from the
    ``.res``/``.ins`` itself, from a sibling file of the same basename, or
    from a SHELX block embedded in the CIF (see
    :func:`fastmolwidget.hkl_io.read_shelx_parameters`).
    """
    params = read_shelx_parameters(path) or ShelxParameters()

    if path.suffix.lower() in ('.res', '.ins'):
        from shelxfile import Shelxfile

        shx = Shelxfile()
        shx.read_file(str(path))
        return small_structure_from_shelx(shx), params

    structure = small_structure_from_cif(path)
    if structure.wavelength:
        params.wavelength = structure.wavelength
    return structure, params


def _merge_to_asu(
    reflections: ReflectionData,
    spacegroup: gemmi.SpaceGroup,
    d_min: float | None,
    cell: gemmi.UnitCell,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge observations into the reciprocal asymmetric unit.

    Symmetry equivalents are averaged with ``1/σ²`` weights, as in a standard
    merging step.  Negative F² values (which occur for weak reflections) are
    clamped to zero before the square root is taken.

    :returns: ``(hkl, |Fo|)`` for the unique reflections.
    """
    ops = spacegroup.operations()
    asu = gemmi.ReciprocalAsu(spacegroup)

    sums: dict[tuple[int, int, int], list[float]] = {}
    for (h, k, l), f_sq, sigma in zip(reflections.hkl,
                                      reflections.f_sq_meas,
                                      reflections.sigma):
        index, _ = asu.to_asu([int(h), int(k), int(l)], ops)
        weight = 1.0 / max(float(sigma), 1e-6) ** 2
        key = (int(index[0]), int(index[1]), int(index[2]))
        entry = sums.get(key)
        if entry is None:
            entry = [0.0, 0.0]
            sums[key] = entry
        entry[0] += weight * float(f_sq)
        entry[1] += weight

    hkl_list, f_obs = [], []
    for index, (weighted, total) in sums.items():
        if total <= 0:
            continue
        if d_min is not None and cell.calculate_d(list(index)) < d_min:
            continue
        hkl_list.append(index)
        f_obs.append(sqrt(max(weighted / total, 0.0)))

    return (np.array(hkl_list, dtype=np.int32).reshape(-1, 3),
            np.array(f_obs, dtype=float))


def _calculated_structure_factors(
    structure: gemmi.SmallStructure,
    hkl: np.ndarray,
    params: ShelxParameters,
    reflections: ReflectionData,
) -> np.ndarray:
    """Return complex *F*\\ :sub:`c` for every reflection in *hkl*.

    Calculated values that came with the reflection file are reused when they
    carry phases; otherwise they are computed by direct summation with the
    real anomalous-dispersion term *f′* included.  SHELXL's isotropic
    extinction correction is applied when ``EXTI`` was refined.
    """
    f_calc_in = reflections.f_calc
    if (f_calc_in is not None and np.iscomplexobj(f_calc_in)
            and np.any(f_calc_in.imag != 0)):
        lookup = {tuple(idx): value
                  for idx, value in zip(reflections.hkl, f_calc_in)}
        if all(tuple(h) in lookup for h in hkl):
            return np.array([lookup[tuple(h)] for h in hkl])

    calculator = gemmi.StructureFactorCalculatorX(structure.cell)
    if params.wavelength:
        calculator.addends.add_cl_fprime(gemmi.hc / params.wavelength)

    f_calc = np.array([
        calculator.calculate_sf_from_small_structure(structure, list(h))
        for h in hkl
    ])
    return _apply_extinction(f_calc, hkl, structure.cell, params)


def _apply_extinction(
    f_calc: np.ndarray,
    hkl: np.ndarray,
    cell: gemmi.UnitCell,
    params: ShelxParameters,
) -> np.ndarray:
    """Apply SHELXL's isotropic extinction correction to *f_calc*.

    SHELXL scales the calculated amplitudes by

    .. math::
        F_c^{corr} = F_c \\left(1 + 0.001\\,x\\,F_c^2\\,
                     \\frac{\\lambda^3}{\\sin 2\\theta}\\right)^{-1/4}

    where *x* is the refined ``EXTI`` parameter.  Returns *f_calc* unchanged
    when no extinction was refined.
    """
    if not params.exti:
        return f_calc

    lambda_ = params.wavelength
    corrected = np.empty_like(f_calc)
    for i, index in enumerate(hkl):
        d = cell.calculate_d(list(index))
        sin_theta = min(lambda_ / (2.0 * d), 1.0)
        sin_2theta = max(2.0 * sin_theta * sqrt(max(1.0 - sin_theta ** 2, 0.0)),
                         1e-6)
        amplitude_sq = abs(f_calc[i]) ** 2
        factor = (1.0 + 0.001 * params.exti * amplitude_sq
                  * lambda_ ** 3 / sin_2theta) ** -0.25
        corrected[i] = f_calc[i] * factor
    return corrected


def _scale_factor(
    params: ShelxParameters,
    f_obs: np.ndarray,
    f_calc_abs: np.ndarray,
) -> float:
    """Return the scale *k* relating the two amplitude sets (``|Fo| ≈ k·|Fc|``).

    SHELXL's refined overall scale factor (the first ``FVAR``) is preferred.
    It is only replaced by a least-squares estimate when it is missing or
    clearly not a scale factor for this data — which happens for plain CIFs
    that carry no SHELX instructions.
    """
    if params.free_variables and params.osf > 0:
        return params.osf
    denominator = float(np.sum(f_calc_abs ** 2))
    if denominator <= 0:
        return 1.0
    return float(np.sum(f_obs * f_calc_abs) / denominator)
