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

import warnings
from dataclasses import dataclass
from math import cos, radians, sin, sqrt
from pathlib import Path

import gemmi
import numpy as np

from fastmolwidget.hkl_io import (
    _REFLECTION_SUFFIXES,
    ReflectionData,
    ShelxParameters,
    _data_blocks,
    find_reflection_file,
    read_reflections,
    read_shelx_parameters,
)

try:
    import density_cpp

    HAS_DENSITY_CPP: bool = True
except ImportError:  # pragma: no cover - depends on the compiled extension
    HAS_DENSITY_CPP = False

#: Grid spacing of the FFT map in Å.  Deliberately a fixed length rather than
#: a multiple of ``d_min``, so that the number of grid points depends only on
#: the size of the unit cell and never on how high the data resolution is.
#: 0.2 Å resolves the shape of individual residual-density features; coarser
#: grids (0.3-0.4 Å) are noticeably blockier once contoured.  Pass
#: ``grid_spacing=`` to :func:`calculate_residual_density` to trade detail
#: against speed and memory.
DEFAULT_GRID_SPACING: float = 0.2

#: Default padding around the displayed atoms, in Å.
DEFAULT_MARGIN: float = 1.5

__all__ = [
    'DEFAULT_GRID_SPACING',
    'DEFAULT_MARGIN',
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
        margin: float = DEFAULT_MARGIN,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract a wireframe isosurface at *level* in Cartesian coordinates.

        Marching cubes runs in fractional-coordinate space and the resulting
        vertices are transformed to Cartesian afterwards, so triclinic and
        monoclinic cells are handled correctly.

        When *atoms* is given, only the density within *margin* of those atoms
        is contoured.  This keeps the surface aligned with grown or packed
        molecules, avoids drawing density where nothing is displayed, and is
        much faster than contouring the whole cell.

        :param level: Contour level in e/Å³.  Use a negative value for the
            negative-density surface.
        :param atoms: Optional ``(N, 3)`` array of Cartesian atom positions
            used to restrict the contoured region.  ``None`` contours exactly
            one unit cell.
        :param margin: Radius around each atom to keep, in Å.
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
            if atoms is not None and len(atoms):
                verts, edges = _clip_to_atoms(verts, edges, atoms, margin)
        return np.ascontiguousarray(verts, dtype=np.float32), edges

    def _region(
        self,
        atoms: np.ndarray | None,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cut the periodic map down to the bounding box around *atoms*.

        This is only a cheap box pre-filter; :func:`_clip_to_atoms` then trims
        the result to the actual per-atom radius.

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
        # Convert the Cartesian margin into a safe fractional padding: the
        # width of the cell along each axis is volume / (area of opposite
        # face), which for the orthogonalisation matrix is 1 / |row of the
        # inverse|.  Using the row norms keeps the padding correct for
        # oblique cells instead of under-padding along the skewed axes.
        inverse = np.linalg.inv(self.orth_matrix)
        pad = margin * np.linalg.norm(inverse, axis=1)
        lo = np.floor((frac.min(axis=0) - pad) * shape).astype(int)
        hi = np.ceil((frac.max(axis=0) + pad) * shape).astype(int) + 1

        idx = [np.arange(a, b) % n for a, b, n in zip(lo, hi, shape)]
        sub = np.ascontiguousarray(self.array[np.ix_(*idx)], dtype=np.float32)
        return sub, lo * step, step


def _clip_to_atoms(
    vertices: np.ndarray,
    edges: np.ndarray,
    atoms: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop every line segment further than *margin* from any atom.

    The bounding box used to cut the grid is necessarily larger than the
    molecule, so this removes the density blobs that sit in the corners of the
    box but are nowhere near an atom.  An edge is kept when **both** of its
    vertices are close enough, which avoids segments dangling into empty space.

    Distances are evaluated with a uniform spatial hash of cell size *margin*,
    so only the 27 neighbouring buckets of each vertex have to be checked.
    That keeps this linear in the number of vertices instead of the
    ``len(vertices) x len(atoms)`` a brute-force test would need.

    :returns: ``(vertices, edges)`` renumbered to the surviving vertices.
    """
    if len(vertices) == 0 or len(edges) == 0:
        return vertices, edges

    atoms = np.asarray(atoms, dtype=np.float32)
    origin = atoms.min(axis=0) - margin
    buckets: dict[tuple[int, int, int], list[int]] = {}
    atom_cells = np.floor((atoms - origin) / margin).astype(int)
    for index, cell in enumerate(map(tuple, atom_cells)):
        buckets.setdefault(cell, []).append(index)

    vertex_cells = np.floor((vertices - origin) / margin).astype(int)
    limit = margin * margin
    keep = np.zeros(len(vertices), dtype=bool)
    offsets = [(dx, dy, dz)
               for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]

    for index, (point, cell) in enumerate(zip(vertices, vertex_cells)):
        near: list[int] = []
        for dx, dy, dz in offsets:
            found = buckets.get((cell[0] + dx, cell[1] + dy, cell[2] + dz))
            if found:
                near.extend(found)
        if not near:
            continue
        delta = atoms[near] - point
        if np.min(np.einsum('ij,ij->i', delta, delta)) <= limit:
            keep[index] = True

    if keep.all():
        return vertices, edges

    edge_mask = keep[edges[:, 0]] & keep[edges[:, 1]]
    if not edge_mask.any():
        return (np.empty((0, 3), dtype=vertices.dtype),
                np.empty((0, 2), dtype=edges.dtype))

    edges = edges[edge_mask]
    used = np.unique(edges)
    renumber = np.full(len(vertices), -1, dtype=np.int64)
    renumber[used] = np.arange(len(used))
    return vertices[used], renumber[edges]


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

def _sanitise_adps(structure: gemmi.SmallStructure) -> list[str]:
    """Replace non-positive-definite ADP tensors with their isotropic equivalent.

    A negative eigenvalue makes the Debye-Waller factor *grow* exponentially
    with resolution, so a single bad tensor turns the whole high-angle part of
    *F*\\ :sub:`c` into nonsense and buries the map under a huge dipole at that
    atom.  Files with such tensors do occur — ``_atom_site_aniso`` values are
    sometimes mangled by CIF writers, and refinements can end on a genuinely
    NPD tensor — so they are neutralised here rather than trusted.

    Positive-definiteness is invariant under the congruence transform that
    takes U\\ :sub:`cif` to U\\ :sub:`cart`, so the test can be applied
    directly to the stored tensor.

    The offending atom keeps its ``u_iso`` (normally U\\ :sub:`eq`, which is
    usually still sensible) and is rendered isotropically.

    :param structure: The structure to clean, modified in place.
    :returns: The labels of the atoms whose ADPs were replaced.
    """
    replaced: list[str] = []
    for site in structure.sites:
        adp = site.aniso
        if adp.u11 == 0.0 and adp.u22 == 0.0 and adp.u33 == 0.0:
            continue  # isotropic already
        matrix = np.array([
            [adp.u11, adp.u12, adp.u13],
            [adp.u12, adp.u22, adp.u23],
            [adp.u13, adp.u23, adp.u33],
        ])
        if np.all(np.linalg.eigvalsh(matrix) > 0.0):
            continue
        fallback = site.u_iso
        if fallback <= 0.0:
            trace = (adp.u11 + adp.u22 + adp.u33) / 3.0
            fallback = trace if trace > 0.0 else 0.05
        site.aniso = gemmi.SMat33d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        site.u_iso = fallback
        replaced.append(site.label)
    return replaced


def small_structure_from_cif(path: str | Path) -> gemmi.SmallStructure:
    """Read a CIF into a :class:`gemmi.SmallStructure` ready for SF calculation.

    A leading ``global_`` block is skipped — it only carries values inherited
    by the blocks that follow, and has no atom sites of its own.  The first
    block that does contain atom sites is used.

    ``change_occupancies_to_crystallographic()`` is required by gemmi before
    the structure factors are summed, so that atoms on special positions
    contribute with the correct multiplicity.

    :param path: Path to the CIF file.
    :raises ValueError: If the file has no block with atom sites.
    """
    doc = gemmi.cif.read(str(path))
    for block in _data_blocks(doc):
        structure = gemmi.make_small_structure_from_block(block)
        if structure.sites:
            structure.change_occupancies_to_crystallographic()
            bad = _sanitise_adps(structure)
            if bad:
                warnings.warn(
                    f'{path}: non-positive-definite ADPs for '
                    f'{", ".join(bad)} - these atoms are treated as '
                    f'isotropic.',
                    RuntimeWarning,
                    stacklevel=2,
                )
            return structure
    raise ValueError(f'No atom sites found in {path}')


def small_structure_from_shelx(shx) -> gemmi.SmallStructure:
    """Build a :class:`gemmi.SmallStructure` from a parsed SHELX model.

    SHELX's **negative U_iso** convention has to be resolved explicitly: a
    riding hydrogen with ``U = -1.5`` means *1.5 × U_eq of the atom it rides
    on*.  ``shelxfile`` (v25) still reports the raw ``-1.5`` from ``uvals``
    and a meaningless negative ``ueq`` for those atoms, so the pivot's U_eq is
    tracked here.

    Occupancies come straight from ``atom.occupancy``; ``shelxfile`` v25
    decodes the SOF free-variable codes correctly, including negative SOFs
    combined with a site occupancy factor and riding atoms inside an ``AFIX``
    group.

    :param shx: A :class:`shelxfile.Shelxfile` that has already read a file.
    :returns: The structure, with the space group and cell filled in.
    :raises ValueError: If the SHELX model has no ``CELL`` instruction.
    """
    if shx.cell is None:
        raise ValueError('SHELX model has no CELL instruction')

    cell = gemmi.UnitCell(shx.cell.a, shx.cell.b, shx.cell.c,
                          shx.cell.alpha, shx.cell.beta, shx.cell.gamma)
    ops = [gemmi.Op(s.to_shelxl().replace(' ', '')) for s in shx.symmcards]
    group = gemmi.GroupOps(ops)
    if shx.latt is not None and shx.latt.centric:
        group.add_inversion()
    spacegroup = gemmi.find_spacegroup_by_ops(group)

    structure = gemmi.SmallStructure()
    structure.cell = cell
    if spacegroup is not None:
        structure.spacegroup = spacegroup
        structure.spacegroup_hm = spacegroup.xhm()

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
        site.occ = atom.occupancy

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
    bad = _sanitise_adps(structure)
    if bad:
        warnings.warn(
            f'Non-positive-definite ADPs for {", ".join(bad)} - these atoms '
            f'are treated as isotropic.',
            RuntimeWarning,
            stacklevel=2,
        )
    return structure


# ---------------------------------------------------------------------------
# The calculation
# ---------------------------------------------------------------------------

def calculate_residual_density(
    model_path: str | Path,
    hkl_path: str | Path | None = None,
    *,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: float | None = None,
) -> ResidualDensityMap:
    """Compute a residual (Fo−Fc) density map from a model and reflection file.

    :param model_path: The refined model — a CIF, or a SHELX ``.res``/``.ins``.
    :param hkl_path: Reflections — a SHELX ``.hkl``, an fcf-style CIF loop, or
        a CIF with an embedded ``_shelx_hkl_file``.  ``None`` looks them up
        automatically with :func:`~fastmolwidget.hkl_io.find_reflection_file`.
    :param grid_spacing: FFT grid spacing in Å.  A fixed length, so the grid
        size depends only on the unit cell and not on the data resolution.
    :param d_min: Optional resolution cut-off in Å.  ``None`` uses all data.
    :returns: The computed map.
    :raises FileNotFoundError: If *hkl_path* is ``None`` and no reflection
        data could be found next to the model.
    :raises ValueError: If the model cannot be interpreted or the reflection
        file contains no usable data.
    """
    model_path = Path(model_path)
    structure, params = _load_model(model_path)
    if structure.spacegroup is None:
        raise ValueError(f'Could not determine the space group of {model_path}')

    if hkl_path is None:
        hkl_path = find_reflection_file(model_path)
        if hkl_path is None:
            raise FileNotFoundError(
                f'No reflection data found for {model_path}. Looked inside the '
                f'file itself and for '
                f'{", ".join(model_path.stem + s for s in _REFLECTION_SUFFIXES)}.'
            )

    reflections = read_reflections(hkl_path)
    hkl, f_obs = _merge_to_asu(reflections, structure.spacegroup, d_min,
                               structure.cell)
    if len(hkl) == 0:
        raise ValueError(f'No usable reflections in {hkl_path or model_path}')

    f_calc = _calculated_structure_factors(structure, hkl, params, reflections)
    scale = _scale_factor(params, f_obs, np.abs(f_calc))

    # Fix the grid from the cell alone, then drop the few reflections that are
    # finer than it can represent, so the map size never follows the data.
    size = _grid_size(structure.cell, grid_spacing)
    fits = _fits_in_grid(hkl, size)
    hkl, f_obs, f_calc = hkl[fits], f_obs[fits], f_calc[fits]
    if len(hkl) == 0:
        raise ValueError(
            f'No reflections fit a {grid_spacing} Å grid - use a smaller '
            f'grid_spacing'
        )

    delta = (f_obs / scale - np.abs(f_calc)) * np.exp(1j * np.angle(f_calc))
    asu = gemmi.ComplexAsuData(structure.cell, structure.spacegroup,
                               hkl, delta.astype(np.complex64))
    grid = asu.transform_f_phi_to_map(exact_size=size)

    resolution = min(structure.cell.calculate_d(list(h)) for h in hkl)
    cell = structure.cell
    return ResidualDensityMap(
        array=np.array(grid, copy=True),
        cell=(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma),
        d_min=resolution,
        scale=scale,
    )


def _fits_in_grid(hkl: np.ndarray, size: list[int]) -> np.ndarray:
    """Mask of the reflections representable on a grid of *size*.

    An FFT grid of ``n`` points along an axis can only carry Miller indices up
    to ``(n - 1) // 2``; anything beyond that would alias back into the map.

    :returns: Boolean mask over *hkl*.
    """
    limits = np.array([(n - 1) // 2 for n in size])
    return np.all(np.abs(hkl) <= limits, axis=1)


def _good_fft_size(minimum: int) -> int:
    """Smallest number ``>= minimum`` that factorises into 2, 3, 5 and 7.

    FFT libraries — including the one behind ``gemmi`` — are only efficient
    for such sizes, and ``gemmi`` rejects grid dimensions it cannot transform.
    """
    size = max(int(minimum), 8)
    while True:
        remainder = size
        for factor in (2, 3, 5, 7):
            while remainder % factor == 0:
                remainder //= factor
        if remainder == 1 and size % 2 == 0:
            return size
        size += 1


def _grid_size(cell: gemmi.UnitCell, spacing: float) -> list[int]:
    """Return the FFT grid dimensions for a fixed real-space *spacing*.

    Deliberately driven by the cell edges alone, so that adding higher-angle
    data does not enlarge the grid.

    :param cell: The unit cell.
    :param spacing: Target grid spacing in Å.
    :returns: ``[nu, nv, nw]`` grid dimensions.
    """
    spacing = max(float(spacing), 0.05)
    return [_good_fft_size(int(np.ceil(length / spacing)))
            for length in (cell.a, cell.b, cell.c)]


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

    **Systematically absent** reflections are discarded.  They have
    ``Fc = 0`` by symmetry, so their measured noise would otherwise enter the
    map as a coefficient of ``|Fo| / scale`` — with a small scale factor that
    produces enormous spurious peaks.

    :returns: ``(hkl, |Fo|)`` for the unique reflections.
    """
    ops = spacegroup.operations()
    asu = gemmi.ReciprocalAsu(spacegroup)

    sums: dict[tuple[int, int, int], list[float]] = {}
    for (h, k, l), f_sq, sigma in zip(reflections.hkl,
                                      reflections.f_sq_meas,
                                      reflections.sigma):
        index = [int(h), int(k), int(l)]
        if ops.is_systematically_absent(index):
            continue
        asu_index, _ = asu.to_asu(index, ops)
        weight = 1.0 / max(float(sigma), 1e-6) ** 2
        key = (int(asu_index[0]), int(asu_index[1]), int(asu_index[2]))
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
