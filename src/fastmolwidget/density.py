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
    _DEFAULT_TWIN_MATRIX,
    _REFLECTION_SUFFIXES,
    CifSource,
    ReflectionData,
    ReflectionSource,
    ShelxParameters,
    _cif_blocks,
    _is_cif_object,
    find_reflection_file,
    has_reflections,
    read_reflections,
    read_shelx_parameters,
)

try:
    from fastmolwidget import density_cpp

    #: ``True`` only when the compiled extension is really importable.  The
    #: attribute is checked rather than just the import, so that a stale or
    #: not-yet-built extension degrades to a clear error instead of an
    #: ``AttributeError`` deep inside :meth:`ResidualDensityMap.isosurface`.
    HAS_DENSITY_CPP: bool = hasattr(density_cpp, 'marching_cubes')
except ImportError:  # pragma: no cover - depends on the compiled extension
    HAS_DENSITY_CPP = False

#: Grid spacing of the FFT map in Å.  Deliberately a fixed length rather than
#: a multiple of ``d_min``, so that the number of grid points depends only on
#: the size of the unit cell and never on how high the data resolution is.
#: 0.15 Å resolves the shape of individual residual-density features; coarser
#: grids (0.3-0.4 Å) are noticeably blockier once contoured.  Pass
#: ``grid_spacing=`` to :func:`calculate_residual_density` to trade detail
#: against speed and memory.
DEFAULT_GRID_SPACING: float = 0.15

#: Default padding around the displayed atoms, in Å.
DEFAULT_MARGIN: float = 1.5

#: Default contour level, as a multiple of the map's RMS.  A fixed absolute
#: level cannot suit every dataset — the RMS of a residual map varies by an
#: order of magnitude between structures — whereas 3σ is the usual
#: crystallographic threshold for "significant" residual density and gives a
#: comparable picture for good and poor refinements alike.
DEFAULT_SIGMA: float = 3.0

#: Default strength of the down-weighting of weak data.  Every Fourier
#: coefficient is multiplied by ``1 / (1 + w·(σ(F)/|Fc|)³)``, so reflections
#: whose σ approaches their calculated amplitude — the noisy, mostly
#: high-angle ones — barely contribute, while strong data passes through
#: untouched.  This is the only smoothing applied: it acts in reciprocal
#: space, on the data, rather than blurring the finished map.  ``0.0``
#: switches it off.
DEFAULT_WEAK_WEIGHT: float = 1.0

#: Exponent of the ``σ(F)/|Fc|`` ratio in the down-weighting of weak data.
#: The higher it is, the more abruptly a reflection is cut once its σ becomes
#: comparable with its calculated amplitude.
WEAK_DATA_EXPONENT: float = 3.0

__all__ = [
    'DEFAULT_GRID_SPACING',
    'DEFAULT_MARGIN',
    'DEFAULT_SIGMA',
    'DEFAULT_WEAK_WEIGHT',
    'HAS_DENSITY_CPP',
    'WEAK_DATA_EXPONENT',
    'ModelSource',
    'ResidualDensityMap',
    'calculate_residual_density',
    'small_structure_from_block',
    'small_structure_from_cif',
    'small_structure_from_shelx',
]

#: Everything :func:`calculate_residual_density` accepts as a model: a path to
#: a CIF or SHELX file, an in-memory CIF document or block, or a structure that
#: was built elsewhere.  The in-memory forms let a host application use the
#: document it is already editing instead of writing a temporary file.
ModelSource = CifSource | gemmi.SmallStructure


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

    def sigma_level(self, sigma: float = DEFAULT_SIGMA) -> float:
        """Return the contour level *sigma* times the map RMS, in e/Å³.

        Residual maps differ hugely in scale between structures, so a level
        expressed in σ transfers between datasets where an absolute one does
        not.  The value is rounded to two decimals to match the precision the
        viewer's spin box offers, and never returns zero.

        :param sigma: Multiple of the RMS to contour at.
        """
        level = round(sigma * self.rms, 2)
        return max(level, 0.01)

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
    ``len(vertices) x len(atoms)`` a brute-force test would need.  The whole
    search is vectorised — the buckets are a sorted array of cell keys that all
    vertices query at once with :func:`numpy.searchsorted`, one pass per
    neighbour offset — because this runs on the interactive path: every change
    of the contour level, of the hydrogen filter or of the visible disorder
    parts re-clips both lobes.

    :returns: ``(vertices, edges)`` renumbered to the surviving vertices.
    """
    if len(vertices) == 0 or len(edges) == 0:
        return vertices, edges
    if margin <= 0.0:
        return (np.empty((0, 3), dtype=vertices.dtype),
                np.empty((0, 2), dtype=edges.dtype))

    atoms = np.asarray(atoms, dtype=np.float32)
    vertices = np.asarray(vertices)
    origin = atoms.min(axis=0) - margin

    # Bucket the atoms by cell and sort them, so a bucket is a contiguous
    # slice that searchsorted can locate in O(log n).
    atom_cells = np.floor((atoms - origin) / margin).astype(np.int64)
    shape = atom_cells.max(axis=0) + 1
    atom_keys = (atom_cells[:, 0] * shape[1] + atom_cells[:, 1]) * shape[2] \
        + atom_cells[:, 2]
    order = np.argsort(atom_keys, kind='stable')
    atom_keys = atom_keys[order]
    sorted_atoms = atoms[order]

    vertex_cells = np.floor((vertices - origin) / margin).astype(np.int64)
    limit = margin * margin
    keep = np.zeros(len(vertices), dtype=bool)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                query = vertex_cells + (dx, dy, dz)
                # Only vertices not yet accepted, whose neighbour cell exists.
                todo = np.flatnonzero(
                    ~keep & np.all((query >= 0) & (query < shape), axis=1))
                if todo.size == 0:
                    continue
                cells = query[todo]
                keys = (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] \
                    + cells[:, 2]
                start = np.searchsorted(atom_keys, keys, side='left')
                stop = np.searchsorted(atom_keys, keys, side='right')
                counts = stop - start
                nonempty = counts > 0
                if not nonempty.any():
                    continue
                todo, start, counts = (todo[nonempty], start[nonempty],
                                       counts[nonempty])
                # Flatten the ragged (vertex -> its bucket's atoms) pairs.
                total = int(counts.sum())
                offsets = np.repeat(np.cumsum(counts) - counts, counts)
                atom_index = np.repeat(start, counts) + \
                    (np.arange(total) - offsets)
                vertex_index = np.repeat(todo, counts)
                delta = sorted_atoms[atom_index] - vertices[vertex_index]
                close = np.einsum('ij,ij->i', delta, delta) <= limit
                keep[vertex_index[close]] = True

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


def small_structure_from_cif(source: CifSource) -> gemmi.SmallStructure:
    """Read a CIF into a :class:`gemmi.SmallStructure` ready for SF calculation.

    A leading ``global_`` block is skipped — it only carries values inherited
    by the blocks that follow, and has no atom sites of its own.  The first
    block that does contain atom sites is used.

    :param source: Path to the CIF file, a parsed document, or a single block.
    :raises ValueError: If the source has no block with atom sites.
    """
    for block in _cif_blocks(source):
        structure = small_structure_from_block(block)
        if structure is not None:
            return structure
    raise ValueError(f'No atom sites found in {_source_name(source)}')


def small_structure_from_block(block) -> gemmi.SmallStructure | None:
    """Build a :class:`gemmi.SmallStructure` from a single CIF block.

    ``change_occupancies_to_crystallographic()`` is required by gemmi before
    the structure factors are summed, so that atoms on special positions
    contribute with the correct multiplicity.

    :param block: A :class:`gemmi.cif.Block`.
    :returns: The structure, or ``None`` when the block has no atom sites.
    """
    structure = gemmi.make_small_structure_from_block(block)
    if not structure.sites:
        return None
    structure.change_occupancies_to_crystallographic()
    bad = _sanitise_adps(structure)
    if bad:
        warnings.warn(
            f'{block.name}: non-positive-definite ADPs for '
            f'{", ".join(bad)} - these atoms are treated as '
            f'isotropic.',
            RuntimeWarning,
            stacklevel=2,
        )
    return structure


def _source_name(source: object) -> str:
    """A short, printable name for a model or reflection source."""
    if isinstance(source, gemmi.cif.Block):
        return f'data_{source.name}'
    if isinstance(source, gemmi.cif.Document):
        return source.source or '<cif document>'
    if isinstance(source, gemmi.SmallStructure):
        return source.name or '<structure>'
    if isinstance(source, ReflectionData):
        return '<reflections>'
    return str(source)


#: SHELX ``LATT`` centring translations, in fractional coordinates.  The sign
#: of the ``LATT`` number selects centrosymmetry, its magnitude the lattice
#: type.  ``LATT 1`` (primitive) has no extra translation.
_LATT_CENTRING: dict[int, tuple[tuple[float, float, float], ...]] = {
    1: (),                                                    # P
    2: ((0.5, 0.5, 0.5),),                                    # I
    3: ((2 / 3, 1 / 3, 1 / 3), (1 / 3, 2 / 3, 2 / 3)),        # R (obverse)
    4: ((0.0, 0.5, 0.5), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0)),   # F
    5: ((0.0, 0.5, 0.5),),                                    # A
    6: ((0.5, 0.0, 0.5),),                                    # B
    7: ((0.5, 0.5, 0.0),),                                    # C
}


def _shelx_spacegroup(shx) -> gemmi.SpaceGroup | None:
    """Derive the space group of a SHELX model.

    The ``SYMM`` cards alone are **not** the full group: ``LATT`` adds the
    lattice centring translations, and a positive ``LATT`` additionally implies
    an inversion centre.  Leaving the centring out silently yields a primitive
    subgroup (``C2/c`` becomes ``P2/c``), which halves the number of symmetry
    mates and makes every calculated structure factor wrong.

    :param shx: A parsed :class:`shelxfile.Shelxfile`.
    :returns: The space group, or ``None`` if gemmi cannot identify it.
    """
    base = [gemmi.Op(card.to_shelxl().replace(' ', '')) for card in shx.symmcards]

    latt = shx.latt.N if shx.latt is not None else 1
    centring = _LATT_CENTRING.get(abs(int(latt)), ())

    denominator = gemmi.Op.DEN
    shifts = [(0, 0, 0)] + [
        tuple(round(component * denominator) for component in vector)
        for vector in centring
    ]

    ops: list[gemmi.Op] = []
    for op in base:
        for shift in shifts:
            translated = gemmi.Op(op.triplet())
            translated.tran = [
                (value + offset) % denominator
                for value, offset in zip(op.tran, shift)
            ]
            ops.append(translated)

    group = gemmi.GroupOps(ops)
    if latt > 0:  # positive LATT => centrosymmetric
        group.add_inversion()
    group.add_missing_elements()
    return gemmi.find_spacegroup_by_ops(group)


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
    spacegroup = _shelx_spacegroup(shx)

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

def _apply_hklf_transform(
    reflections: ReflectionData,
    params: ShelxParameters,
) -> ReflectionData:
    """Apply the ``HKLF`` index transformation and scale factors.

    ``HKLF N S r11…r33 sm`` lets the reflection file be indexed on a different
    setting from the model: the new indices are ``h' = R h`` (so
    ``h' = r11·h + r12·k + r13·l``).  The cell, symmetry and coordinates in the
    ``.res`` refer to the *transformed* indices, so this has to happen before
    anything else touches the data.  ``S`` scales F² and σ, ``sm`` scales σ
    again.

    :param reflections: The data as read from the file.
    :param params: Refinement parameters carrying the ``HKLF`` card.
    :returns: The transformed data, or the input unchanged when the card is
        the default ``HKLF 4``.
    :raises ValueError: If the transformation matrix is singular or has a
        negative determinant, which SHELXL does not allow.
    """
    matrix = params.hklf_matrix
    scale = params.hklf_scale
    sigma_scale = params.hklf_sigma_scale
    if matrix is None and scale == 1.0 and sigma_scale == 1.0:
        return reflections

    hkl = reflections.hkl
    if matrix is not None:
        law = np.array(matrix, dtype=float).reshape(3, 3)
        determinant = float(np.linalg.det(law))
        if determinant <= 0.0:
            raise ValueError(
                f'HKLF matrix must have a positive determinant, got '
                f'{determinant:.3f}'
            )
        # h'_i = sum_j R_ij h_j  ->  row vectors transform with R transposed.
        transformed = np.asarray(hkl, dtype=float) @ law.T
        hkl = np.rint(transformed).astype(np.int32)

    return ReflectionData(
        hkl=hkl,
        f_sq_meas=reflections.f_sq_meas * scale,
        sigma=reflections.sigma * scale * sigma_scale,
        f_calc=reflections.f_calc,
        batch=reflections.batch,
        sigma_known=reflections.sigma_known,
    )


def _twin_domain_indices(
    hkl: np.ndarray,
    matrix: tuple[float, ...],
    components: int,
    *,
    racemic: bool = False,
) -> list[np.ndarray]:
    """Return the Miller indices of every twin domain for each reflection.

    Domain *k* is reached by applying the ``TWIN`` matrix *k* times to the
    prime indices, using the same convention as the ``HKLF`` card:
    ``h' = r11·h + r12·k + r13·l``, i.e. ``h' = M h`` for a column vector.
    When *racemic* is set the ``TWIN`` count was negative, meaning general and
    racemic twinning together: the matrix generates components ``1…m`` (with
    ``m = components / 2``) and components ``m+1…2m`` are their Friedel
    opposites.

    :param hkl: ``(N, 3)`` prime indices.
    :param matrix: The ``TWIN`` matrix in row-major order.
    :param components: Total number of twin components.
    :param racemic: Whether the second half are the inverted components.
    :returns: A list of ``components`` ``(N, 3)`` float arrays.
    """
    law = np.array(matrix, dtype=float).reshape(3, 3)
    generated = components // 2 if racemic else components

    # Row-vector arrays transform with the transpose of the column-vector law.
    indices = [np.asarray(hkl, dtype=float)]
    for _ in range(max(generated - 1, 0)):
        indices.append(indices[-1] @ law.T)

    if racemic:
        indices += [-block for block in indices[:generated]]
    return indices[:components]


class _StructureFactorCache:
    """Memoised ``|Fc|²`` lookups for a structure.

    Twin domains map many reflections onto indices that are already needed
    elsewhere, and direct summation is the expensive part of the calculation,
    so results are cached by Miller index.
    """

    def __init__(self, structure: gemmi.SmallStructure,
                 calculator: gemmi.StructureFactorCalculatorX) -> None:
        self._structure = structure
        self._calculator = calculator
        self._cache: dict[tuple[int, int, int], float] = {}

    def intensity(self, index: tuple[int, int, int]) -> float:
        """Return ``|Fc|²`` for one Miller index."""
        value = self._cache.get(index)
        if value is None:
            amplitude = self._calculator.calculate_sf_from_small_structure(
                self._structure, list(index))
            value = float(abs(amplitude) ** 2)
            self._cache[index] = value
        return value


def _detwin_observations(
    reflections: ReflectionData,
    structure: gemmi.SmallStructure,
    params: ShelxParameters,
) -> ReflectionData:
    """Split twinned intensities into the contribution of the first domain.

    Each measured intensity of a twinned crystal is the sum over domains,
    ``Io = Σ_k b_k |Fc(h_k)|²``.  The model tells us how that sum divides, so
    the part belonging to the domain we are mapping is recovered as

    .. math::
        F_o^2(h_1) = I_{obs}\\,\\frac{|F_c(h_1)|^2}{\\sum_k b_k |F_c(h_k)|^2}

    which reduces to ``|Fc(h₁)|²`` for a perfect model — i.e. the detwinned
    data is on the same scale as the single-domain calculated values, exactly
    what the difference map needs.

    Two data layouts are handled:

    * **HKLF 4 + TWIN** — one record per observation; the other domains'
      indices are generated from the twin law.
    * **HKLF 5** — the domains are listed explicitly, one record each, with a
      negative component number on every record of an overlap group except the
      last.  ``HKLF 5`` may not be combined with ``TWIN``, so the format alone
      decides which layout applies — a negative batch number in ``HKLF 4``
      data means something else entirely (an *R*\\ :sub:`free` flag).

    :param reflections: The measured data.
    :param structure: The refined model, used to apportion the intensities.
    :param params: Refinement parameters carrying ``TWIN`` / ``BASF``.
    :returns: New :class:`ReflectionData` holding one detwinned observation per
        group, indexed by the primary domain.

    .. note::
       A pure **inversion (racemic) twin** is a no-op here.  Splitting the
       intensity relies on the domains having different ``|Fc|``, but for
       ``h`` and ``-h`` that difference comes entirely from the imaginary
       anomalous term *f″*, which gemmi's real-valued addends cannot express.
       The map is therefore left slightly too large for racemic twins — an
       error of the size of the anomalous signal, which is small for light
       atoms.
    """
    calculator = gemmi.StructureFactorCalculatorX(structure.cell)
    if params.wavelength:
        calculator.addends.add_cl_fprime(gemmi.hc / params.wavelength)
    cache = _StructureFactorCache(structure, calculator)
    fractions = params.twin_fractions()

    if params.hklf == 5:
        groups = _hklf5_groups(reflections)
    else:
        groups = _hklf4_groups(reflections, params)

    hkl_out: list[tuple[int, int, int]] = []
    f_sq_out: list[float] = []
    sigma_out: list[float] = []

    for primary, members, f_sq, sigma in groups:
        total = 0.0
        for component, index in members:
            weight = fractions[component] if component < len(fractions) else 0.0
            total += weight * cache.intensity(index)
        if total <= 0.0:
            continue
        share = cache.intensity(primary) / total
        hkl_out.append(primary)
        f_sq_out.append(f_sq * share)
        sigma_out.append(sigma * share)

    if not hkl_out:
        raise ValueError('Detwinning left no usable reflections')

    return ReflectionData(
        hkl=np.array(hkl_out, dtype=np.int32),
        f_sq_meas=np.array(f_sq_out, dtype=float),
        sigma=np.array(sigma_out, dtype=float),
        sigma_known=reflections.sigma_known,
    )


def _hklf4_groups(reflections: ReflectionData, params: ShelxParameters):
    """Yield ``(primary, members, F², σ)`` for ``HKLF 4`` data with a twin law.

    The domains are generated from the ``TWIN`` matrix, so every record is one
    complete observation.
    """
    matrix = params.twin_matrix or _DEFAULT_TWIN_MATRIX
    domains = _twin_domain_indices(reflections.hkl, matrix,
                                   params.twin_components,
                                   racemic=params.twin_racemic)
    rounded = [np.rint(block).astype(int) for block in domains]

    for position in range(len(reflections)):
        members = []
        for component, block in enumerate(rounded):
            index = (int(block[position, 0]), int(block[position, 1]),
                     int(block[position, 2]))
            members.append((component, index))
        yield (members[0][1], members,
               float(reflections.f_sq_meas[position]),
               float(reflections.sigma[position]))


def _hklf5_groups(reflections: ReflectionData):
    """Yield ``(primary, members, F², σ)`` for ``HKLF 5`` overlap groups.

    Records belonging to one measured intensity are consecutive; all but the
    last carry a negative component number.  They share ``F²`` and ``σ``, so
    the values of the closing record are used.  The map is built for domain 1,
    so that record is chosen as the primary one when the group contains it —
    groups without a domain-1 contribution fall back to their first record.
    """
    batch = reflections.batch
    members: list[tuple[int, tuple[int, int, int]]] = []

    for position in range(len(reflections)):
        component = max(abs(int(batch[position])) - 1, 0)  # BASF is 0-based
        index = (int(reflections.hkl[position, 0]),
                 int(reflections.hkl[position, 1]),
                 int(reflections.hkl[position, 2]))
        members.append((component, index))

        if batch[position] > 0:  # last record of the group
            primary = next((idx for comp, idx in members if comp == 0),
                           members[0][1])
            yield (primary, members,
                   float(reflections.f_sq_meas[position]),
                   float(reflections.sigma[position]))
            members = []


def _is_model_object(source: object) -> bool:
    """True when *source* is an in-memory model rather than a file path."""
    return _is_cif_object(source) or isinstance(source, gemmi.SmallStructure)


def _find_reflections_for(model: ModelSource) -> ReflectionSource | None:
    """Locate the reflection data belonging to *model*.

    An in-memory CIF can only carry its reflections itself, since there is no
    directory to look in; a path additionally gets the sibling search of
    :func:`~fastmolwidget.hkl_io.find_reflection_file`.

    :returns: The reflection source, or ``None`` when there is none.
    """
    if _is_cif_object(model):
        return model if has_reflections(model) else None
    if isinstance(model, gemmi.SmallStructure):
        return None
    return find_reflection_file(model)


def _no_reflections_message(model: ModelSource) -> str:
    """The error text for a model whose reflection data could not be found."""
    if _is_model_object(model):
        return (f'No reflection data found for {_source_name(model)}. Pass the '
                f'reflections explicitly - an in-memory model has no file to '
                f'search next to.')
    path = Path(model)
    return (f'No reflection data found for {path}. Looked inside the file '
            f'itself and for '
            f'{", ".join(path.stem + s for s in _REFLECTION_SUFFIXES)}.')


def calculate_residual_density(
    model_path: ModelSource,
    hkl_path: ReflectionSource | None = None,
    *,
    grid_spacing: float = DEFAULT_GRID_SPACING,
    d_min: float | None = None,
    weak_weight: float = DEFAULT_WEAK_WEIGHT,
) -> ResidualDensityMap:
    """Compute a residual (Fo−Fc) density map from a model and reflection data.

    :param model_path: The refined model — a CIF or SHELX ``.res``/``.ins``
        path, an in-memory :class:`gemmi.cif.Document` or
        :class:`gemmi.cif.Block`, or a ready :class:`gemmi.SmallStructure`.
    :param hkl_path: Reflections — a SHELX ``.hkl``, an fcf-style CIF loop, a
        CIF with an embedded ``_shelx_hkl_file``, an in-memory document or
        block, or already read :class:`~fastmolwidget.hkl_io.ReflectionData`.
        ``None`` looks them up in the model itself; for a model *path* the
        sibling files found by
        :func:`~fastmolwidget.hkl_io.find_reflection_file` are searched too.
    :param grid_spacing: FFT grid spacing in Å.  A fixed length, so the grid
        size depends only on the unit cell and not on the data resolution.
    :param d_min: Optional resolution cut-off in Å.  ``None`` uses all data.
    :param weak_weight: Strength of the down-weighting of weak data, see
        :func:`_weak_data_damping`.  ``0.0`` disables it.
    :returns: The computed map.
    :raises FileNotFoundError: If *hkl_path* is ``None`` and no reflection
        data could be found for the model.
    :raises ValueError: If the model cannot be interpreted or the reflection
        data contains nothing usable.
    """
    model = model_path if _is_model_object(model_path) else Path(model_path)
    structure, params = _load_model(model)
    if structure.spacegroup is None:
        raise ValueError(
            f'Could not determine the space group of {_source_name(model)}')

    if hkl_path is None:
        hkl_path = _find_reflections_for(model)
        if hkl_path is None:
            raise FileNotFoundError(_no_reflections_message(model))

    reflections = read_reflections(hkl_path)

    if not reflections.has_f_calc:
        # HKLF can re-index the data into the model's setting; everything
        # downstream assumes that has already happened.
        reflections = _apply_hklf_transform(reflections, params)

        # Twinned data has to be split into single-domain intensities before
        # the map is built, otherwise the other domains' scattering shows up
        # as residual density all over the map.
        if params.is_twinned:
            reflections = _detwin_observations(reflections, structure, params)

    hkl, f_obs, sigma = _merge_to_asu(reflections, structure.spacegroup, d_min,
                                      structure.cell)
    if len(hkl) == 0:
        raise ValueError(f'No usable reflections in {_source_name(hkl_path)}')

    f_calc = _calculated_structure_factors(structure, hkl, params, reflections)
    scale = _scale_factor(params, f_obs, np.abs(f_calc))

    # Fix the grid from the cell alone, then drop the few reflections that are
    # finer than it can represent, so the map size never follows the data.
    size = _grid_size(structure.cell, grid_spacing)
    fits = _fits_in_grid(hkl, size)
    hkl, f_obs, f_calc = hkl[fits], f_obs[fits], f_calc[fits]
    sigma = sigma[fits]
    if len(hkl) == 0:
        raise ValueError(
            f'No reflections fit a {grid_spacing} Å grid - use a smaller '
            f'grid_spacing'
        )

    delta = (f_obs / scale - np.abs(f_calc)) * np.exp(1j * np.angle(f_calc))
    if reflections.sigma_known:
        # σ has to be brought onto the calculated scale, just like |Fo|.
        delta = delta * _weak_data_damping(sigma / scale, np.abs(f_calc),
                                           weak_weight)
    asu = gemmi.ComplexAsuData(structure.cell, structure.spacegroup,
                               hkl, delta.astype(np.complex64))
    grid = asu.transform_f_phi_to_map(exact_size=size)

    resolution = float(structure.cell.calculate_d_array(hkl).min())
    cell = structure.cell
    return ResidualDensityMap(
        array=np.array(grid, copy=True),
        cell=(cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma),
        d_min=resolution,
        scale=scale,
    )


def _weak_data_damping(
    sigma: np.ndarray,
    f_calc_abs: np.ndarray,
    weight: float,
) -> np.ndarray:
    """Return the per-reflection factor that damps weak, noisy data.

    Each Fourier coefficient is multiplied by

    .. math::
        \\frac{1}{1 + w \\left(\\frac{\\sigma(F)}{|F_c|}\\right)^3}

    A reflection measured well compared with what the model predicts passes
    through unchanged, while one whose σ approaches its calculated amplitude
    is suppressed; the third power (:data:`WEAK_DATA_EXPONENT`) makes that
    transition gradual enough to leave genuinely weak but real data in the
    map.  Since the poorly measured reflections are predominantly the
    high-angle ones, the net effect is a data-driven, resolution-dependent
    low-pass filter — the map is smoothed *before* the FFT rather than
    blurred afterward, so no feature is displaced.

    Reflections with a vanishing ``|Fc|``, where the ratio is meaningless,
    are left alone.

    :param sigma: σ(F) of each unique reflection, on the calculated scale.
    :param f_calc_abs: ``|Fc|`` of the same reflections.
    :param weight: The strength *w*; ``<= 0`` disables the filter.
    :returns: Factors in ``(0, 1]``, one per reflection.
    """
    if weight <= 0.0:
        return np.ones_like(f_calc_abs, dtype=float)
    ratio = np.zeros_like(f_calc_abs, dtype=float)
    usable = f_calc_abs > 1e-6
    ratio[usable] = sigma[usable] / f_calc_abs[usable]
    return 1.0 / (1.0 + weight * ratio ** WEAK_DATA_EXPONENT)


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


def _load_model(source: ModelSource) -> tuple[gemmi.SmallStructure, ShelxParameters]:
    """Read the refined model and its SHELX refinement parameters.

    SHELX files are parsed with :mod:`shelxfile`; anything else is read as a
    CIF.  The refined ``FVAR`` / ``WGHT`` / ``EXTI`` values are taken from the
    ``.res``/``.ins`` itself, from a sibling file of the same basename, or
    from a SHELX block embedded in the CIF (see
    :func:`fastmolwidget.hkl_io.read_shelx_parameters`).

    A :class:`gemmi.SmallStructure` handed in directly is used as it is; it
    carries no SHELX instructions, so the defaults apply unless the caller has
    already scaled the data.
    """
    if isinstance(source, gemmi.SmallStructure):
        return source, ShelxParameters()

    if _is_cif_object(source):
        params = read_shelx_parameters(source) or ShelxParameters()
        structure = small_structure_from_cif(source)
        if structure.wavelength:
            params.wavelength = structure.wavelength
        return structure, params

    path = Path(source)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge observations into the reciprocal asymmetric unit.

    Symmetry equivalents are averaged with ``1/σ²`` weights, as in a standard
    merging step.  Negative F² values (which occur for weak reflections) are
    clamped to zero before the square root is taken.

    **Systematically absent** reflections are discarded.  They have
    ``Fc = 0`` by symmetry, so their measured noise would otherwise enter the
    map as a coefficient of ``|Fo| / scale`` — with a small scale factor that
    produces enormous spurious peaks.

    The grouping is done without a Python loop over the (often > 10\\ :sup:`4`)
    observations: :func:`_equivalence_classes` labels the symmetry-equivalence
    classes with pure NumPy, and the weighted means are accumulated with
    :func:`numpy.bincount`.  Only one
    :meth:`gemmi.ReciprocalAsu.to_asu` call per *unique* reflection is left,
    instead of one per observation.

    :returns: ``(hkl, |Fo|, σ(F))`` for the unique reflections.  The merged
        standard uncertainty is propagated onto the amplitude scale as
        ``σ(F) = sqrt(F² + σ(F²)) − |Fo|``, with
        ``σ(F²) = 1/sqrt(Σ 1/σ_i²)`` from the merge.
    """
    ops = spacegroup.operations()
    asu = gemmi.ReciprocalAsu(spacegroup)

    hkl = np.asarray(reflections.hkl, dtype=np.int32).reshape(-1, 3)
    f_sq = np.asarray(reflections.f_sq_meas, dtype=float)
    sigma = np.asarray(reflections.sigma, dtype=float)

    present = ~np.asarray(ops.systematic_absences(hkl), dtype=bool)
    hkl, f_sq, sigma = hkl[present], f_sq[present], sigma[present]
    if len(hkl) == 0:
        empty = np.empty(0, dtype=float)
        return np.empty((0, 3), dtype=np.int32), empty, empty.copy()

    labels, representatives = _equivalence_classes(hkl, ops)
    asu_hkl = np.array(
        [asu.to_asu([int(h), int(k), int(l)], ops)[0]
         for h, k, l in representatives],
        dtype=np.int32,
    ).reshape(-1, 3)

    weight = 1.0 / np.maximum(sigma, 1e-6) ** 2
    count = len(asu_hkl)
    weighted = np.bincount(labels, weights=weight * f_sq, minlength=count)
    total = np.bincount(labels, weights=weight, minlength=count)

    usable = total > 0
    if d_min is not None:
        usable &= cell.calculate_d_array(asu_hkl) >= d_min

    asu_hkl = asu_hkl[usable]
    mean_f_sq = np.maximum(weighted[usable] / total[usable], 0.0)
    f_obs = np.sqrt(mean_f_sq)
    sigma_f_sq = 1.0 / np.sqrt(total[usable])
    sigma_f = np.sqrt(mean_f_sq + sigma_f_sq) - f_obs
    return asu_hkl, f_obs, sigma_f


def _equivalence_classes(
    hkl: np.ndarray,
    ops: gemmi.GroupOps,
) -> tuple[np.ndarray, np.ndarray]:
    """Label the symmetry-equivalence classes of *hkl*.

    Two reflections belong to the same class when a rotation of the space
    group (optionally combined with inversion) maps one onto the other —
    exactly the grouping :meth:`gemmi.ReciprocalAsu.to_asu` produces, which
    always merges Friedel pairs.  Translations, including lattice centring, do
    not act on Miller indices and are therefore ignored.

    Each class is identified by its lexicographically largest member, encoded
    as a single integer so that :func:`numpy.unique` can do the grouping.

    :param hkl: ``(N, 3)`` integer Miller indices.
    :param ops: The symmetry operations of the space group.
    :returns: ``(labels, representatives)`` — an ``(N,)`` array of class
        numbers and the ``(U, 3)`` representative of every class, in the order
        the labels refer to.
    """
    indices = hkl.astype(np.int64)
    rotations = np.array([np.array(op.rot, dtype=np.int64)
                          for op in ops.sym_ops])
    # Miller indices transform as row vectors: h' = h . R  (gemmi stores the
    # rotation parts multiplied by Op.DEN).
    equivalents = np.einsum('nj,mjk->mnk', indices, rotations) // gemmi.Op.DEN
    equivalents = np.concatenate((equivalents, -equivalents))

    # Lexicographic ordering as one number; the base is wide enough that the
    # lower two indices can never outweigh a higher one.
    base = 4 * (int(np.abs(equivalents).max()) + 1)
    codes = ((equivalents[..., 0] * base + equivalents[..., 1]) * base
             + equivalents[..., 2])
    best = codes.argmax(axis=0)
    columns = np.arange(len(indices))

    _, first, labels = np.unique(codes[best, columns], return_index=True,
                                 return_inverse=True)
    return labels, equivalents[best[first], first]


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

    f_calc = _summed_structure_factors(structure, hkl, calculator)
    return _apply_extinction(f_calc, hkl, structure.cell, params)


def _summed_structure_factors(
    structure: gemmi.SmallStructure,
    hkl: np.ndarray,
    calculator: gemmi.StructureFactorCalculatorX,
) -> np.ndarray:
    """Sum *F*\\ :sub:`c` over the model for every reflection in *hkl*.

    Direct summation is the single most expensive step of a residual-density
    map, and gemmi's implementation deliberately leaves it unoptimised (it is
    only meant as a reference for the FFT route, which needs a macromolecular
    ``Model`` and does not apply to a :class:`gemmi.SmallStructure`).  So the
    summation is handed to :func:`density_cpp.structure_factors`, which
    tabulates the separable phase factor over the Miller-index range instead of
    evaluating a sine and a cosine per atom, symmetry image and reflection.

    Falls back to gemmi whenever the compiled extension is unavailable or the
    model uses a scatterer gemmi has no IT92 parametrisation for, so the result
    is always defined.

    :returns: ``(N,)`` complex array of calculated structure factors.
    """
    prepared = _structure_factor_arrays(structure, hkl, calculator)
    if prepared is None:
        return np.array([
            calculator.calculate_sf_from_small_structure(structure, list(h))
            for h in hkl
        ])
    return density_cpp.structure_factors(**prepared)


def _structure_factor_arrays(
    structure: gemmi.SmallStructure,
    hkl: np.ndarray,
    calculator: gemmi.StructureFactorCalculatorX,
) -> dict | None:
    """Flatten the model into the plain arrays :mod:`density_cpp` expects.

    :returns: The keyword arguments for
        :func:`density_cpp.structure_factors`, or ``None`` when the fast path
        cannot be used and gemmi has to do the summation.
    """
    if not HAS_DENSITY_CPP or not hasattr(density_cpp, 'structure_factors'):
        return None
    sites = structure.sites
    if not sites or len(hkl) == 0:
        return None

    cell = structure.cell
    hkl = np.ascontiguousarray(hkl, dtype=np.int32)
    stol2 = 0.25 * np.asarray(cell.calculate_1_d2_array(hkl), dtype=float)

    # gemmi keeps the identity out of UnitCell.images.
    rotations = [np.eye(3)]
    translations = [np.zeros(3)]
    for image in cell.images:
        rotations.append(np.array(image.mat.tolist(), dtype=float))
        translations.append(np.array(image.vec.tolist(), dtype=float))

    # One scattering-factor curve per distinct element, as gemmi caches them.
    form_rows: list[np.ndarray] = []
    form_of_element: dict[str, int] = {}
    form_index = np.empty(len(sites), dtype=np.int32)
    for position, site in enumerate(sites):
        name = site.element.name
        row = form_of_element.get(name)
        if row is None:
            coefficients = site.element.it92
            if coefficients is None:  # no IT92 parametrisation - let gemmi fail
                return None
            a = np.asarray(coefficients.a, dtype=float)[:, None]
            b = np.asarray(coefficients.b, dtype=float)[:, None]
            row = len(form_rows)
            form_rows.append(
                (a * np.exp(-b * stol2[None, :])).sum(axis=0)
                + coefficients.c + calculator.addends.get(site.element)
            )
            form_of_element[name] = row
        form_index[position] = row

    reciprocal = cell.reciprocal()
    return {
        'hkl': hkl,
        'stol2': stol2,
        'rotations': np.array(rotations, dtype=float),
        'translations': np.array(translations, dtype=float),
        'fract': np.array([[s.fract.x, s.fract.y, s.fract.z] for s in sites],
                          dtype=float),
        'occupancies': np.array([s.occ for s in sites], dtype=float),
        'u_iso': np.array([s.u_iso for s in sites], dtype=float),
        'aniso': np.array(
            [[s.aniso.u11, s.aniso.u22, s.aniso.u33,
              s.aniso.u12, s.aniso.u13, s.aniso.u23] for s in sites],
            dtype=float),
        'form_index': form_index,
        'form_factors': np.array(form_rows, dtype=float),
        'reciprocal': (reciprocal.a, reciprocal.b, reciprocal.c),
    }


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
