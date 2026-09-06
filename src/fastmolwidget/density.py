"""Residual ``Fo-Fc`` density maps from refined models and reflections.

Workflow:
1. read and merge reflections into the reciprocal ASU;
2. reuse or compute ``F_c`` (with real anomalous term ``f'``);
3. scale observed amplitudes with SHELXL's OSF so ``|Fo| / OSF`` matches
   ``|Fc|``;
4. build unweighted SHELXL-style coefficients
   ``ΔF = (|Fo| / OSF - |Fc|) * exp(i φc)``;
5. expand over the space group and FFT to a periodic unit-cell map in e/Å³.

``WGHT`` is intentionally not applied to Fourier coefficients. Isosurfaces use
the optional :mod:`density_cpp` extension; without it, the feature fails
cleanly. This module is Qt-free.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
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

    #: True only when the compiled extension exposes the needed bindings.
    HAS_DENSITY_CPP: bool = hasattr(density_cpp, 'marching_cubes')
except ImportError:  # pragma: no cover - depends on the compiled extension
    HAS_DENSITY_CPP = False

#: FFT grid spacing in Å. Fixed by cell size, not by data resolution.
#: ``0.15`` preserves residual-feature shape; ``0.3-0.4`` looks blocky.
DEFAULT_GRID_SPACING: float = 0.15

#: Default padding around the displayed atoms, in Å.
DEFAULT_MARGIN: float = 1.5

#: Default contour level as a multiple of map RMS. ``3σ`` is the usual
#: crystallographic threshold, and an absolute level does not transfer between
#: structures.
DEFAULT_SIGMA: float = 3.0

#: Default strength of the weak-data damping
#: ``1 / (1 + w·(σ(F)/|Fc|)^3)``. This is the only smoothing and acts in
#: reciprocal space. ``0.0`` disables it.
DEFAULT_WEAK_WEIGHT: float = 1.0

#: Exponent of ``σ(F)/|Fc|`` in weak-data damping.
WEAK_DATA_EXPONENT: float = 3.0

#: Cubes per cell of the coarse marching-cubes mask. ``8`` keeps the mask small
#: while still skipping most of a molecule's bounding box.
CUBE_MASK_BLOCK: int = 8

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
    'force_isotropic_adps',
    'small_structure_from_block',
    'small_structure_from_cif',
    'small_structure_from_shelx',
]

#: Model source accepted by :func:`calculate_residual_density`.
ModelSource = CifSource | gemmi.SmallStructure


# ---------------------------------------------------------------------------
# The map
# ---------------------------------------------------------------------------

@dataclass
class ResidualDensityMap:
    """Residual ``Fo-Fc`` density over one periodic unit cell."""

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
        """Return ``sigma * rms`` in e/Å³, rounded to 2 decimals, never zero."""
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
        """Extract one wireframe isosurface in Cartesian coordinates.

        Marching cubes runs in fractional space; vertices are converted to
        Cartesian afterwards. With ``atoms``, contour only within ``margin`` Å
        of those atoms.
        """
        return self.isosurfaces((level,), atoms=atoms, margin=margin)[0]

    def isosurfaces(
        self,
        levels: Sequence[float],
        *,
        atoms: np.ndarray | None = None,
        margin: float = DEFAULT_MARGIN,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extract several isosurfaces from one shared grid cut-out."""
        if not HAS_DENSITY_CPP:
            raise RuntimeError(
                'Residual-density isosurfaces need the compiled "density_cpp" '
                'extension. Build it with:  '
                'pip install -e . --no-build-isolation'
            )

        sub, origin_frac, step_frac = self._region(atoms, margin)
        origin = tuple(float(v) for v in origin_frac)
        step = tuple(float(v) for v in step_frac)
        mask = self._cube_mask(sub.shape, origin_frac, step_frac, atoms, margin)

        surfaces: list[tuple[np.ndarray, np.ndarray]] = []
        for level in levels:
            verts, edges = density_cpp.marching_cubes(
                sub, float(level), origin, step,
                mask=mask, block=CUBE_MASK_BLOCK,
            )
            if len(verts):
                verts = verts @ self.orth_matrix.T
                if atoms is not None and len(atoms):
                    verts, edges = _clip_to_atoms(verts, edges, atoms, margin)
            surfaces.append(
                (np.ascontiguousarray(verts, dtype=np.float32), edges))
        return surfaces

    def _cube_mask(
        self,
        shape: tuple[int, ...],
        origin_frac: np.ndarray,
        step_frac: np.ndarray,
        atoms: np.ndarray | None,
        margin: float,
    ) -> np.ndarray | None:
        """Mark cut-out blocks that lie near *atoms*.

        The mask is a conservative over-estimate of each atom's ``margin``
        sphere, so no cube that could contain a wanted vertex is skipped.
        """
        if atoms is None or len(atoms) == 0:
            return None

        cubes = np.array([max(int(n) - 1, 0) for n in shape])
        blocks = (cubes + CUBE_MASK_BLOCK - 1) // CUBE_MASK_BLOCK
        if np.any(blocks <= 0):
            return None

        # Atom positions and margin, both in cut-out grid steps.
        inverse = np.linalg.inv(self.orth_matrix)
        frac = np.asarray(atoms, dtype=float) @ inverse.T
        position = (frac - origin_frac) / step_frac
        pad = (margin * np.linalg.norm(inverse, axis=1)) / step_frac

        low = np.floor((position - pad) / CUBE_MASK_BLOCK).astype(int)
        high = np.floor((position + pad) / CUBE_MASK_BLOCK).astype(int)
        np.clip(low, 0, blocks - 1, out=low)
        np.clip(high, 0, blocks - 1, out=high)

        mask = np.zeros(tuple(int(n) for n in blocks), dtype=bool)
        for (x0, y0, z0), (x1, y1, z1) in zip(low, high):
            mask[x0:x1 + 1, y0:y1 + 1, z0:z1 + 1] = True
        return mask

    def _region(
        self,
        atoms: np.ndarray | None,
        margin: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cut the periodic map to the bounding box around *atoms*.

        This is only a box pre-filter; :func:`_clip_to_atoms` applies the exact
        per-atom radius later.
        """
        shape = np.array(self.array.shape)
        step = 1.0 / shape

        if atoms is None or len(atoms) == 0:
            # One cell plus a duplicated outer layer, so surfaces close across
            # the periodic boundary.
            sub = np.empty(shape + 1, dtype=np.float32)
            idx = [np.arange(n + 1) % n for n in shape]
            sub[:] = self.array[np.ix_(*idx)]
            return sub, np.zeros(3), step

        frac = np.asarray(atoms, dtype=float) @ np.linalg.inv(self.orth_matrix).T
        # Convert Cartesian margin to safe fractional padding. Using inverse-row
        # norms keeps the padding correct for oblique cells.
        inverse = np.linalg.inv(self.orth_matrix)
        pad = margin * np.linalg.norm(inverse, axis=1)
        lo = np.floor((frac.min(axis=0) - pad) * shape).astype(int)
        hi = np.ceil((frac.max(axis=0) + pad) * shape).astype(int) + 1

        idx = [np.arange(a, b) % n for a, b, n in zip(lo, hi, shape)]
        sub = np.ascontiguousarray(self.array[np.ix_(*idx)], dtype=np.float32)
        return sub, lo * step, step


def _neighbour_offsets() -> np.ndarray:
    """The 27 neighbour-cell offsets, ordered nearest first."""
    offsets = np.array([(x, y, z)
                        for x in (-1, 0, 1)
                        for y in (-1, 0, 1)
                        for z in (-1, 0, 1)], dtype=np.int64)
    return offsets[np.argsort((offsets ** 2).sum(axis=1), kind='stable')]


_NEIGHBOUR_OFFSETS: np.ndarray = _neighbour_offsets()


def _dilated(mask: np.ndarray) -> np.ndarray:
    """Grow a boolean 3-D mask by one cell in every direction.

    Three separable passes give the same Chebyshev-distance-1 result as 27
    shifted ORs.
    """
    grown = mask.copy()
    for axis in range(3):
        source = grown.copy()
        front = [slice(None)] * 3
        back = [slice(None)] * 3
        front[axis] = slice(None, -1)
        back[axis] = slice(1, None)
        grown[tuple(front)] |= source[tuple(back)]
        grown[tuple(back)] |= source[tuple(front)]
    return grown


def _clip_candidates(
    vertex_cells: np.ndarray, atom_cells: np.ndarray, shape: np.ndarray,
) -> np.ndarray:
    """Indices of vertices whose bucket neighbourhood contains an atom.

    This is a cheap exact pre-filter for :func:`_clip_to_atoms`.
    """
    # One-cell padding: a vertex may sit one bucket outside the occupied region
    # and still be within ``margin`` of an atom.
    padded = np.zeros(shape + 2, dtype=bool)
    padded[atom_cells[:, 0] + 1, atom_cells[:, 1] + 1, atom_cells[:, 2] + 1] = True
    near_atoms = _dilated(padded)

    cells = vertex_cells + 1
    inside = np.flatnonzero(
        np.all((cells >= 0) & (cells < np.asarray(near_atoms.shape)), axis=1))
    if inside.size == 0:
        return inside
    lookup = cells[inside]
    return inside[near_atoms[lookup[:, 0], lookup[:, 1], lookup[:, 2]]]


def _clip_to_atoms(
    vertices: np.ndarray,
    edges: np.ndarray,
    atoms: np.ndarray,
    margin: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop segments farther than *margin* from every atom.

    The marching-cubes box is larger than the molecule, so corner blobs must be
    removed afterwards. The search uses a vectorised spatial hash with bucket
    size ``margin`` and a cheap dilated-bucket pre-filter; an edge survives
    only if both vertices do.
    """
    if len(vertices) == 0 or len(edges) == 0:
        return vertices, edges
    if margin <= 0.0:
        return (np.empty((0, 3), dtype=vertices.dtype),
                np.empty((0, 2), dtype=edges.dtype))

    atoms = np.asarray(atoms, dtype=np.float32)
    vertices = np.asarray(vertices)
    origin = atoms.min(axis=0) - margin

    # Sort atoms by bucket so each bucket is one contiguous slice.
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

    pending = _clip_candidates(vertex_cells, atom_cells, shape)
    for offset in _NEIGHBOUR_OFFSETS:
        if pending.size == 0:
            break
        cells = vertex_cells[pending] + offset
        inside = np.all((cells >= 0) & (cells < shape), axis=1)
        if not inside.any():
            continue
        todo, cells = pending[inside], cells[inside]
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
        # Flatten the ragged ``vertex -> bucket atoms`` relation.
        total = int(counts.sum())
        offsets = np.repeat(np.cumsum(counts) - counts, counts)
        atom_index = np.repeat(start, counts) + \
            (np.arange(total) - offsets)
        vertex_index = np.repeat(todo, counts)
        delta = sorted_atoms[atom_index] - vertices[vertex_index]
        close = np.einsum('ij,ij->i', delta, delta) <= limit
        keep[vertex_index[close]] = True
        pending = pending[~keep[pending]]

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
    """Replace non-positive-definite ADPs with isotropic equivalents.

    Negative eigenvalues make the Debye-Waller factor grow with resolution and
    corrupt high-angle ``F_c``. The test is valid directly on the stored tensor;
    offending atoms keep ``u_iso`` (or a trace-based fallback) and become
    isotropic.
    """
    replaced: list[str] = []
    sites = structure.sites
    if not sites:
        return replaced

    tensors = np.array(
        [[[s.aniso.u11, s.aniso.u12, s.aniso.u13],
          [s.aniso.u12, s.aniso.u22, s.aniso.u23],
          [s.aniso.u13, s.aniso.u23, s.aniso.u33]] for s in sites],
        dtype=float)

    diagonal = tensors[:, [0, 1, 2], [0, 1, 2]]
    anisotropic = np.any(diagonal != 0.0, axis=1)
    if not np.any(anisotropic):
        return replaced

    # One stacked decomposition instead of one call per atom; large models have
    # thousands of sites and the per-atom route dominated model loading.
    positive = np.zeros(len(sites), dtype=bool)
    positive[anisotropic] = np.all(
        np.linalg.eigvalsh(tensors[anisotropic]) > 0.0, axis=1)

    for position in np.flatnonzero(anisotropic & ~positive):
        site = sites[int(position)]
        fallback = site.u_iso
        if fallback <= 0.0:
            trace = float(diagonal[position].sum()) / 3.0
            fallback = trace if trace > 0.0 else 0.05
        site.aniso = gemmi.SMat33d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        site.u_iso = fallback
        replaced.append(site.label)
    return replaced


def small_structure_from_cif(source: CifSource) -> gemmi.SmallStructure:
    """Read a CIF into a :class:`gemmi.SmallStructure` for SF calculation.

    Leading ``global_`` blocks are skipped; the first block with atom sites is
    used.
    """
    for block in _cif_blocks(source):
        structure = small_structure_from_block(block)
        if structure is not None:
            return structure
    raise ValueError(f'No atom sites found in {_source_name(source)}')


def small_structure_from_block(block) -> gemmi.SmallStructure | None:
    """Build a :class:`gemmi.SmallStructure` from one CIF block.

    ``change_occupancies_to_crystallographic()`` is required so special-position
    atoms contribute with the correct multiplicity.
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


#: SHELX ``LATT`` centring translations in fractional coordinates. The sign
#: selects centrosymmetry; the magnitude selects lattice type.
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

    ``SYMM`` cards are incomplete on their own: ``LATT`` adds centring
    translations, and positive ``LATT`` also adds inversion. Omitting centring
    would silently reduce the group to a primitive subgroup.
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

    Resolves SHELX negative ``U_iso`` riding-atom conventions explicitly:
    ``-1.5`` means ``1.5 * U_eq`` of the pivot atom. Occupancies come from
    ``atom.occupancy``.
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
        # ``type_symbol`` is only a label; scattering factors use ``.element``.
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

    # gemmi reads symmetry from ``UnitCell.images``, not from the space group.
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
    """Apply the ``HKLF`` index transform and scale factors.

    ``HKLF N S r11…r33 sm`` can re-index reflections into the model setting.
    That must happen first, because the cell, symmetry and coordinates in the
    model already refer to the transformed indices. ``S`` scales F² and σ;
    ``sm`` scales σ again.
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
        # ``h'_i = sum_j R_ij h_j``; row vectors therefore use ``R.T``.
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
    """Return each reflection's Miller indices in every twin domain.

    A negative ``TWIN`` count means general plus racemic twinning: the matrix
    generates components ``1…m`` and components ``m+1…2m`` are their Friedel
    opposites.
    """
    law = np.array(matrix, dtype=float).reshape(3, 3)
    generated = components // 2 if racemic else components

    # Row-vector arrays use the transpose of the column-vector law.
    indices = [np.asarray(hkl, dtype=float)]
    for _ in range(max(generated - 1, 0)):
        indices.append(indices[-1] @ law.T)

    if racemic:
        indices += [-block for block in indices[:generated]]
    return indices[:components]


def _unique_index_map(indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduce an ``(..., 3)`` index array to its distinct Miller indices.

    The three components are packed into a single ``int64`` key so that the
    reduction is an ordinary 1-D :func:`numpy.unique` rather than the far
    slower lexicographic ``axis=0`` variant.

    :returns: ``(unique, inverse)`` where ``unique`` is ``(U, 3)`` int32 and
        ``inverse`` has the shape of *indices* without its last axis.
    """
    flat = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    low = flat.min(axis=0)
    span = (flat.max(axis=0) - low + 1)
    shifted = flat - low
    keys = (shifted[:, 0] * span[1] + shifted[:, 1]) * span[2] + shifted[:, 2]

    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    unique = np.ascontiguousarray(flat[first], dtype=np.int32)
    return unique, inverse.reshape(np.shape(indices)[:-1])


def _friedel_folded(flat: np.ndarray) -> np.ndarray:
    """Map every Miller index onto a canonical member of its Friedel pair.

    Only valid for users of ``|Fc|``: the addends are real, so
    ``Fc(-h) = conj(Fc(h))`` and the magnitudes are exactly equal.  Phases are
    *not* preserved, so the merged map coefficients must never fold this way.
    """
    h, k, l = flat[:, 0], flat[:, 1], flat[:, 2]
    negative = (h < 0) | ((h == 0) & ((k < 0) | ((k == 0) & (l < 0))))
    return np.where(negative[:, None], -flat, flat)


def _domain_intensities(
    structure: gemmi.SmallStructure,
    calculator: gemmi.StructureFactorCalculatorX,
    indices: np.ndarray,
) -> np.ndarray:
    """Return ``|Fc|²`` for every Miller index in an ``(..., 3)`` array.

    Detwinning needs ``|Fc|²`` for every twin component of every observation,
    which for a real data set is millions of lookups.  Asking gemmi for them
    one index at a time dominated the whole map calculation, so the indices are
    folded onto Friedel pairs, reduced to the distinct ones and handed to
    :func:`_summed_structure_factors` in a single batched, OpenMP-parallel
    call.  The values are identical to the per-index route; only the number of
    summations changes.

    :returns: A float array shaped like *indices* without its last axis.
    """
    flat = np.asarray(indices, dtype=np.int64).reshape(-1, 3)
    unique, inverse = _unique_index_map(_friedel_folded(flat))
    f_calc = _summed_structure_factors(structure, unique, calculator)
    values = (np.abs(f_calc) ** 2)[inverse]
    return values.reshape(np.shape(indices)[:-1])


class _StructureFactorCache:
    """Memoised ``|Fc|²`` lookups keyed by Miller index."""

    def __init__(self, structure: gemmi.SmallStructure,
                 calculator: gemmi.StructureFactorCalculatorX) -> None:
        self._structure = structure
        self._calculator = calculator
        self._cache: dict[tuple[int, int, int], float] = {}

    def prime(self, indices: np.ndarray) -> None:
        """Pre-compute ``|Fc|²`` for an ``(N, 3)`` block of indices at once."""
        flat = np.asarray(indices, dtype=np.int32).reshape(-1, 3)
        if len(flat) == 0:
            return
        unique, _ = _unique_index_map(flat)
        values = _domain_intensities(self._structure, self._calculator, unique)
        self._cache.update(zip(map(tuple, unique.tolist()), values.tolist()))

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
    """Split twinned intensities into the primary-domain contribution.

    Uses

    .. math::
        F_o^2(h_1) = I_{obs}\\,\\frac{|F_c(h_1)|^2}{\\sum_k b_k |F_c(h_k)|^2}

    Handles ``HKLF 4 + TWIN`` and ``HKLF 5``. In ``HKLF 4``, negative batch
    values are *not* overlap markers; they may be *R*\\ :sub:`free` flags.

    A pure inversion (racemic) twin is effectively a no-op here: ``h`` and
    ``-h`` differ only through the imaginary anomalous term ``f''``, which
    gemmi's real-valued addends cannot express, so the map stays slightly too
    large.
    """
    calculator = gemmi.StructureFactorCalculatorX(structure.cell)
    if params.wavelength:
        calculator.addends.add_cl_fprime(gemmi.hc / params.wavelength)
    fractions = params.twin_fractions()

    if params.hklf != 5:
        return _detwin_hklf4(reflections, structure, params, calculator,
                             fractions)

    cache = _StructureFactorCache(structure, calculator)
    cache.prime(reflections.hkl)
    groups = _hklf5_groups(reflections)

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


def _detwin_hklf4(
    reflections: ReflectionData,
    structure: gemmi.SmallStructure,
    params: ShelxParameters,
    calculator: gemmi.StructureFactorCalculatorX,
    fractions: list[float],
) -> ReflectionData:
    """Detwin ``HKLF 4`` data without a per-reflection Python loop.

    Every record contributes exactly one observation whose twin components are
    generated from the twin law, so the whole data set is a regular
    ``(N, components, 3)`` index block.  All of it goes through
    :func:`_domain_intensities` in one batched call, and the share is then a
    plain array expression - the result is identical to evaluating the groups
    one at a time.
    """
    matrix = params.twin_matrix or _DEFAULT_TWIN_MATRIX
    domains = _twin_domain_indices(reflections.hkl, matrix,
                                   params.twin_components,
                                   racemic=params.twin_racemic)
    # (N, components, 3)
    indices = np.rint(np.stack(domains, axis=1)).astype(np.int32)
    intensities = _domain_intensities(structure, calculator, indices)

    weights = np.zeros(indices.shape[1], dtype=float)
    usable = min(len(fractions), len(weights))
    weights[:usable] = fractions[:usable]

    total = intensities @ weights
    keep = total > 0.0
    if not np.any(keep):
        raise ValueError('Detwinning left no usable reflections')

    share = intensities[keep, 0] / total[keep]
    return ReflectionData(
        hkl=np.ascontiguousarray(indices[keep, 0, :]),
        f_sq_meas=reflections.f_sq_meas[keep] * share,
        sigma=reflections.sigma[keep] * share,
        sigma_known=reflections.sigma_known,
    )


def _hklf5_groups(reflections: ReflectionData):
    """Yield ``(primary, members, F², σ)`` for ``HKLF 5`` overlap groups.

    Consecutive records form one observation; all but the last have a negative
    component number. Domain 1 is preferred as the primary index.
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
    """Locate the reflection data for *model*.

    In-memory CIF objects can only supply embedded data; file paths also get
    same-basename sibling lookup.
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
    iso_u_override: float | None = None,
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
    :param iso_u_override: When given, every atom's ADP is replaced by an
        isotropic ``U`` of this value (Å²) before *F*\\ :sub:`c` is summed,
        see :func:`force_isotropic_adps`.  Used for interactive disorder-
        moiety fitting, where refined ADPs would otherwise bias the shape of
        the alternate-site peak.  ``None`` (the default) uses the model's own
        ADPs.
    :returns: The computed map.
    :raises FileNotFoundError: If *hkl_path* is ``None`` and no reflection
        data could be found for the model.
    :raises ValueError: If the model cannot be interpreted or the reflection
        data contains nothing usable.
    """
    model = model_path if _is_model_object(model_path) else Path(model_path)
    structure, params = _load_model(model)
    if iso_u_override is not None:
        structure = force_isotropic_adps(structure, iso_u_override)
    if structure.spacegroup is None:
        raise ValueError(
            f'Could not determine the space group of {_source_name(model)}')

    if hkl_path is None:
        hkl_path = _find_reflections_for(model)
        if hkl_path is None:
            raise FileNotFoundError(_no_reflections_message(model))

    reflections = read_reflections(hkl_path)

    if not reflections.has_f_calc:
        # HKLF re-indexing must be applied before any downstream use.
        reflections = _apply_hklf_transform(reflections, params)

        # Twinned data must be split before map calculation.
        if params.is_twinned:
            reflections = _detwin_observations(reflections, structure, params)

    hkl, f_obs, sigma = _merge_to_asu(reflections, structure.spacegroup, d_min,
                                      structure.cell)
    if len(hkl) == 0:
        raise ValueError(f'No usable reflections in {_source_name(hkl_path)}')

    f_calc = _calculated_structure_factors(structure, hkl, params, reflections)
    scale = _scale_factor(params, f_obs, np.abs(f_calc))

    # Fix the grid from the cell alone; drop reflections finer than it can
    # represent so map size never follows data resolution.
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
        # Bring σ onto the calculated scale, just like ``|Fo|``.
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


def force_isotropic_adps(
    structure: gemmi.SmallStructure, u_iso: float,
) -> gemmi.SmallStructure:
    """Return a copy of *structure* with every site forced isotropic at *u_iso*.

    Used to compute a residual-density map for interactively fitting a
    disorder moiety: a refined ADP tends to "suction" the electron density of
    its own disorder partner into itself, distorting or hiding the very peak
    the map is meant to reveal.  Flattening every atom to the same small,
    plausible ``U`` (0.04-0.05 Å² is a typical well-behaved value) removes that
    bias so the alternate-site peak shows its own shape.

    :param structure: The structure to copy (not modified in place).
    :param u_iso: The isotropic displacement parameter to assign to every
        site, in Å².
    :returns: A new :class:`gemmi.SmallStructure` with the same sites,
        occupancies and symmetry, but flattened ADPs.
    """
    flat = gemmi.SmallStructure()
    flat.name = structure.name
    flat.cell = structure.cell
    flat.spacegroup = structure.spacegroup
    flat.wavelength = structure.wavelength
    for site in structure.sites:
        new_site = site.clone()
        new_site.aniso = gemmi.SMat33d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        new_site.u_iso = u_iso
        flat.add_site(new_site)
    # The structure-factor calculator reads symmetry from the cell images,
    # not from the space group alone (see small_structure_from_shelx); a
    # freshly built SmallStructure does not inherit that cache.
    flat.setup_cell_images()
    return flat


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

    Evaluated over the whole reflection array at once - the per-reflection
    Python loop this replaces spent most of its time in one
    :meth:`gemmi.UnitCell.calculate_d` call per Miller index, which
    :meth:`~gemmi.UnitCell.calculate_d_array` does for the entire set in a
    single call.  The result is identical to machine precision.
    """
    if not params.exti:
        return f_calc

    lambda_ = params.wavelength
    d = cell.calculate_d_array(hkl)
    sin_theta = np.minimum(lambda_ / (2.0 * d), 1.0)
    sin_2theta = np.maximum(
        2.0 * sin_theta * np.sqrt(np.maximum(1.0 - sin_theta ** 2, 0.0)), 1e-6)
    factor = (1.0 + 0.001 * params.exti * np.abs(f_calc) ** 2
              * lambda_ ** 3 / sin_2theta) ** -0.25
    return f_calc * factor


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
