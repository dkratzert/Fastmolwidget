"""
Interactive disorder-moiety dragging: pure math, no Qt and no OpenGL.

Lets a user pick either one or more fixed "anchor" atoms, or a single
**bond**, as the border between a disordered fragment and the rest of the
molecule, and then drag any atom of the fragment ("moiety") with the mouse to
reposition it, guided by a residual (Fo-Fc) density map towards the
alternate-site peak.  This module only implements the geometry/optimisation;
:mod:`fastmolwidget.molecule3D` wires it to mouse events.

Splitting at a **bond** (pass ``bond=`` to :func:`build_drag_session`) is
handled by :class:`TorsionDrag`: the bond end away from the grabbed atom
becomes the sole anchor, the bond becomes the axis, and the fragment rotates
about it with elastic give on top (:data:`TORSION_FLEXIBILITY`).  The near
bond end travels *with* the fragment rather than being pinned, so it can
drift off the axis and a tumble - not just a clean torsion - can be
modelled.  See :func:`bond_split_ends` for how the two ends are told apart,
including inside a ring, where cutting one bond separates nothing and the
ring simply deforms elastically as it rotates.

Two rigid/elastic modes are implemented, but :func:`build_drag_session`
always picks the elastic one (see below) regardless of anchor count:

* **One anchor** — the moiety is a rigid body pinned to that single point
  (ball-and-socket).  Dragging any atom rotates the whole fragment about the
  anchor; all internal distances are preserved exactly because it is a pure
  rotation (:class:`RigidPivotDrag`).  Still available for direct use/testing,
  but not used by :func:`build_drag_session`.
* **Two or more anchors** — a rigid body would be over-constrained, so the
  moiety is treated as an elastic mass-spring system: every bonded pair (both
  inside the moiety and between the moiety and an anchor) is a distance
  constraint with the original bond length as its rest length, solved with a
  few Gauss-Seidel relaxation passes per drag step (:class:`ElasticDrag`).
  :func:`moiety_angle_pairs` adds a second, weaker kind of constraint between
  every 1,3 pair of atoms (two sharing a bonded neighbour) at their original
  distance, so bond *angles* stay roughly fixed too - without it, a 1,2-only
  spring system can freely fold or flatten around any shared atom while
  every individual bond length still checks out, which looks unstable and
  "floppy".  These 1,3 pairs are deliberately **loose restraints**, applied
  at :data:`ANGLE_CONSTRAINT_STIFFNESS` (30 %) of a real bond's correction
  each iteration (:class:`ElasticDrag`'s ``soft_edges``), so they stabilise
  the shape without fighting - and degrading the convergence of - the real
  1,2 bonds.  Anchors have infinite mass (never move).
  :func:`build_drag_session` always uses this mode, even for a single
  anchor, for a consistently "gummy" feel.  Optional atomic masses
  (:func:`atomic_mass`) let a constraint's correction split unevenly between
  two solved atoms of different weight.

Terminal hydrogens ride exactly rather than take part in the spring solve at
all: :class:`ElasticDrag`'s ``riding`` mapping (atom → its parent) keeps such
an atom at its *exact* original offset from the parent - identical bond
length and direction, always - translated by however far the parent moved.
This is a hard rule, not an approximation, which is what "a hydrogen keeps
the same relative geometry to its carbon" means physically.

:class:`DensityGuide` samples a residual-density grid (trilinear
interpolation) and its numerical gradient, so the moiety can be nudged
towards a nearby density peak while it is dragged, and so the session can
detect when it has settled onto one ("snapped").

:class:`MoietyDragSession` ties everything together and is the only object
:mod:`molecule3D` needs to construct and drive.

Deliberately **not** implemented: automatically testing whether an inverted/
mirrored copy of the dragged fragment fits the density better at each step.
A version of this was built and reverted - it seeded a second, independent
:class:`ElasticDrag` from a mirror-image starting layout (through the single
anchor, or through the grabbed atom itself with none) and picked whichever
of the two parallel solves scored better against the density map every
update.  It was numerically correct for the simple cases it was tested
against (bond lengths, angles and riding offsets all stayed exact in either
orientation), but in practice did not work out nicely enough to keep: with
no anchor at all, or with an anchor several bonds away from the grabbed
atom, the "genuine" mirror solution for the atoms in between is the
reflection across the anchor-to-grabbed *line*, not simply a physically
intuitive flip, so the alternate orientation a user actually expects and
the one this approach converges to often disagreed.  Combined with running
a whole second relaxation every mouse-move and the possibility of the two
candidates flip-flopping between drag steps, it made the interaction feel
unpredictable rather than helpful, so it was removed again.  If this is
revisited, a live preview of both candidates (rather than silently swapping
which one is shown) is probably a precondition for it to be usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

import numpy as np

__all__ = [
    'ANGLE_CONSTRAINT_STIFFNESS',
    'BREAKAWAY_DISTANCE',
    'DEFAULT_ISO_U',
    'DENSITY_GRADIENT_STEP',
    'DENSITY_NUDGE_STRENGTH',
    'SNAP_GRADIENT_TOL',
    'TORSION_FLEXIBILITY',
    'DensityGuide',
    'ElasticDrag',
    'MoietyDragSession',
    'RigidPivotDrag',
    'TorsionDrag',
    'atomic_mass',
    'bond_split_ends',
    'build_drag_session',
    'find_moiety',
    'moiety_angle_pairs',
    'moiety_edges',
]

#: Isotropic U (Å²) used for the dedicated density map computed for fitting a
#: disorder moiety - see :func:`fastmolwidget.density.force_isotropic_adps`.
#: A refined ADP "suctions" its own disorder partner's density into itself;
#: flattening every atom to a small, plausible U removes that bias.
DEFAULT_ISO_U: float = 0.045

#: Finite-difference step (Å) used by :meth:`DensityGuide.gradient`.  A wider
#: step averages the slope over a bigger neighbourhood, so a nearby peak is
#: sensed - and starts pulling the atom - from farther away, effectively
#: enlarging the "capture radius" within which snapping can engage.  A small value as
# ``0.05`` would be a near-pointwise derivative that only reacts
#: once an atom is almost on top of a peak.
DENSITY_GRADIENT_STEP: float = 0.1

#: How far (Å) a moiety atom is nudged, per drag update, along the normalised
#: local density gradient.  Strong enough that the pull towards a peak is
#: clearly felt rather than a gentle suggestion; the elastic/rigid solver
#: re-applies afterward so the fragment's own geometry is not destroyed by
#: the nudge.
DENSITY_NUDGE_STRENGTH: float = 0.15

#: Gradient magnitude (e/Å⁴, roughly) below which the moiety is considered to
#: have settled into a local density maximum ("snapped").  Loosened together
#: with :data:`DENSITY_GRADIENT_STEP`: a wider finite-difference step reads a
#: larger ambient slope even fairly close to a peak, so the threshold that
#: decides "close enough to lock in" has to scale up with it.
SNAP_GRADIENT_TOL: float = 0.2

#: Extra mouse-target displacement (Å) required to leave a snapped pose - the
#: "more force" needed to drag the moiety further once it has snapped.
BREAKAWAY_DISTANCE: float = 0.7

#: Stiffness of the 1,3 (angle-stabilising) distance constraints relative to
#: the real 1,2 bonds (stiffness ``1.0``).  Deliberately loose: these pairs
#: are not real bonds, only there to stop the moiety folding through an
#: angle while it is dragged, so they must not compete with - and degrade
#: the convergence of - the actual bond-length constraints.
ANGLE_CONSTRAINT_STIFFNESS: float = 0.3

#: How far the grabbed atom is allowed to leave the ideal torsion path
#: towards where the mouse actually is, as a fraction of that offset per
#: :meth:`TorsionDrag.update` (``0.0`` = a perfectly rigid rotation about the
#: bond, ``1.0`` = the mouse fully overrides the rotation).  Rotation about
#: the selected bond stays the dominant motion; this is the "bit of
#: flexibility" on top, and it is also what lets the near bond atom drift off
#: the axis so a tumble - rather than only a clean torsion - can be modelled.
TORSION_FLEXIBILITY: float = 0.5


def atomic_mass(element: str) -> float:
    """Return the standard atomic weight of *element* in u (g/mol).

    Thin wrapper around :class:`gemmi.Element` so callers building the
    ``masses`` argument of :func:`build_drag_session` / :class:`ElasticDrag`
    don't need their own periodic table.  Deuterium (``"D"``) is recognised
    with its correct (roughly doubled) mass; unrecognised symbols fall back
    to gemmi's dummy element weight of ``1.0``, which is deliberately light
    so an unknown label still behaves like a hydrogen rather than silently
    acting as an infinitely heavy anchor.
    """
    import gemmi

    return float(gemmi.Element(element).weight)


# ---------------------------------------------------------------------------
# Moiety discovery
# ---------------------------------------------------------------------------

def find_moiety(
    connections: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    anchors: set[int] | frozenset[int],
    start: int,
) -> set[int]:
    """Return the fragment reachable from *start* without passing an anchor.

    Anchors are the fixed, non-disordered atoms at the border; the moiety is
    the connected component containing *start* once every anchor is removed
    from the bond graph.  Starting from a specific atom (rather than
    expanding from every anchor neighbour) is what correctly picks *one*
    branch when an anchor has other, unrelated neighbours on the rigid part
    of the molecule — those must not be swept into the moiety.  Anchors
    themselves are never included in the result.

    :param connections: Iterable of ``(i, j)`` atom-index pairs, as stored in
        ``MoleculeWidget3D.connections``.
    :param anchors: Fixed atom indices marking the border of the moiety.
    :param start: The atom under the cursor when the drag began; must not
        itself be an anchor.
    :returns: The set of atom indices belonging to the moiety.  Empty when
        *start* is itself an anchor or has no bonds at all.
    """
    anchors = set(anchors)
    if start in anchors:
        return set()

    adjacency: dict[int, set[int]] = {}
    for a, b in connections:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    moiety: set[int] = {start}
    frontier: list[int] = [start]
    while frontier:
        current = frontier.pop()
        for neighbour in adjacency.get(current, ()):
            if neighbour in anchors or neighbour in moiety:
                continue
            moiety.add(neighbour)
            frontier.append(neighbour)

    return moiety


def bond_split_ends(
    connections: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    bond: tuple[int, int],
    grabbed_index: int,
) -> tuple[int, int] | None:
    """Return ``(far, near)`` for splitting the molecule at *bond*.

    Splitting at a bond rather than at an atom means one end of the bond
    becomes the fixed anchor and the other end travels with the dragged
    fragment.  Which is which depends on where the user grabbed: the end on
    the same side as *grabbed_index* is the *near* one (it moves), the other
    is the *far* one (it is the anchor, and together with *near* it defines
    the torsion axis the fragment swings about).

    Sidedness is decided by a breadth-first search from *grabbed_index* with
    only the ``bond`` edge removed - not the whole atom - so a genuine split
    point cleanly separates the two ends.  Inside a ring, cutting one bond
    separates nothing and both ends stay reachable; the nearer one by graph
    distance is then taken, which still gives the axis the user pointed at
    and lets the ring rotate about that bond (deforming elastically) rather
    than refusing the drag.

    :param connections: Every bond in the molecule as ``(i, j)`` atom-index
        pairs.
    :param bond: The selected bond as an ``(i, j)`` atom-index pair.
    :param grabbed_index: The atom under the cursor when the drag started.
    :returns: ``(far, near)``, or ``None`` when *bond* is malformed (both
        ends identical) or neither end can be reached from *grabbed_index*,
        i.e. the bond has nothing to do with the fragment being dragged.
    """
    a, b = bond
    if a == b:
        return None

    adjacency: dict[int, set[int]] = {}
    for i, j in connections:
        adjacency.setdefault(i, set()).add(j)
        adjacency.setdefault(j, set()).add(i)

    # Breadth-first distances from the grabbed atom, never crossing the bond
    # itself, so an acyclic fragment only ever reaches the near end.
    cut = {a: b, b: a}
    distance: dict[int, int] = {grabbed_index: 0}
    frontier: list[int] = [grabbed_index]
    while frontier:
        next_frontier: list[int] = []
        for current in frontier:
            for neighbour in adjacency.get(current, ()):
                if cut.get(current) == neighbour:
                    continue  # this is the bond being split - do not cross it
                if neighbour in distance:
                    continue
                distance[neighbour] = distance[current] + 1
                next_frontier.append(neighbour)
        frontier = next_frontier

    d_a, d_b = distance.get(a), distance.get(b)
    if d_a is None and d_b is None:
        return None
    if d_b is None:
        return b, a
    if d_a is None:
        return a, b
    # Both reachable: a ring, so fall back to whichever end is closer.
    return (b, a) if d_a <= d_b else (a, b)


def moiety_edges(
    connections: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    moiety: set[int],
    anchors: set[int],
) -> list[tuple[int, int]]:
    """Return the bonds relevant to dragging *moiety*.

    Includes every bond entirely inside the moiety, plus every bond between a
    moiety atom and one of *anchors* (these are the constraints that hold the
    fragment in place).  Used both as :class:`ElasticDrag`'s 1,2-distance
    constraint set and as the actual bond list of a duplicated moiety (see
    ``MoleculeWidget3D._create_disorder_duplicate``) - so this must keep
    returning *real bonds only*.  Use :func:`moiety_angle_pairs` for the
    additional 1,3 constraints that stabilise the shape without being real
    bonds themselves.
    """
    edges: list[tuple[int, int]] = []
    for a, b in connections:
        a_in, b_in = a in moiety, b in moiety
        if a_in and b_in or a_in and b in anchors or b_in and a in anchors:
            edges.append((a, b))
    return edges


def moiety_angle_pairs(
    connections: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    moiety: set[int],
    anchors: set[int],
) -> list[tuple[int, int]]:
    """Return 1,3 atom pairs (two bonds apart) relevant to dragging *moiety*.

    A 1,2-distance (bond-length) constraint alone lets the *angle* at any
    shared atom swing freely - two atoms bonded to the same third atom can
    fold towards or away from each other while both bonds stay exactly the
    right length.  Adding a distance constraint directly between such a
    1,3-pair, with their original distance as rest length, keeps that angle
    approximately fixed too (the law of cosines ties a triangle's angle to
    its three side lengths), which is what keeps a dragged moiety's shape
    recognisable instead of collapsing into a zig-zag.

    :param connections: Every bond in the molecule as ``(i, j)`` atom-index
        pairs.
    :param moiety: The atom indices being dragged.
    :param anchors: The fixed border atoms.
    :returns: Pairs ``(a, b)`` with a common bonded neighbour, excluding any
        pair that is already a direct bond (e.g. in a 3-membered ring - that
        distance is already constrained by :func:`moiety_edges`) and any
        pair entirely among *anchors* (fixed relative to each other anyway,
        nothing to constrain).
    """
    edges = moiety_edges(connections, moiety, anchors)
    direct_bonds = {frozenset(edge) for edge in edges}

    adjacency: dict[int, set[int]] = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    pairs: set[frozenset[int]] = set()
    for neighbours in adjacency.values():
        for n1, n2 in combinations(neighbours, 2):
            pair = frozenset((n1, n2))
            if pair in direct_bonds:
                continue
            if n1 in moiety or n2 in moiety:
                pairs.add(pair)

    return [tuple(sorted(pair)) for pair in pairs]


# ---------------------------------------------------------------------------
# Rigid single-pivot rotation
# ---------------------------------------------------------------------------

def _rotation_between(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Shortest-arc rotation matrix taking unit vector *u* onto unit vector *v*."""
    cross = np.cross(u, v)
    cos_a = float(np.dot(u, v))
    sin_a = float(np.linalg.norm(cross))
    if sin_a < 1e-12:
        if cos_a > 0.0:
            return np.eye(3)
        # 180 degree rotation: pick any axis perpendicular to u.
        axis = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - u * np.dot(axis, u)
        axis /= np.linalg.norm(axis)
        return _axis_angle(axis, np.pi)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ])
    return np.eye(3) + skew + skew @ skew * ((1.0 - cos_a) / (sin_a * sin_a))


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation matrix for *angle* radians about a unit-length *axis* (Rodrigues)."""
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ])


class RigidPivotDrag:
    """Rotates a rigid fragment about a single fixed pivot to follow a target.

    Positions are always recomputed from the *original* (drag-start)
    geometry, never accumulated incrementally, so repeated calls to
    :meth:`update` cannot drift or lose precision.
    """

    def __init__(self, pivot: np.ndarray, base_positions: dict[int, np.ndarray]):
        """
        :param pivot: The fixed anchor position (Cartesian, Å).
        :param base_positions: The moiety atoms' positions at drag-start,
            keyed by atom index.
        """
        self.pivot = np.asarray(pivot, dtype=float)
        self._base_offsets = {
            i: np.asarray(p, dtype=float) - self.pivot for i, p in base_positions.items()
        }

    def update(self, grabbed_index: int, target: np.ndarray) -> dict[int, np.ndarray]:
        """Return every moiety atom's new position for a *target* grab point.

        :param grabbed_index: The atom index being dragged; its distance
            from the pivot is preserved (the mouse target is projected onto
            that sphere by direction only).
        :param target: The world-space point the grabbed atom is being
            pulled towards.
        """
        target = np.asarray(target, dtype=float)
        v0 = self._base_offsets[grabbed_index]
        r0 = float(np.linalg.norm(v0))
        if r0 < 1e-9:
            return {i: self.pivot + v for i, v in self._base_offsets.items()}

        v1 = target - self.pivot
        r1 = float(np.linalg.norm(v1))
        if r1 < 1e-9:
            rotation = np.eye(3)
        else:
            rotation = _rotation_between(v0 / r0, v1 / r1)
        return {i: self.pivot + rotation @ v for i, v in self._base_offsets.items()}


# ---------------------------------------------------------------------------
# Elastic multi-anchor drag
# ---------------------------------------------------------------------------

class ElasticDrag:
    """Mass-spring distance-constraint solver for a moiety with 2+ anchors.

    A simplified position-based-dynamics (PBD) solver: each bonded pair keeps
    its original length as a rest length, anchors have infinite mass (never
    move), and the grabbed atom is pinned to the mouse target each iteration
    (a kinematic drive point).  A handful of Gauss-Seidel passes per call give
    the fragment a soft, "gummy" feel instead of an instant rigid snap.
    """

    def __init__(
        self,
        positions: dict[int, np.ndarray],
        anchors: set[int],
        edges: list[tuple[int, int]],
        iterations: int = 12,
        masses: dict[int, float] | None = None,
        riding: dict[int, int] | None = None,
        soft_edges: list[tuple[int, int]] | None = None,
    ):
        """
        :param positions: Starting positions of every atom involved (moiety,
            anchors, *and* any riding atoms), keyed by atom index.
        :param anchors: Indices that never move.
        :param edges: Bonded pairs to keep at their original length (full
            stiffness), from :func:`moiety_edges`.
        :param iterations: Constraint-relaxation passes per :meth:`update`.
        :param masses: Optional per-atom mass (any consistent unit, e.g.
            atomic mass units), keyed by atom index.  Between two movable
            atoms of a constrained pair that are *not* riding, the lighter
            one absorbs the larger share of every correction (see
            :meth:`update`).  Missing entries default to ``1.0``; ``None``
            gives every atom equal mass.
        :param riding: Maps a riding atom's index to its parent's index (e.g.
            a terminal hydrogen and the heavy atom it is bonded to).  Riding
            atoms take no part in the spring solve at all - they simply keep
            their exact original offset from the parent, translated by
            however far the parent moved, so a hydrogen stays glued to its
            carbon with identical bond length and direction (true riding,
            not an approximation).  A riding atom that is also
            *grabbed_index* is treated as a normal solved atom instead (it
            must follow the mouse exactly).
        :param soft_edges: Optional additional pairs to keep at their
            original length, but only loosely: each correction is scaled by
            :data:`ANGLE_CONSTRAINT_STIFFNESS` instead of applied in full, so
            these constraints stabilise the shape (typically 1,3 pairs, see
            :func:`moiety_angle_pairs`) without fighting - and degrading the
            convergence of - the real bonds in *edges*.
        """
        self.anchors = set(anchors)
        self.iterations = iterations
        self.masses = dict(masses) if masses else {}
        self._base_positions = {i: np.array(p, dtype=float) for i, p in positions.items()}
        self.riding = dict(riding) if riding else {}
        self.positions = {
            i: np.array(p, dtype=float) for i, p in positions.items()
            if i not in self.riding
        }
        self._rest_length: dict[tuple[int, int], tuple[float, float]] = {}
        for stiffness, pairs in ((1.0, edges), (ANGLE_CONSTRAINT_STIFFNESS, soft_edges or [])):
            for a, b in pairs:
                if a in self.riding or b in self.riding:
                    continue
                length = float(np.linalg.norm(self._base_positions[a] - self._base_positions[b]))
                self._rest_length[(a, b)] = (length, stiffness)

    def _mass(self, index: int) -> float:
        return self.masses.get(index, 1.0)

    def update(
        self,
        grabbed_index: int,
        target: np.ndarray,
        external_forces: dict[int, np.ndarray] | None = None,
        pin_grabbed: bool = True,
    ) -> dict[int, np.ndarray]:
        """Relax the moiety towards *target* and return the new positions.

        :param grabbed_index: The atom index being dragged; pinned to
            *target* at the start of every relaxation pass.  If this is a
            riding atom, it is solved normally for this call instead (it has
            to follow the mouse exactly, so it cannot also just trail its
            parent).
        :param target: The world-space point the grabbed atom follows.
        :param external_forces: Optional per-atom displacement added once
            before the constraint passes (e.g. a density-gradient nudge);
            the constraint solve then partially, but not fully, absorbs it,
            which is what gives the density guidance a soft pull rather than
            an instant jump.
        :param pin_grabbed: When ``True`` (the default) the grabbed atom is
            held exactly at *target* throughout, so it tracks the mouse
            perfectly.  :class:`TorsionDrag` passes ``False``: there the
            fragment's position comes from a rotation about the bond and the
            grabbed atom is only *pulled* towards the mouse by a soft
            external force, so pinning it would override the rotation
            entirely.  With ``False`` the grabbed atom takes part in the
            relaxation like any other movable atom and *target* is ignored.
        :returns: Positions for every atom that is not an anchor, including
            riding atoms.
        """
        if grabbed_index in self.riding:
            # Grabbed a riding atom directly: solve it like any other atom
            # for this (and every subsequent) call instead of deriving its
            # position from a parent it is simultaneously supposed to be
            # dragging around.
            del self.riding[grabbed_index]
            self.positions[grabbed_index] = self._base_positions[grabbed_index].copy()

        pos = self.positions
        if pin_grabbed:
            pos[grabbed_index] = np.asarray(target, dtype=float)

        if external_forces:
            for i, delta in external_forces.items():
                if i not in pos or i in self.anchors:
                    continue
                if pin_grabbed and i == grabbed_index:
                    continue
                pos[i] = pos[i] + delta

        movable = set(pos) - self.anchors
        if pin_grabbed:
            movable -= {grabbed_index}
        for _ in range(self.iterations):
            for (a, b), (rest, stiffness) in self._rest_length.items():
                pa, pb = pos[a], pos[b]
                delta = pb - pa
                dist = float(np.linalg.norm(delta))
                if dist < 1e-9:
                    continue
                correction = (dist - rest) / dist * stiffness
                move_a = a in movable
                move_b = b in movable
                if move_a and move_b:
                    # Mass-weighted split conserving the pair's centre of
                    # mass: the lighter atom's share of the correction is the
                    # heavier atom's mass fraction, and vice versa, so a
                    # light hydrogen trails a heavy neighbour almost exactly
                    # (riding model) while two similar masses split evenly.
                    m_a, m_b = self._mass(a), self._mass(b)
                    total_mass = m_a + m_b
                    frac_a = m_b / total_mass
                    frac_b = m_a / total_mass
                    pos[a] = pa + frac_a * correction * delta
                    pos[b] = pb - frac_b * correction * delta
                elif move_a:
                    pos[a] = pa + correction * delta
                elif move_b:
                    pos[b] = pb - correction * delta
            if pin_grabbed:
                pos[grabbed_index] = np.asarray(target, dtype=float)

        result = {i: p.copy() for i, p in pos.items() if i not in self.anchors}

        # Riding atoms never entered the spring solve: they simply keep
        # their exact original offset from the parent, translated by
        # however far the parent moved - identical bond length and
        # direction to the parent, always, not just approximately.
        for riding_atom, parent in self.riding.items():
            parent_new = pos.get(parent, self._base_positions[parent])
            parent_old = self._base_positions[parent]
            offset = self._base_positions[riding_atom] - parent_old
            result[riding_atom] = parent_new + offset

        return result


# ---------------------------------------------------------------------------
# Torsion drag: rotation about a selected bond, with elastic give
# ---------------------------------------------------------------------------

class TorsionDrag:
    """Rotate a fragment about a selected bond, with a bit of flexibility.

    Used when the split point is a **bond** rather than a set of atoms.  The
    far end of that bond is the fixed anchor, the near end travels with the
    fragment, and the bond itself is the axis the fragment swings about (see
    :func:`bond_split_ends`).

    Each :meth:`update` does two things, in order:

    1. **Rotate.**  The angle about the axis that best carries the grabbed
       atom's *original* position towards the mouse target is computed, and
       every moiety atom is rigidly rotated about the axis by it.  This is
       recomputed from the stored base positions every call - never
       accumulated - so repeated updates cannot drift, exactly as in
       :class:`RigidPivotDrag`.  Rotation about the bond stays the dominant,
       intended motion.
    2. **Relax.**  Those rotated positions seed an internal
       :class:`ElasticDrag`, which is then run with the grabbed atom *not*
       pinned (``pin_grabbed=False``) and only softly pulled towards where
       the mouse actually is, by :data:`TORSION_FLEXIBILITY` of the offset.
       This is the "bit of flexibility": the fragment may deviate from a
       perfect rotation, deform, and be pulled by the density - and, because
       the near bond atom is an ordinary moiety member rather than an anchor,
       it may drift off the axis so a *tumble* rather than only a clean
       torsion can be modelled.

    A bond inside a ring is not rejected: cutting it separates nothing, so
    the moiety simply comes out larger and the ring deforms elastically as it
    rotates.
    """

    def __init__(
        self,
        positions: dict[int, np.ndarray],
        far: int,
        near: int,
        edges: list[tuple[int, int]],
        iterations: int = 12,
        masses: dict[int, float] | None = None,
        riding: dict[int, int] | None = None,
        soft_edges: list[tuple[int, int]] | None = None,
        flexibility: float = TORSION_FLEXIBILITY,
    ):
        """
        :param positions: Starting positions of every atom involved (moiety,
            the anchor, *and* any riding atoms), keyed by atom index.
        :param far: The bond end that stays fixed - the anchor.
        :param near: The bond end that travels with the fragment.  Together
            with *far* it defines the rotation axis.
        :param edges: Bonded pairs to keep at their original length, from
            :func:`moiety_edges`.
        :param iterations: Relaxation passes per :meth:`update`.
        :param masses: Optional per-atom mass, see :class:`ElasticDrag`.
        :param riding: Optional riding-atom mapping, see :class:`ElasticDrag`.
        :param soft_edges: Optional loose 1,3 restraints, see
            :class:`ElasticDrag`.
        :param flexibility: How far the grabbed atom may leave the ideal
            torsion path towards the mouse, see :data:`TORSION_FLEXIBILITY`.
        """
        self.far = far
        self.near = near
        self.anchors = {far}
        self.flexibility = float(flexibility)
        self._base_positions = {i: np.array(p, dtype=float) for i, p in positions.items()}

        self._origin = self._base_positions[far]
        axis = self._base_positions[near] - self._origin
        norm = float(np.linalg.norm(axis))
        # A zero-length bond has no meaningful axis; fall back to +Z so the
        # solver still runs (as a pure elastic drag) instead of dividing by 0.
        self._axis = axis / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

        self._solver = ElasticDrag(
            positions, {far}, edges, iterations=iterations, masses=masses,
            riding=riding, soft_edges=soft_edges,
        )

    @property
    def positions(self) -> dict[int, np.ndarray]:
        """The internal solver's live positions (used for density nudges)."""
        return self._solver.positions

    @property
    def riding(self) -> dict[int, int]:
        """The internal solver's riding-atom mapping."""
        return self._solver.riding

    def _rotation_towards(self, grabbed_index: int, target: np.ndarray) -> np.ndarray:
        """Rotation matrix about the bond axis aiming *grabbed_index* at *target*.

        Both the grabbed atom's original position and the target are
        projected onto the plane perpendicular to the axis; the signed angle
        between those two projections is the torsion angle.  Returns the
        identity when either projection is degenerate (the point lies
        essentially *on* the axis, where no rotation is defined).
        """
        base = self._base_positions[grabbed_index] - self._origin
        wanted = np.asarray(target, dtype=float) - self._origin

        base_perp = base - self._axis * float(np.dot(base, self._axis))
        wanted_perp = wanted - self._axis * float(np.dot(wanted, self._axis))

        base_norm = float(np.linalg.norm(base_perp))
        wanted_norm = float(np.linalg.norm(wanted_perp))
        if base_norm < 1e-9 or wanted_norm < 1e-9:
            return np.eye(3)

        u = base_perp / base_norm
        v = wanted_perp / wanted_norm
        angle = float(np.arctan2(float(np.dot(np.cross(u, v), self._axis)),
                                 float(np.dot(u, v))))
        return _axis_angle(self._axis, angle)

    def update(
        self,
        grabbed_index: int,
        target: np.ndarray,
        external_forces: dict[int, np.ndarray] | None = None,
    ) -> dict[int, np.ndarray]:
        """Rotate about the bond towards *target*, then relax elastically.

        :param grabbed_index: The atom index being dragged.
        :param target: The world-space point the mouse is at.
        :param external_forces: Optional extra per-atom displacements (the
            density-gradient nudges) applied alongside this solver's own
            flexibility pull on the grabbed atom, so density guidance works
            in torsion mode exactly as it does for a plain elastic drag.
        :returns: New positions for every atom that is not the anchor.
        """
        target = np.asarray(target, dtype=float)
        rotation = self._rotation_towards(grabbed_index, target)

        # Step 1: rigid rotation about the bond, always from the base
        # geometry so repeated calls cannot accumulate drift.
        for i in self._solver.positions:
            base = self._base_positions[i] - self._origin
            self._solver.positions[i] = self._origin + rotation @ base

        # Step 2: pull the grabbed atom part-way towards where the mouse
        # really is, then let the constraints redistribute that pull through
        # the fragment.  This is the flexibility on top of the rotation.
        forces = dict(external_forces) if external_forces else {}
        if grabbed_index in self._solver.positions and self.flexibility > 0.0:
            offset = target - self._solver.positions[grabbed_index]
            forces[grabbed_index] = forces.get(grabbed_index, 0.0) + offset * self.flexibility

        return self._solver.update(
            grabbed_index, target, external_forces=forces or None,
            pin_grabbed=False,
        )


# ---------------------------------------------------------------------------
# Density guidance
# ---------------------------------------------------------------------------

class DensityGuide:
    """Trilinear sampling and gradient of a periodic residual-density grid.

    Works with any object exposing ``array`` (an ``(nu, nv, nw)`` grid, as in
    :class:`fastmolwidget.density.ResidualDensityMap`) and ``orth_matrix`` (a
    ``3x3`` fractional-to-Cartesian matrix) - typically a
    :class:`~fastmolwidget.density.ResidualDensityMap` itself.
    """

    def __init__(self, grid: np.ndarray, orth_matrix: np.ndarray):
        self.grid = np.asarray(grid, dtype=float)
        self.orth_matrix = np.asarray(orth_matrix, dtype=float)
        self._frac_from_cart = np.linalg.inv(self.orth_matrix).T

    @classmethod
    def from_map(cls, density_map) -> DensityGuide:
        """Build a guide from a :class:`~fastmolwidget.density.ResidualDensityMap`."""
        return cls(density_map.array, density_map.orth_matrix)

    def _to_fractional(self, cart_point: np.ndarray) -> np.ndarray:
        return np.asarray(cart_point, dtype=float) @ self._frac_from_cart

    def sample(self, cart_point: np.ndarray) -> float:
        """Trilinearly interpolated density (e/Å³) at a Cartesian point.

        The grid is periodic, so the point is wrapped into ``[0, 1)``
        fractional coordinates first.
        """
        shape = np.array(self.grid.shape, dtype=float)
        frac = self._to_fractional(cart_point) % 1.0
        idx = frac * shape
        i0 = np.floor(idx).astype(int)
        t = idx - i0
        i1 = (i0 + 1) % self.grid.shape
        i0 = i0 % self.grid.shape

        c000 = self.grid[i0[0], i0[1], i0[2]]
        c100 = self.grid[i1[0], i0[1], i0[2]]
        c010 = self.grid[i0[0], i1[1], i0[2]]
        c001 = self.grid[i0[0], i0[1], i1[2]]
        c110 = self.grid[i1[0], i1[1], i0[2]]
        c101 = self.grid[i1[0], i0[1], i1[2]]
        c011 = self.grid[i0[0], i1[1], i1[2]]
        c111 = self.grid[i1[0], i1[1], i1[2]]

        c00 = c000 * (1 - t[0]) + c100 * t[0]
        c01 = c001 * (1 - t[0]) + c101 * t[0]
        c10 = c010 * (1 - t[0]) + c110 * t[0]
        c11 = c011 * (1 - t[0]) + c111 * t[0]
        c0 = c00 * (1 - t[1]) + c10 * t[1]
        c1 = c01 * (1 - t[1]) + c11 * t[1]
        return float(c0 * (1 - t[2]) + c1 * t[2])

    def gradient(self, cart_point: np.ndarray, step: float = DENSITY_GRADIENT_STEP) -> np.ndarray:
        """Numerical (central-difference) gradient of the density, in Cartesian Å."""
        p = np.asarray(cart_point, dtype=float)
        grad = np.zeros(3)
        for axis in range(3):
            dp = np.zeros(3)
            dp[axis] = step
            grad[axis] = (self.sample(p + dp) - self.sample(p - dp)) / (2.0 * step)
        return grad

    def score(self, points: dict[int, np.ndarray]) -> float:
        """Sum of the interpolated density at every given point."""
        return float(sum(self.sample(p) for p in points.values()))


# ---------------------------------------------------------------------------
# Session: ties rigid/elastic solving, density guidance and snapping together
# ---------------------------------------------------------------------------

@dataclass
class MoietyDragSession:
    """A single Ctrl+drag interaction moving one moiety.

    Construct with :func:`build_drag_session`; :mod:`molecule3D` calls
    :meth:`update` from ``mouseMoveEvent`` and discards the session on
    release.

    ``mode`` is ``'elastic'`` for the usual atom-anchored drag and
    ``'torsion'`` when the split point was a bond, in which case the solver
    rotates the fragment about that bond (see :class:`TorsionDrag`).  Both
    take the same arguments and are driven identically from here.
    """

    grabbed_index: int
    mode: str  # 'rigid', 'elastic' or 'torsion'
    _solver: RigidPivotDrag | ElasticDrag | TorsionDrag
    density: DensityGuide | None = None
    snapped: bool = False
    _snap_target: np.ndarray | None = field(default=None, repr=False)
    _last_positions: dict[int, np.ndarray] = field(default_factory=dict, repr=False)

    def update(self, target: np.ndarray) -> dict[int, np.ndarray]:
        """Advance the drag towards *target* (a world-space Cartesian point).

        :returns: The moiety's new atom positions, keyed by atom index.
        """
        target = np.asarray(target, dtype=float)

        if self.snapped and self._snap_target is not None:
            if float(np.linalg.norm(target - self._snap_target)) < BREAKAWAY_DISTANCE:
                # Inside the breakaway dead zone: hold the snapped pose - this
                # is the "extra force" needed to pull the moiety further.
                return self._last_positions
            self.snapped = False
            self._snap_target = None

        if self.mode == 'rigid':
            assert isinstance(self._solver, RigidPivotDrag)
            # The rigid body has one controlling point (the grabbed atom),
            # so guidance is expressed as a bias on its target instead of
            # per-atom nudges - rotating towards it carries the whole
            # fragment along.
            positions = self._solver.update(self.grabbed_index, self._bias_target(target))
        else:
            assert isinstance(self._solver, ElasticDrag | TorsionDrag)
            # The grabbed atom is normally pinned to the mouse exactly, so
            # without this it would never itself feel the density pull -
            # only the *other*, passive atoms of the moiety would (see
            # _density_nudges).  That leaves single-atom moieties (a very
            # common case - dragging one terminal atom) with no guidance at
            # all, since there is no other atom to nudge.  Biasing the
            # grabbed atom's own drive point fixes that for every moiety
            # size.  TorsionDrag takes the same two arguments and applies
            # them the same way, on top of its rotation about the bond.
            external_forces = self._density_nudges(self._solver.positions)
            positions = self._solver.update(
                self.grabbed_index, self._bias_target(target), external_forces=external_forces,
            )

        if self.density is not None:
            gradient_norm = sum(
                float(np.linalg.norm(self.density.gradient(p)))
                for p in positions.values()
            )
            if gradient_norm < SNAP_GRADIENT_TOL * max(len(positions), 1):
                self.snapped = True
                self._snap_target = target.copy()

        self._last_positions = positions
        return positions

    def _bias_target(self, target: np.ndarray) -> np.ndarray:
        """Nudge the grabbed atom's own drive point towards a nearby peak.

        Applied in both modes so the atom the user is actively controlling
        is itself pulled towards density, not just whatever the elastic
        solver's other, passive moiety atoms happen to be (see
        :meth:`_density_nudges`) - the only guidance a single-atom moiety
        ever gets, since it has no other atom to nudge.  Returns *target*
        unchanged when there is no density guide or the local gradient is
        (numerically) zero.
        """
        if self.density is None:
            return target
        grad = self.density.gradient(target)
        norm = float(np.linalg.norm(grad))
        if norm <= 1e-9:
            return target
        return target + grad / norm * min(norm, 1.0) * DENSITY_NUDGE_STRENGTH

    def _density_nudges(
        self, positions: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray] | None:
        """Small per-atom step along the local density gradient.

        Only used by the elastic solver's *other*, passive moiety atoms -
        the grabbed atom's own drive point is biased separately by
        :meth:`_bias_target`, and the rigid solver has no other atoms at all
        (dragging any of its atoms rotates the whole rigid body).
        """
        if self.density is None:
            return None
        nudges: dict[int, np.ndarray] = {}
        for i, p in positions.items():
            if i == self.grabbed_index or i in getattr(self._solver, 'anchors', ()):
                continue
            grad = self.density.gradient(p)
            norm = float(np.linalg.norm(grad))
            if norm > 1e-9:
                nudges[i] = grad / norm * min(norm, 1.0) * DENSITY_NUDGE_STRENGTH
        return nudges


def build_drag_session(
    connections: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    positions: dict[int, np.ndarray],
    anchors: set[int],
    grabbed_index: int,
    density: DensityGuide | None = None,
    masses: dict[int, float] | None = None,
    riding_atoms: dict[int, int] | None = None,
    bond: tuple[int, int] | None = None,
) -> MoietyDragSession | None:
    """Build a :class:`MoietyDragSession` for dragging from *grabbed_index*.

    Uses the elastic (mass-spring) solver, even for a single anchor, so the
    drag feel is consistent regardless of how many anchors are selected - see
    the module docstring.  When *bond* is given the split point is that bond
    instead, and the fragment rotates about it (:class:`TorsionDrag`).

    :param connections: Every bond in the molecule as ``(i, j)`` atom-index
        pairs.
    :param positions: Current Cartesian position of every atom, keyed by
        index (only the moiety's and the anchors' entries are used).
    :param anchors: The user-selected fixed border atoms.  An empty set
        means "no anchors at all" - the whole fragment connected to
        *grabbed_index* is then dragged as a free body (nothing holds it in
        place), which is how a whole-molecule disorder is modelled.
        Ignored when *bond* is given, since the bond determines the anchor.
    :param grabbed_index: The atom under the cursor when the drag started.
    :param density: Optional density guide for snapping; ``None`` disables
        density guidance (the moiety still drags, just without a target).
    :param masses: Optional per-atom mass, keyed by atom index (see
        :class:`ElasticDrag`).  ``None`` gives every solved atom equal
        weight.
    :param riding_atoms: Optional mapping of a riding atom's index to its
        parent's index (typically a terminal hydrogen and the heavy atom it
        is bonded to, see :class:`ElasticDrag`).  Riding atoms keep their
        exact original offset from the parent - same bond length, same
        direction - rather than taking part in the spring solve.
    :param bond: Optional ``(i, j)`` bond to split at instead of using
        *anchors*.  Its far end (the one away from *grabbed_index*, see
        :func:`bond_split_ends`) becomes the sole anchor and the bond becomes
        the axis the fragment rotates about; the near end travels with the
        fragment and may drift off that axis, so a tumble rather than only a
        clean torsion can be modelled.
    :returns: The session, or ``None`` when *grabbed_index* does not belong
        to a moiety reachable from *anchors*, or when *bond* is unrelated to
        the fragment being dragged (nothing to drag either way).
    """
    far: int | None = None
    near: int | None = None
    if bond is not None:
        ends = bond_split_ends(connections, bond, grabbed_index)
        if ends is None:
            return None
        far, near = ends
        anchors = {far}

    moiety = find_moiety(connections, anchors, grabbed_index)
    if grabbed_index not in moiety:
        return None

    bonds = moiety_edges(connections, moiety, anchors)
    angle_pairs = moiety_angle_pairs(connections, moiety, anchors)
    combined = {i: positions[i] for i in moiety}
    combined.update({i: positions[i] for i in anchors})
    riding = {
        i: p for i, p in (riding_atoms or {}).items()
        if i in combined and p in combined
    }

    if far is not None and near is not None:
        torsion = TorsionDrag(
            combined, far, near, bonds, masses=masses, riding=riding,
            soft_edges=angle_pairs,
        )
        return MoietyDragSession(
            grabbed_index=grabbed_index, mode='torsion', _solver=torsion,
            density=density,
        )

    solver = ElasticDrag(
        combined, set(anchors), bonds, masses=masses, riding=riding,
        soft_edges=angle_pairs,
    )
    return MoietyDragSession(
        grabbed_index=grabbed_index, mode='elastic', _solver=solver, density=density,
    )
