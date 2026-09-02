"""
Interactive disorder-moiety dragging: pure math, no Qt and no OpenGL.

Lets a user pick one or more fixed "anchor" atoms (the border between a
disordered fragment and the rest of the molecule) and then drag any atom of
the fragment ("moiety") with the mouse to reposition it, guided by a residual
(Fo-Fc) density map towards the alternate-site peak.  This module only
implements the geometry/optimisation; :mod:`fastmolwidget.molecule3D` wires it
to mouse events.

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
  Anchors have infinite mass (never move).  :func:`build_drag_session` always
  uses this mode, even for a single anchor, for a consistently "gummy" feel.

:class:`DensityGuide` samples a residual-density grid (trilinear
interpolation) and its numerical gradient, so the moiety can be nudged
towards a nearby density peak while it is dragged, and so the session can
detect when it has settled onto one ("snapped").

:class:`MoietyDragSession` ties everything together and is the only object
:mod:`molecule3D` needs to construct and drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    'BREAKAWAY_DISTANCE',
    'DEFAULT_ISO_U',
    'DENSITY_GRADIENT_STEP',
    'DENSITY_NUDGE_STRENGTH',
    'SNAP_GRADIENT_TOL',
    'DensityGuide',
    'ElasticDrag',
    'MoietyDragSession',
    'RigidPivotDrag',
    'build_drag_session',
    'find_moiety',
    'moiety_edges',
]

#: Isotropic U (Å²) used for the dedicated density map computed for fitting a
#: disorder moiety - see :func:`fastmolwidget.density.force_isotropic_adps`.
#: A refined ADP "suctions" its own disorder partner's density into itself;
#: flattening every atom to a small, plausible U removes that bias.
DEFAULT_ISO_U: float = 0.045

#: Finite-difference step (Å) used by :meth:`DensityGuide.gradient`.
DENSITY_GRADIENT_STEP: float = 0.05

#: How far (Å) a moiety atom is nudged, per drag update, along the normalised
#: local density gradient.  Small enough to feel like gentle guidance rather
#: than a hard snap; the elastic/rigid solver re-applies afterwards so the
#: fragment's own geometry is not destroyed by the nudge.
DENSITY_NUDGE_STRENGTH: float = 0.06

#: Gradient magnitude (e/Å⁴, roughly) below which the moiety is considered to
#: have settled into a local density maximum ("snapped").
SNAP_GRADIENT_TOL: float = 0.05

#: Extra mouse-target displacement (Å) required to leave a snapped pose - the
#: "more force" needed to drag the moiety further once it has snapped.
BREAKAWAY_DISTANCE: float = 0.6


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


def moiety_edges(
    connections: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    moiety: set[int],
    anchors: set[int],
) -> list[tuple[int, int]]:
    """Return the bonds relevant to dragging *moiety*.

    Includes every bond entirely inside the moiety, plus every bond between a
    moiety atom and one of *anchors* (these are the constraints that hold the
    fragment in place).  Used to build :class:`ElasticDrag`'s constraint set.
    """
    edges: list[tuple[int, int]] = []
    for a, b in connections:
        a_in, b_in = a in moiety, b in moiety
        if a_in and b_in or a_in and b in anchors or b_in and a in anchors:
            edges.append((a, b))
    return edges


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
    ):
        """
        :param positions: Starting positions of every atom involved (moiety
            *and* anchors), keyed by atom index.
        :param anchors: Indices that never move.
        :param edges: Bonded pairs to keep at their original length, from
            :func:`moiety_edges`.
        :param iterations: Constraint-relaxation passes per :meth:`update`.
        """
        self.positions = {i: np.array(p, dtype=float) for i, p in positions.items()}
        self.anchors = set(anchors)
        self.iterations = iterations
        self._rest_length = {
            (a, b): float(np.linalg.norm(self.positions[a] - self.positions[b]))
            for a, b in edges
        }

    def update(
        self,
        grabbed_index: int,
        target: np.ndarray,
        external_forces: dict[int, np.ndarray] | None = None,
    ) -> dict[int, np.ndarray]:
        """Relax the moiety towards *target* and return the new positions.

        :param grabbed_index: The atom index being dragged; pinned to
            *target* at the start of every relaxation pass.
        :param target: The world-space point the grabbed atom follows.
        :param external_forces: Optional per-atom displacement added once
            before the constraint passes (e.g. a density-gradient nudge);
            the constraint solve then partially, but not fully, absorbs it,
            which is what gives the density guidance a soft pull rather than
            an instant jump.
        :returns: Positions for every atom that is not an anchor.
        """
        pos = self.positions
        pos[grabbed_index] = np.asarray(target, dtype=float)

        if external_forces:
            for i, delta in external_forces.items():
                if i in pos and i != grabbed_index and i not in self.anchors:
                    pos[i] = pos[i] + delta

        movable = (set(pos) - self.anchors) - {grabbed_index}
        for _ in range(self.iterations):
            for (a, b), rest in self._rest_length.items():
                pa, pb = pos[a], pos[b]
                delta = pb - pa
                dist = float(np.linalg.norm(delta))
                if dist < 1e-9:
                    continue
                correction = (dist - rest) / dist
                move_a = a in movable
                move_b = b in movable
                if move_a and move_b:
                    pos[a] = pa + 0.5 * correction * delta
                    pos[b] = pb - 0.5 * correction * delta
                elif move_a:
                    pos[a] = pa + correction * delta
                elif move_b:
                    pos[b] = pb - correction * delta
            pos[grabbed_index] = np.asarray(target, dtype=float)

        return {i: p.copy() for i, p in pos.items() if i not in self.anchors}


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
    """

    grabbed_index: int
    mode: str  # 'rigid' or 'elastic'
    _solver: RigidPivotDrag | ElasticDrag
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
            biased_target = target
            if self.density is not None:
                # The rigid body has one controlling point (the grabbed
                # atom), so guidance is expressed as a bias on its target
                # instead of per-atom nudges - rotating towards it carries
                # the whole fragment along.
                grad = self.density.gradient(target)
                norm = float(np.linalg.norm(grad))
                if norm > 1e-9:
                    biased_target = target + grad / norm * min(norm, 1.0) * DENSITY_NUDGE_STRENGTH
            positions = self._solver.update(self.grabbed_index, biased_target)
        else:
            assert isinstance(self._solver, ElasticDrag)
            external_forces = self._density_nudges(self._solver.positions)
            positions = self._solver.update(
                self.grabbed_index, target, external_forces=external_forces,
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

    def _density_nudges(
        self, positions: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray] | None:
        """Small per-atom step along the local density gradient.

        Only used by the elastic (multi-anchor) solver: the rigid solver
        biases its *target* instead, see :func:`build_drag_session`'s caller.
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
) -> MoietyDragSession | None:
    """Build a :class:`MoietyDragSession` for dragging from *grabbed_index*.

    Always uses the elastic (mass-spring) solver, even for a single anchor,
    so the drag feel is consistent regardless of how many anchors are
    selected - see the module docstring.

    :param connections: Every bond in the molecule as ``(i, j)`` atom-index
        pairs.
    :param positions: Current Cartesian position of every atom, keyed by
        index (only the moiety's and the anchors' entries are used).
    :param anchors: The user-selected fixed border atoms.  An empty set
        means "no anchors at all" - the whole fragment connected to
        *grabbed_index* is then dragged as a free body (nothing holds it in
        place), which is how a whole-molecule disorder is modelled.
    :param grabbed_index: The atom under the cursor when the drag started.
    :param density: Optional density guide for snapping; ``None`` disables
        density guidance (the moiety still drags, just without a target).
    :returns: The session, or ``None`` when *grabbed_index* does not belong
        to a moiety reachable from *anchors* (nothing to drag).
    """
    moiety = find_moiety(connections, anchors, grabbed_index)
    if grabbed_index not in moiety:
        return None

    edges = moiety_edges(connections, moiety, anchors)
    combined = {i: positions[i] for i in moiety}
    combined.update({i: positions[i] for i in anchors})
    solver = ElasticDrag(combined, set(anchors), edges)
    return MoietyDragSession(
        grabbed_index=grabbed_index, mode='elastic', _solver=solver, density=density,
    )
