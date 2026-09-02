"""Tests for :mod:`fastmolwidget.disorder_drag` — interactive moiety dragging.

Pure-math module, no Qt/GL involved, so these are plain unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastmolwidget.disorder_drag import (
    BREAKAWAY_DISTANCE,
    DensityGuide,
    ElasticDrag,
    RigidPivotDrag,
    build_drag_session,
    find_moiety,
    moiety_edges,
)

# ---------------------------------------------------------------------------
# find_moiety / moiety_edges
# ---------------------------------------------------------------------------

def test_find_moiety_simple_chain():
    """A chain anchored at one end: the whole rest is the moiety."""
    connections = [(0, 1), (1, 2), (2, 3)]
    assert find_moiety(connections, {0}, start=1) == {1, 2, 3}


def test_find_moiety_stops_at_second_anchor():
    """Two anchors bracket the moiety; nothing beyond the far anchor is included."""
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert find_moiety(connections, {0, 4}, start=1) == {1, 2, 3}


def test_find_moiety_branching():
    """A branch point inside the moiety is fully included."""
    connections = [(0, 1), (1, 2), (1, 3)]
    assert find_moiety(connections, {0}, start=1) == {1, 2, 3}


def test_find_moiety_only_follows_the_branch_started_from():
    """An anchor with an unrelated second branch must not pull it in."""
    connections = [(0, 1), (1, 2), (0, 3), (3, 4)]
    assert find_moiety(connections, {0}, start=1) == {1, 2}
    assert find_moiety(connections, {0}, start=3) == {3, 4}


def test_find_moiety_start_is_anchor_is_empty():
    assert find_moiety([(0, 1)], {0}, start=0) == set()


def test_find_moiety_isolated_start_atom():
    assert find_moiety([(0, 1)], {5}, start=9) == {9}


def test_moiety_edges_includes_anchor_bonds():
    connections = [(0, 1), (1, 2), (2, 3)]
    moiety = {1, 2}
    edges = moiety_edges(connections, moiety, {0, 3})
    assert set(edges) == {(0, 1), (1, 2), (2, 3)}


# ---------------------------------------------------------------------------
# RigidPivotDrag
# ---------------------------------------------------------------------------

def test_rigid_pivot_preserves_distances():
    pivot = np.array([0.0, 0.0, 0.0])
    base = {
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
    }
    drag = RigidPivotDrag(pivot, base)
    new = drag.update(2, np.array([0.0, 5.0, 0.0]))
    for i, base_pos in base.items():
        assert np.linalg.norm(new[i]) == pytest.approx(np.linalg.norm(base_pos), abs=1e-9)
    # Grabbed atom points exactly towards the target direction.
    direction = new[2] / np.linalg.norm(new[2])
    assert direction == pytest.approx(np.array([0.0, 1.0, 0.0]), abs=1e-9)


def test_rigid_pivot_no_rotation_when_target_matches_current():
    pivot = np.array([0.0, 0.0, 0.0])
    base = {1: np.array([2.0, 0.0, 0.0])}
    drag = RigidPivotDrag(pivot, base)
    new = drag.update(1, np.array([2.0, 0.0, 0.0]))
    assert new[1] == pytest.approx(base[1])


def test_rigid_pivot_is_stateless_across_calls():
    """Repeated updates never drift — always solved from the original geometry."""
    pivot = np.array([0.0, 0.0, 0.0])
    base = {1: np.array([1.0, 0.0, 0.0])}
    drag = RigidPivotDrag(pivot, base)
    drag.update(1, np.array([0.0, 1.0, 0.0]))
    drag.update(1, np.array([0.0, 0.0, 1.0]))
    final = drag.update(1, np.array([1.0, 0.0, 0.0]))
    assert final[1] == pytest.approx(base[1], abs=1e-6)


def test_rigid_pivot_180_degree_flip_is_stable():
    """The degenerate antiparallel case must not raise or return NaN."""
    pivot = np.array([0.0, 0.0, 0.0])
    base = {1: np.array([1.0, 0.0, 0.0])}
    drag = RigidPivotDrag(pivot, base)
    new = drag.update(1, np.array([-3.0, 0.0, 0.0]))
    assert np.all(np.isfinite(new[1]))
    assert np.linalg.norm(new[1]) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# ElasticDrag
# ---------------------------------------------------------------------------

def test_elastic_drag_keeps_anchors_fixed():
    connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
    moiety = {1, 2, 3}
    anchors = {0, 4}
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([4.5, 0.0, 0.0]),
        4: np.array([6.0, 0.0, 0.0]),
    }
    edges = moiety_edges(connections, moiety, anchors)
    combined = {**{i: positions[i] for i in moiety}, **{i: positions[i] for i in anchors}}
    drag = ElasticDrag(combined, anchors, edges)
    new = drag.update(2, np.array([3.0, 1.5, 0.0]))
    assert 0 not in new and 4 not in new  # anchors are not returned as movable
    # The solver never touches anchors internally either.
    assert drag.positions[0] == pytest.approx(positions[0])
    assert drag.positions[4] == pytest.approx(positions[4])


def test_elastic_drag_approximately_preserves_bond_lengths():
    connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
    moiety = {1, 2, 3}
    anchors = {0, 4}
    # Anchors closer together than the fully-stretched chain length (6.0 Å),
    # so there is slack for the moiety to bend into when dragged.
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([4.5, 0.0, 0.0]),
        4: np.array([4.0, 3.9, 0.0]),
    }
    edges = moiety_edges(connections, moiety, anchors)
    combined = {**{i: positions[i] for i in moiety}, **{i: positions[i] for i in anchors}}
    drag = ElasticDrag(combined, anchors, edges, iterations=40)
    new = drag.update(2, np.array([2.0, 1.0, 0.0]))
    full = {0: positions[0], 4: positions[4], **new}
    for a, b in edges:
        rest = np.linalg.norm(positions[a] - positions[b])
        actual = np.linalg.norm(full[a] - full[b])
        assert actual == pytest.approx(rest, rel=0.05)


def test_elastic_drag_grabbed_atom_reaches_target():
    connections = [(0, 1), (1, 2), (2, 3)]
    moiety = {1, 2}
    anchors = {0, 3}
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([4.5, 0.0, 0.0]),
    }
    edges = moiety_edges(connections, moiety, anchors)
    combined = {**{i: positions[i] for i in moiety}, **{i: positions[i] for i in anchors}}
    drag = ElasticDrag(combined, anchors, edges)
    target = np.array([1.5, 1.0, 0.0])
    new = drag.update(1, target)
    assert new[1] == pytest.approx(target)


# ---------------------------------------------------------------------------
# DensityGuide
# ---------------------------------------------------------------------------

def _gaussian_grid(n: int, cell: float, center: tuple[float, float, float], sigma: float = 0.3):
    xs = np.linspace(0, cell, n, endpoint=False)
    X, Y, Z = np.meshgrid(xs, xs, xs, indexing='ij')
    cx, cy, cz = center
    return 10.0 * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2) / (2 * sigma ** 2))


def test_density_guide_samples_peak():
    grid = _gaussian_grid(40, 5.0, (2.5, 2.5, 2.5))
    guide = DensityGuide(grid, np.eye(3) * 5.0)
    assert guide.sample(np.array([2.5, 2.5, 2.5])) == pytest.approx(10.0, abs=0.05)


def test_density_guide_gradient_points_uphill():
    grid = _gaussian_grid(40, 5.0, (2.5, 2.5, 3.0))
    guide = DensityGuide(grid, np.eye(3) * 5.0)
    grad = guide.gradient(np.array([2.5, 2.5, 2.5]))
    # Gradient should point from the offset point towards the peak (+z).
    assert grad[2] > 0.0
    assert abs(grad[0]) < 1e-6
    assert abs(grad[1]) < 1e-6


def test_density_guide_is_periodic():
    grid = _gaussian_grid(40, 5.0, (0.0, 2.5, 2.5))
    guide = DensityGuide(grid, np.eye(3) * 5.0)
    # A point just outside the cell should wrap and see the peak at 0.
    near_peak = guide.sample(np.array([4.99, 2.5, 2.5]))
    far = guide.sample(np.array([2.5, 2.5, 2.5]))
    assert near_peak > far


# ---------------------------------------------------------------------------
# build_drag_session / MoietyDragSession
# ---------------------------------------------------------------------------

def test_build_drag_session_drags_whole_molecule_when_no_anchors():
    """No anchors: the whole connected fragment drags as a free body."""
    connections = [(0, 1)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    session = build_drag_session(connections, positions, set(), 1)
    assert session is not None
    assert session.mode == 'elastic'
    new = session.update(np.array([1.5, 2.0, 0.0]))
    # No anchors: even atom 0 (the other end of the bond) is free to move.
    assert 0 in new and 1 in new


def test_build_drag_session_none_when_grabbed_atom_not_in_moiety():
    connections = [(0, 1)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    # atom 0 is itself the anchor, not part of the moiety.
    assert build_drag_session(connections, positions, {0}, 0) is None


def test_build_drag_session_always_uses_elastic_even_for_one_anchor():
    connections = [(0, 1)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    session = build_drag_session(connections, positions, {0}, 1)
    assert session is not None
    assert session.mode == 'elastic'


def test_build_drag_session_picks_elastic_for_two_anchors():
    connections = [(0, 1), (1, 2), (2, 3)]
    positions = {
        0: np.zeros(3),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([4.5, 0.0, 0.0]),
    }
    session = build_drag_session(connections, positions, {0, 3}, 1)
    assert session is not None
    assert session.mode == 'elastic'


def test_session_snaps_near_density_peak_and_resists_breakaway():
    connections = [(0, 1)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    peak = np.array([0.0, 1.5, 0.0])
    grid = _gaussian_grid(50, 6.0, tuple(peak + 3.0), sigma=0.4)
    guide = DensityGuide(grid, np.eye(3) * 6.0)
    session = build_drag_session(connections, positions, {0}, 1, density=guide)
    assert session is not None

    # Drag right onto the peak repeatedly so the gradient settles near zero.
    for _ in range(5):
        session.update(peak)
    assert session.snapped is True

    # A small further move (inside the breakaway distance) should not move
    # the moiety away from the snapped pose.
    before = session._last_positions[1].copy()
    tiny_move = peak + np.array([0.0, 0.0, BREAKAWAY_DISTANCE * 0.3])
    after = session.update(tiny_move)
    assert after[1] == pytest.approx(before)

    # A move clearly beyond the breakaway distance should resume tracking.
    far_move = peak + np.array([0.0, 0.0, BREAKAWAY_DISTANCE * 3.0])
    resumed = session.update(far_move)
    assert not np.allclose(resumed[1], before)
