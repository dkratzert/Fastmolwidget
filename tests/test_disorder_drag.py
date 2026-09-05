"""Tests for :mod:`fastmolwidget.disorder_drag` — interactive moiety dragging.

Pure-math module, no Qt/GL involved, so these are plain unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from fastmolwidget.disorder_controller import DisorderDragMixin
from fastmolwidget.disorder_drag import (
    ANGLE_CONSTRAINT_STIFFNESS,
    BREAKAWAY_DISTANCE,
    DEFAULT_ISO_U,
    DensityGuide,
    HYDROGEN_ISO_U,
    DisorderSplit,
    ElasticDrag,
    PLANAR_CONSTRAINT_STIFFNESS,
    RigidPivotDrag,
    TorsionDrag,
    atomic_mass,
    bond_split_ends,
    build_drag_session,
    detect_planar_groups,
    find_moiety,
    moiety_angle_pairs,
    moiety_edges,
    next_disorder_label,
    plan_disorder_duplicate,
    riding_atoms,
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


def test_moiety_angle_pairs_linear_chain():
    """Every pair of atoms two bonds apart gets a 1,3 constraint."""
    connections = [(0, 1), (1, 2), (2, 3)]
    moiety = {1, 2, 3}
    pairs = moiety_angle_pairs(connections, moiety, {0})
    assert set(pairs) == {(0, 2), (1, 3)}


def test_moiety_angle_pairs_excludes_direct_bonds():
    """A 1,3 pair that happens to also be directly bonded (e.g. a 3-membered
    ring) must not be duplicated - moiety_edges already constrains it."""
    connections = [(0, 1), (1, 2), (0, 2)]  # triangle
    moiety = {1, 2}
    pairs = moiety_angle_pairs(connections, moiety, {0})
    assert (0, 2) not in pairs and (2, 0) not in pairs


def test_moiety_angle_pairs_excludes_anchor_only_pairs():
    """A 1,3 pair entirely among anchors needs no constraint (never moves)."""
    connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
    moiety = {2}
    anchors = {0, 1, 3, 4}
    # 1 and 3 share neighbour 2, but both are anchors - no constraint needed.
    pairs = moiety_angle_pairs(connections, moiety, anchors)
    assert (1, 3) not in pairs and (3, 1) not in pairs


def test_moiety_angle_pairs_branching():
    """A branch point produces a 1,3 pair between its two branches."""
    connections = [(0, 1), (1, 2), (1, 3)]
    moiety = {1, 2, 3}
    pairs = moiety_angle_pairs(connections, moiety, {0})
    assert set(pairs) == {(0, 2), (0, 3), (2, 3)}


def test_detect_planar_groups_finds_ring():
    """A roughly planar five-membered ring is detected as one group."""
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([1.7, 1.0, 0.0]),
        3: np.array([0.7, 1.9, 0.0]),
        4: np.array([-0.3, 1.0, 0.0]),
    }
    groups = detect_planar_groups(connections, positions, set(positions), set())
    assert groups and set(frozenset(group) for group in groups) == {frozenset(positions)}


def test_elastic_drag_keeps_planar_group_near_original_plane():
    """Planar constraints are weak but still restore a ring toward its plane."""
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([1.7, 1.0, 0.0]),
        3: np.array([0.7, 1.9, 0.0]),
        4: np.array([-0.3, 1.0, 0.0]),
    }
    anchors = {0}
    edges = moiety_edges(connections, set(positions), anchors)
    drag = ElasticDrag(
        dict(positions), anchors, edges, iterations=30,
        planar_groups=[[0, 1, 2, 3, 4]],
    )
    moved = drag.update(2, np.array([1.7, 1.0, 1.5]))
    max_z = max(abs(moved[i][2]) for i in positions if i not in {0, 2})
    assert max_z < 1.0
    assert PLANAR_CONSTRAINT_STIFFNESS > 0.0


def test_detect_planar_groups_include_nearby_in_plane_substituent():
    """Only atoms truly near the ring plane are added to the planar group."""
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([1.7, 1.0, 0.0]),
        3: np.array([0.7, 1.9, 0.0]),
        4: np.array([-0.3, 1.0, 0.0]),
        5: np.array([1.2, -0.8, 0.05]),
    }
    groups = detect_planar_groups(connections, positions, set(positions), set())
    assert any({0, 1, 2, 3, 4, 5} <= set(group) for group in groups)

    tetrahedral = {**positions, 6: np.array([1.2, -0.8, 0.8])}
    tetrahedral_connections = [*connections, (1, 6)]
    tetra_groups = detect_planar_groups(tetrahedral_connections, tetrahedral, set(tetrahedral), set())
    assert not any({0, 1, 2, 3, 4, 5, 6} <= set(group) for group in tetra_groups)


def test_detect_planar_groups_excludes_hydrogens_from_search_and_group():
    """Hydrogen atoms are left out of the ring-search and ring-plane correction."""
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.0, 0.0, 0.0]),
        2: np.array([1.7, 1.0, 0.0]),
        3: np.array([0.7, 1.9, 0.0]),
        4: np.array([-0.3, 1.0, 0.0]),
        5: np.array([1.2, -0.8, 0.02]),
    }
    groups = detect_planar_groups(connections, positions, set(positions), set(), exclude={5})
    assert groups
    assert all(5 not in group for group in groups)


def test_dragged_atoms_are_flattened_to_isotropic_adps():
    """Making a moiety split permanent also flattens its ADPs to isotropic."""

    class DummyAtom:
        def __init__(self, atom_type, u_cart, u_iso):
            self.type_ = atom_type
            self.u_cart = u_cart
            self.u_iso = u_iso
            self.adp_valid = True
            self.u_eigvals = np.array([0.2, 0.3, 0.4])
            self.u_eigvecs = np.eye(3)
            self.adp_A_matrix = np.eye(3)
            self.adp_billboard_r = 1.2
            self.npd_half_edge = 0.3

    class DummyRenderer(DisorderDragMixin):
        def __init__(self):
            self.atoms = [
                DummyAtom('C', np.eye(3), 0.04),
                DummyAtom('H', np.eye(3), 0.05),
            ]

    renderer = DummyRenderer()
    for idx in (0, 1):
        renderer._make_atom_isotropic(idx)

    assert renderer.atoms[0].u_iso == pytest.approx(DEFAULT_ISO_U)
    assert renderer.atoms[1].u_iso == pytest.approx(HYDROGEN_ISO_U)
    for atom in renderer.atoms:
        assert atom.u_cart is None
        assert atom.adp_valid is True
        assert atom.u_eigvals is None
        assert atom.u_eigvecs is None
        assert atom.adp_A_matrix is None
        assert atom.adp_billboard_r == pytest.approx(0.0)
        assert atom.npd_half_edge == pytest.approx(0.0)


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


def test_atomic_mass_hydrogen_lighter_than_carbon():
    assert atomic_mass('H') < atomic_mass('C')
    assert atomic_mass('D') > atomic_mass('H')  # deuterium is heavier than protium
    assert atomic_mass('D') < atomic_mass('C')


def test_atomic_mass_unknown_symbol_falls_back_to_light_dummy():
    """An unrecognised label must not act like an infinitely heavy anchor."""
    assert atomic_mass('Zz') == pytest.approx(1.0)


def test_elastic_drag_light_atom_trails_heavy_neighbour_more():
    """Between two movable atoms, the lighter one absorbs most of the
    correction (riding-model-like), the heavier one barely moves."""
    # anchor(0, heavy) -- grabbed(1, heavy) -- movable_heavy(2) -- movable_light(3)
    connections = [(0, 1), (1, 2), (2, 3)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([4.0, 0.0, 0.0]),
    }
    edges = moiety_edges(connections, {1, 2, 3}, {0})
    combined = {0: positions[0], 1: positions[1], 2: positions[2], 3: positions[3]}
    masses = {0: 12.0, 1: 12.0, 2: 12.0, 3: 1.0}

    drag = ElasticDrag(combined, {0}, edges, masses=masses)
    target = np.array([1.5, 3.0, 0.0])
    new = drag.update(1, target)

    unweighted = ElasticDrag(dict(combined), {0}, edges)
    new_unweighted = unweighted.update(1, target)

    # The light atom (3) ends up farther from its pre-drag position than the
    # heavy one (2) does, relative to the unweighted (equal-mass) baseline.
    heavy_displacement = float(np.linalg.norm(new[2] - positions[2]))
    heavy_displacement_unweighted = float(np.linalg.norm(new_unweighted[2] - positions[2]))
    assert heavy_displacement < heavy_displacement_unweighted


def test_elastic_drag_equal_masses_matches_unweighted_default():
    """Explicit equal masses must reproduce the default (no masses) result."""
    connections = [(0, 1), (1, 2), (2, 3)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([4.5, 0.0, 0.0]),
    }
    edges = moiety_edges(connections, {1, 2}, {0, 3})
    combined_a = {i: positions[i] for i in (0, 1, 2, 3)}
    combined_b = {i: positions[i] for i in (0, 1, 2, 3)}
    target = np.array([1.5, 2.0, 0.0])

    default_drag = ElasticDrag(combined_a, {0, 3}, edges)
    equal_mass_drag = ElasticDrag(combined_b, {0, 3}, edges, masses={0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0})

    new_default = default_drag.update(1, target)
    new_equal = equal_mass_drag.update(1, target)
    for i in new_default:
        assert new_default[i] == pytest.approx(new_equal[i])


def test_build_drag_session_accepts_masses():
    connections = [(0, 1), (1, 2)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0]), 2: np.array([1.5, 1.0, 0.0])}
    masses = {0: 12.0, 1: 12.0, 2: 1.0}
    session = build_drag_session(connections, positions, {0}, 1, masses=masses)
    assert session is not None
    new = session.update(np.array([1.5, 3.0, 0.0]))
    assert 2 in new  # the light atom is still dragged along


# ---------------------------------------------------------------------------
# Loose 1,3 (soft) restraints
# ---------------------------------------------------------------------------

def test_soft_edge_applies_partial_correction_per_iteration():
    """A single relaxation pass on a soft-only edge must move the free atom
    by exactly ``stiffness`` of the full correction, not the whole way."""
    positions = {1: np.zeros(3), 2: np.array([5.0, 0.0, 0.0])}  # rest length 5
    drag = ElasticDrag(positions, set(), edges=[], iterations=1, soft_edges=[(1, 2)])
    new = drag.update(1, np.array([15.0, 0.0, 0.0]))  # stretches dist(1,2) to 10
    expected_dist = 10.0 - ANGLE_CONSTRAINT_STIFFNESS * (10.0 - 5.0)
    assert float(np.linalg.norm(new[2] - new[1])) == pytest.approx(expected_dist)


def test_soft_edges_converge_more_loosely_than_hard_edges():
    """With equal iteration counts, a hard (1,2) constraint must end up much
    closer to its rest length than a soft (1,3) one under the same stretch."""
    hard = ElasticDrag(
        {1: np.zeros(3), 2: np.array([5.0, 0.0, 0.0])}, set(), edges=[(1, 2)], iterations=8,
    )
    soft = ElasticDrag(
        {1: np.zeros(3), 2: np.array([5.0, 0.0, 0.0])}, set(), edges=[],
        iterations=8, soft_edges=[(1, 2)],
    )
    target = np.array([15.0, 0.0, 0.0])
    hard_new = hard.update(1, target)
    soft_new = soft.update(1, target)

    hard_error = abs(float(np.linalg.norm(hard_new[2] - hard_new[1])) - 5.0)
    soft_error = abs(float(np.linalg.norm(soft_new[2] - soft_new[1])) - 5.0)
    assert hard_error < soft_error


def test_moiety_bond_lengths_converge_better_with_loose_angle_constraints():
    """The 1,3 constraints must not noticeably degrade how well the real
    1,2 bonds converge, compared with treating everything at full stiffness."""
    connections = [(0, 1), (1, 2), (2, 3)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([4.5, 0.0, 0.0]),
    }
    moiety = {1, 2, 3}
    bonds = moiety_edges(connections, moiety, {0})
    angle_pairs = moiety_angle_pairs(connections, moiety, {0})
    target = np.array([1.5, 3.0, 0.0])

    loose = ElasticDrag(
        dict(positions), {0}, bonds, iterations=12, soft_edges=angle_pairs,
    )
    full_strength = ElasticDrag(
        dict(positions), {0}, bonds + angle_pairs, iterations=12,
    )
    new_loose = loose.update(1, target)
    new_full = full_strength.update(1, target)

    def bond_error(new):
        d12 = float(np.linalg.norm(new[2] - new[1]))
        d23 = float(np.linalg.norm(new[3] - new[2]))
        return abs(d12 - 1.5) + abs(d23 - 1.5)

    assert bond_error(new_loose) < bond_error(new_full)


# ---------------------------------------------------------------------------
# Riding atoms (exact rigid attachment, e.g. terminal hydrogens)
# ---------------------------------------------------------------------------

def _angle(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Angle a-vertex-b in radians."""
    u = a - vertex
    v = b - vertex
    cos = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def test_elastic_drag_riding_atom_keeps_its_geometry_at_the_parent():
    """A riding atom must keep its *internal* geometry exactly: the bond
    length to its parent and the angle to the parent's other bond, both of
    which requires the offset to rotate with the parent's frame."""
    # anchor(0) - C(1, grabbed) - H(2, riding)
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([1.5, 1.0, 0.0]),
    }
    edges = [(0, 1), (1, 2)]
    combined = {i: positions[i] for i in (0, 1, 2)}
    drag = ElasticDrag(combined, {0}, edges, riding={2: 1})

    target = np.array([1.5, 3.0, 0.0])
    new = drag.update(1, target)

    assert new[1] == pytest.approx(target)
    assert np.linalg.norm(new[2] - new[1]) == pytest.approx(
        np.linalg.norm(positions[2] - positions[1]), abs=1e-12,
    )
    assert _angle(positions[0], positions[1], positions[2]) == pytest.approx(
        _angle(positions[0], new[1], new[2]), abs=1e-9,
    )


def test_elastic_drag_riding_offset_rotates_with_the_parent_frame():
    """Guards against a regression to translation-only riding: once the
    parent's frame has turned, the lab-frame offset *must* have turned too."""
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([1.5, 1.0, 0.0]),
    }
    edges = [(0, 1), (1, 2)]
    combined = {i: positions[i] for i in (0, 1, 2)}
    drag = ElasticDrag(combined, {0}, edges, riding={2: 1})

    new = drag.update(1, np.array([1.5, 3.0, 0.0]))

    base_offset = positions[2] - positions[1]
    new_offset = new[2] - new[1]
    assert np.linalg.norm(new_offset - base_offset) > 0.1


def test_elastic_drag_riding_atom_follows_parent_through_relaxation():
    """The geometry stays exact even when the parent itself is only indirectly
    moved by the spring solve (not the grabbed atom)."""
    # anchor(0) - C1(1) - C2(2, grabbed) with a riding H(3) on C1.
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([1.5, 1.0, 0.0]),
    }
    edges = [(0, 1), (1, 2), (1, 3)]
    combined = {i: positions[i] for i in (0, 1, 2, 3)}
    drag = ElasticDrag(combined, {0}, edges, riding={3: 1}, iterations=40)

    target = np.array([3.0, 2.0, 0.0])
    new = drag.update(2, target)

    assert np.linalg.norm(new[3] - new[1]) == pytest.approx(
        np.linalg.norm(positions[3] - positions[1]), abs=1e-9,
    )
    # C1's frame is spanned by the anchor and C2, so both angles at C1 hold.
    for neighbour, moved in ((0, new.get(0, positions[0])), (2, new[2])):
        assert _angle(positions[neighbour], positions[1], positions[3]) == pytest.approx(
            _angle(moved, new[1], new[3]), abs=1e-9,
        )


def test_elastic_drag_riding_group_keeps_internal_geometry():
    """Every riding atom on one parent shares a single rotation, so a methyl
    group's H...H distances survive the drag untouched."""
    #   anchor(0) - C(1, grabbed) with three riding H(2, 3, 4)
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([1.9, 1.0, 0.0]),
        3: np.array([1.9, -0.5, 0.87]),
        4: np.array([1.9, -0.5, -0.87]),
    }
    edges = [(0, 1), (1, 2), (1, 3), (1, 4)]
    combined = {i: positions[i] for i in range(5)}
    drag = ElasticDrag(combined, {0}, edges, riding={2: 1, 3: 1, 4: 1})

    new = drag.update(1, np.array([0.5, 1.4, 0.3]))

    for a, b in ((2, 3), (2, 4), (3, 4)):
        assert np.linalg.norm(new[a] - new[b]) == pytest.approx(
            np.linalg.norm(positions[a] - positions[b]), abs=1e-9,
        )
    for h in (2, 3, 4):
        assert np.linalg.norm(new[h] - new[1]) == pytest.approx(
            np.linalg.norm(positions[h] - positions[1]), abs=1e-9,
        )
        assert _angle(positions[0], positions[1], positions[h]) == pytest.approx(
            _angle(positions[0], new[1], new[h]), abs=1e-9,
        )


def test_torsion_drag_riding_hydrogen_keeps_its_geometry():
    """The case that actually broke: a torsion rotates the fragment through a
    large angle, and the hydrogen has to turn with it."""
    #   far(0) - near(1) - C(2, grabbed) - H(3, riding on C)
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([2.2, 1.2, 0.0]),
        3: np.array([2.2, 1.9, 0.9]),
    }
    edges = [(0, 1), (1, 2), (2, 3)]
    combined = {i: positions[i] for i in range(4)}
    drag = TorsionDrag(
        combined, far=0, near=1, edges=edges, riding={3: 2}, flexibility=0.0,
    )

    # Pull the grabbed atom right round to the other side of the axis.
    new = drag.update(2, np.array([2.2, -1.2, 0.0]))

    assert np.linalg.norm(new[3] - new[2]) == pytest.approx(
        np.linalg.norm(positions[3] - positions[2]), abs=1e-9,
    )
    assert _angle(positions[1], positions[2], positions[3]) == pytest.approx(
        _angle(new[1], new[2], new[3]), abs=1e-9,
    )


def test_elastic_drag_riding_atom_grabbed_directly_is_solved_normally():
    """Grabbing the riding atom itself must let it follow the mouse exactly,
    not force it to stay glued to its parent's rigid offset."""
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([1.5, 1.0, 0.0]),
    }
    edges = [(0, 1), (1, 2)]
    combined = {i: positions[i] for i in (0, 1, 2)}
    drag = ElasticDrag(combined, {0}, edges, riding={2: 1})

    target = np.array([1.5, 5.0, 0.0])
    new = drag.update(2, target)
    assert new[2] == pytest.approx(target)


def test_elastic_drag_riding_atom_without_frame_translates():
    """A parent with no bonded, non-riding neighbour has no frame to rotate
    with, so its riding atoms are simply carried along."""
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([0.0, 0.96, 0.0]),
    }
    edges = [(0, 1)]
    drag = ElasticDrag(dict(positions), set(), edges, riding={1: 0})

    target = np.array([2.0, 0.5, -1.0])
    new = drag.update(0, target)
    assert new[1] - new[0] == pytest.approx(positions[1] - positions[0], abs=1e-12)


def test_elastic_drag_riding_atom_on_an_anchor_does_not_move():
    """A hydrogen riding on an *anchor* must stay put: the anchor never
    moves, so its moving neighbours must not swing the hydrogen either."""
    #   H(3, riding on the anchor) - anchor(0) - C(1) - C(2, grabbed)
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([-0.4, 0.9, 0.0]),
    }
    edges = [(0, 1), (1, 2), (0, 3)]
    drag = ElasticDrag(dict(positions), {0}, edges, riding={3: 0})

    new = drag.update(2, np.array([3.0, 2.0, 0.0]))
    assert new[3] == pytest.approx(positions[3], abs=1e-12)


def test_build_drag_session_riding_atoms_ride_exactly():
    """End-to-end through build_drag_session with a riding_atoms mapping."""
    connections = [(0, 1), (1, 2)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([1.5, 1.0, 0.0]),
    }
    session = build_drag_session(
        connections, positions, {0}, 1, riding_atoms={2: 1},
    )
    assert session is not None
    target = np.array([1.5, 4.0, 0.0])
    new = session.update(target)
    assert np.linalg.norm(new[2] - new[1]) == pytest.approx(
        np.linalg.norm(positions[2] - positions[1]), abs=1e-9,
    )
    assert _angle(positions[0], positions[1], positions[2]) == pytest.approx(
        _angle(positions[0], new[1], new[2]), abs=1e-9,
    )


def test_build_drag_session_riding_atoms_outside_moiety_are_ignored():
    """A riding_atoms entry pointing outside the actual moiety/anchors must
    be silently dropped rather than raising a KeyError."""
    connections = [(0, 1)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    # Index 99 does not exist in this tiny graph at all.
    session = build_drag_session(
        connections, positions, {0}, 1, riding_atoms={99: 1},
    )
    assert session is not None
    new = session.update(np.array([1.5, 2.0, 0.0]))
    assert 99 not in new


def test_build_drag_session_keeps_1_3_distances_loosely():
    """End-to-end: dragging a chain must keep 1,3 distances close to (but
    not necessarily exactly) their original value, unlike a chain with no
    angle constraints at all which is free to fold arbitrarily."""
    connections = [(0, 1), (1, 2), (2, 3)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([3.0, 0.0, 0.0]),
        3: np.array([4.5, 0.0, 0.0]),
    }
    original_13 = float(np.linalg.norm(positions[2] - positions[0]))

    session = build_drag_session(connections, positions, {0}, 1)
    new = session.update(np.array([1.5, 3.0, 0.0]))
    new_13 = float(np.linalg.norm(new[2] - positions[0]))
    # Loose, not exact: within 10% of the original 1,3 distance.
    assert new_13 == pytest.approx(original_13, rel=0.1)


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


def test_single_atom_moiety_is_pulled_towards_a_nearby_peak():
    """The grabbed atom itself must be nudged towards density - not only the
    (non-existent, for a single-atom moiety) other passive atoms - otherwise
    a lone dragged atom would never feel any snapping at all."""
    connections = [(0, 1)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    peak = np.array([3.0, 3.0, 3.0])
    grid = _gaussian_grid(60, 8.0, tuple(peak), sigma=0.4)
    guide = DensityGuide(grid, np.eye(3) * 8.0)
    session = build_drag_session(connections, positions, {0}, 1, density=guide)
    assert session is not None

    # Simulate the mouse approaching the peak in a few discrete steps, as a
    # real drag gesture would (never landing exactly on the peak itself).
    mouse_path = [peak + np.array([d, 0.0, 0.0]) for d in np.linspace(0.6, 0.0, 8)]
    pos = None
    for mouse_target in mouse_path:
        pos = session.update(mouse_target)[1]

    assert session.snapped is True
    assert float(np.linalg.norm(pos - peak)) < 0.1


def test_bias_target_moves_towards_the_gradient():
    """MoietyDragSession._bias_target must nudge, not just pass through,
    when a density guide with a non-zero local gradient is present."""
    connections = [(0, 1)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    peak = np.array([3.0, 3.0, 3.0])
    grid = _gaussian_grid(60, 8.0, tuple(peak), sigma=0.4)
    guide = DensityGuide(grid, np.eye(3) * 8.0)
    session = build_drag_session(connections, positions, {0}, 1, density=guide)
    assert session is not None

    off_peak = peak + np.array([0.3, 0.0, 0.0])
    biased = session._bias_target(off_peak)
    assert not np.allclose(biased, off_peak)
    # Biasing must move it closer to the peak, not farther away or sideways.
    assert float(np.linalg.norm(biased - peak)) < float(np.linalg.norm(off_peak - peak))


def test_bias_target_is_identity_without_density():
    connections = [(0, 1)]
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    session = build_drag_session(connections, positions, {0}, 1, density=None)
    assert session is not None
    target = np.array([2.0, 3.0, 4.0])
    assert session._bias_target(target) is target



# ---------------------------------------------------------------------------
# bond_split_ends
# ---------------------------------------------------------------------------

def test_bond_split_ends_picks_the_near_end_from_either_side():
    """The bond end on the grabbed atom's side is the near (moving) one."""
    connections = [(0, 1), (1, 2), (2, 3), (3, 4)]
    # Grabbing beyond atom 2 -> 2 is near, 1 is the far anchor.
    assert bond_split_ends(connections, (1, 2), grabbed_index=4) == (1, 2)
    assert bond_split_ends(connections, (1, 2), grabbed_index=3) == (1, 2)
    # Grabbing on the other side flips which end anchors.
    assert bond_split_ends(connections, (1, 2), grabbed_index=0) == (2, 1)


def test_bond_split_ends_grabbing_a_bond_end_itself():
    connections = [(0, 1), (1, 2), (2, 3)]
    assert bond_split_ends(connections, (1, 2), grabbed_index=2) == (1, 2)
    assert bond_split_ends(connections, (1, 2), grabbed_index=1) == (2, 1)


def test_bond_split_ends_ring_falls_back_to_graph_distance():
    """Cutting one ring bond separates nothing, so the closer end wins."""
    # Ring 0-1-2-3-0 with a tail 3-4.
    connections = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4)]
    # From atom 4: 0 is two bonds away, 1 is three, so 0 is near, 1 anchors.
    assert bond_split_ends(connections, (0, 1), grabbed_index=4) == (1, 0)


def test_bond_split_ends_none_for_unrelated_bond():
    connections = [(0, 1), (5, 6)]
    assert bond_split_ends(connections, (5, 6), grabbed_index=0) is None


def test_bond_split_ends_none_for_degenerate_bond():
    connections = [(0, 1), (1, 2)]
    assert bond_split_ends(connections, (1, 1), grabbed_index=2) is None


# ---------------------------------------------------------------------------
# ElasticDrag pin_grabbed
# ---------------------------------------------------------------------------

def test_elastic_drag_pin_grabbed_false_does_not_force_the_target():
    """Without pinning, the grabbed atom relaxes instead of jumping onto the
    mouse target."""
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    drag = ElasticDrag(positions, {0}, [(0, 1)])
    target = np.array([5.0, 0.0, 0.0])
    new = drag.update(1, target, pin_grabbed=False)
    assert not np.allclose(new[1], target)
    # The bond length constraint is still honoured.
    assert float(np.linalg.norm(new[1] - positions[0])) == pytest.approx(1.5, abs=1e-6)


def test_elastic_drag_pin_grabbed_true_is_the_default():
    positions = {0: np.zeros(3), 1: np.array([1.5, 0.0, 0.0])}
    drag = ElasticDrag(positions, {0}, [(0, 1)])
    target = np.array([1.5, 2.0, 0.0])
    assert drag.update(1, target)[1] == pytest.approx(target)


# ---------------------------------------------------------------------------
# TorsionDrag
# ---------------------------------------------------------------------------

def _torsion_setup():
    """far(0) - near(1) along +X, with a bent tail 2, 3 off the axis."""
    connections = [(0, 1), (1, 2), (2, 3)]
    positions = {
        0: np.array([0.0, 0.0, 0.0]),
        1: np.array([1.5, 0.0, 0.0]),
        2: np.array([2.2, 1.2, 0.0]),
        3: np.array([3.4, 1.6, 0.0]),
    }
    moiety = find_moiety(connections, {0}, 2)
    edges = moiety_edges(connections, moiety, {0})
    soft = moiety_angle_pairs(connections, moiety, {0})
    return connections, positions, edges, soft


def _distance_to_x_axis(point):
    return float(np.linalg.norm(np.asarray(point)[1:]))


def test_torsion_drag_zero_flexibility_is_a_pure_rotation():
    """With no flexibility every atom keeps its exact distance to the axis."""
    _, positions, edges, soft = _torsion_setup()
    drag = TorsionDrag(positions, 0, 1, edges, soft_edges=soft, flexibility=0.0)
    new = drag.update(2, np.array([2.2, 0.0, 1.2]))

    assert 0 not in new  # the far end is the anchor and never moves
    for i in (1, 2, 3):
        assert _distance_to_x_axis(new[i]) == pytest.approx(
            _distance_to_x_axis(positions[i]), abs=1e-9)
    # The grabbed atom reaches a target that lies on its rotation circle.
    assert new[2] == pytest.approx(np.array([2.2, 0.0, 1.2]), abs=1e-9)


def test_torsion_drag_zero_flexibility_keeps_the_near_end_on_the_axis():
    _, positions, edges, soft = _torsion_setup()
    drag = TorsionDrag(positions, 0, 1, edges, soft_edges=soft, flexibility=0.0)
    new = drag.update(2, np.array([2.2, 0.0, 1.2]))
    assert new[1] == pytest.approx(positions[1], abs=1e-9)


def test_torsion_drag_flexibility_lets_the_near_end_tumble():
    """With flexibility the near bond end drifts off the axis - that is what
    makes a tumble, rather than only a clean torsion, possible."""
    _, positions, edges, soft = _torsion_setup()
    off_circle = np.array([3.5, 0.0, 2.5])  # not reachable by rotation alone

    rigid = TorsionDrag(positions, 0, 1, edges, soft_edges=soft, flexibility=0.0)
    flexible = TorsionDrag(positions, 0, 1, edges, soft_edges=soft, flexibility=0.5)
    rigid_new = rigid.update(2, off_circle)
    flexible_new = flexible.update(2, off_circle)

    assert _distance_to_x_axis(rigid_new[1]) == pytest.approx(0.0, abs=1e-9)
    assert _distance_to_x_axis(flexible_new[1]) > 0.01
    # Flexibility gets the grabbed atom closer to where the mouse actually is.
    assert float(np.linalg.norm(flexible_new[2] - off_circle)) < float(
        np.linalg.norm(rigid_new[2] - off_circle))


def test_torsion_drag_flexibility_still_respects_the_anchor_bond():
    _, positions, edges, soft = _torsion_setup()
    drag = TorsionDrag(positions, 0, 1, edges, soft_edges=soft, flexibility=0.5)
    new = drag.update(2, np.array([3.5, 0.0, 2.5]))
    assert float(np.linalg.norm(new[1] - positions[0])) == pytest.approx(1.5, rel=0.05)


def test_torsion_drag_is_stateless_across_calls():
    """Repeated updates must not accumulate drift - each is solved from the
    original geometry, like RigidPivotDrag."""
    _, positions, edges, soft = _torsion_setup()
    drag = TorsionDrag(positions, 0, 1, edges, soft_edges=soft, flexibility=0.0)
    drag.update(2, np.array([2.2, 0.0, 1.2]))
    drag.update(2, np.array([2.2, -1.2, 0.0]))
    final = drag.update(2, np.array([2.2, 1.2, 0.0]))  # back to the start
    for i in (1, 2, 3):
        assert final[i] == pytest.approx(positions[i], abs=1e-6)


def test_torsion_drag_anchor_never_moves():
    _, positions, edges, soft = _torsion_setup()
    drag = TorsionDrag(positions, 0, 1, edges, soft_edges=soft, flexibility=0.8)
    drag.update(2, np.array([4.0, 3.0, 2.0]))
    assert drag.positions[0] == pytest.approx(positions[0], abs=1e-12)


def test_torsion_drag_degenerate_axis_does_not_raise():
    """A zero-length bond has no axis; the solver must still run."""
    positions = {0: np.zeros(3), 1: np.zeros(3), 2: np.array([1.5, 0.0, 0.0])}
    drag = TorsionDrag(positions, 0, 1, [(0, 1), (1, 2)])
    new = drag.update(2, np.array([0.0, 1.5, 0.0]))
    assert np.all(np.isfinite(new[2]))


# ---------------------------------------------------------------------------
# build_drag_session with a bond
# ---------------------------------------------------------------------------

def test_build_drag_session_bond_builds_a_torsion_session():
    connections, positions, _, _ = _torsion_setup()
    session = build_drag_session(connections, positions, set(), 2, bond=(0, 1))
    assert session is not None
    assert session.mode == 'torsion'
    assert session._solver.far == 0
    assert session._solver.near == 1


def test_build_drag_session_bond_anchors_the_far_end_only():
    connections, positions, _, _ = _torsion_setup()
    session = build_drag_session(connections, positions, set(), 2, bond=(0, 1))
    assert session is not None
    new = session.update(np.array([2.2, 0.0, 1.2]))
    assert 0 not in new       # far end is the anchor
    assert 1 in new           # near end travels with the fragment
    assert 2 in new and 3 in new


def test_build_drag_session_bond_ignores_supplied_anchors():
    """The bond determines the anchor, so a stale atom selection cannot
    silently pin the wrong atom."""
    connections, positions, _, _ = _torsion_setup()
    session = build_drag_session(connections, positions, {3}, 2, bond=(0, 1))
    assert session is not None
    assert session.mode == 'torsion'
    new = session.update(np.array([2.2, 0.0, 1.2]))
    assert 3 in new  # would have been an anchor (absent) without the override


def test_build_drag_session_bond_none_when_unrelated():
    connections, positions, _, _ = _torsion_setup()
    assert build_drag_session(connections, positions, set(), 2, bond=(7, 8)) is None


def test_build_drag_session_without_bond_is_still_elastic():
    connections, positions, _, _ = _torsion_setup()
    session = build_drag_session(connections, positions, {0}, 2)
    assert session is not None
    assert session.mode == 'elastic'


# ---------------------------------------------------------------------------
# Shared helpers extracted from molecule3D (renderer-independent)
# ---------------------------------------------------------------------------

def test_next_disorder_label_uses_letter_suffixes():
    assert next_disorder_label('O1', set()) == 'O1B'
    assert next_disorder_label('O1', {'O1B'}) == 'O1C'


def test_next_disorder_label_falls_back_to_numeric_suffix():
    used = {f'C1{letter}' for letter in 'BCDEFGHIJKLMNOPQRSTUVWXYZ'}
    assert next_disorder_label('C1', used) == 'C1_dup2'


def test_riding_atoms_maps_every_bonded_hydrogen():
    types = ['C', 'H', 'C']
    positions = [np.zeros(3), np.array([0.0, 1.0, 0.0]), np.array([1.5, 0.0, 0.0])]
    assert riding_atoms(types, positions, [(0, 1), (0, 2)]) == {1: 0}


def test_riding_atoms_prefers_the_closest_heavy_partner():
    """A hydrogen the bond table gave two partners rides on the nearer one."""
    types = ['O', 'H', 'O']
    positions = [np.zeros(3), np.array([0.0, 0.98, 0.0]), np.array([0.0, 2.1, 0.0])]
    assert riding_atoms(types, positions, [(0, 1), (1, 2)]) == {1: 0}


def test_riding_atoms_ignores_unbonded_hydrogens():
    types = ['C', 'H']
    positions = [np.zeros(3), np.array([0.0, 1.0, 0.0])]
    assert riding_atoms(types, positions, []) == {}


def test_riding_atoms_falls_back_to_a_hydrogen_partner():
    """H2 has only hydrogen neighbours, so the closest one is the parent."""
    types = ['H', 'H']
    positions = [np.zeros(3), np.array([0.0, 0.74, 0.0])]
    assert riding_atoms(types, positions, [(0, 1)]) == {0: 1, 1: 0}


def test_disorder_split_resolves_atoms_onto_the_matching_part():
    split = DisorderSplit()
    split.register({1: 5, 2: 6})
    assert split.is_duplicate == {5, 6}
    # side_of is a duplicate -> map originals onto their copies
    assert split.matching_split_atom(1, 5) == 5
    # side_of is an original -> map copies back onto their originals
    assert split.matching_split_atom(6, 1) == 2
    # an atom that was never duplicated (an anchor) is returned unchanged
    assert split.matching_split_atom(9, 5) == 9
    assert split.matching_split_atom(9, 1) == 9


def test_disorder_split_clear_forgets_everything():
    split = DisorderSplit()
    split.register({1: 5})
    split.clear()
    assert not split.duplicate_of and not split.is_duplicate


def test_plan_disorder_duplicate_names_and_bonds_the_copies():
    #   anchor(0) - 1 - 2
    connections = [(0, 1), (1, 2)]
    plan, new_edges = plan_disorder_duplicate(
        connections, {1, 2}, {0}, {'C0', 'C1', 'C2'}, ['C0', 'C1', 'C2'], 3,
    )
    assert plan == [(1, 'C1B', 3), (2, 'C2B', 4)]
    # The copies bond to each other and to the shared anchor, exactly as the
    # originals did; the anchor itself is never duplicated.
    assert set(new_edges) == {(0, 3), (3, 4)}


def test_plan_disorder_duplicate_empty_moiety():
    assert plan_disorder_duplicate([(0, 1)], set(), set(), set(), ['A', 'B'], 2) == ([], ())


def test_part_fade_and_fading_are_qt_free_and_monotonic():
    from fastmolwidget.atoms import PART_FADE_MAX, fade_towards_white, part_fade

    assert part_fade(0) == 0.0
    assert part_fade(1) == 0.0
    assert 0.0 < part_fade(2) < part_fade(3)
    assert part_fade(99) == pytest.approx(PART_FADE_MAX)
    # Fading keeps the hue but moves every channel towards white.
    assert fade_towards_white((0.0, 0.0, 0.0), 0.5) == pytest.approx((0.5, 0.5, 0.5))
    assert fade_towards_white((0.2, 0.4, 0.6), 0.0) == pytest.approx((0.2, 0.4, 0.6))
    assert fade_towards_white((0.2, 0.4, 0.6), 1.0) == pytest.approx((1.0, 1.0, 1.0))


def test_disorder_controller_imports_no_qt():
    """The controller is shared by every renderer (including a future non-Qt
    host), so importing it must not pull Qt in."""
    import subprocess
    import sys

    code = (
        'import sys, fastmolwidget.disorder_controller\n'
        'bad = [m for m in sys.modules if m.split(".")[0] in '
        '("PyQt5", "PyQt6", "PySide2", "PySide6", "qtpy")]\n'
        'print(",".join(sorted(bad)))\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == ''
