"""Renderer-independent driving of the interactive disorder-moiety drag.

:mod:`fastmolwidget.disorder_drag` is the pure geometry/optimisation of a
drag; this module is the layer above it that turns *user gestures* into drag
sessions: deciding what the anchors are, splitting a moiety off into a
permanent "part 2" the first time it is dragged, keeping track of which atom
is a copy of which, obtaining the density guide, and feeding new positions
back to whoever is drawing.

Like :mod:`~fastmolwidget.disorder_drag` it imports **neither Qt nor
OpenGL** - it is shared by the OpenGL widget and is ready for the QPainter
and Qt Quick renderers, so coordinates cross its boundary as plain floats
and NumPy arrays, never as ``QPointF``.

Everything renderer-specific is delegated to a small set of hooks the host
class implements (see :class:`DisorderDragMixin`): reading its own atom list,
picking, projecting the cursor into world space, cloning an atom, and writing
positions back.  That last one is deliberately a hook rather than shared code
poking at atom objects: :class:`~fastmolwidget.molecule_painter.MoleculeRendererMixin`
stores *view-frame* coordinates and tracks the model→view mapping separately,
so only the host knows how to apply a world-space result correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fastmolwidget.disorder_drag import MoietyDragSession
    from fastmolwidget.molecule_base import ModelSourceMixin

    #: The mixin needs its host's model/reflection lookup for the density
    #: guide.  Declaring the base only while type checking keeps the runtime
    #: MRO - and this module's Qt-freeness - untouched.
    _Base = ModelSourceMixin
else:
    _Base = object

__all__ = ['DisorderDragMixin']


class DisorderDragMixin(_Base):
    """Drives moiety and single-atom dragging for a molecule renderer.

    Mixed in *before* the renderer's Qt base class.  The host must call
    :meth:`_init_disorder_drag` from its constructor, reset the split state
    with :meth:`_reset_disorder_split` whenever it loads a new atom list, and
    implement the hooks at the bottom of this class.

    The gesture protocol is:

    * :meth:`try_start_moiety_drag` / :meth:`try_start_single_atom_drag` on
      the first confirmed movement of a press (never on the press itself, so
      a plain click keeps its normal selection meaning),
    * :meth:`update_moiety_drag` / :meth:`update_single_atom_drag` on every
      further move,
    * :meth:`end_drag` on release.
    """

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _init_disorder_drag(self) -> None:
        """Initialise all drag state; call once from the host's constructor."""
        from fastmolwidget.disorder_drag import DisorderSplit

        #: The active moiety drag session, or ``None`` between drags.
        self._disorder_drag_session: MoietyDragSession | None = None
        #: Cached dedicated density map (flattened isotropic ADPs) used only
        #: for snapping a dragged moiety - separate from the displayed map.
        self._disorder_density_guide = None
        #: Which atoms have been split into a part-2 copy, and how.
        self._disorder_split = DisorderSplit()
        #: Index of the atom being freely repositioned (Ctrl+Shift+drag), or
        #: ``None`` between drags.  No moiety, no anchors, no duplication -
        #: just that one atom's position changes.
        self._single_atom_drag_index: int | None = None

    @property
    def _disorder_duplicate_of(self) -> dict[int, int]:
        """Maps an original atom index to its permanent "part 2" duplicate.

        Created the first time that atom's moiety was split off, so a later
        drag of the same original moves the existing copy instead of making
        another one.
        """
        return self._disorder_split.duplicate_of

    @property
    def _disorder_is_duplicate(self) -> set[int]:
        """Indices of atoms that *are* such a duplicate (part 2), so grabbing
        one directly is recognised without creating yet another copy."""
        return self._disorder_split.is_duplicate

    def _reset_disorder_split(self) -> None:
        """Forget every split - a reloaded atom list invalidates all indices."""
        self._disorder_split.clear()

    def _matching_split_atom(self, index: int, side_of: int) -> int:
        """Resolve *index* onto the same split part as *side_of*.

        See :meth:`fastmolwidget.disorder_drag.DisorderSplit.matching_split_atom`.
        """
        return self._disorder_split.matching_split_atom(index, side_of)

    # ------------------------------------------------------------------
    # Moiety drag (Ctrl+drag)
    # ------------------------------------------------------------------

    def try_start_moiety_drag(
        self, x: float, y: float, anchor_labels: set[str],
        selected_bond: tuple[str, ...] | None = None,
    ) -> bool:
        """Begin a moiety drag for the atom at screen position *(x, y)*.

        Call this only once real movement of a Ctrl+left-button press is
        confirmed, so a plain Ctrl+click that never moves keeps behaving as
        ordinary add/remove from selection instead of side-effecting a split
        into existence.  *(x, y)* is normally the press position, so the user
        grabs whatever was under the cursor when the gesture started.

        *anchor_labels* are the fixed border atoms; the picked atom must
        belong to the fragment reachable from them (see
        :func:`~fastmolwidget.disorder_drag.find_moiety`).  With **no**
        anchors there is nothing holding the fragment: the whole connected
        fragment under the cursor is then dragged as a free body, which is how
        a whole-molecule disorder (a disordered solvent molecule with no fixed
        attachment point, say) is modelled.

        Alternatively a single **bond** may be given instead of anchors (the
        two are mutually exclusive in every host's selection handling).  The
        split point is then that bond: its far end becomes the anchor and the
        fragment rotates about the bond, with elastic give and with the near
        end free to drift off the axis so a tumble can be modelled - see
        :class:`~fastmolwidget.disorder_drag.TorsionDrag`.

        The *original* atoms never move: the very first time a given moiety is
        dragged it is duplicated into a permanent "part 2" and the duplicate is
        dragged instead, so the split stays available afterwards.  Dragging the
        same moiety again - either by grabbing an original that already has a
        split, or by grabbing the copy directly - moves the existing duplicate.

        :returns: ``True`` when a session was started.
        """
        from fastmolwidget.disorder_drag import (
            atomic_mass,
            bond_split_ends,
            build_drag_session,
            detect_planar_groups,
            find_moiety,
            riding_atoms,
        )

        count = self._drag_atom_count()
        if not count:
            return False

        grabbed_index = self._pick_atom_index(x, y)
        if grabbed_index is None:
            return False

        labels = [self._drag_atom_label(i) for i in range(count)]
        label_to_index = {label: i for i, label in enumerate(labels)}
        anchor_indices = {
            label_to_index[label] for label in anchor_labels
            if label in label_to_index
        }
        if grabbed_index in anchor_indices:
            return False

        connections = self._drag_connections()

        # A single selected bond defines the split point instead of anchor
        # atoms: its far end becomes the anchor and the fragment rotates
        # about the bond.
        split_bond: tuple[int, int] | None = None
        ends: tuple[int, int] | None = None
        if selected_bond is not None and not anchor_indices:
            if len(selected_bond) != 2:
                return False
            index_a = label_to_index.get(selected_bond[0])
            index_b = label_to_index.get(selected_bond[1])
            if index_a is None or index_b is None:
                return False
            # Once a split exists the bond's atoms exist twice, and the
            # selection still names whichever copy was clicked.  Resolve both
            # ends onto the side actually being grabbed first, or the search
            # below would walk out through the *other* part and anchor the
            # wrong atom.
            index_a = self._matching_split_atom(index_a, grabbed_index)
            index_b = self._matching_split_atom(index_b, grabbed_index)
            ends = bond_split_ends(connections, (index_a, index_b), grabbed_index)
            if ends is None:
                return False  # the bond has nothing to do with the fragment
            split_bond = (index_a, index_b)
            anchor_indices = {ends[0]}
            if grabbed_index in anchor_indices:
                return False

        if grabbed_index in self._disorder_is_duplicate:
            # Grabbed the split copy (part 2) directly: drag it as it is.
            drag_grabbed_index = grabbed_index
        elif grabbed_index in self._disorder_duplicate_of:
            # Grabbed an original (part 1) that has already been split off.
            # Drag that original, so the residual part can be positioned
            # independently of its part-2 counterpart and both halves of the
            # disorder can be adjusted.  Nothing is duplicated a second time.
            drag_grabbed_index = grabbed_index
        else:
            moiety = find_moiety(connections, anchor_indices, grabbed_index)
            if grabbed_index not in moiety:
                return False
            duplicate_map = self._create_disorder_duplicate(moiety, anchor_indices)
            if not duplicate_map:
                return False
            self._disorder_split.register(duplicate_map)
            drag_grabbed_index = duplicate_map[grabbed_index]
            connections = self._drag_connections()

        if split_bond is not None and ends is not None:
            # The bond's near end travels with the fragment and therefore
            # exists twice once a split has been made.  The torsion axis has
            # to point at whichever copy is actually being dragged - part 1 or
            # part 2 - while the far end anchors and is never duplicated.
            far, near = ends
            split_bond = (far, self._matching_split_atom(near, drag_grabbed_index))

        count = self._drag_atom_count()
        types = [self._drag_atom_type(i) for i in range(count)]
        positions = {i: self._drag_atom_position(i) for i in range(count)}
        moiety = find_moiety(connections, anchor_indices, drag_grabbed_index)
        hydrogen_indices = {i for i, element in enumerate(types) if element in {'H', 'D'}}
        planar_groups = detect_planar_groups(
            connections, positions, moiety, anchor_indices,
            exclude=hydrogen_indices,
        )
        session = build_drag_session(
            connections, positions, anchor_indices, drag_grabbed_index,
            density=self._get_disorder_density_guide(),
            masses={i: atomic_mass(t) for i, t in enumerate(types)},
            riding_atoms=riding_atoms(types, [positions[i] for i in range(count)],
                                      connections),
            bond=split_bond,
            planar_groups=planar_groups,
            planar_excluded=hydrogen_indices,
        )
        if session is None:
            return False
        if not self._begin_drag_projection(drag_grabbed_index, x, y):
            return False

        self._disorder_drag_session = session
        return True

    def update_moiety_drag(self, x: float, y: float) -> None:
        """Advance the active moiety drag towards screen position *(x, y)*."""
        session = self._disorder_drag_session
        if session is None:
            return
        target = self._drag_target(x, y)
        if target is None:
            return
        self._apply_drag_positions(session.update(target))

    # ------------------------------------------------------------------
    # Single free-atom drag (Ctrl+Shift+drag)
    # ------------------------------------------------------------------

    def try_start_single_atom_drag(self, x: float, y: float) -> bool:
        """Begin freely repositioning the single atom at *(x, y)*, if any.

        Unlike a moiety drag this never looks at the selection, never
        duplicates anything and never touches any other atom: it simply lets
        the one picked atom follow the cursor.  Call it only once real
        movement is confirmed, so a plain Ctrl+Shift+click does nothing.

        :returns: ``True`` when a drag was started.
        """
        if not self._drag_atom_count():
            return False

        index = self._pick_atom_index(x, y)
        if index is None:
            return False
        if not self._begin_drag_projection(index, x, y):
            return False

        self._single_atom_drag_index = index
        return True

    def update_single_atom_drag(self, x: float, y: float) -> None:
        """Move the grabbed atom to follow the cursor.

        Only that one atom's position changes - no bonds, parts, labels or any
        other atom are touched.
        """
        index = self._single_atom_drag_index
        if index is None:
            return
        target = self._drag_target(x, y)
        if target is None:
            return
        self._apply_drag_positions({index: target})

    def end_drag(self) -> None:
        """Finish any drag in progress.  Positions already applied are kept."""
        self._disorder_drag_session = None
        self._single_atom_drag_index = None
        self._end_drag_projection()

    # ------------------------------------------------------------------
    # Splitting
    # ------------------------------------------------------------------

    def _create_disorder_duplicate(
        self, moiety: set[int], anchors: set[int],
    ) -> dict[int, int]:
        """Duplicate *moiety* into a new, permanent disorder "part 2".

        The originals are forced to part 1 and are never touched again; the
        copies get unique labels (see
        :func:`~fastmolwidget.disorder_drag.next_disorder_label`) and part 2,
        bonded exactly like the originals were - both amongst themselves and
        to the shared *anchors* - so the result renders as an ordinary
        two-part disorder split.

        :returns: Mapping from each original atom index in *moiety* to its new
            duplicate's index, or ``{}`` when *moiety* is empty.
        """
        from fastmolwidget.disorder_drag import plan_disorder_duplicate

        count = self._drag_atom_count()
        labels = [self._drag_atom_label(i) for i in range(count)]
        plan, new_edges = plan_disorder_duplicate(
            self._drag_connections(), moiety, anchors, set(labels), labels, count,
        )
        if not plan:
            return {}

        duplicate_map: dict[int, int] = {}
        for original_index, label, duplicate_index in plan:
            self._set_atom_part(original_index, 1)
            created = self._clone_atom_for_split(original_index, label, 2)
            if created != duplicate_index:  # pragma: no cover - host contract
                raise RuntimeError(
                    'clone_atom_for_split must append the new atom: expected '
                    f'index {duplicate_index}, got {created}',
                )
            duplicate_map[original_index] = created

        self._add_connections(new_edges)
        self._on_split_parts_changed()
        return duplicate_map

    # ------------------------------------------------------------------
    # Density guidance
    # ------------------------------------------------------------------

    def _get_disorder_density_guide(self):
        """Lazily compute and cache the flattened-ADP density map for snapping.

        Uses the same model/reflection sources as the displayed residual
        density (``_density_sources``, from
        :class:`~fastmolwidget.molecule_base.ModelSourceMixin`), but with every
        ADP forced isotropic (see
        :func:`~fastmolwidget.density.force_isotropic_adps`) so a refined ADP
        does not bias the shape of the alternate-site peak.  Returns ``None``
        (dragging still works, just without guidance) when no model or
        reflections are available or the map cannot be computed.
        """
        if self._disorder_density_guide is not None:
            return self._disorder_density_guide

        from fastmolwidget.density import calculate_residual_density
        from fastmolwidget.disorder_drag import DEFAULT_ISO_U, DensityGuide

        try:
            model, reflections = self._density_sources(None, None)
            density_map = calculate_residual_density(
                model, reflections, iso_u_override=DEFAULT_ISO_U,
            )
        except Exception:  # noqa: BLE001 - guidance is optional
            return None

        self._disorder_density_guide = DensityGuide.from_map(density_map)
        return self._disorder_density_guide

    # ------------------------------------------------------------------
    # Host contract - every renderer implements these
    # ------------------------------------------------------------------

    def _drag_atom_count(self) -> int:
        """Number of atoms currently loaded."""
        raise NotImplementedError

    def _drag_atom_label(self, index: int) -> str:
        """Label of the atom at *index*."""
        raise NotImplementedError

    def _drag_atom_type(self, index: int) -> str:
        """Element symbol of the atom at *index*."""
        raise NotImplementedError

    def _drag_atom_position(self, index: int) -> np.ndarray:
        """World-space (model-frame) position of the atom at *index*."""
        raise NotImplementedError

    def _drag_connections(self) -> tuple[tuple[int, int], ...]:
        """Every bond as an ``(i, j)`` index pair - the table actually drawn."""
        raise NotImplementedError

    def _pick_atom_index(self, x: float, y: float) -> int | None:
        """Index of the atom under screen position *(x, y)*, or ``None``."""
        raise NotImplementedError

    def _begin_drag_projection(self, index: int, x: float, y: float) -> bool:
        """Prepare screen→world mapping for a drag grabbing *index* at *(x, y)*.

        Whatever the renderer needs to turn later cursor positions into world
        points (an inverse model-view matrix and a depth plane, in the 3-D
        case) is set up here and used by :meth:`_drag_target`.

        :returns: ``False`` to abort the drag, e.g. when the matrix is
            singular.
        """
        raise NotImplementedError

    def _drag_target(self, x: float, y: float) -> np.ndarray | None:
        """World-space point the grabbed atom should follow for *(x, y)*."""
        raise NotImplementedError

    def _end_drag_projection(self) -> None:
        """Release whatever :meth:`_begin_drag_projection` cached."""

    def _set_atom_part(self, index: int, part: int) -> None:
        """Set the disorder part of the atom at *index*."""
        raise NotImplementedError

    def _clone_atom_for_split(self, index: int, label: str, part: int) -> int:
        """Append a copy of the atom at *index* with *label* and *part*.

        The copy must look exactly like the original (same ADP/radius data)
        until it is refined separately.  Must **append**, and return the new
        atom's index.
        """
        raise NotImplementedError

    def _add_connections(self, edges: tuple[tuple[int, int], ...]) -> None:
        """Append *edges* to the bond table."""
        raise NotImplementedError

    def _apply_drag_positions(self, positions: dict[int, np.ndarray]) -> None:
        """Write new world positions back and refresh the display.

        Renderers that keep their atom coordinates in a *view* frame (see
        :class:`~fastmolwidget.molecule_painter.MoleculeRendererMixin`, which
        rotates coordinates in place and tracks the model→view mapping as
        ``_view_rotation``/``_view_offset``) must convert here, and must keep
        their cached coordinate arrays and any residual-density geometry
        consistent - which is exactly why this is a hook and not shared code.
        """
        raise NotImplementedError

    def _on_split_parts_changed(self) -> None:
        """Called after a split added a new disorder part.

        Hosts update whatever they expose as the available parts and notify
        their viewer (``available_parts`` / ``partsChanged`` in the Qt
        renderers).
        """
