"""Qt-free test host for the disorder drag, shared by the model-edit tests.

Implements the :class:`~fastmolwidget.disorder_controller.DisorderDragMixin`
hooks on plain lists and the tiny part of the renderer API that
:class:`~fastmolwidget.loader.MoleculeLoader` needs, so the whole chain -
load, drag, commit, reload - runs without Qt or OpenGL.
"""

from __future__ import annotations

import numpy as np

from fastmolwidget.disorder_controller import DisorderDragMixin
from fastmolwidget.tools import build_conntable


class FakeRenderer(DisorderDragMixin):
    """Stores atoms as arrays; picking and projection are scripted."""

    def __init__(self) -> None:
        self._init_disorder_drag()
        self.labels: list[str] = []
        self.types: list[str] = []
        self.parts: list[int] = []
        self.positions: list[np.ndarray] = []
        self.connections: tuple[tuple[int, int], ...] = ()
        self.edits: list = []
        self.model_source = None
        self.pick: int | None = None
        self.target: np.ndarray | None = None

    # -- the MoleculeLoader side ---------------------------------------

    def set_model_source(self, model, reflections=None) -> None:
        self.model_source = model

    def open_molecule(self, atoms, cell=None, keep_view: bool = False) -> None:
        self._reset_disorder_split()
        self.labels = [a.label for a in atoms]
        self.types = [a.type for a in atoms]
        self.parts = [int(a.part or 0) for a in atoms]
        self.positions = [np.array([a.x, a.y, a.z], dtype=float) for a in atoms]
        symmgen = [a.symm_matrix is not None and not np.allclose(a.symm_matrix, np.eye(3))
                   for a in atoms]
        self.connections = tuple(build_conntable(
            np.array(self.positions), self.types, self.parts, symmgen=symmgen))

    def index(self, label: str, part: int | None = None) -> int:
        for i, (name, atom_part) in enumerate(zip(self.labels, self.parts)):
            if name == label and (part is None or atom_part == part):
                return i
        raise KeyError(label)

    # -- scripted gestures ---------------------------------------------

    def drag(self, grabbed: int, shift, anchors: set[str] = frozenset(),
             bond: tuple[str, str] | None = None, steps: int = 5) -> None:
        """Ctrl+drag atom *grabbed* by *shift* (Å) and release."""
        self.pick = grabbed
        start = self.positions[grabbed].copy()
        assert self.try_start_moiety_drag(0.0, 0.0, set(anchors), bond)
        for step in range(1, steps + 1):
            self.target = start + np.asarray(shift, float) * step / steps
            self.update_moiety_drag(0.0, 0.0)
        self.end_drag()

    def move_single(self, index: int, shift) -> None:
        """Ctrl+Shift+drag one atom by *shift* (Å) and release."""
        self.pick = index
        assert self.try_start_single_atom_drag(0.0, 0.0)
        self.target = self.positions[index] + np.asarray(shift, float)
        self.update_single_atom_drag(0.0, 0.0)
        self.end_drag()

    # -- DisorderDragMixin hooks ---------------------------------------

    def _drag_atom_count(self) -> int:
        return len(self.labels)

    def _drag_atom_label(self, index: int) -> str:
        return self.labels[index]

    def _drag_atom_type(self, index: int) -> str:
        return self.types[index]

    def _drag_atom_position(self, index: int) -> np.ndarray:
        return self.positions[index].copy()

    def _drag_connections(self):
        return self.connections

    def _pick_atom_index(self, x: float, y: float) -> int | None:
        return self.pick

    def _begin_drag_projection(self, index: int, x: float, y: float) -> bool:
        return True

    def _drag_target(self, x: float, y: float):
        return self.target

    def _get_disorder_density_guide(self):
        return None

    def _set_atom_part(self, index: int, part: int) -> None:
        self.parts[index] = part

    def _clone_atom_for_split(self, index: int, label: str, part: int) -> int:
        self.labels.append(label)
        self.types.append(self.types[index])
        self.parts.append(part)
        self.positions.append(self.positions[index].copy())
        return len(self.labels) - 1

    def _add_connections(self, edges) -> None:
        self.connections = tuple(self.connections) + tuple(edges)

    def _apply_drag_positions(self, positions) -> None:
        for index, position in positions.items():
            self.positions[index] = np.asarray(position, dtype=float)

    def _on_model_edited(self, edit) -> None:
        self.edits.append(edit)
