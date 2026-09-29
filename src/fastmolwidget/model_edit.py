"""Writing interactive model edits back to a SHELX ``.res``/``.ins`` file.

The disorder drag (:mod:`fastmolwidget.disorder_controller`) changes only the
atoms a renderer draws.  This module keeps the refinement model in step:

* :class:`ModelEditSession` owns a :class:`shelxfile.edit.ShelxDocument` of
  the loaded file.  Every finished drag is committed to it as **one** undo
  step, the loader re-reads the displayed atoms from it (so edits survive
  Grow/Pack), and :meth:`ModelEditSession.save` writes it out.
* :class:`AtomSource` ties each displayed atom to its line in the file plus
  the symmetry operation that produced the displayed copy, so a dragged
  symmetry image is written back into the asymmetric unit.
* :func:`spring_restraints` turns the drag's springs into SHELXL restraints:
  ``SADI`` between part-1 and part-2 counterparts (bonds at
  :data:`SADI_BOND_ESD`, 1,3 pairs at :data:`SADI_ANGLE_ESD`), ``FLAT`` for
  every planar group of each part, and ``RIGU`` + ``SIMU`` over both parts.

A split writes the originals as ``PART 1`` on a new free variable (sof
``10·fv + p``) and the copies as ``PART 2`` (sof ``−(10·fv + p)``), *p* being
the atom's own fixed occupancy.  Moieties on a special position get a
negative ``PART -2`` for the copies, so SHELXL applies no special-position
constraints to them:

* atoms with a reduced site occupancy (``p < 1``, e.g. ``10.33333`` on a
  three-fold axis) keep *p*: copy sof ``−(10·fv + p)``;
* when the dragged moiety holds *k* symmetry images of the same atoms (grown
  across the special position), the whole moiety is written as explicit
  atoms where it is shown, each with occupancy ``(1 − fv)/k``.

The restraints go before the first atom, outside every ``PART``/``RESI``
scope; restraints across symmetry use ``EQIV`` names.

Nothing here imports Qt.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import gemmi
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:  # pragma: no cover - typing only
    from shelxfile import Shelxfile
    from shelxfile.atoms.atom import Atom
    from shelxfile.edit import ShelxDocument

    from fastmolwidget.disorder_drag import DragEdit, DragSplit
    from fastmolwidget.sdm import Atomtuple

__all__ = [
    'DISORDER_FVAR_START',
    'SADI_ANGLE_ESD',
    'SADI_BOND_ESD',
    'AtomSource',
    'CommitReport',
    'ModelEditError',
    'ModelEditSession',
    'SplitPair',
    'spring_restraints',
    'symop_from_arrays',
    'symop_invert',
    'symop_to_shelx',
]

#: ``SADI`` esd for the full-stiffness springs (real bonds).
SADI_BOND_ESD: float = 0.02
#: ``SADI`` esd for the loose 1,3 springs.
SADI_ANGLE_ESD: float = 0.04
#: Starting value of the free variable that refines the split occupancy.
DISORDER_FVAR_START: float = 0.5
#: Two positions of the same atom closer than this (Å) are the same position.
_SAME_POSITION = 0.01
#: Occupancy codes ``10 < sof < 15`` mean "fixed at sof − 10".
_FIXED_OCCUPANCY_MIN = 10.0
_FIXED_OCCUPANCY_MAX = 15.0


class ModelEditError(ValueError):
    """An edit cannot be written back to the model; nothing was changed."""


# ---------------------------------------------------------------- symmetry

#: ``X, Y, Z`` -- how an atom of the asymmetric unit is displayed.
IDENTITY_OP = gemmi.Op()


def symop_from_arrays(rotation: ArrayLike, translation: ArrayLike) -> gemmi.Op:
    """A crystallographic operation ``x' = R·x + t`` from float arrays.

    :class:`gemmi.Op` holds both parts as integer multiples of
    ``gemmi.Op.DEN`` (24), which snaps the floats to their exact grid and
    makes two spellings of one operation compare -- and hash -- equal.
    That is what the lookups here rely on.
    """
    op = gemmi.Op()
    op.rot = np.rint(np.asarray(rotation, dtype=float) * gemmi.Op.DEN).astype(int).tolist()
    op.tran = np.rint(np.asarray(translation, dtype=float) * gemmi.Op.DEN).astype(int).tolist()
    return op


def symop_to_shelx(op: gemmi.Op) -> str:
    """SHELXL text such as ``-X+1, Y, -Z+1/2``."""
    return op.triplet().upper().replace(',', ', ')


def symop_invert(op: gemmi.Op, frac: ArrayLike) -> np.ndarray:
    """The point that *op* maps onto *frac*."""
    point = [float(value) for value in np.asarray(frac, dtype=float)]
    return np.asarray(op.inverse().apply_to_xyz(point), dtype=float)


@dataclass(frozen=True)
class AtomSource:
    """Where a displayed atom comes from: a file atom and an operation.

    :ivar name: The atom's ``fullname_short`` (``C1`` or ``C1_3`` in residue 3).
    :ivar part: Its ``PART`` number, which with *name* identifies it.
    :ivar op: Maps the file atom's fractional coordinates onto the
        displayed copy.
    """

    name: str
    part: int
    op: gemmi.Op = field(default_factory=gemmi.Op)

    @property
    def key(self) -> tuple[str, int]:
        return self.name.upper(), self.part


@dataclass(frozen=True)
class SplitPair:
    """One split atom and its copy, by name, for re-registering after reload.

    :ivar first: The part-1 atom's name.
    :ivar second: The copy's name.
    :ivar op: ``None`` when both are displayed through the same operation
        (an ordinary split).  For a special-position split the copy is an
        explicit atom displayed as itself, while *first* is displayed through
        *op*.
    """

    first: str
    second: str
    op: gemmi.Op | None = None


@dataclass
class CommitReport:
    """What committing one drag did."""

    label: str
    messages: list[str] = field(default_factory=list)
    restraints: list[str] = field(default_factory=list)


# ------------------------------------------------------------- restraints

def spring_restraints(
    split: DragSplit,
    first: dict[int, str],
    second: dict[int, str],
    hydrogens: set[int] | frozenset[int] = frozenset(),
) -> list[str]:
    """SHELXL restraints mirroring the springs of a split drag.

    :param split: The recorded split.  Indices refer to the originals and
        anchors.
    :param first: Instruction name of every original and anchor atom
        (``C1A``, ``C1A_$2``, ...).
    :param second: Instruction name of every original's copy; anchors are
        shared and absent here.
    :param hydrogens: Indices of hydrogen atoms; they ride in SHELXL and get
        no geometry or ADP restraints.
    :returns: Instruction lines, in the order they should be written.
    """
    def counterpart(index: int) -> str:
        return second.get(index, first[index])

    lines: list[str] = []
    for esd, pairs in ((SADI_BOND_ESD, split.bonds), (SADI_ANGLE_ESD, split.angle_pairs)):
        for a, b in pairs:
            if a in hydrogens or b in hydrogens:
                continue
            if a not in second and b not in second:
                continue  # both anchors: nothing was split here
            lines.append(f'SADI {esd:.2f} {first[a]} {first[b]} {counterpart(a)} {counterpart(b)}')

    for group in split.planar_groups:
        atoms = [i for i in group if i not in hydrogens]
        if len(atoms) < 4:
            continue
        lines.append('FLAT ' + ' '.join(first[i] for i in atoms))
        lines.append('FLAT ' + ' '.join(counterpart(i) for i in atoms))

    adp_atoms: list[str] = []
    for index in sorted(split.anchors) + sorted(split.duplicates):
        if index in hydrogens:
            continue
        for name in (first[index], second.get(index)):
            # Symmetry images are left out of ADP restraints.
            if name and '_$' not in name and name not in adp_atoms:
                adp_atoms.append(name)
    if len(adp_atoms) >= 2:
        lines.append('RIGU ' + ' '.join(adp_atoms))
        lines.append('SIMU ' + ' '.join(adp_atoms))
    return lines


# ---------------------------------------------------------------- session

def _fixed_occupancy(atom: Atom) -> float:
    """The atom's own fixed occupancy *p* from ``sof = 10 + p``."""
    sof = float(atom.sof)
    if _FIXED_OCCUPANCY_MIN < sof < _FIXED_OCCUPANCY_MAX:
        return round(sof - _FIXED_OCCUPANCY_MIN, 5)
    raise ModelEditError(
        f'{atom.fullname_short}: its occupancy ({sof:g}) is already refined or tied '
        f'to a free variable, so it cannot be split automatically'
    )


class ModelEditSession:
    """The editable SHELX model behind a displayed structure.

    :param document: The model.
    :param path: The file it was read from, used as the default save target
        and for :meth:`default_save_path`.
    """

    def __init__(self, document: ShelxDocument, path: Path) -> None:
        self._document = document
        self._path = Path(path)
        self._pairs: tuple[SplitPair, ...] = ()
        document.set_state_provider(lambda: self._pairs)
        document.mark_saved()

    @classmethod
    def from_file(cls, path: str | Path) -> ModelEditSession:
        """Read a ``.res``/``.ins`` file.

        :raises ValueError: when the file has no ``CELL``.
        """
        from shelxfile.edit import ShelxDocument

        document = ShelxDocument.from_file(path)
        if not document.shelxfile.cell:
            raise ValueError(f'No CELL instruction found in SHELX file: {path}')
        return cls(document, Path(path))

    # ----------------------------------------------------------- queries

    @property
    def document(self) -> ShelxDocument:
        return self._document

    @property
    def shelxfile(self) -> Shelxfile:
        """The current model.  A new object after every undo/redo."""
        return self._document.shelxfile

    @property
    def path(self) -> Path:
        return self._path

    @property
    def pairs(self) -> tuple[SplitPair, ...]:
        return self._pairs

    @property
    def is_modified(self) -> bool:
        """Whether there are edits not yet saved."""
        return self._document.is_modified

    @property
    def has_edits(self) -> bool:
        """Whether anything was ever committed (undone steps included)."""
        return self._document.can_undo or self._document.can_redo

    @property
    def can_undo(self) -> bool:
        return self._document.can_undo

    @property
    def can_redo(self) -> bool:
        return self._document.can_redo

    @property
    def undo_label(self) -> str | None:
        return self._document.history.undo_label

    @property
    def redo_label(self) -> str | None:
        return self._document.history.redo_label

    def default_save_path(self) -> Path:
        """``<basename>.ins`` next to the loaded file."""
        return self._path.with_suffix('.ins')

    # --------------------------------------------------------- history

    def undo(self) -> str | None:
        """Undo the last committed edit; returns its label, or ``None``."""
        result = self._document.undo()
        if result is None:
            return None
        self._pairs = tuple(result.payload or ())
        return result.label

    def redo(self) -> str | None:
        """Redo the last undone edit; returns its label, or ``None``."""
        result = self._document.redo()
        if result is None:
            return None
        self._pairs = tuple(result.payload or ())
        return result.label

    # ------------------------------------------------------------ saving

    def save(self, path: str | Path | None = None, *, backup: bool = True) -> Path:
        """Write the model to *path* (default :meth:`default_save_path`).

        An existing file is first copied to ``<name>.bak`` when *backup*.
        """
        target = Path(path) if path is not None else self.default_save_path()
        if backup and target.exists():
            shutil.copy2(target, target.with_name(target.name + '.bak'))
        self._document.write(target)
        return target

    # ------------------------------------------------- display mapping

    def _atoms_by_key(self) -> dict[tuple[str, int], Atom]:
        return {
            (atom.fullname_short.upper(), atom.part.n): atom
            for atom in self.shelxfile.atoms if not atom.qpeak
        }

    def _cell(self) -> list[float]:
        cell = self.shelxfile.cell
        if cell is None:
            raise ModelEditError('The SHELX model has no CELL')
        return list(cell)

    def _orthogonalisation(self) -> np.ndarray:
        """Matrix ``M`` with ``cartesian = M @ fractional``, in shelxfile's convention.

        Built from shelxfile's own conversion so fractional coordinates
        computed here agree with the ones it writes.
        """
        from shelxfile.misc.misc import frac_to_cart

        cell = self._cell()
        return np.array([frac_to_cart(list(axis), cell) for axis in np.eye(3)], dtype=float).T

    def sources_for(self, atoms: Sequence[Atomtuple]) -> list[AtomSource | None]:
        """Map each displayed atom (as handed to the renderer) to its source.

        The operation is reconstructed from the displayed position, the file
        position and the rotation part stored in ``Atomtuple.symm_matrix``,
        so grown and packed atoms map back exactly, lattice translations
        included.  Atoms that match nothing in the file map to ``None``.
        """
        by_key = self._atoms_by_key()
        if not atoms:
            return []
        to_frac = np.linalg.inv(self._orthogonalisation())
        cartesian = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
        fractional = cartesian @ to_frac.T
        sources: list[AtomSource | None] = []
        for displayed, frac in zip(atoms, fractional):
            key = (str(displayed.label).upper(), int(displayed.part or 0))
            atom = by_key.get(key)
            if atom is None:
                sources.append(None)
                continue
            if displayed.symm_matrix is None:
                rotation = np.eye(3)
            else:
                # Stored column-major: displayed = symm_matrixᵀ · asu + t.
                rotation = np.asarray(displayed.symm_matrix, dtype=float).T
            translation = frac - rotation @ np.asarray(atom.frac_coords, dtype=float)
            sources.append(AtomSource(atom.fullname_short, atom.part.n,
                                      symop_from_arrays(rotation, translation)))
        return sources

    def pairs_in(self, sources: Sequence[AtomSource | None]) -> list[tuple[int, int]]:
        """``(part-1 index, part-2 index)`` of every recorded split pair shown."""
        by_name: dict[str, list[tuple[int, gemmi.Op]]] = {}
        for index, source in enumerate(sources):
            if source is not None:
                by_name.setdefault(source.name.upper(), []).append((index, source.op))
        found: list[tuple[int, int]] = []
        for pair in self._pairs:
            firsts = by_name.get(pair.first.upper(), [])
            seconds = by_name.get(pair.second.upper(), [])
            if pair.op is None:
                second_by_op = {op: index for index, op in seconds}
                for index, op in firsts:
                    if op in second_by_op:
                        found.append((index, second_by_op[op]))
            else:
                first = next((i for i, op in firsts if op == pair.op), None)
                second = next((i for i, op in seconds if op == IDENTITY_OP), None)
                if first is not None and second is not None:
                    found.append((first, second))
        return found

    # ------------------------------------------------------------ commit

    def commit(self, edit: DragEdit, sources: Sequence[AtomSource | None],
               *, hydrogens: Iterable[int] = ()) -> CommitReport:
        """Write one finished drag into the model as a single undo step.

        :param edit: What the drag changed.
        :param sources: The mapping of the displayed atoms at the time of the
            drag, from :meth:`sources_for`.  Copies made during the drag
            have indices beyond its end.
        :param hydrogens: Indices of hydrogen atoms among the displayed ones.
        :raises ModelEditError: when the edit cannot be expressed; the model
            and this session's bookkeeping are left unchanged.  Every
            failure is reported this way, so a caller never has to guess
            which exception the editing layer below might raise.
        """
        hydrogen_set = set(hydrogens)
        split = edit.split
        copies = set(split.duplicates.values()) if split is not None else set()
        label = self._label(edit, sources)
        report = CommitReport(label)
        pairs_before = self._pairs
        try:
            with self._document.batch(label):
                if split is not None:
                    self._commit_split(split, edit, sources, hydrogen_set, report)
                moves = {i: p for i, p in edit.positions.items() if i not in copies}
                self._commit_moves(moves, sources, report)
        except ModelEditError:
            # ShelxDocument.batch() rolled the model back; the split pairs
            # recorded inside the block are ours to undo.
            self._pairs = pairs_before
            raise
        except Exception as error:
            self._pairs = pairs_before
            raise ModelEditError(str(error) or type(error).__name__) from error
        return report

    def _label(self, edit: DragEdit, sources: Sequence[AtomSource | None]) -> str:
        def name(index: int) -> str:
            source = sources[index] if index < len(sources) else None
            return source.name if source is not None else f'#{index}'

        if edit.split is not None:
            originals = sorted(edit.split.duplicates)
            return f'Split {name(originals[0])} ({len(originals)} atoms)'
        moved = sorted(edit.positions)
        if len(moved) == 1:
            return f'Move {name(moved[0])}'
        return f'Move {len(moved)} atoms'

    def _source(self, sources: Sequence[AtomSource | None], index: int) -> AtomSource:
        source = sources[index] if 0 <= index < len(sources) else None
        if source is None:
            raise ModelEditError('A dragged atom has no counterpart in the SHELX file')
        return source

    def _atom(self, source: AtomSource) -> Atom:
        atom = self._atoms_by_key().get(source.key)
        if atom is None:
            raise ModelEditError(f'{source.name} is no longer in the SHELX file')
        return atom

    def _frac(self, cartesian: ArrayLike) -> np.ndarray:
        from shelxfile.misc.misc import cart_to_frac

        point = [float(value) for value in np.asarray(cartesian, dtype=float)]
        return np.asarray(cart_to_frac(point, self._cell()), dtype=float)

    def _commit_moves(self, moves: dict[int, np.ndarray],
                      sources: Sequence[AtomSource | None], report: CommitReport) -> None:
        if not moves:
            return
        orthogonalisation = self._orthogonalisation()
        targets: dict[tuple[str, int], tuple[Atom, np.ndarray]] = {}
        for index, position in sorted(moves.items()):
            source = self._source(sources, index)
            atom = self._atom(source)
            frac = symop_invert(source.op, self._frac(position))
            previous = targets.get(source.key)
            if previous is not None:
                gap = np.linalg.norm(orthogonalisation @ (previous[1] - frac))
                if gap > _SAME_POSITION:
                    raise ModelEditError(
                        f'{atom.fullname_short} was moved through two different symmetry '
                        f'images to two different places')
            targets[source.key] = (atom, frac)
        result = self._document.move_atoms(
            [(atom, frac) for atom, frac in targets.values()], cartesian=False)
        report.messages.extend(result.messages)

    def _uvals_for(self, atom: Atom, isotropic: bool) -> list[float] | None:
        """U values of a split atom: flattened to isotropic, or ``None``."""
        from fastmolwidget.disorder_drag import DEFAULT_ISO_U

        if not isotropic or atom.is_hydrogen:
            return None  # hydrogens keep their riding U codes
        return [DEFAULT_ISO_U]

    def _commit_split(self, split: DragSplit, edit: DragEdit,
                      sources: Sequence[AtomSource | None], hydrogens: set[int],
                      report: CommitReport) -> None:
        document = self._document
        originals = sorted(split.duplicates)
        original_sources = [self._source(sources, i) for i in originals]
        anchor_sources = {i: self._source(sources, i) for i in split.anchors}
        # Several images of the moiety were dragged together: the moiety sits
        # on a special position and is written out explicitly, all images.
        ops = list(dict.fromkeys(source.op for source in original_sources))
        explicit = len(ops) > 1

        # One file atom per distinct key, in drag order.
        file_atoms: dict[tuple[str, int], Atom] = {}
        for source in original_sources:
            if source.key not in file_atoms:
                file_atoms[source.key] = self._atom(source)
        for atom in file_atoms.values():
            if atom.part.n != 0:
                raise ModelEditError(
                    f'{atom.fullname_short} is already in PART {atom.part.n}; '
                    f'splitting existing disorder is not supported')
        occupancy = {key: _fixed_occupancy(atom) for key, atom in file_atoms.items()}
        problem = document.atom_problem_for_part(file_atoms.values())
        if problem:
            raise ModelEditError(f'Cannot split: {problem}')
        # A reduced site occupancy marks atoms on a special position, whose
        # displaced copy must not be held there by symmetry: negative PART.
        special = explicit or min(occupancy.values()) < 1.0 - 1e-3
        second_part = -2 if special else 2

        fvar = document.add_free_variable(DISORDER_FVAR_START)
        base = 10 * fvar
        taken: set[str] = set()
        first_names: dict[tuple[str, int], str] = {}
        for key, atom in file_atoms.items():
            first_names[key] = document.split_name(atom, 'A', taken, keep_own_name=True)
            taken.add(first_names[key].upper())

        groups: dict[gemmi.Op, list[int]] = {}
        for index, source in zip(originals, original_sources):
            groups.setdefault(source.op if explicit else ops[0], []).append(index)
        copy_atoms: dict[int, Atom] = {}
        for indices in groups.values():
            atoms, names, coordinates, uvals, sofs = [], [], [], [], []
            for index in indices:
                source = self._source(sources, index)
                atom = file_atoms[source.key]
                atoms.append(atom)
                name = document.split_name(atom, 'B', taken)
                taken.add(name.upper())
                names.append(name)
                position = self._frac(edit.positions[split.duplicates[index]])
                # Explicit copies are written where they are shown; otherwise
                # the copy goes back into the asymmetric unit.
                coordinates.append(position if explicit
                                   else symop_invert(source.op, position))
                uvals.append(self._uvals_for(atom, split.isotropic))
                share = 1.0 / len(ops) if explicit else occupancy[source.key]
                sofs.append(-(base + round(share, 5)))
            made = document.duplicate_atoms(
                atoms, names, second_part, sofs, coordinates,
                cartesian=False, uvals=uvals).added
            copy_atoms.update(zip(indices, made))
        if special:
            report.messages.append(
                f'Special position: the copy is written as PART {second_part} on '
                f'free variable {fvar}, so no special-position constraints apply to it')
        part_one = list(file_atoms.values())
        document.assign_part(part_one, 1, [base + occupancy[key] for key in file_atoms])
        flatten = [(atom, u) for atom in part_one
                   if (u := self._uvals_for(atom, split.isotropic)) is not None]
        if flatten:
            document.set_uvals(flatten)
        for key, atom in file_atoms.items():
            document.rename_atom(atom, first_names[key])

        # Restraint names, all seen from the frame of the first original.
        reference = original_sources[0].op
        into_reference_frame = reference.inverse()

        def name_of(atom: Atom, op: gemmi.Op) -> str:
            return document.name_for_operation(
                atom, symop_to_shelx(into_reference_frame * op))

        first: dict[int, str] = {}
        second: dict[int, str] = {}
        for index, source in zip(originals, original_sources):
            first[index] = name_of(file_atoms[source.key], source.op)
            copy_op = IDENTITY_OP if explicit else source.op
            second[index] = name_of(copy_atoms[index], copy_op)
        for index, source in anchor_sources.items():
            first[index] = name_of(self._atom(source), source.op)

        restraints = spring_restraints(split, first, second, hydrogens)
        for line in restraints:
            document.add_restraint(line, header=True)
        report.restraints.extend(restraints)

        pairs = list(self._pairs)
        for index, source in zip(originals, original_sources):
            first_name = file_atoms[source.key].fullname_short
            copy = copy_atoms[index]
            if explicit:
                pairs.append(SplitPair(first_name, copy.fullname_short, source.op))
            else:
                pair = SplitPair(first_name, copy.fullname_short)
                if pair not in pairs:
                    pairs.append(pair)
        self._pairs = tuple(pairs)
