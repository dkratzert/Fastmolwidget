"""Writing dragged disorder back into a SHELX model (``model_edit``).

The drag itself runs on :class:`fake_drag_host.FakeRenderer`, a Qt-free host
of :class:`~fastmolwidget.disorder_controller.DisorderDragMixin`, so the whole
chain - load, drag, commit, reload, undo, save - is exercised without OpenGL.
Every result is re-read with ``Shelxfile(debug=True)``, because what counts
is what SHELXL will see.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fake_drag_host import FakeRenderer
from shelxfile import Shelxfile

from fastmolwidget.atoms import HYDROGEN_ELEMENTS
from fastmolwidget.disorder_drag import DragEdit, DragSplit
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.model_edit import (
    IDENTITY_OP,
    ModelEditError,
    spring_restraints,
    symop_from_arrays,
    symop_invert,
    symop_to_shelx,
)

DATA = Path('tests/test-data')
P31C = DATA / 'p31c-finalcif.res'

#: Cl-C-C-Cl across an inversion centre: the asymmetric unit holds one half,
#: growing shows the whole molecule as two symmetry images.
INVERSION_RES = """TITL inv in P-1
CELL 0.71073 10 10 10 90 90 90
ZERR 1 0 0 0 0 0 0
LATT 1
SFAC C CL
UNIT 2 2
L.S. 4
FVAR 1.0
C1    1    0.075000    0.000000    0.000000    11.00000    0.03000
CL1   2    0.250000    0.000000    0.000000    11.00000    0.03000
HKLF 4
END
"""


def load(path) -> tuple[FakeRenderer, MoleculeLoader]:
    host = FakeRenderer()
    loader = MoleculeLoader(host)
    loader.load_file(path)
    return host, loader


def commit_last(host: FakeRenderer, loader: MoleculeLoader):
    hydrogens = [i for i, t in enumerate(host.types) if t in HYDROGEN_ELEMENTS]
    report = loader.edit_session.commit(host.edits[-1], loader.atom_sources,
                                        hydrogens=hydrogens)
    loader.reload()
    return report


def reparse(loader: MoleculeLoader) -> Shelxfile:
    shx = Shelxfile(debug=True)
    shx.read_string(loader.edit_session.document.text)
    return shx


def by_name(shx: Shelxfile, name: str):
    atom = shx.atoms.get_atom_by_name(name)
    assert atom is not None, name
    return atom


def translation_of(op) -> tuple[float, float, float]:
    """The fractional translation of a :class:`gemmi.Op`."""
    import gemmi

    return tuple(value / gemmi.Op.DEN for value in op.tran)


# ------------------------------------------------------------ symmetry

class TestSymOp:
    def test_text(self):
        op = symop_from_arrays(-np.eye(3), [1.0, 0.5, 0.0])
        assert symop_to_shelx(op) == '-X+1, -Y+1/2, -Z'
        assert symop_to_shelx(IDENTITY_OP) == 'X, Y, Z'

    def test_invert_undoes_apply(self):
        op = symop_from_arrays([[0, -1, 0], [1, -1, 0], [0, 0, 1]], [1.0, 0.0, 0.5])
        point = [0.1, 0.2, 0.3]
        np.testing.assert_allclose(symop_invert(op, op.apply_to_xyz(point)), point)

    def test_relative_to_itself_is_identity(self):
        op = symop_from_arrays(-np.eye(3), [1.0, 1.0, 0.0])
        assert op.inverse() * op == IDENTITY_OP

    def test_translations_snap_to_an_exact_grid(self):
        op = symop_from_arrays(np.eye(3), [0.33334, 0.0, 0.4999])
        assert translation_of(op) == pytest.approx((1 / 3, 0.0, 0.5))
        assert symop_to_shelx(op) == 'X+1/3, Y, Z+1/2'


# ---------------------------------------------------------- restraints

def test_spring_restraints_pair_each_spring_with_its_counterpart():
    split = DragSplit(
        duplicates={1: 10, 2: 11, 3: 12, 4: 13, 5: 14},
        anchors=(0,),
        bonds=((0, 1), (1, 2), (2, 3), (3, 4), (2, 5)),
        angle_pairs=((0, 2), (1, 5)),
        planar_groups=((0, 1, 2, 3),),
    )
    first = {0: 'N1', 1: 'C1A', 2: 'C2A', 3: 'C3A', 4: 'C4A', 5: 'H2A'}
    second = {1: 'C1B', 2: 'C2B', 3: 'C3B', 4: 'C4B', 5: 'H2B'}
    lines = spring_restraints(split, first, second, hydrogens={5})
    assert lines == [
        'SADI 0.02 N1 C1A N1 C1B',
        'SADI 0.02 C1A C2A C1B C2B',
        'SADI 0.02 C2A C3A C2B C3B',
        'SADI 0.02 C3A C4A C3B C4B',
        'SADI 0.04 N1 C2A N1 C2B',
        'FLAT N1 C1A C2A C3A',
        'FLAT N1 C1B C2B C3B',
        'RIGU N1 C1A C1B C2A C2B C3A C3B C4A C4B',
        'SIMU N1 C1A C1B C2A C2B C3A C3B C4A C4B',
    ]


def test_spring_restraints_leave_symmetry_images_out_of_adp_restraints():
    split = DragSplit(duplicates={0: 2, 1: 3}, bonds=((0, 1),))
    lines = spring_restraints(split, {0: 'C1A', 1: 'C1A_$1'}, {0: 'C1B', 1: 'C1C'})
    assert lines[0] == 'SADI 0.02 C1A C1A_$1 C1B C1C'
    assert lines[1] == 'RIGU C1A C1B C1C'


# ------------------------------------------------------------- drag edit

def test_drag_reports_the_split_and_every_copy():
    host, _ = load(P31C)
    host.drag(host.index('C4'), (0.4, 0.3, 0.0), anchors={'P1'})
    edit = host.edits[-1]
    names = sorted(host.labels[i] for i in edit.split.duplicates)
    assert names == ['C4', 'H4A', 'H4B', 'H4C']
    assert set(edit.positions) == set(edit.split.duplicates.values())
    assert edit.split.anchors == (host.index('P1'),)


def test_single_atom_drag_reports_only_that_atom():
    host, _ = load(P31C)
    index = host.index('CL1')
    host.move_single(index, (0.2, 0.0, 0.0))
    edit = host.edits[-1]
    assert edit.split is None
    assert list(edit.positions) == [index]


# ---------------------------------------------------------------- split

class TestOrdinarySplit:
    @pytest.fixture
    def split(self):
        host, loader = load(P31C)
        host.drag(host.index('C4'), (0.4, 0.3, 0.0), anchors={'P1'})
        target = host.positions[host.index('C4B')].copy()
        report = commit_last(host, loader)
        return host, loader, report, target

    def test_parts_occupancies_and_names(self, split):
        _, loader, _, _ = split
        shx = reparse(loader)
        fvar = len(shx.fvars)
        assert shx.fvars[fvar] == pytest.approx(0.5)
        for name in ('C4A', 'H4AA', 'H4BA', 'H4CA'):
            assert by_name(shx, name).part.n == 1
            assert by_name(shx, name).sof == pytest.approx(10 * fvar + 1)
        for name in ('C4B', 'H4AB', 'H4BB', 'H4CB'):
            assert by_name(shx, name).part.n == 2
            assert by_name(shx, name).sof == pytest.approx(-(10 * fvar + 1))
        assert by_name(shx, 'H4AB').afix.mn == 137
        assert by_name(shx, 'C4B').is_isotropic

    def test_copy_is_written_where_it_was_dropped(self, split):
        _, loader, _, target = split
        np.testing.assert_allclose(by_name(reparse(loader), 'C4B').cart_coords,
                                   target, atol=1e-3)

    def test_restraints(self, split):
        _, loader, report, _ = split
        assert report.restraints == [
            'SADI 0.02 P1 C4A P1 C4B', 'RIGU P1 C4A C4B', 'SIMU P1 C4A C4B']
        text = loader.edit_session.document.text
        for line in report.restraints:
            assert line in text

    def test_reload_registers_the_pairs(self, split):
        host, _, _, _ = split
        c4a, c4b = host.index('C4A'), host.index('C4B')
        assert host._disorder_split.duplicate_of[c4a] == c4b

    def test_dragging_the_copy_again_moves_it(self, split):
        host, loader, _, _ = split
        host.drag(host.index('C4B'), (0.1, 0.0, 0.0), anchors={'P1'})
        assert host.edits[-1].split is None
        moved = host.positions[host.index('C4B')].copy()
        report = commit_last(host, loader)
        assert report.label.startswith('Move')
        np.testing.assert_allclose(by_name(reparse(loader), 'C4B').cart_coords,
                                   moved, atol=1e-3)
        assert loader.edit_session.document.history.undo_labels == [
            'Split C4 (4 atoms)', report.label]

    def test_undo_and_redo(self, split):
        host, loader, _, _ = split
        session = loader.edit_session
        count = len(host.labels)
        assert session.undo() == 'Split C4 (4 atoms)'
        loader.reload()
        assert 'C4' in host.labels and 'C4B' not in host.labels
        assert session.pairs == ()
        assert not session.is_modified
        assert session.redo() == 'Split C4 (4 atoms)'
        loader.reload()
        assert len(host.labels) == count
        assert host._disorder_split.duplicate_of[host.index('C4A')] == host.index('C4B')

    def test_split_survives_grow_and_pack(self, split):
        host, loader, _, _ = split
        loader.set_grow(True)
        assert 'C4B' in host.labels
        loader.set_grow(False)
        loader.set_pack(True)
        assert 'C4B' in host.labels
        loader.set_pack(False)
        assert host.labels.count('C4B') == 1

    def test_loading_the_file_again_discards_the_edits(self, split):
        host, loader, _, _ = split
        loader.load_file(P31C)
        assert 'C4B' not in host.labels
        assert not loader.edit_session.has_edits


class TestSpecialPositions:
    def test_reduced_site_occupancy_gives_a_negative_part(self):
        host, loader = load(P31C)  # acetonitrile on a three-fold axis, sof 10.33333
        host.drag(host.index('C23'), (0.5, 0.2, 0.1))
        report = commit_last(host, loader)
        shx = reparse(loader)
        fvar = len(shx.fvars)
        assert by_name(shx, 'N3A').part.n == 1
        assert by_name(shx, 'N3A').sof == pytest.approx(10 * fvar + 1 / 3, abs=1e-4)
        assert by_name(shx, 'N3B').part.n == -2
        assert by_name(shx, 'N3B').sof == pytest.approx(-(10 * fvar + 1 / 3), abs=1e-4)
        assert any('Special position' in m for m in report.messages)

    def test_grown_images_are_written_explicitly(self, tmp_path):
        res = tmp_path / 'inv.res'
        res.write_text(INVERSION_RES)
        host, loader = load(res)
        loader.set_grow(True)
        assert host.labels == ['C1', 'CL1', 'C1', 'CL1']
        host.drag(host.index('CL1'), (0.0, 0.8, 0.0))
        report = commit_last(host, loader)
        shx = reparse(loader)
        part_one = [a for a in shx.atoms if a.part.n == 1]
        part_two = [a for a in shx.atoms if a.part.n == -2]
        assert len(part_one) == 2 and len(part_two) == 4
        for atom in part_one:
            assert atom.occupancy == pytest.approx(0.5)
        for atom in part_two:
            assert atom.occupancy == pytest.approx(0.25)  # (1 - 0.5) / 2 images
        assert 'EQIV $1 -X, -Y, -Z' in loader.edit_session.document.text
        assert 'SADI 0.02 C1A C1A_$1 C1B C1C' in report.restraints


def test_dragged_symmetry_image_is_written_back_into_the_asymmetric_unit(tmp_path):
    res = tmp_path / 'inv.res'
    res.write_text(INVERSION_RES)
    host, loader = load(res)
    loader.set_grow(True)
    image = next(i for i, s in enumerate(loader.atom_sources)
                 if s.name == 'CL1' and s.op != IDENTITY_OP)
    host.move_single(image, (0.0, 0.0, 0.5))  # +0.05 in z on the image
    commit_last(host, loader)
    assert by_name(reparse(loader), 'CL1').frac_coords == pytest.approx((0.25, 0.0, -0.05))


def test_an_existing_disorder_part_is_refused_and_nothing_changes():
    host, loader = load(P31C)
    n1 = host.index('N1')
    edit = DragEdit(positions={len(host.labels): host.positions[n1]},
                    split=DragSplit(duplicates={n1: len(host.labels)}))
    before = loader.edit_session.document.text
    with pytest.raises(ModelEditError, match='already in PART'):
        loader.edit_session.commit(edit, loader.atom_sources)
    assert loader.edit_session.document.text == before
    assert not loader.edit_session.can_undo


def test_non_shelx_files_have_no_session():
    _, loader = load(DATA / 'p21c.cif')
    assert loader.edit_session is None
    assert loader.atom_sources == []


# ----------------------------------------------------------------- saving

def test_save_writes_ins_and_backs_up(tmp_path):
    source = tmp_path / 'model.res'
    source.write_text(P31C.read_text())
    host, loader = load(source)
    host.drag(host.index('C4'), (0.4, 0.3, 0.0), anchors={'P1'})
    commit_last(host, loader)
    session = loader.edit_session
    assert session.is_modified
    target = session.save()
    assert target == tmp_path / 'model.ins'
    assert not session.is_modified
    assert 'C4B' in target.read_text()
    session.save()
    assert (tmp_path / 'model.ins.bak').exists()


def test_grown_sources_are_exact_crystallographic_operations():
    """Grown and packed atoms map back onto the file with exact operations.

    Checks on a hexagonal cell that the loader's Cartesian convention and
    shelxfile's agree: every translation must come out as a multiple of 1/12.
    """
    host, loader = load(P31C)
    for toggle in (loader.set_grow, loader.set_pack):
        toggle(True)
        sources = loader.atom_sources
        assert len(sources) == len(host.labels)
        assert all(source is not None for source in sources)
        for source in sources:
            twelfths = np.asarray(translation_of(source.op)) * 12
            np.testing.assert_allclose(twelfths, np.round(twelfths), atol=1e-6)
        toggle(False)


def test_a_failed_commit_leaves_no_trace_of_the_split(monkeypatch):
    """Every commit failure is a ModelEditError and rolls back completely.

    The document restores the model itself; the session's own split-pair
    bookkeeping is recorded inside the batch and has to be restored too,
    or the phantom pairs are baked into every later undo snapshot.
    """
    host, loader = load(P31C)
    host.drag(host.index('C4'), (0.4, 0.3, 0.0), anchors={'P1'})
    session = loader.edit_session
    before_text = session.document.text
    before_pairs = session._pairs

    def explode(*args, **kwargs):
        raise KeyError('no such atom')

    monkeypatch.setattr(type(session), '_commit_moves', explode)
    with pytest.raises(ModelEditError, match='no such atom'):
        session.commit(host.edits[-1], loader.atom_sources)

    assert session.document.text == before_text
    assert session._pairs == before_pairs
    assert not session.can_undo


def test_a_moiety_whose_afix_is_closed_after_a_comment_can_be_split():
    """REM lines between the last riding H and its AFIX 0 are comments.

    ``BB_LJ45_a.res`` carries an embedded ``REM <hkl>`` block right there,
    which used to push the duplicated block inside the AFIX 23 bracket and
    made the whole CHCl2 moiety un-splittable.
    """
    host, loader = load(DATA / 'BB_LJ45_a.res')
    host.drag(host.index('C1X'), (0.4, 0.3, 0.0), anchors=set())
    report = commit_last(host, loader)
    assert report.label.endswith('(5 atoms)')

    shx = reparse(loader)
    for name in ('CL3A', 'CL2A', 'C1XA', 'H1AA', 'H1BA'):
        assert by_name(shx, name).part.n == 1, name
    for name in ('CL3B', 'CL2B', 'C1XB', 'H1AB', 'H1BB'):
        assert by_name(shx, name).part.n != 0, name
    # Both AFIX 23 groups survived intact, riders next to their pivot.
    assert by_name(shx, 'H1AA').afix.mn == 23
    assert by_name(shx, 'H1AB').afix.mn == 23
