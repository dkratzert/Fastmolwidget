"""Undo, redo and Save Model in :class:`MoleculeViewer3DWidget`.

The drag is scripted on the Qt-free :class:`fake_drag_host.FakeRenderer`
loaded from the same file, so its atom indices match the viewer's, and the
resulting edit is emitted through the real ``modelEdited`` signal.  No
OpenGL context is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fake_drag_host import FakeRenderer
from qtpy import QtWidgets

from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.viewer_widget3D import MoleculeViewer3DWidget

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

P31C = Path('tests/test-data/p31c-finalcif.res')


def scripted_split(path: Path):
    host = FakeRenderer()
    MoleculeLoader(host).load_file(path)
    host.drag(host.index('C4'), (0.4, 0.3, 0.0), anchors={'P1'})
    return host.edits[-1]


@pytest.fixture
def viewer(tmp_path):
    source = tmp_path / 'model.res'
    source.write_text(P31C.read_text())
    widget = MoleculeViewer3DWidget()
    widget.load_file(source)
    return widget, source


def labels(viewer: MoleculeViewer3DWidget) -> list[str]:
    return [atom.label for atom in viewer.render_widget.atoms]


def test_controls_start_disabled(viewer):
    widget, _ = viewer
    assert not widget._undo_button.isEnabled()
    assert not widget._redo_button.isEnabled()
    assert widget._save_model_button.isEnabled()


def test_edit_undo_redo_save(viewer, tmp_path):
    widget, source = viewer
    widget.render_widget.modelEdited.emit(scripted_split(source))
    assert 'C4B' in labels(widget)
    assert widget._undo_button.isEnabled()
    assert 'Split C4' in widget._undo_button.toolTip()
    assert widget.last_commit_report.restraints[0] == 'SADI 0.02 P1 C4A P1 C4B'

    assert widget.undo() == 'Split C4 (4 atoms)'
    assert 'C4B' not in labels(widget)
    assert widget._redo_button.isEnabled()
    assert widget.redo() == 'Split C4 (4 atoms)'
    assert 'C4B' in labels(widget)

    written = widget.save_model()
    assert written == tmp_path / 'model.ins'
    assert 'C4B' in written.read_text()
    assert not widget.edit_session.is_modified


def test_edit_survives_grow(viewer):
    widget, source = viewer
    widget.render_widget.modelEdited.emit(scripted_split(source))
    widget.grow()
    assert 'C4B' in labels(widget)


def test_refused_edit_is_reported_and_discarded(viewer, monkeypatch):
    widget, _ = viewer
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning',
                        lambda *args, **kwargs: warnings.append(args))
    from fastmolwidget.disorder_drag import DragEdit, DragSplit

    atoms = widget.render_widget.atoms
    n1 = next(i for i, atom in enumerate(atoms) if atom.label == 'N1')
    widget.render_widget.modelEdited.emit(DragEdit(
        positions={len(atoms): atoms[n1].center},
        split=DragSplit(duplicates={n1: len(atoms)})))
    assert warnings
    assert not widget.edit_session.can_undo


def test_cif_models_cannot_be_saved():
    widget = MoleculeViewer3DWidget()
    widget.load_file('tests/test-data/p21c.cif')
    assert widget.edit_session is None
    assert not widget._save_model_button.isEnabled()
    with pytest.raises(RuntimeError):
        widget.save_model()


def test_undo_shortcuts_do_not_claim_the_whole_window(viewer):
    """An embeddable widget must not shadow the host application's undo.

    A window-scoped Ctrl+Z would be ambiguous with the host's own binding,
    and Qt then fires neither.
    """
    from qtpy import QtCore, QtGui

    assert widget_shortcut_keys(viewer[0]) == {'Ctrl+Z', 'Ctrl+Y', 'Ctrl+Shift+Z'}
    for shortcut in viewer[0]._shortcuts:
        assert shortcut.context() == QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut
        assert isinstance(shortcut, QtGui.QShortcut)


def widget_shortcut_keys(widget) -> set[str]:
    return {s.key().toString() for s in widget._shortcuts}


def test_a_commit_that_fails_unexpectedly_is_reported_and_discarded(viewer, monkeypatch):
    """Any commit failure must reach the user, not abort the host process.

    ``_on_model_edited`` is a Qt slot, so an exception escaping it is fatal
    under PyQt6.  The display must also be resynchronised with the model,
    which the document has rolled back.
    """
    widget, source = viewer
    edit = scripted_split(source)
    session = widget._loader.edit_session

    def explode(*args, **kwargs):
        raise KeyError('C4')

    monkeypatch.setattr(type(session), '_commit_split', explode)
    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning',
                        lambda *args, **kwargs: warnings.append(args[2]))

    before = session.document.text
    widget.render_widget.modelEdited.emit(edit)

    assert warnings and 'C4' in warnings[0]
    assert widget.last_commit_report is None
    assert session.document.text == before
    assert 'C4B' not in labels(widget)
    assert not widget._undo_button.isEnabled()
