"""The SHELX text editor embedded in :class:`MoleculeViewer3DWidget`.

Both panes are views of a single :class:`~shelxfile.edit.ShelxDocument`, so
these tests check that an edit made on either side reaches the other, and
that the 3-D view follows the text *without* the user pressing Apply.

No OpenGL context is needed: nothing here paints, it only loads models and
reads back the atoms the renderer was given.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from qtpy import QtWidgets

from fastmolwidget.viewer_widget3D import EDITOR_COLUMNS, MoleculeViewer3DWidget

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

P31C = Path('tests/test-data/p31c-finalcif.res')


@pytest.fixture
def viewer(tmp_path):
    source = tmp_path / 'model.res'
    source.write_text(P31C.read_text())
    widget = MoleculeViewer3DWidget()
    widget.load_file(source)
    return widget


def labels(widget: MoleculeViewer3DWidget) -> list[str]:
    return [atom.label for atom in widget.render_widget.atoms]


def type_into(widget: MoleculeViewer3DWidget, text: str) -> None:
    """Put *text* in the editor as if the user had typed it."""
    widget.editor.editor.setPlainText(text)


def scripted_split(source: Path):
    """A drag that splits ``C4``, scripted on the Qt-free fake renderer.

    Loaded from the same file, so its atom indices match the viewer's.
    """
    from fake_drag_host import FakeRenderer

    from fastmolwidget.loader import MoleculeLoader

    host = FakeRenderer()
    MoleculeLoader(host).load_file(source)  # type: ignore[arg-type]
    host.drag(host.index('C4'), (0.4, 0.3, 0.0), anchors={'P1'})
    return host.edits[-1]


def loaded_path(widget: MoleculeViewer3DWidget) -> Path:
    return Path(widget.windowTitle())


# --------------------------------------------------------------- binding


def test_editor_and_view_share_one_document(viewer):
    assert viewer.editor.document is viewer.edit_session.document
    assert viewer.editor.shelxfile is viewer.edit_session.shelxfile
    assert viewer.editor.isEnabled()


def test_editor_text_is_the_loaded_model(viewer):
    assert viewer.editor.editor.toPlainText() == viewer.edit_session.document.text
    assert 'CELL' in viewer.editor.editor.toPlainText()


def test_editor_is_unbound_for_a_cif():
    widget = MoleculeViewer3DWidget()
    widget.load_file('tests/test-data/p21c.cif')
    assert widget.edit_session is None
    assert widget.editor.document is None
    assert not widget.editor.isEnabled()
    assert widget.editor.editor.toPlainText() == ''


def test_loading_another_model_rebinds_the_editor(viewer, tmp_path):
    first = viewer.editor.document
    other = tmp_path / 'other.res'
    other.write_text(P31C.read_text())
    viewer.load_file(other)
    assert viewer.editor.document is not first
    assert viewer.editor.document is viewer.edit_session.document


def test_switching_to_a_cif_and_back_rebinds(viewer, tmp_path):
    viewer.load_file('tests/test-data/p21c.cif')
    assert viewer.editor.document is None

    source = tmp_path / 'again.res'
    source.write_text(P31C.read_text())
    viewer.load_file(source)
    assert viewer.editor.document is viewer.edit_session.document
    assert viewer.editor.isEnabled()


# ----------------------------------------------------------- text -> 3-D


def test_typing_moves_the_3d_atoms_without_pressing_apply(viewer):
    text = viewer.editor.editor.toPlainText()
    assert 'C4' in labels(viewer)

    type_into(viewer, text.replace('\nC4 ', '\nC99 ', 1))
    viewer.flush_editor_text()

    assert 'C99' in labels(viewer)
    assert 'C4' not in labels(viewer)


def test_the_debounce_timer_applies_the_text(viewer, qtbot=None):
    """Typing starts the timer; firing it is what reaches the model."""
    text = viewer.editor.editor.toPlainText()
    type_into(viewer, text.replace('\nC4 ', '\nC98 ', 1))
    assert viewer._apply_timer.isActive()

    viewer._apply_timer.timeout.emit()
    assert 'C98' in labels(viewer)
    assert not viewer.editor.is_dirty


def test_unparsable_text_keeps_the_previous_model(viewer):
    before = labels(viewer)
    type_into(viewer, 'this is not a shelx file at all')

    assert viewer.flush_editor_text() is False
    assert labels(viewer) == before
    assert not viewer.editor.error_label.isHidden()


def test_live_apply_does_not_reformat_the_text(viewer):
    """Reformatting under the cursor while the user types is unacceptable."""
    typed = viewer.editor.editor.toPlainText() + '\nREM    spaced   out\n'
    type_into(viewer, typed)
    viewer.flush_editor_text()
    assert viewer.editor.editor.toPlainText() == typed


def test_a_burst_of_edits_is_one_undo_step(viewer):
    text = viewer.editor.editor.toPlainText()
    for n in range(3):
        type_into(viewer, text + f'\nREM burst {n}\n')
        viewer.flush_editor_text()

    history = viewer.edit_session.document.history
    assert history.undo_labels.count('Edit text') == 1
    assert viewer.undo() == 'Edit text'
    assert 'REM burst' not in viewer.edit_session.document.text


# ----------------------------------------------------------- 3-D -> text


def test_a_drag_regenerates_the_editor_text(viewer):
    viewer.render_widget.modelEdited.emit(scripted_split(loaded_path(viewer)))

    assert 'C4B' in labels(viewer)
    assert 'C4B' in viewer.editor.editor.toPlainText()
    assert viewer.editor.editor.toPlainText() == viewer.edit_session.document.text


def test_undo_after_a_text_edit_still_reaches_the_drag(viewer):
    """The document must survive the re-parse, history and all."""
    viewer.render_widget.modelEdited.emit(scripted_split(loaded_path(viewer)))
    assert 'C4B' in labels(viewer)

    type_into(viewer, viewer.editor.editor.toPlainText() + '\nREM after the drag\n')
    viewer.flush_editor_text()

    assert viewer.undo() == 'Edit text'
    assert 'C4B' in labels(viewer)
    assert viewer.undo() == 'Split C4 (4 atoms)'
    assert 'C4B' not in labels(viewer)


def test_a_drag_on_unparsable_text_is_refused(viewer, monkeypatch):
    edit = scripted_split(loaded_path(viewer))

    warnings = []
    monkeypatch.setattr(QtWidgets.QMessageBox, 'warning',
                        lambda *args, **kwargs: warnings.append(args))
    type_into(viewer, 'this is not a shelx file at all')
    viewer.render_widget.modelEdited.emit(edit)

    assert warnings
    assert 'C4B' not in labels(viewer)
    assert not viewer.edit_session.can_undo


def test_saving_writes_what_the_editor_shows(viewer, tmp_path):
    type_into(viewer, viewer.editor.editor.toPlainText() + '\nREM unapplied\n')
    written = viewer.save_model()
    assert 'REM unapplied' in written.read_text()


# ------------------------------------------------------- shared selection


def test_clicking_an_atom_scrolls_the_editor_to_it(viewer):
    document = viewer.edit_session.document
    viewer.render_widget.atomClicked.emit('C4')
    assert viewer.editor.editor.textCursor().blockNumber() == document.line_of_atom('C4')


def test_the_text_cursor_highlights_the_atom_in_3d(viewer):
    viewer.editor.atom_selected.emit('C4')
    assert viewer.render_widget.selected_atoms == {'C4'}

    viewer.editor.atom_selected.emit('N1')
    assert viewer.render_widget.selected_atoms == {'N1'}


def test_selection_sync_does_not_loop(viewer):
    """``jump_to_atom`` must not echo back as another selection."""
    seen = []
    viewer.editor.atom_selected.connect(seen.append)
    viewer.render_widget.atomClicked.emit('C4')
    assert seen == []


# --------------------------------------------------------------- layout


def test_the_editor_sits_in_a_resizable_splitter(viewer):
    from qtpy import QtCore

    splitter = viewer._splitter
    assert splitter.orientation() == QtCore.Qt.Orientation.Horizontal
    assert splitter.indexOf(viewer.render_widget) == 0
    assert splitter.indexOf(viewer.editor) == 1
    assert splitter.isCollapsible(1)
    assert not splitter.isCollapsible(0)


def test_the_editor_toolbar_is_out_of_the_pane(viewer):
    """Its buttons are wider than 84 columns and would force the pane open."""
    assert viewer.editor.toolbar.parent() is not viewer.editor
    assert viewer._splitter.indexOf(viewer.editor.toolbar) == -1
    assert viewer.editor.toolbar.parent() is viewer


def test_the_editor_pane_opens_84_columns_wide(viewer):
    viewer.resize(1400, 800)
    viewer.show()
    try:
        expected = viewer.editor.character_width(EDITOR_COLUMNS)
        assert viewer._splitter.sizes()[1] == expected
    finally:
        viewer.close()


def test_the_structure_gets_the_leftover_height(viewer):
    """The control rows are as tall as their widgets, the splitter takes the rest.

    A row of buttons with the default size policy grows to claim a share of
    the spare vertical space, which leaves the buttons floating in the middle
    of a white gap and the structure squeezed into the top half.
    """
    viewer.resize(1400, 800)
    viewer.show()
    try:
        toolbar = viewer.editor.toolbar
        assert toolbar.height() == toolbar.sizeHint().height()
        # Everything not taken by the three control rows.
        rows = sum(w.height() for w in (toolbar, viewer._open_file_button,
                                        viewer._bond_color_button))
        assert viewer._splitter.height() > viewer.height() - rows - 60
    finally:
        viewer.close()
