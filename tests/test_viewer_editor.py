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
from qtpy import QtCore, QtWidgets

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
    assert splitter.indexOf(viewer._editor_pane) == 1
    assert viewer.editor.parent() is viewer._editor_pane
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


# ------------------------------------------------------------ SHELXL log


def test_the_refine_log_sits_under_the_editor_collapsed(viewer):
    panel = viewer._log_panel
    assert panel.parent() is viewer._editor_pane
    layout = viewer._editor_pane.layout()
    assert layout.indexOf(viewer.editor) < layout.indexOf(panel)
    assert not panel.is_expanded
    assert panel.tabs.isHidden()


def test_clicking_the_log_header_opens_and_closes_it(viewer):
    panel = viewer._log_panel
    panel.toggle_button.click()
    assert panel.is_expanded and not panel.tabs.isHidden()
    panel.toggle_button.click()
    assert not panel.is_expanded and panel.tabs.isHidden()


def test_the_log_opens_during_a_refinement_and_closes_after(viewer):
    panel = viewer._log_panel
    viewer.editor.refine_started.emit()
    assert panel.is_expanded
    viewer.editor.refine_output.emit(' R1 =  0.0300 for    1234 Fo > 4sig(Fo)\n')
    assert 'R1 =  0.0300' in panel.log.toPlainText()
    viewer.editor.refine_finished.emit(True, '')
    assert not panel.is_expanded
    # The output stays readable after the fact.
    panel.toggle_button.click()
    assert 'R1 =  0.0300' in panel.log.toPlainText()


def test_a_failed_refinement_keeps_the_log_open(viewer):
    panel = viewer._log_panel
    viewer.editor.refine_started.emit()
    viewer.editor.refine_finished.emit(False, 'SHELXL refinement failed.')
    assert panel.is_expanded
    assert 'SHELXL refinement failed.' in panel.log.toPlainText()


def test_a_new_refinement_clears_the_old_log(viewer):
    panel = viewer._log_panel
    viewer.editor.refine_started.emit()
    viewer.editor.refine_output.emit('old run\n')
    viewer.editor.refine_finished.emit(True, '')
    viewer.editor.refine_started.emit()
    assert 'old run' not in panel.log.toPlainText()


def show_list_tab(widget: MoleculeViewer3DWidget) -> None:
    from fastmolwidget.refine_log import LIST_TAB

    widget._log_panel.set_expanded(True)
    widget._log_panel.tabs.setCurrentIndex(LIST_TAB)


def test_the_list_file_is_the_models_lst(viewer):
    assert viewer._log_panel.list_file == loaded_path(viewer).with_suffix('.lst')


def test_the_list_file_is_read_only_when_its_tab_is_shown(tmp_path):
    source = tmp_path / 'model.res'
    source.write_text(P31C.read_text())
    source.with_suffix('.lst').write_text(' R1 =  0.0312 for 1000 Fo > 4sig(Fo)\n')
    widget = MoleculeViewer3DWidget()
    widget.load_file(source)
    panel = widget._log_panel
    assert panel.list_view.toPlainText() == ''
    show_list_tab(widget)
    assert 'R1 =  0.0312' in panel.list_view.toPlainText()


def test_a_missing_list_file_says_so(viewer):
    show_list_tab(viewer)
    assert viewer._log_panel.list_view.toPlainText() == ''
    assert 'does not exist yet' in viewer._log_panel.list_view.placeholderText()


def test_the_list_file_is_reread_when_it_changed(viewer):
    lst = loaded_path(viewer).with_suffix('.lst')
    lst.write_text('first run\n')
    show_list_tab(viewer)
    assert viewer._log_panel.list_view.toPlainText() == 'first run\n'
    lst.write_text('second, longer run\n')
    viewer.editor.refine_finished.emit(False, 'failed')
    viewer._log_panel.tabs.setCurrentIndex(1)
    assert 'second, longer run' in viewer._log_panel.list_view.toPlainText()


def test_a_cif_has_no_list_file():
    widget = MoleculeViewer3DWidget()
    widget.load_file('tests/test-data/p21c.cif')
    assert widget._log_panel.list_file is None
    show_list_tab(widget)
    assert 'No SHELX model' in widget._log_panel.list_view.placeholderText()


def test_the_open_log_takes_a_quarter_of_the_pane(viewer):
    viewer.resize(1400, 800)
    viewer.show()
    try:
        panel = viewer._log_panel
        pane = viewer._editor_pane
        collapsed = panel.height()
        assert collapsed == panel.sizeHint().height()
        panel.set_expanded(True)
        app.processEvents()
        assert panel.height() == pytest.approx(pane.height() / 4, abs=2)
    finally:
        viewer.close()


@pytest.fixture
def refining_viewer(tmp_path):
    from shelxfile.refine.refine import find_shelxl_exe

    if not find_shelxl_exe():
        pytest.skip('SHELXL not found')
    source = tmp_path / 'p31c.res'
    source.write_text(P31C.read_text())
    (tmp_path / 'p31c.hkl').write_bytes(P31C.with_suffix('.hkl').read_bytes())
    widget = MoleculeViewer3DWidget()
    widget.load_file(source)
    return widget


def wait_for_refinement(widget: MoleculeViewer3DWidget, timeout_s: float = 60) -> list[tuple]:
    """Spin the event loop until the running refinement has finished."""
    import time

    finished: list[tuple] = []
    widget.editor.refine_finished.connect(lambda ok, msg: finished.append((ok, msg)))
    deadline = time.monotonic() + timeout_s
    while not finished and time.monotonic() < deadline:
        app.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
    assert finished, 'SHELXL did not finish in time'
    return finished


def test_a_real_refinement_streams_into_the_log(refining_viewer):
    widget = refining_viewer
    shown_while_running: list[bool] = []
    widget.editor.refine_output.connect(
        lambda _line: shown_while_running.append(widget._log_panel.is_expanded))
    assert widget.editor.refine(cycles=1) is True
    assert wait_for_refinement(widget) == [(True, '')]
    assert shown_while_running and all(shown_while_running)
    assert 'R1' in widget._log_panel.log.toPlainText()
    assert not widget._log_panel.is_expanded
    assert 'L.S. 1' in widget.editor.editor.toPlainText()
    # The list file SHELXL just wrote is one click away.
    show_list_tab(widget)
    assert 'SHELXL' in widget._log_panel.list_view.toPlainText()
    # The summary has the final values, including the .lst-only WGHT.
    summary = widget._summary_widget
    assert summary.title == 'Refinement: finished'
    assert summary.value('cycle') == '1 / 1'
    assert summary.value('r1_obs') != '–'
    assert summary.value('wght') != '–'


def test_the_refinement_does_not_block_and_locks_editing(refining_viewer):
    widget = refining_viewer
    before = labels(widget)
    assert widget.editor.refine(cycles=1) is True
    # refine() came back while SHELXL is still running.
    assert widget.editor.is_refining
    assert widget._log_panel.is_expanded
    assert not widget._undo_button.isEnabled()
    assert not widget._save_model_button.isEnabled()
    assert widget.undo() is None
    assert labels(widget) == before
    wait_for_refinement(widget)
    assert not widget.editor.is_refining
    assert widget._save_model_button.isEnabled()
    assert widget._undo_button.isEnabled()  # the refinement is one undo step
    assert widget.undo() == 'Refine (SHELXL)'


# ----------------------------------------------------- refinement summary


def test_a_model_without_a_list_file_has_an_empty_summary(viewer):
    summary = viewer._summary_widget
    assert not summary.isHidden()
    assert summary.title == 'Refinement: no refinement yet'
    assert summary.value('r1_obs') == '–'


def test_a_never_refined_model_still_shows_its_space_group(viewer):
    # LATT -1 + SYMM: P31c, although the TITL calls it P-31c.
    assert viewer._summary_widget.space_group == 'Space group: P31c'


def test_the_space_group_follows_edited_symmetry(viewer):
    text = viewer.editor.editor.toPlainText().replace('LATT -1', 'LATT 1')
    type_into(viewer, text)
    assert viewer.flush_editor_text()
    assert viewer._summary_widget.space_group == 'Space group: P-31c'


def test_a_cif_has_no_space_group_in_the_summary():
    widget = MoleculeViewer3DWidget()
    widget.load_file('tests/test-data/p21c.cif')
    assert widget._summary_widget.space_group == 'Space group: –'


def test_the_summary_is_read_from_the_list_file(tmp_path):
    from test_shelxl_summary import LIST_TAIL

    source = tmp_path / 'model.res'
    source.write_text(P31C.read_text())
    source.with_suffix('.lst').write_text(LIST_TAIL)
    widget = MoleculeViewer3DWidget()
    widget.load_file(source)
    summary = widget._summary_widget
    assert summary.title == 'Refinement: from model.lst'
    assert summary.value('r1_obs') == '0.0308'
    assert summary.value('wght') == '0.0348  0.6275'
    assert summary.value('peak') == '0.22 (C5)'
    assert summary.value('counts') == '5352 / 357 / 287'
    assert summary.value('flack') == '-0.018(65) / 0.007(31)'


def test_a_cif_has_no_summary():
    widget = MoleculeViewer3DWidget()
    widget.load_file('tests/test-data/p21c.cif')
    assert widget._summary_widget.isHidden()


def test_the_summary_follows_the_console_while_refining(viewer):
    from test_shelxl_summary import CONSOLE

    viewer.editor.refine_started.emit()
    summary = viewer._summary_widget
    assert summary.title == 'Refinement: running…'
    lines = CONSOLE.splitlines(keepends=True)
    for line in lines[:5]:
        viewer.editor.refine_output.emit(line)
    assert summary.value('cycle') == '1 / 10'  # L.S. 10 in the model
    assert summary.value('shift') == '0.013 (mean 0.002)'
    for line in lines[5:]:
        viewer.editor.refine_output.emit(line)
    viewer.editor.refine_finished.emit(False, 'failed')
    assert summary.title == 'Refinement: failed'
    assert summary.value('r1_all') == '0.0343'


def test_the_summary_sits_under_the_editor_pane_as_wide_as_it(viewer):
    viewer.resize(2400, 900)
    viewer.show()
    try:
        app.processEvents()
        summary = viewer._summary_widget
        pane = viewer._editor_pane
        assert summary.width() == pane.width()
        assert summary.mapTo(viewer, QtCore.QPoint(0, 0)).x() == pane.mapTo(viewer, QtCore.QPoint(0, 0)).x()
        assert summary.mapTo(viewer, QtCore.QPoint(0, 0)).y() > pane.mapTo(viewer, QtCore.QPoint(0, pane.height())).y()
        viewer._splitter.setSizes([1700, 600])
        app.processEvents()
        assert summary.width() == pane.width() == 600
    finally:
        viewer.close()


def test_the_summary_never_makes_the_window_wider(viewer):
    """With no room beside the control rows it narrows instead."""
    with_summary = viewer.minimumSizeHint().width()
    viewer._summary_widget.hide()
    assert with_summary <= viewer.minimumSizeHint().width() + 10