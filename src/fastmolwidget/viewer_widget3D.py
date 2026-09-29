"""Ready-to-use 3-D viewer widget."""

from __future__ import annotations

"""
TODO:
* Split-Rotate about selected bond
* While splitting, check for inverted moiety
* While dragging, adjust occupancy, so that residual density fits better. 
"""

from argparse import ArgumentParser
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets
from shelxfile.gui.editor_widget import ShelxEditorWidget

from fastmolwidget.density_controls import DensityControlsMixin
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.model_edit import CommitReport, ModelEditError
from fastmolwidget.molecule3D import MoleculeWidget3D
from fastmolwidget.part_combo import PartFilterWidget

#: How long the editor has to be idle before its text is applied to the
#: model, in milliseconds.  Long enough not to re-parse mid-word, short
#: enough that the 3-D view feels like it is following along.
LIVE_APPLY_DELAY_MS = 300

#: Width of the editor pane, in characters of its own font.  SHELXL writes
#: at most 80 columns, so this is that plus a little.
EDITOR_COLUMNS = 84


class MoleculeViewer3DWidget(DensityControlsMixin, QtWidgets.QWidget):
    """3-D :class:`MoleculeWidget3D` plus its control bar and a SHELX editor.

    A ``.res``/``.ins`` file is shown twice: as atoms in the 3-D view, and
    as text in a :class:`~shelxfile.gui.editor_widget.ShelxEditorWidget` in
    the right-hand pane of a resizable splitter.  **Both are views of one
    :class:`~shelxfile.edit.ShelxDocument`**, the one owned by this
    viewer's :class:`~fastmolwidget.model_edit.ModelEditSession`, so a drag
    in 3-D shows up in the text and an edit in the text shows up in 3-D.
    The 3-D side is refreshed by subscribing to that document, not by
    calling :meth:`MoleculeLoader.reload` after each kind of edit.

    Typed text reaches the model without pressing Apply: it is applied
    after :data:`LIVE_APPLY_DELAY_MS` of idleness, as one coalesced undo
    step per burst, and the text is deliberately *not* reformatted while
    that happens.  Text that does not parse leaves the last good model on
    screen and shows the editor's inline error.

    CIF and XYZ files cannot be represented as SHELX text, so the editor
    pane is unbound and disabled for them.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # ── molecule renderer ────────────────────────────────────────────────
        self._render_widget = MoleculeWidget3D()
        # MoleculeLoader only needs the open_molecule() API.
        self._loader = MoleculeLoader(self._render_widget)  # type: ignore[arg-type]

        # ── SHELX text editor ────────────────────────────────────────────────
        self._editor = ShelxEditorWidget()
        # Nothing is loaded yet, so there is no document to edit.
        self._editor.setEnabled(False)
        #: The document both views are bound to, or ``None``.
        self._bound_document = None
        #: Set while a document change is being turned into a redisplay, so
        #: the two paths into it cannot reload twice for one edit.
        self._reloading = False
        #: Whether the current edit already caused a redisplay.
        self._reloaded_for_edit = False
        self._sized_splitter = False

        self._apply_timer = QtCore.QTimer(self)
        self._apply_timer.setSingleShot(True)
        self._apply_timer.setInterval(LIVE_APPLY_DELAY_MS)
        self._apply_timer.timeout.connect(self._apply_editor_text)
        self._editor.editor.textChanged.connect(self._on_editor_text_changed)

        # ── control bar ──────────────────────────────────────────────────────
        self._grow_checkbox = QtWidgets.QCheckBox("Grow")
        self._pack_checkbox = QtWidgets.QCheckBox("Pack Unit Cell")
        self._adp_checkbox = QtWidgets.QCheckBox("Show ADP")
        self._label_checkbox = QtWidgets.QCheckBox("Show Labels")
        self._hydrogens_checkbox = QtWidgets.QCheckBox("Hide Hydrogens")

        self._bw_label = QtWidgets.QLabel("Bond Width:")
        self._bond_width_spinbox = QtWidgets.QSpinBox()
        self._bond_width_spinbox.setRange(0, 15)
        self._bond_width_spinbox.setValue(3)
        self._bond_color_button = QtWidgets.QPushButton("Bond Color…")
        self._reset_center_button = QtWidgets.QPushButton("Reset Rotation Center")
        self._best_view_button = QtWidgets.QPushButton("Best View")
        self._open_file_button = QtWidgets.QPushButton("Open File…")
        self._save_image_button = QtWidgets.QPushButton("Save Image…")
        self._undo_button = QtWidgets.QPushButton("Undo")
        self._redo_button = QtWidgets.QPushButton("Redo")
        self._save_model_button = QtWidgets.QPushButton("Save Model…")
        self._init_density_controls()

        # "Hide Hydrogens" unchecked -> visible by default.
        self._adp_checkbox.setChecked(True)
        self._hydrogens_checkbox.setChecked(False)

        # Wire controls to the renderer.
        self._adp_checkbox.toggled.connect(self._render_widget.show_adps)
        self._label_checkbox.toggled.connect(self._render_widget.show_labels)
        self._hydrogens_checkbox.toggled.connect(
            lambda checked: self._render_widget.show_hydrogens(not checked)
        )
        self._bond_width_spinbox.valueChanged.connect(self._render_widget.set_bond_width)
        self._bond_color_button.clicked.connect(self._choose_bond_color)
        self._reset_center_button.clicked.connect(self._render_widget.reset_rotation_center)
        self._best_view_button.clicked.connect(self._render_widget.align_best_view)
        self._open_file_button.clicked.connect(self._open_file_dialog)
        self._save_image_button.clicked.connect(self._save_image_dialog)
        self._grow_checkbox.toggled.connect(self._on_grow_toggled)
        self._pack_checkbox.toggled.connect(self._on_pack_toggled)

        # Model editing: every finished drag is written into the SHELX model.
        self._render_widget.modelEdited.connect(self._on_model_edited)
        self._undo_button.clicked.connect(self.undo)
        self._redo_button.clicked.connect(self.redo)
        self._save_model_button.clicked.connect(self._save_model_dialog)
        #: What the last committed edit reported (restraints, messages).
        self.last_commit_report: CommitReport | None = None
        self._shortcuts = []
        for keys, slot in (('Ctrl+Z', self._undo_shortcut),
                           ('Ctrl+Y', self._redo_shortcut),
                           ('Ctrl+Shift+Z', self._redo_shortcut)):
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(keys), self)
            # This is an embeddable widget: a window-wide binding would
            # collide with the host's own undo and disable both.
            shortcut.setContext(QtCore.Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)
        self._update_edit_controls()

        # Selection is shared: clicking an atom scrolls the editor to its
        # line, and moving the text cursor onto an atom highlights it in 3-D.
        # Neither direction echoes back (jump_to_atom suppresses
        # atom_selected, select_atoms emits nothing).
        self._render_widget.atomClicked.connect(self._editor.jump_to_atom)
        self._editor.atom_selected.connect(self._on_editor_atom_selected)
        # Connected after the editor's own handler, so it runs once the
        # refinement is over: the residual-density map was computed from
        # the pre-refinement Fc and means nothing afterwards.
        self._editor.refine_button.clicked.connect(self._on_refine_clicked)

        # Apply initial defaults.
        self._render_widget.set_bond_width(3)
        self._render_widget.show_labels(False)

        # ── Part filter ───────────────────────────────────────────────────────
        self._part_widget = PartFilterWidget()
        self._part_widget.selectionChanged.connect(self._apply_part_filter)

        # Also handles programmatic open_molecule() calls.
        self._render_widget.partsChanged.connect(self._update_part_controls)

        # ── layout ───────────────────────────────────────────────────────────
        # The 3-D view and the editor text side by side; every control row
        # underneath, spanning both.  The editor's toolbar is one of those
        # rows rather than part of the editor pane: a row of buttons is far
        # wider than 80 columns, and inside the splitter its minimum width
        # would stop the pane from ever being as narrow as a SHELX file.
        self._splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._render_widget)
        self._splitter.addWidget(self._editor)
        self._splitter.setCollapsible(0, False)
        self._splitter.setCollapsible(1, True)
        # Resizing the window grows the 3-D view; the editor keeps its width.
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)

        # Row 1: structure toggles.
        control_bar = QtWidgets.QHBoxLayout()
        control_bar.addWidget(self._open_file_button)
        control_bar.addWidget(self._undo_button)
        control_bar.addWidget(self._redo_button)
        control_bar.addWidget(self._grow_checkbox)
        control_bar.addWidget(self._pack_checkbox)
        control_bar.addWidget(self._adp_checkbox)
        control_bar.addWidget(self._label_checkbox)
        control_bar.addWidget(self._hydrogens_checkbox)
        control_bar.addStretch()

        # Row 2: bond and view controls.
        control_bar2 = QtWidgets.QHBoxLayout()
        control_bar2.addWidget(self._bw_label)
        control_bar2.addWidget(self._bond_width_spinbox)
        control_bar2.addWidget(self._bond_color_button)
        control_bar2.addWidget(self._reset_center_button)
        control_bar2.addWidget(self._best_view_button)
        control_bar2.addWidget(self._save_image_button)
        control_bar2.addWidget(self._save_model_button)
        control_bar2.addWidget(self._residual_density_button)
        control_bar2.addWidget(self._density_level_label)
        control_bar2.addWidget(self._density_level_spinbox)
        control_bar2.addWidget(self._part_widget)
        control_bar2.addStretch()

        vl = QtWidgets.QVBoxLayout(self)
        # The splitter takes every bit of leftover height; the control rows
        # are only as tall as their widgets.
        vl.addWidget(self._splitter, 1)
        vl.addLayout(control_bar)
        vl.addLayout(control_bar2)
        # Row 3: the SHELX editor's own actions.
        vl.addWidget(self._editor.toolbar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def render_widget(self) -> MoleculeWidget3D:
        """Underlying :class:`MoleculeWidget3D`."""
        return self._render_widget

    @property
    def editor(self) -> ShelxEditorWidget:
        """The SHELX text editor in the right-hand pane.

        Bound to the same document as the 3-D view for ``.res``/``.ins``
        files, and unbound (and disabled) for everything else.  Its
        toolbar lives in this widget's control bar, not in the pane.
        """
        return self._editor

    def load_file(self, filename: str | Path) -> None:
        """Load and display a structure file.  Unsaved model edits are lost."""
        self._apply_timer.stop()
        self._loader.load_file(filename)
        self.setWindowTitle(str(Path(filename).resolve()))
        self._bind_editor()
        # Keep the controls in sync if a new model cleared density.
        self._sync_density_controls()
        self._update_edit_controls()

    # ------------------------------------------------------------------
    # The shared document
    # ------------------------------------------------------------------

    def _bind_editor(self) -> None:
        """Point the editor at the document the 3-D view is showing.

        Also what subscribes this widget to that document, so every later
        change -- from either side -- redisplays the structure.
        """
        session = self._loader.edit_session
        document = None if session is None else session.document
        if document is self._bound_document:
            return
        if self._bound_document is not None:
            self._bound_document.unsubscribe(self._on_document_changed)
        self._bound_document = document
        if document is None:
            self._editor.clear()
            self._editor.setEnabled(False)
            return
        self._editor.setEnabled(True)
        self._editor.set_shelxfile(document)
        document.subscribe(self._on_document_changed)

    def _on_document_changed(self, _document) -> None:
        """Redisplay the structure after anything changed the model."""
        self._reloaded_for_edit = True
        self._reload_display()

    def _reload_display(self) -> None:
        """Rebuild the 3-D view from the current model, once."""
        if self._reloading:
            return
        self._reloading = True
        try:
            self._loader.reload()
        finally:
            self._reloading = False
        self._update_edit_controls()

    # ------------------------------------------------------------------
    # Live text editing
    # ------------------------------------------------------------------

    def _on_editor_text_changed(self) -> None:
        if self._bound_document is not None:
            self._apply_timer.start()

    def _apply_editor_text(self) -> bool:
        """Apply whatever is in the editor right now.

        The text is applied without being reformatted -- the user may
        still be typing in it -- and as a coalescing undo step, so a burst
        of edits does not bury the drags and splits underneath it.
        """
        self._apply_timer.stop()
        if self._bound_document is None or not self._editor.is_dirty:
            return True
        return self._editor.apply(normalise=False, coalesce=True)

    def flush_editor_text(self) -> bool:
        """Apply any pending text edit now, ahead of something that needs it.

        :returns: ``False`` when the text does not parse, in which case the
            model is unchanged and the editor shows why.
        """
        return self._apply_editor_text()

    def _undo_shortcut(self) -> None:
        """Ctrl+Z means text undo in the editor, model undo everywhere else."""
        if self._editor.editor.hasFocus():
            self._editor.editor.undo()
        else:
            self.undo()

    def _redo_shortcut(self) -> None:
        if self._editor.editor.hasFocus():
            self._editor.editor.redo()
        else:
            self.redo()

    def _on_editor_atom_selected(self, name: str) -> None:
        self._render_widget.select_atoms({name})

    def _on_refine_clicked(self) -> None:
        """A refinement replaced the coordinates, so the density map is stale.

        It was computed from the pre-refinement ``Fc``, and the model's
        path has not changed, so nothing else would drop it.
        """
        if self._render_widget.residual_density_map is not None:
            self.clear_residual_density()
            self._sync_density_controls()

    # ------------------------------------------------------------------
    # Model editing
    # ------------------------------------------------------------------

    @property
    def edit_session(self):
        """The editable SHELX model (:class:`~fastmolwidget.model_edit.ModelEditSession`),
        or ``None`` when the loaded file is not a ``.res``/``.ins``."""
        return self._loader.edit_session

    def undo(self) -> str | None:
        """Undo the last model edit; returns its label, or ``None``."""
        return self._step_history(redo=False)

    def redo(self) -> str | None:
        """Redo the last undone model edit; returns its label, or ``None``."""
        return self._step_history(redo=True)

    def save_model(self, path: str | Path | None = None) -> Path:
        """Write the edited model (default ``<basename>.ins``).

        An existing file is backed up to ``<name>.bak`` first.

        :raises RuntimeError: when no SHELX model is loaded.
        """
        session = self._loader.edit_session
        if session is None:
            raise RuntimeError('Only SHELX .res/.ins models can be saved')
        # What is on screen in the editor is what the user means to save.
        self._apply_editor_text()
        written = session.save(path)
        self._update_edit_controls()
        return written

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        """Give the editor pane its 84-column width, the first time only."""
        super().showEvent(event)
        if self._sized_splitter:
            return
        self._sized_splitter = True
        editor_width = self._editor.character_width(EDITOR_COLUMNS)
        total = max(self._splitter.width(), editor_width + 200)
        self._splitter.setSizes([total - editor_width, editor_width])

    def _step_history(self, *, redo: bool) -> str | None:
        session = self._loader.edit_session
        if session is None or self._render_widget.drag_in_progress:
            return None
        self._apply_timer.stop()
        self._reloaded_for_edit = False
        label = session.redo() if redo else session.undo()
        if label is not None and not self._reloaded_for_edit:
            self._reload_display()
        self._update_edit_controls()
        return label

    def _on_model_edited(self, edit) -> None:
        """Write a finished drag into the SHELX model and redisplay it."""
        session = self._loader.edit_session
        if session is None:
            return  # not an editable format: the drag stays visual only
        from fastmolwidget.atoms import HYDROGEN_ELEMENTS

        # A pending text edit has to reach the model before the drag does,
        # or the drag is committed against a model the user has already
        # moved on from.
        if not self._apply_editor_text():
            self.last_commit_report = None
            QtWidgets.QMessageBox.warning(
                self, 'Cannot change the model',
                'The text in the editor does not parse, so the drag cannot '
                'be written to the model.\n\nThe drag was discarded.')
            self._reload_display()
            return

        hydrogens = [i for i, atom in enumerate(self._render_widget.atoms)
                     if atom.type_ in HYDROGEN_ELEMENTS]
        self._reloaded_for_edit = False
        try:
            self.last_commit_report = session.commit(
                edit, self._loader.atom_sources, hydrogens=hydrogens)
        except ModelEditError as error:
            self.last_commit_report = None
            QtWidgets.QMessageBox.warning(
                self, 'Cannot change the model',
                f'{error}\n\nThe drag was discarded.')
        finally:
            # The model was rolled back, so the display has to follow it
            # even when the commit failed and left the model untouched.
            if not self._reloaded_for_edit:
                self._reload_display()
            self._update_edit_controls()

    def _update_edit_controls(self) -> None:
        session = self._loader.edit_session
        can_undo = session is not None and session.can_undo
        can_redo = session is not None and session.can_redo
        self._undo_button.setEnabled(can_undo)
        self._redo_button.setEnabled(can_redo)
        self._undo_button.setToolTip(
            f'Undo {session.undo_label} (Ctrl+Z)' if can_undo else 'Nothing to undo')
        self._redo_button.setToolTip(
            f'Redo {session.redo_label} (Ctrl+Y)' if can_redo else 'Nothing to redo')
        self._save_model_button.setEnabled(session is not None)
        self._save_model_button.setToolTip(
            'Write the edited model as a SHELX .ins file' if session is not None
            else 'Only SHELX .res/.ins models can be edited and saved')

    def _save_model_dialog(self) -> None:
        session = self._loader.edit_session
        if session is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Model', str(session.default_save_path()),
            'SHELX Instruction File (*.ins);;SHELX Result File (*.res);;All Files (*)',
        )
        if path:
            self.save_model(path)

    def _confirm_discard_edits(self) -> bool:
        """Ask before unsaved model edits are thrown away."""
        session = self._loader.edit_session
        if session is None:
            return True
        if not (session.is_modified or self._editor.is_dirty):
            return True
        answer = QtWidgets.QMessageBox.question(
            self, 'Unsaved model changes',
            'The model has unsaved changes. Discard them?',
            QtWidgets.QMessageBox.StandardButton.Discard
            | QtWidgets.QMessageBox.StandardButton.Cancel,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Discard

    def grow(self) -> None:
        """Grow the current structure to full molecules."""
        if self._pack_checkbox.isChecked():
            self._pack_checkbox.blockSignals(True)
            self._pack_checkbox.setChecked(False)
            self._pack_checkbox.blockSignals(False)
            self._loader.set_pack(False)
        self._grow_checkbox.blockSignals(True)
        self._grow_checkbox.setChecked(True)
        self._grow_checkbox.blockSignals(False)
        self._loader.set_grow(True)

    def set_bond_color(
        self,
        color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set the default color for non-selected 3-D bonds."""
        self._render_widget.set_bond_color(color)

    def show_residual_density(self, hkl_path: str | Path | None = None,
                              level: float | None = None) -> None:
        """Show a residual-density map."""
        super().show_residual_density(hkl_path, level)

    def _on_grow_toggled(self, checked: bool) -> None:
        """Activate grow mode; deactivate pack mode when grow is switched on."""
        if checked and self._pack_checkbox.isChecked():
            self._pack_checkbox.blockSignals(True)
            self._pack_checkbox.setChecked(False)
            self._pack_checkbox.blockSignals(False)
            self._loader.set_pack(False)
        self._loader.set_grow(checked)

    def _on_pack_toggled(self, checked: bool) -> None:
        """Activate pack mode; deactivate grow mode when pack is switched on."""
        if checked and self._grow_checkbox.isChecked():
            self._grow_checkbox.blockSignals(True)
            self._grow_checkbox.setChecked(False)
            self._grow_checkbox.blockSignals(False)
            self._loader.set_grow(False)
        self._loader.set_pack(checked)
        if checked:
            self._render_widget.reset_rotation_center()
            self._render_widget._align_to_reciprocal_axis(1)

    def _update_part_controls(self, parts: frozenset[int]) -> None:
        """Refresh the Part filter after a load."""
        self._part_widget.update_parts(parts)
        # None means all parts.
        if len(parts) > 1:
            self._render_widget.set_visible_parts(None)

    def _apply_part_filter(self) -> None:
        """Apply the current Part filter."""
        checked = set(self._part_widget.checked_values())
        # None means all parts.
        if checked == self._render_widget.available_parts:
            self._render_widget.set_visible_parts(None)
        else:
            self._render_widget.set_visible_parts(checked)

    def _choose_bond_color(self) -> None:
        """Open a color picker for bonds."""
        current = QtGui.QColor.fromRgbF(*self._render_widget._bond_rgb)
        color = QtWidgets.QColorDialog.getColor(current, self, "Choose Bond Color")
        if color.isValid():
            self._render_widget.set_bond_color(color)

    def _open_file_dialog(self) -> None:
        """Open a file dialog and load the chosen structure."""
        if not self._confirm_discard_edits():
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Structure File",
            "",
            "Structure Files (*.cif *.res *.ins *.xyz);;All Files (*)",
        )
        if path:
            self.load_file(path)

    def _save_image_dialog(self) -> None:
        """Open a file dialog and save the current view."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self._render_widget.save_image(Path(path))


if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    parse = ArgumentParser(description="Test the 3-D molecule viewer widget with a sample CIF file.")
    parse.add_argument("cif_file", nargs="?", default=None, help="Path to a CIF file to load (optional).")
    args = parse.parse_args()

    w = MoleculeViewer3DWidget()
    # Path is relative to the repository root.
    # w.load_file(Path("tests/test-data/p31c.cif"))
    # w.load_file(r"D:\frames\CK-B874-finalcif.cif")
    # A good file to test disoder dragging:
    # w.load_file('tests/test-data/1000007.cif')
    # w.load_file('tests/test-data/p21c.cif')
    # w.load_file('tests/test-data/1548072_many_atoms.cif')
    # w.load_file(Path('tests/test-data/4060314.cif'))
    # w.load_file(Path('tests/test-data/1979688_small.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1367_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1368_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1369_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1370_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1371_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1372_ESM.cif'))
    # w.load_file(Path('tests/test-data/IKmjs421_2_0m_sump.res'))
    # w.load_file("tests\\test-data\\nospera2.cif")

    w.load_file(r"tests/test-data/BB_LJ45_a.res")
    w.show_residual_density(r"tests/test-data/BB_LJ45_a.hkl")

    # w.load_file(r"tests/test-data/SW-7-Ti_a-finalcif.res")
    # w.show_residual_density(r"tests/test-data/SW-7-Ti_a-finalcif.hkl")

    if args.cif_file:
        w.load_file(args.cif_file)
    w.show()
    w.grow()
    # app.processEvents()
    w.showMaximized()
    # w.render_widget.set_visible_parts({0, 1})
    app.exec()
