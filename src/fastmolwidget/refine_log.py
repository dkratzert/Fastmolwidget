"""Collapsible panel under the SHELX editor: SHELXL's console output and its list file."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

from fastmolwidget.shelxl_summary import Peak, RefinementSummary

#: Header text of the panel.
TITLE = 'SHELXL output'

#: Tab indices.
OUTPUT_TAB = 0
LIST_TAB = 1


def _text_view() -> QtWidgets.QPlainTextEdit:
    view = QtWidgets.QPlainTextEdit()
    view.setReadOnly(True)
    view.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
    view.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
    return view


class RefineLogPanel(QtWidgets.QWidget):
    """A clickable header over two read-only tabs: *Output* and *List file*.

    *Output* is SHELXL's console output: :meth:`start` clears it and opens the
    panel, :meth:`append` adds SHELXL's lines as they arrive and :meth:`finish`
    collapses the panel again after a successful refinement.  A failed
    refinement leaves it open, with the reason appended, because that output
    is exactly what has to be read.

    *List file* shows the ``.lst`` file set with :meth:`set_list_file`.  It
    is read only when that tab is shown, and read again only if the file
    changed since -- a list file can run to megabytes.

    Clicking the header opens or closes the panel at any time.  Collapsed,
    only the header is shown and the panel does not grow; expanded it takes
    whatever share of the height its layout's stretch gives it.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._toggle = QtWidgets.QToolButton()
        self._toggle.setText(TITLE)
        self._toggle.setCheckable(True)
        self._toggle.setAutoRaise(True)
        self._toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setToolTip('Show or hide the output and list file of SHELXL')
        self._toggle.toggled.connect(self.set_expanded)

        self._log = _text_view()
        self._list = _text_view()
        #: The list file shown in the *List file* tab, or ``None``.
        self._list_path: Path | None = None
        #: ``(mtime_ns, size)`` of the list file as last read, or ``None``.
        self._list_stamp: tuple[int, int] | None = None

        self._tabs = QtWidgets.QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.addTab(self._log, 'Output')
        self._tabs.addTab(self._list, 'List file')
        self._tabs.currentChanged.connect(self._on_tab_changed)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._toggle)
        layout.addWidget(self._tabs)
        self.set_list_file(None)
        self.set_expanded(False)

    @property
    def toggle_button(self) -> QtWidgets.QToolButton:
        """The header; clicking it opens or closes the panel."""
        return self._toggle

    @property
    def tabs(self) -> QtWidgets.QTabWidget:
        """The *Output* and *List file* tabs."""
        return self._tabs

    @property
    def log(self) -> QtWidgets.QPlainTextEdit:
        """The text area holding SHELXL's console output."""
        return self._log

    @property
    def list_view(self) -> QtWidgets.QPlainTextEdit:
        """The text area showing the list file."""
        return self._list

    @property
    def list_file(self) -> Path | None:
        """The ``.lst`` file shown in the *List file* tab."""
        return self._list_path

    @property
    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        """Show (``True``) or hide the tabs below the header."""
        if self._toggle.isChecked() != expanded:
            # Re-enters through toggled.
            self._toggle.setChecked(expanded)
            return
        self._toggle.setArrowType(QtCore.Qt.ArrowType.DownArrow if expanded
                                  else QtCore.Qt.ArrowType.RightArrow)
        self._tabs.setVisible(expanded)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Expanding if expanded
                           else QtWidgets.QSizePolicy.Policy.Fixed)
        if expanded:
            self._on_tab_changed(self._tabs.currentIndex())

    # ------------------------------------------------------------ output

    def start(self) -> None:
        """A refinement begins: clear the output and show it."""
        self._log.clear()
        self._toggle.setText(f'{TITLE} (running…)')
        self._tabs.setCurrentIndex(OUTPUT_TAB)
        self.set_expanded(True)

    def append(self, line: str) -> None:
        """Add one line of output and keep the end of the log in view."""
        self._log.appendPlainText(line.rstrip('\r\n'))
        bar = self._log.verticalScrollBar()
        bar.setValue(bar.maximum())

    def finish(self, success: bool, message: str = '') -> None:
        """The refinement is over: collapse on success, explain a failure."""
        self._toggle.setText(TITLE)
        if success:
            self.set_expanded(False)
            return
        if message:
            self.append(f'*** {message} ***')
        self.set_expanded(True)

    # --------------------------------------------------------- list file

    def set_list_file(self, path: str | Path | None) -> None:
        """Show *path* in the *List file* tab, re-reading it even if unchanged.

        It is read now if that tab is on screen, otherwise when it is next
        shown.  A file that does not exist yet is fine: SHELXL writes it.
        """
        self._list_path = None if path is None else Path(path)
        self._list_stamp = None
        self._tabs.setTabToolTip(LIST_TAB, '' if self._list_path is None else str(self._list_path))
        if self._list_shown():
            self.reload_list_file()

    def reload_list_file(self) -> None:
        """Read the list file into its tab, unless it is unchanged since last time."""
        path = self._list_path
        if path is None:
            self._show_list_text('', 'No SHELX model is loaded.')
            return
        try:
            stat = path.stat()
        except OSError:
            self._show_list_text('', f'{path.name} does not exist yet; '
                                     'SHELXL writes it when it refines.')
            return
        stamp = (stat.st_mtime_ns, stat.st_size)
        if stamp == self._list_stamp:
            return
        try:
            # SHELXL writes plain ASCII; latin-1 cannot fail on stray bytes.
            text = path.read_text(encoding='latin-1')
        except OSError as exc:
            self._show_list_text('', f'Cannot read {path.name}: {exc}')
            return
        self._show_list_text(text, '')
        self._list_stamp = stamp

    def _show_list_text(self, text: str, placeholder: str) -> None:
        self._list_stamp = None
        self._list.setPlainText(text)
        self._list.setPlaceholderText(placeholder)

    def _list_shown(self) -> bool:
        return self.is_expanded and self._tabs.currentIndex() == LIST_TAB

    def _on_tab_changed(self, index: int) -> None:
        if index == LIST_TAB and self.is_expanded:
            self.reload_list_file()


#: ``(key, label, row, column)`` of every value in :class:`RefinementSummaryWidget`.
SUMMARY_FIELDS = (
    ('cycle', 'Cycle', 0, 0),
    ('r1_obs', 'R1 [Fo>4σ]', 0, 1),
    ('goof', 'GooF', 0, 2),
    ('counts', 'Data/restr./param.', 1, 0),
    ('r1_all', 'R1 (all data)', 1, 1),
    ('restrained_goof', 'Restr. GooF', 1, 2),
    ('shift', 'Shift/esd', 2, 0),
    ('wr2', 'wR2', 2, 1),
    ('flack', 'Flack x / z', 2, 2),
    ('peak', 'Peak [e/Å³]', 3, 0),
    ('hole', 'Hole [e/Å³]', 3, 1),
    ('wght', 'WGHT', 3, 2),
)

#: Shown for a value that is not known (yet).
MISSING = '–'


def _format_summary(summary: RefinementSummary) -> dict[str, tuple[str, str]]:
    """``key -> (text, tooltip)`` for every field of :data:`SUMMARY_FIELDS`."""
    s = summary

    def number(value: float | None, digits: int) -> str:
        return MISSING if value is None else f'{value:.{digits}f}'

    def peak(p: Peak | None, what: str) -> tuple[str, str]:
        if p is None:
            return MISSING, ''
        text = f'{p.height:.2f}'
        tooltip = f'{what}: {p.height:.2f} e/Å³ at {p.position}'
        if p.atom is not None:
            text += f' ({p.atom})'
            tooltip += f', {p.distance:.2f} Å from {p.atom}'
        return text, tooltip

    if s.cycle is None:
        cycle = MISSING
    else:
        cycle = f'{s.cycle} / {s.cycles}' if s.cycles else str(s.cycle)
    counts = ' / '.join(MISSING if n is None else str(n)
                        for n in (s.data, s.restraints, s.parameters))
    ratio = ''
    if s.data and s.parameters:
        ratio = f'Data/parameter ratio {s.data / s.parameters:.1f}'
    if s.max_shift is None:
        shift, shift_tip = MISSING, ''
    else:
        shift = f'{s.max_shift:.3f} (mean {s.mean_shift:.3f})'
        shift_tip = f'Largest shift/esd, for {s.max_shift_parameter}'
    # Both in one field: x by classical fit, z from Parsons' quotients.
    if s.flack is None and s.flack_parsons is None:
        flack = MISSING
    else:
        flack = f'{s.flack or MISSING} / {s.flack_parsons or MISSING}'
    flack_tip = '\n'.join(text for text in (
        s.flack and f'Flack x = {s.flack} by classical fit to all intensities',
        s.flack_parsons and f"Parsons z = {s.flack_parsons} from selected quotients") if text)
    wght = MISSING if s.wght is None else '  '.join(f'{w:g}' for w in s.wght)
    return {
        'cycle': (cycle, 'Least-squares cycle, of those requested by L.S./CGLS'),
        'r1_obs': (number(s.r1_obs, 4),
                   '' if s.n_obs is None else f'For {s.n_obs} reflections with Fo > 4σ(Fo)'),
        'r1_all': (number(s.r1_all, 4), '' if s.n_all is None else f'For all {s.n_all} data'),
        'wr2': (number(s.wr2, 4), 'wR2 for all data'),
        'goof': (number(s.goof, 3), 'Goodness of fit S'),
        'restrained_goof': (number(s.restrained_goof, 3), 'Restrained goodness of fit'),
        'counts': (counts, ratio),
        'shift': (shift, shift_tip),
        'flack': (flack, flack_tip),
        'peak': peak(s.peak, 'Highest difference peak'),
        'hole': peak(s.hole, 'Deepest difference hole'),
        'wght': (wght, 'Weighting scheme SHELXL recommends for the next refinement'),
    }


class RefinementSummaryWidget(QtWidgets.QWidget):
    """The headline numbers of the current or last SHELXL refinement.

    A title saying where the numbers come from, over a grid of
    :data:`SUMMARY_FIELDS`.  Its preferred width is set from outside with
    :meth:`set_target_width`, so it can line up with another widget.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._target_width: int | None = None
        self._title = QtWidgets.QLabel()
        font = self._title.font()
        font.setBold(True)
        self._title.setFont(font)

        grid = QtWidgets.QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(1)
        self._values: dict[str, QtWidgets.QLabel] = {}
        muted = self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText).name()
        for key, label, row, column in SUMMARY_FIELDS:
            name = QtWidgets.QLabel(f'{label}:')
            name.setStyleSheet(f'color: {muted};')
            value = QtWidgets.QLabel(MISSING)
            value.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            grid.addWidget(name, row, 2 * column)
            grid.addWidget(value, row, 2 * column + 1)
            self._values[key] = value
        for column in range(3):
            grid.setColumnStretch(2 * column + 1, 1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        # The space group belongs to the model, not the refinement, so it is
        # shown even for a file that was never refined.
        self._space_group = QtWidgets.QLabel()
        self._space_group.setFont(font)
        self._space_group.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        header = QtWidgets.QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(self._title)
        header.addStretch(1)
        header.addWidget(self._space_group)
        layout.addLayout(header)
        layout.addLayout(grid)
        layout.addStretch(1)
        # Never force the window wider: in a narrow one the values are cut
        # off rather than pushing the control rows beside it out of view.
        self.setMinimumSize(1, 1)
        self.set_space_group(None)
        self.show_summary(None, 'no refinement yet')

    @property
    def title(self) -> str:
        return self._title.text()

    @property
    def space_group(self) -> str:
        """The space-group text in the header."""
        return self._space_group.text()

    def set_space_group(self, name: str | None, tooltip: str = '') -> None:
        """Show the model's space group (``None``: not known)."""
        self._space_group.setText(f'Space group: {name or MISSING}')
        self._space_group.setToolTip(tooltip)

    def value(self, key: str) -> str:
        """The text shown for *key* (see :data:`SUMMARY_FIELDS`)."""
        return self._values[key].text()

    def show_summary(self, summary: RefinementSummary | None, status: str) -> None:
        """Show *summary* (``None``: nothing known) with *status* in the title."""
        self._title.setText(f'Refinement: {status}')
        values = _format_summary(summary if summary is not None else RefinementSummary())
        for key, (text, tooltip) in values.items():
            label = self._values[key]
            label.setText(text)
            label.setToolTip(tooltip)

    def set_target_width(self, width: int) -> None:
        """Prefer *width*, and never grow beyond it."""
        self._target_width = max(width, 1)
        self.setMaximumWidth(self._target_width)
        self.updateGeometry()

    def sizeHint(self) -> QtCore.QSize:
        hint = super().sizeHint()
        if self._target_width is not None:
            hint.setWidth(self._target_width)
        return hint
