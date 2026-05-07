"""Checkable multi-select QComboBox for disorder-part filtering.

Internal helper shared by :mod:`~fastmolwidget.viewer_widget` and
:mod:`~fastmolwidget.viewer_widget3D`.  Not part of the public API.
"""

from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

# ---------------------------------------------------------------------------
# Qt-binding–agnostic event-type constant
# ---------------------------------------------------------------------------
try:
    _MOUSE_PRESS: int | Qt = QtCore.QEvent.Type.MouseButtonPress  # type: ignore[attr-defined]
except AttributeError:  # PyQt5
    _MOUSE_PRESS = QtCore.QEvent.MouseButtonPress  # type: ignore[attr-defined]


class _CheckableComboBox(QtWidgets.QComboBox):
    """A :class:`~qtpy.QtWidgets.QComboBox` whose items each have a checkbox.

    The popup stays open while the user ticks/unticks items.
    The button face shows a summary text such as ``"Parts: 0, 1"`` or
    ``"All Parts"``.

    Usage::

        combo = _CheckableComboBox()
        combo.add_part(0)
        combo.add_part(1)
        combo.add_part(2)
        combo.selectionChanged.connect(lambda: print(combo.checked_values()))
    """

    #: Emitted whenever a checked state changes.
    selectionChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Make the button text area editable but read-only so we can set
        # arbitrary summary text without the standard "current item" logic.
        self.setEditable(True)
        le = self.lineEdit()
        le.setReadOnly(True)
        # Forward clicks on the line-edit to toggle popup open/closed.
        le.installEventFilter(self)

        model = QtGui.QStandardItemModel(self)
        self.setModel(model)
        self.view().viewport().installEventFilter(self)
        model.dataChanged.connect(self._on_data_changed)
        self._update_button_text()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_part(self, part: int, *, checked: bool = True) -> None:
        """Append one disorder *part* number to the combo."""
        item = QtGui.QStandardItem(f"Part {part}")
        item.setData(part, Qt.ItemDataRole.UserRole)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        self.model().appendRow(item)
        self._update_button_text()

    def clear_parts(self) -> None:
        """Remove all items and reset the button text."""
        self.model().clear()
        self._update_button_text()

    def checked_values(self) -> list[int]:
        """Return the part numbers whose checkbox is currently ticked."""
        result: list[int] = []
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def eventFilter(  # type: ignore[override]
        self,
        obj: QtCore.QObject,
        event: QtCore.QEvent,
    ) -> bool:
        viewport = self.view().viewport()
        le = self.lineEdit()

        if obj is viewport and event.type() == _MOUSE_PRESS:
            idx = self.view().indexAt(event.pos())
            if idx.isValid():
                item = self.model().item(idx.row())
                if item is not None:
                    new = (
                        Qt.CheckState.Unchecked
                        if item.checkState() == Qt.CheckState.Checked
                        else Qt.CheckState.Checked
                    )
                    item.setCheckState(new)
            # Always return True: prevents the default handler from
            # activating/closing the popup on item click.
            return True

        if obj is le and event.type() == _MOUSE_PRESS:
            # Toggle popup when the read-only line-edit is clicked.
            if self.view().isVisible():
                self.hidePopup()
            else:
                self.showPopup()
            return True

        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_data_changed(self, *_args: object) -> None:
        self._update_button_text()
        self.selectionChanged.emit()

    def _update_button_text(self) -> None:
        vals = self.checked_values()
        n_total = self.model().rowCount()
        if n_total == 0 or len(vals) == n_total:
            text = "All Parts"
        elif not vals:
            text = "No Parts"
        else:
            text = "Parts: " + ", ".join(str(v) for v in vals)
        le = self.lineEdit()
        if le is not None:
            le.setText(text)


class PartFilterWidget(QtWidgets.QWidget):
    """A label + checkable combo box for disorder-part filtering.

    Wraps :class:`_CheckableComboBox` together with a descriptive label into
    a single :class:`~qtpy.QtWidgets.QWidget` that shows/hides itself
    automatically depending on how many distinct parts are present.

    Parameters
    ----------
    label:
        Text shown to the left of the combo box. Defaults to ``"Show Parts:"``.
    min_combo_width:
        Minimum width (pixels) for the combo box.  When ``None`` (default)
        the width is derived from the rendered size of the label text
        ``"All Parts"`` plus room for the drop-down arrow, so the button
        face never clips its content.
    parent:
        Optional parent widget.

    Signals
    -------
    selectionChanged
        Re-emitted from the inner :class:`_CheckableComboBox` whenever a
        checked state changes.

    Usage::

        widget = PartFilterWidget()
        widget.selectionChanged.connect(lambda: do_something(widget.checked_values()))
        layout.addWidget(widget)

        # When a new molecule is loaded, call:
        widget.update_parts(frozenset({0, 1, 2}))
    """

    #: Emitted whenever a checked state changes inside the combo.
    selectionChanged = QtCore.Signal()

    def __init__(
        self,
        label: str = "Show Parts:",
        *,
        min_combo_width: int | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Fixed,
        )

        self._combo = _CheckableComboBox()
        self._combo.selectionChanged.connect(self.selectionChanged)

        if min_combo_width is None:
            # Derive the minimum width from the widest default label text
            # ("All Parts") so the button face is never clipped.
            fm = self._combo.fontMetrics()
            # Arrow-button width from the current Qt style; fall back to 28 px.
            try:
                arrow_w: int = self._combo.style().pixelMetric(
                    QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent  # type: ignore[attr-defined]
                )
            except AttributeError:
                arrow_w = 28
            min_combo_width = fm.horizontalAdvance("All Parts") + arrow_w + 8
        self._combo.setMinimumWidth(min_combo_width)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QtWidgets.QLabel(label))
        layout.addWidget(self._combo)

        # Hidden until update_parts() reveals disorder.
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_parts(self, parts: frozenset[int]) -> None:
        """Rebuild the combo from *parts* and show/hide the widget.

        - If *parts* has **≤ 1** element the widget hides itself.
        - Otherwise each part number is added as a checked item and the
          widget becomes visible.

        Parameters
        ----------
        parts:
            The set of distinct disorder-part integers from the current
            molecule.
        """
        self._combo.clear_parts()
        if len(parts) <= 1:
            self.hide()
            return
        for part in sorted(parts):
            self._combo.add_part(part, checked=True)
        self.show()

    def checked_values(self) -> list[int]:
        """Return part numbers whose checkbox is currently ticked."""
        return self._combo.checked_values()

