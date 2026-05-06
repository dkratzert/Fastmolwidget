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

