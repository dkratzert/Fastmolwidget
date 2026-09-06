"""Checkable combo-box helpers for disorder-part filtering."""

from __future__ import annotations

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

# ---------------------------------------------------------------------------
# Qt-binding-agnostic event-type constant
# ---------------------------------------------------------------------------
try:
    _MOUSE_PRESS: int | Qt = QtCore.QEvent.Type.MouseButtonPress  # type: ignore[attr-defined]
except AttributeError:  # PyQt5
    _MOUSE_PRESS = QtCore.QEvent.MouseButtonPress  # type: ignore[attr-defined]


class _CheckableComboBox(QtWidgets.QComboBox):
    """QComboBox with checkable items that keeps its popup open while toggling."""

    #: Emitted whenever a checked state changes.
    selectionChanged = QtCore.Signal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # Use the line edit for custom summary text.
        self.setEditable(True)
        le = self.lineEdit()
        le.setReadOnly(True)
        # Clicking the line edit toggles the popup.
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
            # Keep the popup open on item clicks.
            return True

        if obj is le and event.type() == _MOUSE_PRESS:
            # Toggle the popup from the read-only line edit.
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
            text = "All"
        elif not vals:
            text = "None"
        else:
            text = ", ".join(str(v) for v in vals)
        le = self.lineEdit()
        if le is not None:
            le.setText(text)
        self.view().adjustSize()


class PartFilterWidget(QtWidgets.QWidget):
    """Label plus checkable combo box for disorder-part filtering."""

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
            # Size for the widest default label text.
            fm = self._combo.fontMetrics()
            # Arrow width from the current style; fall back to 28 px.
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

        # Hidden until multiple parts are present.
        self.hide()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_parts(self, parts: frozenset[int]) -> None:
        """Rebuild the combo from *parts* and hide it for 0 or 1 part."""
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
