"""
Qt Quick-based molecule viewer, analogous to
:class:`~fastmolwidget.viewer_widget.MoleculeViewerWidget`.

Architecture
------------
* :class:`MoleculeViewerBackend` — ``QObject`` exposed to QML as the context
  property ``"backend"``.  Manages :class:`~fastmolwidget.loader.MoleculeLoader`
  and wires QML controls to the renderer.
* :class:`MoleculeViewerQuickWidget` — thin ``QWidget`` wrapper that hosts a
  ``QQuickWidget`` (so the Qt Quick scene is embeddable in any QWidget layout).

The QML scene (``qml/MoleculeViewer.qml``) creates a ``MoleculeItem`` instance
and calls ``backend.registerRenderItem(mol)`` from ``Component.onCompleted`` to
hand the item reference back to Python.

Usage::

    viewer = MoleculeViewerQuickWidget()
    viewer.load_file("structure.cif")
    viewer.show()
"""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Slot

from fastmolwidget.loader import MoleculeLoader

try:
    from qtpy.QtCore import Property
except ImportError:
    from qtpy.QtCore import pyqtProperty as Property  # type: ignore[no-redef]

try:
    from qtpy.QtQuickWidgets import QQuickWidget
    from qtpy.QtQml import qmlRegisterType
    from fastmolwidget.molecule_quick import MoleculeQuickItem
    _HAS_QTQUICK = True
except (ImportError, RuntimeError):
    _HAS_QTQUICK = False


# ---------------------------------------------------------------------------
# Backend QObject
# ---------------------------------------------------------------------------

class MoleculeViewerBackend(QtCore.QObject):
    """Qt Quick backend that owns the loader and exposes all controls as slots.

    Wired to QML via ``engine.rootContext().setContextProperty("backend", self)``.
    """

    # -- Notify signals for bindable properties --
    growActiveChanged = QtCore.Signal(bool)
    packActiveChanged = QtCore.Signal(bool)
    showAdpsChanged = QtCore.Signal(bool)
    showLabelsChanged = QtCore.Signal(bool)
    hideHydrogensChanged = QtCore.Signal(bool)
    partsModelChanged = QtCore.Signal(list)
    hasPartsChanged = QtCore.Signal(bool)

    def __init__(self, parent: QtCore.QObject | None = None) -> None:
        super().__init__(parent)
        self._render_item: MoleculeQuickItem | None = None
        self._loader: MoleculeLoader | None = None

        self._grow_active = False
        self._pack_active = False
        self._show_adps = True
        self._show_labels = False
        self._hide_hydrogens = False
        self._parts_model: list[int] = []
        self._manual_parts: set[int] | None = None  # None = show all

    # ------------------------------------------------------------------
    # Called from QML Component.onCompleted
    # ------------------------------------------------------------------

    @Slot(QtCore.QObject)
    def registerRenderItem(self, item: object) -> None:
        """Receive the ``MoleculeItem`` instance created by QML and wire it up."""
        self._render_item = item  # type: ignore[assignment]
        self._loader = MoleculeLoader(self._render_item)

        # Sync initial state to the item
        self._render_item.show_adps(self._show_adps)
        self._render_item.show_labels(self._show_labels)
        self._render_item.show_hydrogens(not self._hide_hydrogens)

        # Connect partsChanged signal from the renderer
        self._render_item.partsChanged.connect(self._on_parts_changed)

    # ------------------------------------------------------------------
    # Properties (read-only from QML; mutated only through slots)
    # ------------------------------------------------------------------

    @Property(bool, notify=growActiveChanged)
    def growActive(self) -> bool:
        return self._grow_active

    @Property(bool, notify=packActiveChanged)
    def packActive(self) -> bool:
        return self._pack_active

    @Property(bool, notify=showAdpsChanged)
    def showAdps(self) -> bool:
        return self._show_adps

    @Property(bool, notify=showLabelsChanged)
    def showLabels(self) -> bool:
        return self._show_labels

    @Property(bool, notify=hideHydrogensChanged)
    def hideHydrogens(self) -> bool:
        return self._hide_hydrogens

    @Property(list, notify=partsModelChanged)
    def partsModel(self) -> list[int]:
        return self._parts_model

    @Property(bool, notify=hasPartsChanged)
    def hasParts(self) -> bool:
        return len(self._parts_model) > 1

    # ------------------------------------------------------------------
    # File I/O slots
    # ------------------------------------------------------------------

    @Slot()
    def openFileDialog(self) -> None:
        """Open a platform file dialog and load the chosen file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            None,
            "Open Structure File",
            "",
            "Structure Files (*.cif *.res *.ins *.xyz);;All Files (*)",
        )
        if path:
            self.load_file(path)

    @Slot()
    def saveImageDialog(self) -> None:
        """Open a save dialog and export the current view as an image."""
        if self._render_item is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            None,
            "Save Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self._render_item.save_image(Path(path))

    def load_file(self, filename: str | Path) -> None:
        """Load a structure file (called from Python *or* via slot)."""
        if self._loader is not None:
            self._loader.load_file(filename)

    # ------------------------------------------------------------------
    # Structure toggle slots
    # ------------------------------------------------------------------

    @Slot(bool)
    def setGrow(self, checked: bool) -> None:
        if checked and self._pack_active:
            self._pack_active = False
            self.packActiveChanged.emit(False)
            if self._loader:
                self._loader.set_pack(False)
        self._grow_active = checked
        self.growActiveChanged.emit(checked)
        if self._loader:
            self._loader.set_grow(checked)

    @Slot(bool)
    def setPack(self, checked: bool) -> None:
        if checked and self._grow_active:
            self._grow_active = False
            self.growActiveChanged.emit(False)
            if self._loader:
                self._loader.set_grow(False)
        self._pack_active = checked
        self.packActiveChanged.emit(checked)
        if self._loader:
            self._loader.set_pack(checked)
        if checked and self._render_item is not None:
            self._render_item.reset_rotation_center()
            self._render_item._align_to_reciprocal_axis(1)

    @Slot(bool)
    def setShowAdps(self, value: bool) -> None:
        self._show_adps = value
        self.showAdpsChanged.emit(value)
        if self._render_item:
            self._render_item.show_adps(value)

    @Slot(bool)
    def setShowLabels(self, value: bool) -> None:
        self._show_labels = value
        self.showLabelsChanged.emit(value)
        if self._render_item:
            self._render_item.show_labels(value)

    @Slot(bool)
    def setHideHydrogens(self, value: bool) -> None:
        self._hide_hydrogens = value
        self.hideHydrogensChanged.emit(value)
        if self._render_item:
            self._render_item.show_hydrogens(not value)

    # ------------------------------------------------------------------
    # Bond slots
    # ------------------------------------------------------------------

    @Slot(int)
    def setBondWidth(self, width: int) -> None:
        if self._render_item:
            self._render_item.set_bond_width(width)

    @Slot()
    def chooseBondColor(self) -> None:
        if self._render_item is None:
            return
        current = QtGui.QColor(self._render_item.bond_color)
        color = QtWidgets.QColorDialog.getColor(current, None, "Choose Bond Color")
        if color.isValid():
            self._render_item.set_bond_color(color)

    def set_bond_color(
        self,
        color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set bond colour from Python code (not a slot)."""
        if self._render_item:
            self._render_item.set_bond_color(color)

    # ------------------------------------------------------------------
    # View slots
    # ------------------------------------------------------------------

    @Slot()
    def resetCenter(self) -> None:
        if self._render_item:
            self._render_item.reset_rotation_center()

    @Slot()
    def bestView(self) -> None:
        if self._render_item:
            self._render_item.align_best_view()

    @Slot(int)
    def alignAxis(self, axis: int) -> None:
        """Called from QML key handlers for F1/F2/F3."""
        if self._render_item:
            self._render_item._align_to_reciprocal_axis(axis)

    # ------------------------------------------------------------------
    # Parts filter slots
    # ------------------------------------------------------------------

    @Slot(int, bool)
    def togglePart(self, part: int, checked: bool) -> None:
        """Called from each part CheckBox in QML."""
        if self._render_item is None:
            return
        if self._manual_parts is None:
            self._manual_parts = set(self._render_item.available_parts)
        if checked:
            self._manual_parts.add(part)
        else:
            self._manual_parts.discard(part)
        if self._manual_parts == self._render_item.available_parts:
            self._render_item.set_visible_parts(None)
        else:
            self._render_item.set_visible_parts(self._manual_parts)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_parts_changed(self, parts: frozenset[int]) -> None:
        self._manual_parts = None  # reset manual filter on new load
        new_model = sorted(parts) if len(parts) > 1 else []
        had_parts = len(self._parts_model) > 1
        self._parts_model = new_model
        self.partsModelChanged.emit(new_model)
        if had_parts != (len(new_model) > 1):
            self.hasPartsChanged.emit(len(new_model) > 1)
        if len(new_model) > 1 and self._render_item is not None:
            self._render_item.set_visible_parts(None)


# ---------------------------------------------------------------------------
# QWidget wrapper
# ---------------------------------------------------------------------------

class MoleculeViewerQuickWidget(QtWidgets.QWidget):
    """A ready-to-use Qt Quick viewer, analogous to
    :class:`~fastmolwidget.viewer_widget.MoleculeViewerWidget`.

    Embeds a ``QQuickWidget`` containing ``qml/MoleculeViewer.qml``.  The
    :class:`MoleculeViewerBackend` is exposed to QML as the context property
    ``"backend"``.

    :param parent: Optional parent widget.
    :raises RuntimeError: If Qt Quick is not available (``QtQuick`` not
        installed or no QML engine could be created).
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        if not _HAS_QTQUICK:
            # Degrade gracefully: show an informative label instead of crashing.
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel(
                "Qt Quick is not available.\n"
                "Install the 'pyside6' or 'pyqt6' extras to enable the Quick viewer."
            ))
            self._backend: MoleculeViewerBackend | None = None
            return

        # Register the MoleculeQuickItem type with QML (idempotent)
        qmlRegisterType(MoleculeQuickItem, "Fastmolwidget", 1, 0, "MoleculeItem")

        # Create backend before setting up the QML engine
        self._backend = MoleculeViewerBackend(self)

        # QQuickWidget hosts the entire QML scene
        self._quick_widget = QQuickWidget(self)
        self._quick_widget.setResizeMode(
            QQuickWidget.ResizeMode.SizeRootObjectToView
        )

        # Expose backend to the QML context
        engine = self._quick_widget.engine()
        ctx = engine.rootContext()
        ctx.setContextProperty("backend", self._backend)

        # Load the bundled QML file
        qml_path = Path(__file__).parent / "qml" / "MoleculeViewer.qml"
        self._quick_widget.setSource(
            QtCore.QUrl.fromLocalFile(str(qml_path.resolve()))
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._quick_widget)

    # ------------------------------------------------------------------
    # Public API (mirrors MoleculeViewerWidget)
    # ------------------------------------------------------------------

    @property
    def render_widget(self) -> MoleculeQuickItem | None:
        """The underlying :class:`MoleculeQuickItem` (``None`` before the QML
        ``Component.onCompleted`` fires or when Qt Quick is unavailable)."""
        return self._backend._render_item if self._backend else None

    def load_file(self, filename: str | Path) -> None:
        """Load a structure file and display it.

        :param filename: Path to the file (``.cif``, ``.res``, ``.ins``, ``.xyz``).
        """
        if self._backend:
            self._backend.load_file(filename)

    def set_bond_color(
        self,
        color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set the default colour used for non-selected bonds."""
        if self._backend:
            self._backend.set_bond_color(color)


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = MoleculeViewerQuickWidget()
    w.resize(900, 650)
    w.show()
    # Load after show so the QML item is fully initialised
    QtCore.QTimer.singleShot(
        100, lambda: w.load_file('tests/test-data/1979688_small.cif')
    )
    sys.exit(app.exec())

