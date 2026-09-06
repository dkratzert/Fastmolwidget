"""Qt Quick viewer and backend."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Slot

from fastmolwidget.density_controls import (
    HAS_DENSITY_CPP,
    ask_for_reflection_file,
    auto_reflection_file,
    density_statistics_text,
)
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule_base import DENSITY_LEVEL_MAX, DENSITY_LEVEL_MIN

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
    """Backend exposed to QML as ``backend``."""

    # Notify signals for bindable properties.
    growActiveChanged = QtCore.Signal(bool)
    packActiveChanged = QtCore.Signal(bool)
    showAdpsChanged = QtCore.Signal(bool)
    showLabelsChanged = QtCore.Signal(bool)
    hideHydrogensChanged = QtCore.Signal(bool)
    partsModelChanged = QtCore.Signal(list)
    hasPartsChanged = QtCore.Signal(bool)
    densityActiveChanged = QtCore.Signal(bool)
    densityLevelChanged = QtCore.Signal(float)
    densityStatisticsChanged = QtCore.Signal(str)

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
        self._density_active = False
        self._density_level = 0.30
        self._density_statistics = ''

    # ------------------------------------------------------------------
    # Called from QML Component.onCompleted.
    # ------------------------------------------------------------------

    @Slot(QtCore.QObject)
    def registerRenderItem(self, item: object) -> None:
        """Store the QML ``MoleculeItem`` and wire it up."""
        self._render_item = item  # type: ignore[assignment]
        self._loader = MoleculeLoader(self._render_item)

        # Sync initial state.
        self._render_item.show_adps(self._show_adps)
        self._render_item.show_labels(self._show_labels)
        self._render_item.show_hydrogens(not self._hide_hydrogens)

        # Connect renderer signals.
        self._render_item.partsChanged.connect(self._on_parts_changed)
        # Keep QML in sync with Ctrl+wheel level changes.
        self._render_item.densityLevelChanged.connect(self._on_density_level_changed)

    # ------------------------------------------------------------------
    # Read-only QML properties.
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

    @Property(bool, constant=True)
    def densityAvailable(self) -> bool:
        """Whether ``density_cpp`` is available."""
        return HAS_DENSITY_CPP

    @Property(bool, notify=densityActiveChanged)
    def densityActive(self) -> bool:
        """Whether residual density is shown."""
        return self._density_active

    @Property(float, notify=densityLevelChanged)
    def densityLevel(self) -> float:
        """Residual-density level in e/Å³."""
        return self._density_level

    @Property(float, constant=True)
    def densityLevelMin(self) -> float:
        return DENSITY_LEVEL_MIN

    @Property(float, constant=True)
    def densityLevelMax(self) -> float:
        return DENSITY_LEVEL_MAX

    @Property(str, notify=densityStatisticsChanged)
    def densityStatistics(self) -> str:
        """Tooltip text for the density button."""
        return self._density_statistics

    # ------------------------------------------------------------------
    # File I/O slots.
    # ------------------------------------------------------------------

    @Slot()
    def openFileDialog(self) -> None:
        """Open a file dialog and load the chosen structure."""
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
        """Open a save dialog and export the current view."""
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
        """Load a structure file."""
        if self._loader is not None:
            self._loader.load_file(filename)
            # Keep QML in sync if a new model cleared density.
            self._sync_density_state()

    # ------------------------------------------------------------------
    # Residual-density slots.
    # ------------------------------------------------------------------

    @Slot(bool)
    def setDensity(self, checked: bool) -> None:
        """Show or hide residual density from the QML toggle."""
        if self._render_item is None:
            return
        if not checked:
            self.clear_residual_density()
            return

        hkl_path = auto_reflection_file(self._render_item)
        if hkl_path is None:
            hkl_path = ask_for_reflection_file(None, self._render_item)
            if hkl_path is None:  # dialog cancelled
                self._set_density_active(False)
                return

        error_message = ""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            # level=None -> contour at 3 sigma for this map.
            self.show_residual_density(hkl_path)
        except Exception as exc:  # noqa: BLE001 - never take the host app down
            error_message = str(exc)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if error_message:
            self._set_density_active(False)
            QtWidgets.QMessageBox.warning(None, "Residual density", error_message)

    @Slot(float)
    def setDensityLevel(self, level: float) -> None:
        """Re-contour the cached map at *level*."""
        if self._render_item is not None:
            self._render_item.set_residual_density_level(level)

    def show_residual_density(self, hkl_path: str | Path | None = None,
                              level: float | None = None) -> None:
        """Show a residual-density map and sync QML state."""
        if self._render_item is None:
            raise RuntimeError('The QML render item is not ready yet.')
        self._render_item.show_residual_density(hkl_path, level)
        self._on_density_level_changed(self._render_item.residual_density_level)
        self._set_density_active(True)

    def clear_residual_density(self) -> None:
        """Clear the residual-density isosurface."""
        if self._render_item is not None:
            self._render_item.clear_residual_density()
        self._set_density_active(False)

    def _set_density_active(self, active: bool) -> None:
        """Publish density state and tooltip to QML.

        Emit unconditionally so failed toggles push ``False`` back into QML.
        """
        density_map = (self._render_item.residual_density_map
                       if active and self._render_item is not None else None)
        statistics = density_statistics_text(density_map)
        if statistics != self._density_statistics:
            self._density_statistics = statistics
            self.densityStatisticsChanged.emit(statistics)
        self._density_active = active
        self.densityActiveChanged.emit(active)

    def _sync_density_state(self) -> None:
        """Match QML state to the renderer."""
        self._set_density_active(
            self._render_item is not None
            and self._render_item.residual_density_map is not None
        )

    def _on_density_level_changed(self, level: float) -> None:
        """Mirror view-side level changes into QML."""
        if level == self._density_level:
            return
        self._density_level = level
        self.densityLevelChanged.emit(level)

    # ------------------------------------------------------------------
    # Structure toggle slots.
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
    # Bond slots.
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
        """Set bond color from Python code."""
        if self._render_item:
            self._render_item.set_bond_color(color)

    # ------------------------------------------------------------------
    # View slots.
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
        """Handle QML F1/F2/F3 shortcuts."""
        if self._render_item:
            self._render_item._align_to_reciprocal_axis(axis)

    # ------------------------------------------------------------------
    # Parts filter slots.
    # ------------------------------------------------------------------

    @Slot(int, bool)
    def togglePart(self, part: int, checked: bool) -> None:
        """Toggle one disorder part."""
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
    # Internal helpers.
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
    """Qt Quick viewer hosted in a ``QQuickWidget``."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        if not _HAS_QTQUICK:
            # Degrade gracefully instead of crashing.
            layout = QtWidgets.QVBoxLayout(self)
            layout.addWidget(QtWidgets.QLabel(
                "Qt Quick is not available.\n"
                "Install the 'pyside6' or 'pyqt6' extras to enable the Quick viewer."
            ))
            self._backend: MoleculeViewerBackend | None = None
            return

        # Register the QML item type.
        qmlRegisterType(MoleculeQuickItem, "Fastmolwidget", 1, 0, "MoleculeItem")

        # Create the backend first.
        self._backend = MoleculeViewerBackend(self)

        # Host the full QML scene.
        self._quick_widget = QQuickWidget(self)
        self._quick_widget.setResizeMode(
            QQuickWidget.ResizeMode.SizeRootObjectToView
        )

        # Expose the backend to QML.
        engine = self._quick_widget.engine()
        ctx = engine.rootContext()
        ctx.setContextProperty("backend", self._backend)

        # Load the bundled QML file.
        qml_path = Path(__file__).parent / "qml" / "MoleculeViewer.qml"
        self._quick_widget.setSource(
            QtCore.QUrl.fromLocalFile(str(qml_path.resolve()))
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._quick_widget)

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------

    @property
    def render_widget(self) -> MoleculeQuickItem | None:
        """Underlying :class:`MoleculeQuickItem`, or ``None`` if unavailable."""
        return self._backend._render_item if self._backend else None

    def load_file(self, filename: str | Path) -> None:
        """Load and display a structure file."""
        if self._backend:
            self._backend.load_file(filename)
            self.setWindowTitle(str(Path(filename).resolve()))

    def set_bond_color(
            self,
            color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set the default color for non-selected bonds."""
        if self._backend:
            self._backend.set_bond_color(color)

    def show_residual_density(self, hkl_path: str | Path | None = None,
                              level: float | None = None) -> None:
        """Show a residual-density map."""
        if self._backend is None:
            raise RuntimeError('Qt Quick is not available.')
        self._backend.show_residual_density(hkl_path, level)

    def clear_residual_density(self) -> None:
        """Clear the residual-density isosurface."""
        if self._backend:
            self._backend.clear_residual_density()


# ---------------------------------------------------------------------------
# Development entry point.
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = MoleculeViewerQuickWidget()
    w.resize(900, 650)
    w.show()
    # Load after show so the QML item exists.
    QtCore.QTimer.singleShot(
        # 100, lambda: w.load_file('tests/test-data/1979688_small.cif')
        10, lambda: w.load_file('tests/test-data/p21c.cif')
    )
    sys.exit(app.exec())
