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
        # Ctrl+wheel in the view changes the contour level; keep the Level
        # spin box showing what is actually contoured.
        self._render_item.densityLevelChanged.connect(self._on_density_level_changed)

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

    @Property(bool, constant=True)
    def densityAvailable(self) -> bool:
        """Whether the compiled ``density_cpp`` extension is usable."""
        return HAS_DENSITY_CPP

    @Property(bool, notify=densityActiveChanged)
    def densityActive(self) -> bool:
        """Whether a residual-density isosurface is currently displayed."""
        return self._density_active

    @Property(float, notify=densityLevelChanged)
    def densityLevel(self) -> float:
        """Contour level of the residual density, in e/Å³."""
        return self._density_level

    @Property(float, constant=True)
    def densityLevelMin(self) -> float:
        return DENSITY_LEVEL_MIN

    @Property(float, constant=True)
    def densityLevelMax(self) -> float:
        return DENSITY_LEVEL_MAX

    @Property(str, notify=densityStatisticsChanged)
    def densityStatistics(self) -> str:
        """Tooltip text for the Residual Density button."""
        return self._density_statistics

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
            # Loading a different structure drops its residual density; make
            # the control bar follow whatever the renderer ended up with.
            self._sync_density_state()

    # ------------------------------------------------------------------
    # Residual-density slots
    # ------------------------------------------------------------------

    @Slot(bool)
    def setDensity(self, checked: bool) -> None:
        """Show or hide the residual density from the QML toggle button.

        The QML button is the single source of truth for the on/off state, so
        any failure (no model, no reflection data, or a cancelled file dialog)
        pushes it back out again via ``densityActiveChanged``.
        """
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
            # level=None -> contour at 3 sigma of this particular map.
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
        """Re-contour the cached map at *level* (e/Å³) from the QML spin box."""
        if self._render_item is not None:
            self._render_item.set_residual_density_level(level)

    def show_residual_density(self, hkl_path: str | Path | None = None,
                              level: float | None = None) -> None:
        """Compute and show a residual electron-density map.

        Mirrors :meth:`fastmolwidget.viewer_widget.MoleculeViewerWidget.show_residual_density`;
        the QML button and Level spin box are updated to match.

        :param hkl_path: Reflection file; ``None`` finds it automatically.
        :param level: Contour level in e/Å³; ``None`` uses 3σ of the map.
        """
        if self._render_item is None:
            raise RuntimeError('The QML render item is not ready yet.')
        self._render_item.show_residual_density(hkl_path, level)
        self._on_density_level_changed(self._render_item.residual_density_level)
        self._set_density_active(True)

    def clear_residual_density(self) -> None:
        """Remove the residual electron-density isosurface."""
        if self._render_item is not None:
            self._render_item.clear_residual_density()
        self._set_density_active(False)

    def _set_density_active(self, active: bool) -> None:
        """Publish the on/off state to QML, tooltip included.

        ``densityActiveChanged`` is emitted **unconditionally**: when turning
        the density on fails, the QML button has already flipped itself to
        checked, so the backend has to push ``False`` back out even though its
        own state never changed.  Re-assigning the same value in QML is a
        no-op, so the redundant notification is harmless.
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
        """Match the QML controls to the renderer's actual state."""
        self._set_density_active(
            self._render_item is not None
            and self._render_item.residual_density_map is not None
        )

    def _on_density_level_changed(self, level: float) -> None:
        """Mirror a level change made in the view (Ctrl+wheel) into QML."""
        if level == self._density_level:
            return
        self._density_level = level
        self.densityLevelChanged.emit(level)

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
        """Toggle a single part's visibility and update the renderer."""
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

    def show_residual_density(self, hkl_path: str | Path | None = None,
                              level: float | None = None) -> None:
        """Compute and show a residual (Fo−Fc) electron-density map.

        The QML Residual Density button and Level spin box follow along, so
        the controls never disagree with the view.

        :param hkl_path: Reflection file; ``None`` finds it automatically from
            the loaded model.
        :param level: Contour level in e/Å³; ``None`` contours at 3σ of the map.
        :raises RuntimeError: If Qt Quick is unavailable, the QML item is not
            ready yet, no model is loaded, or ``density_cpp`` is missing.
        """
        if self._backend is None:
            raise RuntimeError('Qt Quick is not available.')
        self._backend.show_residual_density(hkl_path, level)

    def clear_residual_density(self) -> None:
        """Remove the residual electron-density isosurface."""
        if self._backend:
            self._backend.clear_residual_density()


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
        # 100, lambda: w.load_file('tests/test-data/1979688_small.cif')
        10, lambda: w.load_file('tests/test-data/p21c.cif')
    )
    sys.exit(app.exec())
