"""Qt Quick wrapper around the shared 2-D molecule renderer."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QImage, QPainter

try:
    from qtpy.QtQuick import QQuickPaintedItem
    _HAS_QTQUICK = True
except ImportError:
    _HAS_QTQUICK = False
    QQuickPaintedItem = object  # type: ignore[assignment, misc]

from fastmolwidget.molecule_painter import MoleculeRendererMixin


class MoleculeQuickItem(MoleculeRendererMixin, QQuickPaintedItem):  # type: ignore[misc]
    """``QQuickPaintedItem`` molecule renderer."""

    atomClicked = QtCore.Signal(str)
    bondClicked = QtCore.Signal(str, str)
    #: Emitted after loading with the frozenset of disorder parts present.
    partsChanged = QtCore.Signal(object)
    #: Emitted when the residual-density contour level changes.
    densityLevelChanged = QtCore.Signal(float)

    def __init__(self, parent: QQuickPaintedItem | None = None) -> None:
        QQuickPaintedItem.__init__(self, parent)
        self._init_renderer()

        self.setAcceptedMouseButtons(Qt.MouseButton.AllButtons)
        self.setAcceptHoverEvents(True)
        self.setFlag(QQuickPaintedItem.Flag.ItemAcceptsInputMethod, True)
        self.setFlag(QQuickPaintedItem.Flag.ItemIsFocusScope, True)

        # Prefer an FBO when Qt supports it.
        if hasattr(QQuickPaintedItem, 'RenderTarget'):
            try:
                self.setRenderTarget(
                    QQuickPaintedItem.RenderTarget.FramebufferObject
                )
            except Exception:
                pass  # older Qt versions may not support this

    # ------------------------------------------------------------------
    # QQuickPaintedItem interface
    # ------------------------------------------------------------------

    def paint(self, painter: QPainter) -> None:
        """Called by the Qt Quick scene graph to repaint the item."""
        painter.fillRect(
            QtCore.QRectF(0, 0, self.width(), self.height()),
            self._bg_color,
        )
        if not self.atoms:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = painter.font()
        font.setPixelSize(max(1, int(self.fontsize * self.zoom * 4)))
        painter.setFont(font)
        self._painter = painter
        try:
            self.draw()
        except (ValueError, IndexError) as e:
            print(f'MoleculeQuickItem draw crashed: {e}')
        finally:
            self._painter = None

    # ------------------------------------------------------------------
    # Geometry change (replaces QWidget.resizeEvent)
    # ------------------------------------------------------------------

    def geometryChange(
        self,
        new_geometry: QtCore.QRectF,
        old_geometry: QtCore.QRectF,
    ) -> None:
        """Scale zoom when the item is resized."""
        self._on_resize(
            old_geometry.width(), old_geometry.height(),
            new_geometry.width(), new_geometry.height(),
        )
        super().geometryChange(new_geometry, old_geometry)

    # ------------------------------------------------------------------
    # Hover events
    # ------------------------------------------------------------------

    def hoverMoveEvent(self, event: QtGui.QHoverEvent) -> None:  # type: ignore[override]
        pos = event.position()
        self._update_hover(pos.x(), pos.y())

    def hoverLeaveEvent(self, event: QtGui.QHoverEvent) -> None:  # type: ignore[override]
        self._on_leave()
        super().hoverLeaveEvent(event)

    # ------------------------------------------------------------------
    # Background colour
    # ------------------------------------------------------------------

    def set_background_color(self, color: QColor) -> None:
        """Set the background fill colour."""
        self._bg_color = color
        self.update()

    # ------------------------------------------------------------------
    # Image export
    # ------------------------------------------------------------------

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Render the current view to an image file."""
        w = int(self.width() * image_scale)
        h = int(self.height() * image_scale)
        if w <= 0 or h <= 0:
            return
        image = QImage(w, h, QImage.Format.Format_RGB32)
        image.fill(self._bg_color)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.scale(image_scale, image_scale)
        font = painter.font()
        font.setPixelSize(max(1, int(self.fontsize * self.zoom * 4)))
        painter.setFont(font)
        self._painter = painter
        try:
            if self.atoms:
                self.draw()
        except (ValueError, IndexError) as e:
            print(f'MoleculeQuickItem save_image crashed: {e}')
        finally:
            self._painter = None
            painter.end()
        image.save(str(filename.resolve()))
