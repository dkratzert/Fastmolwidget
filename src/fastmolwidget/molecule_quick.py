"""
``QQuickPaintedItem``-based renderer for the Qt Quick viewer.

:class:`MoleculeQuickItem` provides the same 2-D molecule drawing as
:class:`~fastmolwidget.molecule2D.MoleculeWidget` but lives inside a
Qt Quick scene graph.  It is registered as the QML type ``MoleculeItem``
(module ``Fastmolwidget``, version 1.0) by
:mod:`~fastmolwidget.viewer_widget_quick`.

Hover tracking
--------------
Because ``QQuickItem`` fires ``mouseMoveEvent`` **only** when a mouse button
is pressed, hover labels require ``setAcceptHoverEvents(True)`` and overriding
``hoverMoveEvent`` / ``hoverLeaveEvent``.  ``MoleculeRendererMixin._update_hover``
is called from both paths.

Import guard
------------
The entire module is guarded so that importing it on a system without Qt Quick
does not crash the host application.
"""

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
    """``QQuickPaintedItem`` molecule renderer.

    Shares all drawing and interaction logic with
    :class:`~fastmolwidget.molecule2D.MoleculeWidget` via
    :class:`~fastmolwidget.molecule_painter.MoleculeRendererMixin`.

    Register with QML before creating any engine::

        from qtpy.QtQml import qmlRegisterType
        qmlRegisterType(MoleculeQuickItem, "Fastmolwidget", 1, 0, "MoleculeItem")

    Then in QML::

        import Fastmolwidget 1.0
        MoleculeItem { id: mol; anchors.fill: parent }
    """

    atomClicked = QtCore.Signal(str)
    bondClicked = QtCore.Signal(str, str)
    #: Emitted after every load with the frozenset of disorder-part numbers.
    partsChanged = QtCore.Signal(object)

    def __init__(self, parent: QQuickPaintedItem | None = None) -> None:
        QQuickPaintedItem.__init__(self, parent)
        self._init_renderer()

        # Allow mouse and keyboard interaction
        self.setAcceptedMouseButtons(Qt.MouseButton.AllButtons)
        self.setAcceptHoverEvents(True)
        self.setFlag(QQuickPaintedItem.Flag.ItemAcceptsInputMethod, True)
        self.setFlag(QQuickPaintedItem.Flag.ItemIsFocusScope, True)

        # Use an FBO so the scene graph composites efficiently
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
        # Fill background
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
    # Hover events (QQuickItem fires these without a button pressed)
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
