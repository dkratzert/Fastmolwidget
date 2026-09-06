"""Qt widget wrapper around the shared 2-D molecule renderer."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QImage, QPainter, QPalette, QResizeEvent

from fastmolwidget.molecule_painter import (
    Atom,  # re-exported for backwards compatibility
    MoleculeRendererMixin,
    RenderItem,  # re-exported for backwards compatibility
    calc_volume,  # re-exported for backwards compatibility
)
from fastmolwidget.sdm import Atomtuple  # noqa: F401


class MoleculeWidget(MoleculeRendererMixin, QtWidgets.QWidget):
    """Interactive 2-D molecule widget."""

    _AUTO_ZOOM_PADDING = 1.1

    atomClicked = QtCore.Signal(str)
    bondClicked = QtCore.Signal(str, str)
    #: Emitted after loading with the frozenset of disorder parts present.
    partsChanged = QtCore.Signal(object)
    #: Emitted when the residual-density contour level changes.
    densityLevelChanged = QtCore.Signal(float)

    def __init__(self, parent: QtGui.QWidget | None = None) -> None:
        # QWidget must exist before _init_renderer() uses update() or signals.
        QtWidgets.QWidget.__init__(self, parent)
        self._init_renderer()

        pal = QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtCore.Qt.GlobalColor.white)
        self.setAutoFillBackground(True)
        self.setPalette(pal)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.setMouseTracking(True)

    # ------------------------------------------------------------------
    # Qt widget overrides
    # ------------------------------------------------------------------

    def sizeHint(self) -> QtCore.QSize:
        """Preferred starting size so the render area stays visible in layouts."""
        return QtCore.QSize(640, 480)

    def minimumSizeHint(self) -> QtCore.QSize:
        """Reasonable minimum size for molecule rendering."""
        return QtCore.QSize(320, 220)

    def set_background_color(self, color: QColor) -> None:
        """Set the background colour of the widget."""
        self._bg_color = color
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, color)
        self.setPalette(palette)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        """Repaint the widget by re-rendering the molecule scene."""
        if self.atoms:
            self._painter = QPainter(self)
            self._painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            font = self._painter.font()
            font.setPixelSize(max(1, int(self.fontsize * self.zoom * 4)))
            self._painter.setFont(font)
            try:
                self.draw()
            except (ValueError, IndexError) as e:
                print(f'Draw structure crashed: {e}')
            finally:
                self._painter.end()

    def resizeEvent(self, event: QResizeEvent) -> None:
        """Keep the molecule at the same relative size after resize."""
        old = event.oldSize()
        new = event.size()
        self._on_resize(old.width(), old.height(), new.width(), new.height())
        super().resizeEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        """Clear hover labels when the cursor leaves the widget."""
        self._on_leave()
        super().leaveEvent(event)

    # ------------------------------------------------------------------
    # Image export
    # ------------------------------------------------------------------

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Render the current molecule view to an image file."""
        image = QImage(self.size() * image_scale, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        painter = QPainter(image)
        painter.scale(image_scale, image_scale)
        self.render(painter, QtCore.QPoint(0, 0))
        painter.end()
        image.save(str(filename.resolve()))


# ---------------------------------------------------------------------------
# Compatibility re-exports
# ---------------------------------------------------------------------------
__all__ = ['Atom', 'MoleculeWidget', 'RenderItem', 'calc_volume']


if __name__ == '__main__':
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])
    from fastmolwidget.viewer_widget import MoleculeViewerWidget
    w = MoleculeViewerWidget()
    w.load_file('tests/test-data/1548072_many_atoms.cif')
    w.grow()
    w.show()
    w.showMaximized()
    app.exec()
