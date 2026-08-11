"""
A versatile 2-D molecule drawing widget for PyQt/PySide.

Renders molecules as ORTEP-style thermal ellipsoid plots (when anisotropic
displacement parameters are provided) or as simple ball-and-stick diagrams.
The widget supports interactive mouse rotation, zooming, and panning.

All rendering logic lives in
:class:`~fastmolwidget.molecule_painter.MoleculeRendererMixin`.
This module keeps only the Qt-widget boilerplate (``paintEvent``,
``resizeEvent``, palette setup, ``save_image``, etc.).

Mouse controls:

- **Left drag**:   Rotate the molecule.
- **Right drag**:  Zoom in / out.
- **Middle drag**: Pan the view.
- **Scroll wheel**: Increase / decrease label font size.
- **Left click**:  Select a single atom or bond.
- **Ctrl + Left click**: Toggle multi-selection.
- **Alt/Option + Left click**: Recentre pivot (middle-click alternative).
- **Middle click**: Recentre rotation pivot on clicked atom.
- **F1/F2/F3**: Align view to real-space axes a/b/c.
"""

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
    """Interactive Qt widget that renders a molecule as a 2-D projection.

    Supports ORTEP-style anisotropic displacement parameter (ADP) ellipsoids
    at 50 % probability level, isotropic spheres, and ball-and-stick
    representations.  The molecule can be rotated (left-drag), zoomed
    (right-drag), and panned (middle-drag) with the mouse.

    Typical usage::

        widget = MoleculeWidget(parent)
        widget.open_molecule(atoms=atom_list, cell=cell_params)

    :param parent: Optional parent widget.
    """

    _AUTO_ZOOM_PADDING = 1.1

    atomClicked = QtCore.Signal(str)
    bondClicked = QtCore.Signal(str, str)
    #: Emitted after every :meth:`open_molecule` / :meth:`grow_molecule` call
    #: with the frozenset of disorder-part numbers present in the loaded atoms.
    partsChanged = QtCore.Signal(object)

    def __init__(self, parent: QtGui.QWidget | None = None) -> None:
        # Qt base class must be initialised first so that update() and signals
        # are available before _init_renderer() is called.
        QtWidgets.QWidget.__init__(self, parent)
        self._init_renderer()

        # Widget-specific setup
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
        """Scale zoom proportionally so the molecule fills the same fraction
        of the viewport after resize."""
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

    def show_residual_density(self, hkl_path: str | Path | None = None,
                              level: float = 0.30) -> None:
        """3D-only feature; residual-density isosurfaces are not rendered in 2-D."""

    def clear_residual_density(self) -> None:
        """3D-only feature; clearing residual-density isosurfaces is a no-op in 2-D."""


# ---------------------------------------------------------------------------
# Backwards-compatibility re-exports
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
