"""
A self-contained QWidget that embeds a :class:`~fastmolwidget.molecule3D.MoleculeWidget3D`
together with its control bar.

The layout and controls are identical to
:class:`~fastmolwidget.viewer_widget.MoleculeViewerWidget` (the 2-D variant).

Usage::

    viewer = MoleculeViewer3DWidget()
    viewer.load_file("structure.cif")
    viewer.show()

"""

from __future__ import annotations

from pathlib import Path

from qtpy import QtGui, QtWidgets

from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule3D import MoleculeWidget3D


class MoleculeViewer3DWidget(QtWidgets.QWidget):
    """A ready-to-use 3-D viewer that combines a :class:`MoleculeWidget3D`
    with a control bar.

    The control bar provides the same toggles as
    :class:`~fastmolwidget.viewer_widget.MoleculeViewerWidget`:

    * **Grow** – expand the asymmetric unit to complete molecules.
    * **Show ADP** – toggle ADP ellipsoid / sphere display.
    * **Show Labels** – toggle atom-name labels.
    * **Round Bonds** – switch between round (8-segment) and angular (4-segment)
      cylinders.
    * **Show Hydrogens** – toggle hydrogen visibility.
    * **Bond Width** – spinbox controlling cylinder radius.
    * **Bond Color** – button opening a color picker for all non-selected bonds.

    The loader (:class:`~fastmolwidget.loader.MoleculeLoader`) is identical to
    the 2-D widget so all supported file formats (CIF, SHELX .res/.ins, XYZ)
    work without modification.

    :param parent: Optional parent widget.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # ── molecule renderer ────────────────────────────────────────────────
        self._render_widget = MoleculeWidget3D()
        # MoleculeLoader accepts any widget with the open_molecule() API
        self._loader = MoleculeLoader(self._render_widget)  # type: ignore[arg-type]

        # ── control bar ──────────────────────────────────────────────────────
        self._grow_checkbox = QtWidgets.QCheckBox("Grow")
        self._adp_checkbox = QtWidgets.QCheckBox("Show ADP")
        self._label_checkbox = QtWidgets.QCheckBox("Show Labels")
        self._bond_type_checkbox = QtWidgets.QCheckBox("Round Bonds")
        self._hydrogens_checkbox = QtWidgets.QCheckBox("Show Hydrogens")

        self._bw_label = QtWidgets.QLabel("Bond Width:")
        self._bond_width_spinbox = QtWidgets.QSpinBox()
        self._bond_width_spinbox.setRange(1, 15)
        self._bond_width_spinbox.setValue(3)
        self._bond_color_button = QtWidgets.QPushButton("Bond Color…")

        # Initial checked state matches the renderer defaults
        self._adp_checkbox.setChecked(True)
        self._bond_type_checkbox.setChecked(True)
        self._hydrogens_checkbox.setChecked(True)

        # Wire controls to renderer
        self._adp_checkbox.toggled.connect(self._render_widget.show_adps)
        self._label_checkbox.toggled.connect(self._render_widget.show_labels)
        self._bond_type_checkbox.toggled.connect(self._render_widget.show_round_bonds)
        self._hydrogens_checkbox.toggled.connect(self._render_widget.show_hydrogens)
        self._bond_width_spinbox.valueChanged.connect(self._render_widget.set_bond_width)
        self._bond_color_button.clicked.connect(self._choose_bond_color)
        self._grow_checkbox.toggled.connect(self._loader.set_grow)

        # Apply initial defaults to the renderer
        self._render_widget.set_bond_width(3)
        self._render_widget.show_labels(False)
        self._render_widget.show_round_bonds(True)

        # ── layout ───────────────────────────────────────────────────────────
        control_bar = QtWidgets.QHBoxLayout()
        control_bar.addWidget(self._grow_checkbox)
        control_bar.addWidget(self._adp_checkbox)
        control_bar.addWidget(self._label_checkbox)
        control_bar.addWidget(self._bond_type_checkbox)
        control_bar.addWidget(self._hydrogens_checkbox)
        control_bar.addWidget(self._bw_label)
        control_bar.addWidget(self._bond_width_spinbox)
        control_bar.addWidget(self._bond_color_button)
        control_bar.addStretch()

        vl = QtWidgets.QVBoxLayout(self)
        vl.addWidget(self._render_widget)
        vl.addLayout(control_bar)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def render_widget(self) -> MoleculeWidget3D:
        """The underlying :class:`MoleculeWidget3D` (read-only)."""
        return self._render_widget

    def load_file(self, filename: str | Path) -> None:
        """Load a structure file and display it in 3-D.

        Supported formats: ``.cif``, ``.res``, ``.ins``, ``.xyz``.

        :param filename: Path to the structure file.
        :raises ValueError: If the file format is not supported.
        :raises FileNotFoundError: If the file does not exist.
        """
        self._loader.load_file(filename)

    def set_bond_color(
        self,
        color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set the default colour used for non-selected 3-D bonds."""
        self._render_widget.set_bond_color(color)

    def _choose_bond_color(self) -> None:
        """Open a colour picker for the bond colour."""
        current = QtGui.QColor.fromRgbF(*self._render_widget._bond_rgb)
        color = QtWidgets.QColorDialog.getColor(current, self, "Choose Bond Color")
        if color.isValid():
            self._render_widget.set_bond_color(color)


if __name__ == "__main__":

    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])


    w = MoleculeViewer3DWidget()
    # Path is relative to the repository root; adjust as needed for your setup
    #w.load_file(Path(__file__).parent.parent.parent / "tests" / "test-data" / "p31c.cif")
    w.load_file('../../tests/test-data/p21c.cif')
    #w.load_file('../../tests/test-data/1548072_many_atoms.cif')
    w.show()
    app.exec()
