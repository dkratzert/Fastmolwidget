"""
A self-contained QWidget that embeds a :class:`~fastmolwidget.molecule2D.MoleculeWidget`
together with its control bar.

Usage::

    viewer = MoleculeViewerWidget()
    viewer.load_file("structure.cif")
    viewer.show()

"""

from __future__ import annotations

from pathlib import Path

from qtpy import QtGui, QtWidgets

from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget


class MoleculeViewerWidget(QtWidgets.QWidget):
    """A ready-to-use viewer widget that combines a :class:`MoleculeWidget`
    with a control bar.

    The control bar provides the following controls:

    * **Grow** – expand the asymmetric unit to complete molecules.
    * **Show ADP** – toggle ADP ellipsoid / sphere display.
    * **Show Labels** – toggle atom-name labels.
    * **Hide Hydrogens** – toggle hydrogen visibility.
    * **Bond Width** – spinbox controlling bond width (second row).
    * **Bond Color** – button opening a color picker for all non-selected bonds (second row).

    The interface is intentionally minimal: call :meth:`load_file` to display a
    structure.

    :param parent: Optional parent widget.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # ── molecule renderer ────────────────────────────────────────────────
        self._render_widget = MoleculeWidget()
        self._loader = MoleculeLoader(self._render_widget)

        # ── control bar ──────────────────────────────────────────────────────
        self._adp_checkbox = QtWidgets.QCheckBox("Show ADP")
        self._label_checkbox = QtWidgets.QCheckBox("Show Labels")
        self._hydrogens_checkbox = QtWidgets.QCheckBox("Hide Hydrogens")
        self._grow_checkbox = QtWidgets.QCheckBox("Grow")
        self._pack_checkbox = QtWidgets.QCheckBox("Pack Unit Cell")

        self._bw_label = QtWidgets.QLabel("Bond Width:")
        self._bond_width_spinbox = QtWidgets.QSpinBox()
        self._bond_width_spinbox.setRange(1, 15)
        self._bond_width_spinbox.setValue(3)
        self._bond_color_button = QtWidgets.QPushButton("Bond Color…")
        self._reset_center_button = QtWidgets.QPushButton("Reset Rotation Center")
        self._open_file_button = QtWidgets.QPushButton("Open File…")

        # default state
        # "Hide Hydrogens" unchecked → hydrogens are visible by default
        self._adp_checkbox.setChecked(True)
        self._hydrogens_checkbox.setChecked(False)

        # wire controls to renderer
        self._adp_checkbox.toggled.connect(self._render_widget.show_adps)
        self._label_checkbox.toggled.connect(self._render_widget.show_labels)
        self._hydrogens_checkbox.toggled.connect(
            lambda checked: self._render_widget.show_hydrogens(not checked)
        )
        self._bond_width_spinbox.valueChanged.connect(self._render_widget.set_bond_width)
        self._bond_color_button.clicked.connect(self._choose_bond_color)
        self._open_file_button.clicked.connect(self._open_file_dialog)
        self._reset_center_button.clicked.connect(self._render_widget.reset_rotation_center)
        self._grow_checkbox.toggled.connect(self._on_grow_toggled)
        self._pack_checkbox.toggled.connect(self._on_pack_toggled)

        # apply initial defaults to the renderer
        self._render_widget.set_bond_width(3)
        self._render_widget.show_labels(False)

        # ── layout ───────────────────────────────────────────────────────────
        # Row 1: structure toggles
        control_bar = QtWidgets.QHBoxLayout()
        control_bar.addWidget(self._open_file_button)
        control_bar.addWidget(self._grow_checkbox)
        control_bar.addWidget(self._pack_checkbox)
        control_bar.addWidget(self._adp_checkbox)
        control_bar.addWidget(self._label_checkbox)
        control_bar.addWidget(self._hydrogens_checkbox)
        control_bar.addStretch()

        # Row 2: bond controls
        control_bar2 = QtWidgets.QHBoxLayout()
        control_bar2.addWidget(self._bw_label)
        control_bar2.addWidget(self._bond_width_spinbox)
        control_bar2.addWidget(self._bond_color_button)
        control_bar2.addWidget(self._reset_center_button)
        control_bar2.addStretch()

        vl = QtWidgets.QVBoxLayout(self)
        vl.addWidget(self._render_widget)
        vl.addLayout(control_bar)
        vl.addLayout(control_bar2)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def render_widget(self) -> MoleculeWidget:
        """The underlying :class:`MoleculeWidget` (read-only)."""
        return self._render_widget

    def load_file(self, filename: str | Path) -> None:
        """Load a structure file and display it.

        The file format is determined from the extension (``.cif``, ``.res``,
        ``.ins``, ``.xyz``).

        :param filename: Path to the structure file.
        :raises ValueError: If the file format is not supported.
        :raises FileNotFoundError: If the file does not exist.
        """
        self._loader.load_file(filename)

    def set_bond_color(
        self,
        color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set the default colour used for non-selected bonds."""
        self._render_widget.set_bond_color(color)

    def _on_grow_toggled(self, checked: bool) -> None:
        """Activate grow mode; deactivate pack mode when grow is switched on."""
        if checked and self._pack_checkbox.isChecked():
            self._pack_checkbox.blockSignals(True)
            self._pack_checkbox.setChecked(False)
            self._pack_checkbox.blockSignals(False)
            self._loader.set_pack(False)
        self._loader.set_grow(checked)

    def _on_pack_toggled(self, checked: bool) -> None:
        """Activate pack mode; deactivate grow mode when pack is switched on."""
        if checked and self._grow_checkbox.isChecked():
            self._grow_checkbox.blockSignals(True)
            self._grow_checkbox.setChecked(False)
            self._grow_checkbox.blockSignals(False)
            self._loader.set_grow(False)
        self._loader.set_pack(checked)

    def _choose_bond_color(self) -> None:
        """Open a colour picker for the bond colour."""
        current = QtGui.QColor(self._render_widget.bond_color)
        color = QtWidgets.QColorDialog.getColor(current, self, "Choose Bond Color")
        if color.isValid():
            self._render_widget.set_bond_color(color)

    def _open_file_dialog(self) -> None:
        """Open a file dialog to select and load a structure file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Structure File",
            "",
            "Structure Files (*.cif *.res *.ins *.xyz);;All Files (*)",
        )
        if path:
            self.load_file(path)

if __name__ == '__main__':

    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    w = MoleculeViewerWidget()
    #w.load_file('../../tests/test-data/p31c.cif')
    #w.load_file('../../tests/test-data/p31c-finalcif.res')
    #w.load_file('../../tests/test-data/1548072_many_atoms.cif')
    w.load_file('../../tests/test-data/p21c.cif')
    w.show()
    w.showMaximized()
    app.exec()