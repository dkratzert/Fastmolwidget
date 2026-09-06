"""Ready-to-use 2-D viewer widget."""

from __future__ import annotations

from pathlib import Path

from qtpy import QtGui, QtWidgets

from fastmolwidget.density_controls import DensityControlsMixin
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.part_combo import PartFilterWidget


class MoleculeViewerWidget(DensityControlsMixin, QtWidgets.QWidget):
    """2-D :class:`MoleculeWidget` plus its control bar."""

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
        self._best_view_button = QtWidgets.QPushButton("Best View")
        self._open_file_button = QtWidgets.QPushButton("Open File…")
        self._save_image_button = QtWidgets.QPushButton("Save Image…")
        self._init_density_controls()

        # "Hide Hydrogens" unchecked -> visible by default.
        self._adp_checkbox.setChecked(True)
        self._hydrogens_checkbox.setChecked(False)

        # Wire controls to the renderer.
        self._adp_checkbox.toggled.connect(self._render_widget.show_adps)
        self._label_checkbox.toggled.connect(self._render_widget.show_labels)
        self._hydrogens_checkbox.toggled.connect(
            lambda checked: self._render_widget.show_hydrogens(not checked)
        )
        self._bond_width_spinbox.valueChanged.connect(self._render_widget.set_bond_width)
        self._bond_color_button.clicked.connect(self._choose_bond_color)
        self._open_file_button.clicked.connect(self._open_file_dialog)
        self._reset_center_button.clicked.connect(self._render_widget.reset_rotation_center)
        self._best_view_button.clicked.connect(self._render_widget.align_best_view)
        self._save_image_button.clicked.connect(self._save_image_dialog)
        self._grow_checkbox.toggled.connect(self._on_grow_toggled)
        self._pack_checkbox.toggled.connect(self._on_pack_toggled)

        # Apply initial defaults.
        self._render_widget.set_bond_width(3)
        self._render_widget.show_labels(False)

        # ── Part filter ───────────────────────────────────────────────────────
        self._part_widget = PartFilterWidget()
        self._part_widget.selectionChanged.connect(self._apply_part_filter)

        self._render_widget.partsChanged.connect(self._update_part_controls)

        # ── layout ───────────────────────────────────────────────────────────
        # Row 1: structure toggles.
        control_bar = QtWidgets.QHBoxLayout()
        control_bar.addWidget(self._open_file_button)
        control_bar.addWidget(self._grow_checkbox)
        control_bar.addWidget(self._pack_checkbox)
        control_bar.addWidget(self._adp_checkbox)
        control_bar.addWidget(self._label_checkbox)
        control_bar.addWidget(self._hydrogens_checkbox)
        control_bar.addStretch()

        # Row 2: bond controls.
        control_bar2 = QtWidgets.QHBoxLayout()
        control_bar2.addWidget(self._bw_label)
        control_bar2.addWidget(self._bond_width_spinbox)
        control_bar2.addWidget(self._bond_color_button)
        control_bar2.addWidget(self._reset_center_button)
        control_bar2.addWidget(self._best_view_button)
        control_bar2.addWidget(self._save_image_button)
        control_bar2.addWidget(self._residual_density_button)
        control_bar2.addWidget(self._density_level_label)
        control_bar2.addWidget(self._density_level_spinbox)
        control_bar2.addWidget(self._part_widget)
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
        """Underlying :class:`MoleculeWidget`."""
        return self._render_widget

    def load_file(self, filename: str | Path) -> None:
        """Load and display a structure file."""
        self._loader.load_file(filename)
        self.setWindowTitle(str(Path(filename).resolve()))
        # Keep the controls in sync if a new model cleared density.
        self._sync_density_controls()

    def grow(self) -> None:
        """Grow the current structure to full molecules."""
        if self._pack_checkbox.isChecked():
            self._pack_checkbox.blockSignals(True)
            self._pack_checkbox.setChecked(False)
            self._pack_checkbox.blockSignals(False)
            self._loader.set_pack(False)
        self._grow_checkbox.blockSignals(True)
        self._grow_checkbox.setChecked(True)
        self._grow_checkbox.blockSignals(False)
        self._loader.set_grow(True)

    def set_bond_color(
        self,
        color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set the default color for non-selected bonds."""
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
        if checked:
            self._render_widget.reset_rotation_center()
            self._render_widget._align_to_reciprocal_axis(1)

    def _update_part_controls(self, parts: frozenset[int]) -> None:
        """Refresh the Part filter after a load."""
        self._part_widget.update_parts(parts)
        if len(parts) > 1:
            self._render_widget.set_visible_parts(None)

    def _apply_part_filter(self) -> None:
        """Apply the current Part filter."""
        checked = set(self._part_widget.checked_values())
        if checked == self._render_widget.available_parts:
            self._render_widget.set_visible_parts(None)
        else:
            self._render_widget.set_visible_parts(checked)

    def _choose_bond_color(self) -> None:
        """Open a color picker for bonds."""
        current = QtGui.QColor(self._render_widget.bond_color)
        color = QtWidgets.QColorDialog.getColor(current, self, "Choose Bond Color")
        if color.isValid():
            self._render_widget.set_bond_color(color)

    def _open_file_dialog(self) -> None:
        """Open a file dialog and load the chosen structure."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Structure File",
            "",
            "Structure Files (*.cif *.res *.ins *.xyz);;All Files (*)",
        )
        if path:
            self.load_file(path)

    def _save_image_dialog(self) -> None:
        """Open a file dialog and save the current view."""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self._render_widget.save_image(Path(path))


if __name__ == '__main__':

    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    w = MoleculeViewerWidget()
    # w.load_file('tests/test-data/p31c.cif')
    # w.load_file('tests/test-data/p31c-finalcif.res')
    # w.load_file('tests/test-data/1548072_many_atoms.cif')
    # w.load_file('tests/test-data/p21c.cif')
    # w.load_file("tests\\test-data\\nospera2.cif")
    w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1369_ESM.cif'))
    # w.load_file(Path('tests/test-data/IKmjs421_2_0m_sump.res'))
    w.grow()
    w._reset_center_button.click()
    w.show()
    w.showMaximized()
    # w.render_widget.set_visible_parts({0,1})
    app.exec()
