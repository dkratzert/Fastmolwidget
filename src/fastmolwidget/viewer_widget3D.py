"""Ready-to-use 3-D viewer widget."""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

from qtpy import QtGui, QtWidgets

from fastmolwidget.density_controls import DensityControlsMixin
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule3D import MoleculeWidget3D
from fastmolwidget.part_combo import PartFilterWidget


class MoleculeViewer3DWidget(DensityControlsMixin, QtWidgets.QWidget):
    """3-D :class:`MoleculeWidget3D` plus its control bar."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # ── molecule renderer ────────────────────────────────────────────────
        self._render_widget = MoleculeWidget3D()
        # MoleculeLoader only needs the open_molecule() API.
        self._loader = MoleculeLoader(self._render_widget)  # type: ignore[arg-type]

        # ── control bar ──────────────────────────────────────────────────────
        self._grow_checkbox = QtWidgets.QCheckBox("Grow")
        self._pack_checkbox = QtWidgets.QCheckBox("Pack Unit Cell")
        self._adp_checkbox = QtWidgets.QCheckBox("Show ADP")
        self._label_checkbox = QtWidgets.QCheckBox("Show Labels")
        self._hydrogens_checkbox = QtWidgets.QCheckBox("Hide Hydrogens")

        self._bw_label = QtWidgets.QLabel("Bond Width:")
        self._bond_width_spinbox = QtWidgets.QSpinBox()
        self._bond_width_spinbox.setRange(0, 15)
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
        self._reset_center_button.clicked.connect(self._render_widget.reset_rotation_center)
        self._best_view_button.clicked.connect(self._render_widget.align_best_view)
        self._open_file_button.clicked.connect(self._open_file_dialog)
        self._save_image_button.clicked.connect(self._save_image_dialog)
        self._grow_checkbox.toggled.connect(self._on_grow_toggled)
        self._pack_checkbox.toggled.connect(self._on_pack_toggled)

        # Apply initial defaults.
        self._render_widget.set_bond_width(3)
        self._render_widget.show_labels(False)

        # ── Part filter ───────────────────────────────────────────────────────
        self._part_widget = PartFilterWidget()
        self._part_widget.selectionChanged.connect(self._apply_part_filter)

        # Also handles programmatic open_molecule() calls.
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

        # Row 2: bond and view controls.
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
    def render_widget(self) -> MoleculeWidget3D:
        """Underlying :class:`MoleculeWidget3D`."""
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
        """Set the default color for non-selected 3-D bonds."""
        self._render_widget.set_bond_color(color)

    def show_residual_density(self, hkl_path: str | Path | None = None,
                              level: float | None = None) -> None:
        """Show a residual-density map."""
        super().show_residual_density(hkl_path, level)

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
        # None means all parts.
        if len(parts) > 1:
            self._render_widget.set_visible_parts(None)

    def _apply_part_filter(self) -> None:
        """Apply the current Part filter."""
        checked = set(self._part_widget.checked_values())
        # None means all parts.
        if checked == self._render_widget.available_parts:
            self._render_widget.set_visible_parts(None)
        else:
            self._render_widget.set_visible_parts(checked)

    def _choose_bond_color(self) -> None:
        """Open a color picker for bonds."""
        current = QtGui.QColor.fromRgbF(*self._render_widget._bond_rgb)
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


if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    parse = ArgumentParser(description="Test the 3-D molecule viewer widget with a sample CIF file.")
    parse.add_argument("cif_file", nargs="?", default=None, help="Path to a CIF file to load (optional).")
    args = parse.parse_args()

    w = MoleculeViewer3DWidget()
    # Path is relative to the repository root.
    # w.load_file(Path("tests/test-data/p31c.cif"))
    # w.load_file(r"D:\frames\CK-B874-finalcif.cif")
    # w.load_file('tests/test-data/1000007.cif')
    # w.load_file('tests/test-data/p21c.cif')
    # w.load_file('tests/test-data/1548072_many_atoms.cif')
    # w.load_file(Path('tests/test-data/4060314.cif'))
    # w.load_file(Path('tests/test-data/1979688_small.cif'))
    w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1367_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1368_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1369_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1370_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1371_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1372_ESM.cif'))
    # w.load_file(Path('tests/test-data/IKmjs421_2_0m_sump.res'))
    # w.load_file("tests\\test-data\\nospera2.cif")
    if args.cif_file:
        w.load_file(args.cif_file)
    w.show()
    w.grow()
    # app.processEvents()
    w.showMaximized()
    # w.render_widget.set_visible_parts({0, 1})
    app.exec()
