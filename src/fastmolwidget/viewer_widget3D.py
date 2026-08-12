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

from argparse import ArgumentParser
from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule3D import MoleculeWidget3D
from fastmolwidget.part_combo import PartFilterWidget

try:
    from fastmolwidget.density import HAS_DENSITY_CPP
except ImportError:
    HAS_DENSITY_CPP = False


#: Stylesheet for the checkable "Residual Density" button.  The checked state
#: gets the same green as the positive isosurface plus a sunken border, so it
#: is obvious whether density is currently displayed — relief alone is easy to
#: miss, and is barely visible in some Qt styles.
_DENSITY_BUTTON_STYLE = """
QPushButton:checked {
    background-color: #cdebcd;
    border: 2px inset #3c8c3c;
    font-weight: bold;
}
QPushButton:checked:hover {
    background-color: #bce0bc;
}
"""

_DENSITY_TOOLTIP_OFF = "Show the residual Fo-Fc density map."


class MoleculeViewer3DWidget(QtWidgets.QWidget):
    """A ready-to-use 3-D viewer that combines a :class:`MoleculeWidget3D`
    with a control bar.

    The control bar provides the same toggles as
    :class:`~fastmolwidget.viewer_widget.MoleculeViewerWidget`:

    * **Grow** – expand the asymmetric unit to complete molecules.
    * **Show ADP** – toggle ADP ellipsoid / sphere display.
    * **Show Labels** – toggle atom-name labels.
    * **Show Hydrogens** – toggle hydrogen visibility.
    * **Bond Width** – spinbox controlling cylinder radius.
    * **Bond Color** – button opening a color picker for all non-selected bonds.
    * **Reset Rotation Center** – restores the rotation pivot to the molecule's
      geometric centre (undoes a middle-click recentring).
    * **Residual Density** – checkable button; when pressed (shown sunken and
      tinted green) the Fo-Fc isosurfaces are displayed, clicking again hides
      them.
    * **Level** – spinbox changing the residual-density contour level; enabled
      only while density is shown.
    * **Parts** *Shown only when disorder parts are present)* –
      a checkable combo box listing every disorder-part number found in the
      loaded structure.  All parts are selected by default; unticking a part
      hides those atoms and their bonds.

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
        self._residual_density_button = QtWidgets.QPushButton("Residual Density")
        # Checkable so the button itself shows whether density is on: Qt draws
        # a checked QPushButton sunken, and the stylesheet adds a green tint on
        # top so the state is obvious at a glance and not only by relief.
        self._residual_density_button.setCheckable(True)
        self._residual_density_button.setStyleSheet(_DENSITY_BUTTON_STYLE)
        self._density_level_label = QtWidgets.QLabel("Level:")
        self._density_level_spinbox = QtWidgets.QDoubleSpinBox()
        self._density_level_spinbox.setRange(0.01, 9.99)
        self._density_level_spinbox.setSingleStep(0.01)
        self._density_level_spinbox.setDecimals(2)
        self._density_level_spinbox.setValue(0.30)
        self._density_level_spinbox.setSuffix(" e/Å³")
        # Nothing to contour until a map is loaded.
        self._density_level_spinbox.setEnabled(False)
        self._density_level_label.setEnabled(False)
        if HAS_DENSITY_CPP:
            self._residual_density_button.setToolTip(_DENSITY_TOOLTIP_OFF)
        else:
            self._residual_density_button.setEnabled(False)
            self._residual_density_button.setToolTip(
                "Residual density requires the compiled density_cpp extension."
            )

        # Initial checked state matches the renderer defaults
        # "Hide Hydrogens" unchecked → hydrogens are visible by default
        self._adp_checkbox.setChecked(True)
        self._hydrogens_checkbox.setChecked(False)

        # Wire controls to renderer
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
        self._residual_density_button.toggled.connect(self._on_density_toggled)
        self._density_level_spinbox.valueChanged.connect(self._render_widget.set_residual_density_level)
        self._grow_checkbox.toggled.connect(self._on_grow_toggled)
        self._pack_checkbox.toggled.connect(self._on_pack_toggled)

        # Apply initial defaults to the renderer
        self._render_widget.set_bond_width(3)
        self._render_widget.show_labels(False)

        # ── Part filter (Row 3) ───────────────────────────────────────────────
        self._part_widget = PartFilterWidget()
        self._part_widget.selectionChanged.connect(self._apply_part_filter)

        # React to partsChanged from the renderer (also fires on programmatic
        # open_molecule() calls from outside the viewer).
        self._render_widget.partsChanged.connect(self._update_part_controls)

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

        # Row 2: bond / view controls
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
        # Loading a different structure drops its residual density; make the
        # control bar follow whatever the renderer actually ended up with.
        self._sync_density_controls()

    def grow(self) -> None:
        """Grow the current structure to complete molecules.

        Expands the asymmetric unit using crystal symmetry (SDM algorithm).
        Deactivates Pack Unit Cell if it is currently enabled.
        No-op when no file has been loaded or the file has no symmetry (XYZ).
        """
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
        """Set the default colour used for non-selected 3-D bonds."""
        self._render_widget.set_bond_color(color)

    def show_residual_density(self, hkl_path: str | Path | None = None,
                              level: float | None = None) -> None:
        """Compute and show a residual electron-density map.

        The control-bar button is switched to its pressed (green) state and the
        Level spin box is set to the level actually used, so the view and the
        controls stay consistent when this is called from code.

        :param hkl_path: Path to the reflection file.  ``None`` finds the data
            automatically from the loaded model.
        :param level: Absolute contour level in e/Å³.  ``None`` contours at
            3σ of the map, which adapts to each structure.
        :raises RuntimeError: If no model is loaded or density support is unavailable.
        :raises FileNotFoundError: If no reflection data could be found.
        :raises ValueError: If the reflection data cannot be used.
        """
        self._render_widget.show_residual_density(hkl_path, level)
        # Show the level that was really used - it is computed from the map
        # when *level* is None.  Setting it must not trigger a re-contour.
        self._density_level_spinbox.blockSignals(True)
        self._density_level_spinbox.setValue(self._render_widget._density_level)
        self._density_level_spinbox.blockSignals(False)
        self._set_density_controls_active(True)

    def clear_residual_density(self) -> None:
        """Remove the residual electron-density isosurface."""
        self._render_widget.clear_residual_density()
        self._set_density_controls_active(False)

    def _on_density_toggled(self, checked: bool) -> None:
        """Show or hide the residual density when the button is toggled.

        The button is the single source of truth for the on/off state, so any
        failure (no model, no reflection data, or a cancelled file dialog)
        pops it back out again.
        """
        if not checked:
            self.clear_residual_density()
            return

        hkl_path = self._auto_reflection_file()
        if hkl_path is None:
            hkl_path = self._ask_for_reflection_file()
            if hkl_path is None:  # dialog cancelled
                self._set_density_button_checked(False)
                self._set_density_controls_active(False)
                return

        error_message = ""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            # level=None -> contour at 3 sigma of this particular map, and let
            # show_residual_density() put the resulting value in the spin box.
            self.show_residual_density(hkl_path)
        except Exception as exc:  # noqa: BLE001 - never take the host app down
            error_message = str(exc)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if error_message:
            self._set_density_button_checked(False)
            self._set_density_controls_active(False)
            QtWidgets.QMessageBox.warning(self, "Residual density", error_message)

    def _set_density_button_checked(self, checked: bool) -> None:
        """Set the button's checked state without re-entering the handler."""
        self._residual_density_button.blockSignals(True)
        self._residual_density_button.setChecked(checked)
        self._residual_density_button.blockSignals(False)

    def _set_density_controls_active(self, active: bool) -> None:
        """Reflect the on/off state in the button, tooltip and level spinbox.

        :param active: ``True`` when a density map is currently displayed.
        """
        self._set_density_button_checked(active)
        self._density_level_spinbox.setEnabled(active)
        self._density_level_label.setEnabled(active)
        if not HAS_DENSITY_CPP:
            return
        if active:
            self._update_residual_density_tooltip()
        else:
            self._residual_density_button.setToolTip(_DENSITY_TOOLTIP_OFF)

    def _sync_density_controls(self) -> None:
        """Match the density controls to the renderer's actual state.

        Used after operations that may drop the map behind the control bar's
        back — loading a different structure, for instance.
        """
        self._set_density_controls_active(
            self._render_widget.residual_density_map is not None
        )

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
        """Rebuild the Part combo whenever the renderer loads a new molecule.

        Hides the Part row when only one unique part value exists (no disorder).
        """
        self._part_widget.update_parts(parts)
        # All parts visible → pass None (skip per-atom set lookup in renderer).
        if len(parts) > 1:
            self._render_widget.set_visible_parts(None)

    def _apply_part_filter(self) -> None:
        """Forward the current combo selection to the renderer."""
        checked = set(self._part_widget.checked_values())
        # Pass None when everything is ticked — avoids per-atom set lookup.
        if checked == self._render_widget.available_parts:
            self._render_widget.set_visible_parts(None)
        else:
            self._render_widget.set_visible_parts(checked)

    def _choose_bond_color(self) -> None:
        """Open a colour picker for the bond colour."""
        current = QtGui.QColor.fromRgbF(*self._render_widget._bond_rgb)
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

    def _save_image_dialog(self) -> None:
        """Open a file dialog and save a screenshot via :meth:`save_image`.

        The current label visibility state is preserved as-is; labels appear
        in the screenshot only if they are active at the time of saving.
        """
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Image",
            "",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All Files (*)",
        )
        if path:
            self._render_widget.save_image(Path(path))

    def _auto_reflection_file(self) -> Path | None:
        """Return the model file when it carries its own reflection data.

        Deliberately does *not* look at sibling files: picking up a separate
        ``.hkl`` silently would hide which dataset is being used, so that case
        goes through the file dialog instead.

        :returns: The model path when it contains reflections, else ``None``.
        """
        model_path = getattr(self._render_widget, "_model_path", None)
        if model_path is None:
            return None
        try:
            from fastmolwidget.hkl_io import has_reflections

            model_path = Path(model_path)
            if model_path.suffix.lower() != ".hkl" and has_reflections(model_path):
                return model_path
        except Exception:  # noqa: BLE001 - fall back to asking the user
            return None
        return None

    def _ask_for_reflection_file(self) -> Path | None:
        """Last resort: let the user pick a reflection file."""
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Reflection File",
            self._residual_density_start_path(),
            "Reflection files (*.hkl *.fcf *.cif);;SHELX HKL (*.hkl);;All files (*)",
        )
        return Path(path) if path else None

    def _residual_density_start_path(self) -> str:
        """Return the best starting path for the residual-density file dialog."""
        model_path = getattr(self._render_widget, "_model_path", None)
        if model_path is None:
            return ""

        model_path = Path(model_path)
        hkl_path = model_path.with_suffix(".hkl")
        if hkl_path.exists():
            return str(hkl_path)
        return str(model_path.parent)

    def _update_residual_density_tooltip(self) -> None:
        """Show the map statistics on the button while density is displayed."""
        density_map = self._render_widget.residual_density_map
        if density_map is None:
            return
        self._residual_density_button.setToolTip(
            f"Residual density shown - click to hide.\n"
            f"max {density_map.max:+.3f}, min {density_map.min:+.3f}, "
            f"rms {density_map.rms:.3f} e/Å³"
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    parse = ArgumentParser(description="Test the 3-D molecule viewer widget with a sample CIF file.")
    parse.add_argument("cif_file", nargs="?", default=None, help="Path to a CIF file to load (optional).")
    args = parse.parse_args()

    w = MoleculeViewer3DWidget()
    # Path is relative to the repository root; adjust as needed for your setup
    # w.load_file(Path("tests/test-data/p31c.cif"))
    w.load_file('tests/test-data/p21c.cif')
    # w.load_file('tests/test-data/1000007.cif')
    # w.load_file('tests/test-data/1548072_many_atoms.cif')
    # w.load_file(Path('tests/test-data/4060314.cif'))
    # w.load_file(Path('tests/test-data/1979688_small.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1367_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1368_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1369_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1370_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1371_ESM.cif'))
    # w.load_file(Path('tests/test-data/41467_2015_BFncomms9288_MOESM1372_ESM.cif'))
    # w.load_file(Path('tests/test-data/IKmjs421_2_0m_sump.res'))
    if args.cif_file:
        w.load_file(args.cif_file)
    w.show()
    w.grow()
    # app.processEvents()
    w.showMaximized()
    # w.render_widget.set_visible_parts({0, 1})
    app.exec()
