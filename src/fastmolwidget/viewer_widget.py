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

from qtpy import QtWidgets

from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule2D import MoleculeWidget


class MoleculeViewerWidget(QtWidgets.QWidget):
    """A ready-to-use viewer widget that combines a :class:`MoleculeWidget`
    with a control bar (ADP toggle, label toggle, bond style, hydrogen
    visibility, bond-width adjustment, and difference density map controls).

    The interface is intentionally minimal: call :meth:`load_file` to display a
    structure, and :meth:`load_diff_map` to overlay the residual electron density.

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
        self._bond_type_checkbox = QtWidgets.QCheckBox("Round Bonds")
        self._hydrogens_checkbox = QtWidgets.QCheckBox("Show Hydrogens")
        self._grow_checkbox = QtWidgets.QCheckBox("Grow")

        self._bw_label = QtWidgets.QLabel("Bond Width:")
        self._bond_width_spinbox = QtWidgets.QSpinBox()
        self._bond_width_spinbox.setRange(1, 15)
        self._bond_width_spinbox.setValue(3)

        # default state
        self._adp_checkbox.setChecked(True)
        self._bond_type_checkbox.setChecked(True)
        self._hydrogens_checkbox.setChecked(True)

        # wire controls to renderer
        self._adp_checkbox.toggled.connect(self._render_widget.show_adps)
        self._label_checkbox.toggled.connect(self._render_widget.show_labels)
        self._bond_type_checkbox.toggled.connect(self._render_widget.show_round_bonds)
        self._hydrogens_checkbox.toggled.connect(self._render_widget.show_hydrogens)
        self._bond_width_spinbox.valueChanged.connect(self._render_widget.set_bond_width)
        self._grow_checkbox.toggled.connect(self._loader.set_grow)

        # apply initial defaults to the renderer
        self._render_widget.set_bond_width(3)
        self._render_widget.show_labels(False)
        self._render_widget.show_round_bonds(True)

        # ── density map controls ─────────────────────────────────────────────
        self._density_checkbox = QtWidgets.QCheckBox("Show Map")
        self._density_checkbox.setChecked(True)
        self._density_checkbox.setToolTip(
            "Toggle the difference electron density (Fo − Fc) wireframe map"
        )
        self._density_checkbox.toggled.connect(self._render_widget.show_density)

        self._density_level_label = QtWidgets.QLabel("Map σ:")
        self._density_level_spin = QtWidgets.QDoubleSpinBox()
        self._density_level_spin.setRange(1.0, 20.0)
        self._density_level_spin.setSingleStep(0.5)
        self._density_level_spin.setValue(3.0)
        self._density_level_spin.setDecimals(1)
        self._density_level_spin.setToolTip(
            "Contour level for the difference map expressed as multiples of σ"
        )

        self._load_map_button = QtWidgets.QPushButton("Load Map")
        self._load_map_button.setToolTip(
            "Compute and display the (Fo − Fc) difference electron density map "
            "from the currently loaded CIF and its associated HKL file"
        )
        self._load_map_button.clicked.connect(self._on_load_map_clicked)

        # ── layout ───────────────────────────────────────────────────────────
        control_bar = QtWidgets.QHBoxLayout()
        control_bar.addWidget(self._grow_checkbox)
        control_bar.addWidget(self._adp_checkbox)
        control_bar.addWidget(self._label_checkbox)
        control_bar.addWidget(self._bond_type_checkbox)
        control_bar.addWidget(self._hydrogens_checkbox)
        control_bar.addWidget(self._bw_label)
        control_bar.addWidget(self._bond_width_spinbox)
        control_bar.addStretch()

        # density controls in a second row
        density_bar = QtWidgets.QHBoxLayout()
        density_bar.addWidget(self._load_map_button)
        density_bar.addWidget(self._density_level_label)
        density_bar.addWidget(self._density_level_spin)
        density_bar.addWidget(self._density_checkbox)
        density_bar.addStretch()

        vl = QtWidgets.QVBoxLayout(self)
        vl.addWidget(self._render_widget)
        vl.addLayout(control_bar)
        vl.addLayout(density_bar)

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

    def load_diff_map(
        self,
        hkl_path: str | Path | None = None,
        level_sigma: float | None = None,
    ) -> None:
        """Compute and overlay the (Fo − Fc) residual electron density map.

        Must be called after :meth:`load_file` with a CIF file.  The method
        delegates to :meth:`~fastmolwidget.loader.MoleculeLoader.load_diff_map`.

        :param hkl_path: Explicit path to a SHELX4 HKL file, or ``None`` to
            auto-detect.
        :param level_sigma: Contour level in σ units (defaults to the value
            shown in the σ spinbox).
        """
        if level_sigma is None:
            level_sigma = self._density_level_spin.value()
        self._loader.load_diff_map(hkl_path=hkl_path, level_sigma=level_sigma)

    # ------------------------------------------------------------------
    # Private slots
    # ------------------------------------------------------------------

    def _on_load_map_clicked(self) -> None:
        """Slot: load the difference density map at the current sigma level."""
        level = self._density_level_spin.value()
        try:
            self._loader.load_diff_map(level_sigma=level)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Difference Map Error",
                f"Could not compute the difference electron density map:\n\n{exc}",
            )


if __name__ == '__main__':

    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    w = MoleculeViewerWidget()
    #w.load_file('../../tests/test-data/p31c.cif')
    w.load_file('../../tests/test-data/p31c-finalcif.res')
    w.show()
    app.exec()