"""
A self-contained QWidget that embeds a :class:`~fastmolwidget.molecule2D.MoleculeWidget`
together with its control bar.

Usage::

    viewer = MoleculeViewerWidget()
    viewer.load_file("structure.cif")
    viewer.show()

Or pass the file name directly to the constructor::

    viewer = MoleculeViewerWidget("structure.cif")
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
    visibility, and bond-width adjustment).

    The interface is intentionally minimal: provide a *filename* to display a
    structure immediately, or call :meth:`load_file` at any later time.

    :param filename: Optional path to a structure file (``.cif``, ``.res``,
        ``.ins``, or ``.xyz``).  Passed straight to :meth:`load_file`.
    :param parent: Optional parent widget.
    """

    def __init__(self, filename: str | Path | None = None, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # ── molecule renderer ────────────────────────────────────────────────
        self._render_widget = MoleculeWidget()
        self._loader = MoleculeLoader(self._render_widget)

        # ── control bar ──────────────────────────────────────────────────────
        self._adp_checkbox = QtWidgets.QCheckBox("Show ADP")
        self._label_checkbox = QtWidgets.QCheckBox("Show Labels")
        self._bond_type_checkbox = QtWidgets.QCheckBox("Round Bonds")
        self._hydrogens_checkbox = QtWidgets.QCheckBox("Show Hydrogens")

        self._bw_label = QtWidgets.QLabel("Bond Width:")
        self._bond_width_spinbox = QtWidgets.QSpinBox()
        self._bond_width_spinbox.setRange(1, 15)
        self._bond_width_spinbox.setValue(3)

        # default state
        self._adp_checkbox.setChecked(True)
        self._bond_type_checkbox.setChecked(True)
        self._hydrogens_checkbox.setChecked(True)

        # wire controls to renderer
        self._adp_checkbox.toggled.connect(self._render_widget.show_adp)
        self._label_checkbox.toggled.connect(self._render_widget.show_labels)
        self._bond_type_checkbox.toggled.connect(self._render_widget.show_round_bonds)
        self._hydrogens_checkbox.toggled.connect(self._render_widget.show_hydrogens)
        self._bond_width_spinbox.valueChanged.connect(self._render_widget.set_bond_width)

        # apply initial defaults to the renderer
        self._render_widget.set_bond_width(3)
        self._render_widget.show_labels(False)
        self._render_widget.show_round_bonds(True)

        # ── layout ───────────────────────────────────────────────────────────
        control_bar = QtWidgets.QHBoxLayout()
        control_bar.addWidget(self._adp_checkbox)
        control_bar.addWidget(self._label_checkbox)
        control_bar.addWidget(self._bond_type_checkbox)
        control_bar.addWidget(self._hydrogens_checkbox)
        control_bar.addWidget(self._bw_label)
        control_bar.addWidget(self._bond_width_spinbox)
        control_bar.addStretch()

        vl = QtWidgets.QVBoxLayout(self)
        vl.addWidget(self._render_widget)
        vl.addLayout(control_bar)

        # ── optional initial file ─────────────────────────────────────────────
        if filename is not None:
            self.load_file(filename)

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

if __name__ == '__main__':

    app = QtWidgets.QApplication.instance()
    if not app:
        app = QtWidgets.QApplication([])

    w = MoleculeViewerWidget()
    w.load_file('../../tests/test-data/1979688_small.cif')
    w.show()
    app.exec()