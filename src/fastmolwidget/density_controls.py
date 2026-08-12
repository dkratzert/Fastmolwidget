"""Shared "Residual Density" control-bar behaviour for the viewer widgets.

:class:`DensityControlsMixin` owns the checkable *Residual Density* button and
the *Level* spin box, together with the logic that keeps them in step with the
renderer: finding the reflection data, reporting failures without taking the
host application down, and reflecting the on/off state in the button.

Both :class:`~fastmolwidget.viewer_widget.MoleculeViewerWidget` (2-D) and
:class:`~fastmolwidget.viewer_widget3D.MoleculeViewer3DWidget` (3-D) mix it in;
the underlying renderers implement the same API
(:meth:`~fastmolwidget.molecule_base.MoleculeWidgetProtocol.show_residual_density`
and friends), so the controls do not care which one they are driving.

Contract for the host widget
----------------------------
* Be a ``QWidget`` (the mixin parents dialogs and message boxes to it).
* Create ``self._render_widget`` **before** calling
  :meth:`~DensityControlsMixin._init_density_controls`.
* Add ``_residual_density_button``, ``_density_level_label`` and
  ``_density_level_spinbox`` to its control bar.
* Call :meth:`~DensityControlsMixin._sync_density_controls` after loading a
  file, so the controls follow a map that was dropped by the loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qtpy import QtCore, QtWidgets

try:
    from fastmolwidget.density import HAS_DENSITY_CPP
except ImportError:  # pragma: no cover - density support is optional
    HAS_DENSITY_CPP = False

if TYPE_CHECKING:
    # The mixin is only ever combined with a QWidget host, and it parents
    # dialogs to ``self``; saying so here keeps type checkers honest without
    # putting a second QWidget into the runtime MRO.
    _HostBase = QtWidgets.QWidget
else:
    _HostBase = object


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

__all__ = [
    'HAS_DENSITY_CPP',
    'DensityControlsMixin',
]


class DensityControlsMixin(_HostBase):
    """Residual-density button and level spin box for a viewer widget."""

    #: The renderer this control bar drives, created by the host widget before
    #: :meth:`_init_density_controls` is called.  Deliberately untyped: the
    #: hosts narrow it to their own concrete renderer class, and only the
    #: residual-density part of
    #: :class:`~fastmolwidget.molecule_base.MoleculeWidgetProtocol` is used
    #: here.
    _render_widget: Any

    def _init_density_controls(self) -> None:
        """Create and wire the density controls.

        Call from the host's ``__init__``, after ``self._render_widget`` has
        been created and before the control bar is laid out.
        """
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

        self._residual_density_button.toggled.connect(self._on_density_toggled)
        self._density_level_spinbox.valueChanged.connect(
            self._render_widget.set_residual_density_level
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
        self._density_level_spinbox.setValue(
            self._render_widget.residual_density_level)
        self._density_level_spinbox.blockSignals(False)
        self._set_density_controls_active(True)

    def clear_residual_density(self) -> None:
        """Remove the residual electron-density isosurface."""
        self._render_widget.clear_residual_density()
        self._set_density_controls_active(False)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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
