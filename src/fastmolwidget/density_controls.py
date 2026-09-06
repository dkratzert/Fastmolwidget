"""Shared residual-density controls for the viewer widgets.

Host contract:
* be a ``QWidget``;
* create ``self._render_widget`` before :meth:`_init_density_controls`;
* add the button, label and spin box to the control bar;
* call :meth:`_sync_density_controls` after loading a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qtpy import QtCore, QtWidgets

from fastmolwidget.molecule_base import (
    DENSITY_LEVEL_MAX,
    DENSITY_LEVEL_MIN,
    DENSITY_LEVEL_STEP,
)

try:
    from fastmolwidget.density import HAS_DENSITY_CPP
except ImportError:  # pragma: no cover - density support is optional
    HAS_DENSITY_CPP = False

if TYPE_CHECKING:
    # Runtime host is always a QWidget; this keeps type checkers happy without
    # adding QWidget twice to the MRO.
    _HostBase = QtWidgets.QWidget
else:
    _HostBase = object


#: Checked-state styling for the residual-density button.
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

_DENSITY_TOOLTIP_NO_DATA = ("No usable reflection data for this structure - "
                            "residual density is unavailable.")

__all__ = [
    'HAS_DENSITY_CPP',
    'DensityControlsMixin',
    'ResidualDensityControls',
    'ask_for_reflection_file',
    'auto_reflection_file',
    'density_statistics_text',
    'reflection_file_start_path',
]


# ---------------------------------------------------------------------------
# UI-toolkit-agnostic helpers, shared with the Qt Quick backend
# ---------------------------------------------------------------------------

def auto_reflection_file(render_widget: Any) -> object | None:
    """Return the renderer's declared or embedded reflection source.

    Sibling files are not searched here; that should stay explicit via the file
    dialog.
    """
    declared = getattr(render_widget, "reflection_source", None)
    if declared is not None:
        return declared

    model = getattr(render_widget, "model_source", None)
    if model is None:
        model = getattr(render_widget, "_model_path", None)
    if model is None:
        return None
    try:
        from fastmolwidget.hkl_io import has_reflections

        if isinstance(model, (str, Path)):
            model = Path(model)
            if model.suffix.lower() == ".hkl":
                return None
        return model if has_reflections(model) else None
    except Exception:  # noqa: BLE001 - fall back to asking the user
        return None


def reflection_file_start_path(render_widget: Any) -> str:
    """Return the best starting path for the reflection-file dialog."""
    model_path = getattr(render_widget, "_model_path", None)
    if model_path is None:
        return ""

    model_path = Path(model_path)
    hkl_path = model_path.with_suffix(".hkl")
    if hkl_path.exists():
        return str(hkl_path)
    return str(model_path.parent)


def ask_for_reflection_file(parent: Any, render_widget: Any) -> Path | None:
    """Last resort: ask the user for a reflection file."""
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        parent,
        "Open Reflection File",
        reflection_file_start_path(render_widget),
        "Reflection files (*.hkl *.fcf *.cif);;SHELX HKL (*.hkl);;All files (*)",
    )
    return Path(path) if path else None


def density_statistics_text(density_map: Any) -> str:
    """One-line ``max / min / rms`` summary of *density_map* for a tooltip."""
    if density_map is None:
        return _DENSITY_TOOLTIP_OFF
    return (
        f"Residual density shown - click to hide.\n"
        f"max {density_map.max:+.3f}, min {density_map.min:+.3f}, "
        f"rms {density_map.rms:.3f} e/Å³"
    )


class DensityControlsMixin(_HostBase):
    """Residual-density button and level spin box for a viewer widget."""

    #: Renderer driven by these controls. Hosts narrow the type themselves.
    _render_widget: Any

    #: Whether missing reflection data may trigger a file dialog.
    allow_reflection_dialog: bool = True

    def _init_density_controls(self) -> None:
        """Create and connect the density controls."""
        self._residual_density_button = QtWidgets.QPushButton("Residual Density")
        # Keep the on/off state visible on the button itself.
        self._residual_density_button.setCheckable(True)
        self._residual_density_button.setStyleSheet(_DENSITY_BUTTON_STYLE)

        self._density_level_label = QtWidgets.QLabel("Level:")
        self._density_level_spinbox = QtWidgets.QDoubleSpinBox()
        self._density_level_spinbox.setRange(DENSITY_LEVEL_MIN, DENSITY_LEVEL_MAX)
        self._density_level_spinbox.setSingleStep(DENSITY_LEVEL_STEP)
        self._density_level_spinbox.setDecimals(2)
        self._density_level_spinbox.setValue(0.30)
        self._density_level_spinbox.setSuffix(" e/Å³")
        self._density_level_spinbox.setToolTip(
            "Residual-density contour level.\n"
            "Ctrl + mouse wheel over the structure changes it too."
        )
        # Disabled until a map is loaded.
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
        # Keep the spin box in sync with Ctrl+wheel changes in the view.
        self._render_widget.densityLevelChanged.connect(self._on_density_level_changed)

    def _on_density_level_changed(self, level: float) -> None:
        """Mirror a view-driven level change into the spin box."""
        self._density_level_spinbox.blockSignals(True)
        self._density_level_spinbox.setValue(level)
        self._density_level_spinbox.blockSignals(False)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_residual_density(self, hkl_path: object | None = None,
                              level: float | None = None) -> None:
        """Compute and show a residual electron-density map.

        ``level=None`` contours at 3σ of that map and updates the controls to
        the level actually used.
        """
        self._render_widget.show_residual_density(hkl_path, level)
        # ``level=None`` resolves to a map-specific absolute level.
        self._density_level_spinbox.blockSignals(True)
        self._density_level_spinbox.setValue(
            self._render_widget.residual_density_level)
        self._density_level_spinbox.blockSignals(False)
        self._set_density_controls_active(True)

    def clear_residual_density(self) -> None:
        """Remove the residual electron-density isosurface."""
        self._render_widget.clear_residual_density()
        self._set_density_controls_active(False)

    def update_density_availability(self) -> None:
        """Enable the button only when the renderer can show density."""
        if not HAS_DENSITY_CPP:
            return
        available = bool(getattr(self._render_widget,
                                 'has_residual_density_data', False))
        self._residual_density_button.setEnabled(available)
        if available:
            self._sync_density_controls()
            return
        self.clear_residual_density()
        self._residual_density_button.setToolTip(_DENSITY_TOOLTIP_NO_DATA)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _on_density_toggled(self, checked: bool) -> None:
        """Show or hide residual density from the button state.

        Any failure resets the button to unchecked.
        """
        if not checked:
            self.clear_residual_density()
            return

        hkl_source = self._auto_reflection_file()
        if hkl_source is None:
            hkl_source = self._ask_for_reflection_file()
            if hkl_source is None:  # dialog cancelled, or none offered
                self._set_density_button_checked(False)
                self._set_density_controls_active(False)
                return

        error_message = ""
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        try:
            # ``level=None`` means 3σ for this map.
            self.show_residual_density(hkl_source)
        except Exception as exc:  # noqa: BLE001 - never take the host app down
            error_message = str(exc)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        if error_message:
            self._set_density_button_checked(False)
            self._set_density_controls_active(False)
            if self.allow_reflection_dialog:
                QtWidgets.QMessageBox.warning(self, "Residual density", error_message)

    def _set_density_button_checked(self, checked: bool) -> None:
        """Set the button's checked state without re-entering the handler."""
        self._residual_density_button.blockSignals(True)
        self._residual_density_button.setChecked(checked)
        self._residual_density_button.blockSignals(False)

    def _set_density_controls_active(self, active: bool) -> None:
        """Reflect the on/off state in the button, tooltip and level widgets."""
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
        """Match the controls to the renderer's actual density state."""
        self._set_density_controls_active(
            self._render_widget.residual_density_map is not None
        )

    def _auto_reflection_file(self) -> object | None:
        """Return the renderer's declared or embedded reflection source."""
        return auto_reflection_file(self._render_widget)

    def _ask_for_reflection_file(self) -> Path | None:
        """Last resort: ask for a reflection file.

        Returns ``None`` when :attr:`allow_reflection_dialog` is ``False``.
        """
        if not self.allow_reflection_dialog:
            return None
        return ask_for_reflection_file(self, self._render_widget)

    def _residual_density_start_path(self) -> str:
        """Return the best starting path for the residual-density file dialog."""
        return reflection_file_start_path(self._render_widget)

    def _update_residual_density_tooltip(self) -> None:
        """Show the map statistics on the button while density is displayed."""
        density_map = self._render_widget.residual_density_map
        if density_map is None:
            return
        self._residual_density_button.setToolTip(
            density_statistics_text(density_map))


class ResidualDensityControls(DensityControlsMixin, QtWidgets.QWidget):
    """Ready-made residual-density controls for arbitrary host layouts."""

    def __init__(self, render_widget: Any,
                 parent: QtWidgets.QWidget | None = None,
                 *, allow_reflection_dialog: bool = True) -> None:
        super().__init__(parent)
        self._render_widget = render_widget
        self.allow_reflection_dialog = allow_reflection_dialog
        self._init_density_controls()

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._residual_density_button)
        layout.addWidget(self._density_level_label)
        layout.addWidget(self._density_level_spinbox)

    @property
    def button(self) -> QtWidgets.QPushButton:
        """The checkable *Residual Density* button."""
        return self._residual_density_button

    @property
    def level_spinbox(self) -> QtWidgets.QDoubleSpinBox:
        """The *Level* spin box, in e/Å³."""
        return self._density_level_spinbox
