"""Shared protocol and source-tracking helpers for molecule widgets."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from fastmolwidget.sdm import Atomtuple

#: Residual-density level step for Ctrl+wheel, in e/Å³.
DENSITY_LEVEL_STEP: float = 0.01

#: Lowest allowed contour level, in e/Å³.
DENSITY_LEVEL_MIN: float = 0.01

#: Highest allowed contour level, in e/Å³.
DENSITY_LEVEL_MAX: float = 9.99


def _same_source(new: object | None, old: object | None) -> bool:
    """Return ``True`` when two model or reflection sources are equivalent.

    Paths compare by value; in-memory sources compare by identity.
    """
    if new is old:
        return True
    if isinstance(new, (str, Path)) and isinstance(old, (str, Path)):
        return Path(new) == Path(old)
    return False


class ModelSourceMixin:
    """Track the model and reflection sources behind the displayed atoms."""

    #: Backing model source.
    _model_source: object | None = None
    #: Reflection source, or ``None`` to resolve it from the model.
    _reflection_source: object | None = None
    #: Path form of the current model source, when applicable.
    _model_path: Path | None = None

    def set_model_source(self, model: object | None = None,
                         reflections: object | None = None) -> None:
        """Declare the model and reflection sources for the displayed atoms."""
        changed = not (_same_source(model, self._model_source)
                       and _same_source(reflections, self._reflection_source))
        self._model_source = model
        self._reflection_source = reflections
        self._model_path = Path(model) if isinstance(model, (str, Path)) else None
        if changed:
            self.clear_residual_density()  # type: ignore[attr-defined]

    @property
    def model_source(self) -> object | None:
        """The model the residual density is calculated from, if any."""
        return self._model_source if self._model_source is not None else self._model_path

    @property
    def reflection_source(self) -> object | None:
        """The declared reflection source, or ``None`` when it is implicit."""
        return self._reflection_source

    @property
    def has_residual_density_data(self) -> bool:
        """``True`` when a residual-density map could be calculated."""
        from fastmolwidget.density import HAS_DENSITY_CPP
        from fastmolwidget.hkl_io import find_reflection_file, has_reflections

        if not HAS_DENSITY_CPP:
            return False
        source = self._reflection_source
        if source is None:
            source = self.model_source
        if source is None:
            return False
        try:
            if isinstance(source, (str, Path)):
                return find_reflection_file(source) is not None
            return has_reflections(source)
        except Exception:  # noqa: BLE001 - "no data" is an answer, not a failure
            return False

    def _density_sources(
        self,
        model: object | None = None,
        reflections: object | None = None,
    ) -> tuple[object, object | None]:
        """Resolve ``show_residual_density`` arguments against widget state."""
        if model is None:
            model = self.model_source
        if model is None:
            raise RuntimeError(
                'No structure model available - load a .res/.ins/.cif file '
                'or call set_model_source() before showing residual density.'
            )
        return model, reflections if reflections is not None else self._reflection_source


@runtime_checkable
class MoleculeWidgetProtocol(Protocol):
    """Common public API for all molecule widgets.

    Expected signals: ``atomClicked(str)``, ``bondClicked(str, str)``, and
    ``densityLevelChanged(float)``.
    """

    # ------------------------------------------------------------------
    # Molecule data
    # ------------------------------------------------------------------

    def open_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None = None,
        keep_view: bool = False,
    ) -> None:
        """Load atoms and redraw."""
        ...

    def clear(self) -> None:
        """Remove all atoms and bonds from the widget."""
        ...

    # ------------------------------------------------------------------
    # Display toggles
    # ------------------------------------------------------------------

    def show_adps(self, value: bool) -> None:
        """Toggle ADP ellipsoids / isotropic spheres."""
        ...

    def show_labels(self, value: bool) -> None:
        """Toggle atom-label display."""
        ...


    def show_hydrogens(self, value: bool) -> None:
        """Toggle hydrogen atom visibility."""
        ...

    def set_visible_parts(self, parts: set[int] | None) -> None:
        """Set the visible disorder parts."""
        ...

    @property
    def dragged_atoms_are_isotropic(self) -> bool:
        """Whether dragged disorder atoms are flattened to isotropic ADPs."""
        ...

    def set_dragged_atoms_isotropic(self, value: bool) -> None:
        """Toggle whether dragged disorder atoms are flattened to isotropic ADPs."""
        ...

    def set_bond_width(self, width: int) -> None:
        """Set the bond width (screen pixels or world-space scaling factor)."""
        ...

    def set_bond_color(self, color: object) -> None:
        """Set the default colour for non-selected bonds."""
        ...

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    def set_background_color(self, color: object) -> None:
        """Set the background colour."""
        ...

    def set_labels_visible(self, visible: bool) -> None:
        """Toggle visibility of atom labels (alias for :meth:`show_labels`)."""
        ...

    def setLabelFont(self, font_size: int) -> None:
        """Set the pixel size used for atom labels."""
        ...

    # ------------------------------------------------------------------
    # View control
    # ------------------------------------------------------------------

    def reset_view(self) -> None:
        """Reset zoom, rotation and pan to defaults."""
        ...

    def align_best_view(self) -> None:
        """Align the structure with a PCA-based best view."""
        ...

    def set_model_source(self, model: object | None = None,
                         reflections: object | None = None) -> None:
        """Declare the model and reflection sources behind the displayed atoms."""
        ...

    @property
    def has_residual_density_data(self) -> bool:
        """``True`` when a residual-density map could be computed right now."""
        ...

    def show_residual_density(self, hkl_path: object | None = None,
                              level: float | None = None) -> None:
        """Compute and display a residual (Fo−Fc) isosurface."""
        ...

    def set_residual_density_level(self, level: float) -> None:
        """Re-contour the cached residual-density map."""
        ...

    def step_residual_density_level(self, steps: int) -> bool:
        """Adjust the contour level by *steps* wheel notches."""
        ...

    def clear_residual_density(self) -> None:
        """Remove any residual-density isosurface currently displayed."""
        ...

    def refresh_residual_density(self) -> None:
        """Re-clip the cached map around the atoms that are visible now."""
        ...

    @property
    def residual_density_map(self) -> object | None:
        """The current ``ResidualDensityMap``, or ``None``."""
        ...

    @property
    def residual_density_level(self) -> float:
        """The contour level the isosurface is currently drawn at, in e/Å³."""
        ...

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Render the current view to an image file."""
        ...
