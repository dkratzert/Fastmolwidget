"""
Shared interface protocol for molecule display widgets.

Both :class:`~fastmolwidget.molecule2D.MoleculeWidget` and
:class:`~fastmolwidget.molecule3D.MoleculeWidget3D` satisfy this protocol, so
either can be used wherever a :class:`MoleculeWidgetProtocol` is expected.

Usage::

    from fastmolwidget.molecule_base import MoleculeWidgetProtocol
    from fastmolwidget.molecule2D import MoleculeWidget

    def render(widget: MoleculeWidgetProtocol) -> None:
        widget.open_molecule(atoms, cell=cell)

"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from fastmolwidget.sdm import Atomtuple

#: Change in the residual-density contour level per Ctrl+wheel notch, in e/Å³.
DENSITY_LEVEL_STEP: float = 0.01

#: Lowest contour level the interactive controls allow, in e/Å³.  A level of
#: zero would contour the whole map, so it is never reached.
DENSITY_LEVEL_MIN: float = 0.01

#: Highest contour level the interactive controls allow, in e/Å³.
DENSITY_LEVEL_MAX: float = 9.99


def _same_source(new: object | None, old: object | None) -> bool:
    """Whether two model / reflection sources mean the same thing.

    File names are compared by value, so reloading the same file (what grow
    and pack do) is recognised as unchanged; in-memory objects are compared by
    identity, because two documents can carry the same block name without
    describing the same structure.
    """
    if new is old:
        return True
    if isinstance(new, (str, Path)) and isinstance(old, (str, Path)):
        return Path(new) == Path(old)
    return False


class ModelSourceMixin:
    """Bookkeeping of the model and reflections behind the displayed atoms.

    A widget that was filled by :class:`~fastmolwidget.loader.MoleculeLoader`
    knows the file it is showing, but one that was handed a list of
    :class:`~fastmolwidget.sdm.Atomtuple` through ``open_molecule()`` does not
    — and residual density needs the model to calculate *F*\\ :sub:`c` from.
    This mixin lets such a host declare the sources once
    (:meth:`set_model_source`) and answers the two questions the renderers and
    control bars ask about them.

    Both renderers mix it in, so 2-D, Qt Quick and 3-D behave identically.
    """

    #: The model backing the displayed atoms: a path, an in-memory
    #: :class:`gemmi.cif.Document` / :class:`gemmi.cif.Block`, or a
    #: :class:`gemmi.SmallStructure`.
    _model_source: object | None = None
    #: Where its reflections come from; ``None`` means "look in / next to the
    #: model".
    _reflection_source: object | None = None
    #: Path of the structure file last loaded, kept in step with
    #: :attr:`_model_source` when that is a real file.
    _model_path: Path | None = None

    def set_model_source(self, model: object | None = None,
                         reflections: object | None = None) -> None:
        """Declare which model and reflections back the displayed atoms.

        A cached density map belongs to the previous model's reflections, so
        it is dropped whenever the sources really change.

        :param model: The refined model — a path, a :class:`gemmi.cif.Document`
            or :class:`gemmi.cif.Block`, or a :class:`gemmi.SmallStructure`.
            ``None`` forgets it.
        :param reflections: Where its reflections come from — the same kinds of
            source, or already read
            :class:`~fastmolwidget.hkl_io.ReflectionData`.  ``None`` means
            "look inside the model, and next to it when it is a file".
        """
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
        """``True`` when a map could be calculated for the current model.

        Only the declared sources are inspected — no map is computed — so a
        host can enable or disable its density control right after loading a
        structure.
        """
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
        """Resolve the arguments of ``show_residual_density`` against the state.

        :returns: ``(model, reflections)`` ready for
            :func:`~fastmolwidget.density.calculate_residual_density`.
        :raises RuntimeError: If no model is available at all.
        """
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
    """Protocol defining the common public API shared by all molecule widgets.

    Both :class:`~fastmolwidget.molecule2D.MoleculeWidget` and
    :class:`~fastmolwidget.molecule3D.MoleculeWidget3D` implement this
    protocol.  Any class that provides all of these methods is a valid
    molecule display widget regardless of inheritance.

    Expected signals (not enforceable via Protocol):

    * ``atomClicked(str)`` – emitted with the atom label when an atom is clicked.
    * ``bondClicked(str, str)`` – emitted with the two atom labels when a bond
      is clicked.
    * ``densityLevelChanged(float)`` – emitted with the new contour level
      whenever the residual-density level changes, so that a control bar can
      follow a Ctrl+wheel adjustment made in the view.
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
        """Load a new set of atoms and redraw.

        :param atoms: List of :class:`~fastmolwidget.sdm.Atomtuple` in
            Cartesian coordinates (Å).
        :param cell: Unit-cell parameters ``(a, b, c, α, β, γ)`` needed to
            convert fractional ADP tensors to Cartesian.  ``None`` for
            molecules with no periodic boundary.
        :param keep_view: If ``True`` the current zoom / rotation / pan is
            preserved.
        """
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
        """Set which disorder parts are rendered.

        :param parts: A set of part numbers to display, or ``None`` to show
            all parts (no filtering).  An empty set hides every atom.
        """
        ...

    def set_bond_width(self, width: int) -> None:
        """Set the bond width (screen pixels or world-space scaling factor)."""
        ...

    def set_bond_color(self, color: object) -> None:
        """Set the default colour used for non-selected bonds.

        :param color: A :class:`~qtpy.QtGui.QColor`, hex string, or RGB tuple.
        """
        ...

    # ------------------------------------------------------------------
    # Appearance
    # ------------------------------------------------------------------

    def set_background_color(self, color: object) -> None:
        """Set the background colour.

        :param color: A :class:`~qtpy.QtGui.QColor` instance.
        """
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
        """Rotate the structure to the orientation that maximises atom visibility.

        Uses PCA on the currently visible atom positions so that the direction
        with the least spread points towards the camera (Z-axis) and the widest
        face of the molecule faces the viewer.  Hydrogen / deuterium atoms are
        excluded when ``show_hydrogens_flag`` is ``False``.  No-op when fewer
        than two visible atoms are loaded.
        """
        ...

    def set_model_source(self, model: object | None = None,
                         reflections: object | None = None) -> None:
        """Declare the model and reflections behind the displayed atoms.

        See
        :meth:`~fastmolwidget.molecule_base.ModelSourceMixin.set_model_source`.
        """
        ...

    @property
    def has_residual_density_data(self) -> bool:
        """``True`` when a residual-density map could be computed right now."""
        ...

    def show_residual_density(self, hkl_path: object | None = None,
                              level: float | None = None) -> None:
        """Compute and display a residual (Fo−Fc) electron-density isosurface.

        The map is calculated from the reflection data together with the
        refined model, so nothing has to be pre-computed by another program.
        All three renderers implement this: the 3-D widget draws a true
        wireframe isosurface, while the 2-D and Qt Quick renderers project the
        same cage into their 2-D view.

        :param hkl_path: Path to a raw SHELX ``.hkl`` reflection file (or a
            ``.cif``/``.fcf`` file with an embedded reflection loop) paired
            with the currently loaded structure.  ``None`` finds the data
            automatically.
        :param level: Isosurface contour level in e/Å³.  ``None`` contours at
            3σ of the map, which adapts to each structure.
            A positive-density surface is drawn at ``+level`` (green) and a
            negative-density surface at ``-level`` (red).
        :raises RuntimeError: If no model is loaded, or the compiled
            ``density_cpp`` extension is missing.
        :raises FileNotFoundError: If no reflection data could be found.
        """
        ...

    def set_residual_density_level(self, level: float) -> None:
        """Re-contour the residual-density map at a new level.

        The map itself is reused, so this is much cheaper than
        :meth:`show_residual_density`.  A no-op when no map is loaded.

        :param level: Contour level in e/Å³.
        """
        ...

    def step_residual_density_level(self, steps: int) -> bool:
        """Raise or lower the contour level by *steps* wheel notches.

        Backs Ctrl+wheel in the view.  Each notch is
        :data:`DENSITY_LEVEL_STEP` e/Å³ and the result is clamped to
        :data:`DENSITY_LEVEL_MIN` … :data:`DENSITY_LEVEL_MAX`.

        :param steps: Number of notches; positive raises the level.
        :returns: ``True`` when a map was loaded and the level was adjusted.
        """
        ...

    def clear_residual_density(self) -> None:
        """Remove any residual-density isosurface currently displayed."""
        ...

    def refresh_residual_density(self) -> None:
        """Re-clip the cached map around the atoms that are visible now."""
        ...

    @property
    def residual_density_map(self) -> object | None:
        """The computed ``ResidualDensityMap``, or ``None`` when none is shown.

        Useful for reporting the map statistics (``max``, ``min``, ``rms``).
        """
        ...

    @property
    def residual_density_level(self) -> float:
        """The contour level the isosurface is currently drawn at, in e/Å³."""
        ...

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Render the current view to an image file."""
        ...
