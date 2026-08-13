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

    def show_residual_density(self, hkl_path: str | Path | None = None,
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
