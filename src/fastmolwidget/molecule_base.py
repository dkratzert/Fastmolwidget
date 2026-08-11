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

    def show_residual_density(self, hkl_path: str | Path, level: float = 0.10) -> None:
        """Compute and display a residual (Fo−Fc) electron-density isosurface.

        3D-only feature: real rendering is implemented in
        :class:`~fastmolwidget.molecule3D.MoleculeWidget3D`. The 2-D
        (:class:`~fastmolwidget.molecule2D.MoleculeWidget`) and Qt Quick
        (:class:`~fastmolwidget.molecule_quick.MoleculeQuickItem`) renderers
        implement this as a documented no-op so ``isinstance`` checks against
        :class:`MoleculeWidgetProtocol` keep working uniformly across renderers.

        :param hkl_path: Path to a raw SHELX ``.hkl`` reflection file (or a
            ``.cif``/``.fcf`` file with an embedded reflection loop) paired
            with the currently loaded structure.
        :param level: Isosurface contour level in e/Å³ (default ``0.10``).
            A positive-density surface is drawn at ``+level`` (green) and a
            negative-density surface at ``-level`` (red).
        """
        ...

    def clear_residual_density(self) -> None:
        """Remove any residual-density isosurface currently displayed.

        3D-only feature; a documented no-op on the 2-D and Qt Quick renderers.
        """
        ...

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Render the current view to an image file."""
        ...
