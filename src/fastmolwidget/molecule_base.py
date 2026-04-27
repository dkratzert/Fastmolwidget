"""
Shared interface protocol for molecule display widgets.

Both :class:`~fastmolwidget.molecule2D.MoleculeWidget` and
:class:`~fastmolwidget.molecule3D.MoleculeWidget3D` satisfy this protocol, so
either can be used wherever a :class:`MoleculeWidgetProtocol` is expected.

Usage::

    from fastmolwidget.molecule_base import MoleculeWidgetProtocol
    from fastmolwidget.molecule2D import MoleculeWidget

    def render(widget: MoleculeWidgetProtocol) -> None:
        widget.open_molecule(atoms, cell=cell, adps=adps)

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
        adps: dict[str, tuple[float, float, float, float, float, float]] | None = None,
        keep_view: bool = False,
    ) -> None:
        """Load a new set of atoms and redraw.

        :param atoms: List of :class:`~fastmolwidget.sdm.Atomtuple` in
            Cartesian coordinates (Å).
        :param cell: Unit-cell parameters ``(a, b, c, α, β, γ)`` needed to
            convert fractional ADP tensors to Cartesian.  ``None`` for
            molecules with no periodic boundary.
        :param adps: Mapping of atom label → ``(U11, U22, U33, U23, U13,
            U12)`` anisotropic displacement parameters.  ``None`` to show
            plain spheres.
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

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Render the current view to an image file."""
        ...
