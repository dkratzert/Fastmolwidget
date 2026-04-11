"""
Loader class for :class:`~fastmolwidget.molecule2D.MoleculeWidget`.

Uses composition to provide file-format–aware loading of molecular data into a
:class:`MoleculeWidget`.  Supported formats:

* **CIF** – Crystallographic Information File (``.cif``)
* **SHELX** – SHELXL instruction file (``.res`` / ``.ins``)
* **XYZ** – Standard XYZ coordinate file (``.xyz``)

Example usage::

    widget = MoleculeWidget()
    loader = MoleculeLoader(widget)
    loader.load_file("structure.cif")
"""

from __future__ import annotations

from pathlib import Path

from fastmolwidget.cif.cif_file_io import CifReader
from fastmolwidget.dsrmath import frac_to_cart
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.sdm import Atomtuple


# SHELX instruction keywords – lines starting with one of these are *not*
# atom coordinate lines.
_SHELX_KEYWORDS = frozenset((
    'TITL', 'CELL', 'ZERR', 'LATT', 'SYMM', 'SFAC', 'UNIT', 'L.S.', 'LIST',
    'FVAR', 'WGHT', 'FMAP', 'PLAN', 'BOND', 'ACTA', 'CONF', 'HTAB', 'HKLF',
    'END', 'TEMP', 'SIZE', 'EXTI', 'SWAT', 'MOLE', 'PART', 'AFIX', 'HFIX',
    'SHEL', 'BASF', 'TWIN', 'DFIX', 'DANG', 'SADI', 'SAME', 'FLAT', 'SIMU',
    'DELU', 'ISOR', 'FREE', 'CONN', 'MPLA', 'RTAB', 'ABIN', 'ANSC', 'ANSR',
    'BLOC', 'BUMP', 'CGLS', 'CHIV', 'DEFS', 'EADP', 'EQIV', 'EXYZ', 'GRID',
    'HOPE', 'LAUE', 'MERG', 'MORE', 'MOVE', 'OMIT', 'PRIG', 'REM', 'RESI',
    'RIGU', 'SUMP', 'SPEC', 'STIR', 'TWST', 'WIGL', 'WPDB', 'XNPD',
    'BIND', 'LONE', 'ANIS', 'DISP',
))


class MoleculeLoader:
    """Load molecular structures from various file formats into a
    :class:`MoleculeWidget`.

    The loader uses **composition**: it holds a reference to the widget and
    delegates rendering to it via :meth:`MoleculeWidget.open_molecule`.

    :param widget: The molecule widget to populate.
    """

    # Maps file suffixes (lower-cased) to the internal loader method names.
    _FORMAT_MAP: dict[str, str] = {
        '.cif': '_load_cif',
        '.res': '_load_shelx',
        '.ins': '_load_shelx',
        '.xyz': '_load_xyz',
    }

    def __init__(self, widget: MoleculeWidget) -> None:
        self._widget = widget

    @property
    def widget(self) -> MoleculeWidget:
        """The :class:`MoleculeWidget` this loader populates."""
        return self._widget

    def load_file(self, path: str | Path, *, keep_view: bool = False) -> None:
        """Load a molecular structure from *path*.

        The file format is determined from the file extension.

        :param path: Path to the file.
        :param keep_view: If ``True``, preserve the current zoom / rotation.
        :raises ValueError: If the file format is not supported.
        :raises FileNotFoundError: If the file does not exist.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        loader_name = self._FORMAT_MAP.get(suffix)
        if loader_name is None:
            supported = ', '.join(sorted(self._FORMAT_MAP))
            raise ValueError(
                f"Unsupported file format '{suffix}'. "
                f"Supported extensions: {supported}"
            )
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        loader = getattr(self, loader_name)
        loader(path, keep_view=keep_view)

    # ------------------------------------------------------------------
    # CIF loading
    # ------------------------------------------------------------------

    def _load_cif(self, path: Path, *, keep_view: bool = False) -> None:
        """Load a CIF file using the existing :class:`CifReader`."""
        cif = CifReader(path)
        atoms = list(cif.atoms_orth)
        cell = cif.cell[:6]
        adps = self._widget.load_adps_from_cif(cif.displacement_parameters())
        self._widget.open_molecule(atoms=atoms, cell=cell, adps=adps,
                                   keep_view=keep_view)

    # ------------------------------------------------------------------
    # SHELX .res / .ins loading
    # ------------------------------------------------------------------

    def _load_shelx(self, path: Path, *, keep_view: bool = False) -> None:
        """Load a SHELX instruction (.res / .ins) file."""
        atoms, cell = self._parse_shelx(path)
        self._widget.open_molecule(atoms=atoms, cell=cell, adps=None,
                                   keep_view=keep_view)

    # ------------------------------------------------------------------
    # XYZ loading
    # ------------------------------------------------------------------

    def _load_xyz(self, path: Path, *, keep_view: bool = False) -> None:
        """Load a standard XYZ coordinate file.

        The format is::

            <number of atoms>
            <comment line>
            <element> <x> <y> <z>
            ...

        Coordinates are assumed to be in Ångströms (Cartesian).
        XYZ files have no unit-cell or ADP information.
        """
        atoms = self._parse_xyz(path)
        self._widget.open_molecule(atoms=atoms, cell=None, adps=None,
                                   keep_view=keep_view)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_shelx(
        path: Path,
    ) -> tuple[
        list[Atomtuple],
        tuple[float, float, float, float, float, float],
    ]:
        """Parse a SHELX .res / .ins file.

        Returns the atom list (in Cartesian coordinates) and the unit-cell
        parameters.
        """
        text = path.read_text()
        lines = text.splitlines()

        cell: tuple[float, float, float, float, float, float] | None = None
        sfac_elements: list[str] = []
        atoms: list[Atomtuple] = []

        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith('!'):
                continue

            parts = line.split()
            keyword = parts[0].upper()

            if keyword == 'CELL':
                # CELL lambda a b c alpha beta gamma
                if len(parts) >= 8:
                    cell = (
                        float(parts[2]), float(parts[3]), float(parts[4]),
                        float(parts[5]), float(parts[6]), float(parts[7]),
                    )
                continue

            if keyword == 'SFAC':
                # SFAC element1 element2 ...
                # The extended SFAC form may include numeric scattering
                # factor parameters after the element symbol; skip those.
                sfac_elements.extend(p.capitalize() for p in parts[1:] if not _is_number(p))
                continue

            if keyword in _SHELX_KEYWORDS:
                continue

            if keyword in ('END', 'HKLF'):
                break

            # Potential atom line: label sfac_num x y z sof Uiso
            if len(parts) >= 6 and _is_number(parts[1]):
                sfac_idx = int(parts[1])
                if sfac_idx < 1 or sfac_idx > len(sfac_elements):
                    continue
                try:
                    frac_x = float(parts[2])
                    frac_y = float(parts[3])
                    frac_z = float(parts[4])
                except ValueError:
                    continue

                element = sfac_elements[sfac_idx - 1]
                label = parts[0]

                # Parse the disorder part from the occupancy sign/prefix
                sof_str = parts[5] if len(parts) > 5 else '11.00000'
                part = 0
                try:
                    sof_val = float(sof_str)
                    part_digit = int(sof_val / 10)
                    if part_digit not in (0, 1):
                        part = part_digit
                except ValueError:
                    pass

                if cell is not None:
                    cart = frac_to_cart([frac_x, frac_y, frac_z], list(cell))
                    atoms.append(Atomtuple(
                        label=label, type=element,
                        x=cart[0], y=cart[1], z=cart[2],
                        part=part,
                    ))

        if cell is None:
            raise ValueError(f"No CELL instruction found in SHELX file: {path}")
        if not atoms:
            raise ValueError(f"No atoms found in SHELX file: {path}")

        return atoms, cell

    @staticmethod
    def _parse_xyz(path: Path) -> list[Atomtuple]:
        """Parse a standard XYZ file and return a list of
        :class:`Atomtuple`."""
        lines = path.read_text().splitlines()
        if len(lines) < 3:
            raise ValueError(f"XYZ file too short: {path}")

        try:
            natoms = int(lines[0].strip())
        except ValueError:
            raise ValueError(
                f"First line of XYZ file must be the atom count, "
                f"got: {lines[0].strip()!r}"
            )

        atom_lines = lines[2:]  # skip count and comment
        atoms: list[Atomtuple] = []
        for idx, line in enumerate(atom_lines):
            parts = line.split()
            if len(parts) < 4:
                continue  # skip blank / malformed lines
            element = parts[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            label = f"{element}{idx + 1}"
            atoms.append(
                Atomtuple(label=label, type=element, x=x, y=y, z=z, part=0)
            )

        if len(atoms) != natoms:
            raise ValueError(
                f"XYZ file declares {natoms} atoms but {len(atoms)} were "
                f"parsed from {path}"
            )
        return atoms


def _is_number(s: str) -> bool:
    """Return ``True`` if *s* can be interpreted as a float."""
    try:
        float(s)
        return True
    except ValueError:
        return False
