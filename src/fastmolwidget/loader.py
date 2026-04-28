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
from typing import Any, Generator

from shelxfile import Shelxfile

from fastmolwidget.cif.cif_file_io import CifReader, adp
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.sdm import Atomtuple
from fastmolwidget.tools import to_float


class MoleculeLoader:
    """Load molecular structures from various file formats into a
    :class:`MoleculeWidget`.

    The loader uses **composition**: it holds a reference to the widget and
    delegates rendering to it via :meth:`MoleculeWidget.open_molecule`.

    :param widget: The molecule widget to populate.
    """

    _FORMAT_MAP: dict[str, str] = {
        '.cif': '_load_cif',
        '.res': '_load_shelx',
        '.ins': '_load_shelx',
        '.xyz': '_load_xyz',
    }

    def __init__(self, widget: MoleculeWidget) -> None:
        self._widget = widget
        self._grow_enabled: bool = False
        self._pack_enabled: bool = False
        self._pack_symmop_indices: list[int] | None = None
        self._last_path: Path | None = None

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
        self._last_path = path
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

    _GROWABLE_FORMATS: frozenset[str] = frozenset({'.cif', '.res', '.ins'})

    def set_grow(self, enabled: bool) -> None:
        """Toggle SDM grow mode and reload the current file if it supports
        crystal symmetry (CIF or SHELX).

        Grow expands the asymmetric unit to complete molecules using crystal
        symmetry.  Has no effect when the last loaded file is an XYZ (no
        symmetry information).  When :meth:`set_pack` is active, pack mode
        takes priority and grow does nothing until pack is disabled.

        :param enabled: ``True`` to enable structure growing, ``False`` to
            revert to the bare asymmetric unit.
        """
        self._grow_enabled = enabled
        if (self._last_path is not None
                and self._last_path.suffix.lower() in self._GROWABLE_FORMATS
                and not self._pack_enabled):
            self.load_file(self._last_path, keep_view=True)

    def set_pack(
        self,
        enabled: bool,
        symmop_indices: list[int] | None = None,
    ) -> None:
        """Toggle unit-cell packing mode and reload the current file.

        Packing generates all symmetry-equivalent positions within one unit
        cell.  Positions closer than the default tolerance to an already-added
        atom are discarded as duplicates.  When pack is enabled it takes
        priority over :meth:`set_grow`.

        :param enabled: ``True`` to enable packing, ``False`` to revert to the
            asymmetric unit (or to grown molecules if grow is still active).
        :param symmop_indices: Optional list of 0-based symmetry-operation
            indices (referring to the internal ``SymmCards`` list, including the
            identity at index 0) to apply.  ``None`` applies all operations
            from the space group, including the inversion centre for
            centrosymmetric structures.
        """
        self._pack_enabled = enabled
        self._pack_symmop_indices = symmop_indices
        if (self._last_path is not None
                and self._last_path.suffix.lower() in self._GROWABLE_FORMATS):
            self.load_file(self._last_path, keep_view=True)

    # ------------------------------------------------------------------
    # CIF loading
    # ------------------------------------------------------------------

    def _load_cif(self, path: Path, *, keep_view: bool = False) -> None:
        """Load a CIF file using :class:`CifReader`."""
        cif = CifReader(path)
        if self._pack_enabled:
            atoms = self._compute_packed_atoms_cif(cif, self._pack_symmop_indices)
        elif self._grow_enabled:
            atoms = self._compute_grown_atoms(cif)
        else:
            atoms = list(cif.atoms_orth)
        self._widget.open_molecule(
            atoms=atoms,
            cell=cif.cell[:6],
            adps=self._load_adps_from_cif(cif.displacement_parameters()),
            keep_view=keep_view,
        )

    @staticmethod
    def _compute_packed_atoms_cif(
        cif: CifReader,
        symmop_indices: list[int] | None = None,
    ) -> list:
        """Pack one complete unit cell from a CIF.

        Applies all (or selected) symmetry operations to the fractional-
        coordinate atoms and folds every position back into [0, 1).
        Near-duplicate positions are discarded automatically.

        :param cif: The parsed CIF to pack.
        :param symmop_indices: Optional subset of 0-based symmetry-operation
            indices.  ``None`` uses all operations from the space group.
        :returns: A list of :class:`~fastmolwidget.sdm.Atomtuple` in Cartesian
            coordinates covering one unit cell.
        """
        from fastmolwidget.sdm import SDM

        fract_atoms = list(cif.atoms_fract)
        sdm = SDM(fract_atoms, cif.symmops, cif.cell, centric=cif.is_centrosymm)
        return sdm.pack_unit_cell(symmop_indices=symmop_indices)

    @staticmethod
    def _compute_grown_atoms(cif: CifReader) -> list:
        """Expand the asymmetric unit to complete molecules via the SDM.

        Reads fractional-coordinate atoms, runs the Shortest Distance Matrix
        algorithm with the CIF's symmetry operators, and returns the packed
        Cartesian-coordinate atom list.

        :param cif: The parsed CIF to grow.
        :returns: A list of :class:`~fastmolwidget.sdm.Atomtuple` in Cartesian
            coordinates including all symmetry-generated atoms that complete
            the molecule(s).
        """
        from fastmolwidget.sdm import SDM

        # SDM.calc_molindex mutates the atom lists in place – pass a fresh copy
        fract_atoms = list(cif.atoms_fract)
        sdm = SDM(fract_atoms, cif.symmops, cif.cell, centric=cif.is_centrosymm)
        need_symm = sdm.calc_sdm()
        return sdm.packer(sdm, need_symm)

    @staticmethod
    def _load_adps_from_cif(
        adps: Generator[adp, Any, None],
    ) -> dict[str, tuple[float, float, float, float, float, float]]:
        """Convert a generator of CIF displacement parameters into the ADP
        mapping expected by :meth:`MoleculeWidget.open_molecule`.

        :param adps: Generator of :class:`~fastmolwidget.cif.cif_file_io.adp`
            named-tuples produced by
            :meth:`~fastmolwidget.cif.cif_file_io.CifReader.displacement_parameters`.
        :returns: A dict mapping atom labels to ``(U11, U22, U33, U23, U13,
            U12)`` tuples of floats.
        """
        adp_dict: dict[str, tuple[float, float, float, float, float, float]] = {}
        for dp in adps:
            adp_dict[dp.label] = (
                to_float(dp.U11), to_float(dp.U22), to_float(dp.U33),
                to_float(dp.U23), to_float(dp.U13), to_float(dp.U12),
            )
        return adp_dict

    # ------------------------------------------------------------------
    # SHELX .res / .ins loading
    # ------------------------------------------------------------------

    def _load_shelx(self, path: Path, *, keep_view: bool = False) -> None:
        """Load a SHELX instruction (.res / .ins) file using the
        :mod:`shelxfile` library."""
        atoms, cell, adps = self._parse_shelx(path)
        if self._pack_enabled:
            atoms = self._compute_packed_atoms_shelx(path, self._pack_symmop_indices)
        elif self._grow_enabled:
            atoms = self._compute_grown_atoms_shelx(path)
        self._widget.open_molecule(atoms=atoms, cell=cell, adps=adps,
                                   keep_view=keep_view)

    @staticmethod
    def _compute_grown_atoms_shelx(path: Path) -> list:
        """Expand the asymmetric unit of a SHELX file to complete molecules
        via the SDM, analogous to :meth:`_compute_grown_atoms` for CIF files.

        :param path: Path to the SHELX ``.res`` / ``.ins`` file.
        :returns: A list of :class:`~fastmolwidget.sdm.Atomtuple` in Cartesian
            coordinates including all symmetry-generated atoms.
        """
        from fastmolwidget.sdm import SDM

        shx = Shelxfile()
        shx.read_file(path)

        cell_params: tuple[float, float, float, float, float, float] = (
            shx.cell.a, shx.cell.b, shx.cell.c,
            shx.cell.alpha, shx.cell.beta, shx.cell.gamma,
        )

        # Build fractional-coordinate atom lists (mutable – SDM mutates them)
        fract_atoms: list[list] = []
        for at in shx.atoms:
            if at.qpeak:
                continue
            x, y, z = at.frac_coords
            fract_atoms.append(
                [at.name, at.element, x, y, z, at.part.n, at.occupancy, at.ueq]
            )

        # Collect symmetry operations as comma-separated strings (skip identity)
        symmops: list[str] = []
        for s in shx.symmcards:
            op_str = s.to_shelxl()
            # SDM already includes identity; only add non-identity ops
            symmops.append(op_str)

        centric = shx.latt.centric if shx.latt else False

        sdm = SDM(fract_atoms, symmops, cell_params, centric=centric)
        need_symm = sdm.calc_sdm()
        return sdm.packer(sdm, need_symm)

    @staticmethod
    def _compute_packed_atoms_shelx(
        path: Path,
        symmop_indices: list[int] | None = None,
    ) -> list:
        """Pack one complete unit cell from a SHELX file.

        Applies all (or selected) symmetry operations to the fractional-
        coordinate atoms and folds every position back into [0, 1).
        Near-duplicate positions are discarded automatically.

        :param path: Path to the SHELX ``.res`` / ``.ins`` file.
        :param symmop_indices: Optional subset of 0-based symmetry-operation
            indices.  ``None`` uses all operations from the space group.
        :returns: A list of :class:`~fastmolwidget.sdm.Atomtuple` in Cartesian
            coordinates covering one unit cell.
        """
        from fastmolwidget.sdm import SDM

        shx = Shelxfile()
        shx.read_file(path)

        cell_params: tuple[float, float, float, float, float, float] = (
            shx.cell.a, shx.cell.b, shx.cell.c,
            shx.cell.alpha, shx.cell.beta, shx.cell.gamma,
        )

        fract_atoms: list[list] = []
        for at in shx.atoms:
            if at.qpeak:
                continue
            x, y, z = at.frac_coords
            fract_atoms.append(
                [at.name, at.element, x, y, z, at.part.n, at.occupancy, at.ueq]
            )

        symmops: list[str] = [s.to_shelxl() for s in shx.symmcards]
        centric = shx.latt.centric if shx.latt else False

        sdm = SDM(fract_atoms, symmops, cell_params, centric=centric)
        return sdm.pack_unit_cell(symmop_indices=symmop_indices)

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
        dict[str, tuple[float, float, float, float, float, float]],
    ]:
        """Parse a SHELX .res / .ins file using the :mod:`shelxfile` library.

        Returns the atom list (in Cartesian coordinates), the unit-cell
        parameters, and a dictionary of anisotropic displacement parameters.

        Q-peaks (residual electron-density peaks) are excluded.
        """
        shx = Shelxfile()
        shx.read_file(path)

        if shx.cell is None:
            raise ValueError(f"No CELL instruction found in SHELX file: {path}")

        cell = shx.cell
        cell_params: tuple[float, float, float, float, float, float] = (
            cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma,
        )

        atoms: list[Atomtuple] = []
        adp_dict: dict[str, tuple[float, float, float, float, float, float]] = {}
        for at in shx.atoms:
            # Skip Q-peaks (residual electron-density peaks, not real atoms).
            if at.qpeak:
                continue

            x, y, z = at.cart_coords
            atoms.append(Atomtuple(
                label=at.name,
                type=at.element,
                x=x,
                y=y,
                z=z,
                part=at.part.n,
            ))

            # Collect anisotropic displacement parameters for non-isotropic atoms.
            if not at.is_isotropic:
                u11, u22, u33, u23, u13, u12 = at.uvals
                adp_dict[at.name] = (u11, u22, u33, u23, u13, u12)

        return atoms, cell_params, adp_dict

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
