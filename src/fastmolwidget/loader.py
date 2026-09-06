"""File-format loaders for :class:`~fastmolwidget.molecule2D.MoleculeWidget`."""

from __future__ import annotations

from pathlib import Path

from shelxfile import Shelxfile

from fastmolwidget.cif.cif_file_io import CifReader
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.sdm import Atomtuple
from fastmolwidget.tools import to_float


class MoleculeLoader:
    """Load CIF, SHELX and XYZ files into a :class:`MoleculeWidget`."""

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

        # A new model invalidates the old density map. Reloading the same path
        # keeps it, which is what grow/pack rely on.
        self._widget.set_model_source(path)
        loader = getattr(self, loader_name)
        loader(path, keep_view=keep_view)

    _GROWABLE_FORMATS: frozenset[str] = frozenset({'.cif', '.res', '.ins'})

    def set_grow(self, enabled: bool) -> None:
        """Toggle SDM grow mode.

        Reloads the current CIF or SHELX file with ``keep_view=True``. XYZ has
        no symmetry, and pack mode still takes priority.
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

        Packing applies all or selected symmetry operations, folds positions
        into ``[0, 1)``, drops near-duplicates, and takes priority over grow.
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
        adp_by_label: dict[str, tuple] = {
            dp.label: (
                to_float(dp.U11), to_float(dp.U22), to_float(dp.U33),
                to_float(dp.U23), to_float(dp.U13), to_float(dp.U12),
            )
            for dp in cif.displacement_parameters()
        }
        if self._pack_enabled:
            atoms = self._compute_packed_atoms_cif(cif, self._pack_symmop_indices)
        elif self._grow_enabled:
            atoms = self._compute_grown_atoms(cif)
        else:
            atoms = [
                Atomtuple(label=at.label, type=at.type, x=at.x, y=at.y, z=at.z,
                          part=at.part, adp=adp_by_label.get(at.label))
                for at in cif.atoms_orth
            ]
        self._widget.open_molecule(
            atoms=atoms,
            cell=cif.cell[:6],
            keep_view=keep_view,
        )
        self._widget._is_packed = self._pack_enabled

    @staticmethod
    def _compute_packed_atoms_cif(
        cif: CifReader,
        symmop_indices: list[int] | None = None,
    ) -> list:
        """Pack one CIF unit cell.

        Applies all or selected symmetry operations, folds positions into
        ``[0, 1)``, and drops near-duplicates.
        """
        from fastmolwidget.sdm import SDM

        adp_by_label: dict[str, tuple] = {
            dp.label: (
                to_float(dp.U11), to_float(dp.U22), to_float(dp.U33),
                to_float(dp.U23), to_float(dp.U13), to_float(dp.U12),
            )
            for dp in cif.displacement_parameters()
        }
        fract_atoms = list(cif.atoms_fract)
        sdm = SDM(fract_atoms, cif.symmops, cif.cell, centric=cif.is_centrosymm)
        cart_atoms = sdm.pack_unit_cell(symmop_indices=symmop_indices)
        return [at._replace(adp=adp_by_label.get(at.label)) for at in cart_atoms]

    @staticmethod
    def _compute_grown_atoms(cif: CifReader) -> list:
        """Expand a CIF asymmetric unit to complete molecules via the SDM."""
        from fastmolwidget.sdm import SDM

        adp_by_label: dict[str, tuple] = {
            dp.label: (
                to_float(dp.U11), to_float(dp.U22), to_float(dp.U33),
                to_float(dp.U23), to_float(dp.U13), to_float(dp.U12),
            )
            for dp in cif.displacement_parameters()
        }
        # SDM mutates the atom lists in place.
        fract_atoms = list(cif.atoms_fract)
        sdm = SDM(fract_atoms, cif.symmops, cif.cell, centric=cif.is_centrosymm)
        need_symm = sdm.calc_sdm()
        cart_atoms = sdm.packer(sdm, need_symm)
        return [at._replace(adp=adp_by_label.get(at.label)) for at in cart_atoms]

    # ------------------------------------------------------------------
    # SHELX .res / .ins loading
    # ------------------------------------------------------------------

    def _load_shelx(self, path: Path, *, keep_view: bool = False) -> None:
        """Load a SHELX instruction (.res / .ins) file using the
        :mod:`shelxfile` library."""
        atoms, cell = self._parse_shelx(path)
        if self._pack_enabled:
            atoms = self._compute_packed_atoms_shelx(path, self._pack_symmop_indices)
        elif self._grow_enabled:
            atoms = self._compute_grown_atoms_shelx(path)
        self._widget.open_molecule(atoms=atoms, cell=cell, keep_view=keep_view)
        self._widget._is_packed = self._pack_enabled

    @staticmethod
    def _compute_grown_atoms_shelx(path: Path) -> list:
        """Expand a SHELX asymmetric unit to complete molecules via the SDM."""
        from fastmolwidget.sdm import SDM

        shx = Shelxfile()
        shx.read_file(path)

        cell_params: tuple[float, float, float, float, float, float] = (
            shx.cell.a, shx.cell.b, shx.cell.c,
            shx.cell.alpha, shx.cell.beta, shx.cell.gamma,
        )

        adp_by_lp: dict[tuple, tuple] = {}
        # SDM mutates these fractional-coordinate atom lists.
        fract_atoms: list[list] = []
        for at in shx.atoms:
            if at.qpeak:
                continue
            x, y, z = at.frac_coords
            label = at.fullname_short  # residue-unique, e.g. "C1_1"
            fract_atoms.append(
                [label, at.element, x, y, z, at.part.n, at.occupancy, at.ueq]
            )
            if not at.is_isotropic:
                u11, u22, u33, u23, u13, u12 = at.uvals
                adp_by_lp[(label, at.part.n)] = (u11, u22, u33, u23, u13, u12)

        # SHELX SYMM cards, without the implicit identity.
        symmops: list[str] = [s.to_shelxl() for s in shx.symmcards]
        centric = shx.latt.centric if shx.latt else False

        sdm = SDM(fract_atoms, symmops, cell_params, centric=centric)
        need_symm = sdm.calc_sdm()
        cart_atoms = sdm.packer(sdm, need_symm)
        return [
            at._replace(adp=adp_by_lp.get((at.label, at.part)))
            for at in cart_atoms
        ]

    @staticmethod
    def _compute_packed_atoms_shelx(
        path: Path,
        symmop_indices: list[int] | None = None,
    ) -> list:
        """Pack one SHELX unit cell.

        Applies all or selected symmetry operations, folds positions into
        ``[0, 1)``, drops near-duplicates, and re-attaches ADPs by
        ``(label, part)``.
        """
        from fastmolwidget.sdm import SDM

        shx = Shelxfile()
        shx.read_file(path)

        cell_params: tuple[float, float, float, float, float, float] = (
            shx.cell.a, shx.cell.b, shx.cell.c,
            shx.cell.alpha, shx.cell.beta, shx.cell.gamma,
        )

        adp_by_lp: dict[tuple, tuple] = {}
        fract_atoms: list[list] = []
        for at in shx.atoms:
            if at.qpeak:
                continue
            x, y, z = at.frac_coords
            label = at.fullname_short  # residue-unique, e.g. "C1_1"
            fract_atoms.append(
                [label, at.element, x, y, z, at.part.n, at.occupancy, at.ueq]
            )
            if not at.is_isotropic:
                u11, u22, u33, u23, u13, u12 = at.uvals
                adp_by_lp[(label, at.part.n)] = (u11, u22, u33, u23, u13, u12)

        symmops: list[str] = [s.to_shelxl() for s in shx.symmcards]
        centric = shx.latt.centric if shx.latt else False

        sdm = SDM(fract_atoms, symmops, cell_params, centric=centric)
        cart_atoms = sdm.pack_unit_cell(symmop_indices=symmop_indices)
        return [
            at._replace(adp=adp_by_lp.get((at.label, at.part)))
            for at in cart_atoms
        ]

    # ------------------------------------------------------------------
    # XYZ loading
    # ------------------------------------------------------------------

    def _load_xyz(self, path: Path, *, keep_view: bool = False) -> None:
        """Load a standard XYZ file.

        Coordinates are Cartesian Å. XYZ carries no cell or ADP data.
        """
        atoms = self._parse_xyz(path)
        self._widget.open_molecule(atoms=atoms, cell=None, keep_view=keep_view)

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
        """Parse a SHELX ``.res``/``.ins`` file.

        Returns Cartesian atoms with embedded ADPs and the unit cell. Q-peaks
        are skipped.
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
        for at in shx.atoms:
            # Skip Q-peaks: residual density, not atoms.
            if at.qpeak:
                continue

            x, y, z = at.cart_coords
            adp_vals: tuple | None = None
            if not at.is_isotropic:
                u11, u22, u33, u23, u13, u12 = at.uvals
                adp_vals = (u11, u22, u33, u23, u13, u12)

            atoms.append(Atomtuple(
                label=at.fullname_short,
                type=at.element,
                x=x,
                y=y,
                z=z,
                part=at.part.n,
                adp=adp_vals,
            ))

        return atoms, cell_params

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
