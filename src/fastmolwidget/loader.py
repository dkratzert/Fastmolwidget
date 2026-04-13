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

import numpy as np
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
        self._last_path: Path | None = None
        self._density_enabled: bool = False
        self._density_sigma: float = 3.0
        self._hkl_path: Path | None = None

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
        symmetry information).

        :param enabled: ``True`` to enable structure growing, ``False`` to
            revert to the bare asymmetric unit.
        """
        self._grow_enabled = enabled
        if (self._last_path is not None
                and self._last_path.suffix.lower() in self._GROWABLE_FORMATS):
            self.load_file(self._last_path, keep_view=True)
            # Re-compute density with updated symmetry replication
            if self._density_enabled:
                self._load_density()

    def set_density(self, enabled: bool) -> None:
        """Toggle residual electron-density wireframe display.

        When enabled, the loader computes and draws the Fo−Fc difference
        density isosurface.  Requires a CIF file with HKL data (either
        embedded or as a separate ``.hkl`` file).

        :param enabled: ``True`` to show density, ``False`` to hide it.
        """
        self._density_enabled = enabled
        self._widget.show_density(enabled)
        if enabled and self._last_path is not None:
            self._load_density()

    def set_density_sigma(self, sigma: float) -> None:
        """Set the isosurface level in multiples of the RMS density.

        :param sigma: Number of standard deviations for the iso-level
            (e.g. 3.0 for ±3σ).
        """
        self._density_sigma = max(0.5, sigma)
        if self._density_enabled and self._last_path is not None:
            self._load_density()

    def set_hkl_path(self, path: str | Path | None) -> None:
        """Set an explicit path to a ``.hkl`` file.

        When ``None``, the loader tries the embedded ``_shelx_hkl_file`` data
        from the CIF.

        :param path: Path to a SHELX HKL file, or ``None``.
        """
        self._hkl_path = Path(path) if path is not None else None

    def _load_density(self) -> None:
        """Compute and set the residual density wireframe on the widget.

        When grow mode is active, the density wireframe is replicated to cover
        all unit cells that contain atoms of the grown structure, ensuring the
        density display is consistent with the expanded crystal symmetry.
        """
        if self._last_path is None:
            return

        suffix = self._last_path.suffix.lower()
        if suffix != '.cif':
            return

        try:
            from fastmolwidget.density import compute_difference_density
            from fastmolwidget.isosurface import marching_cubes_wireframe
            import gemmi

            # Determine HKL path: explicit > auto-detect sibling > embedded
            hkl_path = self._hkl_path
            if hkl_path is None:
                # Try to find a sibling .hkl file
                candidates = list(self._last_path.parent.glob('*.hkl'))
                stem = self._last_path.stem
                for c in candidates:
                    if stem in c.stem:
                        hkl_path = c
                        break

            density_arr, cell, spacegroup = compute_difference_density(
                self._last_path, hkl_path=hkl_path,
            )
            rms = float(np.sqrt(np.mean(density_arr ** 2)))
            if rms < 1e-12:
                self._widget.set_density_wireframe(
                    np.empty((0, 6)), np.empty((0, 6)),
                )
                return

            pos_level = self._density_sigma * rms
            neg_level = -self._density_sigma * rms

            pos_segs = marching_cubes_wireframe(density_arr, pos_level, cell)
            neg_segs = marching_cubes_wireframe(density_arr, neg_level, cell)

            # When grow is enabled, replicate density for neighboring unit cells
            # to match the expanded structure.
            if self._grow_enabled and len(self._widget.atoms) > 0:
                pos_segs = self._replicate_density_segments(
                    pos_segs, cell,
                )
                neg_segs = self._replicate_density_segments(
                    neg_segs, cell,
                )

            self._widget.set_density_wireframe(pos_segs, neg_segs)
        except Exception as e:
            print(f"Failed to compute residual density: {e}")
            self._widget.set_density_wireframe(
                np.empty((0, 6)), np.empty((0, 6)),
            )

    def _replicate_density_segments(
        self,
        segments: np.ndarray,
        cell: 'gemmi.UnitCell',
    ) -> np.ndarray:
        """Replicate density wireframe segments to cover all unit cells
        that overlap the bounding box of the currently displayed atoms.

        The residual density map has the periodicity of the unit cell. When
        the structure is grown to show complete molecules across cell
        boundaries, the density wireframe must be translated by lattice
        vectors to cover those regions.

        :param segments: ``(N, 6)`` array of line segments in Cartesian.
        :param cell: The unit cell for fractional ↔ Cartesian conversion.
        :returns: Expanded ``(M, 6)`` array of segments.
        """
        import gemmi

        if len(segments) == 0 or len(self._widget.atoms) == 0:
            return segments

        # Determine the bounding box of atoms in fractional coordinates
        min_frac = np.array([1e9, 1e9, 1e9])
        max_frac = np.array([-1e9, -1e9, -1e9])
        for atom in self._widget.atoms:
            pos = gemmi.Position(*atom.coordinate)
            frac = cell.fractionalize(pos)
            f = np.array([frac.x, frac.y, frac.z])
            min_frac = np.minimum(min_frac, f)
            max_frac = np.maximum(max_frac, f)

        # Determine integer translation range (unit cells to replicate)
        n_min = np.floor(min_frac).astype(int)
        n_max = np.floor(max_frac).astype(int)

        # The base segments already cover (0,0,0).  Generate translations.
        translations = []
        for na in range(n_min[0], n_max[0] + 1):
            for nb in range(n_min[1], n_max[1] + 1):
                for nc in range(n_min[2], n_max[2] + 1):
                    if na == 0 and nb == 0 and nc == 0:
                        continue  # Skip identity (already present)
                    frac_shift = gemmi.Fractional(float(na), float(nb), float(nc))
                    cart_shift = cell.orthogonalize(frac_shift)
                    translations.append(
                        np.array([cart_shift.x, cart_shift.y, cart_shift.z])
                    )

        if not translations:
            return segments

        all_segments = [segments]
        for shift in translations:
            shifted = segments.copy()
            shifted[:, :3] += shift
            shifted[:, 3:] += shift
            all_segments.append(shifted)

        return np.concatenate(all_segments, axis=0)

    # ------------------------------------------------------------------
    # CIF loading
    # ------------------------------------------------------------------

    def _load_cif(self, path: Path, *, keep_view: bool = False) -> None:
        """Load a CIF file using :class:`CifReader`."""
        cif = CifReader(path)
        atoms = (
            self._compute_grown_atoms(cif)
            if self._grow_enabled
            else list(cif.atoms_orth)
        )
        self._widget.open_molecule(
            atoms=atoms,
            cell=cif.cell[:6],
            adps=self._load_adps_from_cif(cif.displacement_parameters()),
            keep_view=keep_view,
        )

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
        if self._grow_enabled:
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
