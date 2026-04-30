"""Tests for the negative-PART / symmgen logic in build_conntable."""

from pathlib import Path

import numpy as np
import pytest

from fastmolwidget.tools import build_conntable

# ---------------------------------------------------------------------------
# Helper: place two atoms at a bondable C–O distance (~1.4 Å)
# ---------------------------------------------------------------------------
_CO_DIST = 1.4  # typical C–O single bond


def _two_atom_coords(dist: float = _CO_DIST) -> np.ndarray:
    return np.array([[0.0, 0.0, 0.0], [dist, 0.0, 0.0]])


# ===========================================================================
# Basic sanity
# ===========================================================================


class TestNegativePartBasic:
    """Unit tests for the negative-PART symmetry-boundary filter."""

    def test_same_neg_part_both_base_bonds(self):
        """Two base atoms with same negative part should bond."""
        coords = _two_atom_coords()
        bonds = build_conntable(
            coords, ["O", "C"], [-1, -1],
            symmgen=[False, False],
        )
        assert (0, 1) in bonds

    def test_same_neg_part_both_symmgen_bonds(self):
        """Two symmgen atoms with same negative part should bond (intra-copy)."""
        coords = _two_atom_coords()
        bonds = build_conntable(
            coords, ["O", "C"], [-1, -1],
            symmgen=[True, True],
        )
        assert (0, 1) in bonds

    def test_neg_part_cross_boundary_excluded(self):
        """A base atom with neg part should NOT bond to a symmgen atom with neg part."""
        coords = _two_atom_coords()
        bonds = build_conntable(
            coords, ["O", "C"], [-1, -1],
            symmgen=[False, True],
        )
        assert (0, 1) not in bonds

    def test_neg_part_cross_boundary_excluded_reverse(self):
        """A symmgen atom with neg part should NOT bond to a base atom with neg part."""
        coords = _two_atom_coords()
        bonds = build_conntable(
            coords, ["O", "C"], [-1, -1],
            symmgen=[True, False],
        )
        assert (0, 1) not in bonds

    def test_neg_part_base_to_symmgen_part0_allowed(self):
        """Base atom (neg part) bonding to symmgen atom (part=0) is unaffected
        because neither has negative part on the symmgen side AND the existing
        part filter allows part≠0 to part=0 bonds."""
        coords = _two_atom_coords()
        # atom 0: part=-1, base; atom 1: part=0, symmgen
        bonds = build_conntable(
            coords, ["O", "C"], [-1, 0],
            symmgen=[False, True],
        )
        # Excluded because atom 0 has neg part and they cross boundary
        assert (0, 1) not in bonds

    def test_positive_parts_cross_boundary_allowed(self):
        """Cross-boundary bond is allowed when parts are positive (no neg-part rule)."""
        coords = _two_atom_coords()
        bonds = build_conntable(
            coords, ["O", "C"], [0, 0],
            symmgen=[False, True],
        )
        assert (0, 1) in bonds

    def test_no_symmgen_param_no_filter(self):
        """When symmgen is None, negative-part filter is not applied."""
        coords = _two_atom_coords()
        bonds = build_conntable(
            coords, ["O", "C"], [-1, -1],
            symmgen=None,
        )
        assert (0, 1) in bonds

    def test_different_neg_parts_excluded_by_standard_filter(self):
        """Different non-zero parts are excluded by the standard part filter
        regardless of symmgen."""
        coords = _two_atom_coords()
        bonds = build_conntable(
            coords, ["O", "C"], [-1, -2],
            symmgen=[False, False],
        )
        # Standard part filter: both non-zero and different → excluded
        assert (0, 1) not in bonds


# ===========================================================================
# Integration test with real CIF data
# ===========================================================================

data = Path("tests/test-data")


class TestNegativePartCIF:
    """Integration test using 1979688_small.cif which has O13/C39 in PART -1."""

    @pytest.fixture()
    def grown_atoms(self):
        from fastmolwidget.cif.cif_file_io import CifReader
        from fastmolwidget.sdm import SDM

        cif = CifReader(str(data / "1979688_small.cif"))
        atoms_fract = list(cif.atoms_fract)
        cell = cif.cell
        symmops = cif.symmops

        sdm = SDM(atoms_fract, symmops, cell, centric=cif.is_centrosymm)
        need_symm = sdm.calc_sdm()
        return sdm.packer(sdm, need_symm)

    def test_o13_c39_base_copy_bonded(self, grown_atoms):
        """O13 and C39 in the base asymmetric unit should be bonded."""
        coords = np.array([[at.x, at.y, at.z] for at in grown_atoms], dtype=np.float64)
        types = [at.type for at in grown_atoms]
        parts = [at.part for at in grown_atoms]
        symmgen = [
            not np.allclose(np.array(at.symm_matrix, dtype=float), np.eye(3))
            if at.symm_matrix is not None else False
            for at in grown_atoms
        ]

        bonds = build_conntable(coords, types, parts, symmgen=symmgen)

        # Find base O13 and base C39
        base_o13 = next(
            i for i, at in enumerate(grown_atoms)
            if at.label == "O13" and not symmgen[i]
        )
        base_c39 = next(
            i for i, at in enumerate(grown_atoms)
            if at.label == "C39" and not symmgen[i]
        )
        pair = tuple(sorted((base_o13, base_c39)))
        assert pair in bonds

    def test_o13_c39_symmgen_copy_bonded(self, grown_atoms):
        """O13 and C39 in the symmetry-generated copy should be bonded."""
        coords = np.array([[at.x, at.y, at.z] for at in grown_atoms], dtype=np.float64)
        types = [at.type for at in grown_atoms]
        parts = [at.part for at in grown_atoms]
        symmgen = [
            not np.allclose(np.array(at.symm_matrix, dtype=float), np.eye(3))
            if at.symm_matrix is not None else False
            for at in grown_atoms
        ]

        bonds = build_conntable(coords, types, parts, symmgen=symmgen)

        # Find symmgen O13 and symmgen C39
        sg_o13 = next(
            i for i, at in enumerate(grown_atoms)
            if at.label == "O13" and symmgen[i]
        )
        sg_c39 = next(
            i for i, at in enumerate(grown_atoms)
            if at.label == "C39" and symmgen[i]
        )
        pair = tuple(sorted((sg_o13, sg_c39)))
        assert pair in bonds

    def test_no_cross_boundary_bonds_for_neg_part(self, grown_atoms):
        """No bonds should cross the base/symmgen boundary for PART -1 atoms."""
        coords = np.array([[at.x, at.y, at.z] for at in grown_atoms], dtype=np.float64)
        types = [at.type for at in grown_atoms]
        parts = [at.part for at in grown_atoms]
        symmgen = [
            not np.allclose(np.array(at.symm_matrix, dtype=float), np.eye(3))
            if at.symm_matrix is not None else False
            for at in grown_atoms
        ]

        bonds = build_conntable(coords, types, parts, symmgen=symmgen)

        neg_indices = {i for i, at in enumerate(grown_atoms) if at.part < 0}
        for i, j in bonds:
            if i in neg_indices or j in neg_indices:
                # Both must be on the same side of the boundary
                assert symmgen[i] == symmgen[j], (
                    f"Cross-boundary bond: {grown_atoms[i].label}(sg={symmgen[i]}) "
                    f"-- {grown_atoms[j].label}(sg={symmgen[j]})"
                )

