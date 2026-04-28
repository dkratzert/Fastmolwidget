# möp
#
# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <dkratzert@gmx.de> wrote this file. As long as you retain
# this notice you can do whatever you want with this stuff. If we meet some day,
# and you think this stuff is worth it, you can buy me a beer in return.
# Daniel Kratzert
# ----------------------------------------------------------------------------
#
from __future__ import annotations

import time
from collections import namedtuple
from math import sqrt, cos, radians, sin, floor
from operator import attrgetter
from typing import TYPE_CHECKING

try:
    import sdm_cpp

    HAS_CPP = True
except ImportError:
    HAS_CPP = False

from fastmolwidget.atoms import get_radius_from_element
from fastmolwidget.dsrmath import SymmetryElement, frac_to_cart

if TYPE_CHECKING:
    pass

DEBUG = False
Atomtuple = namedtuple('Atomtuple', ('label', 'type', 'x', 'y', 'z', 'part', 'symm_matrix'), defaults=(None,))


class SymmCards:
    """
    Contains the list of SYMM cards
    """

    def __init__(self):
        self._symmcards = [SymmetryElement(['X', 'Y', 'Z'])]

    def _as_str(self) -> str:
        return "\n".join([str(x) for x in self._symmcards])

    def __repr__(self) -> str:
        return self._as_str()

    def __str__(self) -> str:
        return self._as_str()

    def __getitem__(self, item):
        return self._symmcards[item]

    def __iter__(self):
        yield from self._symmcards

    def __len__(self):
        return len(self._symmcards)

    def append(self, symmData: list) -> None:
        """
        Add the content of a Shelxl SYMM command to generate the appropriate SymmetryElement instance.
        :param symmData: list of strings. eg.['1/2+X', '1/2+Y', '1/2+Z']
        :return: None
        """
        newSymm = SymmetryElement(symmData)
        if newSymm not in self._symmcards:
            self._symmcards.append(newSymm)


class SDMItem:
    __slots__ = ['a1', 'a2', 'atom1', 'atom2', 'covalent', 'dddd', 'dist', 'symmetry_number']

    def __init__(self):
        self.dist = 0.0
        self.atom1 = None
        self.a1 = 0
        self.atom2 = None
        self.a2 = 0
        self.symmetry_number = 0
        self.covalent = True
        self.dddd = 0

    def __lt__(self, a2) -> bool:
        return True if self.dist < a2.dist else False

    def __eq__(self, other: SDMItem) -> bool:
        if other.a1 == self.a2 and other.a2 == self.a1:
            return True
        return False

    def __repr__(self):
        return f'{self.atom1.name} {self.atom2.name} {self.a1} {self.a2} dist: {self.dist} coval: {self.covalent} sn: {self.symmetry_number} {self.dddd}'


class SDM:
    sdm_list: list[SDMItem]

    def __init__(self, atoms: tuple[list], symmlist: list, cell: tuple[float, float, float, float, float, float],
                 centric=False):
        """
        Calculates the shortest distance matrix
                        0      1      2  3  4   5     6          7
        :param atoms: [Name, Element, X, Y, Z, Part, ocuupancy, molindex -> (later)]
        :param symmlist:
        :param cell:
        """
        self.atoms = atoms
        self.symmcards = SymmCards()
        if centric:
            self.symmcards.append(['-X', '-Y', '-Z'])
            self.symmcards[-1].centric = True
        for s in symmlist:
            self.symmcards.append(s.split(','))
        self.cell = cell
        self.cosal = cos(radians(cell[3]))
        self.cosbe = cos(radians(cell[4]))
        self.cosga = cos(radians(cell[5]))
        self.aga = self.cell[0] * self.cell[1] * self.cosga
        self.bbe = self.cell[0] * self.cell[2] * self.cosbe
        self.cal = self.cell[1] * self.cell[2] * self.cosal
        self.asq = self.cell[0] ** 2
        self.bsq = self.cell[1] ** 2
        self.csq = self.cell[2] ** 2
        self.sdm_list = []  # list of sdmitems
        self.maxmol = 1
        self.sdmtime = 0

    def calc_sdm(self) -> list:
        t1 = time.perf_counter()
        h = {'H', 'D'}
        nlen = len(self.symmcards)

        symm_m = []
        symm_t = []
        for s in self.symmcards:
            symm_m.append(tuple(map(tuple, s.matrix.T)))
            symm_t.append(tuple(s.trans))

        # C++ Fast Path
        if HAS_CPP:
            coords = [[at[2], at[3], at[4]] for at in self.atoms]
            radii = [get_radius_from_element(at[1]) for at in self.atoms]
            is_h = [at[1] in h for at in self.atoms]
            parts = [at[5] for at in self.atoms]

            cpp_results = sdm_cpp.calc_sdm_cpp(
                coords, symm_m, symm_t,
                self.aga, self.bbe, self.cal,
                self.asq, self.bsq, self.csq,
                radii, is_h, parts
            )

            for (i, j, best_n, mind, dddd, covalent) in cpp_results:
                sdm_item = SDMItem()
                sdm_item.dist = mind
                sdm_item.atom1 = self.atoms[i]
                sdm_item.atom2 = self.atoms[j]
                sdm_item.a1 = i
                sdm_item.a2 = j
                sdm_item.symmetry_number = best_n
                sdm_item.dddd = dddd
                sdm_item.covalent = covalent
                self.sdm_list.append(sdm_item)

        # Pure Python Fallback Path
        else:
            at2_plushalf = [(x[2] + 0.5, x[3] + 0.5, x[4] + 0.5) for x in self.atoms]
            aga, bbe, cal = self.aga, self.bbe, self.cal
            asq, bsq, csq = self.asq, self.bsq, self.csq

            for i, at1 in enumerate(self.atoms):
                x1, y1, z1 = at1[2], at1[3], at1[4]

                prime_array = []
                for m, t in zip(symm_m, symm_t):
                    px = x1 * m[0][0] + y1 * m[1][0] + z1 * m[2][0] + t[0]
                    py = x1 * m[0][1] + y1 * m[1][1] + z1 * m[2][1] + t[1]
                    pz = x1 * m[0][2] + y1 * m[1][2] + z1 * m[2][2] + t[2]
                    prime_array.append((px, py, pz))

                for j, at2 in enumerate(self.atoms):
                    mind = 1000000.0
                    hma = False
                    atp_x, atp_y, atp_z = at2_plushalf[j]

                    sdm_item = SDMItem()

                    for n in range(nlen):
                        px, py, pz = prime_array[n]

                        dx = px - atp_x
                        dy = py - atp_y
                        dz = pz - atp_z

                        dpx = dx - floor(dx) - 0.5
                        dpy = dy - floor(dy) - 0.5
                        dpz = dz - floor(dz) - 0.5

                        A = 2.0 * (dpx * dpy * aga + dpx * dpz * bbe + dpy * dpz * cal)
                        dk2 = dpx * dpx * asq + dpy * dpy * bsq + dpz * dpz * csq + A

                        if dk2 > 16.0:
                            continue

                        dk = sqrt(dk2)
                        if n:
                            dk += 0.0001

                        if (dk > 0.01) and (mind >= dk):
                            mind = dk
                            sdm_item.dist = mind
                            sdm_item.atom1 = at1
                            sdm_item.atom2 = at2
                            sdm_item.a1 = i
                            sdm_item.a2 = j
                            sdm_item.symmetry_number = n
                            hma = True

                    if not sdm_item.atom1:
                        continue
                    if ((sdm_item.atom1[1] not in h and sdm_item.atom2[1] not in h) and
                        sdm_item.atom1[5] * sdm_item.atom2[5] == 0) or sdm_item.atom1[5] == sdm_item.atom2[5]:
                        dddd = (get_radius_from_element(at1[1]) + get_radius_from_element(at2[1])) * 1.2
                        sdm_item.dddd = dddd
                    else:
                        dddd = 0.0
                    if sdm_item.dist < dddd:
                        if hma:
                            sdm_item.covalent = True
                    else:
                        sdm_item.covalent = False
                    if hma:
                        self.sdm_list.append(sdm_item)

        t2 = time.perf_counter()
        self.sdmtime = t2 - t1
        print(f'Time for sdm {"(C++)" if HAS_CPP else "(Python fallback)"}:', round(self.sdmtime, 4), 's')

        self.sdm_list.sort(key=attrgetter('dist'))
        self.calc_molindex(self.atoms)
        need_symm = self.collect_needed_symmetry()
        if DEBUG:
            print(f"The asymmetric unit contains {self.maxmol} fragments.")
        return need_symm

    def collect_needed_symmetry(self) -> list:
        need_symm = []
        h = ('H', 'D')

        symm_m = []
        symm_t = []
        for s in self.symmcards:
            symm_m.append(tuple(map(tuple, s.matrix.T)))
            symm_t.append(tuple(s.trans))

        aga, bbe, cal = self.aga, self.bbe, self.cal
        asq, bsq, csq = self.asq, self.bsq, self.csq

        # Collect needsymm list:
        for sdm_item in self.sdm_list:
            if sdm_item.covalent:
                if sdm_item.atom1[-1] < 1 or sdm_item.atom1[-1] > 6:
                    continue

                x1, y1, z1 = sdm_item.atom1[2], sdm_item.atom1[3], sdm_item.atom1[4]
                x2, y2, z2 = sdm_item.atom2[2], sdm_item.atom2[3], sdm_item.atom2[4]

                for n, (m, t) in enumerate(zip(symm_m, symm_t)):
                    if sdm_item.atom1[5] * sdm_item.atom2[5] != 0 and \
                            sdm_item.atom1[5] != sdm_item.atom2[5]:
                        continue
                    # Both the same atomic number and number 0 (hydrogen)
                    if sdm_item.atom1[1] == sdm_item.atom2[1] and sdm_item.atom1[1] in h:
                        continue

                    px = x1 * m[0][0] + y1 * m[1][0] + z1 * m[2][0] + t[0]
                    py = x1 * m[0][1] + y1 * m[1][1] + z1 * m[2][1] + t[1]
                    pz = x1 * m[0][2] + y1 * m[1][2] + z1 * m[2][2] + t[2]

                    Dx = px - x2 + 0.5
                    Dy = py - y2 + 0.5
                    Dz = pz - z2 + 0.5

                    fDx, fDy, fDz = floor(Dx), floor(Dy), floor(Dz)

                    dpx = Dx - fDx - 0.5
                    dpy = Dy - fDy - 0.5
                    dpz = Dz - fDz - 0.5

                    if n == 0 and fDx == 0 and fDy == 0 and fDz == 0:
                        continue

                    A = 2.0 * (dpx * dpy * aga + dpx * dpz * bbe + dpy * dpz * cal)
                    dk2 = dpx * dpx * asq + dpy * dpy * bsq + dpz * dpz * csq + A

                    if dk2 <= 0.000001:
                        continue

                    dk = sqrt(dk2)
                    dddd = sdm_item.dist + 0.2
                    if sdm_item.atom1[1] in h and sdm_item.atom2[1] in h:
                        dddd = 1.8

                    if dk <= dddd:
                        bs = [n + 1, int(5 - fDx), int(5 - fDy), int(5 - fDz), sdm_item.atom1[-1]]
                        if bs not in need_symm:
                            need_symm.append(bs)
        return need_symm

    def calc_molindex(self, all_atoms):
        """Assign a molecule index to every atom using Union-Find (path-halving +
        union-by-rank).  Replaces the original O(K·|sdm_list|) repeated-scan loop
        with an essentially linear O(N + M·α(N)) algorithm.

        The last element of each atom list is set to its molecule index (1-based),
        matching the contract expected by collect_needed_symmetry() and packer().
        """
        n = len(all_atoms)
        for at in all_atoms:
            at.append(-1)  # reserve the molindex slot (keeps original API)

        # ── Union-Find with path-halving and union-by-rank ────────────────────
        parent = list(range(n))
        rank = [0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path halving (no recursion)
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            rx, ry = find(x), find(y)
            if rx == ry:
                return
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1

        for sdm_item in self.sdm_list:
            if sdm_item.covalent:
                union(sdm_item.a1, sdm_item.a2)

        # ── assign sequential 1-based molecule indices ────────────────────────
        root_to_mol: dict[int, int] = {}
        mol_counter = 0
        for i in range(n):
            root = find(i)
            if root not in root_to_mol:
                mol_counter += 1
                root_to_mol[root] = mol_counter
            all_atoms[i][-1] = root_to_mol[root]

        self.maxmol = mol_counter

    def pack_unit_cell(
            self,
            symmop_indices: list[int] | None = None,
            *,
            cart_tolerance: float = 0.2,
    ) -> list[Atomtuple]:
        """Pack all symmetry-equivalent positions into one unit cell.

        For every atom in the asymmetric unit every selected symmetry
        operation is applied and the result is folded back into [0, 1)
        fractional coordinates.  Positions that are already occupied within
        *cart_tolerance* Ångström (with periodic boundary conditions) are
        discarded as duplicates.  The threshold matches the one used by the
        molecule-grow :meth:`packer`.

        This call can be made on a fresh :class:`SDM` object before
        :meth:`calc_sdm` — it does **not** require the SDM to have been run.

        :param symmop_indices: 0-based indices into the internal
            :class:`SymmCards` list (identity is always index 0).  ``None``
            applies all operations including the inversion centre when the
            structure is centrosymmetric.
        :param cart_tolerance: Cartesian distance threshold (Å) for duplicate
            detection with periodic boundary conditions.  Default 0.2 Å
            matches the grow packer.
        :returns: List of :class:`Atomtuple` in Cartesian Ångström coordinates.
        """
        selected: list[int] = (
            list(range(len(self.symmcards)))
            if symmop_indices is None
            else list(symmop_indices)
        )

        symm_m: list = []
        symm_t: list = []
        for s in self.symmcards:
            symm_m.append(tuple(map(tuple, s.matrix.T)))
            symm_t.append(tuple(s.trans))

        cell = self.cell[:6]

        # packed entries: [label, type, fx, fy, fz, part, cx, cy, cz, matrix]
        # cx/cy/cz are Cartesian Å (for fast distance checks)
        packed: list[list] = []

        for at in self.atoms:
            x1, y1, z1 = at[2], at[3], at[4]
            part = at[5]
            label = at[0]
            elem = at[1]

            for idx in selected:
                m = symm_m[idx]
                t = symm_t[idx]

                # Apply symmetry operation (column-major, matching packer())
                px = x1 * m[0][0] + y1 * m[1][0] + z1 * m[2][0] + t[0]
                py = x1 * m[0][1] + y1 * m[1][1] + z1 * m[2][1] + t[1]
                pz = x1 * m[0][2] + y1 * m[1][2] + z1 * m[2][2] + t[2]

                # Fold into [0, 1)
                px %= 1.0
                py %= 1.0
                pz %= 1.0

                # Duplicate check using Cartesian distances with PBCs.
                # Fractional difference folded to [−0.5, 0.5] → then
                # converted to Å via vector_length() — same as packer().
                is_dup = False
                for ex in packed:
                    # Atoms in different (non-zero) disorder parts cannot
                    # be duplicates of each other.
                    if ex[5] != 0 and part != 0 and ex[5] != part:
                        continue
                    ddx = px - ex[2];
                    ddx -= round(ddx)
                    ddy = py - ex[3];
                    ddy -= round(ddy)
                    ddz = pz - ex[4];
                    ddz -= round(ddz)
                    if self.vector_length(ddx, ddy, ddz) < cart_tolerance:
                        is_dup = True
                        break

                if not is_dup:
                    cx, cy, cz = frac_to_cart([px, py, pz], cell)
                    packed.append([label, elem, px, py, pz, part, cx, cy, cz, m])

        cart_atoms: list[Atomtuple] = []
        for at in packed:
            cart_atoms.append(
                Atomtuple(
                    label=at[0], type=at[1], x=at[6], y=at[7], z=at[8],
                    part=at[5], symm_matrix=at[9],
                )
            )
        return cart_atoms

    def vector_length(self, x: float, y: float, z: float) -> float:
        """
        Calculates the vector length given in fractional coordinates.
        """
        A = 2.0 * (x * y * self.aga + x * z * self.bbe + y * z * self.cal)
        return sqrt(x ** 2 * self.asq + y ** 2 * self.bsq + z ** 2 * self.csq + A)

    def packer(self, sdm: SDM, need_symm: list, with_qpeaks=False) -> list[Atomtuple]:
        """
        Packs atoms of the asymmetric unit to real molecules.
        """
        showatoms = []
        # Append base atoms with an identity matrix
        identity_matrix = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        for at in self.atoms:
            showatoms.append(list(at) + ['base', identity_matrix])

        new_atoms = []

        symm_m = []
        symm_t = []
        for s in self.symmcards:
            symm_m.append(tuple(map(tuple, s.matrix.T)))
            symm_t.append(tuple(s.trans))

        for symm in need_symm:
            s, h, k, l, symmgroup = symm
            h -= 5
            k -= 5
            l -= 5
            s -= 1

            m = symm_m[s]
            t = symm_t[s]

            for atom in self.atoms:
                if atom[-1] == symmgroup:
                    x1, y1, z1 = atom[2], atom[3], atom[4]
                    px = x1 * m[0][0] + y1 * m[1][0] + z1 * m[2][0] + t[0] + h
                    py = x1 * m[0][1] + y1 * m[1][1] + z1 * m[2][1] + t[1] + k
                    pz = x1 * m[0][2] + y1 * m[1][2] + z1 * m[2][2] + t[2] + l

                    # The new atom with the symmetry matrix appended:
                    new = [atom[0], atom[1], px, py, pz, atom[5], atom[6], atom[7], 'symmgen', m]
                    new_atoms.append(new)
                    isthere = False
                    # Only add atom if its occupancy (new[5]) is greater zero:
                    if new[5] >= 0:
                        for existing in showatoms:
                            if existing[5] != new[5]:
                                continue
                            length = sdm.vector_length(px - existing[2],
                                                       py - existing[3],
                                                       pz - existing[4])
                            if length < 0.2:
                                isthere = True
                                break
                    if not isthere:
                        showatoms.append(new)

        cart_atoms = []
        cell = self.cell[:6]
        for at in showatoms:
            x, y, z = frac_to_cart([at[2], at[3], at[4]], cell)
            # at[-1] contains the symm_matrix
            cart_atoms.append(Atomtuple(label=at[0], type=at[1], x=x, y=y, z=z, part=at[5], symm_matrix=at[-1]))
        return cart_atoms


if __name__ == "__main__":
    import sys
    from pathlib import Path
    from qtpy import QtWidgets
    from fastmolwidget.viewer_widget import MoleculeViewerWidget

    app = QtWidgets.QApplication(sys.argv)
    viewer = MoleculeViewerWidget()
    viewer.load_file(Path('../../tests/test-data/4060314.cif'))
    viewer.show()
    sys.exit(app.exec())
