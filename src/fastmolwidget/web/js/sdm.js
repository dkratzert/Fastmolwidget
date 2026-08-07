/**
 * Port of `fastmolwidget.sdm.SDM` (Shortest-Distance-Matrix growing and
 * unit-cell packing). Pure JS, no native acceleration — fine for the atom
 * counts typical of a single asymmetric unit / unit cell.
 *
 * Input atoms use **fractional** coordinates: `{label, type, x, y, z, part}`.
 * `part` defaults to 0. Output atoms (from `grow()` / `packUnitCell()`) use
 * **Cartesian Ångström** coordinates and match the `Atomtuple`-style shape
 * consumed by `MoleculeWidget2D.openMolecule()`:
 * `{label, type, x, y, z, part, symm_matrix}`.
 */

import { getRadiusFromElement } from './elements.js';
import { transpose } from './linalg.js';
import { applySymmOp, fracToCart, identitySymmOp, matEquals, parseSymmOp, transEquivalent } from './symmetry.js';

const HYDROGENS = new Set(['H', 'D']);

class UnionFind {
  constructor(n) {
    this.parent = Array.from({ length: n }, (_, i) => i);
    this.rank = new Array(n).fill(0);
  }

  find(x) {
    while (this.parent[x] !== x) {
      this.parent[x] = this.parent[this.parent[x]];
      x = this.parent[x];
    }
    return x;
  }

  union(x, y) {
    const rx = this.find(x), ry = this.find(y);
    if (rx === ry) return;
    if (this.rank[rx] < this.rank[ry]) {
      this._link(ry, rx);
    } else {
      this._link(rx, ry);
      if (this.rank[rx] === this.rank[ry]) this.rank[rx] += 1;
    }
  }

  _link(root, child) {
    this.parent[child] = root;
  }
}

export class SDM {
  /**
   * @param {Array<{label:string,type:string,x:number,y:number,z:number,part?:number}>} atoms
   *   Fractional-coordinate asymmetric-unit atoms.
   * @param {string[]} symmops SHELX-style symmetry-operation strings (e.g. `"X,Y,Z"`), identity excluded.
   * @param {number[]} cell `[a,b,c,alpha,beta,gamma]`.
   * @param {boolean} [centric=false] Whether to add the inversion centre `-X,-Y,-Z`.
   */
  constructor(atoms, symmops, cell, centric = false) {
    this.atoms = atoms.map((a) => ({ ...a, part: a.part ?? 0, molindex: -1 }));
    this.symmops = [identitySymmOp()];
    const extra = centric ? ['-X,-Y,-Z', ...symmops] : symmops;
    for (const s of extra) {
      const op = parseSymmOp(s);
      const dup = this.symmops.some((o) => matEquals(o.matrix, op.matrix) && transEquivalent(o.trans, op.trans));
      if (!dup) this.symmops.push(op);
    }
    this.cell = cell;
    const [a, b, c, alpha, beta, gamma] = cell;
    const rad = (d) => (d * Math.PI) / 180;
    this.aga = a * b * Math.cos(rad(gamma));
    this.bbe = a * c * Math.cos(rad(beta));
    this.cal = b * c * Math.cos(rad(alpha));
    this.asq = a * a;
    this.bsq = b * b;
    this.csq = c * c;
    this.sdmList = [];
    this.maxmol = 1;
  }

  vectorLength(x, y, z) {
    const A = 2.0 * (x * y * this.aga + x * z * this.bbe + y * z * this.cal);
    return Math.sqrt(x * x * this.asq + y * y * this.bsq + z * z * this.csq + A);
  }

  /** Compute the shortest-distance matrix and return the "needed symmetry" list. */
  calcSdm() {
    const n = this.atoms.length;
    const nlen = this.symmops.length;
    const { aga, bbe, cal, asq, bsq, csq } = this;
    const at2PlusHalf = this.atoms.map((a) => [a.x + 0.5, a.y + 0.5, a.z + 0.5]);
    this.sdmList = [];

    for (let i = 0; i < n; i++) {
      const at1 = this.atoms[i];
      const v1 = [at1.x, at1.y, at1.z];
      const primeArray = this.symmops.map((op) => applySymmOp(op, v1));

      for (let j = 0; j < n; j++) {
        const at2 = this.atoms[j];
        const [atpX, atpY, atpZ] = at2PlusHalf[j];
        let mind = 1000000.0;
        let item = null;

        for (let nIdx = 0; nIdx < nlen; nIdx++) {
          const [px, py, pz] = primeArray[nIdx];
          const dx = px - atpX, dy = py - atpY, dz = pz - atpZ;
          const dpx = dx - Math.floor(dx) - 0.5;
          const dpy = dy - Math.floor(dy) - 0.5;
          const dpz = dz - Math.floor(dz) - 0.5;
          const A = 2.0 * (dpx * dpy * aga + dpx * dpz * bbe + dpy * dpz * cal);
          const dk2 = dpx * dpx * asq + dpy * dpy * bsq + dpz * dpz * csq + A;
          if (dk2 > 16.0) continue;
          let dk = Math.sqrt(dk2);
          if (nIdx) dk += 0.0001;
          if (dk > 0.01 && mind >= dk) {
            mind = dk;
            item = {
              dist: mind, atom1: at1, atom2: at2, a1: i, a2: j,
              symmetryNumber: nIdx, covalent: true, dddd: 0,
            };
          }
        }

        if (!item) continue;
        const isH1 = HYDROGENS.has(item.atom1.type);
        const isH2 = HYDROGENS.has(item.atom2.type);
        let dddd;
        if ((!isH1 && !isH2 && item.atom1.part * item.atom2.part === 0) || item.atom1.part === item.atom2.part) {
          dddd = (getRadiusFromElement(item.atom1.type) + getRadiusFromElement(item.atom2.type)) * 1.2;
          item.dddd = dddd;
        } else {
          dddd = 0.0;
        }
        item.covalent = item.dist < dddd;
        this.sdmList.push(item);
      }
    }

    this.sdmList.sort((a, b) => a.dist - b.dist);
    this._calcMolindex();
    return this._collectNeededSymmetry();
  }

  _calcMolindex() {
    const n = this.atoms.length;
    const uf = new UnionFind(n);
    for (const item of this.sdmList) {
      if (item.covalent) uf.union(item.a1, item.a2);
    }
    const rootToMol = new Map();
    let counter = 0;
    for (let i = 0; i < n; i++) {
      const root = uf.find(i);
      if (!rootToMol.has(root)) {
        counter += 1;
        rootToMol.set(root, counter);
      }
      this.atoms[i].molindex = rootToMol.get(root);
    }
    this.maxmol = counter;
  }

  _collectNeededSymmetry() {
    const needSymm = [];
    const seen = new Set();
    const { aga, bbe, cal, asq, bsq, csq } = this;

    for (const item of this.sdmList) {
      if (!item.covalent) continue;
      const molindex = item.atom1.molindex;
      if (molindex < 1 || molindex > 6) continue;

      const x1 = item.atom1.x, y1 = item.atom1.y, z1 = item.atom1.z;
      const x2 = item.atom2.x, y2 = item.atom2.y, z2 = item.atom2.z;

      for (let n = 0; n < this.symmops.length; n++) {
        if (item.atom1.part * item.atom2.part !== 0 && item.atom1.part !== item.atom2.part) continue;
        if (item.atom1.type === item.atom2.type && HYDROGENS.has(item.atom1.type)) continue;

        const [px, py, pz] = applySymmOp(this.symmops[n], [x1, y1, z1]);
        const Dx = px - x2 + 0.5, Dy = py - y2 + 0.5, Dz = pz - z2 + 0.5;
        const fDx = Math.floor(Dx), fDy = Math.floor(Dy), fDz = Math.floor(Dz);
        const dpx = Dx - fDx - 0.5, dpy = Dy - fDy - 0.5, dpz = Dz - fDz - 0.5;
        if (n === 0 && fDx === 0 && fDy === 0 && fDz === 0) continue;

        const A = 2.0 * (dpx * dpy * aga + dpx * dpz * bbe + dpy * dpz * cal);
        const dk2 = dpx * dpx * asq + dpy * dpy * bsq + dpz * dpz * csq + A;
        if (dk2 <= 0.000001) continue;
        const dk = Math.sqrt(dk2);
        let dddd = item.dist + 0.2;
        if (HYDROGENS.has(item.atom1.type) && HYDROGENS.has(item.atom2.type)) dddd = 1.8;

        if (dk <= dddd) {
          const bs = [n + 1, Math.trunc(5 - fDx), Math.trunc(5 - fDy), Math.trunc(5 - fDz), molindex];
          const key = bs.join(',');
          if (!seen.has(key)) {
            seen.add(key);
            needSymm.push(bs);
          }
        }
      }
    }
    return needSymm;
  }

  /** Expand the asymmetric unit to whole molecules (call after `calcSdm()`). */
  packer(needSymm) {
    const showAtoms = this.atoms.map((a) => ({
      label: a.label, type: a.type, x: a.x, y: a.y, z: a.z, part: a.part, matrix: null,
    }));

    for (const [s, h, k, l, symmgroup] of needSymm) {
      const h2 = h - 5, k2 = k - 5, l2 = l - 5;
      const op = this.symmops[s - 1];

      for (const atom of this.atoms) {
        if (atom.molindex !== symmgroup) continue;
        const [px0, py0, pz0] = applySymmOp(op, [atom.x, atom.y, atom.z]);
        const px = px0 + h2, py = py0 + k2, pz = pz0 + l2;
        const newAtom = { label: atom.label, type: atom.type, x: px, y: py, z: pz, part: atom.part, matrix: transpose(op.matrix) };

        let isThere = false;
        if (atom.part >= 0) {
          for (const existing of showAtoms) {
            if (existing.part !== atom.part) continue;
            const length = this.vectorLength(px - existing.x, py - existing.y, pz - existing.z);
            if (length < 0.2) {
              isThere = true;
              break;
            }
          }
        }
        if (!isThere) showAtoms.push(newAtom);
      }
    }

    return showAtoms.map((at) => {
      const [x, y, z] = fracToCart([at.x, at.y, at.z], this.cell);
      return { label: at.label, type: at.type, x, y, z, part: at.part, symm_matrix: at.matrix };
    });
  }

  /** Grow the asymmetric unit to complete molecules (runs `calcSdm()` + `packer()`). */
  grow() {
    const needSymm = this.calcSdm();
    return this.packer(needSymm);
  }

  /**
   * Pack every symmetry-equivalent position into one unit cell.
   * @param {number[]|null} [symmopIndices] 0-based indices into the internal
   *   symmetry-operation list (identity is always index 0). `null` uses all.
   * @param {number} [cartTolerance=0.2] Cartesian duplicate-detection tolerance (Å).
   */
  packUnitCell(symmopIndices = null, cartTolerance = 0.2) {
    const selected = symmopIndices ?? this.symmops.map((_, i) => i);
    const tolSq = cartTolerance * cartTolerance;
    const { aga, bbe, cal, asq, bsq, csq } = this;
    const packed = [];

    for (const at of this.atoms) {
      for (const idx of selected) {
        const op = this.symmops[idx];
        const [px0, py0, pz0] = applySymmOp(op, [at.x, at.y, at.z]);
        const px = ((px0 % 1) + 1) % 1;
        const py = ((py0 % 1) + 1) % 1;
        const pz = ((pz0 % 1) + 1) % 1;

        let isDup = false;
        for (const ex of packed) {
          if (ex.part !== 0 && at.part !== 0 && ex.part !== at.part) continue;
          let ddx = px - ex.fx, ddy = py - ex.fy, ddz = pz - ex.fz;
          ddx -= Math.round(ddx);
          ddy -= Math.round(ddy);
          ddz -= Math.round(ddz);
          const d2 = ddx * ddx * asq + ddy * ddy * bsq + ddz * ddz * csq
            + 2.0 * (ddx * ddy * aga + ddx * ddz * bbe + ddy * ddz * cal);
          if (d2 < tolSq) {
            isDup = true;
            break;
          }
        }
        if (!isDup) {
          const [cx, cy, cz] = fracToCart([px, py, pz], this.cell);
          packed.push({ fx: px, fy: py, fz: pz, part: at.part, label: at.label, type: at.type, cx, cy, cz, matrix: transpose(op.matrix) });
        }
      }
    }

    return packed.map((p) => ({
      label: p.label, type: p.type, x: p.cx, y: p.cy, z: p.cz, part: p.part, symm_matrix: p.matrix,
    }));
  }
}
