/**
 * High-level convenience wrapper combining the SDM grow/pack-unit-cell logic
 * (`sdm.js`) with the Canvas renderer (`molecule2d.js`).
 *
 * Python only needs to parse the structure file (CIF/SHELX) and send the
 * **asymmetric unit** as fractional coordinates plus the symmetry
 * operations — see `fastmolwidget.web_export` for a ready-made exporter.
 * Growing molecules to completion and packing whole unit cells both happen
 * here, in the browser.
 *
 * Expected JSON contract (see README.md for details):
 * ```json
 * {
 *   "cell": [a, b, c, alpha, beta, gamma],
 *   "centric": false,
 *   "symmops": ["X,Y,Z", "-X,1/2+Y,1/2-Z", ...],
 *   "atoms": [
 *     {"label": "C1", "type": "C", "x": 0.1, "y": 0.2, "z": 0.3, "part": 0,
 *      "adp": [U11, U22, U33, U23, U13, U12] }
 *   ]
 * }
 * ```
 */

import { SDM } from './sdm.js';
import { MoleculeWidget2D } from './molecule2d.js';
import { fracToCart } from './symmetry.js';

export class MoleculeViewer2D {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} [options] forwarded to `MoleculeWidget2D`.
   */
  constructor(canvas, options = {}) {
    this.widget = new MoleculeWidget2D(canvas, options);
    this._structure = null; // last loaded JSON (fractional, asymmetric unit)
    this._adpByLabel = new Map();
    this._growEnabled = false;
    this._packEnabled = false;
    this._packSymmopIndices = null;
  }

  /** Load a structure from the fractional-coordinate JSON contract. */
  loadStructure(data) {
    this._structure = data;
    this._adpByLabel = new Map(data.atoms.filter((a) => a.adp).map((a) => [a.label, a.adp]));
    this._refresh(false);
  }

  /** Enable/disable growing the asymmetric unit to complete molecules. */
  setGrow(enabled) {
    this._growEnabled = enabled;
    if (enabled) this._packEnabled = false;
    if (this._structure) this._refresh(true);
  }

  /** Enable/disable packing every symmetry-equivalent position into one unit cell.
   * @param {number[]|null} [symmopIndices] 0-based indices (identity is always 0); `null` = all. */
  setPack(enabled, symmopIndices = null) {
    this._packEnabled = enabled;
    this._packSymmopIndices = symmopIndices;
    if (enabled) this._growEnabled = false;
    if (this._structure) this._refresh(true);
  }

  _makeSdm() {
    const { atoms, symmops, cell, centric } = this._structure;
    return new SDM(atoms, symmops ?? [], cell, !!centric);
  }

  _refresh(keepView) {
    const { cell, atoms } = this._structure;
    let cartAtoms;
    if (this._packEnabled) {
      cartAtoms = this._makeSdm().packUnitCell(this._packSymmopIndices);
    } else if (this._growEnabled) {
      cartAtoms = this._makeSdm().grow();
    } else {
      cartAtoms = atoms.map((a) => {
        const [x, y, z] = fracToCart([a.x, a.y, a.z], cell);
        return { label: a.label, type: a.type, x, y, z, part: a.part ?? 0, symm_matrix: null };
      });
    }
    const withAdp = cartAtoms.map((a) => ({ ...a, adp: this._adpByLabel.get(a.label) ?? null }));
    this.widget.isPacked = this._packEnabled;
    if (keepView) {
      this.widget.growMolecule({ atoms: withAdp, cell });
      // Growing/packing changes the atom set, but `growMolecule` keeps the
      // view and therefore the bounding sphere of the *previous* (smaller)
      // atom set. Re-centre and re-fit — the rotation is preserved.
      this.widget.fitToView();
    } else {
      this.widget.openMolecule({ atoms: withAdp, cell, keepView: false });
    }
  }
}
