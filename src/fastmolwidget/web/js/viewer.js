/**
 * Convenience wrapper combining `sdm.js` grow/pack logic with the Canvas
 * renderer in `molecule2d.js`.
 *
 * Loads the fractional-coordinate JSON from `fastmolwidget.web_export`;
 * growing and packing happen in the browser.
 */

import { SDM } from './sdm.js';
import { MoleculeWidget2D } from './molecule2d.js';
import { DensityMap } from './density.js';
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
    /** Decoded density map, or `null` when absent. */
    this._densityPromise = Promise.resolve(null);
  }

  /** Load a structure from the fractional-coordinate JSON contract. */
  loadStructure(data) {
    this._structure = data;
    this._adpByLabel = new Map(data.atoms.filter((a) => a.adp).map((a) => [a.label, a.adp]));
    this.widget.clearResidualDensity();
    // Decode early so toggling density later is immediate.
    this._densityPromise = data.density
      ? DensityMap.fromPayload(data.density)
      : Promise.resolve(null);
    this._densityPromise.catch(() => {});
    this._refresh(false);
    // Lets pre-built controls resync density availability.
    this.widget.dispatchEvent(new CustomEvent('structureChanged', {
      detail: { hasDensity: this.hasDensity },
    }));
  }

  /** Whether the loaded structure ships a residual-density map. */
  get hasDensity() {
    return !!(this._structure && this._structure.density);
  }

  /**
   * Show or hide the residual (Fo-Fc) density wireframe.
   *
   * @param {boolean} visible
   * @param {number} [level] contour level in e/A^3; defaults to 3 sigma.
   * @returns {Promise<boolean>} whether density ended up being shown.
   */
  async setDensityVisible(visible, level) {
    if (!visible) {
      this.widget.clearResidualDensity();
      return false;
    }
    const map = await this._densityPromise;
    if (!map) return false;
    this.widget.showResidualDensity(map, level);
    return true;
  }

  /** Re-contour the density at *level* (e/A^3). No-op when none is shown. */
  setDensityLevel(level) {
    this.widget.setResidualDensityLevel(level);
  }

  /** Suggested contour level from the payload, in e/A^3, or `null`. */
  densitySuggestedLevel() {
    const payload = this._structure && this._structure.density;
    return payload ? payload.level : null;
  }

  /** Enable/disable growing the asymmetric unit to complete molecules. */
  setGrow(enabled) {
    this._growEnabled = enabled;
    if (enabled) this._packEnabled = false;
    if (this._structure) this._refresh(true);
  }

  /**
   * Enable/disable packing every symmetry-equivalent position into one unit cell.
   * @param {number[]|null} [symmopIndices] 0-based indices; `null` = all.
   */
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
      // `growMolecule` keeps the rotation; re-fit the new atom set.
      this.widget.fitToView();
    } else {
      this.widget.openMolecule({ atoms: withAdp, cell, keepView: false });
    }
  }
}
