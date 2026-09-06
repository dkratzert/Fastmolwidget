import { getRadiusFromElement } from './elements.js';

/**
 * Port of `fastmolwidget.tools.build_conntable`.
 * @param {number[][]} coords Nx3 cartesian coordinates.
 * @param {string[]} types element symbols.
 * @param {number[]} parts SHELX disorder-part numbers.
 * @param {object} [opts]
 * @param {number[]} [opts.radii] Precomputed covalent radii.
 * @param {number} [opts.extraParam=1.2] bond-detection tolerance multiplier.
 * @param {boolean[]} [opts.symmgen] per-atom symmetry-generated flag.
 * @returns {Array<[number, number]>} list of `[i, j]` bonded pairs (i < j).
 */
export function buildConnTable(coords, types, parts, opts = {}) {
  const n = coords.length;
  if (n === 0) return [];

  const extraParam = opts.extraParam ?? 1.2;
  const radii = opts.radii ?? types.map((t) => getRadiusFromElement(t));
  const symmgen = opts.symmgen ?? null;
  const isH = types.map((t) => t === 'H' || t === 'D');

  const bonds = [];
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = coords[i][0] - coords[j][0];
      const dy = coords[i][1] - coords[j][1];
      const dz = coords[i][2] - coords[j][2];
      const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (d <= 0.01 || d > 4.0) continue;
      const radiiSum = (radii[i] + radii[j]) * extraParam;
      if (d >= radiiSum) continue;
      if (parts[i] !== 0 && parts[j] !== 0 && parts[i] !== parts[j]) continue;
      if (symmgen) {
        const eitherNeg = parts[i] < 0 || parts[j] < 0;
        const crossBoundary = symmgen[i] !== symmgen[j];
        if (eitherNeg && crossBoundary) continue;
      }
      if (isH[i] && isH[j]) continue;
      bonds.push([i, j]);
    }
  }
  return bonds;
}
