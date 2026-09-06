/** Symmetry parsing and fractional/cartesian conversion helpers. */

import { identity3, matVec } from './linalg.js';

function partitionAxis(symm, axisChar) {
  const idx = symm.indexOf(axisChar);
  if (idx === -1) return { coef: 0, rest: symm };
  const before = symm.slice(0, idx);
  const after = symm.slice(idx + 1);
  if (before.length > 0 && before[before.length - 1] === '-') {
    return { coef: -1, rest: before.slice(0, -1) + after };
  }
  return { coef: 1, rest: (before + after).replace(/\+/g, '') };
}

function parseTransFloat(str) {
  if (!str) return 0;
  if (/^[+-]?\d*\.?\d+$/.test(str)) return parseFloat(str);
  if (str.includes('/')) {
    const [num, denom] = str.split('/');
    return parseFloat(num) / parseFloat(denom);
  }
  const v = parseFloat(str);
  return Number.isNaN(v) ? 0 : v;
}

function parseAxisExpr(expr) {
  let rest = expr.toUpperCase().replace(/\s+/g, '');
  const row = [0, 0, 0];
  const axes = ['X', 'Y', 'Z'];
  for (let i = 0; i < 3; i++) {
    const { coef, rest: newRest } = partitionAxis(rest, axes[i]);
    row[i] = coef;
    rest = newRest;
  }
  const trans = rest ? parseTransFloat(rest) : 0;
  return { row, trans };
}

/**
 * Parse a SHELX-style symmetry op into `{ matrix, trans }`.
 * @param {string|string[]} symm Comma-joined string or 3-element array.
 */
export function parseSymmOp(symm) {
  const parts = Array.isArray(symm) ? symm : symm.split(',');
  const matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  const trans = [0, 0, 0];
  for (let i = 0; i < 3; i++) {
    const { row, trans: t } = parseAxisExpr(parts[i]);
    matrix[i] = row;
    trans[i] = t;
  }
  return { matrix, trans };
}

export function applySymmOp(op, v) {
  const p = matVec(op.matrix, v);
  return [p[0] + op.trans[0], p[1] + op.trans[1], p[2] + op.trans[2]];
}

export function identitySymmOp() {
  return { matrix: identity3(), trans: [0, 0, 0] };
}

export function matEquals(a, b, tol = 1e-6) {
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (Math.abs(a[i][j] - b[i][j]) > tol) return false;
    }
  }
  return true;
}

function mod1(x) {
  return ((x % 1) + 1) % 1;
}

export function transEquivalent(a, b, tol = 1e-6) {
  for (let i = 0; i < 3; i++) {
    if (Math.abs(mod1(a[i]) - mod1(b[i])) > tol) return false;
  }
  return true;
}

export function calcVolume(a, b, c, alpha, beta, gamma) {
  const rad = (d) => (d * Math.PI) / 180;
  const ca = Math.cos(rad(alpha));
  const cb = Math.cos(rad(beta));
  const cg = Math.cos(rad(gamma));
  return a * b * c * Math.sqrt(1 + 2 * ca * cb * cg - ca * ca - cb * cb - cg * cg);
}

/** Port of `dsrmath.frac_to_cart`. `cell` is `[a,b,c,alpha,beta,gamma]` in degrees. */
export function fracToCart(fracCoord, cell) {
  const [a, b, c, alpha, beta, gamma] = cell;
  const [x, y, z] = fracCoord;
  const rad = (d) => (d * Math.PI) / 180;
  const al = rad(alpha), be = rad(beta), ga = rad(gamma);
  const cosastar = (Math.cos(be) * Math.cos(ga) - Math.cos(al)) / (Math.sin(be) * Math.sin(ga));
  const sinastar = Math.sqrt(1 - cosastar * cosastar);
  const xc = a * x + b * Math.cos(ga) * y + c * Math.cos(be) * z;
  const yc = b * Math.sin(ga) * y + -c * Math.sin(be) * cosastar * z;
  const zc = c * Math.sin(be) * sinastar * z;
  return [xc, yc, zc];
}

/** Port of `dsrmath.cart_to_frac`. */
export function cartToFrac(cartCoord, cell) {
  const [a, b, c, alpha, beta, gamma] = cell;
  const [xc, yc, zc] = cartCoord;
  const rad = (d) => (d * Math.PI) / 180;
  const al = rad(alpha), be = rad(beta), ga = rad(gamma);
  const cosastar = (Math.cos(be) * Math.cos(ga) - Math.cos(al)) / (Math.sin(be) * Math.sin(ga));
  const sinastar = Math.sqrt(1 - cosastar * cosastar);
  const z = zc / (c * Math.sin(be) * sinastar);
  const y = (yc - -c * Math.sin(be) * cosastar * z) / (b * Math.sin(ga));
  const x = (xc - b * Math.cos(ga) * y - c * Math.cos(be) * z) / a;
  return [x, y, z];
}
