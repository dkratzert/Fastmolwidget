/**
 * Canvas 2D molecule renderer — JavaScript port of
 * `fastmolwidget.molecule2D.MoleculeWidget` + `fastmolwidget.molecule_painter`.
 *
 * Renders molecules as ORTEP-style thermal-ellipsoid plots (when anisotropic
 * displacement parameters are supplied) or ball-and-stick diagrams, with
 * mouse rotate/zoom/pan, hover tooltips, click-selection, best-view (PCA),
 * and F1/F2/F3 real-space-axis alignment (requires a unit cell).
 *
 * This module only renders. Structure parsing (CIF/SHELX) and the SDM
 * grow / pack-unit-cell logic live in `sdm.js` — see `viewer.js` for a
 * convenience wrapper that wires the two together.
 *
 * Mouse controls
 * --------------
 * - Left drag:    rotate
 * - Right drag:   zoom
 * - Middle drag:  pan
 * - Scroll wheel: label font size
 * - Left click:   select atom/bond; Ctrl = multi-select; Alt = recentre pivot
 * - Middle click: recentre rotation pivot on the clicked atom
 * - F1/F2/F3:     align view to real-space axes a/b/c (needs a unit cell + focus)
 */

import { getElementColor, getRadiusFromElement } from './elements.js';
import {
  cross, eigSym3, identity3, inv3, matMul, matVec, norm, normalize, orthonormalize3, transpose, vecAdd, vecScale, vecSub,
} from './linalg.js';
import { darker, lighter } from './color.js';
import { buildConnTable } from './conntable.js';
import { calcVolume } from './symmetry.js';

const HYDROGENS = new Set(['H', 'D']);
const AUTO_ZOOM_PADDING = 1.1;

/** Half-edge of the NPD placeholder cube, as a fraction of `atomsSize`. */
const NPD_CUBE_HALF_FACTOR = 0.4;
/** Bounding-circle radius of the NPD cube, as a fraction of `atomsSize`. */
const NPD_CUBE_BOUND_FACTOR = NPD_CUBE_HALF_FACTOR * 1.7320508075688772;

// Light direction in view space (x right, y down, z away from the viewer).
const NPD_LIGHT = normalize([-0.3, -0.5, -1.0]);

// Corner index is i*4 + j*2 + k with i/j/k selecting the sign of u/v/w.
// Each face is wound so that (p1 - p0) x (p2 - p1) points outwards.
const NPD_CUBE_FACE_INDICES = [
  [4, 6, 7, 5], // +u
  [0, 1, 3, 2], // -u
  [2, 3, 7, 6], // +v
  [0, 4, 5, 1], // -v
  [1, 5, 7, 3], // +w
  [0, 2, 6, 4], // -w
];

/**
 * Projected faces of the NPD placeholder cube — port of
 * `fastmolwidget.molecule_painter.npd_cube_faces`.
 *
 * The cube is axis-aligned in the *molecular* Cartesian frame and is brought
 * into view space with `R` (the accumulated view rotation), so it turns
 * together with the rest of the structure.
 *
 * @param {number[][]} R 3x3 view rotation matrix (`cumulativeR`).
 * @param {number} half Half-edge length of the cube in screen pixels.
 * @returns {{corners: number[][], meanZ: number, normal: number[]}[]} faces
 *   sorted back-to-front (descending depth; smaller z is nearer the viewer).
 *   `corners` are screen-space offsets relative to the atom centre.
 */
export function npdCubeFaces(R, half) {
  const u = [R[0][0] * half, R[1][0] * half, R[2][0] * half];
  const v = [R[0][1] * half, R[1][1] * half, R[2][1] * half];
  const w = [R[0][2] * half, R[1][2] * half, R[2][2] * half];
  const corners = [];
  for (const si of [-1, 1]) {
    for (const sj of [-1, 1]) {
      for (const sk of [-1, 1]) {
        corners.push([
          si * u[0] + sj * v[0] + sk * w[0],
          si * u[1] + sj * v[1] + sk * w[1],
          si * u[2] + sj * v[2] + sk * w[2],
        ]);
      }
    }
  }
  const faces = NPD_CUBE_FACE_INDICES.map((idx) => {
    const pts = idx.map((i) => corners[i]);
    let normal = cross(vecSub(pts[1], pts[0]), vecSub(pts[2], pts[1]));
    const n = norm(normal);
    if (n > 1e-12) normal = vecScale(normal, 1 / n);
    return {
      corners: pts.map((p) => [p[0], p[1]]),
      meanZ: (pts[0][2] + pts[1][2] + pts[2][2] + pts[3][2]) / 4,
      normal,
    };
  });
  faces.sort((a, b) => b.meanZ - a.meanZ);
  return faces;
}

/** Lambert brightness factor for a cube face normal (1.0 = base colour). */
export function npdFaceShade(normal) {
  const diffuse = Math.max(
    0, normal[0] * NPD_LIGHT[0] + normal[1] * NPD_LIGHT[1] + normal[2] * NPD_LIGHT[2],
  );
  return Math.min(1.6, Math.max(0.45, 0.6 + 0.85 * diffuse));
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function isIdentity(m, tol = 1e-9) {
  const I = identity3();
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) if (Math.abs(m[i][j] - I[i][j]) > tol) return false;
  return true;
}

function sortedBondKey(a, b) {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}

function dot3(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

/** Internal per-atom render state (mirrors `molecule_painter.Atom`). */
class Atom {
  constructor(x, y, z, name, type_, part) {
    this.coordinate = [x, y, z];
    this.name = name;
    this.type = type_;
    this.part = part;
    this.symmgen = false;
    this.radius = getRadiusFromElement(type_);
    this.screenx = 0;
    this.screeny = 0;
    this.z = z;
    this.color = getElementColor(type_);
    this.colorLight = lighter(this.color, 160);
    this.colorDark = darker(this.color, 180);
    this.uCart = null; // 3x3 or null
    this.uIso = null;
    this.uEigvals = null; // ascending [l0,l1,l2]
    this.uEigvecs = null; // columns
    this.uInv = null;
    this.adpValid = true;
  }
}

export class MoleculeWidget2D extends EventTarget {
  static AUTO_ZOOM_PADDING = AUTO_ZOOM_PADDING;

  /**
   * @param {HTMLCanvasElement} canvas
   * @param {object} [options]
   * @param {boolean} [options.attachEvents=true] Wire up mouse/wheel/keyboard handlers.
   */
  constructor(canvas, options = {}) {
    super();
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');

    // HiDPI support: the canvas backing store is allocated at
    // devicePixelRatio while all drawing happens in logical (CSS) pixels,
    // exactly like Qt renders its widget at the screen's device pixel ratio.
    // This keeps lines crisp on high-density displays and makes line
    // thicknesses match the Qt QPainter widget 1:1. `options.devicePixelRatio`
    // forces a fixed ratio (e.g. for deterministic tests / exports).
    this._forcedDpr = options.devicePixelRatio ?? null;
    this.dpr = this._detectDpr();
    // Device scale of the frame currently being rendered; see `render()`.
    this._renderScale = this.dpr;
    // Fallback for callers that never call `resize()`; note that an
    // unstyled <canvas> reports the HTML default of 300x150, which is not a
    // meaningful viewport size. `_sized` therefore records whether a real
    // measurement has been seen yet — see `resize()`.
    this._cssWidth = canvas.width;
    this._cssHeight = canvas.height;
    this._sized = false;

    this.zoom = 1.0;
    this.fontsize = 10;
    this.bondWidth = 3;
    this.labels = false;
    this.showAdpsFlag = true;
    this.showHydrogensFlag = true;

    this.availableParts = new Set();
    this.visibleParts = null;

    this.selectedAtoms = new Set();
    this.selectedBonds = new Set(); // "name1|name2" sorted

    this.adpScale = 1.5382;

    this.moleculeCenter = [0, 0, 0];
    this.moleculeRadius = 10;

    this.lastPos = { x: 0, y: 0 };
    this.pressPos = { x: 0, y: 0 };

    this.scale = 150.0;
    this.cxGlobal = 0.0;
    this.cyGlobal = 0.0;
    this.cumulativeR = identity3();

    this.bgColor = '#ffffff';
    this.bondColor = '#555555';
    this.fallbackPenColor = '#000000';
    this.adpPenColor = 'rgba(0,0,0,1)';

    this.atoms = [];
    this.connections = [];
    this.objects = [];
    this.screenCenter = [0, 0];

    this.hoveredAtom = null;
    this.hoveredBond = null;
    this.hoveredBondDistance = null;
    this.hoverCursor = null;

    this.cachedAdpLineWidth = 1.0;

    this.cell = null;
    this.amatrix = null;
    this.astar = null;
    this.bstar = null;
    this.cstar = null;
    this.isPacked = false;

    this._rafPending = false;
    this._dragButton = undefined;

    if (options.attachEvents !== false) this._attachEvents();
  }

  // ------------------------------------------------------------------
  // Public API — display settings
  // ------------------------------------------------------------------

  setBondWidth(width) {
    this.bondWidth = width;
    this.update();
  }

  setBondColor(color) {
    this.bondColor = color;
    this.update();
  }

  setLabelsVisible(visible) {
    this.labels = visible;
    this.update();
  }

  showLabels(value) {
    this.labels = value;
    this.update();
  }

  showHydrogens(value) {
    this.showHydrogensFlag = value;
    this.update();
  }

  showAdps(value) {
    this.showAdpsFlag = value;
    this.update();
  }

  setVisibleParts(parts) {
    this.visibleParts = parts;
    this.update();
  }

  setBackgroundColor(color) {
    this.bgColor = color;
    this.update();
  }

  setLabelFont(size) {
    this.fontsize = size < 0 ? 1 : size;
    this.update();
  }

  resetView() {
    this.zoom = this._autoZoom();
    this.cumulativeR = identity3();
    this.update();
  }

  resetRotationCenter() {
    this._getCenterAndRadius();
    this.update();
  }

  /** Re-centre the rotation pivot on the current atoms and fit them into the
   * viewport, *keeping* the current rotation.
   *
   * There is no single Qt counterpart: it is `reset_rotation_center()`
   * followed by the zoom part of `reset_view()` — the combination the Qt
   * desktop applications use after growing or packing a structure. Needed
   * because loading with `keepView` deliberately does not touch the bounding
   * sphere, so the auto-zoom would otherwise still fit the asymmetric unit. */
  fitToView() {
    this._getCenterAndRadius();
    this.zoom = this._autoZoom();
    this.update();
  }

  clear() {
    this.openMolecule({ atoms: [] });
  }

  // ------------------------------------------------------------------
  // Molecule loading
  // ------------------------------------------------------------------

  /**
   * @param {object} params
   * @param {Array<{label,type,x,y,z,part?,symm_matrix?,adp?}>} params.atoms Cartesian Å coordinates.
   * @param {number[]} [params.cell] `[a,b,c,alpha,beta,gamma]`.
   * @param {boolean} [params.keepView=false]
   */
  openMolecule({ atoms, cell = null, keepView = false }) {
    this.isPacked = false;
    this._loadMolecule(atoms, cell, keepView);
  }

  growMolecule({ atoms, cell = null }) {
    this._loadMolecule(atoms, cell, true);
  }

  _loadMolecule(atomsInput, cell, keepView) {
    this.cell = cell;
    if (this.cell) this._calcAmatrix();

    this._makeAdps(atomsInput ?? []);
    this.connections = this._getConntableFromAtoms();

    this.availableParts = new Set(this.atoms.map((a) => a.part));
    this.visibleParts = null;
    this.dispatchEvent(new CustomEvent('partsChanged', { detail: this.availableParts }));

    if (!keepView) {
      this._getCenterAndRadius();
      this.cumulativeR = identity3();
      this.selectedAtoms.clear();
      this.selectedBonds.clear();
    }

    this.objects = [];
    for (const [n1, n2] of this.connections) {
      this.objects.push({ isBond: true, zOrder: 0, atom1: this.atoms[n1], atom2: this.atoms[n2] });
    }
    for (const atom of this.atoms) {
      if (atom.type === 'H' || atom.type === 'D') atom.uIso = 0.01;
      this.objects.push({ isBond: false, zOrder: 0, atom1: atom, atom2: null });
    }

    const carryRotation = keepView && !isIdentity(this.cumulativeR);
    for (const at of this.atoms) {
      if (carryRotation) {
        at.coordinate = vecAdd(matVec(this.cumulativeR, vecSub(at.coordinate, this.moleculeCenter)), this.moleculeCenter);
      }
      at.z = at.coordinate[2];
      if (at.uCart) {
        if (carryRotation) {
          at.uCart = matMul(this.cumulativeR, matMul(at.uCart, transpose(this.cumulativeR)));
        }
        this._refreshAdpDerived(at);
      }
    }

    if (!keepView) this.zoom = this._autoZoom();
    this.update();
  }

  _refreshAdpDerived(at) {
    try {
      const { values, vectors } = eigSym3(at.uCart);
      at.adpValid = values.every((v) => v > 0);
      const invU = inv3(at.uCart);
      if (!invU) throw new Error('singular');
      at.uEigvals = values;
      at.uEigvecs = vectors;
      at.uInv = invU;
    } catch {
      at.adpValid = false;
      at.uCart = null;
      at.uIso = null;
    }
  }

  _calcAmatrix() {
    const [a, b, c, alpha, beta, gamma] = this.cell;
    const rad = (d) => (d * Math.PI) / 180;
    const V = calcVolume(a, b, c, alpha, beta, gamma);
    this.astar = (b * c * Math.sin(rad(alpha))) / V;
    this.bstar = (c * a * Math.sin(rad(beta))) / V;
    this.cstar = (a * b * Math.sin(rad(gamma))) / V;
    this.amatrix = [
      [a, b * Math.cos(rad(gamma)), c * Math.cos(rad(beta))],
      [0, b * Math.sin(rad(gamma)), (c * (Math.cos(rad(alpha)) - Math.cos(rad(beta)) * Math.cos(rad(gamma)))) / Math.sin(rad(gamma))],
      [0, 0, V / (a * b * Math.sin(rad(gamma)))],
    ];
  }

  _makeAdps(atomsInput) {
    this.atoms = [];
    const nameCounts = new Map();
    for (const at of atomsInput) {
      const base = at.label;
      const count = nameCounts.get(base) ?? 0;
      const internalName = count === 0 ? base : `${base}>>${count}`;
      nameCounts.set(base, count + 1);

      const a = new Atom(at.x, at.y, at.z, internalName, at.type, at.part ?? 0);
      const symmMatrix = at.symm_matrix ?? null;
      if (symmMatrix) a.symmgen = !isIdentity(symmMatrix);
      if (at.adp && this.cell) {
        try {
          a.uCart = this._uijToCart(at.adp, symmMatrix);
          a.uIso = (a.uCart[0][0] + a.uCart[1][1] + a.uCart[2][2]) / 3.0;
        } catch {
          a.uCart = null;
          a.uIso = null;
        }
      }
      this.atoms.push(a);
    }
  }

  _uijToCart(uvals, symmMatrix) {
    const [U11, U22, U33, U23, U13, U12] = uvals;
    let Uij = [
      [U11, U12, U13],
      [U12, U22, U23],
      [U13, U23, U33],
    ];
    if (symmMatrix) Uij = matMul(transpose(symmMatrix), matMul(Uij, symmMatrix));
    const N = [[this.astar, 0, 0], [0, this.bstar, 0], [0, 0, this.cstar]];
    return matMul(matMul(matMul(matMul(this.amatrix, N), Uij), transpose(N)), transpose(this.amatrix));
  }

  _getConntableFromAtoms(extraParam = 1.2) {
    const coords = this.atoms.map((a) => a.coordinate);
    const types = this.atoms.map((a) => a.type);
    const parts = this.atoms.map((a) => a.part);
    const radii = this.atoms.map((a) => a.radius);
    const symmgen = this.atoms.map((a) => a.symmgen);
    return buildConnTable(coords, types, parts, { radii, extraParam, symmgen });
  }

  _getCenterAndRadius() {
    if (this.atoms.length === 0) {
      this.moleculeCenter = [0, 0, 0];
      this.moleculeRadius = 10;
      return;
    }
    const min_ = [Infinity, Infinity, Infinity];
    const max_ = [-Infinity, -Infinity, -Infinity];
    for (const at of this.atoms) {
      for (let j = 0; j < 3; j++) {
        min_[j] = Math.min(min_[j], at.coordinate[j]);
        max_[j] = Math.max(max_[j], at.coordinate[j]);
      }
    }
    const c = [0, 1, 2].map((j) => (max_[j] + min_[j]) / 2);
    let r = 0;
    for (const at of this.atoms) {
      const d = norm(vecSub(at.coordinate, c)) + 1.5;
      if (d > r) r = d;
    }
    this.moleculeCenter = c;
    this.moleculeRadius = r || 10;
  }

  // ------------------------------------------------------------------
  // View control
  // ------------------------------------------------------------------

  _alignToReciprocalAxis(axisIndex) {
    if (!this.amatrix || !this.cell) return;
    const direct = normalize([this.amatrix[0][axisIndex], this.amatrix[1][axisIndex], this.amatrix[2][axisIndex]]);
    const zAxis = direct;
    let up = [0, 1, 0];
    if (Math.abs(dot3(zAxis, up)) > 0.99) up = [1, 0, 0];
    const xAxis = normalize(cross(up, zAxis));
    const yAxis = normalize(cross(zAxis, xAxis));
    this._rotateTo([xAxis, yAxis, zAxis]);
  }

  alignBestView() {
    if (this.atoms.length < 2) return;
    const visible = this.showHydrogensFlag ? this.atoms : this.atoms.filter((a) => !HYDROGENS.has(a.type));
    if (visible.length < 2) return;
    const mean = [0, 0, 0];
    for (const at of visible) for (let j = 0; j < 3; j++) mean[j] += at.coordinate[j] / visible.length;
    const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (const at of visible) {
      const d = vecSub(at.coordinate, mean);
      for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += d[i] * d[j];
    }
    const { vectors } = eigSym3(cov);
    // eigSym3 returns ascending eigenvalues; PCA wants descending variance order.
    const order = [2, 1, 0];
    const cols = order.map((i) => [vectors[0][i], vectors[1][i], vectors[2][i]]);
    const [xAxis, yAxis] = cols;
    let zAxis = cols[2];
    if (dot3(cross(xAxis, yAxis), zAxis) < 0) zAxis = vecScale(zAxis, -1);
    this._rotateTo([xAxis, yAxis, zAxis]);
  }

  _rotateTo(targetR) {
    const invCurrent = inv3(this.cumulativeR) ?? identity3();
    const deltaR = matMul(targetR, invCurrent);
    this._applyDeltaRotation(deltaR);
    this.cumulativeR = orthonormalize3(targetR);
    this.update();
  }

  _applyDeltaRotation(deltaR) {
    // Rotate the cached eigen-decomposition rigidly instead of recomputing it
    // from scratch (mirrors the Python/Qt renderer's `rotate_molecule`).
    // Re-deriving eigenvectors every drag frame via the analytic eigSym3
    // solver is numerically unstable whenever an atom has two (near-)equal
    // ADP eigenvalues (e.g. an atom sitting on a symmetry axis): the
    // null-space computation degenerates and eigSym3 falls back to an
    // arbitrary fixed world-axis vector, unrelated to the atom's actual
    // orientation. That made the principal-axis cross-section lines jump to
    // nonsensical directions while the ellipse body (computed independently
    // from the 2x2 projected covariance) stayed correct. Rotating the
    // eigenvectors themselves is exact (eigenvalues are rotation-invariant)
    // and keeps everything consistent frame to frame.
    for (const at of this.atoms) {
      at.coordinate = vecAdd(matVec(deltaR, vecSub(at.coordinate, this.moleculeCenter)), this.moleculeCenter);
      at.z = at.coordinate[2];
      if (at.uCart) {
        at.uCart = matMul(deltaR, matMul(at.uCart, transpose(deltaR)));
        if (at.uEigvecs) at.uEigvecs = matMul(deltaR, at.uEigvecs);
        if (at.uInv) at.uInv = matMul(deltaR, matMul(at.uInv, transpose(deltaR)));
      }
    }
  }

  // ------------------------------------------------------------------
  // Geometry helpers
  // ------------------------------------------------------------------

  get atomsSize() {
    return this.zoom * 70;
  }

  _detectDpr() {
    if (this._forcedDpr != null) return this._forcedDpr;
    return (typeof window !== 'undefined' && window.devicePixelRatio) || 1;
  }

  /** CSS (logical) pixel width of the drawing surface. */
  get cssWidth() { return this._cssWidth; }

  /** CSS (logical) pixel height of the drawing surface. */
  get cssHeight() { return this._cssHeight; }

  _autoZoom() {
    const w = this._cssWidth, h = this._cssHeight;
    const r = this.moleculeRadius;
    if (w <= 0 || h <= 0 || r <= 0) return AUTO_ZOOM_PADDING / 100;
    return (AUTO_ZOOM_PADDING * Math.min(w, h)) / 2 / r / 100;
  }

  _adpIntersectionLineWidth() {
    return clamp(this.zoom * 3.0, 1.0, 6.0);
  }

  /** Resize the drawing surface. `cssWidth`/`cssHeight` are logical (CSS)
   * pixels — normally the element's `getBoundingClientRect()` size. The
   * backing store is allocated at `devicePixelRatio` so lines stay crisp on
   * HiDPI displays, mirroring Qt rendering the widget at the screen's device
   * pixel ratio. Keeps the on-screen scale proportional (like Qt's
   * `resizeEvent`).
   *
   * The *first* call with positive dimensions re-fits the molecule instead of
   * scaling proportionally: until then the widget only knows the placeholder
   * `canvas.width`/`canvas.height` (300x150 by default), so scaling from that
   * baseline would produce a wildly wrong zoom. This matters for viewers
   * created inside a hidden container (e.g. an inactive report tab), whose
   * real size only arrives later via a `ResizeObserver`. */
  resize(cssWidth, cssHeight) {
    if (!(cssWidth > 0) || !(cssHeight > 0)) return;
    this.dpr = this._detectDpr();
    const oldMin = this._sized ? Math.min(this._cssWidth, this._cssHeight) : 0;
    this._cssWidth = cssWidth;
    this._cssHeight = cssHeight;
    this._sized = true;
    this.canvas.width = Math.max(1, Math.round(cssWidth * this.dpr));
    this.canvas.height = Math.max(1, Math.round(cssHeight * this.dpr));
    const newMin = Math.min(cssWidth, cssHeight);
    if (oldMin > 0) this.zoom *= newMin / oldMin;
    else this.zoom = this._autoZoom();
    this.update();
  }

  /** Backwards-compatible resize entry point. `newW`/`newH` are treated as
   * logical (CSS) pixels and forwarded to {@link resize}. */
  handleResize(oldW, oldH, newW, newH) {
    this.resize(newW, newH);
  }

  getSphericalRadius(atom) {
    if (atom.uCart && !atom.adpValid) {
      return (this.atomsSize * NPD_CUBE_BOUND_FACTOR) / this.scale;
    }
    if (this.showAdpsFlag && atom.uIso != null) return Math.sqrt(atom.uIso);
    return 0.23;
  }

  getDirectionalRadius(atom, v) {
    const d = norm(v);
    if (d < 1e-8) return 0.23;
    if (!atom.adpValid) return 0.23;
    if (this.showAdpsFlag && atom.uInv) {
      const u = vecScale(v, 1 / d);
      const t = matVec(atom.uInv, u);
      const val = u[0] * t[0] + u[1] * t[1] + u[2] * t[2];
      if (val > 0) return this.adpScale / Math.sqrt(val);
    }
    if (this.showAdpsFlag && atom.uIso != null) return Math.sqrt(atom.uIso) * this.adpScale;
    return 0.23;
  }

  // ------------------------------------------------------------------
  // Hit-testing
  // ------------------------------------------------------------------

  isPointInsideAtom(atom, px, py) {
    const dx = px - atom.screenx, dy = py - atom.screeny;
    if (atom.uCart && !atom.adpValid) {
      const bound = this.atomsSize * NPD_CUBE_BOUND_FACTOR;
      return dx * dx + dy * dy <= bound * bound;
    }
    if (this.showAdpsFlag && atom.uCart) {
      const a = atom.uCart[0][0], b = atom.uCart[0][1], c = atom.uCart[1][1];
      const T = a + c, D = a * c - b * b, diff = T * T * 0.25 - D;
      if (diff >= 0) {
        const sq = Math.sqrt(diff);
        const eig1 = T * 0.5 - sq, eig2 = T * 0.5 + sq;
        if (eig1 > 0 && eig2 > 0) {
          const r1 = Math.sqrt(eig1) * this.scale * this.adpScale;
          const r2 = Math.sqrt(eig2) * this.scale * this.adpScale;
          const angle = Math.abs(b) > 1e-8 ? Math.atan2(eig1 - a, b) : (a < c ? 0 : Math.PI / 2);
          const cosA = Math.cos(angle), sinA = Math.sin(angle);
          const localX = dx * cosA + dy * sinA;
          const localY = -dx * sinA + dy * cosA;
          return (localX * localX) / (r1 * r1) + (localY * localY) / (r2 * r2) <= 1.0;
        }
      }
    }
    let circleSize = this.atomsSize;
    if (this.showAdpsFlag && atom.uIso != null) circleSize = Math.sqrt(atom.uIso) * this.scale * this.adpScale * 2;
    return dx * dx + dy * dy <= (circleSize / 2) ** 2;
  }

  _getBondLine(at1, at2) {
    const c1 = at1.coordinate, c2 = at2.coordinate;
    const v = vecSub(c2, c1);
    const d = norm(v);
    const r1 = this.getDirectionalRadius(at1, v);
    const r2 = this.getDirectionalRadius(at2, vecScale(v, -1));
    if (d <= r1 + r2) return null;
    const vNorm = vecScale(v, 1 / d);
    const p1 = vecAdd(c1, vecScale(vNorm, r1));
    const p2 = vecSub(c2, vecScale(vNorm, r2));
    const x1 = p1[0] * this.scale + this.cxGlobal, y1 = p1[1] * this.scale + this.cyGlobal;
    const x2 = p2[0] * this.scale + this.cxGlobal, y2 = p2[1] * this.scale + this.cyGlobal;
    const dynamicWidth = Math.max(1, Math.trunc(this.bondWidth * this.zoom * 5));
    return { x1, y1, x2, y2, width: dynamicWidth };
  }

  isPointNearBond(at1, at2, px, py) {
    const line = this._getBondLine(at1, at2);
    if (!line) return false;
    const { x1, y1, x2, y2, width } = line;
    const lx = x2 - x1, ly = y2 - y1;
    const lenSq = lx * lx + ly * ly;
    if (lenSq === 0) return false;
    const t = clamp(((px - x1) * lx + (py - y1) * ly) / lenSq, 0, 1);
    const projX = x1 + t * lx, projY = y1 + t * ly;
    const distSq = (px - projX) ** 2 + (py - projY) ** 2;
    const tolerance = Math.max(5.0, width / 2.0 + 4.0);
    return distSq <= tolerance * tolerance;
  }

  // ------------------------------------------------------------------
  // Mouse / keyboard interaction
  // ------------------------------------------------------------------

  _attachEvents() {
    const canvas = this.canvas;
    if (canvas.tabIndex < 0) canvas.tabIndex = 0;
    if (!canvas.style.touchAction) canvas.style.touchAction = 'none';
    // Prevent the browser from starting a text/element selection drag over
    // the canvas — without this, dragging (e.g. to rotate) can trigger the
    // browser's "auto-scroll toward viewport edge to extend selection"
    // behaviour, which visibly shifts the whole page.
    canvas.style.userSelect = 'none';
    canvas.style.webkitUserSelect = 'none';

    canvas.addEventListener('contextmenu', (e) => e.preventDefault());
    canvas.addEventListener('mousedown', (e) => this._onMouseDown(e));
    canvas.addEventListener('mousemove', (e) => {
      if (this._dragButton === undefined) {
        const pos = this._eventPos(e);
        this._updateHover(pos.x, pos.y);
      }
    });
    canvas.addEventListener('mouseleave', () => this._onLeave());
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.deltaY < 0) this.setLabelFont(this.fontsize + 2);
      else if (e.deltaY > 0) this.setLabelFont(this.fontsize - 2);
    }, { passive: false });
    canvas.addEventListener('keydown', (e) => {
      if (e.key === 'F1') { e.preventDefault(); this._alignToReciprocalAxis(0); }
      else if (e.key === 'F2') { e.preventDefault(); this._alignToReciprocalAxis(1); }
      else if (e.key === 'F3') { e.preventDefault(); this._alignToReciprocalAxis(2); }
    });
  }

  _eventPos(e) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left) * (this._cssWidth / rect.width),
      y: (e.clientY - rect.top) * (this._cssHeight / rect.height),
    };
  }

  _onMouseDown(e) {
    // preventScroll avoids the browser scrolling the canvas into view (which
    // would otherwise shift the whole page on every click-drag).
    this.canvas.focus?.({ preventScroll: true });
    const pos = this._eventPos(e);
    this.lastPos = pos;
    this.pressPos = pos;
    this._dragButton = e.button;
    this._boundMove = (ev) => this._onDragMove(ev);
    this._boundUp = (ev) => this._onMouseUp(ev);
    window.addEventListener('mousemove', this._boundMove);
    window.addEventListener('mouseup', this._boundUp);
    e.preventDefault();
  }

  _onDragMove(e) {
    // Prevent the browser from starting/extending a text selection while
    // dragging, which on many browsers auto-scrolls the page toward the
    // viewport edge the cursor approaches (visible as the toolbar "shifting
    // upward" during an upward rotate-drag).
    e.preventDefault();
    const pos = this._eventPos(e);
    if (this._dragButton === 0) {
      this.rotateMolecule(pos.x - this.lastPos.x, pos.y - this.lastPos.y);
      this._clearHoverState();
    } else if (this._dragButton === 2) {
      this.zoomMolecule(this.lastPos.y - pos.y);
      this._clearHoverState();
    } else if (this._dragButton === 1) {
      this.panMolecule(this.lastPos.x - pos.x, this.lastPos.y - pos.y);
      this._clearHoverState();
    }
    this.lastPos = pos;
  }

  _onMouseUp(e) {
    window.removeEventListener('mousemove', this._boundMove);
    window.removeEventListener('mouseup', this._boundUp);
    const pos = this._eventPos(e);
    const dx = pos.x - this.pressPos.x, dy = pos.y - this.pressPos.y;
    const isClick = Math.abs(dx) < 5 && Math.abs(dy) < 5;

    if (e.button === 1) {
      if (isClick) this._recenterOnClick(pos.x, pos.y);
    } else if (e.button === 0 && isClick) {
      if (e.altKey) {
        this._recenterOnClick(pos.x, pos.y);
      } else {
        this._handleLeftClick(pos.x, pos.y, e.ctrlKey || e.metaKey);
      }
    }
    this._dragButton = undefined;
  }

  _handleLeftClick(x, y, ctrl) {
    let clickedAtom = null, clickedBond = null, frontZ = Infinity;
    for (const item of this.objects) {
      if (!this.showHydrogensFlag) {
        if (HYDROGENS.has(item.atom1.type) || (item.isBond && HYDROGENS.has(item.atom2.type))) continue;
      }
      if (item.isBond) {
        if (this.isPointNearBond(item.atom1, item.atom2, x, y) && item.zOrder < frontZ) {
          frontZ = item.zOrder;
          clickedBond = [item.atom1.name, item.atom2.name].sort();
          clickedAtom = null;
        }
      } else if (this.isPointInsideAtom(item.atom1, x, y) && item.zOrder < frontZ) {
        frontZ = item.zOrder;
        clickedAtom = item.atom1;
        clickedBond = null;
      }
    }

    let changed = false;
    if (clickedAtom) {
      if (ctrl) {
        if (this.selectedAtoms.has(clickedAtom.name)) this.selectedAtoms.delete(clickedAtom.name);
        else this.selectedAtoms.add(clickedAtom.name);
      } else {
        this.selectedAtoms = new Set([clickedAtom.name]);
        this.selectedBonds.clear();
      }
      changed = true;
      this.dispatchEvent(new CustomEvent('atomClicked', { detail: clickedAtom.name }));
    } else if (clickedBond) {
      const key = sortedBondKey(clickedBond[0], clickedBond[1]);
      if (ctrl) {
        if (this.selectedBonds.has(key)) this.selectedBonds.delete(key);
        else this.selectedBonds.add(key);
      } else {
        this.selectedBonds = new Set([key]);
        this.selectedAtoms.clear();
      }
      changed = true;
      this.dispatchEvent(new CustomEvent('bondClicked', { detail: clickedBond }));
    } else if (!ctrl && (this.selectedAtoms.size || this.selectedBonds.size)) {
      this.selectedAtoms.clear();
      this.selectedBonds.clear();
      changed = true;
    }
    if (changed) this.update();
  }

  rotateMolecule(dxScreen, dyScreen) {
    const yAngle = -dxScreen / 100;
    const xAngle = dyScreen / 100;
    const Rx = [[1, 0, 0], [0, Math.cos(xAngle), -Math.sin(xAngle)], [0, Math.sin(xAngle), Math.cos(xAngle)]];
    const Ry = [[Math.cos(yAngle), 0, Math.sin(yAngle)], [0, 1, 0], [-Math.sin(yAngle), 0, Math.cos(yAngle)]];
    const R = matMul(Rx, Ry);
    // Re-orthonormalize after composing: many small rotation multiplications
    // in a row (every mousemove of a drag, over a long session) otherwise
    // accumulate floating-point drift away from a proper rotation matrix,
    // which would gradually distort the rendered geometry.
    this.cumulativeR = orthonormalize3(matMul(R, this.cumulativeR));
    this._applyDeltaRotation(R);
    this.update();
  }

  zoomMolecule(deltaYScreen) {
    const delta = deltaYScreen / 400;
    this.zoom = Math.max(0.005, this.zoom - delta);
    this.update();
  }

  panMolecule(dxScreen, dyScreen) {
    this.moleculeCenter[0] += dxScreen / 50;
    this.moleculeCenter[1] += dyScreen / 50;
    this.update();
  }

  _recenterOnClick(px, py) {
    if (!this.atoms.length) return;
    let clicked = null, frontZ = Infinity;
    for (const item of this.objects) {
      if (item.isBond) continue;
      if (!this.showHydrogensFlag && HYDROGENS.has(item.atom1.type)) continue;
      if (this.isPointInsideAtom(item.atom1, px, py) && item.zOrder < frontZ) {
        frontZ = item.zOrder;
        clicked = item.atom1;
      }
    }
    if (clicked) {
      this.moleculeCenter = [...clicked.coordinate];
      this.update();
    }
  }

  _clearHoverState() {
    if (this.hoveredAtom !== null || this.hoveredBond !== null) {
      this.hoveredAtom = null;
      this.hoveredBond = null;
      this.hoveredBondDistance = null;
      this.hoverCursor = null;
    }
  }

  _onLeave() {
    const changed = this.hoveredAtom !== null || this.hoveredBond !== null;
    this._clearHoverState();
    if (changed) this.update();
  }

  _updateHover(px, py) {
    if (!this.atoms.length) return;
    let newAtom = null, newBond = null, newDist = null, frontZ = Infinity;
    for (const item of this.objects) {
      if (item.isBond) {
        const at1 = item.atom1, at2 = item.atom2;
        if (!this.showHydrogensFlag && (HYDROGENS.has(at1.type) || HYDROGENS.has(at2.type))) continue;
        if (this.visibleParts && (!this.visibleParts.has(at1.part) || !this.visibleParts.has(at2.part))) continue;
        if (this.isPointNearBond(at1, at2, px, py) && item.zOrder < frontZ) {
          frontZ = item.zOrder;
          newBond = [at1.name, at2.name].sort();
          newAtom = null;
          newDist = norm(vecSub(at1.coordinate, at2.coordinate));
        }
      } else {
        const atom = item.atom1;
        if (!this.showHydrogensFlag && HYDROGENS.has(atom.type)) continue;
        if (this.visibleParts && !this.visibleParts.has(atom.part)) continue;
        if (this.isPointInsideAtom(atom, px, py) && item.zOrder < frontZ) {
          frontZ = item.zOrder;
          newAtom = atom.name;
          newBond = null;
          newDist = null;
        }
      }
    }
    const changed = newAtom !== this.hoveredAtom || newBond?.join() !== this.hoveredBond?.join()
      || (newBond !== null && (this.hoverCursor?.x !== px || this.hoverCursor?.y !== py));
    this.hoveredAtom = newAtom;
    this.hoveredBond = newBond;
    this.hoveredBondDistance = newDist;
    this.hoverCursor = newBond !== null ? { x: px, y: py } : null;
    if (changed) this.update();
  }

  // ------------------------------------------------------------------
  // Scheduling
  // ------------------------------------------------------------------

  /** Schedule a re-render on the next animation frame (debounced). */
  update() {
    if (this._rafPending) return;
    this._rafPending = true;
    requestAnimationFrame(() => {
      this._rafPending = false;
      this.render();
    });
  }

  /** Render immediately onto `this.canvas` (or a supplied context/size for export).
   * `scale` sets the baseline transform (used by `toDataURL`/`saveImage` to
   * render at higher resolution without the caller having to pre-apply
   * `ctx.scale()`, which would otherwise be wiped out by the reset below). */
  render(ctx = this.ctx, width = this._cssWidth, height = this._cssHeight,
    scale = (this._cssWidth > 0 ? this.canvas.width / this._cssWidth : 1)) {
    // Always start from a clean, known state. If a previous frame threw
    // between a ctx.save() and its matching ctx.restore() (e.g. an
    // unexpected NaN or an invalid colour string reaching a canvas API that
    // throws), the active transform could otherwise stay leaked — every
    // following frame would then render translated/rotated/skewed, which
    // looks like darkened, streaky "stripes" across the canvas. Resetting
    // the transform up front makes that failure mode structurally
    // impossible regardless of what caused the earlier exception.
    ctx.setTransform(scale, 0, 0, scale, 0, 0);
    // Remember the device scale actually in effect for this frame. Qt strokes
    // the ADP intersection lines with a *cosmetic* pen, whose width is in
    // device pixels; the canvas draws in logical pixels, so those strokes have
    // to divide by this factor to come out the same thickness. Taken from
    // `scale` rather than `this.dpr` so high-resolution exports match too.
    this._renderScale = scale || 1;
    ctx.save();
    try {
      this._renderScene(ctx, width, height);
    } catch (err) {
      // Mirrors the Python widget's paintEvent try/except: never let a
      // single bad frame corrupt the canvas or crash the caller.
      console.error('MoleculeWidget2D render failed:', err);
    } finally {
      ctx.restore();
    }
  }

  _renderScene(ctx, width, height) {
    ctx.fillStyle = this.bgColor;
    ctx.fillRect(0, 0, width, height);

    if (this.atoms.length === 0) {
      return;
    }

    ctx.font = `${Math.max(1, Math.round(this.fontsize * this.zoom * 4))}px sans-serif`;
    ctx.textBaseline = 'alphabetic';

    this.scale = this.zoom * 130;
    this.screenCenter = [width / 2, height / 2];
    this.cxGlobal = this.screenCenter[0] - this.moleculeCenter[0] * this.scale;
    this.cyGlobal = this.screenCenter[1] - this.moleculeCenter[1] * this.scale;
    this.cachedAdpLineWidth = this._adpIntersectionLineWidth();

    const margin = this.scale * this.adpScale * 2.0 + 40.0;
    const vpLeft = -margin, vpTop = -margin, vpRight = width + margin, vpBottom = height + margin;

    for (const atom of this.atoms) {
      atom.screenx = atom.coordinate[0] * this.scale + this.cxGlobal;
      atom.screeny = atom.coordinate[1] * this.scale + this.cyGlobal;
    }

    this._calculateZOrder();

    const labelAtoms = [];
    for (const item of this.objects) {
      if (!this.showHydrogensFlag) {
        if (HYDROGENS.has(item.atom1.type) || (item.isBond && HYDROGENS.has(item.atom2.type))) continue;
      }
      if (this.visibleParts) {
        if (!this.visibleParts.has(item.atom1.part)) continue;
        if (item.isBond && !this.visibleParts.has(item.atom2.part)) continue;
      }
      if (item.isBond) {
        const a1 = item.atom1, a2 = item.atom2;
        if ((a1.screenx < vpLeft && a2.screenx < vpLeft) || (a1.screenx > vpRight && a2.screenx > vpRight)
          || (a1.screeny < vpTop && a2.screeny < vpTop) || (a1.screeny > vpBottom && a2.screeny > vpBottom)) continue;
        this._drawBondRounded(ctx, a1, a2);
      } else {
        const sx = item.atom1.screenx, sy = item.atom1.screeny;
        if (sx < vpLeft || sx > vpRight || sy < vpTop || sy > vpBottom) continue;
        this._drawAtom(ctx, item.atom1);
        const isHovered = item.atom1.name === this.hoveredAtom;
        if (isHovered) labelAtoms.push(item.atom1);
        else if (this.labels && !HYDROGENS.has(item.atom1.type)) labelAtoms.push(item.atom1);
      }
    }

    for (const atom of labelAtoms) this._drawLabel(ctx, atom, atom.name === this.hoveredAtom);

    if (this.hoveredAtom === null && this.hoveredBond !== null && this.hoveredBondDistance != null && this.hoverCursor) {
      this._drawHoverDistanceLabel(ctx, `${this.hoveredBondDistance.toFixed(3)} \u00c5`, this.hoverCursor.x, this.hoverCursor.y, width, height);
    }

    if (this.isPacked) this._drawAxisIndicator(ctx, height);
  }

  _calculateZOrder() {
    for (const item of this.objects) {
      item.zOrder = item.isBond ? (item.atom1.z + item.atom2.z) / 2.0 : item.atom1.z;
    }
    this.objects.sort((a, b) => b.zOrder - a.zOrder);
  }

  // ------------------------------------------------------------------
  // Bond drawing
  // ------------------------------------------------------------------

  _drawBondRounded(ctx, at1, at2) {
    const line = this._getBondLine(at1, at2);
    if (!line) return;
    const { x1, y1, x2, y2, width } = line;
    const dx = x2 - x1, dy = y2 - y1;
    const length = Math.hypot(dx, dy);
    if (length < 0.0001) return;

    const bondKey = sortedBondKey(at1.name, at2.name);
    if (this.selectedBonds.has(bondKey)) this._drawBondSelection(ctx, x1, y1, x2, y2, width);

    const angle = Math.atan2(dy, dx);
    ctx.save();
    ctx.translate(x1, y1);
    ctx.rotate(angle);

    const dark = darker(this.bondColor, 170);
    const light = lighter(this.bondColor, 160);
    const shadow = darker(this.bondColor, 280);
    const grad = ctx.createLinearGradient(0, -width / 2, 0, width / 2);
    grad.addColorStop(0, dark);
    grad.addColorStop(0.2, light);
    grad.addColorStop(1, shadow);
    ctx.fillStyle = grad;

    ctx.beginPath();
    ctx.moveTo(0, -width / 2);
    ctx.lineTo(length, -width / 2);
    ctx.arc(length, 0, width / 2, -Math.PI / 2, Math.PI / 2, false);
    ctx.lineTo(0, width / 2);
    ctx.arc(0, 0, width / 2, Math.PI / 2, -Math.PI / 2, false);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
  }

  _drawBondSelection(ctx, x1, y1, x2, y2, width) {
    const selWidth = width + Math.max(4, Math.trunc(12 * this.zoom));
    ctx.save();
    ctx.strokeStyle = 'rgb(0,190,255)';
    ctx.lineWidth = selWidth;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();
    ctx.restore();
  }

  // ------------------------------------------------------------------
  // Atom drawing
  // ------------------------------------------------------------------

  _sphereFill(ctx, cx, cy, maxR, colorLight, color, colorDark) {
    const gx = cx - maxR * 0.3, gy = cy - maxR * 0.3;
    const grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, maxR * 1.5);
    grad.addColorStop(0.0, colorLight);
    grad.addColorStop(0.4, color);
    grad.addColorStop(1.0, colorDark);
    return grad;
  }

  _drawSelection(ctx, cx, cy, r1, r2, angleRad) {
    const padding = 4.0;
    ctx.save();
    ctx.strokeStyle = 'rgb(0,190,255)';
    ctx.lineWidth = Math.max(3, 12 * this.zoom);
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.ellipse(cx, cy, r1 + padding, r2 + padding, angleRad, 0, 2 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  _drawAtom(ctx, atom) {
    if (atom.uCart && !atom.adpValid) {
      // Non-positive-definite tensor: show the cube placeholder in both ADP
      // and isotropic mode so the broken atom is never hidden.
      this._drawInvalidAdp(ctx, atom);
      return;
    }
    const cx = atom.screenx, cy = atom.screeny;

    if (this.showAdpsFlag && atom.uCart) {
      const a = atom.uCart[0][0], b = atom.uCart[0][1], c = atom.uCart[1][1];
      const T = a + c, D = a * c - b * b, diff = T * T * 0.25 - D;
      if (diff >= 0) {
        const sq = Math.sqrt(diff);
        const eig1 = T * 0.5 - sq, eig2 = T * 0.5 + sq;
        if (eig1 > 0 && eig2 > 0) {
          const r1 = Math.sqrt(eig1) * this.scale * this.adpScale;
          const r2 = Math.sqrt(eig2) * this.scale * this.adpScale;
          const angle = Math.abs(b) > 1e-8 ? Math.atan2(eig1 - a, b) : (a < c ? 0 : Math.PI / 2);
          const maxR = Math.max(r1, r2);

          if (this.selectedAtoms.has(atom.name)) this._drawSelection(ctx, cx, cy, r1, r2, angle);

          ctx.save();
          ctx.beginPath();
          ctx.ellipse(cx, cy, r1, r2, angle, 0, 2 * Math.PI);
          ctx.fillStyle = this._sphereFill(ctx, cx, cy, maxR, atom.colorLight, atom.color, atom.colorDark);
          ctx.strokeStyle = this.adpPenColor;
          ctx.lineWidth = 1;
          ctx.fill();
          ctx.stroke();

          this._drawPrincipalArcs(ctx, atom, cx, cy, r1, r2, angle);
          ctx.restore();
          return;
        }
      }
    }

    let circleSize = this.atomsSize;
    if (this.showAdpsFlag && atom.uIso != null) circleSize = Math.sqrt(atom.uIso) * this.scale * this.adpScale * 2;
    const radius = circleSize / 2;
    if (this.selectedAtoms.has(atom.name)) this._drawSelection(ctx, cx, cy, radius, radius, 0);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, radius, 0, 2 * Math.PI);
    ctx.fillStyle = this._sphereFill(ctx, cx, cy, radius, atom.colorLight, atom.color, atom.colorDark);
    ctx.strokeStyle = this.fallbackPenColor;
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
    ctx.restore();
  }

  /**
   * Non-positive-definite ADP fallback: a real 3-D cube oriented in the
   * molecular frame and projected through the current view rotation, so it
   * turns with the structure. All six faces are painted back-to-front.
   */
  _drawInvalidAdp(ctx, atom) {
    const cx = atom.screenx, cy = atom.screeny;
    const half = this.atomsSize * NPD_CUBE_HALF_FACTOR;
    if (this.selectedAtoms.has(atom.name)) {
      const bound = this.atomsSize * NPD_CUBE_BOUND_FACTOR;
      this._drawSelection(ctx, cx, cy, bound, bound, 0);
    }

    ctx.save();
    ctx.strokeStyle = this.fallbackPenColor;
    ctx.lineWidth = 1;
    // Match Qt's QPen defaults (BevelJoin, miter limit 2).  The canvas
    // default is 'miter' with a limit of 10, which grows a long spike out of
    // the acute corners of a face that is projected nearly edge-on.
    ctx.lineJoin = 'bevel';
    ctx.miterLimit = 2;
    for (const face of npdCubeFaces(this.cumulativeR, half)) {
      const k = npdFaceShade(face.normal);
      ctx.fillStyle = k >= 1 ? lighter(atom.color, k * 100) : darker(atom.color, 100 / k);
      ctx.beginPath();
      ctx.moveTo(cx + face.corners[0][0], cy + face.corners[0][1]);
      for (let i = 1; i < face.corners.length; i++) {
        ctx.lineTo(cx + face.corners[i][0], cy + face.corners[i][1]);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  _drawLabel(ctx, atom, enlarged) {
    ctx.save();
    ctx.fillStyle = 'rgb(100,50,5)';
    const rPix = this.getSphericalRadius(atom) * this.scale;
    if (enlarged) {
      ctx.font = `bold ${Math.max(1, Math.round((this.fontsize + 4) * this.zoom * 2))}px sans-serif`;
    }
    ctx.fillText(atom.name, atom.screenx + rPix + 2, atom.screeny - rPix - 2);
    ctx.restore();
  }

  _drawPrincipalArcs(ctx, atom, cx, cy, r1, r2, angle) {
    const eigvals = atom.uEigvals;
    // Qt uses a *cosmetic* pen here, i.e. a width measured in device pixels
    // and unaffected by any transform. The canvas draws in logical pixels
    // scaled by `_renderScale`, so divide to land on the same thickness.
    const lineWidth = this.cachedAdpLineWidth / (this._renderScale || 1);
    if (!eigvals || eigvals[0] <= 0 || eigvals[1] <= 0 || eigvals[2] <= 0) {
      // Fallback cross spanning the ellipse along its principal axes, matching
      // `molecule_painter._draw_principal_arcs` (which draws it under the
      // painter's `rotate(angle)`).
      const ca = Math.cos(angle), sa = Math.sin(angle);
      ctx.save();
      ctx.strokeStyle = `rgba(0,0,0,${120 / 255})`;
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(cx - r1 * ca, cy - r1 * sa);
      ctx.lineTo(cx + r1 * ca, cy + r1 * sa);
      ctx.moveTo(cx + r2 * sa, cy - r2 * ca);
      ctx.lineTo(cx - r2 * sa, cy + r2 * ca);
      ctx.stroke();
      ctx.restore();
      return;
    }
    const eigenvectors = atom.uEigvecs;
    const c = this.adpScale, s = this.scale;
    ctx.save();
    ctx.strokeStyle = `rgba(0,0,0,${120 / 255})`;
    ctx.lineWidth = lineWidth;

    const pairs = [[1, 2], [0, 2], [0, 1]];
    for (const [i, j] of pairs) {
      const li = eigvals[i], lj = eigvals[j];
      if (li <= 0 || lj <= 0) continue;
      const ri3d = c * Math.sqrt(li), rj3d = c * Math.sqrt(lj);
      const vi = [eigenvectors[0][i], eigenvectors[1][i], eigenvectors[2][i]];
      const vj = [eigenvectors[0][j], eigenvectors[1][j], eigenvectors[2][j]];

      // Qt applies a rotation-compensated brush transform here; the
      // compensation cancels out algebraically, leaving a transform built
      // directly from the raw (world-frame) eigenvector x/y components.
      const Ax = s * ri3d * vi[0], Ay = s * ri3d * vi[1];
      const Bx = s * rj3d * vj[0], By = s * rj3d * vj[1];

      // This cross-section curve lies ON the ellipsoid surface, so the
      // visible/hidden split is the silhouette (front-facing surface),
      // determined by the surface NORMAL — not by the depth of the curve
      // point. For a point P(t) = ri3d*cos(t)*vi + rj3d*sin(t)*vj on the
      // surface, the outward normal is proportional to
      //   (cos(t)/ri3d)*vi + (sin(t)/rj3d)*vj,
      // hence the z-amplitude below uses division by the radius, not
      // multiplication. For a spherical ADP (ri3d == rj3d) the two agree,
      // but for elongated ellipsoids the depth-based split is wrong.
      const AzN = vi[2] / ri3d, BzN = vj[2] / rj3d;
      const zAmp = Math.hypot(AzN, BzN);

      // Sample the arc directly in screen space (rather than stroking a
      // unit circle through a ctx.transform) so the stroke width stays
      // exactly `cachedAdpLineWidth` device pixels (see `lineWidth` above),
      // mirroring Qt's cosmetic pen. Relying on ctx.transform + a
      // compensating 1/det line-width scale breaks down when the ellipsoid's
      // principal plane is viewed near edge-on: the transform becomes nearly
      // singular, 1/det blows up, and the "circle" degenerates into a huge
      // filled band across the canvas instead of a thin arc.
      //
      // The visible (front-facing) half of this on-surface cross-section is
      // where the surface normal's depth component n_z(t) = AzN*cos(t) +
      // BzN*sin(t) is <= 0 (normal pointing toward the viewer; smaller z =
      // closer, matching the z-order painter's-algorithm sort below).
      // n_z(t) = zAmp*cos(t - phiN), which is <= 0 exactly for t in
      // [phiN + pi/2, phiN + 3*pi/2] — a plain half turn starting at
      // phiN + pi/2. (Qt's `drawArc` needs the negated/offset angle here
      // because its angle parameter is measured the opposite way from the
      // cos(t)/sin(t) sampling used below; since we sample directly there
      // is no such flip to compensate for.)
      let startAngle = 0;
      let sweep = 2 * Math.PI;
      if (zAmp >= 1e-8) {
        const phiN = Math.atan2(BzN, AzN);
        startAngle = phiN + Math.PI / 2;
        sweep = Math.PI;
      }
      const steps = 48;
      ctx.save();
      ctx.beginPath();
      for (let k = 0; k <= steps; k++) {
        const t = startAngle + (sweep * k) / steps;
        const lx = Math.cos(t), ly = Math.sin(t);
        const px = cx + Ax * lx + Bx * ly;
        const py = cy + Ay * lx + By * ly;
        if (k === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.lineWidth = lineWidth;
      ctx.stroke();
      ctx.restore();
    }
    ctx.restore();
  }

  _drawHoverDistanceLabel(ctx, text, cx, cy, width, height) {
    ctx.save();
    const px = Math.max(1, Math.round(this.fontsize * this.zoom * 2));
    ctx.font = `bold ${px}px sans-serif`;
    const metrics = ctx.measureText(text);
    const padX = 6, padY = 3;
    const tw = metrics.width;
    const th = px * 1.2;
    const boxW = tw + 2 * padX, boxH = th + 2 * padY;
    let x = cx + 14, y = cy + 14;
    if (x + boxW > width) x = cx - 14 - boxW;
    if (y + boxH > height) y = cy - 14 - boxH;

    ctx.fillStyle = 'rgba(143,230,193,0.86)';
    ctx.strokeStyle = 'rgba(60,60,60,0.86)';
    ctx.lineWidth = 1;
    const r = 5;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + boxW, y, x + boxW, y + boxH, r);
    ctx.arcTo(x + boxW, y + boxH, x, y + boxH, r);
    ctx.arcTo(x, y + boxH, x, y, r);
    ctx.arcTo(x, y, x + boxW, y, r);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = 'rgb(20,20,20)';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(text, x + boxW / 2, y + boxH / 2);
    ctx.restore();
  }

  _drawAxisIndicator(ctx, height) {
    if (!this.amatrix || !this.cell) return;
    let axes = [0, 1, 2].map((i) => normalize([this.amatrix[0][i], this.amatrix[1][i], this.amatrix[2][i]]));
    axes = axes.map((v) => matVec(this.cumulativeR, v));
    const arrowLen = 40.0;
    const originX = 55.0, originY = height - 55.0;
    const colors = ['rgb(220,30,30)', 'rgb(30,160,30)', 'rgb(30,30,220)'];
    const labels = ['a', 'b', 'c'];

    ctx.save();
    ctx.font = 'bold 12px sans-serif';
    for (let i = 0; i < 3; i++) {
      const [vx, vy] = axes[i];
      const tipX = originX + vx * arrowLen, tipY = originY + vy * arrowLen;
      ctx.strokeStyle = colors[i];
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(originX, originY);
      ctx.lineTo(tipX, tipY);
      ctx.stroke();

      const ddx = tipX - originX, ddy = tipY - originY;
      const length = Math.hypot(ddx, ddy);
      if (length > 1e-6) {
        const ux = ddx / length, uy = ddy / length;
        const px2 = -uy, py2 = ux;
        const headLen = 8.0, headW = 3.5;
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - ux * headLen + px2 * headW, tipY - uy * headLen + py2 * headW);
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - ux * headLen - px2 * headW, tipY - uy * headLen - py2 * headW);
        ctx.stroke();
      }

      ctx.fillStyle = colors[i];
      ctx.textAlign = 'left';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(labels[i], tipX + 4 * (vx >= 0 ? 1 : -2), tipY + 4 * (vy >= 0 ? -1 : 2));
    }
    ctx.restore();
  }

  // ------------------------------------------------------------------
  // Image export
  // ------------------------------------------------------------------

  /** Render at `scale`x resolution and return a data URL (default PNG). */
  toDataURL(type = 'image/png', scale = 1.5) {
    const off = document.createElement('canvas');
    off.width = Math.max(1, Math.round(this._cssWidth * scale));
    off.height = Math.max(1, Math.round(this._cssHeight * scale));
    const octx = off.getContext('2d');
    this.render(octx, this._cssWidth, this._cssHeight, scale);
    return off.toDataURL(type);
  }

  /** Render at `scale`x resolution and trigger a browser download. */
  saveImage(filename = 'molecule.png', scale = 1.5) {
    const url = this.toDataURL('image/png', scale);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
  }
}

export { Atom };
