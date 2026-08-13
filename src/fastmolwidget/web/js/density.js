/**
 * Residual (Fo-Fc) electron-density wireframes in the browser.
 *
 * The map itself is **computed in Python** (`fastmolwidget.density`) and shipped
 * inside the page by `fastmolwidget.web_export.export_density()`: one unit cell,
 * quantised to a byte per grid point, zeroed outside the region that can ever be
 * displayed, gzipped and base64-encoded. Everything downstream — decoding,
 * cutting the periodic sub-grid out around the atoms, contouring it and clipping
 * the result — happens here, so the contour level stays adjustable and growing
 * or packing in the browser still gets density around the new atoms.
 *
 * `marchingCubes()` is a port of `density_cpp.cpp`, so the wireframe matches the
 * Qt renderers edge for edge (up to the quantisation of the shipped map).
 *
 * @see js/README.md for the JSON contract.
 */

import { CORNER_OFFSETS, EDGE_CORNERS, EDGE_COUNT, lookupTables } from './mc_tables.js';

const EPSILON = 1e-12;

/** Canonical grid-edge identity, so neighbouring cubes share vertices. */
function canonicalEdgeKey(x, y, z, edgeId) {
  switch (edgeId) {
    case 0: return [x, y, z, 0];
    case 1: return [x + 1, y, z, 1];
    case 2: return [x, y + 1, z, 0];
    case 3: return [x, y, z, 1];
    case 4: return [x, y, z + 1, 0];
    case 5: return [x + 1, y, z + 1, 1];
    case 6: return [x, y + 1, z + 1, 0];
    case 7: return [x, y, z + 1, 1];
    case 8: return [x, y, z, 2];
    case 9: return [x + 1, y, z, 2];
    case 10: return [x + 1, y + 1, z, 2];
    default: return [x, y + 1, z, 2];
  }
}

function interpolateMu(level, valueA, valueB) {
  if (Math.abs(level - valueA) < EPSILON) return 0.0;
  if (Math.abs(level - valueB) < EPSILON) return 1.0;
  if (Math.abs(valueA - valueB) < EPSILON) return 0.0;
  const mu = (level - valueA) / (valueB - valueA);
  return Math.min(Math.max(mu, 0.0), 1.0);
}

/**
 * Largest absolute corner value of every cube in the grid.
 *
 * Contouring the same block at many levels (dragging the level control) spends
 * almost all its time deciding that a cube is not crossed at all. A cube whose
 * eight corners are all smaller in magnitude than |level| cannot be crossed by
 * either lobe, so one comparison against this table replaces eight grid reads
 * and the case-index arithmetic. It pays for itself many times over on a map
 * that has been masked to zero away from the atoms, where most cubes are flat
 * zero.
 *
 * @returns {Float32Array} `(nx-1) * (ny-1) * (nz-1)`, C-ordered.
 */
export function cellAbsMax(values, size) {
  const [nx, ny, nz] = size;
  if (nx < 2 || ny < 2 || nz < 2) return new Float32Array(0);
  const cx = nx - 1, cy = ny - 1, cz = nz - 1;
  const out = new Float32Array(cx * cy * cz);
  let at = 0;
  for (let x = 0; x < cx; x++) {
    for (let y = 0; y < cy; y++) {
      const base = (x * ny + y) * nz;
      const bx = ((x + 1) * ny + y) * nz;
      const by = (x * ny + y + 1) * nz;
      const bxy = ((x + 1) * ny + y + 1) * nz;
      for (let z = 0; z < cz; z++) {
        let m = Math.abs(values[base + z]);
        let v = Math.abs(values[base + z + 1]); if (v > m) m = v;
        v = Math.abs(values[bx + z]); if (v > m) m = v;
        v = Math.abs(values[bx + z + 1]); if (v > m) m = v;
        v = Math.abs(values[by + z]); if (v > m) m = v;
        v = Math.abs(values[by + z + 1]); if (v > m) m = v;
        v = Math.abs(values[bxy + z]); if (v > m) m = v;
        v = Math.abs(values[bxy + z + 1]); if (v > m) m = v;
        out[at++] = m;
      }
    }
  }
  return out;
}

/**
 * Extract a wireframe isosurface from a regular 3-D scalar grid.
 *
 * Port of `density_cpp.marching_cubes`: classic Lorensen-Cline marching cubes
 * whose triangles are emitted as deduplicated undirected line segments. Vertices
 * are keyed by the identity of the grid edge they sit on, so adjacent cubes
 * re-use them and the cage has no doubled lines.
 *
 * @param {Float32Array} values C-ordered `(nx, ny, nz)` grid.
 * @param {number[]} size `[nx, ny, nz]`.
 * @param {number} level Contour level.
 * @param {number[]} origin Coordinates of `values[0,0,0]`.
 * @param {number[]} step Spacing along the three grid axes.
 * @param {Float32Array} [absMax] optional `cellAbsMax()` table used to skip
 *   cubes the surface cannot cross. Purely an optimisation — the output is
 *   identical with or without it.
 * @returns {{vertices: Float64Array, segments: Int32Array}} `(M, 3)` vertices
 *   and `(K, 2)` segment indices, flattened.
 */
export function marchingCubes(values, size, level, origin, step, absMax) {
  const [nx, ny, nz] = size;
  if (nx < 2 || ny < 2 || nz < 2) {
    return { vertices: new Float64Array(0), segments: new Int32Array(0) };
  }

  const { edgeMasks, triTable, triWidth } = lookupTables();
  const vertexLookup = new Map();
  const vertices = [];
  const edgeLookup = new Set();
  const segments = [];
  const magnitude = Math.abs(level);
  const skipStride = (ny - 1) * (nz - 1);

  const at = (x, y, z) => values[(x * ny + y) * nz + z];

  const addWireEdge = (a, b) => {
    if (a === b) return;
    const key = a < b ? a * 4294967296 + b : b * 4294967296 + a;
    if (edgeLookup.has(key)) return;
    edgeLookup.add(key);
    segments.push(a, b);
  };

  const getOrCreateVertex = (cx, cy, cz, edgeId) => {
    const k = canonicalEdgeKey(cx, cy, cz, edgeId);
    const key = ((k[0] * (ny + 2) + k[1]) * (nz + 2) + k[2]) * 3 + k[3];
    const known = vertexLookup.get(key);
    if (known !== undefined) return known;

    const cornerA = EDGE_CORNERS[edgeId][0];
    const cornerB = EDGE_CORNERS[edgeId][1];
    const x0 = cx + CORNER_OFFSETS[cornerA][0];
    const y0 = cy + CORNER_OFFSETS[cornerA][1];
    const z0 = cz + CORNER_OFFSETS[cornerA][2];
    const x1 = cx + CORNER_OFFSETS[cornerB][0];
    const y1 = cy + CORNER_OFFSETS[cornerB][1];
    const z1 = cz + CORNER_OFFSETS[cornerB][2];

    const mu = interpolateMu(level, at(x0, y0, z0), at(x1, y1, z1));
    const index = vertices.length / 3;
    vertices.push(
      origin[0] + (x0 + mu * (x1 - x0)) * step[0],
      origin[1] + (y0 + mu * (y1 - y0)) * step[1],
      origin[2] + (z0 + mu * (z1 - z0)) * step[2]
    );
    vertexLookup.set(key, index);
    return index;
  };

  const local = new Int32Array(EDGE_COUNT);

  for (let cx = 0; cx < nx - 1; cx++) {
    for (let cy = 0; cy < ny - 1; cy++) {
      const skipRow = absMax ? cx * skipStride + cy * (nz - 1) : 0;
      for (let cz = 0; cz < nz - 1; cz++) {
        if (absMax && absMax[skipRow + cz] < magnitude) continue;
        let caseIndex = 0;
        for (let corner = 0; corner < 8; corner++) {
          const value = at(cx + CORNER_OFFSETS[corner][0],
                           cy + CORNER_OFFSETS[corner][1],
                           cz + CORNER_OFFSETS[corner][2]);
          if (value < level) caseIndex |= 1 << corner;
        }

        const edgeMask = edgeMasks[caseIndex];
        if (edgeMask === 0) continue;

        local.fill(-1);
        for (let edge = 0; edge < EDGE_COUNT; edge++) {
          if ((edgeMask & (1 << edge)) === 0) continue;
          local[edge] = getOrCreateVertex(cx, cy, cz, edge);
        }

        const row = caseIndex * triWidth;
        for (let p = 0; p + 2 < triWidth && triTable[row + p] !== -1; p += 3) {
          const v0 = local[triTable[row + p]];
          const v1 = local[triTable[row + p + 1]];
          const v2 = local[triTable[row + p + 2]];
          if (v0 < 0 || v1 < 0 || v2 < 0) continue;
          if (v0 === v1 || v1 === v2 || v0 === v2) continue;
          addWireEdge(v0, v1);
          addWireEdge(v1, v2);
          addWireEdge(v2, v0);
        }
      }
    }
  }

  return { vertices: new Float64Array(vertices), segments: new Int32Array(segments) };
}

/**
 * Drop every segment further than *margin* from any atom.
 *
 * Port of `density._clip_to_atoms`: the bounding box cut out of the map is
 * necessarily larger than the molecule, so this removes the blobs sitting in
 * the corners of the box. A segment survives only when **both** its endpoints
 * are close enough, which stops lines dangling into empty space.
 *
 * Distances go through a uniform spatial hash of cell size *margin*, so only
 * the 27 neighbouring buckets of a vertex are ever examined.
 *
 * @returns {{vertices: Float64Array, segments: Int32Array}} renumbered.
 */
export function clipToAtoms(vertices, segments, atoms, margin) {
  const vertexCount = vertices.length / 3;
  const segmentCount = segments.length / 2;
  if (vertexCount === 0 || segmentCount === 0 || margin <= 0) {
    return { vertices: new Float64Array(0), segments: new Int32Array(0) };
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < atoms.length; i += 3) {
    if (atoms[i] < minX) minX = atoms[i];
    if (atoms[i + 1] < minY) minY = atoms[i + 1];
    if (atoms[i + 2] < minZ) minZ = atoms[i + 2];
    if (atoms[i] > maxX) maxX = atoms[i];
    if (atoms[i + 1] > maxY) maxY = atoms[i + 1];
    if (atoms[i + 2] > maxZ) maxZ = atoms[i + 2];
  }
  const originX = minX - margin, originY = minY - margin, originZ = minZ - margin;
  // Grid wide enough that any vertex worth keeping falls inside it; anything
  // outside is trivially too far from every atom.
  const nj = Math.floor((maxY - originY) / margin) + 3;
  const nk = Math.floor((maxZ - originZ) / margin) + 3;
  const bucketOf = (i, j, k) => (i * nj + j) * nk + k;

  const buckets = new Map();
  for (let a = 0; a < atoms.length; a += 3) {
    const i = Math.floor((atoms[a] - originX) / margin);
    const j = Math.floor((atoms[a + 1] - originY) / margin);
    const k = Math.floor((atoms[a + 2] - originZ) / margin);
    const key = bucketOf(i, j, k);
    const list = buckets.get(key);
    if (list) list.push(a);
    else buckets.set(key, [a]);
  }

  const limit = margin * margin;
  const keep = new Uint8Array(vertexCount);
  for (let v = 0; v < vertexCount; v++) {
    const px = vertices[3 * v], py = vertices[3 * v + 1], pz = vertices[3 * v + 2];
    const ci = Math.floor((px - originX) / margin);
    const cj = Math.floor((py - originY) / margin);
    const ck = Math.floor((pz - originZ) / margin);
    let near = false;
    for (let di = -1; di <= 1 && !near; di++) {
      for (let dj = -1; dj <= 1 && !near; dj++) {
        for (let dk = -1; dk <= 1 && !near; dk++) {
          const list = buckets.get(bucketOf(ci + di, cj + dj, ck + dk));
          if (list === undefined) continue;
          for (let n = 0; n < list.length; n++) {
            const a = list[n];
            const dx = atoms[a] - px, dy = atoms[a + 1] - py, dz = atoms[a + 2] - pz;
            if (dx * dx + dy * dy + dz * dz <= limit) { near = true; break; }
          }
        }
      }
    }
    keep[v] = near ? 1 : 0;
  }

  const renumber = new Int32Array(vertexCount).fill(-1);
  const keptVertices = [];
  const keptSegments = [];
  for (let s = 0; s < segmentCount; s++) {
    const a = segments[2 * s], b = segments[2 * s + 1];
    if (keep[a] === 0 || keep[b] === 0) continue;
    if (renumber[a] < 0) {
      renumber[a] = keptVertices.length / 3;
      keptVertices.push(vertices[3 * a], vertices[3 * a + 1], vertices[3 * a + 2]);
    }
    if (renumber[b] < 0) {
      renumber[b] = keptVertices.length / 3;
      keptVertices.push(vertices[3 * b], vertices[3 * b + 1], vertices[3 * b + 2]);
    }
    keptSegments.push(renumber[a], renumber[b]);
  }
  return {
    vertices: new Float64Array(keptVertices),
    segments: new Int32Array(keptSegments),
  };
}

function modulo(a, b) {
  const r = a % b;
  return r >= 0 ? r : r + b;
}

function invert3(m) {
  const [a, b, c] = m[0], [d, e, f] = m[1], [g, h, i] = m[2];
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-12) return null;
  return [
    [(e * i - f * h) / det, (c * h - b * i) / det, (b * f - c * e) / det],
    [(f * g - d * i) / det, (a * i - c * g) / det, (c * d - a * f) / det],
    [(d * h - e * g) / det, (b * g - a * h) / det, (a * e - b * d) / det],
  ];
}

/** Standard fractional-to-Cartesian matrix, matching `density._orthogonalisation_matrix`. */
function orthMatrix(cell) {
  const [a, b, c, alpha, beta, gamma] = cell;
  const rad = Math.PI / 180;
  const ca = Math.cos(alpha * rad), cb = Math.cos(beta * rad), cg = Math.cos(gamma * rad);
  const sg = Math.sin(gamma * rad);
  const volume = Math.sqrt(Math.max(1 - ca * ca - cb * cb - cg * cg + 2 * ca * cb * cg, 1e-12));
  return [
    [a, b * cg, c * cb],
    [0, b * sg, c * (ca - cb * cg) / sg],
    [0, 0, c * volume / sg],
  ];
}

function base64ToBytes(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function gunzip(bytes) {
  if (typeof DecompressionStream === 'undefined') {
    throw new Error(
      'This browser cannot inflate the density map (no DecompressionStream). ' +
      'Re-export it with compress=False.'
    );
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/**
 * A residual-density map covering one unit cell, as shipped to the browser.
 *
 * The map is periodic, so it can be sampled outside `[0, 1)` in fractional
 * coordinates by wrapping the grid indices — which is what `region()` does to
 * cover grown or packed molecules.
 */
export class DensityMap {
  constructor({ values, size, cell, rms, level, max, min, margin }) {
    /** @type {Float32Array} `(nx, ny, nz)` grid of rho in e/A^3. */
    this.values = values;
    this.size = size;
    this.cell = cell;
    this.rms = rms;
    this.max = max;
    this.min = min;
    /** Level the exporter suggests, in e/A^3. */
    this.level = level;
    /** Radius in A the shipped map was masked to. */
    this.margin = margin;
    this.orth = orthMatrix(cell);
    this.inverse = invert3(this.orth);
  }

  /**
   * Decode the `density` object of the JSON contract.
   *
   * @param {object} payload as produced by `web_export.export_density()`.
   * @returns {Promise<DensityMap>}
   */
  static async fromPayload(payload) {
    let bytes = base64ToBytes(payload.data);
    if (payload.encoding === 'gzip+base64') bytes = await gunzip(bytes);
    const quantised = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const values = new Float32Array(quantised.length);
    const scale = payload.scale;
    for (let i = 0; i < quantised.length; i++) values[i] = quantised[i] * scale;
    return new DensityMap({
      values,
      size: payload.size,
      cell: payload.cell,
      rms: payload.rms,
      level: payload.level,
      max: payload.max,
      min: payload.min,
      margin: payload.margin,
    });
  }

  /** Contour level at *sigma* times the map RMS, rounded like the Qt viewers. */
  sigmaLevel(sigma = 3.0) {
    return Math.max(Math.round(sigma * this.rms * 100) / 100, 0.01);
  }

  /**
   * Cut the periodic map down to the bounding box around *atoms*.
   *
   * Port of `ResidualDensityMap._region`; only a cheap box pre-filter, with
   * `clipToAtoms()` doing the per-atom trimming afterwards.
   *
   * @param {Float64Array} atoms flattened `(N, 3)` Cartesian positions.
   * @param {number} margin padding in A.
   */
  region(atoms, margin) {
    const [nx, ny, nz] = this.size;
    const dims = [nx, ny, nz];
    const step = [1 / nx, 1 / ny, 1 / nz];
    const inv = this.inverse;

    // Fractional padding: the row norms of the inverse keep this correct for
    // oblique cells instead of under-padding along the skewed axes.
    const pad = inv.map((row) => margin * Math.hypot(row[0], row[1], row[2]));
    const lo = [Infinity, Infinity, Infinity];
    const hi = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < atoms.length; i += 3) {
      for (let axis = 0; axis < 3; axis++) {
        const f = inv[axis][0] * atoms[i] + inv[axis][1] * atoms[i + 1] + inv[axis][2] * atoms[i + 2];
        if (f < lo[axis]) lo[axis] = f;
        if (f > hi[axis]) hi[axis] = f;
      }
    }

    const start = [0, 0, 0];
    const shape = [0, 0, 0];
    for (let axis = 0; axis < 3; axis++) {
      start[axis] = Math.floor((lo[axis] - pad[axis]) * dims[axis]);
      const end = Math.ceil((hi[axis] + pad[axis]) * dims[axis]) + 1;
      shape[axis] = Math.max(end - start[axis], 0);
    }

    const sub = new Float32Array(shape[0] * shape[1] * shape[2]);
    let out = 0;
    for (let i = 0; i < shape[0]; i++) {
      const gi = modulo(start[0] + i, nx);
      for (let j = 0; j < shape[1]; j++) {
        const gj = modulo(start[1] + j, ny);
        const base = (gi * ny + gj) * nz;
        for (let k = 0; k < shape[2]; k++) {
          sub[out++] = this.values[base + modulo(start[2] + k, nz)];
        }
      }
    }
    return {
      values: sub,
      size: shape,
      origin: [start[0] * step[0], start[1] * step[1], start[2] * step[2]],
      step,
    };
  }

  /**
   * The extracted block for *atoms*, with its cube-magnitude table, memoised.
   *
   * Changing only the contour level re-uses everything: cutting the block out
   * of the periodic map and building the skip table are exactly the parts that
   * do not depend on the level. The cache is keyed on the *identity* of the
   * atom array, which the renderer keeps stable until the visible atoms
   * actually change.
   */
  _blockFor(atoms, margin) {
    const cache = this._blockCache;
    if (cache && cache.atoms === atoms && cache.margin === margin) return cache.block;
    const block = this.region(atoms, margin);
    block.absMax = cellAbsMax(block.values, block.size);
    this._blockCache = { atoms, margin, block };
    return block;
  }

  /**
   * Extract a wireframe isosurface at *level*, in Cartesian coordinates.
   *
   * Marching cubes runs in fractional-coordinate space and the vertices are
   * transformed to Cartesian afterwards, so oblique cells come out right.
   *
   * @param {number} level contour level in e/A^3; negative for the negative lobe.
   * @param {Float64Array} atoms flattened Cartesian positions to clip against.
   * @param {number} margin radius around each atom to keep, in A.
   * @returns {{vertices: Float64Array, segments: Int32Array}} Cartesian.
   */
  isosurface(level, atoms, margin) {
    if (!atoms || atoms.length === 0) {
      return { vertices: new Float64Array(0), segments: new Int32Array(0) };
    }
    const block = this._blockFor(atoms, margin);
    const mesh = marchingCubes(block.values, block.size, level,
                               block.origin, block.step, block.absMax);

    const orth = this.orth;
    const cartesian = new Float64Array(mesh.vertices.length);
    for (let i = 0; i < mesh.vertices.length; i += 3) {
      const x = mesh.vertices[i], y = mesh.vertices[i + 1], z = mesh.vertices[i + 2];
      cartesian[i] = orth[0][0] * x + orth[0][1] * y + orth[0][2] * z;
      cartesian[i + 1] = orth[1][0] * x + orth[1][1] * y + orth[1][2] * z;
      cartesian[i + 2] = orth[2][0] * x + orth[2][1] * y + orth[2][2] * z;
    }
    return clipToAtoms(cartesian, mesh.segments, atoms, margin);
  }
}
