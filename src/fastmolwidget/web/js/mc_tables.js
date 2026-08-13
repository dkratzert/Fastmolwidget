/**
 * Marching-cubes lookup tables.
 *
 * A direct port of `build_lookup_tables()` in
 * `fastmolwidget/density_cpp/density_cpp.cpp`, so the wireframe the browser
 * draws has exactly the same topology as the one the Qt renderers draw.
 *
 * The 256 cases are *derived* from the cube topology rather than pasted in as
 * literal data: it is about the same amount of code, it cannot silently
 * disagree with the C++ version, and it keeps the package free of
 * third-party table data. Building them takes well under a millisecond and
 * happens once, lazily, on the first contour.
 */

/** Corner offsets of the unit cube, in the canonical marching-cubes order. */
const CORNER_OFFSETS = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];

/** The two corners spanned by each of the 12 cube edges. */
const EDGE_CORNERS = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

/** Corners of each of the 6 cube faces, walked in order around the face. */
const FACE_CORNERS = [
  [0, 1, 2, 3],  // z = 0
  [4, 5, 6, 7],  // z = 1
  [0, 1, 5, 4],  // y = 0
  [1, 2, 6, 5],  // x = 1
  [2, 3, 7, 6],  // y = 1
  [3, 0, 4, 7],  // x = 0
];

/** Edges of each face, in the same order as `FACE_CORNERS`. */
const FACE_EDGES = [
  [0, 1, 2, 3],
  [4, 5, 6, 7],
  [0, 9, 4, 8],
  [1, 10, 5, 9],
  [2, 11, 6, 10],
  [3, 8, 7, 11],
];

const EDGE_COUNT = 12;
const TRI_TABLE_WIDTH = 16;

let cachedTables = null;

function connect(adjacency, a, b) {
  if (a === b) return;
  adjacency[a][b] = true;
  adjacency[b][a] = true;
}

/**
 * Walk the face-crossing graph of one case and emit its triangle fans.
 *
 * Each face cut by the isosurface contributes one (or, for an ambiguous
 * 4-cut face, two) connections between the edges it cuts. Following those
 * connections traces the closed loops the surface makes through the cube;
 * every loop is then fanned into triangles.
 */
function buildCase(caseIndex) {
  let edgeMask = 0;
  for (let edge = 0; edge < EDGE_COUNT; edge++) {
    const insideA = ((caseIndex >> EDGE_CORNERS[edge][0]) & 1) !== 0;
    const insideB = ((caseIndex >> EDGE_CORNERS[edge][1]) & 1) !== 0;
    if (insideA !== insideB) edgeMask |= 1 << edge;
  }

  const triangles = new Int32Array(TRI_TABLE_WIDTH).fill(-1);
  if (edgeMask === 0) return { edgeMask, triangles };

  const adjacency = [];
  for (let i = 0; i < EDGE_COUNT; i++) adjacency.push(new Array(EDGE_COUNT).fill(false));

  for (let face = 0; face < FACE_CORNERS.length; face++) {
    const inside = [];
    const cut = [];
    let cutCount = 0;
    for (let i = 0; i < 4; i++) {
      const corner = FACE_CORNERS[face][i];
      const next = FACE_CORNERS[face][(i + 1) % 4];
      inside.push(((caseIndex >> corner) & 1) !== 0);
      cut.push(((caseIndex >> corner) & 1) !== ((caseIndex >> next) & 1));
      if (cut[i]) cutCount++;
    }
    if (cutCount === 2) {
      const [first, second] = [0, 1, 2, 3].filter((i) => cut[i]);
      connect(adjacency, FACE_EDGES[face][first], FACE_EDGES[face][second]);
    } else if (cutCount === 4) {
      // Ambiguous face: the two resolutions differ, pick the one that keeps
      // the "inside" corners connected (same choice as the C++ version).
      if (inside[0]) {
        connect(adjacency, FACE_EDGES[face][3], FACE_EDGES[face][0]);
        connect(adjacency, FACE_EDGES[face][1], FACE_EDGES[face][2]);
      } else {
        connect(adjacency, FACE_EDGES[face][0], FACE_EDGES[face][1]);
        connect(adjacency, FACE_EDGES[face][2], FACE_EDGES[face][3]);
      }
    }
  }

  const used = [];
  for (let i = 0; i < EDGE_COUNT; i++) used.push(new Array(EDGE_COUNT).fill(false));
  let triPos = 0;

  for (let start = 0; start < EDGE_COUNT; start++) {
    for (let neighbour = 0; neighbour < EDGE_COUNT; neighbour++) {
      if (!adjacency[start][neighbour] || used[start][neighbour]) continue;

      let loop = [start];
      let prev = -1;
      let curr = start;
      let next = neighbour;

      for (;;) {
        used[curr][next] = true;
        used[next][curr] = true;
        prev = curr;
        curr = next;
        if (curr === start) break;
        loop.push(curr);

        let candidate = -1;
        for (let e = 0; e < EDGE_COUNT; e++) {
          if (adjacency[curr][e] && e !== prev && !used[curr][e]) { candidate = e; break; }
        }
        if (candidate < 0) {
          for (let e = 0; e < EDGE_COUNT; e++) {
            if (adjacency[curr][e] && e !== prev) { candidate = e; break; }
          }
        }
        if (candidate < 0) { loop = []; break; }
        next = candidate;
      }

      if (loop.length < 3) continue;
      for (let i = 1; i + 1 < loop.length && triPos + 2 < TRI_TABLE_WIDTH; i++) {
        triangles[triPos++] = loop[0];
        triangles[triPos++] = loop[i];
        triangles[triPos++] = loop[i + 1];
      }
    }
  }

  return { edgeMask, triangles };
}

/**
 * The 256-case marching-cubes tables, built once and memoised.
 *
 * @returns {{edgeMasks: Int32Array, triTable: Int32Array, triWidth: number}}
 *   `edgeMasks[case]` is the bitmask of intersected edges; `triTable` holds
 *   `triWidth` entries per case, `-1`-terminated, as edge-id triples.
 */
export function lookupTables() {
  if (cachedTables) return cachedTables;
  const edgeMasks = new Int32Array(256);
  const triTable = new Int32Array(256 * TRI_TABLE_WIDTH).fill(-1);
  for (let caseIndex = 0; caseIndex < 256; caseIndex++) {
    const { edgeMask, triangles } = buildCase(caseIndex);
    edgeMasks[caseIndex] = edgeMask;
    triTable.set(triangles, caseIndex * TRI_TABLE_WIDTH);
  }
  cachedTables = { edgeMasks, triTable, triWidth: TRI_TABLE_WIDTH };
  return cachedTables;
}

export { CORNER_OFFSETS, EDGE_CORNERS, EDGE_COUNT, TRI_TABLE_WIDTH };
