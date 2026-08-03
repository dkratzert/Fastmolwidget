/**
 * Minimal 3x3 linear algebra helpers used by the molecule renderer.
 * All matrices are plain arrays-of-arrays: `[[a,b,c],[d,e,f],[g,h,i]]`.
 * All vectors are plain 3-element arrays `[x, y, z]`.
 */

export function identity3() {
  return [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
}

export function matMul(a, b) {
  const r = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += a[i][k] * b[k][j];
      r[i][j] = s;
    }
  }
  return r;
}

export function matVec(a, v) {
  return [
    a[0][0] * v[0] + a[0][1] * v[1] + a[0][2] * v[2],
    a[1][0] * v[0] + a[1][1] * v[1] + a[1][2] * v[2],
    a[2][0] * v[0] + a[2][1] * v[1] + a[2][2] * v[2],
  ];
}

export function transpose(a) {
  return [
    [a[0][0], a[1][0], a[2][0]],
    [a[0][1], a[1][1], a[2][1]],
    [a[0][2], a[1][2], a[2][2]],
  ];
}

export function matSub(a, b) {
  return a.map((row, i) => row.map((v, j) => v - b[i][j]));
}

export function det3(a) {
  return (
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
  );
}

export function inv3(a) {
  const d = det3(a);
  if (Math.abs(d) < 1e-14) return null;
  const invD = 1 / d;
  const c = [
    [
      (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * invD,
      (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * invD,
      (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * invD,
    ],
    [
      (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * invD,
      (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * invD,
      (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * invD,
    ],
    [
      (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * invD,
      (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * invD,
      (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * invD,
    ],
  ];
  return c;
}

export function vecSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function vecAdd(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function vecScale(a, s) {
  return [a[0] * s, a[1] * s, a[2] * s];
}

export function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

export function norm(a) {
  return Math.sqrt(dot(a, a));
}

export function normalize(a) {
  const n = norm(a);
  if (n < 1e-12) return [0, 0, 0];
  return vecScale(a, 1 / n);
}

/**
 * Re-orthonormalize a near-rotation 3x3 matrix (rows) via Gram-Schmidt,
 * returning the nearest proper rotation matrix. Used to correct the tiny
 * floating-point drift that accumulates after composing many incremental
 * rotations (e.g. hundreds of mouse-drag rotate steps in one session) —
 * without this, a rotation matrix can very slowly stop being orthonormal,
 * eventually distorting rendered geometry (elongated/skewed shapes).
 */
export function orthonormalize3(m) {
  const r0 = normalize(m[0]);
  let r1 = vecSub(m[1], vecScale(r0, dot(r0, m[1])));
  r1 = normalize(r1);
  const r2 = cross(r0, r1);
  return [r0, r1, r2];
}

/**
 * Analytic eigen-decomposition of a symmetric 3x3 matrix, equivalent to
 * `numpy.linalg.eigh`: returns ascending eigenvalues and matching
 * orthonormal eigenvectors as matrix columns.
 * Reference: https://en.wikipedia.org/wiki/Eigenvalue_algorithm#3x3_matrices
 */
export function eigSym3(A) {
  const a00 = A[0][0], a01 = A[0][1], a02 = A[0][2];
  const a11 = A[1][1], a12 = A[1][2], a22 = A[2][2];

  const p1 = a01 * a01 + a02 * a02 + a12 * a12;
  if (p1 < 1e-18) {
    // Already diagonal.
    const vals = [a00, a11, a22];
    const order = [0, 1, 2].sort((i, j) => vals[i] - vals[j]);
    const evals = order.map((i) => vals[i]);
    const evecs = identity3();
    const cols = order.map((i) => [evecs[0][i], evecs[1][i], evecs[2][i]]);
    return {
      values: evals,
      vectors: [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
      ],
    };
  }

  const q = (a00 + a11 + a22) / 3;
  const b00 = a00 - q, b11 = a11 - q, b22 = a22 - q;
  const p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2 * p1;
  const p = Math.sqrt(p2 / 6);
  const invP = 1 / p;
  const B = [
    [b00 * invP, a01 * invP, a02 * invP],
    [a01 * invP, b11 * invP, a12 * invP],
    [a02 * invP, a12 * invP, b22 * invP],
  ];
  let r = det3(B) / 2;
  r = Math.max(-1, Math.min(1, r));
  const phi = Math.acos(r) / 3;

  const eig0 = q + 2 * p * Math.cos(phi);
  const eig2 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  const eig1 = 3 * q - eig0 - eig2; // trace is invariant

  const vals = [eig0, eig1, eig2].sort((a, b) => a - b);
  const [l0, l1, l2] = vals;

  const eigenvectorFor = (lambda) => {
    // Solve (A - lambda*I) v = 0 using the row-cross-product trick. Only
    // reliable when lambda is well separated from the other two
    // eigenvalues (i.e. (A - lambda*I) has rank 2); callers must ensure that.
    const M = [
      [a00 - lambda, a01, a02],
      [a01, a11 - lambda, a12],
      [a02, a12, a22 - lambda],
    ];
    const candidates = [
      cross(M[0], M[1]),
      cross(M[0], M[2]),
      cross(M[1], M[2]),
    ];
    let best = candidates[0];
    let bestLen = norm(best);
    for (let i = 1; i < candidates.length; i++) {
      const len = norm(candidates[i]);
      if (len > bestLen) {
        best = candidates[i];
        bestLen = len;
      }
    }
    if (bestLen < 1e-9) return null;
    return normalize(best);
  };

  // Repeated eigenvalues make the row-cross-product null-space trick above
  // ill-conditioned (or outright rank-deficient), which used to make this
  // function fall back to an arbitrary fixed world-axis vector — unrelated
  // to the matrix's actual orientation. This is common in practice: any ADP
  // that sits on crystallographic symmetry axis has two exactly equal
  // eigenvalues. Instead, pick whichever eigenvalue is best separated from
  // the *other two* (always one of the two extremes, never the middle one,
  // since its isolation is the min of the two outer gaps) and solve for
  // that eigenvector robustly, then resolve the remaining (possibly
  // degenerate) pair by reducing to a well-conditioned 2x2 symmetric
  // eigenproblem within the plane orthogonal to it — never guessing.
  const isoLow = l1 - l0;
  const isoHigh = l2 - l1;
  const isoIndex = isoLow >= isoHigh ? 0 : 2;
  let vIso = eigenvectorFor(vals[isoIndex]);
  if (vIso === null) {
    // All three eigenvalues effectively equal (isotropic): any orthonormal
    // basis is a valid eigenbasis.
    return { values: vals, vectors: identity3() };
  }

  // Helper vector: whichever world axis is least aligned with vIso, to keep
  // the Gram-Schmidt step well-conditioned.
  const absComp = [Math.abs(vIso[0]), Math.abs(vIso[1]), Math.abs(vIso[2])];
  const helperAxis = absComp[0] <= absComp[1] && absComp[0] <= absComp[2] ? 0
    : absComp[1] <= absComp[2] ? 1 : 2;
  const helper = [0, 0, 0];
  helper[helperAxis] = 1;
  const vA = normalize(vecSub(helper, vecScale(vIso, dot(helper, vIso))));
  const vB = cross(vIso, vA);

  // Project A onto the (vA, vB) plane and solve the 2x2 eigenproblem
  // analytically — this correctly splits the remaining two eigenvectors
  // even when they are close but not exactly equal, and is exact (any
  // basis works) when they are equal.
  const Av_A = matVec3(A, vA);
  const Av_B = matVec3(A, vB);
  const Aaa = dot(vA, Av_A);
  const Abb = dot(vB, Av_B);
  const Aab = dot(vA, Av_B);
  const T = Aaa + Abb;
  const D = Aaa * Abb - Aab * Aab;
  const disc = Math.sqrt(Math.max(0, (T * T) / 4 - D));
  const eLo = T / 2 - disc;
  const ph = Math.abs(Aab) > 1e-12 ? Math.atan2(eLo - Aaa, Aab) : (Aaa <= Abb ? 0 : Math.PI / 2);
  const cosPh = Math.cos(ph), sinPh = Math.sin(ph);
  const vLo = vecAdd(vecScale(vA, cosPh), vecScale(vB, sinPh));
  const vHi = vecAdd(vecScale(vA, -sinPh), vecScale(vB, cosPh));

  const [v0, v1, v2] = isoIndex === 0 ? [vIso, vLo, vHi] : [vLo, vHi, vIso];

  return {
    values: vals,
    vectors: [
      [v0[0], v1[0], v2[0]],
      [v0[1], v1[1], v2[1]],
      [v0[2], v1[2], v2[2]],
    ],
  };
}

function matVec3(A, v) {
  return [
    A[0][0] * v[0] + A[0][1] * v[1] + A[0][2] * v[2],
    A[1][0] * v[0] + A[1][1] * v[1] + A[1][2] * v[2],
    A[2][0] * v[0] + A[2][1] * v[1] + A[2][2] * v[2],
  ];
}
