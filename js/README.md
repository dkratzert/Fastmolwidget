# Fastmolwidget — JavaScript 2D renderer

A browser/Canvas port of `fastmolwidget.molecule2D` (the 2-D QPainter
renderer) plus the SDM grow / pack-unit-cell logic (`fastmolwidget.sdm`).
Structure **file parsing** (CIF via `gemmi`, SHELX via `shelxfile`) stays in
Python — see `fastmolwidget.web_export` — everything downstream (growing
molecules to completion, packing unit cells, and rendering) runs in plain,
dependency-free JavaScript (ES modules, Canvas 2D — no build step required).

## Files

| File            | Purpose |
|-----------------|---------|
| `elements.js`   | Element colours/covalent radii (port of `atoms.py`) |
| `linalg.js`     | 3×3 matrix/vector helpers + analytic symmetric eigen-decomposition |
| `color.js`      | Minimal `QColor.lighter()`/`darker()` equivalents |
| `conntable.js`  | Bond-detection (port of `tools.build_conntable`) |
| `symmetry.js`   | SHELX symmetry-operation parsing, `frac_to_cart`/`cart_to_frac` |
| `sdm.js`        | `SDM` class — grow asymmetric unit / pack unit cell (port of `sdm.py`) |
| `molecule2d.js` | `MoleculeWidget2D` — the Canvas renderer (port of `molecule2D.py` + `molecule_painter.py`) |
| `viewer.js`     | `MoleculeViewer2D` — wires `SDM` growing/packing into the renderer |
| `part_filter.js`| `createPartFilter(widget)` — checkable disorder-part dropdown (port of `part_combo.PartFilterWidget`) |
| `demo/`         | Minimal standalone demo page + sample structure |

All algorithmic ports (`eigSym3`, `buildConnTable`, `SDM.grow()`,
`SDM.packUnitCell()`) have been cross-checked against the real Python
implementations on real structures and produce identical results.

## Quick start

```bash
cd js
python3 -m http.server 8000
# open http://localhost:8000/demo/index.html
```

```js
import { MoleculeViewer2D } from './viewer.js';

const canvas = document.getElementById('canvas');
const viewer = new MoleculeViewer2D(canvas);

const data = await (await fetch('/structure.json')).json();
viewer.loadStructure(data);

viewer.setGrow(true);          // expand asymmetric unit to whole molecules
viewer.setPack(true);          // or: pack one full unit cell instead
viewer.widget.showAdps(true);
viewer.widget.setBondColor('#336699');
viewer.widget.alignBestView();
viewer.widget.saveImage('molecule.png');
```

### HiDPI / crisp lines

Size the canvas through `widget.resize(cssWidth, cssHeight)` rather than
setting `canvas.width`/`canvas.height` yourself. The widget allocates the
backing store at `window.devicePixelRatio` (all drawing still happens in
logical CSS pixels), so lines stay crisp on Retina / high-DPI displays and
line thicknesses match the Qt QPainter widget 1:1. Let CSS control the
displayed size (e.g. `width: 100%; height: 100%`) and call `resize()` with the
element's `getBoundingClientRect()` size on load and on every `resize` event:

```js
function fit() {
  const r = canvas.getBoundingClientRect();
  viewer.widget.resize(r.width, r.height);
}
window.addEventListener('resize', fit);
fit();
```

Pass `new MoleculeWidget2D(canvas, { devicePixelRatio: 1 })` to force a fixed
ratio (e.g. for deterministic tests or exports).

### Disorder-part filter

`createPartFilter(widget)` builds the same checkable "Show Parts:" dropdown as
the Qt viewers. Drop the returned element into your control bar; it wires
itself to the widget's `partsChanged` event and forwards the ticked parts via
`setVisibleParts(...)`. It stays hidden unless the structure has more than one
disorder part, and all parts are ticked by default:

```js
import { createPartFilter } from './part_filter.js';
document.getElementById('bar').append(createPartFilter(viewer.widget));
```


Or use `MoleculeWidget2D` directly if you already have Cartesian atoms and
don't need growing/packing (mirrors `MoleculeWidget.open_molecule()` in
Python):

```js
import { MoleculeWidget2D } from './molecule2d.js';

const widget = new MoleculeWidget2D(canvas);
widget.openMolecule({
  atoms: [
    { label: 'C1', type: 'C', x: 0, y: 0, z: 0 },
    { label: 'O1', type: 'O', x: 1.2, y: 0, z: 0 },
  ],
  cell: null, // or [a,b,c,alpha,beta,gamma] to enable ADPs / F1-F3 alignment
});
```

## JSON contract (Python → JS)

`fastmolwidget.web_export.export_cif(path)` / `export_shelx(path)` produce:

```jsonc
{
  "cell": [a, b, c, alpha, beta, gamma],
  "centric": false,                       // adds "-X,-Y,-Z" automatically
  "symmops": ["x, y, z", "-x, 1/2+y, -z"], // SHELX-style, identity may be omitted
  "atoms": [
    {
      "label": "C1", "type": "C",
      "x": 0.1234, "y": 0.5678, "z": 0.9012,   // FRACTIONAL coordinates
      "part": 0,
      "adp": [U11, U22, U33, U23, U13, U12]     // or null (isotropic/no ADP)
    }
  ]
}
```

This is the **asymmetric unit** in fractional coordinates — `MoleculeViewer2D`
runs `SDM` in the browser to grow/pack it before handing Cartesian atoms to
`MoleculeWidget2D.openMolecule()`. If you don't need growing/packing (e.g. a
plain XYZ file, or you've already computed Cartesian atoms in Python), skip
`web_export`/`viewer.js` and call `MoleculeWidget2D.openMolecule()` directly
with atoms in the same shape as `Atomtuple` (`label, type, x, y, z, part,
symm_matrix, adp`), Cartesian Å coordinates.

## API mapping to `MoleculeWidgetProtocol`

| Python (`MoleculeWidget`)      | JavaScript (`MoleculeWidget2D`) |
|---------------------------------|----------------------------------|
| `open_molecule(atoms, cell, keep_view)` | `openMolecule({atoms, cell, keepView})` |
| `clear()`                       | `clear()` |
| `show_adps(bool)`                | `showAdps(bool)` |
| `show_labels(bool)`               | `showLabels(bool)` |
| `show_hydrogens(bool)`            | `showHydrogens(bool)` |
| `set_visible_parts(set\|None)`     | `setVisibleParts(Set\|null)` |
| `set_bond_width(int)`             | `setBondWidth(int)` |
| `set_bond_color(color)`           | `setBondColor(cssColor)` |
| `set_labels_visible(bool)`        | `setLabelsVisible(bool)` |
| `set_background_color(color)`    | `setBackgroundColor(cssColor)` |
| `setLabelFont(size)`             | `setLabelFont(size)` |
| `reset_view()`                   | `resetView()` |
| `align_best_view()`              | `alignBestView()` |
| `save_image(path)`               | `saveImage(filename)` / `toDataURL()` |
| `reset_rotation_center()`         | `resetRotationCenter()` |
| `atomClicked` / `bondClicked` / `partsChanged` signals | `EventTarget` events of the same names (`widget.addEventListener('atomClicked', e => ...)`, `e.detail` holds the payload) |

## Mouse / keyboard controls

Identical to the Python widget: left-drag rotate, right-drag zoom,
middle-drag pan, scroll wheel changes label size, left-click selects
(Ctrl = multi-select, Alt = recentre pivot), middle-click recentres the
rotation pivot, F1/F2/F3 align to real-space axes a/b/c (requires a unit
cell and canvas focus).

## Notes on fidelity

- ADP ellipsoids, principal-axis arcs, bond gradients, hover tooltips,
  selection highlighting, best-view (PCA), and the unit-cell axis indicator
  are all ported with the same geometry/math as the Python renderer.
  Gradient shading is a close visual approximation of Qt's
  `QRadialGradient`/`QLinearGradient` (exact pixel parity isn't required —
  the underlying geometry is exact).
- The analytic 3×3 symmetric eigen-decomposition (`linalg.js: eigSym3`) was
  validated against `numpy.linalg.eigh` (max error ~1e-14 on 2000 random
  matrices), including atoms with (near-)degenerate ADP eigenvalues (common
  for atoms sitting on a crystallographic symmetry axis, e.g. `N3`/`C23`/`C24`
  in `p31c.cif`): the row-cross-product null-space trick used for a single
  eigenvector is ill-conditioned when its eigenvalue isn't well separated
  from the others, so `eigSym3` first solves for whichever eigenvalue *is*
  well separated (always one of the two extremes) and then resolves the
  remaining (possibly degenerate) pair via a well-conditioned 2×2 eigenproblem
  in the orthogonal-complement plane, instead of falling back to an arbitrary
  fixed world-axis vector.
- Interactive rotation (`_applyDeltaRotation`) rigidly rotates each atom's
  cached ADP eigenvectors/inverse by the incremental delta rotation rather
  than re-running `eigSym3` on every drag frame, mirroring the Python/Qt
  renderer's `rotate_molecule`. `eigSym3` is only invoked once per file
  load/reload. This matters because eigenvector choice within a degenerate
  eigenspace, while always mathematically valid, is an arbitrary pick each
  time it's recomputed — re-deriving it every frame made the principal-axis
  cross-section arcs jump to a different (correct but visually
  "differently-rotated") basis on almost every mouse-move event.
- `SDM.grow()`, `SDM.packUnitCell()`, and `buildConnTable()` were validated
  against the real `fastmolwidget.sdm.SDM` / `fastmolwidget.tools.build_conntable`
  on real CIF test structures and produce byte-for-byte identical atom sets.
- The `symm_matrix` returned by `SDM.grow()`/`SDM.packUnitCell()` must be the
  **transpose** of the raw symmetry-operation matrix, matching the convention
  `fastmolwidget.sdm.SDM` uses internally (it stores `SymmetryElement.matrix.T`).
  `MoleculeWidget2D._uijToCart()` applies `symm_matrix.T @ Uij @ symm_matrix`
  (a literal port of `molecule_painter._uij_to_cart`), which only produces the
  correct tensor-transform law `U' = M @ U @ M.T` when `symm_matrix` is `M.T`.
  Storing the raw (non-transposed) matrix silently produces wrong — often
  non-positive-definite — ADP ellipsoids for every symmetry-generated
  (grown/packed) atom while leaving asymmetric-unit atoms unaffected. Verified
  against the real Python renderer on `tests/test-data/p31c.cif`: all 52
  symmetry-generated ADP atoms match to `1e-16` after the fix (vs. 0.04–0.22 Ų
  off before it).
