# AGENTS.md — Fastmolwidget

Embeddable PyQt/PySide6 widget for crystal-structure display. Three parallel renderers (2D QPainter, Qt Quick QPainter, 3D OpenGL) share a single public API.

## Architecture (read these first)

- `src/fastmolwidget/__init__.py` — the *only* public surface; everything re-exported here is API.
- `molecule_base.py` — `MoleculeWidgetProtocol` (`@runtime_checkable`). Every renderer must satisfy it. New display methods MUST be added here too.
- `molecule2D.py` (`MoleculeWidget`) — pure-Python QPainter renderer; ORTEP ellipsoids, no OpenGL. Concrete `QWidget` subclass that mixes `MoleculeRendererMixin` with `QWidget`.
- `molecule_painter.py` (`MoleculeRendererMixin`) — Qt-base-class-agnostic mixin containing **all** 2-D rendering state and drawing logic (atoms, bonds, ADPs, hover, selection, PCA best-view). Both `MoleculeWidget` (2D) and `MoleculeQuickItem` (Quick) inherit from it; the concrete class only provides the Qt plumbing (`paintEvent`/`paint`, signals, `save_image`).
- `molecule_quick.py` (`MoleculeQuickItem`) — `QQuickPaintedItem` renderer that re-uses `MoleculeRendererMixin` inside a Qt Quick scene graph. Registered as the QML type `MoleculeItem` (module `Fastmolwidget 1.0`). Guarded by `_HAS_QTQUICK`; import never crashes when Qt Quick is missing.
- `molecule3D.py` (`MoleculeWidget3D`) — `QOpenGLWidget` with sphere/ellipsoid impostors and tessellated cylinder bonds. GLSL targets `#version 120` on macOS and `#version 140` on Windows/Linux. Must degrade gracefully when `PyOpenGL` is missing or context creation fails — see the `_HAS_PYOPENGL` guard at module top; never let the host app crash.
- `molecule3D.py` has priority now.
- `viewer_widget.py` / `viewer_widget3D.py` — `MoleculeViewerWidget` / `MoleculeViewer3DWidget` bundle a renderer + two-row control bar. Both expose `.render_widget` and `.load_file(path)`.
  - **Row 1** (both): Open File… button, Grow, Pack Unit Cell, Show ADP, Show Labels, Hide Hydrogens (`checked` → `show_hydrogens(False)`). Grow and Pack are mutually exclusive (toggling one unchecks the other).
  - **Row 2** (both): Bond Width spinbox (2D: 1–15, 3D: 0–15), Bond Color button (opens `QColorDialog`), Reset Rotation Center button → calls `render_widget.reset_rotation_center()`, Best View button (`align_best_view()`), Save Image… button (`save_image(...)`), and a Parts filter (shown when multiple disorder parts are present). The **3D viewer additionally** has a checkable Residual Density button (pressed + green while shown; `_on_density_toggled`) and a Level `QDoubleSpinBox` (→ `set_residual_density_level`, re-contours without recomputing, enabled only while density is shown); the button is disabled with an explanatory tooltip when `density.HAS_DENSITY_CPP` is `False`.
  - Both viewers also expose a `.set_bond_color(color)` method that forwards to the renderer; the 3D viewer also forwards `.show_residual_density(hkl_path, level)` and `.clear_residual_density()`.
- `viewer_widget_quick.py` — `MoleculeViewerQuickWidget` / `MoleculeViewerBackend` bundle a `QQuickWidget` hosting a QML scene with the same two-row control bar as the widget viewers. `MoleculeViewerBackend` is a `QObject` exposed to QML as the context property `"backend"`; it owns the `MoleculeLoader` and forwards all QML controls to the `MoleculeQuickItem` render item. The QML scene lives in `qml/MoleculeViewer.qml`; the parts-filter dropdown is in `qml/PartFilterComboBox.qml` (opens a `Popup` upward to stay within the `QQuickWidget` bounds). The viewer exposes `.render_widget`, `.load_file(path)`, and `.set_bond_color(color)`.
  - Gracefully degrades when Qt Quick is unavailable (`_HAS_QTQUICK` guard).
- `loader.py` (`MoleculeLoader`) — composition, not inheritance. Dispatch table `_FORMAT_MAP` maps suffix → method (`.cif`→`_load_cif`, `.res`/`.ins`→`_load_shelx`, `.xyz`→`_load_xyz`). To add a format: add an entry, implement `_load_<fmt>(self, path, *, keep_view)`, and (if it has symmetry) include the suffix in `_GROWABLE_FORMATS`. Also exposes `set_pack(enabled, symmop_indices)` for unit-cell packing (mutually exclusive with grow; pack takes priority). `load_file` also records the loaded path on the widget as `widget._model_path`, which `molecule3D.py` needs to compute residual-density maps.
- `hkl_io.py` — Qt-free readers for reflection data and SHELX refinement parameters: `read_shelx_hkl`/`parse_shelx_hkl` (fixed-format `HKLF 4`, free-format fallback, stops at the `0 0 0` terminator), `read_cif_reflections` (fcf-style `_refln_*` loops, reuses `F_calc` when present), `embedded_shelx_hkl` (`_shelx_hkl_file` — self-contained SHELXL CIFs need no separate `.hkl`), and `read_shelx_parameters` → `ShelxParameters(osf, wght_a, wght_b, exti, wavelength, free_variables)`. Parameters are looked up in the order: the `.res`/`.ins` itself → a sibling `.res`/`.ins` of the same basename → a SHELX block embedded in the CIF (`embedded_shelx_res`, tags `_shelx_res_file` / `_iucr_refine_instructions_details`). `_data_blocks()` skips a leading `global_` block; **every** CIF reader here and in `density.py` must go through it.
  - `find_reflection_file(model)` / `has_reflections(path)` implement the automatic lookup: the model file itself first (embedded `_shelx_hkl_file` or `_refln_*` loop), then siblings of the same basename with a `_REFLECTION_SUFFIXES` extension (`.hkl`, `.fcf`, `.fco`, `.cif`). `has_reflections` is deliberately cheap — it never parses the reflections, only checks for the tags. **The viewer button is stricter than this**: `MoleculeViewer3DWidget._auto_reflection_file()` only accepts reflections *embedded in the model file* and otherwise opens the file dialog (pre-selecting a sibling `.hkl`), so it is always visible which dataset a map came from. The wider sibling search is only used by the programmatic `calculate_residual_density(model)` / `show_residual_density()` path.
- `density.py` — residual (Fo−Fc) maps, **Qt-free**. `calculate_residual_density(model, hkl=None)` → `ResidualDensityMap` (`.array` numpy grid over one unit cell, `.max/.min/.rms/.d_min/.scale`, `.isosurface(level, atoms=…, margin=…)` → Cartesian `(vertices, edges)` for `GL_LINES`). `hkl` defaults to the model itself, which is all a self-contained SHELXL CIF needs. Uses gemmi for the SF summation and FFT. Three non-obvious requirements when building a `gemmi.SmallStructure` by hand — all three silently give wrong results if missed: (1) `site.element` must be set (`type_symbol` is only a label, otherwise everything scatters as `X`); (2) `structure.setup_cell_images()` must be called (the SF calculator reads symmetry from `UnitCell`, not from the space group); (3) CIF-read structures need `change_occupancies_to_crystallographic()`. Map coefficients follow SHELXL's own convention: unweighted `(|Fo|/OSF − |Fc|)·exp(iφc)` — `WGHT` is *not* applied to Fourier coefficients.
  - **Grid size is fixed by the cell, not the data**: `_grid_size()` divides the cell edges by `DEFAULT_GRID_SPACING` (0.2 Å — chosen so individual residual-density features keep their shape; coarser grids look blocky once contoured) and rounds up to a 2/3/5/7-smooth FFT size, then `_fits_in_grid()` drops reflections beyond `(n-1)//2` so nothing aliases. `exact_size` (not `min_size`/`sample_rate`) is passed to gemmi, otherwise high-resolution data silently enlarges the grid.
  - **Two data defences that matter a lot**: systematically absent reflections are dropped in `_merge_to_asu` (their `Fc` is 0, so their noise enters the map multiplied by `1/scale`), and `_sanitise_adps()` downgrades non-positive-definite ADP tensors to isotropic with a `RuntimeWarning` (a negative eigenvalue makes the Debye-Waller factor grow with resolution — `tests/test-data/p21c.cif` has exactly this defect on `Al1` and without the guard the map peaks at ±9 e/Å³ instead of ±1.3).
  - **SHELX symmetry**: `_shelx_spacegroup()` must combine the `SYMM` cards with the `LATT` **centring translations** (`_LATT_CENTRING`) and, for a positive `LATT`, an inversion centre. Using the `SYMM` cards alone silently produces a primitive subgroup (C2/c → P2/c), which halves the symmetry mates and makes every Fc wrong — it showed up as a residual map peaking at +31 e/Å³ instead of +0.5.
  - **Twinning**: `_detwin_observations()` splits twinned intensities before the map is built, using `Fo²(h₁) = Io · |Fc(h₁)|² / Σ_k b_k |Fc(h_k)|²`. `HKLF 4` + `TWIN` generates the other domains from the twin law (`h' = h M`, row-vector convention, `_hklf4_groups`); `HKLF 5` reads them from the file, where consecutive records ending in a *positive* batch number form one observation (`_hklf5_groups`). `BASF` gives the fractions of components 2…n. Without this, a twinned refinement's map is several times too large (`Esser_JW316`, `BASF 0.35`: rms 0.21 → 0.05 against SHELXL's 0.04). Two documented gaps: a **negative** `TWIN` component count selects an ordering that is not implemented (warns), and a pure **inversion/racemic** twin is a mathematical no-op because `h` and `-h` only differ through *f″*, which gemmi's real-valued addends cannot express.
  - **SHELX negative `U_iso`** (`-1.5` = 1.5 × U_eq of the pivot atom) is still resolved in `small_structure_from_shelx`: *shelxfile* v25 returns the raw `-1.5` from `uvals` and a meaningless negative `ueq` for riding atoms. Occupancies, by contrast, are taken straight from `atom.occupancy` — v25 decodes the SOF free-variable codes correctly (this needs `shelxfile>=25`; v24 mis-decoded negative SOFs with a site-occupancy factor ≠ 1 and dropped riding atoms' SOFs).
- `fastmolwidget/density_cpp` (`src/fastmolwidget/density_cpp/density_cpp.cpp`) — optional pybind11 marching-cubes extension, built by `setup.py` alongside `sdm_cpp` and imported as `from fastmolwidget import density_cpp`. `marching_cubes(grid, level, origin, step) -> (vertices, edges)` with deduplicated wireframe edges. Guarded by `density.HAS_DENSITY_CPP`, which checks `hasattr(density_cpp, 'marching_cubes')` rather than just the import — a bare import can succeed against a stale or unbuilt module. There is **no** pure-Python fallback: the feature raises a clear `RuntimeError` and the viewer disables its button instead. Tests must `importorskip("fastmolwidget.density_cpp")` (the fully qualified name), otherwise they silently skip.
- `sdm.py` — Shortest-Distance-Matrix algorithm to grow asymmetric units to whole molecules. Optional C++ acceleration via `import sdm_cpp` (`HAS_CPP` flag); the Python path must keep working. `SDM.calc_sdm` mutates its input atom list — always pass a fresh copy (see `_compute_grown_atoms`). Also provides `SDM.pack_unit_cell(symmop_indices)` for unit-cell packing (does not require `calc_sdm` to have been run first).
- `cif/cif_file_io.py` — `CifReader` (uses `gemmi`) yields `Atomtuple` via `.atoms_orth` / `.atoms_fract` and ADPs via `.displacement_parameters()`.
- `atoms.py`, `dsrmath.py`, `tools.py` — element radii/colours, vector/matrix helpers, `to_float` for CIF strings (strips esd parentheses).

## Core data type

`Atomtuple = namedtuple('Atomtuple', ('label','type','x','y','z','part','symm_matrix','adp'), defaults=(None, None))` (defined in `sdm.py`). Coordinates are Cartesian Å when fed to widgets; `part` is SHELX disorder part. `adp` is `(U11,U22,U33,U23,U13,U12)` or `None` for isotropic atoms — embedded directly in the tuple. `cell` is `(a,b,c,α,β,γ)`.

## Conventions

- **Qt binding-agnostic**: always `from qtpy import ...`; never import `PySide6` / `PyQt6` directly. Stubs use `pyside6-stubs`.
- **3D fallback path**: any code path in `molecule3D.py` that touches `gl.*` must be guarded so the widget reverts to a `QWidget` text overlay instead of raising.
- **Growing structures**: enabled via `MoleculeLoader.set_grow(True)`; reloads the last file in-place with `keep_view=True`. XYZ has no symmetry → grow is a no-op.
- **Packing structures**: enabled via `MoleculeLoader.set_pack(True)`; applies all (or selected) symmetry operations and folds atoms into one unit cell. Pack takes priority over grow when both are active.
- **Public API additions** must be reflected in: the relevant widget class, `MoleculeWidgetProtocol`, `__init__.py` `__all__` (**and** its `_LAZY_IMPORTS` map + `TYPE_CHECKING` block — the top-level package imports lazily), and `README.md`. Current protocol methods: `open_molecule`, `clear`, `show_adps`, `show_labels`, `show_hydrogens`, `set_visible_parts`, `set_bond_width`, `set_bond_color`, `set_labels_visible` (alias for `show_labels`), `set_background_color`, `setLabelFont`, `reset_view`, `align_best_view`, `show_residual_density`, `clear_residual_density`, `save_image`. `Atomtuple`, `MoleculeQuickItem`, `MoleculeViewerBackend`, `MoleculeRendererMixin`, `ResidualDensityMap`, `calculate_residual_density`, `ReflectionData`, `ShelxParameters`, `read_reflections`, `read_shelx_parameters`, `bundle_js`, `structure_json`, `render_html`, and `write_html` are exported in `__all__`.
- **`shelxfile`** is a required runtime dependency (used in `loader.py` for `.res`/`.ins`); it is listed in `pyproject.toml` `dependencies`, not in extras.
- **3D mouse controls**: left-drag rotate, right-drag zoom, middle-drag pan, **middle-click recentres the rotation pivot** on the clicked atom (`reset_rotation_center()` restores the default), scroll-wheel adjusts label font size.
- **Keyboard shortcuts** (both 2D and 3D, requires unit cell): **F1** aligns the view so that real-space axis **a** points towards the viewer (i.e. the b–c face is seen flat-on); **F2** → **b**; **F3** → **c**. For orthogonal cells this is the same as aligning along the reciprocal axis; for non-orthogonal cells it correctly places the chosen cell edge perpendicular to the screen. No-op when no cell is loaded. A unit-cell axis indicator (a=red, b=green, c=blue) is drawn in the bottom-left corner while Pack Unit Cell is active.
- **No new top-level deps** without updating `pyproject.toml` extras (`pyside6`, `pyqt6`, `gl3d`, `cpp`, `density`).
- **Residual density (3D only)**: `show_residual_density(hkl_path=None, level=None, *, model_path=None)`, `set_residual_density_level(level)`, `clear_residual_density()` and the `residual_density_map` property live on `MoleculeWidget3D`. `hkl_path=None` (the default) resolves the reflections via `hkl_io.find_reflection_file`; the viewer button, however, only auto-uses reflections embedded in the model file and opens a dialog for everything else (see `hkl_io.py` above). `level=None` contours at `density.DEFAULT_SIGMA` (3.0) × the map RMS via `ResidualDensityMap.sigma_level()` — a fixed absolute level is wrong here, because the RMS of a residual map varies by an order of magnitude between structures. The viewer pushes the resulting value into the Level spin box. The viewer's button is **checkable** and is the single source of truth for the on/off state — `_on_density_toggled` pops it back out on any failure (no model, no data, cancelled dialog) — while `show_/clear_residual_density` push the state back into it via `_set_density_controls_active()`, which also enables/disables the Level spinbox.
  - **Lifetime**: `MoleculeLoader.load_file` clears the density when the path differs from `widget._model_path`, since the map belongs to the previous model's reflections; reloading the *same* path (what grow and pack do) keeps it, and `_load_molecule` then re-runs `_build_density_geometry()` so the surface is re-clipped around the new atom set. `MoleculeViewer3DWidget.load_file` calls `_sync_density_controls()` afterwards so the button never claims density is shown when it is not. `show_/clear_residual_density` are part of `MoleculeWidgetProtocol`; the 2D and Quick renderers implement them as documented no-ops. Rendering uses `shaders.LINE_VERT/LINE_FRAG` and one shared VBO holding both lobes (positive lobe first, `_density_pos_count` indices, then the negative lobe), drawn as `GL_LINES` after the solid geometry. The surface is clipped to `molecule3D.DENSITY_MARGIN` (1.5 Å) around `_visible_atom_positions()` — the atoms actually drawn, i.e. after the hydrogen and disorder-part filters — so `show_hydrogens()` and `set_visible_parts()` re-contour the density (cheap; the map itself is cached and not recomputed).
- Python 3.12+ syntax (`X | None`, `from __future__ import annotations` is used widely). Per repo policy assume Python 3.14, PyQt5 *only when explicitly editing user code*; library code stays on `qtpy` + Python ≥ 3.12.

## Developer workflow

This repo uses **uv** (`uv.lock`) for development, but the build backend is **setuptools** (`build-backend = "setuptools.build_meta"`) because it supports the optional `sdm_cpp` C++ extension built via `setup.py`.

## JavaScript renderer & web export (`src/fastmolwidget/web/`)

A dependency-free browser/Canvas port of `molecule2D.py` + `molecule_painter.py`
plus the SDM grow/pack-unit-cell logic (`sdm.py`), for embedding molecule
views in web pages without Qt. CIF/SHELX file **parsing** still happens in
Python (`fastmolwidget.web_export` exports the asymmetric unit + symmetry
ops as JSON); growing, packing, and rendering all run in the browser. All
algorithmic ports were numerically cross-validated against the real Python
implementations (see `src/fastmolwidget/web/js/README.md` "Notes on fidelity",
which also documents the file layout, JSON contract and API mapping).

- `web/js/` — the ES modules, shipped as package data (`pyproject.toml`
  `[tool.setuptools.package-data]`). `index.js` is the public entry point;
  `embed.js` provides `createViewer(container, structure, options)` and the
  shared control bar.
- `web/assets.py` — `importlib.resources` access to the shipped JS/templates.
- `web/bundle.py` — `bundle_js()` turns the ES modules into **one classic
  `<script>` blob** defining `window.Fastmolwidget` (a CommonJS-style module
  registry inside an IIFE). It only understands the static named
  `import`/`export` syntax used in `web/js/` and raises
  `UnsupportedJsSyntaxError` on anything else — including import cycles — so a
  JS change can never silently produce a broken bundle. Result is cached
  (`bundle_js.cache_clear()` in dev servers).
- `web/templates/viewer.html` — the standalone page, rendered with
  `string.Template` (`$placeholder`), **not** `str.format`/f-strings (literal
  CSS/JS braces) and **not** Jinja2 (no runtime dependency).
- `web/html.py` — `structure_json()` (escapes `<`/`>`/`&` as `\uXXXX` so it
  stays valid JSON *and* a safe `<script>` payload) and
  `render_html()` / `write_html()` for a fully self-contained page.
- `web_demo_server.py` — threaded dev server serving that page plus the raw ES
  modules.
- **`fastmolwidget.web` must never import Qt.** `fastmolwidget/__init__.py` is
  therefore lazy (:pep:`562` `__getattr__`), so report generators can use the
  web helpers without a Qt binding installed. Keep it that way when adding
  exports.
- JS is bundled at runtime from the packaged sources (no build step), so an
  editable checkout always reflects the current JS.
- `tests/test_web_bundle.py` executes the bundle with Node.js when available
  (skipped otherwise) and cross-checks `SDM.grow()`/`packUnitCell()` against the
  Python `SDM`.

## Copilot instructions
- Install dependencies in a new virtual environment. For 3D support, the host system must have OpenGL drivers and `libegl1` installed (on Debian/Ubuntu):
```bash
apt-get install -y libegl1
```

```bash
uv sync --extra pyside6 --extra gl3d   # install with 3D + Qt binding
uv run pytest                          # run all tests (cwd must be repo root — tests use Path('tests/test-data'))
uv run ruff check src tests
uv run ty check                        # type checker configured in dev group
```

Tests instantiate a process-wide `QApplication` at module import (see top of `tests/test_molecule2D.py`). Tests requiring real OpenGL are skipped on headless CI — pattern shown in `tests/test_viewer_widget3D.py`. Test data lives in `tests/test-data/` (CIF, SHELX `.res`, `.xyz`); reuse those files instead of generating new fixtures.

## Comminication
- Keep sentences short and to the point. Use bullet points, numbered lists, and tables where appropriate. Avoid long paragraphs.
- When asking for clarification, be specific about what information is missing and why it is needed.

## Per-user rules (from `global-copilot-instructions`)

If information or code is missing, **ask** — do not guess or invent. Refuse rather than fabricate. Before writing new code, request a detailed specification.
