# AGENTS.md — Fastmolwidget

Embeddable PyQt/PySide6 widget for crystal-structure display. Two parallel renderers (2D QPainter + 3D OpenGL) share a single public API.

## Architecture (read these first)

- `src/fastmolwidget/__init__.py` — the *only* public surface; everything re-exported here is API.
- `molecule_base.py` — `MoleculeWidgetProtocol` (`@runtime_checkable`). Every renderer must satisfy it. New display methods MUST be added here too.
- `molecule2D.py` (`MoleculeWidget`) — pure-Python QPainter renderer; ORTEP ellipsoids, no OpenGL.
- `molecule3D.py` (`MoleculeWidget3D`) — `QOpenGLWidget` with sphere/ellipsoid impostors and tessellated cylinder bonds. GLSL targets `#version 120`. Must degrade gracefully when `PyOpenGL` is missing or context creation fails — see the `_HAS_PYOPENGL` guard at module top; never let the host app crash.
- `molecule3D.py` has priority now.
- `viewer_widget.py` / `viewer_widget3D.py` — `MoleculeViewerWidget` / `MoleculeViewer3DWidget` bundle a renderer + two-row control bar. Both expose `.render_widget` and `.load_file(path)`.
  - **Row 1** (both): Open File… button, Grow, Pack Unit Cell, Show ADP, Show Labels, Hide Hydrogens (`checked` → `show_hydrogens(False)`). Grow and Pack are mutually exclusive (toggling one unchecks the other).
  - **Row 2** (both): Bond Width spinbox (2D: 1–15, 3D: 0–15), Bond Color button (opens `QColorDialog`), Reset Rotation Center button → calls `render_widget.reset_rotation_center()`
  - Both viewers also expose a `.set_bond_color(color)` method that forwards to the renderer.
- `loader.py` (`MoleculeLoader`) — composition, not inheritance. Dispatch table `_FORMAT_MAP` maps suffix → method (`.cif`→`_load_cif`, `.res`/`.ins`→`_load_shelx`, `.xyz`→`_load_xyz`). To add a format: add an entry, implement `_load_<fmt>(self, path, *, keep_view)`, and (if it has symmetry) include the suffix in `_GROWABLE_FORMATS`. Also exposes `set_pack(enabled, symmop_indices)` for unit-cell packing (mutually exclusive with grow; pack takes priority).
- `sdm.py` — Shortest-Distance-Matrix algorithm to grow asymmetric units to whole molecules. Optional C++ acceleration via `import sdm_cpp` (`HAS_CPP` flag); the Python path must keep working. `SDM.calc_sdm` mutates its input atom list — always pass a fresh copy (see `_compute_grown_atoms`). Also provides `SDM.pack_unit_cell(symmop_indices)` for unit-cell packing (does not require `calc_sdm` to have been run first).
- `cif/cif_file_io.py` — `CifReader` (uses `gemmi`) yields `Atomtuple` via `.atoms_orth` / `.atoms_fract` and ADPs via `.displacement_parameters()`.
- `atoms.py`, `dsrmath.py`, `tools.py` — element radii/colours, vector/matrix helpers, `to_float` for CIF strings (strips esd parentheses).

## Core data type

`Atomtuple = namedtuple('Atomtuple', ('label','type','x','y','z','part','symm_matrix'), defaults=(None,))` (defined in `sdm.py`). Coordinates are Cartesian Å when fed to widgets; `part` is SHELX disorder part. ADPs are dicts `{label: (U11,U22,U33,U23,U13,U12)}`. `cell` is `(a,b,c,α,β,γ)`.

## Conventions

- **Qt binding-agnostic**: always `from qtpy import ...`; never import `PySide6` / `PyQt6` directly. Stubs use `pyside6-stubs`.
- **3D fallback path**: any code path in `molecule3D.py` that touches `gl.*` must be guarded so the widget reverts to a `QWidget` text overlay instead of raising.
- **Growing structures**: enabled via `MoleculeLoader.set_grow(True)`; reloads the last file in-place with `keep_view=True`. XYZ has no symmetry → grow is a no-op.
- **Packing structures**: enabled via `MoleculeLoader.set_pack(True)`; applies all (or selected) symmetry operations and folds atoms into one unit cell. Pack takes priority over grow when both are active.
- **Public API additions** must be reflected in: the relevant widget class, `MoleculeWidgetProtocol`, `__init__.py` `__all__`, and `README.md`. Current protocol methods: `open_molecule`, `clear`, `show_adps`, `show_labels`, `show_hydrogens`, `set_bond_width`, `set_bond_color`, `set_labels_visible` (alias for `show_labels`), `set_background_color`, `setLabelFont`, `reset_view`, `save_image`.
- **`shelxfile`** is a required runtime dependency (used in `loader.py` for `.res`/`.ins`); it is listed in `pyproject.toml` `dependencies`, not in extras.
- **3D mouse controls**: left-drag rotate, right-drag zoom, middle-drag pan, **middle-click recentres the rotation pivot** on the clicked atom (`reset_rotation_center()` restores the default), scroll-wheel adjusts label font size.
- **Keyboard shortcuts** (both 2D and 3D, requires unit cell): **F1** aligns the view so that reciprocal axis a* points towards the viewer; **F2** → b*; **F3** → c*. No-op when no cell is loaded. A unit-cell axis indicator (a=red, b=green, c=blue) is drawn in the bottom-left corner while Pack Unit Cell is active.
- **No new top-level deps** without updating `pyproject.toml` extras (`pyside6`, `pyqt6`, `gl3d`, `cpp`).
- Python 3.12+ syntax (`X | None`, `from __future__ import annotations` is used widely). Per repo policy assume Python 3.14, PyQt5 *only when explicitly editing user code*; library code stays on `qtpy` + Python ≥ 3.12.

## Developer workflow

This repo uses **uv** (`uv.lock`) for development, but the build backend is **setuptools** (`build-backend = "setuptools.build_meta"`) because it supports the optional `sdm_cpp` C++ extension built via `setup.py`.

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
