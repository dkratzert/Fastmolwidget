# AGENTS.md — Fastmolwidget

Embeddable PyQt/PySide6 widget for crystal-structure display. Two parallel renderers (2D QPainter + 3D OpenGL) share a single public API.

## Architecture (read these first)

- `src/fastmolwidget/__init__.py` — the *only* public surface; everything re-exported here is API.
- `molecule_base.py` — `MoleculeWidgetProtocol` (`@runtime_checkable`). Every renderer must satisfy it. New display methods MUST be added here too.
- `molecule2D.py` (`MoleculeWidget`) — pure-Python QPainter renderer; ORTEP ellipsoids, no OpenGL.
- `molecule3D.py` (`MoleculeWidget3D`) — `QOpenGLWidget` with sphere/ellipsoid impostors and tessellated cylinder bonds. GLSL targets `#version 120`. Must degrade gracefully when `PyOpenGL` is missing or context creation fails — see the `_HAS_PYOPENGL` guard at module top; never let the host app crash.
- `molecule3D.py` has priority now.
- `viewer_widget.py` / `viewer_widget3D.py` — `MoleculeViewerWidget` / `MoleculeViewer3DWidget` bundle a renderer + control bar (Grow, Show ADP, Show Labels, Round Bonds, Show Hydrogens, Bond Width 1–15). Both viewers expose `.render_widget` and `.load_file(path)`.
- `loader.py` (`MoleculeLoader`) — composition, not inheritance. Dispatch table `_FORMAT_MAP` maps suffix → method (`.cif`→`_load_cif`, `.res`/`.ins`→`_load_shelx`, `.xyz`→`_load_xyz`). To add a format: add an entry, implement `_load_<fmt>(self, path, *, keep_view)`, and (if it has symmetry) include the suffix in `_GROWABLE_FORMATS`.
- `sdm.py` — Shortest-Distance-Matrix algorithm to grow asymmetric units to whole molecules. Optional C++ acceleration via `import sdm_cpp` (`HAS_CPP` flag); the Python path must keep working. `SDM.calc_sdm` mutates its input atom list — always pass a fresh copy (see `_compute_grown_atoms`).
- `cif/cif_file_io.py` — `CifReader` (uses `gemmi`) yields `Atomtuple` via `.atoms_orth` / `.atoms_fract` and ADPs via `.displacement_parameters()`.
- `atoms.py`, `dsrmath.py`, `tools.py` — element radii/colours, vector/matrix helpers, `to_float` for CIF strings (strips esd parentheses).

## Core data type

`Atomtuple = namedtuple('Atomtuple', ('label','type','x','y','z','part','symm_matrix'), defaults=(None,))` (defined in `sdm.py`). Coordinates are Cartesian Å when fed to widgets; `part` is SHELX disorder part. ADPs are dicts `{label: (U11,U22,U33,U23,U13,U12)}`. `cell` is `(a,b,c,α,β,γ)`.

## Conventions

- **Qt binding-agnostic**: always `from qtpy import ...`; never import `PySide6` / `PyQt6` directly. Stubs use `pyside6-stubs`.
- **OpenGL setup**: callers must invoke `configure_opengl_format()` *before* `QApplication(...)` (mandatory on macOS). It silently swallows platform errors — keep that behaviour.
- **3D fallback path**: any code path in `molecule3D.py` that touches `gl.*` must be guarded so the widget reverts to a `QWidget` text overlay instead of raising.
- **Growing structures**: enabled via `MoleculeLoader.set_grow(True)`; reloads the last file in-place with `keep_view=True`. XYZ has no symmetry → grow is a no-op.
- **Public API additions** must be reflected in: the relevant widget class, `MoleculeWidgetProtocol`, `__init__.py` `__all__`, and `README.md`.
- **No new top-level deps** without updating `pyproject.toml` extras (`pyside6`, `pyqt6`, `gl3d`).
- Python 3.12+ syntax (`X | None`, `from __future__ import annotations` is used widely). Per repo policy assume Python 3.14, PyQt5 *only when explicitly editing user code*; library code stays on `qtpy` + Python ≥ 3.12.

## Developer workflow

This repo uses **uv** (`uv.lock`, `build-backend = "uv_build"`).

```bash
uv sync --extra pyside6 --extra gl3d   # install with 3D + Qt binding
uv run pytest                          # run all tests (cwd must be repo root — tests use Path('tests/test-data'))
uv run pytest tests/test_molecule2D.py::test_calc_volume
uv run ruff check src tests
uv run ty check                        # type checker configured in dev group
```

Tests instantiate a process-wide `QApplication` at module import (see top of `tests/test_molecule2D.py`). Tests requiring real OpenGL are skipped on headless CI — pattern shown in `tests/test_viewer_widget3D.py`. Test data lives in `tests/test-data/` (CIF, SHELX `.res`, `.xyz`); reuse those files instead of generating new fixtures.

## Release

Tag-driven via GitHub Actions to **TestPyPI** only. Bump `project.version` in `pyproject.toml`, then `git tag version-X.Y.Z && git push origin version-X.Y.Z`.

## Per-user rules (from `global-copilot-instructions`)

If information or code is missing, **ask** — do not guess or invent. Refuse rather than fabricate. Before writing new code, request a detailed specification.

