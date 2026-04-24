# Fastmolwidget

**A PyQt/PySide6 widget to display crystal structures**

Fastmolwidget is a lightweight, embeddable Qt widget that renders molecular and crystal structures in both 2D projection and real-time 3D OpenGL.
It supports anisotropic displacement parameter (ADP) ellipsoids, ball-and-stick diagrams, and plain sphere representations.
The 2D backend uses a pure-Python QPainter renderer (no OpenGL required); the 3D backend uses hardware-accelerated OpenGL with sphere and ellipsoid impostors.

## Screenshots

| 2D (QPainter) | 3D (OpenGL) |
|---|---|
| ![Fastmolwidget 2D ORTEP view](https://github.com/dkratzert/Fastmolwidget/raw/main/docs/images/screenshot.png) | ![Fastmolwidget 3D OpenGL view](https://github.com/dkratzert/Fastmolwidget/raw/main/docs/images/screenshot3d.png) |
| *ORTEP-style crystal structure with ADP ellipsoids (2D QPainter backend)* | *Real-time 3D ball-and-stick view with depth-shaded spheres and cylinder bonds (OpenGL backend)* |

## Features

- **ADP ellipsoids** at the 50 % probability level, rendered from anisotropic displacement parameters
- **Ball-and-stick** and **isotropic sphere** representations as fallbacks or when speed is more important than detail
- **Real-time 3D rendering** via `MoleculeWidget3D` — sphere impostors and tessellated cylinder bonds in hardware-accelerated OpenGL
- **Graceful degradation** — if PyOpenGL is not installed or OpenGL fails, `MoleculeWidget3D` falls back to an informational text overlay; the host application never crashes
- **Interactive mouse controls**: rotate (left-drag), zoom (right-drag), pan (middle-drag), scroll wheel to resize labels
- **Atom and bond selection**: single click or Ctrl+click for multi-selection; emits `atomClicked` / `bondClicked` Qt signals
- **Hydrogen visibility toggle**
- **Atom label display toggle** with adjustable font size
- **Bond width** adjustment via spin box
- **Multiple file formats**: CIF, SHELX `.res`/`.ins`, and plain XYZ. More to come...
- **Embeddable** — both `MoleculeWidget` (2D) and `MoleculeWidget3D` (3D) are plain `QWidget` subclasses; drop either into any layout
- **Ready-to-use viewers** — `MoleculeViewerWidget` (2D) and `MoleculeViewer3DWidget` (3D) bundle the renderer with a full control bar
- **Common protocol** — `MoleculeWidgetProtocol` lets you write code that works with either widget interchangeably

## Supported File Formats

| Extension | Format | Notes |
|-----------|--------|-------|
| `.cif` | Crystallographic Information File | Reads atoms, unit cell, and ADPs |
| `.res` / `.ins` | SHELXL instruction file | Reads atoms and unit cell via [shelxfile](https://github.com/dkratzert/ShelXFile) |
| `.xyz` | Standard XYZ coordinate file | Cartesian coordinates, no cell or ADPs |

## Installation

```bash
pip install fastmolwidget
```

By default, `fastmolwidget` installs **without a concrete Qt binding**.
Install one binding explicitly via extras:

```bash
pip install "fastmolwidget[pyside6]"
pip install "fastmolwidget[pyqt6]"
```

For **3D OpenGL rendering**, also install PyOpenGL:

```bash
pip install PyOpenGL
```

Without PyOpenGL the 3D widget still loads and displays a helpful message — the host application never crashes.

**Requirements**: Python >= 3.12, NumPy, gemmi, shelxfile, qtpy, and either PySide6 or PyQt6.

## Quick Start

### Standalone 2D viewer

```python
from qtpy.QtWidgets import QApplication
from fastmolwidget import MoleculeViewerWidget

app = QApplication([])
viewer = MoleculeViewerWidget()
viewer.load_file("structure.cif")
viewer.show()
app.exec()
```

### Standalone 3D viewer

```python
from qtpy.QtWidgets import QApplication
from fastmolwidget import MoleculeViewer3DWidget, configure_opengl_format

# Must be called before QApplication is created
configure_opengl_format()

app = QApplication([])
viewer = MoleculeViewer3DWidget()
viewer.load_file("structure.cif")
viewer.show()
app.exec()
```

### Embedding the 3D widget in your own layout

```python
from fastmolwidget import MoleculeWidget3D, configure_opengl_format

configure_opengl_format()   # before QApplication

mol = MoleculeWidget3D(parent=self)
mol.open_molecule(atoms, cell=cell, adps=adps)
layout.addWidget(mol)
```

### Embedding in your own layout (2D)

```python
from fastmolwidget import MoleculeWidget, MoleculeLoader

mol = MoleculeWidget(parent=self)
loader = MoleculeLoader(mol)
# The loader recognizes the file format from the extension and populates `mol` accordingly
loader.load_file("structure.cif")

# drop `mol` into any QLayout
layout.addWidget(mol)
```

### Loading a different file at runtime

```python
viewer.load_file("new_structure.res")
```

### Reacting to atom / bond clicks

```python
mol.atomClicked.connect(lambda label: print(f"Clicked atom: {label}"))
mol.bondClicked.connect(lambda a, b: print(f"Clicked bond: {a}–{b}"))
```

## Mouse Controls

| Action | Effect |
|--------|--------|
| Left-drag | Rotate the molecule |
| Right-drag | Zoom in / out |
| Middle-drag | Pan the view |
| Scroll wheel | Increase / decrease label font size |
| Left-click | Select a single atom or bond |
| Ctrl + Left-click | Toggle multi-selection |

## Control Bar Options

### `MoleculeViewerWidget` (2D) and `MoleculeViewer3DWidget` (3D)

Both viewers expose the same control bar:

| Control | Default | Description |
|---------|---------|-------------|
| Grow | ✗ | Expand the asymmetric unit to complete molecules |
| Show ADP | ✓ | Toggle ORTEP ellipsoid / isotropic sphere rendering |
| Show Labels | ✗ | Toggle non-hydrogen atom labels |
| Round Bonds | ✓ | Switch between round cylinder and flat bond drawing |
| Show Hydrogens | ✓ | Show or hide hydrogen atoms and their bonds |
| Bond Width | 3 | Stroke width / cylinder radius for bonds (1–15) |

## API Overview

### `configure_opengl_format()`

```python
from fastmolwidget import configure_opengl_format
configure_opengl_format()   # call before QApplication(...)
```

Sets a sensible `QSurfaceFormat` default (24-bit depth, double-buffer, 4× MSAA) for all platforms, including macOS where the format **must** be configured before any GL context is created.  Safe to call multiple times; any platform error is silently swallowed.

### `MoleculeViewer3DWidget(parent=None)`

A self-contained 3D viewer combining `MoleculeWidget3D` with the control bar.

- `load_file(path)` — load a structure file (format auto-detected from extension: `.cif`, `.res`, `.ins`, `.xyz`)
- `render_widget` — read-only property exposing the underlying `MoleculeWidget3D`

### `MoleculeWidget3D(parent=None)`

Hardware-accelerated OpenGL renderer.  A `QOpenGLWidget` (Qt ≥ 6) or `QWidget` subclass that can be dropped into any layout.

**Rendering technique**

| Primitive | Technique |
|-----------|-----------|
| Atoms | Billboard sphere impostors — each atom is a quad; the fragment shader ray-casts a sphere and writes corrected depth values |
| ADP ellipsoids | Impostor quads — the fragment shader ray-casts an exact ellipsoid using the inverse U_cart tensor passed as a `mat3` uniform |
| Bonds | Tessellated cylinder mesh (8-segment, 4-segment for angular style) built on the CPU and uploaded as a single VBO |
| Labels | `QPainter` overlay drawn after the OpenGL pass |

All GLSL shaders target `#version 120` (OpenGL 2.1 / GLSL 1.20) for maximum hardware compatibility.

If *PyOpenGL* is not installed, or if OpenGL context creation fails at runtime, the widget falls back to a plain text overlay — the host application never crashes.

#### Qt Signals

| Signal | Signature | Emitted when |
|--------|-----------|--------------|
| `atomClicked` | `(label: str)` | The user clicks on an atom |
| `bondClicked` | `(label1: str, label2: str)` | The user clicks on a bond |

#### Data Methods

- **`open_molecule(atoms, cell=None, adps=None, keep_view=False)`**  
  Load a new set of atoms and redraw.  
  - `atoms` — list of `Atomtuple(label, type, x, y, z, part)` in Cartesian coordinates (Å)  
  - `cell` — optional `(a, b, c, α, β, γ)` tuple; required for ADP rendering  
  - `adps` — optional `dict` mapping atom labels to `(U11, U22, U33, U23, U13, U12)` tensors  
  - `keep_view` — preserve current zoom, rotation, and pan when `True`

- **`grow_molecule(atoms, cell=None, adps=None)`**  
  Replace atoms while preserving the view. Equivalent to `open_molecule(..., keep_view=True)`.

- **`clear()`**  
  Remove all atoms and bonds.

#### Display Methods

- **`show_adps(value: bool)`** — toggle ADP ellipsoid rendering; falls back to isotropic spheres when `False`
- **`show_labels(value: bool)`** — show / hide atom labels
- **`show_hydrogens(value: bool)`** — show / hide hydrogen atoms and bonds
- **`show_round_bonds(value: bool)`** — switch between 8-segment (`True`) and 4-segment (`False`) cylinder bonds
- **`set_bond_width(width: int)`** — set cylinder radius scale (1–15)
- **`setLabelFont(font_size: int)`** — set label font pixel size
- **`set_background_color(color: QColor)`** — change background colour
- **`reset_view()`** — reset zoom, rotation, and pan to defaults
- **`save_image(filename, image_scale=1.5)`** — render the current view to an image file

#### Example — feeding atom data directly to `MoleculeWidget3D`

```python
from fastmolwidget import MoleculeWidget3D, configure_opengl_format
from fastmolwidget.sdm import Atomtuple

configure_opengl_format()   # before QApplication

mol = MoleculeWidget3D(parent=self)

atoms = [
    Atomtuple(label="C1", type="C", x=0.0,  y=0.0,  z=0.0,  part=0),
    Atomtuple(label="O1", type="O", x=1.22, y=0.0,  z=0.0,  part=0),
    Atomtuple(label="H1", type="H", x=-0.5, y=0.94, z=0.0,  part=0),
]

adps = {
    "C1": (0.02, 0.02, 0.02, 0.0, 0.0, 0.0),
    "O1": (0.03, 0.03, 0.03, 0.0, 0.0, 0.0),
}

cell = (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)

mol.open_molecule(atoms=atoms, cell=cell, adps=adps)
mol.atomClicked.connect(lambda label: print(f"Selected: {label}"))

layout.addWidget(mol)
```

### `MoleculeViewerWidget(parent=None)`

A self-contained 2D viewer combining `MoleculeWidget` with the control bar.

- `load_file(path)` — load a structure file (format auto-detected from extension)
- `render_widget` — read-only property exposing the underlying `MoleculeWidget`

### `MoleculeWidget(parent=None)`

The 2D QPainter renderer. A plain `QWidget` subclass you can drop into any layout.

#### Qt Signals

| Signal | Signature | Emitted when |
|--------|-----------|--------------|
| `atomClicked` | `(label: str)` | The user clicks on an atom; `label` is the atom name (e.g. `"C1"`) |
| `bondClicked` | `(label1: str, label2: str)` | The user clicks on a bond; both atom labels are passed |

#### Data Methods

- **`open_molecule(atoms, cell=None, adps=None, keep_view=False)`**  
  Load a new set of atoms and reset (or optionally preserve) the view.  
  - `atoms` — list of `Atomtuple(label, type, x, y, z, part)` in Cartesian coordinates (Å)  
  - `cell` — optional `(a, b, c, α, β, γ)` tuple of unit-cell parameters (Å / °); required for ADP rendering  
  - `adps` — optional `dict` mapping atom labels to `(U11, U22, U33, U23, U13, U12)` ADP tensors  
  - `keep_view` — when `True`, the current zoom, pan, and rotation are preserved (useful for live updates)

- **`grow_molecule(atoms, cell=None, adps=None)`**  
  Replace the atom set while always preserving the current view.  
  Equivalent to calling `open_molecule(..., keep_view=True)`.

- **`clear()`**  
  Remove all atoms and bonds from the display.

#### Display Methods

- **`show_adps(value: bool)`**  
  Toggle ORTEP-style ADP ellipsoid rendering. When `False`, atoms are drawn as isotropic spheres.

- **`show_labels(value: bool)`**  
  Show or hide non-hydrogen atom labels.

- **`show_hydrogens(value: bool)`**  
  Show or hide hydrogen / deuterium atoms and their bonds.

- **`show_round_bonds(value: bool)`**  
  Switch between 3D-shaded cylinder-style bonds (`True`, default) and flat single-colour bonds (`False`).

- **`set_bond_width(width: int)`**  
  Set the stroke width for bonds in pixels (valid range: 1–15).

- **`setLabelFont(font_size: int)`**  
  Set the pixel size used for atom labels.

- **`set_background_color(color: QColor)`**  
  Change the widget background colour.

- **`reset_view()`**  
  Reset zoom, pan, and rotation to their defaults.

#### Example — feeding atom data directly to `MoleculeWidget` (2D)

```python
from fastmolwidget import MoleculeWidget
from fastmolwidget.sdm import Atomtuple

mol = MoleculeWidget(parent=self)

atoms = [
    Atomtuple(label="C1", type="C", x=0.0,  y=0.0,  z=0.0,  part=0),
    Atomtuple(label="O1", type="O", x=1.22, y=0.0,  z=0.0,  part=0),
    Atomtuple(label="H1", type="H", x=-0.5, y=0.94, z=0.0,  part=0),
]

# ADP tensors: {atom_label: (U11, U22, U33, U23, U13, U12)}
adps = {
    "C1": (0.02, 0.02, 0.02, 0.0, 0.0, 0.0),
    "O1": (0.03, 0.03, 0.03, 0.0, 0.0, 0.0),
}

cell = (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)   # optional

mol.open_molecule(atoms=atoms, cell=cell, adps=adps)
mol.atomClicked.connect(lambda label: print(f"Selected: {label}"))

layout.addWidget(mol)
```

### `MoleculeWidgetProtocol`

```python
from fastmolwidget import MoleculeWidgetProtocol
```

A `typing.Protocol` (runtime-checkable) that captures the common public API of both `MoleculeWidget` and `MoleculeWidget3D`.  Use it to write renderer-agnostic code:

```python
from fastmolwidget import MoleculeWidgetProtocol

def load_into(widget: MoleculeWidgetProtocol, atoms, cell, adps) -> None:
    widget.open_molecule(atoms, cell=cell, adps=adps)
    widget.show_labels(True)
```

Any class that implements `open_molecule`, `clear`, `show_adps`, `show_labels`, `show_round_bonds`, `show_hydrogens`, `set_bond_width`, `set_background_color`, `setLabelFont`, `reset_view`, and `save_image` satisfies the protocol.

### `MoleculeLoader(widget)`

Format-aware loader that populates any widget satisfying `MoleculeWidgetProtocol`.

- `load_file(path, keep_view=False)` — parse file and call `open_molecule` / `grow_molecule`
- `set_grow(value: bool)` — toggle automatic molecule growing (expand asymmetric unit)

## License

BSD 2-Clause License — see [LICENSE](LICENSE) for details.

© 2026 Daniel Kratzert

## Maintainer Release Workflow

The release workflow is tag-driven and currently publishes to **TestPyPI** only.

1. Ensure `project.version` in `pyproject.toml` is the version to publish.
2. Create and push a matching tag in the format `version-X.Y.Z`.
3. GitHub Actions builds sdist/wheel and uploads to TestPyPI.

Example:

```bash
git tag version-0.1.0
git push origin version-0.1.0
```

