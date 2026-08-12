![Latest Release](https://img.shields.io/github/v/tag/dkratzert/Fastmolwidget?label=Release)
[![Unit Tests](https://github.com/dkratzert/Fastmolwidget/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/dkratzert/Fastmolwidget/actions/workflows/tests.yml)
![Contributions](https://img.shields.io/badge/contributions-welcome-blue)
[![PyPI package](https://repology.org/badge/version-for-repo/pypi/python:fastmolwidget.svg)](https://repology.org/project/python:fastmolwidget/versions)
<a href="https://repology.org/project/fastmolwidget/versions">
<img src="https://repology.org/badge/vertical-allrepos/fastmolwidget.svg" alt="Packaging status" align="right">
</a>

# fastmolwidget

**A PyQt/PySide6 widget to display crystal structures**

fastmolwidget is a lightweight, embeddable Qt widget that renders molecular and crystal structures in both 2D projection and 3D OpenGL.
It supports anisotropic displacement parameter (ADP) ellipsoids, ball-and-stick diagrams, and plain sphere representations.
The 2D backend uses a pure-Python QPainter renderer (no OpenGL required); the 3D backend uses hardware-accelerated OpenGL with sphere and ellipsoid impostors.
A Qt Quick backend is also available for embedding the 2D renderer inside a QML scene.

## Screenshots

| 2D (QPainter)                                                                                                  | 3D (OpenGL)                                                                                                       |
|----------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|
| ![Fastmolwidget 2D ORTEP view](https://github.com/dkratzert/Fastmolwidget/raw/main/docs/images/screenshot.png) | ![Fastmolwidget 3D OpenGL view](https://github.com/dkratzert/Fastmolwidget/raw/main/docs/images/screenshot3d.png) |
| *ORTEP-style crystal structure with ADP ellipsoids (2D QPainter backend)*                                      | *Real-time 3D ball-and-stick view with depth-shaded spheres and cylinder bonds (OpenGL backend)*                  |

## Features

- **ADP ellipsoids** at the 50 % probability level
- **Ball-and-stick** and **isotropic sphere**
- **Real-time 3D rendering** via `MoleculeWidget3D` — sphere impostors and tessellated cylinder bonds in hardware-accelerated OpenGL
- **Interactive mouse controls**: rotate (left-drag), zoom (right-drag), pan (middle-drag), scroll wheel to resize labels
- **Atom and bond selection**: single click or Ctrl+click for multi-selection; emits `atomClicked` / `bondClicked` Qt signals
- **Hover labels**: hovering over an atom shows its label; hovering over a bond shows the distance in Ångströms
- **Hydrogen visibility toggle**
- **Atom label display toggle** with adjustable font size
- **Bond width** adjustment via spin box
- **Configurable bond color** — set programmatically or via the control-bar color picker
- **Residual (Fo−Fc) density maps (3D)** — computed on the fly from a SHELX `.hkl` (or an fcf-style CIF reflection loop) plus the refined model, and drawn as green/red wireframe isosurfaces; no pre-computed map file needed (see [Residual density maps](#residual-fofc-density-maps-3d))
- **Multiple file formats**: CIF, SHELX `.res`/`.ins`, and plain XYZ. More to come...
- **Embeddable** — both `MoleculeWidget` (2D) and `MoleculeWidget3D` (3D) are plain `QWidget` subclasses; drop either into any layout
- **Qt Quick support** — `MoleculeQuickItem` (`QQuickPaintedItem`) and `MoleculeViewerQuickWidget` allow embedding the 2D renderer in a QML scene
- **Ready-to-use viewers** — `MoleculeViewerWidget` (2D), `MoleculeViewer3DWidget` (3D), and `MoleculeViewerQuickWidget` (Qt Quick) bundle the renderer with a full control bar
- **Common protocol** — `MoleculeWidgetProtocol` lets you write code that works with either widget interchangeably
- **HTML / browser output** — a dependency-free JavaScript port of the 2D renderer ships with the package; `fastmolwidget.web` hands you the renderer and the structure as ready-to-embed strings for HTML reports (see [Embedding in HTML reports](#embedding-in-html-reports))

## Supported File Formats

| Extension       | Format                            | Notes                                                                             |
|-----------------|-----------------------------------|-----------------------------------------------------------------------------------|
| `.cif`          | Crystallographic Information File | Reads atoms, unit cell, and ADPs                                                  |
| `.res` / `.ins` | SHELXL instruction file           | Reads atoms and unit cell via [shelxfile](https://github.com/dkratzert/ShelXFile) |
| `.xyz`          | Standard XYZ coordinate file      | Cartesian coordinates, no cell or ADPs                                            |

## Installation

```bash
# with PySide6 (recommended)
uv add "fastmolwidget[pyside6]"

# or PyQt6
uv add "fastmolwidget[pyqt6]"

# add 3D OpenGL support (optional, requires Qt ≥ 6.7 and pyopenGL installed in the Python environment)
uv add "fastmolwidget[pyside6,gl3d]"
```

### Optional C++ Acceleration (`sdm_cpp`)

The symmetry-growing step (SDM) has an optional C++ extension that uses **pybind11** and **OpenMP** for a significant speed-up on large structures. The pure-Python fallback is always available.

```bash
uv pip install pybind11
uv pip install -e . --no-build-isolation

# macOS: optionally install libomp for multi-threaded acceleration
brew install libomp
```

**Requirements**: Python ≥ 3.12, NumPy, gemmi, shelxfile, qtpy, and either PySide6 or PyQt6.

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
from fastmolwidget import MoleculeViewer3DWidget

app = QApplication([])
viewer = MoleculeViewer3DWidget()
viewer.load_file("structure.cif")
viewer.show()
app.exec()
```

### Qt Quick viewer

The Qt Quick viewer embeds the 2D QPainter renderer inside a QML scene with a QML-native control bar.

```python
from qtpy.QtWidgets import QApplication
from qtpy.QtCore import QTimer
from fastmolwidget import MoleculeViewerQuickWidget

app = QApplication([])
viewer = MoleculeViewerQuickWidget()
viewer.resize(900, 650)
viewer.show()
# Load after show so the QML Component.onCompleted has fired
QTimer.singleShot(100, lambda: viewer.load_file("structure.cif"))
app.exec()
```

> **Note:** `load_file` must be called after the widget is shown and the QML scene has initialised. Using a short `QTimer.singleShot` delay is the simplest approach.

## Embedding the 3D widget in your own layout

```python
from fastmolwidget import MoleculeWidget3D

mol = MoleculeWidget3D(parent=self)
mol.open_molecule(atoms, cell=cell)
layout.addWidget(mol)
```


### Embedding the 2D widget in your own layout

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

| Action                  | Effect                                                                                                                                    |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| Left-drag               | Rotate the molecule                                                                                                                       |
| Right-drag              | Zoom in / out                                                                                                                             |
| Middle-drag             | Pan the view                                                                                                                              |
| Middle-click            | Recentre the rotation pivot on the clicked atom (3D only)                                                                                 |
| Alt/Option + Left-click | On systems without a middle mouse button, Alt/Option + Left-click recentres the rotation pivot on the clicked atom (same as Middle-click) |
| Scroll wheel            | Increase / decrease label font size                                                                                                       |
| Left-click              | Select a single atom or bond                                                                                                              |
| Ctrl + Left-click       | Toggle multi-selection                                                                                                                    |
| Hover over atom         | Show the atom label (enlarged when persistent labels are on)                                                                              |
| Hover over bond         | Show the bond distance (Å) in a rounded tooltip near the cursor                                                                           |

## Keyboard Shortcuts

The widget must have keyboard focus (click on it once) for these shortcuts to work.

| Key | Effect                                                                                          |
|-----|-------------------------------------------------------------------------------------------------|
| F1  | Align the view so that the reciprocal axis **a\*** points towards the viewer (requires a unit cell) |
| F2  | Align the view so that the reciprocal axis **b\*** points towards the viewer (requires a unit cell) |
| F3  | Align the view so that the reciprocal axis **c\*** points towards the viewer (requires a unit cell) |

> **Note:** The F-key shortcuts are available in both the 2D (`MoleculeWidget`) and 3D (`MoleculeWidget3D`) renderers. They have no effect when no unit cell is loaded (e.g. plain XYZ files).

## Control Bar Options

### `MoleculeViewerWidget` (2D) and `MoleculeViewer3DWidget` (3D)

Both viewers expose the same two-row control bar:

**Row 1 — structure toggles**

| Control               | Default | Description                                                                                    |
|-----------------------|---------|------------------------------------------------------------------------------------------------|
| Open File…            | —       | Opens a file dialog to load a structure file                                                   |
| Grow                  | ✗       | Expand the asymmetric unit to complete molecules (mutually exclusive with Pack Unit Cell)       |
| Pack Unit Cell        | ✗       | Generate all symmetry-equivalent positions within one unit cell (mutually exclusive with Grow) |
| Show ADP              | ✓       | Toggle ORTEP ellipsoid / isotropic sphere rendering                                            |
| Show Labels           | ✗       | Toggle non-hydrogen atom labels                                                                |
| Hide Hydrogens        | ✗       | When checked, hydrogen atoms and their bonds are hidden                                        |

**Row 2 — bond and view controls**

| Control               | Default | Description                                                                                    |
|-----------------------|---------|------------------------------------------------------------------------------------------------|
| Bond Width            | 3       | Stroke width / cylinder radius for bonds (2D: 1–15, 3D: 0–15)                                 |
| Bond Color            | —       | Opens a colour picker to change the default bond colour                                        |
| Reset Rotation Center | —       | Restores the rotation pivot to the molecule's geometric centre (both 2D and 3D)               |
| Best View             | —       | Rotates the current structure to a visibility-optimized orientation (PCA on visible atoms)     |
| Save Image…           | —       | Opens a file-save dialog and writes the current view to a PNG or JPEG file                    |
| Residual Density      | off     | *(3D only)* Checkable — pressed (sunken, green) while the Fo−Fc isosurface is shown; click again to hide it. Uses reflections embedded in the model file directly, and opens a file dialog when a separate reflection file is needed |
| Level                 | 0.30    | *(3D only)* Contour level of the residual-density isosurface in e/Å³; enabled only while density is shown |
| Parts                 | All     | Filter displayed disorder parts; shown when multiple part values are present                   |

> When **Pack Unit Cell** is active, a unit-cell axis indicator (a = red, b = green, c = blue) is drawn in the bottom-left corner of the widget and rotates with the view.

### `MoleculeViewerQuickWidget` (Qt Quick)

The Qt Quick viewer provides the same two-row control bar as the widget viewers, but implemented in QML (`qml/MoleculeViewer.qml`). All controls and features are identical; the Parts filter uses a QML `Popup` (opens upward) with checkable items instead of the `QComboBox`-based `PartFilterWidget`.

## API Overview

### `MoleculeViewer3DWidget(parent=None)`

A self-contained 3D viewer combining `MoleculeWidget3D` with the control bar.

- `load_file(path)` — load a structure file (format auto-detected from extension: `.cif`, `.res`, `.ins`, `.xyz`)
- `grow()` — expand the asymmetric unit to complete molecules using crystal symmetry; deactivates Pack Unit Cell if active; no-op for XYZ files or when no file is loaded
- `set_bond_color(color)` — set the default color for non-selected bonds
- `render_widget` — read-only property exposing the underlying `MoleculeWidget3D`

### `MoleculeViewerQuickWidget(parent=None)`

A self-contained Qt Quick viewer embedding a `QQuickWidget` with a QML control bar and a `MoleculeQuickItem` renderer. Degrades gracefully to a text label when Qt Quick is unavailable.

- `load_file(path)` — load a structure file (format auto-detected from extension: `.cif`, `.res`, `.ins`, `.xyz`). Must be called **after** the widget is shown and the QML scene has initialised (use `QTimer.singleShot` for a short delay).
- `set_bond_color(color)` — set the default color for non-selected bonds
- `render_widget` — read-only property exposing the underlying `MoleculeQuickItem` (`None` before the QML `Component.onCompleted` fires or when Qt Quick is unavailable)

### `MoleculeQuickItem(parent=None)`

The Qt Quick renderer. A `QQuickPaintedItem` subclass that shares all drawing logic with `MoleculeWidget` via `MoleculeRendererMixin`. Register with QML before use:

```python
from qtpy.QtQml import qmlRegisterType
from fastmolwidget import MoleculeQuickItem

qmlRegisterType(MoleculeQuickItem, "Fastmolwidget", 1, 0, "MoleculeItem")
```

Then in QML:

```qml
import Fastmolwidget 1.0
MoleculeItem { id: mol; anchors.fill: parent }
```

The item exposes the same data and display methods as `MoleculeWidget` (see below): `open_molecule`, `clear`, `show_adps`, `show_labels`, `show_hydrogens`, `set_visible_parts`, `set_bond_width`, `set_bond_color`, `set_labels_visible`, `setLabelFont`, `set_background_color`, `reset_view`, `align_best_view`, `save_image`.

### `MoleculeWidget3D(parent=None)`

Hardware-accelerated OpenGL renderer. A `QOpenGLWidget` (Qt ≥ 6) or `QWidget` subclass that can be dropped into any layout.

**Rendering technique**

| Primitive      | Technique                                                                                                                    |
|----------------|------------------------------------------------------------------------------------------------------------------------------|
| Atoms          | Billboard sphere impostors — each atom is a quad; the fragment shader ray-casts a sphere and writes corrected depth values   |
| ADP ellipsoids | Impostor quads — the fragment shader ray-casts an exact ellipsoid using the inverse U_cart tensor passed as a `mat3` uniform |
| Bonds          | Tessellated cylinder mesh (8-segment, 4-segment for angular style) built on the CPU and uploaded as a single VBO             |
| Labels         | `QPainter` overlay drawn after the OpenGL pass                                                                               |

GLSL shader targets are platform-aware: `#version 120` on macOS (OpenGL 2.1 / GLSL 1.20) and `#version 140` on Windows/Linux (OpenGL 3.1+ / GLSL 1.40).

#### Qt Signals

| Signal        | Signature                    | Emitted when               |
|---------------|------------------------------|----------------------------|
| `atomClicked` | `(label: str)`               | The user clicks on an atom |
| `bondClicked` | `(label1: str, label2: str)` | The user clicks on a bond  |

#### Data Methods

- **`open_molecule(atoms, cell=None, keep_view=False)`**  
  Load a new set of atoms and redraw.
    - `atoms` — list of `Atomtuple(label, type, x, y, z, part, adp=None)` in Cartesian coordinates (Å); embed `adp=(U11,U22,U33,U23,U13,U12)` directly in the tuple for anisotropic atoms
    - `cell` — optional `(a, b, c, α, β, γ)` tuple; required for ADP rendering
    - `keep_view` — preserve current zoom, rotation, and pan when `True`

- **`grow_molecule(atoms, cell=None)`**  
  Replace atoms while preserving the view. Equivalent to `open_molecule(..., keep_view=True)`.

- **`clear()`**  
  Remove all atoms and bonds.

#### Display Methods

- **`show_adps(value: bool)`** — toggle ADP ellipsoid rendering; falls back to isotropic spheres when `False`
- **`show_labels(value: bool)`** — show / hide atom labels
- **`show_hydrogens(value: bool)`** — show / hide hydrogen atoms and bonds
- **`set_visible_parts(parts: set[int] | None)`** — filter by disorder part; `None` shows all atoms; an empty set hides all atoms; e.g. `set_visible_parts({0, 1})` shows only Part 0 and Part 1
- **`set_bond_width(width: int)`** — set cylinder radius scale (0–15)
- **`set_bond_color(color)`** — set the default color for non-selected bonds; accepts `QColor`, hex string, or an RGB tuple
- **`set_labels_visible(visible: bool)`** — alias for `show_labels`
- **`setLabelFont(font_size: int)`** — set label font pixel size
- **`set_background_color(color: QColor)`** — change background colour
- **`reset_view()`** — reset zoom, rotation, and pan to defaults
- **`align_best_view()`** — rotate the structure so the widest face points towards the viewer (PCA on visible atoms; H/D excluded when hydrogen visibility is off)
- **`reset_rotation_center()`** — restore the rotation pivot to the molecule's geometric center (undoes a middle-click recentring)
- **`save_image(filename: Path, image_scale: float = 1.5)`** — capture the current OpenGL framebuffer and write it to a PNG or JPEG file (format inferred from the file extension). The captured image is then scaled by `image_scale` using smooth bilinear filtering before saving. Labels appear in the saved image if they are active at the time of the call.

#### Residual-density Methods

- **`show_residual_density(hkl_path=None, level=0.30, *, model_path=None)`** — compute a residual (Fo−Fc) map and display it as wireframe isosurfaces (green at `+level`, red at `-level`, in e/Å³). `hkl_path=None` finds the reflections automatically — the model file itself, then siblings of the same basename; `model_path` defaults to the file the widget last loaded. Note the *control-bar button* is deliberately stricter and only auto-uses reflections embedded in the model, asking for anything else. On `MoleculeViewer3DWidget` this also presses the Residual Density button in, so the controls never disagree with the view. Raises `RuntimeError` when no model is available or the compiled `density_cpp` extension is missing, and `FileNotFoundError` when no reflection data can be found.
- **`set_residual_density_level(level: float)`** — re-contour the already computed map; much cheaper than recomputing. No-op when no map is loaded.
- **`clear_residual_density()`** — remove the isosurface.
- **`residual_density_map`** *(property)* — the computed `ResidualDensityMap` (with `.max`, `.min`, `.rms`, `.d_min` and the raw `.array` grid), or `None`.

> These are 3D-only. `MoleculeWidget` (2D) and `MoleculeQuickItem` implement `show_residual_density` / `clear_residual_density` as documented no-ops so that `MoleculeWidgetProtocol` checks keep working across all renderers.

#### Example — feeding atom data directly to `MoleculeWidget3D`

```python
from fastmolwidget import MoleculeWidget3D, Atomtuple

mol = MoleculeWidget3D(parent=self)

# Embed ADP tensors directly in each Atomtuple (None = isotropic / no ADP)
atoms = [
    Atomtuple(label="C1", type="C", x=0.0,  y=0.0,  z=0.0,  part=0,
              adp=(0.02, 0.02, 0.02, 0.0, 0.0, 0.0)),
    Atomtuple(label="O1", type="O", x=1.22, y=0.0,  z=0.0,  part=0,
              adp=(0.03, 0.03, 0.03, 0.0, 0.0, 0.0)),
    Atomtuple(label="H1", type="H", x=-0.5, y=0.94, z=0.0,  part=0),
]

cell = (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)

mol.open_molecule(atoms=atoms, cell=cell)
mol.atomClicked.connect(lambda label: print(f"Selected: {label}"))

layout.addWidget(mol)
```

### `MoleculeViewerWidget(parent=None)`

A self-contained 2D viewer combining `MoleculeWidget` with the control bar.

- `load_file(path)` — load a structure file (format auto-detected from extension)
- `grow()` — expand the asymmetric unit to complete molecules using crystal symmetry; deactivates Pack Unit Cell if active; no-op for XYZ files or when no file is loaded
- `set_bond_color(color)` — set the default color for non-selected bonds
- `render_widget` — read-only property exposing the underlying `MoleculeWidget`

### `MoleculeWidget(parent=None)`

The 2D QPainter renderer. A plain `QWidget` subclass you can drop into any layout.

#### Qt Signals

| Signal        | Signature                    | Emitted when                                                       |
|---------------|------------------------------|--------------------------------------------------------------------|
| `atomClicked` | `(label: str)`               | The user clicks on an atom; `label` is the atom name (e.g. `"C1"`) |
| `bondClicked` | `(label1: str, label2: str)` | The user clicks on a bond; both atom labels are passed             |

#### Data Methods

- **`open_molecule(atoms, cell=None, keep_view=False)`**  
  Load a new set of atoms and reset (or optionally preserve) the view.
    - `atoms` — list of `Atomtuple(label, type, x, y, z, part, adp=None)` in Cartesian coordinates (Å); embed `adp=(U11,U22,U33,U23,U13,U12)` for anisotropic atoms
    - `cell` — optional `(a, b, c, α, β, γ)` tuple of unit-cell parameters (Å / °); required for ADP rendering
    - `keep_view` — when `True`, the current zoom, pan, and rotation are preserved (useful for live updates)

- **`grow_molecule(atoms, cell=None)`**  
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

- **`set_visible_parts(parts: set[int] | None)`**  
  Filter by disorder part number.  `None` (the default) shows all parts.  Pass a
  set of integers to restrict rendering to those parts; an empty set hides every
  atom.  Example: `widget.set_visible_parts({0, 1})` shows Part 0 and Part 1.

- **`set_bond_width(width: int)`**  
  Set the stroke width for bonds in pixels (valid range: 1–15).

- **`set_bond_color(color)`**  
  Set the default color for non-selected bonds. Accepts `QColor`, hex string (e.g. `"#d1812a"`), or an RGB tuple (floats in `[0..1]` or integers in `[0..255]`).

- **`set_labels_visible(visible: bool)`**  
  Alias for `show_labels`.

- **`setLabelFont(font_size: int)`**  
  Set the pixel size used for atom labels.

- **`set_background_color(color: QColor)`**  
  Change the widget background color.

- **`reset_view()`**  
  Reset zoom, pan, and rotation to their defaults.

- **`align_best_view()`**  
  Rotate the structure to the orientation that maximises atom visibility for screenshots.  Uses PCA on the currently visible atom positions: the thinnest axis of the atom cloud points towards the camera so the widest face faces the viewer.  Hydrogen / deuterium atoms are excluded when their visibility is turned off.

- **`save_image(filename: Path, image_scale: float = 1.5)`**  
  Render the current structure view to an image file.  
  The widget is redrawn off-screen at `widget_size × image_scale`; the result is saved as PNG or JPEG (format inferred from the file extension).  
  Labels appear in the saved image if they are active at the time of the call.

#### Example — feeding atom data directly to `MoleculeWidget` (2D)

```python
from fastmolwidget import MoleculeWidget, Atomtuple

mol = MoleculeWidget(parent=self)

# Embed ADP tensors directly in each Atomtuple (omit or use None = isotropic)
atoms = [
    Atomtuple(label="C1", type="C", x=0.0,  y=0.0,  z=0.0,  part=0,
              adp=(0.02, 0.02, 0.02, 0.0, 0.0, 0.0)),
    Atomtuple(label="O1", type="O", x=1.22, y=0.0,  z=0.0,  part=0,
              adp=(0.03, 0.03, 0.03, 0.0, 0.0, 0.0)),
    Atomtuple(label="H1", type="H", x=-0.5, y=0.94, z=0.0,  part=0),
]

cell = (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)  # optional

mol.open_molecule(atoms=atoms, cell=cell)
mol.atomClicked.connect(lambda label: print(f"Selected: {label}"))

layout.addWidget(mol)
```

## Advanced API

### `MoleculeWidgetProtocol`

The core rendering interface is defined by `MoleculeWidgetProtocol`. `MoleculeWidget` (2D), `MoleculeWidget3D` (3D), and `MoleculeQuickItem` (Qt Quick) all satisfy this protocol, making them drop-in replacements for each other.

```python
from fastmolwidget.molecule_base import MoleculeWidgetProtocol
from fastmolwidget import MoleculeWidget3D

def do_something_with_widget(widget: MoleculeWidgetProtocol):
    ...
```

### 3D Application Example

```python
import sys
from qtpy.QtWidgets import QApplication
from fastmolwidget import MoleculeViewer3DWidget

app = QApplication(sys.argv)
viewer = MoleculeViewer3DWidget()
viewer.load_file("examples/test_molecule.res")
viewer.show()
sys.exit(app.exec_())
```

### 3D Generic Widget Example

```python
import sys
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from fastmolwidget import MoleculeWidget3D
from fastmolwidget.loader import MoleculeLoader

app = QApplication(sys.argv)

main_window = QMainWindow()
central_widget = QWidget(main_window)
layout = QVBoxLayout(central_widget)

# Create and configure the 3D molecule widget
molecule_widget = MoleculeWidget3D()
molecule_widget.set_bond_color("#FF5733")  # Example: set bond color to a shade of orange

# Load a molecule file (CIF, RES, or XYZ format)
loader = MoleculeLoader(molecule_widget)
loader.load_file("examples/test_molecule.res")

layout.addWidget(molecule_widget)
main_window.setCentralWidget(central_widget)

main_window.show()
sys.exit(app.exec_())
```

## Residual (Fo−Fc) density maps (3D)

`MoleculeWidget3D` can compute and display a residual electron-density map
directly from a reflection file and the refined model — no `.fcf`, `.map` or
any other pre-computed map file is required.

```python
from fastmolwidget import MoleculeViewer3DWidget

viewer = MoleculeViewer3DWidget()
viewer.load_file("structure.cif")   # a self-contained SHELXL CIF
viewer.show_residual_density()      # reflections come from the CIF itself
```

The reflection data is used **without asking only when it lives inside the
model file** — self-contained SHELXL CIFs carry the whole `.hkl` in
`_shelx_hkl_file`, and fcf-style files carry a `_refln_*` loop.

When the reflections are in a **separate file** (the usual `.res` + `.hkl`
pair) the button opens a file dialog, with a matching `.hkl` next to the model
pre-selected — so it is always visible which dataset a map was computed from.

The button is a **toggle**: while density is displayed it stays pressed and is
tinted green, and clicking it again removes the surface. The *Level* spinbox is
enabled only while a map is shown, and the button's tooltip carries the map
statistics.

Loading a **different structure** switches the density off again — the map
belongs to the previous model's reflections. Grow and Pack reload the same
file, so they keep the map and simply re-clip it around the larger set of
displayed atoms.

Pass an explicit path to skip the dialog:

```python
viewer.show_residual_density("other.hkl", level=0.5)

m = viewer.render_widget.residual_density_map
print(f"peak {m.max:+.3f}, hole {m.min:+.3f}, rms {m.rms:.3f} e/Å³")
```

Called programmatically without arguments, `show_residual_density()` searches
more widely than the button does: the model file itself first, then files of
the same basename with a `.hkl`, `.fcf`, `.fco` or `.cif` extension
(`fastmolwidget.hkl_io.find_reflection_file`).

Positive density is drawn as a **green** wireframe at `+level`, negative
density as a **red** wireframe at `-level`; the default level is
**0.3 e/Å³**. Only density **within 1.5 Å of a visible atom** is shown, so
hiding hydrogens or filtering disorder parts re-contours the surface
accordingly, and no density is drawn in empty regions of the unit cell.

### Grid size

The FFT grid uses a **fixed 0.2 Å spacing derived from the unit cell alone**,
so the number of grid points never depends on how high the data resolution is
— sub-Ångström data does not make the grid explode. Reflections finer than the
grid can represent are dropped rather than aliased. Pass `grid_spacing=` to
`calculate_residual_density()` to trade detail against speed and memory.

### How it is calculated

1. Reflections are read from a SHELX `.hkl` (`HKLF 4`) file, from an
   fcf-style CIF reflection loop, or from a `_shelx_hkl_file` block embedded
   in the CIF, and merged into the reciprocal asymmetric unit with 1/σ²
   weights. **Systematically absent** reflections are discarded — their `Fc`
   is zero by symmetry, so their measured noise would enter the map amplified
   by `1/scale`.
2. *F*<sub>c</sub> is taken from the reflection file when it already contains
   phased calculated values, otherwise it is computed by direct summation with
   [gemmi](https://gemmi.readthedocs.io), including the real anomalous term
   *f′*. Atoms whose anisotropic ADP tensor is not positive definite are
   downgraded to isotropic with a `RuntimeWarning` — a negative eigenvalue
   makes the Debye-Waller factor *grow* with resolution and would otherwise
   bury the map under a huge dipole at that atom.
3. The refined overall scale factor (SHELXL's first `FVAR`) puts the two on a
   common scale, and SHELXL's isotropic `EXTI` correction is applied when it
   was refined.
4. The map uses SHELXL's own **unweighted** difference coefficients,
   `(|Fo|/OSF − |Fc|)·exp(iφc)` — the `WGHT` scheme deliberately is *not*
   applied, because SHELXL uses it only for the least-squares objective and
   not for Fourier maps.
5. An FFT over the space group yields ρ in e/Å³, and the isosurface is
   extracted with the `density_cpp` marching-cubes extension.

A leading `global_` block in a CIF is ignored; the first block with atom sites
is used.

### Where the refinement parameters come from

The refined `FVAR` / `WGHT` / `EXTI` values are looked up in this order:

1. the `.res` / `.ins` file itself, when that is what was loaded;
2. a `.res` (then `.ins`) file of the same basename next to a loaded CIF;
3. a complete SHELX `.res` block embedded in the CIF (`_shelx_res_file` or
   `_iucr_refine_instructions_details`) — which most deposited CIFs carry, so
   a CIF on its own is usually enough.

If none of these exist, a least-squares scale factor is estimated from the
data instead; this is an approximation and is documented as such in
`fastmolwidget.density`.

### Requirements and accuracy

Isosurface extraction needs the optional compiled `density_cpp` extension:

```bash
uv pip install pybind11
uv pip install -e . --no-build-isolation
```

Without it the feature degrades gracefully — the control-bar button is
disabled and `show_residual_density()` raises a clear `RuntimeError` instead
of crashing.

For the bundled `p31c` test structure the computed map gives
`max +0.32, min −0.30, rms 0.063 e/Å³` against SHELXL's reported
`+0.224 / −0.252 / 0.053`, and the underlying structure-factor calculation
reproduces the published *R*<sub>1</sub> of 0.0343. The remaining difference in
the extremes comes from SHELXL merging Friedel pairs, neglecting *f″* and
contouring on its own grid; the position and shape of the density features are
unaffected. A ~130-atom structure with 43 000 reflections (`p21c.cif`) takes
about 0.4 s.

### Using the map without Qt

`fastmolwidget.density` and `fastmolwidget.hkl_io` import no Qt at all, so the
map can be computed in headless scripts:

```python
from fastmolwidget import calculate_residual_density

m = calculate_residual_density("structure.res")   # reflections found automatically
print(m.array.shape, m.rms)          # raw numpy grid, one unit cell
vertices, edges = m.isosurface(0.3)   # Cartesian wireframe
```

## Running the Examples

To run the provided examples, you can use the following commands:

```bash
# 2D Viewer example
python -m fastmolwidget.examples.viewer_2d_example

# 3D Viewer example
python -m fastmolwidget.examples.viewer_3d_example

# Generic 3D Widget example
python -m fastmolwidget.examples.generic_3d_widget_example
```

## Embedding in HTML reports

The package ships a dependency-free JavaScript port of the 2D renderer
(`fastmolwidget/web/js`, see its `README.md`). Structure **parsing** stays in
Python; growing, packing and rendering run in the browser on a `<canvas>` — no
Qt, no build step, and no network access at runtime.

`fastmolwidget.web` imports no Qt at all, so it also works in a headless report
generator.

### Drop it into your own template

`bundle_js()` returns the whole renderer as a single classic-`<script>` string
and `structure_json()` the structure. Both are safe to paste inside a
`<script>` element; in Jinja2 inject them with `| safe`:

```python
from fastmolwidget.web import bundle_js, structure_json

html = template.render(
    fastmolwidget_js=bundle_js(),
    structure_json=structure_json('structure.cif'),
)
```

```jinja
<div id="mol" style="height:400px"></div>
<script>
    var mol = {{ structure_json | safe }};
    {{ fastmolwidget_js | safe }}
</script>
<script>
    var viewer = Fastmolwidget.createViewer(
        document.getElementById('mol'), mol, {controls: false, grow: true});
</script>
```

`createViewer(container, structure, options)` fills the container with a
HiDPI-aware canvas and keeps it sized to the element. Options: `controls`,
`grow`, `pack`, `adps`, `labels`, `hydrogens`, `bondWidth`, `bondColor`,
`background`, `bestView`. The returned object is a `MoleculeViewer2D`; its
`.widget` exposes the same API as the Python `MoleculeWidget` (`showAdps()`,
`setBondColor()`, `alignBestView()`, `saveImage()`, …) and emits `atomClicked`,
`bondClicked` and `partsChanged` events.

`controls` accepts `true`/`false` to show/hide the whole bar, or an object to
selectively show/hide individual elements (unspecified keys default to
visible):

```js
Fastmolwidget.createViewer(container, mol, {
  controls: { pack: false, bondWidth: false, saveImage: false },
});
```

Recognised keys: `grow`, `pack`, `adps`, `labels`, `hydrogens`, `partFilter`,
`bondWidth`, `bestView`, `resetView`, `saveImage`.

`window.Fastmolwidget` also exposes `MoleculeViewer2D`, `MoleculeWidget2D`,
`SDM`, `createPartFilter` and `version`.

### Or generate a finished page

```python
from fastmolwidget.web import render_html, write_html

write_html('structure.cif', 'structure.html', controls=True, grow=True)
html = render_html('structure.cif', controls=False, height='400px')
# Selectively hide individual control-bar elements:
html = render_html('structure.cif', controls={'pack': False, 'bondWidth': False})
```

The result is fully self-contained (renderer and structure inlined), so it
works from `file://`, inside an e-mail attachment, or in a Qt app via
`QWebEngineView.setHtml(render_html('structure.cif'))`.

To try it out, serve a structure with the built-in demo server:

```bash
python -m fastmolwidget.web_demo_server --cif tests/test-data/p21c.cif
```
