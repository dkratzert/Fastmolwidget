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
- **Multiple file formats**: CIF, SHELX `.res`/`.ins`, and plain XYZ. More to come...
- **Embeddable** — both `MoleculeWidget` (2D) and `MoleculeWidget3D` (3D) are plain `QWidget` subclasses; drop either into any layout
- **Ready-to-use viewers** — `MoleculeViewerWidget` (2D) and `MoleculeViewer3DWidget` (3D) bundle the renderer with a full control bar
- **Qt Quick / QML support** — `MoleculeQuick3D` is a `QQuickRhiItem` (Qt ≥ 6.7) that renders the same 3D scene inside a QML scene graph
- **Common protocol** — `MoleculeWidgetProtocol` lets you write code that works with either widget interchangeably

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

# add 3D OpenGL support
uv add "fastmolwidget[pyside6,gl3d]"

# Qt Quick / QML support (uses the same pyside6 extra; QtQuick is included)
uv add "fastmolwidget[quickview]"
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

### Embedding the 3D widget in your own layout

```python
from fastmolwidget import MoleculeWidget3D

mol = MoleculeWidget3D(parent=self)
mol.open_molecule(atoms, cell=cell, adps=adps)
layout.addWidget(mol)
```

### Qt Quick / QML integration

```python
import sys
from qtpy.QtGui import QGuiApplication
from qtpy.QtQml import QQmlApplicationEngine, qmlRegisterType
from fastmolwidget import MoleculeQuick3D, MoleculeLoader, setup_opengl_backend

# Must be called before creating QGuiApplication (especially on macOS).
setup_opengl_backend()

app = QGuiApplication(sys.argv)
engine = QQmlApplicationEngine()
qmlRegisterType(MoleculeQuick3D, "MolWidget", 1, 0, "MoleculeQuick3D")
engine.load("main.qml")
sys.exit(app.exec())
```

```qml
// main.qml
import QtQuick 2.15
import MolWidget 1.0

MoleculeQuick3D {
    width: 800; height: 600
    showAdps: true
    showLabels: true

    Repeater {
        model: parent.labelPositions
        delegate: Text {
            required property var modelData
            x: modelData.x + 4;  y: modelData.y - 4
            text: modelData.text
            color: modelData.kind === "hover_atom" ? "#1a6ecc" : "#6b3200"
            font.pixelSize: parent.parent.labelFontSize
            font.bold: modelData.kind !== "atom"
        }
    }
}
```



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

## Control Bar Options

### `MoleculeViewerWidget` (2D) and `MoleculeViewer3DWidget` (3D)

Both viewers expose the same control bar:

| Control               | Default | Description                                                                                    |
|-----------------------|---------|------------------------------------------------------------------------------------------------|
| Grow                  | ✗       | Expand the asymmetric unit to complete molecules                                               |
| Pack Unit Cell        | ✗       | Generate all symmetry-equivalent positions within one unit cell (mutually exclusive with Grow) |
| Show ADP              | ✓       | Toggle ORTEP ellipsoid / isotropic sphere rendering                                            |
| Show Labels           | ✗       | Toggle non-hydrogen atom labels                                                                |
| Round Bonds           | ✓       | Switch between round cylinder and flat bond drawing                                            |
| Show Hydrogens        | ✓       | Show or hide hydrogen atoms and their bonds                                                    |
| Bond Width            | 3       | Stroke width / cylinder radius for bonds (1–15)                                                |
| Bond Color            | —       | Opens a colour picker to change the default bond colour                                        |
| Reset Rotation Center | —       | Restores the rotation pivot to the molecule's geometric centre (3D only)                       |

## API Overview

### `MoleculeViewer3DWidget(parent=None)`

A self-contained 3D viewer combining `MoleculeWidget3D` with the control bar.

- `load_file(path)` — load a structure file (format auto-detected from extension: `.cif`, `.res`, `.ins`, `.xyz`)
- `set_bond_color(color)` — set the default color for non-selected bonds
- `render_widget` — read-only property exposing the underlying `MoleculeWidget3D`

### `MoleculeWidget3D(parent=None)`

Hardware-accelerated OpenGL renderer. A `QOpenGLWidget` (Qt ≥ 6) or `QWidget` subclass that can be dropped into any layout.

**Rendering technique**

| Primitive      | Technique                                                                                                                    |
|----------------|------------------------------------------------------------------------------------------------------------------------------|
| Atoms          | Billboard sphere impostors — each atom is a quad; the fragment shader ray-casts a sphere and writes corrected depth values   |
| ADP ellipsoids | Impostor quads — the fragment shader ray-casts an exact ellipsoid using the inverse U_cart tensor passed as a `mat3` uniform |
| Bonds          | Tessellated cylinder mesh (8-segment, 4-segment for angular style) built on the CPU and uploaded as a single VBO             |
| Labels         | `QPainter` overlay drawn after the OpenGL pass                                                                               |

All GLSL shaders target `#version 120` (OpenGL 2.1 / GLSL 1.20) for maximum hardware compatibility.

#### Qt Signals

| Signal        | Signature                    | Emitted when               |
|---------------|------------------------------|----------------------------|
| `atomClicked` | `(label: str)`               | The user clicks on an atom |
| `bondClicked` | `(label1: str, label2: str)` | The user clicks on a bond  |

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
- **`set_bond_width(width: int)`** — set cylinder radius scale (1–15)
- **`set_bond_color(color)`** — set the default color for non-selected bonds; accepts `QColor`, hex string, or an RGB tuple
- **`set_labels_visible(visible: bool)`** — alias for `show_labels`
- **`setLabelFont(font_size: int)`** — set label font pixel size
- **`set_background_color(color: QColor)`** — change background colour
- **`reset_view()`** — reset zoom, rotation, and pan to defaults
- **`reset_rotation_center()`** — restore the rotation pivot to the molecule's geometric center (undoes a middle-click recentring)
- **`save_image(filename, image_scale=1.5)`** — render the current view to an image file

#### Example — feeding atom data directly to `MoleculeWidget3D`

```python
from fastmolwidget import MoleculeWidget3D
from fastmolwidget.sdm import Atomtuple

mol = MoleculeWidget3D(parent=self)

atoms = [
    Atomtuple(label="C1", type="C", x=0.0, y=0.0, z=0.0, part=0),
    Atomtuple(label="O1", type="O", x=1.22, y=0.0, z=0.0, part=0),
    Atomtuple(label="H1", type="H", x=-0.5, y=0.94, z=0.0, part=0),
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

#### Example — feeding atom data directly to `MoleculeWidget` (2D)

```python
from fastmolwidget import MoleculeWidget
from fastmolwidget.sdm import Atomtuple

mol = MoleculeWidget(parent=self)

atoms = [
    Atomtuple(label="C1", type="C", x=0.0, y=0.0, z=0.0, part=0),
    Atomtuple(label="O1", type="O", x=1.22, y=0.0, z=0.0, part=0),
    Atomtuple(label="H1", type="H", x=-0.5, y=0.94, z=0.0, part=0),
]

# ADP tensors: {atom_label: (U11, U22, U33, U23, U13, U12)}
adps = {
    "C1": (0.02, 0.02, 0.02, 0.0, 0.0, 0.0),
    "O1": (0.03, 0.03, 0.03, 0.0, 0.0, 0.0),
}

cell = (5.0, 5.0, 5.0, 90.0, 90.0, 90.0)  # optional

mol.open_molecule(atoms=atoms, cell=cell, adps=adps)
mol.atomClicked.connect(lambda label: print(f"Selected: {label}"))

layout.addWidget(mol)
```

## Advanced API

### `MoleculeWidgetProtocol`

The core rendering interface is defined by `MoleculeWidgetProtocol`. Both `MoleculeWidget` (2D) and `MoleculeWidget3D` (3D) satisfy this protocol, making them drop-in replacements for each other.

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
