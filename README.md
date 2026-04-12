# Fastmolwidget

**A PyQt/PySide6 widget to display crystal structures**

Fastmolwidget is a lightweight, embeddable Qt widget that renders molecular and crystal structures in 2D projection with interactive mouse controls. 
It supports anisotropic displacement parameter (ADP) ellipsoids, ball-and-stick diagrams, and plain sphere representations — 
all drawn with a pure-Python QPainter backend (no OpenGL required).

## Screenshot

![Fastmolwidget showing an ORTEP-style crystal structure](https://github.com/user-attachments/assets/7946ef73-7e74-475d-a5c6-2012243b9f77)

*View of a crystal structure from a CIF file with ADP ellipsoids. The control bar allows toggling display options interactively.*

## Features

- **ADP ellipsoids** at the 50 % probability level, rendered from anisotropic displacement parameters
- **Ball-and-stick** and **isotropic sphere** representations as fallbacks or when speed is more important than detail
- **Interactive mouse controls**: rotate (left-drag), zoom (right-drag), pan (middle-drag), scroll wheel to resize labels
- **Atom and bond selection**: single click or Ctrl+click for multi-selection; emits `atomClicked` / `bondClicked` Qt signals
- **Hydrogen visibility toggle**
- **Atom label display toggle** with adjustable font size
- **Bond width** adjustment via spin box
- **Multiple file formats**: CIF, SHELX `.res`/`.ins`, and plain XYZ. More to come...
- **Embeddable** — `MoleculeWidget` is a plain `QWidget` subclass; drop it into any layout
- **Ready-to-use** `MoleculeViewerWidget` bundles the renderer with a full control bar

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

**Requirements**: Python ≥ 3.14, PyQt6, NumPy, gemmi, shelxfile, qtpy.

## Quick Start

### Standalone viewer

```python
from qtpy.QtWidgets import QApplication
from fastmolwidget import MoleculeViewerWidget

app = QApplication([])
viewer = MoleculeViewerWidget()
viewer.load_file("structure.cif")
viewer.show()
app.exec()
```

### Embedding in your own layout

```python
from fastmolwidget import MoleculeWidget, MoleculeLoader

mol = MoleculeWidget(parent=self)
loader = MoleculeLoader(mol)
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

## Control Bar Options (`MoleculeViewerWidget`)

| Control | Default | Description |
|---------|---------|-------------|
| Show ADP | ✓ | Toggle ORTEP ellipsoid / isotropic sphere rendering |
| Show Labels | ✗ | Toggle non-hydrogen atom labels |
| Round Bonds | ✓ | Switch between 3D-shaded and flat bond drawing |
| Show Hydrogens | ✓ | Show or hide hydrogen atoms and their bonds |
| Bond Width | 3 | Stroke width for bonds (1–15 px) |

## API Overview

### `MoleculeViewerWidget(parent=None)`

A self-contained widget combining `MoleculeWidget` with the control bar described above.

- `load_file(path)` — load a structure file (format auto-detected from extension)
- `render_widget` — read-only property exposing the underlying `MoleculeWidget`

### `MoleculeWidget(parent=None)`

The low-level renderer widget. It is a plain `QWidget` subclass that you can drop into any layout.
Provide atom data directly via `open_molecule()` instead of loading a file through `MoleculeLoader`.

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

#### Example — feeding atom data directly

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

### `MoleculeLoader(widget)`

Format-aware loader that populates a `MoleculeWidget`.

- `load_file(path, keep_view=False)` — parse file and call `open_molecule` / `grow_molecule`

## License

BSD 2-Clause License — see [LICENSE](LICENSE) for details.

© 2026 Daniel Kratzert
