"""Shared renderer mixin and data types for 2-D perspective molecule drawing.

:class:`MoleculeRendererMixin` holds **all** state and drawing logic without
depending on any specific Qt base class.  Concrete widgets mix it with the
appropriate Qt base:

* :class:`~fastmolwidget.molecule2D.MoleculeWidget`  ← ``QWidget``
* :class:`~fastmolwidget.molecule_quick.MoleculeQuickItem` ← ``QQuickPaintedItem``

:class:`Atom` and :class:`RenderItem` are also defined here and re-exported by
:mod:`fastmolwidget.molecule2D` for backwards compatibility.

Minimal contract that the concrete class must satisfy
------------------------------------------------------
* Inherit ``MoleculeRendererMixin`` **before** the Qt base class so that
  ``super()`` chains work correctly (cooperative multiple inheritance).
* Call ``QWidget.__init__(self, parent)`` (or equivalent) **then**
  ``self._init_renderer()`` from the concrete ``__init__``.
* Declare the three Qt signals as class-level attributes::

      atomClicked  = Signal(str)
      bondClicked  = Signal(str, str)
      partsChanged = Signal(object)

* The concrete class is responsible for creating ``self._painter`` and calling
  ``self.draw()`` at repaint time (e.g. inside ``paintEvent`` / ``paint``).
* Implement :meth:`save_image` using the appropriate Qt image-capture API.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt, cos, sin, dist, radians, atan2, degrees, pi

import numpy as np
from qtpy import QtCore, QtGui
from qtpy.QtCore import Qt, QRectF
from qtpy.QtGui import (
    QPainter, QPen, QBrush, QColor, QMouseEvent,
    QWheelEvent, QRadialGradient, QLinearGradient, QTransform,
)

from fastmolwidget.atoms import get_radius_from_element, element2color
from fastmolwidget.sdm import Atomtuple


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def calc_volume(
    a: float, b: float, c: float,
    alpha: float, beta: float, gamma: float,
) -> float:
    """Return the unit-cell volume from cell parameters."""
    ca, cb, cg = cos(radians(alpha)), cos(radians(beta)), cos(radians(gamma))
    return a * b * c * sqrt(1 + 2 * ca * cb * cg - ca ** 2 - cb ** 2 - cg ** 2)


#: Half-edge of the NPD placeholder cube, as a fraction of ``atoms_size``.
NPD_CUBE_HALF_FACTOR = 0.4

#: Bounding-circle radius of the NPD cube, as a fraction of ``atoms_size``
#: (half-edge times sqrt(3), the cube's body diagonal half-length).
NPD_CUBE_BOUND_FACTOR = NPD_CUBE_HALF_FACTOR * 1.7320508075688772

# Light direction in view space (x right, y down, z away from the viewer):
# upper-left and in front of the molecule.
_NPD_LIGHT = np.array([-0.3, -0.5, -1.0])
_NPD_LIGHT = _NPD_LIGHT / np.linalg.norm(_NPD_LIGHT)

# The six faces of a unit cube as indices into the corner list built by
# :func:`npd_cube_faces`, each wound so that the outward normal is
# ``(p1 - p0) x (p2 - p1)``.  Corner index bits are (i, j, k) -> sign of
# (u, v, w), see below.
_NPD_CUBE_FACES: tuple[tuple[int, int, int, int], ...] = (
    (4, 6, 7, 5),  # +u
    (0, 1, 3, 2),  # -u
    (2, 3, 7, 6),  # +v
    (0, 4, 5, 1),  # -v
    (1, 5, 7, 3),  # +w
    (0, 2, 6, 4),  # -w
)


def npd_cube_faces(
    rotation: np.ndarray, half: float,
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """Return the projected faces of the NPD placeholder cube.

    The cube is axis-aligned in the *molecular* Cartesian frame and is
    brought into view space with *rotation* (the renderer's accumulated view
    rotation), so it turns together with the rest of the structure — the same
    convention the OpenGL renderer uses.

    :param rotation: 3x3 view rotation matrix (``cumulative_R``).
    :param half: Half-edge length of the cube in screen pixels.
    :returns: A list of ``(corners, mean_z, normal)`` tuples, one per face,
        sorted **back-to-front** (descending depth, since smaller ``z`` is
        nearer the viewer here).  ``corners`` is a ``(4, 2)`` array of
        screen-space offsets relative to the atom centre and ``normal`` is the
        outward unit normal in view space.
    """
    R = np.asarray(rotation, dtype=np.float64)
    u, v, w = R[:, 0] * half, R[:, 1] * half, R[:, 2] * half
    # Corner index is i*4 + j*2 + k with i/j/k selecting the sign of u/v/w.
    corners = np.array([
        (si * u) + (sj * v) + (sk * w)
        for si in (-1.0, 1.0) for sj in (-1.0, 1.0) for sk in (-1.0, 1.0)
    ])
    faces: list[tuple[np.ndarray, float, np.ndarray]] = []
    for idx in _NPD_CUBE_FACES:
        pts = corners[list(idx)]
        normal = np.cross(pts[1] - pts[0], pts[2] - pts[1])
        norm = float(np.linalg.norm(normal))
        if norm > 1e-12:
            normal = normal / norm
        faces.append((pts[:, :2], float(pts[:, 2].mean()), normal))
    faces.sort(key=lambda f: f[1], reverse=True)
    return faces


def npd_face_shade(normal: np.ndarray) -> float:
    """Return the Lambert brightness factor for a cube face *normal*.

    ``1.0`` leaves the base colour unchanged; larger values brighten it.
    """
    diffuse = max(0.0, float(np.dot(normal, _NPD_LIGHT)))
    return min(1.60, max(0.45, 0.60 + 0.85 * diffuse))


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RenderItem:
    """A single renderable element (atom or bond) in the z-ordered draw list.

    :param is_bond: ``True`` for a bond, ``False`` for an atom.
    :param z_order: Depth value used to sort back-to-front (painter's algorithm).
    :param atom1: First (or only) atom involved.
    :param atom2: Second atom of a bond, ``None`` for an atom item.
    """

    is_bond: bool
    z_order: float = 0.0
    atom1: Atom = None          # type: ignore[assignment]
    atom2: Atom | None = None


class Atom:
    """Internal representation of a single atom for 2-D rendering."""

    __slots__ = [
        'coordinate', 'name', 'part', 'symmgen', 'radius',
        'screenx', 'screeny', 'type_', 'u_cart', 'color',
        'color_light', 'color_dark', 'u_iso', 'z',
        'u_eigvals', 'u_eigvecs', 'u_inv',
        'sphere_brush', 'adp_brush', 'adp_valid',
    ]

    def __init__(
        self,
        x: float, y: float, z: float,
        name: str, type_: str, part: int,
    ) -> None:
        self.coordinate = np.array([x, y, z], dtype=np.float32)
        self.adp_valid = True
        self.z = z
        self.name = name
        self.part = part
        self.symmgen = False
        self.type_ = type_
        self.screenx = 0.0
        self.screeny = 0.0
        self.radius = get_radius_from_element(type_)
        self.u_cart = None
        self.color = QColor(element2color.get(self.type_, '#000000'))
        self.color_light = self.color.lighter(160)
        self.color_dark = self.color.darker(180)
        self.u_iso = None
        self.u_eigvals = None
        self.u_eigvecs = None
        self.u_inv = None

        sg = QRadialGradient(0.35, 0.35, 1.0)
        sg.setCoordinateMode(QRadialGradient.CoordinateMode.ObjectBoundingMode)
        sg.setColorAt(0.0, self.color_light)
        sg.setColorAt(0.4, self.color)
        sg.setColorAt(1.0, self.color_dark)
        self.sphere_brush = QBrush(sg)

        ag = QRadialGradient(0.0, 0.0, 1.0)
        ag.setColorAt(0.0, self.color_light)
        ag.setColorAt(0.4, self.color)
        ag.setColorAt(1.0, self.color_dark)
        self.adp_brush = QBrush(ag)

    def __repr__(self) -> str:
        return str((self.name, self.type_, self.coordinate))


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class MoleculeRendererMixin:
    """Pure-Python mixin providing all 2-D molecule-rendering logic.

    Contains no Qt base-class dependencies: every Qt call goes through
    ``self.width()``, ``self.height()``, ``self.update()``, and
    ``self._painter`` — all provided by the concrete subclass.
    """

    _AUTO_ZOOM_PADDING = 1.1
    # Bounding rect for a unit circle, shared by _draw_principal_arcs
    _UNIT_RECT = QRectF(-1.0, -1.0, 2.0, 2.0)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_renderer(self) -> None:
        """Initialise all renderer state.

        Call this **once** from the concrete ``__init__``, **after** the
        Qt base-class constructor has run (so that ``self.update()`` and
        the signal infrastructure are already in place).
        """
        self._astar: float | None = None
        self._bstar: float | None = None
        self._cstar: float | None = None
        self._amatrix: np.ndarray | None = None
        self._cell: tuple[float, float, float, float, float, float] | None = None
        self._is_packed = False

        self.zoom = 1.0
        self.fontsize = 13
        self.bond_width = 3
        self.labels = True
        self._show_adps = True
        self.show_hydrogens_flag = True

        # Part filter
        self.available_parts: frozenset[int] = frozenset()
        self._visible_parts: set[int] | None = None

        # Selection
        self.selected_atoms: set[str] = set()
        self.selected_bonds: set[tuple[str, ...]] = set()

        # ORTEP scaling factor for 50 % probability ellipsoids
        self.adp_scale = 1.5382

        self.molecule_center = np.array([0, 0, 0], dtype=np.float32)
        self.molecule_radius = 10

        # Last / press position for drag tracking
        self._lastPos = QtCore.QPointF(0.0, 0.0)
        self._pressPos = QtCore.QPointF(0.0, 0.0)

        # Active painter (set by the concrete paint method before draw())
        self._painter: QPainter | None = None

        self.x_angle = 0.0
        self.y_angle = 0.0
        self.scale = 150.0
        self.cx_global = 0.0
        self.cy_global = 0.0

        # Cumulative rotation matrix (preserves orientation during grow)
        self.cumulative_R = np.eye(3, dtype=np.float32)

        # Background colour (used by QQuickPaintedItem's paint() fill)
        self._bg_color = QColor(Qt.GlobalColor.white)

        # Bond colours
        self.bond_color = QColor('#555555')
        self.fallback_pen_color = QColor(Qt.GlobalColor.black)
        self.adp_pen_color = QColor(0, 0, 0, 255)
        self.bond_grad_dark = QColor(60, 60, 60)
        self.bond_grad_light = QColor(140, 140, 140)
        self.bond_grad_shadow = QColor(10, 10, 10)
        self._rebuild_bond_brush()

        # Atom / object lists
        self.atoms: list[Atom] = []
        self.connections: tuple = ()
        self.objects: list[RenderItem] = []
        self.screen_center: list[float] = [0.0, 0.0]

        # Vectorised numpy arrays (filled in _load_molecule)
        self._coords_array = np.empty((0, 3))
        self._ucart_array = np.empty((0, 3, 3))
        self._has_adp = np.empty(0, dtype=bool)
        self._eigenvalues_array = np.empty((0, 3))
        self._eigenvectors_array = np.empty((0, 3, 3))
        self._u_inv_array = np.empty((0, 3, 3))

        # Hover state
        self.hovered_atom: str | None = None
        self.hovered_bond: tuple[str, str] | None = None
        self._hovered_bond_distance: float | None = None
        self._hover_cursor: QtCore.QPointF | None = None

        # Per-frame cache filled in draw()
        self._cached_adp_line_width: float = 1.0

    # ------------------------------------------------------------------
    # Public API — display settings
    # ------------------------------------------------------------------

    def set_bond_width(self, width: int) -> None:
        """Set the width of the bonds."""
        self.bond_width = width
        self.update()  # type: ignore[misc]

    def set_bond_color(
        self,
        color: QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set the default colour used for all non-selected bonds."""
        if isinstance(color, QColor):
            self.bond_color = color
        elif isinstance(color, str):
            self.bond_color = QColor(color)
        elif isinstance(color, tuple) and len(color) == 3:
            r, g, b = color
            if all(isinstance(c, (int, float)) for c in (r, g, b)):
                if all(c <= 1.0 for c in (r, g, b)):
                    self.bond_color = QColor(int(r * 255), int(g * 255), int(b * 255))
                else:
                    self.bond_color = QColor(int(r), int(g), int(b))
            else:
                raise ValueError("RGB tuple components must be numeric.")
        else:
            raise ValueError(
                "Bond color must be a QColor, hex string, or RGB tuple (0..1 or 0..255)."
            )
        self._rebuild_bond_brush()
        self.update()  # type: ignore[misc]

    def set_labels_visible(self, visible: bool) -> None:
        """Toggle visibility of atom labels."""
        self.labels = visible
        self.update()  # type: ignore[misc]

    def show_hydrogens(self, value: bool) -> None:
        """Toggle display of hydrogen atoms and their bonds."""
        self.show_hydrogens_flag = value
        self.update()  # type: ignore[misc]

    def set_visible_parts(self, parts: set[int] | None) -> None:
        """Set which disorder parts are rendered (``None`` = all)."""
        self._visible_parts = parts
        self.update()  # type: ignore[misc]

    def reset_view(self) -> None:
        """Reset zoom and rotation to defaults."""
        self.zoom = self._auto_zoom()
        self.x_angle = 180.0
        self.y_angle = 180.0
        self.z_angle = 0.0  # type: ignore[attr-defined]
        self.x_shift_screen = 0  # type: ignore[attr-defined]
        self.y_shift_screen = 0  # type: ignore[attr-defined]
        self.cumulative_R = np.eye(3, dtype=np.float32)
        self.update()  # type: ignore[misc]

    def setLabelFont(self, font_size: int) -> None:
        """Set the pixel size used for atom labels and schedule a repaint."""
        if font_size < 0:
            font_size = 1
        self.fontsize = font_size
        self.update()  # type: ignore[misc]

    def clear(self) -> None:
        """Remove all atoms and bonds from the widget."""
        self.open_molecule(atoms=[])

    def show_labels(self, value: bool) -> None:
        """Toggle the display of non-hydrogen atom labels."""
        self.labels = value
        self.update()  # type: ignore[misc]

    def show_adps(self, value: bool) -> None:
        """Toggle the display of ADP ellipsoids / isotropic spheres."""
        self._show_adps = value
        self.update()  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Molecule loading
    # ------------------------------------------------------------------

    def open_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None = None,
        keep_view: bool = False,
    ) -> None:
        """Load a new molecule and reset the view (unless *keep_view* is set)."""
        self._is_packed = False
        self._load_molecule(atoms, cell, keep_view=keep_view)

    def grow_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        """Update the molecule while preserving the current view."""
        self._load_molecule(atoms, cell, keep_view=True)

    def _load_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None = None,
        keep_view: bool = False,
    ) -> None:
        self._cell = cell
        if self._cell is not None:
            self.calc_amatrix()

        self.atoms.clear()
        self.make_adps(atoms)
        self.connections = self.get_conntable_from_atoms()

        self.available_parts = frozenset(a.part for a in self.atoms)
        self._visible_parts = None
        self.partsChanged.emit(self.available_parts)  # type: ignore[attr-defined]

        if not keep_view:
            self.get_center_and_radius()
            self.cumulative_R = np.eye(3, dtype=np.float32)
            self.selected_atoms.clear()
            self.selected_bonds.clear()

        self.objects.clear()
        for n1, n2 in self.connections:
            at1 = self.atoms[n1]
            at2 = self.atoms[n2]
            self.objects.append(RenderItem(is_bond=True, atom1=at1, atom2=at2))

        for atom in self.atoms:
            if atom.type_ in ('H', 'D'):
                atom.u_iso = 0.01
            self.objects.append(RenderItem(is_bond=False, atom1=atom))

        # Build numpy arrays for vectorised rotation
        self._coords_array = np.array([at.coordinate for at in self.atoms])
        self._ucart_array = np.zeros((len(self.atoms), 3, 3))
        self._has_adp = np.zeros(len(self.atoms), dtype=bool)
        self._eigenvalues_array = np.zeros((len(self.atoms), 3))
        self._eigenvectors_array = np.zeros((len(self.atoms), 3, 3))
        self._u_inv_array = np.zeros((len(self.atoms), 3, 3))

        if keep_view and not np.allclose(self.cumulative_R, np.eye(3)):
            self._coords_array = (
                np.dot(self._coords_array - self.molecule_center, self.cumulative_R.T)
                + self.molecule_center
            )

        for i, at in enumerate(self.atoms):
            if keep_view and not np.allclose(self.cumulative_R, np.eye(3)):
                at.coordinate = self._coords_array[i]
            at.z = at.coordinate[2]
            if at.u_cart is not None:
                if keep_view and not np.allclose(self.cumulative_R, np.eye(3)):
                    at.u_cart = np.matmul(
                        self.cumulative_R,
                        np.matmul(at.u_cart, self.cumulative_R.T),
                    )
                try:
                    evals, evecs = np.linalg.eigh(at.u_cart)
                    at.adp_valid = bool(np.all(evals > 0))
                    u_invers = np.linalg.inv(at.u_cart)
                    self._ucart_array[i] = at.u_cart
                    self._eigenvalues_array[i] = evals
                    self._eigenvectors_array[i] = evecs
                    self._u_inv_array[i] = u_invers
                    self._has_adp[i] = True
                    at.u_eigvals = evals
                    at.u_eigvecs = evecs
                    at.u_inv = u_invers
                except np.linalg.LinAlgError:
                    at.adp_valid = False
                    at.u_cart = None
                    at.u_iso = None

        if not keep_view:
            self.zoom = self._auto_zoom()

        self.update()  # type: ignore[misc]

    def calc_amatrix(self) -> None:
        """Compute the orthogonalisation matrix and reciprocal-lattice lengths."""
        a, b, c, alpha, beta, gamma = self._cell  # type: ignore[misc]
        V = calc_volume(a, b, c, alpha, beta, gamma)
        self._astar = (b * c * sin(radians(alpha))) / V
        self._bstar = (c * a * sin(radians(beta))) / V
        self._cstar = (a * b * sin(radians(gamma))) / V
        self._amatrix = np.array([
            [a, b * cos(radians(gamma)), c * cos(radians(beta))],
            [
                0,
                b * sin(radians(gamma)),
                c * (
                    cos(radians(alpha))
                    - cos(radians(beta)) * cos(radians(gamma))
                ) / sin(radians(gamma)),
            ],
            [0, 0, V / (a * b * sin(radians(gamma)))],
        ], dtype=float)

    def make_adps(self, atoms: list[Atomtuple]) -> None:
        """Convert ``Atomtuple`` list to internal :class:`Atom` objects."""
        self.atoms.clear()
        name_counts: dict[str, int] = {}

        for at in atoms:
            base_name = at.label
            count = name_counts.get(base_name, 0)
            internal_name = base_name if count == 0 else f"{base_name}>>{count}"
            name_counts[base_name] = count + 1

            a = Atom(at.x, at.y, at.z, internal_name, at.type, at.part)
            symm_matrix = getattr(at, 'symm_matrix', None)
            if symm_matrix is not None:
                symm_np = np.array(symm_matrix, dtype=float)
                a.symmgen = not np.allclose(symm_np, np.eye(3))
            adp_vals = getattr(at, 'adp', None)
            if adp_vals is not None and self._cell:
                try:
                    sm = np.array(symm_matrix, dtype=float) if symm_matrix is not None else None
                    a.u_cart = self._uij_to_cart(adp_vals, sm)
                    a.u_iso = np.trace(a.u_cart) / 3.0
                except Exception:
                    a.u_cart = None
                    a.u_iso = None
            self.atoms.append(a)

    # ------------------------------------------------------------------
    # View control
    # ------------------------------------------------------------------

    def reset_rotation_center(self) -> None:
        """Reset the rotation pivot to the geometric centre of the molecule."""
        self.get_center_and_radius()
        self.update()  # type: ignore[misc]

    def _align_to_reciprocal_axis(self, axis_index: int) -> None:
        """Align the view so that real-space axis *axis_index* (0=a,1=b,2=c)
        points towards the viewer."""
        if self._amatrix is None or self._cell is None:
            return
        direct_vec = self._amatrix[:, axis_index].copy()
        direct_vec = direct_vec / np.linalg.norm(direct_vec)
        z_axis = direct_vec.astype(np.float32)
        up_candidate = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(z_axis, up_candidate)) > 0.99:
            up_candidate = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        x_axis = np.cross(up_candidate, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        target_R = np.array([x_axis, y_axis, z_axis], dtype=np.float32)
        delta_R = target_R @ np.linalg.inv(self.cumulative_R)
        if self.atoms:
            self._coords_array = (
                np.dot(self._coords_array - self.molecule_center, delta_R.T)
                + self.molecule_center
            )
            if np.any(self._has_adp):
                self._ucart_array = np.matmul(delta_R, np.matmul(self._ucart_array, delta_R.T))
                self._eigenvectors_array = np.matmul(delta_R, self._eigenvectors_array)
                self._u_inv_array = np.matmul(delta_R, np.matmul(self._u_inv_array, delta_R.T))
            for i, at in enumerate(self.atoms):
                at.coordinate = self._coords_array[i]
                at.z = at.coordinate[2]
                if self._has_adp[i]:
                    at.u_cart = self._ucart_array[i]
                    at.u_eigvecs = self._eigenvectors_array[i]
                    at.u_inv = self._u_inv_array[i]
        self.cumulative_R = target_R
        self.update()  # type: ignore[misc]

    def align_best_view(self) -> None:
        """Rotate the structure to maximise atom visibility (PCA)."""
        if not self.atoms or len(self._coords_array) < 2:
            return
        if self.show_hydrogens_flag:
            visible = np.arange(len(self.atoms))
        else:
            visible = np.array(
                [i for i, at in enumerate(self.atoms) if at.type_ not in ('H', 'D')],
                dtype=np.intp,
            )
        if len(visible) < 2:
            return
        coords = self._coords_array[visible].astype(np.float64)
        centred = coords - coords.mean(axis=0)
        cov = centred.T @ centred
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]
        x_axis = evecs[:, 0].astype(np.float32)
        y_axis = evecs[:, 1].astype(np.float32)
        z_axis = evecs[:, 2].astype(np.float32)
        if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
            z_axis = -z_axis
        target_R = np.array([x_axis, y_axis, z_axis], dtype=np.float32)
        delta_R = target_R @ np.linalg.inv(self.cumulative_R)
        self._coords_array = (
            np.dot(self._coords_array - self.molecule_center, delta_R.T)
            + self.molecule_center
        )
        if np.any(self._has_adp):
            self._ucart_array = np.matmul(delta_R, np.matmul(self._ucart_array, delta_R.T))
            self._eigenvectors_array = np.matmul(delta_R, self._eigenvectors_array)
            self._u_inv_array = np.matmul(delta_R, np.matmul(self._u_inv_array, delta_R.T))
        for i, at in enumerate(self.atoms):
            at.coordinate = self._coords_array[i]
            at.z = at.coordinate[2]
            if self._has_adp[i]:
                at.u_cart = self._ucart_array[i]
                at.u_eigvecs = self._eigenvectors_array[i]
                at.u_inv = self._u_inv_array[i]
        self.cumulative_R = target_R
        self.update()  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @property
    def atoms_size(self) -> float:
        """Atom circle diameter in screen pixels."""
        return self.zoom * 70

    def _auto_zoom(self) -> float:
        """Compute the zoom level that fits the molecule into the viewport."""
        w = self.width()  # type: ignore[attr-defined]
        h = self.height()  # type: ignore[attr-defined]
        r = self.molecule_radius
        if w <= 0 or h <= 0 or r <= 0:
            return self._AUTO_ZOOM_PADDING / 100
        return self._AUTO_ZOOM_PADDING * min(w, h) / 2 / r / 100

    def _adp_intersection_line_width(self) -> float:
        return max(1.0, min(6.0, self.zoom * 3.0))

    def _on_resize(
        self,
        old_w: float, old_h: float,
        new_w: float, new_h: float,
    ) -> None:
        """Scale zoom proportionally when the viewport is resized."""
        if old_w > 0 and old_h > 0:
            old_min = min(old_w, old_h)
            new_min = min(new_w, new_h)
            if old_min > 0:
                self.zoom *= new_min / old_min

    def _on_leave(self) -> None:
        """Clear hover state when the cursor leaves the render area."""
        changed = False
        if self.hovered_atom is not None:
            self.hovered_atom = None
            changed = True
        if self.hovered_bond is not None:
            self.hovered_bond = None
            self._hovered_bond_distance = None
            self._hover_cursor = None
            changed = True
        if changed:
            self.update()  # type: ignore[misc]

    def get_center_and_radius(self) -> None:
        """Compute the bounding sphere of the current atom set."""
        min_ = [999999.0, 999999.0, 999999.0]
        max_ = [-999999.0, -999999.0, -999999.0]
        for at in self.atoms:
            for j in range(3):
                v = float(at.coordinate[j])
                if v < min_[j]:
                    min_[j] = v
                if v > max_[j]:
                    max_[j] = v
        c = np.array([(max_[j] + min_[j]) / 2 for j in range(3)], dtype=np.float32)
        r = 0.0
        for atom in self.atoms:
            d = dist(atom.coordinate, c) + 1.5
            if d > r:
                r = d
        self.molecule_center = c
        self.molecule_radius = r or 10

    def get_spherical_radius(self, atom: Atom) -> float:
        """Return an approximate isotropic radius for label-offset calculations."""
        if atom.u_cart is not None and not atom.adp_valid:
            # NPD cube: bounding radius in Angstrom (atoms_size / scale is
            # zoom-invariant, so this is a constant).
            return self.atoms_size * NPD_CUBE_BOUND_FACTOR / self.scale
        if self._show_adps and atom.u_iso is not None:
            return sqrt(atom.u_iso)
        return 0.23

    def get_directional_radius(self, atom: Atom, v: np.ndarray) -> float:
        """Return the distance from the atom centre to its ellipsoid surface along *v*."""
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        d = sqrt(vx * vx + vy * vy + vz * vz)
        if d < 1e-8:
            return 0.23
        if not atom.adp_valid:
            return 0.23
        if self._show_adps and atom.u_inv is not None:
            inv_d = 1.0 / d
            ux, uy, uz = vx * inv_d, vy * inv_d, vz * inv_d
            M = atom.u_inv
            tx = M[0, 0] * ux + M[0, 1] * uy + M[0, 2] * uz
            ty = M[1, 0] * ux + M[1, 1] * uy + M[1, 2] * uz
            tz = M[2, 0] * ux + M[2, 1] * uy + M[2, 2] * uz
            val = ux * tx + uy * ty + uz * tz
            if val > 0:
                return self.adp_scale / sqrt(val)
        if self._show_adps and atom.u_iso is not None:
            return sqrt(atom.u_iso) * self.adp_scale
        return 0.23

    def get_conntable_from_atoms(self, extra_param: float = 1.2) -> tuple:
        """Build a connectivity table from atomic coordinates and covalent radii."""
        from fastmolwidget.tools import build_conntable
        coords = np.array([a.coordinate for a in self.atoms], dtype=np.float64)
        types = [a.type_ for a in self.atoms]
        parts = [a.part for a in self.atoms]
        radii = np.array([a.radius for a in self.atoms], dtype=np.float64)
        symmgen = [a.symmgen for a in self.atoms]
        return build_conntable(
            coords, types, parts,
            radii=radii, extra_param=extra_param, symmgen=symmgen,
        )

    # ------------------------------------------------------------------
    # Rotation matrices
    # ------------------------------------------------------------------

    def rotate_x(self) -> np.typing.NDArray[np.float32]:
        """3×3 rotation matrix around X by :attr:`x_angle` radians."""
        return np.array([
            [1, 0, 0],
            [0, cos(self.x_angle), -sin(self.x_angle)],
            [0, sin(self.x_angle), cos(self.x_angle)],
        ], dtype=np.float32)

    def rotate_y(self) -> np.typing.NDArray[np.float32]:
        """3×3 rotation matrix around Y by :attr:`y_angle` radians."""
        return np.array([
            [cos(self.y_angle), 0, sin(self.y_angle)],
            [0, 1, 0],
            [-sin(self.y_angle), 0, cos(self.y_angle)],
        ], dtype=np.float32)

    def _uij_to_cart(
        self,
        uvals: tuple[float, float, float, float, float, float],
        symm_matrix: np.ndarray | None = None,
    ) -> np.ndarray:
        """Convert fractional *Uij* displacement parameters to Cartesian tensor."""
        U11, U22, U33, U23, U13, U12 = uvals
        Uij = np.array([
            [U11, U12, U13],
            [U12, U22, U23],
            [U13, U23, U33],
        ], dtype=float)
        if symm_matrix is not None:
            Uij = symm_matrix.T @ Uij @ symm_matrix
        N = np.diag([self._astar, self._bstar, self._cstar])
        return self._amatrix.dot(N).dot(Uij).dot(N.T).dot(self._amatrix.T)

    # ------------------------------------------------------------------
    # Hit-testing
    # ------------------------------------------------------------------

    def is_point_inside_atom(self, atom: Atom, px: float, py: float) -> bool:
        """Return ``True`` if (px, py) is inside the atom's 2-D projection."""
        cx = atom.screenx
        cy = atom.screeny
        dx = px - cx
        dy = py - cy
        if atom.u_cart is not None and not atom.adp_valid:
            bound = self.atoms_size * NPD_CUBE_BOUND_FACTOR
            return dx ** 2 + dy ** 2 <= bound ** 2
        if self._show_adps and atom.u_cart is not None:
            a = atom.u_cart[0, 0]
            b = atom.u_cart[0, 1]
            c = atom.u_cart[1, 1]
            T = a + c
            D = a * c - b * b
            diff = T * T * 0.25 - D
            if diff >= 0:
                sq = sqrt(diff)
                eig1 = T * 0.5 - sq
                eig2 = T * 0.5 + sq
                if eig1 > 0 and eig2 > 0:
                    r1 = sqrt(eig1) * self.scale * self.adp_scale
                    r2 = sqrt(eig2) * self.scale * self.adp_scale
                    angle = (
                        degrees(atan2(eig1 - a, b)) if abs(b) > 1e-8
                        else (0.0 if a < c else 90.0)
                    )
                    rad = radians(angle)
                    cos_a, sin_a = cos(rad), sin(rad)
                    local_x = dx * cos_a + dy * sin_a
                    local_y = -dx * sin_a + dy * cos_a
                    return (local_x ** 2 / r1 ** 2) + (local_y ** 2 / r2 ** 2) <= 1.0
        circle_size = self.atoms_size
        if self._show_adps and atom.u_iso is not None:
            circle_size = sqrt(atom.u_iso) * self.scale * self.adp_scale * 2
        return dx ** 2 + dy ** 2 <= (circle_size / 2) ** 2

    def is_point_near_bond(self, at1: Atom, at2: Atom, px: float, py: float) -> bool:
        """Return ``True`` if (px, py) is near the projected bond segment."""
        line_data = self._get_bond_line(at1, at2)
        if not line_data:
            return False
        x1, y1, x2, y2, dynamic_width = line_data
        line_vec = np.array([x2 - x1, y2 - y1])
        p_vec = np.array([px - x1, py - y1])
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0.0:
            return False
        t = max(0.0, min(1.0, np.dot(p_vec, line_vec) / line_len_sq))
        proj = np.array([x1, y1]) + t * line_vec
        dist_sq = (px - proj[0]) ** 2 + (py - proj[1]) ** 2
        tolerance = max(5.0, dynamic_width / 2.0 + 4.0)
        return dist_sq <= tolerance ** 2

    # ------------------------------------------------------------------
    # Mouse / keyboard event handlers (compatible with both QWidget and
    # QQuickItem because QMouseEvent / QWheelEvent / QKeyEvent are shared)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._lastPos = event.position()
        self._pressPos = event.position()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: C901
        if event.button() == Qt.MouseButton.MiddleButton:
            dx = event.position().x() - self._pressPos.x()
            dy = event.position().y() - self._pressPos.y()
            if abs(dx) < 5 and abs(dy) < 5:
                self._recenter_on_click(event.position().x(), event.position().y())
            super().mouseReleaseEvent(event)  # type: ignore[misc]
            return

        if event.button() == Qt.MouseButton.LeftButton:
            dx = event.position().x() - self._pressPos.x()
            dy = event.position().y() - self._pressPos.y()
            if abs(dx) < 5 and abs(dy) < 5:
                x = event.position().x()
                y = event.position().y()
                modifiers = event.modifiers()
                if bool(modifiers & Qt.KeyboardModifier.AltModifier):
                    self._recenter_on_click(x, y)
                    super().mouseReleaseEvent(event)  # type: ignore[misc]
                    return

                clicked_atom = None
                clicked_bond = None
                front_z = float('inf')
                for item in self.objects:
                    if not self.show_hydrogens_flag:
                        if (item.atom1.type_ in ('H', 'D')
                                or (item.is_bond and item.atom2.type_ in ('H', 'D'))):
                            continue
                    if item.is_bond:
                        if self.is_point_near_bond(item.atom1, item.atom2, x, y):
                            if item.z_order < front_z:
                                front_z = item.z_order
                                clicked_bond = tuple(sorted((item.atom1.name, item.atom2.name)))
                                clicked_atom = None
                    else:
                        if self.is_point_inside_atom(item.atom1, x, y):
                            if item.z_order < front_z:
                                front_z = item.z_order
                                clicked_atom = item.atom1
                                clicked_bond = None

                ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                changed = False
                if clicked_atom:
                    if ctrl:
                        if clicked_atom.name in self.selected_atoms:
                            self.selected_atoms.remove(clicked_atom.name)
                        else:
                            self.selected_atoms.add(clicked_atom.name)
                    else:
                        self.selected_atoms = {clicked_atom.name}
                        self.selected_bonds.clear()
                    changed = True
                    self.atomClicked.emit(clicked_atom.name)  # type: ignore[attr-defined]
                elif clicked_bond:
                    if ctrl:
                        if clicked_bond in self.selected_bonds:
                            self.selected_bonds.remove(clicked_bond)
                        else:
                            self.selected_bonds.add(clicked_bond)
                    else:
                        self.selected_bonds = {clicked_bond}
                        self.selected_atoms.clear()
                    changed = True
                    self.bondClicked.emit(clicked_bond[0], clicked_bond[1])  # type: ignore[attr-defined]
                else:
                    if not ctrl and (self.selected_atoms or self.selected_bonds):
                        self.selected_atoms.clear()
                        self.selected_bonds.clear()
                        changed = True
                if changed:
                    self.update()  # type: ignore[misc]

        super().mouseReleaseEvent(event)  # type: ignore[misc]

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.rotate_molecule(event)
            self._clear_hover_state()
        elif event.buttons() == Qt.MouseButton.RightButton:
            self.zoom_molecule(event)
            self._clear_hover_state()
        elif event.buttons() == Qt.MouseButton.MiddleButton:
            self.pan_molecule(event)
            self._clear_hover_state()
        else:
            self._update_hover(event.position().x(), event.position().y())
        self._lastPos = event.position()

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Increase or decrease label font size on scroll."""
        if event.angleDelta().y() > 0:
            self.setLabelFont(self.fontsize + 2)
        elif event.angleDelta().y() < 0:
            self.setLabelFont(self.fontsize - 2)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """F1=a, F2=b, F3=c axis alignment."""
        if event.key() == Qt.Key.Key_F1:
            self._align_to_reciprocal_axis(0)
        elif event.key() == Qt.Key.Key_F2:
            self._align_to_reciprocal_axis(1)
        elif event.key() == Qt.Key.Key_F3:
            self._align_to_reciprocal_axis(2)
        else:
            super().keyPressEvent(event)  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Interaction helpers
    # ------------------------------------------------------------------

    def rotate_molecule(self, event: QMouseEvent) -> None:
        self.y_angle = -(event.position().x() - self._lastPos.x()) / 100
        self.x_angle = (event.position().y() - self._lastPos.y()) / 100
        R_y = self.rotate_y()
        R_x = self.rotate_x()
        R = np.dot(R_x, R_y)
        self.cumulative_R = np.dot(R, self.cumulative_R)
        if self.atoms:
            self._coords_array = (
                np.dot(self._coords_array - self.molecule_center, R.T)
                + self.molecule_center
            )
            if np.any(self._has_adp):
                self._ucart_array = np.matmul(R, np.matmul(self._ucart_array, R.T))
                self._eigenvectors_array = np.matmul(R, self._eigenvectors_array)
                self._u_inv_array = np.matmul(R, np.matmul(self._u_inv_array, R.T))
            for i, at in enumerate(self.atoms):
                at.coordinate = self._coords_array[i]
                at.z = at.coordinate[2]
                if self._has_adp[i]:
                    at.u_cart = self._ucart_array[i]
                    at.u_eigvecs = self._eigenvectors_array[i]
                    at.u_inv = self._u_inv_array[i]
        self.update()  # type: ignore[misc]

    def zoom_molecule(self, event: QMouseEvent) -> None:
        delta = (self._lastPos.y() - event.position().y()) / 400
        self.zoom = max(0.005, self.zoom - delta)
        self.update()  # type: ignore[misc]

    def pan_molecule(self, event: QMouseEvent) -> None:
        self.molecule_center[0] += (self._lastPos.x() - event.position().x()) / 50
        self.molecule_center[1] += (self._lastPos.y() - event.position().y()) / 50
        self.update()  # type: ignore[misc]

    def _recenter_on_click(self, px: float, py: float) -> None:
        """Recentre the rotation pivot on the atom under the cursor."""
        if not self.atoms:
            return
        clicked_atom: Atom | None = None
        front_z = float('inf')
        for item in self.objects:
            if item.is_bond:
                continue
            if not self.show_hydrogens_flag and item.atom1.type_ in ('H', 'D'):
                continue
            if self.is_point_inside_atom(item.atom1, px, py):
                if item.z_order < front_z:
                    front_z = item.z_order
                    clicked_atom = item.atom1
        if clicked_atom is not None:
            self.molecule_center = clicked_atom.coordinate.copy()
            self.update()  # type: ignore[misc]

    def _clear_hover_state(self) -> None:
        if self.hovered_atom is not None or self.hovered_bond is not None:
            self.hovered_atom = None
            self.hovered_bond = None
            self._hovered_bond_distance = None
            self._hover_cursor = None

    def _update_hover(self, px: float, py: float) -> None:
        if not self.atoms:
            return
        hydrogens = ('H', 'D')
        new_atom: str | None = None
        new_bond: tuple[str, str] | None = None
        new_dist: float | None = None
        front_z = float('inf')
        for item in self.objects:
            if item.is_bond:
                at1, at2 = item.atom1, item.atom2
                if not self.show_hydrogens_flag and (
                    at1.type_ in hydrogens or at2.type_ in hydrogens
                ):
                    continue
                if self._visible_parts is not None and (
                    at1.part not in self._visible_parts
                    or at2.part not in self._visible_parts
                ):
                    continue
                if self.is_point_near_bond(at1, at2, px, py):
                    if item.z_order < front_z:
                        front_z = item.z_order
                        new_bond = tuple(sorted((at1.name, at2.name)))  # type: ignore[assignment]
                        new_atom = None
                        new_dist = float(np.linalg.norm(at1.coordinate - at2.coordinate))
            else:
                atom = item.atom1
                if not self.show_hydrogens_flag and atom.type_ in hydrogens:
                    continue
                if self._visible_parts is not None and atom.part not in self._visible_parts:
                    continue
                if self.is_point_inside_atom(atom, px, py):
                    if item.z_order < front_z:
                        front_z = item.z_order
                        new_atom = atom.name
                        new_bond = None
                        new_dist = None
        cursor = QtCore.QPointF(px, py)
        changed = (
            new_atom != self.hovered_atom
            or new_bond != self.hovered_bond
            or (new_bond is not None and self._hover_cursor != cursor)
        )
        self.hovered_atom = new_atom
        self.hovered_bond = new_bond
        self._hovered_bond_distance = new_dist
        self._hover_cursor = cursor if new_bond is not None else None
        if changed:
            self.update()  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Main draw method (called from paintEvent / paint)
    # ------------------------------------------------------------------

    def draw(self) -> None:
        """Execute the full rendering pass using :attr:`_painter`."""
        self.scale = self.zoom * 130
        self.screen_center = [self.width() / 2, self.height() / 2]  # type: ignore[attr-defined]
        self.cx_global = self.screen_center[0] - self.molecule_center[0] * self.scale
        self.cy_global = self.screen_center[1] - self.molecule_center[1] * self.scale
        self._cached_adp_line_width = self._adp_intersection_line_width()

        hydrogens = ('H', 'D')
        margin = self.scale * self.adp_scale * 2.0 + 40.0
        vp_left = -margin
        vp_top = -margin
        vp_right = self.width() + margin   # type: ignore[attr-defined]
        vp_bottom = self.height() + margin  # type: ignore[attr-defined]

        for atom in self.atoms:
            c = atom.coordinate
            atom.screenx = c[0] * self.scale + self.cx_global
            atom.screeny = c[1] * self.scale + self.cy_global

        self.calculate_z_order()

        label_atoms: list[Atom] = []
        for item in self.objects:
            if not self.show_hydrogens_flag:
                if (item.atom1.type_ in hydrogens
                        or (item.is_bond and item.atom2.type_ in hydrogens)):
                    continue
            if self._visible_parts is not None:
                if item.atom1.part not in self._visible_parts:
                    continue
                if item.is_bond and item.atom2.part not in self._visible_parts:
                    continue
            if item.is_bond:
                a1, a2 = item.atom1, item.atom2
                if (
                    (a1.screenx < vp_left and a2.screenx < vp_left)
                    or (a1.screenx > vp_right and a2.screenx > vp_right)
                    or (a1.screeny < vp_top and a2.screeny < vp_top)
                    or (a1.screeny > vp_bottom and a2.screeny > vp_bottom)
                ):
                    continue
                self._draw_bond_rounded(a1, a2)
            else:
                sx = item.atom1.screenx
                sy = item.atom1.screeny
                if sx < vp_left or sx > vp_right or sy < vp_top or sy > vp_bottom:
                    continue
                self.draw_atom(item.atom1)
                is_hovered = item.atom1.name == self.hovered_atom
                if is_hovered:
                    label_atoms.append(item.atom1)
                elif self.labels and item.atom1.type_ not in hydrogens:
                    label_atoms.append(item.atom1)

        for atom in label_atoms:
            self.draw_label(atom, enlarged=(atom.name == self.hovered_atom))

        if (
            self.hovered_atom is None
            and self.hovered_bond is not None
            and self._hovered_bond_distance is not None
            and self._hover_cursor is not None
        ):
            self._draw_hover_distance_label(
                f"{self._hovered_bond_distance:.3f} Å",
                self._hover_cursor.x(),
                self._hover_cursor.y(),
            )

        if self._is_packed:
            self._draw_axis_indicator()

    def calculate_z_order(self) -> None:
        """Sort :attr:`objects` back-to-front by depth."""
        for item in self.objects:
            item.z_order = (
                (item.atom1.z + item.atom2.z) / 2.0 if item.is_bond
                else item.atom1.z
            )
        self.objects.sort(reverse=True, key=lambda item: item.z_order)

    # ------------------------------------------------------------------
    # Bond drawing
    # ------------------------------------------------------------------

    def _rebuild_bond_brush(self) -> None:
        base = self.bond_color
        dark = base.darker(170)
        light = base.lighter(160)
        shadow = base.darker(280)
        bg = QLinearGradient(0, 1, 0, -1)
        bg.setColorAt(0.0, dark)
        bg.setColorAt(0.2, light)
        bg.setColorAt(1.0, shadow)
        self.bond_brush = QBrush(bg)

    def _get_bond_line(
        self, at1: Atom, at2: Atom,
    ) -> tuple[float, float, float, float, int] | None:
        c1 = at1.coordinate
        c2 = at2.coordinate
        v = c2 - c1
        vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
        d = sqrt(vx * vx + vy * vy + vz * vz)
        r1 = self.get_directional_radius(at1, v)
        r2 = self.get_directional_radius(at2, -v)
        if d <= r1 + r2:
            return None
        v_norm = v / d
        p1 = c1 + v_norm * r1
        p2 = c2 - v_norm * r2
        x1 = p1[0] * self.scale + self.cx_global
        y1 = p1[1] * self.scale + self.cy_global
        x2 = p2[0] * self.scale + self.cx_global
        y2 = p2[1] * self.scale + self.cy_global
        dynamic_width = max(1, int(self.bond_width * self.zoom * 5))
        return x1, y1, x2, y2, dynamic_width

    def _draw_bond_selection(
        self, x1: float, y1: float, x2: float, y2: float, dynamic_width: int,
    ) -> None:
        sel_width = dynamic_width + max(4, int(12 * self.zoom))
        pen = QPen(QColor(0, 190, 255), sel_width, Qt.PenStyle.SolidLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._painter.setPen(pen)  # type: ignore[union-attr]
        self._painter.drawLine(int(x1), int(y1), int(x2), int(y2))  # type: ignore[union-attr]

    def _draw_bond_rounded(self, at1: Atom, at2: Atom) -> None:
        line_data = self._get_bond_line(at1, at2)
        if not line_data:
            return
        x1, y1, x2, y2, dynamic_width = line_data
        dx = x2 - x1
        dy = y2 - y1
        length = sqrt(dx * dx + dy * dy)
        if length < 0.0001:
            return
        bond_key = tuple(sorted((at1.name, at2.name)))
        if bond_key in self.selected_bonds:
            self._draw_bond_selection(x1, y1, x2, y2, dynamic_width)
        nx = -dy / length
        ny = dx / length
        Lx, Ly = -1.0, -1.0
        if (nx * Lx + ny * Ly) < 0:
            nx, ny = -nx, -ny
        t = QTransform(
            ny * dynamic_width / 2.0, -nx * dynamic_width / 2.0,
            nx * dynamic_width / 2.0, ny * dynamic_width / 2.0,
            x1, y1,
        )
        self.bond_brush.setTransform(t)
        pen = QPen(self.bond_brush, dynamic_width, Qt.PenStyle.SolidLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._painter.setPen(pen)  # type: ignore[union-attr]
        self._painter.drawLine(int(x1), int(y1), int(x2), int(y2))  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Atom drawing
    # ------------------------------------------------------------------

    def draw_atom(self, atom: Atom) -> None:
        """Draw a single atom as ADP ellipsoid, sphere, or fixed circle."""
        if atom.u_cart is not None and not atom.adp_valid:
            # Non-positive-definite tensor: show the cube placeholder in both
            # ADP and isotropic mode so the broken atom is never hidden.
            self._draw_invalid_adp(atom)
            return
        cx = atom.screenx
        cy = atom.screeny

        if self._show_adps and atom.u_cart is not None:
            a = atom.u_cart[0, 0]
            b = atom.u_cart[0, 1]
            c = atom.u_cart[1, 1]
            T = a + c
            D = a * c - b * b
            diff = T * T * 0.25 - D
            if diff >= 0:
                sq = sqrt(diff)
                eig1 = T * 0.5 - sq
                eig2 = T * 0.5 + sq
                if eig1 > 0 and eig2 > 0:
                    r1 = sqrt(eig1) * self.scale * self.adp_scale
                    r2 = sqrt(eig2) * self.scale * self.adp_scale
                    angle = (
                        degrees(atan2(eig1 - a, b)) if abs(b) > 1e-8
                        else (0.0 if a < c else 90.0)
                    )
                    self._painter.save()  # type: ignore[union-attr]
                    self._painter.translate(cx, cy)  # type: ignore[union-attr]
                    self._painter.rotate(angle)  # type: ignore[union-attr]
                    if atom.name in self.selected_atoms:
                        self._draw_selection(r1, r2)
                    max_r = max(r1, r2)
                    sx, sy = -max_r * 0.3, -max_r * 0.3
                    rad = radians(angle)
                    fx = sx * cos(rad) + sy * sin(rad)
                    fy = -sx * sin(rad) + sy * cos(rad)
                    t = QTransform()
                    t.translate(fx, fy)
                    t.scale(max_r * 1.5, max_r * 1.5)
                    atom.adp_brush.setTransform(t)
                    self._painter.setBrush(atom.adp_brush)  # type: ignore[union-attr]
                    self._painter.setPen(QPen(self.adp_pen_color, 1, Qt.PenStyle.SolidLine))  # type: ignore[union-attr]
                    self._painter.drawEllipse(QRectF(-r1, -r2, 2 * r1, 2 * r2))  # type: ignore[union-attr]
                    cross_pen = QPen(QColor(0, 0, 0, 120), self._cached_adp_line_width,
                                     Qt.PenStyle.SolidLine)
                    cross_pen.setCosmetic(True)
                    self._painter.setPen(cross_pen)  # type: ignore[union-attr]
                    self._draw_principal_arcs(atom, r1, r2, angle)
                    self._painter.restore()  # type: ignore[union-attr]
                    return

        circle_size = self.atoms_size
        if self._show_adps and atom.u_iso is not None:
            circle_size = sqrt(atom.u_iso) * self.scale * self.adp_scale * 2
        radius = circle_size / 2
        self._painter.save()  # type: ignore[union-attr]
        self._painter.translate(cx, cy)  # type: ignore[union-attr]
        if atom.name in self.selected_atoms:
            self._draw_selection(radius, radius)
        self._painter.setPen(QPen(self.fallback_pen_color, 1, Qt.PenStyle.SolidLine))  # type: ignore[union-attr]
        self._painter.setBrush(atom.sphere_brush)  # type: ignore[union-attr]
        self._painter.drawEllipse(QRectF(-radius, -radius, circle_size, circle_size))  # type: ignore[union-attr]
        self._painter.restore()  # type: ignore[union-attr]

    def _draw_invalid_adp(self, atom: Atom) -> None:
        """Draw the placeholder cube used for non-positive-definite ADPs.

        The cube is a real 3-D box oriented in the molecular frame, projected
        through the current view rotation, so it turns with the structure.
        All six faces are painted back-to-front (painter's algorithm), which
        gives correct hidden-face removal for any orientation.
        """
        half = self.atoms_size * NPD_CUBE_HALF_FACTOR
        self._painter.save()  # type: ignore[union-attr]
        self._painter.translate(atom.screenx, atom.screeny)  # type: ignore[union-attr]
        if atom.name in self.selected_atoms:
            bound = self.atoms_size * NPD_CUBE_BOUND_FACTOR
            self._draw_selection(bound, bound)
        self._painter.setPen(QPen(self.fallback_pen_color, 1))  # type: ignore[union-attr]
        for corners, _mean_z, normal in npd_cube_faces(self.cumulative_R, half):
            shade = npd_face_shade(normal)
            self._painter.setBrush(  # type: ignore[union-attr]
                QBrush(atom.color.lighter(max(1, round(shade * 100))))
            )
            polygon = QtGui.QPolygonF(
                [QtCore.QPointF(float(px), float(py)) for px, py in corners]
            )
            self._painter.drawPolygon(polygon)  # type: ignore[union-attr]
        self._painter.restore()  # type: ignore[union-attr]

    def draw_npd_text(self, dx: float, dy: float, s: float) -> None:
        self._painter.setPen(QPen(Qt.GlobalColor.white))  # type: ignore[union-attr]
        font = self._painter.font()  # type: ignore[union-attr]
        old_size = font.pixelSize()
        font.setPixelSize(max(int(self.atoms_size * 0.3), 1))
        self._painter.setFont(font)  # type: ignore[union-attr]
        front_rect = QRectF(-s - dx / 2, -s - dy / 2, 2 * s, 2 * s)
        self._painter.drawText(front_rect, Qt.AlignmentFlag.AlignCenter, "NPD")  # type: ignore[union-attr]
        font.setPixelSize(old_size)
        self._painter.setFont(font)  # type: ignore[union-attr]

    def _draw_selection(self, r1: float, r2: float) -> None:
        padding = 4.0
        pen = QPen(QColor(0, 190, 255), max(3, 12 * self.zoom), Qt.PenStyle.SolidLine)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        self._painter.setPen(pen)  # type: ignore[union-attr]
        self._painter.setBrush(Qt.BrushStyle.NoBrush)  # type: ignore[union-attr]
        self._painter.drawEllipse(  # type: ignore[union-attr]
            QRectF(-r1 - padding, -r2 - padding, (r1 + padding) * 2, (r2 + padding) * 2)
        )

    def draw_label(self, atom: Atom, enlarged: bool = False) -> None:
        """Draw the atom's name next to its ellipsoid/sphere."""
        self._painter.setPen(QPen(QColor(100, 50, 5), 2, Qt.PenStyle.SolidLine))  # type: ignore[union-attr]
        r_pix = self.get_spherical_radius(atom) * self.scale
        if enlarged:
            base_font = self._painter.font()  # type: ignore[union-attr]
            hover_font = QtGui.QFont(base_font)
            hover_font.setPixelSize(max(1, int((self.fontsize + 4) * self.zoom * 2)))
            hover_font.setBold(True)
            self._painter.setFont(hover_font)  # type: ignore[union-attr]
            self._painter.drawText(  # type: ignore[union-attr]
                int(atom.screenx + r_pix + 2), int(atom.screeny - r_pix - 2), atom.name
            )
            self._painter.setFont(base_font)  # type: ignore[union-attr]
        else:
            self._painter.drawText(  # type: ignore[union-attr]
                int(atom.screenx + r_pix + 2), int(atom.screeny - r_pix - 2), atom.name
            )

    def _draw_principal_arcs(self, atom: Atom, r1: float, r2: float, angle: float) -> None:
        eigvals = atom.u_eigvals
        if eigvals is None or eigvals[0] <= 0 or eigvals[1] <= 0 or eigvals[2] <= 0:
            self._painter.drawLine(int(-r1), 0, int(r1), 0)  # type: ignore[union-attr]
            self._painter.drawLine(0, int(-r2), 0, int(r2))  # type: ignore[union-attr]
            return
        eigenvectors = atom.u_eigvecs
        angle_rad = radians(angle)
        cos_a = cos(angle_rad)
        sin_a = sin(angle_rad)
        c = self.adp_scale
        s = self.scale
        pen = self._painter.pen()  # type: ignore[union-attr]
        pen.setCosmetic(True)
        pen.setWidthF(self._cached_adp_line_width)
        self._painter.setPen(pen)  # type: ignore[union-attr]
        self._painter.setBrush(Qt.BrushStyle.NoBrush)  # type: ignore[union-attr]
        base_transform = self._painter.transform()  # type: ignore[union-attr]
        for i_ax, j_ax in ((1, 2), (0, 2), (0, 1)):
            li = eigvals[i_ax]
            lj = eigvals[j_ax]
            if li <= 0 or lj <= 0:
                continue
            ri_3d = c * sqrt(li)
            rj_3d = c * sqrt(lj)
            vi = eigenvectors[:, i_ax]
            vj = eigenvectors[:, j_ax]
            vi0, vi1, vi2 = float(vi[0]), float(vi[1]), float(vi[2])
            vj0, vj1, vj2 = float(vj[0]), float(vj[1]), float(vj[2])
            Ax = s * ri_3d * (vi0 * cos_a + vi1 * sin_a)
            Bx = s * rj_3d * (vj0 * cos_a + vj1 * sin_a)
            Ay = s * ri_3d * (-vi0 * sin_a + vi1 * cos_a)
            By = s * rj_3d * (-vj0 * sin_a + vj1 * cos_a)
            arc_xform = QTransform(Ax, Ay, Bx, By, 0.0, 0.0)
            self._painter.setTransform(arc_xform * base_transform)  # type: ignore[union-attr]
            # This cross-section lies ON the ellipsoid surface, so the
            # visible half is the front-facing part (silhouette split),
            # determined by the surface NORMAL, not by the depth of the
            # curve point.  The outward normal at
            # P(t) = ri*cos(t)*vi + rj*sin(t)*vj is proportional to
            # (cos(t)/ri)*vi + (sin(t)/rj)*vj, so the z-amplitude divides by
            # the radius rather than multiplying.  For a spherical ADP both
            # agree; for elongated ellipsoids the depth-based split is wrong.
            Az_n = vi2 / ri_3d
            Bz_n = vj2 / rj_3d
            z_amp = sqrt(Az_n * Az_n + Bz_n * Bz_n)
            if z_amp < 1e-8:
                self._painter.drawArc(self._UNIT_RECT, 0, 5760)  # type: ignore[union-attr]
            else:
                phi_n = atan2(Bz_n, Az_n)
                start_deg = degrees(-(phi_n + 1.5 * pi))
                self._painter.drawArc(self._UNIT_RECT, int(start_deg * 16), 2880)  # type: ignore[union-attr]
        self._painter.setTransform(base_transform)  # type: ignore[union-attr]

    def _draw_hover_distance_label(self, text: str, cx: float, cy: float) -> None:
        font = self._painter.font()  # type: ignore[union-attr]
        hover_font = QtGui.QFont(font)
        hover_font.setPixelSize(max(1, int(self.fontsize * self.zoom * 2)))
        hover_font.setBold(True)
        self._painter.setFont(hover_font)  # type: ignore[union-attr]
        metrics = QtGui.QFontMetrics(hover_font)
        pad_x, pad_y = 6.0, 3.0
        tw = metrics.horizontalAdvance(text)
        th = metrics.height()
        box_w = tw + 2 * pad_x
        box_h = th + 2 * pad_y
        x = cx + 14.0
        y = cy + 14.0
        w = max(1, self.width())   # type: ignore[attr-defined]
        h = max(1, self.height())  # type: ignore[attr-defined]
        if x + box_w > w:
            x = cx - 14.0 - box_w
        if y + box_h > h:
            y = cy - 14.0 - box_h
        rect = QRectF(x, y, box_w, box_h)
        self._painter.save()  # type: ignore[union-attr]
        self._painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)  # type: ignore[union-attr]
        self._painter.setBrush(QColor(143, 230, 193, 220))  # type: ignore[union-attr]
        self._painter.setPen(QPen(QColor(60, 60, 60, 220), 1.0))  # type: ignore[union-attr]
        self._painter.drawRoundedRect(rect, 5.0, 5.0)  # type: ignore[union-attr]
        self._painter.setPen(QColor(20, 20, 20))  # type: ignore[union-attr]
        self._painter.drawText(rect, int(Qt.AlignmentFlag.AlignCenter), text)  # type: ignore[union-attr]
        self._painter.restore()  # type: ignore[union-attr]
        self._painter.setFont(font)  # type: ignore[union-attr]

    def _draw_axis_indicator(self) -> None:
        """Draw unit-cell axis arrows (a=red, b=green, c=blue) in the bottom-left."""
        if self._amatrix is None or self._cell is None:
            return
        axes = [self._amatrix[:, i].astype(np.float64) for i in range(3)]
        axes = [v / np.linalg.norm(v) for v in axes]
        R = self.cumulative_R.astype(np.float64)
        axes = [R @ v for v in axes]
        arrow_len = 40.0
        origin_x = 55.0
        origin_y = self.height() - 55.0  # type: ignore[attr-defined]
        colors = [QColor(220, 30, 30), QColor(30, 160, 30), QColor(30, 30, 220)]
        labels = ['a', 'b', 'c']
        self._painter.save()  # type: ignore[union-attr]
        self._painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)  # type: ignore[union-attr]
        font = QtGui.QFont()
        font.setPixelSize(12)
        font.setBold(True)
        self._painter.setFont(font)  # type: ignore[union-attr]
        for i in range(3):
            vx, vy = float(axes[i][0]), float(axes[i][1])
            tip_x = origin_x + vx * arrow_len
            tip_y = origin_y + vy * arrow_len
            pen = QPen(colors[i], 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._painter.setPen(pen)  # type: ignore[union-attr]
            self._painter.drawLine(  # type: ignore[union-attr]
                QtCore.QPointF(origin_x, origin_y),
                QtCore.QPointF(tip_x, tip_y),
            )
            dx, dy = tip_x - origin_x, tip_y - origin_y
            length = sqrt(dx * dx + dy * dy)
            if length > 1e-6:
                ux, uy = dx / length, dy / length
                px2, py2 = -uy, ux
                head_len, head_w = 8.0, 3.5
                self._painter.drawLine(  # type: ignore[union-attr]
                    QtCore.QPointF(tip_x, tip_y),
                    QtCore.QPointF(
                        tip_x - ux * head_len + px2 * head_w,
                        tip_y - uy * head_len + py2 * head_w,
                    ),
                )
                self._painter.drawLine(  # type: ignore[union-attr]
                    QtCore.QPointF(tip_x, tip_y),
                    QtCore.QPointF(
                        tip_x - ux * head_len - px2 * head_w,
                        tip_y - uy * head_len - py2 * head_w,
                    ),
                )
            self._painter.setPen(colors[i])  # type: ignore[union-attr]
            self._painter.drawText(  # type: ignore[union-attr]
                QtCore.QPointF(
                    tip_x + 4 * (1 if vx >= 0 else -2),
                    tip_y + 4 * (-1 if vy >= 0 else 2),
                ),
                labels[i],
            )
        self._painter.restore()  # type: ignore[union-attr]

