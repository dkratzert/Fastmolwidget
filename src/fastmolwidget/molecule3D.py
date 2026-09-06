"""
Real 3D OpenGL molecule-display widget.

Provides :class:`MoleculeWidget3D`, a drop-in replacement for
:class:`~fastmolwidget.molecule2D.MoleculeWidget` that uses hardware-accelerated
OpenGL rendering.  The public API is identical to :class:`MoleculeWidget`.

If *PyOpenGL* is not installed the widget degrades gracefully to a plain
:class:`~qtpy.QtWidgets.QWidget` that shows an informational message.  Any
OpenGL initialisation failure is caught at runtime and the same text fallback
is activated, so the host application never crashes.

Rendering overview
------------------
* **Atoms** – sphere impostors: each atom is rendered as a billboard quad; the
  fragment shader ray-casts a sphere and writes the correct depth value so that
  atoms overlap correctly regardless of draw order.
* **ADP ellipsoids** – same impostor technique, but the fragment shader is given
  the inverse of the scaled U_cart tensor via a ``mat3`` uniform and
  ray-casts an exact ellipsoid.  One draw call per ADP atom.
* **Bonds** – tessellated cylinder mesh (8 sides) generated on the CPU and
  uploaded as a single VBO.  No end caps are needed because atom spheres visually
  close the cylinder ends.
* **Labels** – rendered with :class:`~qtpy.QtGui.QPainter` as an overlay after
  the OpenGL pass.

All GLSL shaders target ``#version 120`` on macOS (OpenGL 2.1 / GLSL 1.20,
the maximum available in a macOS compatibility-profile context) and
``#version 140`` on Windows/Linux (OpenGL 3.1+ / GLSL 1.40).
See :mod:`fastmolwidget.shaders` for the shader templates.
The widget always requests a compatibility-profile context so that Qt's
``QPainter`` GL paint engine (used for the label overlay) keeps working
alongside the molecule shaders.

Mouse controls
--------------
* **Left drag**  – rotate.
* **Right drag** – zoom.
* **Middle drag** – pan.
* **Middle click** – centre the view on the clicked atom (becomes the new
  rotation pivot).
* **Scroll wheel** – increase / decrease label font size.
* **Ctrl + scroll wheel** – raise / lower the residual-density contour level.
* **Left click** – select atom or bond; emit ``atomClicked`` / ``bondClicked``.
* **Ctrl + left click** – add to / remove from selection.
* **Ctrl + left drag** (starting on an atom) – interactively drag the
  disordered fragment beyond the currently selected anchor atoms to
  reposition it, guided by a residual-density map towards the alternate-site
  peak; the original atoms are duplicated into a permanent "part 2" the
  first time this happens and only the copy moves.  With **no** atoms
  selected, the whole connected molecule under the cursor is dragged as a
  free body instead, for modelling whole-molecule disorder.  With a single
  **bond** selected instead of atoms, that bond is the split point: its far
  end is the anchor and the fragment rotates about the bond, with elastic
  give and with the near end free to drift off the axis so a tumble rather
  than only a clean torsion can be modelled.  The geometry lives in
  :mod:`fastmolwidget.disorder_drag` and the gesture handling in
  :mod:`fastmolwidget.disorder_controller` - both renderer-independent, so
  this widget only supplies the hooks that need OpenGL (picking, the
  screen→world projection, atom cloning and the geometry rebuild).
* **Ctrl + Shift + left drag** (starting on an atom) – freely reposition just
  that one atom.  No moiety, no anchors, no duplication - only the picked
  atom's position changes, everything else stays exactly as it was.
"""

from __future__ import annotations

import ctypes
from math import cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from fastmolwidget import atoms as _atoms
from fastmolwidget.atoms import (
    display_radius_for_element,
    element2color,
    fade_towards_white as _fade_towards_white,
    hex_to_rgb_float as _hex_to_rgb_float,
    part_fade as _part_fade,
)
from fastmolwidget.disorder_controller import DisorderDragMixin
from fastmolwidget.molecule2D import calc_volume
from fastmolwidget.molecule_base import (
    DENSITY_LEVEL_MAX,
    DENSITY_LEVEL_MIN,
    DENSITY_LEVEL_STEP,
    ModelSourceMixin,
)
from fastmolwidget.disorder_drag import isotropic_u_for_atom_type
from fastmolwidget.sdm import Atomtuple
from fastmolwidget import shaders as _shaders

# Backwards-compatible aliases: the disorder-part colouring now lives in the
# Qt-free atoms module so every renderer can share it.
_PART_FADE_BASE = _atoms.PART_FADE_BASE
_PART_FADE_STEP = _atoms.PART_FADE_STEP
_PART_FADE_MAX = _atoms.PART_FADE_MAX

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    import OpenGL.GL as gl
    import OpenGL.GL.shaders as _glshaders

    _HAS_PYOPENGL: bool = True
except Exception:  # ImportError or any platform error
    _HAS_PYOPENGL = False
    gl = None  # type: ignore[assignment]
    _glshaders = None  # type: ignore[assignment]

# Pick the best available QOpenGLWidget base class.
_QOGLBase: type | None = None
try:
    from qtpy.QtOpenGLWidgets import QOpenGLWidget as _QOGLBase  # Qt ≥ 6 / qtpy shim
except ImportError:
    try:
        from qtpy.QtWidgets import QOpenGLWidget as _QOGLBase  # Qt 5
    except (ImportError, AttributeError):
        _QOGLBase = None

_WidgetBase: type = _QOGLBase if _QOGLBase is not None else QtWidgets.QWidget
_IS_GL_WIDGET: bool = _QOGLBase is not None

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

__all__ = ["MoleculeWidget3D"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_rgb_color(color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int]
                         ) -> tuple[float, float, float]:
    """Normalise a QColor/hex/RGB triple to float RGB in ``[0, 1]``."""
    if isinstance(color, QtGui.QColor):
        return (color.redF(), color.greenF(), color.blueF())

    if isinstance(color, str):
        return _hex_to_rgb_float(color)

    if len(color) != 3:
        raise ValueError("Bond color must have exactly three RGB components.")

    rgb = tuple(float(component) for component in color)
    if any(component < 0.0 for component in rgb):
        raise ValueError("Bond color components must be non-negative.")
    if any(component > 1.0 for component in rgb):
        if any(component > 255.0 for component in rgb):
            raise ValueError("Integer RGB bond colors must be in the range 0..255.")
        rgb = tuple(component / 255.0 for component in rgb)

    return (
        min(1.0, max(0.0, rgb[0])),
        min(1.0, max(0.0, rgb[1])),
        min(1.0, max(0.0, rgb[2])),
    )


def _make_cylinder(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    color: tuple[float, float, float],
    n_seg: int = 20,
    selected: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Generate a cylinder mesh between *p1* and *p2*."""
    axis = p2 - p1
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None, None

    u = axis / length

    # Two vectors perpendicular to the cylinder axis.
    if abs(u[0]) < 0.9:
        v = np.cross(u, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    else:
        v = np.cross(u, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)

    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Outward segment normals.
    normals = cos_a[:, None] * v[None, :] + sin_a[:, None] * w[None, :]  # (n_seg, 3)

    verts = np.zeros((2 * n_seg, 10), dtype=np.float32)
    sel_flag = 1.0 if selected else 0.0

    # Bottom ring.
    for i in range(n_seg):
        verts[i, :3] = p1 + radius * normals[i]
        verts[i, 3:6] = normals[i]
        verts[i, 6:9] = color
        verts[i, 9] = sel_flag

    # Top ring.
    for i in range(n_seg):
        verts[n_seg + i, :3] = p2 + radius * normals[i]
        verts[n_seg + i, 3:6] = normals[i]
        verts[n_seg + i, 6:9] = color
        verts[n_seg + i, 9] = sel_flag

    # Side-surface triangles.
    idx_list = []
    for i in range(n_seg):
        next_i = (i + 1) % n_seg
        b0, b1 = i, next_i
        t0, t1 = i + n_seg, next_i + n_seg
        idx_list.extend([b0, t0, b1, b1, t0, t1])

    return verts, np.array(idx_list, dtype=np.uint32)


# ---------------------------------------------------------------------------
# Internal atom representation
# ---------------------------------------------------------------------------

class _Atom3D:
    """Internal 3-D atom representation used by :class:`MoleculeWidget3D`."""

    __slots__ = [
        "center", "label", "type_", "part", "symmgen",
        "color_f", "display_radius",
        "u_cart", "u_iso", "adp_valid", "u_eigvals", "u_eigvecs",
        "adp_billboard_r", "adp_A_matrix", "npd_half_edge",
    ]

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        label: str,
        type_: str,
        part: int,
        u_eq: float | None = None,
    ) -> None:
        self.center = np.array([x, y, z], dtype=np.float32)
        self.label = label
        self.type_ = type_
        self.part = part
        self.symmgen = False

        hex_color = element2color.get(type_, "#808080")
        self.color_f: tuple[float, float, float] = _hex_to_rgb_float(hex_color)

        # World-space display radius (Å).
        self.display_radius: float = display_radius_for_element(type_)

        self.u_cart: np.ndarray | None = None
        # Isotropic U (Å²); H/D keeps None unless refined anisotropically, so
        # hydrogen always draws at the fixed HYDROGEN_DISPLAY_RADIUS like in the
        # 2-D and JS renderers.
        self.u_iso: float | None = u_eq if type_ not in ('H', 'D') else None
        self.adp_valid: bool = True
        self.u_eigvals: np.ndarray | None = None
        self.u_eigvecs: np.ndarray | None = None

        # Set by _build_geometry when ADP data is present.
        self.adp_billboard_r: float = 0.0
        self.adp_A_matrix: np.ndarray | None = None
        # Half-edge of the NPD placeholder cube (Å).
        self.npd_half_edge: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Atom3D {self.label} {self.type_} {self.center}>"


# ---------------------------------------------------------------------------
# GLSL shader sources  (see fastmolwidget/shaders.py for the templates)
# ---------------------------------------------------------------------------

# Platform-selected at import time: GLSL 1.20 on macOS, 1.40 elsewhere.
_SPHERE_VERT = _shaders.SPHERE_VERT
_SPHERE_FRAG = _shaders.SPHERE_FRAG
_CYLINDER_VERT = _shaders.CYLINDER_VERT
_CYLINDER_FRAG = _shaders.CYLINDER_FRAG
_ELLIPSOID_BATCH_VERT = _shaders.ELLIPSOID_BATCH_VERT
_ELLIPSOID_BATCH_FRAG = _shaders.ELLIPSOID_BATCH_FRAG
_LINE_VERT = _shaders.LINE_VERT
_LINE_FRAG = _shaders.LINE_FRAG

# Residual-density wireframe colours.
_DENSITY_POS_COLOR: tuple[float, float, float] = (0.0, 0.85, 0.0)
_DENSITY_NEG_COLOR: tuple[float, float, float] = (0.9, 0.0, 0.0)

#: Residual density is only shown within this distance (Å) of a visible atom.
DENSITY_MARGIN: float = 1.5

# Selection highlight colour.
_SEL_COLOR: tuple[float, float, float] = (0.0, 0.75, 1.0)

# Default bond colour.
_DEFAULT_BOND_COLOR: tuple[float, float, float] = _hex_to_rgb_float("#d1812a")

# ORTEP 50 % scale factor.
_ADP_SCALE: float = 1.5382

# Screen-space tolerance in pixels for bond hit-testing.
_BOND_HIT_TOLERANCE_PX: float = 6.0

# Bounding-sphere radius of the NPD placeholder cube.
_NPD_BOUND_FACTOR: float = 1.7320508075688772


def _npd_bound_radius(atom: _Atom3D) -> float:
    """Return the bounding-sphere radius of an atom's NPD placeholder cube."""
    return float(atom.npd_half_edge) * _NPD_BOUND_FACTOR


def _next_disorder_label(base_label: str, used: set[str]) -> str:
    """Deprecated alias of
    :func:`fastmolwidget.disorder_drag.next_disorder_label`, kept so existing
    callers and tests importing it from here keep working."""
    from fastmolwidget.disorder_drag import next_disorder_label

    return next_disorder_label(base_label, used)


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class MoleculeWidget3D(DisorderDragMixin, ModelSourceMixin, _WidgetBase):  # type: ignore[valid-type,misc]
    """Real 3-D OpenGL crystal-structure display widget.

    Drop-in replacement for :class:`~fastmolwidget.molecule2D.MoleculeWidget`
    with an identical public API.  Rendering is GPU-accelerated via
    hardware-accelerated OpenGL 3.2 (compatibility) sphere / ellipsoid
    impostors and tessellated cylinder bonds.

    If *PyOpenGL* is unavailable or OpenGL initialisation fails the widget
    gracefully shows an informational message rather than crashing.

    Parameters
    ----------
    parent:
        Optional parent widget.
    """

    atomClicked = QtCore.Signal(str)
    bondClicked = QtCore.Signal(str, str)
    #: Emitted after loading with the frozenset of disorder parts present.
    partsChanged = QtCore.Signal(object)
    #: Emitted when the residual-density contour level changes.
    densityLevelChanged = QtCore.Signal(float)

    # Vertical half-extent multiplier used for orthographic framing.
    _ORTHO_VIEW_MARGIN: float = 1.6

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # ---- Molecule data ------------------------------------------------
        self.atoms: list[_Atom3D] = []
        self.connections: tuple = ()
        self._cell: tuple[float, ...] | None = None
        self._is_packed: bool = False
        self._astar: float = 0.0
        self._bstar: float = 0.0
        self._cstar: float = 0.0
        self._amatrix: np.ndarray = np.eye(3, dtype=float)

        # ---- Public display state (mirrors MoleculeWidget) ----------------
        self.fontsize: int = 18
        self.label_color = QtGui.QColor(100, 50, 5)
        self.bond_width: int = 3
        self.atoms_size: int = 12  # kept for API compatibility
        self.labels: bool = True
        self.show_hydrogens_flag: bool = True
        self.selected_atoms: set[str] = set()
        self.selected_bonds: set[tuple[str, str]] = set()

        # ---- Part filter -------------------------------------------------
        #: Frozenset of all disorder-part numbers in the current atom list.
        self.available_parts: frozenset[int] = frozenset()
        #: Parts to render; ``None`` means *all parts* (no filtering).
        self._visible_parts: set[int] | None = None

        self._show_adps: bool = True

        # ---- 3-D view state -----------------------------------------------
        self._rot_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self._zoom: float = 1.0
        self._pan: np.ndarray = np.zeros(2, dtype=np.float32)
        self._molecule_center: np.ndarray = np.zeros(3, dtype=np.float32)
        self._molecule_radius: float = 10.0
        self.cumulative_R: np.ndarray = np.eye(3, dtype=np.float32)

        self._bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._bond_rgb: tuple[float, float, float] = _DEFAULT_BOND_COLOR

        # ---- OpenGL failure state -----------------------------------------
        _no_pyopengl_msg = (
            "Install PyOpenGL to enable 3D rendering:\n"
            "    pip install PyOpenGL"
        )
        self._gl_failed: bool = not (_HAS_PYOPENGL and _IS_GL_WIDGET)
        self._gl_fail_reason: str = _no_pyopengl_msg if not _HAS_PYOPENGL else (
            "QOpenGLWidget not available in this Qt installation."
            if not _IS_GL_WIDGET else ""
        )
        self._gl_initialized: bool = False

        # ---- GL object handles (allocated in initializeGL) ----------------
        self._sphere_prog: int = 0
        self._ellipsoid_batch_prog: int = 0  # draws all ellipsoids in one call
        self._cylinder_prog: int = 0
        self._vao: int = 0  # scratch VAO (GL 3.0+; skipped gracefully on macOS 2.1)
        self._sphere_vbo: int = 0
        self._sphere_ibo: int = 0
        self._cylinder_vbo: int = 0
        self._cylinder_ibo: int = 0
        self._ellipsoid_batch_vbo: int = 0
        self._ellipsoid_batch_ibo: int = 0
        # NPD-cube placeholder mesh (reuses the cylinder shader: same vertex
        # layout = position3 + normal3 + color3 + selected1).
        self._cube_vbo: int = 0
        self._cube_ibo: int = 0
        # Residual-density isosurface wireframe (positive and negative lobes).
        self._density_vbo: int = 0
        self._density_ibo: int = 0

        # ---- CPU-side geometry buffers ------------------------------------
        self._sphere_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._sphere_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._sphere_count: int = 0
        self._cylinder_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._cylinder_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._cylinder_count: int = 0
        self._ellipsoid_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._ellipsoid_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._ellipsoid_count: int = 0
        self._cube_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._cube_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._cube_count: int = 0

        # ---- Residual-density isosurface ----------------------------------
        #: The computed map, or ``None`` when no density is loaded.
        self._density_map = None
        #: Path of the last loaded structure file, when applicable.
        self._model_path: Path | None = None
        self._density_level: float = 0.30
        self._density_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._density_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        #: Index counts of the positive and negative lobe, in that order.
        self._density_pos_count: int = 0
        self._density_neg_count: int = 0
        self._density_dirty: bool = False

        # ADP atoms for batched ellipsoid draw call
        self._adp_draw_list: list[_Atom3D] = []
        # NPD atoms for batched cube draw call
        self._npd_draw_list: list[_Atom3D] = []

        self._geometry_dirty: bool = False

        # ---- Mouse tracking -----------------------------------------------
        self._lastPos: QtCore.QPointF | None = None
        self._pressPos: QtCore.QPointF | None = None
        self._mouse_moved: bool = False
        # Label of the atom under the cursor, if any.
        self._hover_atom_label: str | None = None
        # Hovered bond, its length in Å, and the label anchor position.
        self._hover_bond: tuple[str, str] | None = None
        self._hover_bond_distance: float | None = None
        self._hover_cursor: QtCore.QPointF | None = None
        # Enable mouse-move events without a button held (for hover detection)
        self.setMouseTracking(True)

        # ---- Interactive disorder-moiety dragging (Ctrl+drag) --------------
        # State and orchestration live in the renderer-independent
        # DisorderDragMixin; only the projection below is 3-D specific.
        self._init_disorder_drag()
        #: Fixed eye-space depth plane the grabbed atom is dragged within.
        self._disorder_drag_eye_z: float = 0.0
        #: Inverse model-view matrix cached for the duration of one drag.
        self._disorder_drag_mv_inv: np.ndarray | None = None

        # ---- Single free-atom dragging (Ctrl+Shift+drag) -------------------
        # ``_single_atom_drag_index`` is owned by DisorderDragMixin; the
        # projection state below is shared with the moiety drag.

        # ---- Widget appearance --------------------------------------------
        # QOpenGLWidget + autoFillBackground would clear the GL buffer.
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtCore.Qt.GlobalColor.white)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setPalette(pal)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Request a compatible surface format for this widget.
        if _IS_GL_WIDGET and not self._gl_failed:
            self._setup_surface_format()

        # Connect default no-op handlers so emits are always safe.
        self.atomClicked.connect(lambda _x: None)
        self.bondClicked.connect(lambda _x, _y: None)

    def paintGL(self):
        if not self._gl_initialized and not self._gl_failed:
            try:
                # paintGL runs with a current context.
                self._do_initializeGL()
                self._gl_initialized = True
                if self._geometry_dirty and self.atoms:
                    self._upload_geometry()
            except Exception as exc:
                self._gl_failed = True
                self._gl_fail_reason = f"OpenGL initialisation failed:\n{exc}"
                print(self._gl_fail_reason)
                # Fallback painting continues.
        if self._gl_failed:
            self._paint_fallback_on_gl()
            return
        self._do_paintGL()

    # ------------------------------------------------------------------
    # Surface format
    # ------------------------------------------------------------------

    def _setup_surface_format(self) -> None:
        """Request a compatibility context with depth buffer and 4× MSAA.

        Compatibility profile is required for the QPainter overlay path.
        """
        try:
            fmt = QtGui.QSurfaceFormat()
            fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
            fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
            if _shaders.MACOS:
                fmt.setVersion(2, 1)
            else:
                fmt.setVersion(3, 2)
            fmt.setDepthBufferSize(24)
            fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
            fmt.setSamples(4)
            # setFormat exists on QOpenGLWidget, not QWidget.
            if hasattr(self, "setFormat"):
                self.setFormat(fmt)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # QOpenGLWidget interface
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        """Called once by Qt after the OpenGL context has been created."""
        if self._gl_failed:
            return
        try:
            self._do_initializeGL()
            self._gl_initialized = True
            if self._geometry_dirty and self.atoms:
                self._upload_geometry()
        except Exception as exc:
            self._gl_failed = True
            self._gl_fail_reason = f"OpenGL initialisation failed:\n{exc}"
            print(f"[MoleculeWidget3D] {self._gl_fail_reason}")

    def _do_initializeGL(self) -> None:
        """Actual GL setup – any exception disables 3-D rendering."""
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        try:
            gl.glEnable(gl.GL_MULTISAMPLE)
        except Exception:
            pass  # not fatal

        # Use one scratch VAO when available; macOS 2.1 may not support it.
        try:
            self._vao = int(gl.glGenVertexArrays(1))
            gl.glBindVertexArray(self._vao)
        except Exception as vao_exc:
            self._vao = 0
            print(f"[MoleculeWidget3D] VAO unavailable, using default VAO 0: {vao_exc}")

        # Compile shaders.
        self._sphere_prog = self._compile_program(
            _SPHERE_VERT, _SPHERE_FRAG, "sphere"
        )
        self._ellipsoid_batch_prog = self._compile_program(
            _ELLIPSOID_BATCH_VERT, _ELLIPSOID_BATCH_FRAG, "ellipsoid_batch"
        )
        self._cylinder_prog = self._compile_program(
            _CYLINDER_VERT, _CYLINDER_FRAG, "cylinder"
        )
        self._line_prog = self._compile_program(
            _LINE_VERT, _LINE_FRAG, "line"
        )

        # Allocate VBOs / IBOs.
        buffers = gl.glGenBuffers(10)
        (
            self._sphere_vbo, self._sphere_ibo,
            self._cylinder_vbo, self._cylinder_ibo,
            self._ellipsoid_batch_vbo, self._ellipsoid_batch_ibo,
            self._cube_vbo, self._cube_ibo,
            self._density_vbo, self._density_ibo,
        ) = buffers

        # Warn if the driver fell back to no MSAA.
        try:
            samples = int(self.format().samples())
            if samples < 2:
                print(f"[MoleculeWidget3D] OpenGL surface has samples={samples} (no MSAA).")
        except Exception:
            pass

    def _compile_program(self, vert_src: str, frag_src: str, name: str) -> int:
        """Compile and link a shader program."""
        try:
            vert = _glshaders.compileShader(vert_src, gl.GL_VERTEX_SHADER)
            frag = _glshaders.compileShader(frag_src, gl.GL_FRAGMENT_SHADER)
            # Skip validation here; the draw FBO may not be ready yet.
            try:
                prog = _glshaders.compileProgram(vert, frag, validate=False)
            except TypeError:
                # Older PyOpenGL lacks the validate kwarg.
                prog = _glshaders.compileProgram(vert, frag)
            return int(prog)
        except Exception as exc:
            raise RuntimeError(f"Failed to compile '{name}' shaders: {exc}") from exc

    def resizeGL(self, w: int, h: int) -> None:
        """Called by Qt when the widget is resized."""
        if self._gl_failed:
            return
        try:
            gl.glViewport(0, 0, max(1, w), max(1, h))
        except Exception:
            pass

    def paintGL(self) -> None:
        """Called by Qt to render the scene."""
        if self._gl_failed:
            self._paint_fallback_on_gl()
            return
        try:
            self._do_paintGL()
        except Exception as exc:
            print(f"[MoleculeWidget3D] paintGL error (continuing): {exc}")

    def _do_paintGL(self) -> None:
        if self._geometry_dirty and self.atoms:
            self._upload_geometry()

        # QPainter mutates GL state; restore what the renderer expects.
        self._reassert_gl_state()

        r, g, b = self._bg_rgb
        gl.glClearColor(r, g, b, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if not self.atoms:
            return

        mv = self._compute_mv_matrix()
        proj = self._compute_proj_matrix()

        # Bonds first.
        if self._cylinder_count > 0:
            self._render_cylinders(mv, proj)

        # Regular atom spheres.
        if self._sphere_count > 0:
            self._render_spheres(mv, proj)

        # NPD placeholders stay visible in both display modes.
        if self._cube_count > 0:
            self._render_cubes(mv, proj)

        # ADP ellipsoids.
        if self._show_adps and self._ellipsoid_count > 0:
            self._render_ellipsoids_batched(mv, proj)

        # Draw the density wireframe last.
        if self._density_dirty:
            self._upload_density_geometry()
        if self._density_pos_count or self._density_neg_count:
            self._render_density(mv, proj)

        # Build the 2-D overlay off-screen; Qt's GL painter is unreliable here.
        overlay = self._compose_overlay_image(mv, proj)
        painter = QtGui.QPainter(self)
        try:
            if overlay is not None:
                painter.drawImage(0, 0, overlay)
        finally:
            painter.end()

    def _reassert_gl_state(self) -> None:
        """Restore the GL state expected by the renderer."""
        try:
            gl.glEnable(gl.GL_DEPTH_TEST)
            gl.glDepthFunc(gl.GL_LEQUAL)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glDisable(gl.GL_CULL_FACE)
            try:
                gl.glEnable(gl.GL_MULTISAMPLE)
            except Exception:
                pass  # not fatal on contexts without MSAA
            # QPainter may leave a different VAO bound.
            if self._vao:
                gl.glBindVertexArray(self._vao)
            dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
            w = max(1, int(self.width() * dpr))
            h = max(1, int(self.height() * dpr))
            gl.glViewport(0, 0, w, h)
        except Exception:
            # Never let GL state recovery crash the host app.
            pass

    # ------------------------------------------------------------------
    # paintEvent: OpenGL path or QPainter fallback
    # ------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        """Route paint requests to the appropriate renderer."""
        if _IS_GL_WIDGET:
            # Let QOpenGLWidget handle makeCurrent → paintGL → doneCurrent.
            try:
                super().paintEvent(event)
            except Exception:
                painter = QtGui.QPainter(self)
                self._draw_fallback_text(painter)
                painter.end()
        else:
            # Pure QWidget fallback: no OpenGL context.
            painter = QtGui.QPainter(self)
            self._draw_fallback_text(painter)
            painter.end()

    def _paint_fallback_on_gl(self) -> None:
        """Draw a text fallback when GL is unavailable (inside a GL context)."""
        try:
            gl.glClearColor(0.94, 0.94, 0.94, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        except Exception:
            pass
        painter = QtGui.QPainter(self)
        self._draw_fallback_text(painter)
        painter.end()

    def _draw_fallback_text(self, painter: QtGui.QPainter) -> None:
        painter.fillRect(self.rect(), QtGui.QColor(240, 240, 240))
        painter.setPen(QtGui.QColor(80, 80, 80))
        msg = (
            "3D OpenGL rendering unavailable.\n"
            + self._gl_fail_reason
        )
        painter.drawText(
            self.rect(), Qt.AlignmentFlag.AlignCenter, msg
        )
        print(self._gl_fail_reason)

    # ------------------------------------------------------------------
    # Geometry building
    # ------------------------------------------------------------------

    def _build_geometry(self) -> None:
        """(Re)build all CPU-side geometry from :attr:`atoms`."""
        self._build_sphere_geometry()
        self._build_ellipsoid_geometry_batched()
        self._build_cube_geometry()
        self._build_cylinder_geometry()
        self._geometry_dirty = True

    def _atom_color(self, atom: _Atom3D) -> tuple[float, float, float]:
        """The colour *atom* is drawn in, honouring selection and its part.

        A selected atom always wins (it has to stay obvious), otherwise the
        element colour is washed out according to the atom's disorder part so
        the parts of a split can be told apart - see :func:`_part_fade`.
        """
        if atom.label in self.selected_atoms:
            return _SEL_COLOR
        fade = _part_fade(atom.part)
        if fade <= 0.0:
            return atom.color_f
        return _fade_towards_white(atom.color_f, fade)

    def _bond_color(self, at1: _Atom3D, at2: _Atom3D) -> tuple[float, float, float]:
        """The colour the bond between *at1* and *at2* is drawn in.

        Faded by whichever of the two atoms belongs to the later disorder
        part, so a bond joining a shared atom to a part-2 copy reads as
        belonging to that part rather than to the undisordered backbone.
        """
        fade = _part_fade(max(at1.part, at2.part))
        if fade <= 0.0:
            return self._bond_rgb
        return _fade_towards_white(self._bond_rgb, fade)

    def _build_sphere_geometry(self) -> None:
        """Create billboard quad data for atoms rendered as spheres."""
        corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        quad_idx_tpl = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

        self._adp_draw_list = []
        self._npd_draw_list = []
        sphere_atoms: list[_Atom3D] = []

        for atom in self.atoms:
            if not self.show_hydrogens_flag and atom.type_ in ("H", "D"):
                continue
            if self._visible_parts is not None and atom.part not in self._visible_parts:
                continue
            if atom.u_cart is not None and not atom.adp_valid:
                self._npd_draw_list.append(atom)
            elif self._show_adps and atom.u_cart is not None:
                self._adp_draw_list.append(atom)
            else:
                sphere_atoms.append(atom)

        n = len(sphere_atoms)
        if n == 0:
            self._sphere_verts = np.empty(0, dtype=np.float32)
            self._sphere_idx = np.empty(0, dtype=np.uint32)
            self._sphere_count = 0
            return

        # Vertex layout: [cx, cy, cz, r, g, b, radius, corner_x, corner_y, selected]
        # 10 floats per vertex, 4 vertices per atom, 6 indices per atom
        verts = np.zeros((n * 4, 10), dtype=np.float32)
        idx = np.zeros(n * 6, dtype=np.uint32)

        for i, atom in enumerate(sphere_atoms):
            c = atom.center
            is_selected = atom.label in self.selected_atoms
            col = self._atom_color(atom)
            sel_flag = 1.0 if is_selected else 0.0
            r = (
                sqrt(atom.u_iso) * _ADP_SCALE
                if self._show_adps and atom.u_iso is not None
                else atom.display_radius
            )
            for j in range(4):
                vi = i * 4 + j
                verts[vi, 0:3] = c
                verts[vi, 3:6] = col
                verts[vi, 6] = r
                verts[vi, 7:9] = corners[j]
                verts[vi, 9] = sel_flag
            idx[i * 6: i * 6 + 6] = quad_idx_tpl + i * 4

        self._sphere_verts = verts.ravel()
        self._sphere_idx = idx
        self._sphere_count = n * 6

    def _build_cylinder_geometry(self) -> None:
        """Build tessellated cylinder meshes for all bonds."""
        n_seg = 20  # 20-segment cylinders avoid visible Gouraud facet bands
        # base cylinder radius, scaled by bond_width
        cyl_r = 0.016 * max(0, self.bond_width)

        all_verts: list[np.ndarray] = []
        all_idx: list[np.ndarray] = []
        v_offset = 0

        for n1, n2 in self.connections:
            at1 = self.atoms[n1]
            at2 = self.atoms[n2]

            if not self.show_hydrogens_flag:
                if at1.type_ in ("H", "D") or at2.type_ in ("H", "D"):
                    continue
            if self._visible_parts is not None:
                if at1.part not in self._visible_parts or at2.part not in self._visible_parts:
                    continue

            bond_key: tuple[str, str] = tuple(sorted((at1.label, at2.label)))  # type: ignore[assignment]
            is_selected = bond_key in self.selected_bonds
            if is_selected:
                bond_color = _SEL_COLOR
            else:
                bond_color = self._bond_color(at1, at2)

            verts, bond_idx = _make_cylinder(
                at1.center, at2.center, cyl_r, bond_color, n_seg,
                selected=is_selected,
            )
            if verts is None:
                continue

            all_verts.append(verts)
            all_idx.append(bond_idx + v_offset)
            v_offset += len(verts)

        if all_verts:
            self._cylinder_verts = np.concatenate(all_verts, axis=0).ravel()
            self._cylinder_idx = np.concatenate(all_idx)
            self._cylinder_count = int(len(self._cylinder_idx))
        else:
            self._cylinder_verts = np.empty(0, dtype=np.float32)
            self._cylinder_idx = np.empty(0, dtype=np.uint32)
            self._cylinder_count = 0

    def _build_ellipsoid_geometry_batched(self) -> None:
        """Pack all ADP ellipsoids into a single VBO for one-call rendering.

        Vertex layout: 28 floats per vertex, 4 vertices per atom.
        See ``_ELLIPSOID_BATCH_VERT`` for the attribute offsets.
        Must be called **after** :meth:`_build_sphere_geometry` which populates
        :attr:`_adp_draw_list`.
        """
        atoms = [
            a for a in self._adp_draw_list
            if a.adp_A_matrix is not None and a.u_eigvecs is not None
        ]
        n = len(atoms)
        if n == 0:
            self._ellipsoid_verts = np.empty(0, dtype=np.float32)
            self._ellipsoid_idx = np.empty(0, dtype=np.uint32)
            self._ellipsoid_count = 0
            return

        # ── per-atom data (vectorised) ──────────────────────────────────────
        _corners = np.array(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        # Repeat corner pattern for n atoms: shape (n*4, 2)
        corners_tiled = np.tile(_corners, (n, 1))

        # Per-atom arrays – each repeated 4× to fill all 4 quad vertices
        centers = np.repeat(
            np.array([a.center for a in atoms], dtype=np.float32), 4, axis=0
        )
        colors = np.repeat(
            np.array(
                [self._atom_color(a) for a in atoms],
                dtype=np.float32,
            ),
            4, axis=0,
        )
        radii = np.repeat(
            np.array([a.adp_billboard_r for a in atoms], dtype=np.float32), 4
        )
        sel_flags = np.repeat(
            np.array(
                [1.0 if a.label in self.selected_atoms else 0.0 for a in atoms],
                dtype=np.float32,
            ),
            4,
        )
        # Columns of the A-matrix and eigenvector matrix
        # mat3(col0, col1, col2) in GLSL uses column-major order → A[:, j]
        A_col0 = np.repeat(
            np.array([a.adp_A_matrix[:, 0] for a in atoms], dtype=np.float32), 4, axis=0
        )
        A_col1 = np.repeat(
            np.array([a.adp_A_matrix[:, 1] for a in atoms], dtype=np.float32), 4, axis=0
        )
        A_col2 = np.repeat(
            np.array([a.adp_A_matrix[:, 2] for a in atoms], dtype=np.float32), 4, axis=0
        )
        evec0 = np.repeat(
            np.array([a.u_eigvecs[:, 0] for a in atoms], dtype=np.float32), 4, axis=0
        )
        evec1 = np.repeat(
            np.array([a.u_eigvecs[:, 1] for a in atoms], dtype=np.float32), 4, axis=0
        )
        evec2 = np.repeat(
            np.array([a.u_eigvecs[:, 2] for a in atoms], dtype=np.float32), 4, axis=0
        )

        # Assemble interleaved VBO: (n*4, 28) → ravel
        verts = np.hstack([
            corners_tiled,  # 2
            centers,  # 3
            colors,  # 3
            radii[:, None],  # 1
            sel_flags[:, None],  # 1
            A_col0, A_col1, A_col2,  # 9
            evec0, evec1, evec2,  # 9
        ])  # → (n*4, 28)

        # Indices: 6 per atom quad
        quad_tpl = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        offsets = np.arange(n, dtype=np.uint32) * 4
        idx = (quad_tpl[None, :] + offsets[:, None]).ravel()

        self._ellipsoid_verts = verts.astype(np.float32).ravel()
        self._ellipsoid_idx = idx
        self._ellipsoid_count = int(len(idx))

    def _build_cube_geometry(self) -> None:
        """Build one tessellated cube placeholder per NPD atom."""
        atoms = self._npd_draw_list
        if not atoms:
            self._cube_verts = np.empty(0, dtype=np.float32)
            self._cube_idx = np.empty(0, dtype=np.uint32)
            self._cube_count = 0
            return

        # 6 face normals (outward).  Each face has 4 unique vertices so
        # that each face can carry its own normal (flat shading).
        face_normals = np.array([
            (0.0, 0.0, 1.0),  # +Z (front)
            (0.0, 0.0, -1.0),  # -Z (back)
            (1.0, 0.0, 0.0),  # +X (right)
            (-1.0, 0.0, 0.0),  # -X (left)
            (0.0, 1.0, 0.0),  # +Y (top)
            (0.0, -1.0, 0.0),  # -Y (bottom)
        ], dtype=np.float32)

        # Local-space corner offsets (half-edge = 1) for each face, in
        # CCW order when viewed from outside.
        face_corners = np.array([
            # +Z
            [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
            # -Z
            [(1, -1, -1), (-1, -1, -1), (-1, 1, -1), (1, 1, -1)],
            # +X
            [(1, -1, 1), (1, -1, -1), (1, 1, -1), (1, 1, 1)],
            # -X
            [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
            # +Y
            [(-1, 1, 1), (1, 1, 1), (1, 1, -1), (-1, 1, -1)],
            # -Y
            [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
        ], dtype=np.float32)

        # Per-face quad → 2 triangles (CCW)
        face_quad = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

        n = len(atoms)
        verts = np.zeros((n * 24, 10), dtype=np.float32)
        idx = np.zeros(n * 36, dtype=np.uint32)

        for ai, atom in enumerate(atoms):
            half = float(atom.npd_half_edge) if atom.npd_half_edge > 0.0 \
                else 0.5 * float(atom.display_radius)
            is_selected = atom.label in self.selected_atoms
            color = self._atom_color(atom)
            sel_flag = 1.0 if is_selected else 0.0
            base_v = ai * 24
            base_i = ai * 36

            for f in range(6):
                normal = face_normals[f]
                for c in range(4):
                    vi = base_v + f * 4 + c
                    verts[vi, 0:3] = atom.center + face_corners[f, c] * half
                    verts[vi, 3:6] = normal
                    verts[vi, 6:9] = color
                    verts[vi, 9] = sel_flag
                idx[base_i + f * 6: base_i + f * 6 + 6] = (
                    face_quad + base_v + f * 4
                )

        self._cube_verts = verts.ravel()
        self._cube_idx = idx
        self._cube_count = int(len(idx))

    def _upload_geometry(self) -> None:
        """Upload CPU geometry arrays to GPU VBOs."""
        if not self._gl_initialized or self._gl_failed:
            return

        if self._sphere_verts.size > 0:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sphere_vbo)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                self._sphere_verts.nbytes,
                self._sphere_verts,
                gl.GL_DYNAMIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._sphere_ibo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER,
                self._sphere_idx.nbytes,
                self._sphere_idx,
                gl.GL_DYNAMIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        if self._cylinder_verts.size > 0:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._cylinder_vbo)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                self._cylinder_verts.nbytes,
                self._cylinder_verts,
                gl.GL_DYNAMIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._cylinder_ibo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER,
                self._cylinder_idx.nbytes,
                self._cylinder_idx,
                gl.GL_DYNAMIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        if self._ellipsoid_verts.size > 0:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._ellipsoid_batch_vbo)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                self._ellipsoid_verts.nbytes,
                self._ellipsoid_verts,
                gl.GL_DYNAMIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ellipsoid_batch_ibo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER,
                self._ellipsoid_idx.nbytes,
                self._ellipsoid_idx,
                gl.GL_DYNAMIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        if self._cube_verts.size > 0:
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._cube_vbo)
            gl.glBufferData(
                gl.GL_ARRAY_BUFFER,
                self._cube_verts.nbytes,
                self._cube_verts,
                gl.GL_DYNAMIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._cube_ibo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER,
                self._cube_idx.nbytes,
                self._cube_idx,
                gl.GL_DYNAMIC_DRAW,
            )
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

        self._geometry_dirty = False

    def _upload_density_geometry(self) -> None:
        """Upload the residual-density wireframe to its VBO / IBO."""
        if not self._gl_initialized or self._gl_failed:
            return
        self._density_dirty = False
        if self._density_verts.size == 0 or self._density_idx.size == 0:
            return

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._density_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER,
            self._density_verts.nbytes,
            self._density_verts,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._density_ibo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            self._density_idx.nbytes,
            self._density_idx,
            gl.GL_DYNAMIC_DRAW,
        )
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_spheres(self, mv: np.ndarray, proj: np.ndarray) -> None:
        prog = self._sphere_prog
        gl.glUseProgram(prog)

        _set_mat4(prog, b"u_mv", mv)
        _set_mat4(prog, b"u_proj", proj)

        stride = 10 * 4  # 10 floats × 4 bytes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sphere_vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._sphere_ibo)

        _bind_attrib(prog, b"a_center", 3, stride, 0)
        _bind_attrib(prog, b"a_color", 3, stride, 12)
        _bind_attrib(prog, b"a_radius", 1, stride, 24)
        _bind_attrib(prog, b"a_corner", 2, stride, 28)
        _bind_attrib(prog, b"a_selected", 1, stride, 36)

        gl.glDrawElements(
            gl.GL_TRIANGLES, self._sphere_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0)
        )

        _unbind_attrib(prog, [b"a_center", b"a_color", b"a_radius", b"a_corner", b"a_selected"])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)

    def _render_cylinders(self, mv: np.ndarray, proj: np.ndarray) -> None:
        prog = self._cylinder_prog
        gl.glUseProgram(prog)

        _set_mat4(prog, b"u_mv", mv)
        _set_mat4(prog, b"u_proj", proj)

        # Normal matrix = inverse-transpose of the upper-left 3×3 of MV
        try:
            nm = np.linalg.inv(mv[:3, :3]).T.astype(np.float32)
        except np.linalg.LinAlgError:
            nm = np.eye(3, dtype=np.float32)
        loc_nm = gl.glGetUniformLocation(prog, b"u_normal_mat")
        if loc_nm >= 0:
            gl.glUniformMatrix3fv(loc_nm, 1, False, nm.T.copy())

        stride = 10 * 4  # 10 floats × 4 bytes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._cylinder_vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._cylinder_ibo)

        _bind_attrib(prog, b"a_position", 3, stride, 0)
        _bind_attrib(prog, b"a_normal", 3, stride, 12)
        _bind_attrib(prog, b"a_color", 3, stride, 24)
        _bind_attrib(prog, b"a_selected", 1, stride, 36)

        gl.glDrawElements(
            gl.GL_TRIANGLES, self._cylinder_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0)
        )

        _unbind_attrib(prog, [b"a_position", b"a_normal", b"a_color", b"a_selected"])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)

    def _render_ellipsoids_batched(self, mv: np.ndarray, proj: np.ndarray) -> None:
        """Render all ADP ellipsoids with a **single** glDrawElements call."""
        prog = self._ellipsoid_batch_prog
        gl.glUseProgram(prog)

        _set_mat4(prog, b"u_mv", mv)
        _set_mat4(prog, b"u_proj", proj)

        stride = 28 * 4  # 28 floats × 4 bytes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._ellipsoid_batch_vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ellipsoid_batch_ibo)

        _bind_attrib(prog, b"a_corner", 2, stride, 0)
        _bind_attrib(prog, b"a_center", 3, stride, 8)
        _bind_attrib(prog, b"a_color", 3, stride, 20)
        _bind_attrib(prog, b"a_radius", 1, stride, 32)
        _bind_attrib(prog, b"a_selected", 1, stride, 36)
        _bind_attrib(prog, b"a_A_col0", 3, stride, 40)
        _bind_attrib(prog, b"a_A_col1", 3, stride, 52)
        _bind_attrib(prog, b"a_A_col2", 3, stride, 64)
        _bind_attrib(prog, b"a_evec0", 3, stride, 76)
        _bind_attrib(prog, b"a_evec1", 3, stride, 88)
        _bind_attrib(prog, b"a_evec2", 3, stride, 100)

        gl.glDrawElements(
            gl.GL_TRIANGLES, self._ellipsoid_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0)
        )

        _unbind_attrib(prog, [
            b"a_corner", b"a_center", b"a_color", b"a_radius", b"a_selected",
            b"a_A_col0", b"a_A_col1", b"a_A_col2",
            b"a_evec0", b"a_evec1", b"a_evec2",
        ])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)

    def _render_cubes(self, mv: np.ndarray, proj: np.ndarray) -> None:
        """Render NPD-cube placeholders using the cylinder shader.

        Cubes share the cylinder vertex layout (position3 + normal3 + color3
        + selected1, 10 floats / vert) so a dedicated shader is unnecessary.
        """
        prog = self._cylinder_prog
        gl.glUseProgram(prog)

        _set_mat4(prog, b"u_mv", mv)
        _set_mat4(prog, b"u_proj", proj)

        try:
            nm = np.linalg.inv(mv[:3, :3]).T.astype(np.float32)
        except np.linalg.LinAlgError:
            nm = np.eye(3, dtype=np.float32)
        loc_nm = gl.glGetUniformLocation(prog, b"u_normal_mat")
        if loc_nm >= 0:
            gl.glUniformMatrix3fv(loc_nm, 1, False, nm.T.copy())

        stride = 10 * 4  # 10 floats × 4 bytes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._cube_vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._cube_ibo)

        _bind_attrib(prog, b"a_position", 3, stride, 0)
        _bind_attrib(prog, b"a_normal", 3, stride, 12)
        _bind_attrib(prog, b"a_color", 3, stride, 24)
        _bind_attrib(prog, b"a_selected", 1, stride, 36)

        gl.glDrawElements(
            gl.GL_TRIANGLES, self._cube_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0)
        )

        _unbind_attrib(prog, [b"a_position", b"a_normal", b"a_color", b"a_selected"])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)

    def _render_density(self, mv: np.ndarray, proj: np.ndarray) -> None:
        """Draw the residual-density isosurface as a coloured wireframe.

        Both lobes live in one VBO; the positive lobe occupies the first
        ``_density_pos_count`` indices and the negative lobe the rest, so a
        single buffer binding serves two draw calls that differ only in the
        ``u_color`` uniform.
        """
        prog = self._line_prog
        gl.glUseProgram(prog)

        _set_mat4(prog, b"u_mv", mv)
        _set_mat4(prog, b"u_proj", proj)

        stride = 3 * 4  # position3 × 4 bytes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._density_vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._density_ibo)
        _bind_attrib(prog, b"a_position", 3, stride, 0)

        if self._density_pos_count:
            _set_vec3(prog, b"u_color", np.array(_DENSITY_POS_COLOR, dtype=np.float32))
            gl.glDrawElements(
                gl.GL_LINES, self._density_pos_count,
                gl.GL_UNSIGNED_INT, ctypes.c_void_p(0),
            )
        if self._density_neg_count:
            _set_vec3(prog, b"u_color", np.array(_DENSITY_NEG_COLOR, dtype=np.float32))
            gl.glDrawElements(
                gl.GL_LINES, self._density_neg_count, gl.GL_UNSIGNED_INT,
                ctypes.c_void_p(self._density_pos_count * 4),
            )

        _unbind_attrib(prog, [b"a_position"])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)

    def _draw_labels_overlay(self, mv: np.ndarray, proj: np.ndarray) -> None:
        """Draw the QPainter overlay used by the fallback path."""
        if not self.atoms:
            return
        overlay = self._compose_overlay_image(mv, proj)
        if overlay is None:
            return
        painter = QtGui.QPainter(self)
        try:
            painter.drawImage(0, 0, overlay)
        finally:
            painter.end()

    def _compose_overlay_image(
        self,
        mv: np.ndarray,
        proj: np.ndarray,
    ) -> QtGui.QImage | None:
        """Render the full 2-D overlay into a transparent QImage."""
        w = max(1, int(self.width()))
        h = max(1, int(self.height()))
        if w <= 0 or h <= 0:
            return None

        # Render at device resolution so high-DPI text stays crisp.
        dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
        if dpr <= 0.0:
            dpr = 1.0
        pw = max(1, int(round(w * dpr)))
        ph = max(1, int(round(h * dpr)))

        # Premultiplied alpha is best for compositing.
        image = QtGui.QImage(pw, ph, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(dpr)
        image.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QtGui.QPainter(image)
        try:
            painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            painter.setRenderHint(QtGui.QPainter.RenderHint.SmoothPixmapTransform, True)
            # The devicePixelRatio keeps painter coordinates logical.
            self._draw_labels_with_painter(painter, mv, proj)
            if self._is_packed:
                self._draw_axis_indicator(painter)
        finally:
            painter.end()
        return image

    def _draw_labels_with_painter(
        self,
        painter: QtGui.QPainter,
        mv: np.ndarray,
        proj: np.ndarray,
    ) -> None:
        """Draw atom labels using an already-active ``QPainter``."""
        if not self.atoms:
            return

        w = max(1, self.width())
        h = max(1, self.height())
        hydrogens = ("H", "D")

        base_size = max(1, int(self.fontsize))
        hover_size = base_size + 4  # enlarge hovered label

        font = QtGui.QFont()
        font.setPixelSize(base_size)
        painter.setFont(font)
        painter.setPen(self.label_color)

        hover_label = self._hover_atom_label
        hover_atom = None

        # Label offsets scale with zoom.
        _, half_h = self._ortho_half_extents()
        px_per_angstrom = (h * 0.5) / half_h if half_h > 1e-8 else 1.0

        # Rotation part of the model-view matrix.
        R_view = np.asarray(mv[:3, :3], dtype=np.float64)

        # Label offset direction in view-space xy.
        _label_dir2 = np.array([1.0, 1.0], dtype=np.float64) / sqrt(2.0)

        def _atom_screen_radius(atom: _Atom3D) -> float:
            """Return the screen-space radius of *atom* along the label direction."""
            if self._show_adps and atom.u_cart is not None and atom.adp_valid:
                # 50 % ORTEP covariance in world space.
                C_world = np.asarray(atom.u_cart, dtype=np.float64)
                # The 2x2 marginal gives the orthographic silhouette.
                C_view = R_view @ C_world @ R_view.T
                C2 = C_view[:2, :2]
                quad = float(_label_dir2 @ C2 @ _label_dir2)
                r = sqrt(max(quad, 0.0))
            elif atom.u_cart is not None and not atom.adp_valid and atom.npd_half_edge > 0.0:
                # ``u_iso`` may be negative here, so use the cube bound radius.
                r = _npd_bound_radius(atom)
            elif self._show_adps and atom.u_iso is not None:
                r = sqrt(atom.u_iso)
            else:
                r = atom.display_radius
            return r * px_per_angstrom

        def project(atom: _Atom3D) -> tuple[int, int] | None:
            pos4 = np.array([*atom.center, 1.0], dtype=np.float32)
            eye = mv @ pos4
            clip = proj @ eye
            if abs(clip[3]) < 1e-8:
                return None
            ndc = clip[:3] / clip[3]
            if not (
                -1.0 <= ndc[0] <= 1.0
                and -1.0 <= ndc[1] <= 1.0
                and -1.0 <= ndc[2] <= 1.0
            ):
                return None
            return (
                int((ndc[0] + 1.01) * 0.5 * w),
                int((1.01 - ndc[1]) * 0.5 * h),
            )

        # Persistent labels (only when "Show Labels" is on). Hidden hydrogens
        # never get a label, and the hovered atom is drawn separately below
        # with a larger font.
        if self.labels:
            for atom in self.atoms:
                if not self.show_hydrogens_flag and atom.type_ in hydrogens:
                    continue
                if self._visible_parts is not None and atom.part not in self._visible_parts:
                    continue
                if atom.label == hover_label:
                    hover_atom = atom
                    continue
                pt = project(atom)
                if pt is None:
                    continue
                offset = int(_atom_screen_radius(atom))
                painter.drawText(pt[0] + offset, pt[1] - offset, atom.label)
        elif hover_label is not None:
            for atom in self.atoms:
                if self._visible_parts is not None and atom.part not in self._visible_parts:
                    continue
                if atom.label == hover_label:
                    hover_atom = atom
                    break

        # Hover label – enlarged. Only draw if the hovered atom is actually
        # displayed (hydrogens are filtered out by _pick_atom_at when hidden,
        # but we double-check here for safety).
        if hover_atom is not None:
            if not self.show_hydrogens_flag and hover_atom.type_ in hydrogens:
                return
            pt = project(hover_atom)
            if pt is None:
                return
            hover_font = QtGui.QFont(font)
            hover_font.setPixelSize(hover_size)
            hover_font.setBold(True)
            painter.setFont(hover_font)
            offset = int(_atom_screen_radius(hover_atom))
            painter.drawText(pt[0] + offset, pt[1] - offset, hover_atom.label)

        # Bond-distance hover label (only when no atom is hovered).
        if (
            hover_atom is None
            and self._hover_bond is not None
            and self._hover_bond_distance is not None
            and self._hover_cursor is not None
        ):
            self._draw_hover_distance_label(
                painter,
                f"{self._hover_bond_distance:.3f} Å",
                self._hover_cursor.x(),
                self._hover_cursor.y(),
            )

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------

    def _compute_mv_matrix(self) -> np.ndarray:
        """Build the Model-View matrix from current rotation / zoom / pan."""
        dist = max(self._molecule_radius * 3.0, 3.0)

        # Step 1 – translate molecule centre to world origin
        T_centre = np.eye(4, dtype=np.float32)
        T_centre[0, 3] = -self._molecule_center[0]
        T_centre[1, 3] = -self._molecule_center[1]
        T_centre[2, 3] = -self._molecule_center[2]

        # Step 2 – apply accumulated rotation
        R = np.eye(4, dtype=np.float32)
        R[:3, :3] = self._rot_matrix

        # Step 3 – pan in view space
        T_pan = np.eye(4, dtype=np.float32)
        T_pan[0, 3] = self._pan[0]
        T_pan[1, 3] = self._pan[1]

        # Step 4 – pull camera back along -Z
        T_cam = np.eye(4, dtype=np.float32)
        T_cam[2, 3] = -dist

        return (T_cam @ T_pan @ R @ T_centre).astype(np.float32)

    def _ortho_half_extents(self) -> tuple[float, float]:
        """Return orthographic half-width/half-height in view-space units."""
        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / h
        half_h = max(
            self._molecule_radius * self._ORTHO_VIEW_MARGIN / max(self._zoom * 2, 0.01),
            0.5,
        )
        half_w = half_h * aspect
        return half_w, half_h

    def _compute_proj_matrix(self) -> np.ndarray:
        """Build an orthographic projection matrix."""
        half_w, half_h = self._ortho_half_extents()
        near, far = 0.01, 10000.0
        return np.array(
            [
                [1.0 / half_w, 0.0, 0.0, 0.0],
                [0.0, 1.0 / half_h, 0.0, 0.0],
                [0.0, 0.0, 2.0 / (near - far), (far + near) / (near - far)],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Molecule loading
    # ------------------------------------------------------------------

    def open_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None = None,
        keep_view: bool = False,
    ) -> None:
        """Load a new molecule and (unless *keep_view*) reset the view."""
        self._is_packed = False
        self._load_molecule(atoms, cell, keep_view=keep_view)

    def grow_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None = None,
    ) -> None:
        """Update the displayed molecule while preserving the current view."""
        self._load_molecule(atoms, cell, keep_view=True)

    def _load_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None,
        keep_view: bool,
    ) -> None:
        self._cell = cell

        if self._cell is not None:
            self.calc_amatrix()

        # Build internal atom list with optional ADP tensors
        self.atoms = []
        name_counts: dict[str, int] = {}

        # A freshly loaded/reloaded atom list invalidates every atom index,
        # so any moiety-drag split state from before must be discarded too.
        self._reset_disorder_split()

        for at in atoms:
            base_name = at.label
            count = name_counts.get(base_name, 0)
            internal_name = base_name if count == 0 else f"{base_name}>>{count}"
            name_counts[base_name] = count + 1

            a3d = _Atom3D(at.x, at.y, at.z, internal_name, at.type, at.part)
            symm = getattr(at, "symm_matrix", None)
            if symm is not None:
                symm_np = np.array(symm, dtype=float)
                a3d.symmgen = not np.allclose(symm_np, np.eye(3))
            else:
                a3d.symmgen = False

            # Anisotropic H atoms keep their ADP tensor.
            adp_vals = getattr(at, "adp", None)
            if adp_vals is not None and self._cell:
                try:
                    symm_arr = np.array(symm, dtype=float) if symm is not None else None
                    a3d.u_cart = self._uij_to_cart(adp_vals, symm_arr)
                    a3d.u_iso = float(np.trace(a3d.u_cart) / 3.0)
                    evals, evecs = np.linalg.eigh(a3d.u_cart)
                    a3d.u_eigvals = evals
                    a3d.u_eigvecs = evecs
                    if np.any(evals <= 0):
                        # Use a compact cube placeholder for NPD tensors.
                        a3d.adp_valid = False
                        a3d.npd_half_edge = float(
                            0.4 * _ADP_SCALE * np.sqrt(np.max(np.abs(evals)))
                        )
                    else:
                        a3d.adp_valid = True
                        # Billboard radius for the ellipsoid quad.
                        a3d.adp_billboard_r = float(_ADP_SCALE * np.sqrt(np.max(evals)) * 1.2)
                        A = np.linalg.inv(_ADP_SCALE ** 2 * a3d.u_cart)
                        a3d.adp_A_matrix = A.astype(np.float32)
                except Exception:
                    a3d.u_cart = None
                    a3d.u_iso = None
                    a3d.adp_valid = False

            self.atoms.append(a3d)

        self.connections = self._get_conntable()

        self.available_parts = frozenset(a.part for a in self.atoms)
        self._visible_parts = None

        if not keep_view:
            self._compute_molecule_bounds()
            self._rot_matrix = np.eye(3, dtype=np.float32)
            self.cumulative_R = np.eye(3, dtype=np.float32)
            self._zoom = 1.0
            self._pan = np.zeros(2, dtype=np.float32)
            self.selected_atoms.clear()
            self.selected_bonds.clear()
        else:
            # Update the centre so grown structures stay in view.
            self._compute_molecule_bounds()

        self._build_geometry()
        # Same-model reloads keep the map but must re-clip it.
        if self._density_map is not None:
            self._build_density_geometry()
        self.update()

        # Emit after geometry is ready.
        self.partsChanged.emit(self.available_parts)

    def _compute_molecule_bounds(self) -> None:
        """Compute the bounding sphere of the current atom set."""
        if not self.atoms:
            self._molecule_center = np.zeros(3, dtype=np.float32)
            self._molecule_radius = 10.0
            return

        coords = np.array([a.center for a in self.atoms], dtype=np.float32)
        mn = coords.min(axis=0)
        mx = coords.max(axis=0)
        self._molecule_center = ((mn + mx) / 2.0).astype(np.float32)
        dists = np.linalg.norm(coords - self._molecule_center, axis=1)
        self._molecule_radius = float(np.max(dists) + 1.5)
        if self._molecule_radius < 1.0:
            self._molecule_radius = 1.0

    def _get_conntable(self, extra_param: float = 1.2) -> tuple:
        """Build a connectivity table from atomic coordinates and covalent radii.

        Delegates to the shared vectorised implementation in
        :func:`fastmolwidget.tools.build_conntable`.
        """
        from fastmolwidget.tools import build_conntable

        coords = np.array([a.center for a in self.atoms], dtype=np.float64)
        types = [a.type_ for a in self.atoms]
        parts = [a.part for a in self.atoms]
        symmgen = [a.symmgen for a in self.atoms]
        return build_conntable(coords, types, parts, extra_param=extra_param, symmgen=symmgen)

    # ------------------------------------------------------------------
    # ADP crystallography helpers  (ported from molecule2D.py)
    # ------------------------------------------------------------------

    def calc_amatrix(self) -> None:
        """Compute the orthogonalisation matrix from the unit-cell parameters."""
        a, b, c, alpha, beta, gamma = self._cell  # type: ignore[misc]
        V = calc_volume(a, b, c, alpha, beta, gamma)
        self._astar = (b * c * sin(radians(alpha))) / V
        self._bstar = (c * a * sin(radians(beta))) / V
        self._cstar = (a * b * sin(radians(gamma))) / V
        self._amatrix = np.array(
            [
                [a, b * cos(radians(gamma)), c * cos(radians(beta))],
                [
                    0,
                    b * sin(radians(gamma)),
                    c
                    * (
                        cos(radians(alpha))
                        - cos(radians(beta)) * cos(radians(gamma))
                    )
                    / sin(radians(gamma)),
                ],
                [0, 0, V / (a * b * sin(radians(gamma)))],
            ],
            dtype=float,
        )

    def _uij_to_cart(
        self,
        uvals: tuple[float, float, float, float, float, float],
        symm_matrix: Optional[np.ndarray],
    ) -> np.ndarray:
        """Convert fractional *Uij* to a Cartesian ADP tensor."""
        U11, U22, U33, U23, U13, U12 = uvals
        Uij = np.array(
            [[U11, U12, U13], [U12, U22, U23], [U13, U23, U33]], dtype=float
        )
        if symm_matrix is not None:
            Uij = symm_matrix.T @ Uij @ symm_matrix
        N = np.diag([self._astar, self._bstar, self._cstar])
        return self._amatrix @ N @ Uij @ N.T @ self._amatrix.T

    # ------------------------------------------------------------------
    # Public API  (mirrors MoleculeWidget)
    # ------------------------------------------------------------------

    def set_background_color(self, color: QtGui.QColor) -> None:
        """Set the widget background colour."""
        self._bg_rgb = (
            color.redF(),
            color.greenF(),
            color.blueF(),
        )
        pal = self.palette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, color)
        self.setPalette(pal)
        self.update()

    def set_bond_color(self, color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int]) -> None:
        """Set the default color used for all non-selected bonds."""
        self._bond_rgb = _normalize_rgb_color(color)
        if self.atoms:
            self._build_geometry()
        self.update()

    def sizeHint(self) -> QtCore.QSize:
        """Preferred starting size."""
        return QtCore.QSize(640, 480)

    def minimumSizeHint(self) -> QtCore.QSize:
        """Minimum useful size."""
        return QtCore.QSize(320, 220)

    def set_bond_width(self, width: int) -> None:
        """Set the bond width.  Triggers a geometry rebuild."""
        self.bond_width = width
        if self.atoms:
            self._build_geometry()
        self.update()

    def set_labels_visible(self, visible: bool) -> None:
        """Toggle atom label visibility."""
        self.labels = visible
        self.update()

    def show_labels(self, value: bool) -> None:
        """Toggle atom label visibility."""
        self.labels = value
        self.update()

    def show_hydrogens(self, value: bool) -> None:
        """Toggle hydrogen atom and bond display."""
        self.show_hydrogens_flag = value
        if self.atoms:
            self._build_geometry()
        if self._density_map is not None:
            self._build_density_geometry()
        self.update()

    def set_visible_parts(self, parts: set[int] | None) -> None:
        """Set which disorder parts are rendered.

        :param parts: A set of part numbers to display, or ``None`` to show
            all parts (no filtering).  An empty set hides every atom.
        """
        if parts == self._visible_parts:
            return
        self._visible_parts = parts
        if self.atoms:
            self._build_geometry()
        if self._density_map is not None:
            self._build_density_geometry()
        self.update()

    def show_adps(self, value: bool) -> None:
        """Toggle ADP ellipsoid / isotropic sphere display."""
        self._show_adps = value
        if self.atoms:
            self._build_geometry()
        self.update()

    def setLabelFont(self, font_size: int) -> None:
        """Set atom label pixel size."""
        self.fontsize = max(1, font_size)
        self.update()

    # ------------------------------------------------------------------
    # Residual (Fo-Fc) density
    # ------------------------------------------------------------------

    def show_residual_density(
        self,
        hkl_path: object | None = None,
        level: float | None = None,
        *,
        model_path: object | None = None,
    ) -> None:
        """Compute and display a residual (Fo−Fc) isosurface.

        The map is cached, so later level changes only re-contour it.
        """
        from fastmolwidget.density import calculate_residual_density

        model, reflections = self._density_sources(model_path, hkl_path)
        self._density_map = calculate_residual_density(model, reflections)
        self._density_level = (self._density_map.sigma_level()
                               if level is None else abs(float(level)))
        self._build_density_geometry()
        self.update()

    def refresh_residual_density(self) -> None:
        """Re-clip the cached map around the visible atoms."""
        if self._density_map is None:
            return
        self._build_density_geometry()
        self.update()

    def set_residual_density_level(self, level: float) -> None:
        """Re-contour the cached residual-density map."""
        level = abs(float(level))
        if level == self._density_level:
            return
        self._density_level = level
        self.densityLevelChanged.emit(level)
        if self._density_map is not None:
            self._build_density_geometry()
            self.update()

    def step_residual_density_level(self, steps: int) -> bool:
        """Adjust the contour level by *steps* wheel notches."""
        if self._density_map is None:
            return False
        level = self._density_level + steps * DENSITY_LEVEL_STEP
        level = min(max(level, DENSITY_LEVEL_MIN), DENSITY_LEVEL_MAX)
        self.set_residual_density_level(round(level, 2))
        return True

    def clear_residual_density(self) -> None:
        """Remove the residual-density isosurface from the view."""
        self._density_map = None
        self._density_verts = np.empty(0, dtype=np.float32)
        self._density_idx = np.empty(0, dtype=np.uint32)
        self._density_pos_count = 0
        self._density_neg_count = 0
        self._density_dirty = True
        # The dedicated flattened-ADP map used for moiety-drag snapping is
        # tied to the same model/reflections and must be recomputed too.
        self._disorder_density_guide = None
        self.update()

    @property
    def residual_density_map(self):
        """The computed :class:`~fastmolwidget.density.ResidualDensityMap`.

        ``None`` until :meth:`show_residual_density` has been called.  Useful
        for reporting the map statistics (``max``, ``min``, ``rms``).
        """
        return self._density_map

    @property
    def residual_density_level(self) -> float:
        """The contour level the isosurface is currently drawn at, in e/Å³."""
        return self._density_level

    def _visible_atom_positions(self) -> np.ndarray | None:
        """Cartesian positions of the atoms that are currently drawn.

        Applies the same hydrogen and disorder-part filters as the geometry
        builders, so the residual density follows exactly what is on screen.

        :returns: An ``(N, 3)`` array, or ``None`` when nothing is visible.
        """
        positions = [
            atom.center for atom in self.atoms
            if (self.show_hydrogens_flag or atom.type_ not in ("H", "D"))
               and (self._visible_parts is None
                    or atom.part in self._visible_parts)
        ]
        if not positions:
            return None
        return np.asarray(positions, dtype=float)

    def _build_density_geometry(self) -> None:
        """Contour the cached map and pack both lobes into one line buffer.

        The isosurface is restricted to :data:`DENSITY_MARGIN` around the
        *visible* atoms, so grown or packed structures get density around every
        displayed atom while hidden hydrogens and filtered-out disorder parts
        drag nothing in.
        """
        if self._density_map is None:
            self.clear_residual_density()
            return

        positions = self._visible_atom_positions()

        verts_list: list[np.ndarray] = []
        edges_list: list[np.ndarray] = []
        counts: list[int] = []
        offset = 0
        for verts, edges in self._density_map.isosurfaces(
            (self._density_level, -self._density_level),
            atoms=positions, margin=DENSITY_MARGIN,
        ):
            if len(verts) and len(edges):
                verts_list.append(np.asarray(verts, dtype=np.float32))
                edges_list.append(np.asarray(edges, dtype=np.uint32) + offset)
                counts.append(int(edges.size))
                offset += len(verts)
            else:
                counts.append(0)

        if verts_list:
            self._density_verts = np.concatenate(verts_list).ravel()
            self._density_idx = np.concatenate(edges_list).ravel()
        else:
            self._density_verts = np.empty(0, dtype=np.float32)
            self._density_idx = np.empty(0, dtype=np.uint32)
        self._density_pos_count, self._density_neg_count = counts
        self._density_dirty = True

    def clear(self) -> None:
        """Remove all atoms and bonds."""
        self.open_molecule(atoms=[])

    def reset_view(self) -> None:
        """Reset zoom, rotation and pan to initial defaults."""
        self._rot_matrix = np.eye(3, dtype=np.float32)
        self.cumulative_R = np.eye(3, dtype=np.float32)
        self._zoom = 1.0
        self._pan = np.zeros(2, dtype=np.float32)
        self.update()

    def reset_rotation_center(self) -> None:
        """Restore the rotation pivot to the molecule centre."""
        self._compute_molecule_bounds()
        self._pan = np.zeros(2, dtype=np.float32)
        self.update()

    def _align_to_reciprocal_axis(self, axis_index: int) -> None:
        """Align the view so real-space axis 0/1/2 points at the viewer."""
        if self._amatrix is None or self._cell is None:
            return

        # Real-space lattice vectors are the columns of _amatrix.
        direct_vec = self._amatrix[:, axis_index].copy()
        direct_vec = direct_vec / np.linalg.norm(direct_vec)

        # Map direct_vec to +Z.
        z_axis = direct_vec.astype(np.float32)

        # Choose an "up" vector that is not parallel to z_axis.
        up_candidate = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(z_axis, up_candidate)) > 0.99:
            up_candidate = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        x_axis = np.cross(up_candidate, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # Rows are the new basis vectors in the original frame.
        target_R = np.array([x_axis, y_axis, z_axis], dtype=np.float32)

        self._rot_matrix = target_R
        self.cumulative_R = target_R
        self.update()

    def align_best_view(self) -> None:
        """Rotate the structure to the orientation that maximises atom visibility.

        Uses PCA (Principal Component Analysis) on the currently visible atom
        positions.  The eigenvector with the **smallest** eigenvalue (the axis
        along which the cloud is thinnest) is mapped to the camera Z-axis so
        that the widest face of the molecule points towards the viewer.

        Hydrogen / deuterium atoms are excluded from the PCA when
        ``show_hydrogens_flag`` is ``False``.  Does nothing when fewer than
        two visible atoms are loaded.
        """
        if not self.atoms:
            return

        # --- collect visible atom positions ------------------------------
        if self.show_hydrogens_flag:
            visible_coords = np.array([a.center for a in self.atoms], dtype=np.float64)
        else:
            visible_coords = np.array(
                [a.center for a in self.atoms if a.type_ not in ('H', 'D')],
                dtype=np.float64,
            )
        if len(visible_coords) < 2:
            return

        # --- PCA on centred coordinates ----------------------------------
        centred = visible_coords - visible_coords.mean(axis=0)
        cov = centred.T @ centred  # 3×3 scatter matrix

        evals, evecs = np.linalg.eigh(cov)  # eigenvalues ascending, evecs as columns

        # Sort descending: largest variance → X, smallest → Z (towards viewer)
        order = np.argsort(evals)[::-1]
        evecs = evecs[:, order]

        x_axis = evecs[:, 0].astype(np.float32)
        y_axis = evecs[:, 1].astype(np.float32)
        z_axis = evecs[:, 2].astype(np.float32)

        # Enforce right-handed coordinate system
        if np.dot(np.cross(x_axis, y_axis), z_axis) < 0:
            z_axis = -z_axis

        target_R = np.array([x_axis, y_axis, z_axis], dtype=np.float32)

        self._rot_matrix = target_R
        self.cumulative_R = target_R
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key-press events for real-space axis alignment (F1=a, F2=b, F3=c)."""
        if event.key() == Qt.Key.Key_F1:
            self._align_to_reciprocal_axis(0)
        elif event.key() == Qt.Key.Key_F2:
            self._align_to_reciprocal_axis(1)
        elif event.key() == Qt.Key.Key_F3:
            self._align_to_reciprocal_axis(2)
        else:
            super().keyPressEvent(event)

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Save the current view to an image file."""
        image: QtGui.QImage = self.grabFramebuffer()

        if image_scale != 1.0:
            new_w = int(image.width() * image_scale)
            new_h = int(image.height() * image_scale)
            image = image.scaled(
                new_w,
                new_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        image.save(str(Path(filename).resolve()))

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._lastPos = event.position()
        self._pressPos = event.position()
        self._mouse_moved = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._lastPos is None:
            # Hover only — no prior press. Update the hovered-atom label.
            if event.buttons() == Qt.MouseButton.NoButton:
                self._update_hover(event.position())
            return

        pos = event.position()
        dx = float(pos.x() - self._lastPos.x())
        dy = float(pos.y() - self._lastPos.y())
        self._mouse_moved = True

        if event.buttons() == Qt.MouseButton.NoButton:
            # Pure hover (no drag in progress)
            self._update_hover(pos)
            self._lastPos = pos
            return

        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)

        if event.buttons() == Qt.MouseButton.LeftButton and ctrl and shift:
            # Free single-atom drag: only materialises once real movement is
            # confirmed, for the same reason as the moiety drag below - a
            # plain Ctrl+Shift+click must not do anything by itself.
            if self._single_atom_drag_index is None and self._pressPos is not None:
                self.try_start_single_atom_drag(
                    float(self._pressPos.x()), float(self._pressPos.y()),
                )
            if self._single_atom_drag_index is not None:
                self.update_single_atom_drag(float(pos.x()), float(pos.y()))
                self._lastPos = pos
                return

        if event.buttons() == Qt.MouseButton.LeftButton and ctrl and not shift:
            # Only materialise the drag (and duplicate the moiety) once real
            # movement is confirmed, never on a plain Ctrl+click - that must
            # keep behaving as ordinary add/remove-from-selection.  The pick
            # uses the press position, so the user grabs whatever was under
            # the cursor when the gesture started.
            if self._disorder_drag_session is None and self._pressPos is not None:
                selected_bond = (
                    next(iter(self.selected_bonds))
                    if len(self.selected_bonds) == 1 else None
                )
                self.try_start_moiety_drag(
                    float(self._pressPos.x()), float(self._pressPos.y()),
                    set(self.selected_atoms), selected_bond,
                )
            if self._disorder_drag_session is not None:
                self.update_moiety_drag(float(pos.x()), float(pos.y()))
                self._lastPos = pos
                return

        # Any drag suppresses the hover label until the mouse stops moving.
        if self._hover_atom_label is not None:
            self._hover_atom_label = None
        if self._hover_bond is not None:
            self._hover_bond = None
            self._hover_bond_distance = None
            self._hover_cursor = None

        if event.buttons() == Qt.MouseButton.LeftButton:
            # Arcball-style rotation
            angle_y = dx / 100.0
            angle_x = dy / 100.0
            Ry = np.array(
                [
                    [cos(angle_y), 0.0, sin(angle_y)],
                    [0.0, 1.0, 0.0],
                    [-sin(angle_y), 0.0, cos(angle_y)],
                ],
                dtype=np.float32,
            )
            Rx = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, cos(angle_x), -sin(angle_x)],
                    [0.0, sin(angle_x), cos(angle_x)],
                ],
                dtype=np.float32,
            )
            R = Rx @ Ry
            self._rot_matrix = R @ self._rot_matrix
            self.cumulative_R = R @ self.cumulative_R

        elif event.buttons() == Qt.MouseButton.RightButton:
            # Zoom
            self._zoom += dy / 250.0
            self._zoom = max(0.01, self._zoom)

        elif event.buttons() == Qt.MouseButton.MiddleButton:
            # Pan
            pan_scale = self._molecule_radius * 0.001
            self._pan[0] += dx * pan_scale
            self._pan[1] -= dy * pan_scale

        self._lastPos = pos
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        # Alt/Option + left-click emulates middle-click recentering.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._mouse_moved
            and self._pressPos is not None
        ):
            # Emulate middle-click recentering.
            if bool(event.modifiers() & Qt.KeyboardModifier.AltModifier):
                self._handle_middle_click(event)
            else:
                self._handle_click(event)
        elif (
            event.button() == Qt.MouseButton.MiddleButton
            and not self._mouse_moved
            and self._pressPos is not None
        ):
            self._handle_middle_click(event)
        if event.button() == Qt.MouseButton.LeftButton:
            # End of a moiety or single-atom drag, if one was in progress.
            # Positions already applied to self.atoms are kept.
            self.end_drag()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: QtCore.QEvent) -> None:  # type: ignore[override]
        """Clear the hovered-atom label when the cursor leaves the widget."""
        changed = False
        if self._hover_atom_label is not None:
            self._hover_atom_label = None
            changed = True
        if self._hover_bond is not None:
            self._hover_bond = None
            self._hover_bond_distance = None
            self._hover_cursor = None
            changed = True
        if changed:
            self.update()
        super().leaveEvent(event)

    def _update_hover(self, pos: QtCore.QPointF) -> None:
        """Pick the atom (or, if none, the bond) under *pos* and refresh the
        hover state if it changed.

        Hidden hydrogens are excluded from both atom and bond picks.  Atom
        hover takes priority over bond hover, so the rounded distance label
        is only shown when the cursor is over a bond but not over any atom.
        """
        if not self.atoms:
            new_atom: str | None = None
            new_bond: tuple[str, str] | None = None
            new_dist: float | None = None
        else:
            sx, sy = float(pos.x()), float(pos.y())
            mv = self._compute_mv_matrix()
            atom, atom_t = self._pick_atom_at(sx, sy, mv=mv)
            new_atom = atom.label if atom is not None else None
            new_bond = None
            new_dist = None
            if atom is None:
                # Bond pick – reuse exactly the same ray test as click selection.
                proj = self._compute_proj_matrix()
                best_t = float("inf")
                best_pair: tuple[_Atom3D, _Atom3D] | None = None
                for n1, n2 in self.connections:
                    at1, at2 = self.atoms[n1], self.atoms[n2]
                    if not self.show_hydrogens_flag and (at1.type_ in ("H", "D") or at2.type_ in ("H", "D")):
                        continue
                    if self._visible_parts is not None and (
                        at1.part not in self._visible_parts or at2.part not in self._visible_parts
                    ):
                        continue
                    t = self._ray_bond_screen(sx, sy, at1.center, at2.center, mv, proj)
                    if t is not None and t < best_t:
                        best_t = t
                        best_pair = (at1, at2)
                if best_pair is not None:
                    a, b = best_pair
                    new_bond = tuple(sorted((a.label, b.label)))  # type: ignore[assignment]
                    new_dist = float(np.linalg.norm(a.center - b.center))

        changed = (
            new_atom != self._hover_atom_label
            or new_bond != self._hover_bond
            or (new_bond is not None and self._hover_cursor != pos)
        )
        self._hover_atom_label = new_atom
        self._hover_bond = new_bond
        self._hover_bond_distance = new_dist
        self._hover_cursor = QtCore.QPointF(pos) if new_bond is not None else None
        if changed:
            self.update()

    def _draw_hover_distance_label(self, painter: QtGui.QPainter, text: str, cx: float, cy: float) -> None:
        """Render *text* in a rounded, semi-transparent box near *(cx, cy)*.

        The fill is a blend of *Himmelblau* and *Mintgrün* with mild
        transparency; the border is a thin neutral grey.
        """
        font = QtGui.QFont()
        font.setPixelSize(max(1, int(self.fontsize * self._zoom)))
        font.setBold(True)
        painter.setFont(font)
        metrics = QtGui.QFontMetrics(font)
        pad_x, pad_y = 2.0, 0.0
        tw = metrics.horizontalAdvance(text)
        th = metrics.height()

        # Place the box just below-right of the cursor; clamp to widget bounds.
        box_w = tw + 2 * pad_x
        box_h = th + 2 * pad_y
        x = cx + 14.0
        y = cy + 14.0
        w = max(1, self.width())
        h = max(1, self.height())
        if x + box_w > w:
            x = cx - 14.0 - box_w
        if y + box_h > h:
            y = cy - 14.0 - box_h
        rect = QtCore.QRectF(x, y, box_w, box_h)

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(QtGui.QColor(143, 230, 193, 220))  # Himmelblau ↔ Mintgrün
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60, 220), 1.0))
        painter.drawRoundedRect(rect, 5.0, 5.0)
        painter.setPen(QtGui.QColor(20, 20, 20))
        painter.drawText(
            rect,
            int(QtCore.Qt.AlignmentFlag.AlignCenter),
            text,
        )
        painter.restore()

    def _draw_axis_indicator(self, painter: QtGui.QPainter) -> None:
        """Draw unit-cell axis arrows (a=red, b=green, c=blue) in the bottom-left corner.

        The arrows are rotated by the current view rotation so they track the
        molecule orientation.  Does nothing if no unit cell is loaded.
        """
        if self._cell is None or self._amatrix is None:
            return

        # Unit cell vectors in Cartesian (columns of _amatrix), normalised
        axes = [self._amatrix[:, i].astype(np.float64) for i in range(3)]
        axes = [v / np.linalg.norm(v) for v in axes]

        # Rotate by current view rotation
        R = self._rot_matrix.astype(np.float64)
        axes = [R @ v for v in axes]

        arrow_len = 40.0
        origin_x = 55.0
        origin_y = float(self.height()) - 55.0

        colors = [
            QtGui.QColor(220, 30, 30),
            QtGui.QColor(30, 160, 30),
            QtGui.QColor(30, 30, 220),
        ]
        labels = ['a', 'b', 'c']

        painter.save()
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)

        font = QtGui.QFont()
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)

        for i in range(3):
            vx, vy = float(axes[i][0]), float(axes[i][1])
            tip_x = origin_x + vx * arrow_len
            # OpenGL view Y is up; Qt screen Y is down.
            tip_y = origin_y - vy * arrow_len

            pen = QtGui.QPen(colors[i], 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(
                QtCore.QPointF(origin_x, origin_y),
                QtCore.QPointF(tip_x, tip_y),
            )

            # Arrowhead
            dx, dy = tip_x - origin_x, tip_y - origin_y
            length = sqrt(dx * dx + dy * dy)
            if length > 1e-6:
                ux, uy = dx / length, dy / length
                px, py = -uy, ux  # perpendicular
                head_len = 8.0
                head_w = 3.5
                painter.drawLine(
                    QtCore.QPointF(tip_x, tip_y),
                    QtCore.QPointF(tip_x - ux * head_len + px * head_w,
                                   tip_y - uy * head_len + py * head_w),
                )
                painter.drawLine(
                    QtCore.QPointF(tip_x, tip_y),
                    QtCore.QPointF(tip_x - ux * head_len - px * head_w,
                                   tip_y - uy * head_len - py * head_w),
                )

            # Label at the tip
            painter.setPen(colors[i])
            painter.drawText(
                QtCore.QPointF(tip_x + 4 * (1 if vx >= 0 else -2),
                               tip_y + 4 * (-1 if vy >= 0 else 2)),
                labels[i],
            )

        painter.restore()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        """Scroll changes label size; Ctrl+scroll changes density level."""
        delta = event.angleDelta().y()
        if delta == 0:
            return
        steps = 1 if delta > 0 else -1
        if event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Only claim Ctrl+wheel when a map is loaded.
            if self.step_residual_density_level(steps):
                event.accept()
            else:
                event.ignore()
            return
        self.setLabelFont(self.fontsize + 2 * steps)

    def _handle_click(self, event: QtGui.QMouseEvent) -> None:
        """Select atom or bond under the cursor."""
        pos = event.position()
        if self._is_click_drag(pos):
            return

        mv = self._compute_mv_matrix()
        proj = self._compute_proj_matrix()
        sx, sy = float(pos.x()), float(pos.y())

        # Atom pass — shared with middle-click centring.
        best_atom, best_t = self._pick_atom_at(sx, sy, mv=mv)
        best_bond: tuple[str, str] | None = None

        # Compare atom and bond hits in the same view-space t units.
        for n1, n2 in self.connections:
            at1, at2 = self.atoms[n1], self.atoms[n2]
            if not self.show_hydrogens_flag:
                if at1.type_ in ("H", "D") or at2.type_ in ("H", "D"):
                    continue
            t = self._ray_bond_screen(sx, sy, at1.center, at2.center, mv, proj)
            if t is not None and t < best_t:
                best_t = t
                best_bond = tuple(sorted((at1.label, at2.label)))  # type: ignore[assignment]
                best_atom = None

        ctrl = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        changed = False

        if best_atom is not None:
            if ctrl:
                if best_atom.label in self.selected_atoms:
                    self.selected_atoms.discard(best_atom.label)
                else:
                    self.selected_atoms.add(best_atom.label)
            else:
                self.selected_atoms = {best_atom.label}
                self.selected_bonds.clear()
            changed = True
            self.atomClicked.emit(best_atom.label)

        elif best_bond is not None:
            if ctrl:
                if best_bond in self.selected_bonds:
                    self.selected_bonds.discard(best_bond)
                else:
                    self.selected_bonds.add(best_bond)
            else:
                self.selected_bonds = {best_bond}
                self.selected_atoms.clear()
            changed = True
            self.bondClicked.emit(best_bond[0], best_bond[1])

        else:
            if not ctrl and (self.selected_atoms or self.selected_bonds):
                self.selected_atoms.clear()
                self.selected_bonds.clear()
                changed = True

        if changed:
            self._build_geometry()
            self.update()

    def _handle_middle_click(self, event: QtGui.QMouseEvent) -> None:
        """Centre the view on the atom under the cursor (no-op if nothing hit).

        After centring, the picked atom becomes the rotation pivot and is
        moved to the screen centre by resetting the pan offset.
        """
        pos = event.position()
        if self._is_click_drag(pos):
            return

        atom, _ = self._pick_atom_at(float(pos.x()), float(pos.y()))
        if atom is None:
            return

        self._molecule_center = atom.center.astype(np.float32).copy()
        self._pan = np.zeros(2, dtype=np.float32)
        self.update()

    # ------------------------------------------------------------------
    # Interactive dragging - hooks for DisorderDragMixin
    # ------------------------------------------------------------------

    def _drag_atom_count(self) -> int:
        return len(self.atoms)

    def _drag_atom_label(self, index: int) -> str:
        return self.atoms[index].label

    def _drag_atom_type(self, index: int) -> str:
        return self.atoms[index].type_

    def _drag_atom_position(self, index: int) -> np.ndarray:
        return self.atoms[index].center.astype(np.float64)

    def _drag_connections(self) -> tuple[tuple[int, int], ...]:
        return tuple(self.connections)

    def _pick_atom_index(self, x: float, y: float) -> int | None:
        atom, _ = self._pick_atom_at(x, y, mv=self._compute_mv_matrix())
        if atom is None:
            return None
        for index, candidate in enumerate(self.atoms):
            if candidate is atom:
                return index
        return None

    def _begin_drag_projection(self, index: int, x: float, y: float) -> bool:
        """Pin the drag to the grabbed atom's eye-space depth plane.

        The mouse ray is later intersected with that screen-parallel plane, so
        the target stays at a constant depth for the whole gesture instead of
        the atom rushing towards or away from the viewer.
        """
        mv = self._compute_mv_matrix().astype(np.float64)
        try:
            self._disorder_drag_mv_inv = np.linalg.inv(mv)
        except np.linalg.LinAlgError:
            self._disorder_drag_mv_inv = None
            return False
        c4 = np.array([*self.atoms[index].center, 1.0], dtype=np.float64)
        self._disorder_drag_eye_z = float((mv @ c4)[2])
        return True

    def _drag_target(self, x: float, y: float) -> np.ndarray | None:
        if self._disorder_drag_mv_inv is None:
            return None
        return self._screen_to_world_at_depth(
            x, y, self._disorder_drag_eye_z, self._disorder_drag_mv_inv,
        )

    def _end_drag_projection(self) -> None:
        self._disorder_drag_mv_inv = None

    def _set_atom_part(self, index: int, part: int) -> None:
        self.atoms[index].part = part

    def _clone_atom_for_split(self, index: int, label: str, part: int) -> int:
        """Append a copy of atom *index* for the split disorder part.

        The split is created in the current ADP mode: either the copy inherits the
        original anisotropic tensor, or it is flattened to a small isotropic U to
        start from a simpler part-2 model.
        """
        original = self.atoms[index]
        duplicate = _Atom3D(
            float(original.center[0]), float(original.center[1]),
            float(original.center[2]), label, original.type_, part,
        )
        duplicate.symmgen = original.symmgen

        if self.dragged_atoms_are_isotropic:
            duplicate.u_cart = None
            duplicate.u_iso = isotropic_u_for_atom_type(original.type_)
            duplicate.adp_valid = True
            duplicate.u_eigvals = None
            duplicate.u_eigvecs = None
            duplicate.adp_billboard_r = 0.0
            duplicate.adp_A_matrix = None
            duplicate.npd_half_edge = 0.0
        else:
            duplicate.u_cart = None if original.u_cart is None else np.array(original.u_cart, copy=True)
            duplicate.u_iso = None if original.u_iso is None else float(original.u_iso)
            duplicate.adp_valid = bool(original.adp_valid)
            duplicate.u_eigvals = None if original.u_eigvals is None else np.array(original.u_eigvals, copy=True)
            duplicate.u_eigvecs = None if original.u_eigvecs is None else np.array(original.u_eigvecs, copy=True)
            duplicate.adp_billboard_r = float(original.adp_billboard_r)
            duplicate.adp_A_matrix = None if original.adp_A_matrix is None else np.array(original.adp_A_matrix, copy=True)
            duplicate.npd_half_edge = float(original.npd_half_edge)

        self.atoms.append(duplicate)
        return len(self.atoms) - 1

    def _add_connections(self, edges: tuple[tuple[int, int], ...]) -> None:
        self.connections = tuple(self.connections) + tuple(edges)

    def _apply_drag_positions(self, positions: dict[int, np.ndarray]) -> None:
        for index, position in positions.items():
            self.atoms[index].center = np.asarray(position, dtype=np.float32)

        self._build_geometry()
        if self._density_map is not None:
            self._build_density_geometry()
        self.update()

    def _on_split_parts_changed(self) -> None:
        self.available_parts = frozenset(a.part for a in self.atoms)
        self.partsChanged.emit(self.available_parts)

    def _compute_riding_atoms(self) -> dict[int, int]:
        """Map every bonded hydrogen to the atom it rides on.

        Thin wrapper over :func:`fastmolwidget.disorder_drag.riding_atoms`
        for this widget's atom list; see there for the rule.
        """
        from fastmolwidget.disorder_drag import riding_atoms

        return riding_atoms(
            [a.type_ for a in self.atoms],
            [a.center.astype(np.float64) for a in self.atoms],
            self.connections,
        )

    def _screen_to_world_at_depth(
        self, x: float, y: float, eye_z: float, mv_inv: np.ndarray,
    ) -> np.ndarray:
        """World-space point under cursor *(x, y)* at a fixed eye-space depth.

        The mouse ray is intersected with the screen-parallel plane at
        *eye_z* (the grabbed atom's view-space depth when the drag started),
        so the target stays at a constant depth for the whole gesture.
        """
        w = max(1, self.width())
        h = max(1, self.height())
        half_w, half_h = self._ortho_half_extents()
        nx = 2.0 * x / w - 1.0
        ny = 1.0 - 2.0 * y / h
        eye_point = np.array([nx * half_w, ny * half_h, eye_z, 1.0], dtype=np.float64)
        world = mv_inv @ eye_point
        return world[:3]

    def _is_click_drag(self, pos: QtCore.QPointF) -> bool:
        """Return ``True`` when the cursor moved more than 5 px from the
        original press position — i.e. this release should be treated as a
        drag, not a click."""
        if self._pressPos is None:
            return False
        dx = pos.x() - self._pressPos.x()
        dy = pos.y() - self._pressPos.y()
        return dx * dx + dy * dy > 25

    def _pick_atom_at(
        self,
        sx: float,
        sy: float,
        *,
        mv: np.ndarray | None = None,
    ) -> tuple[_Atom3D | None, float]:
        """Return the front-most atom under screen position *(sx, sy)* and
        its viewspace ray *t*, or ``(None, inf)`` if no atom is hit.  Bonds
        are ignored.

        :param mv: Optional precomputed model-view matrix; pass it when the
            caller already needs it (e.g. for a subsequent bond pass) to
            avoid recomputing.
        """
        if mv is None:
            mv = self._compute_mv_matrix()
        ray_origin, ray_dir = self._screen_to_ray_viewspace(sx, sy)

        best_atom: _Atom3D | None = None
        best_t = float("inf")

        for atom in self.atoms:
            if not self.show_hydrogens_flag and atom.type_ in ("H", "D"):
                continue
            if self._visible_parts is not None and atom.part not in self._visible_parts:
                continue
            # Hit test against the *rendered* surface so the entire visible
            # ellipsoid / sphere is selectable.
            if (
                self._show_adps
                and atom.u_cart is not None
                and atom.adp_valid
                and atom.adp_A_matrix is not None
            ):
                t = self._ray_ellipsoid_hit_viewspace(
                    ray_origin, ray_dir, atom.center, atom.adp_A_matrix, mv
                )
            elif (
                atom.u_cart is not None
                and not atom.adp_valid
                and atom.npd_half_edge > 0.0
            ):
                # Cube placeholder: pick against its bounding sphere
                # (radius = half_edge × √3).  Slight over-pick at the
                # corners is acceptable and simpler than ray-AABB.
                radius = _npd_bound_radius(atom)
                t = self._ray_sphere_hit_viewspace(
                    ray_origin, ray_dir, atom.center, radius, mv
                )
            else:
                radius = (
                    sqrt(float(atom.u_iso)) * _ADP_SCALE
                    if self._show_adps and atom.u_iso is not None
                    else atom.display_radius
                )
                t = self._ray_sphere_hit_viewspace(
                    ray_origin, ray_dir, atom.center, radius, mv
                )
            # Nearest-t wins → front-most surface is selected (z-ordering).
            if t is not None and t < best_t:
                best_t = t
                best_atom = atom
        return best_atom, best_t

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _screen_to_ray_viewspace(self, sx: float, sy: float) -> tuple[np.ndarray, np.ndarray]:
        """Return orthographic ray origin and direction for screen position *(sx, sy)*."""
        w = max(1, self.width())
        h = max(1, self.height())
        half_w, half_h = self._ortho_half_extents()
        nx = 2.0 * sx / w - 1.0
        ny = 1.0 - 2.0 * sy / h
        origin = np.array([nx * half_w, ny * half_h, 0.0], dtype=np.float32)
        direction = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return origin, direction

    def _ray_sphere_hit_viewspace(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        world_center: np.ndarray,
        radius: float,
        mv: np.ndarray,
    ) -> float | None:
        """Ray–sphere intersection in view space.  Returns parametric *t* or ``None``."""
        c4 = np.array([*world_center, 1.0], dtype=np.float32)
        c_eye = (mv @ c4)[:3]

        oc = ray_origin - c_eye
        d = ray_dir
        a = float(np.dot(d, d))
        b = 2.0 * float(np.dot(oc, d))
        c = float(np.dot(oc, oc) - radius * radius)
        disc = b * b - 4.0 * a * c
        if disc < 0.0:
            return None
        sqrt_disc = sqrt(disc)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)
        t = t0 if t0 >= 0.0 else t1
        return t if t >= 0.0 else None

    def _ray_ellipsoid_hit_viewspace(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        world_center: np.ndarray,
        a_matrix: np.ndarray,
        mv: np.ndarray,
    ) -> float | None:
        """Ray–ellipsoid intersection in view space."""
        # Transform the quadratic form into view space.
        c4 = np.array([*world_center, 1.0], dtype=np.float32)
        c_eye = (mv @ c4)[:3]

        R = mv[:3, :3].astype(np.float64)
        A = np.asarray(a_matrix, dtype=np.float64)
        M = R @ A @ R.T

        oc = (ray_origin - c_eye).astype(np.float64)
        d = ray_dir.astype(np.float64)

        Md = M @ d
        Moc = M @ oc

        a_c = float(np.dot(d, Md))
        b_c = 2.0 * float(np.dot(oc, Md))
        c_c = float(np.dot(oc, Moc)) - 1.0

        if abs(a_c) < 1e-20:
            return None

        disc = b_c * b_c - 4.0 * a_c * c_c
        if disc < 0.0:
            return None

        sqrt_disc = sqrt(disc)
        t0 = (-b_c - sqrt_disc) / (2.0 * a_c)
        t1 = (-b_c + sqrt_disc) / (2.0 * a_c)
        t = t0 if t0 >= 0.0 else t1
        return float(t) if t >= 0.0 else None

    def _ray_bond_screen(
        self,
        sx: float,
        sy: float,
        p1: np.ndarray,
        p2: np.ndarray,
        mv: np.ndarray,
        proj: np.ndarray,
    ) -> float | None:
        """Return the view-space hit distance for a bond near *(sx, sy)*."""
        w = max(1, self.width())
        h = max(1, self.height())

        def _project(pos: np.ndarray) -> tuple[np.ndarray, float] | None:
            p4 = np.array([*pos, 1.0], dtype=np.float32)
            eye = mv @ p4
            clip = proj @ eye
            if abs(clip[3]) < 1e-8:
                return None
            ndc = clip[:3] / clip[3]
            if ndc[2] < -1.0 or ndc[2] > 1.0:
                return None
            screen = np.array(
                [(ndc[0] + 1.0) * 0.5 * w, (1.0 - ndc[1]) * 0.5 * h],
                dtype=np.float32,
            )
            return screen, float(eye[2])  # viewspace z (negative in front of camera)

        r1 = _project(p1)
        r2 = _project(p2)
        if r1 is None or r2 is None:
            return None

        sp1, z1 = r1
        sp2, z2 = r2

        p = np.array([sx, sy], dtype=np.float32)
        ab = sp2 - sp1
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-6:
            # Both endpoints project to essentially the same pixel.
            dist = float(np.linalg.norm(p - sp1))
            if dist <= _BOND_HIT_TOLERANCE_PX:
                # Use the same interpolation formula as the normal path (t=0.5).
                z_closest = z1 + 0.5 * (z2 - z1)
                return float(-z_closest)
            return None

        t = float(max(0.0, min(1.0, np.dot(p - sp1, ab) / ab_len2)))
        proj_pt = sp1 + t * ab
        dist = float(np.linalg.norm(p - proj_pt))

        if dist <= _BOND_HIT_TOLERANCE_PX:
            # Interpolate viewspace z and negate to get t (positive, smaller = closer).
            z_closest = z1 + t * (z2 - z1)
            return float(-z_closest)
        return None


# ---------------------------------------------------------------------------
# Private GL helpers (module-level to keep the class body shorter)
# ---------------------------------------------------------------------------

# Cache for glGetUniformLocation / glGetAttribLocation – keyed by (prog, name).
# Avoids repeated driver round-trips on every frame.
_UNIFORM_LOC_CACHE: dict[tuple[int, bytes], int] = {}
_ATTRIB_LOC_CACHE: dict[tuple[int, bytes], int] = {}


def _set_mat4(prog: int, name: bytes, mat: np.ndarray) -> None:
    key = (prog, name)
    try:
        loc = _UNIFORM_LOC_CACHE[key]
    except KeyError:
        loc = gl.glGetUniformLocation(prog, name)
        _UNIFORM_LOC_CACHE[key] = loc
    if loc >= 0:
        gl.glUniformMatrix4fv(loc, 1, False, mat.T.astype(np.float32).copy())


def _set_vec3(prog: int, name: bytes, v: np.ndarray) -> None:
    key = (prog, name)
    try:
        loc = _UNIFORM_LOC_CACHE[key]
    except KeyError:
        loc = gl.glGetUniformLocation(prog, name)
        _UNIFORM_LOC_CACHE[key] = loc
    if loc >= 0:
        v = np.asarray(v, dtype=np.float32).ravel()
        gl.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))


def _set_float(prog: int, name: bytes, value: float) -> None:
    key = (prog, name)
    try:
        loc = _UNIFORM_LOC_CACHE[key]
    except KeyError:
        loc = gl.glGetUniformLocation(prog, name)
        _UNIFORM_LOC_CACHE[key] = loc
    if loc >= 0:
        gl.glUniform1f(loc, float(value))


def _bind_attrib(
    prog: int, name: bytes, size: int, stride: int, offset: int
) -> None:
    key = (prog, name)
    try:
        loc = _ATTRIB_LOC_CACHE[key]
    except KeyError:
        loc = gl.glGetAttribLocation(prog, name)
        _ATTRIB_LOC_CACHE[key] = loc
    if loc >= 0:
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, False, stride, ctypes.c_void_p(offset))


def _unbind_attrib(prog: int, names: list[bytes]) -> None:
    for name in names:
        key = (prog, name)
        try:
            loc = _ATTRIB_LOC_CACHE[key]
        except KeyError:
            loc = gl.glGetAttribLocation(prog, name)
            _ATTRIB_LOC_CACHE[key] = loc
        if loc >= 0:
            gl.glDisableVertexAttribArray(loc)
