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

All GLSL shaders target ``#version 120`` (OpenGL 2.1 / GLSL 1.20 compatibility
profile) for the widest possible hardware support.

Module-level helpers
--------------------
Call :func:`configure_opengl_format` **before** creating
:class:`~qtpy.QtWidgets.QApplication` to enable multi-sample anti-aliasing
on all platforms (required for macOS).

Mouse controls
--------------
* **Left drag**  – rotate.
* **Right drag** – zoom.
* **Middle drag** – pan.
* **Scroll wheel** – increase / decrease label font size.
* **Left click** – select atom or bond; emit ``atomClicked`` / ``bondClicked``.
* **Ctrl + left click** – add to / remove from selection.
"""

from __future__ import annotations

import ctypes
from math import cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from fastmolwidget.atoms import element2color, get_radius_from_element
from fastmolwidget.molecule2D import calc_volume
from fastmolwidget.sdm import Atomtuple

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

# Locate the best QOpenGLWidget base class available in the current Qt binding
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

__all__ = ["MoleculeWidget3D", "configure_opengl_format"]


def configure_opengl_format() -> None:
    """Set a sensible default :class:`~qtpy.QtGui.QSurfaceFormat`.

    Call this **before** creating :class:`~qtpy.QtWidgets.QApplication` to
    request depth buffer, double-buffering and 4× MSAA on all platforms
    (including macOS where the format must be set as the default before any
    context is created).

    Example::

        from fastmolwidget.molecule3D import configure_opengl_format
        from qtpy import QtWidgets

        configure_opengl_format()
        app = QtWidgets.QApplication([])
    """
    try:
        fmt = QtGui.QSurfaceFormat()
        # GLSL 1.20 shaders require a compatibility context on macOS.
        fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
        fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
        fmt.setVersion(2, 1)
        fmt.setDepthBufferSize(24)
        fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
        fmt.setSamples(4)
        QtGui.QSurfaceFormat.setDefaultFormat(fmt)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb_float(hex_color: str) -> tuple[float, float, float]:
    """Convert ``#RRGGBB`` hex string to ``(r, g, b)`` float tuple in [0, 1]."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _make_cylinder(
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
    col1: tuple[float, float, float],
    col2: tuple[float, float, float],
    n_seg: int = 8,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Generate a cylinder mesh between *p1* and *p2*.

    Returns ``(vertices, indices)`` arrays, or ``(None, None)`` if the
    endpoints are too close together.

    Vertex layout: ``[x, y, z, nx, ny, nz, r, g, b]`` (9 floats, 36 bytes).
    The bottom ring (at *p1*) uses *col1*; the top ring (at *p2*) uses *col2*.
    No end caps are generated because atom spheres close the bond ends visually.
    """
    axis = p2 - p1
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None, None

    u = axis / length

    # Find two vectors perpendicular to the cylinder axis
    if abs(u[0]) < 0.9:
        v = np.cross(u, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    else:
        v = np.cross(u, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)

    angles = np.linspace(0.0, 2.0 * np.pi, n_seg, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Outward normals for each segment
    normals = cos_a[:, None] * v[None, :] + sin_a[:, None] * w[None, :]  # (n_seg, 3)

    verts = np.zeros((2 * n_seg, 9), dtype=np.float32)

    # Bottom ring (p1) with col1
    for i in range(n_seg):
        verts[i, :3] = p1 + radius * normals[i]
        verts[i, 3:6] = normals[i]
        verts[i, 6:9] = col1

    # Top ring (p2) with col2
    for i in range(n_seg):
        verts[n_seg + i, :3] = p2 + radius * normals[i]
        verts[n_seg + i, 3:6] = normals[i]
        verts[n_seg + i, 6:9] = col2

    # Triangle indices for the side surface (two tris per quad strip segment)
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
        "center", "label", "type_", "part",
        "color_f", "display_radius",
        "u_cart", "u_iso", "adp_valid", "u_eigvals",
        "adp_billboard_r", "adp_A_matrix",
    ]

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        label: str,
        type_: str,
        part: int,
    ) -> None:
        self.center = np.array([x, y, z], dtype=np.float32)
        self.label = label
        self.type_ = type_
        self.part = part

        hex_color = element2color.get(type_, "#808080")
        self.color_f: tuple[float, float, float] = _hex_to_rgb_float(hex_color)

        # World-space visual radius for sphere rendering (Å)
        self.display_radius: float = 0.25

        self.u_cart: np.ndarray | None = None
        self.u_iso: float | None = None
        self.adp_valid: bool = True
        self.u_eigvals: np.ndarray | None = None

        # Set by _build_geometry when ADP data is present
        self.adp_billboard_r: float = 0.0
        self.adp_A_matrix: np.ndarray | None = None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Atom3D {self.label} {self.type_} {self.center}>"


# ---------------------------------------------------------------------------
# GLSL 1.20 shader sources
# ---------------------------------------------------------------------------

# ── Sphere impostor ──────────────────────────────────────────────────────────
_SPHERE_VERT = """\
#version 120
// Per-vertex attributes (one quad per atom)
attribute vec3 a_center;
attribute vec3 a_color;
attribute float a_radius;
attribute vec2 a_corner;

uniform mat4 u_mv;
uniform mat4 u_proj;

varying vec3 v_center_eye;
varying vec3 v_color;
varying float v_radius;
varying vec2 v_corner;

void main() {
    v_color  = a_color;
    v_radius = a_radius;
    v_corner = a_corner;

    vec4 c_eye    = u_mv * vec4(a_center, 1.0);
    v_center_eye  = c_eye.xyz;

    // Billboard: extend by sphere radius in eye-space X/Y.
    // A 5 % safety margin ensures full sphere coverage at any view angle.
    vec4 pos = c_eye;
    pos.xy  += a_corner * a_radius * 1.05;
    gl_Position = u_proj * pos;
}
"""

_SPHERE_FRAG = """\
#version 120
varying vec3  v_center_eye;
varying vec3  v_color;
varying float v_radius;
varying vec2  v_corner;

uniform mat4 u_proj;

void main() {
    // Fragment position on the billboard plane in eye space
    vec3 frag_eye = vec3(v_center_eye.xy + v_corner * v_radius * 1.05,
                         v_center_eye.z);
    vec3 ray_dir  = normalize(frag_eye);

    // Ray-sphere intersection:  | t*d - C |^2 = r^2
    // => t^2 - 2t(d.C) + (C.C - r^2) = 0
    float b    = dot(ray_dir, v_center_eye);
    float c    = dot(v_center_eye, v_center_eye) - v_radius * v_radius;
    float disc = b * b - c;
    if (disc < 0.0) discard;

    float t = b - sqrt(disc);
    if (t < 0.0) discard;

    vec3 hit    = t * ray_dir;
    vec3 normal = normalize(hit - v_center_eye);

    // Phong lighting (fixed light in eye space)
    vec3  light  = normalize(vec3(1.0, 1.5, 2.0));
    float diff   = max(dot(normal, light), 0.0);
    float spec   = pow(max(dot(reflect(-light, normal), normalize(-hit)), 0.0), 48.0);

    vec3 color = v_color * (0.2 + 0.7 * diff) + vec3(0.45) * spec;
    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);

    // Write corrected depth so atoms occlude bonds and each other properly
    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
"""

# ── Ellipsoid impostor ───────────────────────────────────────────────────────
# Vertex shader is identical to the sphere; only the fragment shader differs.
_ELLIPSOID_VERT = """\
#version 120
// Unit quad corners passed as per-vertex data
attribute vec2 a_corner;

uniform mat4 u_mv;
uniform mat4 u_proj;
uniform vec3 u_center;   // atom centre in world space
uniform float u_radius;  // billboard half-size
uniform vec3 u_color;    // atom colour

varying vec3  v_center_eye;
varying vec3  v_color;
varying float v_radius;
varying vec2  v_corner;

void main() {
    v_color  = u_color;
    v_radius = u_radius;
    v_corner = a_corner;

    vec4 c_eye   = u_mv * vec4(u_center, 1.0);
    v_center_eye = c_eye.xyz;

    vec4 pos = c_eye;
    pos.xy  += a_corner * u_radius * 1.05;
    gl_Position = u_proj * pos;
}
"""

_ELLIPSOID_FRAG = """\
#version 120
varying vec3  v_center_eye;
varying vec3  v_color;
varying float v_radius;
varying vec2  v_corner;

uniform mat4 u_proj;
uniform mat3 u_ellipsoid_A;   // A = inv(scale^2 * U_cart)

void main() {
    vec3 frag_eye = vec3(v_center_eye.xy + v_corner * v_radius * 1.05,
                         v_center_eye.z);
    vec3 d = normalize(frag_eye);
    vec3 C = v_center_eye;

    // Ray-ellipsoid: (t*d - C)^T A (t*d - C) = 1
    // => a_c * t^2  -  2 * b_c * t  +  (C^T A C - 1) = 0
    float a_c  = dot(d,  u_ellipsoid_A * d);
    float b_c  = dot(C,  u_ellipsoid_A * d);
    float cc   = dot(C,  u_ellipsoid_A * C) - 1.0;
    float disc = b_c * b_c - a_c * cc;

    if (disc < 0.0 || a_c < 1e-10) discard;

    float t = (b_c - sqrt(disc)) / a_c;
    if (t < 0.0) discard;

    vec3 hit    = t * d;
    // Outward normal = gradient of the ellipsoid equation = 2 A (P - C)
    vec3 normal = normalize(u_ellipsoid_A * (hit - C));

    vec3  light = normalize(vec3(1.0, 1.5, 2.0));
    float diff  = max(dot(normal, light), 0.0);
    float spec  = pow(max(dot(reflect(-light, normal), normalize(-hit)), 0.0), 48.0);

    vec3 color = v_color * (0.2 + 0.7 * diff) + vec3(0.35) * spec;
    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);

    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
"""

# ── Cylinder mesh ────────────────────────────────────────────────────────────
_CYLINDER_VERT = """\
#version 120
attribute vec3 a_position;
attribute vec3 a_normal;
attribute vec3 a_color;

uniform mat4 u_mv;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;   // inverse-transpose of MV upper 3x3

varying vec3 v_normal_eye;
varying vec3 v_pos_eye;
varying vec3 v_color;

void main() {
    v_color      = a_color;
    vec4 pos_e   = u_mv * vec4(a_position, 1.0);
    v_pos_eye    = pos_e.xyz;
    v_normal_eye = normalize(u_normal_mat * a_normal);
    gl_Position  = u_proj * pos_e;
}
"""

_CYLINDER_FRAG = """\
#version 120
varying vec3 v_normal_eye;
varying vec3 v_pos_eye;
varying vec3 v_color;

void main() {
    vec3  normal = normalize(v_normal_eye);
    vec3  light  = normalize(vec3(1.0, 1.5, 2.0));
    float diff   = max(dot(normal, light), 0.0);
    float spec   = pow(max(dot(reflect(-light, normal),
                               normalize(-v_pos_eye)), 0.0), 32.0);

    vec3 color = v_color * (0.2 + 0.7 * diff) + vec3(0.35) * spec;
    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""

# Selection highlight colour (cyan)
_SEL_COLOR: tuple[float, float, float] = (0.0, 0.75, 1.0)

# ORTEP 50 % probability scale factor
_ADP_SCALE: float = 1.5382


# ---------------------------------------------------------------------------
# Main widget
# ---------------------------------------------------------------------------

class MoleculeWidget3D(_WidgetBase):  # type: ignore[valid-type,misc]
    """Real 3-D OpenGL crystal-structure display widget.

    Drop-in replacement for :class:`~fastmolwidget.molecule2D.MoleculeWidget`
    with an identical public API.  Rendering is GPU-accelerated via
    hardware-accelerated OpenGL 2.1 sphere / ellipsoid impostors and
    tessellated cylinder bonds.

    If *PyOpenGL* is unavailable or OpenGL initialisation fails the widget
    gracefully shows an informational message rather than crashing.

    Parameters
    ----------
    parent:
        Optional parent widget.
    """

    atomClicked = QtCore.Signal(str)
    bondClicked = QtCore.Signal(str, str)

    # Field-of-view used by both _compute_proj_matrix and _screen_to_ray_viewspace
    _FOV_DEGREES: float = 45.0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)

        # ---- Molecule data ------------------------------------------------
        self.atoms: list[_Atom3D] = []
        self.connections: tuple = ()
        self._cell: tuple[float, ...] | None = None
        self._adp_map: dict = {}
        self._astar: float = 0.0
        self._bstar: float = 0.0
        self._cstar: float = 0.0
        self._amatrix: np.ndarray = np.eye(3, dtype=float)

        # ---- Public display state (mirrors MoleculeWidget) ----------------
        self.fontsize: int = 13
        self.bond_width: int = 3
        self.atoms_size: int = 12   # kept for API compatibility
        self.labels: bool = True
        self.show_hydrogens_flag: bool = True
        self.selected_atoms: set[str] = set()
        self.selected_bonds: set[tuple[str, str]] = set()

        self._show_adps: bool = True
        self._round_bonds: bool = True   # True → 8-segment cyl, False → 4-segment

        # ---- 3-D view state -----------------------------------------------
        self._rot_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self._zoom: float = 1.0
        self._pan: np.ndarray = np.zeros(2, dtype=np.float32)
        self._molecule_center: np.ndarray = np.zeros(3, dtype=np.float32)
        self._molecule_radius: float = 10.0
        self.cumulative_R: np.ndarray = np.eye(3, dtype=np.float32)

        self._bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)

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
        self._ellipsoid_prog: int = 0
        self._cylinder_prog: int = 0
        self._sphere_vbo: int = 0
        self._sphere_ibo: int = 0
        self._cylinder_vbo: int = 0
        self._cylinder_ibo: int = 0
        self._unit_quad_vbo: int = 0   # shared 2-D corner quad for ellipsoids
        self._unit_quad_ibo: int = 0

        # ---- CPU-side geometry buffers ------------------------------------
        self._sphere_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._sphere_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._sphere_count: int = 0
        self._cylinder_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._cylinder_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._cylinder_count: int = 0

        # ADP atoms for per-atom ellipsoid draw calls
        self._adp_draw_list: list[_Atom3D] = []

        self._geometry_dirty: bool = False

        # ---- Mouse tracking -----------------------------------------------
        self._lastPos: QtCore.QPointF | None = None
        self._pressPos: QtCore.QPointF | None = None
        self._mouse_moved: bool = False

        # ---- Widget appearance --------------------------------------------
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtCore.Qt.GlobalColor.white)
        self.setAutoFillBackground(True)
        self.setPalette(pal)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Request a sensible OpenGL surface format for this widget instance
        if _IS_GL_WIDGET and not self._gl_failed:
            self._setup_surface_format()

        # Connect default no-op handlers so signals can be emitted safely
        self.atomClicked.connect(lambda _x: None)
        self.bondClicked.connect(lambda _x, _y: None)

    # ------------------------------------------------------------------
    # Surface format
    # ------------------------------------------------------------------

    def _setup_surface_format(self) -> None:
        """Request depth buffer, double-buffering and 4× MSAA."""
        try:
            fmt = QtGui.QSurfaceFormat()
            fmt.setRenderableType(QtGui.QSurfaceFormat.RenderableType.OpenGL)
            fmt.setProfile(QtGui.QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
            fmt.setVersion(2, 1)
            fmt.setDepthBufferSize(24)
            fmt.setSwapBehavior(QtGui.QSurfaceFormat.SwapBehavior.DoubleBuffer)
            fmt.setSamples(4)
            # setFormat exists on QOpenGLWidget but not QWidget
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

        # Compile shaders
        self._sphere_prog = self._compile_program(
            _SPHERE_VERT, _SPHERE_FRAG, "sphere"
        )
        self._ellipsoid_prog = self._compile_program(
            _ELLIPSOID_VERT, _ELLIPSOID_FRAG, "ellipsoid"
        )
        self._cylinder_prog = self._compile_program(
            _CYLINDER_VERT, _CYLINDER_FRAG, "cylinder"
        )

        # Allocate VBOs / IBOs
        buffers = gl.glGenBuffers(6)
        (
            self._sphere_vbo, self._sphere_ibo,
            self._cylinder_vbo, self._cylinder_ibo,
            self._unit_quad_vbo, self._unit_quad_ibo,
        ) = buffers

        # Upload the shared unit quad used for all ellipsoid draw calls
        corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        quad_idx = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._unit_quad_vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, corners.nbytes, corners, gl.GL_STATIC_DRAW
        )
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._unit_quad_ibo)
        gl.glBufferData(
            gl.GL_ELEMENT_ARRAY_BUFFER, quad_idx.nbytes, quad_idx, gl.GL_STATIC_DRAW
        )
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)

    def _compile_program(self, vert_src: str, frag_src: str, name: str) -> int:
        """Compile and link a GLSL 1.20 shader program.

        Raises :class:`RuntimeError` on compilation failure so that
        :meth:`initializeGL` can catch and set the failure flag.
        """
        try:
            vert = _glshaders.compileShader(vert_src, gl.GL_VERTEX_SHADER)
            frag = _glshaders.compileShader(frag_src, gl.GL_FRAGMENT_SHADER)
            # compileProgram validates by default; in QOpenGLWidget.initializeGL the
            # draw FBO may not be fully ready yet, which can trigger false negatives.
            try:
                prog = _glshaders.compileProgram(vert, frag, validate=False)
            except TypeError:
                # Older PyOpenGL without validate kwarg: keep previous behavior.
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
        r, g, b = self._bg_rgb
        gl.glClearColor(r, g, b, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if not self.atoms:
            return

        if self._geometry_dirty:
            self._upload_geometry()

        mv = self._compute_mv_matrix()
        proj = self._compute_proj_matrix()

        # Bonds first (behind atom spheres)
        if self._cylinder_count > 0:
            self._render_cylinders(mv, proj)

        # Regular atom spheres
        if self._sphere_count > 0:
            self._render_spheres(mv, proj)

        # ADP ellipsoids (one draw call each)
        if self._show_adps and self._adp_draw_list:
            for atom in self._adp_draw_list:
                if atom.adp_A_matrix is not None:
                    self._render_one_ellipsoid(atom, mv, proj)

        # Labels overlay
        if self.labels:
            self._draw_labels_overlay(mv, proj)

    # ------------------------------------------------------------------
    # paintEvent – routes to OpenGL path or pure-QPainter fallback
    # ------------------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        """Route paint requests to the appropriate renderer."""
        if _IS_GL_WIDGET:
            # Let QOpenGLWidget's paintEvent call makeCurrent → paintGL → doneCurrent
            try:
                super().paintEvent(event)
            except Exception:
                painter = QtGui.QPainter(self)
                self._draw_fallback_text(painter)
                painter.end()
        else:
            # Pure QWidget fallback: no OpenGL context
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

    # ------------------------------------------------------------------
    # Geometry building
    # ------------------------------------------------------------------

    def _build_geometry(self) -> None:
        """(Re)build all CPU-side geometry from :attr:`atoms`."""
        self._build_sphere_geometry()
        self._build_cylinder_geometry()
        self._geometry_dirty = True

    def _build_sphere_geometry(self) -> None:
        """Create billboard quad data for non-ADP atoms."""
        corners = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=np.float32)
        quad_idx_tpl = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)

        self._adp_draw_list = []
        sphere_atoms: list[_Atom3D] = []

        for atom in self.atoms:
            if not self.show_hydrogens_flag and atom.type_ in ("H", "D"):
                continue
            if self._show_adps and atom.u_cart is not None and atom.adp_valid:
                self._adp_draw_list.append(atom)
            else:
                sphere_atoms.append(atom)

        n = len(sphere_atoms)
        if n == 0:
            self._sphere_verts = np.empty(0, dtype=np.float32)
            self._sphere_idx = np.empty(0, dtype=np.uint32)
            self._sphere_count = 0
            return

        # Vertex layout: [cx, cy, cz, r, g, b, radius, corner_x, corner_y]
        # 9 floats per vertex, 4 vertices per atom, 6 indices per atom
        verts = np.zeros((n * 4, 9), dtype=np.float32)
        idx = np.zeros(n * 6, dtype=np.uint32)

        for i, atom in enumerate(sphere_atoms):
            c = atom.center
            col = (
                _SEL_COLOR
                if atom.label in self.selected_atoms
                else atom.color_f
            )
            r = atom.display_radius
            for j in range(4):
                vi = i * 4 + j
                verts[vi, 0:3] = c
                verts[vi, 3:6] = col
                verts[vi, 6] = r
                verts[vi, 7:9] = corners[j]
            idx[i * 6: i * 6 + 6] = quad_idx_tpl + i * 4

        self._sphere_verts = verts.ravel()
        self._sphere_idx = idx
        self._sphere_count = n * 6

    def _build_cylinder_geometry(self) -> None:
        """Build tessellated cylinder meshes for all bonds."""
        n_seg = 8 if self._round_bonds else 4
        # base cylinder radius, scaled by bond_width
        cyl_r = 0.022 * max(1, self.bond_width)

        all_verts: list[np.ndarray] = []
        all_idx: list[np.ndarray] = []
        v_offset = 0

        for n1, n2 in self.connections:
            at1 = self.atoms[n1]
            at2 = self.atoms[n2]

            if not self.show_hydrogens_flag:
                if at1.type_ in ("H", "D") or at2.type_ in ("H", "D"):
                    continue

            bond_key: tuple[str, str] = tuple(sorted((at1.label, at2.label)))  # type: ignore[assignment]
            if bond_key in self.selected_bonds:
                col1 = col2 = _SEL_COLOR
            else:
                col1, col2 = at1.color_f, at2.color_f

            verts, bond_idx = _make_cylinder(at1.center, at2.center, cyl_r, col1, col2, n_seg)
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

        self._geometry_dirty = False

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_spheres(self, mv: np.ndarray, proj: np.ndarray) -> None:
        prog = self._sphere_prog
        gl.glUseProgram(prog)

        _set_mat4(prog, b"u_mv", mv)
        _set_mat4(prog, b"u_proj", proj)

        stride = 9 * 4  # 9 floats × 4 bytes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._sphere_vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._sphere_ibo)

        _bind_attrib(prog, b"a_center", 3, stride, 0)
        _bind_attrib(prog, b"a_color",  3, stride, 12)
        _bind_attrib(prog, b"a_radius", 1, stride, 24)
        _bind_attrib(prog, b"a_corner", 2, stride, 28)

        gl.glDrawElements(
            gl.GL_TRIANGLES, self._sphere_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0)
        )

        _unbind_attrib(prog, [b"a_center", b"a_color", b"a_radius", b"a_corner"])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)

    def _render_one_ellipsoid(
        self, atom: _Atom3D, mv: np.ndarray, proj: np.ndarray
    ) -> None:
        prog = self._ellipsoid_prog
        gl.glUseProgram(prog)

        _set_mat4(prog, b"u_mv", mv)
        _set_mat4(prog, b"u_proj", proj)

        # Per-atom uniforms
        _set_vec3(prog, b"u_center", atom.center)
        _set_vec3(
            prog,
            b"u_color",
            np.array(
                _SEL_COLOR if atom.label in self.selected_atoms else atom.color_f,
                dtype=np.float32,
            ),
        )
        _set_float(prog, b"u_radius", atom.adp_billboard_r)

        A = atom.adp_A_matrix
        if A is not None:
            loc = gl.glGetUniformLocation(prog, b"u_ellipsoid_A")
            if loc >= 0:
                # OpenGL is column-major; numpy is row-major → transpose
                gl.glUniformMatrix3fv(loc, 1, False, A.astype(np.float32).T.copy())

        # Draw shared unit quad
        stride = 2 * 4  # 2 floats × 4 bytes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._unit_quad_vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._unit_quad_ibo)

        _bind_attrib(prog, b"a_corner", 2, stride, 0)

        gl.glDrawElements(
            gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0)
        )

        _unbind_attrib(prog, [b"a_corner"])
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

        stride = 9 * 4  # 9 floats × 4 bytes
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._cylinder_vbo)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._cylinder_ibo)

        _bind_attrib(prog, b"a_position", 3, stride, 0)
        _bind_attrib(prog, b"a_normal",   3, stride, 12)
        _bind_attrib(prog, b"a_color",    3, stride, 24)

        gl.glDrawElements(
            gl.GL_TRIANGLES, self._cylinder_count, gl.GL_UNSIGNED_INT, ctypes.c_void_p(0)
        )

        _unbind_attrib(prog, [b"a_position", b"a_normal", b"a_color"])
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
        gl.glUseProgram(0)

    def _draw_labels_overlay(self, mv: np.ndarray, proj: np.ndarray) -> None:
        """Draw atom labels as a QPainter overlay on top of the GL scene."""
        if not self.atoms:
            return

        w = max(1, self.width())
        h = max(1, self.height())
        hydrogens = ("H", "D")

        painter = QtGui.QPainter(self)
        try:
            font = QtGui.QFont()
            font.setPixelSize(max(1, self.fontsize))
            painter.setFont(font)
            painter.setPen(QtGui.QColor(100, 50, 5))

            for atom in self.atoms:
                if atom.type_ in hydrogens:
                    continue

                pos4 = np.array([*atom.center, 1.0], dtype=np.float32)
                eye = mv @ pos4
                clip = proj @ eye
                if abs(clip[3]) < 1e-8 or clip[3] < 0:
                    continue
                ndc = clip[:3] / clip[3]
                if not (-1.0 <= ndc[0] <= 1.0 and -1.0 <= ndc[1] <= 1.0):
                    continue

                sx = int((ndc[0] + 1.0) * 0.5 * w)
                sy = int((1.0 - ndc[1]) * 0.5 * h)
                painter.drawText(sx + 4, sy - 4, atom.label)
        finally:
            painter.end()

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------

    def _compute_mv_matrix(self) -> np.ndarray:
        """Build the Model-View matrix from current rotation / zoom / pan."""
        dist = self._molecule_radius * 3.0 / max(self._zoom, 0.001)

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

    def _compute_proj_matrix(self) -> np.ndarray:
        """Build a perspective projection matrix (45° FoV)."""
        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / h
        near, far = 0.01, 10000.0
        f = 1.0 / float(np.tan(np.radians(self._FOV_DEGREES) / 2.0))
        return np.array(
            [
                [f / aspect, 0.0, 0.0,                        0.0],
                [0.0,        f,   0.0,                        0.0],
                [0.0,        0.0, (far + near) / (near - far), (2 * far * near) / (near - far)],
                [0.0,        0.0, -1.0,                       0.0],
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
        adps: dict[str, tuple[float, float, float, float, float, float]] | None = None,
        keep_view: bool = False,
    ) -> None:
        """Load a new molecule and (unless *keep_view*) reset the view."""
        self._load_molecule(atoms, cell, adps, keep_view=keep_view)

    def grow_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None = None,
        adps: dict[str, tuple[float, float, float, float, float, float]] | None = None,
    ) -> None:
        """Update the displayed molecule while preserving the current view."""
        self._load_molecule(atoms, cell, adps, keep_view=True)

    def _load_molecule(
        self,
        atoms: list[Atomtuple],
        cell: tuple[float, float, float, float, float, float] | None,
        adps: dict[str, tuple[float, float, float, float, float, float]] | None,
        keep_view: bool,
    ) -> None:
        self._cell = cell
        self._adp_map = adps if adps is not None else {}

        if self._cell is not None:
            self.calc_amatrix()

        # Build internal atom list with optional ADP tensors
        self.atoms = []
        name_counts: dict[str, int] = {}

        for at in atoms:
            base_name = at.label
            count = name_counts.get(base_name, 0)
            internal_name = base_name if count == 0 else f"{base_name}>>{count}"
            name_counts[base_name] = count + 1

            a3d = _Atom3D(at.x, at.y, at.z, internal_name, at.type, at.part)

            if self._adp_map and self._cell and base_name in self._adp_map:
                try:
                    uvals = self._adp_map[base_name]
                    symm = getattr(at, "symm_matrix", None)
                    if symm is not None:
                        symm = np.array(symm, dtype=float)
                    a3d.u_cart = self._uij_to_cart(uvals, symm)
                    a3d.u_iso = float(np.trace(a3d.u_cart) / 3.0)
                    evals, _ = np.linalg.eigh(a3d.u_cart)
                    if np.any(evals <= 0):
                        a3d.adp_valid = False
                    else:
                        a3d.adp_valid = True
                        a3d.u_eigvals = evals
                        # Billboard radius for the ellipsoid impostor quad
                        a3d.adp_billboard_r = float(
                            _ADP_SCALE * np.sqrt(np.max(evals)) * 1.2
                        )
                        A = np.linalg.inv(_ADP_SCALE**2 * a3d.u_cart)
                        a3d.adp_A_matrix = A.astype(np.float32)
                except Exception:
                    a3d.u_cart = None
                    a3d.u_iso = None
                    a3d.adp_valid = False

            if at.type in ("H", "D"):
                a3d.u_iso = a3d.u_iso if a3d.u_iso else 0.01

            self.atoms.append(a3d)

        self.connections = self._get_conntable()

        if not keep_view:
            self._compute_molecule_bounds()
            self._rot_matrix = np.eye(3, dtype=np.float32)
            self.cumulative_R = np.eye(3, dtype=np.float32)
            self._zoom = 1.0
            self._pan = np.zeros(2, dtype=np.float32)
            self.selected_atoms.clear()
            self.selected_bonds.clear()
        else:
            # Update centre so grown structures stay in view
            self._compute_molecule_bounds()

        self._build_geometry()
        self.update()

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
        """Build a connectivity table from atomic coordinates and covalent radii."""
        connections = []
        h_types = ("H", "D")
        n = len(self.atoms)
        for i in range(n):
            at1 = self.atoms[i]
            for j in range(i + 1, n):
                at2 = self.atoms[j]
                if (at1.part != 0 and at2.part != 0) and at1.part != at2.part:
                    continue
                d = float(np.linalg.norm(at1.center - at2.center))
                if d > 4.0:
                    continue
                r1 = get_radius_from_element(at1.type_)
                r2 = get_radius_from_element(at2.type_)
                if (r1 + r2) * extra_param > d:
                    if at1.type_ in h_types and at2.type_ in h_types:
                        continue
                    connections.append((i, j))
        return tuple(connections)

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
        self.update()

    def show_adps(self, value: bool) -> None:
        """Toggle ADP ellipsoid / isotropic sphere display."""
        self._show_adps = value
        if self.atoms:
            self._build_geometry()
        self.update()

    def show_round_bonds(self, bond_type: bool = True) -> None:
        """Switch between 3-D (more segments) and angular (fewer segments) bonds."""
        self._round_bonds = bond_type
        if self.atoms:
            self._build_geometry()
        self.update()

    def setLabelFont(self, font_size: int) -> None:
        """Set atom label pixel size."""
        self.fontsize = max(1, font_size)
        self.update()

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

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Save the current view to an image file."""
        pixmap = self.grab()
        if image_scale != 1.0:
            new_w = int(pixmap.width() * image_scale)
            new_h = int(pixmap.height() * image_scale)
            pixmap = pixmap.scaled(
                new_w,
                new_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        pixmap.save(str(Path(filename).resolve()))

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._lastPos = event.position()
        self._pressPos = event.position()
        self._mouse_moved = False

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._lastPos is None:
            return

        pos = event.position()
        dx = float(pos.x() - self._lastPos.x())
        dy = float(pos.y() - self._lastPos.y())
        self._mouse_moved = True

        if event.buttons() == Qt.MouseButton.LeftButton:
            # Arcball-style rotation
            angle_y = dx / 80.0
            angle_x = dy / 80.0
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
            self._zoom += dy / 100.0
            self._zoom = max(0.01, self._zoom)

        elif event.buttons() == Qt.MouseButton.MiddleButton:
            # Pan
            pan_scale = self._molecule_radius * 0.008
            self._pan[0] -= dx * pan_scale
            self._pan[1] += dy * pan_scale

        self._lastPos = pos
        self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._mouse_moved
            and self._pressPos is not None
        ):
            self._handle_click(event)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        """Scroll wheel adjusts label font size."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.setLabelFont(self.fontsize + 2)
        elif delta < 0:
            self.setLabelFont(self.fontsize - 2)

    def _handle_click(self, event: QtGui.QMouseEvent) -> None:
        """Select atom or bond under the cursor."""
        pos = event.position()
        # Guard: if position moved more than 5 pixels treat as drag
        if self._pressPos is not None:
            dx = pos.x() - self._pressPos.x()
            dy = pos.y() - self._pressPos.y()
            if dx * dx + dy * dy > 25:
                return

        mv = self._compute_mv_matrix()
        proj = self._compute_proj_matrix()
        sx, sy = float(pos.x()), float(pos.y())

        ray_dir = self._screen_to_ray_viewspace(sx, sy)

        best_atom: _Atom3D | None = None
        best_bond: tuple[str, str] | None = None
        best_t = float("inf")

        for atom in self.atoms:
            if not self.show_hydrogens_flag and atom.type_ in ("H", "D"):
                continue
            t = self._ray_sphere_hit_viewspace(ray_dir, atom.center, atom.display_radius, mv)
            if t is not None and t < best_t:
                best_t = t
                best_atom = atom
                best_bond = None

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

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _screen_to_ray_viewspace(self, sx: float, sy: float) -> np.ndarray:
        """Return a ray direction in view/eye space for screen position *(sx, sy)*."""
        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / h
        f = 1.0 / float(np.tan(np.radians(self._FOV_DEGREES) / 2.0))
        nx = (2.0 * sx / w - 1.0) / (f / aspect)
        ny = -(2.0 * sy / h - 1.0) / f
        return np.array([nx, ny, -1.0], dtype=np.float32)

    def _ray_sphere_hit_viewspace(
        self,
        ray_dir: np.ndarray,
        world_center: np.ndarray,
        radius: float,
        mv: np.ndarray,
    ) -> float | None:
        """Ray–sphere intersection in view space.  Returns parametric *t* or ``None``."""
        c4 = np.array([*world_center, 1.0], dtype=np.float32)
        c_eye = (mv @ c4)[:3]

        d = ray_dir
        b = float(np.dot(d, c_eye))
        disc = b * b - (float(np.dot(c_eye, c_eye)) - radius * radius)
        if disc < 0.0:
            return None
        t = b - sqrt(disc)
        return t if t >= 0.0 else None

    def _ray_bond_screen(
        self,
        sx: float,
        sy: float,
        p1: np.ndarray,
        p2: np.ndarray,
        mv: np.ndarray,
        proj: np.ndarray,
    ) -> float | None:
        """Return average NDC depth of bond *p1–p2* if the screen click *(sx,sy)*
        is within 6 pixels of its projected line segment, else ``None``."""
        w = max(1, self.width())
        h = max(1, self.height())

        def _project(pos: np.ndarray) -> np.ndarray | None:
            p4 = np.array([*pos, 1.0], dtype=np.float32)
            clip = proj @ (mv @ p4)
            if abs(clip[3]) < 1e-8 or clip[3] < 0:
                return None
            ndc = clip[:3] / clip[3]
            if ndc[2] < -1.0 or ndc[2] > 1.0:
                return None
            return np.array(
                [(ndc[0] + 1.0) * 0.5 * w, (1.0 - ndc[1]) * 0.5 * h, ndc[2]],
                dtype=np.float32,
            )

        sp1 = _project(p1)
        sp2 = _project(p2)
        if sp1 is None or sp2 is None:
            return None

        p = np.array([sx, sy], dtype=np.float32)
        a = sp1[:2]
        b = sp2[:2]
        ab = b - a
        ab_len2 = float(np.dot(ab, ab))
        if ab_len2 < 1e-6:
            return None
        t = float(max(0.0, min(1.0, np.dot(p - a, ab) / ab_len2)))
        proj_pt = a + t * ab
        dist = float(np.linalg.norm(p - proj_pt))

        if dist <= 6.0:
            return float((sp1[2] + sp2[2]) / 2.0)
        return None


# ---------------------------------------------------------------------------
# Private GL helpers (module-level to keep the class body shorter)
# ---------------------------------------------------------------------------

def _set_mat4(prog: int, name: bytes, mat: np.ndarray) -> None:
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniformMatrix4fv(loc, 1, False, mat.T.astype(np.float32).copy())


def _set_vec3(prog: int, name: bytes, v: np.ndarray) -> None:
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        v = np.asarray(v, dtype=np.float32).ravel()
        gl.glUniform3f(loc, float(v[0]), float(v[1]), float(v[2]))


def _set_float(prog: int, name: bytes, value: float) -> None:
    loc = gl.glGetUniformLocation(prog, name)
    if loc >= 0:
        gl.glUniform1f(loc, float(value))


def _bind_attrib(
    prog: int, name: bytes, size: int, stride: int, offset: int
) -> None:
    loc = gl.glGetAttribLocation(prog, name)
    if loc >= 0:
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, size, gl.GL_FLOAT, False, stride, ctypes.c_void_p(offset))


def _unbind_attrib(prog: int, names: list[bytes]) -> None:
    for name in names:
        loc = gl.glGetAttribLocation(prog, name)
        if loc >= 0:
            gl.glDisableVertexAttribArray(loc)
