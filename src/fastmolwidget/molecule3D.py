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

Mouse controls
--------------
* **Left drag**  – rotate.
* **Right drag** – zoom.
* **Middle drag** – pan.
* **Middle click** – centre the view on the clicked atom (becomes the new
  rotation pivot).
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

__all__ = ["MoleculeWidget3D"]



# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _hex_to_rgb_float(hex_color: str) -> tuple[float, float, float]:
    """Convert ``#RRGGBB`` hex string to ``(r, g, b)`` float tuple in [0, 1]."""
    h = hex_color.lstrip("#")
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


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
    """Generate a cylinder mesh between *p1* and *p2*.

    Returns ``(vertices, indices)`` arrays, or ``(None, None)`` if the
    endpoints are too close together.

    Vertex layout: ``[x, y, z, nx, ny, nz, r, g, b, selected]`` (10 floats,
    40 bytes).  ``selected`` is 1.0 for selected bonds (rendered flat, no
    shading) and 0.0 otherwise.  The full cylinder uses one uniform colour.
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

    verts = np.zeros((2 * n_seg, 10), dtype=np.float32)
    sel_flag = 1.0 if selected else 0.0

    # Bottom ring (p1)
    for i in range(n_seg):
        verts[i, :3] = p1 + radius * normals[i]
        verts[i, 3:6] = normals[i]
        verts[i, 6:9] = color
        verts[i, 9] = sel_flag

    # Top ring (p2)
    for i in range(n_seg):
        verts[n_seg + i, :3] = p2 + radius * normals[i]
        verts[n_seg + i, 3:6] = normals[i]
        verts[n_seg + i, 6:9] = color
        verts[n_seg + i, 9] = sel_flag

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
        "center", "label", "type_", "part", "symmgen",
        "color_f", "display_radius",
        "u_cart", "u_iso", "adp_valid", "u_eigvals", "u_eigvecs",
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
        u_eq: float = 0.04,
    ) -> None:
        self.center = np.array([x, y, z], dtype=np.float32)
        self.label = label
        self.type_ = type_
        self.part = part
        self.symmgen = False

        hex_color = element2color.get(type_, "#808080")
        self.color_f: tuple[float, float, float] = _hex_to_rgb_float(hex_color)

        # World-space visual radius for sphere rendering (Å) – covalent radius
        try:
            self.display_radius: float = get_radius_from_element(type_)
        except (KeyError, Exception):
            self.display_radius: float = 0.5

        self.u_cart: np.ndarray | None = None
        # Store the isotropic U value (Å²); radius = sqrt(u_iso) * _ADP_SCALE
        self.u_iso: float | None = u_eq if type_ not in ('H', 'D') else None
        self.adp_valid: bool = True
        self.u_eigvals: np.ndarray | None = None
        self.u_eigvecs: np.ndarray | None = None

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
attribute float a_selected;

uniform mat4 u_mv;
uniform mat4 u_proj;

varying vec3 v_center_eye;
varying vec3 v_color;
varying float v_radius;
varying vec2 v_corner;
varying float v_selected;

void main() {
    v_color    = a_color;
    v_radius   = a_radius;
    v_corner   = a_corner;
    v_selected = a_selected;

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
varying float v_selected;

uniform mat4 u_proj;

void main() {
    // Orthographic projection: all rays are parallel to -Z.
    vec2 local_xy = v_corner * v_radius * 1.05;
    float xy2 = dot(local_xy, local_xy);
    float r2 = v_radius * v_radius;
    if (xy2 > r2) discard;

    float z_hit = sqrt(r2 - xy2);
    vec3 local_hit = vec3(local_xy, z_hit);
    vec3 hit    = v_center_eye + local_hit;
    vec3 normal = normalize(local_hit);

    // Bright, low-shadow lighting for crisp atom colours.
    // Selected atoms are coloured via v_color upstream, so no branch needed.
    vec3  light     = normalize(vec3(1.0, 1.5, 2.0));
    float diff      = max(dot(normal, light), 0.0);
    float soft_diff = 0.25 + 0.75 * diff;
    float spec      = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 72.0);

    vec3 base_color = clamp(v_color * 1.08, 0.0, 1.0);
    vec3 color      = base_color * (0.50 + 0.35 * soft_diff) + vec3(0.16) * spec;
    gl_FragColor    = vec4(clamp(color, 0.0, 1.0), 1.0);

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

uniform mat4 u_mv;
uniform mat4 u_proj;
uniform mat3 u_ellipsoid_A;   // A = inv(scale^2 * U_cart)
uniform mat3 u_world_evecs;
uniform float u_selected;     // 1.0 → render flat, no shading

void main() {
    vec2 local_xy = v_corner * v_radius * 1.05;
    vec3 q0 = vec3(local_xy, 0.0);
    vec3 ez = vec3(0.0, 0.0, 1.0);
    
    mat3 inv_mv = transpose(mat3(u_mv));
    vec3 ray_o = inv_mv * q0;
    vec3 ray_d = inv_mv * ez;

    vec3 Aq0 = u_ellipsoid_A * ray_o;
    vec3 Aez = u_ellipsoid_A * ray_d;

    // Orthographic projection: solve the local +Z intersection.
    float a_c  = dot(ray_d, Aez);
    float b_c  = 2.0 * dot(ray_o, Aez);
    float cc   = dot(ray_o, Aq0) - 1.0;
    float disc = b_c * b_c - 4.0 * a_c * cc;

    if (disc < 0.0 || a_c < 1e-10) discard;

    float t_hit = (-b_c + sqrt(disc)) / (2.0 * a_c);
    vec3 hit_world = ray_o + t_hit * ray_d;
    vec3 local_hit = q0 + vec3(0.0, 0.0, t_hit);
    vec3 hit    = v_center_eye + local_hit;

    vec3 color;
    if (u_selected > 0.5) {
        vec3 normal_world = normalize(u_ellipsoid_A * hit_world);
        vec3 normal = normalize(mat3(u_mv) * normal_world);

        vec3  light     = normalize(vec3(2.0, 1.5, 2.0));
        float diff      = max(dot(normal, light), 0.0);
        float soft_diff = 0.25 + 0.75 * diff;
        float spec      = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 72.0);

        vec3 base_color = clamp(v_color * 1.08, 0.0, 1.0);
        color = base_color * (0.50 + 0.35 * soft_diff) + vec3(0.14) * spec;

        float p0 = dot(hit_world, u_world_evecs[0]);
        float p1 = dot(hit_world, u_world_evecs[1]);
        float p2 = dot(hit_world, u_world_evecs[2]);
        float lw = v_radius * 0.04;
        if (abs(p0) < lw || abs(p1) < lw || abs(p2) < lw) {
            color *= 0.15;
        }
    } else {
        // Outward normal = gradient of the ellipsoid equation = 2 A (P - C)
        vec3 normal_world = normalize(u_ellipsoid_A * hit_world);
        vec3 normal = normalize(mat3(u_mv) * normal_world);

        vec3  light     = normalize(vec3(1.0, 1.5, 2.0));
        float diff      = max(dot(normal, light), 0.0);
        float soft_diff = 0.25 + 0.75 * diff;
        float spec      = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 72.0);

        vec3 base_color = clamp(v_color * 1.08, 0.0, 1.0);
        color = base_color * (0.50 + 0.35 * soft_diff) + vec3(0.14) * spec;

        float p0 = dot(hit_world, u_world_evecs[0]);
        float p1 = dot(hit_world, u_world_evecs[1]);
        float p2 = dot(hit_world, u_world_evecs[2]);
        float lw = v_radius * 0.04;
        if (abs(p0) < lw || abs(p1) < lw || abs(p2) < lw) {
            color *= 0.15;
        }
    }

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
attribute float a_selected;

uniform mat4 u_mv;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;   // inverse-transpose of MV upper 3x3

varying vec3 v_normal_eye;
varying vec3 v_pos_eye;
varying vec3 v_color;
varying float v_selected;

void main() {
    v_color      = a_color;
    v_selected   = a_selected;
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
varying float v_selected;

void main() {
    vec3 color;
    if (v_selected > 0.5) {
        // Selected bonds render perfectly flat — no diffuse/specular shading
        color = v_color;
    } else {
        vec3  normal = normalize(v_normal_eye);
        vec3  light  = normalize(vec3(1.0, 1.5, 2.0));
        float diff   = max(dot(normal, light), 0.0);
        float spec   = pow(max(dot(reflect(-light, normal),
                                   normalize(-v_pos_eye)), 0.0), 32.0);

        color = v_color * (0.45 + 0.55 * diff) + vec3(0.30) * spec;
    }
    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""

# ── Batched ellipsoid impostor ────────────────────────────────────────────────
# All per-atom data (center, A-matrix, eigenvectors, …) are packed as vertex
# attributes so that every ellipsoid can be drawn with a single glDrawElements
# call instead of one call per atom.
#
# Vertex layout (28 floats = 112 bytes, stride):
#   offset  0 ( 0 B) – a_corner   vec2
#   offset  2 ( 8 B) – a_center   vec3
#   offset  5 (20 B) – a_color    vec3
#   offset  8 (32 B) – a_radius   float
#   offset  9 (36 B) – a_selected float
#   offset 10 (40 B) – a_A_col0   vec3  (column 0 of the A-matrix)
#   offset 13 (52 B) – a_A_col1   vec3
#   offset 16 (64 B) – a_A_col2   vec3
#   offset 19 (76 B) – a_evec0    vec3  (column 0 of the eigenvector matrix)
#   offset 22 (88 B) – a_evec1    vec3
#   offset 25 (100 B)– a_evec2    vec3
_ELLIPSOID_BATCH_VERT = """\
#version 120
attribute vec2  a_corner;
attribute vec3  a_center;
attribute vec3  a_color;
attribute float a_radius;
attribute float a_selected;
attribute vec3  a_A_col0;
attribute vec3  a_A_col1;
attribute vec3  a_A_col2;
attribute vec3  a_evec0;
attribute vec3  a_evec1;
attribute vec3  a_evec2;

uniform mat4 u_mv;
uniform mat4 u_proj;

varying vec3  v_center_eye;
varying vec3  v_color;
varying float v_radius;
varying vec2  v_corner;
varying float v_selected;
varying vec3  v_A_col0;
varying vec3  v_A_col1;
varying vec3  v_A_col2;
varying vec3  v_evec0;
varying vec3  v_evec1;
varying vec3  v_evec2;

void main() {
    v_color    = a_color;
    v_radius   = a_radius;
    v_corner   = a_corner;
    v_selected = a_selected;
    v_A_col0   = a_A_col0;
    v_A_col1   = a_A_col1;
    v_A_col2   = a_A_col2;
    v_evec0    = a_evec0;
    v_evec1    = a_evec1;
    v_evec2    = a_evec2;

    vec4 c_eye   = u_mv * vec4(a_center, 1.0);
    v_center_eye = c_eye.xyz;

    vec4 pos = c_eye;
    pos.xy  += a_corner * a_radius * 1.05;
    gl_Position = u_proj * pos;
}
"""

_ELLIPSOID_BATCH_FRAG = """\
#version 120
varying vec3  v_center_eye;
varying vec3  v_color;
varying float v_radius;
varying vec2  v_corner;
varying float v_selected;
varying vec3  v_A_col0;
varying vec3  v_A_col1;
varying vec3  v_A_col2;
varying vec3  v_evec0;
varying vec3  v_evec1;
varying vec3  v_evec2;

uniform mat4 u_mv;
uniform mat4 u_proj;

void main() {
    mat3 A     = mat3(v_A_col0, v_A_col1, v_A_col2);
    mat3 evecs = mat3(v_evec0,  v_evec1,  v_evec2);

    vec2 local_xy = v_corner * v_radius * 1.05;
    vec3 q0 = vec3(local_xy, 0.0);
    vec3 ez = vec3(0.0, 0.0, 1.0);

    mat3 inv_mv = transpose(mat3(u_mv));
    vec3 ray_o  = inv_mv * q0;
    vec3 ray_d  = inv_mv * ez;

    vec3 Aq0 = A * ray_o;
    vec3 Aez = A * ray_d;

    float a_c  = dot(ray_d, Aez);
    float b_c  = 2.0 * dot(ray_o, Aez);
    float cc   = dot(ray_o, Aq0) - 1.0;
    float disc = b_c * b_c - 4.0 * a_c * cc;

    if (disc < 0.0 || a_c < 1e-10) discard;

    float t_hit    = (-b_c + sqrt(disc)) / (2.0 * a_c);
    vec3 hit_world = ray_o + t_hit * ray_d;
    vec3 local_hit = q0 + vec3(0.0, 0.0, t_hit);
    vec3 hit       = v_center_eye + local_hit;

    vec3 normal_world = normalize(A * hit_world);
    vec3 normal       = normalize(mat3(u_mv) * normal_world);

    vec3  light     = v_selected > 0.5
                        ? normalize(vec3(2.0, 1.5, 2.0))
                        : normalize(vec3(1.0, 1.5, 2.0));
    float diff      = max(dot(normal, light), 0.0);
    float soft_diff = 0.25 + 0.75 * diff;
    float spec      = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 72.0);

    vec3 base_color = clamp(v_color * 1.08, 0.0, 1.0);
    vec3 color = base_color * (0.50 + 0.35 * soft_diff) + vec3(0.14) * spec;

    float lw = v_radius * 0.04;
    if (abs(dot(hit_world, evecs[0])) < lw ||
        abs(dot(hit_world, evecs[1])) < lw ||
        abs(dot(hit_world, evecs[2])) < lw) {
        color *= 0.15;
    }

    gl_FragColor = vec4(clamp(color, 0.0, 1.0), 1.0);

    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
"""

# Selection highlight colour (cyan)
_SEL_COLOR: tuple[float, float, float] = (0.0, 0.75, 1.0)

# Default bond colour (grey-brown)
_DEFAULT_BOND_COLOR: tuple[float, float, float] = _hex_to_rgb_float("#d1812a")

# ORTEP 50 % probability scale factor
_ADP_SCALE: float = 1.5382

# Screen-space tolerance in pixels for bond hit-testing.
_BOND_HIT_TOLERANCE_PX: float = 6.0


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
        self._adp_map: dict = {}
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
        self._ellipsoid_prog: int = 0  # kept but no longer used for rendering
        self._ellipsoid_batch_prog: int = 0  # new: draws all ellipsoids in one call
        self._cylinder_prog: int = 0
        self._sphere_vbo: int = 0
        self._sphere_ibo: int = 0
        self._cylinder_vbo: int = 0
        self._cylinder_ibo: int = 0
        self._ellipsoid_batch_vbo: int = 0
        self._ellipsoid_batch_ibo: int = 0
        self._unit_quad_vbo: int = 0  # kept for backward compat, not used in main path
        self._unit_quad_ibo: int = 0

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

        # ADP atoms for batched ellipsoid draw call
        self._adp_draw_list: list[_Atom3D] = []

        self._geometry_dirty: bool = False

        # ---- Mouse tracking -----------------------------------------------
        self._lastPos: QtCore.QPointF | None = None
        self._pressPos: QtCore.QPointF | None = None
        self._mouse_moved: bool = False
        # Label of the atom currently under the cursor (None if no atom hovered)
        self._hover_atom_label: str | None = None
        # Bond currently under the cursor (sorted label tuple) and its
        # Cartesian length in Å, plus the latest cursor position used to
        # anchor the rounded distance label.
        self._hover_bond: tuple[str, str] | None = None
        self._hover_bond_distance: float | None = None
        self._hover_cursor: QtCore.QPointF | None = None
        # Enable mouse-move events without a button held (for hover detection)
        self.setMouseTracking(True)

        # ---- Widget appearance --------------------------------------------
        # NB: do NOT enable autoFillBackground on a QOpenGLWidget.  The Qt
        # paint engine would erase our GL framebuffer to the palette Window
        # colour the moment we construct a QPainter for the label overlay,
        # leaving only the painter's output visible.  The background is
        # cleared explicitly with glClearColor in _do_paintGL instead.
        pal = QtGui.QPalette()
        pal.setColor(QtGui.QPalette.ColorRole.Window, QtCore.Qt.GlobalColor.white)
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.setPalette(pal)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Request a sensible OpenGL surface format for this widget instance
        if _IS_GL_WIDGET and not self._gl_failed:
            self._setup_surface_format()

        # Connect default no-op handlers so signals can be emitted safely
        self.atomClicked.connect(lambda _x: None)
        self.bondClicked.connect(lambda _x, _y: None)

    def paintGL(self):
        if not self._gl_initialized and not self._gl_failed:
            try:
                # try to initialize; this happens with a current context (paintGL guaranteed current)
                self._do_initializeGL()
                self._gl_initialized = True
                if self._geometry_dirty and self.atoms:
                    self._upload_geometry()
            except Exception as exc:
                self._gl_failed = True
                self._gl_fail_reason = f"OpenGL initialisation failed:\n{exc}"
                print(self._gl_fail_reason)
                # fallback painting continues
        if self._gl_failed:
            self._paint_fallback_on_gl()
            return
        self._do_paintGL()

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
        self._ellipsoid_batch_prog = self._compile_program(
            _ELLIPSOID_BATCH_VERT, _ELLIPSOID_BATCH_FRAG, "ellipsoid_batch"
        )
        self._cylinder_prog = self._compile_program(
            _CYLINDER_VERT, _CYLINDER_FRAG, "cylinder"
        )

        # Allocate VBOs / IBOs (sphere, cylinder, ellipsoid batch; unit quad no longer used)
        buffers = gl.glGenBuffers(6)
        (
            self._sphere_vbo, self._sphere_ibo,
            self._cylinder_vbo, self._cylinder_ibo,
            self._ellipsoid_batch_vbo, self._ellipsoid_batch_ibo,
        ) = buffers

        # One-shot MSAA diagnostic: warn when the actual sample count is < 2,
        # which usually means the driver fell back to a single-sample format.
        try:
            samples = int(self.format().samples())
            if samples < 2:
                print(f"[MoleculeWidget3D] OpenGL surface has samples={samples} (no MSAA).")
        except Exception:
            pass

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
        if self._geometry_dirty and self.atoms:
            self._upload_geometry()

        # Re-assert GL state every frame.  Once a QPainter has run on this
        # QOpenGLWidget (for atom labels or the bond-distance hover), Qt's
        # paint engine leaves several state bits flipped — most importantly
        # GL_MULTISAMPLE, GL_DEPTH_TEST and the viewport — which would
        # otherwise persist into the next paintGL and degrade the image
        # ("view gets worse after the first label was shown").
        self._reassert_gl_state()

        r, g, b = self._bg_rgb
        gl.glClearColor(r, g, b, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        if not self.atoms:
            return

        mv = self._compute_mv_matrix()
        proj = self._compute_proj_matrix()

        # Bonds first (behind atom spheres)
        if self._cylinder_count > 0:
            self._render_cylinders(mv, proj)

        # Regular atom spheres
        if self._sphere_count > 0:
            self._render_spheres(mv, proj)

        # ADP ellipsoids – all in a single batched draw call
        if self._show_adps and self._ellipsoid_count > 0:
            self._render_ellipsoids_batched(mv, proj)

        # Construct QPainter *after* every raw GL draw is done.  Always
        # creating it (even when no labels/hover are active) keeps the
        # GL→QPainter transition state identical on every frame, so the
        # very first time a label appears no longer triggers a visible
        # state-change regression.
        painter = QtGui.QPainter(self)
        try:
            # painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            # painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            self._draw_labels_with_painter(painter, mv, proj)
            if self._is_packed:
                self._draw_axis_indicator(painter)
        finally:
            painter.end()

    def _reassert_gl_state(self) -> None:
        """Restore the raw-GL state we depend on.

        QPainter (used for atom-label / bond-distance overlays) silently
        toggles GL_DEPTH_TEST, GL_BLEND, GL_SCISSOR_TEST, GL_MULTISAMPLE and
        the viewport.  Re-assert everything we rely on at the top of every
        paintGL so a previous frame's overlay cannot poison this one.
        """
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
            dpr = float(self.devicePixelRatioF()) if hasattr(self, "devicePixelRatioF") else 1.0
            w = max(1, int(self.width() * dpr))
            h = max(1, int(self.height() * dpr))
            gl.glViewport(0, 0, w, h)
        except Exception:
            # Never let a state hiccup take the host app down.
            pass

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
        self._build_ellipsoid_geometry_batched()
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

        # Vertex layout: [cx, cy, cz, r, g, b, radius, corner_x, corner_y, selected]
        # 10 floats per vertex, 4 vertices per atom, 6 indices per atom
        verts = np.zeros((n * 4, 10), dtype=np.float32)
        idx = np.zeros(n * 6, dtype=np.uint32)

        for i, atom in enumerate(sphere_atoms):
            c = atom.center
            is_selected = atom.label in self.selected_atoms
            col = _SEL_COLOR if is_selected else atom.color_f
            sel_flag = 1.0 if is_selected else 0.0
            r = (sqrt(atom.u_iso) * _ADP_SCALE) if atom.u_iso is not None else atom.display_radius
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

            bond_key: tuple[str, str] = tuple(sorted((at1.label, at2.label)))  # type: ignore[assignment]
            is_selected = bond_key in self.selected_bonds
            if is_selected:
                bond_color = _SEL_COLOR
            else:
                bond_color = self._bond_rgb

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
                [_SEL_COLOR if a.label in self.selected_atoms else a.color_f
                 for a in atoms],
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

        self._geometry_dirty = False

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

    def _render_one_ellipsoid(self, atom: _Atom3D, mv: np.ndarray, proj: np.ndarray) -> None:
        prog = self._ellipsoid_prog
        gl.glUseProgram(prog)

        _set_mat4(prog, b"u_mv", mv)
        _set_mat4(prog, b"u_proj", proj)

        # Per-atom uniforms
        _set_vec3(prog, b"u_center", atom.center)
        is_selected = atom.label in self.selected_atoms
        _set_vec3(
            prog,
            b"u_color",
            np.array(
                _SEL_COLOR if is_selected else atom.color_f,
                dtype=np.float32,
            ),
        )
        _set_float(prog, b"u_radius", atom.adp_billboard_r)
        _set_float(prog, b"u_selected", 1.0 if is_selected else 0.0)

        A = atom.adp_A_matrix
        if A is not None:
            loc = gl.glGetUniformLocation(prog, b"u_ellipsoid_A")
            if loc >= 0:
                # OpenGL is column-major; numpy is row-major → transpose
                gl.glUniformMatrix3fv(loc, 1, False, A.astype(np.float32).T.copy())

        evecs = atom.u_eigvecs
        if evecs is not None:
            loc = gl.glGetUniformLocation(prog, b"u_world_evecs")
            if loc >= 0:
                gl.glUniformMatrix3fv(loc, 1, False, evecs.astype(np.float32).T.copy())

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

    def _draw_labels_overlay(self, mv: np.ndarray, proj: np.ndarray) -> None:
        """Draw atom labels as a QPainter overlay on top of the GL scene.

        Standalone entry point used by the pure-QWidget fallback path.  Inside
        the GL paintGL flow the painter is owned by ``_do_paintGL`` and the
        labels are drawn by :meth:`_draw_labels_with_painter` directly.
        """
        if not self.atoms:
            return
        painter = QtGui.QPainter(self)
        try:
            #painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
            #painter.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
            self._draw_labels_with_painter(painter, mv, proj)
        finally:
            painter.end()

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

        base_size = max(1, self.fontsize)
        hover_size = base_size + 4  # enlarge hovered label

        font = QtGui.QFont()
        font.setPixelSize(base_size)
        painter.setFont(font)
        painter.setPen(self.label_color)

        hover_label = self._hover_atom_label
        hover_atom = None

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
                int((ndc[0] + 1.0) * 0.5 * w),
                int((1.0 - ndc[1]) * 0.5 * h),
            )

        # Persistent labels (only when "Show Labels" is on). Hidden hydrogens
        # never get a label, and the hovered atom is drawn separately below
        # with a larger font.
        if self.labels:
            for atom in self.atoms:
                if not self.show_hydrogens_flag and atom.type_ in hydrogens:
                    continue
                if atom.label == hover_label:
                    hover_atom = atom
                    continue
                pt = project(atom)
                if pt is None:
                    continue
                painter.drawText(pt[0] + 4, pt[1] - 4, atom.label)
        elif hover_label is not None:
            for atom in self.atoms:
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
            painter.drawText(pt[0] + 4, pt[1] - 4, hover_atom.label)

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
        adps: dict[str, tuple[float, float, float, float, float, float]] | None = None,
        keep_view: bool = False,
    ) -> None:
        """Load a new molecule and (unless *keep_view*) reset the view."""
        self._is_packed = False
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
            symm = getattr(at, "symm_matrix", None)
            if symm is not None:
                symm_np = np.array(symm, dtype=float)
                a3d.symmgen = not np.allclose(symm_np, np.eye(3))
            else:
                a3d.symmgen = False

            if self._adp_map and self._cell and base_name in self._adp_map:
                try:
                    uvals = self._adp_map[base_name]
                    symm = getattr(at, "symm_matrix", None)
                    if symm is not None:
                        symm = np.array(symm, dtype=float)
                    a3d.u_cart = self._uij_to_cart(uvals, symm)
                    a3d.u_iso = float(np.trace(a3d.u_cart) / 3.0)
                    evals, evecs = np.linalg.eigh(a3d.u_cart)
                    if np.any(evals <= 0):
                        a3d.adp_valid = False
                    else:
                        a3d.adp_valid = True
                        a3d.u_eigvals = evals
                        a3d.u_eigvecs = evecs
                        # Billboard radius for the ellipsoid impostor quad
                        a3d.adp_billboard_r = float(
                            _ADP_SCALE * np.sqrt(np.max(evals)) * 1.2
                        )
                        A = np.linalg.inv(_ADP_SCALE ** 2 * a3d.u_cart)
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
        """Restore the rotation pivot to the molecule's geometric centre.

        Undoes any previous middle-click recentring: the pivot is moved back
        to the centre of the atom bounding box and the pan offset is cleared
        so the molecule is re-framed.  Rotation, zoom and selection are kept.
        """
        self._compute_molecule_bounds()
        self._pan = np.zeros(2, dtype=np.float32)
        self.update()

    def _align_to_reciprocal_axis(self, axis_index: int) -> None:
        """Align the view so that the reciprocal axis *axis_index* (0=a*, 1=b*, 2=c*) points towards the viewer.

        Does nothing if no unit cell is available.
        """
        if self._amatrix is None or self._cell is None:
            return

        # Reciprocal lattice vectors in Cartesian are rows of M^{-1}
        M_inv = np.linalg.inv(self._amatrix)
        recip_vec = M_inv[axis_index]
        recip_vec = recip_vec / np.linalg.norm(recip_vec)

        # Build rotation that maps recip_vec → +Z (screen normal, towards viewer)
        z_axis = recip_vec.astype(np.float32)

        # Choose an initial "up" vector; avoid degeneracy if z_axis is parallel to Y
        up_candidate = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(np.dot(z_axis, up_candidate)) > 0.99:
            up_candidate = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        x_axis = np.cross(up_candidate, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)

        # Target rotation: rows are the new basis vectors expressed in the original frame
        target_R = np.array([x_axis, y_axis, z_axis], dtype=np.float32)

        self._rot_matrix = target_R
        self.cumulative_R = target_R
        self.update()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Handle key-press events for reciprocal-axis alignment."""
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
        # Support Macs / trackpads without a middle mouse button by allowing
        # Alt/Option + left-click to act as a middle-click centring gesture.
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._mouse_moved
            and self._pressPos is not None
        ):
            # Alt/Option modifier recentres the rotation pivot (emulate middle-click)
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
        font.setPixelSize(max(1, self.fontsize))
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
            tip_y = origin_y - vy * arrow_len  # screen Y is inverted

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
        """Scroll wheel adjusts label font size."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.setLabelFont(self.fontsize + 2)
        elif delta < 0:
            self.setLabelFont(self.fontsize - 2)

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

        # Bonds are tested in the same pass so that a bond visually in front of
        # an atom behind it can still be selected.  _ray_bond_screen returns
        # viewspace t (same unit as the atom ray-casters), so the comparison is
        # consistent and the front-most object always wins.
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
            else:
                radius = (sqrt(float(atom.u_iso)) * _ADP_SCALE) if atom.u_iso is not None else atom.display_radius
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
        """Ray–ellipsoid intersection in view space.

        The ellipsoid is defined in world space by
        ``(P - C)^T · A · (P - C) = 1`` where ``A = a_matrix`` is the same
        matrix uploaded to the ellipsoid fragment shader (i.e.
        ``inv(_ADP_SCALE² · U_cart)``).  Picking against this matrix
        guarantees that the clickable region equals the visible silhouette.

        Returns parametric *t* of the front-face hit (smallest non-negative
        root) or ``None`` if the ray misses the ellipsoid.
        """
        # World→eye is mv.  For a point P_world, P_eye = R · P_world + T,
        # so (P_world - C) = R^T · (P_eye - C_eye).  The ellipsoid form
        # transforms to view space as q^T · M · q = 1 with M = R · A · R^T,
        # where q = P_eye - C_eye and R = mv[:3, :3].
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
        """Return the viewspace *t* of the closest point on bond *p1–p2* to the
        screen click *(sx, sy)* if that point is within 6 pixels of the projected
        line segment, else ``None``.

        The returned value uses the same unit as
        :meth:`_ray_sphere_hit_viewspace` and
        :meth:`_ray_ellipsoid_hit_viewspace` (``t = −z_eye``, positive, smaller
        = closer to the camera), so all three can be compared directly and the
        front-most object wins.
        """
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
