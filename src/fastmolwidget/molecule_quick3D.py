"""Qt Quick 3-D molecule display item.

Provides :class:`MoleculeQuick3D`, a
:class:`~PySide6.QtQuick.QQuickRhiItem` subclass (Qt ≥ 6.7) that renders the
same crystal-structure scene as
:class:`~fastmolwidget.molecule3D.MoleculeWidget3D` but lives inside a Qt
Quick scene graph, making it composable with QML.

Requires Qt **6.7 or later** (``QQuickRhiItem`` was added in Qt 6.7).

Quick-start
-----------

Register the type and create a QML engine::

    from fastmolwidget.molecule_quick3D import MoleculeQuick3D, setup_opengl_backend

    # Must be called BEFORE creating QGuiApplication / QApplication.
    setup_opengl_backend()

    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    qmlRegisterType(MoleculeQuick3D, "MolWidget", 1, 0, "MoleculeQuick3D")
    engine.load("main.qml")

Then in ``main.qml``::

    import QtQuick 2.15
    import MolWidget 1.0

    MoleculeQuick3D {
        id: mol
        width: 800; height: 600
        showAdps: true
        showLabels: false

        // Atom labels positioned by Python; rendered by QML
        Repeater {
            model: mol.labelPositions
            delegate: Text {
                x: modelData.x + 4;  y: modelData.y - 4
                text: modelData.text
                color: modelData.kind === "hover_atom" ? "blue" : "saddlebrown"
                font.pixelSize: mol.labelFontSize
                font.bold: modelData.kind !== "atom"
            }
        }
    }

Rendering overview
------------------
The same sphere impostor, batched-ellipsoid impostor and tessellated cylinder
pipeline used by :class:`~fastmolwidget.molecule3D.MoleculeWidget3D` is
re-used here.  All GLSL shaders target ``#version 150 core`` (OpenGL 3.2 Core
Profile).  View matrices are passed via a std140 uniform buffer object (UBO),
which is managed through the Qt RHI API.

Mouse controls
--------------
* **Left drag**   – rotate.
* **Right drag**  – zoom.
* **Middle drag** – pan.
* **Middle click** or **Alt + left click** – recentre the rotation pivot on
  the clicked atom.
* **Scroll wheel** – adjust atom-label font size.
* **Left click**  – select atom or bond; emit ``atomClicked`` / ``bondClicked``.
* **Ctrl + left click** – add to / remove from selection.

Threading model
---------------
All Python state lives on the GUI thread.  The :class:`_MolRhiRenderer` object
is created by Qt once per item and used exclusively on the render thread.
:meth:`_MolRhiRenderer.synchronize` is called by Qt while the GUI thread is
blocked; it copies the CPU-side geometry arrays and view matrices from the item
to the renderer.  After ``synchronize`` returns,
:meth:`_MolRhiRenderer.render` records RHI draw commands.

This module bypasses ``qtpy`` for ``QQuickRhiItem`` and ``QQuickRhiItemRenderer``
because those classes are not exposed uniformly by qtpy.  All other Qt types
are imported through qtpy as per project convention.

Platform notes
--------------
* **macOS** – Qt 6 on macOS defaults to the Metal rendering backend; you must
  call :func:`setup_opengl_backend` **before** creating ``QGuiApplication``
  (this forces the OpenGL backend for Qt Quick).
* **Linux / Windows** – works with mesa-gl, NVIDIA, AMD, Intel drivers.
  Requires ``libegl1`` on Debian/Ubuntu headless systems.
"""

from __future__ import annotations

from math import cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional

import numpy as np
from qtpy import QtCore, QtGui
from qtpy.QtCore import Property, Qt, Signal

from fastmolwidget.molecule2D import calc_volume
from fastmolwidget.molecule3D import (
    _ADP_SCALE,
    _BOND_HIT_TOLERANCE_PX,
    _DEFAULT_BOND_COLOR,
    _SEL_COLOR,
    _Atom3D,
    _make_cylinder,
    _normalize_rgb_color,
)
from fastmolwidget.sdm import Atomtuple

# ---------------------------------------------------------------------------
# Optional: QQuickRhiItem / QQuickRhiItemRenderer (Qt ≥ 6.7, not in qtpy)
# ---------------------------------------------------------------------------

_HAS_RHI: bool = False
_RhiItemBase: type
_RhiRendererBase: type

# RHI type aliases — populated in the try/except below; set to None as
# fallback so that method bodies referencing them never raise NameError at
# import time (they are guarded by the _HAS_RHI flag at runtime).
QQuickRhiItem = None  # type: ignore[assignment]
QQuickRhiItemRenderer = None  # type: ignore[assignment]
QRhiBuffer = None  # type: ignore[assignment]
QRhiGraphicsPipeline = None  # type: ignore[assignment]
QRhiShaderStage = None  # type: ignore[assignment]
QRhiVertexInputLayout = None  # type: ignore[assignment]
QRhiVertexInputAttribute = None  # type: ignore[assignment]
QRhiVertexInputBinding = None  # type: ignore[assignment]
QRhiShaderResourceBinding = None  # type: ignore[assignment]
QRhiCommandBuffer = None  # type: ignore[assignment]
QRhiViewport = None  # type: ignore[assignment]
QRhiDepthStencilClearValue = None  # type: ignore[assignment]
QShader = None  # type: ignore[assignment]
QShaderCode = None  # type: ignore[assignment]
QShaderKey = None  # type: ignore[assignment]
QShaderVersion = None  # type: ignore[assignment]

try:
    from PySide6.QtQuick import (  # type: ignore[import-not-found,no-redef]
        QQuickRhiItem,
        QQuickRhiItemRenderer,
    )
    from PySide6.QtGui import (  # type: ignore[import-not-found,no-redef]
        QRhiBuffer,
        QRhiCommandBuffer,
        QRhiDepthStencilClearValue,
        QRhiGraphicsPipeline,
        QRhiShaderResourceBinding,
        QRhiShaderStage,
        QRhiVertexInputAttribute,
        QRhiVertexInputBinding,
        QRhiVertexInputLayout,
        QRhiViewport,
        QShader,
        QShaderCode,
        QShaderKey,
        QShaderVersion,
    )

    _RhiItemBase = QQuickRhiItem
    _RhiRendererBase = QQuickRhiItemRenderer
    _HAS_RHI = True
except ImportError:
    try:
        from PyQt6.QtQuick import (  # type: ignore[import-not-found,no-redef]
            QQuickRhiItem,
            QQuickRhiItemRenderer,
        )
        from PyQt6.QtGui import (  # type: ignore[import-not-found,no-redef]
            QRhiBuffer,
            QRhiCommandBuffer,
            QRhiDepthStencilClearValue,
            QRhiGraphicsPipeline,
            QRhiShaderResourceBinding,
            QRhiShaderStage,
            QRhiVertexInputAttribute,
            QRhiVertexInputBinding,
            QRhiVertexInputLayout,
            QRhiViewport,
            QShader,
            QShaderCode,
            QShaderKey,
            QShaderVersion,
        )

        _RhiItemBase = QQuickRhiItem
        _RhiRendererBase = QQuickRhiItemRenderer
        _HAS_RHI = True
    except ImportError:
        pass

if not _HAS_RHI:
    from qtpy import QtWidgets as _QtWidgets

    _RhiItemBase = _QtWidgets.QWidget
    _RhiRendererBase = object

__all__ = ["MoleculeQuick3D", "setup_opengl_backend"]

# ---------------------------------------------------------------------------
# Surface format helper
# ---------------------------------------------------------------------------


def setup_opengl_backend() -> None:
    """Force Qt Quick to use an OpenGL scene graph.

    Call this **before** creating :class:`~qtpy.QtWidgets.QApplication` /
    ``QGuiApplication`` to ensure :class:`MoleculeQuick3D` gets an OpenGL
    context, especially on macOS where the default backend is Metal.

    Usage::

        from fastmolwidget.molecule_quick3D import setup_opengl_backend
        setup_opengl_backend()
        app = QGuiApplication(sys.argv)
    """
    try:
        try:
            from PySide6.QtQuick import QQuickWindow  # type: ignore[import-not-found]
        except ImportError:
            from PyQt6.QtQuick import QQuickWindow  # type: ignore[import-not-found]
        QQuickWindow.setSceneGraphBackend("opengl")
    except Exception:
        pass  # best-effort; failure is non-fatal


# ---------------------------------------------------------------------------
# GLSL 1.50 core-profile shaders (view matrices passed via std140 UBO)
# ---------------------------------------------------------------------------

# ── Sphere impostor ──────────────────────────────────────────────────────────
_SPHERE_VERT = """\
#version 150 core
layout(std140) uniform Matrices {
    mat4 u_mv;
    mat4 u_proj;
};
layout(location = 0) in vec3  a_center;
layout(location = 1) in vec3  a_color;
layout(location = 2) in float a_radius;
layout(location = 3) in vec2  a_corner;
layout(location = 4) in float a_selected;

out vec3  v_center_eye;
out vec3  v_color;
out float v_radius;
out vec2  v_corner;
out float v_selected;

void main() {
    v_color    = a_color;
    v_radius   = a_radius;
    v_corner   = a_corner;
    v_selected = a_selected;

    vec4 c_eye   = u_mv * vec4(a_center, 1.0);
    v_center_eye = c_eye.xyz;

    vec4 pos = c_eye;
    pos.xy  += a_corner * a_radius * 1.05;
    gl_Position = u_proj * pos;
}
"""

_SPHERE_FRAG = """\
#version 150 core
layout(std140) uniform Matrices {
    mat4 u_mv;
    mat4 u_proj;
};
in vec3  v_center_eye;
in vec3  v_color;
in float v_radius;
in vec2  v_corner;
in float v_selected;

out vec4 fragColor;

void main() {
    vec2  local_xy = v_corner * v_radius * 1.05;
    float xy2 = dot(local_xy, local_xy);
    float r2  = v_radius * v_radius;
    if (xy2 > r2) discard;

    float z_hit   = sqrt(r2 - xy2);
    vec3  local_hit = vec3(local_xy, z_hit);
    vec3  hit    = v_center_eye + local_hit;
    vec3  normal = normalize(local_hit);

    vec3  light     = normalize(vec3(1.0, 1.5, 2.0));
    float diff      = max(dot(normal, light), 0.0);
    float soft_diff = 0.25 + 0.75 * diff;
    float spec      = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 72.0);

    vec3 base_color = clamp(v_color * 1.08, 0.0, 1.0);
    vec3 color      = base_color * (0.50 + 0.35 * soft_diff) + vec3(0.16) * spec;
    fragColor       = vec4(clamp(color, 0.0, 1.0), 1.0);

    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
"""

# ── Cylinder mesh ─────────────────────────────────────────────────────────────
_CYLINDER_VERT = """\
#version 150 core
layout(std140) uniform Matrices {
    mat4 u_mv;
    mat4 u_proj;
    mat4 u_normal_mat;
};
layout(location = 0) in vec3  a_position;
layout(location = 1) in vec3  a_normal;
layout(location = 2) in vec3  a_color;
layout(location = 3) in float a_selected;

out vec3  v_normal_eye;
out vec3  v_pos_eye;
out vec3  v_color;
out float v_selected;

void main() {
    v_color      = a_color;
    v_selected   = a_selected;
    vec4 pos_e   = u_mv * vec4(a_position, 1.0);
    v_pos_eye    = pos_e.xyz;
    v_normal_eye = normalize(mat3(u_normal_mat) * a_normal);
    gl_Position  = u_proj * pos_e;
}
"""

_CYLINDER_FRAG = """\
#version 150 core
in vec3  v_normal_eye;
in vec3  v_pos_eye;
in vec3  v_color;
in float v_selected;

out vec4 fragColor;

void main() {
    vec3 color;
    if (v_selected > 0.5) {
        color = v_color;
    } else {
        vec3  normal = normalize(v_normal_eye);
        vec3  light  = normalize(vec3(1.0, 1.5, 2.0));
        float diff   = max(dot(normal, light), 0.0);
        float spec   = pow(max(dot(reflect(-light, normal),
                                   normalize(-v_pos_eye)), 0.0), 32.0);
        color = v_color * (0.45 + 0.55 * diff) + vec3(0.30) * spec;
    }
    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
"""

# ── Batched ellipsoid impostor ────────────────────────────────────────────────
_ELLIPSOID_BATCH_VERT = """\
#version 150 core
layout(std140) uniform Matrices {
    mat4 u_mv;
    mat4 u_proj;
};
layout(location =  0) in vec2  a_corner;
layout(location =  1) in vec3  a_center;
layout(location =  2) in vec3  a_color;
layout(location =  3) in float a_radius;
layout(location =  4) in float a_selected;
layout(location =  5) in vec3  a_A_col0;
layout(location =  6) in vec3  a_A_col1;
layout(location =  7) in vec3  a_A_col2;
layout(location =  8) in vec3  a_evec0;
layout(location =  9) in vec3  a_evec1;
layout(location = 10) in vec3  a_evec2;

out vec3  v_center_eye;
out vec3  v_color;
out float v_radius;
out vec2  v_corner;
out float v_selected;
out vec3  v_A_col0;
out vec3  v_A_col1;
out vec3  v_A_col2;
out vec3  v_evec0;
out vec3  v_evec1;
out vec3  v_evec2;

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
#version 150 core
layout(std140) uniform Matrices {
    mat4 u_mv;
    mat4 u_proj;
};
in vec3  v_center_eye;
in vec3  v_color;
in float v_radius;
in vec2  v_corner;
in float v_selected;
in vec3  v_A_col0;
in vec3  v_A_col1;
in vec3  v_A_col2;
in vec3  v_evec0;
in vec3  v_evec1;
in vec3  v_evec2;

out vec4 fragColor;

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

    fragColor = vec4(clamp(color, 0.0, 1.0), 1.0);

    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
"""

# ---------------------------------------------------------------------------
# Renderer (lives on the Qt Quick render thread)
# ---------------------------------------------------------------------------

_ORTHO_VIEW_MARGIN: float = 1.6


def _make_glsl_shader(glsl_src: str, stage: object) -> object:
    """Wrap a GLSL source string in a :class:`QShader` for the RHI backend.

    Only called when *_HAS_RHI* is True, so the RHI type aliases are valid.
    """
    s = QShader()  # type: ignore[call-arg]
    s.setStage(stage)  # type: ignore[arg-type]
    key = QShaderKey(  # type: ignore[call-arg]
        QShader.Source.GlslShader,  # type: ignore[union-attr]
        QShaderVersion(150),  # type: ignore[call-arg]
    )
    s.setShader(key, QShaderCode(glsl_src.encode()))  # type: ignore[call-arg]
    return s


def _pack_mat4(mat: np.ndarray) -> bytes:
    """Return the 64-byte column-major (std140) representation of *mat*."""
    return mat.astype(np.float32).T.copy().tobytes()


class _MolRhiRenderer(_RhiRendererBase):  # type: ignore[valid-type,misc]
    """Qt RHI renderer for :class:`MoleculeQuick3D`.

    Lives exclusively on the Qt Quick render thread.  All GPU resource
    allocation happens in :meth:`initialize`; draw commands are recorded in
    :meth:`render`.  CPU-side geometry and view-matrix data arrive via
    :meth:`synchronize` while the GUI thread is blocked.
    """

    def __init__(self) -> None:
        super().__init__()

        self._initialized: bool = False

        # ── RHI resources (created in initialize()) ─────────────────────────
        # Uniform buffer objects
        self._sphere_ubo = None
        self._cylinder_ubo = None
        self._ellipsoid_ubo = None
        # Geometry vertex / index buffers (resized as needed)
        self._sphere_vbo = None
        self._sphere_ibo = None
        self._cylinder_vbo = None
        self._cylinder_ibo = None
        self._ellipsoid_vbo = None
        self._ellipsoid_ibo = None
        # Shader resource bindings
        self._sphere_srb = None
        self._cylinder_srb = None
        self._ellipsoid_srb = None
        # Graphics pipelines
        self._sphere_pipeline = None
        self._cylinder_pipeline = None
        self._ellipsoid_pipeline = None

        # ── Geometry arrays (copied from item in synchronize) ────────────────
        self._sphere_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._sphere_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._sphere_count: int = 0
        self._cylinder_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._cylinder_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._cylinder_count: int = 0
        self._ellipsoid_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._ellipsoid_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._ellipsoid_count: int = 0
        self._geometry_dirty: bool = False

        # ── View state ───────────────────────────────────────────────────────
        self._mv: np.ndarray = np.eye(4, dtype=np.float32)
        self._proj: np.ndarray = np.eye(4, dtype=np.float32)
        self._bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._show_adps: bool = True
        self._has_atoms: bool = False

    # ------------------------------------------------------------------
    # QQuickRhiItemRenderer interface
    # ------------------------------------------------------------------

    def initialize(self, cb: object) -> None:  # type: ignore[override]
        """Create all RHI resources.

        Called on the render thread when the renderer is first created (and
        again if the underlying QRhi instance is replaced).
        """
        if not _HAS_RHI:
            return

        rhi = self.rhi()  # type: ignore[attr-defined]
        if self._initialized:
            # Guard against re-entry without a genuine QRhi change.
            return

        try:
            self._create_resources(rhi)
            self._initialized = True
        except Exception as exc:
            print(f"[MoleculeQuick3D] RHI init failed: {exc}")
            self._initialized = False

    def synchronize(self, item: object) -> None:  # type: ignore[override]
        """Copy state from *item* (GUI thread, which is blocked right now)."""
        mol: MoleculeQuick3D = item  # type: ignore[assignment]

        if mol._geometry_dirty:
            self._sphere_verts = mol._sphere_verts.copy()
            self._sphere_idx = mol._sphere_idx.copy()
            self._sphere_count = mol._sphere_count
            self._cylinder_verts = mol._cylinder_verts.copy()
            self._cylinder_idx = mol._cylinder_idx.copy()
            self._cylinder_count = mol._cylinder_count
            self._ellipsoid_verts = mol._ellipsoid_verts.copy()
            self._ellipsoid_idx = mol._ellipsoid_idx.copy()
            self._ellipsoid_count = mol._ellipsoid_count
            self._geometry_dirty = True
            mol._geometry_dirty = False

        self._mv = mol._compute_mv_matrix()
        self._proj = mol._compute_proj_matrix()
        self._bg_rgb = mol._bg_rgb
        self._show_adps = mol._show_adps
        self._has_atoms = bool(mol.atoms)

    def render(self, cb: object) -> None:  # type: ignore[override]
        """Record draw commands into *cb*."""
        if not _HAS_RHI or not self._initialized:
            return

        rhi = self.rhi()  # type: ignore[attr-defined]
        rub = rhi.nextResourceUpdateBatch()  # type: ignore[union-attr]

        if self._geometry_dirty:
            self._upload_geometry(rub)
            self._geometry_dirty = False

        self._upload_ubos(rub)

        r, g, b = self._bg_rgb
        clear_color = QtGui.QColor(int(r * 255), int(g * 255), int(b * 255))
        clear_ds = QRhiDepthStencilClearValue()  # type: ignore[call-arg]
        clear_ds.setDepthClearValue(1.0)
        clear_ds.setStencilClearValue(0)

        rt = self.renderTarget()  # type: ignore[attr-defined]
        cb.beginPass(rt, clear_color, clear_ds, rub)  # type: ignore[union-attr]

        sz = rt.pixelSize()
        w, h = float(sz.width()), float(sz.height())
        viewport = QRhiViewport()  # type: ignore[call-arg]
        viewport.setViewport(0.0, 0.0, w, h)

        if self._has_atoms:
            if self._cylinder_count > 0:
                cb.setGraphicsPipeline(self._cylinder_pipeline)  # type: ignore[union-attr]
                cb.setViewport(viewport)  # type: ignore[union-attr]
                cb.setShaderResources(self._cylinder_srb)  # type: ignore[union-attr]
                cb.setVertexInput(  # type: ignore[union-attr]
                    0, [(self._cylinder_vbo, 0)],
                    self._cylinder_ibo, 0,
                    QRhiCommandBuffer.IndexFormat.IndexUInt32,  # type: ignore[union-attr]
                )
                cb.drawIndexed(self._cylinder_count)  # type: ignore[union-attr]

            if self._sphere_count > 0:
                cb.setGraphicsPipeline(self._sphere_pipeline)  # type: ignore[union-attr]
                cb.setViewport(viewport)  # type: ignore[union-attr]
                cb.setShaderResources(self._sphere_srb)  # type: ignore[union-attr]
                cb.setVertexInput(  # type: ignore[union-attr]
                    0, [(self._sphere_vbo, 0)],
                    self._sphere_ibo, 0,
                    QRhiCommandBuffer.IndexFormat.IndexUInt32,  # type: ignore[union-attr]
                )
                cb.drawIndexed(self._sphere_count)  # type: ignore[union-attr]

            if self._show_adps and self._ellipsoid_count > 0:
                cb.setGraphicsPipeline(self._ellipsoid_pipeline)  # type: ignore[union-attr]
                cb.setViewport(viewport)  # type: ignore[union-attr]
                cb.setShaderResources(self._ellipsoid_srb)  # type: ignore[union-attr]
                cb.setVertexInput(  # type: ignore[union-attr]
                    0, [(self._ellipsoid_vbo, 0)],
                    self._ellipsoid_ibo, 0,
                    QRhiCommandBuffer.IndexFormat.IndexUInt32,  # type: ignore[union-attr]
                )
                cb.drawIndexed(self._ellipsoid_count)  # type: ignore[union-attr]

        cb.endPass()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Resource creation helpers
    # ------------------------------------------------------------------

    def _create_resources(self, rhi: object) -> None:
        """Allocate all GPU resources (UBOs, geometry buffers, SRBs, pipelines)."""
        rpd = self.renderTarget().renderPassDescriptor()  # type: ignore[union-attr]

        BufType = QRhiBuffer.Type  # type: ignore[union-attr]
        BufUsage = QRhiBuffer.UsageFlag  # type: ignore[union-attr]

        # ── Uniform buffer objects ───────────────────────────────────────────
        # Sphere / ellipsoid UBO: u_mv (64) + u_proj (64) = 128 bytes
        self._sphere_ubo = rhi.newBuffer(BufType.Dynamic, BufUsage.UniformBuffer, 128)  # type: ignore[union-attr]
        self._sphere_ubo.create()
        self._ellipsoid_ubo = rhi.newBuffer(BufType.Dynamic, BufUsage.UniformBuffer, 128)  # type: ignore[union-attr]
        self._ellipsoid_ubo.create()
        # Cylinder UBO: u_mv (64) + u_proj (64) + u_normal_mat as mat4 (64) = 192 bytes
        self._cylinder_ubo = rhi.newBuffer(BufType.Dynamic, BufUsage.UniformBuffer, 192)  # type: ignore[union-attr]
        self._cylinder_ubo.create()

        # ── Geometry buffers (placeholder minimum size) ──────────────────────
        self._sphere_vbo = rhi.newBuffer(BufType.Dynamic, BufUsage.VertexBuffer, 4)  # type: ignore[union-attr]
        self._sphere_vbo.create()
        self._sphere_ibo = rhi.newBuffer(BufType.Dynamic, BufUsage.IndexBuffer, 4)  # type: ignore[union-attr]
        self._sphere_ibo.create()
        self._cylinder_vbo = rhi.newBuffer(BufType.Dynamic, BufUsage.VertexBuffer, 4)  # type: ignore[union-attr]
        self._cylinder_vbo.create()
        self._cylinder_ibo = rhi.newBuffer(BufType.Dynamic, BufUsage.IndexBuffer, 4)  # type: ignore[union-attr]
        self._cylinder_ibo.create()
        self._ellipsoid_vbo = rhi.newBuffer(BufType.Dynamic, BufUsage.VertexBuffer, 4)  # type: ignore[union-attr]
        self._ellipsoid_vbo.create()
        self._ellipsoid_ibo = rhi.newBuffer(BufType.Dynamic, BufUsage.IndexBuffer, 4)  # type: ignore[union-attr]
        self._ellipsoid_ibo.create()

        # ── Shader resource bindings ─────────────────────────────────────────
        stages = (
            QRhiShaderResourceBinding.StageFlag.VertexStage  # type: ignore[union-attr]
            | QRhiShaderResourceBinding.StageFlag.FragmentStage  # type: ignore[union-attr]
        )

        def make_srb(ubo: object) -> object:
            srb = rhi.newShaderResourceBindings()  # type: ignore[union-attr]
            srb.setBindings([
                QRhiShaderResourceBinding.uniformBuffer(0, stages, ubo)  # type: ignore[union-attr]
            ])
            if not srb.create():
                raise RuntimeError("SRB creation failed")
            return srb

        self._sphere_srb = make_srb(self._sphere_ubo)
        self._cylinder_srb = make_srb(self._cylinder_ubo)
        self._ellipsoid_srb = make_srb(self._ellipsoid_ubo)

        # ── Graphics pipelines ───────────────────────────────────────────────
        self._sphere_pipeline = self._make_pipeline(
            rhi, rpd, _SPHERE_VERT, _SPHERE_FRAG,
            self._sphere_srb,
            stride=10 * 4,
            attributes=[
                (0, QRhiVertexInputAttribute.Format.Float3, 0),   # a_center  # type: ignore[union-attr]
                (1, QRhiVertexInputAttribute.Format.Float3, 12),  # a_color   # type: ignore[union-attr]
                (2, QRhiVertexInputAttribute.Format.Float,  24),  # a_radius  # type: ignore[union-attr]
                (3, QRhiVertexInputAttribute.Format.Float2, 28),  # a_corner  # type: ignore[union-attr]
                (4, QRhiVertexInputAttribute.Format.Float,  36),  # a_selected# type: ignore[union-attr]
            ],
        )
        self._cylinder_pipeline = self._make_pipeline(
            rhi, rpd, _CYLINDER_VERT, _CYLINDER_FRAG,
            self._cylinder_srb,
            stride=10 * 4,
            attributes=[
                (0, QRhiVertexInputAttribute.Format.Float3,  0),  # a_position# type: ignore[union-attr]
                (1, QRhiVertexInputAttribute.Format.Float3, 12),  # a_normal  # type: ignore[union-attr]
                (2, QRhiVertexInputAttribute.Format.Float3, 24),  # a_color   # type: ignore[union-attr]
                (3, QRhiVertexInputAttribute.Format.Float,  36),  # a_selected# type: ignore[union-attr]
            ],
        )
        self._ellipsoid_pipeline = self._make_pipeline(
            rhi, rpd, _ELLIPSOID_BATCH_VERT, _ELLIPSOID_BATCH_FRAG,
            self._ellipsoid_srb,
            stride=28 * 4,
            attributes=[
                (0,  QRhiVertexInputAttribute.Format.Float2,   0),  # a_corner  # type: ignore[union-attr]
                (1,  QRhiVertexInputAttribute.Format.Float3,   8),  # a_center  # type: ignore[union-attr]
                (2,  QRhiVertexInputAttribute.Format.Float3,  20),  # a_color   # type: ignore[union-attr]
                (3,  QRhiVertexInputAttribute.Format.Float,   32),  # a_radius  # type: ignore[union-attr]
                (4,  QRhiVertexInputAttribute.Format.Float,   36),  # a_selected# type: ignore[union-attr]
                (5,  QRhiVertexInputAttribute.Format.Float3,  40),  # a_A_col0  # type: ignore[union-attr]
                (6,  QRhiVertexInputAttribute.Format.Float3,  52),  # a_A_col1  # type: ignore[union-attr]
                (7,  QRhiVertexInputAttribute.Format.Float3,  64),  # a_A_col2  # type: ignore[union-attr]
                (8,  QRhiVertexInputAttribute.Format.Float3,  76),  # a_evec0   # type: ignore[union-attr]
                (9,  QRhiVertexInputAttribute.Format.Float3,  88),  # a_evec1   # type: ignore[union-attr]
                (10, QRhiVertexInputAttribute.Format.Float3, 100),  # a_evec2   # type: ignore[union-attr]
            ],
        )

    def _make_pipeline(
        self,
        rhi: object,
        rpd: object,
        vert_src: str,
        frag_src: str,
        srb: object,
        stride: int,
        attributes: list[tuple[int, object, int]],
    ) -> object:
        """Create and return a :class:`QRhiGraphicsPipeline`."""
        vs = _make_glsl_shader(vert_src, QShader.Stage.VertexStage)    # type: ignore[union-attr]
        fs = _make_glsl_shader(frag_src, QShader.Stage.FragmentStage)  # type: ignore[union-attr]
        if not vs.isValid() or not fs.isValid():
            raise RuntimeError("Shader creation failed")

        vs_stage = QRhiShaderStage()  # type: ignore[call-arg]
        vs_stage.setType(QRhiShaderStage.Type.Vertex)  # type: ignore[union-attr]
        vs_stage.setShader(vs)
        fs_stage = QRhiShaderStage()  # type: ignore[call-arg]
        fs_stage.setType(QRhiShaderStage.Type.Fragment)  # type: ignore[union-attr]
        fs_stage.setShader(fs)

        binding = QRhiVertexInputBinding()  # type: ignore[call-arg]
        binding.setStride(stride)

        attrs = []
        for loc, fmt, offset in attributes:
            a = QRhiVertexInputAttribute()  # type: ignore[call-arg]
            a.setLocation(loc)
            a.setBinding(0)
            a.setFormat(fmt)
            a.setOffset(offset)
            attrs.append(a)

        layout = QRhiVertexInputLayout()  # type: ignore[call-arg]
        layout.setBindings([binding])
        layout.setAttributes(attrs)

        blend = QRhiGraphicsPipeline.TargetBlend()  # type: ignore[union-attr]

        pipeline = rhi.newGraphicsPipeline()  # type: ignore[union-attr]
        pipeline.setShaderStages([vs_stage, fs_stage])
        pipeline.setVertexInputLayout(layout)
        pipeline.setShaderResourceBindings(srb)
        pipeline.setRenderPassDescriptor(rpd)
        pipeline.setTopology(QRhiGraphicsPipeline.Topology.Triangles)  # type: ignore[union-attr]
        pipeline.setDepthTest(True)
        pipeline.setDepthWrite(True)
        pipeline.setDepthOp(QRhiGraphicsPipeline.CompareOp.LessOrEqual)  # type: ignore[union-attr]
        pipeline.setCullMode(QRhiGraphicsPipeline.CullMode.None_)  # type: ignore[union-attr]
        pipeline.setTargetBlends([blend])
        if not pipeline.create():
            raise RuntimeError(f"Pipeline creation failed for {vert_src[:30]!r}")
        return pipeline

    # ------------------------------------------------------------------
    # Per-frame resource upload helpers
    # ------------------------------------------------------------------

    def _resize_and_upload(self, buf: object, data: bytes, rub: object) -> None:
        """Ensure *buf* is large enough for *data* and upload it."""
        n = len(data)
        if n == 0:
            return
        if buf.size() < n:  # type: ignore[union-attr]
            buf.setSize(n)  # type: ignore[union-attr]
            if not buf.create():  # type: ignore[union-attr]
                raise RuntimeError("Buffer resize failed")
        rub.updateDynamicBuffer(buf, 0, n, data)  # type: ignore[union-attr]

    def _upload_geometry(self, rub: object) -> None:
        """Upload CPU-side geometry arrays to the GPU buffers."""
        if self._sphere_verts.size > 0:
            self._resize_and_upload(
                self._sphere_vbo, self._sphere_verts.tobytes(), rub)
            self._resize_and_upload(
                self._sphere_ibo, self._sphere_idx.tobytes(), rub)
        if self._cylinder_verts.size > 0:
            self._resize_and_upload(
                self._cylinder_vbo, self._cylinder_verts.tobytes(), rub)
            self._resize_and_upload(
                self._cylinder_ibo, self._cylinder_idx.tobytes(), rub)
        if self._ellipsoid_verts.size > 0:
            self._resize_and_upload(
                self._ellipsoid_vbo, self._ellipsoid_verts.tobytes(), rub)
            self._resize_and_upload(
                self._ellipsoid_ibo, self._ellipsoid_idx.tobytes(), rub)

    def _upload_ubos(self, rub: object) -> None:
        """Upload view matrices to the uniform buffer objects."""
        mv_bytes = _pack_mat4(self._mv)
        proj_bytes = _pack_mat4(self._proj)
        sphere_data = mv_bytes + proj_bytes  # 128 bytes

        rub.updateDynamicBuffer(self._sphere_ubo, 0, 128, sphere_data)  # type: ignore[union-attr]
        rub.updateDynamicBuffer(self._ellipsoid_ubo, 0, 128, sphere_data)  # type: ignore[union-attr]

        # Normal matrix for cylinder lighting (mat4 storing 3×3 in top-left).
        try:
            nm3 = np.linalg.inv(self._mv[:3, :3]).T.astype(np.float32)
        except np.linalg.LinAlgError:
            nm3 = np.eye(3, dtype=np.float32)
        nm4 = np.eye(4, dtype=np.float32)
        nm4[:3, :3] = nm3
        nm_bytes = _pack_mat4(nm4)  # 64 bytes
        rub.updateDynamicBuffer(  # type: ignore[union-attr]
            self._cylinder_ubo, 0, 192, sphere_data + nm_bytes)


# ---------------------------------------------------------------------------
# Qt Quick item (lives on the GUI thread)
# ---------------------------------------------------------------------------

class MoleculeQuick3D(_RhiItemBase):  # type: ignore[valid-type,misc]
    """3-D molecule display item for Qt Quick / QML.

    Provides the same crystal-structure rendering as
    :class:`~fastmolwidget.molecule3D.MoleculeWidget3D` inside a Qt Quick
    scene, allowing it to be embedded in QML layouts and composed with other
    Quick items.

    **Prerequisite**: call :func:`setup_opengl_backend` before creating
    ``QGuiApplication`` so that Qt Quick uses an OpenGL scene graph.

    **QML registration**::

        from qtpy.QtQml import qmlRegisterType
        qmlRegisterType(MoleculeQuick3D, "MolWidget", 1, 0, "MoleculeQuick3D")

    Parameters
    ----------
    parent:
        Optional parent :class:`~PySide6.QtQuick.QQuickItem`.

    Signals
    -------
    atomClicked(str):
        Emitted when the user clicks on an atom; carries the atom label.
    bondClicked(str, str):
        Emitted when the user clicks on a bond; carries the two atom labels.
    labelPositionsChanged():
        Emitted whenever the ``labelPositions`` property changes (molecule
        load, view change, resize).

    Notes
    -----
    Atom labels and hover text are **not** rendered by OpenGL; they are
    exposed through the ``labelPositions`` property so that QML ``Text``
    delegates can position them.  See the module docstring for an example.
    """

    atomClicked = Signal(str)
    bondClicked = Signal(str, str)
    labelPositionsChanged = Signal()

    # Per-property change signals (used as QML notify)
    _showAdpsChanged = Signal(bool)
    _showLabelsChanged = Signal(bool)
    _showHydrogensChanged = Signal(bool)
    _bondWidthChanged = Signal(int)
    _bondColorChanged = Signal(str)
    _backgroundColorChanged = Signal(str)
    _labelFontSizeChanged = Signal(int)

    _ORTHO_VIEW_MARGIN: float = _ORTHO_VIEW_MARGIN

    def __init__(self, parent: object = None) -> None:
        if not _HAS_RHI:
            raise RuntimeError(
                "MoleculeQuick3D requires QQuickRhiItem (Qt ≥ 6.7) which is not "
                "available in the current Qt installation.  Install PySide6 ≥ 6.7 "
                "or PyQt6 ≥ 6.7 with QtQuick support."
            )
        super().__init__(parent)  # type: ignore[call-arg]

        # ── Molecule data ────────────────────────────────────────────────────
        self.atoms: list[_Atom3D] = []
        self.connections: tuple = ()
        self._cell: tuple[float, ...] | None = None
        self._adp_map: dict = {}
        self._astar: float = 0.0
        self._bstar: float = 0.0
        self._cstar: float = 0.0
        self._amatrix: np.ndarray = np.eye(3, dtype=float)

        # ── Display state ────────────────────────────────────────────────────
        self.fontsize: int = 18
        self.bond_width: int = 3
        self.labels: bool = False
        self.show_hydrogens_flag: bool = True
        self.selected_atoms: set[str] = set()
        self.selected_bonds: set[tuple[str, str]] = set()
        self._show_adps: bool = True
        self._bg_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0)
        self._bond_rgb: tuple[float, float, float] = _DEFAULT_BOND_COLOR

        # ── 3-D view state ───────────────────────────────────────────────────
        self._rot_matrix: np.ndarray = np.eye(3, dtype=np.float32)
        self._zoom: float = 1.0
        self._pan: np.ndarray = np.zeros(2, dtype=np.float32)
        self._molecule_center: np.ndarray = np.zeros(3, dtype=np.float32)
        self._molecule_radius: float = 10.0
        self.cumulative_R: np.ndarray = np.eye(3, dtype=np.float32)

        # ── CPU-side geometry buffers ────────────────────────────────────────
        self._sphere_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._sphere_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._sphere_count: int = 0
        self._cylinder_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._cylinder_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._cylinder_count: int = 0
        self._ellipsoid_verts: np.ndarray = np.empty(0, dtype=np.float32)
        self._ellipsoid_idx: np.ndarray = np.empty(0, dtype=np.uint32)
        self._ellipsoid_count: int = 0
        self._adp_draw_list: list[_Atom3D] = []
        self._geometry_dirty: bool = False

        # ── Mouse tracking ───────────────────────────────────────────────────
        self._lastPos: QtCore.QPointF | None = None
        self._pressPos: QtCore.QPointF | None = None
        self._mouse_moved: bool = False
        self._hover_atom_label: str | None = None
        self._hover_bond: tuple[str, str] | None = None
        self._hover_bond_distance: float | None = None
        self._hover_cursor: QtCore.QPointF | None = None

        # ── Label positions exposed to QML ───────────────────────────────────
        # List of dicts: {"kind": "atom"|"hover_atom"|"hover_bond",
        #                  "text": str, "x": float, "y": float}
        self._label_positions: list[dict] = []

        # ── Accept mouse / hover events ──────────────────────────────────────
        self.setAcceptedMouseButtons(
            Qt.MouseButton.LeftButton
            | Qt.MouseButton.RightButton
            | Qt.MouseButton.MiddleButton
        )
        self.setAcceptHoverEvents(True)

        # Default no-op connections so signals can always be emitted safely.
        self.atomClicked.connect(lambda _x: None)
        self.bondClicked.connect(lambda _x, _y: None)

    # ------------------------------------------------------------------
    # QQuickRhiItem interface
    # ------------------------------------------------------------------

    def createRenderer(self) -> _MolRhiRenderer:  # type: ignore[override]
        """Return the render-thread renderer object."""
        return _MolRhiRenderer()

    # ------------------------------------------------------------------
    # QML properties
    # ------------------------------------------------------------------

    @Property(bool, notify=_showAdpsChanged)  # type: ignore[misc]
    def showAdps(self) -> bool:
        """Toggle ADP ellipsoid display."""
        return self._show_adps

    @showAdps.setter  # type: ignore[misc]
    def showAdps(self, value: bool) -> None:
        if self._show_adps != value:
            self._show_adps = value
            self._showAdpsChanged.emit(value)
            if self.atoms:
                self._build_geometry()
            self._request_update()

    @Property(bool, notify=_showLabelsChanged)  # type: ignore[misc]
    def showLabels(self) -> bool:
        """Toggle atom label overlay."""
        return self.labels

    @showLabels.setter  # type: ignore[misc]
    def showLabels(self, value: bool) -> None:
        if self.labels != value:
            self.labels = value
            self._showLabelsChanged.emit(value)
            self._request_update()

    @Property(bool, notify=_showHydrogensChanged)  # type: ignore[misc]
    def showHydrogens(self) -> bool:
        """Toggle hydrogen atom visibility."""
        return self.show_hydrogens_flag

    @showHydrogens.setter  # type: ignore[misc]
    def showHydrogens(self, value: bool) -> None:
        if self.show_hydrogens_flag != value:
            self.show_hydrogens_flag = value
            self._showHydrogensChanged.emit(value)
            if self.atoms:
                self._build_geometry()
            self._request_update()

    @Property(int, notify=_bondWidthChanged)  # type: ignore[misc]
    def bondWidth(self) -> int:
        """Bond cylinder radius control."""
        return self.bond_width

    @bondWidth.setter  # type: ignore[misc]
    def bondWidth(self, value: int) -> None:
        if self.bond_width != value:
            self.bond_width = value
            self._bondWidthChanged.emit(value)
            if self.atoms:
                self._build_geometry()
            self._request_update()

    @Property(str, notify=_bondColorChanged)  # type: ignore[misc]
    def bondColor(self) -> str:
        """Bond colour as ``#RRGGBB`` hex string."""
        r, g, b = self._bond_rgb
        return "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )

    @bondColor.setter  # type: ignore[misc]
    def bondColor(self, value: str) -> None:
        self.set_bond_color(value)

    @Property(str, notify=_backgroundColorChanged)  # type: ignore[misc]
    def backgroundColor(self) -> str:
        """Background colour as ``#RRGGBB`` hex string."""
        r, g, b = self._bg_rgb
        return "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )

    @backgroundColor.setter  # type: ignore[misc]
    def backgroundColor(self, value: str) -> None:
        self.set_background_color(QtGui.QColor(value))

    @Property(int, notify=_labelFontSizeChanged)  # type: ignore[misc]
    def labelFontSize(self) -> int:
        """Atom label font size in pixels."""
        return self.fontsize

    @labelFontSize.setter  # type: ignore[misc]
    def labelFontSize(self, value: int) -> None:
        new_size = max(1, value)
        if self.fontsize != new_size:
            self.fontsize = new_size
            self._labelFontSizeChanged.emit(new_size)
            self._request_update()

    @Property("QVariantList", notify=labelPositionsChanged)  # type: ignore[misc]
    def labelPositions(self) -> list[dict]:
        """Screen-space positions for atom labels and hover overlays.

        Returns a list of dicts, each with keys:

        * ``"kind"`` – ``"atom"``, ``"hover_atom"``, or ``"hover_bond"``
        * ``"text"`` – label / distance string
        * ``"x"`` – screen X in logical pixels
        * ``"y"`` – screen Y in logical pixels
        """
        return self._label_positions

    # ------------------------------------------------------------------
    # Python-style public API  (mirrors MoleculeWidget3D)
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

    def clear(self) -> None:
        """Remove all atoms and bonds."""
        self.open_molecule(atoms=[])

    def show_adps(self, value: bool) -> None:
        """Toggle ADP ellipsoid / isotropic sphere display."""
        self.showAdps = value

    def show_labels(self, value: bool) -> None:
        """Toggle atom label visibility."""
        self.showLabels = value

    def set_labels_visible(self, visible: bool) -> None:
        """Toggle atom label visibility (alias for :meth:`show_labels`)."""
        self.show_labels(visible)

    def show_hydrogens(self, value: bool) -> None:
        """Toggle hydrogen atom and bond display."""
        self.showHydrogens = value

    def set_bond_width(self, width: int) -> None:
        """Set the bond width.  Triggers a geometry rebuild."""
        self.bondWidth = width

    def set_bond_color(
        self,
        color: QtGui.QColor | str | tuple[float, float, float] | tuple[int, int, int],
    ) -> None:
        """Set the default colour used for all non-selected bonds."""
        self._bond_rgb = _normalize_rgb_color(color)
        hex_str = "#{:02x}{:02x}{:02x}".format(
            int(self._bond_rgb[0] * 255),
            int(self._bond_rgb[1] * 255),
            int(self._bond_rgb[2] * 255),
        )
        self._bondColorChanged.emit(hex_str)
        if self.atoms:
            self._build_geometry()
        self._request_update()

    def set_background_color(self, color: QtGui.QColor) -> None:
        """Set the widget background colour."""
        self._bg_rgb = (color.redF(), color.greenF(), color.blueF())
        hex_str = "#{:02x}{:02x}{:02x}".format(
            int(self._bg_rgb[0] * 255),
            int(self._bg_rgb[1] * 255),
            int(self._bg_rgb[2] * 255),
        )
        self._backgroundColorChanged.emit(hex_str)
        self._request_update()

    def setLabelFont(self, font_size: int) -> None:
        """Set atom label pixel size."""
        self.labelFontSize = font_size

    def reset_view(self) -> None:
        """Reset zoom, rotation and pan to initial defaults."""
        self._rot_matrix = np.eye(3, dtype=np.float32)
        self.cumulative_R = np.eye(3, dtype=np.float32)
        self._zoom = 1.0
        self._pan = np.zeros(2, dtype=np.float32)
        self._request_update()

    def reset_rotation_center(self) -> None:
        """Restore the rotation pivot to the molecule's geometric centre.

        Undoes any previous middle-click recentring.
        """
        self._compute_molecule_bounds()
        self._pan = np.zeros(2, dtype=np.float32)
        self._request_update()

    def save_image(self, filename: Path, image_scale: float = 1.5) -> None:
        """Save the current view to an image file.

        Uses :meth:`QQuickWindow.grabWindow` to capture the scene and then
        crops to the item's bounding rectangle within the window.

        Parameters
        ----------
        filename:
            Output path; the format is inferred from the extension.
        image_scale:
            Scale factor applied after cropping.
        """
        window = self.window()
        if window is None:
            return
        image = window.grabWindow()
        pixmap = QtGui.QPixmap.fromImage(image)

        # Crop to this item's scene rect.
        scene_rect = self.mapRectToScene(
            QtCore.QRectF(0.0, 0.0, self.width(), self.height())
        )
        crop = scene_rect.toRect()
        if crop.isValid():
            pixmap = pixmap.copy(crop)

        if image_scale != 1.0:
            new_w = int(pixmap.width() * image_scale)
            new_h = int(pixmap.height() * image_scale)
            pixmap = pixmap.scaled(
                new_w, new_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        pixmap.save(str(Path(filename).resolve()))

    # ------------------------------------------------------------------
    # Molecule loading (internal)
    # ------------------------------------------------------------------

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
                    symm2 = getattr(at, "symm_matrix", None)
                    if symm2 is not None:
                        symm2 = np.array(symm2, dtype=float)
                    a3d.u_cart = self._uij_to_cart(uvals, symm2)
                    a3d.u_iso = float(np.trace(a3d.u_cart) / 3.0)
                    evals, evecs = np.linalg.eigh(a3d.u_cart)
                    if np.any(evals <= 0):
                        a3d.adp_valid = False
                    else:
                        a3d.adp_valid = True
                        a3d.u_eigvals = evals
                        a3d.u_eigvecs = evecs
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
            self._compute_molecule_bounds()

        self._build_geometry()
        self._request_update()

    # ------------------------------------------------------------------
    # Geometry building
    # ------------------------------------------------------------------

    def _build_geometry(self) -> None:
        self._build_sphere_geometry()
        self._build_ellipsoid_geometry_batched()
        self._build_cylinder_geometry()
        self._geometry_dirty = True

    def _build_sphere_geometry(self) -> None:
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

        verts = np.zeros((n * 4, 10), dtype=np.float32)
        idx = np.zeros(n * 6, dtype=np.uint32)

        for i, atom in enumerate(sphere_atoms):
            c = atom.center
            is_selected = atom.label in self.selected_atoms
            col = _SEL_COLOR if is_selected else atom.color_f
            sel_flag = 1.0 if is_selected else 0.0
            r = atom.u_iso or atom.display_radius
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
        n_seg = 20
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
            bond_color = _SEL_COLOR if is_selected else self._bond_rgb

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

        _corners = np.array(
            [[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]],
            dtype=np.float32,
        )
        corners_tiled = np.tile(_corners, (n, 1))
        centers = np.repeat(
            np.array([a.center for a in atoms], dtype=np.float32), 4, axis=0
        )
        colors = np.repeat(
            np.array(
                [_SEL_COLOR if a.label in self.selected_atoms else a.color_f
                 for a in atoms],
                dtype=np.float32,
            ), 4, axis=0,
        )
        radii = np.repeat(
            np.array([a.adp_billboard_r for a in atoms], dtype=np.float32), 4
        )
        sel_flags = np.repeat(
            np.array(
                [1.0 if a.label in self.selected_atoms else 0.0 for a in atoms],
                dtype=np.float32,
            ), 4,
        )
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

        verts = np.hstack([
            corners_tiled, centers, colors, radii[:, None], sel_flags[:, None],
            A_col0, A_col1, A_col2, evec0, evec1, evec2,
        ])

        quad_tpl = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)
        offsets = np.arange(n, dtype=np.uint32) * 4
        idx = (quad_tpl[None, :] + offsets[:, None]).ravel()

        self._ellipsoid_verts = verts.astype(np.float32).ravel()
        self._ellipsoid_idx = idx
        self._ellipsoid_count = int(len(idx))

    # ------------------------------------------------------------------
    # Molecule / ADP helpers
    # ------------------------------------------------------------------

    def _compute_molecule_bounds(self) -> None:
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
        from fastmolwidget.tools import build_conntable

        coords = np.array([a.center for a in self.atoms], dtype=np.float64)
        types = [a.type_ for a in self.atoms]
        parts = [a.part for a in self.atoms]
        symmgen = [a.symmgen for a in self.atoms]
        return build_conntable(coords, types, parts,
                               extra_param=extra_param, symmgen=symmgen)

    def calc_amatrix(self) -> None:
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
                    c * (cos(radians(alpha)) - cos(radians(beta))
                         * cos(radians(gamma))) / sin(radians(gamma)),
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
        U11, U22, U33, U23, U13, U12 = uvals
        Uij = np.array(
            [[U11, U12, U13], [U12, U22, U23], [U13, U23, U33]], dtype=float
        )
        if symm_matrix is not None:
            Uij = symm_matrix.T @ Uij @ symm_matrix
        N = np.diag([self._astar, self._bstar, self._cstar])
        return self._amatrix @ N @ Uij @ N.T @ self._amatrix.T

    # ------------------------------------------------------------------
    # Matrix helpers
    # ------------------------------------------------------------------

    def _compute_mv_matrix(self) -> np.ndarray:
        dist = max(self._molecule_radius * 3.0, 3.0)

        T_centre = np.eye(4, dtype=np.float32)
        T_centre[0, 3] = -self._molecule_center[0]
        T_centre[1, 3] = -self._molecule_center[1]
        T_centre[2, 3] = -self._molecule_center[2]

        R = np.eye(4, dtype=np.float32)
        R[:3, :3] = self._rot_matrix

        T_pan = np.eye(4, dtype=np.float32)
        T_pan[0, 3] = self._pan[0]
        T_pan[1, 3] = self._pan[1]

        T_cam = np.eye(4, dtype=np.float32)
        T_cam[2, 3] = -dist

        return (T_cam @ T_pan @ R @ T_centre).astype(np.float32)

    def _ortho_half_extents(self) -> tuple[float, float]:
        w = max(1.0, self.width())
        h = max(1.0, self.height())
        aspect = w / h
        half_h = max(
            self._molecule_radius * self._ORTHO_VIEW_MARGIN
            / max(self._zoom * 2, 0.01),
            0.5,
        )
        half_w = half_h * aspect
        return half_w, half_h

    def _compute_proj_matrix(self) -> np.ndarray:
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
    # Label positions
    # ------------------------------------------------------------------

    def _request_update(self) -> None:
        """Recompute label positions and schedule a Qt Quick repaint."""
        self._recompute_label_positions()
        self.update()

    def _recompute_label_positions(self) -> None:
        """Update :attr:`_label_positions` from current view state."""
        w = self.width()
        h = self.height()
        if w <= 0 or h <= 0 or not self.atoms:
            if self._label_positions:
                self._label_positions = []
                self.labelPositionsChanged.emit()
            return

        mv = self._compute_mv_matrix()
        proj = self._compute_proj_matrix()
        positions: list[dict] = []

        def project(atom: _Atom3D) -> tuple[float, float] | None:
            pos4 = np.array([*atom.center, 1.0], dtype=np.float32)
            eye = mv @ pos4
            clip = proj @ eye
            if abs(clip[3]) < 1e-8:
                return None
            ndc = clip[:3] / clip[3]
            if not (-1.0 <= ndc[0] <= 1.0
                    and -1.0 <= ndc[1] <= 1.0
                    and -1.0 <= ndc[2] <= 1.0):
                return None
            return (
                float((ndc[0] + 1.0) * 0.5 * w),
                float((1.0 - ndc[1]) * 0.5 * h),
            )

        hover_label = self._hover_atom_label

        if self.labels:
            for atom in self.atoms:
                if not self.show_hydrogens_flag and atom.type_ in ("H", "D"):
                    continue
                if atom.label == hover_label:
                    continue  # drawn separately as hover_atom
                pt = project(atom)
                if pt is not None:
                    positions.append(
                        {"kind": "atom", "text": atom.label,
                         "x": pt[0], "y": pt[1]}
                    )

        # Hover atom label (shown even when persistent labels are off).
        if hover_label is not None:
            for atom in self.atoms:
                if atom.label == hover_label:
                    if not self.show_hydrogens_flag and atom.type_ in ("H", "D"):
                        break
                    pt = project(atom)
                    if pt is not None:
                        positions.append(
                            {"kind": "hover_atom", "text": atom.label,
                             "x": pt[0], "y": pt[1]}
                        )
                    break

        # Hover bond distance label.
        if (
            hover_label is None
            and self._hover_bond is not None
            and self._hover_bond_distance is not None
            and self._hover_cursor is not None
        ):
            cx = float(self._hover_cursor.x()) + 14.0
            cy = float(self._hover_cursor.y()) + 14.0
            positions.append(
                {
                    "kind": "hover_bond",
                    "text": f"{self._hover_bond_distance:.3f} Å",
                    "x": cx,
                    "y": cy,
                }
            )

        self._label_positions = positions
        self.labelPositionsChanged.emit()

    # ------------------------------------------------------------------
    # QQuickItem geometry change
    # ------------------------------------------------------------------

    def geometryChange(  # type: ignore[override]
        self,
        newGeometry: QtCore.QRectF,
        oldGeometry: QtCore.QRectF,
    ) -> None:
        """Recompute labels when the item is resized."""
        super().geometryChange(newGeometry, oldGeometry)
        if newGeometry.size() != oldGeometry.size():
            self._recompute_label_positions()
            self.update()

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self._lastPos = event.position()
        self._pressPos = event.position()
        self._mouse_moved = False
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._lastPos is None:
            event.accept()
            return

        pos = event.position()
        dx = float(pos.x() - self._lastPos.x())
        dy = float(pos.y() - self._lastPos.y())
        self._mouse_moved = True

        # Clear hover during drag.
        if self._hover_atom_label is not None:
            self._hover_atom_label = None
        if self._hover_bond is not None:
            self._hover_bond = None
            self._hover_bond_distance = None
            self._hover_cursor = None

        if event.buttons() == Qt.MouseButton.LeftButton:
            angle_y = dx / 80.0
            angle_x = dy / 80.0
            Ry = np.array(
                [[cos(angle_y), 0.0, sin(angle_y)],
                 [0.0, 1.0, 0.0],
                 [-sin(angle_y), 0.0, cos(angle_y)]],
                dtype=np.float32,
            )
            Rx = np.array(
                [[1.0, 0.0, 0.0],
                 [0.0, cos(angle_x), -sin(angle_x)],
                 [0.0, sin(angle_x), cos(angle_x)]],
                dtype=np.float32,
            )
            R = Rx @ Ry
            self._rot_matrix = R @ self._rot_matrix
            self.cumulative_R = R @ self.cumulative_R

        elif event.buttons() == Qt.MouseButton.RightButton:
            self._zoom += dy / 100.0
            self._zoom = max(0.01, self._zoom)

        elif event.buttons() == Qt.MouseButton.MiddleButton:
            pan_scale = self._molecule_radius * 0.005
            self._pan[0] += dx * pan_scale
            self._pan[1] -= dy * pan_scale

        self._lastPos = pos
        self._request_update()
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not self._mouse_moved
            and self._pressPos is not None
        ):
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

        self._lastPos = None
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # type: ignore[override]
        """Scroll wheel adjusts label font size."""
        delta = event.angleDelta().y()
        if delta > 0:
            self.setLabelFont(self.fontsize + 2)
        elif delta < 0:
            self.setLabelFont(self.fontsize - 2)
        event.accept()

    def hoverMoveEvent(self, event: object) -> None:  # type: ignore[override]
        """Update hover state for atom / bond under cursor."""
        try:
            pos = event.position()  # type: ignore[union-attr]
        except AttributeError:
            return
        self._update_hover(pos)

    def hoverLeaveEvent(self, event: object) -> None:  # type: ignore[override]
        """Clear hover state when cursor leaves the item."""
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
            self._request_update()

    def _update_hover(self, pos: QtCore.QPointF) -> None:
        if not self.atoms:
            new_atom: str | None = None
            new_bond: tuple[str, str] | None = None
            new_dist: float | None = None
        else:
            sx, sy = float(pos.x()), float(pos.y())
            mv = self._compute_mv_matrix()
            atom, _ = self._pick_atom_at(sx, sy, mv=mv)
            new_atom = atom.label if atom is not None else None
            new_bond = None
            new_dist = None
            if atom is None:
                proj = self._compute_proj_matrix()
                best_t = float("inf")
                best_pair: tuple[_Atom3D, _Atom3D] | None = None
                for n1, n2 in self.connections:
                    at1, at2 = self.atoms[n1], self.atoms[n2]
                    if not self.show_hydrogens_flag and (
                        at1.type_ in ("H", "D") or at2.type_ in ("H", "D")
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
            self._request_update()

    def _handle_click(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position()
        if self._is_click_drag(pos):
            return

        mv = self._compute_mv_matrix()
        proj = self._compute_proj_matrix()
        sx, sy = float(pos.x()), float(pos.y())

        best_atom, best_t = self._pick_atom_at(sx, sy, mv=mv)
        best_bond: tuple[str, str] | None = None

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
            self._request_update()

    def _handle_middle_click(self, event: QtGui.QMouseEvent) -> None:
        pos = event.position()
        if self._is_click_drag(pos):
            return
        atom, _ = self._pick_atom_at(float(pos.x()), float(pos.y()))
        if atom is None:
            return
        self._molecule_center = atom.center.astype(np.float32).copy()
        self._pan = np.zeros(2, dtype=np.float32)
        self._request_update()

    def _is_click_drag(self, pos: QtCore.QPointF) -> bool:
        if self._pressPos is None:
            return False
        dx = pos.x() - self._pressPos.x()
        dy = pos.y() - self._pressPos.y()
        return dx * dx + dy * dy > 25

    # ------------------------------------------------------------------
    # Hit testing
    # ------------------------------------------------------------------

    def _pick_atom_at(
        self,
        sx: float,
        sy: float,
        *,
        mv: np.ndarray | None = None,
    ) -> tuple[_Atom3D | None, float]:
        if mv is None:
            mv = self._compute_mv_matrix()
        ray_origin, ray_dir = self._screen_to_ray_viewspace(sx, sy)

        best_atom: _Atom3D | None = None
        best_t = float("inf")

        for atom in self.atoms:
            if not self.show_hydrogens_flag and atom.type_ in ("H", "D"):
                continue
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
                radius = float(atom.u_iso or atom.display_radius)
                t = self._ray_sphere_hit_viewspace(
                    ray_origin, ray_dir, atom.center, radius, mv
                )
            if t is not None and t < best_t:
                best_t = t
                best_atom = atom
        return best_atom, best_t

    def _screen_to_ray_viewspace(
        self, sx: float, sy: float
    ) -> tuple[np.ndarray, np.ndarray]:
        w = max(1.0, self.width())
        h = max(1.0, self.height())
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
        w = max(1.0, self.width())
        h = max(1.0, self.height())

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
            return screen, float(eye[2])

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
            dist = float(np.linalg.norm(p - sp1))
            if dist <= _BOND_HIT_TOLERANCE_PX:
                return float(-(z1 + 0.5 * (z2 - z1)))
            return None

        t = float(max(0.0, min(1.0, np.dot(p - sp1, ab) / ab_len2)))
        proj_pt = sp1 + t * ab
        dist = float(np.linalg.norm(p - proj_pt))

        if dist <= _BOND_HIT_TOLERANCE_PX:
            z_closest = z1 + t * (z2 - z1)
            return float(-z_closest)
        return None
