"""
GLSL shader sources for :class:`~fastmolwidget.molecule3D.MoleculeWidget3D`.

Platform selection
------------------
* **macOS** — GLSL 1.20 / OpenGL 2.1 (compatibility profile tops out here).
  Uses ``attribute``/``varying``/``gl_FragColor`` syntax.
* **Other** — GLSL 1.40 / OpenGL 3.1+.
  Uses ``in``/``out``/``fragColor`` syntax.

The six public constants exported by this module are ready-to-compile strings:

* :data:`SPHERE_VERT` / :data:`SPHERE_FRAG` — sphere impostor
* :data:`CYLINDER_VERT` / :data:`CYLINDER_FRAG` — tessellated cylinder bonds
* :data:`ELLIPSOID_BATCH_VERT` / :data:`ELLIPSOID_BATCH_FRAG` — ADP ellipsoid
  impostor (all atoms in one draw call)
"""

from __future__ import annotations

import sys
from string import Template

# ---------------------------------------------------------------------------
# Platform-specific syntax tokens
# ---------------------------------------------------------------------------

MACOS: bool = sys.platform == "darwin"

if MACOS:
    _VER   = "120"
    _INV   = "attribute"  # vertex-shader input (from VBO)
    _OUTV  = "varying"    # vertex-shader output → fragment stage
    _INF   = "varying"    # fragment-shader input
    _FDECL = ""           # no explicit output-variable declaration
    _FOUT  = "gl_FragColor"
else:
    _VER   = "140"
    _INV   = "in"
    _OUTV  = "out"
    _INF   = "in"
    _FDECL = "out vec4 fragColor;\n"
    _FOUT  = "fragColor"

_SUB: dict[str, str] = dict(
    VER=_VER, INV=_INV, OUTV=_OUTV, INF=_INF, FDECL=_FDECL, FOUT=_FOUT
)


def _t(src: str) -> str:
    """Substitute platform tokens in *src* and return the final GLSL string."""
    return Template(src).substitute(_SUB)


# ---------------------------------------------------------------------------
# Sphere impostor
# ---------------------------------------------------------------------------

SPHERE_VERT: str = _t("""\
#version $VER
// Per-vertex attributes (one quad per atom)
$INV vec3 a_center;
$INV vec3 a_color;
$INV float a_radius;
$INV vec2 a_corner;
$INV float a_selected;

uniform mat4 u_mv;
uniform mat4 u_proj;

$OUTV vec3 v_center_eye;
$OUTV vec3 v_color;
$OUTV float v_radius;
$OUTV vec2 v_corner;
$OUTV float v_selected;

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
""")

SPHERE_FRAG: str = _t("""\
#version $VER
$INF vec3  v_center_eye;
$INF vec3  v_color;
$INF float v_radius;
$INF vec2  v_corner;
$INF float v_selected;

uniform mat4 u_proj;

${FDECL}
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
    $FOUT           = vec4(clamp(color, 0.0, 1.0), 1.0);

    // Write corrected depth so atoms occlude bonds and each other properly
    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
""")

# ---------------------------------------------------------------------------
# Cylinder mesh (tessellated, 8 sides)
# ---------------------------------------------------------------------------

CYLINDER_VERT: str = _t("""\
#version $VER
$INV vec3 a_position;
$INV vec3 a_normal;
$INV vec3 a_color;
$INV float a_selected;

uniform mat4 u_mv;
uniform mat4 u_proj;
uniform mat3 u_normal_mat;   // inverse-transpose of MV upper 3x3

$OUTV vec3 v_normal_eye;
$OUTV vec3 v_pos_eye;
$OUTV vec3 v_color;
$OUTV float v_selected;

void main() {
    v_color      = a_color;
    v_selected   = a_selected;
    vec4 pos_e   = u_mv * vec4(a_position, 1.0);
    v_pos_eye    = pos_e.xyz;
    v_normal_eye = normalize(u_normal_mat * a_normal);
    gl_Position  = u_proj * pos_e;
}
""")

CYLINDER_FRAG: str = _t("""\
#version $VER
$INF vec3 v_normal_eye;
$INF vec3 v_pos_eye;
$INF vec3 v_color;
$INF float v_selected;

${FDECL}
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
    $FOUT = vec4(clamp(color, 0.0, 1.0), 1.0);
}
""")

# ---------------------------------------------------------------------------
# Batched ellipsoid impostor
# ---------------------------------------------------------------------------
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

ELLIPSOID_BATCH_VERT: str = _t("""\
#version $VER
$INV vec2  a_corner;
$INV vec3  a_center;
$INV vec3  a_color;
$INV float a_radius;
$INV float a_selected;
$INV vec3  a_A_col0;
$INV vec3  a_A_col1;
$INV vec3  a_A_col2;
$INV vec3  a_evec0;
$INV vec3  a_evec1;
$INV vec3  a_evec2;

uniform mat4 u_mv;
uniform mat4 u_proj;

$OUTV vec3  v_center_eye;
$OUTV vec3  v_color;
$OUTV float v_radius;
$OUTV vec2  v_corner;
$OUTV float v_selected;
$OUTV vec3  v_A_col0;
$OUTV vec3  v_A_col1;
$OUTV vec3  v_A_col2;
$OUTV vec3  v_evec0;
$OUTV vec3  v_evec1;
$OUTV vec3  v_evec2;

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
""")

ELLIPSOID_BATCH_FRAG: str = _t("""\
#version $VER
$INF vec3  v_center_eye;
$INF vec3  v_color;
$INF float v_radius;
$INF vec2  v_corner;
$INF float v_selected;
$INF vec3  v_A_col0;
$INF vec3  v_A_col1;
$INF vec3  v_A_col2;
$INF vec3  v_evec0;
$INF vec3  v_evec1;
$INF vec3  v_evec2;

uniform mat4 u_mv;
uniform mat4 u_proj;

${FDECL}
void main() {
    mat3 A     = mat3(v_A_col0, v_A_col1, v_A_col2);
    mat3 evecs = mat3(v_evec0,  v_evec1,  v_evec2);

    vec2 local_xy = v_corner * v_radius * 1.05;
    // Orthographic projection: solve the local +Z intersection.
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

    $FOUT = vec4(clamp(color, 0.0, 1.0), 1.0);

    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
""")

