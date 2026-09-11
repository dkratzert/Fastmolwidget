"""GLSL source strings for the 3-D renderer."""

from __future__ import annotations

import sys
from string import Template

# ---------------------------------------------------------------------------
# Platform-specific syntax tokens
# ---------------------------------------------------------------------------

MACOS: bool = sys.platform == "darwin"

if MACOS:
    _VER   = "120"
    _INV   = "attribute"  # vertex input
    _OUTV  = "varying"    # vertex output
    _INF   = "varying"    # fragment input
    _FDECL = ""           # no explicit fragment output
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
    """Substitute platform tokens into *src*."""
    return Template(src).substitute(_SUB)


# ---------------------------------------------------------------------------
# Shading constants
# ---------------------------------------------------------------------------
# The 2-D QPainter renderer shades atoms with a radial gradient running from
# ``QColor.lighter(160)`` at the highlight through the plain colour to
# ``QColor.darker(180)`` at the rim, and bonds with a linear gradient whose
# darkest stop is ``darker(280)``.  Qt scales the HSV *value* by those
# percentages, so the equivalent multipliers on the base colour are
# 1.60 / 1.00 / 0.556 and 0.357.
#
# The 3-D shaders follow the same idea but with the peak pulled back: a Lambert
# term spreads the highlight over a far larger part of the surface than Qt's
# radial gradient does, so the literal 1.60 washes light colours out.
#
# The 2-D highlight sits at 35 % / 35 % of the atom's bounding box, i.e. above
# and to the *left* of the centre, hence the negative X in the light vector.

_LIGHT = "normalize(vec3(-1.0, 1.5, 2.0))"
_LIGHT_SELECTED = "normalize(vec3(-2.0, 1.5, 2.0))"
_ATOM_RIM = "0.60"    # ~QColor.darker(170)
_ATOM_PEAK = "1.28"
_BOND_RIM = "0.45"
_BOND_PEAK = "1.22"
# Principal-axis bands: the 2-D renderer draws them as QColor(0, 0, 0, 120)
# arcs over the finished gradient, which multiplies the colour by 1 - 120/255.
_ADP_LINE_SHADE = "0.43"

_SUB.update(
    LIGHT=_LIGHT,
    LIGHT_SELECTED=_LIGHT_SELECTED,
    ATOM_RIM=_ATOM_RIM,
    ATOM_PEAK=_ATOM_PEAK,
    BOND_RIM=_BOND_RIM,
    BOND_PEAK=_BOND_PEAK,
    ADP_LINE_SHADE=_ADP_LINE_SHADE,
)


# ---------------------------------------------------------------------------
# Sphere impostor
# ---------------------------------------------------------------------------

SPHERE_VERT: str = _t("""\
#version $VER
// One quad per atom.
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

    // Expand the quad in eye-space X/Y.
    // 5% margin avoids clipping.
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

    // Bright shading matching the 2-D renderer's radial gradient.
    vec3  light = $LIGHT;
    float diff  = max(dot(normal, light), 0.0);
    float shade = $ATOM_RIM + ($ATOM_PEAK - $ATOM_RIM) * pow(diff, 1.0);
    float spec  = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 72.0);

    vec3 color = v_color * shade + vec3(0.12) * spec;
    $FOUT      = vec4(clamp(color, 0.0, 1.0), 1.0);

    // Write corrected depth.
    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
""")

# ---------------------------------------------------------------------------
# Cylinder mesh (8 sides)
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
        // Selected bonds stay flat.
        color = v_color;
    } else {
        vec3  normal = normalize(v_normal_eye);
        vec3  light  = $LIGHT;
        float diff   = max(dot(normal, light), 0.0);
        float spec   = pow(max(dot(reflect(-light, normal),
                                   normalize(-v_pos_eye)), 0.0), 32.0);

        float shade = $BOND_RIM + ($BOND_PEAK - $BOND_RIM) * pow(diff, 1.0);
        color = v_color * shade + vec3(0.18) * spec;
    }
    $FOUT = vec4(clamp(color, 0.0, 1.0), 1.0);
}
""")

# ---------------------------------------------------------------------------
# Batched ellipsoid impostor
# ---------------------------------------------------------------------------
# Per-atom data is packed into vertex attributes.
# Vertex layout (28 floats = 112 bytes):
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

    vec3  light     = v_selected > 0.5 ? $LIGHT_SELECTED : $LIGHT;
    float diff      = max(dot(normal, light), 0.0);
    float shade     = $ATOM_RIM + ($ATOM_PEAK - $ATOM_RIM) * pow(diff, 1.0);
    float spec      = pow(max(dot(reflect(-light, normal), vec3(0.0, 0.0, 1.0)), 0.0), 72.0);

    vec3 color = v_color * shade + vec3(0.12) * spec;

    float lw = v_radius * 0.04;
    if (abs(dot(hit_world, evecs[0])) < lw ||
        abs(dot(hit_world, evecs[1])) < lw ||
        abs(dot(hit_world, evecs[2])) < lw) {
        color *= $ADP_LINE_SHADE;
    }

    $FOUT = vec4(clamp(color, 0.0, 1.0), 1.0);

    vec4 clip_pos = u_proj * vec4(hit, 1.0);
    gl_FragDepth  = (clip_pos.z / clip_pos.w + 1.0) * 0.5;
}
""")



# ---------------------------------------------------------------------------
# Flat-colour line shader for residual-density wireframes
# ---------------------------------------------------------------------------

LINE_VERT: str = _t("""\
#version $VER
$INV vec3 a_position;

uniform mat4 u_mv;
uniform mat4 u_proj;

void main() {
    gl_Position = u_proj * u_mv * vec4(a_position, 1.0);
}
""")

LINE_FRAG: str = _t("""\
#version $VER
$FDECL
uniform vec3 u_color;

void main() {
    $FOUT = vec4(u_color, 1.0);
}
""")
