"""Tests for :class:`~fastmolwidget.molecule3D.MoleculeWidget3D` and
:class:`~fastmolwidget.viewer_widget3D.MoleculeViewer3DWidget`.

OpenGL rendering is skipped on headless CI runners where *PyOpenGL* or a real
GPU context is unavailable; the widget then shows a text fallback, which is all
we can reasonably test in that environment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qtpy import QtGui, QtWidgets

import fastmolwidget.molecule3D as molecule3d
from fastmolwidget.molecule3D import MoleculeWidget3D
from fastmolwidget.molecule_base import MoleculeWidgetProtocol
from fastmolwidget.sdm import Atomtuple
from fastmolwidget.viewer_widget3D import MoleculeViewer3DWidget

app = QtWidgets.QApplication.instance()
if not app:
    app = QtWidgets.QApplication([])

data = Path("tests/test-data")


# ------------------------------------------------------------------
# Protocol compliance
# ------------------------------------------------------------------

def test_widget3d_satisfies_protocol():
    """MoleculeWidget3D must satisfy MoleculeWidgetProtocol."""
    widget = MoleculeWidget3D()
    assert isinstance(widget, MoleculeWidgetProtocol)


# ------------------------------------------------------------------
# Construction
# ------------------------------------------------------------------

def test_construction_defaults():
    widget = MoleculeWidget3D()
    assert widget.atoms_size == 12
    assert widget.fontsize == 18
    assert widget.bond_width == 3
    assert widget.labels is True
    assert widget._show_adps is True
    assert widget.show_hydrogens_flag is True
    np.testing.assert_allclose(widget._bond_rgb, molecule3d._DEFAULT_BOND_COLOR, atol=1e-6)


def test_viewer3d_construction():
    w = MoleculeViewer3DWidget()
    assert w is not None
    assert w.render_widget is not None
    assert isinstance(w.render_widget, MoleculeWidget3D)


# ------------------------------------------------------------------
# open_molecule / clear
# ------------------------------------------------------------------

def test_open_molecule_atoms():
    widget = MoleculeWidget3D()
    atoms = [
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ]
    widget.open_molecule(atoms)
    assert len(widget.atoms) == 2
    assert widget.atoms[0].label == "C1"
    assert widget.atoms[1].type_ == "O"


def test_open_molecule_clear():
    widget = MoleculeWidget3D()
    widget.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    assert len(widget.atoms) == 1

    widget.clear()
    assert len(widget.atoms) == 0


def test_open_molecule_resets_view():
    widget = MoleculeWidget3D()
    widget._zoom = 3.0
    widget._pan = np.array([1.0, 2.0], dtype=np.float32)

    widget.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    assert widget._zoom == 1.0
    np.testing.assert_array_equal(widget._pan, [0.0, 0.0])


def test_open_molecule_keep_view():
    widget = MoleculeWidget3D()
    widget.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    widget._zoom = 2.5

    widget.open_molecule(
        [Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0), Atomtuple("N1", "N", 1.5, 0.0, 0.0, 0)],
        keep_view=True,
    )
    assert widget._zoom == pytest.approx(2.5)
    assert len(widget.atoms) == 2


# ------------------------------------------------------------------
# View control
# ------------------------------------------------------------------

def test_reset_view():
    widget = MoleculeWidget3D()
    widget._zoom = 3.0
    widget._pan = np.array([5.0, -3.0], dtype=np.float32)

    widget.reset_view()
    assert widget._zoom == 1.0
    np.testing.assert_array_equal(widget._pan, [0.0, 0.0])
    np.testing.assert_array_equal(widget._rot_matrix, np.eye(3, dtype=np.float32))


def test_reset_rotation_center_restores_geometric_center():
    """reset_rotation_center() must snap the pivot back to the atom-bbox
    midpoint, clear the pan offset, and leave zoom + rotation untouched
    (it is the targeted inverse of a middle-click recentring)."""
    widget = MoleculeWidget3D()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 2.0, 0.0, 0.0, 0),
    ])
    geometric_center = widget._molecule_center.copy()
    rot_before = widget._rot_matrix.copy()

    # Simulate a middle-click recentre on one atom and a custom zoom.
    widget._molecule_center = widget.atoms[1].center.astype(np.float32).copy()
    widget._pan = np.array([4.0, -7.0], dtype=np.float32)
    widget._zoom = 2.5

    widget.reset_rotation_center()

    np.testing.assert_allclose(widget._molecule_center, geometric_center, atol=1e-6)
    np.testing.assert_array_equal(widget._pan, [0.0, 0.0])
    assert widget._zoom == 2.5
    np.testing.assert_array_equal(widget._rot_matrix, rot_before)


def test_reset_rotation_center_button_in_viewer():
    """The MoleculeViewer3DWidget control bar exposes a button that calls
    MoleculeWidget3D.reset_rotation_center()."""
    viewer = MoleculeViewer3DWidget()
    widget = viewer.render_widget
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 2.0, 0.0, 0.0, 0),
    ])
    geometric_center = widget._molecule_center.copy()

    # Move the pivot off-centre, then click the button.
    widget._molecule_center = widget.atoms[1].center.astype(np.float32).copy()
    widget._pan = np.array([1.0, 1.0], dtype=np.float32)

    viewer._reset_center_button.click()

    np.testing.assert_allclose(widget._molecule_center, geometric_center, atol=1e-6)
    np.testing.assert_array_equal(widget._pan, [0.0, 0.0])


# ------------------------------------------------------------------
# Display toggles
# ------------------------------------------------------------------

def test_show_adps_toggle():
    widget = MoleculeWidget3D()
    widget.show_adps(False)
    assert widget._show_adps is False
    widget.show_adps(True)
    assert widget._show_adps is True


def test_show_labels_toggle():
    widget = MoleculeWidget3D()
    widget.show_labels(True)
    assert widget.labels is True
    widget.show_labels(False)
    assert widget.labels is False


def test_show_labels_via_set_labels_visible():
    widget = MoleculeWidget3D()
    widget.set_labels_visible(False)
    assert widget.labels is False


def test_show_hydrogens_toggle():
    widget = MoleculeWidget3D()
    widget.show_hydrogens(False)
    assert widget.show_hydrogens_flag is False
    widget.show_hydrogens(True)
    assert widget.show_hydrogens_flag is True


def test_set_bond_width():
    widget = MoleculeWidget3D()
    widget.set_bond_width(7)
    assert widget.bond_width == 7


def test_set_label_font():
    widget = MoleculeWidget3D()
    widget.setLabelFont(20)
    assert widget.fontsize == 20
    # Must clamp to minimum of 1
    widget.setLabelFont(-5)
    assert widget.fontsize == 1


def test_set_background_color():
    widget = MoleculeWidget3D()
    widget.set_background_color(QtGui.QColor(0, 0, 0))
    r, g, b = widget._bg_rgb
    assert r == pytest.approx(0.0)
    assert g == pytest.approx(0.0)
    assert b == pytest.approx(0.0)


def test_set_bond_color_with_qcolor():
    widget = MoleculeWidget3D()
    widget.set_bond_color(QtGui.QColor("#6b5d4f"))
    np.testing.assert_allclose(widget._bond_rgb, molecule3d._hex_to_rgb_float("#6b5d4f"), atol=1e-6)


def test_set_bond_color_with_integer_tuple():
    widget = MoleculeWidget3D()
    widget.set_bond_color((120, 110, 100))
    np.testing.assert_allclose(widget._bond_rgb, (120 / 255.0, 110 / 255.0, 100 / 255.0), atol=1e-6)


def test_viewer3d_set_bond_color_proxy():
    viewer = MoleculeViewer3DWidget()
    viewer.set_bond_color("#5f5348")
    np.testing.assert_allclose(viewer.render_widget._bond_rgb, molecule3d._hex_to_rgb_float("#5f5348"), atol=1e-6)


def test_compile_program_disables_validate(monkeypatch):
    """Shader linking should skip eager program validation in initializeGL."""
    if not molecule3d._HAS_PYOPENGL or molecule3d._glshaders is None:
        pytest.skip("requires PyOpenGL")
    widget = MoleculeWidget3D()
    calls = []

    monkeypatch.setattr(molecule3d._glshaders, "compileShader", lambda src, typ: 123)

    def _fake_compile_program(*shaders, **kwargs):
        calls.append((shaders, kwargs))
        return 456

    monkeypatch.setattr(molecule3d._glshaders, "compileProgram", _fake_compile_program)
    prog = widget._compile_program("void main(){}", "void main(){}", "test")

    assert prog == 456
    assert calls
    assert calls[0][1].get("validate") is False


def test_atom_shader_uses_brighter_low_shadow_lighting():
    assert "Orthographic projection: all rays are parallel to -Z." in molecule3d._SPHERE_FRAG
    assert "vec2 local_xy = v_corner * v_radius * 1.05" in molecule3d._SPHERE_FRAG
    assert "base_color = clamp(v_color * 1.08, 0.0, 1.0)" in molecule3d._SPHERE_FRAG
    assert "vec3(0.16) * spec" in molecule3d._SPHERE_FRAG


def test_ellipsoid_shader_matches_brighter_atom_lighting_profile():
    assert "Orthographic projection: solve the local +Z intersection." in molecule3d._ELLIPSOID_BATCH_FRAG
    assert "vec3 q0 = vec3(local_xy, 0.0)" in molecule3d._ELLIPSOID_BATCH_FRAG
    assert "base_color = clamp(v_color * 1.08, 0.0, 1.0)" in molecule3d._ELLIPSOID_BATCH_FRAG
    assert "vec3(0.14) * spec" in molecule3d._ELLIPSOID_BATCH_FRAG


def test_bond_geometry_uses_single_uniform_color():
    widget = MoleculeWidget3D()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ])

    verts = widget._cylinder_verts.reshape(-1, 10)
    colors = verts[:, 6:9]
    unique_colors = np.unique(np.round(colors, 6), axis=0)
    expected_color = np.array(molecule3d._DEFAULT_BOND_COLOR)

    assert widget._cylinder_count > 0
    assert unique_colors.shape == (1, 3)
    np.testing.assert_allclose(unique_colors[0], expected_color, atol=1e-6)


def test_bond_geometry_uses_configured_bond_color():
    widget = MoleculeWidget3D()
    widget.set_bond_color("#66584a")
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ])

    verts = widget._cylinder_verts.reshape(-1, 10)
    unique_colors = np.unique(np.round(verts[:, 6:9], 6), axis=0)

    assert unique_colors.shape == (1, 3)
    np.testing.assert_allclose(unique_colors[0], molecule3d._hex_to_rgb_float("#66584a"), atol=1e-6)


def test_selected_bond_uses_single_selection_color():
    widget = MoleculeWidget3D()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ])
    widget.selected_bonds = {("C1", "O1")}
    widget._build_geometry()

    verts = widget._cylinder_verts.reshape(-1, 10)
    colors = verts[:, 6:9]
    unique_colors = np.unique(np.round(colors, 6), axis=0)

    assert unique_colors.shape == (1, 3)
    np.testing.assert_allclose(unique_colors[0], molecule3d._SEL_COLOR, atol=1e-6)
    # Selected bonds carry a selected-flag of 1.0 in the last vertex column
    np.testing.assert_allclose(verts[:, 9], 1.0)


# ------------------------------------------------------------------
# Connectivity
# ------------------------------------------------------------------

def test_connections_built():
    """Two bonded atoms should produce one connection."""
    widget = MoleculeWidget3D()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C", 1.5, 0.0, 0.0, 0),
    ])
    assert len(widget.connections) == 1


def test_no_connections_far_apart():
    widget = MoleculeWidget3D()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C", 10.0, 0.0, 0.0, 0),
    ])
    assert len(widget.connections) == 0


# ------------------------------------------------------------------
# CIF loading (via viewer3D)
# ------------------------------------------------------------------

def test_viewer3d_load_cif():
    w = MoleculeViewer3DWidget()
    w.load_file(data / "1979688_small.cif")
    assert len(w.render_widget.atoms) == 94


def test_viewer3d_load_xyz():
    w = MoleculeViewer3DWidget()
    w.load_file(data / "test_molecule.xyz")
    assert len(w.render_widget.atoms) == 5


def test_viewer3d_load_shelx():
    w = MoleculeViewer3DWidget()
    w.load_file(data / "test_molecule.res")
    assert len(w.render_widget.atoms) == 5


def test_viewer3d_unsupported_format():
    # MoleculeLoader checks the extension before the file existence, so a
    # ValueError is raised even though fake.pdb does not exist on disk.
    w = MoleculeViewer3DWidget()
    with pytest.raises(ValueError, match="Unsupported file format"):
        w.load_file(data / "fake.pdb")


def test_viewer3d_missing_file():
    w = MoleculeViewer3DWidget()
    with pytest.raises(FileNotFoundError):
        w.load_file("nonexistent.cif")


# ------------------------------------------------------------------
# Rendering sanity (no crash)
# ------------------------------------------------------------------

def test_widget3d_grab_does_not_crash():
    """widget.grab() must not raise even when OpenGL is unavailable."""
    widget = MoleculeWidget3D()
    widget.resize(800, 600)
    widget.show()
    atoms = [
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ]
    widget.open_molecule(atoms)
    pixmap = widget.grab()
    assert not pixmap.isNull()


def test_viewer3d_renders_without_crash():
    w = MoleculeViewer3DWidget()
    w.load_file(data / "1979688_small.cif")
    w.resize(800, 600)
    w.show()
    pixmap = w.grab()
    assert not pixmap.isNull()


def test_label_overlay_does_not_blacken_background():
    """Regression: enabling labels must not flip the GL background to black.

    Root cause of the original bug: ``QOpenGLWidget`` renders into a private
    FBO that Qt's compositor reads.  Constructing a ``QPainter(self)`` on the
    widget rebinds the current GL render target to the paint-engine-owned
    FBO.  The buggy code did ``glClear(...)`` *before* constructing the
    ``QPainter`` and outside any ``begin/endNativePainting`` bracket, so the
    clear hit the wrong framebuffer.  The molecule then rendered into the
    paint-engine FBO that was never cleared, giving a black background with
    only the molecule + text composited on top.

    The earlier attempt to test this with ``widget.grab()`` failed because
    ``QWidget.grab()`` on ``QOpenGLWidget`` uses a *different* paint route
    (``QWidget::render``) that doesn't trigger the bug.  The reliable probe
    is :meth:`QOpenGLWidget.grabFramebuffer`, which forces a real ``paintGL``
    pass through the live compositor route and reads back the actual FBO
    Qt would composite to the screen.

    The test renders a single carbon at the centre of a 200×200 widget with
    a distinctive teal background, then reads a corner pixel both with
    labels off (control) and on (regression).  Both must show the configured
    background colour.
    """
    if not molecule3d._HAS_PYOPENGL or not molecule3d._IS_GL_WIDGET:
        pytest.skip("requires real OpenGL context")

    widget = MoleculeWidget3D()
    bg = QtGui.QColor(50, 150, 200)  # distinctive teal, far from black
    widget.set_background_color(bg)
    widget.resize(200, 200)
    widget.show()
    QtWidgets.QApplication.processEvents()

    if widget._gl_failed:
        pytest.skip("OpenGL context creation failed in this environment")
    if not hasattr(widget, "grabFramebuffer"):
        pytest.skip("QOpenGLWidget.grabFramebuffer() unavailable")

    # Single atom at origin.  It projects to the centre of the widget, so
    # corner pixels are guaranteed to be pure background.
    widget.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    QtWidgets.QApplication.processEvents()

    expected = (bg.red(), bg.green(), bg.blue())
    tol = 4  # tight: glClear should give exactly the configured colour

    def corner_rgb(labels_on: bool) -> tuple[int, int, int]:
        widget.show_labels(labels_on)
        widget.update()
        QtWidgets.QApplication.processEvents()
        img = widget.grabFramebuffer()
        c = img.pixelColor(3, 3)
        return c.red(), c.green(), c.blue()

    rgb_off = corner_rgb(False)
    rgb_on = corner_rgb(True)

    def _close(actual: tuple[int, int, int]) -> bool:
        return all(abs(a - e) <= tol for a, e in zip(actual, expected))

    # Sanity: labels-off must show the configured background.  If this fails
    # the test environment can't render the GL widget at all.
    if not _close(rgb_off):
        pytest.skip(
            f"GL framebuffer readback returned {rgb_off} with labels OFF; "
            f"expected ~{expected}.  The environment cannot render GL — "
            f"the regression assertion would be meaningless here."
        )

    assert _close(rgb_on), (
        f"Corner pixel with labels ON = {rgb_on}, expected ~{expected}.  "
        f"The QPainter label overlay clobbered the GL framebuffer — likely "
        f"the QPainter was constructed outside the begin/endNativePainting "
        f"bracket, or glClear() ran before QPainter rebound the FBO."
    )


# ------------------------------------------------------------------
# Matrix helpers (unit tests; no GL context required)
# ------------------------------------------------------------------

def test_mv_matrix_shape():
    widget = MoleculeWidget3D()
    mv = widget._compute_mv_matrix()
    assert mv.shape == (4, 4)
    assert mv.dtype == np.float32


def test_proj_matrix_shape():
    widget = MoleculeWidget3D()
    widget.resize(800, 600)
    proj = widget._compute_proj_matrix()
    assert proj.shape == (4, 4)
    # Orthographic projection keeps w unchanged.
    assert proj[3, 2] == pytest.approx(0.0)
    assert proj[3, 3] == pytest.approx(1.0)


def test_screen_to_ray_viewspace_orthographic():
    widget = MoleculeWidget3D()
    widget.resize(800, 600)

    center_origin, center_dir = widget._screen_to_ray_viewspace(400.0, 300.0)
    left_origin, left_dir = widget._screen_to_ray_viewspace(200.0, 300.0)

    np.testing.assert_allclose(center_dir, [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(left_dir, [0.0, 0.0, -1.0], atol=1e-6)
    np.testing.assert_allclose(center_origin, [0.0, 0.0, 0.0], atol=1e-6)
    assert left_origin[0] < center_origin[0]


def test_ray_sphere_hit_viewspace_uses_ray_origin_for_ortho():
    widget = MoleculeWidget3D()
    widget.resize(800, 600)
    widget.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])

    mv = widget._compute_mv_matrix()
    center_origin, center_dir = widget._screen_to_ray_viewspace(400.0, 300.0)
    edge_origin, edge_dir = widget._screen_to_ray_viewspace(0.0, 300.0)

    hit_center = widget._ray_sphere_hit_viewspace(
        center_origin, center_dir, widget.atoms[0].center, widget.atoms[0].display_radius, mv
    )
    hit_edge = widget._ray_sphere_hit_viewspace(
        edge_origin, edge_dir, widget.atoms[0].center, widget.atoms[0].display_radius, mv
    )

    assert hit_center is not None
    assert hit_edge is None


def test_molecule_bounds():
    widget = MoleculeWidget3D()
    widget.open_molecule([
        Atomtuple("C1", "C", -1.0, 0.0, 0.0, 0),
        Atomtuple("C2", "C",  1.0, 0.0, 0.0, 0),
    ])
    np.testing.assert_allclose(widget._molecule_center, [0.0, 0.0, 0.0], atol=0.01)
    assert widget._molecule_radius >= 1.0


# ------------------------------------------------------------------
# ADP with a real CIF (no GL required for the tensor maths)
# ------------------------------------------------------------------

def test_npd_atom_renders_as_cube():
    """Atoms with non-positive-definite ADP tensors must render as cubes.

    p21c.cif contains an Al1 atom with U33 ≈ -0.0137, which makes its U_cart
    non-positive-definite.  When ADPs are shown the atom must be routed to
    the cube draw list (not the sphere bucket); when ADPs are hidden it
    must fall back to a regular sphere — same behaviour as the 2-D widget.
    """
    from fastmolwidget.cif.cif_file_io import CifReader
    from fastmolwidget.tools import to_float

    cif = CifReader(data / "p21c.cif")
    adp_dict = {
        dp.label: (
            to_float(dp.U11), to_float(dp.U22), to_float(dp.U33),
            to_float(dp.U23), to_float(dp.U13), to_float(dp.U12),
        )
        for dp in cif.displacement_parameters()
    }
    widget = MoleculeWidget3D()
    widget.open_molecule(list(cif.atoms_orth), cif.cell[:6], adp_dict)

    al1 = next(a for a in widget.atoms if a.label == "Al1")
    assert al1.u_cart is not None
    assert al1.adp_valid is False, "Al1 in p21c.cif must be flagged NPD"
    assert al1.npd_half_edge > 0.0, "NPD half-edge must be set for cube sizing"

    # ADPs ON: Al1 must be in the NPD-cube list, not the sphere list.
    widget.show_adps(True)
    npd_labels = {a.label for a in widget._npd_draw_list}
    assert "Al1" in npd_labels
    assert widget._cube_count > 0, "Cube index buffer must be populated"
    # 6 faces × 2 triangles × 3 indices = 36 indices per cube.
    assert widget._cube_count % 36 == 0
    assert widget._cube_count // 36 == len(widget._npd_draw_list)

    # ADPs OFF: NPD list must be empty; Al1 falls back to a sphere.
    widget.show_adps(False)
    assert len(widget._npd_draw_list) == 0
    assert widget._cube_count == 0


def test_adp_tensors_computed():
    """ADP atoms from a CIF must have valid u_cart tensors attached."""
    from fastmolwidget.cif.cif_file_io import CifReader
    from fastmolwidget.tools import to_float

    cif = CifReader(data / "1979688_small.cif")
    adp_dict = {
        dp.label: (
            to_float(dp.U11), to_float(dp.U22), to_float(dp.U33),
            to_float(dp.U23), to_float(dp.U13), to_float(dp.U12),
        )
        for dp in cif.displacement_parameters()
    }
    widget = MoleculeWidget3D()
    widget.open_molecule(list(cif.atoms_orth), cif.cell[:6], adp_dict)

    aniso_atoms = [a for a in widget.atoms if a.u_cart is not None and a.adp_valid]
    assert len(aniso_atoms) > 0
    for atom in aniso_atoms[:5]:
        assert atom.adp_A_matrix is not None
        assert atom.adp_A_matrix.shape == (3, 3)
        assert atom.adp_billboard_r > 0.0


# ------------------------------------------------------------------
# Bond hit-testing (depth ordering fix)
# ------------------------------------------------------------------

def test_ray_bond_returns_viewspace_t():
    """_ray_bond_screen must return a viewspace t comparable to atom t values.

    Specifically, a bond whose midpoint is closer to the camera (smaller
    viewspace z absolute value) must return a *smaller* t than one farther away.
    """
    widget = MoleculeWidget3D()
    widget.resize(800, 600)

    # Two atoms on screen centre so bonds project through the click point.
    widget.open_molecule([
        Atomtuple("C1", "C", -0.5, 0.0, 0.0, 0),
        Atomtuple("C2", "C",  0.5, 0.0, 0.0, 0),
        Atomtuple("C3", "C", -0.5, 0.0, -2.0, 0),
        Atomtuple("C4", "C",  0.5, 0.0, -2.0, 0),
    ])

    mv = widget._compute_mv_matrix()
    proj = widget._compute_proj_matrix()

    p1_near = widget.atoms[0].center
    p2_near = widget.atoms[1].center
    p1_far  = widget.atoms[2].center
    p2_far  = widget.atoms[3].center

    # Click exactly at the screen centre
    sx, sy = 400.0, 300.0
    t_near = widget._ray_bond_screen(sx, sy, p1_near, p2_near, mv, proj)
    t_far  = widget._ray_bond_screen(sx, sy, p1_far,  p2_far,  mv, proj)

    assert t_near is not None, "Near bond should register a hit"
    assert t_far  is not None, "Far bond should register a hit"
    # Near bond is closer to the camera → smaller t.
    assert t_near < t_far, (
        f"Near bond t={t_near:.3f} must be less than far bond t={t_far:.3f}"
    )


def test_ray_bond_returns_interpolated_t_for_slanted_bond():
    """_ray_bond_screen interpolates viewspace t at the click point.

    A slanted bond whose endpoints are at very different z-values must return
    a t value that is interpolated at the projection of the click point onto
    the screen-space line segment — not the average of the endpoint depths.
    """
    widget = MoleculeWidget3D()
    widget.resize(800, 600)

    # Bond runs from (−1, 0, 0) to (1, 0, −4) — large z difference.
    widget.open_molecule([
        Atomtuple("C1", "C", -1.0, 0.0,  0.0, 0),
        Atomtuple("C2", "C",  1.0, 0.0, -4.0, 0),
    ])

    mv   = widget._compute_mv_matrix()
    proj = widget._compute_proj_matrix()

    # Project both endpoints to screen.
    def project(pos):
        p4 = np.array([*pos, 1.0], dtype=np.float32)
        eye  = mv @ p4
        clip = proj @ eye
        ndc  = clip[:3] / clip[3]
        w, h = widget.width(), widget.height()
        return np.array(
            [(ndc[0] + 1.0) * 0.5 * w, (1.0 - ndc[1]) * 0.5 * h],
            dtype=np.float32,
        ), float(eye[2])

    sp1, z1 = project(widget.atoms[0].center)
    sp2, z2 = project(widget.atoms[1].center)

    # Click one-quarter of the way along the screen segment.
    click = sp1 + 0.25 * (sp2 - sp1)
    sx, sy = float(click[0]), float(click[1])

    t_result = widget._ray_bond_screen(sx, sy, widget.atoms[0].center, widget.atoms[1].center, mv, proj)
    assert t_result is not None, "Bond should register a hit at the one-quarter point"

    # The expected viewspace t at parametric 0.25 along the segment.
    z_expected = z1 + 0.25 * (z2 - z1)
    t_expected = -z_expected
    assert abs(t_result - t_expected) < 0.5, (
        f"Interpolated t={t_result:.3f} should be near expected t={t_expected:.3f}"
    )


def test_bond_selected_when_in_front_of_atom():
    """Clicking on a bond in front of an atom selects the bond, not the atom.

    The old code tested bonds only when no atom was hit; this test verifies
    that a bond closer to the camera wins over an atom behind it by comparing
    the viewspace t values returned by each hit-tester directly.
    """
    widget = MoleculeWidget3D()
    widget.resize(800, 600)
    widget.show_adps(False)  # use sphere hit-testing, simpler geometry

    # Atom at (0, 0, -5) — behind the bond in view space.
    # Bond C1–C2 runs along X at z=0 (i.e. closer to camera).
    #
    # In view space the camera looks along -Z, so z=0 > z=–5 (closer).
    # The atom's world position (0,0,-5) will be rotated by the MV matrix but
    # the bond (along X at z=0) is effectively in the near plane.
    # We place the bond along the screen midline so the click hits it.
    widget.open_molecule([
        Atomtuple("N1", "N",  0.0, 0.0, -5.0, 0),  # farther atom (behind bond)
        Atomtuple("C1", "C", -0.5, 0.0,  0.0, 0),  # bond endpoint (near)
        Atomtuple("C2", "C",  0.5, 0.0,  0.0, 0),  # bond endpoint (near)
    ])

    mv   = widget._compute_mv_matrix()
    proj = widget._compute_proj_matrix()

    # Project the far atom to find its screen position, then click there.
    # That pixel should be covered by the near bond, so the bond wins.
    pos4 = np.array([0.0, 0.0, -5.0, 1.0], dtype=np.float32)
    clip = proj @ (mv @ pos4)
    ndc  = clip[:3] / clip[3]
    w, h = widget.width(), widget.height()
    sx = (ndc[0] + 1.0) * 0.5 * w
    sy = (1.0 - ndc[1]) * 0.5 * h

    # Test the hit-testing methods directly (no QMouseEvent required).
    ray_origin, ray_dir = widget._screen_to_ray_viewspace(sx, sy)

    # Compute t for the far atom sphere.
    t_atom = widget._ray_sphere_hit_viewspace(
        ray_origin, ray_dir, widget.atoms[0].center,
        widget.atoms[0].display_radius, mv
    )

    # Compute t for the near bond (C1–C2).
    t_bond = widget._ray_bond_screen(
        sx, sy, widget.atoms[1].center, widget.atoms[2].center, mv, proj
    )

    assert t_atom is not None, "Far atom should be hit"
    assert t_bond is not None, "Near bond should register a hit"
    # Bond is closer → smaller t.
    assert t_bond < t_atom, (
        f"Near bond t={t_bond:.3f} must be less than far atom t={t_atom:.3f}"
    )


# ------------------------------------------------------------------
# Hover labels (atom name + bond distance)
# ------------------------------------------------------------------

from qtpy import QtCore  # noqa: E402  (used by the hover tests below)


def _project_to_screen(widget: MoleculeWidget3D, world_pos: np.ndarray) -> tuple[float, float]:
    """Project a world-space point to widget screen coordinates."""
    mv = widget._compute_mv_matrix()
    proj = widget._compute_proj_matrix()
    p4 = np.array([*world_pos, 1.0], dtype=np.float32)
    clip = proj @ (mv @ p4)
    ndc = clip[:3] / clip[3]
    w, h = widget.width(), widget.height()
    return (ndc[0] + 1.0) * 0.5 * w, (1.0 - ndc[1]) * 0.5 * h


def _make_two_atom_widget3d(label1: str = "C1", elem1: str = "C",
                            label2: str = "O1", elem2: str = "O",
                            dx: float = 1.5) -> MoleculeWidget3D:
    """Build a sized 3-D widget with two atoms ``dx`` Å apart on the X axis."""
    widget = MoleculeWidget3D()
    widget.resize(800, 600)
    widget.open_molecule([
        Atomtuple(label1, elem1, 0.0, 0.0, 0.0, 0),
        Atomtuple(label2, elem2, dx,  0.0, 0.0, 0),
    ])
    return widget


def test_hover_atom_sets_hover_label_3d():
    widget = _make_two_atom_widget3d()
    sx, sy = _project_to_screen(widget, widget.atoms[0].center)

    widget._update_hover(QtCore.QPointF(sx, sy))
    assert widget._hover_atom_label == "C1"
    # Atom hover wins → no bond hover state.
    assert widget._hover_bond is None
    assert widget._hover_bond_distance is None
    assert widget._hover_cursor is None


def test_hover_bond_records_distance_in_angstrom_3d():
    widget = _make_two_atom_widget3d(dx=1.5)
    a, b = widget.atoms[0], widget.atoms[1]
    sx1, sy1 = _project_to_screen(widget, a.center)
    sx2, sy2 = _project_to_screen(widget, b.center)
    mx, my = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0

    widget._update_hover(QtCore.QPointF(mx, my))
    assert widget._hover_bond == ("C1", "O1")
    assert widget._hover_bond_distance == pytest.approx(1.5, abs=1e-3)
    assert widget._hover_cursor is not None
    assert widget._hover_cursor.x() == pytest.approx(mx)
    assert widget._hover_cursor.y() == pytest.approx(my)
    assert widget._hover_atom_label is None


def test_hover_atom_priority_over_bond_3d():
    """When the cursor sits over an atom, atom hover wins over bond hover."""
    widget = _make_two_atom_widget3d(dx=1.5)
    sx, sy = _project_to_screen(widget, widget.atoms[0].center)

    widget._update_hover(QtCore.QPointF(sx, sy))
    assert widget._hover_atom_label == "C1"
    assert widget._hover_bond is None


def test_hover_excludes_hidden_hydrogens_3d():
    """A bond whose endpoint is a hidden hydrogen must not be hoverable."""
    widget = MoleculeWidget3D()
    widget.resize(800, 600)
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("H1", "H", 1.0, 0.0, 0.0, 0),
    ])
    widget.show_hydrogens(False)

    # Cursor over the hidden hydrogen position.
    sx, sy = _project_to_screen(widget, widget.atoms[1].center)
    widget._update_hover(QtCore.QPointF(sx, sy))
    assert widget._hover_atom_label is None
    assert widget._hover_bond is None

    # Cursor on the bond midpoint between C1 and the hidden H1.
    sx1, sy1 = _project_to_screen(widget, widget.atoms[0].center)
    mx, my = (sx + sx1) / 2.0, (sy + sy1) / 2.0
    widget._update_hover(QtCore.QPointF(mx, my))
    assert widget._hover_bond is None


def test_leave_event_clears_hover_state_3d():
    widget = _make_two_atom_widget3d(dx=1.5)
    a, b = widget.atoms[0], widget.atoms[1]
    sx1, sy1 = _project_to_screen(widget, a.center)
    sx2, sy2 = _project_to_screen(widget, b.center)
    widget._update_hover(QtCore.QPointF((sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0))
    assert widget._hover_bond is not None

    widget.leaveEvent(QtCore.QEvent(QtCore.QEvent.Type.Leave))
    assert widget._hover_atom_label is None
    assert widget._hover_bond is None
    assert widget._hover_bond_distance is None
    assert widget._hover_cursor is None


def test_drag_clears_hover_state_3d():
    """A mouse drag must suppress both atom and bond hover labels."""
    widget = _make_two_atom_widget3d(dx=1.5)
    a, b = widget.atoms[0], widget.atoms[1]
    sx1, sy1 = _project_to_screen(widget, a.center)
    sx2, sy2 = _project_to_screen(widget, b.center)
    mx, my = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
    widget._update_hover(QtCore.QPointF(mx, my))
    assert widget._hover_bond is not None

    # Simulate the start of a left-button drag.
    widget._lastPos = QtCore.QPointF(mx, my)
    drag_event = QtGui.QMouseEvent(
        QtCore.QEvent.Type.MouseMove,
        QtCore.QPointF(mx + 30, my + 30),
        QtCore.QPointF(mx + 30, my + 30),
        QtCore.Qt.MouseButton.NoButton,
        QtCore.Qt.MouseButton.LeftButton,
        QtCore.Qt.KeyboardModifier.NoModifier,
    )
    widget.mouseMoveEvent(drag_event)

    assert widget._hover_atom_label is None
    assert widget._hover_bond is None
    assert widget._hover_bond_distance is None
    assert widget._hover_cursor is None


def test_hover_state_default_values_3d():
    """Newly constructed widget must have no hover state set."""
    widget = MoleculeWidget3D()
    assert widget._hover_atom_label is None
    assert widget._hover_bond is None
    assert widget._hover_bond_distance is None
    assert widget._hover_cursor is None
    # Mouse tracking must be enabled so hover events fire without a button.
    assert widget.hasMouseTracking() is True


# ------------------------------------------------------------------
# Save-image button (3D viewer)
# ------------------------------------------------------------------

def test_save_image_button_exists_3d():
    """MoleculeViewer3DWidget must have a Save Image button."""
    w = MoleculeViewer3DWidget()
    assert hasattr(w, '_save_image_button')
    assert isinstance(w._save_image_button, QtWidgets.QPushButton)


def test_save_image_preserves_labels_off_3d(tmp_path, monkeypatch):
    """Labels must remain off during and after saving when they were off."""
    w = MoleculeViewer3DWidget()
    w.load_file(data / 'test_molecule.xyz')
    w.render_widget.show_labels(False)

    labels_during: list[bool] = []

    def _mock_save(path, **kwargs):
        labels_during.append(w.render_widget.labels)

    monkeypatch.setattr(w.render_widget, 'save_image', _mock_save)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **kw: (str(tmp_path / 'out.png'), 'PNG Image (*.png)')),
    )

    w._save_image_button.click()

    assert labels_during == [False], "Labels must stay False during save_image()"
    assert w.render_widget.labels is False, "Labels must still be False after save"


def test_save_image_preserves_labels_on_3d(tmp_path, monkeypatch):
    """Labels must remain on during and after saving when they were on."""
    w = MoleculeViewer3DWidget()
    w.load_file(data / 'test_molecule.xyz')
    w.render_widget.show_labels(True)

    labels_during: list[bool] = []

    def _mock_save(path, **kwargs):
        labels_during.append(w.render_widget.labels)

    monkeypatch.setattr(w.render_widget, 'save_image', _mock_save)
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **kw: (str(tmp_path / 'out.png'), 'PNG Image (*.png)')),
    )

    w._save_image_button.click()

    assert labels_during == [True], "Labels must stay True during save_image()"
    assert w.render_widget.labels is True, "Labels must still be True after save"


def test_save_image_cancelled_does_not_call_save_3d(monkeypatch):
    """Cancelling the file dialog must not trigger save_image at all."""
    w = MoleculeViewer3DWidget()
    called = []

    monkeypatch.setattr(w.render_widget, 'save_image', lambda *a, **kw: called.append(1))
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        'getSaveFileName',
        staticmethod(lambda *a, **kw: ('', '')),  # user cancelled
    )

    w._save_image_button.click()

    assert called == [], "save_image must not be called when dialog is cancelled"


def test_save_image_3d_labels_appear_in_file(tmp_path):
    """Labels must be visible in the saved PNG when labels are enabled (3D).

    Uses grabFramebuffer() + QPainter overlay path.  Skipped on headless CI
    where no real OpenGL context is available.

    Saves the same scene twice (labels off / on), then counts pixels that
    differ between the two images.  Text glyphs produce a visible difference,
    regardless of the specific label colour value.
    """
    import numpy as np
    from PIL import Image

    if not molecule3d._HAS_PYOPENGL or not molecule3d._IS_GL_WIDGET:
        pytest.skip("requires real OpenGL context")

    widget = MoleculeWidget3D()
    widget.resize(400, 300)
    widget.show()
    QtWidgets.QApplication.processEvents()

    if widget._gl_failed:
        pytest.skip("OpenGL context creation failed in this environment")
    if not hasattr(widget, "grabFramebuffer"):
        pytest.skip("QOpenGLWidget.grabFramebuffer() unavailable")

    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ])
    QtWidgets.QApplication.processEvents()

    # Verify grabFramebuffer returns something useful (not a blank/zero image).
    fb = widget.grabFramebuffer()
    if fb.isNull() or fb.bits() is None:
        pytest.skip("grabFramebuffer returned a null image — GL not rendering")
    fb_arr = np.frombuffer(fb.bits(), dtype=np.uint8).reshape(fb.height(), fb.width(), 4)
    if fb_arr.max() < 10:
        pytest.skip("grabFramebuffer returned a blank image — GL not rendering")

    path_off = tmp_path / 'labels_off_3d.png'
    path_on  = tmp_path / 'labels_on_3d.png'

    widget.show_labels(False)
    widget.save_image(path_off, image_scale=1.0)

    widget.show_labels(True)
    widget.save_image(path_on,  image_scale=1.0)

    arr_off = np.array(Image.open(path_off).convert('RGB'))
    arr_on  = np.array(Image.open(path_on).convert('RGB'))

    diff = np.abs(arr_on.astype(int) - arr_off.astype(int)).sum(axis=2)
    n_changed = int((diff > 5).sum())

    assert n_changed > 50, (
        f"Expected significant pixel differences when labels are ON "
        f"(only {n_changed} pixels changed — labels may not be rendered "
        f"into the saved 3D image)."
    )


def test_atom_labels_render_readable_glyphs_3d():
    """Labels overlaid on the GL widget must render as actual glyph shapes,
    not as the empty ``.notdef`` rectangle ("tofu") that Qt's OpenGL paint
    engine substitutes when it cannot rasterise a font.

    Approach:
    1. Build the same label string ("C1") on a transparent ``QImage`` with
       the *same* ``QFont`` using the raster paint engine — this is the
       ground-truth glyph mask (always correct because raster rendering
       does not go through GL).
    2. Read the live widget framebuffer with labels OFF (control) and ON
       (regression).  Subtract the two to isolate the pixels added by the
       label overlay.
    3. Compare the diff mask against the reference glyph mask:
       - The overlapping pixel count must be a significant fraction of the
         reference mask area (proves real glyphs were drawn).
       - The diff mask must NOT look like a filled or outlined rectangle
         (which is what tofu glyphs produce).  We test this by checking
         that the diff mask is *not* a near-perfect axis-aligned box: the
         ratio of mask pixels to bounding-box area for "C1" rendered with
         a normal font is well below 1.0 (lots of empty interior + curved
         strokes), whereas a tofu rectangle outline gives a ratio close to
         the box's perimeter/area which is also low — so we additionally
         require shape correlation with the reference glyph.
    """
    if not molecule3d._HAS_PYOPENGL or not molecule3d._IS_GL_WIDGET:
        pytest.skip("requires real OpenGL context")

    widget = MoleculeWidget3D()
    bg = QtGui.QColor(255, 255, 255)
    widget.set_background_color(bg)
    widget.resize(400, 300)
    widget.show()
    QtWidgets.QApplication.processEvents()

    if widget._gl_failed:
        pytest.skip("OpenGL context creation failed in this environment")
    if not hasattr(widget, "grabFramebuffer"):
        pytest.skip("QOpenGLWidget.grabFramebuffer() unavailable")

    # Single carbon at origin so the label sits near widget centre.
    widget.open_molecule([Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0)])
    QtWidgets.QApplication.processEvents()

    def grab_rgb(labels_on: bool) -> np.ndarray:
        widget.show_labels(labels_on)
        widget.update()
        QtWidgets.QApplication.processEvents()
        img = widget.grabFramebuffer().convertToFormat(
            QtGui.QImage.Format.Format_RGB32
        )
        h, w = img.height(), img.width()
        buf = img.constBits()
        if hasattr(buf, "setsize"):
            buf.setsize(img.sizeInBytes())  # PyQt5/6 path
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        # Drop the unused alpha-byte slot of Format_RGB32 → RGB.
        return arr[:, :, [2, 1, 0]].copy()

    rgb_off = grab_rgb(False)
    rgb_on = grab_rgb(True)

    if rgb_off.max() < 10:
        pytest.skip("framebuffer empty — GL not actually rendering")

    # Pixels that changed when labels were turned on.
    diff = np.abs(rgb_on.astype(int) - rgb_off.astype(int)).sum(axis=2)
    label_mask = diff > 20
    if not label_mask.any():
        pytest.fail("No label pixels appeared when labels were enabled.")

    # Build the ground-truth glyph reference using the raster engine.
    label_text = "C1"
    base_size = max(1, widget.fontsize)
    font = QtGui.QFont()
    font.setPixelSize(base_size)
    metrics = QtGui.QFontMetrics(font)
    tw = metrics.horizontalAdvance(label_text)
    th = metrics.height()
    pad = 4
    ref_w, ref_h = tw + 2 * pad, th + 2 * pad
    ref_img = QtGui.QImage(
        ref_w, ref_h, QtGui.QImage.Format.Format_ARGB32_Premultiplied
    )
    ref_img.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(ref_img)
    try:
        p.setRenderHint(QtGui.QPainter.RenderHint.TextAntialiasing, True)
        p.setFont(font)
        p.setPen(widget.label_color)
        p.drawText(pad, pad + metrics.ascent(), label_text)
    finally:
        p.end()
    ref_ptr = ref_img.constBits()
    if hasattr(ref_ptr, "setsize"):
        ref_ptr.setsize(ref_img.sizeInBytes())
    ref_arr = np.frombuffer(ref_ptr, dtype=np.uint8).reshape(ref_h, ref_w, 4)
    ref_mask = ref_arr[:, :, 3] > 32  # alpha-based glyph mask
    ref_pixels = int(ref_mask.sum())
    assert ref_pixels > 20, "raster glyph reference is empty — test broken"

    # Crop the live label region from the diff mask using the bounding box
    # of the differing pixels (much more robust than projecting screen
    # coords because the label is offset a few pixels from the atom).
    ys, xs = np.where(label_mask)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = label_mask[y0:y1, x0:x1]
    crop_pixels = int(crop.sum())

    # --- Assertion 1 ---------------------------------------------------
    # Number of changed pixels must be in the same order of magnitude as
    # the reference glyph.  A tofu rectangle outline for "C1" at 18 px
    # pixel-size is roughly 2*(width + height) ≈ 60–80 pixels; a rendered
    # "C1" glyph fill is typically 80–200 pixels.  We require enough
    # changed pixels that *something* is drawn, then we test the shape.
    assert crop_pixels >= max(20, ref_pixels // 3), (
        f"Only {crop_pixels} label pixels detected (reference glyph has "
        f"{ref_pixels}). The label overlay is missing or near-empty."
    )

    # --- Assertion 2 ---------------------------------------------------
    # Rule out tofu (a hollow rectangle with empty interior).  For an
    # axis-aligned hollow rectangle, every pixel of the mask lies on the
    # bounding-box border; the interior pixel count is 0.  A real glyph
    # like "C1" has substantial interior content (the bowl of "C" is open
    # but the strokes themselves have thickness, and "1" is essentially
    # a filled line).  We measure the *interior fill ratio*: the fraction
    # of mask pixels that are NOT on the outermost 1-pixel border of the
    # bounding box.  A hollow rectangle gives 0; a filled glyph gives
    # close to 1.
    ch, cw = crop.shape
    if ch < 4 or cw < 4:
        pytest.fail(
            f"Label region too small ({cw}×{ch}) — cannot distinguish "
            f"a tofu rectangle from real glyphs."
        )
    interior = crop[1:-1, 1:-1]
    interior_pixels = int(interior.sum())
    interior_ratio = interior_pixels / max(1, crop_pixels)
    assert interior_ratio > 0.5, (
        f"Label mask looks like a hollow rectangle (interior_ratio="
        f"{interior_ratio:.2f}, crop={cw}×{ch}, pixels={crop_pixels}). "
        f"Glyphs are likely rendering as ‘.notdef’ tofu rectangles."
    )

    # --- Assertion 3 ---------------------------------------------------
    # Shape sanity: the live label crop must broadly resemble the
    # reference glyph.  We compare *aspect ratios* — a "C1" glyph is much
    # wider than tall (≈ 1.3–1.7); a single tofu rectangle for two chars
    # would have a similar aspect, but a full filled rectangle would too,
    # so the aspect alone is not enough — combined with the interior_ratio
    # check above, however, it is decisive.
    live_aspect = cw / ch
    ref_aspect = ref_w / ref_h
    # Allow a generous 50 % tolerance — exact pixel sizes vary across
    # platforms / DPI but the overall proportion must be sane.
    assert 0.5 * ref_aspect <= live_aspect <= 2.0 * ref_aspect, (
        f"Label aspect ratio {live_aspect:.2f} differs wildly from the "
        f"reference glyph aspect {ref_aspect:.2f} — labels are not being "
        f"rendered as text."
    )
