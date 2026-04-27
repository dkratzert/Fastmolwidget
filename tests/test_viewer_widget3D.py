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
    assert widget.fontsize == 13
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


def test_show_round_bonds_toggle():
    widget = MoleculeWidget3D()
    widget.show_round_bonds(False)
    assert widget._round_bonds is False
    widget.show_round_bonds(True)
    assert widget._round_bonds is True


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
    assert "Orthographic projection: solve the local +Z intersection." in molecule3d._ELLIPSOID_FRAG
    assert "vec3 q0 = vec3(local_xy, 0.0)" in molecule3d._ELLIPSOID_FRAG
    assert "base_color = clamp(v_color * 1.08, 0.0, 1.0)" in molecule3d._ELLIPSOID_FRAG
    assert "vec3(0.14) * spec" in molecule3d._ELLIPSOID_FRAG


def test_bond_geometry_uses_single_uniform_color():
    widget = MoleculeWidget3D()
    widget.open_molecule([
        Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
        Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
    ])

    verts = widget._cylinder_verts.reshape(-1, 9)
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

    verts = widget._cylinder_verts.reshape(-1, 9)
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

    verts = widget._cylinder_verts.reshape(-1, 9)
    colors = verts[:, 6:9]
    unique_colors = np.unique(np.round(colors, 6), axis=0)

    assert unique_colors.shape == (1, 3)
    np.testing.assert_allclose(unique_colors[0], molecule3d._SEL_COLOR, atol=1e-6)


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


def test_screen_to_ray_viewspace_is_orthographic():
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
    assert t_bond is not None, "Near bond should be hit"
    # Bond is closer → smaller t.
    assert t_bond < t_atom, (
        f"Near bond t={t_bond:.3f} must be less than far atom t={t_atom:.3f}"
    )
