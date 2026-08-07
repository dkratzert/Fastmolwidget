"""Tests for the JavaScript bundle shipped with the package.

The bundle is what report generators paste into a ``<script>`` element, so the
important properties are: it contains every module, no ES-module syntax
survives the transform, every ``__fmw_require`` target exists, and it is safe to
embed in HTML.  When a Node.js interpreter is available the bundle is also
executed for real.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from fastmolwidget.web import assets, bundle_js
from fastmolwidget.web.bundle import UnsupportedJsSyntaxError, _transform

CIF = Path('tests/test-data/p21c.cif')
GROWABLE_CIF = Path('tests/test-data/p31c.cif')
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='Node.js not available')

# Stubs the bundle needs to run headless in Node: a canvas that never draws
# and a no-op requestAnimationFrame (so `update()` never reaches `render()`).
JS_HEADLESS_PRELUDE = """
globalThis.requestAnimationFrame = () => 0;
const canvas = { width: 300, height: 150, getContext: () => ({}),
                 addEventListener: () => {}, style: {} };
const F = (require('./bundle.js'), globalThis.Fastmolwidget);
const d = require('./structure.json');
const viewer = new F.MoleculeViewer2D(canvas, {attachEvents: false, devicePixelRatio: 1});
const w = viewer.widget;
const screenExtent = () => {
  const s = w.zoom * 130;
  const cx = w.cssWidth / 2 - w.moleculeCenter[0] * s;
  const cy = w.cssHeight / 2 - w.moleculeCenter[1] * s;
  const xs = w.atoms.map((a) => a.coordinate[0] * s + cx);
  const ys = w.atoms.map((a) => a.coordinate[1] * s + cy);
  return [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
};
"""


@pytest.fixture(scope='module')
def node() -> str:
    assert NODE is not None
    return NODE


@pytest.fixture(scope='module')
def bundle() -> str:
    return bundle_js()


def test_js_assets_are_shipped():
    names = assets.js_module_names()
    for expected in ('index.js', 'embed.js', 'viewer.js', 'molecule2d.js', 'sdm.js'):
        assert expected in names
    assert assets.js_directory().is_dir()


def test_bundle_has_no_module_syntax_left(bundle: str):
    offenders = [
        line for line in bundle.splitlines() if re.match(r'^[ \t]*(import|export)\b', line)
    ]
    assert offenders == []


def test_bundle_contains_every_module(bundle: str):
    for name in assets.js_module_names():
        assert f"__fmw_modules['{name}']" in bundle


def test_every_require_target_is_registered(bundle: str):
    required = set(re.findall(r"__fmw_require\('([^']+)'\)", bundle))
    registered = set(re.findall(r"__fmw_modules\['([^']+)'\]", bundle))
    assert required - {'./' + n for n in registered} - registered == set()


def test_bundle_is_script_tag_safe(bundle: str):
    assert '</script' not in bundle
    assert '<!--' not in bundle


def test_bundle_defines_the_global(bundle: str):
    assert 'root.Fastmolwidget' in bundle
    assert "typeof window !== 'undefined' ? window : globalThis" in bundle


def test_bundle_is_cached():
    assert bundle_js() is bundle_js()


def test_unsupported_syntax_raises():
    with pytest.raises(UnsupportedJsSyntaxError):
        _transform('bad.js', 'export default function () {}\n')
    with pytest.raises(UnsupportedJsSyntaxError):
        _transform('bad.js', "import x from './y.js';\n")
    with pytest.raises(UnsupportedJsSyntaxError):
        _transform('bad.js', "import { a } from 'lodash';\n")


def test_supported_syntax_is_rewritten():
    body, deps = _transform(
        'a.js',
        "import { a, b as c } from './x.js';\nexport const q = 1;\nexport { q };\n",
    )
    assert deps == ['x.js']
    assert "const { a, b: c } = __fmw_require('x.js');" in body
    assert 'const q = 1;' in body
    assert '__fmw_exports.q = q;' in body


@needs_node
def test_bundle_is_valid_javascript(bundle: str, tmp_path: Path, node: str):
    path = tmp_path / 'bundle.js'
    path.write_text(bundle, encoding='utf-8')
    subprocess.run([node, '--check', str(path)], check=True, capture_output=True)


@needs_node
def test_bundle_exposes_the_public_api(bundle: str, tmp_path: Path, node: str):
    path = tmp_path / 'bundle.js'
    path.write_text(bundle, encoding='utf-8')
    result = subprocess.run(
        [node, '-e', f'require({str(path)!r}); console.log(Object.keys(Fastmolwidget).join(","))'],
        check=True, capture_output=True, text=True,
    )
    exported = result.stdout.strip().split(',')
    for name in ('createViewer', 'MoleculeViewer2D', 'MoleculeWidget2D', 'SDM',
                 'createPartFilter', 'version'):
        assert name in exported


@needs_node
def test_js_sdm_matches_python_sdm(bundle: str, tmp_path: Path, node: str):
    """The bundled SDM must grow/pack exactly like the Python implementation."""
    from fastmolwidget.sdm import SDM
    from fastmolwidget.web import structure_data

    data = structure_data(CIF)
    (tmp_path / 'bundle.js').write_text(bundle, encoding='utf-8')
    (tmp_path / 'structure.json').write_text(json.dumps(data), encoding='utf-8')
    script = """
const F = (require('./bundle.js'), globalThis.Fastmolwidget);
const d = require('./structure.json');
const mk = () => new F.SDM(d.atoms, d.symmops, d.cell, !!d.centric);
const grown = mk().grow();
const packed = mk().packUnitCell(null);
console.log(JSON.stringify({
  grown: grown.length,
  packed: packed.length,
  first: [grown[0].label, grown[0].x, grown[0].y, grown[0].z],
}));
"""
    (tmp_path / 'run.js').write_text(script, encoding='utf-8')
    out = subprocess.run([node, 'run.js'], cwd=tmp_path, check=True, capture_output=True, text=True)
    js = json.loads(out.stdout)

    def fresh():
        return [[a['label'], a['type'], a['x'], a['y'], a['z'], a['part']] for a in data['atoms']]

    sdm = SDM(fresh(), data['symmops'], tuple(data['cell']), data['centric'])
    py_grown = sdm.packer(sdm, sdm.calc_sdm())
    py_packed = SDM(fresh(), data['symmops'], tuple(data['cell']), data['centric']).pack_unit_cell()

    assert js['grown'] == len(py_grown)
    assert js['packed'] == len(py_packed)
    assert js['first'][0] == py_grown[0].label
    for got, expected in zip(js['first'][1:], py_grown[0][2:5], strict=True):
        assert got == pytest.approx(expected, abs=1e-12)


def _run_js(node: str, bundle: str, tmp_path: Path, structure: dict, script: str) -> dict:
    """Run *script* (appended to :data:`JS_HEADLESS_PRELUDE`) in Node and return
    the JSON object it prints."""
    (tmp_path / 'bundle.js').write_text(bundle, encoding='utf-8')
    (tmp_path / 'structure.json').write_text(json.dumps(structure), encoding='utf-8')
    (tmp_path / 'run.js').write_text(JS_HEADLESS_PRELUDE + script, encoding='utf-8')
    out = subprocess.run(
        [node, 'run.js'], cwd=tmp_path, check=True, capture_output=True, text=True,
    )
    return json.loads(out.stdout)


@needs_node
def test_grown_structure_is_fitted_into_a_late_measured_canvas(
    bundle: str, tmp_path: Path, node: str,
):
    """Regression: a viewer created in a hidden container (0x0) and grown must
    fit the *completed* molecule once its real size arrives."""
    from fastmolwidget.web import structure_data

    script = """
viewer.loadStructure(d);
viewer.setGrow(true);
w.resize(0, 0);          // hidden container: must not poison the zoom
w.resize(902, 623);      // first real measurement -> refit
const [x0, x1, y0, y1] = screenExtent();
console.log(JSON.stringify({x0, x1, y0, y1, zoom: w.zoom, radius: w.moleculeRadius,
                            atoms: w.atoms.length}));
"""
    js = _run_js(node, bundle, tmp_path, structure_data(GROWABLE_CIF), script)

    assert js['atoms'] > 0
    assert js['x0'] >= 0 and js['x1'] <= 902
    assert js['y0'] >= 0 and js['y1'] <= 623


@needs_node
def test_js_fit_matches_qt_fit_after_growing(bundle: str, tmp_path: Path, node: str):
    """`fitToView()` must reproduce the Qt widget's
    ``reset_rotation_center()`` + ``reset_view()`` fit exactly."""
    from qtpy import QtWidgets

    from fastmolwidget.loader import MoleculeLoader
    from fastmolwidget.molecule2D import MoleculeWidget
    from fastmolwidget.web import structure_data

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None
    qt_widget = MoleculeWidget()
    qt_widget.resize(902, 623)
    loader = MoleculeLoader(qt_widget)
    loader.load_file(GROWABLE_CIF)
    loader.set_grow(True)
    qt_widget.reset_rotation_center()
    py_radius = qt_widget.molecule_radius
    py_zoom = qt_widget._AUTO_ZOOM_PADDING * min(902, 623) / 2 / py_radius / 100

    script = """
viewer.loadStructure(d);
viewer.setGrow(true);
w.resize(902, 623);
console.log(JSON.stringify({zoom: w.zoom, radius: w.moleculeRadius,
                            center: w.moleculeCenter, atoms: w.atoms.length}));
"""
    js = _run_js(node, bundle, tmp_path, structure_data(GROWABLE_CIF), script)

    assert js['atoms'] == len(qt_widget.atoms)
    # Qt stores the molecule centre as float32, hence the loose tolerance.
    assert js['radius'] == pytest.approx(py_radius, rel=1e-6)
    assert js['zoom'] == pytest.approx(py_zoom, rel=1e-6)
    for got, expected in zip(js['center'], qt_widget.molecule_center, strict=True):
        assert got == pytest.approx(expected, abs=1e-5)
