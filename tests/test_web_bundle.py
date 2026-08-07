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
NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='Node.js not available')


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
