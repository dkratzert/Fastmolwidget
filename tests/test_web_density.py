"""Tests for the residual-density map shipped to the JavaScript renderer.

The Python half — computing, masking, quantising and packing the map — runs
everywhere.  The JavaScript half is exercised with Node.js when it is
available, exactly like ``tests/test_web_bundle.py``; those tests skip
otherwise.

The important claims verified here:

* masking the grid outside the displayed envelope is **lossless** for the
  contour, so the payload can be cut down without changing the picture;
* the quantised map still contours within a fraction of a grid step of the
  full-precision one;
* the JavaScript marching cubes reproduces the C++ one edge for edge, so the
  browser cage matches the Qt renderers;
* a structure exported without density carries no payload at all.
"""

from __future__ import annotations

import base64
import gzip
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fastmolwidget.density import (
    HAS_DENSITY_CPP,
    ResidualDensityMap,
    calculate_residual_density,
)
from fastmolwidget.web import bundle_js, structure_data
from fastmolwidget.web_export import (
    _MASK_SLACK_STEPS,
    _coverage_atoms,
    _envelope_mask,
    export_cif,
    export_density,
)

DATA = Path('tests/test-data')
CIF = DATA / 'p21c.cif'
RES = DATA / 'p31c-finalcif.res'
HKL = DATA / 'p31c-finalcif.hkl'

NODE = shutil.which('node')
needs_node = pytest.mark.skipif(NODE is None, reason='Node.js not available')
needs_cpp = pytest.mark.skipif(
    not HAS_DENSITY_CPP, reason='density_cpp C++ extension not built')


@pytest.fixture(scope='module')
def payload() -> dict:
    return export_density(CIF, grid_spacing=0.3)


def decode(payload: dict) -> np.ndarray:
    """Undo the transport encoding, the way the browser does."""
    raw = base64.b64decode(payload['data'])
    if payload['encoding'] == 'gzip+base64':
        raw = gzip.decompress(raw)
    quantised = np.frombuffer(raw, dtype=np.int8).reshape(payload['size'])
    return quantised.astype(np.float32) * payload['scale']


# ---------------------------------------------------------------------------
# The exported payload
# ---------------------------------------------------------------------------

@needs_cpp
def test_payload_describes_one_unit_cell(payload):
    reference = calculate_residual_density(CIF, grid_spacing=0.3)

    assert payload['mode'] == 'grid'
    assert payload['size'] == list(reference.array.shape)
    assert payload['cell'] == pytest.approx(list(reference.cell))
    assert payload['rms'] == pytest.approx(reference.rms)
    assert payload['level'] == pytest.approx(reference.sigma_level())


@needs_cpp
def test_payload_round_trips_within_a_quantisation_step(payload):
    reference = calculate_residual_density(CIF, grid_spacing=0.3)
    values = decode(payload)
    kept = values != 0.0

    assert kept.any()
    # int8 quantisation: at most half a step anywhere that survived the mask.
    assert np.abs(values[kept] - reference.array[kept]).max() <= payload['scale']
    # ... and that step is small compared with the contour level.
    assert payload['scale'] < 0.05 * payload['level']


@needs_cpp
def test_masking_is_lossless_for_the_contour():
    """Zeroing the grid outside the envelope must not move the contour."""
    reference = calculate_residual_density(CIF, grid_spacing=0.3)
    atoms = _coverage_atoms(export_cif(CIF), 'asu')
    spacing = float((np.linalg.norm(reference.orth_matrix, axis=0)
                     / np.asarray(reference.array.shape)).max())
    mask = _envelope_mask(reference.array.shape, reference.orth_matrix, atoms,
                          1.5 + _MASK_SLACK_STEPS * spacing)
    masked = np.array(reference.array, dtype=np.float32)
    masked[~mask] = 0.0
    trimmed = ResidualDensityMap(array=masked, cell=reference.cell,
                                 d_min=reference.d_min, scale=reference.scale)

    for level in (reference.sigma_level(), -reference.sigma_level()):
        full = reference.isosurface(level, atoms=atoms, margin=1.5)
        cut = trimmed.isosurface(level, atoms=atoms, margin=1.5)
        assert np.array_equal(full[0], cut[0])
        assert np.array_equal(full[1], cut[1])


@needs_cpp
def test_wider_coverage_keeps_more_of_the_map():
    asu = decode(export_density(CIF, grid_spacing=0.3, coverage='asu'))
    cell = decode(export_density(CIF, grid_spacing=0.3, coverage='cell'))

    assert (cell != 0).sum() > (asu != 0).sum()


@needs_cpp
def test_uncompressed_payload_is_still_readable():
    plain = export_density(CIF, grid_spacing=0.3, compress=False)

    assert plain['encoding'] == 'base64'
    assert decode(plain).shape == tuple(plain['size'])


@needs_cpp
def test_explicit_level_is_honoured():
    assert export_density(CIF, grid_spacing=0.3, level=0.55)['level'] == pytest.approx(0.55)


def test_unknown_coverage_is_rejected():
    with pytest.raises(ValueError, match='coverage'):
        _coverage_atoms(export_cif(CIF), 'everything')


@needs_cpp
def test_a_shelx_model_with_a_separate_hkl_exports():
    data = export_density(RES, HKL, grid_spacing=0.3)

    assert data['size'] and data['level'] > 0
    assert decode(data).shape == tuple(data['size'])


# ---------------------------------------------------------------------------
# Density is opt-in
# ---------------------------------------------------------------------------

def test_no_density_key_by_default():
    """A page that does not ask for a map must not carry one."""
    assert 'density' not in structure_data(CIF)


@needs_cpp
def test_density_is_added_on_request():
    data = structure_data(CIF, density=True,
                          density_options={'grid_spacing': 0.3})

    assert data['density']['mode'] == 'grid'
    assert data['atoms'] == structure_data(CIF)['atoms']


@needs_cpp
def test_density_payload_can_be_passed_in(payload):
    data = structure_data(CIF, density=payload)

    assert data['density'] is payload


def test_density_true_needs_a_file():
    with pytest.raises(ValueError, match='model file'):
        structure_data(structure_data(CIF), density=True)


@needs_cpp
def test_density_grows_the_page_only_when_asked(payload):
    from fastmolwidget.web import render_html

    plain = render_html(CIF, controls=False)
    with_map = render_html(CIF, controls=False, density=payload)

    assert len(with_map) > len(plain)
    # Sanity: the map must actually dominate the difference.
    assert len(with_map) - len(plain) > 0.5 * len(payload['data'])


# ---------------------------------------------------------------------------
# The JavaScript side
# ---------------------------------------------------------------------------

_SHIM = """
globalThis.requestAnimationFrame = () => 0;
const F = (require('./bundle.js'), globalThis.Fastmolwidget);
const input = require('./input.json');
"""


@pytest.fixture(scope='module')
def node() -> str:
    assert NODE is not None
    return NODE


def run_node(node: str, tmp_path: Path, script: str, payload: dict) -> dict:
    (tmp_path / 'bundle.js').write_text(bundle_js(), encoding='utf-8')
    (tmp_path / 'input.json').write_text(json.dumps(payload), encoding='utf-8')
    (tmp_path / 'run.js').write_text(_SHIM + script, encoding='utf-8')
    result = subprocess.run(
        [node, str(tmp_path / 'run.js')],
        check=True, capture_output=True, text=True, cwd=tmp_path,
    )
    return json.loads(result.stdout)


@needs_node
@needs_cpp
@pytest.mark.parametrize('sigma', [3.0, 2.0])
def test_js_marching_cubes_matches_the_cpp_one(node, tmp_path, sigma):
    """The browser cage must match the Qt one edge for edge."""
    from fastmolwidget import density_cpp

    reference = calculate_residual_density(CIF, grid_spacing=0.3)
    atoms = _coverage_atoms(export_cif(CIF), 'asu')
    sub, origin, step = reference._region(atoms, 1.5)
    sub = np.ascontiguousarray(sub, dtype=np.float32)
    level = float(reference.sigma_level(sigma))

    expected_v, expected_e = density_cpp.marching_cubes(
        sub, level, tuple(map(float, origin)), tuple(map(float, step)))

    out = run_node(node, tmp_path, """
const r = F.marchingCubes(Float32Array.from(input.values), input.size,
                          input.level, input.origin, input.step);
console.log(JSON.stringify({v: Array.from(r.vertices), s: Array.from(r.segments)}));
""", {
        'values': sub.ravel().tolist(),
        'size': [int(n) for n in sub.shape],
        'level': level,
        'origin': [float(v) for v in origin],
        'step': [float(v) for v in step],
    })

    vertices = np.array(out['v']).reshape(-1, 3)
    segments = np.array(out['s'], dtype=np.int64).reshape(-1, 2)

    assert vertices.shape == expected_v.shape
    assert np.allclose(vertices, expected_v, atol=1e-9)
    assert ({tuple(sorted(e)) for e in segments}
            == {tuple(sorted(e)) for e in expected_e})


@needs_node
@needs_cpp
def test_js_isosurface_matches_python_end_to_end(node, tmp_path, payload):
    """Decode, region, contour and clip in the browser vs. the same in Python."""
    atoms = _coverage_atoms(export_cif(CIF), 'asu')
    values = decode(payload)
    web_map = ResidualDensityMap(
        array=values, cell=tuple(payload['cell']), d_min=1.0, scale=1.0)
    expected_v, expected_e = web_map.isosurface(
        payload['level'], atoms=atoms, margin=payload['margin'])

    out = run_node(node, tmp_path, """
(async () => {
  const map = await F.DensityMap.fromPayload(input.payload);
  const r = map.isosurface(input.level, Float64Array.from(input.atoms), input.margin);
  console.log(JSON.stringify({v: Array.from(r.vertices), s: Array.from(r.segments)}));
})();
""", {
        'payload': payload,
        'level': payload['level'],
        'margin': payload['margin'],
        'atoms': atoms.ravel().tolist(),
    })

    vertices = np.array(out['v']).reshape(-1, 3)
    segments = np.array(out['s'], dtype=np.int64).reshape(-1, 2)

    assert len(segments) == len(expected_e)

    # Vertices are renumbered by the clipping, so compare the segments by their
    # midpoints instead of by index.  The tolerance only has to absorb
    # float32/float64 rounding — it is five orders of magnitude below the grid
    # step, so it cannot hide a genuinely different contour.
    def sorted_midpoints(v, e):
        mid = (v[e[:, 0]] + v[e[:, 1]]) / 2.0
        return mid[np.lexsort(np.round(mid, 6).T)]

    assert np.allclose(sorted_midpoints(vertices, segments),
                       sorted_midpoints(expected_v, expected_e), atol=1e-5)


@needs_node
@needs_cpp
def test_js_viewer_shows_and_hides_the_density(node, tmp_path, payload):
    data = structure_data(CIF, density=payload)

    out = run_node(node, tmp_path, """
(async () => {
  const canvas = { width: 300, height: 150, getContext: () => ({}),
                   addEventListener: () => {}, style: {} };
  const viewer = new F.MoleculeViewer2D(canvas, {attachEvents: false, devicePixelRatio: 1});
  viewer.loadStructure(input);
  const before = viewer.widget._densityPos;
  const shown = await viewer.setDensityVisible(true);
  const segments = viewer.widget._densityPos.segments.length / 2;
  const level = viewer.widget.densityLevel;
  viewer.setDensityLevel(level * 1.5);
  const fewer = viewer.widget._densityPos.segments.length / 2;
  viewer.widget.clearResidualDensity();
  console.log(JSON.stringify({
    hasDensity: viewer.hasDensity, before: before, shown: shown,
    segments: segments, level: level, fewer: fewer,
    cleared: viewer.widget._densityPos === null,
  }));
})();
""", data)

    assert out['hasDensity'] is True
    assert out['before'] is None
    assert out['shown'] is True
    assert out['segments'] > 0
    assert out['level'] == pytest.approx(payload['level'])
    assert out['fewer'] < out['segments']   # a higher level contours less
    assert out['cleared'] is True


@needs_node
def test_js_viewer_without_density_is_unaffected(node, tmp_path):
    data = structure_data(CIF)

    out = run_node(node, tmp_path, """
(async () => {
  const canvas = { width: 300, height: 150, getContext: () => ({}),
                   addEventListener: () => {}, style: {} };
  const viewer = new F.MoleculeViewer2D(canvas, {attachEvents: false, devicePixelRatio: 1});
  viewer.loadStructure(input);
  const shown = await viewer.setDensityVisible(true);
  console.log(JSON.stringify({hasDensity: viewer.hasDensity, shown: shown,
                              level: viewer.densitySuggestedLevel()}));
})();
""", data)

    assert out['hasDensity'] is False
    assert out['shown'] is False
    assert out['level'] is None


@needs_node
@needs_cpp
def test_js_density_follows_the_rotation(node, tmp_path, payload):
    """The wireframe and the atoms must undergo exactly the same motion."""
    data = structure_data(CIF, density=payload)

    out = run_node(node, tmp_path, """
(async () => {
  const canvas = { width: 300, height: 150, getContext: () => ({}),
                   addEventListener: () => {}, style: {} };
  const viewer = new F.MoleculeViewer2D(canvas, {attachEvents: false, devicePixelRatio: 1});
  viewer.loadStructure(input);
  await viewer.setDensityVisible(true);
  const w = viewer.widget;
  const kept = Array.from(w._densityPos.vertices);
  w.alignBestView();
  w.moleculeCenter[0] += 3.0;          // a pan between two rotations
  w._alignToReciprocalAxis(2);
  // The stored segments must be untouched (model frame) ...
  const unchanged = JSON.stringify(Array.from(w._densityPos.vertices)) === JSON.stringify(kept);
  // ... and replaying the tracked transform on the model coordinates must
  // reproduce the displayed atom positions.
  const mapped = w._toViewFrame(w._modelCoords);
  let worst = 0;
  for (let i = 0; i < w.atoms.length; i++) {
    for (let j = 0; j < 3; j++) {
      worst = Math.max(worst, Math.abs(mapped[3 * i + j] - w.atoms[i].coordinate[j]));
    }
  }
  console.log(JSON.stringify({unchanged: unchanged, worst: worst}));
})();
""", data)

    assert out['unchanged'] is True
    assert out['worst'] < 1e-9


@needs_node
@needs_cpp
def test_js_density_reclips_when_atoms_are_hidden(node, tmp_path, payload):
    data = structure_data(CIF, density=payload)

    out = run_node(node, tmp_path, """
(async () => {
  const canvas = { width: 300, height: 150, getContext: () => ({}),
                   addEventListener: () => {}, style: {} };
  const viewer = new F.MoleculeViewer2D(canvas, {attachEvents: false, devicePixelRatio: 1});
  viewer.loadStructure(input);
  await viewer.setDensityVisible(true);
  const all = viewer.widget._densityPos.segments.length / 2;
  viewer.widget.showHydrogens(false);
  const noH = viewer.widget._densityPos.segments.length / 2;
  console.log(JSON.stringify({all: all, noH: noH}));
})();
""", data)

    assert out['all'] > 0
    assert out['noH'] < out['all']


@needs_node
@needs_cpp
def test_js_level_box_applies_while_typing(node, tmp_path, payload):
    """The number box must re-contour on `input`, not only on Enter/blur."""
    data = structure_data(CIF, density=payload)

    out = run_node(node, tmp_path, """
(async () => {
  const frames = [];
  globalThis.requestAnimationFrame = (f) => { frames.push(f); return frames.length; };
  const flush = () => { const q = frames.splice(0); q.forEach((f) => f()); };
  const stub = () => new Proxy({}, {get: (t, k) =>
      k === 'measureText' ? () => ({width: 10})
    : (k === 'createRadialGradient' || k === 'createLinearGradient')
      ? () => ({addColorStop: () => {}}) : () => {}});
  const node = () => ({style: {}, children: [], _l: {}, value: '', type: '',
    disabled: false, checked: false, textContent: '',
    append(...a) { this.children.push(...a); }, appendChild(a) { this.children.push(a); },
    addEventListener(t, f) { (this._l[t] = this._l[t] || []).push(f); },
    fire(t) { (this._l[t] || []).forEach((f) => f()); },
    getContext: stub});
  globalThis.document = {createElement: node, createTextNode: (t) => ({text: t}),
                         getElementById: () => null};

  const canvas = node();
  const viewer = new F.MoleculeViewer2D(canvas, {attachEvents: false, devicePixelRatio: 1});
  const bar = F.createControlBar(viewer, {grow: false, pack: false, adps: true,
      labels: false, hydrogens: true, bondWidth: 3, density: false,
      densityLevel: null, saveFileName: 'm.png'}, {partFilter: false});
  viewer.loadStructure(input);
  await viewer.setDensityVisible(true);

  let box = null;
  for (const c of bar.children) for (const g of (c.children || [])) {
    if (g.type === 'number') box = g;
  }
  box.disabled = false;
  const seen = [];
  const type = (value, event) => {
    box.value = value; box.fire(event); flush();
    seen.push({value: value, level: viewer.widget.densityLevel, field: box.value});
  };
  type('0.', 'input');        // half-typed: must be ignored
  type('99', 'input');        // out of range: must be ignored
  type('0.60', 'input');      // applied without Enter
  type('0.25', 'input');
  type('50', 'change');       // blur clamps
  console.log(JSON.stringify({found: box !== null, seen: seen}));
})();
""", data)

    assert out['found'] is True
    partial, out_of_range, high, low, clamped = out['seen']
    assert partial['level'] == pytest.approx(payload['level'])
    assert out_of_range['level'] == pytest.approx(payload['level'])
    assert high['level'] == pytest.approx(0.60)     # no Enter needed
    assert high['field'] == '0.60'                  # and not rewritten underneath
    assert low['level'] == pytest.approx(0.25)
    assert clamped['level'] == pytest.approx(9.99)


@needs_node
@needs_cpp
def test_js_level_change_reuses_the_extracted_block(node, tmp_path, payload):
    """Re-contouring must not redo the periodic sub-grid extraction."""
    data = structure_data(CIF, density=payload)

    out = run_node(node, tmp_path, """
(async () => {
  const canvas = { width: 300, height: 150, getContext: () => ({}),
                   addEventListener: () => {}, style: {} };
  const viewer = new F.MoleculeViewer2D(canvas, {attachEvents: false, devicePixelRatio: 1});
  viewer.loadStructure(input);
  await viewer.setDensityVisible(true);
  const map = viewer.widget.densityMap;
  const first = map._blockCache.block;
  viewer.setDensityLevel(viewer.widget.densityLevel * 1.2);
  const second = map._blockCache.block;
  // ... but a change to the visible atoms must invalidate it.
  viewer.widget.showHydrogens(false);
  const third = map._blockCache.block;
  console.log(JSON.stringify({reused: first === second, rebuilt: third !== first,
                              hasSkipTable: first.absMax.length > 0}));
})();
""", data)

    assert out['reused'] is True
    assert out['rebuilt'] is True
    assert out['hasSkipTable'] is True


@needs_node
@needs_cpp
def test_js_skip_table_does_not_change_the_contour(node, tmp_path, payload):
    """The cube-magnitude table is an optimisation, never a change in output."""
    atoms = _coverage_atoms(export_cif(CIF), 'asu')

    out = run_node(node, tmp_path, """
(async () => {
  const map = await F.DensityMap.fromPayload(input.payload);
  const atoms = Float64Array.from(input.atoms);
  const block = map._blockFor(atoms, input.margin);
  const fast = F.marchingCubes(block.values, block.size, input.level,
                               block.origin, block.step, block.absMax);
  const slow = F.marchingCubes(block.values, block.size, input.level,
                               block.origin, block.step);
  const same = fast.vertices.length === slow.vertices.length
    && fast.segments.length === slow.segments.length
    && fast.vertices.every((v, i) => v === slow.vertices[i])
    && fast.segments.every((v, i) => v === slow.segments[i]);
  console.log(JSON.stringify({identical: same, segments: fast.segments.length / 2}));
})();
""", {'payload': payload, 'atoms': atoms.ravel().tolist(),
      'level': payload['level'], 'margin': payload['margin']})

    assert out['identical'] is True
    assert out['segments'] > 0
