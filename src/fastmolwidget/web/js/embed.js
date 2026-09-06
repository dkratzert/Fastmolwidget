/**
 * DOM embedding entry point.
 *
 * `createViewer(container, structure, options)` creates a HiDPI-aware canvas,
 * keeps it sized to the container, and loads the JSON from
 * `fastmolwidget.web.structure_json()`.
 */

import { MoleculeViewer2D } from './viewer.js';
import { createPartFilter } from './part_filter.js';

const DEFAULTS = {
  controls: false,
  grow: false,
  pack: false,
  adps: true,
  labels: false,
  hydrogens: true,
  bondWidth: 3,
  bondColor: null,
  background: null,
  bestView: false,
  saveFileName: 'molecule.png',
  density: false,
  densityLevel: null,
};

/**
 * Per-element visibility for the optional control bar.
 * When `options.controls` is an object, omitted keys default to `true`.
 */
const CONTROL_ELEMENT_DEFAULTS = {
  grow: true,
  pack: true,
  adps: true,
  labels: true,
  hydrogens: true,
  partFilter: true,
  bondWidth: true,
  density: true,
  bestView: true,
  resetView: true,
  saveImage: true,
};

function el(tag, style, props = {}) {
  const node = document.createElement(tag);
  if (style) node.style.cssText = style;
  Object.assign(node, props);
  return node;
}

function checkbox(labelText, checked, onChange) {
  const label = el('label', 'display:flex; align-items:center; gap:4px; font-size:13px;');
  const input = el('input', null, { type: 'checkbox', checked: !!checked });
  input.addEventListener('change', () => onChange(input.checked));
  label.append(input, document.createTextNode(labelText));
  label.inputEl = input;
  return label;
}

function button(text, onClick) {
  const b = el('button', 'font-size:13px; cursor:pointer;', { type: 'button', textContent: text });
  b.addEventListener('click', onClick);
  return b;
}

/**
 * Build the shared control bar.
 * @param {MoleculeViewer2D} viewer
 * @param {object} opts effective options
 * @param {object} [elements] per-element visibility
 * @returns {HTMLElement}
 */
function createControlBar(viewer, opts, elements = CONTROL_ELEMENT_DEFAULTS) {
  const show = { ...CONTROL_ELEMENT_DEFAULTS, ...elements };
  const bar = el('div', 'display:flex; gap:8px; align-items:center; padding:6px 10px; '
    + 'border-bottom:1px solid #ccc; flex-wrap:wrap; font-family:sans-serif;');

  let growChk = null;
  let packChk = null;
  if (show.grow) {
    growChk = checkbox('Grow', opts.grow, (checked) => {
      if (checked && packChk) packChk.inputEl.checked = false;
      viewer.setGrow(checked);
    });
  }
  if (show.pack) {
    packChk = checkbox('Pack unit cell', opts.pack, (checked) => {
      if (checked && growChk) growChk.inputEl.checked = false;
      viewer.setPack(checked);
    });
  }

  if (growChk) bar.append(growChk);
  if (packChk) bar.append(packChk);
  if (show.adps) bar.append(checkbox('ADPs', opts.adps, (c) => viewer.widget.showAdps(c)));
  if (show.labels) bar.append(checkbox('Labels', opts.labels, (c) => viewer.widget.showLabels(c)));
  if (show.hydrogens) bar.append(checkbox('Show H', opts.hydrogens, (c) => viewer.widget.showHydrogens(c)));
  if (show.partFilter) bar.append(createPartFilter(viewer.widget));

  if (show.bondWidth) {
    const widthLabel = el('label', 'display:flex; align-items:center; gap:4px; font-size:13px;');
    const widthInput = el('input', null, {
      type: 'range', min: '1', max: '15', value: String(opts.bondWidth),
    });
    widthInput.addEventListener('input', () => viewer.widget.setBondWidth(parseInt(widthInput.value, 10)));
    widthLabel.append(document.createTextNode('Bond width'), widthInput);
    bar.append(widthLabel);
  }

  // Hide density controls until a structure with a map is loaded.
  if (show.density) {
    const levelLabel = el('label',
      'display:flex; align-items:center; gap:4px; font-size:13px; opacity:0.45;');
    const levelInput = el('input', 'width:70px;', {
      type: 'number', min: '0.01', max: '9.99', step: '0.02',
      value: String(opts.densityLevel ?? 0.3),
      disabled: true,
    });

    // Re-contour on input, not just Enter/blur. Coalesce updates per frame.
    let pendingFrame = 0;
    let applyingFromInput = false;
    const applyLevel = () => {
      pendingFrame = 0;
      const value = parseFloat(levelInput.value);
      // Ignore half-typed or out-of-range values; `change` clamps on commit.
      if (!Number.isFinite(value) || value < 0.01 || value > 9.99) return;
      // Do not rewrite the field while the user is still typing.
      applyingFromInput = true;
      try {
        viewer.setDensityLevel(value);
      } finally {
        applyingFromInput = false;
      }
    };
    const scheduleLevel = () => {
      if (levelInput.disabled || pendingFrame) return;
      pendingFrame = typeof requestAnimationFrame === 'function'
        ? requestAnimationFrame(applyLevel)
        : setTimeout(applyLevel, 0);
    };
    levelInput.addEventListener('input', scheduleLevel);
    levelInput.addEventListener('change', () => {
      const value = parseFloat(levelInput.value);
      if (!Number.isFinite(value)) {
        levelInput.value = viewer.widget.densityLevel.toFixed(2);
        return;
      }
      const clamped = Math.min(Math.max(value, 0.01), 9.99);
      if (clamped !== value) levelInput.value = clamped.toFixed(2);
      if (!levelInput.disabled) viewer.setDensityLevel(clamped);
    });
    levelLabel.append(document.createTextNode('Level'), levelInput,
                      document.createTextNode('e/\u00c5\u00b3'));

    const densityChk = checkbox('Density', false, (checked) => {
      // Decoding is async; trust the checkbox only after the viewer confirms.
      viewer.setDensityVisible(checked, parseFloat(levelInput.value)).then((shown) => {
        densityChk.inputEl.checked = shown;
        levelInput.disabled = !shown;
        levelLabel.style.opacity = shown ? '1' : '0.45';
        if (shown) levelInput.value = viewer.widget.densityLevel.toFixed(2);
      }, () => {
        densityChk.inputEl.checked = false;
        levelInput.disabled = true;
      });
    });
    viewer.widget.addEventListener('densityLevelChanged', (e) => {
      if (applyingFromInput) return;   // don't fight the field being typed in
      levelInput.value = Number(e.detail).toFixed(2);
    });

    const syncDensityControls = () => {
      const available = viewer.hasDensity;
      densityChk.style.display = available ? '' : 'none';
      levelLabel.style.display = available ? '' : 'none';
      if (!available) return;
      const suggested = viewer.densitySuggestedLevel();
      if (opts.densityLevel == null && suggested != null) {
        levelInput.value = Number(suggested).toFixed(2);
      }
      if (opts.density && !densityChk.inputEl.checked) densityChk.inputEl.click();
    };
    viewer.widget.addEventListener('structureChanged', syncDensityControls);
    syncDensityControls();

    bar.append(densityChk, levelLabel);
  }

  if (show.bestView) bar.append(button('Best view', () => viewer.widget.alignBestView()));  if (show.resetView) {
    bar.append(button('Reset view', () => {
      viewer.widget.resetRotationCenter();
      viewer.widget.resetView();
    }));
  }
  if (show.saveImage) bar.append(button('Save image', () => viewer.widget.saveImage(opts.saveFileName)));
  return bar;
}

/**
 * Create a viewer inside *container*.
 *
 * @param {HTMLElement|string} container element or element id to fill.
 * @param {object|null} [structure] fractional-coordinate JSON, or `null`.
 * @param {object} [options]
 * @param {boolean|object} [options.controls=false] show all controls, or pass
 *   an object of per-element visibility flags.
 * @param {boolean} [options.grow=false] grow the asymmetric unit.
 * @param {boolean} [options.pack=false] pack one unit cell.
 * @param {boolean} [options.adps=true] draw ADP ellipsoids.
 * @param {boolean} [options.labels=false] draw atom labels.
 * @param {boolean} [options.hydrogens=true] show hydrogen atoms.
 * @param {number} [options.bondWidth=3]
 * @param {string} [options.bondColor] CSS colour.
 * @param {string} [options.background] CSS colour.
 * @param {boolean} [options.bestView=false] align to the PCA best view after loading.
 * @param {string} [options.saveFileName='molecule.png']
 * @param {number} [options.devicePixelRatio] fixed ratio for tests/exports.
 * @returns {MoleculeViewer2D} with extra `.destroy()` and `.container` properties.
 */
export function createViewer(container, structure = null, options = {}) {
  const host = typeof container === 'string' ? document.getElementById(container) : container;
  if (!host) throw new Error('createViewer: container not found');
  const opts = { ...DEFAULTS, ...options };
  const controlsEnabled = !!options.controls;
  const controlElements = typeof options.controls === 'object' && options.controls !== null
    ? options.controls
    : {};

  host.textContent = '';
  const style = window.getComputedStyle(host);
  if (style.position === 'static') host.style.position = 'relative';
  const layout = el('div', 'display:flex; flex-direction:column; width:100%; height:100%;');
  const canvas = el('canvas', 'flex:1 1 auto; display:block; width:100%; height:100%; '
    + 'min-height:0; cursor:grab;');
  canvas.tabIndex = 0; // needed for the F1-F3 axis-alignment shortcuts

  const viewerOptions = {};
  if (options.devicePixelRatio !== undefined) viewerOptions.devicePixelRatio = options.devicePixelRatio;
  const viewer = new MoleculeViewer2D(canvas, viewerOptions);

  if (controlsEnabled) layout.append(createControlBar(viewer, opts, controlElements));
  layout.append(canvas);
  host.append(layout);

  if (opts.background !== null) viewer.widget.setBackgroundColor(opts.background);
  if (opts.bondColor !== null) viewer.widget.setBondColor(opts.bondColor);
  viewer.widget.bondWidth = opts.bondWidth;
  viewer.widget.showAdpsFlag = !!opts.adps;
  viewer.widget.labels = !!opts.labels;
  viewer.widget.showHydrogensFlag = !!opts.hydrogens;

  const fit = () => {
    const rect = canvas.getBoundingClientRect();
    if (rect.width > 0 && rect.height > 0) viewer.widget.resize(rect.width, rect.height);
  };

  let observer = null;
  if (typeof ResizeObserver !== 'undefined') {
    observer = new ResizeObserver(fit);
    observer.observe(host);
  } else {
    window.addEventListener('resize', fit);
  }

  viewer.container = host;
  viewer.canvas = canvas;
  viewer.fit = fit;
  viewer.destroy = () => {
    if (observer) observer.disconnect();
    else window.removeEventListener('resize', fit);
    host.textContent = '';
  };

  if (structure) {
    // Measure first; otherwise auto-zoom uses the placeholder 300x150 canvas size.
    fit();
    viewer.loadStructure(structure);
    if (opts.pack) viewer.setPack(true);
    else if (opts.grow) viewer.setGrow(true);
    fit();
    if (opts.bestView) viewer.widget.alignBestView();
    viewer.widget.fitToView();
    // Without controls, nothing else turns density on.
    if (opts.density && !controlsEnabled) {
      viewer.setDensityVisible(true, opts.densityLevel ?? undefined);
    }
  } else {
    fit();
  }
  return viewer;
}

export { createControlBar, CONTROL_ELEMENT_DEFAULTS };
