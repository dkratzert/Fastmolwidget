/**
 * Embedding entry point — the counterpart of `new Miew({container: ...})`.
 *
 * `createViewer(container, structure, options)` takes a plain DOM element
 * (typically an empty `<div style="height:400px">` in an HTML report), creates
 * the canvas inside it, keeps it sized to the container (HiDPI-aware), and
 * loads a structure in the fractional-coordinate JSON contract produced by
 * `fastmolwidget.web.structure_json()`.
 *
 * ```html
 * <div id="mol" style="height:400px"></div>
 * <script>
 *   var mol = { ...structure JSON... };
 *   var viewer = Fastmolwidget.createViewer(
 *       document.getElementById('mol'), mol, {controls: false, grow: true});
 * </script>
 * ```
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
 * Build the control bar shared by the demo page, the generated standalone HTML
 * and any report embedding that asks for `controls: true`.
 *
 * @param {MoleculeViewer2D} viewer
 * @param {object} opts effective (merged) options
 * @returns {HTMLElement}
 */
function createControlBar(viewer, opts) {
  const bar = el('div', 'display:flex; gap:8px; align-items:center; padding:6px 10px; '
    + 'border-bottom:1px solid #ccc; flex-wrap:wrap; font-family:sans-serif;');

  const growChk = checkbox('Grow', opts.grow, (checked) => {
    if (checked) packChk.inputEl.checked = false;
    viewer.setGrow(checked);
  });
  const packChk = checkbox('Pack unit cell', opts.pack, (checked) => {
    if (checked) growChk.inputEl.checked = false;
    viewer.setPack(checked);
  });

  bar.append(
    growChk,
    packChk,
    checkbox('ADPs', opts.adps, (c) => viewer.widget.showAdps(c)),
    checkbox('Labels', opts.labels, (c) => viewer.widget.showLabels(c)),
    checkbox('Show H', opts.hydrogens, (c) => viewer.widget.showHydrogens(c)),
    createPartFilter(viewer.widget),
  );

  const widthLabel = el('label', 'display:flex; align-items:center; gap:4px; font-size:13px;');
  const widthInput = el('input', null, {
    type: 'range', min: '1', max: '15', value: String(opts.bondWidth),
  });
  widthInput.addEventListener('input', () => viewer.widget.setBondWidth(parseInt(widthInput.value, 10)));
  widthLabel.append(document.createTextNode('Bond width'), widthInput);

  bar.append(
    widthLabel,
    button('Best view', () => viewer.widget.alignBestView()),
    button('Reset view', () => viewer.widget.resetView()),
    button('Save image', () => viewer.widget.saveImage(opts.saveFileName)),
  );
  return bar;
}

/**
 * Create a molecule viewer inside *container*.
 *
 * @param {HTMLElement|string} container element or element id to fill.
 * @param {object|null} [structure] structure in the fractional-coordinate JSON
 *   contract; may be `null` and loaded later via `viewer.loadStructure(...)`.
 * @param {object} [options]
 * @param {boolean} [options.controls=false] show the control bar.
 * @param {boolean} [options.grow=false] grow the asymmetric unit to whole molecules.
 * @param {boolean} [options.pack=false] pack one complete unit cell.
 * @param {boolean} [options.adps=true] draw ADP ellipsoids.
 * @param {boolean} [options.labels=false] draw atom labels.
 * @param {boolean} [options.hydrogens=true] show hydrogen atoms.
 * @param {number}  [options.bondWidth=3]
 * @param {string}  [options.bondColor] CSS colour.
 * @param {string}  [options.background] CSS colour.
 * @param {boolean} [options.bestView=false] align to the PCA best view after loading.
 * @param {string}  [options.saveFileName='molecule.png']
 * @param {number}  [options.devicePixelRatio] force a fixed ratio (tests/exports).
 * @returns {MoleculeViewer2D} with an extra `.destroy()` and `.container` property.
 */
export function createViewer(container, structure = null, options = {}) {
  const host = typeof container === 'string' ? document.getElementById(container) : container;
  if (!host) throw new Error('createViewer: container not found');
  const opts = { ...DEFAULTS, ...options };

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

  if (opts.controls) layout.append(createControlBar(viewer, opts));
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
    viewer.loadStructure(structure);
    if (opts.pack) viewer.setPack(true);
    else if (opts.grow) viewer.setGrow(true);
    fit();
    if (opts.bestView) viewer.widget.alignBestView();
    else viewer.widget.resetView();
  } else {
    fit();
  }
  return viewer;
}

export { createControlBar };
