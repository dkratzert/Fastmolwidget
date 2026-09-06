/** Small colour helper matching the `QColor` subset used by the renderer. */

function hexToRgb(hex) {
  let h = hex.replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const num = parseInt(h, 16);
  return [(num >> 16) & 255, (num >> 8) & 255, num & 255];
}

function rgbToHex(r, g, b) {
  const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
  return `#${[r, g, b].map((v) => clamp(v).toString(16).padStart(2, '0')).join('')}`;
}

function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r) h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return [h, s, v];
}

function hsvToRgb(h, s, v) {
  const c = v * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = v - c;
  let r = 0, g = 0, b = 0;
  if (h < 60) [r, g, b] = [c, x, 0];
  else if (h < 120) [r, g, b] = [x, c, 0];
  else if (h < 180) [r, g, b] = [0, c, x];
  else if (h < 240) [r, g, b] = [0, x, c];
  else if (h < 300) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  return [(r + m) * 255, (g + m) * 255, (b + m) * 255];
}

/**
 * Parse a colour into `{r,g,b,a}`.
 * Accepts hex, `rgb()`/`rgba()`, arrays, or any CSS colour the canvas can parse.
 * Never throws: invalid input falls back to opaque black, like `QColor`.
 */
export function parseColor(color) {
  if (Array.isArray(color)) {
    const [r, g, b, a = 1] = color;
    return { r, g, b, a };
  }
  if (typeof color === 'string') {
    if (color.startsWith('#')) {
      const [r, g, b] = hexToRgb(color);
      return { r, g, b, a: 1 };
    }
    const m = color.match(/rgba?\(([^)]+)\)/);
    if (m) {
      const parts = m[1].split(',').map((v) => parseFloat(v));
      return { r: parts[0], g: parts[1], b: parts[2], a: parts[3] ?? 1 };
    }
    return parseColorViaCanvas(color);
  }
  return { r: 0, g: 0, b: 0, a: 1 };
}

// Lazy 1x1 canvas for CSS colour syntaxes handled by the browser parser.
// Invalid `fillStyle` assignments are ignored rather than throwing.
let _colorProbeCtx = null;
function parseColorViaCanvas(color) {
  try {
    if (!_colorProbeCtx) {
      const c = document.createElement('canvas');
      c.width = 1;
      c.height = 1;
      _colorProbeCtx = c.getContext('2d');
    }
    _colorProbeCtx.fillStyle = '#000000';
    _colorProbeCtx.fillStyle = color; // silently ignored if invalid
    _colorProbeCtx.fillRect(0, 0, 1, 1);
    const [r, g, b, a] = _colorProbeCtx.getImageData(0, 0, 1, 1).data;
    return { r, g, b, a: a / 255 };
  } catch {
    return { r: 0, g: 0, b: 0, a: 1 };
  }
}

export function toCss({ r, g, b, a = 1 }) {
  if (a >= 1) return rgbToHex(r, g, b);
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${a})`;
}

/** Qt `QColor.lighter(factor)` equivalent (HSV value scaling, factor=100 is a no-op). */
export function lighter(color, factor = 150) {
  const { r, g, b, a } = parseColor(color);
  const [h, s, v] = rgbToHsv(r, g, b);
  const newV = Math.min(1, v * (factor / 100));
  const [nr, ng, nb] = hsvToRgb(h, s, newV || (factor > 100 ? 0.05 : 0));
  return toCss({ r: nr, g: ng, b: nb, a });
}

/** Qt `QColor.darker(factor)` equivalent (factor=100 is a no-op, >100 darkens). */
export function darker(color, factor = 200) {
  const { r, g, b, a } = parseColor(color);
  const [h, s, v] = rgbToHsv(r, g, b);
  const newV = v * (100 / factor);
  const [nr, ng, nb] = hsvToRgb(h, s, newV);
  return toCss({ r: nr, g: ng, b: nb, a });
}

export function withAlpha(color, alpha) {
  const { r, g, b } = parseColor(color);
  return toCss({ r, g, b, a: alpha });
}
