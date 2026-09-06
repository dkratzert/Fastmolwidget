/**
 * Multi-select disorder-part filter, mirroring the Qt part-filter widget.
 *
 * All parts start checked. `null` means "show all". The control hides itself
 * unless the structure has more than one part.
 */

/**
 * @param {import('./molecule2d.js').MoleculeWidget2D} widget Renderer to
 *   filter. Must dispatch `partsChanged` and expose `setVisibleParts(Set|null)`.
 * @param {object} [options]
 * @param {string} [options.label='Show Parts:'] Text shown before the button.
 * @returns {HTMLElement} Container element, hidden until >1 part is present.
 */
export function createPartFilter(widget, { label = 'Show Parts:' } = {}) {
  const container = document.createElement('span');
  container.className = 'part-filter';
  container.style.cssText = 'display:none; align-items:center; gap:4px; position:relative; font-size:13px;';

  const labelEl = document.createElement('span');
  labelEl.textContent = label;

  const button = document.createElement('button');
  button.type = 'button';
  button.textContent = 'All';
  button.style.cssText = 'min-width:72px; text-align:left; cursor:pointer;';

  const popup = document.createElement('div');
  popup.style.cssText = 'display:none; position:absolute; top:100%; left:0; z-index:1000; '
    + 'background:#fff; border:1px solid #ccc; box-shadow:0 2px 6px rgba(0,0,0,0.2); '
    + 'padding:4px; white-space:nowrap;';

  container.append(labelEl, button, popup);

  let parts = []; // sorted part numbers currently offered
  const checked = new Set(); // subset of `parts` that is ticked

  const summaryText = () => {
    if (parts.length === 0 || checked.size === parts.length) return 'All';
    if (checked.size === 0) return 'None';
    return [...checked].sort((a, b) => a - b).join(', ');
  };

  const apply = () => {
    button.textContent = summaryText();
    // Match Qt: all checked => show everything (`null`).
    widget.setVisibleParts(checked.size === parts.length ? null : new Set(checked));
  };

  const rebuild = (available) => {
    parts = [...available].sort((a, b) => a - b);
    checked.clear();
    for (const p of parts) checked.add(p);
    popup.replaceChildren();
    for (const p of parts) {
      const row = document.createElement('label');
      row.style.cssText = 'display:flex; align-items:center; gap:6px; padding:2px 6px; cursor:pointer;';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.checked = true;
      cb.addEventListener('change', () => {
        if (cb.checked) checked.add(p); else checked.delete(p);
        apply();
      });
      const text = document.createElement('span');
      text.textContent = `Part ${p}`;
      row.append(cb, text);
      popup.append(row);
    }
    button.textContent = summaryText();
    // Hide unless disorder is present.
    container.style.display = parts.length > 1 ? 'inline-flex' : 'none';
    if (parts.length <= 1) popup.style.display = 'none';
  };

  const closePopup = () => { popup.style.display = 'none'; };

  button.addEventListener('click', (e) => {
    e.stopPropagation();
    popup.style.display = popup.style.display === 'none' ? 'block' : 'none';
  });
  // Keep the popup open while toggling checkboxes.
  popup.addEventListener('click', (e) => e.stopPropagation());
  // Close when clicking anywhere else.
  document.addEventListener('click', closePopup);

  widget.addEventListener('partsChanged', (e) => rebuild(e.detail));

  return container;
}
