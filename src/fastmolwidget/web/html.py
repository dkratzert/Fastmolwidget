"""Build HTML for the JavaScript renderer: the structure payload, a complete
self-contained page, and the helpers a report generator needs.

Two typical uses:

*Embed into an existing (e.g. Jinja2) report template* — inject the renderer
and the structure as plain strings::

    from fastmolwidget.web import bundle_js, structure_json

    template.render(fastmolwidget_js=bundle_js(),
                    structure_json=structure_json('structure.cif'))

.. code-block:: jinja

    <div id="mol" style="height:400px"></div>
    <script>
        var mol = {{ structure_json | safe }};
        {{ fastmolwidget_js | safe }}
    </script>
    <script>
        var viewer = Fastmolwidget.createViewer(
            document.getElementById('mol'), mol, {controls: false, grow: true});
    </script>

*Or generate a finished standalone page* (no network access needed, works from
``file://`` and in a ``QWebEngineView`` via ``setHtml()``)::

    from fastmolwidget.web import render_html, write_html

    write_html('structure.cif', 'structure.html')

This module imports nothing from Qt.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

from fastmolwidget.web.assets import render_template
from fastmolwidget.web.bundle import bundle_js

__all__ = ['TEMPLATE_NAME', 'render_html', 'structure_data', 'structure_json', 'write_html']

TEMPLATE_NAME = 'viewer.html'

Structure = str | Path | dict[str, Any]


def _escape_json_for_script_tag(text: str) -> str:
    """Escape ``<``, ``>`` and ``&`` in a JSON string as ``\\uXXXX``.

    The result stays valid JSON *and* a valid JavaScript literal, and can never
    close the surrounding ``<script>`` element (the same trick Django's
    ``json_script`` uses).  ``"</script"`` cannot be escaped as ``"<\\/script"``
    here, because that is a JavaScript-only escape which would break
    ``JSON.parse``/:func:`json.loads`.
    """
    return text.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def structure_data(
    structure: Structure,
    *,
    density: dict[str, Any] | bool | None = None,
    density_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the fractional-coordinate JSON contract for *structure*.

    *structure* may be a path to a ``.cif``/``.res``/``.ins`` file or an
    already-built dictionary (which is returned unchanged unless *density*
    adds to it).

    :param density: Residual (Fo−Fc) density to embed.  ``None`` (the default)
        adds nothing at all, so pages that do not want a map stay small.
        ``True`` computes one with :func:`~fastmolwidget.web_export.export_density`;
        a dict is used as the already-exported payload.
    :param density_options: Keyword arguments for
        :func:`~fastmolwidget.web_export.export_density` when *density* is
        ``True`` (e.g. ``{'coverage': 'cell', 'grid_spacing': 0.3}``).
    :raises ValueError: If ``density=True`` and *structure* is a dict, since
        the map can only be computed from the model file.
    """
    from fastmolwidget.web_export import export_cif, export_density, export_shelx

    if isinstance(structure, dict):
        data = structure
    else:
        path = Path(structure)
        suffix = path.suffix.lower()
        if suffix == '.cif':
            data = export_cif(path)
        elif suffix in ('.res', '.ins'):
            data = export_shelx(path)
        else:
            raise ValueError(f'Unsupported file type: {path.suffix}')

    if density is None or density is False:
        return data
    if density is True:
        if isinstance(structure, dict):
            raise ValueError(
                'density=True needs the model file to compute the map from; '
                'pass an already-exported payload instead.'
            )
        density = export_density(Path(structure), **(density_options or {}))
    return {**data, 'density': density}


def structure_json(structure: Structure, **kwargs: Any) -> str:
    """Return *structure* as a JSON string ready to be embedded in a
    ``<script>`` element (e.g. ``var mol = {{ structure_json | safe }};``).

    Extra keyword arguments are passed to :func:`structure_data`.
    """
    return _escape_json_for_script_tag(json.dumps(structure_data(structure, **kwargs)))


def render_html(
    structure: Structure,
    *,
    title: str | None = None,
    controls: bool | dict[str, bool] = True,
    height: str = '100%',
    background: str = '#ffffff',
    grow: bool = False,
    pack: bool = False,
    adps: bool = True,
    labels: bool = False,
    hydrogens: bool = True,
    bond_width: int = 3,
    bond_color: str | None = None,
    best_view: bool = False,
    density: dict[str, Any] | bool | None = None,
    density_options: dict[str, Any] | None = None,
    density_level: float | None = None,
) -> str:
    """Render *structure* as a complete, fully self-contained HTML document.

    The renderer and the structure are inlined, so the result needs no network
    access and can be written to disk, mailed, or handed to
    ``QWebEngineView.setHtml()``.

    :param structure: path to a ``.cif``/``.res``/``.ins`` file, or an exported
        structure dictionary.
    :param title: document title; defaults to the file name.
    :param controls: show the control bar (grow, pack, ADPs, labels, hydrogens,
        disorder-part filter, bond width, best view, reset view, save image).
        Pass ``True``/``False`` to show/hide the whole bar, or a dict to
        selectively show/hide individual elements, e.g.
        ``{'pack': False, 'bondWidth': False}`` (unspecified keys default to
        visible). Recognised keys: ``grow``, ``pack``, ``adps``, ``labels``,
        ``hydrogens``, ``partFilter``, ``bondWidth``, ``bestView``,
        ``resetView``, ``saveImage``.
    :param height: CSS height of the viewer container.
    :param background: CSS page/canvas background colour.
    :param density: Residual (Fo−Fc) density to embed — ``None`` (the default)
        embeds none and keeps the page small, ``True`` computes one, or pass an
        already-exported payload.  See :func:`structure_data`.
    :param density_options: Keyword arguments for
        :func:`~fastmolwidget.web_export.export_density` when ``density=True``.
    :param density_level: Contour level the page starts at, in e/Å³.  ``None``
        uses the level stored in the payload (3σ of the map).
    """
    if title is None:
        title = Path(structure).name if isinstance(structure, str | Path) else 'Fastmolwidget'

    options = {
        'controls': controls,
        'grow': grow,
        'pack': pack,
        'adps': adps,
        'labels': labels,
        'hydrogens': hydrogens,
        'bondWidth': bond_width,
        'bestView': best_view,
        'background': background,
    }
    if bond_color is not None:
        options['bondColor'] = bond_color
    if density_level is not None:
        options['densityLevel'] = density_level

    return render_template(
        TEMPLATE_NAME,
        title=escape(title),
        background=background,
        height=height,
        bundle_js=bundle_js(),
        structure_json=structure_json(
            structure, density=density, density_options=density_options),
        options_json=json.dumps(options),
    )


def write_html(structure: Structure, out_path: str | Path, **kwargs: Any) -> Path:
    """Render *structure* with :func:`render_html` and write it to *out_path*.

    Returns the written path.
    """
    path = Path(out_path)
    path.write_text(render_html(structure, **kwargs), encoding='utf-8')
    return path
