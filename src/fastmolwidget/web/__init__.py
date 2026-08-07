"""Web/HTML support: ship the JavaScript renderer to browsers and reports.

Public helpers:

- :func:`bundle_js` — the whole renderer as one classic ``<script>`` blob
  exposing ``window.Fastmolwidget``.
- :func:`structure_json` / :func:`structure_data` — a parsed structure in the
  fractional-coordinate JSON contract consumed by the renderer.
- :func:`render_html` / :func:`write_html` — a complete, self-contained page.
- :func:`js_directory` — filesystem location of the shipped ES modules.

Nothing in this package imports Qt.
"""

from fastmolwidget.web.assets import (
    js_directory,
    js_module_names,
    js_source_map,
    read_js,
    read_template,
    render_template,
    template_directory,
)
from fastmolwidget.web.bundle import GLOBAL_NAME, UnsupportedJsSyntaxError, bundle_js
from fastmolwidget.web.html import (
    TEMPLATE_NAME,
    render_html,
    structure_data,
    structure_json,
    write_html,
)

__all__ = [
    'GLOBAL_NAME',
    'TEMPLATE_NAME',
    'UnsupportedJsSyntaxError',
    'bundle_js',
    'js_directory',
    'js_module_names',
    'js_source_map',
    'read_js',
    'read_template',
    'render_html',
    'render_template',
    'structure_data',
    'structure_json',
    'template_directory',
    'write_html',
]
