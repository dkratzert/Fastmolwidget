"""Web helpers for the bundled JS renderer and self-contained HTML pages."""

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
