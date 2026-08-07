"""Access to the JavaScript renderer and HTML templates shipped inside the
package (``fastmolwidget/web/js`` and ``fastmolwidget/web/templates``).

The assets are ordinary package data, so they are available from an installed
wheel as well as from a source checkout::

    from fastmolwidget.web.assets import js_directory, read_js

    print(js_directory())          # .../site-packages/fastmolwidget/web/js
    print(read_js('viewer.js'))

This module deliberately imports nothing from Qt so it can be used in headless
report generators.
"""

from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path
from string import Template

__all__ = [
    'js_directory',
    'js_module_names',
    'js_source_map',
    'read_js',
    'read_template',
    'render_template',
    'template_directory',
]

_PACKAGE = 'fastmolwidget.web'
_JS_DIR = 'js'
_TEMPLATE_DIR = 'templates'


def _resource(*parts: str):
    """Return the ``importlib.resources`` traversable for a shipped data file."""
    resource = files(_PACKAGE)
    for part in parts:
        resource = resource.joinpath(part)
    return resource


def _package_dir(*parts: str) -> Path:
    """Return a real filesystem directory for shipped data.

    For a normal (extracted) wheel install ``files()`` already yields a
    :class:`pathlib.Path`; ``as_file`` is only used as a fallback for exotic
    (zipped) installs.
    """
    resource = _resource(*parts)
    if isinstance(resource, Path):
        return resource
    with as_file(resource) as path:
        return Path(path)


def js_directory() -> Path:
    """Filesystem directory holding the ES-module sources of the JS renderer."""
    return _package_dir(_JS_DIR)


def template_directory() -> Path:
    """Filesystem directory holding the HTML page templates."""
    return _package_dir(_TEMPLATE_DIR)


def js_module_names() -> list[str]:
    """Sorted names of all shipped JavaScript modules (``'viewer.js'`` …)."""
    return sorted(p.name for p in _resource(_JS_DIR).iterdir() if p.name.endswith('.js'))


def read_js(name: str) -> str:
    """Return the source of the shipped JavaScript module *name*."""
    resource = _resource(_JS_DIR, name)
    if not resource.is_file():
        raise FileNotFoundError(f'No such JavaScript module: {name}')
    return resource.read_text(encoding='utf-8')


def js_source_map() -> dict[str, str]:
    """Map of ``module name -> source`` for every shipped JavaScript module."""
    return {name: read_js(name) for name in js_module_names()}


def read_template(name: str) -> str:
    """Return the raw text of the shipped HTML template *name*."""
    resource = _resource(_TEMPLATE_DIR, name)
    if not resource.is_file():
        raise FileNotFoundError(f'No such template: {name}')
    return resource.read_text(encoding='utf-8')


def render_template(name: str, /, **values: str) -> str:
    """Render the shipped template *name* with :class:`string.Template`.

    ``string.Template`` (``$placeholder``) is used instead of ``str.format`` or
    f-strings because the templates contain literal CSS/JavaScript braces, and
    instead of Jinja2 to avoid a runtime dependency.  Missing placeholders raise
    :class:`KeyError`.
    """
    return Template(read_template(name)).substitute(**values)
