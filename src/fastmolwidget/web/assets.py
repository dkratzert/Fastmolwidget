"""Access shipped JavaScript and HTML assets."""

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
    """Return the ``importlib.resources`` traversable for a shipped file."""
    resource = files(_PACKAGE)
    for part in parts:
        resource = resource.joinpath(part)
    return resource


def _package_dir(*parts: str) -> Path:
    """Return a real filesystem path for shipped data.

    ``as_file`` is only needed for zipped installs.
    """
    resource = _resource(*parts)
    if isinstance(resource, Path):
        return resource
    with as_file(resource) as path:
        return Path(path)


def js_directory() -> Path:
    """Return the directory of shipped JS modules."""
    return _package_dir(_JS_DIR)


def template_directory() -> Path:
    """Return the directory of shipped HTML templates."""
    return _package_dir(_TEMPLATE_DIR)


def js_module_names() -> list[str]:
    """Return sorted names of shipped JS modules."""
    return sorted(p.name for p in _resource(_JS_DIR).iterdir() if p.name.endswith('.js'))


def read_js(name: str) -> str:
    """Return the source of shipped JS module *name*."""
    resource = _resource(_JS_DIR, name)
    if not resource.is_file():
        raise FileNotFoundError(f'No such JavaScript module: {name}')
    return resource.read_text(encoding='utf-8')


def js_source_map() -> dict[str, str]:
    """Return ``module name -> source`` for shipped JS modules."""
    return {name: read_js(name) for name in js_module_names()}


def read_template(name: str) -> str:
    """Return the raw text of shipped template *name*."""
    resource = _resource(_TEMPLATE_DIR, name)
    if not resource.is_file():
        raise FileNotFoundError(f'No such template: {name}')
    return resource.read_text(encoding='utf-8')


def render_template(name: str, /, **values: str) -> str:
    """Render shipped template *name* with :class:`string.Template`.

    ``string.Template`` avoids escaping CSS/JS braces and adds no dependency.
    """
    return Template(read_template(name)).substitute(**values)
