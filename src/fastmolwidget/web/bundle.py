"""Bundle shipped ES modules into one classic ``<script>`` blob."""

from __future__ import annotations

import re
from functools import lru_cache

from fastmolwidget.web.assets import js_source_map

__all__ = ['ENTRY_MODULE', 'GLOBAL_NAME', 'UnsupportedJsSyntaxError', 'bundle_js']

ENTRY_MODULE = 'index.js'
GLOBAL_NAME = 'Fastmolwidget'

#: ``import { a, b as c } from './x.js';`` (multi-line too)
_IMPORT_RE = re.compile(
    r'^[ \t]*import\s*\{(?P<names>[^}]*)\}\s*from\s*[\'"](?P<source>[^\'"]+)[\'"];?[ \t]*$',
    re.MULTILINE | re.DOTALL,
)
#: ``export { A, B } from './y.js';``
_REEXPORT_RE = re.compile(
    r'^[ \t]*export\s*\{(?P<names>[^}]*)\}\s*from\s*[\'"](?P<source>[^\'"]+)[\'"];?[ \t]*$',
    re.MULTILINE | re.DOTALL,
)
#: ``export { A, B as C };``
_EXPORT_LIST_RE = re.compile(r'^[ \t]*export\s*\{(?P<names>[^}]*)\};?[ \t]*$', re.MULTILINE | re.DOTALL)
#: ``export function f(...)`` / ``export class C`` / ``export const X = ...``
_EXPORT_DECL_RE = re.compile(
    r'^(?P<indent>[ \t]*)export\s+(?P<kind>function\*?|class|const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)',
    re.MULTILINE,
)
#: Any remaining top-level import/export is unsupported.
_LEFTOVER_RE = re.compile(r'^[ \t]*(?P<word>import|export)\b', re.MULTILINE)

_RUNTIME = """\
var __fmw_modules = {};
var __fmw_cache = {};
function __fmw_require(name) {
  name = String(name).replace(/^\\.\\//, '');
  if (Object.prototype.hasOwnProperty.call(__fmw_cache, name)) return __fmw_cache[name];
  var factory = __fmw_modules[name];
  if (!factory) throw new Error('Fastmolwidget: unknown module ' + name);
  var exports = {};
  __fmw_cache[name] = exports;
  factory(exports, __fmw_require);
  return exports;
}
"""


class UnsupportedJsSyntaxError(ValueError):
    """Raised for unsupported JS module syntax."""


def _split_specifiers(names: str) -> list[tuple[str, str]]:
    """Parse ``'a, b as c'`` into ``[('a', 'a'), ('b', 'c')]``."""
    specs: list[tuple[str, str]] = []
    for raw in names.split(','):
        item = raw.strip()
        if not item:
            continue
        if ' as ' in item:
            source, _, alias = item.partition(' as ')
            specs.append((source.strip(), alias.strip()))
        else:
            specs.append((item, item))
    return specs


def _normalize(specifier: str, module: str) -> str:
    """Turn a relative import specifier into a bare module name."""
    if not specifier.startswith('./'):
        raise UnsupportedJsSyntaxError(
            f"{module}: only './name.js' import specifiers are supported, got {specifier!r}"
        )
    name = specifier[2:]
    if '/' in name:
        raise UnsupportedJsSyntaxError(f'{module}: nested module paths are not supported ({specifier!r})')
    return name


def _transform(module: str, source: str) -> tuple[str, list[str]]:
    """Rewrite one ES module into a registry factory body."""
    deps: list[str] = []
    export_lines: list[str] = []

    def on_import(match: re.Match[str]) -> str:
        dep = _normalize(match['source'], module)
        deps.append(dep)
        bindings = ', '.join(
            name if name == alias else f'{name}: {alias}' for name, alias in _split_specifiers(match['names'])
        )
        return f"const {{ {bindings} }} = __fmw_require('{dep}');"

    def on_reexport(match: re.Match[str]) -> str:
        dep = _normalize(match['source'], module)
        deps.append(dep)
        for name, alias in _split_specifiers(match['names']):
            export_lines.append(f"__fmw_exports.{alias} = __fmw_require('{dep}').{name};")
        return ''

    def on_export_list(match: re.Match[str]) -> str:
        for name, alias in _split_specifiers(match['names']):
            export_lines.append(f'__fmw_exports.{alias} = {name};')
        return ''

    def on_export_decl(match: re.Match[str]) -> str:
        export_lines.append(f"__fmw_exports.{match['name']} = {match['name']};")
        return f"{match['indent']}{match['kind']} {match['name']}"

    body = _IMPORT_RE.sub(on_import, source)
    body = _REEXPORT_RE.sub(on_reexport, body)
    body = _EXPORT_LIST_RE.sub(on_export_list, body)
    body = _EXPORT_DECL_RE.sub(on_export_decl, body)

    leftover = _LEFTOVER_RE.search(body)
    if leftover:
        line = body[: leftover.start()].count('\n') + 1
        snippet = body[leftover.start():].splitlines()[0].strip()
        raise UnsupportedJsSyntaxError(
            f'{module}:{line}: unsupported {leftover["word"]} statement for the '
            f'classic-script bundler: {snippet!r}'
        )
    # Export assignments go last: `const`/`class` are not bound earlier.
    if export_lines:
        body = body + '\n' + '\n'.join(export_lines) + '\n'
    return body, deps


def _check_no_cycles(graph: dict[str, list[str]], entry: str) -> None:
    """Raise if the module graph contains an import cycle.

    A cycle would expose incomplete exports.
    """
    visiting: set[str] = set()
    done: set[str] = set()

    def walk(name: str, stack: list[str]) -> None:
        if name in done:
            return
        if name in visiting:
            cycle = ' -> '.join([*stack, name])
            raise UnsupportedJsSyntaxError(f'Import cycle in the JavaScript modules: {cycle}')
        visiting.add(name)
        for dep in graph.get(name, ()):
            walk(dep, [*stack, name])
        visiting.discard(name)
        done.add(name)

    walk(entry, [])


def _escape_for_script_tag(text: str) -> str:
    """Make *text* safe inside an HTML ``<script>`` tag."""
    return text.replace('</script', r'<\/script').replace('<!--', r'<\!--')


@lru_cache(maxsize=None)  # noqa: UP033  (keyword form keeps `.cache_clear()` obvious)
def bundle_js(entry: str = ENTRY_MODULE, *, version: str | None = None) -> str:
    """Return the full renderer as one classic-script source."""
    if version is None:
        from fastmolwidget import __version__

        version = __version__

    sources = js_source_map()
    if entry not in sources:
        raise FileNotFoundError(f'No such JavaScript module: {entry}')

    bodies: dict[str, str] = {}
    graph: dict[str, list[str]] = {}
    for name, source in sources.items():
        body, deps = _transform(name, source)
        for dep in deps:
            if dep not in sources:
                raise UnsupportedJsSyntaxError(f'{name}: imports unknown module {dep!r}')
        bodies[name] = body
        graph[name] = deps
    _check_no_cycles(graph, entry)

    parts = [
        '/* Fastmolwidget JavaScript renderer — generated bundle, do not edit. */',
        f";(function (root) {{\n'use strict';\n{_RUNTIME}",
    ]
    for name in sorted(bodies):
        parts.append(
            f"__fmw_modules['{name}'] = function (__fmw_exports, __fmw_require) {{\n"
            f'{bodies[name]}\n}};'
        )
    parts.append(
        f"root.{GLOBAL_NAME} = __fmw_require('{entry}');\n"
        f'root.{GLOBAL_NAME}.version = {version!r};\n'
        "})(typeof window !== 'undefined' ? window : globalThis);"
    )
    return _escape_for_script_tag('\n'.join(parts) + '\n')
