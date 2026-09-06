"""HTML helpers for the web renderer."""

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
    """Escape ``<``, ``>`` and ``&`` as ``\\uXXXX`` so the result stays valid
    JSON and can't close the surrounding ``<script>`` tag. (``<\\/script>``
    would work in JS but isn't valid JSON, so it can't be used here.)
    """
    return text.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def structure_data(
    structure: Structure,
    *,
    density: dict[str, Any] | bool | None = None,
    density_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the web JSON contract for *structure*.

    *structure* may be a file path or a prebuilt dict. ``density=True`` needs
    the model file so the map can be computed.
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
    """Return *structure* as JSON safe for a ``<script>`` tag."""
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
    """Render *structure* as a self-contained HTML document."""
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
    """Render *structure* and write it to *out_path*."""
    path = Path(out_path)
    path.write_text(render_html(structure, **kwargs), encoding='utf-8')
    return path
