"""Tests for the HTML/report-embedding helpers (:mod:`fastmolwidget.web.html`)."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fastmolwidget.web import (
    assets,
    render_html,
    structure_data,
    structure_json,
    write_html,
)

CIF = Path('tests/test-data/p21c.cif')
RES = Path('tests/test-data/p31c-finalcif.res')


def test_template_is_shipped():
    assert 'viewer.html' in [p.name for p in assets.template_directory().iterdir()]
    assert '$bundle_js' in assets.read_template('viewer.html')


def test_structure_data_from_cif():
    data = structure_data(CIF)
    assert data['cell'] and len(data['cell']) == 6
    assert data['atoms']
    assert {'label', 'type', 'x', 'y', 'z', 'part', 'adp'} <= set(data['atoms'][0])


def test_structure_data_passes_dicts_through():
    payload = {'cell': [1, 1, 1, 90, 90, 90], 'symmops': [], 'centric': False, 'atoms': []}
    assert structure_data(payload) is payload


def test_structure_data_rejects_unknown_suffix():
    with pytest.raises(ValueError, match='Unsupported file type'):
        structure_data('structure.xyz')


def test_structure_json_is_script_tag_safe():
    payload = {'atoms': [{'label': '</script><!--', 'type': 'C', 'x': 0, 'y': 0, 'z': 0, 'part': 0}]}
    text = structure_json(payload)
    assert '</script' not in text
    assert '<!--' not in text
    assert json.loads(text)['atoms'][0]['label'] == '</script><!--'


def test_render_html_is_self_contained():
    html = render_html(CIF)
    assert html.startswith('<!doctype html>')
    assert 'Fastmolwidget.createViewer' in html
    assert 'root.Fastmolwidget' in html  # the inlined bundle
    assert 'src=' not in html  # no external resources
    assert '$' not in html.split('<script>')[0]  # every placeholder substituted
    assert 'p21c.cif' in html


def test_render_html_options_reach_the_page():
    html = render_html(CIF, controls=False, grow=True, labels=True, bond_width=7,
                       bond_color='#123456', background='#101010', height='400px')
    match = re.search(r'var fastmolwidgetOptions = (\{.*?\});', html)
    assert match is not None
    options = json.loads(match.group(1))
    assert options['controls'] is False
    assert options['grow'] is True
    assert options['labels'] is True
    assert options['bondWidth'] == 7
    assert options['bondColor'] == '#123456'
    assert 'height: 400px' in html
    assert '#101010' in html


def test_render_html_escapes_the_title():
    assert '<b>' not in render_html(CIF, title='<b>x</b>')


def test_render_html_from_shelx():
    assert 'Fastmolwidget.createViewer' in render_html(RES)


def test_write_html(tmp_path: Path):
    out = write_html(CIF, tmp_path / 'structure.html')
    assert out.is_file()
    assert out.read_text(encoding='utf-8').startswith('<!doctype html>')


def test_web_package_imports_without_qt():
    """Report generators must be able to use the web helpers with no Qt binding.

    Checked by asserting that neither ``fastmolwidget`` nor
    ``fastmolwidget.web`` pulls in ``qtpy`` (or a Qt binding) at import time.
    """
    code = (
        'import sys, fastmolwidget, fastmolwidget.web; '
        'fastmolwidget.web.bundle_js(); '
        'fastmolwidget.web.render_html'
        '; print(sorted(m for m in sys.modules '
        "if m in ('qtpy', 'PySide6', 'PyQt6', 'PyQt5')))"
    )
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == '[]'
