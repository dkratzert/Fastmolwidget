"""Tests for the threaded JavaScript demo server."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from fastmolwidget.web_demo_server import run_server

CIF = Path('tests/test-data/p21c.cif')


@pytest.fixture
def server():
    srv = run_server(CIF, port=0, open_browser=False)
    yield srv
    srv.shutdown()
    srv.server_close()


def _get(server, path: str) -> str:
    host, port = server.server_address[:2]
    with urllib.request.urlopen(f'http://{host}:{port}{path}', timeout=10) as response:
        return response.read().decode('utf-8')


def test_serves_a_self_contained_page(server):
    html = _get(server, '/')
    assert html.startswith('<!doctype html>')
    assert 'Fastmolwidget.createViewer' in html
    assert 'root.Fastmolwidget' in html  # the inlined bundle
    assert 'p21c.cif' in html


def test_serves_the_structure_json(server):
    data = json.loads(_get(server, '/structure.json'))
    assert data['atoms']
    assert len(data['cell']) == 6


def test_serves_the_raw_es_modules(server):
    assert 'export class MoleculeViewer2D' in _get(server, '/viewer.js')


def test_missing_file_is_reported():
    with pytest.raises(FileNotFoundError):
        run_server(Path('does-not-exist.cif'), port=0, open_browser=False)
