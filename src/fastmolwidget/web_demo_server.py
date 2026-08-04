"""Threaded demo web server that exposes a CIF structure through the
JavaScript renderer in ``js/`` (see ``js/README.md``).

Parses the CIF in Python (:mod:`fastmolwidget.web_export`) and serves the
exported asymmetric-unit JSON alongside the ``js/`` renderer modules and a
generated demo page, using :class:`http.server.ThreadingHTTPServer` so the
several concurrent requests a browser makes (HTML, JS modules, the JSON
payload) are each handled in their own thread.

Run directly::

    uv run python -m fastmolwidget.web_demo_server
    uv run python -m fastmolwidget.web_demo_server --cif tests/test-data/p31c.cif --port 8080

Then open the printed URL (a browser tab is opened automatically unless
``--no-browser`` is passed).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastmolwidget.web_export import export_cif

REPO_ROOT = Path(__file__).resolve().parents[2]
JS_DIR = REPO_ROOT / 'js'
DEFAULT_CIF = REPO_ROOT / 'tests' / 'test-data' / 'p21c.cif'

__all__ = ['main', 'run_server']

# Same controls/layout as js/demo/index.html, adapted to be served from the
# js/ directory root (so the relative './viewer.js' import resolves) and to
# load the dynamically generated '/structure.json' instead of a static file.
_DEMO_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Fastmolwidget — {title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 0; display: flex; flex-direction: column; height: 100vh; }}
    #bar {{ display: flex; gap: 8px; align-items: center; padding: 6px 10px; border-bottom: 1px solid #ccc; flex-wrap: wrap; }}
    #bar label {{ display: flex; align-items: center; gap: 4px; font-size: 13px; }}
    canvas {{ flex: 1; display: block; width: 100%; height: 100%; cursor: grab; }}
    #status {{ font-size: 12px; color: #555; margin-left: auto; }}
  </style>
</head>
<body>
  <div id="bar">
    <label><input id="growChk" type="checkbox"> Grow</label>
    <label><input id="packChk" type="checkbox"> Pack unit cell</label>
    <label><input id="adpChk" type="checkbox" checked> ADPs</label>
    <label><input id="labelChk" type="checkbox"> Labels</label>
    <label><input id="hChk" type="checkbox" checked> Show H</label>
    <label>Bond width <input id="bondWidth" type="range" min="1" max="15" value="3"></label>
    <button id="bestViewBtn">Best view</button>
    <button id="resetBtn">Reset view</button>
    <button id="saveBtn">Save image</button>
    <span id="status">Loading {title}…</span>
  </div>
  <canvas id="canvas"></canvas>

  <script type="module">
    import {{ MoleculeViewer2D }} from './viewer.js';

    const canvas = document.getElementById('canvas');
    function fitCanvas() {{
      const rect = canvas.getBoundingClientRect();
      if (viewer) viewer.widget.resize(rect.width, rect.height);
    }}

    const viewer = new MoleculeViewer2D(canvas);
    window.viewer = viewer; // for manual console poking

    window.addEventListener('resize', fitCanvas);
    fitCanvas();

    const status = document.getElementById('status');
    viewer.widget.addEventListener('atomClicked', (e) => {{ status.textContent = `Atom: ${{e.detail}}`; }});
    viewer.widget.addEventListener('bondClicked', (e) => {{ status.textContent = `Bond: ${{e.detail.join('-')}}`; }});

    document.getElementById('growChk').addEventListener('change', (e) => {{
      if (e.target.checked) document.getElementById('packChk').checked = false;
      viewer.setGrow(e.target.checked);
    }});
    document.getElementById('packChk').addEventListener('change', (e) => {{
      if (e.target.checked) document.getElementById('growChk').checked = false;
      viewer.setPack(e.target.checked);
    }});
    document.getElementById('adpChk').addEventListener('change', (e) => viewer.widget.showAdps(e.target.checked));
    document.getElementById('labelChk').addEventListener('change', (e) => viewer.widget.showLabels(e.target.checked));
    document.getElementById('hChk').addEventListener('change', (e) => viewer.widget.showHydrogens(e.target.checked));
    document.getElementById('bondWidth').addEventListener('input', (e) => viewer.widget.setBondWidth(parseInt(e.target.value, 10)));
    document.getElementById('bestViewBtn').addEventListener('click', () => viewer.widget.alignBestView());
    document.getElementById('resetBtn').addEventListener('click', () => viewer.widget.resetView());
    document.getElementById('saveBtn').addEventListener('click', () => viewer.widget.saveImage('molecule.png'));

    fetch('/structure.json')
      .then((r) => r.json())
      .then((data) => {{
        viewer.loadStructure(data);
        fitCanvas();
        viewer.widget.resetView();
        status.textContent = `Loaded {title}: ${{data.atoms.length}} atoms`;
      }})
      .catch((err) => {{ status.textContent = `Failed to load structure: ${{err}}`; }});
  </script>
</body>
</html>
"""


def _make_handler(structure_json: bytes, demo_html: bytes) -> type[SimpleHTTPRequestHandler]:
    """Build a request-handler class closing over the pre-rendered response
    bodies for ``/`` and ``/structure.json``; every other path falls back to
    serving static files from :data:`JS_DIR` (the renderer modules)."""

    class DemoHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(JS_DIR), **kwargs)

        def do_GET(self) -> None:
            if self.path in ('/', '/index.html'):
                self._send_bytes(demo_html, 'text/html; charset=utf-8')
            elif self.path == '/structure.json':
                self._send_bytes(structure_json, 'application/json')
            else:
                super().do_GET()

        def _send_bytes(self, data: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def end_headers(self) -> None:
            # Disable all caching so a reload always fetches the current
            # HTML/JS/JSON instead of a stale, browser-cached copy.
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            super().end_headers()

        def log_message(self, format: str, *args) -> None:
            sys.stderr.write(f'[web_demo_server] {self.address_string()} - {format % args}\n')

    return DemoHandler


def run_server(
    cif_path: Path = DEFAULT_CIF,
    host: str = '127.0.0.1',
    port: int = 8000,
    open_browser: bool = True,
) -> ThreadingHTTPServer:
    """Parse *cif_path* and start the threaded demo server in a background
    thread. Returns the running :class:`~http.server.ThreadingHTTPServer` so
    the caller can ``server.shutdown()`` it later (e.g. in tests)."""
    cif_path = Path(cif_path)
    data = export_cif(cif_path)
    structure_json = json.dumps(data).encode('utf-8')
    demo_html = _DEMO_HTML_TEMPLATE.format(title=cif_path.name).encode('utf-8')

    handler_cls = _make_handler(structure_json, demo_html)
    server = ThreadingHTTPServer((host, port), handler_cls)

    thread = threading.Thread(target=server.serve_forever, name='fastmolwidget-web-demo', daemon=True)
    thread.start()

    url = f'http://{host}:{port}/'
    print(f'Serving {cif_path.name} ({len(data["atoms"])} atoms) at {url} (Ctrl+C to stop)')
    if open_browser:
        webbrowser.open(url)
    return server


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cif', type=Path, default=DEFAULT_CIF, help='CIF file to display')
    parser.add_argument('--host', default='127.0.0.1', help='Bind address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on (default: 8000)')
    parser.add_argument('--no-browser', action='store_true', help="Don't open a browser window automatically")
    args = parser.parse_args(argv)

    server = run_server(args.cif, host=args.host, port=args.port, open_browser=not args.no_browser)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print('\nShutting down…')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
