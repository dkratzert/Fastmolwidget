"""Threaded demo web server for the JavaScript renderer shipped in
``fastmolwidget/web/js``.

Parses a CIF in Python (:mod:`fastmolwidget.web_export`) and serves the
self-contained page produced by :func:`fastmolwidget.web.render_html`, using
:class:`http.server.ThreadingHTTPServer`.  The raw ES modules are served as
well, so they can still be loaded directly during JavaScript development.

Run directly::

    uv run python -m fastmolwidget.web_demo_server
    uv run python -m fastmolwidget.web_demo_server --cif tests/test-data/p31c.cif --port 8080

Then open the printed URL (a browser tab is opened automatically unless
``--no-browser`` is passed).
"""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from fastmolwidget.web import (
    bundle_js,
    js_directory,
    render_html,
    structure_data,
    structure_json,
)

DEFAULT_CIF = Path('tests') / 'test-data' / 'p21c.cif'

__all__ = ['DEFAULT_CIF', 'main', 'run_server']


def _make_handler(cif_path: Path) -> type[SimpleHTTPRequestHandler]:
    """Build a request-handler class serving the viewer page for *cif_path*.

    The page is rendered per request (with the bundle cache cleared) so editing
    a JavaScript module and reloading the browser shows the change immediately.
    Every other path falls back to the shipped ES modules.
    """
    js_dir = str(js_directory())

    class DemoHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=js_dir, **kwargs)

        def do_GET(self) -> None:
            if self.path in ('/', '/index.html'):
                bundle_js.cache_clear()
                html = render_html(cif_path, title=cif_path.name, controls=True)
                self._send_bytes(html.encode('utf-8'), 'text/html; charset=utf-8')
            elif self.path == '/structure.json':
                self._send_bytes(structure_json(cif_path).encode('utf-8'), 'application/json')
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
    if not cif_path.is_file():
        raise FileNotFoundError(f'No such structure file: {cif_path} (pass one with --cif)')
    n_atoms = len(structure_data(cif_path)['atoms'])

    server = ThreadingHTTPServer((host, port), _make_handler(cif_path))

    thread = threading.Thread(target=server.serve_forever, name='fastmolwidget-web-demo', daemon=True)
    thread.start()

    url = f'http://{host}:{port}/'
    print(f'Serving {cif_path.name} ({n_atoms} atoms) at {url} (Ctrl+C to stop)')
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
