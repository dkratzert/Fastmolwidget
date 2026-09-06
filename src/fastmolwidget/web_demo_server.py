"""Threaded demo server for the web renderer."""

from __future__ import annotations

import argparse
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from fastmolwidget.web import (
    bundle_js,
    js_directory,
    render_html,
    structure_data,
    structure_json,
)

DEFAULT_CIF = Path('tests') / 'test-data' / 'p21c.cif'

__all__ = ['DEFAULT_CIF', 'main', 'run_server']


def _make_handler(
    cif_path: Path,
    density: dict[str, Any] | None = None,
) -> type[SimpleHTTPRequestHandler]:
    """Build a handler serving the viewer page for *cif_path*.

    The page is rendered per request so JS edits show up on reload.
    """
    js_dir = str(js_directory())

    class DemoHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=js_dir, **kwargs)

        def do_GET(self) -> None:
            if self.path in ('/', '/index.html'):
                bundle_js.cache_clear()
                html = render_html(cif_path, title=cif_path.name, controls=True,
                                   density=density)
                self._send_bytes(html.encode('utf-8'), 'text/html; charset=utf-8')
            elif self.path == '/structure.json':
                payload = structure_json(cif_path, density=density)
                self._send_bytes(payload.encode('utf-8'), 'application/json')
            else:
                super().do_GET()

        def _send_bytes(self, data: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def end_headers(self) -> None:
            # Disable caching so reloads always fetch current assets.
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
    density: bool = False,
    density_options: dict[str, Any] | None = None,
) -> ThreadingHTTPServer:
    """Start the threaded demo server and return it."""
    cif_path = Path(cif_path)
    if not cif_path.is_file():
        raise FileNotFoundError(f'No such structure file: {cif_path} (pass one with --cif)')
    n_atoms = len(structure_data(cif_path)['atoms'])

    payload = None
    if density:
        from fastmolwidget.web_export import export_density

        print(f'Computing the residual density of {cif_path.name}…')
        payload = export_density(cif_path, **(density_options or {}))
        print(f'  {"x".join(str(n) for n in payload["size"])} grid, '
              f'level {payload["level"]:.2f} e/A^3, '
              f'{len(payload["data"]) / 1024:.0f} KB embedded')

    server = ThreadingHTTPServer((host, port), _make_handler(cif_path, payload))

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
    parser.add_argument('--density', action='store_true',
                        help='Compute and embed the residual (Fo-Fc) density map')
    parser.add_argument('--density-coverage', choices=('asu', 'grow', 'cell'), default='cell',
                        help='Which atoms to keep density around (default: cell, so Grow '
                             'and Pack both keep their density)')
    parser.add_argument('--density-spacing', type=float, default=None,
                        help='Density grid spacing in A (default: 0.25)')
    args = parser.parse_args(argv)

    density_options: dict[str, Any] = {'coverage': args.density_coverage}
    if args.density_spacing is not None:
        density_options['grid_spacing'] = args.density_spacing

    server = run_server(args.cif, host=args.host, port=args.port,
                        open_browser=not args.no_browser,
                        density=args.density, density_options=density_options)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        print('\nShutting down…')
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
