"""Fastmolwidget — embeddable widgets for displaying crystal structures.

The Qt widgets are imported lazily (:pep:`562`) so that Qt-free consumers — for
example HTML report generators using :mod:`fastmolwidget.web` — can import this
package without a Qt binding installed.  ``from fastmolwidget import
MoleculeWidget`` keeps working unchanged.
"""

from typing import TYPE_CHECKING

__version__ = "1.0.0"

if TYPE_CHECKING:
    from fastmolwidget.loader import MoleculeLoader
    from fastmolwidget.molecule2D import MoleculeWidget
    from fastmolwidget.molecule3D import MoleculeWidget3D
    from fastmolwidget.molecule_base import MoleculeWidgetProtocol
    from fastmolwidget.molecule_painter import MoleculeRendererMixin
    from fastmolwidget.molecule_quick import MoleculeQuickItem
    from fastmolwidget.sdm import Atomtuple
    from fastmolwidget.viewer_widget import MoleculeViewerWidget
    from fastmolwidget.viewer_widget3D import MoleculeViewer3DWidget
    from fastmolwidget.viewer_widget_quick import (
        MoleculeViewerBackend,
        MoleculeViewerQuickWidget,
    )
    from fastmolwidget.web import bundle_js, render_html, structure_json, write_html

# public name -> module it lives in
_LAZY_IMPORTS = {
    "MoleculeViewerWidget": "fastmolwidget.viewer_widget",
    "MoleculeViewerQuickWidget": "fastmolwidget.viewer_widget_quick",
    "MoleculeViewerBackend": "fastmolwidget.viewer_widget_quick",
    "MoleculeQuickItem": "fastmolwidget.molecule_quick",
    "MoleculeWidget": "fastmolwidget.molecule2D",
    "MoleculeLoader": "fastmolwidget.loader",
    "MoleculeWidget3D": "fastmolwidget.molecule3D",
    "MoleculeViewer3DWidget": "fastmolwidget.viewer_widget3D",
    "MoleculeWidgetProtocol": "fastmolwidget.molecule_base",
    "MoleculeRendererMixin": "fastmolwidget.molecule_painter",
    "Atomtuple": "fastmolwidget.sdm",
    "bundle_js": "fastmolwidget.web",
    "structure_json": "fastmolwidget.web",
    "render_html": "fastmolwidget.web",
    "write_html": "fastmolwidget.web",
}

__all__ = [
    "Atomtuple",
    "MoleculeLoader",
    "MoleculeQuickItem",
    "MoleculeRendererMixin",
    "MoleculeViewer3DWidget",
    "MoleculeViewerBackend",
    "MoleculeViewerQuickWidget",
    "MoleculeViewerWidget",
    "MoleculeWidget",
    "MoleculeWidget3D",
    "MoleculeWidgetProtocol",
    "__version__",
    "bundle_js",
    "render_html",
    "structure_json",
    "write_html",
]


def __getattr__(name: str):
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value  # cache so later lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


def main() -> None:
    import argparse
    import sys

    from qtpy.QtWidgets import QApplication

    from fastmolwidget.viewer_widget import MoleculeViewerWidget
    from fastmolwidget.viewer_widget3D import MoleculeViewer3DWidget

    parser = argparse.ArgumentParser(description="Fastmolwidget crystal structure viewer.")
    parser.add_argument("mode", choices=["2D", "3D", "2d", "3d"], type=str.upper, help="Display mode: 2D or 3D")
    parser.add_argument("file", type=str, help="Path to a molecule file (CIF, RES, XYZ, etc.)")
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv)

    if args.mode == "2D":
        viewer = MoleculeViewerWidget()
    else:
        viewer = MoleculeViewer3DWidget()

    viewer.load_file(args.file)
    viewer.show()

    if hasattr(app, "exec"):
        sys.exit(app.exec())
    else:
        sys.exit(app.exec_())
