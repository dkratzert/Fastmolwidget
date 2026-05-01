from fastmolwidget.viewer_widget import MoleculeViewerWidget
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule3D import MoleculeWidget3D, configure_opengl_format
from fastmolwidget.viewer_widget3D import MoleculeViewer3DWidget
from fastmolwidget.molecule_base import MoleculeWidgetProtocol

__version__ = "0.6.3"

__all__ = [
    "MoleculeViewerWidget",
    "MoleculeWidget",
    "MoleculeLoader",
    "MoleculeWidget3D",
    "MoleculeViewer3DWidget",
    "MoleculeWidgetProtocol",
    "configure_opengl_format",
    "__version__",
]


def main() -> None:
    print("Hello from fastmolwidget!")
