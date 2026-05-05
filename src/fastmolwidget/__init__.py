from fastmolwidget.viewer_widget import MoleculeViewerWidget
from fastmolwidget.molecule2D import MoleculeWidget
from fastmolwidget.loader import MoleculeLoader
from fastmolwidget.molecule3D import MoleculeWidget3D
from fastmolwidget.viewer_widget3D import MoleculeViewer3DWidget
from fastmolwidget.molecule_base import MoleculeWidgetProtocol

__version__ = "0.7.1"

__all__ = [
    "MoleculeViewerWidget",
    "MoleculeWidget",
    "MoleculeLoader",
    "MoleculeWidget3D",
    "MoleculeViewer3DWidget",
    "MoleculeWidgetProtocol",
    "__version__",
]


def main() -> None:
    import argparse
    import sys
    from qtpy.QtWidgets import QApplication

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
