/**
 * MolViewer3D.qml
 *
 * Ready-to-use Qt Quick viewer that wraps a MoleculeQuick3D item with a
 * minimal control panel.  Mirrors the controls of MoleculeViewer3DWidget.
 *
 * Prerequisites
 * -------------
 * Register MoleculeQuick3D from Python before loading this file::
 *
 *     from fastmolwidget.molecule_quick3D import MoleculeQuick3D, setup_opengl_backend
 *     from qtpy.QtQml import qmlRegisterType
 *     setup_opengl_backend()
 *     app = QGuiApplication(sys.argv)
 *     engine = QQmlApplicationEngine()
 *     qmlRegisterType(MoleculeQuick3D, "MolWidget", 1, 0, "MoleculeQuick3D")
 *     engine.load("MolViewer3D.qml")
 *
 * Exposed properties
 * ------------------
 * molecule  (MoleculeQuick3D, read-only) – the underlying renderer item.
 */

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import MolWidget 1.0

Item {
    id: root

    // Read-only access to the underlying renderer.
    readonly property var molecule: mol

    // ── 3-D molecule renderer ─────────────────────────────────────────────
    MoleculeQuick3D {
        id: mol
        anchors {
            top:    parent.top
            left:   parent.left
            right:  parent.right
            bottom: controlPanel.top
        }

        // Atom / bond labels positioned by Python, rendered here.
        Repeater {
            model: mol.labelPositions
            delegate: Text {
                required property var modelData
                x: modelData.x + 4
                y: modelData.y - 4
                text: modelData.text
                color: {
                    if (modelData.kind === "hover_atom")  return "#1a6ecc";
                    if (modelData.kind === "hover_bond")  return "#147a41";
                    return "#6b3200";
                }
                font.pixelSize: mol.labelFontSize
                font.bold:      modelData.kind !== "atom"
                // Slight text shadow to keep labels readable on bright backgrounds.
                layer.enabled:  true
            }
        }
    }

    // ── Control panel ─────────────────────────────────────────────────────
    RowLayout {
        id: controlPanel
        anchors {
            left:   parent.left
            right:  parent.right
            bottom: bondRow.top
        }
        spacing: 8
        leftPadding:  4
        rightPadding: 4

        CheckBox {
            text: "Show ADP"
            checked: mol.showAdps
            onToggled: mol.showAdps = checked
        }
        CheckBox {
            text: "Show Labels"
            checked: mol.showLabels
            onToggled: mol.showLabels = checked
        }
        CheckBox {
            id: hideHydrogensCb
            text: "Hide Hydrogens"
            checked: false
            onToggled: mol.showHydrogens = !checked
        }
        Item { Layout.fillWidth: true }
    }

    // ── Bond / reset row ──────────────────────────────────────────────────
    RowLayout {
        id: bondRow
        anchors {
            left:   parent.left
            right:  parent.right
            bottom: parent.bottom
        }
        spacing: 8
        leftPadding:  4
        rightPadding: 4

        Label { text: "Bond Width:" }
        SpinBox {
            from: 0
            to:   15
            value: mol.bondWidth
            onValueModified: mol.bondWidth = value
        }

        Button {
            text: "Reset View"
            onClicked: mol.reset_view()
        }
        Button {
            text: "Reset Rotation Center"
            onClicked: mol.reset_rotation_center()
        }
        Item { Layout.fillWidth: true }
    }
}
