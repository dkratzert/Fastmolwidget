/**
 * MoleculeViewer.qml
 *
 * Qt Quick control bar + molecule canvas.
 *
 * Loaded by MoleculeViewerQuickWidget (viewer_widget_quick.py).
 * The Python backend is exposed via the "backend" context property.
 *
 * Layout
 * ------
 *   ColumnLayout
 *     MoleculeItem   ← fills all available space (registered Python type)
 *     RowLayout      ← Row 1: structure toggles
 *     RowLayout      ← Row 2: bond controls + optional parts filter
 */

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Fastmolwidget 1.0

ColumnLayout {
    id: root
    anchors.fill: parent
    spacing: 2

    // ── Molecule renderer ────────────────────────────────────────────────
    MoleculeItem {
        id: mol
        Layout.fillWidth: true
        Layout.fillHeight: true

        // Keyboard focus so F1/F2/F3 work when the item is clicked
        focus: true
        Keys.onPressed: function(event) {
            if (event.key === Qt.Key_F1) {
                backend.alignAxis(0)
                event.accepted = true
            } else if (event.key === Qt.Key_F2) {
                backend.alignAxis(1)
                event.accepted = true
            } else if (event.key === Qt.Key_F3) {
                backend.alignAxis(2)
                event.accepted = true
            }
        }

        // Hand the live item reference back to the Python backend so it can
        // create the MoleculeLoader and wire signals.
        Component.onCompleted: backend.registerRenderItem(mol)
    }

    // ── Row 1: structure toggles ─────────────────────────────────────────
    RowLayout {
        Layout.fillWidth: true
        spacing: 6
        Layout.leftMargin: 4
        Layout.rightMargin: 4
        Layout.bottomMargin: 2

        Button {
            text: "Open File…"
            onClicked: backend.openFileDialog()
        }

        CheckBox {
            id: growCheck
            text: "Grow"
            checked: backend.growActive
            onToggled: backend.setGrow(checked)
            // Keep in sync when the backend changes state (mutual exclusion)
            Connections {
                target: backend
                function onGrowActiveChanged(v) { growCheck.checked = v }
            }
        }

        CheckBox {
            id: packCheck
            text: "Pack Unit Cell"
            checked: backend.packActive
            onToggled: backend.setPack(checked)
            Connections {
                target: backend
                function onPackActiveChanged(v) { packCheck.checked = v }
            }
        }

        CheckBox {
            text: "Show ADP"
            checked: backend.showAdps
            onToggled: backend.setShowAdps(checked)
        }

        CheckBox {
            text: "Show Labels"
            checked: backend.showLabels
            onToggled: backend.setShowLabels(checked)
        }

        CheckBox {
            text: "Hide Hydrogens"
            checked: backend.hideHydrogens
            onToggled: backend.setHideHydrogens(checked)
        }

        // Spacer
        Item { Layout.fillWidth: true }
    }

    // ── Row 2: bond controls + parts filter ──────────────────────────────
    RowLayout {
        Layout.fillWidth: true
        spacing: 6
        Layout.leftMargin: 4
        Layout.rightMargin: 4
        Layout.bottomMargin: 4

        Label { text: "Bond Width:" }

        SpinBox {
            id: bondWidthSpin
            from: 1
            to: 15
            value: 3
            onValueChanged: backend.setBondWidth(value)
        }

        Button {
            text: "Bond Color…"
            onClicked: backend.chooseBondColor()
        }

        Button {
            text: "Reset Rotation Center"
            onClicked: backend.resetCenter()
        }

        Button {
            text: "Best View"
            onClicked: backend.bestView()
        }

        Button {
            text: "Save Image…"
            onClicked: backend.saveImageDialog()
        }

        // ── Inline parts filter ──────────────────────────────────────────
        // Visible only when the structure has multiple disorder parts.
        RowLayout {
            visible: backend.hasParts
            spacing: 4

            Label {
                text: "Parts:"
                visible: backend.hasParts
            }

            Repeater {
                model: backend.partsModel
                CheckBox {
                    text: "Part " + modelData
                    checked: true
                    onToggled: backend.togglePart(modelData, checked)
                }
            }
        }

        // Spacer
        Item { Layout.fillWidth: true }
    }
}

