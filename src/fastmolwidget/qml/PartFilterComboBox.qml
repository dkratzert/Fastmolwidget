/**
 * PartFilterComboBox.qml
 *
 * QML counterpart of the Python ``PartFilterWidget`` / ``_CheckableComboBox``
 * (see ``part_combo.py``).
 *
 * Displays a "Show Parts:" label and a button whose text summarises the
 * current selection ("All", "None", or comma-separated part numbers).
 * Clicking the button opens a Popup with one CheckBox per disorder part.
 * The Popup opens **upward** so it stays within the QQuickWidget bounds
 * (the button sits at the bottom of the window).
 *
 * Properties
 * ----------
 *   partsModel : list<int>   – set by the backend (empty → hidden)
 *
 * Signals
 * -------
 *   partToggled(int part, bool checked)
 */

import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    id: root

    // ── public interface ─────────────────────────────────────────────────
    /** Disorder-part numbers to display (provided by the backend). */
    property var partsModel: []

    /** Emitted when the user ticks/unticks a part checkbox. */
    signal partToggled(int part, bool checked)

    // ── sizing ───────────────────────────────────────────────────────────
    implicitWidth: row.implicitWidth
    implicitHeight: row.implicitHeight

    // ── internal state ───────────────────────────────────────────────────
    /** Map  part-number → bool.  Initialised to all-checked on model change. */
    property var _checkedParts: ({})
    property string _summaryText: "All"

    onPartsModelChanged: {
        // Reset all parts to checked whenever the model is replaced.
        var m = {};
        for (var i = 0; i < partsModel.length; ++i)
            m[partsModel[i]] = true;
        _checkedParts = m;
        _updateSummary();
    }

    function _updateSummary() {
        var total = partsModel.length;
        if (total === 0) { _summaryText = "All"; return; }

        var checkedList = [];
        for (var i = 0; i < total; ++i) {
            var p = partsModel[i];
            if (_checkedParts[p] !== false)
                checkedList.push(p);
        }
        if (checkedList.length === total)
            _summaryText = "All";
        else if (checkedList.length === 0)
            _summaryText = "None";
        else
            _summaryText = checkedList.join(", ");
    }

    // ── visual tree ──────────────────────────────────────────────────────
    RowLayout {
        id: row
        spacing: 4

        Label { text: "Show Parts:" }

        Button {
            id: btn
            text: root._summaryText
            // Keep a reasonable minimum so the button never collapses to zero.
            implicitWidth: Math.max(fontMetrics.advanceWidth("All Parts") + 40, 80)
            onClicked: popup.opened ? popup.close() : popup.open()

            FontMetrics { id: fontMetrics; font: btn.font }
        }
    }

    Popup {
        id: popup
        // Open UPWARD: position the popup so its bottom edge aligns with
        // the top of the button row.  This keeps it inside the QQuickWidget.
        x: btn.x
        y: -popup.height - 2

        width: Math.max(btn.width, col.implicitWidth + 2 * padding)
        padding: 6

        // Keep the popup open while the user clicks checkboxes; only close
        // on Escape or clicking outside.
        closePolicy: Popup.CloseOnEscape | Popup.CloseOnPressOutside

        ColumnLayout {
            id: col
            spacing: 2

            Repeater {
                model: root.partsModel

                CheckBox {
                    text: "Part " + modelData
                    checked: root._checkedParts[modelData] !== false
                    onToggled: {
                        var cp = root._checkedParts;
                        cp[modelData] = checked;
                        root._checkedParts = cp;
                        root._updateSummary();
                        root.partToggled(modelData, checked);
                    }
                }
            }
        }
    }
}