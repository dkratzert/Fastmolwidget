/**
 * DensityLevelSpinBox.qml
 *
 * Fractional spin box for the residual-density contour level, the QML
 * counterpart of the ``QDoubleSpinBox`` in ``density_controls.py``.
 *
 * QtQuick.Controls' SpinBox is integer-only, so the level is held internally
 * as hundredths of an e/Å³ and converted for display; ``decimalValue`` and
 * ``setLevel()`` are the only things callers need to touch.
 *
 * Properties
 * ----------
 *   minimumLevel : real   – lowest selectable level in e/Å³
 *   maximumLevel : real   – highest selectable level in e/Å³
 *   decimalValue : real   – the level currently shown, in e/Å³
 *
 * Signals
 * -------
 *   levelEdited(real level)  – emitted only for *user* edits, never when the
 *                              value is pushed in with setLevel(); that is
 *                              what keeps the backend round-trip from looping.
 */

import QtQuick 2.15
import QtQuick.Controls 2.15

SpinBox {
    id: control

    // ── public interface ─────────────────────────────────────────────────
    property real minimumLevel: 0.01
    property real maximumLevel: 9.99
    readonly property real decimalValue: value / 100.0

    signal levelEdited(real level)

    /** Show *level* without emitting levelEdited(). */
    function setLevel(level) {
        var ticks = Math.round(level * 100.0)
        if (ticks !== value)
            value = ticks
    }

    // ── implementation ───────────────────────────────────────────────────
    from: Math.round(minimumLevel * 100.0)
    to: Math.round(maximumLevel * 100.0)
    stepSize: 2               // 0.02 e/Å³, matching DENSITY_LEVEL_STEP
    editable: true

    validator: DoubleValidator {
        bottom: control.minimumLevel
        top: control.maximumLevel
        decimals: 2
        notation: DoubleValidator.StandardNotation
    }

    textFromValue: function(value, locale) {
        return Number(value / 100.0).toLocaleString(locale, 'f', 2)
    }

    valueFromText: function(text, locale) {
        return Math.round(Number.fromLocaleString(locale, text) * 100.0)
    }

    // valueModified fires for user interaction only, not for bindings or
    // programmatic assignment - exactly the distinction we need here.
    onValueModified: control.levelEdited(control.decimalValue)
}
