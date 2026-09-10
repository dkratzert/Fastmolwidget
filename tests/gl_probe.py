"""Crash-safe detection of a usable OpenGL context for the 3-D tests.

``MoleculeWidget3D`` can only be *shown* on a machine with a working GL
stack.  On headless CI (``QT_QPA_PLATFORM=offscreen`` plus Mesa software
rendering) the sequence the framebuffer tests perform — show the widget,
pump the event loop, read the framebuffer back — can abort the whole
process with ``SIGABRT`` from inside Qt/Mesa: no Qt warning, no Python
exception, nothing that could be caught here.

The flags ``molecule3D._HAS_PYOPENGL`` / ``._IS_GL_WIDGET`` only say that
the *imports* worked, and ``MoleculeWidget3D._gl_failed`` is set too late
(the process is already gone), so neither can prevent that.  The risky
sequence is therefore executed once in a **separate process**; if that
process dies, the tests needing a live framebuffer are skipped instead of
taking the whole test run down with them.

The probe deliberately shows *several* widgets and keeps them alive: the
observed CI abort happened on the third live ``QOpenGLWidget`` of the
module, not on the first.
"""

from __future__ import annotations

import subprocess
import sys
from functools import cache

import pytest

#: Number of live GL widgets the probe creates.  At least as many as
#: ``test_viewer_widget3D.py`` keeps alive at once.
_PROBE_WIDGETS = 3

_PROBE_SOURCE = f"""
import sys
import tempfile
from pathlib import Path

from qtpy import QtWidgets

app = QtWidgets.QApplication([])

import fastmolwidget.molecule3D as m3
from fastmolwidget.sdm import Atomtuple

if not m3._HAS_PYOPENGL or not m3._IS_GL_WIDGET:
    sys.exit(2)

atoms = [
    Atomtuple("C1", "C", 0.0, 0.0, 0.0, 0),
    Atomtuple("O1", "O", 1.5, 0.0, 0.0, 0),
]

# Kept alive on purpose: the abort this probe guards against only showed
# up once several QOpenGLWidgets existed at the same time.
widgets = []
for _ in range({_PROBE_WIDGETS}):
    widget = m3.MoleculeWidget3D()
    widgets.append(widget)
    widget.resize(400, 300)
    widget.show()
    app.processEvents()

    if widget._gl_failed:
        sys.exit(3)
    if not hasattr(widget, "grabFramebuffer"):
        sys.exit(4)

    widget.open_molecule(atoms)
    app.processEvents()

    for labels in (False, True):
        widget.show_labels(labels)
        widget.update()
        app.processEvents()
        image = widget.grabFramebuffer()
        if image is None or image.isNull():
            sys.exit(5)

with tempfile.TemporaryDirectory() as tmp:
    widgets[-1].save_image(Path(tmp) / "probe.png", image_scale=1.0)

for widget in widgets:
    widget.close()
app.processEvents()
print("GL_OK")
"""

_PROBE_TIMEOUT = 180.0


@cache
def real_gl_available() -> bool:
    """Return ``True`` when live ``MoleculeWidget3D`` framebuffers work.

    The result is cached for the whole test session, so the subprocess is
    started at most once.
    """
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _PROBE_SOURCE],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and "GL_OK" in completed.stdout


def skip_without_real_gl() -> None:
    """Skip the calling test unless a live GL framebuffer is available."""
    if not real_gl_available():
        pytest.skip(
            "requires a real OpenGL context (the probe subprocess could not "
            "show and read back MoleculeWidget3D framebuffers in this "
            "environment)"
        )
