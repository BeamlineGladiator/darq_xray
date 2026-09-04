"""Shared Qt test helpers, importable by pytest files AND tests/gui_smoke.py.

Not a test module (no ``test_`` prefix). gui_smoke.py puts the repo root on
sys.path, so both worlds import this as ``tests.qt_helpers``.
"""

from __future__ import annotations

import contextlib
import time

from PySide6.QtWidgets import QApplication


@contextlib.contextmanager
def applied_theme(app: QApplication, mode: str):
    """Apply a theme for the duration of a test, then put the app back.

    `darq_xray.gui.theme.apply_theme` changes three process-wide things — the style, the
    palette and the stylesheet — and pytest shares one QApplication across every
    Qt module in the run. A test that restored only `app.styleSheet()` left
    every later widget under Fusion with the themed palette but the *old*
    stylesheet, a combination no production path produces; anything measuring
    geometry, colour or `style()` behaviour afterwards then depends on
    collection order, and surfaces as an unrelated test flaking.

    The style is restored by name rather than by handing the previous QStyle
    object back: `QApplication.setStyle` takes ownership and deletes what it
    replaces, so the saved pointer is not safe to reuse.
    """
    from darq_xray.gui.theme import apply_theme

    style_name = app.style().objectName()
    palette = app.palette()
    sheet = app.styleSheet()
    try:
        yield apply_theme(app, mode)
    finally:
        app.setStyleSheet(sheet)
        if style_name:
            app.setStyle(style_name)
        app.setPalette(palette)


def wait_builder_idle(w, timeout_s: float = 30.0) -> None:
    """Drive the event loop until the figure-builder window has no live or
    pending compose worker."""
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_s
    while w._worker is not None or w._pending_render or w._pending_export is not None:
        assert time.monotonic() < deadline, "compose worker did not finish in time"
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()  # flush any just-queued result delivery


def render_and_wait(w, timeout_s: float = 30.0):
    """Async twin of the old synchronous ``render_now()`` contract: request a
    render, wait for it, return the ComposeResult of THIS request (None on
    error or nothing-to-render — the notes bar carries the explanation)."""
    w.render_now()
    wait_builder_idle(w, timeout_s)
    return w._last_outcome


def export_and_wait(w, timeout_s: float = 30.0):
    """Same, for ``export_now()`` (async from the busy-indication Task 4 on)."""
    w.export_now()
    wait_builder_idle(w, timeout_s)
    return w._last_outcome


def wait_batch_idle(dialog, timeout_s: float = 60.0) -> None:
    """Drive the event loop until *dialog*._batch (a DialogBatchRunner) is idle."""
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_s
    while dialog._batch.running:
        assert time.monotonic() < deadline, "replot batch did not finish in time"
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
