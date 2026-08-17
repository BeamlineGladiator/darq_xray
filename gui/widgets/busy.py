"""Shared busy-indication vocabulary: overlay spinner, wait cursor, thread pins.

Everything user-visible about "the app is working" lives here (spec
2026-08-17-busy-indication-design.md): :class:`BusyOverlay` (animated
indeterminate spinner or determinate progress over a host widget),
:func:`busy_cursor` (honest wait-cursor for short synchronous blocks),
:func:`keep_alive` (pins running QThreads so they are never garbage-collected
mid-flight) and its shutdown-only counterpart :func:`wait_for_workers` (joins
every pinned worker — the one place the app is allowed to block the GUI
thread on a QThread), and the batch machinery (:class:`BatchWorker`/
:class:`DialogBatchRunner`) used by the replot dialogs to run a per-item
render loop on a worker thread under a cancellable, determinate overlay.
Module-level :data:`_RENDER_LOCK` serializes the actual render/export call
inside :class:`BatchWorker` and (imported from here) `figure_builder`'s
``_ComposeWorker`` — two worker threads must never call into matplotlib at
the same time.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from PySide6.QtCore import QEvent, QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from dfxm.common.eta import EtaEstimator

from ..theme import ThemeController

# Running QThreads pinned here until their finished signal fires — a running
# QThread that gets garbage-collected aborts the process.
_LIVE_WORKERS: set = set()

# BatchWorker and figure_builder._ComposeWorker each run on their own QThread
# and can therefore end up rendering concurrently; matplotlib's mathtext
# parser (and some of the loader-cache machinery) is not thread-safe, so both
# worker classes serialize their actual render/export call through this one
# lock rather than through a per-class lock each.
_RENDER_LOCK = threading.Lock()


def keep_alive(worker) -> None:
    """Pin *worker* (a QThread) until it finishes."""
    _LIVE_WORKERS.add(worker)
    worker.finished.connect(lambda w=worker: _LIVE_WORKERS.discard(w))


def wait_for_workers(timeout_ms: int | None = None) -> None:
    """Block the calling (GUI) thread until every pinned worker has finished.

    Joining a running ``QThread`` from the GUI thread is normally forbidden —
    it freezes the UI and defeats the point of a worker thread — but at
    interpreter/window teardown there is no event loop left to consume a
    worker's results anyway, and a still-running ``QThread`` that gets
    garbage-collected at that point aborts the process
    (``QThread: Destroyed while thread is still running``). This is the one
    sanctioned exception to that rule: call it only from a shutdown path
    (``MainWindow.closeEvent``, defensively after ``app.exec()`` in
    ``gui/app.py``), never from routine UI code.

    *timeout_ms* is passed to each worker's ``wait()`` (``None`` waits
    indefinitely, matching ``QThread.wait()``'s own default). The live-worker
    set is copied before iterating so a worker finishing (and discarding
    itself via its own ``finished`` signal) concurrently with this call never
    mutates the set out from under it.
    """
    for worker in list(_LIVE_WORKERS):
        if timeout_ms is None:
            worker.wait()
        else:
            worker.wait(timeout_ms)


@contextmanager
def busy_cursor(text: str = "", widget=None):
    """Wait-cursor (and optional status text) around a short synchronous block.

    Forces one ``processEvents()`` so the cursor/text actually appear BEFORE
    the block runs; always restores the cursor, including on raise. The status
    text is deliberately left for the call site's completion message to
    overwrite.

    Re-entrancy note: that forced ``processEvents()`` call pumps the Qt event
    queue — it can dispatch a pending paint, a timer, or (if this block is
    nested inside another ``busy_cursor``/dialog) a signal from further up the
    call stack, before this block's body ever runs. Callers must tolerate
    events firing out of order relative to when they were queued and must not
    assume the block executes atomically with respect to the rest of the app.
    """
    app = QApplication.instance()
    if app is not None:
        app.setOverrideCursor(Qt.CursorShape.WaitCursor)
    if widget is not None and text:
        widget.setText(text)
    if app is not None:
        app.processEvents()
    try:
        yield
    finally:
        if app is not None:
            app.restoreOverrideCursor()


class BusyOverlay(QWidget):
    """Translucent, input-swallowing overlay over a host widget.

    Indeterminate mode (``start``): a rotating KIT-green arc painted in
    :meth:`paintEvent`, driven by a 50 ms QTimer, plus a one-line text label.
    Determinate mode (``set_progress``): a progress bar plus a
    ``"{done}/{total} — {eta}"`` sub-label. ``stop()`` in EVERY finish path —
    success, error, cancel, close — is the call sites' contract.
    """

    cancelRequested = Signal()

    def __init__(self, host: QWidget, cancellable: bool = False) -> None:
        super().__init__(host)
        self._host = host
        self._cancellable = cancellable
        self._angle = 0
        self._determinate = False
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._spin)

        self._label = QLabel("", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub = QLabel("", self)
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bar = QProgressBar(self)
        self._bar.setFixedWidth(220)
        self._bar.setTextVisible(False)
        self._bar.hide()
        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.clicked.connect(self.cancelRequested.emit)
        self._cancel_btn.hide()

        lay = QVBoxLayout(self)
        lay.addStretch(2)
        lay.addSpacing(56)  # room for the painted arc above the label
        lay.addWidget(self._label)
        lay.addWidget(self._bar, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self._sub)
        lay.addWidget(self._cancel_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch(3)

        host.installEventFilter(self)
        self.hide()

    @property
    def active(self) -> bool:
        return self.isVisible()

    def start(self, text: str) -> None:
        self._determinate = False
        self._label.setText(text)
        self._sub.setText("")
        self._bar.hide()
        self._cancel_btn.setVisible(self._cancellable)
        self.setGeometry(self._host.rect())
        self.raise_()
        self.show()
        self._timer.start()

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def set_progress(self, done: int, total: int, eta_text: str = "") -> None:
        self._determinate = True
        self._bar.setRange(0, max(1, int(total)))
        self._bar.setValue(int(done))
        self._bar.show()
        self._sub.setText(f"{done}/{total} — {eta_text}" if eta_text else f"{done}/{total}")
        self.update()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    # -- internals --------------------------------------------------------
    def _spin(self) -> None:
        self._angle = (self._angle - 12) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt override
        p = ThemeController.instance().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor(p.surface)
        bg.setAlpha(190)
        painter.fillRect(self.rect(), bg)
        if not self._determinate:
            pen = QPen(QColor(p.accent), 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            r = 18
            cx, cy = self.width() // 2, self.height() // 2 - 44
            painter.drawArc(cx - r, cy - r, 2 * r, 2 * r, self._angle * 16, 100 * 16)
        painter.end()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 — Qt override
        if obj is self._host and event.type() == QEvent.Type.Resize and self.isVisible():
            self.setGeometry(self._host.rect())
        return False

    # swallow interaction with the host while active
    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt override
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt override
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt override
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802 — Qt override
        event.accept()


class BatchWorker(QThread):
    """Per-item batch on a worker thread: ``fn(item) -> list[str]`` per item.

    Emits ``itemDone(done, total)`` after each item and ``batchFinished
    (written, error_text)`` once — error_text "" on success AND on cancel
    (``cancelled`` distinguishes). ``request_stop()`` is cooperative: the
    current item always completes. Exceptions are formatted with a
    StageUserError hint when present and carried as data, with the partial
    ``written`` list preserved (those files are really on disk).
    """

    itemDone = Signal(int, int)
    batchFinished = Signal(list, str)

    def __init__(self, items: list, fn) -> None:
        super().__init__()
        self._items = list(items)
        self._fn = fn
        self._stop = False
        self.written: list[str] = []
        self.cancelled = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # worker thread — no Qt widgets in here
        total = len(self._items)
        err = ""
        try:
            for i, item in enumerate(self._items):
                if self._stop:
                    self.cancelled = True
                    break
                with _RENDER_LOCK:
                    self.written.extend(self._fn(item))
                self.itemDone.emit(i + 1, total)
        except Exception as exc:  # noqa: BLE001 — delivered to the dialog as data
            hint = getattr(exc, "hint", "")
            err = f"{exc} — {hint}" if hint else str(exc)
        self.batchFinished.emit(list(self.written), err)


class DialogBatchRunner(QObject):
    """Owns one batch run for a dialog: cancellable determinate BusyOverlay
    over the dialog, ETA from EtaEstimator, button gating, and GUI-thread
    delivery (it is a QObject parented to the dialog, so the worker's signals
    queue onto the GUI thread). Single-item batches keep the indeterminate
    spinner — a 0/1 progress bar with an ETA is noise."""

    def __init__(self, dialog: QWidget, buttons: tuple) -> None:
        super().__init__(dialog)
        self._buttons = tuple(buttons)
        self._overlay = BusyOverlay(dialog, cancellable=True)
        self._overlay.cancelRequested.connect(self.request_cancel)
        self._eta = EtaEstimator()
        self._worker: BatchWorker | None = None
        self._on_finished_cb = None

    @property
    def running(self) -> bool:
        return self._worker is not None

    def start(self, items: list, fn, on_finished, text: str = "Rendering…") -> None:
        if self._worker is not None:
            return
        self._on_finished_cb = on_finished
        for b in self._buttons:
            b.setEnabled(False)
        self._eta.reset()
        self._overlay.start(text)
        if len(items) > 1:
            self._overlay.set_progress(0, len(items), "")
        worker = BatchWorker(items, fn)
        worker.itemDone.connect(self._on_item_done)  # bound methods -> queued to GUI thread
        worker.batchFinished.connect(self._on_batch_finished)
        self._worker = worker
        keep_alive(worker)
        worker.start()

    def request_cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._overlay.set_text("Cancelling — finishing current item…")

    def _on_item_done(self, done: int, total: int) -> None:
        self._eta.update(done / total)
        if total > 1:
            self._overlay.set_progress(done, total, self._eta.eta_text())

    def _on_batch_finished(self, written: list, error: str) -> None:
        worker, self._worker = self._worker, None
        cancelled = bool(worker.cancelled) if worker is not None else False
        self._overlay.stop()  # stop() in EVERY finish path: success, error, cancel
        for b in self._buttons:
            b.setEnabled(True)
        if self._on_finished_cb is not None:
            self._on_finished_cb(written, error, cancelled)
