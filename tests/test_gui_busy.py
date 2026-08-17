"""BusyOverlay / busy_cursor / keep_alive (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui.widgets.busy import BusyOverlay, busy_cursor  # noqa: E402

_hosts: list = []


def _host():
    # Kept in _hosts and disposed deterministically in the autouse fixture
    # below (rather than left for Python's GC): BusyOverlay's Cancel button
    # connects a bound method of itself (`self.cancelRequested.emit`),
    # forming a reference cycle that only the cyclic collector breaks — if
    # that collection happens to run during a later test's processEvents(),
    # Qt can dispatch a queued event to an object mid-teardown. Deleting each
    # host (and its overlay child) right after its own test sidesteps that
    # race entirely.
    host = QWidget()
    host.resize(300, 200)
    host.show()
    _hosts.append(host)
    return host


@pytest.fixture(autouse=True)
def _cleanup_hosts():
    yield
    for h in _hosts:
        h.deleteLater()
    _hosts.clear()
    _app.processEvents()


def test_overlay_start_stop_visibility_and_text():
    host = _host()
    ov = BusyOverlay(host)
    assert not ov.active
    ov.start("Rendering…")
    assert ov.active and ov._label.text() == "Rendering…"
    assert ov.geometry() == host.rect()
    ov.stop()
    assert not ov.active and not ov._timer.isActive()


def test_overlay_determinate_progress_line():
    ov = BusyOverlay(_host())
    ov.start("Rendering…")
    ov.set_progress(2, 5, "~10 s left")
    assert ov._bar.value() == 2 and ov._bar.maximum() == 5
    assert ov._sub.text() == "2/5 — ~10 s left"
    ov.set_progress(3, 5)
    assert ov._sub.text() == "3/5"
    ov.stop()


def test_overlay_cancel_button_only_when_cancellable():
    ov = BusyOverlay(_host())
    ov.start("x")
    assert not ov._cancel_btn.isVisible()
    ov.stop()
    ov2 = BusyOverlay(_host(), cancellable=True)
    hits = []
    ov2.cancelRequested.connect(lambda: hits.append(1))
    ov2.start("x")
    assert ov2._cancel_btn.isVisible()
    ov2._cancel_btn.click()
    assert hits == [1]
    ov2.stop()


def test_overlay_tracks_host_resize():
    host = _host()
    ov = BusyOverlay(host)
    ov.start("x")
    host.resize(500, 400)
    _app.processEvents()
    assert ov.geometry() == host.rect()
    ov.stop()


def test_busy_cursor_sets_and_restores():
    assert _app.overrideCursor() is None
    with busy_cursor():
        assert _app.overrideCursor() is not None
    assert _app.overrideCursor() is None


def test_busy_cursor_restores_on_exception_and_writes_text():
    from PySide6.QtWidgets import QLabel

    label = QLabel("")
    with pytest.raises(RuntimeError):
        with busy_cursor("loading…", widget=label):
            assert label.text() == "loading…"
            raise RuntimeError("boom")
    assert _app.overrideCursor() is None


def _run_batch(worker):
    worker.start()
    assert worker.wait(20000)
    _app.processEvents()


def test_batch_worker_runs_items_and_reports():
    from gui.widgets.busy import BatchWorker

    ticks, finishes = [], []
    w = BatchWorker([1, 2, 3], lambda i: [f"p{i}"])
    w.itemDone.connect(lambda d, t: ticks.append((d, t)))
    w.batchFinished.connect(lambda paths, err: finishes.append((paths, err)))
    _run_batch(w)
    assert ticks == [(1, 3), (2, 3), (3, 3)]
    assert finishes == [(["p1", "p2", "p3"], "")]
    assert not w.cancelled


def test_batch_worker_cancel_stops_after_current_item():
    from gui.widgets.busy import BatchWorker

    seen: list[int] = []
    holder: list = []

    def fn(i):
        seen.append(i)
        holder[0].request_stop()
        return [f"p{i}"]

    w = BatchWorker([1, 2, 3], fn)
    holder.append(w)
    results: list = []
    w.batchFinished.connect(lambda paths, err: results.append((paths, err)))
    _run_batch(w)
    assert seen == [1] and w.cancelled
    assert results == [(["p1"], "")]


def test_batch_worker_error_carries_partial_written_and_hint():
    from dfxm.common.errors import StageUserError
    from gui.widgets.busy import BatchWorker

    def fn(i):
        if i == 2:
            raise StageUserError("bad item", hint="fix it")
        return [f"p{i}"]

    results: list = []
    w = BatchWorker([1, 2, 3], fn)
    w.batchFinished.connect(lambda paths, err: results.append((paths, err)))
    _run_batch(w)
    assert results == [(["p1"], "bad item — fix it")]


def test_dialog_batch_runner_overlay_buttons_and_finish():
    from PySide6.QtWidgets import QPushButton

    from gui.widgets.busy import DialogBatchRunner

    host = _host()
    btn = QPushButton("Render", host)
    runner = DialogBatchRunner(host, (btn,))
    done: list = []
    runner.start([1, 2], lambda i: [f"p{i}"], lambda w, e, c: done.append((w, e, c)))
    assert runner.running and runner._overlay.active and not btn.isEnabled()
    import time as _t

    deadline = _t.monotonic() + 20
    while runner.running:
        assert _t.monotonic() < deadline
        _app.processEvents()
        _t.sleep(0.01)
    _app.processEvents()
    assert done == [(["p1", "p2"], "", False)]
    assert not runner._overlay.active and btn.isEnabled()
