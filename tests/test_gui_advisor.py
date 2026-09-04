"""Profile caching and the debounced advisory worker (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import threading  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from darq_xray.config.models import CostEstimate, Param, ParamType, StageSpec  # noqa: E402
from darq_xray.gui import advisor as A  # noqa: E402

GB = 1024**3

_SPEC = StageSpec(
    name="demo",
    label="Demo",
    description="",
    params=(Param("output_dir", ParamType.DIR, "Out"),),
    estimate="tests.test_gui_advisor:_cheap_estimate",
)


def _cheap_estimate(params):
    return CostEstimate(1 * GB, 1 * GB, (10, 100, 100), True)


# A stand-in estimator that blocks until the test releases it, so a test can
# hold one worker genuinely in flight while it fires more `request()` calls —
# the only way to actually exercise the `_pending` latest-wins collapse rather
# than just the debounce-timer restart.
_SLOW_SPEC = StageSpec(
    name="slow",
    label="Slow",
    description="",
    params=(Param("output_dir", ParamType.DIR, "Out"),),
    estimate="tests.test_gui_advisor:_slow_estimate",
)
_slow_calls: list[int] = []
_slow_release = threading.Event()


def _slow_estimate(params):
    _slow_calls.append(1)
    _slow_release.wait(timeout=5.0)
    return CostEstimate(1 * GB, 1 * GB, (10, 100, 100), True)


def _drain(timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _clean_advisor_state():
    A.clear_profile_cache()
    A._set_gl_ready(False)
    A._gl_worker = None
    yield
    A.clear_profile_cache()
    A._set_gl_ready(False)
    A._gl_worker = None


def test_profile_is_cached_within_the_ttl(monkeypatch):
    calls = []
    real = A.machine.profile
    monkeypatch.setattr(
        A.machine,
        "profile",
        lambda **kw: (calls.append(kw), real(**kw))[1],
    )
    A.clear_profile_cache()
    A.cached_profile(os.getcwd())
    A.cached_profile(os.getcwd())
    assert len(calls) == 1


def test_cache_is_per_directory(monkeypatch, tmp_path):
    calls = []
    real = A.machine.profile
    monkeypatch.setattr(
        A.machine,
        "profile",
        lambda **kw: (calls.append(kw), real(**kw))[1],
    )
    A.clear_profile_cache()
    A.cached_profile(os.getcwd())
    A.cached_profile(str(tmp_path))
    assert len(calls) == 2


def test_cached_profile_never_probes_gl(monkeypatch):
    monkeypatch.setattr(
        A.machine,
        "probe_gl",
        lambda **kw: pytest.fail("cached_profile must never probe GL"),
    )
    A.clear_profile_cache()
    prof = A.cached_profile(os.getcwd())
    assert prof.gl_status == "unprobed"


def test_request_debounces_and_emits_once():
    seen = []
    adv = A.StageAdvisor(_SPEC, lambda: {}, debounce_ms=50)
    adv.advisoryReady.connect(seen.append)
    for _ in range(5):
        adv.request()
    _drain(5.0)
    assert len(seen) == 1
    assert "expected to run in memory" in seen[0].headline


def test_request_collapses_requests_made_while_a_worker_is_in_flight():
    """The `_pending` re-run path, not just the debounce-timer restart.

    `test_request_debounces_and_emits_once` fires all its `request()` calls
    before the timer ever expires, so every call just restarts the same
    single-shot timer — `_pending` is never touched and a deleted
    `if self._worker is not None: ...` guard in `_start()` would not fail it.
    This test holds a worker genuinely running (blocked inside the estimator)
    and fires requests while it is in flight, so it is `_pending`'s collapse
    — one more run after the current one lands, not one per request — that
    is actually under test.
    """
    _slow_calls.clear()
    _slow_release.clear()
    seen = []
    adv = A.StageAdvisor(_SLOW_SPEC, lambda: {}, debounce_ms=10)
    adv.advisoryReady.connect(seen.append)

    adv.request()
    deadline = time.monotonic() + 5.0
    while not _slow_calls and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)
    assert _slow_calls, "the first worker never started"
    # Precondition: a worker is genuinely running right now, blocked on the
    # release event — without this, the test below would be meaningless.
    assert adv._worker is not None

    for _ in range(5):
        adv.request()
        _drain(0.2)  # let each debounce timer expire while the worker is still blocked

    # None of those five requests may have started a second worker — the
    # worker is still blocked on _slow_release, so any new estimator call
    # would already show up here.
    assert len(_slow_calls) == 1
    assert adv._pending is True

    _slow_release.set()  # let the in-flight worker (and its one collapsed re-run) finish
    deadline = time.monotonic() + 5.0
    while (len(_slow_calls) < 2 or len(seen) < 2) and time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)

    # Exactly ONE further worker ran after the five collapsed requests — not five.
    assert len(_slow_calls) == 2
    assert len(seen) == 2


def test_compute_blocking_returns_and_stores_latest():
    adv = A.StageAdvisor(_SPEC, lambda: {}, debounce_ms=50)
    got = adv.compute_blocking()
    assert got.plan is not None
    assert adv.latest is got


def test_gl_is_not_probed_until_asked(monkeypatch):
    monkeypatch.setattr(
        A.machine,
        "probe_gl",
        lambda **kw: pytest.fail("GL must not be probed by the cost path"),
    )
    A.clear_profile_cache()
    A._set_gl_ready(False)
    A.cached_profile(os.getcwd())


def test_once_probed_the_cached_profile_carries_gl(monkeypatch):
    from darq_xray.common.machine import GLInfo

    monkeypatch.setattr(
        A.machine,
        "probe_gl",
        lambda **kw: (GLInfo("llvmpipe", "Mesa", "4.5", 2048, True), "ok"),
    )
    A.clear_profile_cache()
    A._set_gl_ready(False)
    A.probe_gl_async()
    _drain(10.0)
    assert A.gl_ready() is True
    A.clear_profile_cache()
    assert A.cached_profile(os.getcwd()).gl is not None


def test_probe_gl_async_uses_the_bounded_timeout_not_probe_gls_120s_default(monkeypatch):
    """M9: `_GlProbeWorker` is pinned via `keep_alive` and joined with no
    timeout of its own by `MainWindow.closeEvent`'s `wait_for_workers()`, so a
    hanging driver would otherwise make closing the window look frozen for up
    to `machine.probe_gl`'s full 120 s default."""
    from darq_xray.common.machine import GLInfo

    seen_kwargs: dict = {}

    def _fake_probe_gl(**kw):
        seen_kwargs.update(kw)
        return GLInfo("llvmpipe", "Mesa", "4.5", 2048, True), "ok"

    monkeypatch.setattr(A.machine, "probe_gl", _fake_probe_gl)
    A.clear_profile_cache()
    A._set_gl_ready(False)
    A.probe_gl_async()
    _drain(10.0)
    # Precondition: the constant this worker is expected to pass really is
    # shorter than probe_gl's own default, or the assertion below would be
    # trivially satisfied by any timeout at all.
    assert A.GL_PROBE_TIMEOUT_S < 120.0
    assert seen_kwargs.get("timeout") == A.GL_PROBE_TIMEOUT_S
