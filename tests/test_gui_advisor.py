"""Profile caching and the debounced advisory worker (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import CostEstimate, Param, ParamType, StageSpec  # noqa: E402
from gui import advisor as A  # noqa: E402

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


def _drain(timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)


@pytest.fixture(autouse=True)
def _clean_advisor_state():
    A.clear_profile_cache()
    yield
    A.clear_profile_cache()


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
    assert "runs in memory" in seen[0].headline


def test_compute_blocking_returns_and_stores_latest():
    adv = A.StageAdvisor(_SPEC, lambda: {}, debounce_ms=50)
    got = adv.compute_blocking()
    assert got.plan is not None
    assert adv.latest is got
