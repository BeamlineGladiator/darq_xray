"""StageUserError hints must survive the child-process boundary.

Also covers `StageRunner.pid` and the `tests/peak_rss.py` measurement harness
that watches the child that `pid` identifies.
"""

import os
import time

import pytest

from darq_xray.runner import Failed, StageRunner


def _fail_with_hint(params, progress=None):
    from darq_xray.common.errors import StageUserError

    raise StageUserError("maps.h5 not found in /nowhere", hint="Run darfix first.")


def _fail_plain(params, progress=None):
    raise RuntimeError("boom")


def _run_to_failure(fn) -> Failed:
    runner = StageRunner(fn, {}, start_method="fork")
    runner.start()
    t0 = time.time()
    while not runner.finished and time.time() - t0 < 30:
        runner.poll()
        time.sleep(0.02)
    assert runner.failure is not None
    return runner.failure


def test_stage_user_error_attrs():
    from darq_xray.common.errors import StageUserError

    exc = StageUserError("bad input", hint="fix it like so")
    assert isinstance(exc, ValueError)  # existing pytest.raises(ValueError) keep working
    assert str(exc) == "bad input"
    assert exc.hint == "fix it like so"


def test_hint_round_trips_through_runner():
    failure = _run_to_failure(_fail_with_hint)
    assert failure.error == "maps.h5 not found in /nowhere"
    assert failure.hint == "Run darfix first."


def test_plain_exception_has_empty_hint():
    failure = _run_to_failure(_fail_plain)
    assert failure.error == "boom"
    assert failure.hint == ""


def test_stage_runner_exposes_child_pid():
    runner = StageRunner("tests.peak_rss:_sleepy_target", {"seconds": 0.2})
    assert runner.pid is None, "no child before start()"
    runner.start()
    try:
        assert isinstance(runner.pid, int) and runner.pid > 0
        assert runner.pid != os.getpid(), "pid must be the child's, not the parent's"
    finally:
        runner.cancel()


# --- tests/peak_rss.py: the harness must not be able to measure nothing ------
MIB = 1 << 20


def test_peak_rss_sees_a_large_allocation():
    """The harness's own reason to exist: it must observe a real allocation."""
    from tests.peak_rss import measure_peak_rss

    small, baseline = measure_peak_rss("tests.peak_rss:_hungry_target", {"mib": 8})
    big, hungry = measure_peak_rss("tests.peak_rss:_hungry_target", {"mib": 256})
    # Precondition: the target really ran (and really allocated) in both runs,
    # rather than the child dying early and leaving two comparable near-zeros.
    assert small == {"sum": 1.0} and big == {"sum": 1.0}
    assert baseline > MIB, f"baseline {baseline} B is not a real child's RSS"
    assert hungry - baseline > 128 * MIB, (
        f"harness did not observe the allocation: {baseline} -> {hungry}"
    )


def test_peak_rss_raises_when_no_sample_can_be_read(monkeypatch):
    """A sampler that reads nothing must fail loudly, not return 0."""
    from tests import peak_rss

    monkeypatch.setattr(peak_rss, "_sample_rss", lambda proc: None)
    with pytest.raises(RuntimeError, match="measurement is dead"):
        peak_rss.measure_peak_rss("tests.peak_rss:_sleepy_target", {"seconds": 0.2})


def test_peak_rss_raises_on_an_implausible_peak(monkeypatch):
    """A sampler stuck at zero must fail loudly too — 0 is never an answer."""
    from tests import peak_rss

    monkeypatch.setattr(peak_rss, "_sample_rss", lambda proc: 0)
    with pytest.raises(RuntimeError, match="not reading the child"):
        peak_rss.measure_peak_rss("tests.peak_rss:_sleepy_target", {"seconds": 0.2})


def test_assert_peak_under_cannot_pass_on_a_dead_measurement(monkeypatch):
    """The vacuous pass this harness exists to prevent: dead measurement, huge limit."""
    from tests import peak_rss

    monkeypatch.setattr(peak_rss, "_sample_rss", lambda proc: 0)
    with pytest.raises(RuntimeError, match="not reading the child"):
        peak_rss.assert_peak_under("tests.peak_rss:_sleepy_target", {"seconds": 0.2}, 8 << 30)


def test_assert_peak_under_passes_and_catches_an_overrun():
    """Same target, two limits: one it meets, one it cannot."""
    from tests.peak_rss import assert_peak_under

    result = assert_peak_under("tests.peak_rss:_hungry_target", {"mib": 8}, 2 << 30)
    assert result == {"sum": 1.0}
    with pytest.raises(AssertionError, match="over the"):
        assert_peak_under("tests.peak_rss:_hungry_target", {"mib": 256}, 64 * MIB)


def test_assert_peak_under_rejects_an_unmeetable_limit():
    """A limit below a bare interpreter's floor says nothing about the stage."""
    from tests.peak_rss import assert_peak_under

    with pytest.raises(ValueError, match="floor"):
        assert_peak_under("tests.peak_rss:_sleepy_target", {"seconds": 0.05}, 1024)


def test_peak_rss_surfaces_a_stage_failure():
    from tests.peak_rss import measure_peak_rss

    with pytest.raises(RuntimeError, match="boom"):
        measure_peak_rss("tests.peak_rss:_boom_target", {})
