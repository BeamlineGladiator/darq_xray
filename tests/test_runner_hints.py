"""StageUserError hints must survive the child-process boundary."""

import time

from dfxm.runner import Failed, StageRunner


def _fail_with_hint(params, progress=None):
    from dfxm.common.errors import StageUserError

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
    from dfxm.common.errors import StageUserError

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
