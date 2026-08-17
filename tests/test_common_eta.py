"""Qt-free ETA helpers (dfxm/common/eta.py)."""

from dfxm.common.eta import EtaEstimator, format_eta


def test_format_eta_quiet_below_thresholds():
    assert format_eta(1.0, 0.5) == ""  # < 2 s elapsed
    assert format_eta(10.0, 0.01) == ""  # < 5 % done
    assert format_eta(10.0, 1.0) == ""  # finished — nothing left to estimate


def test_format_eta_seconds_and_minutes():
    assert format_eta(10.0, 0.5) == "~10 s left"
    assert format_eta(60.0, 0.25) == "~3 min left"  # 180 s remaining -> minutes


def test_estimator_estimates_and_smooths():
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    assert est.eta_text() == ""
    t[0] = 10.0
    est.update(0.5)
    assert est.eta_text() == "~10 s left"
    t[0] = 12.0
    est.update(0.6)  # raw remaining = 8 s; EMA(0.7*10 + 0.3*8) = 9.4 -> ~9 s
    assert est.eta_text() == "~9 s left"


def test_estimator_ignores_regressing_frac():
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.5)
    before = est.eta_text()
    t[0] = 11.0
    est.update(0.4)  # monotonic clamp: a regression never poisons the estimate
    assert est.eta_text() == before


def test_estimator_reset_forgets_everything():
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.5)
    assert est.eta_text() != ""
    est.reset()
    assert est.eta_text() == ""
