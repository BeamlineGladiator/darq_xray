"""Qt-free ETA helpers (darq_xray/common/eta.py)."""

from darq_xray.common.eta import EtaEstimator, format_eta


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
    """A regressing sample changes nothing; only the clock moves the readout.

    Compared against an estimator fed the same clock but no regression, so the
    assertion separates the two effects: any difference between them is the
    regression poisoning the estimate, while the drop both share is the
    wall-clock countdown `eta_text()` now applies.
    """
    t = [0.0]
    poisoned = EtaEstimator(clock=lambda: t[0])
    clean = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    poisoned.update(0.5)
    clean.update(0.5)
    t[0] = 11.0
    poisoned.update(0.4)  # monotonic clamp: a regression never poisons the estimate
    assert poisoned.eta_text() == clean.eta_text() == "~9 s left"


def test_estimator_reset_forgets_everything():
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.5)
    assert est.eta_text() != ""
    est.reset()
    assert est.eta_text() == ""


# -- what a real stage's progress trace does to the estimator ------------------
# Three defects met on a real run, two of them in here and pinned below:
#
#  * A stage that re-emits the SAME fraction (visualize reports `progress(0.6,
#    ...)` repeatedly while the whole strain half grinds away) made
#    `elapsed * (1 - frac) / frac` grow with every sample, so the ETA counted
#    UP.
#  * Between samples the text was frozen: `eta_text()` replayed the last
#    computed number no matter how much wall-clock had passed, so a stage that
#    reports nothing for four minutes showed a stale "~30 s left" throughout.
#  * The rate was the whole-run average, so a run that changes pace late (a
#    fast per-layer loop followed by one slow whole-volume alignment) kept
#    quoting the fast phase's rate.
#
# The third root cause is stage-side and NOT fixable here: `frac` is assigned by
# milestone rather than by work, so no estimator can make a signal that sits at
# 0.6 for 40 % of the run accurate.


def _seconds(text: str) -> float:
    """ "~7 s left" / "~3 min left" -> seconds. "" -> -1 (no claim made)."""
    if not text:
        return -1.0
    n = float(text.strip("~").split()[0])
    return n * 60.0 if "min" in text else n


def test_a_stalled_fraction_never_inflates_the_estimate():
    """Re-reporting the same frac is not progress and must not grow the ETA."""
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.6)
    first = _seconds(est.eta_text())
    assert first > 0
    for now in (20.0, 40.0, 80.0):
        t[0] = now
        est.update(0.6)  # same frac, more elapsed — no new information
        assert _seconds(est.eta_text()) <= first, (
            f"ETA grew to {est.eta_text()!r} at t={now} while frac stood still"
        )


def test_the_estimate_counts_down_between_samples():
    """Wall-clock passing is information even when the stage says nothing."""
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.5)
    assert est.eta_text() == "~10 s left"
    t[0] = 15.0  # five silent seconds
    assert est.eta_text() == "~5 s left"


def test_the_estimate_stops_claiming_once_it_is_overrun():
    """Past its own estimate the honest answer is no number, not "~1 s left"."""
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.5)
    t[0] = 25.0  # 15 s of silence against a 10 s estimate
    assert est.eta_text() == ""


def test_a_late_slowdown_re_bases_the_estimate():
    """The rate comes from the recent window, not the whole run.

    Half the work goes by in 10 s, then the stage slows ~30x. At t=70 the
    whole-run average says ~47 s remain; the true figure at the current pace is
    240 s. The windowed estimate must land nearer the truth.
    """
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.5)
    for i in range(1, 7):  # 0.5 -> 0.6 over the next 60 s
        t[0] = 10.0 + 10.0 * i
        est.update(0.5 + 0.1 * i / 6.0)
    assert _seconds(est.eta_text()) > 100.0, (
        f"still quoting the fast phase: {est.eta_text()!r} (whole-run average says ~47 s)"
    )
