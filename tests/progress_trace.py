"""Record a stage's `progress()` reports and assert they are well-formed.

**Not a pytest file** — no `test_` prefix; imported by `test_stage_progress.py`,
the same arrangement as `tests/peak_rss.py`.

Why this exists. Every stage takes `progress(frac, text)` and the GUI turns the
stream of `frac` values into a progress bar and, via
`darq_xray.common.eta.EtaEstimator`, a "~N left" readout. Stages used to assign
`frac` by **milestone** rather than by work — `visualize` reported `0.6` and
then said nothing at all for the whole strain half of the run, `paraview`
emitted four fractions for an entire 17 GB export — and a bar that stands still
for 40 % of the wall-clock cannot be extrapolated by any estimator. The
estimator was fixed separately (it now measures rate over a trailing window, so
phase-to-phase *cost* differences no longer matter); what it still needs from a
stage is that `frac` keeps **moving**.

That is the invariant here, and it is checkable on a synthetic fixture without
timing anything: the largest jump between consecutive reports is a proxy for the
longest unreported stretch of work, measured in units of the stage's own idea of
its progress. `MAX_FRAC_GAP` is therefore a ceiling on silence.

The gap from `0.0` to the *first* report counts. A stage that does a third of
its work before saying anything is exactly as silent as one that goes quiet in
the middle, and only counting gaps between reports would miss it.
"""

from __future__ import annotations

from typing import Callable

# No stage may advance its own progress by more than this in one step, counting
# the implicit step from 0.0 to its first report.
#
# Not a round number picked for looks: before the sweep, `visualize` jumped 0.40
# (0.6 -> 1.0) and `paraview` jumped 0.45 (0.10 -> 0.55) and 0.43 (0.55 -> 0.98),
# while the best-behaved stages already reported per item. 0.15 is comfortably
# under every pre-sweep offender and comfortably over what a per-item loop
# produces once the long inner operations report too.
#
# **Tighten this, never loosen it.** Raising it to admit a stage converts the
# one check that would have caught that stage's silence into a rubber stamp; the
# fix is to make the stage report, which is always possible — every long
# operation here is a loop over blocks, layers, volumes or jobs.
MAX_FRAC_GAP = 0.15

# A run that reports twice cannot demonstrate anything about smoothness, so a
# trace shorter than this is treated as a broken fixture rather than a pass.
MIN_REPORTS = 4


def trace(run_fn: Callable, params: dict) -> list[tuple[float, str]]:
    """Run *run_fn* in-process, returning its `(frac, text)` reports in order.

    In-process rather than through `StageRunner`: this asserts on what the
    stage *reports*, which the child-process hop only relays, and staying in
    one process keeps the whole sweep to seconds.
    """
    seen: list[tuple[float, str]] = []

    def record(frac, text=""):
        seen.append((float(frac), str(text)))

    run_fn(params, record)
    return seen


def assert_progress_wellformed(
    seen: list[tuple[float, str]],
    *,
    label: str,
    max_gap: float = MAX_FRAC_GAP,
    min_reports: int = MIN_REPORTS,
) -> float:
    """Assert *seen* is a usable progress stream. Returns the largest gap.

    Four properties, each a real failure mode met in this codebase:

    * **It reports at all**, and more than a handful of times.
    * **It ends at exactly 1.0** — the GUI clears the bar on the terminal
      message, and a run finishing at 0.98 leaves a bar that never fills.
    * **It never goes backwards.** `EtaEstimator.update` ignores a regressing
      fraction, so a stage that regresses silently loses those samples.
    * **It never jumps more than *max_gap***, the silence ceiling above.
    """
    assert seen, f"{label}: the run reported no progress at all"
    fracs = [f for f, _ in seen]
    assert len(seen) >= min_reports, (
        f"{label}: only {len(seen)} progress report(s) — too few to say anything "
        f"about smoothness: {seen}"
    )
    assert fracs[-1] == 1.0, f"{label}: final progress is {fracs[-1]}, not 1.0"
    assert all(0.0 <= f <= 1.0 for f in fracs), f"{label}: fraction outside [0, 1]: {fracs}"

    for i, (a, b) in enumerate(zip(fracs, fracs[1:])):
        assert b >= a, (
            f"{label}: progress went backwards at report {i + 1}, {a} -> {b} "
            f"({seen[i][1]!r} -> {seen[i + 1][1]!r})"
        )

    # The leading 0.0 makes the first report's own jump count.
    gaps = [b - a for a, b in zip([0.0] + fracs, fracs)]
    worst = max(gaps)
    where = gaps.index(worst)
    assert worst <= max_gap, (
        f"{label}: progress jumps {worst:.3f} in one step (limit {max_gap}), from "
        f"{([0.0] + fracs)[where]} to {fracs[where]} at {seen[where][1]!r} — that "
        f"stretch of the run reports nothing. Full trace: {[round(f, 3) for f in fracs]}"
    )
    return worst
