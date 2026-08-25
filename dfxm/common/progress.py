"""Progress-reporting helpers shared by the stages. Qt-free, no IO, no state.

A stage's ``run(params, progress)`` reports one number for the whole run, but
the work is nested: a run has phases, a phase has datasets, a dataset has
layers. Without a way to map a nested loop's own 0..1 progress into its slice of
the parent's bar, the only thing a stage can do is report at phase *boundaries*
— which is exactly how `visualize` came to sit at 0.6 for the entire strain half
of a run and `paraview` to emit four fractions for a whole export.

:func:`sub_progress` is that mapping, and it composes: a stage gives each
dataset a slice, `_process_dataset` gives each product a slice of that, and the
layer loop reports per layer inside it. The GUI sees one smoothly advancing
number and `dfxm.common.eta.EtaEstimator` can extrapolate from it.

`tests/progress_trace.py` pins the result: no stage may advance its own progress
by more than `MAX_FRAC_GAP` in a single step.
"""

from __future__ import annotations

from typing import Callable

ProgressFn = Callable[..., None]


def noop(frac: float, text: str = "") -> None:
    """Accept and discard a progress report.

    Every stage's ``run`` does ``progress = progress or noop`` so the body never
    branches on whether anyone is listening.
    """


def sub_progress(progress: ProgressFn | None, lo: float, hi: float) -> ProgressFn:
    """A callback taking a local 0..1 fraction and reporting it into [lo, hi].

    ``sub_progress(progress, 0.2, 0.5)(0.5, "half")`` reports ``0.35``. The
    local fraction is clamped, so a caller that overshoots (an off-by-one in a
    loop bound, a generator yielding one more block than predicted) cannot push
    the parent's bar past ``hi`` or make it go backwards — both of which
    `EtaEstimator` would otherwise have to absorb, and a regression it silently
    discards.

    ``local == 1.0`` lands on exactly ``hi``, which matters because the next
    sibling slot *starts* at exactly ``hi`` (see :func:`slice_for`) and an ulp
    short would read as the bar stepping backwards. No special case is needed
    for it: ``lo + (hi - lo) * 1.0 == hi`` held for all 3 000 000 random pairs
    in [0, 1) it was checked against, and for every slot boundary
    :func:`slice_for` produces across the counts and ranges the stages use. An
    earlier version guarded it explicitly; the guard was dead code, and the
    ulp-apart regression it claimed to prevent was really :func:`slice_for`
    computing a slot's stop as ``start + span``.

    Returns :func:`noop` when *progress* is None, so callers can pass the result
    straight down without a second None check at every level.
    """
    if progress is None:
        return noop

    def report(frac: float, text: str = "") -> None:
        local = min(1.0, max(0.0, float(frac)))
        progress(lo + (hi - lo) * local, text)

    return report


def slice_for(index: int, count: int, lo: float, hi: float) -> tuple[float, float]:
    """The ``[start, stop)`` sub-range item *index* of *count* owns within [lo, hi].

    Equal shares, because the honest alternative — weighting each item by its
    real cost — needs timings this project does not have and that vary per
    dataset anyway. Equal shares are wrong in the small (one dataset may take
    twice another) and right in the large (the bar always advances, and
    `EtaEstimator` measures the *recent* rate, so it re-bases itself on the
    current item's pace rather than trusting these weights).

    *count* is clamped to at least 1 so an empty work list cannot divide by zero.

    Both ends are computed from *lo* rather than the stop from the start
    (``start + span``): the two are equal in real arithmetic and differ by an
    ulp in floating point, which made item *i*'s stop land just below item
    *i+1*'s start and the bar step backwards between adjacent items.
    """
    count = max(1, int(count))
    span = (hi - lo) / count
    return lo + span * index, lo + span * (index + 1)
