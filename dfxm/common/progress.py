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

    The *absolute* value is clamped to [0, 1] too. Clamping the local fraction
    alone bounds the result to [lo, hi], which is worth nothing when the slot
    itself is wrong: an out-of-range index into :func:`slice_for` used to hand
    back a range starting below zero, and that negative fraction went straight
    to the GUI bar and to `EtaEstimator.update`. For any slot a sane caller
    passes the outer clamp is a no-op, so it costs nothing and ``local == 1.0``
    still lands on exactly ``hi``.

    Returns :func:`noop` when *progress* is None, so callers can pass the result
    straight down without a second None check at every level.
    """
    if progress is None:
        return noop

    def report(frac: float, text: str = "") -> None:
        local = min(1.0, max(0.0, float(frac)))
        progress(min(1.0, max(0.0, lo + (hi - lo) * local)), text)

    return report


def weighted_slices(weights: dict) -> dict:
    """Map ``{key: weight}`` to ``{key: (start, stop)}`` over [0, 1].

    For a stage whose slice is spent on a fixed set of *optional* products —
    layer PNGs, an animation, a 3-D scene, a rotation video — where the user
    decides which of them run. Shares are normalised over the weights actually
    given, so switching a product off shortens the bar rather than leaving a
    hole in it, and the boundary of a product that did not run is never reported
    because the branch that would report it never executes.

    Fixed fractions were wrong in both directions here: a `report(0.6, "layers
    done")` that fired whether or not the layer loop ran jumped the bar by 0.56
    in one step *and* printed a step line asserting work that never happened.

    A weight of 0 means "not running": that key gets a zero-width range at its
    position, which its (unreached) branch would report as a no-op. All-zero
    weights — a dataset with every product off — collapse to zero-width ranges
    rather than dividing by zero.

    Both `visualize._process_dataset` and `rocking._render` allocate their
    products this way; the helper is shared so the two cannot drift.
    """
    total = sum(float(w) for w in weights.values()) or 1.0
    out: dict = {}
    acc = 0.0
    for key, w in weights.items():
        start = acc / total
        acc += float(w)
        out[key] = (start, acc / total)
    return out


def slice_for(index: int, count: int, lo: float, hi: float) -> tuple[float, float]:
    """The ``[start, stop)`` sub-range item *index* of *count* owns within [lo, hi].

    Equal shares, because the honest alternative — weighting each item by its
    real cost — needs timings this project does not have and that vary per
    dataset anyway. Equal shares are wrong in the small (one dataset may take
    twice another) and right in the large (the bar always advances, and
    `EtaEstimator` measures the *recent* rate, so it re-bases itself on the
    current item's pace rather than trusting these weights).

    *count* is clamped to at least 1 so an empty work list cannot divide by
    zero, and *index* to a slot that actually exists: ``slice_for(-1, 0, 0.05,
    0.99)`` used to return ``(-0.89, 0.05)``, a range starting a whole span
    *below* `lo`. Neither clamp repairs the caller's arithmetic — a stage that
    miscounts its items still misallocates the bar — but the damage stays inside
    [lo, hi] instead of surfacing as a negative progress fraction.

    Both ends are computed from *lo* rather than the stop from the start
    (``start + span``): the two are equal in real arithmetic and differ by an
    ulp in floating point, which made item *i*'s stop land just below item
    *i+1*'s start and the bar step backwards between adjacent items.
    """
    count = max(1, int(count))
    index = min(max(0, int(index)), count - 1)
    span = (hi - lo) / count
    return lo + span * index, lo + span * (index + 1)
