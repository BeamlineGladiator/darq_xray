"""The progress helpers and the harness that polices them.

Written after a mutation audit of `test_stage_progress.py` found three
mutations that killed nothing: loosening `MAX_FRAC_GAP`, dropping the
leading-`0.0` term that makes the *first* report's jump count, and deleting the
per-layer report inside `render.save_layer_pngs`. The first is benign — a
weakened threshold cannot fail a test by construction — but the other two were
real gaps: the behaviour existed and nothing checked it. Stage traces are the
wrong instrument for that, because whether a stage happens to exercise a rule
depends on its fixture's size. These check the rules directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfxm.common import progress as P
from tests.progress_trace import MAX_FRAC_GAP, assert_progress_wellformed


# -- sub_progress -------------------------------------------------------------
def test_sub_progress_maps_a_local_fraction_into_its_range():
    seen = []
    report = P.sub_progress(lambda f, t="": seen.append(f), 0.2, 0.5)
    for local in (0.0, 0.5, 1.0):
        report(local)
    assert seen == [0.2, 0.35, 0.5]


def test_sub_progress_reports_exactly_hi_at_the_top():
    """`local == 1.0` must land on exactly `hi`, not an ulp below it.

    The next sibling slot starts at exactly `hi` (see `slice_for`), so an ulp
    short reads as the bar stepping backwards between adjacent items. Asserted
    as bit equality across every slot boundary `slice_for` actually produces,
    rather than on one hand-picked pair — this is a property of the arithmetic,
    and checking one pair would not notice it failing for another.
    """
    for count in (2, 3, 5, 7, 16, 76):
        for lo, hi in ((0.05, 0.99), (0.02, 0.97), (0.1, 0.95), (0.0, 1.0)):
            for i in range(count):
                a, b = P.slice_for(i, count, lo, hi)
                seen: list[float] = []
                P.sub_progress(lambda f, t="", s=seen: s.append(f), a, b)(1.0)
                assert seen == [b], f"count={count} slot={i}: {seen[0]!r} != {b!r}"


def test_sub_progress_clamps_an_overshooting_local_fraction():
    seen = []
    report = P.sub_progress(lambda f, t="": seen.append(f), 0.1, 0.4)
    report(1.7)
    report(-0.5)
    assert seen == [0.4, 0.1]


def test_sub_progress_of_none_is_a_usable_noop():
    report = P.sub_progress(None, 0.0, 1.0)
    assert report(0.5, "anything") is None  # callable, silent, no branch needed upstream


# -- slice_for ----------------------------------------------------------------
def test_slice_for_hands_adjacent_items_exactly_the_same_boundary():
    """Item i's stop must be bit-identical to item i+1's start.

    Computed from `lo` at both ends rather than as `start + span`: the two are
    equal in real arithmetic and differ by an ulp in floating point, which is
    what made `slices` report 0.7071428571428572 then 0.7071428571428571 and
    trip the no-regression rule.
    """
    for count in (2, 3, 7, 16, 76):
        bounds = [P.slice_for(i, count, 0.1, 0.95) for i in range(count)]
        for (_, stop), (start, _) in zip(bounds, bounds[1:]):
            assert stop == start, f"count={count}: {stop!r} != {start!r}"
        assert bounds[0][0] == 0.1
        assert bounds[-1][1] == pytest.approx(0.95)


def test_slice_for_survives_an_empty_work_list():
    assert P.slice_for(0, 0, 0.0, 1.0) == (0.0, 1.0)


# -- the harness itself -------------------------------------------------------
def _ok_trace(n=10):
    return [((i + 1) / n, f"step {i}") for i in range(n)]


def test_harness_accepts_a_smooth_trace():
    assert assert_progress_wellformed(_ok_trace(), label="ok") <= MAX_FRAC_GAP


def test_harness_rejects_a_big_jump_on_the_very_first_report():
    """The leading 0.0 term. A stage silent through its first third is exactly
    as silent as one that goes quiet in the middle, and only comparing
    *consecutive* reports would wave it through."""
    # Long enough to clear the MIN_REPORTS check, so the failure that fires is
    # the gap rule and not the too-few-reports one.
    trace = [(0.5, "half the run, unreported")] + [(0.5 + 0.1 * i, f"s{i}") for i in range(1, 6)]
    with pytest.raises(AssertionError, match="jumps 0.500"):
        assert_progress_wellformed(trace, label="late-starter")


def test_harness_rejects_a_regression():
    trace = [(0.2, "a"), (0.4, "b"), (0.3, "backwards"), (1.0, "d")]
    with pytest.raises(AssertionError, match="went backwards"):
        assert_progress_wellformed(trace, label="regressing")


def test_harness_rejects_a_run_that_never_reaches_one():
    trace = [(0.25, "a"), (0.5, "b"), (0.75, "c"), (0.99, "so close")]
    with pytest.raises(AssertionError, match="final progress is 0.99"):
        assert_progress_wellformed(trace, label="unfinished")


def test_harness_rejects_a_trace_too_short_to_judge():
    with pytest.raises(AssertionError, match="too few to say anything"):
        assert_progress_wellformed([(0.5, "a"), (1.0, "b")], label="terse")


def test_harness_rejects_silence():
    with pytest.raises(AssertionError, match="reported no progress at all"):
        assert_progress_wellformed([], label="mute")


# -- the per-layer report the stages depend on --------------------------------
def test_save_layer_pngs_reports_once_per_layer(tmp_path):
    """Pinned directly, because no stage fixture is big enough to pin it.

    Deleting this report leaves every stage test passing — their volumes have
    few enough layers that the per-product reports around the loop already keep
    the gaps small. On a real volume it is the difference between a bar that
    advances ~78 times per dataset and one that advances four times.
    """
    from dfxm.common import render as Rnd

    n_layers = 5
    vol = np.random.default_rng(0).standard_normal((n_layers, 4, 6))
    seen: list[tuple[float, str]] = []
    Rnd.save_layer_pngs(
        vol,
        np.arange(n_layers, dtype=float),
        str(tmp_path),
        "field",
        -1.0,
        1.0,
        "viridis",
        "t",
        "c",
        0.1,
        0.2,
        progress=lambda f, t="": seen.append((f, t)),
    )
    assert len(seen) == n_layers, f"expected one report per layer, got {seen}"
    assert [f for f, _ in seen] == [(i + 1) / n_layers for i in range(n_layers)]
    # After the write, not before: the last report means every PNG is on disk.
    assert len(list(tmp_path.glob("field_layers/*.png"))) == n_layers
