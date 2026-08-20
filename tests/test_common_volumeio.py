"""Bounded-memory volume IO (dfxm/common/volumeio.py)."""

from __future__ import annotations

import tracemalloc

import h5py
import numpy as np
import pytest

from dfxm.common import volumeio


@pytest.fixture
def volume(tmp_path):
    """A (7, 5, 3) float32 volume with distinct, exactly-representable values."""
    data = np.arange(7 * 5 * 3, dtype=np.float32).reshape(7, 5, 3)
    path = tmp_path / "vol.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("vol", data=data)
    return str(path), data


def test_volume_bytes_is_shape_times_itemsize(volume):
    path, data = volume
    with h5py.File(path, "r") as f:
        assert volumeio.volume_bytes(f["vol"]) == data.nbytes


def test_iter_blocks_covers_every_element_exactly_once(volume):
    path, data = volume
    with h5py.File(path, "r") as f:
        seen = np.zeros(data.shape[0], dtype=int)
        for sl, block in volumeio.iter_blocks(f["vol"], budget_bytes=data.nbytes // 3):
            seen[sl] += 1
            assert np.array_equal(block, data[sl])
    assert (seen == 1).all()


@pytest.mark.parametrize("divisor", [1, 2, 3, 5, 100])
def test_iter_blocks_reassembles_the_original_for_any_budget(volume, divisor):
    path, data = volume
    with h5py.File(path, "r") as f:
        blocks = [b for _, b in volumeio.iter_blocks(f["vol"], budget_bytes=data.nbytes // divisor)]
    assert np.array_equal(np.concatenate(blocks, axis=0), data)


def test_iter_blocks_always_yields_at_least_one_layer(volume):
    """A budget smaller than a single layer must still make progress, not hang."""
    path, data = volume
    with h5py.File(path, "r") as f:
        blocks = list(volumeio.iter_blocks(f["vol"], budget_bytes=1))
    assert len(blocks) == data.shape[0]
    assert all(b.shape[0] == 1 for _, b in blocks)


def test_load_or_stream_returns_an_array_when_it_fits(volume):
    path, data = volume
    with h5py.File(path, "r") as f:
        result = volumeio.load_or_stream(f["vol"], budget_bytes=data.nbytes * 10)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, data)


def test_load_or_stream_returns_a_reader_when_it_does_not_fit(volume):
    path, data = volume
    with h5py.File(path, "r") as f:
        result = volumeio.load_or_stream(f["vol"], budget_bytes=data.nbytes // 4)
        assert isinstance(result, volumeio.BlockReader)
        assert result.shape == data.shape
        assert np.array_equal(np.concatenate([b for _, b in result], axis=0), data)


@pytest.fixture
def wide_volume(tmp_path):
    """Values spanning many magnitudes — where naive summation loses bits.

    Exponent range widened from the plan's (-6, 7) to (-15, 16): at the
    narrower range a naive per-block ``np.nansum`` summed in a different
    order per budget still landed on the same float64 bits every time (see
    task-12-report.md), so the negative harness test never actually
    exercised the failure path it exists to catch. (-15, 16) spans enough
    magnitudes that catastrophic cancellation differs across block orderings.
    """
    rng = np.random.default_rng(20260820)
    data = (rng.standard_normal((13, 9, 7)) * 10.0 ** rng.integers(-15, 16, (13, 9, 7))).astype(
        np.float64
    )
    path = tmp_path / "wide.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("vol", data=data)
    return str(path), data


def test_neumaier_sum_continues_across_calls():
    values = np.array([1e16, 1.0, -1e16, 1.0])
    whole = volumeio.neumaier_sum(values)
    state = volumeio.neumaier_sum(values[:2])
    part = volumeio.neumaier_sum(values[2:], state=state)
    assert whole == part
    assert whole[0] + whole[1] == 2.0  # the naive result would be 0.0


@pytest.mark.parametrize("divisor", [1, 2, 3, 7, 13, 1000])
def test_block_nansum_is_bit_identical_across_budgets(wide_volume, divisor):
    """The core guarantee: the memory budget must not change the answer."""
    path, data = wide_volume
    with h5py.File(path, "r") as f:
        reference = volumeio.block_nansum(f["vol"], budget_bytes=data.nbytes * 10)
        result = volumeio.block_nansum(f["vol"], budget_bytes=max(1, data.nbytes // divisor))
    assert result == reference  # exact equality, not approx


def test_block_nansum_ignores_nan(tmp_path):
    data = np.array([[[1.0, np.nan]], [[2.0, 4.0]]])
    path = tmp_path / "n.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("vol", data=data)
    with h5py.File(path, "r") as f:
        assert volumeio.block_nansum(f["vol"], budget_bytes=1) == 7.0


def _running_max(acc, block):
    return max(acc, float(np.nanmax(block)))


@pytest.mark.parametrize("divisor", [1, 4, 100])
def test_block_reduce_is_bit_identical_across_budgets(wide_volume, divisor):
    path, data = wide_volume
    with h5py.File(path, "r") as f:
        reference = volumeio.block_reduce(
            f["vol"], _running_max, budget_bytes=data.nbytes * 10, init=-np.inf
        )
        result = volumeio.block_reduce(
            f["vol"], _running_max, budget_bytes=max(1, data.nbytes // divisor), init=-np.inf
        )
    assert result == reference


@pytest.mark.parametrize("divisor", [1, 3, 50])
def test_two_pass_mean_subtraction_is_bit_identical_across_budgets(wide_volume, divisor):
    """The bucket-2 pattern: a global statistic, then a block-wise application."""
    path, data = wide_volume
    with h5py.File(path, "r") as f:

        def stat(acc, block):
            total, comp = volumeio.neumaier_sum(block.ravel(), state=acc[:2])
            return (total, comp, acc[2] + block.size)

        def apply(stat_value, block):
            total, comp, count = stat_value
            return block - ((total + comp) / count)

        budgets = (data.nbytes * 10, max(1, data.nbytes // divisor))
        outs = []
        for budget in budgets:
            blocks = list(
                volumeio.two_pass(f["vol"], stat, apply, budget_bytes=budget, init=(0.0, 0.0, 0))
            )
            outs.append(np.concatenate([b for _, b in blocks], axis=0))
    assert np.array_equal(outs[0], outs[1])  # bitwise, including NaN placement


def test_scratch_array_is_writable_and_persists_within_the_block(tmp_path):
    with volumeio.scratch_array((4, 3), np.float64, dirpath=str(tmp_path)) as arr:
        arr[:] = 2.5
        arr[0, 0] = 7.0
        assert arr.shape == (4, 3)
        assert arr[0, 0] == 7.0
        assert arr[3, 2] == 2.5


def test_scratch_array_deletes_its_file_on_exit(tmp_path):
    with volumeio.scratch_array((4, 3), np.float64, dirpath=str(tmp_path)) as arr:
        path = arr.filename
        assert path is not None
        assert list(tmp_path.iterdir())
    assert not list(tmp_path.iterdir()), "scratch file outlived the context"


def test_scratch_array_deletes_its_file_on_exception(tmp_path):
    """A crash mid-run must not leave gigabytes of scratch behind."""
    with pytest.raises(RuntimeError):
        with volumeio.scratch_array((4, 3), np.float64, dirpath=str(tmp_path)) as arr:
            arr[:] = 1.0
            raise RuntimeError("simulated stage failure")
    assert not list(tmp_path.iterdir())


def test_scratch_array_behaves_like_an_ndarray(tmp_path):
    """Numerical code must not need to know it is disk-backed."""
    with volumeio.scratch_array((5,), np.float64, dirpath=str(tmp_path)) as arr:
        arr[:] = np.arange(5)
        np.multiply(arr, 2.0, out=arr)  # in-place: the bucket-3 discipline
        assert np.array_equal(np.asarray(arr), np.arange(5) * 2.0)
        assert float(np.nanmean(arr)) == 4.0


def test_harness_passes_a_budget_independent_function(wide_volume):
    from tests.equivalence import assert_budget_independent

    path, data = wide_volume
    with h5py.File(path, "r") as f:
        assert_budget_independent(
            lambda d, budget_bytes: volumeio.block_nansum(d, budget_bytes=budget_bytes),
            f["vol"],
            nbytes=data.nbytes,
        )


def test_harness_catches_a_budget_dependent_function(wide_volume):
    """The harness is only worth having if it actually fails on a bad impl."""
    from tests.equivalence import assert_budget_independent

    path, data = wide_volume

    def naive_sum(dset, *, budget_bytes):
        # np.sum per block then adding up: pairwise ordering varies with budget
        return float(
            sum(
                float(np.nansum(b))
                for _sl, b in volumeio.iter_blocks(dset, budget_bytes=budget_bytes)
            )
        )

    with h5py.File(path, "r") as f:
        with pytest.raises(AssertionError, match="different bits"):
            assert_budget_independent(naive_sum, f["vol"], nbytes=data.nbytes)


# -- block context, second axis, streaming reductions -------------------------
def test_iter_with_context_appends_the_next_block_head():
    data = np.arange(20 * 3 * 4, dtype=np.float64).reshape(20, 3, 4)
    seen = np.zeros(20, dtype=int)
    windows = []
    blocks = volumeio.iter_blocks(data, budget_bytes=data.nbytes // 4)
    for interior, window, within in volumeio.iter_with_context(blocks, trailing=1):
        assert np.array_equal(window[within], data[interior])
        seen[interior] += 1
        windows.append((interior, window))
    assert np.array_equal(seen, np.ones(20, dtype=int)), "interiors must tile exactly once"
    # Every block but the last carries one row of the next.
    for interior, window in windows[:-1]:
        assert window.shape[0] == (interior.stop - interior.start) + 1
        assert np.array_equal(window[-1], data[interior.stop])
    last_interior, last_window = windows[-1]
    assert last_window.shape[0] == last_interior.stop - last_interior.start


def test_iter_with_context_works_on_a_generated_stream():
    """The point of taking a stream, not a dataset: generated blocks work too."""
    data = np.arange(12 * 2 * 2, dtype=np.float64).reshape(12, 2, 2)

    def generated():
        for start in range(0, 12, 5):
            stop = min(start + 5, 12)
            yield slice(start, stop), data[start:stop] * 2.0

    rebuilt = np.concatenate(
        [w[i] for _sl, w, i in volumeio.iter_with_context(generated(), trailing=1)], axis=0
    )
    assert np.array_equal(rebuilt, data * 2.0)


def test_iter_with_context_single_block():
    data = np.zeros((4, 2, 2))
    items = list(volumeio.iter_with_context(volumeio.iter_blocks(data, budget_bytes=1 << 30)))
    assert len(items) == 1
    interior, window, within = items[0]
    assert interior == slice(0, 4) and window.shape[0] == 4 and within == slice(0, 4)


def test_iter_with_context_carries_two_rows_when_asked():
    data = np.arange(9 * 2 * 2, dtype=np.float64).reshape(9, 2, 2)

    def generated():
        for start in range(0, 9, 3):
            yield slice(start, start + 3), data[start : start + 3]

    windows = list(volumeio.iter_with_context(generated(), trailing=2))
    assert [w.shape[0] for _i, w, _wi in windows] == [5, 5, 3]
    for interior, window, within in windows[:-1]:
        assert np.array_equal(window[within], data[interior])
        assert np.array_equal(window[within.stop :], data[interior.stop : interior.stop + 2])


def test_iter_with_context_refuses_a_successor_too_short_for_trailing():
    """A short successor must fail loudly, not hand back a truncated window.

    Silently returning one halo row where two were asked for corrupts exactly
    one block edge — invisible to the budget-independence tests, because every
    budget corrupts it the same way.
    """
    data = np.arange(7 * 2 * 2, dtype=np.float64).reshape(7, 2, 2)

    def generated():
        for sl in (slice(0, 3), slice(3, 6), slice(6, 7)):
            yield sl, data[sl]

    with pytest.raises(ValueError, match="trailing=2.*only 1"):
        list(volumeio.iter_with_context(generated(), trailing=2))


def test_iter_with_context_over_axis_1_blocks():
    """Context must follow the blocking axis, or it is context from nowhere.

    Over an ``axis=1`` stream the interior mapping happens to survive an
    axis-0 implementation (``within`` spans the whole of axis 0), so only the
    appended rows expose the bug: they would be the next block's first
    *Z-layer* rather than its first *column*.
    """
    data = np.arange(4 * 12 * 3, dtype=np.float64).reshape(4, 12, 3)
    blocks = volumeio.iter_blocks(data, budget_bytes=data.nbytes // 3, axis=1)
    windows = list(volumeio.iter_with_context(blocks, trailing=1, axis=1))
    assert len(windows) == 3
    for interior, window, within in windows:
        assert np.array_equal(window[within], data[:, interior])
    for interior, window, _within in windows[:-1]:
        assert window.shape[1] == (interior.stop - interior.start) + 1
        assert np.array_equal(window[:, -1], data[:, interior.stop])
    last_interior, last_window, _ = windows[-1]
    assert last_window.shape[1] == last_interior.stop - last_interior.start


def test_iter_blocks_axis_1():
    data = np.arange(4 * 12 * 3, dtype=np.float64).reshape(4, 12, 3)
    rebuilt = np.concatenate(
        [block for _sl, block in volumeio.iter_blocks(data, budget_bytes=data.nbytes // 3, axis=1)],
        axis=1,
    )
    assert np.array_equal(rebuilt, data)


def test_stream_mean_matches_nanmean_closely_and_is_budget_independent():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(30, 8, 8))
    data[data > 2] = np.nan
    means = [
        volumeio.stream_mean(volumeio.dataset_blocks(data, budget_bytes=data.nbytes // d))
        for d in (1, 3, 7, 1000)
    ]
    assert len({m.hex() for m in means}) == 1, "mean must not depend on the budget"
    assert means[0] == pytest.approx(float(np.nanmean(data)), rel=1e-12)


def test_stream_minmax_ignores_non_finite():
    data = np.array([[[1.0, np.nan]], [[np.inf, -3.0]], [[5.0, 2.0]]])
    lo, hi = volumeio.stream_minmax(volumeio.dataset_blocks(data, budget_bytes=8))
    assert (lo, hi) == (-3.0, 5.0)


def test_stream_mean_of_nothing_is_nan():
    data = np.full((4, 2, 2), np.nan)
    assert np.isnan(volumeio.stream_mean(volumeio.dataset_blocks(data, budget_bytes=16)))


@pytest.mark.parametrize(
    ("values", "expect_lo", "expect_hi"),
    [
        ([-0.0] * 8 + [0.0] * 8 + [1.0, 2.0], -0.0, 2.0),  # mixed zeros, positive signal
        ([0.0] * 8 + [-0.0] * 8 + [1.0, 2.0], -0.0, 2.0),  # same data, other order
        ([-2.0, -1.0] + [0.0] * 4 + [-0.0] * 4, -2.0, 0.0),  # zeros are the maximum
        ([-0.0] * 6, -0.0, -0.0),  # every zero negative
        ([0.0] * 6, 0.0, 0.0),  # every zero positive
    ],
    ids=["mixed-min", "mixed-min-reordered", "mixed-max", "all-negative", "all-positive"],
)
def test_stream_minmax_zero_sign_is_budget_independent(values, expect_lo, expect_hi):
    """`-0.0` and `0.0` compare equal, so a block-wise min keeps whichever came first.

    That made the *sign* of a zero bound a function of the budget: on the
    mixed-min dataset this returned `0x0.0p+0` whole-array and `-0x0.0p+0` at
    every smaller budget — a budget-dependent bit in a helper whose governing
    guarantee is bit-identical results at any budget. The project's own harness
    cannot catch it: `tests/equivalence.py` documents that `array_equal` calls
    `-0.0` and `+0.0` equal, so only `.hex()` sees this.

    There is no numpy parity to honour — `np.min` on mixed zeros is just as
    arbitrary — so the rule is the sign-aware one: the minimum prefers `-0.0`,
    the maximum prefers `+0.0`, decided by the data and not by the blocking.
    """
    data = np.array(values).reshape(len(values), 1, 1)
    results = {
        tuple(
            v.hex()
            for v in volumeio.stream_minmax(
                volumeio.dataset_blocks(data, budget_bytes=max(1, data.nbytes // d))
            )
        )
        for d in (1, 2, 3, 5, len(values))
    }
    assert len(results) == 1, f"the zero sign moves with the budget: {results}"
    assert results == {(expect_lo.hex(), expect_hi.hex())}


def test_stream_minmax_zero_sign_costs_no_extra_copy():
    """Deciding the sign must not allocate a second copy of the block.

    Selecting the zeros first (`np.signbit(finite[finite == 0.0])`) copies
    them, which on a zero-heavy block is a whole extra full-size array — and
    zero-heavy is the shipping case, since the paraview volume's masked voxels
    are exactly 0.0, so `min == 0.0` fires this branch in *every* block. That
    would put a memory-budget helper over its own budget on the very data the
    module was hardened for. `np.signbit(finite)` is one byte per value and
    equivalent (a block whose min is zero holds no negative value).

    Measured as the *marginal* peak against an identical block with no zeros,
    so the pre-existing cost of `block[np.isfinite(block)]` cancels out.
    """
    count = 1 << 20  # 4 MB of float32
    zero_heavy = np.zeros(count, dtype=np.float32).reshape(-1, 32, 32)
    zero_heavy[0, 0, 0] = 5.0  # so the max is not itself a zero
    no_zeros = np.full(count, 3.0, dtype=np.float32).reshape(-1, 32, 32)

    def peak_bytes(data):
        tracemalloc.start()
        try:
            tracemalloc.reset_peak()
            volumeio.stream_minmax(volumeio.dataset_blocks(data, budget_bytes=data.nbytes))
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    marginal = peak_bytes(zero_heavy) - peak_bytes(no_zeros)
    limit = zero_heavy.nbytes // 2
    assert marginal <= limit, (
        f"the zero-sign branch adds {marginal / 1e6:.1f} MB on a "
        f"{zero_heavy.nbytes / 1e6:.1f} MB block — it is copying the zeros"
    )


# -- exact streaming quantile -------------------------------------------------
@pytest.mark.parametrize("q", [0.0, 1.0, 25.0, 50.0, 99.0, 100.0])
def test_stream_quantile_matches_numpy_exactly(q):
    rng = np.random.default_rng(7)
    data = rng.normal(size=(40, 6, 6))
    data[data > 1.8] = np.nan
    finite = data[np.isfinite(data)]
    got = volumeio.stream_quantile(
        lambda: volumeio.dataset_blocks(data, budget_bytes=data.nbytes // 5), q
    )
    assert got == float(np.percentile(finite, q)), "must be bit-equal, not merely close"


def test_stream_quantile_is_budget_independent():
    rng = np.random.default_rng(11)
    data = rng.normal(size=(33, 5, 5))
    values = [
        volumeio.stream_quantile(
            lambda d=d: volumeio.dataset_blocks(data, budget_bytes=max(1, data.nbytes // d)), 99.0
        )
        for d in (1, 2, 7, 10_000)
    ]
    assert len({v.hex() for v in values}) == 1


def test_stream_quantile_handles_constant_data():
    data = np.full((10, 3, 3), 4.25)
    got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=72), 50.0)
    assert got == 4.25


def test_stream_quantile_of_nothing_is_nan():
    data = np.full((4, 2, 2), np.nan)
    got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=16), 50.0)
    assert np.isnan(got)


def test_stream_quantile_single_finite_value():
    data = np.full((4, 2, 2), np.nan)
    data[1, 1, 1] = 2.5
    got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=16), 90.0)
    assert got == 2.5


@pytest.mark.parametrize("q", [-1.0, 101.0, float("nan")])
def test_stream_quantile_rejects_a_percentile_outside_0_to_100(q):
    """Out of range is an error, as it is in numpy — never a clamped answer.

    Before the guard, q=-1 returned the minimum and q=101 the maximum: a
    plausible-looking number for a caller that had computed its percentile
    wrongly, which is precisely the failure this helper exists to rule out.
    """
    data = np.arange(8.0).reshape(8, 1, 1)
    with pytest.raises(ValueError, match=r"\[0, 100\]"):
        volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=8), q)


@pytest.mark.parametrize("q", [0.5, 2.0, 50.0, 98.0, 99.5])
def test_stream_quantile_is_exact_on_adversarial_data_at_eight_budgets(q):
    """Wide dynamic range + heavy NaN, at eight budgets, bit-equal to numpy.

    The colour-limit call sites run over volumes spanning many decades with a
    large fraction of the voxels masked out; an implementation that is exact
    only on tame Gaussian data would not serve them. Eight budgets (down to one
    that forces single-layer blocks) is the standard Task 5's review applied.
    """
    rng = np.random.default_rng(23)
    data = 10.0 ** rng.uniform(-12, 12, size=(24, 7, 7))
    data[rng.random(data.shape) < 0.1] = np.nan
    data[rng.random(data.shape) < 0.02] = np.inf  # non-finite values must be ignored
    data[0, 0, :3] = data[0, 0, 0]  # a tie, so the rank must break it consistently
    expected = float(np.percentile(data[np.isfinite(data)], q))
    values = [
        volumeio.stream_quantile(
            lambda d=d: volumeio.dataset_blocks(data, budget_bytes=max(1, data.nbytes // d)), q
        )
        for d in (1, 2, 3, 5, 7, 13, 1000, 10_000)
    ]
    assert len({v.hex() for v in values}) == 1, "the budget must not move the answer"
    assert values[0] == expected, "must be bit-equal to np.percentile, not merely close"


def test_stream_quantile_refines_when_a_bin_is_overfull(monkeypatch):
    """Force the narrowing branch: with 2 bins, a bin holds more than the cap.

    Without refinement the helper would either sort the whole overfull bin
    (unbounded memory, the thing this exists to avoid) or return a bin edge.
    Shrinking both constants makes the multi-round path the one under test.
    """
    monkeypatch.setattr(volumeio, "_QUANTILE_BINS", 2)
    monkeypatch.setattr(volumeio, "_QUANTILE_EXACT_CAP", 4)
    rng = np.random.default_rng(3)
    data = rng.normal(size=(20, 4, 4))
    for q in (10.0, 50.0, 90.0):
        got = volumeio.stream_quantile(
            lambda: volumeio.dataset_blocks(data, budget_bytes=data.nbytes // 4), q
        )
        assert got == float(np.percentile(data, q))


def test_stream_quantile_uses_numpys_two_form_interpolation():
    """A triple where the naive one-form interpolation is one ulp off.

    numpy switches from ``lo + (hi - lo) * t`` to ``hi - (hi - lo) * (1 - t)``
    at ``t >= 0.5``; here q=30 puts the target at t=0.6 between the first two
    values and the two forms disagree in the last bit. Without this case the
    random-data tests pass with either form, so the switch would look optional.
    """
    data = np.array([[[-69938.68585729932]], [[16485.831736166052]], [[16486.831736166052]]])
    expected = float(np.percentile(data, 30.0))
    assert expected == -18083.975301220096  # the one-form value is ...220104
    got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=8), 30.0)
    assert got == expected


@pytest.mark.parametrize("q", [0.0, 22.5, 25.0, 50.0, 62.5, 75.0, 97.5, 100.0])
def test_stream_quantile_bins_values_that_sit_exactly_on_bin_edges(q, monkeypatch):
    """Values landing exactly on a bin edge must be counted where they are collected.

    With four bins over 0..8 the edges are 0, 2, 4, 6, 8 and every value is an
    edge or an interior point, so the histogram convention is fully exercised:
    ``side="right"`` puts an edge value in the bin it *starts*, matching the
    survivor pass's ``[bin_lo, bin_hi)``, and the final bin is closed so the
    maximum belongs somewhere. Either half of that wrong and the counted bin
    disagrees with the collected one — an IndexError or a silently wrong value,
    not a rounding difference.
    """
    monkeypatch.setattr(volumeio, "_QUANTILE_BINS", 4)
    data = np.repeat(np.arange(9.0), 9).reshape(9, 3, 3)
    got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=72), q)
    assert got == float(np.percentile(data, q))


def test_stream_quantile_terminates_when_bin_edges_collapse(monkeypatch):
    """A run of identical values narrows the window until it cannot be split.

    Refinement halves toward the repeated value until `hi` is the float
    immediately above `lo`. Every interior edge then rounds onto an endpoint,
    the chosen bin *is* the whole window and re-bisecting reproduces the same
    round forever — and `lo == hi` never becomes true, so the guard at the top
    of the loop does not end it; `_select_among_ties` does. At the shipped
    constants this needs a run of more than 2**20 identical values, which a
    constant background region in a full-size volume supplies. The traversal
    cap turns the failure mode into a fast assertion instead of a hung suite.
    """
    monkeypatch.setattr(volumeio, "_QUANTILE_BINS", 2)
    monkeypatch.setattr(volumeio, "_QUANTILE_EXACT_CAP", 4)
    data = np.full((12, 3, 3), 0.5)
    data[0, 0, 0] = 0.0
    data[0, 0, 1] = 1.0
    traversals = 0

    def make_blocks():
        nonlocal traversals
        traversals += 1
        if traversals > 20_000:
            raise AssertionError("refinement never terminated")
        return volumeio.dataset_blocks(data, budget_bytes=data.nbytes // 3)

    assert volumeio.stream_quantile(make_blocks, 50.0) == float(np.percentile(data, 50.0)) == 0.5


@pytest.mark.parametrize("q", [0.0, 50.0, 80.0, 95.0, 100.0])
def test_stream_quantile_selects_across_an_unsplittable_window(q, monkeypatch):
    """The whole window is two adjacent floats, and the answer is the upper one.

    `stream_quantile`'s min and max here differ by a single ulp, so the very
    first histogram cannot split them and the search lands straight in the
    tie handler with a target that runs past the repeated lower value. A
    handler that only ever returned the window's lower end — or that walked
    the distinct values in the wrong order — would answer 0.5 for the top
    percentiles instead of the float above it. The 16/5 split with q=80 puts
    the target exactly on the first upper value (rank 16 of 21), the rank an
    off-by-one in the running count gets wrong and every other rank forgives.
    """
    monkeypatch.setattr(volumeio, "_QUANTILE_BINS", 2)
    monkeypatch.setattr(volumeio, "_QUANTILE_EXACT_CAP", 4)
    above = float(np.nextafter(0.5, 1.0))
    values = np.array([0.5] * 16 + [above] * 5)
    data = values.reshape(21, 1, 1)
    traversals = 0

    def make_blocks():
        nonlocal traversals
        traversals += 1
        if traversals > 20_000:
            raise AssertionError("refinement never terminated")
        return volumeio.dataset_blocks(data, budget_bytes=8)

    got = volumeio.stream_quantile(make_blocks, q)
    assert got == float(np.percentile(values, q))
    if q >= 80.0:
        assert got == above  # and not 0.5, which is only one ulp away


def test_stream_quantile_skips_interpolation_when_the_rank_is_exact():
    """On a rank that needs no interpolation, no interpolation is performed.

    A spread wide enough to overflow `hi - lo` makes this visible: numpy runs
    its lerp regardless and returns `nan` here (its `subtract` overflows to
    inf, then `inf * 0`), while the value asked for is simply the smaller of
    two order statistics. This is the one input class where this helper
    deliberately does **not** reproduce `np.percentile` — it returns the exact
    order statistic instead of an overflow artefact — and it is unreachable
    for the colour limits it serves, whose data never spans 1e308.
    """
    data = np.array([[[-1e308]], [[1e308]]])
    with np.errstate(over="ignore", invalid="ignore"):  # the overflow is the subject
        got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=8), 0.0)
        assert got == -1e308
        assert np.isnan(float(np.percentile(data, 0.0)))  # what we decline to reproduce
        # Ranks that genuinely interpolate still track numpy, overflow included.
        for q in (25.0, 50.0, 100.0):
            streamed = volumeio.stream_quantile(
                lambda: volumeio.dataset_blocks(data, budget_bytes=8), q
            )
            assert streamed == float(np.percentile(data, q))


def test_stream_quantile_passes_the_shared_budget_independence_harness():
    """Checked by the same instrument as every other phase-5 conversion."""
    from tests.equivalence import assert_budget_independent

    rng = np.random.default_rng(31)
    data = 10.0 ** rng.uniform(-12, 12, size=(24, 7, 7))
    data[rng.random(data.shape) < 0.1] = np.nan

    def run(dset, *, budget_bytes):
        return volumeio.stream_quantile(
            lambda: volumeio.dataset_blocks(dset, budget_bytes=budget_bytes), 99.0
        )

    assert_budget_independent(
        run, data, budgets=[max(1, data.nbytes // d) for d in (1, 3, 7, 1000)]
    )


def test_stream_quantile_subtracts_in_the_arrays_dtype():
    """numpy computes `hi - lo` in float32 for a float32 array; so must this.

    This pair brackets q≈59.9 and the two order statistics differ by ~2000×,
    so rounding the difference to float32 (what numpy does) and keeping it in
    float64 (what a dtype-blind implementation does) disagree in the 8th
    significant figure. The project stores volumes as float32 — paraview.py's
    SAVE_DTYPE, rocking.py, slices.py — and clims come from `np.percentile`
    over exactly those arrays, so this is the shipping path, not a curiosity.
    """
    values = np.array([14.359252, 31153.064], dtype=np.float32)
    data = values.reshape(2, 1, 1)
    expected = float(np.percentile(values, 59.94683904708903))
    assert expected == 18681.02878953133  # float64 subtraction gives ...740256337
    got = volumeio.stream_quantile(
        lambda: volumeio.dataset_blocks(data, budget_bytes=4), 59.94683904708903
    )
    assert got == expected


def test_stream_quantile_agrees_with_its_own_histogram_on_float32():
    """A float32 value on a bin edge must not be counted in one bin, collected from another.

    numpy 1.x demotes a Python float compared against a float32 array
    (value-based casting), so the float64 bin edge the histogram used gets
    rounded in the survivor mask. This array reproduces it: the bin holding
    the target rank counts one value, and the mask then finds none of it —
    `ValueError: need at least one array to concatenate`, the lucky outcome.
    The unlucky one is a selection from the wrong candidates.
    """
    values = np.array(
        [
            7.4209438e03,
            4.1942276e01,
            6.3505867e01,
            2.0944195e08,
            3.0033918e03,
            1.7869806e08,
            4.4543552e07,
        ],
        dtype=np.float32,
    )
    data = values.reshape(7, 1, 1)
    q = 69.60232745249438
    got = volumeio.stream_quantile(lambda: volumeio.dataset_blocks(data, budget_bytes=4), q)
    assert got == float(np.percentile(values, q))


@pytest.mark.parametrize("q", [1.0, 25.0, 50.0, 90.0, 99.0, 99.5])
def test_stream_quantile_matches_numpy_on_float32_volumes(q):
    """float32 is what this project stores, and it is not float64 in miniature.

    Two ways to fail, both live before this test existed. numpy computes
    `hi - lo` in the array's dtype, so a float64 subtraction diverges by ~1e-8
    relative once the bracketing order statistics differ by more than about 2×
    — which a heavy tail under heavy masking produces constantly. And numpy 1.x
    demotes a Python float compared against a float32 array (value-based
    casting), so a float64 bin edge rounds and the survivor mask stops agreeing
    with the histogram — an empty survivor set or a selection from the wrong
    candidates. The old float64-only fuzz could not see either.
    """
    rng = np.random.default_rng(2)
    data = (10.0 ** rng.uniform(0, 10, size=(60, 20, 20))).astype(np.float32)
    data[rng.random(data.shape) < 0.9] = np.nan  # heavily masked, heavy tailed
    finite = data[np.isfinite(data)]
    assert finite.dtype == np.float32
    for divisor in (1, 3, 1000):
        got = volumeio.stream_quantile(
            lambda d=divisor: volumeio.dataset_blocks(data, budget_bytes=max(1, data.nbytes // d)),
            q,
        )
        assert got == float(np.percentile(finite, q)), "float32 must be bit-equal too"


def test_stream_quantile_does_not_walk_the_exponent_range_on_a_zero_background(monkeypatch):
    """A masked background of exact zeros must not cost ~150 reads of the volume.

    Bin edges narrow the *range* geometrically but the *exponent* only
    linearly, so converging onto a tie at 0.0 walks the whole denormal range.
    Measured on a realistic float32 intensity volume at q=50: 142 traversals
    for a 0.0 background against 16 for 1.0 — on the 17 GB paraview volume,
    hours of reading that a user cannot tell from a hang. Narrowing to the
    bin's observed [min, max] collapses a tied bin in one round instead.
    """
    monkeypatch.setattr(volumeio, "_QUANTILE_EXACT_CAP", 4)
    rng = np.random.default_rng(4)
    data = np.zeros(2048, dtype=np.float32)
    data[:256] = rng.gamma(2.0, 500.0, size=256).astype(np.float32)
    data = data.reshape(-1, 8, 8)
    traversals = 0

    def make_blocks():
        nonlocal traversals
        traversals += 1
        return volumeio.dataset_blocks(data, budget_bytes=data.nbytes // 4)

    got = volumeio.stream_quantile(make_blocks, 50.0)
    assert got == float(np.percentile(data, 50.0)) == 0.0
    assert traversals <= 12, f"{traversals} traversals — bisection is walking the exponent range"


@pytest.mark.parametrize(
    "values",
    [
        [-0.0] * 12,  # every value is negative zero
        [-0.0] * 10 + [1.0, 2.0],  # a negative-zero background under signal
    ],
    ids=["all-negative", "background"],
)
@pytest.mark.parametrize("q", [0.0, 10.0, 25.0, 50.0, 75.0])
def test_stream_quantile_carries_numpys_sign_of_zero(values, q, monkeypatch):
    """The sign of a zero follows numpy, which means following its two forms.

    numpy's `_lerp` normalises `-0.0` to `+0.0` in form 1 (`a + diff * t`) and
    keeps it in form 2 (`b - diff * (1 - t)`), so on an all-`-0.0` array the
    sign flips with the fraction: `+0.0` at q=10, `-0.0` at q=25 and q=50.
    Reproducing that needs `_lerp` to be applied even when the endpoints are
    equal — a `hi != lo` short-circuit answers `+0.0` throughout — and needs
    the sign fed into it to come from the data rather than from whichever
    block `stream_minmax` happened to see first.
    """
    monkeypatch.setattr(volumeio, "_QUANTILE_EXACT_CAP", 4)
    data = np.array(values).reshape(len(values), 1, 1)
    expected = float(np.percentile(data, q))
    got = {
        volumeio.stream_quantile(
            lambda d=d: volumeio.dataset_blocks(data, budget_bytes=max(1, data.nbytes // d)), q
        ).hex()
        for d in (1, 3, len(values))
    }
    assert got == {expected.hex()}, f"numpy says {expected.hex()}, we say {got}"


def test_stream_quantile_sign_of_zero_is_budget_independent_when_signs_are_mixed(monkeypatch):
    """Mixed ±0.0 has no numpy bit to match, so the requirement is stability.

    numpy's sign there comes from an arbitrary partition order (it reports
    `-0.0` at q=25 and `+0.0` at q=50 for this very array), and reproducing it
    would mean sorting the whole volume — the thing this helper exists not to
    do. What is *not* negotiable is that the answer stops depending on the
    budget, which it did: `-0.0` at one blocking and `+0.0` at another.
    `assert_budget_independent` cannot see this, since `array_equal` calls the
    two equal; `.hex()` can.
    """
    monkeypatch.setattr(volumeio, "_QUANTILE_EXACT_CAP", 4)
    tiny = float(np.nextafter(0.0, 1.0))
    datasets = {
        "zeros only": [-0.0] * 8 + [0.0] * 8,  # every value a zero
        "under signal": [-0.0] * 8 + [0.0] * 8 + [1.0, 2.0, 3.0, 4.0],  # via selection
        "against a denormal": [-0.0] * 8 + [0.0] * 8 + [tiny] * 4,  # via the tie handler
    }
    for name, values in datasets.items():
        data = np.array(values).reshape(len(values), 1, 1)
        for q in (10.0, 25.0, 50.0):
            got = {
                volumeio.stream_quantile(
                    lambda d=d: volumeio.dataset_blocks(
                        data, budget_bytes=max(1, data.nbytes // d)
                    ),
                    q,
                ).hex()
                for d in (1, 2, 3, 5, len(values))
            }
            assert got == {(0.0).hex()}, f"{name} q={q}: sign depends on the budget, got {got}"


def test_stream_quantile_accepts_a_generated_block_stream():
    """The factory may build blocks that exist nowhere on disk."""
    rng = np.random.default_rng(5)
    layers = [rng.normal(size=(3, 3)) for _ in range(9)]

    def make_blocks():
        return (layer * 2.0 for layer in layers)

    expected = float(np.percentile(np.concatenate([layer.ravel() for layer in layers]) * 2.0, 30.0))
    assert volumeio.stream_quantile(make_blocks, 30.0) == expected


# -- display decimation (render paths) ---------------------------------------
def test_display_decimation_budgets_two_copies(tmp_path):
    """The peak is ~2 copies, so a volume that fits ONCE must still be decimated."""
    path = tmp_path / "d.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("vol", data=np.zeros((16, 16, 16)))  # float64 -> 32768 B
    with h5py.File(path, "r") as f:
        dset = f["vol"]
        assert volumeio.volume_bytes(dset) == 32768
        # One copy fits inside 40000 B; two do not.
        assert volumeio.display_decimation(dset, 40000) > 1
        # Room for both copies -> full fidelity.
        assert volumeio.display_decimation(dset, 1 << 30) == 1


def test_display_decimation_rejects_a_non_3d_dataset(tmp_path):
    """A 2-D dataset gets a legible error, not the caller's IndexError."""
    path = tmp_path / "flat.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("map", data=np.zeros((8, 8)))
    with h5py.File(path, "r") as f:
        with pytest.raises(ValueError, match="3-D volume"):
            volumeio.display_decimation(f["map"], 1 << 30)


def test_decimation_note_prints_shape_in_dataset_order():
    note = volumeio.decimation_note(4, (76, 700, 2891))
    assert "(76, 700, 2891)" in note
    assert "2891x700x76" not in note  # reversed reads as a contradiction
    assert "decimated 4x" in note and "stored data is unchanged" in note
