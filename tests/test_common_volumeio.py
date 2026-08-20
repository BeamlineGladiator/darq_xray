"""Bounded-memory volume IO (dfxm/common/volumeio.py)."""

from __future__ import annotations

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
