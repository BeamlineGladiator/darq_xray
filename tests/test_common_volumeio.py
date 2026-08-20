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
    """Values spanning many magnitudes — where naive summation loses bits."""
    rng = np.random.default_rng(20260820)
    data = (rng.standard_normal((13, 9, 7)) * 10.0 ** rng.integers(-6, 7, (13, 9, 7))).astype(
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
