"""Tests for dfxm.common.alignment and dfxm.common.raster.

The alignment primitives are checked voxel-for-voxel against the legacy PVTI
exporter (export_aligned_volumes_to_paraview_v6_pvti) so the two stay
interchangeable in ParaView world coordinates.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm.common import alignment as A
from dfxm.common import raster as R


def _legacy_export():
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "export_aligned_volumes_to_paraview_v6_pvti.py").exists():
        pytest.skip("legacy PVTI exporter not found")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return __import__("export_aligned_volumes_to_paraview_v6_pvti")


def _synthetic():
    rng = np.random.default_rng(3)
    vol = rng.standard_normal((5, 8, 10))
    vol[0, 0, 0] = np.nan  # ensure NaN handling is exercised
    samy = np.array([0.0, 0.001, 0.0025, 0.004, 0.0061])  # mm
    samz = np.array([0.0, 0.0009, 0.0021, 0.0035, 0.0052])  # mm, irregular
    return vol, samy, samz


def test_samy_shift_matches_legacy():
    legacy = _legacy_export()
    vol, samy, _ = _synthetic()
    mine = A.apply_samy_shifts_to_volume(
        vol, samy, scale_x=legacy.SCALE_X, samy_direction=legacy.SAMY_DIRECTION
    )
    gold = legacy.apply_samy_shifts_to_volume(vol, samy)
    assert mine.shape == gold.shape
    np.testing.assert_array_equal(mine, gold)  # NaNs compare equal


def test_pad_left_matches_legacy():
    legacy = _legacy_export()
    _, samy, _ = _synthetic()
    assert A.compute_pad_left(
        samy, scale_x=legacy.SCALE_X, samy_direction=legacy.SAMY_DIRECTION
    ) == legacy.compute_pad_left(samy)


def test_z_interp_matches_legacy():
    legacy = _legacy_export()
    vol, _, samz = _synthetic()
    v_mine, z_mine, s_mine = A.interpolate_to_uniform_z(vol, samz)
    v_gold, z_gold, s_gold = legacy.interpolate_to_uniform_z(vol, samz)
    np.testing.assert_allclose(v_mine, v_gold, equal_nan=True)
    np.testing.assert_allclose(z_mine, z_gold)
    assert s_mine == pytest.approx(s_gold)


def test_center_and_roi_match_legacy():
    legacy = _legacy_export()
    vol, _, _ = _synthetic()
    for method in ("mean", "median"):
        d_mine, o_mine = A.center_around_zero(vol, method)
        d_gold, o_gold = legacy.center_around_zero(vol, method)
        np.testing.assert_allclose(d_mine, d_gold, equal_nan=True)
        assert o_mine == pytest.approx(o_gold)
    np.testing.assert_array_equal(
        A.apply_roi_3d(vol, (2, 8), (1, 6)), legacy.apply_roi_3d(vol, (2, 8), (1, 6))
    )


def test_align_volume_pipeline_order():
    vol, samy, samz = _synthetic()
    out = A.align_volume(
        vol, samy, samz, scale_x=0.152, roi_y=(1, 7), take_abs=True, center_method="mean"
    )
    # abs -> roi (Y 1:7) -> samy shift (X expand) -> z interp -> centre
    assert out.data.shape[1] == 6  # ROI in Y
    assert out.data.shape[2] >= 10  # X canvas expanded
    assert out.scale_z_um > 0 and out.pad_left >= 0
    finite = out.data[np.isfinite(out.data)]
    assert abs(float(np.mean(finite))) < 1e-9  # centred


def test_center_unknown_method_raises():
    with pytest.raises(ValueError):
        A.center_around_zero(np.zeros((2, 2, 2)), "bogus")


def test_interpolate_to_uniform_z_single_layer():
    """A single-layer volume must return finite data (not NaN) with scale_z > 0 (FIX 3)."""
    rng = np.random.default_rng(42)
    vol = rng.standard_normal((1, 4, 5))
    samz = np.array([0.001])  # mm — single position
    vol_out, z_uniform, scale_z = A.interpolate_to_uniform_z(vol, samz)
    assert vol_out.shape == (1, 4, 5), f"expected (1,4,5), got {vol_out.shape}"
    assert np.all(np.isfinite(vol_out)), "single-layer output must be fully finite"
    np.testing.assert_array_equal(vol_out, vol)
    assert scale_z > 0, f"scale_z must be positive, got {scale_z}"
    assert len(z_uniform) == 1


# -- streamed alignment -------------------------------------------------------
def _streamed_synthetic(nz=17, ny=6, nx=9, seed=3):
    rng = np.random.default_rng(seed)
    vol = rng.normal(size=(nz, ny, nx))
    vol[vol > 1.9] = np.nan
    samy = np.cumsum(rng.normal(scale=0.002, size=nz))
    samz = np.sort(rng.normal(scale=0.01, size=nz))
    return vol, samy, samz


def _drain(streamed):
    """Rebuild the whole volume and report how many blocks it took.

    The block count is returned so parity tests can assert the small-budget
    cases really streamed. Without that assertion a change to the block-size
    arithmetic could quietly collapse every parametrisation to one block and
    the parity test would keep passing while measuring nothing.
    """
    rebuilt = np.empty(streamed.shape, dtype=streamed.dtype)
    n_blocks = 0
    for zsl, block in streamed.blocks():
        rebuilt[zsl] = block
        n_blocks += 1
    return rebuilt, n_blocks


@pytest.mark.parametrize("center_method", [None, "mean", "median"])
@pytest.mark.parametrize("divisor", [1, 3, 8, 10_000])
def test_streamed_alignment_matches_in_core(center_method, divisor):
    vol, samy, samz = _streamed_synthetic()
    kwargs = dict(
        scale_x=0.15,
        samy_direction=1,
        roi_x=(1, 8),
        roi_y=(0, 5),
        take_abs=False,
        center_method=center_method,
    )
    reference = A.align_volume(vol, samy, samz, **kwargs)
    # Budgets are fractions of the ALIGNED volume, not of the input. The
    # aligned canvas here is ~13x the input (27 interpolated Z layers on a
    # samy-widened X), so `vol.nbytes // divisor` puts every divisor — 1
    # included — at one output layer per block, and the parametrisation
    # measures nothing. Sizing from the output is what makes the four cases
    # four different blockings.
    out_bytes = int(np.prod(reference.data.shape)) * reference.data.dtype.itemsize
    streamed = A.align_volume_streamed(
        vol, samy, samz, budget_bytes=max(1, out_bytes // divisor), **kwargs
    )
    assert streamed.shape == reference.data.shape
    assert streamed.pad_left == reference.pad_left
    assert np.array_equal(streamed.z_uniform_um, reference.z_uniform_um)
    assert streamed.scale_z_um == reference.scale_z_um
    assert streamed.center_offset == reference.center_offset
    rebuilt, n_blocks = _drain(streamed)
    # The measurement must be live: each budget has to produce the blocking it
    # is meant to, or this case is silently comparing one whole-volume pass
    # against another.
    expected_blocks = {1: 1, 3: 3, 8: 9, 10_000: reference.data.shape[0]}[divisor]
    assert n_blocks == expected_blocks, f"budget/{divisor} gave {n_blocks} blocks"
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_streamed_blocks_factory_can_be_traversed_twice():
    vol, samy, samz = _streamed_synthetic()
    streamed = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=vol.nbytes // 4)
    first = np.concatenate([b for _sl, b in streamed.blocks()], axis=0)
    second = np.concatenate([b for _sl, b in streamed.blocks()], axis=0)
    assert np.array_equal(first, second, equal_nan=True)


def test_streamed_shape_known_before_reading():
    """A writer must be able to size its output before a voxel is read."""
    vol, samy, samz = _streamed_synthetic()
    streamed = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, roi_x=(2, 7), budget_bytes=64)
    assert streamed.shape[1] == 6  # full Y
    assert streamed.shape[2] == 5 + streamed.pad_left + streamed.pad_right


def test_block_samy_slice_does_not_shrink_the_canvas():
    """The trap: a per-block samy slice implies a narrower pad than the global one."""
    vol, samy, samz = _streamed_synthetic(nz=21)
    samy = samy + np.linspace(0, 0.05, len(samy))  # a strong monotone drift
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    streamed = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=vol.nbytes // 9)
    assert streamed.shape[2] == reference.data.shape[2]
    rebuilt, n_blocks = _drain(streamed)
    assert n_blocks > 1
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_block_samz_slice_does_not_shift_the_z_grid():
    """The Z-grid twin of the pad trap: a block's samz implies a different grid."""
    vol, samy, samz = _streamed_synthetic(nz=19)
    samz = np.sort(samz) * 3.0  # widen the Z span so a block's own grid differs loudly
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    streamed = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=vol.nbytes // 7)
    assert streamed.shape[0] == reference.data.shape[0]
    assert np.array_equal(streamed.z_uniform_um, reference.z_uniform_um)
    rebuilt, n_blocks = _drain(streamed)
    assert n_blocks > 1
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


@pytest.mark.parametrize("divisor", [1, 5, 10_000])
def test_streamed_matches_in_core_for_decreasing_samz(divisor):
    """interp1d sorts internally, so a decreasing samz must stream identically."""
    vol, samy, samz = _streamed_synthetic(nz=15)
    samz = samz[::-1].copy()  # strictly decreasing
    reference = A.align_volume(vol, samy, samz, scale_x=0.15, center_method="mean")
    streamed = A.align_volume_streamed(
        vol,
        samy,
        samz,
        scale_x=0.15,
        center_method="mean",
        budget_bytes=max(1, vol.nbytes // divisor),
    )
    rebuilt, _n = _drain(streamed)
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


@pytest.mark.parametrize("divisor", [1, 5, 10_000])
def test_streamed_matches_in_core_for_nonmonotonic_samz(divisor):
    """A shuffled samz is degenerate but must not produce a *wrong* answer."""
    vol, samy, samz = _streamed_synthetic(nz=13, seed=11)
    samz = samz[np.array([0, 4, 2, 6, 1, 8, 3, 10, 5, 12, 7, 9, 11])]
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    streamed = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, budget_bytes=max(1, vol.nbytes // divisor)
    )
    rebuilt, _n = _drain(streamed)
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_streamed_single_layer_volume_matches_in_core():
    """The interpolator's single-layer early return has to survive streaming."""
    rng = np.random.default_rng(7)
    vol = rng.normal(size=(1, 4, 5))
    samy = np.array([0.002])
    samz = np.array([0.001])
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    streamed = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=8)
    assert streamed.shape == reference.data.shape
    rebuilt, _n = _drain(streamed)
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


@pytest.mark.parametrize("take_abs", [False, True])
def test_streamed_matches_in_core_float32(take_abs):
    """Stage volumes are float32 on disk; the chain upcasts and both paths must agree."""
    vol, samy, samz = _streamed_synthetic(nz=11)
    vol = vol.astype(np.float32)
    reference = A.align_volume(
        vol, samy, samz, scale_x=0.15, take_abs=take_abs, center_method="median"
    )
    streamed = A.align_volume_streamed(
        vol,
        samy,
        samz,
        scale_x=0.15,
        take_abs=take_abs,
        center_method="median",
        budget_bytes=max(1, vol.nbytes // 6),
    )
    assert streamed.dtype == reference.data.dtype
    rebuilt, n_blocks = _drain(streamed)
    assert n_blocks > 1
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_streamed_median_uses_scratch_when_the_volume_will_not_fit(tmp_path):
    """The cached-median branch must give the same answer as the re-run branch."""
    vol, samy, samz = _streamed_synthetic(nz=17)
    budget = max(1, vol.nbytes // 9)
    plain = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, center_method="median", budget_bytes=budget
    )
    cached = A.align_volume_streamed(
        vol,
        samy,
        samz,
        scale_x=0.15,
        center_method="median",
        budget_bytes=budget,
        scratch_dir=str(tmp_path),
    )
    assert cached.center_offset == plain.center_offset
    assert np.array_equal(_drain(cached)[0], _drain(plain)[0], equal_nan=True)
    # The branch under test only runs when the aligned volume exceeds the
    # budget; assert that, so the test cannot go vacuous.
    per_layer = plain.shape[1] * plain.shape[2] * plain.dtype.itemsize
    assert plain.shape[0] * per_layer > budget
    assert not list(tmp_path.iterdir())  # scratch file cleaned up


class _CountingDataset:
    """An array that counts how many times a stream reads slices out of it."""

    def __init__(self, array):
        self._array = array
        self.reads = 0

    @property
    def shape(self):
        return self._array.shape

    @property
    def dtype(self):
        return self._array.dtype

    def __getitem__(self, key):
        self.reads += 1
        return self._array[key]


@pytest.mark.parametrize(
    ("center_method", "expected_reads"),
    [(None, 1), ("mean", 2), ("median", 2)],
)
def test_centring_costs_one_extra_pass_when_the_volume_fits(center_method, expected_reads):
    """The median must not re-align the volume once per quantile traversal.

    The quantile traverses five-plus times. Feeding it the block generator
    directly re-runs the whole alignment chain for every one of those, which is
    what the read counter here would show — and it is pure waste whenever the
    aligned volume fits the budget the caller already granted.
    """
    vol, samy, samz = _streamed_synthetic(nz=13)
    reference = A.align_volume(vol, samy, samz, scale_x=0.15, center_method=center_method)
    dset = _CountingDataset(vol)
    streamed = A.align_volume_streamed(
        dset, samy, samz, scale_x=0.15, center_method=center_method, budget_bytes=1 << 30
    )
    setup_reads = dset.reads
    rebuilt, n_blocks = _drain(streamed)
    assert n_blocks == 1  # the budget is generous, so one block — nothing is hidden
    assert dset.reads == expected_reads, (
        f"{center_method}: {dset.reads} passes over the source, expected {expected_reads}"
    )
    assert setup_reads == expected_reads - 1
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_streamed_rejects_unknown_center_method():
    vol, samy, samz = _streamed_synthetic(nz=5)
    with pytest.raises(ValueError):
        A.align_volume_streamed(
            vol, samy, samz, scale_x=0.15, center_method="bogus", budget_bytes=1 << 20
        )


def test_explicit_pad_defaults_reproduce_the_derived_pad():
    """`pad=None` must be byte-for-byte what the function did before."""
    vol, samy, _ = _streamed_synthetic(nz=9)
    derived = A.apply_samy_shifts_to_volume(vol, samy, 0.15)
    explicit = A.apply_samy_shifts_to_volume(
        vol,
        samy,
        0.15,
        pad=(A.compute_pad_left(samy, 0.15), A.compute_pad_right(samy, 0.15)),
    )
    assert np.array_equal(derived, explicit, equal_nan=True)
    wider = A.apply_samy_shifts_to_volume(vol, samy, 0.15, pad=(7, 5))
    assert wider.shape[2] == vol.shape[2] + 12  # the override really is honoured


def test_explicit_z_grid_defaults_reproduce_the_derived_grid():
    """`z_uniform=None` must be byte-for-byte what the function did before."""
    vol, _, samz = _streamed_synthetic(nz=9)
    derived, z_grid, scale_z = A.interpolate_to_uniform_z(vol, samz)
    explicit, z_out, _s = A.interpolate_to_uniform_z(vol, samz, z_uniform=z_grid)
    assert np.array_equal(derived, explicit, equal_nan=True)
    assert np.array_equal(z_grid, z_out)
    half = A.interpolate_to_uniform_z(vol, samz, z_uniform=z_grid[:3])[0]
    assert half.shape[0] == 3  # the override really is honoured
    assert np.array_equal(half, derived[:3], equal_nan=True)
    assert scale_z > 0


# -- raster -------------------------------------------------------------------
def _write_raw(folder, samy, samz):
    os.makedirs(folder, exist_ok=True)
    name = os.path.basename(folder)
    with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
        f.create_dataset("1.1/instrument/positioners/samy", data=samy)
        f.create_dataset("1.1/instrument/positioners/samz", data=samz)


def test_extract_motor_positions(tmp_path):
    folders = []
    for i, (y, z) in enumerate([(0.0, 0.0), (0.002, 0.001), (0.004, 0.003)]):
        folder = tmp_path / f"layer__{i + 1}"
        _write_raw(str(folder), y, z)
        folders.append(str(folder))
    # one folder without motor data -> skipped
    nomotor = tmp_path / "layer__bad"
    nomotor.mkdir()
    with h5py.File(nomotor / "layer__bad.h5", "w") as f:
        f.create_dataset("x", data=1)
    folders.append(str(nomotor))

    samy, samz, names = R.extract_motor_positions(folders)
    assert len(names) == 3 and names[0] == "layer__1"
    np.testing.assert_allclose(samy, [0.0, 0.002, 0.004])
    np.testing.assert_allclose(samz, [0.0, 0.001, 0.003])


def test_nearest_index():
    samy = np.array([0.0, 0.002, 0.004])
    samz = np.array([0.0, 0.001, 0.003])
    assert R.nearest_index(samy, samz, 0.0039, 0.0029) == 2
    assert R.nearest_index(samy, samz, 0.0, 0.0) == 0
    with pytest.raises(ValueError):
        R.nearest_index(np.array([]), np.array([]), 0, 0)
