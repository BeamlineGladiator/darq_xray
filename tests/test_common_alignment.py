"""Tests for dfxm.common.alignment and dfxm.common.raster.

The alignment primitives are checked voxel-for-voxel against the legacy PVTI
exporter (export_aligned_volumes_to_paraview_v6_pvti) so the two stay
interchangeable in ParaView world coordinates.
"""

from __future__ import annotations

import os
import sys
import tracemalloc
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm.common import alignment as A
from dfxm.common import raster as R
from dfxm.common import volumeio


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
@pytest.mark.parametrize("divisor", [1, 2, 6, 10_000])
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
    # The divisor has to scale the part of the working set that BLOCKING can
    # change, so the budget is the centring pass's blocking-independent
    # scaffolding plus a fraction of the rest. Fractions of the whole thing put
    # the three centring modes on three different blockings for the same divisor
    # (the median's histogram alone is 2.1 MB against this fixture's 0.85 MB of
    # data), which would make the expected counts below mode-dependent for a
    # reason that has nothing to do with what the test measures. Earlier
    # versions used fractions of the input volume (every divisor floored) and
    # then of the output volume (divisors 3 and 8 collapsed together) — neither
    # is the same currency as the budget.
    whole = A.align_volume_streamed(vol, samy, samz, budget_bytes=None, **kwargs)
    scaffold = (
        volumeio.centring_scaffold_bytes(center_method, int(np.prod(whole.shape)))[0]
        if center_method
        else 0
    )
    budget = scaffold + max(1, (whole.working_set_bytes - scaffold) // divisor)
    streamed = A.align_volume_streamed(vol, samy, samz, budget_bytes=budget, **kwargs)
    assert streamed.shape == reference.data.shape
    assert streamed.pad_left == reference.pad_left
    assert np.array_equal(streamed.z_uniform_um, reference.z_uniform_um)
    assert streamed.scale_z_um == reference.scale_z_um
    assert streamed.center_offset == reference.center_offset
    rebuilt, n_blocks = _drain(streamed)
    # The measurement must be live: each budget has to produce the blocking it
    # is meant to, or this case is silently comparing one whole-volume pass
    # against another. The four counts are pairwise distinct by construction —
    # 1, 3, 14 and the 27-layer floor — so no two parametrisations can quietly
    # become the same test, and they are the same for all three centring modes,
    # so a mode cannot quietly stop streaming either.
    expected_blocks = {1: 1, 2: 3, 6: 14, 10_000: reference.data.shape[0]}[divisor]
    assert n_blocks == expected_blocks, f"budget/{divisor} gave {n_blocks} blocks"
    # The advertised block size must be the one the stream actually used.
    assert n_blocks == -(-reference.data.shape[0] // streamed.block_layers)
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_streamed_blocks_factory_can_be_traversed_twice():
    """A second traversal must reproduce the first, block boundaries included.

    Budgeted in working-set currency, like the parity tests above. Sized from
    ``vol.nbytes`` — what this did before — the run landed on the one-layer
    floor and emitted the production overrun warning on every pass, so what it
    actually re-traversed was 27 single-layer blocks. The blocking is asserted
    rather than assumed, so it cannot collapse to the floor again without going
    red.
    """
    vol, samy, samz = _streamed_synthetic()
    whole = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=None)
    streamed = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, budget_bytes=whole.working_set_bytes // 3
    )
    first_blocks = [b for _sl, b in streamed.blocks()]
    assert (streamed.block_layers, len(first_blocks)) == (7, 4)
    first = np.concatenate(first_blocks, axis=0)
    second = np.concatenate([b for _sl, b in streamed.blocks()], axis=0)
    assert np.array_equal(first, second, equal_nan=True)


def test_streamed_shape_known_before_reading():
    """A writer must be able to size its output before a voxel is read."""
    vol, samy, samz = _streamed_synthetic()
    streamed = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, roi_x=(2, 7), budget_bytes=64)
    assert streamed.shape[1] == 6  # full Y
    assert streamed.shape[2] == 5 + streamed.pad_left + streamed.pad_right


def test_block_samy_slice_does_not_shrink_the_canvas():
    """The trap: a per-block samy slice implies a narrower pad than the global one.

    A **multi-layer** block is the case that matters — a one-layer block's samy
    is a single value, which the pad arithmetic cannot get subtly wrong, only
    completely. Sized from ``vol.nbytes`` this test ran 29 blocks of one layer
    and asserted only ``n_blocks > 1``, so it held while covering nothing.
    Working-set currency and an asserted blocking, as elsewhere.
    """
    vol, samy, samz = _streamed_synthetic(nz=21)
    samy = samy + np.linspace(0, 0.05, len(samy))  # a strong monotone drift
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    whole = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=None)
    streamed = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, budget_bytes=whole.working_set_bytes // 4
    )
    assert streamed.shape[2] == reference.data.shape[2]
    rebuilt, n_blocks = _drain(streamed)
    assert (streamed.block_layers, n_blocks) == (6, 5)
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_block_samz_slice_does_not_shift_the_z_grid():
    """The Z-grid twin of the pad trap: a block's samz implies a different grid.

    Same correction as the pad trap above: a one-layer block cannot exercise a
    per-block grid (its samz is one value), and ``vol.nbytes // 7`` floored this
    to 26 such blocks. Working-set currency, with the blocking asserted.
    """
    vol, samy, samz = _streamed_synthetic(nz=19)
    samz = np.sort(samz) * 3.0  # widen the Z span so a block's own grid differs loudly
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    whole = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=None)
    streamed = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, budget_bytes=whole.working_set_bytes // 4
    )
    assert streamed.shape[0] == reference.data.shape[0]
    assert np.array_equal(streamed.z_uniform_um, reference.z_uniform_um)
    rebuilt, n_blocks = _drain(streamed)
    assert (streamed.block_layers, n_blocks) == (5, 6)
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


@pytest.mark.parametrize(("divisor", "expected_blocks"), [(1, 1), (2, 3), (4, 6), (10_000, 23)])
def test_streamed_matches_in_core_for_decreasing_samz(divisor, expected_blocks):
    """interp1d sorts internally, so a decreasing samz must stream identically.

    Budgets are fractions of the *blocking-dependent* working set, as in
    :func:`test_streamed_alignment_matches_in_core`. Sized from ``vol.nbytes``
    — what this test did before — every divisor landed on the one-layer floor
    once blocks were sized against the working set, so all three cases became
    the same one-layer stream and nothing here covered a **multi-layer** block
    on a decreasing samz. The blocking is asserted below so it cannot collapse
    that way again without going red.
    """
    vol, samy, samz = _streamed_synthetic(nz=15)
    samz = samz[::-1].copy()  # strictly decreasing
    kwargs = dict(scale_x=0.15, center_method="mean")
    reference = A.align_volume(vol, samy, samz, **kwargs)
    whole = A.align_volume_streamed(vol, samy, samz, budget_bytes=None, **kwargs)
    scaffold = volumeio.centring_scaffold_bytes("mean", int(np.prod(whole.shape)))[0]
    streamed = A.align_volume_streamed(
        vol,
        samy,
        samz,
        budget_bytes=scaffold + max(1, (whole.working_set_bytes - scaffold) // divisor),
        **kwargs,
    )
    rebuilt, n_blocks = _drain(streamed)
    assert n_blocks == expected_blocks, f"budget/{divisor} gave {n_blocks} blocks"
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def _nonmonotonic_synthetic(nz=17):
    """A samz that is *mostly* ascending but has two distant layers swapped.

    A fully shuffled samz cannot exercise this: `_median_z_step` takes the
    median ``|dz|`` of the layer order, which a shuffle makes large, so the
    uniform grid collapses to three or four layers and no blocking can carry
    more than one. Two swaps leave the median step alone — the grid keeps all
    `nz` layers — while still making `_input_span` degenerate: at `nz=17` the
    five-layer blocking reads spans of 14, 6, 12 and 3 input layers.
    """
    rng = np.random.default_rng(11)
    vol = rng.normal(size=(nz, 6, 9))
    vol[vol > 1.9] = np.nan
    samy = np.cumsum(rng.normal(scale=0.002, size=nz))
    samz = np.linspace(0.0, 0.001 * nz, nz)
    samz[[3, nz - 4]] = samz[[nz - 4, 3]]
    return vol, samy, samz


@pytest.mark.parametrize(
    ("divisor", "expected_blocks", "multi_layer_multi_block"),
    [(1, 1, False), (2, 4, True), (10_000, 17, False)],
)
def test_streamed_matches_in_core_for_nonmonotonic_samz(
    divisor, expected_blocks, multi_layer_multi_block
):
    """A non-monotonic samz is degenerate but must not produce a *wrong* answer.

    The ``/2`` case is the one that matters and the one that had gone missing:
    a **multi-layer block in a multi-block stream**, on a samz whose input span
    is nothing like its output block. `_input_span`'s handling of exactly that
    was wrong once already in this wave, and after the working-set change every
    parametrisation of the old shuffled fixture collapsed to one layer per
    block — so the guarantee was unguarded. The preconditions are asserted, not
    assumed: the samz really is non-monotonic, the span really is wider than the
    block, and the blocking really is the one named.
    """
    vol, samy, samz = _nonmonotonic_synthetic()
    assert not np.all(np.diff(samz) >= 0), "fixture is supposed to be non-monotonic"
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    whole = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=None)
    streamed = A.align_volume_streamed(
        vol, samy, samz, scale_x=0.15, budget_bytes=max(1, whole.working_set_bytes // divisor)
    )
    rebuilt, n_blocks = _drain(streamed)
    assert n_blocks == expected_blocks, f"budget/{divisor} gave {n_blocks} blocks"
    if multi_layer_multi_block:
        # The degenerate read is the point: a block of `block_layers` outputs
        # that pulls in far more input layers than that — and more than one
        # such block, so a block boundary is crossed inside the degeneracy.
        assert streamed.block_layers > 1, "this case must carry a multi-layer block"
        assert n_blocks > 1, "and more than one of them"
        z_um, z_uniform, _s = A._z_grid(samz, vol.shape[0], None)
        spans = [
            A._input_span(z_um, z_uniform[s : s + streamed.block_layers])
            for s in range(0, len(z_uniform), streamed.block_layers)
        ]
        assert max(hi - lo for lo, hi in spans) > 2 * streamed.block_layers, (
            "the span never degenerated, so this is not testing the degenerate path"
        )
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


def test_streamed_block_does_not_retain_its_input_span():
    """The block generator must not pin its raw read across the ``yield``.

    A generator frame stays alive between yields, so a span still bound there is
    retained for as long as the consumer holds the block. That is invisible with
    one stream and expensive with several — `paraview` keeps one open per field,
    and four suspended frames each pinning their own span cost four whole input
    volumes on top of the four blocks. Asserted by weak reference rather than by
    measuring memory, so it cannot pass by being too small to notice.
    """
    import weakref

    vol = np.arange(6 * 4 * 5, dtype=np.float64).reshape(6, 4, 5)
    reads: list = []

    class _Dset:
        """Duck-typed like an h5py dataset: every slice is a fresh array."""

        shape = vol.shape
        dtype = vol.dtype

        def __getitem__(self, item):
            out = np.array(vol[item])
            reads.append(weakref.ref(out))
            return out

    streamed = A.align_volume_streamed(
        _Dset(), np.zeros(6), np.linspace(0.0, 0.005, 6), scale_x=0.15, budget_bytes=None
    )
    blocks = streamed.blocks()
    _zsl, block = next(blocks)  # the generator is now suspended at the yield
    assert reads, "the stream read nothing — this test would measure nothing"
    assert block.shape[1:] == streamed.shape[1:]
    alive = [i for i, ref in enumerate(reads) if ref() is not None]
    assert not alive, f"input span(s) {alive} of {len(reads)} still alive at the yield"
    del blocks


@pytest.mark.parametrize("take_abs", [False, True])
def test_streamed_matches_in_core_float32(take_abs):
    """Stage volumes are float32 on disk; the chain upcasts and both paths must agree.

    The upcast happens *inside* the interpolator, so what this has to cover is a
    block wide enough for the interpolator to run over — one layer takes
    :func:`interpolate_to_uniform_z`'s early return instead, which hands the
    layer back in its own dtype and never exercises the upcast at all.
    ``vol.nbytes // 6`` floored to nine one-layer blocks and so tested exactly
    that early return nine times. Budgeted in working-set currency (the median's
    2.1 MB of blocking-independent scaffolding dwarfs this fixture's data, hence
    the ``scaffold +`` term), with the blocking asserted.
    """
    vol, samy, samz = _streamed_synthetic(nz=11)
    vol = vol.astype(np.float32)
    kwargs = dict(scale_x=0.15, take_abs=take_abs, center_method="median")
    reference = A.align_volume(vol, samy, samz, **kwargs)
    whole = A.align_volume_streamed(vol, samy, samz, budget_bytes=None, **kwargs)
    scaffold = volumeio.centring_scaffold_bytes("median", int(np.prod(whole.shape)))[0]
    streamed = A.align_volume_streamed(
        vol,
        samy,
        samz,
        budget_bytes=scaffold + max(1, (whole.working_set_bytes - scaffold) // 2),
        **kwargs,
    )
    assert streamed.dtype == reference.data.dtype
    rebuilt, n_blocks = _drain(streamed)
    assert (streamed.block_layers, n_blocks) == (3, 3)
    assert np.array_equal(rebuilt, reference.data, equal_nan=True)


def test_streamed_median_uses_scratch_when_the_volume_will_not_fit(tmp_path):
    """The cached-median branch must give the same answer as the re-run branch.

    "Will not fit" is exactly ``block_layers < shape[0]`` — the solved blocking,
    not a comparison of output bytes against the budget, which stopped meaning
    the same thing when blocks began to be sized against the whole working set.
    The old ``vol.nbytes // 9`` budget floored this to 27 one-layer blocks, so
    the cache it filled and read back was 27 single layers; the blocking is
    asserted below in working-set currency so a multi-layer cache block is
    covered and cannot silently disappear again.
    """
    vol, samy, samz = _streamed_synthetic(nz=17)
    kwargs = dict(scale_x=0.15, center_method="median")
    whole = A.align_volume_streamed(vol, samy, samz, budget_bytes=None, **kwargs)
    scaffold = volumeio.centring_scaffold_bytes("median", int(np.prod(whole.shape)))[0]
    budget = scaffold + max(1, (whole.working_set_bytes - scaffold) // 4)
    plain = A.align_volume_streamed(vol, samy, samz, budget_bytes=budget, **kwargs)
    cached = A.align_volume_streamed(
        vol, samy, samz, budget_bytes=budget, scratch_dir=str(tmp_path), **kwargs
    )
    assert cached.center_offset == plain.center_offset
    assert np.array_equal(_drain(cached)[0], _drain(plain)[0], equal_nan=True)
    # The branch under test runs only when the stream cannot yield the whole
    # aligned volume in one block; assert that, so the test cannot go vacuous.
    assert (cached.block_layers, -(-cached.shape[0] // cached.block_layers)) == (5, 6)
    assert cached.block_layers < cached.shape[0]
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


def _peak_bytes(fn):
    """Peak traced allocation during *fn*, numpy's buffers included."""
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_working_set_model_is_pinned_to_the_scipy_it_was_derived_from():
    """An `interp1d` upgrade must fail loudly, not under-price the budget quietly.

    `_WorkingSet.canvas_layer` and `.interp_layer` are counted from scipy's
    source: `interp1d.__init__` making a copy of `y` and then a permuted second
    one via `np.take`, and `_call_linear` holding five block-sized arrays. The
    fragility is asymmetric. If scipy *removes* `interp1d` (it is legacy
    upstream) the import fails loudly; if it allocates *fewer* temporaries the
    model merely leaves budget unspent. But one **extra** temporary silently
    under-prices every block, and the measurement tests carry ~20% headroom, so
    a small regression would slip through them. Hence a version assertion: an
    upgrade is a failing test and a re-derivation, not a quiet overrun.
    """
    import scipy

    assert scipy.__version__ == "1.11.4", (
        f"the working-set model was derived from scipy 1.11.4, found "
        f"{scipy.__version__}. Re-read `interp1d.__init__` (does it still copy "
        "`y` and then `np.take` a permuted second copy, unconditionally?) and "
        "`_call_linear` (still five block-sized arrays at its peak?), then "
        "re-measure `test_streamed_peak_stays_within_the_budget` before "
        "bumping this."
    )


def _budget_fixture(samz_kind="ascending", nz=40):
    """A volume big enough that allocation noise is negligible against its blocks."""
    rng = np.random.default_rng(5)
    vol = rng.normal(size=(nz, 200, 260)).astype(np.float32)
    samy = np.cumsum(rng.normal(scale=0.0005, size=nz))
    samz = np.sort(rng.normal(scale=0.01, size=nz))
    if samz_kind == "descending":
        samz = samz[::-1].copy()
    elif samz_kind == "shuffled":
        samz = samz[rng.permutation(nz)]
    return vol, samy, samz


@pytest.mark.parametrize(
    ("samz_kind", "center_method", "divisor", "floored", "scratch"),
    [
        ("ascending", None, 4, False, False),
        ("ascending", None, 16, False, False),
        ("ascending", "mean", 8, False, False),
        ("ascending", "median", 4, False, False),
        ("descending", None, 4, False, False),
        # A samz that is not already ascending makes `interp1d.__init__` sort,
        # and its `np.take(y, ind)` holds a THIRD copy of the padded canvas.
        # Priced at two copies, this case measured 1.20x its budget and 1.13x
        # its own model. A shuffled samz reads nearly the whole input for any
        # block, so halving the budget cannot halve the working set: this one
        # lands on the one-layer floor, which is the point — the model still
        # has to bound the peak there.
        ("shuffled", None, 2, True, False),
        # The scratch-backed median: a distinct branch with a SECOND traversal
        # the budget has to cover — the quantile pass reading the cache back —
        # and it sat outside this matrix while it handed `budget_bytes` to
        # `dataset_blocks` raw. It measured 1.68x its budget at /4, 3.36x at /8
        # and 3.70x at /16 (1.78x, 3.56x and 2.69x its own reported working set),
        # rising as the budget shrank, which is precisely the signature the
        # working-set model exists to remove. Both a blocked case and a floored
        # one, because the floored one is where a 17 GB paraview volume lands.
        ("ascending", "median", 4, False, True),
        ("ascending", "median", 16, True, True),
    ],
)
def test_streamed_peak_stays_within_the_budget(
    samz_kind, center_method, divisor, floored, scratch, tmp_path
):
    """`budget_bytes` must bound the measured peak, not a fraction of it.

    The defect this pins: blocks used to be sized so the OUTPUT BLOCK fit the
    budget, while producing one costs the input span, the padded canvas, the
    interpolator's temporaries and the statistic's leftovers besides — measured
    at 3.6x to 7.7x the budget, and worse the tighter the budget was. A caller
    that hands over the machine's advised headroom has to get a run that fits
    in it.
    """
    nz = 24 if samz_kind == "shuffled" else 40
    vol, samy, samz = _budget_fixture(samz_kind, nz)
    kwargs = dict(scale_x=0.15, center_method=center_method)
    if scratch:
        kwargs["scratch_dir"] = str(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        whole = A.align_volume_streamed(vol, samy, samz, budget_bytes=None, **kwargs)
        budget = max(1, whole.working_set_bytes // divisor)
        streamed = A.align_volume_streamed(vol, samy, samz, budget_bytes=budget, **kwargs)
    # A floored case cannot promise the budget and says so; a non-floored one
    # must be genuinely blocked, or "within budget" would be asserted of a run
    # that never streamed.
    assert (streamed.working_set_bytes > budget) is floored
    if floored:
        assert streamed.block_layers == 1
    else:
        assert 1 < streamed.block_layers < whole.shape[0], "neither the floor nor one block"
    if scratch:
        # `fits` is exactly `block_layers >= shape[0]`, so this is the assertion
        # that the run took the scratch branch rather than the adopt-one-block
        # one — without it a generous blocking would silently move the case off
        # the branch the row was added to cover.
        assert streamed.block_layers < streamed.shape[0], "this row must use the scratch cache"

    def consume():
        # Iterate WITHOUT accumulating. `_drain` allocates the whole output
        # volume, which is the consumer's own choice and explicitly outside
        # what `budget_bytes` covers — measuring it here would be measuring the
        # test's array, not the chain's working set. The loop variable is still
        # held across each `next()`, which is the part the model does charge.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            blocks = A.align_volume_streamed(
                vol, samy, samz, budget_bytes=budget, **kwargs
            ).blocks()
        for _zsl, block in blocks:
            assert block is not None

    peak = _peak_bytes(consume)
    # Liveness: a real run allocates at least one output block, so a
    # tracemalloc that has stopped seeing numpy's buffers cannot pass this.
    one_block = streamed.block_layers * streamed.shape[1] * streamed.shape[2] * 8
    assert peak > one_block, f"tracemalloc saw only {peak} B; the measurement is dead"
    # The model has to bound the measurement whether or not the budget could be
    # met — that is what makes `working_set_bytes` worth reporting, and it is
    # the assertion the floored case carries.
    assert peak <= streamed.working_set_bytes, (
        f"peak {peak} B is {peak / streamed.working_set_bytes:.2f}x the modelled "
        f"{streamed.working_set_bytes} B working set"
    )
    if not floored:
        assert peak <= budget, (
            f"peak {peak} B is {peak / budget:.2f}x the {budget} B budget "
            f"({streamed.block_layers} layers/block)"
        )


def test_streamed_floors_at_one_layer_and_says_so():
    """Under a budget one layer cannot meet: run anyway, warn, report the truth."""
    vol, samy, samz = _budget_fixture()
    with pytest.warns(UserWarning, match="over the .* budget"):
        streamed = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=1024)
    assert streamed.block_layers == 1
    assert streamed.working_set_bytes > 1024  # the overrun is reported, not hidden
    reference = A.align_volume(vol, samy, samz, scale_x=0.15)
    assert np.array_equal(_drain(streamed)[0], reference.data, equal_nan=True)


def test_budget_none_is_one_block():
    """The in-core façade's request: one block, whatever it costs, and no warning."""
    vol, samy, samz = _streamed_synthetic(nz=13)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        streamed = A.align_volume_streamed(vol, samy, samz, scale_x=0.15, budget_bytes=None)
    assert streamed.block_layers == streamed.shape[0]
    assert _drain(streamed)[1] == 1


def test_median_cache_does_not_hold_a_second_aligned_volume():
    """The `fits` branch must ADOPT its one block, never copy it into a second array.

    ``fits`` is exactly ``out_step == nz``, so the stream yields one block that
    already is the whole aligned volume. Copying it into a ``np.empty(shape)``
    cache holds two at once — and since ``fits`` is true precisely when one
    aligned volume fits ``budget_bytes``, the duplicate puts the run at **twice
    the caller's budget**, on exactly the volume the branch exists to speed up.

    Measured, and dead stable across repeats and shapes: the copy costs exactly
    one extra aligned volume. Peak over the mean path's peak, in units of the
    aligned volume, is 1.27 adopting and 2.27 copying at this fixture's size
    (0.92 / 1.92 at another). The bound below sits between the two.
    """
    rng = np.random.default_rng(5)
    vol = rng.normal(size=(40, 200, 260)).astype(np.float32)
    samy = np.cumsum(rng.normal(scale=0.0005, size=40))
    samz = np.sort(rng.normal(scale=0.01, size=40))

    def streamed(center_method, budget):
        return A.align_volume_streamed(
            vol, samy, samz, scale_x=0.15, center_method=center_method, budget_bytes=budget
        )

    def peak(center_method, budget):
        return _peak_bytes(lambda: streamed(center_method, budget))

    # Each budget is the working set of a single-block run **of that same
    # centring mode**. Probing without `center_method` under-prices the median's
    # own pass (its statistic holds three block-sized arrays the uncentred path
    # never allocates), so the median call gets 43 of 70 layers, `fits` is
    # False, and the adopt branch never executes — the test then passes whether
    # the branch adopts or copies. It went vacuous exactly that way once, when
    # the block-sizing change moved `fits` underneath it; hence the assertion
    # below is made of the *median* call, not of a proxy.
    mean_budget = streamed("mean", None).working_set_bytes
    median_budget = streamed("median", None).working_set_bytes
    base = streamed(None, None)
    nz = base.shape[0]
    aligned = int(np.prod(base.shape)) * base.dtype.itemsize
    assert streamed("median", median_budget).block_layers == nz, (
        "the median call must be a single block, or the adopt branch under test "
        "never runs and this test measures nothing"
    )
    assert streamed("mean", mean_budget).block_layers == nz
    mean_peak = peak("mean", mean_budget)
    median_peak = peak("median", median_budget)

    # Liveness. If a future numpy stops routing its buffers through a traced
    # domain, both figures collapse to a few kB of Python objects and every
    # comparison below passes while measuring nothing. The fixture's aligned
    # volume is ~31 MB, so a live measurement cannot be under it.
    assert aligned > (8 << 20), "fixture too small for allocation noise to be negligible"
    assert mean_peak > aligned, (
        f"tracemalloc saw only {mean_peak} B for a {aligned} B volume — "
        "the measurement is dead, not the code under test"
    )

    assert median_peak - mean_peak < 1.5 * aligned, (
        f"median peak {median_peak} B vs mean {mean_peak} B — a difference of "
        f"{(median_peak - mean_peak) / aligned:.2f} aligned volumes ({aligned} B each). "
        "The median cache is holding a second full-size copy."
    )


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
