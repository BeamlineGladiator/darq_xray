"""Tests for dfxm.stages.rocking — background subtraction, samz-union filtering,
mosa-anchored alignment, and the aligned-HDF5 schema the slicer consumes.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from dfxm.stages import rocking as RK

NF, H, W = 4, 6, 8  # frames, height, width


def _write_motor_folder(
    root, name, samy, samz, frames=None, detector_path="1.1/measurement/pco_ff"
):
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
        f.create_dataset("1.1/instrument/positioners/samy", data=samy)
        f.create_dataset("1.1/instrument/positioners/samz", data=samz)
        if frames is not None:
            f.create_dataset(detector_path, data=frames.astype(np.float32))
    return folder


def _rng_frames(seed):
    return np.random.default_rng(seed).standard_normal((NF, H, W)).astype(np.float32) + 10.0


# -- process_raw_scan ---------------------------------------------------------
def test_process_raw_scan_constant_background_cancels(tmp_path):
    """Identical frames -> background == each frame -> sum and specific are ~0."""
    folder = _write_motor_folder(
        str(tmp_path), "rock__1", 0.0, 0.0, frames=np.full((NF, H, W), 7.0)
    )
    h5p = os.path.join(folder, "rock__1.h5")
    sum_2d, spec_2d, n_frames, idx = RK.process_raw_scan(
        h5p, "1.1/measurement/pco_ff", None, None, None, normalize_sum=False
    )
    assert n_frames == NF and idx == NF // 2
    assert sum_2d.shape == (H, W)
    np.testing.assert_allclose(sum_2d, 0.0, atol=1e-5)
    np.testing.assert_allclose(spec_2d, 0.0, atol=1e-5)


def test_process_raw_scan_roi_and_normalize(tmp_path):
    folder = _write_motor_folder(str(tmp_path), "rock__1", 0.0, 0.0, frames=_rng_frames(0))
    h5p = os.path.join(folder, "rock__1.h5")
    plain, _, nf, _ = RK.process_raw_scan(h5p, "1.1/measurement/pco_ff", None, (1, 5), None, False)
    norm, _, _, _ = RK.process_raw_scan(h5p, "1.1/measurement/pco_ff", None, (1, 5), None, True)
    assert plain.shape == (4, W)  # ROI in Y -> 4 rows
    np.testing.assert_allclose(norm, plain / nf, rtol=1e-5)


# -- full run -----------------------------------------------------------------
def _setup(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # mosa reference: samy_ref = 0.0, samz_ref = 0.0; union start
    for i, z in enumerate([0.0, 0.001, 0.002]):
        _write_motor_folder(str(raw), f"mosa__{i + 1}", 0.0, z)
    # strain extends the samz union up to 0.003
    _write_motor_folder(str(raw), "strain__1", 0.0, 0.003)
    # rocking scans: two outside the [0, 0.003] union (excluded), three inside
    rock = [(-0.001, 0), (0.0, 1), (0.0015, 2), (0.003, 3), (0.005, 4)]
    for z, k in rock:
        _write_motor_folder(str(raw), f"rock__{k}", 0.0001 * k, z, frames=_rng_frames(k))
    return raw


def test_run_builds_aligned_volume(tmp_path):
    raw = _setup(tmp_path)
    out = tmp_path / "rock_out"
    res = RK.run(
        {
            "raw_root": str(raw),
            "rocking_pattern": "rock__*",
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "pixel_size_x_um": 0.152,
            "pixel_size_y_um": 0.385,
            "output_dir": str(out),
            "save_layers": True,
            "save_animation": False,
            "save_topview": False,
        }
    )
    # only the 3 in-union rocking scans are used
    assert res.n_layers_used == 3
    assert res.samy_reference_mm == 0.0 and res.samz_reference_mm == 0.0
    assert res.specific_frame_idx == NF // 2
    assert res.volume_shape[0] == 3  # uniform Z grid from samz [0, 1.5, 3.0] µm

    assert res.aligned_path and os.path.exists(res.aligned_path)
    with h5py.File(res.aligned_path, "r") as f:
        assert set(["sum_intensity", "specific_frame", "z_uniform_um"]).issubset(f.keys())
        assert f["sum_intensity"].shape == res.volume_shape
        assert f.attrs["scale_x_um_per_px"] == pytest.approx(0.152)
        assert f.attrs["scale_z_um_per_px"] > 0
        assert f.attrs["samy_reference_mm"] == pytest.approx(0.0)
        assert "pad_left_px" in f.attrs and "pad_right_px" in f.attrs
        assert f.attrs["specific_frame_idx"] == NF // 2

    # one product per volume, each with a layers dir of one PNG per Z layer
    assert {d.name for d in res.datasets} == {
        "raw_sum_intensity",
        f"raw_specific_frame_{NF // 2:03d}",
    }
    for d in res.datasets:
        pngs = [p for p in os.listdir(d.layers_dir) if p.endswith(".png")]
        assert len(pngs) == res.volume_shape[0]


def test_run_requires_mosa_reference(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_motor_folder(str(raw), "rock__1", 0.0, 0.0, frames=_rng_frames(1))
    with pytest.raises(ValueError, match="mosa"):
        RK.run({"raw_root": str(raw), "rocking_pattern": "rock__*", "mosa_pattern": "mosa__*"})


def test_process_raw_scan_no_background_subtraction(tmp_path):
    """subtract_background=False -> plain sum and raw specific frame (no median removed)."""
    frames = _rng_frames(3)
    folder = _write_motor_folder(str(tmp_path), "rock__1", 0.0, 0.0, frames=frames)
    h5p = os.path.join(folder, "rock__1.h5")
    sum_2d, spec_2d, n_frames, idx = RK.process_raw_scan(
        h5p,
        "1.1/measurement/pco_ff",
        None,
        None,
        None,
        normalize_sum=False,
        subtract_background=False,
    )
    np.testing.assert_allclose(sum_2d, frames.sum(axis=0), rtol=1e-5)
    np.testing.assert_allclose(spec_2d, frames[idx], rtol=1e-5)


def test_run_mosaicity_source_builds_mosa_volume(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    # three mosa layers, each with its own frame stack; these ARE the layers
    for i, z in enumerate([0.0, 0.001, 0.002]):
        _write_motor_folder(str(raw), f"mosa__{i + 1}", 0.0001 * i, z, frames=_rng_frames(i))
    res = RK.run(
        {
            "raw_root": str(raw),
            "source_scan": "mosaicity",
            "mosa_pattern": "mosa__*",
            "pixel_size_x_um": 0.152,
            "pixel_size_y_um": 0.385,
            "save_layers": False,
            "save_animation": False,
            "save_topview": False,
        }
    )
    assert res.n_layers_used == 3
    # default output auto-renamed so it never clobbers the rocking file
    assert res.aligned_path.endswith("aligned_raw_mosa_volumes.h5")
    assert os.path.exists(res.aligned_path)
    assert res.volume_shape[0] == 3
    # source-aware product title
    assert any(d.name == "raw_sum_intensity" for d in res.datasets)
    # figures() returns source-aware titles
    fig_params = {
        "source_scan": "mosaicity",
        "pixel_size_x_um": 0.152,
        "pixel_size_y_um": 0.385,
    }
    specs = RK.figures(res, fig_params)
    sum_titles = [s.title for s in specs if "sum_intensity" in s.figure_id]
    assert sum_titles and sum_titles[0] == "Mosa-integrated Sum Intensity — layer 0"


def test_figures_use_raw_group(tmp_path):
    """Rocking figure specs resolve their cmap from the style's raw group."""
    from dfxm.common.plotting import PlotStyle

    raw = _setup(tmp_path)
    out = tmp_path / "rock_out"
    params = {
        "raw_root": str(raw),
        "rocking_pattern": "rock__*",
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "pixel_size_x_um": 0.152,
        "pixel_size_y_um": 0.385,
        "output_dir": str(out),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
    }
    res = RK.run(params)
    specs = RK.figures(res, params)
    fig = specs[0].build(PlotStyle(cmap_raw="viridis"))
    assert fig.axes[0].images[0].cmap.name == "viridis"
    fig = specs[0].build(None)  # default raw group -> gray (was magma)
    assert fig.axes[0].images[0].cmap.name == "gray"


def test_oversize_volume_becomes_a_note(tmp_path, monkeypatch):
    """A volume wider than the GL 3-D texture limit renders blank silently."""
    import numpy as np

    monkeypatch.setattr(RK.R3, "save_top_view", lambda scene, path, **kw: path)
    monkeypatch.setattr(RK.R3, "volume_texture_limit", lambda *a, **kw: 4)
    p = {
        **RK.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": True,
        "volume_downsample": 1,  # opt out of the auto-fit: this pins the warning
    }
    res = RK.RockingResult(output_dir=str(tmp_path))
    RK._render(
        res,
        np.zeros((2, 4, 5)),
        np.arange(2.0),
        1.0,
        "sum_intensity",
        p,
        str(tmp_path),
        "gray",
        "t",
        "I",
    )
    assert any("texture limit" in n for n in res.datasets[0].notes)

    monkeypatch.setattr(RK.R3, "volume_texture_limit", lambda *a, **kw: 4096)
    res2 = RK.RockingResult(output_dir=str(tmp_path))
    RK._render(
        res2,
        np.zeros((2, 4, 5)),
        np.arange(2.0),
        1.0,
        "sum_intensity",
        p,
        str(tmp_path),
        "gray",
        "t",
        "I",
    )
    assert not res2.datasets[0].notes


# -- replot_catalog + render_replot -------------------------------------------


def _write_aligned(path):
    import h5py
    import numpy as np

    rng = np.random.default_rng(5)
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=rng.standard_normal((2, 4, 5)).astype(np.float32))
        f.create_dataset("specific_frame", data=rng.standard_normal((2, 4, 5)).astype(np.float32))
        f.create_dataset("z_uniform_um", data=np.arange(2, dtype=np.float32))
        f.attrs["scale_x_um_per_px"] = 0.152
        f.attrs["scale_y_um_per_px"] = 0.385
    return path


def test_rocking_replot_catalog_lists_products(tmp_path):
    h5 = str(tmp_path / "aligned.h5")
    _write_aligned(h5)
    cat = RK.replot_catalog(h5)
    keys = {g.key for g in cat}
    assert keys == {"sum_intensity", "specific_frame"}
    assert all(g.shape == (4, 5) for g in cat)  # (Y, X) of the stored layer — ROI hint


def test_rocking_render_replot_writes_pngs_with_clim(tmp_path):
    import os

    h5 = str(tmp_path / "aligned.h5")
    _write_aligned(h5)
    out = str(tmp_path / "replots")
    written = RK.render_replot(
        h5,
        [("sum_intensity", None)],
        style=None,
        clim=(0.0, 2.0),
        out_dir=out,
    )
    assert len(written) == 2
    assert all(os.path.exists(p) for p in written)


# -- F1: blank-clim replot uses percentile scaling, not raw min/max -----------


def _write_aligned_with_hot_pixel(path):
    """Volume where one cell is a 1000-unit outlier; the rest are ~N(0,1)."""
    rng = np.random.default_rng(99)
    vol = rng.standard_normal((2, 8, 8)).astype(np.float32)
    vol[0, 0, 0] = 1000.0  # hot pixel
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=vol)
        f.create_dataset("z_uniform_um", data=np.array([0.0, 1.0], dtype=np.float32))
    return path, vol


def test_rocking_replot_default_clim_uses_percentile(tmp_path):
    """_replot_default_clim must clip the hot-pixel outlier via percentile, not raw min/max."""
    from dfxm.common.plotting import apply_round_clim

    h5 = str(tmp_path / "hot.h5")
    _, vol = _write_aligned_with_hot_pixel(h5)

    with h5py.File(h5, "r") as f:
        got_vmin, got_vmax = RK._replot_default_clim(f["sum_intensity"], {}, style=None)

    # Must NOT equal the raw max (1000)
    assert got_vmax < float(vol.max()), "blank-clim replot must use percentile, not raw max"

    # Must match _colorbar_range + apply_round_clim with the stage's default percentiles
    defaults = RK.STAGE.defaults()
    exp_vmin, exp_vmax = RK._colorbar_range(vol, defaults["cbar_pct_lo"], defaults["cbar_pct_hi"])
    exp_vmin, exp_vmax, _ = apply_round_clim(exp_vmin, exp_vmax, None)
    assert abs(got_vmin - exp_vmin) < 1e-5
    assert abs(got_vmax - exp_vmax) < 1e-5


# -- F2: blank-clim replot titles are source-aware ----------------------------


def test_rocking_replot_title_is_source_aware(tmp_path):
    """render_replot must pass the source-aware run title (not the generic one) to the renderer."""
    from unittest.mock import patch

    h5 = str(tmp_path / "aligned.h5")
    _write_aligned(h5)
    out = str(tmp_path / "replots_f2")

    captured: list[str] = []

    def _capture_title(*args, title, **kwargs):
        captured.append(title)
        return None  # skip rendering; render_replot will skip None figures

    with patch("dfxm.stages.rocking.render_volume_layer", side_effect=_capture_title):
        RK.render_replot(
            h5,
            [("sum_intensity", [0])],
            style=None,
            clim=None,
            out_dir=out,
            params={"source_scan": "mosaicity"},
        )

    assert len(captured) == 1
    expected_title = RK._sum_title("mosaicity")
    assert captured[0] == expected_title, f"got {captured[0]!r}, want {expected_title!r}"


# -- the replot clim is streamed, and the colours must not move ---------------


def _count_clim_blocks(monkeypatch):
    """Record how many blocks each `dataset_blocks` traversal yielded.

    Returned list stays empty if `_replot_default_clim` never streams at all,
    which is what a reverted `dataset[:]` looks like — so the precondition
    assertion below cannot pass vacuously.
    """
    from dfxm.common import volumeio

    # Undo any earlier patch first: a test that calls this once per budget would
    # otherwise wrap the previous wrapper, leaving every earlier `seen` list
    # still recording and making `min(seen)` the smallest count of the WHOLE
    # sweep instead of that budget's. (Observed: two different budgets both
    # reporting 10 blocks.)
    monkeypatch.undo()
    real = volumeio.dataset_blocks
    seen: list[int] = []

    def counting(dset, **kwargs):
        n = 0
        for block in real(dset, **kwargs):
            n += 1
            yield block
        seen.append(n)

    monkeypatch.setattr(volumeio, "dataset_blocks", counting)
    return seen


def _write_clim_volume(path, *, nan_fraction_cut=1.7):
    """A float32 volume with NaNs, both infinities and exact ties, written to *path*.

    The non-finite values are load-bearing, not decoration. `_colorbar_range`
    selects with `np.isfinite` while `volumeio.stream_quantile` drops non-finite
    values by construction; a fixture carrying **only NaNs** cannot tell those
    two selections apart, because `~np.isnan` keeps exactly what `isfinite`
    keeps on it. This fixture was NaN-only, and reverting `_colorbar_range` to
    `~np.isnan` left every test in this module green — the rung boundary was
    pinned by nothing. `±inf` is what separates them, and it is a real value
    here: a pathological darfix fit puts one in a rocking volume, and then the
    in-core rung returns `vmax = inf` (every finite voxel one colour) where the
    streaming rung returns a finite limit — the same data rendering differently
    on two machines.

    Rounding onto a coarse grid puts long runs of **exact ties** around every
    rank, including the 1st and 99th percentiles the replot asks for, which is
    where `stream_quantile`'s rank search and `np.percentile` could disagree
    about which of the equal values they return.

    `_assert_finite_selections_disagree` asserts the first property rather than
    assuming it, so the fixture cannot quietly lose its subject again.
    """
    rng = np.random.default_rng(4)
    volume = (rng.normal(size=(20, 16, 16)) * 1000.0).astype(np.float32)
    volume[volume > nan_fraction_cut * 1000.0] = np.nan
    volume = (np.round(volume / 50.0) * 50.0).astype(np.float32)
    flat = volume.reshape(-1)
    finite_idx = np.flatnonzero(np.isfinite(flat))
    picked = rng.choice(finite_idx, size=320, replace=False)
    flat[picked[:160]] = np.inf
    flat[picked[160:]] = -np.inf
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=volume)
    return volume


def _assert_finite_selections_disagree(volume, lo, hi):
    """`isfinite` and `~isnan` must give different limits on *volume*.

    Without this the rung equality below compares two selections that happen to
    coincide, and reverting `_colorbar_range`'s `np.isfinite` to `~np.isnan`
    would leave it green — which is exactly how this test spent the wave not
    testing anything.
    """
    finite = np.percentile(volume[np.isfinite(volume)], [lo, hi])
    with np.errstate(invalid="ignore"):  # `inf - inf` in numpy's interpolation
        not_nan = np.percentile(volume[~np.isnan(volume)], [lo, hi])
    assert not np.array_equal(finite, not_nan), (
        "the fixture no longer separates `np.isfinite` from `~np.isnan` — they give the "
        f"same limits {tuple(finite)} on it, so reverting `_colorbar_range` to `~np.isnan` "
        "would leave the rung equality below green. Put ±inf back into the fixture."
    )


def test_rocking_replot_clim_is_exactly_the_in_core_percentile(tmp_path, monkeypatch):
    """The streamed percentile must equal the whole-volume one BIT for bit.

    `_colorbar_range` is the in-core form the replot used before it streamed, so
    comparing against it here is comparing against the colours every existing
    PNG was rendered with. `approx` would hide exactly the drift this must not
    have.
    """
    h5p = str(tmp_path / "clim.h5")
    volume = _write_clim_volume(h5p)
    finite = float(np.isfinite(volume).mean())
    assert 0.5 < finite < 1.0, f"fixture must be mostly-finite WITH NaNs present, got {finite:.3f}"

    defaults = RK.STAGE.defaults()
    expected = RK._colorbar_range(volume, defaults["cbar_pct_lo"], defaults["cbar_pct_hi"])

    seen = _count_clim_blocks(monkeypatch)
    with h5py.File(h5p, "r") as f:
        got = RK._replot_default_clim(f["sum_intensity"], {}, None, budget_bytes=16 * 1024)

    assert seen, "_replot_default_clim must stream the volume, not load it"
    assert min(seen) >= 5, f"the budget must have split the volume, got block counts {seen}"
    assert got == expected, f"replot colours moved: {got!r} != in-core {expected!r}"


def test_rocking_replot_clim_is_budget_independent_across_both_rungs(tmp_path, monkeypatch):
    """Same volume, four budgets, both rungs, one answer.

    Which rung runs depends on how much memory the machine has, so a colour that
    differed between them would be a colour that depended on the machine.
    Asserts **both** rungs are actually taken — without that this compares a run
    against itself, the vacuity Task 10 recorded.

    It also asserts the fixture can *see* the two rungs' finite selections
    differ. The in-core rung's `data[np.isfinite(data)]` and the streaming
    rung's implicit non-finite drop agree on a NaN-only volume no matter which
    one the in-core side is written with, so on the fixture as it first stood
    this test passed with `_colorbar_range` reverted to `~np.isnan` — the exact
    defect this wave had already repaired in `visualize`.
    """
    h5p = str(tmp_path / "clim.h5")
    volume = _write_clim_volume(h5p)
    defaults = RK.STAGE.defaults()
    assert np.isinf(volume).any() and np.isnan(volume).any(), "fixture lost its non-finite values"
    _assert_finite_selections_disagree(volume, defaults["cbar_pct_lo"], defaults["cbar_pct_hi"])

    answers = []
    traversals = []  # [] when the in-core rung ran, [n_blocks, ...] when it streamed
    for budget in (4 * 1024, 24 * 1024, 1 << 20, 64 << 20):
        seen = _count_clim_blocks(monkeypatch)
        with h5py.File(h5p, "r") as f:
            answers.append(
                RK._replot_default_clim(f["sum_intensity"], {}, None, budget_bytes=budget)
            )
        traversals.append(seen)

    streamed = [min(s) for s in traversals if s]
    assert streamed, "no budget took the streaming rung"
    assert any(not s for s in traversals), "no budget took the in-core rung"
    assert len(set(streamed)) > 1, (
        f"the streaming budgets must have changed the blocking, got {streamed}"
    )
    assert len(set(answers)) == 1, f"colour limits moved with the budget/rung: {answers}"


def test_rocking_replot_clim_takes_the_in_core_rung_when_it_fits(tmp_path, monkeypatch):
    """A volume that fits the budget must NOT stream.

    Streaming costs ~12 traversals against one, and `compose/adapters.py` calls
    this per panel on an interactive figure-builder preview — so taking the slow
    rung on a volume that fits is the defect, not a missed optimisation.
    """
    h5p = str(tmp_path / "clim.h5")
    _write_clim_volume(h5p)
    with h5py.File(h5p, "r") as f:
        dset = f["sum_intensity"]
        assert RK._fits_in_core(dset, RK.REPLOT_CLIM_WORKING_SET_BYTES), (
            "precondition: this fixture must fit the shipped budget"
        )
        seen = _count_clim_blocks(monkeypatch)
        RK._replot_default_clim(dset, {}, None)  # the shipped default budget
        assert seen == [], f"a fitting volume streamed anyway ({seen} blocks)"


def test_rocking_fits_in_core_bounds_the_measured_working_set():
    """`3 * itemsize + 1` must not sit below what the in-core percentile costs."""
    import gc
    import tracemalloc

    rng = np.random.default_rng(2)
    for dtype, bound_per_elem in (("float32", 13), ("float64", 25), ("uint16", 7)):
        vol = (rng.normal(size=(20, 96, 96)) * 100.0).astype(dtype)
        gc.collect()
        tracemalloc.start()
        base = tracemalloc.get_traced_memory()[0]
        RK._colorbar_range(vol, 1.0, 99.0)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        itemsize = np.dtype(dtype).itemsize
        # `vol` is untraced (allocated before start), so add its own bytes back.
        measured = (peak - base) / vol.size + itemsize
        assert measured <= bound_per_elem, (
            f"{dtype}: the in-core percentile costs {measured:.2f} B/element, over the "
            f"3*itemsize+1 = {bound_per_elem} B that _fits_in_core charges it"
        )


@pytest.mark.parametrize("budget,rung", [(None, "in-core"), (256, "streaming")])
def test_rocking_replot_clim_falls_back_for_an_all_nan_volume(tmp_path, budget, rung):
    """No finite voxel anywhere -> (0.0, 1.0) on BOTH rungs.

    The two signal it differently — `_colorbar_range` returns `(0.0, 1.0)`,
    `stream_quantile` returns NaN — so the conversion between them is a place
    the rungs could silently diverge.
    """
    h5p = str(tmp_path / "nan.h5")
    with h5py.File(h5p, "w") as f:
        f.create_dataset("sum_intensity", data=np.full((8, 16, 16), np.nan, dtype=np.float32))
    with h5py.File(h5p, "r") as f:
        dset = f["sum_intensity"]
        assert RK._fits_in_core(
            dset, RK.REPLOT_CLIM_WORKING_SET_BYTES if budget is None else budget
        ) is (rung == "in-core"), f"precondition: this budget must take the {rung} rung"
        assert RK._replot_default_clim(dset, {}, None, budget_bytes=budget) == (0.0, 1.0)


def test_rocking_render_replot_skips_the_percentile_when_both_limits_are_given(tmp_path):
    """A typed vmin/vmax must not pay for a colour range that is then discarded.

    `_replot_default_clim` used to run before `resolve_clim`, so an explicit
    limit cost a full percentile pass anyway — 12 traversals and 3.55 s on a
    23 MB volume, for a number `_apply_clim` immediately threw away.
    """
    from unittest.mock import patch

    h5 = str(tmp_path / "aligned.h5")
    _write_aligned(h5)
    out = str(tmp_path / "replots_skip")

    with patch.object(RK, "_replot_default_clim", wraps=RK._replot_default_clim) as spy:
        RK.render_replot(h5, [("sum_intensity", [0])], None, {"sum_intensity": (0.0, 2.0)}, out)
        assert spy.call_count == 0, "both limits supplied — the percentile must not run"

        # One blank side still needs it: `_apply_clim` keeps the default there.
        RK.render_replot(h5, [("sum_intensity", [0])], None, {"sum_intensity": (0.0, None)}, out)
        assert spy.call_count == 1, (
            "a half-open override still needs the default for its blank side"
        )

        RK.render_replot(h5, [("sum_intensity", [0])], None, None, out)
        assert spy.call_count == 2, "no override at all must still compute the default"


def test_rocking_render_replot_honours_a_supplied_clim_unchanged(tmp_path):
    """Skipping the percentile must not change which limits reach the renderer."""
    from unittest.mock import patch

    h5 = str(tmp_path / "aligned.h5")
    _write_aligned(h5)
    seen: list[tuple] = []

    def _capture(*args, vmin, vmax, clim, **kwargs):
        seen.append((vmin, vmax, clim))
        return None

    with patch("dfxm.stages.rocking.render_volume_layer", side_effect=_capture):
        RK.render_replot(
            h5, [("sum_intensity", [0])], None, {"sum_intensity": (-3.0, 7.0)}, str(tmp_path / "o")
        )
    assert seen == [(-3.0, 7.0, (-3.0, 7.0))]


def test_rocking_clim_block_budget_never_buys_more_than_the_budget(tmp_path):
    """The conversion rounds DOWN: a block must never cost more working set than asked."""
    h5p = str(tmp_path / "clim.h5")
    _write_clim_volume(h5p)
    with h5py.File(h5p, "r") as f:
        dset = f["sum_intensity"]
        itemsize = dset.dtype.itemsize
        for budget in (1 << 12, 1 << 16, 1 << 24):
            block_bytes = RK._clim_block_budget(dset, budget)
            working_set = block_bytes / itemsize * (itemsize + RK.QUANTILE_WORKING_SET_PER_ELEMENT)
            assert working_set <= budget, (
                f"budget {budget} bought a {working_set:.0f} B working set"
            )


def test_rocking_volume_downsample_auto_fits_the_texture_limit(tmp_path, monkeypatch):
    """Default (0): coarsen to fit rather than save a blank top view."""
    import numpy as np

    captured = {}

    def fake_top(scene, path, **kw):
        captured["scene"] = scene
        return path

    monkeypatch.setattr(RK.R3, "save_top_view", fake_top)
    monkeypatch.setattr(RK.R3, "volume_texture_limit", lambda *a, **kw: 4)
    p = {**RK.STAGE.defaults(), "save_layers": False, "save_animation": False, "save_topview": True}
    res = RK.RockingResult(output_dir=str(tmp_path))
    RK._render(
        res,
        np.zeros((2, 4, 5)),
        np.arange(2.0),
        1.0,
        "sum_intensity",
        p,
        str(tmp_path),
        "gray",
        "t",
        "I",
    )
    assert captured["scene"].downsample == 2
    assert any("coarsened 2x" in n for n in res.datasets[0].notes)
    assert not any("BLANK" in n for n in res.datasets[0].notes)
