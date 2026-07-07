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
