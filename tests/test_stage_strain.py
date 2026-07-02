"""Tests for dfxm.stages.strain — ccmth-only axial strain (cot method): numeric
golden equivalence vs the legacy calc_axial_strain_v7_batch script (self-skips
when the legacy file is absent), an independent cot-formula check, and the
detrend-before-ROI ordering constraint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm.stages import strain as S

CCMTH_PATH = "/entry/ccmth/Center of mass/Center of mass"


def _legacy(modname: str):
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / f"{modname}.py").exists():
        pytest.skip(f"legacy {modname}.py not found")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return __import__(modname)


def _synthetic_ccmth(ny=40, nx=60, seed=1):
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-2, 2, ny))
    return (
        7.144
        + 0.002 * np.arctan(2 * X)
        + 0.001 * np.arctan(1.5 * Y)
        + 0.0001 * rng.standard_normal((ny, nx))
    )


def _write_maps(folder, ccmth):
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, "maps.h5"), "w") as f:
        f.create_dataset(CCMTH_PATH, data=ccmth)


def test_detrend_matches_legacy():
    legacy = _legacy("calc_axial_strain_v7_batch")
    ccmth = _synthetic_ccmth()
    mine_dt, mine_surf = S.detrend_arctan_2d(ccmth.copy())
    leg_dt, leg_surf = legacy.detrend_arctan_2d(ccmth.copy())
    np.testing.assert_allclose(mine_dt, leg_dt, atol=1e-12)
    np.testing.assert_allclose(mine_surf, leg_surf, atol=1e-12)


def test_compute_strain_ccmth_only_matches_legacy_v7():
    legacy = _legacy("calc_axial_strain_v7_batch")
    ccmth = _synthetic_ccmth()
    dt, _ = S.detrend_arctan_2d(ccmth.copy())
    np.testing.assert_allclose(
        S.compute_strain(dt, 7.144), legacy.compute_strain(dt, 7.144), atol=1e-15
    )


def test_compute_strain_is_cot_ccmth():
    """compute_strain returns a single ndarray (not a tuple), shape-preserved,
    equal to cot(ref)·Δ — with an independent hand-computed spot-check."""
    import math

    ccmth = _synthetic_ccmth()
    dt, _ = S.detrend_arctan_2d(ccmth.copy())
    out = S.compute_strain(dt, 7.144)
    assert isinstance(out, np.ndarray)
    assert out.shape == dt.shape

    # independent spot-check (math, not S.cot/np.deg2rad) for a known scalar
    ref_deg, delta_deg = 7.144, 0.001
    ref_rad = math.radians(ref_deg)
    expected = (math.cos(ref_rad) / math.sin(ref_rad)) * math.radians(delta_deg)
    spot = S.compute_strain(np.array([[ref_deg + delta_deg]]), ref_deg)[0, 0]
    np.testing.assert_allclose(spot, expected, rtol=1e-12)


def test_run_single_stacks_volume_and_writes_plots(tmp_path):
    ccmth = _synthetic_ccmth()
    folder = tmp_path / "layer__1"
    _write_maps(str(folder), ccmth)
    res = S.run(
        {
            "mode": "single",
            "input_folder": str(folder),
            "ccmth_ref_deg": 7.144,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert res.n_layers == 1 and res.volume_shape == (1, 40, 60)
    with h5py.File(res.stacked_path, "r") as f:
        vol = f["strain"][:]
        dt, _ = S.detrend_arctan_2d(ccmth.copy())
        np.testing.assert_allclose(vol[0], S.compute_strain(dt, 7.144), atol=1e-15)
        assert f.attrs["description"] == "Stacked 3D strain volume (cot, ccmth-only)"
    pngs = list((tmp_path / "out").glob("*.png"))
    assert any("strain" in p.name for p in pngs)
    assert not any("contributions" in p.name for p in pngs)


def test_run_batch_over_multiple_layers(tmp_path):
    ccmth = _synthetic_ccmth()
    root = tmp_path / "root"
    for name in ["layer__1", "layer__2"]:
        _write_maps(str(root / name), ccmth)
    res = S.run(
        {
            "mode": "batch",
            "root_folder": str(root),
            "folder_pattern": "layer__*",
            "ccmth_ref_deg": 7.144,
            "save_plots": False,
        }
    )
    assert res.n_layers == 2 and res.volume_shape == (2, 40, 60)


def test_batch_missing_maps_file_records_reason(tmp_path):
    root = tmp_path / "root"
    _write_maps(str(root / "layer__1"), _synthetic_ccmth())
    os.makedirs(root / "layer__2")  # matches the pattern but has no maps.h5
    res = S.run(
        {
            "mode": "batch",
            "root_folder": str(root),
            "folder_pattern": "layer__*",
            "ccmth_ref_deg": 7.144,
            "save_plots": False,
        }
    )
    assert res.n_layers == 1
    assert res.skipped == ["layer__2: maps.h5 not found"]


def test_detrend_runs_before_roi(tmp_path):
    """ROI must crop the detrended map, not detrend a pre-cropped map."""
    ccmth = _synthetic_ccmth()
    folder = tmp_path / "layer__1"
    _write_maps(str(folder), ccmth)
    roi = [5, 25, 10, 40]
    res = S.run(
        {
            "mode": "single",
            "input_folder": str(folder),
            "ccmth_ref_deg": 7.144,
            "roi": "5,25,10,40",
            "save_plots": False,
            "output_dir": str(tmp_path / "out"),
        }
    )
    dt_full, _ = S.detrend_arctan_2d(ccmth.copy())
    dt_crop = dt_full[roi[0] : roi[1], roi[2] : roi[3]]
    expected = S.compute_strain(dt_crop, 7.144)
    with h5py.File(res.stacked_path, "r") as f:
        np.testing.assert_allclose(f["strain"][0], expected, atol=1e-15)
        assert f["strain"].shape == (1, 20, 30)


def test_parse_helpers():
    assert S._parse_roi("") is None
    assert S._parse_roi("1,2,3,4") == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        S._parse_roi("1,2,3")
    assert S._parse_float("") is None
    assert S._parse_float("0.5") == 0.5


def test_build_strain_map_legacy_has_no_scale_bar():
    fig = S.build_strain_map(np.random.rand(20, 30) * 1e-3, 0.1, 0.3, None, (None, None))
    ax = fig.axes[0]
    assert len(ax.patches) == 0  # today's strain map has no scale bar


def test_build_strain_map_style_adds_scale_bar():
    from dfxm.common.plotting import PlotStyle

    fig = S.build_strain_map(
        np.random.rand(20, 30) * 1e-3,
        0.1,
        0.3,
        None,
        (None, None),
        style=PlotStyle(scale_bar=True),
    )
    assert len(fig.axes[0].patches) >= 1


def test_build_strain_histogram_returns_none_on_empty_data():
    assert S.build_strain_histogram(np.full((5, 5), np.nan)) is None


def test_build_strain_histogram_styled_returns_figure():
    from matplotlib.figure import Figure

    from dfxm.common.plotting import PlotStyle

    data = np.random.rand(10, 10) * 1e-3
    fig = S.build_strain_histogram(data, style=PlotStyle())
    assert isinstance(fig, Figure)


def test_build_detrend_diag_styled_returns_figure():
    from matplotlib.figure import Figure

    from dfxm.common.plotting import PlotStyle

    arr = np.random.rand(10, 15) * 0.002
    fig = S.build_detrend_diag(arr, arr * 0.9, arr * 0.1, style=PlotStyle())
    assert isinstance(fig, Figure)


def test_strain_map_cmap_follows_style():
    import numpy as np

    from dfxm.common.plotting import PlotStyle
    from dfxm.stages.strain import build_strain_map

    strain = np.random.default_rng(0).standard_normal((6, 8)) * 1e-4
    fig = build_strain_map(strain, 0.152, 0.385, None, (None, None))
    assert fig.axes[0].images[0].cmap.name == "RdBu_r"  # legacy default preserved
    fig = build_strain_map(
        strain, 0.152, 0.385, None, (None, None), style=PlotStyle(cmap_strain="seismic")
    )
    assert fig.axes[0].images[0].cmap.name == "seismic"
