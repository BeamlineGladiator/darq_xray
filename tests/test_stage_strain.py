"""Tests for dfxm.stages.strain — numeric golden equivalence vs the legacy
y_calc_axial_strain_v6_batch (ccmth+mu) and calc_axial_strain_v7_batch
(ccmth-only) scripts, plus the detrend-before-ROI ordering constraint.
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
MU_PATH = "/entry/mu/Center of mass/Center of mass"


def _legacy(modname: str):
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / f"{modname}.py").exists():
        pytest.skip(f"legacy {modname}.py not found")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return __import__(modname)


def _synthetic_maps(ny=40, nx=60, seed=1):
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-2, 2, ny))
    ccmth = (
        7.144
        + 0.002 * np.arctan(2 * X)
        + 0.001 * np.arctan(1.5 * Y)
        + 0.0001 * rng.standard_normal((ny, nx))
    )
    mu = 11.2491 + 0.0015 * X + 0.0008 * Y + 0.0001 * rng.standard_normal((ny, nx))
    return ccmth, mu


def _write_maps(folder, ccmth, mu=None):
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, "maps.h5"), "w") as f:
        f.create_dataset(CCMTH_PATH, data=ccmth)
        if mu is not None:
            f.create_dataset(MU_PATH, data=mu)


def test_detrend_matches_legacy():
    legacy = _legacy("y_calc_axial_strain_v6_batch")
    ccmth, _ = _synthetic_maps()
    mine_dt, mine_surf = S.detrend_arctan_2d(ccmth.copy())
    leg_dt, leg_surf = legacy.detrend_arctan_2d(ccmth.copy())
    np.testing.assert_allclose(mine_dt, leg_dt, atol=1e-12)
    np.testing.assert_allclose(mine_surf, leg_surf, atol=1e-12)


def test_compute_strain_ccmth_mu_matches_legacy():
    legacy = _legacy("y_calc_axial_strain_v6_batch")
    ccmth, mu = _synthetic_maps()
    dt, _ = S.detrend_arctan_2d(ccmth.copy())
    s_mine, c_mine, m_mine = S.compute_strain(dt, 7.144, mu, 11.2491)
    s_leg, c_leg, m_leg = legacy.compute_strain(dt, mu, 7.144, 11.2491)
    np.testing.assert_allclose(s_mine, s_leg, atol=1e-15)
    np.testing.assert_allclose(c_mine, c_leg, atol=1e-15)
    np.testing.assert_allclose(m_mine, m_leg, atol=1e-15)


def test_compute_strain_ccmth_only_matches_legacy_v7():
    legacy = _legacy("calc_axial_strain_v7_batch")
    ccmth, _ = _synthetic_maps()
    dt, _ = S.detrend_arctan_2d(ccmth.copy())
    s_only, _, mu_term = S.compute_strain(dt, 7.144)
    np.testing.assert_allclose(s_only, legacy.compute_strain(dt, 7.144), atol=1e-15)
    assert np.all(mu_term == 0)


def test_run_single_stacks_volume_and_writes_plots(tmp_path):
    ccmth, mu = _synthetic_maps()
    folder = tmp_path / "layer__1"
    _write_maps(str(folder), ccmth, mu)
    res = S.run(
        {
            "method": "ccmth_mu",
            "mode": "single",
            "input_folder": str(folder),
            "ccmth_ref_deg": 7.144,
            "mu_ref_deg": 11.2491,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert res.n_layers == 1 and res.volume_shape == (1, 40, 60)
    with h5py.File(res.stacked_path, "r") as f:
        vol = f["strain"][:]
        dt, _ = S.detrend_arctan_2d(ccmth.copy())
        expected, _, _ = S.compute_strain(dt, 7.144, mu, 11.2491)
        np.testing.assert_allclose(vol[0], expected, atol=1e-15)
        assert f.attrs["method"] == "ccmth_mu"
    pngs = list((tmp_path / "out").glob("*.png"))
    assert any("strain" in p.name for p in pngs)
    assert any("contributions" in p.name for p in pngs)  # only for ccmth_mu


def test_run_batch_over_multiple_layers(tmp_path):
    ccmth, mu = _synthetic_maps()
    root = tmp_path / "root"
    for name in ["layer__1", "layer__2"]:
        _write_maps(str(root / name), ccmth, mu)
    res = S.run(
        {
            "method": "ccmth_only",
            "mode": "batch",
            "root_folder": str(root),
            "folder_pattern": "layer__*",
            "ccmth_ref_deg": 7.144,
            "save_plots": False,
        }
    )
    assert res.n_layers == 2 and res.volume_shape == (2, 40, 60)


def test_detrend_runs_before_roi(tmp_path):
    """ROI must crop the detrended map, not detrend a pre-cropped map."""
    ccmth, mu = _synthetic_maps()
    folder = tmp_path / "layer__1"
    _write_maps(str(folder), ccmth, mu)
    roi = [5, 25, 10, 40]
    res = S.run(
        {
            "method": "ccmth_mu",
            "mode": "single",
            "input_folder": str(folder),
            "ccmth_ref_deg": 7.144,
            "mu_ref_deg": 11.2491,
            "roi": "5,25,10,40",
            "save_plots": False,
            "output_dir": str(tmp_path / "out"),
        }
    )
    # expected: detrend full, THEN crop
    dt_full, _ = S.detrend_arctan_2d(ccmth.copy())
    dt_crop = dt_full[roi[0] : roi[1], roi[2] : roi[3]]
    mu_crop = mu[roi[0] : roi[1], roi[2] : roi[3]]
    expected, _, _ = S.compute_strain(dt_crop, 7.144, mu_crop, 11.2491)
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
