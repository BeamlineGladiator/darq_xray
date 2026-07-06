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
