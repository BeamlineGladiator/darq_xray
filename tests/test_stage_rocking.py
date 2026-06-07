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
