"""Tests for dfxm.stages.matched — (samy,samz) matching, background-subtracted
frame loading, and pixel-aligned grayscale layer output.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from dfxm.stages import matched as M

NF, H, W = 4, 6, 8


def _write_strain(root, name, samy, samz):
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
        f.create_dataset("1.1/instrument/positioners/samy", data=samy)
        f.create_dataset("1.1/instrument/positioners/samz", data=samz)


def _write_rocking(root, name, samy, samz, frames):
    folder = os.path.join(root, name)
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
        f.create_dataset("1.1/instrument/positioners/samy", data=samy)
        f.create_dataset("1.1/instrument/positioners/samz", data=samz)
        f.create_dataset("1.1/measurement/pco_ff", data=frames.astype(np.float32))


# -- units --------------------------------------------------------------------
def test_match_nearest_threshold():
    sy = np.array([0.0, 0.002, 0.004])
    sz = np.array([0.0, 0.001, 0.003])
    ry = np.array([0.0039, 0.5])
    rz = np.array([0.0029, 0.5])
    matches, max_dist = M.match_nearest(sy, sz, ry, rz, threshold_mm=0.001)
    assert matches[2] == 0  # strain layer 2 matches rocking 0 (within threshold)
    assert matches[0] is None  # too far -> no match
    assert max_dist <= 0.001


def test_load_pco_ff_frame_background_subtracted(tmp_path):
    frames = np.ones((NF, H, W), dtype=np.float32) * 5.0
    frames[1, 2, 3] = 50.0  # a spike in frame 1
    _write_rocking(str(tmp_path), "rock__1", 0.0, 0.0, frames)
    h5p = os.path.join(str(tmp_path), "rock__1", "rock__1.h5")
    img = M.load_pco_ff_frame(h5p, "1.1/measurement/pco_ff", frame_index=1)
    assert img.shape == (H, W)
    assert img[2, 3] == pytest.approx(45.0)  # 50 - median(5) = 45
    # flat-background pixels: value - median == 0 (not negative -> not NaN)
    assert np.nansum(img) == pytest.approx(45.0)


# -- end to end ---------------------------------------------------------------
def test_run_saves_matched_layers(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    samy = [0.0, 0.001, 0.002]
    samz = [0.0, 0.001, 0.002]
    for i in range(3):
        _write_strain(str(raw), f"strain__{i + 1}", samy[i], samz[i])
        frames = np.random.default_rng(i).standard_normal((NF, H, W)) + 10.0
        _write_rocking(str(raw), f"rock__{i + 1}", samy[i], samz[i], frames)
    out = tmp_path / "matched_out"
    res = M.run(
        {
            "raw_root": str(raw),
            "strain_pattern": "strain__*",
            "rocking_pattern": "rock__*",
            "frame_index": 0,
            "match_threshold_mm": 0.001,
            "output_dir": str(out),
        }
    )
    assert res.n_strain == 3
    assert res.n_matched == 3 and res.n_saved == 3
    pngs = [p for p in os.listdir(res.layers_dir) if p.endswith(".png")]
    assert len(pngs) == 3


def test_run_skips_mismatched_frame_shape(tmp_path):
    """A matched rocking scan with a different detector shape is skipped, not fatal."""
    raw = tmp_path / "raw"
    raw.mkdir()
    samy = [0.0, 0.001, 0.002]
    samz = [0.0, 0.001, 0.002]
    for i in range(3):
        _write_strain(str(raw), f"strain__{i + 1}", samy[i], samz[i])
    rng = np.random.default_rng(0)
    _write_rocking(str(raw), "rock__1", samy[0], samz[0], rng.standard_normal((NF, H, W)) + 10)
    _write_rocking(str(raw), "rock__2", samy[1], samz[1], rng.standard_normal((NF, H, W)) + 10)
    # rock__3 has a wider detector -> cannot share the canvas built from rock__1
    _write_rocking(str(raw), "rock__3", samy[2], samz[2], rng.standard_normal((NF, H, W + 3)) + 10)
    res = M.run(
        {
            "raw_root": str(raw),
            "strain_pattern": "strain__*",
            "rocking_pattern": "rock__*",
            "match_threshold_mm": 0.001,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert res.n_matched == 3 and res.n_saved == 2  # mismatched one skipped, no crash
    assert any("shape" in s for s in res.skipped)


def test_run_requires_raw_root():
    with pytest.raises(ValueError, match="raw_root"):
        M.run({"raw_root": ""})
