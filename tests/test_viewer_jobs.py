"""Child-process viewer jobs — loader resolution and scene threading (no GL)."""

from __future__ import annotations

import h5py
import numpy as np

from darq_xray import viewer_jobs as VJ


def _h5(tmp_path):
    path = str(tmp_path / "aligned.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=np.ones((2, 3, 4)))
        f.attrs["scale_x_um_per_px"] = 0.15
        f.attrs["scale_y_um_per_px"] = 0.38
        f.attrs["scale_z_um_per_px"] = 2.0
    return path


def test_load_volume_h5_dataset(tmp_path):
    vol, spacing, notes = VJ._load_volume(
        {"kind": "h5_dataset", "path": _h5(tmp_path), "dataset": "sum_intensity"}
    )
    assert vol.shape == (2, 3, 4)
    assert spacing == (0.15, 0.38, 2.0)
    assert notes == ()  # fits -> no decimation, no note


def _big_h5(tmp_path):
    path = str(tmp_path / "big.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=np.zeros((16, 16, 16)))
        f.attrs["scale_x_um_per_px"] = 0.5
        f.attrs["scale_y_um_per_px"] = 2.0
        f.attrs["scale_z_um_per_px"] = 4.0
    return path


def test_load_volume_decimates_and_scales_spacing(tmp_path, monkeypatch):
    """The export child gets the same treatment as the viewer, spacing included."""
    monkeypatch.setattr(VJ.volumeio, "display_headroom_bytes", lambda: 1024)
    vol, spacing, notes = VJ._load_volume(
        {"kind": "h5_dataset", "path": _big_h5(tmp_path), "dataset": "sum_intensity"}
    )
    step = 16 // vol.shape[0]
    assert step > 1
    assert vol.shape == (16 // step, 16 // step, 16 // step)
    assert spacing == (0.5 * step, 2.0 * step, 4.0 * step)
    assert any("decimat" in n.lower() for n in notes)


def test_rotation_video_job_reports_the_decimation(tmp_path, monkeypatch):
    monkeypatch.setattr(VJ.volumeio, "display_headroom_bytes", lambda: 1024)
    monkeypatch.setattr(VJ.R3, "save_rotation_video", lambda *a, **k: a[1] + ".mp4")
    seen = []
    out = VJ.rotation_video_job(
        {
            "loader": {"kind": "h5_dataset", "path": _big_h5(tmp_path), "dataset": "sum_intensity"},
            "scene": {"mode": "volume", "cmap": "gray"},
            "base_path": str(tmp_path / "orbit"),
            "fmt": "mp4",
            "cbar_label": "Intensity",
        },
        progress=lambda frac, text="": seen.append(text),
    )
    assert out["video"] == str(tmp_path / "orbit") + ".mp4"
    assert any("decimat" in n.lower() for n in out["notes"])
    assert any("decimat" in t.lower() for t in seen)  # and said while it runs


def test_rotation_video_job_threads_scene_and_camera(tmp_path, monkeypatch):
    seen = {}

    def fake_save(scene, base_path, fmt, **kw):
        seen.update(scene=scene, base_path=base_path, fmt=fmt, **kw)
        return base_path + ".mp4"

    monkeypatch.setattr(VJ.R3, "save_rotation_video", fake_save)
    out = VJ.rotation_video_job(
        {
            "loader": {"kind": "h5_dataset", "path": _h5(tmp_path), "dataset": "sum_intensity"},
            "scene": {
                "mode": "surface",
                "cmap": "viridis",
                "clim": [0.0, 2.0],
                "log_scale": False,
                "opacity": 0.5,
                "opacity_mapping": "linear",
                "threshold": None,
                "clip": None,
                "downsample": 2,
                "background": "white",
            },
            "base_camera": [[0, 0, 10], [0, 0, 0], [0, 1, 0]],
            "elevation": 0.0,
            "zoom": 1.0,
            "n_frames": 12,
            "fps": 5,
            "base_path": str(tmp_path / "orbit"),
            "fmt": "gif",
            "cbar_label": "Intensity",
            "group": "raw",
            "style_json": "",
        }
    )
    assert out == {"video": str(tmp_path / "orbit") + ".mp4", "notes": []}
    assert seen["scene"].mode == "surface" and seen["scene"].downsample == 2
    assert seen["base_camera"] == ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert seen["n_frames"] == 12 and seen["group"] == "raw"
