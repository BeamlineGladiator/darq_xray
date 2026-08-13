"""Child-process viewer jobs — loader resolution and scene threading (no GL)."""

from __future__ import annotations

import h5py
import numpy as np

from dfxm import viewer_jobs as VJ


def _h5(tmp_path):
    path = str(tmp_path / "aligned.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=np.ones((2, 3, 4)))
        f.attrs["scale_x_um_per_px"] = 0.15
        f.attrs["scale_y_um_per_px"] = 0.38
        f.attrs["scale_z_um_per_px"] = 2.0
    return path


def test_load_volume_h5_dataset(tmp_path):
    vol, spacing = VJ._load_volume(
        {"kind": "h5_dataset", "path": _h5(tmp_path), "dataset": "sum_intensity"}
    )
    assert vol.shape == (2, 3, 4)
    assert spacing == (0.15, 0.38, 2.0)


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
    assert out == {"video": str(tmp_path / "orbit") + ".mp4"}
    assert seen["scene"].mode == "surface" and seen["scene"].downsample == 2
    assert seen["base_camera"] == ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert seen["n_frames"] == 12 and seen["group"] == "raw"
