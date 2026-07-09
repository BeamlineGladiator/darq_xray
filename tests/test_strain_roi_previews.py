"""strain.roi_previews resolves a CoM map layer + pixel scales for the ROI picker."""

from __future__ import annotations

import h5py
import numpy as np

from dfxm.stages import strain as ST


def test_roi_param_is_tagged():
    spec = ST.STAGE
    roi = next(p for p in spec.params if p.name == "roi")
    assert roi.roi_group and roi.roi_axis == "both"


def test_roi_previews_reads_com_map(tmp_path):
    maps = tmp_path / "maps.h5"
    com = np.arange(20 * 12, dtype=float).reshape(20, 12)
    with h5py.File(maps, "w") as f:
        f.create_dataset("/entry/ccmth/Center of mass/Center of mass", data=com)
    params = {
        "mode": "single",
        "input_folder": str(tmp_path),
        "maps_filename": "maps.h5",
        "ccmth_com_path": "/entry/ccmth/Center of mass/Center of mass",
        "pixel_size_x_um": 0.152,
        "pixel_size_y_um": 0.385,
    }
    previews = ST.roi_previews(params)
    assert previews, "expected at least one preview"
    arr, sx, sy = previews[0][1]()
    assert arr.shape == (20, 12)
    assert (sx, sy) == (0.152, 0.385)


def test_roi_previews_missing_file_returns_empty():
    assert (
        ST.roi_previews({"mode": "single", "input_folder": "/no/such", "maps_filename": "maps.h5"})
        == []
    )
