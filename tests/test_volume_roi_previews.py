"""visualize/paraview/slices: roi params tagged + roi_previews read a stacked-volume layer."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm.stages import paraview as PV
from dfxm.stages import slices as SL
from dfxm.stages import visualize as VZ


def _mosa_volume(path):
    with h5py.File(path, "w") as f:
        for grp in ("chi", "mu"):
            g = f.create_group(grp)
            g.create_dataset(
                "Center of mass", data=np.arange(4 * 6 * 5).reshape(4, 6, 5).astype(float)
            )
            g.create_dataset(
                "FWHM", data=np.abs(np.arange(4 * 6 * 5).reshape(4, 6, 5)).astype(float)
            )


@pytest.mark.parametrize(
    "mod,xname,yname",
    [(VZ, "roi_x", "roi_y"), (PV, "roi_x", "roi_y"), (SL, "align_roi_x", "align_roi_y")],
)
def test_roi_params_tagged(mod, xname, yname):
    by_name = {p.name: p for p in mod.STAGE.params}
    assert by_name[xname].roi_group and by_name[xname].roi_axis == "x"
    assert by_name[yname].roi_group and by_name[yname].roi_axis == "y"
    assert by_name[xname].roi_group == by_name[yname].roi_group  # same picker target


@pytest.mark.parametrize("mod", [VZ, PV, SL])
def test_roi_previews_reads_volume_layer(mod, tmp_path):
    vol = tmp_path / "mosa.h5"
    _mosa_volume(str(vol))
    params = {
        "mosa_volume_file": str(vol),
        "strain_volume_file": "",
        "pixel_size_x_um": 0.152,
        "pixel_size_y_um": 0.385,
    }
    previews = mod.roi_previews(params)
    assert previews
    arr, sx, sy = previews[0][1]()
    assert arr.shape == (6, 5)  # middle Z layer of (4,6,5)
    assert (sx, sy) == (0.152, 0.385)


@pytest.mark.parametrize("mod", [VZ, PV, SL])
def test_roi_previews_missing_file_empty(mod):
    assert mod.roi_previews({"mosa_volume_file": "", "strain_volume_file": ""}) == []
