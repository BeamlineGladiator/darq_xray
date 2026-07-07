import h5py
import numpy as np
import pytest

from dfxm.common import figures as F


def _write_vol(path, key="/chi/Center of mass", shape=(2, 4, 5)):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset(key, data=rng.standard_normal(shape).astype(np.float32))
    return key


def test_crop_roi_2d_slices_and_clamps():
    arr = np.arange(20, dtype=float).reshape(4, 5)
    # exact crop
    out = F.crop_roi_2d(arr, (1, 3, 0, 2))
    assert out.shape == (2, 2)
    assert np.array_equal(out, arr[1:3, 0:2])
    # None → unchanged (same object is fine)
    assert F.crop_roi_2d(arr, None) is arr
    # out-of-range bounds are clamped to the array
    assert F.crop_roi_2d(arr, (-5, 999, -5, 999)).shape == (4, 5)
    # empty crop → None
    assert F.crop_roi_2d(arr, (2, 2, 0, 5)) is None


def test_render_volume_layer_applies_clim_and_roi(tmp_path):
    h5 = tmp_path / "vol.h5"
    key = _write_vol(str(h5), shape=(2, 4, 5))
    fig = F.render_volume_layer(
        str(h5),
        key,
        0,
        cmap="magma",
        cmap_group="mosa_com",
        title="t",
        cbar_label="c",
        sx=0.1,
        sy=0.2,
        vmin=0.0,
        vmax=1.0,
        style=None,
        clim=(-3.0, 3.0),
        roi=(1, 3, 0, 2),
    )
    im = fig.axes[0].images[0]
    assert im.norm.vmin == -3.0 and im.norm.vmax == 3.0
    # extent reflects the CROP (2 cols × 0.1, 2 rows × 0.2), origin at 0
    left, right, bottom, top = im.get_extent()
    assert (right, top) == pytest.approx((0.2, 0.4))
    assert (left, bottom) == (0.0, 0.0)


def test_render_volume_layer_empty_crop_returns_none(tmp_path):
    h5 = tmp_path / "vol.h5"
    key = _write_vol(str(h5), shape=(1, 4, 5))
    fig = F.render_volume_layer(
        str(h5),
        key,
        0,
        cmap="magma",
        cmap_group=None,
        title="t",
        cbar_label="c",
        sx=0.1,
        sy=0.1,
        vmin=0.0,
        vmax=1.0,
        style=None,
        roi=(2, 2, 0, 5),
    )
    assert fig is None
