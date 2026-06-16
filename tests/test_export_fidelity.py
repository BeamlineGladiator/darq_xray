"""Export-fidelity tests: a stage's exported/preview figure must match the
figure the run produced (same colour scale, same axes), and the catalog must be
robust to partial/missing inputs.

Covers the review findings:
* strain export uses the run's symmetric zero-centred clim, not raw nanmin/nanmax;
* strain export keeps the ROI axis offset;
* strain detrend export resolves the source maps.h5 for nested folder patterns;
* visualize Center-of-mass export centres the volume like run();
* slices catalog survives a volume group missing an attribute;
* profiles "Export all" does not silently overwrite jobs that share a fig_name.
"""

from __future__ import annotations

import json
import os

import h5py
import numpy as np

from dfxm.stages import mosaicity as Mosaicity
from dfxm.stages import profiles as Profiles
from dfxm.stages import slices as Slices
from dfxm.stages import strain as Strain
from dfxm.stages import visualize as Visualize


def _ccmth(ny=40, nx=60, seed=1):
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-2, 2, ny))
    # deliberately asymmetric about zero so symmetric vs raw clim differ
    return (
        7.144
        + 0.004 * np.arctan(2 * X)
        + 0.001 * np.arctan(1.5 * Y)
        + 0.02 * (X > 1.5)
        + 0.0001 * rng.standard_normal((ny, nx))
    )


def _write_maps(folder, ccmth):
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, "maps.h5"), "w") as f:
        f.create_dataset("/entry/ccmth/Center of mass/Center of mass", data=ccmth)


def _im_clim_extent(fig):
    im = fig.axes[0].images[0]
    return im.get_clim(), im.get_extent()


# ---------------------------------------------------------------------------
# strain
# ---------------------------------------------------------------------------
def test_strain_export_uses_symmetric_clim_like_run(tmp_path):
    folder = tmp_path / "layer__1"
    _write_maps(str(folder), _ccmth())
    params = {
        "mode": "single",
        "input_folder": str(folder),
        "ccmth_ref_deg": 7.144,
        "save_plots": False,
        "output_dir": str(tmp_path / "out"),
    }
    res = Strain.run(params)
    specs = Strain.figures(res, params)
    map_spec = next(s for s in specs if s.kind == "map")
    clim, _ = _im_clim_extent(map_spec.build(None))
    # run() saves the PNG with symmetric_limits when vmin/vmax are blank.
    assert np.isclose(clim[0], -clim[1]), f"export clim not zero-centred: {clim}"


def test_strain_export_keeps_roi_axis_offset(tmp_path):
    folder = tmp_path / "layer__1"
    _write_maps(str(folder), _ccmth())
    params = {
        "mode": "single",
        "input_folder": str(folder),
        "ccmth_ref_deg": 7.144,
        "roi": "5,25,10,40",
        "pixel_size_x_um": 0.152,
        "pixel_size_y_um": 0.385,
        "save_plots": False,
        "output_dir": str(tmp_path / "out"),
    }
    res = Strain.run(params)
    specs = Strain.figures(res, params)
    map_spec = next(s for s in specs if s.kind == "map")
    _, extent = _im_clim_extent(map_spec.build(None))
    # c0 = 10, px = 0.152 -> the X axis must start at 1.52 µm, not 0.
    assert np.isclose(extent[0], 10 * 0.152), f"ROI x-offset dropped: {extent}"


def test_strain_export_detrend_resolves_nested_folder_pattern(tmp_path):
    root = tmp_path / "root"
    _write_maps(str(root / "sub" / "layer__1"), _ccmth())
    params = {
        "mode": "batch",
        "root_folder": str(root),
        "folder_pattern": "sub/layer__*",
        "ccmth_ref_deg": 7.144,
        "save_plots": False,
    }
    res = Strain.run(params)
    assert res.n_layers == 1
    specs = Strain.figures(res, params)
    detrend = next(s for s in specs if "detrend" in s.figure_id)
    # Must NOT raise FileNotFoundError: the source maps.h5 is in root/sub/layer__1.
    fig = detrend.build(None)
    assert fig is not None


# ---------------------------------------------------------------------------
# visualize Center-of-mass centring
# ---------------------------------------------------------------------------
def test_visualize_com_export_centres_like_run(tmp_path, monkeypatch):
    # a CoM volume with a large positive offset so centring clearly matters
    vol = np.linspace(5.0, 7.0, 2 * 6 * 8, dtype=np.float64).reshape(2, 6, 8)
    mosa_h5 = tmp_path / "stacked_mosa.h5"
    with h5py.File(mosa_h5, "w") as f:
        f.create_dataset("chi/Center of mass", data=vol)
    # no motors -> _align is identity (apply_roi_3d + default z), so export == run pre-centring
    monkeypatch.setattr(Visualize, "_read_motors", lambda *a, **k: (np.array([]), np.array([])))

    params = {
        "mosa_volume_file": str(mosa_h5),
        "strain_volume_file": "",
        "raw_root": "",
        "center_method": "midrange",
        "range_pct": 99.5,
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
        "output_dir": str(tmp_path / "out"),
    }
    res = Visualize.run(params)
    specs = Visualize.figures(res, params)
    spec = next(s for s in specs if "chi" in s.figure_id and s.figure_id.endswith("z0000"))
    layer = np.asarray(spec.build(None).axes[0].images[0].get_array())
    # run() centres the whole volume before rendering vol[0]; with no motors the
    # alignment is identity, so the exported layer 0 must equal the centred vol[0].
    centred, _vn, _vx = Visualize._center_com_and_range(vol.copy(), "midrange", 99.5)
    np.testing.assert_allclose(layer, centred[0], atol=1e-9)
    # and it must NOT be the raw (uncentred) data, which sat around 5-6.
    assert float(np.nanmedian(layer)) < 1.0


# ---------------------------------------------------------------------------
# slices catalog robustness
# ---------------------------------------------------------------------------
def _write_one_slice_volume(f, vid, *, drop_attr=None):
    vg = f.create_group(vid)
    attrs = {
        "kind": "mosa_fwhm",
        "cmap": "magma",
        "title": "x",
        "cbar_label": "y",
        "vmin": 0.0,
        "vmax": 1.0,
    }
    if drop_attr:
        attrs.pop(drop_attr)
    for k, v in attrs.items():
        vg.attrs[k] = v
    sg = vg.create_group("z_sweep")
    sg.create_dataset("slices", data=np.ones((1, 4, 5), np.float32))
    sg.create_dataset("u_um", data=np.linspace(0, 4, 5))
    sg.create_dataset("v_um", data=np.linspace(0, 3, 4))
    sg.create_dataset("offsets_um", data=np.zeros(1))


def test_mosaicity_streamed_clim_matches_full(tmp_path):
    rng = np.random.default_rng(0)
    vol = rng.standard_normal((4, 6, 8))
    vol[1, 2, 3] = np.nan  # NaNs must be ignored, like the full-array path
    h5 = tmp_path / "stacked_volumes.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("chi/Center of mass", data=vol)
    with h5py.File(h5, "r") as f:
        vmin, vmax = Mosaicity._streamed_clim(f["chi/Center of mass"])
    assert np.isclose(vmin, float(np.nanmin(vol)))
    assert np.isclose(vmax, float(np.nanmax(vol)))


def test_profiles_export_disambiguates_shared_fig_name(tmp_path):
    # two distinct jobs that happen to share the same fig_name
    jobs = [
        {"name": "a", "fig_name": "dup", "start_uv": [0, 0], "end_uv": [1, 1]},
        {"name": "b", "fig_name": "dup", "start_uv": [0, 0], "end_uv": [1, 1]},
    ]
    params = {"consolidated_h5": str(tmp_path / "x.h5"), "jobs_json": json.dumps(jobs)}
    result = Profiles.ProfilesResult(
        jobs=[
            Profiles.ProfileJobResult(name="a", offset_used_um=0.0, fields=["raw"]),
            Profiles.ProfileJobResult(name="b", offset_used_um=0.0, fields=["raw"]),
        ]
    )
    specs = Profiles.figures(result, params)
    stems = [s.filename for s in specs]
    assert len(stems) == 2
    assert len(set(stems)) == 2, f"two jobs share an export stem -> silent overwrite: {stems}"


def test_slices_catalog_skips_group_missing_attr(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    with h5py.File(h5, "w") as f:
        _write_one_slice_volume(f, "good")
        _write_one_slice_volume(f, "bad", drop_attr="cbar_label")
    res = Slices.SlicesResult(output_h5=str(h5))
    specs = Slices.figures(res, {})
    ids = {s.figure_id for s in specs}
    # the good volume must still be catalogued even though "bad" lacks an attr
    assert any("good" in i for i in ids), f"good volume lost: {ids}"
