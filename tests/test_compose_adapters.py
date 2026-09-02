"""Panel adapters: pure loaders + draw-into-axes dispatch — dfxm.compose.adapters."""

import h5py
import numpy as np
import pytest
from matplotlib.figure import Figure

from dfxm.common.errors import StageUserError
from dfxm.compose.adapters import draw_panel, load_panel
from dfxm.compose.recipe import PanelDef, PanelSource


def _write_mosa(path):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("/chi/Center of mass", data=rng.normal(size=(2, 6, 8)).astype("f4"))
    return str(path)


def _write_strain(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("strain", data=np.linspace(-2e-4, 2e-4, 2 * 6 * 8).reshape(2, 6, 8))
        f.attrs["scale_x_um"] = 0.2
        f.attrs["scale_y_um"] = 0.4
    return str(path)


def _write_obl(path):
    u = np.linspace(-10.0, 10.0, 41)
    v = np.linspace(-8.0, 8.0, 33)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(path, "w") as f:
        for vid, kind in (("raw_sum", "raw_sum"), ("strain", "strain")):
            g = f.create_group(vid)
            g.attrs.update(
                kind=kind, cbar_label="value", cmap="gray", title=vid, vmin=-10.0, vmax=10.0
            )
            sg = g.create_group("obl")
            sg.create_dataset("slices", data=(uu + vv)[None, ...].astype("f4"))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))
    return str(path)


JOB = {"name": "obl", "offset_um": 0.0, "start_uv": [-5.0, -3.0], "end_uv": [5.0, 3.0]}


def test_load_map_layer_mosaicity_extents_and_group(tmp_path):
    h5 = _write_mosa(tmp_path / "stack.h5")
    p = PanelDef(
        "m",
        PanelSource(
            h5,
            "map_layer",
            {"stage": "mosaicity", "dataset": "/chi/Center of mass", "z": 1, "sx": 0.5, "sy": 0.25},
        ),
    )
    d = load_panel(p)
    assert d.kind == "map_layer" and d.group == "mosa_com"
    assert d.ext_x_um == 8 * 0.5 and d.ext_y_um == 6 * 0.25
    assert d.vmin is not None and d.vmax is not None and d.vmin < d.vmax


def test_load_map_layer_strain_defaults_from_attrs_and_symmetric(tmp_path):
    h5 = _write_strain(tmp_path / "strain.h5")
    p = PanelDef("s", PanelSource(h5, "map_layer", {"stage": "strain", "z": 0}))
    d = load_panel(p)
    assert d.group == "strain"
    assert d.ext_x_um == 8 * 0.2 and d.ext_y_um == 6 * 0.4
    assert d.vmin == -d.vmax  # symmetric limits


def test_load_map_layer_roi_crops_extent(tmp_path):
    h5 = _write_mosa(tmp_path / "stack.h5")
    p = PanelDef(
        "m",
        PanelSource(
            h5,
            "map_layer",
            {"stage": "mosaicity", "dataset": "/chi/Center of mass", "z": 0, "sx": 1.0, "sy": 1.0},
        ),
        roi=(1, 4, 2, 6),
    )
    d = load_panel(p)
    assert d.ext_x_um == 4.0 and d.ext_y_um == 3.0


def test_load_slice_plane_and_profiles(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    sp = load_panel(
        PanelDef(
            "p",
            PanelSource(
                h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}
            ),
        )
    )
    assert sp.kind == "slice_plane" and sp.group == "strain"
    assert sp.ext_x_um == 20.0 and sp.ext_y_um == 16.0
    ref = load_panel(PanelDef("r", PanelSource(h5, "profiles_ref", {"job": JOB, "field": None})))
    assert ref.kind == "profiles_ref" and ref.ext_x_um == 20.0
    tr = load_panel(
        PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    )
    assert tr.kind == "profiles_trace"
    assert abs(tr.length_um - np.hypot(10.0, 6.0)) < 1e-9


def test_missing_file_and_missing_key_become_placeholders(tmp_path):
    d = load_panel(
        PanelDef("x", PanelSource(str(tmp_path / "gone.h5"), "map_layer", {"stage": "strain"}))
    )
    assert d.kind == "placeholder" and "gone.h5" in d.payload["reason"]
    h5 = _write_mosa(tmp_path / "stack.h5")
    d2 = load_panel(
        PanelDef(
            "y",
            PanelSource(
                h5,
                "map_layer",
                {"stage": "mosaicity", "dataset": "/nope", "z": 0, "sx": 1, "sy": 1},
            ),
        )
    )
    assert d2.kind == "placeholder"


def test_map_layer_mosaicity_missing_dataset_raises(tmp_path):
    h5 = _write_mosa(tmp_path / "stack.h5")
    p = PanelDef(
        "m", PanelSource(h5, "map_layer", {"stage": "mosaicity", "z": 0, "sx": 1.0, "sy": 1.0})
    )
    with pytest.raises(StageUserError) as exc_info:
        load_panel(p)
    assert "dataset" in str(exc_info.value)
    assert "dataset" in exc_info.value.hint


def test_map_layer_rocking_missing_dataset_raises(tmp_path):
    h5 = _write_mosa(tmp_path / "stack.h5")
    p = PanelDef("m", PanelSource(h5, "map_layer", {"stage": "rocking", "z": 0}))
    with pytest.raises(StageUserError) as exc_info:
        load_panel(p)
    assert "dataset" in str(exc_info.value)


def test_slice_plane_missing_volume_id_raises(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p = PanelDef("p", PanelSource(h5, "slice_plane", {"slice_name": "obl", "plane": 0}))
    with pytest.raises(StageUserError) as exc_info:
        load_panel(p)
    assert "volume_id" in str(exc_info.value)


def test_slice_plane_missing_slice_name_raises(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p = PanelDef("p", PanelSource(h5, "slice_plane", {"volume_id": "strain", "plane": 0}))
    with pytest.raises(StageUserError) as exc_info:
        load_panel(p)
    assert "slice_name" in str(exc_info.value)


def test_profiles_ref_missing_job_raises(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p = PanelDef("r", PanelSource(h5, "profiles_ref", {"field": None}))
    with pytest.raises(StageUserError) as exc_info:
        load_panel(p)
    assert "job" in str(exc_info.value)


def test_profiles_trace_missing_job_raises(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p = PanelDef("t", PanelSource(h5, "profiles_trace", {"field": "strain"}))
    with pytest.raises(StageUserError) as exc_info:
        load_panel(p)
    assert "job" in str(exc_info.value)


def test_profiles_trace_missing_field_raises(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p = PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB}))
    with pytest.raises(StageUserError) as exc_info:
        load_panel(p)
    assert "field" in str(exc_info.value)
    assert "field" in exc_info.value.hint


def test_loader_cache_hit_skips_reread(tmp_path):
    h5 = _write_strain(tmp_path / "strain.h5")
    p = PanelDef("s", PanelSource(h5, "map_layer", {"stage": "strain", "z": 0}))
    cache = {}
    d1 = load_panel(p, cache=cache)
    import os

    os.remove(h5)
    d2 = load_panel(p, cache=cache)  # served from cache, file gone
    assert d2 is d1


def test_draw_panel_dispatch(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    fig = Figure(figsize=(6, 4))
    for sel_kind, sel in (
        ("slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
        ("profiles_ref", {"job": JOB, "field": None}),
        ("profiles_trace", {"job": JOB, "field": "strain"}),
    ):
        ax = fig.add_subplot(111)
        p = PanelDef("p", PanelSource(h5, sel_kind, sel))
        d = load_panel(p)
        draw_panel(ax, p, d, None, colorbar=False, scale_bar=False)
        if sel_kind == "profiles_trace":
            assert ax.lines  # the profile curve
        else:
            assert ax.images
        fig.clear()


def test_draw_placeholder_hatched(tmp_path):
    from dfxm.compose.adapters import draw_placeholder

    fig = Figure()
    ax = fig.add_subplot(111)
    draw_placeholder(ax, "missing file")
    assert any(p.get_hatch() for p in ax.patches)
    assert ax.get_xticks().size == 0


def test_slice_plane_roi_clamps_out_of_range_indices(tmp_path):
    """Out-of-range ROI indices clamp to the plane bounds (parity guard for the
    _crop_uv dedup — same behaviour as the pre-refactor inline clamp)."""
    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0.0, 2.0, 5))
        sg.create_dataset("v_um", data=np.linspace(0.0, 1.5, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0]))
    p = PanelDef(
        "s",
        PanelSource(
            str(h5), "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}
        ),
        roi=(0, 999, 0, 999),
    )
    d = load_panel(p)
    assert d.kind == "slice_plane"
    assert d.ext_x_um == 2.0 and d.ext_y_um == 1.5  # full extents survive the clamp


def test_resolve_trace_opts_follows_style_font_scale_by_default():
    from dfxm.common.plotting import PlotStyle
    from dfxm.compose.adapters import resolve_trace_opts
    from dfxm.compose.recipe import ComposeStyle

    st = PlotStyle(font_scale=2.5)
    o = resolve_trace_opts(ComposeStyle(), st)
    assert o["font_scale"] == 2.5
    assert o["linewidth"] == pytest.approx(1.8 * 2.5)  # scales with the fonts
    assert o["color"] is None  # matplotlib C0
    # style None (bare draw) -> plain 1.0 / 1.8
    o0 = resolve_trace_opts(ComposeStyle(), None)
    assert o0 == {"linewidth": 1.8, "color": None, "font_scale": 1.0}
    # explicit overrides win, linewidth override is absolute (not rescaled)
    o2 = resolve_trace_opts(
        ComposeStyle(trace_linewidth=4.0, trace_color="k", trace_font_scale=1.2), st
    )
    assert o2 == {"linewidth": 4.0, "color": "k", "font_scale": 1.2}
    # font override alone still drives the default linewidth
    o3 = resolve_trace_opts(ComposeStyle(trace_font_scale=2.0), st)
    assert o3["linewidth"] == pytest.approx(3.6) and o3["font_scale"] == 2.0


def test_draw_panel_trace_honours_trace_opts(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    p = PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    d = load_panel(p)
    draw_panel(ax, p, d, None, trace_opts={"linewidth": 5.0, "color": "red", "font_scale": 2.0})
    (line,) = [ln for ln in ax.lines if ln.get_zorder() == 3]
    assert line.get_linewidth() == 5.0
    assert line.get_color() == "red"
    assert ax.yaxis.label.get_fontsize() == pytest.approx(20.0)
    assert ax.xaxis.get_ticklabels()[0].get_fontsize() == pytest.approx(20.0)


def test_data_bbox_roi_finite_box_with_margin_and_within_roi():
    from dfxm.common.figures import data_bbox_roi

    a = np.full((40, 60), np.nan)
    a[10:20, 20:35] = 1.0  # rows 10..19, cols 20..34
    # 5 % margin of the box size (>= 1 px) each side, clamped to the frame
    assert data_bbox_roi(a, margin_frac=0.0) == (10, 20, 20, 35)
    r0, r1, c0, c1 = data_bbox_roi(a, margin_frac=0.1)
    assert (r0, r1, c0, c1) == (9, 21, 18, 37)  # 10% of 10 rows = 1; of 15 cols = 1.5 -> 2
    # nothing finite -> None (caller keeps its own roi)
    assert data_bbox_roi(np.full((3, 3), np.nan)) is None
    # restricted to an explicit ROI: box is found INSIDE it, returned in full-frame coords
    assert data_bbox_roi(a, roi=(0, 40, 0, 25), margin_frac=0.0) == (10, 20, 20, 25)
    # clamps at the frame edge
    b = np.ones((5, 5))
    assert data_bbox_roi(b, margin_frac=0.5) == (0, 5, 0, 5)
    # masked arrays: masked cells count as missing
    m = np.ma.masked_invalid(a)
    assert data_bbox_roi(m, margin_frac=0.0) == (10, 20, 20, 35)


def test_load_panel_crop_to_data_crops_all_image_kinds(tmp_path):
    from dfxm.compose.recipe import PanelDef, PanelSource

    # map layer: strain with NaN border
    h5 = str(tmp_path / "strain.h5")
    arr = np.full((2, 20, 30), np.nan)
    arr[:, 5:15, 10:20] = 1e-4
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=arr)
        f.attrs["scale_x_um"] = 1.0
        f.attrs["scale_y_um"] = 1.0
    p = PanelDef("m", PanelSource(h5, "map_layer", {"stage": "strain", "z": 0}), crop_to_data=True)
    d = load_panel(p)
    # default margin 3 % of the 10-px box = 0.3 -> ceil to 1 px each side
    assert d.payload["layer"].shape == (12, 12)
    assert d.ext_x_um == 12.0 and d.ext_y_um == 12.0
    p0 = PanelDef("m0", PanelSource(h5, "map_layer", {"stage": "strain", "z": 0}))
    assert load_panel(p0).payload["layer"].shape == (20, 30)
    # crop_to_data changes the loader cache key
    from dfxm.compose.adapters import _cache_key

    assert _cache_key(p) != _cache_key(p0)


def test_load_panel_crop_to_data_slice_and_ref(tmp_path):
    from dfxm.compose.recipe import PanelDef, PanelSource

    u = np.linspace(-10.0, 10.0, 41)
    v = np.linspace(-8.0, 8.0, 33)
    plane = np.full((33, 41), np.nan)
    plane[8:24, 10:30] = 2.0
    h5 = str(tmp_path / "obl.h5")
    with h5py.File(h5, "w") as f:
        for vid, kind in (("raw_sum", "raw_sum"), ("strain", "strain")):
            g = f.create_group(vid)
            g.attrs.update(
                kind=kind, cbar_label="value", cmap="gray", title=vid, vmin=-10.0, vmax=10.0
            )
            sg = g.create_group("obl")
            sg.create_dataset("slices", data=plane[None, ...].astype("f4"))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))
    sl = PanelDef(
        "s",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
        crop_to_data=True,
    )
    d = load_panel(sl)
    h, w = d.payload["plane2d"].shape
    assert h < 33 and w < 41 and len(d.payload["u"]) == w and len(d.payload["v"]) == h
    ref = PanelDef(
        "r", PanelSource(h5, "profiles_ref", {"job": JOB, "field": None}), crop_to_data=True
    )
    d2 = load_panel(ref)
    h2, w2 = d2.payload["plane"].shape
    assert h2 < 33 and w2 < 41 and len(d2.payload["u"]) == w2
    # traces ignore the flag
    tr = PanelDef(
        "t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}), crop_to_data=True
    )
    assert load_panel(tr).kind == "profiles_trace"


def test_panel_preview_full_frame_for_roi_picking(tmp_path):
    from dfxm.compose.adapters import panel_preview
    from dfxm.compose.recipe import PanelDef, PanelSource

    h5 = _write_obl(tmp_path / "obl.h5")
    # slice plane: full frame even when the panel itself is cropped
    sl = PanelDef(
        "s",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
        roi=(2, 10, 3, 9),
        crop_to_data=True,
    )
    arr, sx, sy = panel_preview(sl)
    assert arr.shape == (33, 41) and sx == pytest.approx(0.5) and sy == pytest.approx(0.5)
    ref = PanelDef(
        "r", PanelSource(h5, "profiles_ref", {"job": JOB, "field": None}), roi=(1, 5, 1, 5)
    )
    arr2, _sx, _sy = panel_preview(ref)
    assert arr2.shape == (33, 41)
    # map layer: pixel sizes from the selector / attrs
    hm = _write_strain(tmp_path / "strain.h5")
    mp = PanelDef("m", PanelSource(hm, "map_layer", {"stage": "strain", "z": 0}))
    arr3, sx3, sy3 = panel_preview(mp)
    assert arr3.shape == (6, 8) and (sx3, sy3) == (0.2, 0.4)
    # traces have no ROI
    tr = PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    with pytest.raises(ValueError):
        panel_preview(tr)
    # unavailable data -> ValueError with the reason, never a crash
    gone = PanelDef("g", PanelSource(str(tmp_path / "nope.h5"), "profiles_ref", {"job": JOB}))
    with pytest.raises(ValueError):
        panel_preview(gone)


def test_draw_panel_trace_y_tick_labels_off_hides_numbers_keeps_label(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    p = PanelDef(
        "t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}), y_tick_labels=False
    )
    draw_panel(ax, p, load_panel(p), None)
    assert ax.get_yticklabels() == []  # matplotlib drops invisible labels from this list
    assert ax.yaxis.get_offset_text().get_visible() is False
    assert ax.get_ylabel() and ax.yaxis.label.get_visible()
    assert ax.yaxis.get_major_ticks()[0].tick1line.get_visible()  # tick marks stay
    assert ax.get_xticklabels()  # x numbers untouched


def test_draw_panel_y_tick_labels_ignored_by_non_trace_kinds(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    sel = {"volume_id": "strain", "slice_name": "obl", "plane": 0}
    p = PanelDef("s", PanelSource(h5, "slice_plane", sel), y_tick_labels=False)
    draw_panel(ax, p, load_panel(p), None, colorbar=False, scale_bar=False)
    assert ax.yaxis.get_offset_text().get_visible() is True


def test_draw_panel_trace_y_tick_labels_off_hides_scientific_exponent(tmp_path):
    from dfxm.common.plotting import PlotStyle

    h5 = _write_obl(tmp_path / "obl.h5")
    with h5py.File(h5, "r+") as f:
        f["strain/obl/slices"][...] = f["strain/obl/slices"][...] * 1e-3  # strain-sized values
    st = PlotStyle(tickfmt_strain="scientific")
    src = PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"})

    def exponent_visible(ax):
        return any(t.get_visible() and "10^" in t.get_text() for t in ax.texts)

    # control: with the flag on, the scientific format really draws a x10^n text
    ax_on = Figure(figsize=(6, 4)).add_subplot(111)
    p_on = PanelDef("t1", src)
    draw_panel(ax_on, p_on, load_panel(p_on), st)
    assert exponent_visible(ax_on)
    ax_off = Figure(figsize=(6, 4)).add_subplot(111)
    p_off = PanelDef("t2", src, y_tick_labels=False)
    draw_panel(ax_off, p_off, load_panel(p_off), st)
    assert not exponent_visible(ax_off)
    assert ax_off.get_ylabel() and ax_off.yaxis.label.get_visible()  # y-label stays


def _write_png(path, w=40, h=20):
    from matplotlib.image import imsave

    rgb = np.zeros((h, w, 3), "f4")
    rgb[..., 0] = np.linspace(0.0, 1.0, w)[None, :]
    imsave(str(path), rgb)
    return str(path)


def test_load_image_pixels_as_extent_and_float_rgb_payload(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    d = load_panel(PanelDef("i", PanelSource(png, "image", {})))
    assert d.kind == "image" and (d.ext_x_um, d.ext_y_um) == (40.0, 20.0)
    img = d.payload["image"]
    assert img.shape[:2] == (20, 40) and img.dtype.kind == "f"
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert d.group is None and d.vmin is None


@pytest.mark.parametrize("fmt", ["jpeg_rgb_u8", "tiff_grey_u16"])
def test_load_image_integer_formats_normalised_to_unit_floats(tmp_path, fmt):
    # PNG comes back from matplotlib already as float32 in 0-1; JPEG and TIFF
    # arrive as uint8/uint16 and go through the loader's /max normalisation
    from PIL import Image

    if fmt == "jpeg_rgb_u8":
        arr = np.zeros((20, 40, 3), "u1")
        arr[..., 0] = np.linspace(0, 255, 40).astype("u1")[None, :]
        path = tmp_path / "ref.jpg"
        Image.fromarray(arr, mode="RGB").save(str(path))
        ndim = 3
    else:
        arr16 = np.linspace(0, 65535, 40).astype("u2")[None, :].repeat(20, axis=0)
        path = tmp_path / "ref.tif"
        Image.fromarray(arr16, mode="I;16").save(str(path))
        ndim = 2
    d = load_panel(PanelDef("i", PanelSource(str(path), "image", {})))
    assert d.kind == "image"
    img = d.payload["image"]
    assert img.dtype.kind == "f" and img.ndim == ndim
    assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0
    assert float(img.max()) > 0.5  # actually normalised by the dtype max, not squashed
    assert (d.ext_x_um, d.ext_y_um) == (40.0, 20.0)


def test_load_image_roi_is_a_pixel_crop_and_empty_crop_is_placeholder(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    d = load_panel(PanelDef("i", PanelSource(png, "image", {}), roi=(5, 15, 10, 30)))
    assert (d.ext_x_um, d.ext_y_um) == (20.0, 10.0)
    assert d.payload["image"].shape[:2] == (10, 20)
    d2 = load_panel(PanelDef("i", PanelSource(png, "image", {}), roi=(5, 5, 10, 30)))
    assert d2.kind == "placeholder" and "ref.png" in d2.payload["reason"]


def test_load_image_missing_file_is_placeholder_not_error(tmp_path):
    d = load_panel(PanelDef("i", PanelSource(str(tmp_path / "gone.png"), "image", {})))
    assert d.kind == "placeholder" and "gone.png" in d.payload["reason"]


def test_load_image_crop_to_data_is_ignored(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    d = load_panel(PanelDef("i", PanelSource(png, "image", {}), crop_to_data=True))
    assert (d.ext_x_um, d.ext_y_um) == (40.0, 20.0)


def test_panel_preview_refuses_image_panel(tmp_path):
    from dfxm.compose.adapters import panel_preview

    png = _write_png(tmp_path / "ref.png")
    with pytest.raises(ValueError, match="pixel crop"):
        panel_preview(PanelDef("i", PanelSource(png, "image", {})))


def test_draw_panel_image_axis_off_no_title(tmp_path):
    png = _write_png(tmp_path / "ref.png")
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    p = PanelDef("i", PanelSource(png, "image", {}), show_title=True)
    assert draw_panel(ax, p, load_panel(p), None) is None
    assert len(ax.images) == 1 and not ax.axison and ax.get_title() == ""


def test_draw_panel_greyscale_image_drawn_as_stored_not_contrast_stretched(tmp_path):
    from PIL import Image

    # two mid greys (100, 128 of 255): an autoscaled norm would render them
    # as pure black and pure white
    arr = np.full((20, 40), 100, "u1")
    arr[:, 20:] = 128
    path = tmp_path / "grey.png"
    Image.fromarray(arr, "L").save(path)
    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    p = PanelDef("i", PanelSource(str(path), "image", {}))
    assert draw_panel(ax, p, load_panel(p), None) is None
    im = ax.images[0]
    assert im.get_array().ndim == 2
    assert im.get_clim() == (0.0, 1.0)
