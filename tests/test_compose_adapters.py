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
