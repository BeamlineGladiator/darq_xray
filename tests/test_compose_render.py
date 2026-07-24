"""render_recipe / export_recipe — dfxm.compose.render."""

import os

import h5py
import numpy as np
import pytest
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.patches import Rectangle

from dfxm.common.errors import StageUserError
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    PanelSource,
    Row,
)
from dfxm.compose.render import export_recipe, render_recipe


def _scale_bar_box(ax):
    """The AnchoredOffsetbox draw_scale_bar attaches to *ax*, or None."""
    for a in ax.get_children():
        if isinstance(a, AnchoredOffsetbox):
            return a
    return None


def _scale_bar_rect(ax):
    """The scale bar's data-space Rectangle (thickness = get_height()), or None."""
    box = _scale_bar_box(ax)
    if box is None:
        return None
    stack = [box.get_child()]
    while stack:
        node = stack.pop()
        if isinstance(node, Rectangle):
            return node
        if hasattr(node, "get_children"):
            stack.extend(node.get_children())
    return None


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


def _two_panel_recipe(h5, **style):
    p1 = PanelDef(
        "a",
        PanelSource(h5, "slice_plane", {"volume_id": "raw_sum", "slice_name": "obl", "plane": 0}),
    )
    p2 = PanelDef(
        "b",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
    )
    return FigureRecipe(
        "demo",
        {"scale_um_per_cm": 10.0, "show_title": False, **style},
        ComposeStyle(),
        Row([PanelRef("a"), PanelRef("b")]),
        [p1, p2],
    )


def test_render_two_maps_exact_boxes_and_labels(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    res = render_recipe(_two_panel_recipe(h5))
    assert res.n_panels == 2 and res.n_rendered == 2
    from dfxm.common.plotting import measured_box_in

    for pid in ("a", "b"):
        w, h = measured_box_in(res.figure, res.axes_by_id[pid])
        assert abs(w - 20.0 / 10.0 / 2.54) < 0.005 * w
        assert abs(h - 16.0 / 10.0 / 2.54) < 0.005 * h
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert "A" in texts and "B" in texts  # auto label sequence
    assert not any("drift" in n or "scale is off" in n for n in res.notes)


def test_label_template_and_manual_override(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.compose.label_template = "(a)"
    r.panels[1].label = "X9"
    res = render_recipe(r)
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert "(a)" in texts and "X9" in texts and "(b)" not in texts


def test_missing_file_renders_placeholder_and_notes(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels[1].source.h5_path = str(tmp_path / "gone.h5")
    res = render_recipe(r)
    assert res.n_rendered == 1
    assert any("placeholder" in n for n in res.notes)


def test_shared_colorbar_unified_clim_and_single_bar(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p1 = PanelDef(
        "a",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
    )
    p2 = PanelDef(
        "b",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
        clim=(-5.0, 5.0),
    )
    r = FigureRecipe(
        "shared",
        {"scale_um_per_cm": 10.0, "show_title": False},
        ComposeStyle(),
        Col([PanelRef("a"), PanelRef("b")], shared_colorbar=True),
        [p1, p2],
    )
    res = render_recipe(r)
    ima = res.axes_by_id["a"].images[0]
    imb = res.axes_by_id["b"].images[0]
    assert (ima.norm.vmin, ima.norm.vmax) == (imb.norm.vmin, imb.norm.vmax)  # unified
    # exactly one colorbar axes beyond the two panel axes + label texts
    cbar_axes = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(cbar_axes) == 1
    # the bar must span the group's REAL placed envelope (top of "a" to bottom
    # of "b"), not a provisional content-box guess that falls short of it.
    bar_ax = cbar_axes[0]
    pos_a, pos_b = res.axes_by_id["a"].get_position(), res.axes_by_id["b"].get_position()
    group_top, group_bottom = max(pos_a.y1, pos_b.y1), min(pos_a.y0, pos_b.y0)
    bar_pos = bar_ax.get_position()
    tol = 0.03 * (group_top - group_bottom)
    assert abs(bar_pos.y1 - group_top) < tol
    assert abs(bar_pos.y0 - group_bottom) < tol


def test_one_panel_scale_bar_only_designated_panel(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.compose.scale_bar_mode = "one-panel"
    r.compose.scale_bar_panel = "b"
    res = render_recipe(r)
    assert _scale_bar_box(res.axes_by_id["a"]) is None
    assert _scale_bar_box(res.axes_by_id["b"]) is not None


def test_gutter_scale_bar_loc_forced_regardless_of_style(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5, scale_bar_loc="upper right")
    r.compose.scale_bar_mode = "gutter"
    res = render_recipe(r)
    gutter_ax = next(
        ax
        for ax in res.figure.axes
        if ax not in res.axes_by_id.values() and _scale_bar_box(ax) is not None
    )
    box = _scale_bar_box(gutter_ax)
    assert box.loc == AnchoredOffsetbox.codes["center"]


def test_pinned_width_rescales_scale_bar_thickness(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.compose.pinned_width_cm = 6.0  # forces a large rescale off the natural width
    res = render_recipe(r)
    from dfxm.common.plotting import PlotStyle, measured_box_in

    style = PlotStyle()  # matches the recipe's un-overridden scale_bar_thickness_pt
    for pid in ("a", "b"):
        ax = res.axes_by_id[pid]
        w_in, _h_in = measured_box_in(res.figure, ax)
        final_eff = 20.0 / (w_in * 2.54)  # both slice_plane panels span 20 µm in X
        expected_bh = style.scale_bar_thickness_pt * (2.54 / 72.0) * final_eff
        rect = _scale_bar_rect(ax)
        assert rect is not None
        assert abs(rect.get_height() - expected_bh) < 0.02 * expected_bh


def test_shared_colorbar_mixed_groups_refused(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)  # raw + strain groups
    r.layout = Col([PanelRef("a"), PanelRef("b")], shared_colorbar=True)
    with pytest.raises(StageUserError) as e:
        render_recipe(r)
    assert "quantity" in str(e.value) and e.value.hint


def test_shared_x_stack_bottom_labels_only(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    pt = [
        PanelDef(f"t{i}", PanelSource(h5, "profiles_trace", {"job": JOB, "field": vid}))
        for i, vid in enumerate(["raw_sum", "strain"])
    ]
    r = FigureRecipe(
        "stack",
        {"trace_scale_um_per_cm": 5.0, "trace_height_cm": 2.0, "show_title": False},
        ComposeStyle(),
        Col([PanelRef("t0"), PanelRef("t1")], shared_x=True),
        pt,
    )
    res = render_recipe(r)
    top, bot = res.axes_by_id["t0"], res.axes_by_id["t1"]
    assert top.get_xlabel() == "" and bot.get_xlabel() != ""
    assert not any(t.get_visible() for t in top.get_xticklabels())


def test_export_no_tightcrop_all_formats(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    out = tmp_path / "out"
    out.mkdir()
    paths, res = export_recipe(
        _two_panel_recipe(h5), str(out), formats=("png", "pdf", "svg"), dpi=120
    )
    assert sorted(os.path.splitext(p)[1] for p in paths) == [".pdf", ".png", ".svg"]
    import matplotlib.image as mpimg

    img = mpimg.imread([p for p in paths if p.endswith(".png")][0])
    fw, fh = res.figure.get_size_inches()
    # matplotlib's RendererAgg truncates (int()), not rounds, when converting the
    # figure's inch size to a pixel canvas (backend_agg.RendererAgg.__init__) — use
    # the same conversion here so this is an exact-canvas pin, not a float-rounding
    # coin flip at the sub-pixel margin sizes real tick-label/colorbar decoration
    # produces.
    assert img.shape[1] == int(fw * 120) and img.shape[0] == int(fh * 120)  # exact canvas
