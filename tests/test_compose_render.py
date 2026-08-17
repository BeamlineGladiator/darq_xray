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


def _scale_bar_boxes(ax):
    """Every AnchoredOffsetbox on *ax* — used to catch a DUPLICATE scale bar
    (one drawn by the panel's own draw function, one by the deferred
    post-placement pass) that a lone `_scale_bar_box` lookup would miss."""
    return [a for a in ax.get_children() if isinstance(a, AnchoredOffsetbox)]


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


def test_profiles_ref_panel_has_single_scale_bar(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p = PanelDef("r", PanelSource(h5, "profiles_ref", {"job": JOB, "field": None}))
    r = FigureRecipe(
        "ref",
        {"scale_um_per_cm": 10.0, "show_title": False},
        ComposeStyle(),
        PanelRef("r"),
        [p],
    )
    res = render_recipe(r)
    ax = res.axes_by_id["r"]
    boxes = _scale_bar_boxes(ax)
    assert len(boxes) == 1
    # the surviving bar must be the deferred, final-scale one: its Rectangle
    # thickness should match the panel's rendered box width, not some other
    # (e.g. pre-placement) geometry.
    from dfxm.common.plotting import PlotStyle, measured_box_in

    style = PlotStyle()
    w_in, _h_in = measured_box_in(res.figure, ax)
    final_eff = 20.0 / (w_in * 2.54)  # the reference plane spans 20 µm in u
    expected_bh = style.scale_bar_thickness_pt * (2.54 / 72.0) * final_eff
    rect = _scale_bar_rect(ax)
    assert rect is not None
    assert abs(rect.get_height() - expected_bh) < 0.02 * expected_bh


def test_label_template_and_manual_override(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.compose.label_template = "(a)"
    r.panels[1].label = "X9"
    res = render_recipe(r)
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert "(a)" in texts and "X9" in texts and "(b)" not in texts


def test_blank_group_label_is_not_a_group_slot(tmp_path):
    """Item 8: '' group_label = "not a group" (each member gets its own letter),
    distinct from PanelDef.label where '' = suppressed."""
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.layout.group_label = ""  # programmatic edge — JSON load already normalizes
    res = render_recipe(r)
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert "A" in texts and "B" in texts  # two per-panel letters, not one group slot


def test_missing_file_renders_placeholder_and_notes(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels[1].source.h5_path = str(tmp_path / "gone.h5")
    res = render_recipe(r)
    assert res.n_rendered == 1
    assert any("placeholder" in n for n in res.notes)


def test_degenerate_roi_extent_renders_placeholder_not_singular_imshow(tmp_path):
    """A panel whose ROI crops to a single column (zero x-extent) is downgraded
    to a placeholder BOX by size_cells (dfxm/compose/layout.py's _map_cell),
    but that alone used to leave the panel's loaded PanelData.kind untouched
    ("slice_plane") — the per-leaf draw loop in render_recipe dispatches on
    the cell's kind, sees "placeholder", yet still handed draw_panel the
    original (still-degenerate) data, so draw_panel trusted data.kind and
    called imshow with a zero-width extent: matplotlib's "identical low and
    high xlims" UserWarning, reachable any time a recipe/ROI-override crops a
    map or slice panel to a single row or column. render_recipe must draw an
    actual placeholder for it instead, warning-free."""
    import warnings

    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels[1].roi = (0, 33, 5, 6)  # full v range, single u column -> ext_x_um == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = render_recipe(r)
    assert res.n_panels == 2 and res.n_rendered == 1  # "b" now counts as unrendered
    assert any("degenerate extent" in n for n in res.notes)


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
    # of "b"), not a provisional content-box guess that falls short of it —
    # minus the small end-insets that keep its end tick labels INSIDE the
    # span (2026-07-25: a flush bar poked half a tick label past each end,
    # off-canvas or into a neighbouring row).
    bar_ax = cbar_axes[0]
    pos_a, pos_b = res.axes_by_id["a"].get_position(), res.axes_by_id["b"].get_position()
    group_top, group_bottom = max(pos_a.y1, pos_b.y1), min(pos_a.y0, pos_b.y0)
    span = group_top - group_bottom
    bar_pos = bar_ax.get_position()
    eps = 1e-6
    assert group_bottom - eps <= bar_pos.y0 and bar_pos.y1 <= group_top + eps  # inside the span
    assert bar_pos.y1 - bar_pos.y0 > 0.8 * span  # and covering most of it
    res.figure.canvas.draw()
    ren = res.figure.canvas.get_renderer()
    bb = bar_ax.get_tightbbox(ren)
    h_px = res.figure.get_size_inches()[1] * res.figure.dpi
    lo = res.figure.transFigure.transform((0.0, group_bottom))[1]
    hi = res.figure.transFigure.transform((0.0, group_top))[1]
    assert bb.y0 >= lo - 0.5 and bb.y1 <= hi + 0.5, (bb, lo, hi, h_px)  # decorations inside too


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


def test_orphaned_panel_def_tolerated_in_gutter_mode(tmp_path):
    """A PanelDef no longer referenced by any layout leaf (e.g. left behind by
    a GUI delete of a Row/Col that orphaned its nested panels — see
    ``FigureBuilderWindow.delete_selected``) must not crash rendering:
    ``data_by_id``/``panels_by_id`` are keyed by ALL of ``recipe.panels``, but
    ``cell_by_pid`` only holds layout leaves, so any pid iteration sourced from
    the former must tolerate — or filter out — pids absent from the latter."""
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.compose.scale_bar_mode = "gutter"
    r.layout = Row([PanelRef("a")])  # "b" stays in recipe.panels but is orphaned
    res = render_recipe(r)
    assert res.n_panels == 1 and res.n_rendered == 1


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


def test_zero_length_trace_renders_placeholder_lockstep(tmp_path, monkeypatch):
    """Item 6: length_um == 0 joins the degenerate-extent placeholder lockstep —
    placeholder draw + note, never a zero-width trace axes (no mpl warnings)."""
    import warnings

    from dfxm.compose.adapters import PanelData

    monkeypatch.setattr(
        "dfxm.compose.render.load_panel",
        lambda p, cache=None: PanelData(kind="profiles_trace", length_um=0.0, payload={}),
    )
    p = PanelDef("t", PanelSource("/x.h5", "profiles_trace", {"job": JOB, "field": "strain"}))
    r = FigureRecipe("z", {"trace_scale_um_per_cm": 5.0}, ComposeStyle(), PanelRef("t"), [p])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = render_recipe(r)
    assert res.n_rendered == 0
    assert any("degenerate trace length" in n for n in res.notes)


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


def test_orphaned_panel_def_not_loaded_at_all(tmp_path):
    """Item 7: an orphaned PanelDef is skipped WITHOUT an h5 read and reported."""
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.layout = Row([PanelRef("a")])  # "b" orphaned
    cache: dict = {}
    res = render_recipe(r, loader_cache=cache)
    assert res.n_panels == 1 and res.n_rendered == 1
    assert len(cache) == 1  # only "a" was loaded
    assert any("skipped without loading" in n and "b" in n for n in res.notes)


def test_one_panel_scale_bar_unplaced_target_refused(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.layout = Row([PanelRef("a")])  # "b" exists as a def but is not placed
    r.compose.scale_bar_mode = "one-panel"
    r.compose.scale_bar_panel = "b"
    with pytest.raises(StageUserError) as e:
        render_recipe(r)
    assert "not placed" in str(e.value) and e.value.hint


def test_one_panel_scale_bar_trace_target_refused(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels.append(
        PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    )
    r.layout.children.append(PanelRef("t"))
    r.style["trace_scale_um_per_cm"] = 5.0
    r.compose.scale_bar_mode = "one-panel"
    r.compose.scale_bar_panel = "t"
    with pytest.raises(StageUserError) as e:
        render_recipe(r)
    assert "trace panel" in str(e.value) and e.value.hint


def test_one_panel_scale_bar_placeholder_target_degrades_with_note(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels[1].source.h5_path = str(tmp_path / "gone.h5")  # "b" -> placeholder
    r.compose.scale_bar_mode = "one-panel"
    r.compose.scale_bar_panel = "b"
    res = render_recipe(r)
    assert _scale_bar_box(res.axes_by_id["a"]) is None  # no bar leaks elsewhere
    assert any("no scale bar drawn" in n for n in res.notes)


def test_export_dir_uncreatable_raises_user_error(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    blocker = tmp_path / "blocker"
    blocker.write_text("a file standing where the out dir should be")
    with pytest.raises(StageUserError) as e:
        export_recipe(_two_panel_recipe(h5), str(blocker / "out"))
    assert "output directory" in str(e.value) and e.value.hint


def test_export_honours_style_overrides_formats_and_dpi(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    h5 = _write_obl(tmp_path / "obl.h5")
    recorded = {}
    orig = Figure.savefig

    def rec(self, path, **kw):
        recorded[os.path.basename(path)] = kw.get("dpi")
        return orig(self, path, **kw)

    monkeypatch.setattr(Figure, "savefig", rec)
    paths, _res = export_recipe(
        _two_panel_recipe(h5),
        str(tmp_path / "out"),
        style_overrides={"formats": ["svg"], "dpi": 72},
    )
    assert [os.path.splitext(p)[1] for p in paths] == [".svg"]
    assert recorded == {"demo.svg": 72}


def test_shared_colorbar_decorations_reserved_not_overlapping(tmp_path):
    """Real-data finding (2026-07-25): the shared bar was drawn AFTER the
    measure pass, so its tick numbers + label had no reserved space — they ran
    off-canvas (rightmost group) or over the next column's panels. The bar must
    be drawn pre-measure so its decorations are measured like any panel's."""
    import h5py as _h

    # mirror the real failing conditions: axes_mode "none" (panels reserve no
    # margins) + a long vertical colorbar label + strain-magnitude ticks.
    path = tmp_path / "obl.h5"
    u = np.linspace(-10.0, 10.0, 41)
    v = np.linspace(-8.0, 8.0, 33)
    uu, vv = np.meshgrid(u, v)
    with _h.File(path, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(
            kind="strain",
            cbar_label="Strain (cot method)",
            cmap="RdBu_r",
            title="s",
            vmin=-0.00085,
            vmax=0.00085,
        )
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=((uu + vv) * 1e-4)[None, ...].astype("f4"))
        sg.create_dataset("u_um", data=u)
        sg.create_dataset("v_um", data=v)
        sg.create_dataset("offsets_um", data=np.array([0.0]))
    h5 = str(path)
    mk = lambda pid: PanelDef(  # noqa: E731
        pid,
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
    )
    r = FigureRecipe(
        "sharedbb",
        {"scale_um_per_cm": 5.0, "show_title": False, "axes_mode": "none"},
        ComposeStyle(),
        Row(
            [
                Col(
                    [PanelRef("a"), PanelRef("b")],
                    shared_colorbar=True,
                    shared_clim=(-0.00085, 0.00085),
                ),
                PanelRef("c"),
            ]
        ),
        [mk("a"), mk("b"), mk("c")],
    )
    res = render_recipe(r)
    fig = res.figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bar_ax = next(ax for ax in fig.axes if ax not in res.axes_by_id.values())
    bb = bar_ax.get_tightbbox(ren)
    w_px = fig.get_size_inches()[0] * fig.dpi
    h_px = fig.get_size_inches()[1] * fig.dpi
    assert bb.x1 <= w_px + 0.5 and bb.y1 <= h_px + 0.5 and bb.x0 >= -0.5 and bb.y0 >= -0.5, (
        f"bar decorations leave the canvas: {bb} vs {w_px}x{h_px}"
    )
    cb = res.axes_by_id["c"].get_window_extent(ren)
    overlaps = bb.x0 < cb.x1 and cb.x0 < bb.x1 and bb.y0 < cb.y1 and cb.y0 < bb.y1
    assert not overlaps, f"bar decorations overlap panel c: {bb} vs {cb}"


def test_stacked_trace_ylabels_aligned(tmp_path):
    """Albert's 2026-07-25 real-data finding: in a shared-x trace stack the
    strain panel's y label sat at a different x than its neighbours (its tick
    numbers have a different width). Labels in a Col run must share ONE x
    (fig.align_ylabels)."""
    import h5py as _h

    path = tmp_path / "obl3.h5"
    u = np.linspace(-20.0, 20.0, 81)
    v = np.linspace(-15.0, 15.0, 61)
    uu, vv = np.meshgrid(u, v)
    with _h.File(path, "w") as f:
        for vid, val_scale in (("narrow", 1.0), ("wide", 1e-5)):
            g = f.create_group(vid)
            g.attrs.update(
                kind="raw_sum",
                cbar_label="value",
                cmap="gray",
                title=vid,
                vmin=-40.0 * val_scale,
                vmax=40.0 * val_scale,
            )
            sg = g.create_group("obl")
            sg.create_dataset("slices", data=((uu + vv) * val_scale)[None, ...].astype("f4"))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))
    job = {
        "name": "obl",
        "offset_um": 0.0,
        "start_uv": [-8.0, -6.0],
        "end_uv": [8.0, 6.0],
        "reference": "narrow",
    }
    r = FigureRecipe(
        "align",
        {"scale_um_per_cm": 5.0, "trace_scale_um_per_cm": 5.0, "show_title": False},
        ComposeStyle(),
        Col([PanelRef("tn"), PanelRef("tw")], shared_x=True),
        [
            PanelDef(
                "tn", PanelSource(str(path), "profiles_trace", {"job": job, "field": "narrow"})
            ),
            PanelDef("tw", PanelSource(str(path), "profiles_trace", {"job": job, "field": "wide"})),
        ],
    )
    res = render_recipe(r)
    fig = res.figure
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    exts = [res.axes_by_id[pid].yaxis.label.get_window_extent(ren) for pid in ("tn", "tw")]
    # the fixture must genuinely produce different tick widths, else this test
    # could pass vacuously — assert the discrimination premise too
    tick_w = [
        max(
            (t.get_window_extent(ren).width for t in res.axes_by_id[pid].get_yticklabels()),
            default=0.0,
        )
        for pid in ("tn", "tw")
    ]
    assert abs(tick_w[0] - tick_w[1]) > 5.0, f"fixture not discriminating: tick widths {tick_w}"
    assert abs(exts[0].x0 - exts[1].x0) < 1.5, f"ylabels not aligned: {[e.x0 for e in exts]}"


# -- united colorbars (colorbar_mode="united") --------------------------------
def _united_recipe(h5, *, pos="right"):
    def mk(pid, vid):
        return PanelDef(
            pid,
            PanelSource(h5, "slice_plane", {"volume_id": vid, "slice_name": "obl", "plane": 0}),
        )

    return FigureRecipe(
        "united",
        {"scale_um_per_cm": 10.0, "show_title": False},
        ComposeStyle(colorbar_mode="united", colorbar_pos=pos),
        Row([PanelRef("a"), PanelRef("b"), PanelRef("c")]),
        [mk("a", "strain"), mk("b", "raw_sum"), mk("c", "strain")],
    )


def test_united_one_bar_per_quantity_and_clims_unified(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _united_recipe(h5)
    r.panels[2].clim = (-20.0, 5.0)  # "c" widens the strain union
    res = render_recipe(r)
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(extra) == 2  # one strain bar + one raw bar, no per-panel bars
    for pid in ("a", "c"):
        im = res.axes_by_id[pid].images[0]
        assert (im.norm.vmin, im.norm.vmax) == (-20.0, 10.0)
    imb = res.axes_by_id["b"].images[0]
    assert (imb.norm.vmin, imb.norm.vmax) == (-10.0, 10.0)  # raw group untouched


def test_united_right_and_bottom_wrapping(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    res_r = render_recipe(_united_recipe(h5, pos="right"))
    panels = list(res_r.axes_by_id.values())
    max_x1 = max(ax.get_position().x1 for ax in panels)
    extra = [ax for ax in res_r.figure.axes if ax not in panels]
    assert extra and all(ax.get_position().x0 >= max_x1 - 1e-6 for ax in extra)
    res_b = render_recipe(_united_recipe(h5, pos="bottom"))
    panels_b = list(res_b.axes_by_id.values())
    min_y0 = min(ax.get_position().y0 for ax in panels_b)
    extra_b = [ax for ax in res_b.figure.axes if ax not in panels_b]
    assert extra_b and all(ax.get_position().y1 <= min_y0 + 1e-6 for ax in extra_b)


def test_united_ignores_group_flags_with_note(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _united_recipe(h5)
    r.layout = Row([Col([PanelRef("a"), PanelRef("c")], shared_colorbar=True), PanelRef("b")])
    res = render_recipe(r)
    assert any("override 1 group flag" in n for n in res.notes)
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(extra) == 2  # united bars only — the flagged Col added no bar


def test_united_panel_colorbar_true_forces_own_bar(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _united_recipe(h5)
    r.panels[0].colorbar = True  # "a" keeps its own bar, excluded from the union
    r.panels[2].clim = (-20.0, 5.0)
    res = render_recipe(r)
    ima = res.axes_by_id["a"].images[0]
    assert (ima.norm.vmin, ima.norm.vmax) == (-10.0, 10.0)  # NOT unified with "c"
    imc = res.axes_by_id["c"].images[0]
    assert (imc.norm.vmin, imc.norm.vmax) == (-20.0, 5.0)  # union of {c} alone
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(extra) == 3  # a's own cax + strain united bar + raw united bar


def test_united_trace_panels_keep_per_panel_behaviour(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p1 = PanelDef(
        "a",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
    )
    p2 = PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    r = FigureRecipe(
        "mix",
        {"scale_um_per_cm": 10.0, "trace_scale_um_per_cm": 5.0, "show_title": False},
        ComposeStyle(colorbar_mode="united"),
        Row([PanelRef("a"), PanelRef("t")]),
        [p1, p2],
    )
    res = render_recipe(r)
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(extra) == 1  # one united strain bar; the trace contributes nothing


def test_united_zero_groupable_panels_note_no_error(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    pt = [
        PanelDef(f"t{i}", PanelSource(h5, "profiles_trace", {"job": JOB, "field": vid}))
        for i, vid in enumerate(["raw_sum", "strain"])
    ]
    r = FigureRecipe(
        "tunited",
        {"trace_scale_um_per_cm": 5.0, "trace_height_cm": 2.0, "show_title": False},
        ComposeStyle(colorbar_mode="united"),
        Col([PanelRef("t0"), PanelRef("t1")]),
        pt,
    )
    res = render_recipe(r)
    assert res.n_rendered == 2
    assert any("nothing to unite" in n for n in res.notes)


def test_united_bar_stretches_to_scattered_members_union_span(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")

    def mk(pid, vid):
        return PanelDef(
            pid,
            PanelSource(h5, "slice_plane", {"volume_id": vid, "slice_name": "obl", "plane": 0}),
        )

    # vertical stack: strain (top), raw (middle), strain (bottom) — the strain
    # bar must span from the top panel's top to the bottom panel's bottom,
    # bridging the raw panel between them.
    r = FigureRecipe(
        "span",
        {"scale_um_per_cm": 10.0, "show_title": False},
        ComposeStyle(colorbar_mode="united", colorbar_pos="right"),
        Col([PanelRef("a"), PanelRef("b"), PanelRef("c")]),
        [mk("a", "strain"), mk("b", "raw_sum"), mk("c", "strain")],
    )
    res = render_recipe(r)
    top = res.axes_by_id["a"].get_position().y1
    bottom = res.axes_by_id["c"].get_position().y0
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    strain_bar = max(extra, key=lambda ax: ax.get_position().height)
    bp = strain_bar.get_position()
    eps = 1e-6
    assert bottom - eps <= bp.y0 and bp.y1 <= top + eps  # inside the union span
    assert bp.height > 0.8 * (top - bottom)  # and covering most of it


def _two_column_two_quantity_recipe(h5, *, pos):
    """One quantity per column, both columns spanning the full row height —
    the shape the final review flagged: with colorbar_pos="right" both
    groups' bars stretch to (near) the full height in the same right-edge
    column and collide; "bottom" keeps them apart (different x spans)."""

    def mk(pid, vid):
        return PanelDef(
            pid,
            PanelSource(h5, "slice_plane", {"volume_id": vid, "slice_name": "obl", "plane": 0}),
        )

    return FigureRecipe(
        "overlap",
        {"scale_um_per_cm": 10.0, "show_title": False},
        ComposeStyle(colorbar_mode="united", colorbar_pos=pos),
        Row(
            [
                Col([PanelRef("a1"), PanelRef("a2")]),
                Col([PanelRef("b1"), PanelRef("b2")]),
            ]
        ),
        [mk("a1", "strain"), mk("a2", "strain"), mk("b1", "raw_sum"), mk("b2", "raw_sum")],
    )


def test_united_overlapping_bars_produce_a_note(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    res = render_recipe(_two_column_two_quantity_recipe(h5, pos="right"))
    assert any("overlap" in n for n in res.notes)


def test_united_orthogonal_position_produces_no_overlap_note(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    res = render_recipe(_two_column_two_quantity_recipe(h5, pos="bottom"))
    assert not any("overlap" in n for n in res.notes)
