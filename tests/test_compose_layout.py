"""Layout solver, sizing pass — dfxm.compose.layout."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from dfxm.common.errors import StageUserError
from dfxm.common.plotting import PlotStyle, measured_box_in
from dfxm.compose.adapters import PanelData
from dfxm.compose.layout import (
    SizedCell,
    autoscale_traces,
    measure_cells,
    place_tree,
    size_cells,
    trace_column_targets,
)
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    PanelSource,
    Row,
    Spacer,
    TextCell,
)


def _panel(pid, kind="map_layer"):
    return PanelDef(pid, PanelSource("/x.h5", kind, {}))


def _recipe(layout, panels, style=None):
    return FigureRecipe("t", style or {}, ComposeStyle(), layout, panels)


def _map_data(x=20.0, y=10.0):
    return PanelData(kind="map_layer", ext_x_um=x, ext_y_um=y, group="mosa_com")


def _trace_data(length=30.0):
    return PanelData(kind="profiles_trace", length_um=length)


def test_map_and_trace_intrinsic_boxes_exact():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    pa, pt = _panel("a"), _panel("t", "profiles_trace")
    layout = Row([PanelRef("a"), PanelRef("t")])
    cells = size_cells(
        _recipe(layout, [pa, pt]), style, {"a": _map_data(), "t": _trace_data()}, notes := []
    )
    ca, ct = cells[id(layout.children[0])], cells[id(layout.children[1])]
    assert abs(ca.w_in - 20.0 / 10.0 / 2.54) < 1e-9
    assert abs(ca.h_in - 10.0 / 10.0 / 2.54) < 1e-9
    assert abs(ct.w_in - 30.0 / 5.0 / 2.54) < 1e-9
    assert abs(ct.h_in - 2.0 / 2.54) < 1e-9
    assert notes == []


def test_per_panel_scale_override_wins():
    style = PlotStyle(scale_um_per_cm=10.0)
    p = _panel("a")
    p.scale_um_per_cm = 4.0
    layout = PanelRef("a")
    cells = size_cells(_recipe(layout, [p]), style, {"a": _map_data()}, [])
    assert abs(cells[id(layout)].w_in - 20.0 / 4.0 / 2.54) < 1e-9


def test_no_scale_anywhere_refused_with_hint():
    layout = PanelRef("a")
    with pytest.raises(StageUserError) as e:
        size_cells(_recipe(layout, [_panel("a")]), PlotStyle(), {"a": _map_data()}, [])
    assert "scale" in str(e.value).lower() and e.value.hint


def test_pinned_row_height_rescales_and_notes_implied_scale():
    style = PlotStyle(scale_um_per_cm=10.0)
    layout = Row([PanelRef("a")], pinned_height_cm=2.0)  # intrinsic h would be 1 cm
    cells = size_cells(_recipe(layout, [_panel("a")]), style, {"a": _map_data()}, notes := [])
    c = cells[id(layout.children[0])]
    assert abs(c.h_in - 2.0 / 2.54) < 1e-9
    assert abs(c.w_in - 2.0 * (20.0 / 10.0) / 2.54) < 1e-9  # aspect preserved
    assert any("implied" in n and "5" in n for n in notes)  # 10 µm/cm -> implied 5 µm/cm


def test_pinned_col_width_covers_missing_scale():
    layout = Col([PanelRef("a")], pinned_width_cm=4.0)
    cells = size_cells(_recipe(layout, [_panel("a")]), PlotStyle(), {"a": _map_data()}, notes := [])
    c = cells[id(layout.children[0])]
    assert abs(c.w_in - 4.0 / 2.54) < 1e-9
    assert abs(c.h_in - 4.0 * (10.0 / 20.0) / 2.54) < 1e-9
    assert any("implied" in n for n in notes)


def test_spacer_text_placeholder_fixed_boxes():
    style = PlotStyle(scale_um_per_cm=10.0)
    layout = Row([Spacer(1.0, 2.0), TextCell("hdr", 3.0, 1.0), PanelRef("a")])
    data = {"a": PanelData(kind="placeholder", payload={"reason": "gone"})}
    cells = size_cells(_recipe(layout, [_panel("a")]), style, data, notes := [])
    sp = cells[id(layout.children[0])]
    tx = cells[id(layout.children[1])]
    ph = cells[id(layout.children[2])]
    assert (sp.w_in, sp.h_in) == (1.0 / 2.54, 2.0 / 2.54)
    assert (tx.w_in, tx.h_in) == (3.0 / 2.54, 1.0 / 2.54)
    assert (ph.w_in, ph.h_in) == (4.0 / 2.54, 3.0 / 2.54)
    assert any("placeholder" in n for n in notes)


def test_trace_clamp_note_surfaces():
    style = PlotStyle(trace_scale_um_per_cm=0.1)  # 500 µm line -> >30 in, clamps
    layout = PanelRef("t")
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]),
        style,
        {"t": _trace_data(500.0)},
        notes := [],
    )
    assert cells[id(layout)].w_in == 30.0
    assert any("clamp" in n.lower() for n in notes)


def test_map_negative_scale_override_raises_user_error():
    style = PlotStyle(scale_um_per_cm=10.0)
    p = _panel("a")
    p.scale_um_per_cm = -5.0
    layout = PanelRef("a")
    with pytest.raises(StageUserError) as e:
        size_cells(_recipe(layout, [p]), style, {"a": _map_data()}, [])
    assert "scale" in str(e.value).lower() and e.value.hint


def test_map_non_numeric_scale_override_raises_user_error_not_bare_value_error():
    style = PlotStyle(scale_um_per_cm=10.0)
    p = _panel("a")
    p.scale_um_per_cm = "abc"
    layout = PanelRef("a")
    with pytest.raises(StageUserError) as e:
        size_cells(_recipe(layout, [p]), style, {"a": _map_data()}, [])
    assert "scale" in str(e.value).lower() and e.value.hint


def test_pinned_row_height_reaches_trace_with_note():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    layout = Row([PanelRef("t")], pinned_height_cm=4.0)  # natural height would be 2 cm
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]),
        style,
        {"t": _trace_data(30.0)},
        notes := [],
    )
    c = cells[id(layout.children[0])]
    assert abs(c.h_in - 4.0 / 2.54) < 1e-9
    # width is governed purely by length/scale, unaffected by the row's height pin
    assert abs(c.w_in - 30.0 / 5.0 / 2.54) < 1e-9
    assert any("implied" in n and "trace" in n for n in notes)


def test_trace_invalid_scale_override_raises_not_silent_fallback():
    # A valid MAP scale exists, so a silent fallback to it would produce a
    # plausible-looking (but wrong) box instead of surfacing the bad override.
    style = PlotStyle(scale_um_per_cm=10.0)
    p = _panel("t", "profiles_trace")
    p.scale_um_per_cm = -3.0
    layout = PanelRef("t")
    with pytest.raises(StageUserError) as e:
        size_cells(_recipe(layout, [p]), style, {"t": _trace_data(30.0)}, [])
    assert "scale" in str(e.value).lower() and e.value.hint


def test_trace_no_scale_anywhere_refused_with_hint():
    layout = PanelRef("t")
    with pytest.raises(StageUserError) as e:
        size_cells(
            _recipe(layout, [_panel("t", "profiles_trace")]), PlotStyle(), {"t": _trace_data()}, []
        )
    assert "scale" in str(e.value).lower() and e.value.hint


def test_nested_col_under_pinned_row_divides_height_after_gutters():
    """Item 5(b): Row(pinned_height) > Col([a, b]) — each stacked child gets
    (pin − gutter)/2, not the full pin (which overflowed the container)."""
    style = PlotStyle(scale_um_per_cm=10.0)
    a, b = PanelRef("a"), PanelRef("b")
    layout = Row([Col([a, b])], pinned_height_cm=4.0)
    recipe = _recipe(layout, [_panel("a"), _panel("b")])
    recipe.compose.gutter_cm = 0.5
    cells = size_cells(recipe, style, {"a": _map_data(), "b": _map_data()}, notes := [])
    each_in = ((4.0 - 0.5) / 2) / 2.54
    assert abs(cells[id(a)].h_in - each_in) < 1e-9
    assert abs(cells[id(b)].h_in - each_in) < 1e-9
    assert any("split over 2 stacked children" in n for n in notes)


def test_trace_under_both_pins_honours_row_height_too():
    """Item 5(a): Col(pinned_width) inside Row(pinned_height) — the width-pin
    early return used to keep the cosmetic trace height, silently dropping the
    row's height pin."""
    style = PlotStyle(trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    t = PanelRef("t")
    layout = Row([Col([t], pinned_width_cm=6.0)], pinned_height_cm=4.0)
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]), style, {"t": _trace_data()}, notes := []
    )
    c = cells[id(t)]
    assert abs(c.w_in - 6.0 / 2.54) < 1e-9
    assert abs(c.h_in - 4.0 / 2.54) < 1e-9  # NOT trace_height_cm's 2.0
    assert any("pinned row height" in n for n in notes)


def test_map_double_pin_height_wins_with_note():
    style = PlotStyle(scale_um_per_cm=10.0)
    a = PanelRef("a")
    layout = Row([Col([a], pinned_width_cm=6.0)], pinned_height_cm=2.0)
    cells = size_cells(_recipe(layout, [_panel("a")]), style, {"a": _map_data()}, notes := [])
    c = cells[id(a)]
    assert abs(c.h_in - 2.0 / 2.54) < 1e-9
    assert abs(c.w_in - 2.0 * (20.0 / 10.0) / 2.54) < 1e-9  # aspect from the height pin
    assert any("width pin ignored" in n for n in notes)


def test_pin_too_small_for_children_refused():
    layout = Row([Col([PanelRef("a"), PanelRef("b")])], pinned_height_cm=0.4)
    recipe = _recipe(layout, [_panel("a"), _panel("b")])
    recipe.compose.gutter_cm = 0.5  # the gutter alone exceeds the pin
    with pytest.raises(StageUserError) as e:
        size_cells(
            recipe, PlotStyle(scale_um_per_cm=10.0), {"a": _map_data(), "b": _map_data()}, []
        )
    assert "too small" in str(e.value) and e.value.hint


def test_zero_length_trace_becomes_placeholder_with_note():
    style = PlotStyle(trace_scale_um_per_cm=5.0)
    layout = PanelRef("t")
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]),
        style,
        {"t": _trace_data(0.0)},
        notes := [],
    )
    c = cells[id(layout)]
    assert c.kind == "placeholder"
    assert (c.w_in, c.h_in) == (4.0 / 2.54, 3.0 / 2.54)
    assert any("degenerate trace length" in n for n in notes)


def test_zero_length_trace_under_width_pin_still_placeholder():
    t = PanelRef("t")
    layout = Col([t], pinned_width_cm=4.0)
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]),
        PlotStyle(),
        {"t": _trace_data(0.0)},
        notes := [],
    )
    assert cells[id(t)].kind == "placeholder"
    assert any("degenerate trace length" in n for n in notes)


# -- trace autoscale (match column width) -------------------------------------


def _autoscale_recipe(layout, panels):
    r = _recipe(layout, panels)
    r.compose.trace_autoscale = True
    return r


def test_autoscale_matches_trace_to_column_map_width_ratio_kept_with_note():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, t])
    r = _autoscale_recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])
    data = {"m": _map_data(), "t": _trace_data(30.0)}
    cells = size_cells(r, style, data, notes := [])
    ct = cells[id(t)]
    w0, h0 = ct.w_in, ct.h_in  # natural: 30/5 = 6 cm wide, 2 cm tall
    autoscale_traces(r, cells, data, notes)
    assert abs(ct.w_in - cells[id(m)].w_in) < 1e-9  # matched (downscale: 6 cm -> 2 cm)
    assert abs(ct.h_in / ct.w_in - h0 / w0) < 1e-9  # box ratio kept
    # implied scale = 30 µm / 2 cm = 15 µm/cm
    assert any("autoscaled to column width" in n and "15" in n for n in notes)


def test_autoscale_falls_back_to_figure_widest_map_and_upscales():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Row([m, t])  # trace has NO enclosing Col
    r = _autoscale_recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])
    data = {"m": _map_data(40.0, 10.0), "t": _trace_data(10.0)}  # trace 2 cm, map 4 cm
    cells = size_cells(r, style, data, notes := [])
    autoscale_traces(r, cells, data, notes)
    assert abs(cells[id(t)].w_in - cells[id(m)].w_in) < 1e-9  # upscaled: match, not grow-only
    assert any("autoscaled to column width" in n for n in notes)


def test_autoscale_prefers_innermost_enclosing_col_with_a_map():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    mo, mi, t = PanelRef("mo"), PanelRef("mi"), PanelRef("t")
    layout = Col([mo, Col([mi, t])])
    r = _autoscale_recipe(layout, [_panel("mo"), _panel("mi"), _panel("t", "profiles_trace")])
    data = {"mo": _map_data(40.0, 10.0), "mi": _map_data(20.0, 10.0), "t": _trace_data(30.0)}
    cells = size_cells(r, style, data, notes := [])
    autoscale_traces(r, cells, data, notes)
    # the inner Col's 2 cm map wins over the outer Col's 4 cm one
    assert abs(cells[id(t)].w_in - cells[id(mi)].w_in) < 1e-9


def test_autoscale_all_trace_figure_untouched_no_note():
    style = PlotStyle(trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    t1, t2 = PanelRef("t1"), PanelRef("t2")
    layout = Col([t1, t2])
    r = _autoscale_recipe(layout, [_panel("t1", "profiles_trace"), _panel("t2", "profiles_trace")])
    data = {"t1": _trace_data(30.0), "t2": _trace_data(10.0)}
    cells = size_cells(r, style, data, notes := [])
    w1, w2 = cells[id(t1)].w_in, cells[id(t2)].w_in
    autoscale_traces(r, cells, data, notes)
    assert (cells[id(t1)].w_in, cells[id(t2)].w_in) == (w1, w2)
    assert not any("autoscaled" in n for n in notes)


def test_autoscale_skips_pin_sized_trace():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, Col([t], pinned_width_cm=6.0)])
    r = _autoscale_recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])
    data = {"m": _map_data(), "t": _trace_data(30.0)}
    cells = size_cells(r, style, data, notes := [])
    ct = cells[id(t)]
    assert ct.pinned is True
    assert id(t) not in trace_column_targets(r, cells)  # pinned cells never targeted
    w0 = ct.w_in
    autoscale_traces(r, cells, data, notes)
    assert ct.w_in == w0  # the pin wins
    assert not any("autoscaled" in n for n in notes)


def test_size_cells_marks_pin_sized_traces_and_not_free_ones():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    t = PanelRef("t")
    layout = Row([t], pinned_height_cm=4.0)  # the height-pin branch
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]), style, {"t": _trace_data(30.0)}, []
    )
    assert cells[id(t)].pinned is True
    t2 = PanelRef("t")
    cells2 = size_cells(
        _recipe(t2, [_panel("t", "profiles_trace")]), style, {"t": _trace_data(30.0)}, []
    )
    assert cells2[id(t2)].pinned is False


def test_autoscale_leaves_placeholder_trace_untouched():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, t])
    r = _autoscale_recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])
    data = {"m": _map_data(), "t": _trace_data(0.0)}  # degenerate -> placeholder
    cells = size_cells(r, style, data, notes := [])
    autoscale_traces(r, cells, data, notes)
    assert cells[id(t)].kind == "placeholder"
    assert (cells[id(t)].w_in, cells[id(t)].h_in) == (4.0 / 2.54, 3.0 / 2.54)


def test_autoscale_flag_off_is_strict_noop():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, t])
    r = _recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])  # flag stays False
    data = {"m": _map_data(), "t": _trace_data(30.0)}
    cells = size_cells(r, style, data, notes := [])
    w0 = cells[id(t)].w_in
    autoscale_traces(r, cells, data, notes)
    assert cells[id(t)].w_in == w0
    assert not any("autoscaled" in n for n in notes)


# -- measure/align/place ------------------------------------------------------


def _plot_cell(fig, leaf, w_in, h_in, ylabel="y"):
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50)))
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel(ylabel)
    return SizedCell(leaf, None, "trace", w_in, h_in, ax=ax)


def _place_tree_no_share(fig, layout, cells, *, gutter_in, pad_in):
    """Test-only stand-in for ``place_tree`` that skips the margin-sharing
    step (mirrors the bug margin-compensation fixes) — used below to prove
    the row/col fixtures actually discriminate (rather than passing by
    fixture coincidence), the same pattern as
    ``test_stage_profiles.py``'s ``_own_margin_flush``. Never used by
    production code; kept in lockstep with ``place_tree`` minus sharing."""
    env: dict[int, tuple[float, float]] = {}

    def _envelope(node):
        if isinstance(node, Row):
            child_envs = [_envelope(c) for c in node.children]
            w = sum(e[0] for e in child_envs) + gutter_in * max(0, len(child_envs) - 1)
            h = max(e[1] for e in child_envs)
            env[id(node)] = (w, h)
            return (w, h)
        if isinstance(node, Col):
            child_envs = [_envelope(c) for c in node.children]
            w = max(e[0] for e in child_envs)
            h = sum(e[1] for e in child_envs) + gutter_in * max(0, len(child_envs) - 1)
            env[id(node)] = (w, h)
            return (w, h)
        m = cells[id(node)].margins
        e = (m.left + cells[id(node)].w_in + m.right, m.bottom + cells[id(node)].h_in + m.top)
        env[id(node)] = e
        return e

    root_w, root_h = _envelope(layout)
    fig_w, fig_h = root_w + 2 * pad_in, root_h + 2 * pad_in
    fig.set_size_inches(fig_w, fig_h, forward=False)

    def _place(node, x, y):
        if isinstance(node, Row):
            cx = x
            for child in node.children:
                _place(child, cx, y)
                cx += env[id(child)][0] + gutter_in
            return
        if isinstance(node, Col):
            cy = y
            for child in node.children:
                _place(child, x, cy)
                cy += env[id(child)][1] + gutter_in
            return
        c = cells[id(node)]
        if c.ax is None:
            return
        m = c.margins
        x0 = (x + m.left) / fig_w
        y0 = (fig_h - y - m.top - c.h_in) / fig_h
        c.ax.set_position([x0, y0, c.w_in / fig_w, c.h_in / fig_h])
        if c.sync is not None:
            c.sync(fig, c.ax)

    _place(layout, pad_in, pad_in)
    return (fig_w, fig_h)


def test_row_shared_top_bottom_margins_and_exact_boxes():
    fig = Figure(facecolor="white")
    a, b = PanelRef("a"), PanelRef("b")
    layout = Row([a, b])
    ca = _plot_cell(fig, a, 2.0, 1.5)
    # cb gets a title so its natural (pre-share) top margin is genuinely
    # taller than ca's — without this, both cells' top/bottom margins
    # coincide by construction (same h_in, same xlabel) and the "shared"
    # assertion below would pass even if sharing were a no-op.
    cb = _plot_cell(fig, b, 1.2, 1.5, ylabel="a much longer label (units)")
    cb.ax.set_title("a genuinely taller top margin")
    cells = {id(a): ca, id(b): cb}
    measure_cells(fig, [ca, cb])
    pre_top_a, pre_top_b = ca.margins.top, cb.margins.top
    # Discrimination check 1: the fixture must actually produce different
    # pre-share top margins, else "post-share equal" proves nothing.
    assert pre_top_b > pre_top_a, (pre_top_a, pre_top_b)

    # Discrimination check 2: an own-margins-only placement (no sharing)
    # must NOT equalize the tops.
    fig_ns = Figure(facecolor="white")
    ca_ns = _plot_cell(fig_ns, a, 2.0, 1.5)
    cb_ns = _plot_cell(fig_ns, b, 1.2, 1.5, ylabel="a much longer label (units)")
    cb_ns.ax.set_title("a genuinely taller top margin")
    measure_cells(fig_ns, [ca_ns, cb_ns])
    _place_tree_no_share(fig_ns, layout, {id(a): ca_ns, id(b): cb_ns}, gutter_in=0.2, pad_in=0.1)
    assert ca_ns.margins.top != cb_ns.margins.top

    # Real code: place_tree shares the max top/bottom margin over the row.
    place_tree(fig, layout, cells, gutter_in=0.2, pad_in=0.1)
    assert ca.margins.top == cb.margins.top == max(pre_top_a, pre_top_b)
    assert ca.margins.bottom == cb.margins.bottom
    for c, w in ((ca, 2.0), (cb, 1.2)):
        bw, bh = measured_box_in(fig, c.ax)
        assert abs(bw - w) < 0.01 and abs(bh - 1.5) < 0.01
    # boxes top-align: same y1 in figure inches
    figh = fig.get_size_inches()[1]
    y1a = ca.ax.get_position().y1 * figh
    y1b = cb.ax.get_position().y1 * figh
    assert abs(y1a - y1b) < 0.01


def test_col_shared_left_margin_left_aligns_boxes():
    fig = Figure(facecolor="white")
    a, b = PanelRef("a"), PanelRef("b")
    layout = Col([a, b])
    ca = _plot_cell(fig, a, 2.5, 1.0, ylabel="s")
    cb = _plot_cell(fig, b, 1.4, 1.0, ylabel="a very long y label (deg)")
    # A rotated y-label's bbox width is ~the font line height regardless of
    # string length, so the ylabel text alone can't make the pre-share left
    # margins diverge. Give cb a much wider y-data range instead (plain,
    # unshifted tick labels like "100000" are genuinely wider than "0.2").
    ca.ax.set_ylim(0, 1)
    cb.ax.set_ylim(0, 100000)
    cb.ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    cells = {id(a): ca, id(b): cb}
    measure_cells(fig, [ca, cb])
    pre_left_a, pre_left_b = ca.margins.left, cb.margins.left
    # Discrimination check 1: the fixture must actually produce different
    # pre-share left margins, else "post-share equal" proves nothing.
    assert pre_left_b > pre_left_a, (pre_left_a, pre_left_b)

    # Discrimination check 2: an own-margins-only placement (no sharing)
    # must NOT equalize the lefts.
    fig_ns = Figure(facecolor="white")
    ca_ns = _plot_cell(fig_ns, a, 2.5, 1.0, ylabel="s")
    cb_ns = _plot_cell(fig_ns, b, 1.4, 1.0, ylabel="a very long y label (deg)")
    ca_ns.ax.set_ylim(0, 1)
    cb_ns.ax.set_ylim(0, 100000)
    cb_ns.ax.ticklabel_format(axis="y", style="plain", useOffset=False)
    measure_cells(fig_ns, [ca_ns, cb_ns])
    _place_tree_no_share(fig_ns, layout, {id(a): ca_ns, id(b): cb_ns}, gutter_in=0.15, pad_in=0.1)
    assert ca_ns.margins.left != cb_ns.margins.left

    # Real code: place_tree shares the max left/right margin over the col.
    place_tree(fig, layout, cells, gutter_in=0.15, pad_in=0.1)
    assert ca.margins.left == cb.margins.left == max(pre_left_a, pre_left_b)
    assert abs(ca.ax.get_position().x0 - cb.ax.get_position().x0) < 1e-6
    # no vertical overlap, a above b
    assert ca.ax.get_position().y0 > cb.ax.get_position().y1 - 1e-6


def test_ragged_row_of_cols_trailing_padding_aligns_envelopes():
    fig = Figure(facecolor="white")
    a1, a2, b1 = PanelRef("a1"), PanelRef("a2"), PanelRef("b1")
    col_a = Col([a1, a2])  # two panels -> taller envelope
    col_b = Col([b1])  # one panel -> padded at the bottom
    layout = Row([col_a, col_b])
    cells = {
        id(a1): _plot_cell(fig, a1, 1.5, 1.0),
        id(a2): _plot_cell(fig, a2, 1.5, 1.0),
        id(b1): _plot_cell(fig, b1, 1.5, 1.0),
    }
    measure_cells(fig, list(cells.values()))
    fw, fh = place_tree(fig, layout, cells, gutter_in=0.2, pad_in=0.1)
    # col_b's single panel top-aligns with col_a's first panel
    assert abs(cells[id(b1)].ax.get_position().y1 - cells[id(a1)].ax.get_position().y1) < 1e-3
    # figure is exactly the envelope + padding, no auto layout
    assert fig.get_layout_engine() is None
    assert tuple(np.round(fig.get_size_inches(), 3)) == (round(fw, 3), round(fh, 3))


def test_spacer_and_text_cells_occupy_their_boxes():
    fig = Figure(facecolor="white")
    sp = Spacer(2.54, 2.54)  # 1 in
    a = PanelRef("a")
    layout = Row([sp, a])
    csp = SizedCell(sp, None, "spacer", 1.0, 1.0)
    ca = _plot_cell(fig, a, 1.5, 1.0)
    cells = {id(sp): csp, id(a): ca}
    measure_cells(fig, [csp, ca])
    place_tree(fig, layout, cells, gutter_in=0.0, pad_in=0.0)
    figw = fig.get_size_inches()[0]
    # panel starts 1 in (spacer) + its own left margin from the left edge
    assert abs(ca.ax.get_position().x0 * figw - (1.0 + ca.margins.left)) < 0.02


def test_apply_trace_aspect_sets_height_from_width_skipping_pins():
    from dfxm.compose.layout import SizedCell, apply_trace_aspect
    from dfxm.compose.recipe import ComposeStyle

    t = SizedCell(object(), None, "trace", 3.0, 1.0)
    pinned = SizedCell(object(), None, "trace", 3.0, 1.0, pinned=True)
    m = SizedCell(object(), None, "map", 3.0, 3.0)
    cells = {1: t, 2: pinned, 3: m}
    notes = []
    apply_trace_aspect(ComposeStyle(trace_aspect=None), cells, notes)
    assert (t.w_in, t.h_in) == (3.0, 1.0)  # None = untouched
    apply_trace_aspect(ComposeStyle(trace_aspect=4.0), cells, notes)
    assert t.w_in == 3.0 and t.h_in == pytest.approx(0.75)
    assert (pinned.w_in, pinned.h_in) == (3.0, 1.0)  # pins win
    assert (m.w_in, m.h_in) == (3.0, 3.0)  # maps untouched


def test_col_gap_cm_overrides_global_gutter_between_its_children():
    fig = Figure(facecolor="white")
    a1, a2 = PanelRef("a1"), PanelRef("a2")
    col = Col([a1, a2], gap_cm=0.0)  # touching panels inside this column
    b = PanelRef("b")
    layout = Row([col, b])  # ...but the row keeps the global gutter
    cells = {
        id(a1): _plot_cell(fig, a1, 1.5, 1.0),
        id(a2): _plot_cell(fig, a2, 1.5, 1.0),
        id(b): _plot_cell(fig, b, 1.5, 1.0),
    }
    measure_cells(fig, list(cells.values()))
    fw, fh = place_tree(fig, layout, cells, gutter_in=0.5, pad_in=0.0)
    p1 = cells[id(a1)].ax.get_position()
    p2 = cells[id(a2)].ax.get_position()
    m1, m2 = cells[id(a1)].margins, cells[id(a2)].margins
    # vertical distance between the two boxes = only their own decoration margins
    gap_in = (p1.y0 - p2.y1) * fh
    assert gap_in == pytest.approx(m1.bottom + m2.top, abs=1e-6)
    # the row gutter (0.5 in) still separates the column from b
    pb = cells[id(b)].ax.get_position()
    col_right = max(p1.x1, p2.x1) * fw + max(m1.right, m2.right)
    assert (pb.x0 * fw - cells[id(b)].margins.left) - col_right == pytest.approx(0.5, abs=1e-6)


def test_row_col_gap_cm_round_trips_and_validated():
    from dfxm.compose.recipe import recipe_from_json, recipe_to_json, validate_recipe

    p = PanelDef("m", PanelSource("/x.h5", "map_layer", {"stage": "strain"}))
    r = FigureRecipe(
        "g", {}, ComposeStyle(), Row([Col([PanelRef("m")], gap_cm=0.0)], gap_cm=1.25), [p]
    )
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.layout.gap_cm == 1.25 and r2.layout.children[0].gap_cm == 0.0
    r.layout.gap_cm = -1.0
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert "gap_cm" in str(e.value) and e.value.hint


def test_col_fill_height_stretches_its_traces_to_the_row_height():
    fig = Figure(facecolor="white")
    m = PanelRef("m")
    t1, t2 = PanelRef("t1"), PanelRef("t2")
    col = Col([t1, t2], fill_height=True, gap_cm=0.0)
    layout = Row([m, col])
    cm = _plot_cell(fig, m, 2.0, 6.0)  # a tall map
    cm.kind = "map"
    c1 = _plot_cell(fig, t1, 3.0, 1.0)
    c2 = _plot_cell(fig, t2, 3.0, 1.0)
    c1.kind = c2.kind = "trace"
    cells = {id(m): cm, id(t1): c1, id(t2): c2}
    measure_cells(fig, list(cells.values()))
    fw, fh = place_tree(fig, layout, cells, gutter_in=0.2, pad_in=0.0)
    # both traces grew by the same amount and the column now spans the map's envelope height
    assert c1.h_in == pytest.approx(c2.h_in) and c1.h_in > 1.0
    pm = cm.ax.get_position()
    p1 = cells[id(t1)].ax.get_position()
    p2 = cells[id(t2)].ax.get_position()
    top_map = pm.y1 * fh + cm.margins.top
    top_col = p1.y1 * fh + c1.margins.top
    bot_map = pm.y0 * fh - cm.margins.bottom
    bot_col = p2.y0 * fh - c2.margins.bottom
    assert top_col == pytest.approx(top_map, abs=1e-6)
    assert bot_col == pytest.approx(bot_map, abs=1e-6)


def test_fill_flags_round_trip_and_pinned_or_map_cells_untouched():
    from dfxm.compose.recipe import recipe_from_json, recipe_to_json

    p = PanelDef("m", PanelSource("/x.h5", "map_layer", {"stage": "strain"}))
    r = FigureRecipe(
        "f", {}, ComposeStyle(), Row([Col([PanelRef("m")], fill_height=True)], fill_width=True), [p]
    )
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.layout.fill_width is True and r2.layout.children[0].fill_height is True
    fig = Figure(facecolor="white")
    m = PanelRef("m")
    t = PanelRef("t")
    col = Col([t], fill_height=True)
    layout = Row([m, col])
    cm = _plot_cell(fig, m, 2.0, 6.0)
    cm.kind = "map"
    ct = _plot_cell(fig, t, 3.0, 1.0)
    ct.kind = "trace"
    ct.pinned = True  # pins win: a pinned trace is never stretched
    cells = {id(m): cm, id(t): ct}
    measure_cells(fig, list(cells.values()))
    place_tree(fig, layout, cells, gutter_in=0.2, pad_in=0.0)
    assert ct.h_in == 1.0
