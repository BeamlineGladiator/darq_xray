"""Layout solver, sizing pass — dfxm.compose.layout."""

import pytest

from dfxm.common.errors import StageUserError
from dfxm.common.plotting import PlotStyle
from dfxm.compose.adapters import PanelData
from dfxm.compose.layout import size_cells
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
