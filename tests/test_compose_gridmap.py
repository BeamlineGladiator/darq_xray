"""layout_to_grid / flatten_panel_ids / grid_to_layout — darq_xray.compose.gridmap."""

import pytest

from darq_xray.compose.gridmap import flatten_panel_ids, grid_to_layout, layout_to_grid
from darq_xray.compose.recipe import Col, PanelDef, PanelRef, PanelSource, Row, Spacer, TextCell


def _panels(*pids):
    return {
        pid: PanelDef(pid, PanelSource("/x.h5", "map_layer", {"stage": "strain", "z": 0}))
        for pid in pids
    }


@pytest.mark.parametrize(
    "grid",
    [
        [],
        [["a"]],
        [["a", "b"]],
        [["a"], ["b", "c"], ["d"]],
    ],
)
def test_round_trip_law(grid):
    pids = [p for col in grid for p in col]
    assert layout_to_grid(grid_to_layout(grid), _panels(*pids)) == grid


def test_grid_to_layout_shapes():
    lay = grid_to_layout([["a"], ["b", "c"], []])
    assert isinstance(lay, Row)
    assert len(lay.children) == 2  # empty column dropped
    assert isinstance(lay.children[0], PanelRef) and lay.children[0].panel_id == "a"
    col = lay.children[1]
    assert isinstance(col, Col) and [c.panel_id for c in col.children] == ["b", "c"]
    assert isinstance(grid_to_layout([]), Row) and grid_to_layout([]).children == []


def test_layout_to_grid_recognized_shapes():
    p = _panels("a", "b", "c")
    assert layout_to_grid(PanelRef("a"), p) == [["a"]]
    assert layout_to_grid(Col([PanelRef("a"), PanelRef("b")]), p) == [["a", "b"]]
    assert layout_to_grid(Row([PanelRef("a"), Col([PanelRef("b"), PanelRef("c")])]), p) == [
        ["a"],
        ["b", "c"],
    ]
    assert layout_to_grid(Row([]), p) == []


@pytest.mark.parametrize(
    "layout",
    [
        Row([Spacer(1.0, 1.0)]),
        Row([TextCell("t")]),
        Row([Row([PanelRef("a")])]),
        Row([Col([Col([PanelRef("a")])])]),
        Row([PanelRef("ghost")]),
        Row([PanelRef("a")], group_label="auto"),
        Row([PanelRef("a")], shared_colorbar=True),
        Row([PanelRef("a")], pinned_height_cm=3.0),
        Row([Col([PanelRef("a")], shared_colorbar=True)]),  # flagged single-member Col
        TextCell("t"),
    ],
)
def test_unmappable_layouts_return_none(layout):
    assert layout_to_grid(layout, _panels("a")) is None


def test_flagged_multi_member_col_is_mappable():
    p = _panels("a", "b")
    lay = Row([Col([PanelRef("a"), PanelRef("b")], shared_colorbar=True, group_label="auto")])
    assert layout_to_grid(lay, p) == [["a", "b"]]


def test_flatten_panel_ids_dfs_order():
    lay = Row([PanelRef("a"), Col([PanelRef("b"), Spacer(1, 1), PanelRef("c")]), TextCell("t")])
    assert flatten_panel_ids(lay) == ["a", "b", "c"]


def test_panel_group_hint_covers_kinds():
    from darq_xray.compose.gridmap import panel_group_hint

    def p(kind, sel):
        return PanelDef("x", PanelSource("/x.h5", kind, sel))

    assert panel_group_hint(p("map_layer", {"stage": "strain", "z": 0})) == "strain"
    assert panel_group_hint(p("map_layer", {"stage": "rocking", "dataset": "d"})) == "raw"
    assert panel_group_hint(p("map_layer", {"stage": "mosaicity", "dataset": "/chi/FWHM"})) == (
        "mosa_fwhm"
    )
    assert panel_group_hint(p("map_layer", {"stage": "mosaicity", "dataset": "/chi/Center"})) == (
        "mosa_com"
    )
    assert panel_group_hint(p("profiles_trace", {"job": {}, "field": "strain"})) == "trace"
    assert panel_group_hint(p("slice_plane", {"volume_id": "raw_mosa_sum"})) == "raw"
    assert panel_group_hint(p("slice_plane", {"volume_id": "strain"})) == "strain"
    assert panel_group_hint(p("slice_plane", {"volume_id": "mosa_com_chi"})) == "mosa_com"
    assert panel_group_hint(p("profiles_ref", {"job": {}, "field": None})) is None
    assert panel_group_hint(p("image", {})) is None  # neutral grey chip
