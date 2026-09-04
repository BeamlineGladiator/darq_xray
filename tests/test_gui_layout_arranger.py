"""LayoutArranger drag-grid widget (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from darq_xray.gui.widgets.layout_arranger import LayoutArranger, _corner_at  # noqa: E402

_INFO = {
    "a": {"title": "strain / z=0", "group": "strain"},
    "b": {"title": "raw / z=0", "group": "raw"},
    "c": {"title": "job / strain", "group": "trace"},
}


def _arr(grid):
    w = LayoutArranger()
    w.set_grid(grid, _INFO)
    return w


def test_set_grid_grid_round_trip():
    assert _arr([["a"], ["b", "c"]]).grid() == [["a"], ["b", "c"]]


def test_item_move_between_columns_updates_grid_and_signals():
    w = _arr([["a"], ["b", "c"]])
    fired = []
    w.gridChanged.connect(lambda: fired.append(1))
    item = w._columns[1].list.takeItem(0)  # the model move a drag performs
    w._columns[0].list.addItem(item)
    assert w.grid() == [["a", "b"], ["c"]]
    assert fired


def test_add_move_remove_column():
    w = _arr([["a"], ["b"]])
    w._on_add_column()
    assert len(w._columns) == 3
    assert w.grid() == [["a"], ["b"]]  # empty column normalized away in grid()
    w._move_column(w._columns[1], -1)
    assert w.grid() == [["b"], ["a"]]
    w._remove_column(w._columns[0])  # "b" merges into its right neighbour
    assert w.grid() == [["a", "b"]]
    only = w._columns[0]
    n = len(w._columns)
    w._remove_column(only) if n == 1 else None
    assert w.grid() == [["a", "b"]]  # single remaining column: ✕ is a no-op


def test_corner_hotzone_mapping():
    r = QRect(0, 0, 100, 40)
    assert _corner_at(QPoint(3, 3), r) == "upper left"
    assert _corner_at(QPoint(97, 3), r) == "upper right"
    assert _corner_at(QPoint(3, 37), r) == "lower left"
    assert _corner_at(QPoint(97, 37), r) == "lower right"
    assert _corner_at(QPoint(50, 20), r) is None


def test_corner_click_emits_scale_bar_pick_and_marks_tile():
    w = _arr([["a"]])
    picks = []
    w.scaleBarPicked.connect(lambda pid, loc: picks.append((pid, loc)))
    w._on_corner_clicked("a", "lower left")
    assert picks == [("a", "lower left")]
    assert w._scale_bar_panel == "a" and w._scale_bar_loc == "lower left"


def test_schematic_strips_follow_mode_and_pos():
    w = _arr([["a"], ["b"]])
    w.set_bar_schematic("united", "right")
    assert not w._right_strip.isHidden() and w._bottom_strip.isHidden()
    w.set_bar_schematic("united", "bottom")
    assert w._right_strip.isHidden() and not w._bottom_strip.isHidden()
    w.set_bar_schematic("per-panel", "right", {frozenset(["a"])})
    assert w._right_strip.isHidden() and w._bottom_strip.isHidden()
    assert not w._columns[0].flag_strip.isHidden()  # flagged column strip
    assert w._columns[1].flag_strip.isHidden()


def _fb_win():
    from darq_xray.common.plotting import PlotStyle
    from darq_xray.gui.figure_builder import FigureBuilderWindow

    return FigureBuilderWindow(lambda: {}, PlotStyle(scale_um_per_cm=10.0))


def _mk_panel(pid):
    from darq_xray.compose.recipe import PanelDef, PanelSource

    return PanelDef(pid, PanelSource("/x.h5", "map_layer", {"stage": "strain", "z": 0}))


def test_arrange_dialog_clean_grid_preserves_col_flags():
    from darq_xray.compose.recipe import Col, PanelRef
    from darq_xray.gui.widgets.layout_arranger import ArrangeDialog

    w = _fb_win()
    w.add_panels([_mk_panel("a"), _mk_panel("b"), _mk_panel("c")])
    root = w.recipe().layout
    root.children = [
        PanelRef("a"),
        Col(
            [PanelRef("b"), PanelRef("c")],
            pinned_width_cm=4.0,
            group_label="G",
            shared_x=True,
            shared_colorbar=True,
            shared_clim=(-1.0, 1.0),
            gap_cm=0.0,
            fill_height=True,
        ),
    ]
    root.gap_cm = 1.5
    w._rebuild_tree()
    dlg = ArrangeDialog(w.recipe(), w._style)
    assert dlg._warning.isHidden()
    assert dlg._arranger.grid() == [["a"], ["b", "c"]]
    dlg._on_apply()
    col = dlg.result_layout.children[1]
    assert isinstance(col, Col)
    assert col.shared_x and col.shared_colorbar and col.group_label == "G"
    assert col.pinned_width_cm == 4.0 and col.shared_clim == (-1.0, 1.0)
    assert col.gap_cm == 0.0  # per-container gap survives a rearrange
    assert col.fill_height is True
    assert dlg.result_layout.gap_cm == 1.5  # ...and so does the root row's
    w._debounce.stop()


def test_arrange_dialog_flatten_path_warns_and_seeds_one_column():
    from darq_xray.compose.recipe import Col, Spacer
    from darq_xray.gui.widgets.layout_arranger import ArrangeDialog

    w = _fb_win()
    w.add_panels([_mk_panel("a"), _mk_panel("b")])
    w.recipe().layout.children.append(Spacer(1.0, 1.0))  # unmappable
    dlg = ArrangeDialog(w.recipe(), w._style)
    assert not dlg._warning.isHidden()
    assert dlg._arranger.grid() == [["a", "b"]]
    dlg._on_apply()
    assert isinstance(dlg.result_layout.children[0], Col)  # one two-tile column
    w._debounce.stop()
