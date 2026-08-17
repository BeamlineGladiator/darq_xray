"""LayoutArranger drag-grid widget (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui.widgets.layout_arranger import LayoutArranger, _corner_at  # noqa: E402

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
