"""Qt-free tests: marked rows in the plane-selection model."""

from __future__ import annotations

from dfxm.stages.slices import ReplotEntry
from gui.widgets.plane_selection_model import PlaneRow, build_slice_rows, filter_rows


def _entries():
    return [
        ReplotEntry("raw_sum", "oblique_full", 3, [-2.0, 0.0, 2.0], shape=(7, 9), group="raw"),
        ReplotEntry("strain", "oblique_full", 3, [-2.0, 0.0, 2.0], shape=(7, 9), group="strain"),
    ]


def test_build_slice_rows_marks():
    rows = build_slice_rows(_entries(), marks={"oblique_full": [0.1]})  # snaps to 0.0
    by_key = {r.key: r for r in rows}
    assert by_key[("oblique_full", 1)].marked
    assert by_key[("oblique_full", 1)].label.startswith("★ ")
    assert not by_key[("oblique_full", 0)].marked
    assert not by_key[("oblique_full", 0)].label.startswith("★")


def test_build_slice_rows_no_marks_unchanged():
    rows = build_slice_rows(_entries())
    assert all(not r.marked for r in rows)
    assert rows[0].label == "p000  -2.00 µm"


def test_filter_rows_marked_only():
    rows = [
        PlaneRow(key=1, section="", number=1, offset=0.0, label="a", marked=True),
        PlaneRow(key=2, section="", number=2, offset=1.0, label="b"),
    ]
    assert [r.key for r in filter_rows(rows, "", marked_only=True)] == [1]
    assert [r.key for r in filter_rows(rows, "2", marked_only=True)] == []
    assert [r.key for r in filter_rows(rows, "")] == [1, 2]
