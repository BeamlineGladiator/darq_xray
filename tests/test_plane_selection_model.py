"""Tests for the Qt-free planes-first selection model."""

from __future__ import annotations

from gui.widgets.plane_selection_model import (
    build_layer_rows,
    build_slice_rows,
    filter_rows,
    layer_selections,
    parse_tokens,
    slice_selections,
)


class _E:  # duck-typed dfxm.stages.slices.ReplotEntry
    def __init__(self, vid, sname, offsets):
        self.volume_id = vid
        self.slice_name = sname
        self.offsets_um = list(offsets)
        self.n_planes = len(self.offsets_um)


class _G:  # duck-typed dfxm.common.figures.ReplotGroup
    def __init__(self, key, labels):
        self.key = key
        self.item_labels = list(labels)


def _entries():
    return [
        _E("strain", "oblique", [-2.0, 0.0, 2.0]),
        _E("mosa_com_chi", "oblique", [-2.0, 0.0, 2.0]),
        _E("strain", "zsweep", [0.0, 5.0]),
    ]


def test_build_slice_rows_unions_across_volumes():
    rows = build_slice_rows(_entries())
    assert len(rows) == 5  # 3 oblique planes (once, not per volume) + 2 zsweep
    keys = {r.key for r in rows}
    assert ("oblique", 1) in keys and ("zsweep", 1) in keys
    r = next(r for r in rows if r.key == ("oblique", 2))
    assert r.section == "oblique" and r.number == 2 and r.offset == 2.0
    assert r.label == "p002  +2.00 µm"


def test_build_layer_rows_parses_z_and_unions():
    rows = build_layer_rows(
        [
            _G("sum_intensity", ["layer 0  (Z=0.00 µm)", "layer 1  (Z=2.00 µm)"]),
            _G("specific_frame", ["layer 0  (Z=0.00 µm)"]),
        ]
    )
    assert [r.number for r in rows] == [0, 1]
    assert rows[1].offset == 2.0 and rows[0].section == ""


def test_parse_tokens_classification():
    assert parse_tokens("118, 7") == [("number", 118.0), ("number", 7.0)]
    assert parse_tokens("-3.7, +4, 2.0") == [("offset", -3.7), ("offset", 4.0), ("offset", 2.0)]
    assert parse_tokens("") == []
    assert parse_tokens("zz")[0][0] == "invalid"


def test_filter_rows_number_offset_and_no_match():
    rows = build_slice_rows(_entries())
    assert filter_rows(rows, "") == rows  # blank = everything
    vis = filter_rows(rows, "1")  # plane number 1 in BOTH sections
    assert {r.key for r in vis} == {("oblique", 1), ("zsweep", 1)}
    vis = filter_rows(rows, "1.8")  # nearest oblique plane 2.0 (step 2 -> tol 1);
    assert ("oblique", 2) in {r.key for r in vis}  # zsweep nearest 0.0 is off by 1.8 > tol 2.5? no:
    # zsweep offsets [0,5]: step 5, tol 2.5 -> nearest 0.0 within tol, also matches
    assert ("zsweep", 0) in {r.key for r in vis}
    assert filter_rows(rows, "99") == []  # no plane 99 anywhere
    assert filter_rows(rows, "zz") == []  # invalid token matches nothing


def test_slice_selections_product_and_skips():
    entries = _entries()
    sels, skipped = slice_selections(
        entries,
        [("oblique", 1), ("zsweep", 1), ("zsweep", 7)],
        ["strain", "mosa_com_chi"],
    )
    assert ("strain", "oblique", [1]) in sels
    assert ("strain", "zsweep", [1]) in sels
    assert ("mosa_com_chi", "oblique", [1]) in sels
    # mosa has no zsweep group at all; zsweep plane 7 out of range
    assert any("mosa_com_chi/zsweep" in s for s in skipped)
    assert any("no plane" in s for s in skipped)


def test_layer_selections_product_and_skips():
    groups = [_G("sum_intensity", ["l0", "l1", "l2"]), _G("specific_frame", ["l0"])]
    sels, skipped = layer_selections(groups, [0, 2], ["sum_intensity", "specific_frame"])
    assert ("sum_intensity", [0, 2]) in sels
    assert ("specific_frame", [0]) in sels
    assert any("specific_frame" in s for s in skipped)
    assert layer_selections(groups, [], ["sum_intensity"])[0] == []
