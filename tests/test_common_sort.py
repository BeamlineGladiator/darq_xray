"""Tests for dfxm.common.sort."""

from __future__ import annotations

import os

from dfxm.common.sort import find_matching_folders, natural_sort_key


def test_natural_sort_orders_numbers_numerically():
    items = ["x__10", "x__2", "x__1", "x__21", "x__3"]
    assert sorted(items, key=natural_sort_key) == ["x__1", "x__2", "x__3", "x__10", "x__21"]


def test_natural_sort_is_case_insensitive():
    assert sorted(["B", "a", "C"], key=natural_sort_key) == ["a", "B", "C"]


def test_find_matching_folders_returns_dirs_in_natural_order(tmp_path):
    for name in ["layer__10", "layer__2", "layer__1", "other"]:
        (tmp_path / name).mkdir()
    # a file that matches the glob must be ignored (dirs only)
    (tmp_path / "layer__99.txt").write_text("not a dir")

    found = find_matching_folders(str(tmp_path), "layer__*")
    assert [os.path.basename(f) for f in found] == ["layer__1", "layer__2", "layer__10"]


def test_find_matching_folders_empty_when_no_match(tmp_path):
    assert find_matching_folders(str(tmp_path), "nope__*") == []
