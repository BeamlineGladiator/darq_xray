"""Pure-function tests for the ROI picker's rectangle→indices mapping (no QApplication)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")  # module imports PySide6 at import time, but needs no app

from gui.widgets.roi_picker import rect_to_indices  # noqa: E402


def test_basic_floor_ceil_halfopen():
    # a rect from x[12.2, 88.9], y[40.0, 160.7] on a 200x100 (h x w) grid
    assert rect_to_indices(12.2, 88.9, 40.0, 160.7, w=100, h=200) == (40, 161, 12, 89)


def test_swapped_min_max_normalised():
    assert rect_to_indices(88.9, 12.2, 160.7, 40.0, w=100, h=200) == (40, 161, 12, 89)


def test_clamped_to_bounds():
    assert rect_to_indices(-5.0, 500.0, -5.0, 500.0, w=100, h=200) == (0, 200, 0, 100)


def test_degenerate_zero_area():
    assert rect_to_indices(50.0, 50.0, 30.0, 30.0, w=100, h=200) == (30, 30, 50, 50)
