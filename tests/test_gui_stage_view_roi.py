"""StageView shows a Pick ROI… button for roi-grouped specs and writes back per axis."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


def test_pick_roi_writes_pair_encoding(monkeypatch):
    from gui.main_window import MainWindow

    _ = QApplication.instance() or QApplication([])
    win = MainWindow()
    view = win._views["visualize"]  # roi_x/roi_y, roi_axis x/y
    assert view._roi_buttons, "expected a Pick ROI… button for the crop group"

    import gui.stage_view as SV

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (2, 6, 1, 8)  # r0,r1,c0,c1

        def exec(self):
            return 1

    monkeypatch.setattr(SV, "ROIPickerDialog", _FakePicker, raising=False)
    monkeypatch.setattr(SV, "_roi_previews_for", lambda name, vals: [("x", lambda: (None, 1, 1))])
    view._on_pick_roi_group("crop")
    vals = view._form.values()
    assert vals["roi_x"] == "1,8"  # c0,c1
    assert vals["roi_y"] == "2,6"  # r0,r1
    win.close()


def test_pick_roi_writes_both_encoding(monkeypatch):
    from gui.main_window import MainWindow

    _ = QApplication.instance() or QApplication([])
    win = MainWindow()
    view = win._views["strain"]  # single 'roi', roi_axis both

    import gui.stage_view as SV

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (2, 6, 1, 8)

        def exec(self):
            return 1

    monkeypatch.setattr(SV, "ROIPickerDialog", _FakePicker, raising=False)
    monkeypatch.setattr(SV, "_roi_previews_for", lambda name, vals: [("x", lambda: (None, 1, 1))])
    view._on_pick_roi_group("crop")
    assert view._form.values()["roi"] == "2,6,1,8"  # r0,r1,c0,c1
    win.close()
