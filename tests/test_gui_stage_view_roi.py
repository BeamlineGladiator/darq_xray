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


def _exp_sto2_rois():
    from dfxm.config.models import Experiment

    return Experiment(
        darfix_roi="105,230,1832,1266", analysis_roi_x="0,1832", analysis_roi_y="400,1100"
    )


def test_roi_deviation_marker_toggles():
    from dfxm.stages import visualize
    from gui.stage_view import StageView

    _ = QApplication.instance() or QApplication([])
    view = StageView("visualize", visualize.STAGE, _exp_sto2_rois())
    lbl = view._form._labels["roi_x"]
    assert "⚠" not in lbl.text()  # pre-filled value matches the experiment
    view._form.set_values({"roi_x": "1,2"})
    assert "⚠" in lbl.text()
    assert "0,1832" in lbl.toolTip()  # tooltip names the experiment value
    view._form.set_values({"roi_x": "0,1832"})
    assert "⚠" not in lbl.text()


def test_rocking_marker_catches_the_incident_entry():
    from dfxm.stages import rocking
    from gui.stage_view import StageView

    _ = QApplication.instance() or QApplication([])
    view = StageView("rocking", rocking.STAGE, _exp_sto2_rois())
    assert view._form.values()["roi_y"] == "630,1330"  # pre-filled, detector frame
    view._form.set_values({"roi_y": "230,1266"})  # darfix origin+size — the classic mistake
    assert "⚠" in view._form._labels["roi_y"].text()


def test_no_marker_without_experiment_rois():
    from dfxm.config.models import Experiment
    from dfxm.stages import visualize
    from gui.stage_view import StageView

    _ = QApplication.instance() or QApplication([])
    view = StageView("visualize", visualize.STAGE, Experiment())
    view._form.set_values({"roi_x": "1,2"})
    assert "⚠" not in view._form._labels["roi_x"].text()  # nothing to deviate from
