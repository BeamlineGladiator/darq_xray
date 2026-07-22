"""ExperimentDialog: ROI derived read-out + validation on accept/save."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

STO2 = dict(darfix_roi="105,230,1832,1266", analysis_roi_x="0,1832", analysis_roi_y="400,1100")


def _dlg(**fields):
    from dfxm.config.models import Experiment
    from gui.experiment_panel import ExperimentDialog

    _ = QApplication.instance() or QApplication([])
    return ExperimentDialog(Experiment(**fields))


def test_derived_readout_shows_both_frames():
    dlg = _dlg(**STO2)  # keep alive: PySide6 GCs the temp mid-expression otherwise
    text = dlg._roi_note.text()
    assert "x 105→1937" in text and "y 230→1496" in text  # darfix window, detector px
    assert "y 630→1330" in text  # analysis window translated to detector rows


def test_readout_blank_without_darfix():
    assert _dlg()._roi_note.text() == ""


def test_readout_tracks_edits():
    dlg = _dlg(darfix_roi="105,230,1832,1266")
    dlg._form.set_values({"analysis_roi_y": "400,1100"})
    assert "y 630→1330" in dlg._roi_note.text()


def test_readout_survives_malformed_input():
    dlg = _dlg(darfix_roi="105,230,1832,1266")
    dlg._form.set_values({"darfix_roi": "banana"})  # must not raise mid-typing
    assert "expected" in dlg._roi_note.text()


def test_accept_blocked_on_invalid(monkeypatch):
    dlg = _dlg(darfix_roi="105,230,1832,1266", analysis_roi_y="1100,400")
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dlg._on_accept()
    assert warned
    assert dlg.result() != QDialog.DialogCode.Accepted


def test_accept_passes_when_valid():
    dlg = _dlg(**STO2)
    dlg._on_accept()
    assert dlg.result() == QDialog.DialogCode.Accepted


def test_pick_analysis_roi_writes_map_pairs(monkeypatch):
    import dfxm.common.figures as F
    import gui.widgets.roi_picker as RP

    dlg = _dlg(darfix_roi="105,230,1832,1266")
    monkeypatch.setattr(
        F, "stacked_volume_previews", lambda params: [("mosa", lambda: (None, 1.0, 1.0))]
    )

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (400, 1100, 0, 1832)  # r0, r1, c0, c1

        def exec(self):
            return 1

    monkeypatch.setattr(RP, "ROIPickerDialog", _FakePicker)
    dlg._on_pick_analysis_roi()
    vals = dlg._form.values()
    assert vals["analysis_roi_x"] == "0,1832"
    assert vals["analysis_roi_y"] == "400,1100"
    assert "y 630→1330" in dlg._roi_note.text()  # read-out updated by the write-back
