"""append_line_job (pure) + JobsFromMarksDialog checklist (offscreen)."""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from darq_xray.gui.viewers import append_line_job  # noqa: E402


def test_append_line_job_appends_never_updates():
    base = json.dumps([{"name": "oblique_full", "offset_um": 0.0, "start_uv": [9, 9]}])
    out = append_line_job(
        base, "oblique_full", (0.0, 0.0), (1.0, 0.5), 2.0, fields=["strain"], reference="strain"
    )
    jobs = json.loads(out)
    assert len(jobs) == 2  # appended, existing job untouched
    assert jobs[0]["start_uv"] == [9, 9]
    assert jobs[1] == {
        "name": "oblique_full",
        "offset_um": 2.0,
        "start_uv": [0.0, 0.0],
        "end_uv": [1.0, 0.5],
        "fields": ["strain"],
        "reference": "strain",
    }


def test_append_line_job_minimal_and_bad_json():
    out = append_line_job("not json", "s", (0.0, 0.0), (1.0, 0.0), 0.0)
    jobs = json.loads(out)
    assert len(jobs) == 1
    assert "fields" not in jobs[0] and "reference" not in jobs[0]


def test_jobs_from_marks_dialog_selection():
    from darq_xray.gui.widgets.jobs_from_marks import JobsFromMarksDialog

    _app = QApplication.instance() or QApplication([])
    dlg = JobsFromMarksDialog({"b_slice": [1.0], "a_slice": [-2.0, 0.0]})
    assert dlg._list.count() == 3  # sorted by slice then offset, all checked
    dlg._list.item(1).setCheckState(Qt.CheckState.Unchecked)  # drop a_slice @ 0.0
    dlg._on_ok()
    assert dlg.selected == [("a_slice", -2.0), ("b_slice", 1.0)]
