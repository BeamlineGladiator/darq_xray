"""Offscreen test: the Pick-line dialog exposes field checkboxes and returns them."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _mini(path):
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    with h5py.File(path, "w") as f:
        for vid in ("raw_sum", "strain"):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cmap"] = "gray"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "v"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("oblique_full")
            sg.create_dataset("slices", data=np.zeros((1, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))


def test_picker_exposes_field_checkboxes(tmp_path):
    from gui.widgets.line_picker import LinePickerDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = LinePickerDialog(str(h5), "oblique_full")
    # one checkbox per present field, all checked by default
    assert set(dlg.selected_fields()) == {"raw_sum", "strain"}
    # when all boxes are checked field_restriction() returns None (no restriction)
    assert dlg.field_restriction() is None
    # unticking one narrows the returned set
    dlg._field_boxes["raw_sum"].setCheckState(Qt.CheckState.Unchecked)
    assert dlg.selected_fields() == ["strain"]
    # and field_restriction() now returns the restricted list (not None)
    assert dlg.field_restriction() == ["strain"]
    dlg.done(0)


def test_use_button_disabled_when_no_fields_checked(tmp_path):
    """Use button must be disabled when all field boxes are unchecked (FIX 4)."""
    from gui.widgets.line_picker import LinePickerDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = LinePickerDialog(str(h5), "oblique_full")
    # Simulate 2-point pick directly (bypasses mpl event system)
    dlg._pts = [(0.0, 0.0), (1.0, 0.0)]
    dlg._refresh_use_button()  # apply the updated enable logic
    assert dlg._use.isEnabled(), "Use button should be enabled with 2 pts + ≥1 field"
    # Uncheck ALL field boxes — Use button must become disabled
    for box in dlg._field_boxes.values():
        box.setCheckState(Qt.CheckState.Unchecked)
    assert not dlg._use.isEnabled(), "Use button must be disabled when no fields are checked"
    dlg.done(0)


def test_inject_line_into_jobs_with_fields():
    """inject_line_into_jobs is pure — unit-test the fields= kwarg directly."""
    import json

    from gui.viewers import inject_line_into_jobs

    base = json.dumps([{"name": "oblique_full", "offset_um": 0.0}])

    # fields=["strain"] → job has "fields": ["strain"]
    result = inject_line_into_jobs(
        base, "oblique_full", (0.0, 0.0), (1.0, 0.0), 0.0, fields=["strain"]
    )
    job = json.loads(result)[0]
    assert job["fields"] == ["strain"]

    # fields=None → job has NO "fields" key (backward-compatible default)
    result = inject_line_into_jobs(base, "oblique_full", (0.0, 0.0), (1.0, 0.0), 0.0, fields=None)
    job = json.loads(result)[0]
    assert "fields" not in job
