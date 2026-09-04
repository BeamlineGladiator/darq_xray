"""Offscreen tests: shared PlaneBrowser + line-picker background switch."""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


def _mini(path, offsets=(-2.0, 0.0, 2.0)):
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offs = np.asarray(offsets, np.float64)
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
            sg.create_dataset(
                "slices", data=np.zeros((offs.size, v.size, u.size), dtype=np.float32)
            )
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offs)
    return str(path)


def test_browser_open_step_and_group_switch(tmp_path):
    from darq_xray.gui.widgets.plane_browser import PlaneBrowser

    _app = QApplication.instance() or QApplication([])
    b = PlaneBrowser(_mini(tmp_path / "s.h5"))
    assert b.slice_names() == ["oblique_full"]
    b.open_slice("oblique_full", init_offset=1.7)
    assert b.present == ["raw_sum", "strain"]
    assert b.group_id == "raw_sum"  # raw_sum preferred as reference
    assert b.plane_index == 2  # 1.7 snaps to +2.0
    b.step(-1)
    assert b.current_offset() == 0.0
    b.set_group("strain")
    assert b.attrs["title"] == "strain"
    assert b.plane_index == 1  # plane cursor survives the switch
    b.close_file()
    b.deleteLater()  # dispose the top-level canvas so it can't leak into later tests


def test_picker_background_dropdown_and_result_reference(tmp_path):
    from darq_xray.gui.widgets.line_picker import LinePickerDialog

    _app = QApplication.instance() or QApplication([])
    dlg = LinePickerDialog(_mini(tmp_path / "s.h5"), "oblique_full")
    assert dlg._bg.currentText() == "raw_sum"
    dlg._pts = [(0.0, 0.0), (1.0, 0.5)]
    dlg._bg.setCurrentText("strain")  # switch background
    assert dlg._pts == [(0.0, 0.0), (1.0, 0.5)]  # picked points survive
    assert dlg._browser.attrs["title"] == "strain"
    dlg._refresh_use_button()
    dlg._accept()
    start, end, off, fields, reference = dlg.result
    assert (start, end) == ((0.0, 0.0), (1.0, 0.5))
    assert fields is None  # all fields checked -> no restriction
    assert reference == "strain"  # the group the line was drawn against
    dlg.done(0)


def test_inject_line_reference_kwarg():
    from darq_xray.gui.viewers import inject_line_into_jobs

    base = json.dumps([{"name": "oblique_full", "offset_um": 0.0, "reference": "old"}])
    out = inject_line_into_jobs(
        base, "oblique_full", (0.0, 0.0), (1.0, 0.0), 0.0, reference="strain"
    )
    assert json.loads(out)[0]["reference"] == "strain"
    out = inject_line_into_jobs(base, "oblique_full", (0.0, 0.0), (1.0, 0.0), 0.0)
    assert json.loads(out)[0]["reference"] == "old"  # None leaves it untouched


def test_picker_info_shows_star_for_marked_plane(tmp_path):
    from darq_xray.gui.widgets.line_picker import LinePickerDialog
    from darq_xray.stages import slices as sl

    _app = QApplication.instance() or QApplication([])
    h5 = _mini(tmp_path / "m.h5")
    sl.write_marks(h5, "oblique_full", [0.0])
    dlg = LinePickerDialog(h5, "oblique_full", init_offset=0.0)
    assert "★" in dlg._info.text()
    dlg._browser.step(+1)
    assert "★" not in dlg._info.text()
    dlg.done(0)
