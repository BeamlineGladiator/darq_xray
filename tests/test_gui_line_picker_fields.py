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
    # unticking one narrows the returned set
    dlg._field_boxes["raw_sum"].setCheckState(Qt.CheckState.Unchecked)
    assert dlg.selected_fields() == ["strain"]
