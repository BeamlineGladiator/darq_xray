"""Tests for the Pin planes… dialog and its slices-form wiring."""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.widgets.pin_planes import PinPlanesDialog  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _sweep_h5(tmp_path):
    p = str(tmp_path / "oblique_slices.h5")
    with h5py.File(p, "w") as f:
        sg = f.create_group("strain").create_group("oblique")
        sg.create_dataset("slices", data=np.zeros((3, 4, 5), dtype=np.float32))
        sg.create_dataset("u_um", data=np.linspace(0, 4, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 3, 4))
        sg.create_dataset("offsets_um", data=np.array([-2.0, 0.0, 2.0]))
        sg.attrs["normal"] = [0.0, 0.0, 1.0]
        sg.attrs["origin"] = [0.0, 0.0, 0.0]
        sg.attrs["up"] = [0.0, 1.0, 0.0]
        for k, v in (
            ("half_u", 2.0),
            ("half_v", 1.5),
            ("du", 1.0),
            ("dv", 1.0),
            ("sweep_step_um", 2.0),
        ):
            sg.attrs[k] = v
        f["strain"].attrs["kind"] = "strain"
    return p


def test_pin_dialog_writes_pinned_specs_for_checked_planes(tmp_path):
    dlg = PinPlanesDialog(_sweep_h5(tmp_path))
    assert not dlg._panel.checked_plane_keys()  # pinning starts unchecked
    dlg._panel._items[("oblique", 2)].setCheckState(0, Qt.CheckState.Checked)
    dlg._on_ok()
    specs = json.loads(dlg.result_json)
    assert len(specs) == 1
    assert specs[0]["sweep_start_um"] == specs[0]["sweep_stop_um"] == 2.0
    assert specs[0]["half_u"] == 2.0


def test_pin_dialog_empty_selection_or_bad_file_writes_nothing(tmp_path):
    dlg = PinPlanesDialog(_sweep_h5(tmp_path))
    dlg._on_ok()  # nothing checked
    assert dlg.result_json is None
    bad = PinPlanesDialog(str(tmp_path / "missing.h5"))
    assert bad._panel._rows == [] and bad.result_json is None
