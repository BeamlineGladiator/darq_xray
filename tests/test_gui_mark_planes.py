"""Offscreen tests: MarkPlanesDialog toggles and persists marks."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.stages import slices as sl  # noqa: E402


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


def test_mark_toggle_and_save(tmp_path):
    from gui.widgets.mark_planes import MarkPlanesDialog

    _app = QApplication.instance() or QApplication([])
    h5 = _mini(tmp_path / "s.h5")
    dlg = MarkPlanesDialog(h5)
    assert dlg._slice_box.currentText() == "oblique_full"
    dlg._browser.set_plane(2)  # offset +2.0
    dlg._mark_btn.setChecked(True)  # mark it
    dlg._browser.set_plane(0)
    assert not dlg._mark_btn.isChecked()  # button tracks the current plane
    dlg._mark_btn.setChecked(True)  # mark -2.0 too
    assert dlg._dirty()
    dlg._on_save()
    assert dlg.saved
    assert not dlg._dirty()
    assert sl.read_marks(h5) == {"oblique_full": [-2.0, 2.0]}
    # unmark one and save again -> replaced
    dlg._browser.set_plane(0)
    dlg._mark_btn.setChecked(False)
    dlg._on_save()
    dlg.done(0)
    assert sl.read_marks(h5) == {"oblique_full": [2.0]}


def test_dialog_loads_existing_marks(tmp_path):
    from gui.widgets.mark_planes import MarkPlanesDialog

    _app = QApplication.instance() or QApplication([])
    h5 = _mini(tmp_path / "s.h5")
    sl.write_marks(h5, "oblique_full", [0.0])
    dlg = MarkPlanesDialog(h5)
    dlg._browser.set_plane(1)  # offset 0.0
    assert dlg._mark_btn.isChecked()
    assert not dlg._dirty()
    dlg.done(0)
