"""Offscreen construction test for the slices Replot dialog (delegates rendering
to the tested Qt-free core in dfxm.stages.slices)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
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
            sg = g.create_group("plane_a")
            sg.create_dataset("slices", data=np.zeros((2, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))


def test_dialog_populates_tree_and_renders(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    out = tmp_path / "replots"
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(out))
    # two volume groups at the top level
    assert dlg._tree.topLevelItemCount() == 2
    # select everything and render straight through the core
    dlg.select_all()
    assert dlg._selections(), "select_all() left the selection empty"
    written = dlg.render_selection(str(out))
    assert written and all(os.path.exists(p) for p in written)


def test_slice_replot_dialog_passes_roi(tmp_path, monkeypatch):
    from dfxm.stages import slices as sl
    from gui.widgets.slice_replot import SliceReplotDialog

    captured = {}

    def fake_render_replot(h5, selections, style, clim, out_dir, roi=None, **kw):
        captured["roi"] = roi
        return []

    monkeypatch.setattr(sl, "render_replot", fake_render_replot)
    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path))
    dlg.select_all()
    dlg._r0.setText("0")
    dlg._r1.setText("2")
    dlg._c0.setText("0")
    dlg._c1.setText("2")
    dlg.render_selection(str(tmp_path))
    assert captured["roi"] == (0, 2, 0, 2)
