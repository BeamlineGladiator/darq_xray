import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.common.figures import ReplotGroup  # noqa: E402
from gui.widgets.replot_dialog import ReplotDialog  # noqa: E402

_app = QApplication.instance() or QApplication([])


def test_replot_dialog_collects_selection_clim_roi(tmp_path):
    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")  # existence is all the dialog checks before catalog_fn
    captured = {}

    def catalog_fn(path):
        return [ReplotGroup(key="A", label="A", item_labels=["l0", "l1"])]

    def render_fn(path, selections, style, clim, roi, out_dir):
        captured["selections"] = selections
        captured["clim"] = clim
        captured["roi"] = roi
        return [os.path.join(out_dir, "x.png")]

    dlg = ReplotDialog(str(h5), catalog_fn, render_fn, style=None, out_default=str(tmp_path))
    dlg.select_all()
    dlg._vmin.setText("0.5")
    dlg._r0.setText("0")
    dlg._r1.setText("2")
    dlg._c0.setText("0")
    dlg._c1.setText("3")
    written = dlg.render_selection(str(tmp_path))
    assert captured["selections"] == [("A", None)]
    assert captured["clim"] == (0.5, None)
    assert captured["roi"] == (0, 2, 0, 3)
    assert written == [os.path.join(str(tmp_path), "x.png")]


def test_replot_dialog_partial_roi_ignored(tmp_path):
    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")
    captured = {}

    def render_fn(path, selections, style, clim, roi, out_dir):
        captured["roi"] = roi
        return []

    dlg = ReplotDialog(
        str(h5),
        lambda p: [ReplotGroup(key="A", label="A", item_labels=["l0"])],
        render_fn,
        out_default=str(tmp_path),
    )
    dlg.select_all()
    dlg._r0.setText("0")  # only one box filled → ROI ignored
    dlg.render_selection(str(tmp_path))
    assert captured["roi"] is None


def test_replot_dialog_shows_group_pixel_size(tmp_path):
    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")

    def catalog_fn(path):
        return [ReplotGroup(key="A", label="chi", item_labels=["l0"], shape=(4, 5))]

    dlg = ReplotDialog(str(h5), catalog_fn, lambda *a: [], out_default=str(tmp_path))
    node_text = dlg._tree.topLevelItem(0).text(0)
    assert "4×5 px" in node_text  # ROI hint annotated on the group node
