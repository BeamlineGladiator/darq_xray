import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from darq_xray.common.figures import ReplotGroup  # noqa: E402
from darq_xray.gui.widgets.replot_dialog import ReplotDialog  # noqa: E402

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
    dlg._clim._edits["A"][0].setText("0.5")  # vmin for group A
    dlg._r0.setText("0")
    dlg._r1.setText("2")
    dlg._c0.setText("0")
    dlg._c1.setText("3")
    written = dlg.render_selection(str(tmp_path))
    assert captured["selections"] == [("A", [0, 1])]
    assert captured["clim"] == {"A": (0.5, None)}  # per-group mapping keyed by ReplotGroup.key
    assert captured["roi"] == (0, 2, 0, 3)
    assert written == [os.path.join(str(tmp_path), "x.png")]


def test_replot_dialog_collects_per_group_clim(tmp_path):
    """Two groups → one vmin/vmax row each; only filled groups appear in the dict."""
    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")
    captured = {}

    def catalog_fn(path):
        return [
            ReplotGroup(key="mosa_com", label="COM", item_labels=["l0"]),
            ReplotGroup(key="mosa_fwhm", label="FWHM", item_labels=["l0"]),
        ]

    def render_fn(path, selections, style, clim, roi, out_dir):
        captured["clim"] = clim
        return []

    dlg = ReplotDialog(str(h5), catalog_fn, render_fn, out_default=str(tmp_path))
    dlg.select_all()
    dlg._clim._edits["mosa_com"][0].setText("-1")
    dlg._clim._edits["mosa_com"][1].setText("1")
    dlg._clim._edits["mosa_fwhm"][1].setText("0.3")  # vmax only; vmin stays stored
    dlg.render_selection(str(tmp_path))
    assert captured["clim"] == {"mosa_com": (-1.0, 1.0), "mosa_fwhm": (None, 0.3)}


def test_replot_dialog_defaults_to_all_selected(tmp_path):
    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")

    def catalog_fn(path):
        return [
            ReplotGroup(key="A", label="A", item_labels=["l0", "l1"]),
            ReplotGroup(key="B", label="B", item_labels=["l0"]),
        ]

    dlg = ReplotDialog(str(h5), catalog_fn, lambda *a: [], out_default=str(tmp_path))
    sels = dlg._selections()  # every group ticked on open
    assert {key for key, _idxs in sels} == {"A", "B"}


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
    qty_text = dlg._panel._qty.item(0).text()
    assert "4×5 px" in qty_text  # ROI hint annotated on the quantity row


def test_generic_dialog_planes_first_product(tmp_path):
    class _G:
        def __init__(self, key, labels):
            self.key, self.label, self.item_labels, self.shape = key, key, labels, None

    calls = []

    def catalog_fn(_path):
        return [_G("sum_intensity", ["layer 0", "layer 1"]), _G("specific_frame", ["layer 0"])]

    def render_fn(h5, selections, st, clim, roi, out):
        calls.append(selections)
        return ["x.png"]

    h5 = tmp_path / "a.h5"
    h5.write_bytes(b"")
    dlg = ReplotDialog(str(h5), catalog_fn, render_fn, out_default=str(tmp_path))
    dlg.render_selection(str(tmp_path))
    sels = dict(calls[-1])
    assert sels["sum_intensity"] == [0, 1]
    assert sels["specific_frame"] == [0]  # layer 1 skipped for this product, no error


def test_generic_dialog_filter_and_check_all_visible(tmp_path):
    class _G:
        def __init__(self, key, labels):
            self.key, self.label, self.item_labels, self.shape = key, key, labels, None

    def catalog_fn(_path):
        return [_G("A", ["layer 0", "layer 1", "layer 2"])]

    h5 = tmp_path / "a.h5"
    h5.write_bytes(b"")
    dlg = ReplotDialog(str(h5), catalog_fn, lambda *a: [], out_default=str(tmp_path))
    dlg.show()
    _app.processEvents()
    dlg._panel.set_all_checked(False)
    assert not dlg._panel.has_selection()
    dlg._panel._filter.setText("0")
    dlg._panel.check_all_visible()
    assert dlg._panel.has_selection()
    assert dlg._panel.checked_plane_keys() == [0]
    dlg._panel._filter.setText("999")
    assert dlg._panel._no_match.isVisible()


def test_pick_roi_fills_boxes(tmp_path, monkeypatch):
    import numpy as np

    from darq_xray.common.figures import ReplotGroup
    from darq_xray.gui.widgets.replot_dialog import ReplotDialog

    _ = QApplication.instance() or QApplication([])

    def catalog_fn(_h5):
        return [
            ReplotGroup(
                key="/chi/Center of mass", label="χ", item_labels=["layer 0"], shape=(200, 100)
            )
        ]

    def preview_fn(_h5, _key):
        return np.zeros((200, 100)), 0.152, 0.385

    dlg = ReplotDialog("nofile.h5", catalog_fn, lambda *a, **k: [], preview_fn=preview_fn)

    # stub the modal picker: pretend the user dragged (r0,r1,c0,c1)
    import darq_xray.gui.widgets.replot_dialog as RD

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (40, 160, 12, 88)

        def exec(self):
            return 1

    monkeypatch.setattr(RD, "ROIPickerDialog", _FakePicker, raising=False)
    dlg._on_pick_roi()
    assert (dlg._r0.text(), dlg._r1.text(), dlg._c0.text(), dlg._c1.text()) == (
        "40",
        "160",
        "12",
        "88",
    )
    dlg.deleteLater()


# -- busy indication: threaded batch (Task 6) --------------------------------


def test_on_render_runs_batch_with_overlay_and_status(tmp_path):
    from tests.qt_helpers import wait_batch_idle

    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")
    captured = {}

    def catalog_fn(path):
        return [ReplotGroup(key="A", label="A", item_labels=["l0", "l1"])]

    def render_fn(path, selections, style, clim, roi, out_dir):
        captured["selections"] = selections
        captured["clim"] = clim
        captured["roi"] = roi
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "x.png")
        with open(out_path, "wb"):
            pass
        return [out_path]

    dlg = ReplotDialog(str(h5), catalog_fn, render_fn, style=None, out_default=str(tmp_path))
    dlg.show()  # BusyOverlay.active reads isVisible(), which needs a shown ancestor chain
    dlg.select_all()
    dlg._out_edit.setText(str(tmp_path / "out"))
    dlg._on_render()
    assert dlg._batch.running and dlg._batch._overlay.active
    assert not dlg._render_btn.isEnabled()
    wait_batch_idle(dlg)
    assert not dlg._batch._overlay.active and dlg._render_btn.isEnabled()
    assert dlg.written and all(os.path.exists(p) for p in dlg.written)
    assert "wrote" in dlg._status.text()


def test_reject_while_running_cancels_instead_of_closing(tmp_path, monkeypatch):
    import threading

    from tests.qt_helpers import wait_batch_idle

    h5 = tmp_path / "vol.h5"
    h5.write_bytes(b"")
    release = threading.Event()
    calls: list = []

    def catalog_fn(path):
        return [ReplotGroup(key="A", label="A", item_labels=["l0", "l1", "l2"])]

    def render_fn(path, selections, style, clim, roi, out_dir):
        calls.append(selections)
        if len(calls) == 1:
            release.wait(30)  # hold item 1 so the batch is still running when reject() fires
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{len(calls)}.png")
        with open(out_path, "wb"):
            pass
        return [out_path]

    dlg = ReplotDialog(str(h5), catalog_fn, render_fn, style=None, out_default=str(tmp_path))
    dlg.show()
    dlg.select_all()
    dlg._out_edit.setText(str(tmp_path / "out"))
    dlg._on_render()
    assert dlg._batch.running
    dlg.reject()  # cancel request, not a close — dialog stays open, batch keeps running
    assert dlg.isVisible()
    assert dlg._batch.running
    release.set()
    wait_batch_idle(dlg)
    assert "cancelled" in dlg._status.text()
    dlg.reject()  # now it closes normally
    assert not dlg._batch.running
    dlg.deleteLater()
