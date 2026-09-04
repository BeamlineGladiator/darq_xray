"""Offscreen construction + threaded-batch tests for the profiles Replot dialog
(delegates rendering to the tested Qt-free core in darq_xray.stages.profiles).

Fixture pattern mirrors gui_smoke.py step [33]."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


def _mini(path):
    u = np.linspace(-10.0, 10.0, 81)
    v = np.linspace(-8.0, 8.0, 65)
    uu, vv = np.meshgrid(u, v)
    offsets = np.array([-1.0, 0.0, 1.0])
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (
            ("raw_sum", "raw_sum", "gray"),
            ("strain", "strain", "RdBu_r"),
        ):
            g = f.create_group(vid)
            g.attrs["kind"] = kind
            g.attrs["cbar_label"] = "value"
            g.attrs["cmap"] = cmap
            g.attrs["title"] = vid
            g.attrs["vmin"] = -10.0
            g.attrs["vmax"] = 10.0
            sg = g.create_group("oblique_full")
            stack = np.stack([0.7 * uu - 1.3 * vv + o for o in offsets], axis=0).astype(np.float32)
            sg.create_dataset("slices", data=stack)
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offsets)


def _job():
    return {
        "name": "oblique_full",
        "offset_um": 0.0,
        "start_uv": [-5, -3],
        "end_uv": [5, 3],
        "n_samples": 20,
        "width_pixels": 1,
        "fig_name": "test_profiles_replot",
    }


def test_dialog_populates_tree_and_renders(tmp_path):
    from darq_xray.gui.widgets.profiles_replot import ProfilesReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    out = tmp_path / "replots"
    dlg = ProfilesReplotDialog(str(h5), [_job()], style=None, out_default=str(out))
    assert dlg._tree.topLevelItemCount() == 1
    assert dlg._render_btn.isEnabled()
    written = dlg.render_selection(str(out))
    assert written and all(os.path.exists(p) for p in written)
    assert not any(p.endswith(".csv") for p in os.listdir(out))


# -- busy indication: threaded batch (Task 7) --------------------------------


def test_on_render_runs_batch_with_overlay_and_status(tmp_path):
    from darq_xray.gui.widgets.profiles_replot import ProfilesReplotDialog
    from tests.qt_helpers import wait_batch_idle

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    out = tmp_path / "replots"
    dlg = ProfilesReplotDialog(str(h5), [_job()], style=None, out_default=str(out))
    dlg.show()  # BusyOverlay.active reads isVisible(), which needs a shown ancestor chain
    dlg._on_render()
    # profiles is always a single batch item (used_stems dedup + shared trace
    # margins are per-call state) -> indeterminate spinner, no progress bar.
    assert dlg._batch.running and dlg._batch._overlay.active
    assert not dlg._batch._overlay._determinate
    assert not dlg._render_btn.isEnabled()
    wait_batch_idle(dlg)
    assert not dlg._batch._overlay.active and dlg._render_btn.isEnabled()
    assert dlg.written and all(os.path.exists(p) for p in dlg.written)
    assert not any(p.endswith(".csv") for p in os.listdir(out))
    assert "wrote" in dlg._status.text()
    dlg.deleteLater()


def test_on_batch_done_marks_cancelled_result(tmp_path):
    """Pre-first-item cancel race: request_stop() fires before the worker
    thread reaches _whole_batch, so _result_box is still empty when
    _on_batch_done runs. Status must still carry the 'cancelled — ' prefix
    (parity with slice_replot.py / replot_dialog.py)."""
    from darq_xray.gui.widgets.profiles_replot import ProfilesReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = ProfilesReplotDialog(str(h5), [_job()], style=None, out_default=str(tmp_path / "o"))
    dlg._result_box = []
    dlg._last_out_dir = str(tmp_path / "o")
    dlg._on_batch_done([], "", True)
    assert dlg._status.text().startswith("cancelled — ")
    dlg.deleteLater()


def test_reject_while_running_cancels_instead_of_closing(tmp_path, monkeypatch):
    import threading

    from darq_xray.gui.widgets.profiles_replot import ProfilesReplotDialog
    from darq_xray.stages import profiles as pr
    from tests.qt_helpers import wait_batch_idle

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    started = threading.Event()
    release = threading.Event()
    calls: list = []
    real_render_replot = pr.render_replot

    def fake_render_replot(h5_path, jobs, style, clim, out_dir, dpi=150, params=None):
        calls.append(jobs)
        started.set()  # let the main thread know the (only) item is in flight
        release.wait(30)  # hold it so the batch is still running when reject() fires
        return real_render_replot(h5_path, jobs, style, clim, out_dir, dpi=dpi, params=params)

    monkeypatch.setattr(pr, "render_replot", fake_render_replot)
    _app = QApplication.instance() or QApplication([])
    out = tmp_path / "replots"
    dlg = ProfilesReplotDialog(str(h5), [_job()], style=None, out_default=str(out))
    dlg.show()
    dlg._on_render()
    assert dlg._batch.running
    assert started.wait(5), "worker never reached render_replot"
    dlg.reject()  # cancel request, not a close — dialog stays open, batch keeps running
    assert dlg.isVisible()
    assert dlg._batch.running
    release.set()
    wait_batch_idle(dlg)
    # single item + cooperative cancel: the in-flight item always completes,
    # so this finishes as a normal (non-cancelled) batch — see BatchWorker docs.
    assert len(calls) == 1
    assert "wrote" in dlg._status.text()
    dlg.reject()  # now it closes normally
    assert not dlg._batch.running
    dlg.deleteLater()
