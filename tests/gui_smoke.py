"""Headless GUI smoke test (run manually, not collected by pytest).

Exercises the whole vertical slice through the real widgets:

1. build + show the main window (offscreen Qt),
2. confirm the STO2 preset auto-loaded and its calibration caveat is shown,
3. run the concat stage end-to-end through the StageView and check the result,
4. confirm cancelling a long run actually kills the worker process.

Run it with::

    python3 tests/gui_smoke.py

It is named ``gui_smoke.py`` (not ``test_*``) so pytest does not collect it —
it spawns processes and needs a Qt platform, which is awkward under pytest.
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Make the package importable when run as a plain script, and force a
# windowless Qt platform so this works without a display.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

import h5py  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def _make_input() -> str:
    root = tempfile.mkdtemp()
    folder = os.path.join(root, "layer__3")
    os.makedirs(folder)
    with h5py.File(os.path.join(folder, "layer__3.h5"), "w") as f:
        for e, nfr in [("1.1", 3), ("2.1", 2), ("3.1", 4)]:
            g = f.create_group(e)
            pos = g.create_group("instrument/positioners")
            pos.create_dataset("mu", data=np.linspace(11, 11.1, nfr))
            pos.create_dataset("ccmth", data=7.1 + 0.01 * nfr)
            g.create_dataset("instrument/pco_ff/image", data=np.zeros((nfr, 2, 2), dtype="uint16"))
    return folder


def _make_maps(base_folder: str) -> str:
    """A folder with a synthetic darfix maps.h5 (ccmth COM map only)."""
    folder = os.path.join(os.path.dirname(base_folder), "strain_layer__1")
    os.makedirs(folder, exist_ok=True)
    X, Y = np.meshgrid(np.linspace(-3, 3, 30), np.linspace(-2, 2, 24))
    ccmth = 7.144 + 0.002 * np.arctan(2 * X) + 0.001 * np.arctan(1.5 * Y)
    with h5py.File(os.path.join(folder, "maps.h5"), "w") as f:
        f.create_dataset("/entry/ccmth/Center of mass/Center of mass", data=ccmth)
    return folder


def _sleeper(_params, progress=None):
    for i in range(400):
        if progress:
            progress(i / 400, f"step {i}")
        time.sleep(0.05)
    return "should-not-finish"


def main() -> int:
    from dfxm.runner import StageRunner
    from gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.show()
    app.processEvents()
    print("[1] main window built + shown")

    exp = win._experiment_panel.current_experiment()
    assert exp.name == "STO2_overnight", exp.name
    assert exp.ccmth_ref_deg == 7.144
    assert win._experiment_panel._notes.isVisible()
    print("[2] STO2 preset loaded")

    view = win._views["concat"]
    folder = _make_input()
    view._form.set_values({"mode": "single", "input_folder": folder})
    done: list[tuple[str, bool]] = []
    view.runFinished.connect(lambda name, ok: done.append((name, ok)))
    view._on_run()
    t0 = time.time()
    while not done and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.02)
    assert done == [("concat", True)], done
    app.processEvents()
    results = view._results.toPlainText()
    assert "1 ok" in results and "9 frames" in results, results
    assert win._status_items["concat"].text().startswith("✓")
    assert os.path.exists(os.path.join(folder, "layer__3_concat.h5"))
    print(f"[3] concat ran through the UI: {results.splitlines()[0]}; status ✓; output written")

    # Run the strain stage through the UI and confirm the image preview.
    assert set(win._views) == {
        "concat",
        "strain",
        "mosaicity",
        "rocking",
        "visualize",
        "paraview",
        "slices",
        "profiles",
        "matched",
    }
    sview = win._views["strain"]
    sfolder = _make_maps(folder)
    sview._form.set_values(
        {
            "mode": "single",
            "input_folder": sfolder,
            "ccmth_ref_deg": 7.144,
            "output_dir": os.path.join(sfolder, "out"),
        }
    )
    sdone: list[tuple[str, bool]] = []
    sview.runFinished.connect(lambda name, ok: sdone.append((name, ok)))
    sview._on_run()
    t0 = time.time()
    while not sdone and time.time() - t0 < 90:
        app.processEvents()
        time.sleep(0.02)
    assert sdone == [("strain", True)], sdone
    app.processEvents()
    assert win._status_items["strain"].text().startswith("✓")
    assert sview._image.pixmap() is not None and not sview._image.pixmap().isNull()
    print("[4] strain ran through the UI: status ✓; strain-map image previewed")

    # Interactive viewers are present but LAZY: a 3D tab on volume stages, a
    # pick button on profiles, and pyvista must NOT have been imported yet.
    for name in ("visualize", "rocking"):
        tabs = win._views[name]._tabs
        assert any(tabs.tabText(i) == "3D" for i in range(tabs.count())), name
    assert win._views["profiles"]._pick_btn is not None
    assert win._views["visualize"]._vol3d is not None
    assert "pyvista" not in sys.modules and "pyvistaqt" not in sys.modules
    print("[5] interactive viewers wired and lazy (no pyvista import at startup)")

    # Cancel kills the worker.
    runner = StageRunner(_sleeper, {}, start_method="fork")
    runner.start()
    time.sleep(0.4)
    assert runner.is_alive()
    runner.cancel(timeout=2.0)
    assert not runner.is_alive()
    print("[6] cancel terminated a long-running worker")

    # Forms: essentials visible, Advanced expander collapsed, values round-trip.
    for name, view in win._views.items():
        form = view._form
        spec = view._spec
        assert set(form.values()) == {p.name for p in spec.params}, name
        n_adv = sum(1 for p in spec.params if p.advanced)
        if n_adv:
            assert form._adv_toggle is not None, name
            assert f"({n_adv} settings)" in form._adv_toggle.text(), name
            assert not form._adv_box.isVisible(), name
    # focus_param reveals an advanced field
    sform = win._views["strain"]._form
    sform._adv_toggle.setChecked(False)
    sform.focus_param("ccmth_ref_deg")
    assert sform._adv_toggle.isChecked()
    print("[7] grouped forms: essentials/advanced split + value round-trip OK")

    # Help panel follows focus and idles on the stage description.
    sview = win._views["strain"]
    assert "strain" in sview._help._label.text().lower()
    sview._form.focus_param("ccmth_ref_deg")
    app.processEvents()
    help_text = sview._help._label.text()
    assert "Bragg" in help_text and "calibration" in help_text.lower()
    print("[8] help panel idles on description and follows focus")

    # Compact experiment header: summary line + notes + Edit dialog.
    panel = win._experiment_panel
    assert "7.144" in panel._summary.text()
    assert panel._notes.isVisible()
    dlg = panel._make_dialog()
    dlg.show()
    app.processEvents()
    vals = dlg._form.values()
    assert vals["ccmth_ref_deg"] == 7.144
    dlg._form.set_values({"description": "smoke-edited"})
    dlg.accept()
    panel._set_experiment(dlg.experiment())  # what _on_edit does after exec()
    app.processEvents()
    assert panel.current_experiment().description == "smoke-edited"
    print("[9] compact experiment header + edit dialog round-trip")

    # Pipeline rail: Overview first, darfix row disabled, concat optional.
    from PySide6.QtCore import Qt as _Qt

    nav = win._nav
    assert nav.item(0).text().endswith("Overview")
    texts = [nav.item(i).text() for i in range(nav.count())]
    darfix_rows = [i for i, t in enumerate(texts) if "darfix" in t]
    assert len(darfix_rows) == 1
    assert nav.item(darfix_rows[0]).flags() == _Qt.ItemFlag.NoItemFlags
    concat_row = next(i for i, t in enumerate(texts) if "Concatenate" in t)
    assert "(optional)" in texts[concat_row]
    assert darfix_rows[0] == concat_row + 1
    # Overview is the landing page; chips navigate.
    assert win._stack.currentWidget() is win._overview
    win._overview.stageSelected.emit("strain")
    app.processEvents()
    assert win._stack.currentWidget() is win._views["strain"]
    # Status glyphs survived the runs from steps [3]/[4].
    assert win._status_items["concat"].text().startswith("✓")
    assert win._status_items["strain"].text().startswith("✓")
    print("[10] pipeline rail + overview page wired")

    # Success banner from the earlier strain run; progress completed.
    assert sview._banner.isVisible() and sview._banner.text().startswith("✓")
    assert sview._progress.value() == 100
    # Pre-run validation blocks on a missing must_exist path (no child process).
    mview = win._views["mosaicity"]
    win._stack.setCurrentWidget(mview)
    app.processEvents()
    mview._form.set_values({"mode": "batch", "root_folder": "/nonexistent/nowhere"})
    mview._on_run()
    app.processEvents()
    assert mview._banner.isVisible()
    assert "/nonexistent/nowhere" in mview._banner.text()
    assert mview._runner is None  # blocked before launch
    # A real failing run shows the red banner with the error text.
    mview._form.set_values({"root_folder": ""})  # empty: passes must_exist, fails in-stage
    mdone: list[bool] = []
    mview.runFinished.connect(lambda name, ok: mdone.append(ok))
    mview._on_run()
    t0 = time.time()
    while not mdone and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.02)
    assert mdone == [False]
    assert mview._banner.isVisible()
    assert "root_folder" in mview._banner.text()
    assert win._status_items["mosaicity"].text().startswith("✗")
    print("[11] banner + pre-run validation + progress bar")

    print("\nGUI SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
