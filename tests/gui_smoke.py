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

import tempfile as _tempfile_isolate  # noqa: E402

from PySide6.QtCore import QSettings as _QSettingsIsolate  # noqa: E402
from PySide6.QtWidgets import QApplication as _QAppIsolate  # noqa: E402

_QAppIsolate.setOrganizationName("dfxm-smoke")
_QAppIsolate.setApplicationName("pipeline-smoke")
_QSettingsIsolate.setDefaultFormat(_QSettingsIsolate.Format.IniFormat)
_QSettingsIsolate.setPath(
    _QSettingsIsolate.Format.IniFormat,
    _QSettingsIsolate.Scope.UserScope,
    _tempfile_isolate.mkdtemp(),
)

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
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

    # Export button wired on the Output tab after a successful strain run.
    from dfxm.common.plotting import PUBLICATION_STYLE as _PUB_STYLE
    from gui.widgets.export_dialog import ExportDialog as _ExportDialog

    assert hasattr(sview, "_export_btn"), "StageView missing _export_btn"
    assert sview._export_btn.isEnabled(), "Export button should be enabled after a successful run"
    figs = sview._figures()
    assert isinstance(figs, list), f"_figures() returned {type(figs)}, expected list"
    assert len(figs) > 0, "strain _figures() returned empty list after a successful run"
    # Construct the dialog (do NOT call .exec() — that blocks).  Destroy it
    # immediately so it does not steal Qt focus from the main window and
    # break the help-panel assertions in later steps.
    _edlg = _ExportDialog(figs, 0, _PUB_STYLE)
    assert _edlg._canvas.figure is not None
    _edlg.deleteLater()
    app.processEvents()
    # A view that has never had a successful run must start with Export disabled.
    assert not win._views["mosaicity"]._export_btn.isEnabled(), (
        "Export button must be disabled before any successful run"
    )
    print(
        "[5] Export… button enabled after strain run; _figures() non-empty; ExportDialog constructs OK"
        "; Export disabled on un-run mosaicity view"
    )

    # Interactive viewers are present but LAZY: a 3D tab on volume stages, a
    # pick button on profiles, and pyvista must NOT have been imported yet.
    for name in ("visualize", "rocking"):
        tabs = win._views[name]._tabs
        assert any(tabs.tabText(i) == "3D" for i in range(tabs.count())), name
    assert win._views["profiles"]._pick_btn is not None
    assert win._views["visualize"]._vol3d is not None
    assert "pyvista" not in sys.modules and "pyvistaqt" not in sys.modules
    print("[6] interactive viewers wired and lazy (no pyvista import at startup)")

    # Cancel kills the worker.
    runner = StageRunner(_sleeper, {}, start_method="fork")
    runner.start()
    time.sleep(0.4)
    assert runner.is_alive()
    runner.cancel(timeout=2.0)
    assert not runner.is_alive()
    print("[7] cancel terminated a long-running worker")

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
    print("[8] grouped forms: essentials/advanced split + value round-trip OK")

    # Help panel: idles on the stage description, follows focus, reverts on
    # focus-clear, resets on stage switch; tooltips carry the same rich help.
    sview = win._views["strain"]
    assert "strain" in sview._help._label.text().lower()
    sview._form.focus_param("ccmth_ref_deg")
    app.processEvents()
    help_text = sview._help._label.text()
    assert "Bragg" in help_text and "calibration" in help_text.lower()
    # Focus leaving the fields reverts the panel to the stage description.
    sview._form.focusCleared.emit()
    app.processEvents()
    assert sview._help._current is None
    assert "strain" in sview._help._label.text().lower()
    # Switching away and back resets the panel to the stage description.
    sview._help.show_param(sview._spec.params[0])  # force it onto a field
    assert sview._help._current is not None
    win._stack.setCurrentWidget(win._overview)
    app.processEvents()
    win._stack.setCurrentWidget(sview)
    app.processEvents()
    assert sview._help._current is None  # showEvent reset it to idle
    # Enriched hover tooltip on a calibration field.
    tip = sview._form._editors["ccmth_ref_deg"].toolTip()
    assert "Bragg" in tip and "calibration" in tip.lower()
    # Restore the landing page for the later pipeline-rail/overview checks ([11]).
    win._stack.setCurrentWidget(win._overview)
    app.processEvents()
    print("[9] help panel idles/follows/reverts/resets + enriched tooltips")

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
    print("[10] compact experiment header + edit dialog round-trip")

    # [10b] Compute pixel size from a raw scan fills the two calibration fields.
    import h5py as _h5py

    _scan_dir = tempfile.mkdtemp()
    _scan = os.path.join(_scan_dir, "mosa_scan.h5")
    with _h5py.File(_scan, "w") as _f:
        _pos = _f.create_group("1.1/instrument/positioners")
        _pos.create_dataset("mainx", data=5000.0)
        _pos.create_dataset("obx", data=273.0)
        _pos.create_dataset("ffsel", data=-60.0)
        _pos.create_dataset("ffz", data=2100.0)
        _pos.create_dataset("lenssel", data=0.0)
    pdlg = panel._make_dialog()
    pdlg.show()
    app.processEvents()
    pres = pdlg._apply_pixel_size(_scan)  # no modal dialogs on this path
    app.processEvents()
    assert pres.objective == "2x" and pres.condenser_in is True
    vals = pdlg._form.values()
    # abs=1e-6: the pixel-size fields are QDoubleSpinBox(decimals=6), which
    # rounds the stored value to 6 dp even on a programmatic setValue().
    assert vals["pixel_size_x_um"] == pytest.approx(pres.pixel_size_x_um, abs=1e-6)
    assert vals["pixel_size_y_um"] == pytest.approx(pres.pixel_size_y_um, abs=1e-6)
    pdlg.reject()
    app.processEvents()
    print("[10b] compute-pixel-size button fills X/Y from a scan's motors")

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
    print("[11] pipeline rail + overview page wired")

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
    print("[12] banner + pre-run validation + progress bar")

    # Mode-aware pre-run validation: the inactive-mode folder is never checked.
    mview._form.set_values(
        {"mode": "single", "input_folder": "", "root_folder": "/nonexistent/nowhere"}
    )
    assert mview._validate_inputs(mview._form.values()) is None  # single ignores root_folder
    mview._form.set_values(
        {"mode": "batch", "input_folder": "/nonexistent/nowhere", "root_folder": ""}
    )
    assert mview._validate_inputs(mview._form.values()) is None  # batch ignores input_folder
    mview._form.set_values({"mode": "batch", "root_folder": "/nonexistent/nowhere"})
    problem = mview._validate_inputs(mview._form.values())
    assert problem is not None and problem[0] == "root_folder"  # batch still checks root_folder
    print("[13] mode-aware pre-run validation")

    # Export dialog: build a figure spec, render a preview, export 3 formats.
    import os as _os

    import numpy as _np
    from matplotlib.figure import Figure as _Fig

    from dfxm.common import figures as _figs
    from dfxm.common.plotting import PUBLICATION_STYLE
    from gui.widgets.export_dialog import ExportDialog

    # Build a synthetic one-figure catalog so the smoke doesn't need a real run:

    def _mk(style):
        f = _Fig()
        ax = f.add_subplot(111)
        ax.imshow(_np.zeros((8, 8)), extent=[0, 8, 0, 8], origin="lower")
        return f

    spec = _figs.FigureSpec("t", "Test", "map", "test_fig", _mk)
    dlg = ExportDialog([spec], 0, PUBLICATION_STYLE)
    dlg.show()
    app.processEvents()
    assert dlg._canvas.figure is not None
    out = tempfile.mkdtemp()
    dlg._style.formats = ("png", "pdf", "svg")
    paths = dlg.export_to(out)
    app.processEvents()
    assert all(_os.path.exists(p) and _os.path.getsize(p) > 0 for p in paths)
    print("[14] export dialog renders preview + writes png/pdf/svg")

    # [14] Export-dialog robustness: plot kind, filename sanitisation, build-raising spec.
    # 14a: kind="plot" spec — scale bar forced off, export still works.
    def _mk_plot(style):
        f = _Fig()
        f.add_subplot(111).plot([0, 1], [0, 1])
        return f

    plot_spec = _figs.FigureSpec("p", "Plot", "plot", "plot_fig", _mk_plot)
    plot_dlg = ExportDialog([plot_spec], 0, PUBLICATION_STYLE)
    plot_dlg.show()
    app.processEvents()
    plot_out = tempfile.mkdtemp()
    plot_dlg._style.formats = ("png",)
    plot_paths = plot_dlg.export_to(plot_out)
    assert len(plot_paths) == 1 and _os.path.exists(plot_paths[0])

    # 14b: filename with a space is sanitised to underscore.
    def _mk_space(style):
        f = _Fig()
        f.add_subplot(111)
        return f

    space_spec = _figs.FigureSpec("s", "Spaced", "map", "my fig", _mk_space)
    space_dlg = ExportDialog([space_spec], 0, PUBLICATION_STYLE)
    space_out = tempfile.mkdtemp()
    space_dlg._style.formats = ("png",)
    space_paths = space_dlg.export_to(space_out)
    assert len(space_paths) == 1
    assert _os.path.basename(space_paths[0]) == "my_fig.png"

    # 14c: a spec whose build() raises must not crash the ExportDialog constructor.
    def _boom(style):
        raise FileNotFoundError("nope")

    boom_spec = _figs.FigureSpec("b", "Boom", "map", "boom_fig", _boom)
    boom_dlg = ExportDialog([boom_spec], 0, PUBLICATION_STYLE)  # must not raise
    app.processEvents()
    assert boom_dlg._canvas.figure is not None  # error figure was installed
    print("[15] plot-kind round-trip, filename sanitisation, build-raising spec")

    # [16] Session global style: MainWindow.global_plot_style() + StyleControls + ExportDialog.
    from dfxm.common.plotting import PlotStyle as _PlotStyle
    from gui.widgets.export_dialog import StyleControls as _StyleControls

    # global_plot_style() returns a PlotStyle (a copy of PUBLICATION_STYLE).
    session_style = win.global_plot_style()
    assert isinstance(session_style, _PlotStyle), type(session_style)
    # It must be a copy — not the same object as the module constant.
    assert session_style is not PUBLICATION_STYLE

    # StyleControls bound to the session style mutates it in place and emits changed.
    changed_fired: list[int] = []
    sc = _StyleControls(session_style)
    sc.changed.connect(lambda: changed_fired.append(1))
    original_font_scale = session_style.font_scale
    # Simulate a user changing the font scale via the spin box.
    new_val = original_font_scale + 0.5
    sc._w_font_scale.setValue(new_val)
    app.processEvents()
    assert session_style.font_scale == new_val, session_style.font_scale
    assert len(changed_fired) > 0, "StyleControls.changed never fired"

    # ExportDialog constructed from the session style starts with that style.
    def _mk2(style):
        from matplotlib.figure import Figure as _Fig2

        f = _Fig2()
        f.add_subplot(111)
        return f

    spec2 = _figs.FigureSpec("g", "Global", "map", "global_fig", _mk2)
    edlg2 = ExportDialog([spec2], 0, win.global_plot_style())
    assert edlg2._global is win.global_plot_style()
    edlg2.deleteLater()
    sc.deleteLater()
    app.processEvents()
    print(
        "[16] global_plot_style() returns PlotStyle; StyleControls mutates it + emits changed;"
        " ExportDialog uses session style"
    )

    # [17] ExportDialog._on_reset re-syncs controls via set_style().
    from dataclasses import replace as _replace

    from dfxm.common.plotting import PUBLICATION_STYLE as _PUB_STYLE2
    from gui.widgets.export_dialog import ExportDialog as _ExportDialog2

    def _mk3(style):
        from matplotlib.figure import Figure as _Fig3

        f = _Fig3()
        f.add_subplot(111)
        return f

    spec3 = _figs.FigureSpec("r", "Reset", "map", "reset_fig", _mk3)
    global_for_reset = _replace(_PUB_STYLE2)  # independent baseline
    rdlg = _ExportDialog2([spec3], 0, global_for_reset)
    app.processEvents()
    original_font_scale = global_for_reset.font_scale
    # Mutate via the widget so the handler fires and _style is updated.
    rdlg._controls._w_font_scale.setValue(original_font_scale + 1.0)
    app.processEvents()
    assert rdlg._style.font_scale == original_font_scale + 1.0, rdlg._style.font_scale
    # Now reset — should rebind controls to a fresh copy of global_for_reset.
    rdlg._on_reset()
    app.processEvents()
    assert rdlg._style.font_scale == original_font_scale, (
        f"_style.font_scale after reset: {rdlg._style.font_scale} != {original_font_scale}"
    )
    assert rdlg._controls._w_font_scale.value() == original_font_scale, (
        f"widget value after reset: {rdlg._controls._w_font_scale.value()} != {original_font_scale}"
    )
    rdlg.deleteLater()
    app.processEvents()
    print("[17] ExportDialog._on_reset re-syncs _style and widget via set_style()")

    # [18] export_all: batch a stage's catalog; resilient to a failing figure.
    import glob as _glob
    import tempfile as _tempfile

    from dfxm.common.figures import FigureSpec as _FigureSpec

    strain_view = win._views["strain"]
    assert hasattr(strain_view, "export_all"), "StageView missing export_all"
    assert hasattr(strain_view, "_export_all_btn"), "StageView missing _export_all_btn"
    assert strain_view._export_all_btn.isEnabled(), (
        "Export all… button must be enabled after a successful strain run"
    )
    # Not-yet-run view must be disabled.
    assert not win._views["mosaicity"]._export_all_btn.isEnabled(), (
        "Export all… must be disabled before any successful run"
    )

    # 18a: drive export_all on the real strain catalog — every spec should
    # either succeed or record a graceful per-figure failure; the batch must
    # never raise.
    out18 = _tempfile.mkdtemp()
    summary18 = strain_view.export_all(out18)
    assert isinstance(summary18, list), f"export_all returned {type(summary18)}"
    assert len(summary18) > 0, "export_all summary is empty"
    # Each entry is an ExportResult(figure_id, ok, error_or_None).
    for r in summary18:
        assert isinstance(r.figure_id, str), f"figure_id not str: {r.figure_id!r}"
        assert isinstance(r.ok, bool), f"ok not bool: {r.ok!r}"
        assert r.ok or isinstance(r.error, str), (
            f"failed entry {r.figure_id!r} has non-str error: {r.error!r}"
        )
    # Map specs should all succeed (the strain stacked volume exists).
    failed_maps = [(r.figure_id, r.error) for r in summary18 if "map" in r.figure_id and not r.ok]
    assert not failed_maps, f"Map figure(s) failed in export_all: {failed_maps}"
    # Files must exist for every successful spec × every format.
    session_style18 = win.global_plot_style()
    ok_ids = [r.figure_id for r in summary18 if r.ok]
    written = _glob.glob(_os.path.join(out18, "*"))
    assert len(written) >= len(ok_ids) * len(session_style18.formats), (
        f"Expected ≥{len(ok_ids) * len(session_style18.formats)} files, "
        f"got {len(written)}: {written}"
    )

    # 18b: batch resilience — inject a raising spec into a minimal view-like
    # loop and confirm one bad figure never aborts the rest.
    from matplotlib.figure import Figure as _Fig18

    def _good(style):
        f = _Fig18()
        f.add_subplot(111).plot([0, 1])
        return f

    def _bad(style):
        raise RuntimeError("deliberate failure for smoke test")

    good_spec_a = _FigureSpec("g18a", "Good A", "plot", "good_fig18a", _good)
    good_spec_b = _FigureSpec("g18b", "Good B", "plot", "good_fig18b", _good)
    bad_spec = _FigureSpec("b18", "Bad", "plot", "bad_fig18", _bad)
    # Temporarily monkey-patch _figures() to return our synthetic catalog.
    _orig_figures = strain_view._figures
    strain_view._figures = lambda: [good_spec_a, bad_spec, good_spec_b]
    out18b = _tempfile.mkdtemp()
    summary18b = strain_view.export_all(out18b)
    strain_view._figures = _orig_figures  # restore
    assert len(summary18b) == 3, f"Expected 3 entries, got {len(summary18b)}"
    assert summary18b[0].ok is True, f"first good spec should succeed: {summary18b[0]}"
    assert summary18b[1].ok is False, f"bad spec should be recorded as failed: {summary18b[1]}"
    assert summary18b[1].error == "deliberate failure for smoke test"
    assert summary18b[2].ok is True, f"second good spec should succeed: {summary18b[2]}"
    # The two good specs wrote files; the bad one wrote nothing.
    written18b = _glob.glob(_os.path.join(out18b, "*"))
    assert len(written18b) >= 2 * len(session_style18.formats), (
        f"Expected ≥{2 * len(session_style18.formats)} files from good specs, "
        f"got {len(written18b)}: {written18b}"
    )
    print(
        f"[18] export_all: {len(summary18)} strain figures batched "
        f"({sum(1 for r in summary18 if r.ok)} ok); "
        "batch-resilience proven (bad spec recorded, good specs still wrote)"
    )

    # [19] save_spec writes atomically: a per-format failure leaves no partial
    # (".part") or corrupt file at the target, the good formats still write, and
    # the built Figure is cleared afterwards.
    from dfxm.common.plotting import PlotStyle as _PS19
    from gui.widgets.export_dialog import save_spec as _save_spec19

    _built19 = {}

    def _build19(style):
        f = _Fig18()
        f.add_subplot(111).imshow([[0, 1], [1, 0]])
        _built19["fig"] = f
        return f

    spec19 = _FigureSpec("s19", "S19", "map", "weird name/with:chars", _build19)
    out19 = _tempfile.mkdtemp()
    written19 = _save_spec19(spec19, out19, _PS19(formats=("png", "zzz")))
    files19 = _os.listdir(out19)
    assert [_os.path.basename(w) for w in written19] == ["weird_name_with_chars.png"], written19
    assert not any(f.endswith((".part", ".zzz")) for f in files19), (
        f"atomic write left a partial/corrupt file: {files19}"
    )
    assert len(_built19["fig"].axes) == 0, "save_spec did not clear the Figure"
    print("[19] save_spec: atomic write (no .part/corrupt on failure) + Figure cleared")

    # [20] Theme: light by default, toggling restyles the app + embedded canvases.
    from matplotlib.colors import to_hex

    from gui import theme as _theme
    from gui.widgets.mpl_canvas import MplCanvas as _MplCanvas

    tc = _theme.ThemeController.instance()
    tc.set_mode("light")
    assert "#009682" in app.styleSheet()  # KIT green in light QSS
    mc = _MplCanvas()
    assert to_hex(mc.figure.get_facecolor()) == _theme.LIGHT.mpl_facecolor
    # The left-column toggle flips to dark and restyles everything.
    win._theme_btn.setChecked(True)
    win._on_theme_toggle(True)
    app.processEvents()
    assert tc.mode == "dark"
    assert _theme.DARK.accent in app.styleSheet()
    assert to_hex(mc.figure.get_facecolor()) == _theme.DARK.mpl_facecolor  # canvas followed
    assert win._theme_btn.text() == "☾ Dark"
    # The muted concat rail row recoloured to the dark muted ink.
    assert win._status_items["concat"].foreground().color().name() == _theme.DARK.ink_muted
    win._theme_btn.setChecked(False)
    win._on_theme_toggle(False)
    app.processEvents()
    assert tc.mode == "light"
    mc.deleteLater()
    print("[20] theme toggle restyles app QSS + matplotlib canvas + rail; persistence path OK")

    # [21] Stage splitters share one middle|right width via WindowState.
    from PySide6.QtCore import QSettings as _QSettings
    from PySide6.QtWidgets import QSplitter as _QSplitter
    from PySide6.QtWidgets import QWidget as _QWidget

    from gui.window_state import WindowState as _WindowState

    _ws = _WindowState(_QSettings())
    _a = _QSplitter()
    _a.addWidget(_QWidget())
    _a.addWidget(_QWidget())
    _a.resize(1000, 200)
    _a.show()
    _b = _QSplitter()
    _b.addWidget(_QWidget())
    _b.addWidget(_QWidget())
    _b.resize(1000, 200)
    _b.show()
    app.processEvents()
    _ws.register_stage_splitter(_a)
    _ws.register_stage_splitter(_b)
    _a.setSizes([700, 300])
    _a.splitterMoved.emit(700, 1)  # simulate a user drag
    app.processEvents()
    assert _b.sizes() == _a.sizes(), (_a.sizes(), _b.sizes())
    # Real stage views expose an inner splitter registered with the window's
    # WindowState: dragging one real splitter must persist through
    # win._window_state (this assertion fails if MainWindow's registration loop
    # is removed, because then nothing connects the real splitter's move to it).
    assert win._views["strain"].inner_splitter is not None
    win._stack.setCurrentWidget(win._views["mosaicity"])
    app.processEvents()
    real_src = win._views["mosaicity"].inner_splitter
    real_src.setSizes([642, 358])
    real_sizes = real_src.sizes()
    real_src.splitterMoved.emit(642, 1)  # simulate a user drag on a real stage
    app.processEvents()
    assert win._window_state._saved_stage_sizes() == real_sizes, (
        win._window_state._saved_stage_sizes(),
        real_sizes,
    )
    _a.deleteLater()
    _b.deleteLater()
    app.processEvents()
    print("[21] shared stage-splitter width via WindowState")

    # [22] WindowState saves geometry and restores without raising.
    from PySide6.QtCore import QSettings as _QSettings22
    from PySide6.QtWidgets import QMainWindow as _QMainWindow22
    from PySide6.QtWidgets import QSplitter as _QSplitter22
    from PySide6.QtWidgets import QWidget as _QWidget22

    from gui.window_state import WindowState as _WindowState22

    _iso = _QSettings22()
    _ws22 = _WindowState22(_iso)
    _w1 = _QMainWindow22()
    _w1.resize(900, 640)
    _w1.show()
    app.processEvents()
    _ms1 = _QSplitter22()
    _ms1.addWidget(_QWidget22())
    _ms1.addWidget(_QWidget22())
    _ws22.save(_w1, _ms1)
    assert _iso.value("geometry") is not None
    _w2 = _QMainWindow22()
    _ms2 = _QSplitter22()
    _ms2.addWidget(_QWidget22())
    _ms2.addWidget(_QWidget22())
    _ws22.restore(_w2, _ms2)  # must not raise
    app.processEvents()
    # MainWindow persists on close using the app-wide (isolated) settings.
    # Prove the REAL MainWindow persists on close (not the standalone _ws22 above):
    # clear the shared in-process key first, so only MainWindow.closeEvent can
    # re-set it. This fails if MainWindow's closeEvent/save wiring is removed.
    _probe = _QSettings22()
    _probe.remove("geometry")
    _probe.sync()
    assert _probe.value("geometry") is None
    win.resize(880, 610)
    app.processEvents()
    win.close()  # MainWindow.closeEvent -> self._window_state.save(...)
    app.processEvents()
    assert _QSettings22().value("geometry") is not None
    _w1.deleteLater()
    _w2.deleteLater()
    app.processEvents()
    print("[22] window geometry + splitter state persist/restore")

    # [23] publication-style controls expose the four colormap dropdowns and
    # mutate the session style in place.
    from gui.widgets.export_dialog import StyleControls

    style = win.global_plot_style()
    assert style.cmap_mosa_com == "fast"  # new default
    controls = StyleControls(style)
    assert controls._w_cmap_mosa_com.currentText() == style.cmap_mosa_com
    assert controls._w_cmap_raw.currentText() == style.cmap_raw
    controls._w_cmap_strain.setCurrentText("seismic")
    app.processEvents()
    assert style.cmap_strain == "seismic"
    controls.deleteLater()
    app.processEvents()
    print("[23] StyleControls colormap dropdowns mutate the session style")

    # [24] _on_run injects the live publication style into the worker params.
    import gui.stage_view as _SV

    captured: dict = {}
    _real_runner = _SV.StageRunner

    class _RecordingRunner(_real_runner):  # type: ignore[misc,valid-type]
        def __init__(self, target, params, **kw):
            captured.update(params)
            super().__init__(target, params, **kw)

    # NB: `view` was shadowed by the forms loop above — re-fetch concat's view.
    cview = win._views["concat"]
    cdone: list[tuple[str, bool]] = []
    cview.runFinished.connect(lambda name, ok: cdone.append((name, ok)))
    _SV.StageRunner = _RecordingRunner
    try:
        cview._on_run()  # concat form still holds the valid single-folder params
        t0 = time.time()
        while not cdone and time.time() - t0 < 60:
            app.processEvents()
            time.sleep(0.02)
    finally:
        _SV.StageRunner = _real_runner
    assert cdone == [("concat", True)], cdone
    assert captured.get("plot_style", {}).get("cmap_strain") == "seismic", captured.get(
        "plot_style"
    )
    assert captured["plot_style"]["cmap_mosa_com"] == "fast"
    print("[24] runs receive the live publication style (plot_style params key)")

    # [25] the style (incl. colormaps) round-trips through QSettings on save.
    from PySide6.QtCore import QSettings as _QSettings25

    from dfxm.common.plotting import style_from_json as _style_from_json

    win._save_plot_style()
    restored = _style_from_json(_QSettings25().value("plot_style", ""))
    assert restored is not None and restored.cmap_strain == "seismic"
    assert restored.font_scale == style.font_scale
    print("[25] publication style round-trips through QSettings")

    print("\nGUI SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
