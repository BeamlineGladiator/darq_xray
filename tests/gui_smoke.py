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
            # STO2's analysis window (map-frame y 400,1100) pre-fills "roi" and is
            # sized for real ~700x1832 maps; clear it for this tiny 24x30 synthetic
            # fixture so the crop doesn't fall entirely outside the array.
            "roi": "",
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
    # Confirm field-checkbox API is present and documented on the dialog class.
    # No real oblique_slices.h5 available here; full behaviour coverage lives in
    # tests/test_gui_line_picker_fields.py.
    from gui.widgets.line_picker import LinePickerDialog

    assert callable(getattr(LinePickerDialog, "selected_fields", None)), (
        "LinePickerDialog must expose callable selected_fields()"
    )
    assert callable(getattr(LinePickerDialog, "field_restriction", None)), (
        "LinePickerDialog must expose callable field_restriction()"
    )
    assert LinePickerDialog.field_restriction.__doc__, "field_restriction() must have a docstring"
    print(
        "[6] interactive viewers wired and lazy (no pyvista import at startup)"
        "; LinePickerDialog.selected_fields + field_restriction present and documented"
    )

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

    sc._w_title_scale.setValue(0.4)
    assert session_style.title_scale == 0.4, "Title scale widget did not mutate the style"
    sc._w_round_clim.setChecked(True)
    assert session_style.round_clim is True, "Round colour limits widget did not mutate the style"
    sc._w_tickfmt["strain"].setCurrentIndex(3)  # "0 decimals (plain numbers)" in new _TICK_FMTS
    assert session_style.tickfmt_strain == "0", (
        "Per-group tick-format combo must store the format value"
    )

    # trace height knob writes through to the style (blank -> None -> 3.0 default)
    sc._w_trace_height_cm.setText("4.5")
    assert session_style.trace_height_cm == 4.5, session_style.trace_height_cm
    sc._w_trace_height_cm.setText("")
    assert session_style.trace_height_cm is None

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

    # [26] SliceReplotDialog: opens from the slices view, selects all, renders one plane.
    import h5py as _h5py26
    import numpy as _np26

    from gui.widgets.slice_replot import SliceReplotDialog as _SRD

    _slice_tmp = tempfile.mkdtemp()
    _h5_path26 = os.path.join(_slice_tmp, "oblique_slices.h5")
    _u26 = _np26.linspace(-4.0, 4.0, 9)
    _v26 = _np26.linspace(-3.0, 3.0, 7)
    with _h5py26.File(_h5_path26, "w") as _f26:
        _g26 = _f26.create_group("strain")
        _g26.attrs["kind"] = "strain"
        _g26.attrs["cmap"] = "RdBu_r"
        _g26.attrs["title"] = "strain"
        _g26.attrs["cbar_label"] = "ε"
        _g26.attrs["vmin"] = -1.0
        _g26.attrs["vmax"] = 1.0
        _sg26 = _g26.create_group("oblique")
        _sg26.create_dataset(
            "slices", data=_np26.zeros((1, _v26.size, _u26.size), dtype=_np26.float32)
        )
        _sg26.create_dataset("u_um", data=_u26)
        _sg26.create_dataset("v_um", data=_v26)
        _sg26.create_dataset("offsets_um", data=_np26.array([0.0]))
        _sg26.attrs["normal"] = [0.0, 0.0, 1.0]
        _sg26.attrs["origin"] = [0.0, 0.0, 0.0]
        _sg26.attrs["up"] = [0.0, 1.0, 0.0]
        for _k26, _v26_attr in (
            ("half_u", 4.0),
            ("half_v", 3.0),
            ("du", 1.0),
            ("dv", 1.0),
            ("sweep_step_um", 1.0),
        ):
            _sg26.attrs[_k26] = _v26_attr

    # verify the Replot… button is present on the slices view
    slices_view = win._views["slices"]
    assert slices_view._replot_btn is not None, "slices view missing _replot_btn"

    _out26 = os.path.join(_slice_tmp, "replots")
    _dlg26 = _SRD(_h5_path26, style=None, out_default=_out26)
    assert len(_dlg26._panel._rows) == 1
    _dlg26.select_all()
    _written26 = _dlg26.render_selection(_out26)
    assert len(_written26) == 1 and os.path.exists(_written26[0]), _written26
    print("[26] SliceReplotDialog: Replot… button wired; select_all + render_selection writes PNGs")

    # [27] Per-experiment form state saves on edit/flush and restores in a fresh StageView.
    from PySide6.QtCore import QSettings as _QSettings27

    from dfxm.config.models import Experiment as _Experiment27
    from gui.bindings import STAGE_SPECS as _SPECS27
    from gui.form_state import FormStateStore as _FSS27
    from gui.stage_view import StageView as _SV27

    _state_ini = os.path.join(tempfile.mkdtemp(), "form_state.ini")  # isolated from real settings
    _store27 = _FSS27(_QSettings27(_state_ini, _QSettings27.Format.IniFormat))
    _exp27 = _Experiment27(name="smoke_exp")
    _v27 = _SV27("strain", _SPECS27["strain"], _exp27, store=_store27)
    _v27._form.set_values({"root_folder": "/smoke/data"})
    _v27._form.set_values({"pixel_size_x_um": 0.111})  # calibration — must NOT persist
    _v27.flush()
    _v27b = _SV27("strain", _SPECS27["strain"], _exp27, store=_store27)
    assert _v27b._form.values()["root_folder"] == "/smoke/data", "form state not restored"
    assert "pixel_size_x_um" not in (_store27.load("smoke_exp", "strain") or {}), "calib leaked"
    print(
        "[27] per-experiment form state: save-on-flush restores in a fresh StageView (calib excluded)"
    )

    # [28] ROI picker button: schema-driven "Pick ROI…" button wired on roi-grouped stages.
    vview = win._views["visualize"]
    assert getattr(vview, "_roi_buttons", None), "visualize StageView missing Pick ROI… button"
    assert "crop" in vview._roi_buttons, "expected 'crop' roi_group button on visualize"
    sview28 = win._views["strain"]
    assert getattr(sview28, "_roi_buttons", None), "strain StageView missing Pick ROI… button"
    # Stages without roi_group params (e.g. concat, profiles) must have an empty dict.
    concat_view = win._views["concat"]
    assert getattr(concat_view, "_roi_buttons", None) == {}, (
        "concat StageView should have no ROI buttons"
    )
    print("[28] schema-driven Pick ROI… buttons present on roi-grouped stages (visualize, strain)")

    # [29] StyleControls: Scale (µm/cm) field parses defensively and mutates the style.
    from dfxm.common.plotting import PlotStyle as _PS29
    from gui.widgets.export_dialog import StyleControls as _SC29

    _st29 = _PS29()
    _sc29 = _SC29(_st29)
    _sc29._w_scale_umcm.setText("50")
    assert _st29.scale_um_per_cm == 50.0
    _sc29._w_scale_umcm.setText("junk")
    assert _st29.scale_um_per_cm is None
    _sc29._w_scale_umcm.setText("-2")
    assert _st29.scale_um_per_cm is None
    _sc29._w_scale_umcm.setText("")
    assert _st29.scale_um_per_cm is None
    print("[29] StyleControls Scale (µm/cm) field mutates the style defensively")

    # [30] Planes-first slices replot: filter narrows visibility; check-all-visible selects.
    from gui.widgets.slice_replot import SliceReplotDialog as _SRD30

    _dlg30 = _SRD30(_h5_path26, style=None, out_default=_out26)  # reuse [26]'s file
    _dlg30.show()
    app.processEvents()
    _dlg30._panel.set_all_checked(False)
    assert not _dlg30._panel.has_selection()
    _dlg30._panel._filter.setText("0")
    _dlg30._panel.check_all_visible()
    assert _dlg30._panel.has_selection()
    _dlg30._panel._filter.setText("999")
    assert _dlg30._panel._no_match.isVisible()
    print("[30] planes-first slices replot: filter + check-all-visible + no-match hint")

    # [31] Planes-first generic replot dialog (strain/mosaicity/rocking): product
    # selection across checked layers x checked quantity groups + filter hint.
    from gui.widgets.replot_dialog import ReplotDialog as _RD31

    class _G31:
        def __init__(self, key, labels):
            self.key, self.label, self.item_labels, self.shape = key, key, labels, None

    _calls31: list = []

    def _catalog_fn31(_path):
        return [_G31("sum_intensity", ["layer 0", "layer 1"]), _G31("specific_frame", ["layer 0"])]

    def _render_fn31(h5, selections, st, clim, roi, out):
        _calls31.append(selections)
        return ["x.png"]

    _h5_31 = os.path.join(tempfile.mkdtemp(), "a.h5")
    with open(_h5_31, "wb"):
        pass
    _dlg31 = _RD31(_h5_31, _catalog_fn31, _render_fn31, out_default=tempfile.mkdtemp())
    _dlg31.show()
    app.processEvents()
    _dlg31.select_all()
    _dlg31.render_selection(_dlg31._out_edit.text())
    _sels31 = dict(_calls31[-1])
    assert _sels31["sum_intensity"] == [0, 1]
    assert _sels31["specific_frame"] == [0]  # layer 1 skipped for this product, no error
    _dlg31._panel._filter.setText("999")
    assert _dlg31._panel._no_match.isVisible()

    from tests.qt_helpers import wait_batch_idle

    _dlg31._on_render()
    assert _dlg31._batch.running and _dlg31._batch._overlay.active, "replot overlay missing"
    wait_batch_idle(_dlg31)
    assert not _dlg31._batch._overlay.active
    assert "wrote" in _dlg31._status.text()
    print("[31] generic replot dialog planes-first: product selection + filter")

    # [32] Pin planes… dialog: reuses [26]'s file, checks one plane, emits a
    # pinned spec; the button is wired on the slices stage view.
    import json as _json32

    from PySide6.QtCore import Qt as _Qt32

    from gui.widgets.pin_planes import PinPlanesDialog as _PPD32

    assert win._views["slices"]._pin_btn is not None, "slices view missing _pin_btn"
    _dlg32 = _PPD32(_h5_path26)
    _dlg32._panel._items[("oblique", 0)].setCheckState(0, _Qt32.CheckState.Checked)
    _dlg32._on_ok()
    assert _json32.loads(_dlg32.result_json)[0]["sweep_start_um"] == 0.0
    print("[32] Pin planes… dialog emits pinned specs; button wired on slices view")

    # [33] ProfilesReplotDialog: opens from the profiles view, renders checked jobs.
    import h5py as _h5py33
    import numpy as _np33

    from gui.widgets.profiles_replot import ProfilesReplotDialog as _PRD33

    profiles_view = win._views["profiles"]
    assert profiles_view._replot_btn is not None, "profiles view missing _replot_btn"

    _tmp33 = tempfile.mkdtemp()
    _h5_33 = os.path.join(_tmp33, "oblique_slices.h5")
    _u33 = _np33.linspace(-10.0, 10.0, 81)
    _v33 = _np33.linspace(-8.0, 8.0, 65)
    _uu33, _vv33 = _np33.meshgrid(_u33, _v33)
    _offsets33 = _np33.array([-1.0, 0.0, 1.0])
    with _h5py33.File(_h5_33, "w") as _f33:
        for _vid33, _kind33, _cmap33 in (
            ("raw_sum", "raw_sum", "gray"),
            ("strain", "strain", "RdBu_r"),
        ):
            _g33 = _f33.create_group(_vid33)
            _g33.attrs["kind"] = _kind33
            _g33.attrs["cbar_label"] = "value"
            _g33.attrs["cmap"] = _cmap33
            _g33.attrs["title"] = _vid33
            _g33.attrs["vmin"] = -10.0
            _g33.attrs["vmax"] = 10.0
            _sg33 = _g33.create_group("oblique_full")
            _stack33 = _np33.stack(
                [0.7 * _uu33 - 1.3 * _vv33 + _o for _o in _offsets33], axis=0
            ).astype(_np33.float32)
            _sg33.create_dataset("slices", data=_stack33)
            _sg33.create_dataset("u_um", data=_u33)
            _sg33.create_dataset("v_um", data=_v33)
            _sg33.create_dataset("offsets_um", data=_offsets33)

    _jobs33 = [
        {
            "name": "oblique_full",
            "offset_um": 0.0,
            "start_uv": [-5, -3],
            "end_uv": [5, 3],
            "n_samples": 20,
            "width_pixels": 1,
            "fig_name": "smoke33",
        }
    ]
    _dlg33 = _PRD33(_h5_33, _jobs33, style=None, out_default="")
    assert _dlg33._tree.topLevelItemCount() == 1
    assert _dlg33._render_btn.isEnabled()  # opens with fields checked
    # F2: no "fields" key on the original job + all children checked -> pass
    # the job through unchanged so run-default [ref] + sorted(others) ordering
    # applies, instead of pinning tree order via an added "fields" key.
    _checked33 = _dlg33._checked_jobs()
    assert len(_checked33) == 1 and "fields" not in _checked33[0]
    _out33 = os.path.join(_tmp33, "replots")
    _written33 = _dlg33.render_selection(_out33)
    assert _written33 and all(os.path.exists(p) for p in _written33)
    assert not any(p.endswith(".csv") for p in os.listdir(_out33))
    print("[33] ProfilesReplotDialog: Replot… button wired; tree + render writes PNGs, no CSVs")

    # [34] Experiment editor ROI: derived read-out translates map -> detector px
    # and validation catches an inverted analysis pair.
    from dfxm.config.models import Experiment as _Exp34
    from gui.experiment_panel import ExperimentDialog as _ED34

    _dlg34 = _ED34(
        _Exp34(darfix_roi="105,230,1832,1266", analysis_roi_x="0,1832", analysis_roi_y="400,1100")
    )
    assert "y 630→1330" in _dlg34._roi_note.text()
    _dlg34._form.set_values({"analysis_roi_y": "0,700"})
    assert "y 230→930" in _dlg34._roi_note.text()  # read-out is live
    assert not _dlg34._roi_problems()
    _dlg34._form.set_values({"analysis_roi_y": "1100,400"})
    assert _dlg34._roi_problems()  # end <= start is unsaveable
    print("[34] experiment editor ROI: derived read-out live + validation blocks bad pairs")

    # [35] Initialize from data: detect on a synthetic raw tree -> review dialog
    # pre-checks blank fields -> apply lands in the experiment form.
    import h5py as _h535

    _raw35 = os.path.join(tempfile.mkdtemp(prefix="smoke_detect_"), "RAW")
    for _fam35, _n35 in (("s1_strain", 2), ("s1_mosa", 1), ("s1_rocking", 1)):
        for _i35 in range(_n35):
            os.makedirs(os.path.join(_raw35, f"{_fam35}__{_i35}"))
    with _h535.File(os.path.join(_raw35, "s1_strain__0", "s1_strain__0.h5"), "w") as _f35:
        _pos35 = _f35.create_group("1.1/instrument/positioners")
        for _k35, _v35 in dict(
            mainx=-5000.0, obx=273.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0, ccmth=7.1
        ).items():
            _pos35.create_dataset(_k35, data=_v35)
    from gui.widgets.detect_review import DetectReviewDialog as _DRD35

    _dlg35 = _ED34(_Exp34(raw_root=_raw35))
    _rows35 = _dlg35._detect(_dlg35._form.values())
    assert {d.field: d.value for d in _rows35 if d.value}["folder_pattern"] == "s1_strain__*"
    _rev35 = _DRD35(_rows35, current=_dlg35._form.values(), defaults=_Exp34().to_dict())
    _applied35 = _rev35.applied_values()  # pre-checked = blank/default fields
    assert _applied35["folder_pattern"] == "s1_strain__*"
    assert _applied35["mosa_pattern"] == "s1_mosa__*"
    assert 0 < _applied35["pixel_size_x_um"] < _applied35["pixel_size_y_um"]
    assert "darfix_roi" not in _applied35  # skip row pre-darfix — never auto-applied
    _dlg35._form.set_values(_applied35)
    assert _dlg35.experiment().folder_pattern == "s1_strain__*"
    print("[35] initialize-from-data: detectors → review pre-checks → applied into the form")

    # [36] Axes mode: StyleControls dropdown mutates axes_mode; sync restores; JSON round-trips
    from dfxm.common.plotting import PlotStyle as _PS36
    from dfxm.common.plotting import style_from_json as _sfj36
    from dfxm.common.plotting import style_to_json as _stj36

    s36 = _PS36()
    sc36 = _StyleControls(s36)
    idx36 = sc36._w_axes_mode.findData("none")
    assert idx36 >= 0, "Axes combo missing the 'none' entry"
    sc36._w_axes_mode.setCurrentIndex(idx36)
    assert s36.axes_mode == "none", "Axes combo did not mutate style.axes_mode"
    assert _sfj36(_stj36(s36)).axes_mode == "none", "axes_mode lost in JSON round-trip"
    s36.axes_mode = "no_frame"
    sc36.sync_from_style()
    assert sc36._w_axes_mode.currentData() == "no_frame", "sync_from_style did not restore combo"
    print("[36] axes-mode: dropdown mutates style + sync restores + JSON round-trip")

    # [37] figure builder: open from the main window, build a 1-panel recipe from a
    # synthetic slices h5, preview renders, export writes a PNG without tight-crop.
    import h5py as _h5b
    import numpy as _npb

    from dfxm.compose.recipe import PanelDef as _PD
    from dfxm.compose.recipe import PanelSource as _PS

    _bdir = tempfile.mkdtemp()
    _bh5 = os.path.join(_bdir, "obl.h5")
    with _h5b.File(_bh5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=_npb.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=_npb.linspace(0.0, 2.0, 5))
        sg.create_dataset("v_um", data=_npb.linspace(0.0, 1.5, 4))
        sg.create_dataset("offsets_um", data=_npb.array([0.0]))
    win._on_figure_builder()
    fb = win._figure_builder
    assert fb.isVisible()
    fb._style.scale_um_per_cm = 10.0
    fb._sync_style_to_recipe()
    fb.add_panels(
        [
            _PD(
                "s0",
                _PS(_bh5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
            )
        ]
    )
    from tests.qt_helpers import wait_builder_idle

    fb.render_now()
    assert fb._overlay.active, "busy overlay should cover the preview during a render"
    wait_builder_idle(fb)
    assert not fb._overlay.active
    res = fb._last_outcome
    assert res is not None and res.n_rendered == 1, fb._notes_label.text()
    _bout = os.path.join(_bdir, "export")
    import gui.figure_builder as _fbmod

    _orig_dir = _fbmod.QFileDialog.getExistingDirectory
    _fbmod.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: _bout)
    try:
        fb.export_now()
        wait_builder_idle(fb)
    finally:
        _fbmod.QFileDialog.getExistingDirectory = _orig_dir
    assert os.path.exists(os.path.join(_bout, "untitled.png"))
    _rp = os.path.join(_bdir, "r.json")
    fb.save_recipe_file(_rp)
    fb.load_recipe_file(_rp)
    assert not fb.is_dirty()
    print("[37] figure builder: open, preview, export, recipe save/load round-trip")

    # [38] Mark planes… dialog: reuses [26]'s file, marks one plane, Save
    # persists into /marks; the button is wired on the slices stage view.
    from dfxm.stages import slices as _sl38
    from gui.widgets.mark_planes import MarkPlanesDialog as _MPD38

    assert win._views["slices"]._mark_btn is not None, "slices view missing _mark_btn"
    _dlg38 = _MPD38(_h5_path26)  # reuse [26]'s file
    _dlg38._browser.set_plane(0)
    _dlg38._mark_btn.setChecked(True)
    _dlg38._on_save()
    assert _dlg38.saved
    _marks38 = _sl38.read_marks(_h5_path26)
    assert _dlg38._browser.slice_name in _marks38 and _marks38[_dlg38._browser.slice_name]
    print("[38] Mark planes… dialog saves /marks; button wired on slices view")

    # [39] Jobs from marks: button wired on the profiles stage view; the
    # checklist dialog sorts by slice/offset and reports only checked rows.
    from PySide6.QtCore import Qt as _Qt39

    from gui.widgets.jobs_from_marks import JobsFromMarksDialog as _JFMD39

    assert win._views["profiles"]._jobs_marks_btn is not None, (
        "profiles view missing _jobs_marks_btn"
    )
    _dlg39 = _JFMD39({"s": [0.0, 2.0]})
    _dlg39._list.item(1).setCheckState(_Qt39.CheckState.Unchecked)  # drop the 2.0 µm row
    _dlg39._on_ok()
    assert _dlg39.selected == [("s", 0.0)]
    print("[39] Jobs from marks: button wired on profiles view; checklist selection")

    # [40] figure builder interactive: arranger grid, Arrange… apply, united
    # colorbar mode, two-step Add-panels dialog.
    from gui.widgets.layout_arranger import ArrangeDialog, LayoutArranger
    from gui.widgets.panel_picker import AddPanelDialog as _APD40

    _la40 = LayoutArranger()
    _la40.set_grid([["s0"]], {"s0": {"title": "strain slice", "group": "strain"}})
    assert _la40.grid() == [["s0"]]
    fb.recipe().compose.colorbar_mode = "united"
    fb._load_compose_into_widgets()
    assert fb._compose_cbar_mode.currentData() == "united"
    _adlg40 = ArrangeDialog(fb.recipe(), fb._style)
    assert _adlg40._arranger.grid() == [["s0"]]
    _adlg40._on_apply()
    fb.apply_arranged_layout(_adlg40.result_layout)
    fb.render_now()
    wait_builder_idle(fb)
    _res40 = fb._last_outcome
    assert _res40 is not None and _res40.n_rendered == 1, fb._notes_label.text()
    _pdlg40 = _APD40({"slices": {"h5": _bh5, "sx": 0.5, "sy": 0.5, "jobs": []}})
    _pdlg40._stage.setCurrentText("slices")
    _pdlg40._reload()
    _pdlg40._check_all()
    _pdlg40._on_next()
    assert _pdlg40._stack.currentIndex() == 1
    _pdlg40.accept()
    assert _pdlg40.selected_panels and _pdlg40.selected_layout is not None
    assert _pdlg40.selected_panels[0].title  # picker captured a data name
    # trace-autoscale toggle: recipe -> widget -> recipe, and a re-render each way
    fb.recipe().compose.trace_autoscale = True
    fb._load_compose_into_widgets()
    assert fb._compose_trace_autoscale.isChecked()
    fb.render_now()
    wait_builder_idle(fb)
    _res40b = fb._last_outcome
    assert _res40b is not None and _res40b.n_rendered == 1, fb._notes_label.text()
    fb._compose_trace_autoscale.setChecked(False)  # widget -> recipe via _on_compose_edited
    assert fb.recipe().compose.trace_autoscale is False
    fb.render_now()
    wait_builder_idle(fb)
    _res40c = fb._last_outcome
    assert _res40c is not None and _res40c.n_rendered == 1, fb._notes_label.text()
    # fb is never closed in this script (unlike the pytest _no_leaked_debounce_timers
    # fixture, which stops every live window's debounce on teardown) — the last
    # setChecked() above re-armed the 300 ms debounce via schedule_preview(); left
    # running, it can fire deep into a later step's processEvents() call and start
    # an async worker with nothing left to await it, racing a QThread against
    # process exit ("QThread: Destroyed while thread '' is still running" -> abort).
    fb._debounce.stop()
    print(
        "[40] figure builder: arranger + Arrange… + united mode + two-step Add panels"
        " + trace-autoscale toggle"
    )

    # [41] 3-D viewer: launcher on the visualize view opens a Viewer3DWindow;
    # controls mutate the scene; close prunes the window list (offscreen: the
    # GL canvas degrades to its placeholder and export buttons disable).
    from gui.viewers import LoadedVolume as _LV41
    from gui.viewers import VolumeSourceSpec as _VSS41

    _lv41 = _LV41(np.ones((2, 3, 4)), (0.15, 0.38, 2.0), "magma", (0.5, 1.0), "I", "raw")
    _spec41 = _VSS41(
        "smoke_vol", lambda: _lv41, {"kind": "h5_dataset", "path": "/x", "dataset": "d"}
    )
    _panel41 = win._views["visualize"]._vol3d
    _panel41.set_sources({"smoke_vol": _spec41}, "visualize")
    _panel41._open_btn.click()
    assert len(_panel41._windows) == 1
    _win41 = _panel41._windows[0]
    _win41._mode_combo.setCurrentText("surface")
    assert _win41.scene.mode == "surface"
    _win41.close()
    app.processEvents()
    assert len(_panel41._windows) == 0
    print("[41] 3-D viewer: launcher opens window, controls live, close frees")

    print("\nGUI SMOKE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
