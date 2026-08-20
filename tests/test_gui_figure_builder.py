"""Figure-builder window + panel picker (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.plotting import PlotStyle  # noqa: E402
from dfxm.compose.recipe import PanelDef, PanelRef, PanelSource, Row  # noqa: E402
from gui.figure_builder import FigureBuilderWindow  # noqa: E402

# Every FigureBuilderWindow built by a test lives only for that test, but
# add_panels()/etc. arm its 300 ms render debounce (schedule_preview) — a
# QTimer parented to the window, so it outlives the test unless stopped. Left
# alone, that timer is still pending when the file's tests are done; the next
# test file's QApplication.processEvents() call fires it, running render_now()
# against whatever (possibly stale/mid-edit, e.g. a deliberately degenerate
# ROI from a "no mutation" test) recipe state the window was left in — the
# actual mechanism behind a warning that used to show up attributed to an
# unrelated test in tests/test_gui_replot_dialog.py. _track()/the autouse
# fixture below stop the timer and schedule the window's deletion at the end
# of every test, regardless of which helper created it.
_live_windows: list[FigureBuilderWindow] = []


def _track(w: FigureBuilderWindow) -> FigureBuilderWindow:
    _live_windows.append(w)
    return w


def _win():
    return _track(FigureBuilderWindow(lambda: {}, PlotStyle(scale_um_per_cm=10.0)))


@pytest.fixture(autouse=True)
def _no_leaked_debounce_timers():
    yield
    while _live_windows:
        w = _live_windows.pop()
        w._debounce.stop()  # never let a pending render fire into a later test
        w._pending_render = False
        w._pending_export = None
        if w._worker is not None:
            w._worker.wait(30000)  # tests may join; production code must not
        _app.processEvents()  # deliver the worker's queued result/finished
        w.deleteLater()
    _app.processEvents()  # let deleteLater actually run before the next test


def _panel(pid):
    return PanelDef(pid, PanelSource("/x.h5", "map_layer", {"stage": "strain", "z": 0}))


def test_add_panels_and_outline_roundtrip():
    w = _win()
    w.add_panels([_panel("a"), _panel("b")])
    r = w.recipe()
    assert [p.id for p in r.panels] == ["a", "b"]
    assert w.is_dirty()


def test_move_and_delete_edit_the_recipe():
    w = _win()
    w.add_panels([_panel("a"), _panel("b")])
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(1))  # select "b"
    w.move_selected(-1)
    assert [getattr(x, "panel_id", None) for x in _row(w).children] == ["b", "a"]
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    w.delete_selected()
    assert [getattr(x, "panel_id", None) for x in _row(w).children] == ["a"]


def _row(w):
    layout = w.recipe().layout
    assert isinstance(layout, Row)
    return layout


def test_delete_row_purges_nested_panel_defs_and_gutter_renders(tmp_path):
    """Deleting a Row/Col removes its PanelRef leaves from the layout, but
    used to leave their backing PanelDefs in recipe.panels — orphaned defs
    that crash a subsequent gutter-mode render (render_recipe loads data for
    ALL recipe.panels, but the post-placement pid loops only cover layout
    leaves). delete_selected must purge any PanelDef no longer referenced by
    the layout."""
    import h5py
    import numpy as np

    from dfxm.compose.recipe import Col

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0.0, 2.0, 5))
        sg.create_dataset("v_um", data=np.linspace(0.0, 1.5, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0]))

    def _slice_panel(pid):
        return PanelDef(
            pid,
            PanelSource(
                str(h5), "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}
            ),
        )

    w = _win()
    w.add_panels([_slice_panel("keep")])
    # Nest a second panel inside a Col so deleting the Col orphans it.
    root = w.recipe().layout
    doomed_col = Col([])
    root.children.append(doomed_col)
    w.recipe().panels.append(_slice_panel("doomed"))
    doomed_col.children.append(PanelRef("doomed"))
    w._rebuild_tree()

    # select the Col node (second top-level child) and delete it
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(1))
    w.delete_selected()

    assert [p.id for p in w.recipe().panels] == ["keep"]

    w.recipe().compose.scale_bar_mode = "gutter"
    res = render_and_wait(w)
    assert res is not None and res.n_panels == 1 and res.n_rendered == 1


def test_save_open_round_trip_and_dirty_state(tmp_path):
    w = _win()
    w.add_panels([_panel("a")])
    w.recipe().name = ""  # blank name — load must fall back to the file stem
    path = str(tmp_path / "r.json")
    w.save_recipe_file(path)
    assert not w.is_dirty()
    w2 = _win()
    w2.load_recipe_file(path)
    assert [p.id for p in w2.recipe().panels] == ["a"]
    assert not w2.is_dirty()
    assert w2.recipe().name == "r"


def test_set_selected_label_on_spacer_is_a_noop():
    w = _win()
    w.add_spacer()
    w._dirty = False  # isolate: only interested in set_selected_label's own effect
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    w.set_selected_label("x")
    assert not w.is_dirty()


def test_panel_picker_builds_slice_panel_defs(tmp_path):
    import h5py
    import numpy as np

    from gui.widgets.panel_picker import AddPanelDialog

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((2, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0, 1, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 1, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))
    dlg = AddPanelDialog({"slices": {"h5": str(h5), "sx": 0.5, "sy": 0.5, "jobs": []}})
    dlg._stage.setCurrentText("slices")
    dlg._reload()
    dlg._check_all()  # helper that checks every leaf, mirrors replot select_all
    panels = dlg._build_panels()
    assert len(panels) == 2
    assert panels[0].source.kind == "slice_plane"
    assert panels[0].source.selector["volume_id"] == "strain"


def test_panel_picker_pins_loaded_h5_path_not_live_field_text(tmp_path):
    import h5py
    import numpy as np

    from gui.widgets.panel_picker import AddPanelDialog

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0, 1, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 1, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0]))
    dlg = AddPanelDialog({"slices": {"h5": str(h5), "sx": 0.5, "sy": 0.5, "jobs": []}})
    dlg._stage.setCurrentText("slices")
    dlg._reload()
    dlg._check_all()
    dlg._h5_edit.setText("/typo/does/not/match.h5")  # edited after Load, before OK
    panels = dlg._build_panels()
    assert panels
    assert panels[0].source.h5_path == str(h5)


# -- preview + cache ----------------------------------------------------------
def _obl_recipe_panels(tmp_path):
    import h5py
    import numpy as np

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0.0, 2.0, 5))
        sg.create_dataset("v_um", data=np.linspace(0.0, 1.5, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0]))
    from dfxm.compose.recipe import PanelDef, PanelSource

    return [
        PanelDef(
            "a",
            PanelSource(
                str(h5), "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}
            ),
        )
    ]


def test_render_now_populates_preview_and_notes(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    res = render_and_wait(w)
    assert res is not None and res.n_rendered == 1
    assert w._canvas is not None  # a live FigureCanvasQTAgg wrapping res.figure
    assert w._canvas.figure is res.figure


def test_render_error_lands_in_notes_bar_not_crash(tmp_path):
    w = _track(FigureBuilderWindow(lambda: {}, PlotStyle()))  # NO scale anywhere
    w.add_panels(_obl_recipe_panels(tmp_path))
    res = render_and_wait(w)
    assert res is None
    assert "scale" in w._notes_label.text().lower()
    assert "hint" in w._notes_label.text().lower() or "Set Scale" in w._notes_label.text()


def test_cache_survives_file_deletion_until_refresh(tmp_path):
    w = _win()
    panels = _obl_recipe_panels(tmp_path)
    w.add_panels(panels)
    assert render_and_wait(w) is not None
    os.remove(panels[0].source.h5_path)
    res2 = render_and_wait(w)  # served from cache
    assert res2 is not None and res2.n_rendered == 1
    w.refresh_data()  # cache cleared -> placeholder now
    res3 = render_and_wait(w)
    assert res3 is not None and res3.n_rendered == 0
    assert "placeholder" in w._notes_label.text()


def test_click_preview_selects_outline_node(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    res = render_and_wait(w)
    ax = res.axes_by_id["a"]
    w._on_preview_pick(ax)  # the slot the mpl button_press handler calls
    item = w._tree.currentItem()
    assert item is not None and "a" in item.text(0)


# -- style pane + overrides + export ------------------------------------------
def test_style_controls_edit_lands_in_recipe_style(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w._style.font_scale = 2.0
    w._sync_style_to_recipe()
    assert w.recipe().style["font_scale"] == 2.0
    assert w.is_dirty()


def test_compose_knob_edits_land_in_recipe(tmp_path):
    w = _win()
    w._compose_template.setText("(A)")
    w._on_compose_edited()
    assert w.recipe().compose.label_template == "(A)"


def test_panel_override_editor_applies(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    panel = w.recipe().panels[0]
    w._apply_panel_overrides(
        panel,
        {
            "roi": "0,3,1,4",
            "clim": "-2,",
            "cmap": "viridis",
            "label": "Z1",
            "show_title": None,
            "scale_um_per_cm": 4.0,
            "colorbar": False,
        },
    )
    assert panel.roi == (0, 3, 1, 4)
    assert panel.clim == (-2.0, None)
    assert panel.cmap == "viridis" and panel.label == "Z1"
    assert panel.scale_um_per_cm == 4.0 and panel.colorbar is False


def test_panel_override_editor_malformed_roi_no_mutation(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    panel = w.recipe().panels[0]
    panel.roi = (1, 2, 3, 4)
    w._apply_panel_overrides(
        panel,
        {
            "roi": "not,a,roi",
            "clim": "",
            "cmap": "",
            "label": "",
            "show_title": None,
            "scale_um_per_cm": None,
            "colorbar": None,
        },
    )
    assert panel.roi == (1, 2, 3, 4)  # unchanged
    assert "roi" in w._notes_label.text().lower()


def test_panel_override_editor_malformed_clim_no_mutation(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    panel = w.recipe().panels[0]
    panel.clim = (1.0, 2.0)
    w._apply_panel_overrides(
        panel,
        {
            "roi": "",
            "clim": "nope",
            "cmap": "",
            "label": "",
            "show_title": None,
            "scale_um_per_cm": None,
            "colorbar": None,
        },
    )
    assert panel.clim == (1.0, 2.0)  # unchanged
    assert "clim" in w._notes_label.text().lower()


def test_export_now_writes_files(tmp_path, monkeypatch):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "gui.figure_builder.QFileDialog.getExistingDirectory", lambda *a, **k: str(out)
    )
    export_and_wait(w)
    assert os.path.exists(out / "untitled.png")
    assert "wrote" in w._notes_label.text()


def test_export_now_zero_files_written_still_reports_chosen_dir(tmp_path, monkeypatch):
    """All export formats unchecked -> export_recipe returns paths=[] — the
    notes bar must still name the directory the user chose (review finding on
    e4386ed: the async _on_worker_result briefly derived the reported dir
    from os.path.dirname(paths[0]), which is empty when paths == [] — parity
    with the old synchronous export_now, which always printed the chosen
    `out` directly, never one derived from paths[0])."""
    from matplotlib.figure import Figure

    from dfxm.compose.render import ComposeResult

    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "gui.figure_builder.QFileDialog.getExistingDirectory", lambda *a, **k: str(out)
    )
    monkeypatch.setattr(
        "dfxm.compose.render.export_recipe",
        lambda *a, **k: ([], ComposeResult(figure=Figure())),
    )
    export_and_wait(w)
    assert w._notes_label.text() == f"wrote 0 file(s) → {out}"


# -- fix wave 1: partial-submit override editor + export never crashes -------
def test_apply_panel_overrides_unrelated_field_preserves_suppressed_label(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    panel = w.recipe().panels[0]
    panel.label = ""  # explicit "no label" (distinct from None/auto) — set via
    # the outline Label… dialog in real use (set_selected_label supports it)
    w._apply_panel_overrides(panel, {"roi": "0,3,1,4"})  # only the ROI key submitted
    assert panel.roi == (0, 3, 1, 4)
    assert panel.label == ""  # must NOT clobber to None (auto-lettering)


def test_override_widget_edit_preserves_suppressed_label_and_precise_clim(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    panel = w.recipe().panels[0]
    panel.label = ""
    panel.clim = (-0.123456789, 0.987654321)  # more than 6 sig figs (%g would round)
    w._select_outline_panel(panel.id)  # loads the override editor for this panel
    w._ov_roi.setText("0,3,1,4")  # the real widget signal path, editing only ROI
    assert panel.roi == (0, 3, 1, 4)
    assert panel.label == ""  # unrelated field edit must not touch label
    assert panel.clim == (-0.123456789, 0.987654321)  # ...or silently round clim


def test_override_crop_to_data_checkbox_writes_field_and_reloads(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    panel = w.recipe().panels[0]
    assert panel.crop_to_data is False
    w._select_outline_panel(panel.id)
    assert not w._ov_crop.isChecked()
    w._ov_crop.setChecked(True)  # real widget signal path
    assert panel.crop_to_data is True
    assert w.is_dirty()
    w._ov_crop.setChecked(False)
    assert panel.crop_to_data is False
    # reload path from the panel's stored value
    panel.crop_to_data = True
    w._load_panel_page(panel)
    assert w._ov_crop.isChecked()


def test_export_now_unexpected_error_reports_to_notes_bar_not_crash(tmp_path, monkeypatch):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "gui.figure_builder.QFileDialog.getExistingDirectory", lambda *a, **k: str(out)
    )

    def _raise(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr("dfxm.compose.render.export_recipe", _raise)
    export_and_wait(w)  # must not raise
    assert "export failed" in w._notes_label.text()


# -- lifecycle: closing must not leave a render debounce ticking -------------
def test_close_stops_pending_debounce_timer(tmp_path):
    """A schedule_preview() call arms a 300 ms singleShot QTimer; closing the
    window (the not-dirty path — no Save/Discard prompt) must stop it, or the
    timer fires render_now() on a closed/hidden window from a later event-loop
    turn (this is the same leak class the tests worked around via _track/
    _no_leaked_debounce_timers — closeEvent should not have the same wart in
    the real app)."""
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))  # arms the debounce via schedule_preview
    w._dirty = False  # isolate: only interested in the not-dirty close path here
    assert w._debounce.isActive()
    w.close()
    assert not w._debounce.isActive()


def test_close_with_unsaved_changes_cancel_leaves_debounce_running(tmp_path, monkeypatch):
    """Cancelling the Save/Discard/Cancel prompt aborts the close, so the
    timer legitimately stays armed — closeEvent must only stop it on the
    paths that actually close the window."""
    from PySide6.QtWidgets import QMessageBox

    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    assert w.is_dirty()
    assert w._debounce.isActive()
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Cancel)
    w.close()
    assert w._debounce.isActive()  # close was aborted — nothing should change


# -- task 13: main-window launch wiring ---------------------------------------
def test_main_window_launches_builder_non_modal():
    from gui.main_window import MainWindow

    win = MainWindow()
    win._on_figure_builder()
    assert win._figure_builder is not None
    assert win._figure_builder.isVisible()
    first = win._figure_builder
    win._on_figure_builder()  # reuse, not a second window
    assert win._figure_builder is first


def test_builder_defaults_prefill_stage_output_h5_not_folder():
    """strain/mosaicity/rocking panel-picker defaults must be the stacked/
    aligned OUTPUT h5 the stage's catalog reads (mirroring how the stage's
    own run() derives it — see strain.py/mosaicity.py's
    ``os.path.join(default_out_root, stacked_filename)`` and rocking.py's
    ``os.path.join(out_dir, aligned_h5_name)``), never the bare input
    directory field (root_folder/raw_root) the picker can't load a catalog
    from. slices/profiles are untouched (already correct)."""
    from dataclasses import replace as _dc_replace

    from gui.main_window import MainWindow

    win = MainWindow()
    exp = _dc_replace(
        win._experiment_panel.current_experiment(),
        raw_root="/data/raw",
        processed_root="/data/processed",
    )
    win._experiment_panel._set_experiment(exp)

    defaults = win._builder_defaults()
    for stage in ("strain", "mosaicity", "rocking"):
        h5 = defaults[stage]["h5"]
        assert h5 == "" or h5.endswith(".h5"), f"{stage} default {h5!r} is not an h5 path"


# -- node inspector (T4) ------------------------------------------------------
def _select_child(w, i):
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(i))


def test_inspector_switches_page_per_node_type():
    w = _win()
    w.add_panels([_panel("a")])
    w._tree.setCurrentItem(None)
    w.add_row()
    w._tree.setCurrentItem(None)
    w.add_col()
    w._tree.setCurrentItem(None)
    w.add_spacer()
    w._tree.setCurrentItem(None)
    w.add_text()
    w._tree.setCurrentItem(None)
    assert w._inspector.currentWidget() is w._page_hint
    _select_child(w, 0)
    assert w._inspector.currentWidget() is w._page_panel
    _select_child(w, 1)
    assert w._inspector.currentWidget() is w._page_row
    _select_child(w, 2)
    assert w._inspector.currentWidget() is w._page_col
    _select_child(w, 3)
    assert w._inspector.currentWidget() is w._page_spacer
    _select_child(w, 4)
    assert w._inspector.currentWidget() is w._page_text


def test_row_page_pin_and_shared_controls_write_fields():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    w._row_pin_h.setValue(3.5)
    assert row.pinned_height_cm == 3.5
    w._row_pin_h.setValue(0.0)
    assert row.pinned_height_cm is None
    w._row_shared_cb.setChecked(True)
    assert row.shared_colorbar is True
    w._row_shared_clim.setText("-1,2")
    assert row.shared_clim == (-1.0, 2.0)
    w._row_shared_clim.setText("")
    assert row.shared_clim is None


def test_col_page_shared_x_and_pin_width():
    w = _win()
    w.add_col()
    _select_child(w, 0)
    col = w.recipe().layout.children[0]
    w._col_shared_x.setChecked(True)
    assert col.shared_x is True
    w._col_pin_w.setValue(6.0)
    assert col.pinned_width_cm == 6.0


def test_shared_clim_parse_failure_notes_no_mutation():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    row.shared_clim = (0.0, 1.0)
    w._dirty = False
    w._row_shared_clim.setText("nope")
    assert row.shared_clim == (0.0, 1.0)
    assert "shared clim" in w._notes_label.text()
    assert not w.is_dirty()


def test_spacer_and_text_pages_edit_boxes():
    w = _win()
    w.add_spacer()
    _select_child(w, 0)
    sp = w.recipe().layout.children[0]
    w._spacer_w.setValue(1.25)
    assert sp.w_cm == 1.25
    w._tree.setCurrentItem(None)
    w.add_text()
    _select_child(w, 1)
    tx = w.recipe().layout.children[1]
    w._text_edit.setText("Header")
    assert tx.text == "Header"
    w._text_h.setValue(0.75)
    assert tx.h_cm == 0.75


def test_panel_label_three_state_control():
    w = _win()
    w.add_panels([_panel("a")])
    _select_child(w, 0)
    panel = w.recipe().panels[0]
    assert panel.label is None
    w._ov_label_mode.setCurrentIndex(1)  # "No label"
    assert panel.label == ""
    w._ov_label_mode.setCurrentIndex(2)  # "Custom…"
    w._ov_label.setText("Z9")
    assert panel.label == "Z9"
    w._ov_label_mode.setCurrentIndex(0)  # back to auto
    assert panel.label is None


def test_group_mode_control_and_no_auto_sentinel_leak(monkeypatch):
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    w._row_group_mode.setCurrentIndex(1)  # Auto letter
    assert row.group_label == "auto"
    w._tree.setCurrentItem(None)
    w.add_spacer()  # a second node to bounce selection off
    _select_child(w, 1)
    _select_child(w, 0)  # reload the Row page
    assert w._row_group_label.text() == ""  # sentinel never shown
    assert w._row_group_mode.currentData() == "auto"
    captured = {}

    def fake_get_text(*_a, **kw):
        captured["prefill"] = kw.get("text", "")
        return ("", False)

    monkeypatch.setattr("gui.figure_builder.QInputDialog.getText", fake_get_text)
    w._on_label_selected()
    assert captured["prefill"] == ""  # the Label… dialog leak, fixed
    w._row_group_mode.setCurrentIndex(2)  # Custom…
    w._row_group_label.setText("M1")
    assert row.group_label == "M1"


def test_noop_inspector_edit_does_not_dirty():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    w._dirty = False
    w._apply_node_field(row, "shared_colorbar", False)  # already False
    assert not w.is_dirty()
    w._tree.setCurrentItem(None)
    w.add_panels([_panel("a")])
    panel = w.recipe().panels[0]
    w._dirty = False
    w._apply_panel_overrides(panel, {"cmap": ""})  # "" maps to None == current
    assert not w.is_dirty()


def test_inspector_edit_updates_item_text_in_place_without_rebuild():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    item_before = w._tree.currentItem()
    w._row_group_mode.setCurrentIndex(1)  # -> "[group]" marker
    assert w._tree.currentItem() is item_before  # same item object: no rebuild
    assert "[group]" in item_before.text(0)


# -- selection persistence + stale canvas (T5) --------------------------------
def test_move_selected_is_repeatable_selection_persists():
    w = _win()
    w.add_panels([_panel("a"), _panel("b"), _panel("c")])
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(2))  # select "c"
    w.move_selected(-1)
    assert [getattr(x, "panel_id", None) for x in _row(w).children] == ["a", "c", "b"]
    assert getattr(w._selected_node(), "panel_id", None) == "c"  # survived the rebuild
    w.move_selected(-1)  # pressing ↑ again must keep working
    assert [getattr(x, "panel_id", None) for x in _row(w).children] == ["c", "a", "b"]


def test_delete_selects_parent_container():
    w = _win()
    w.add_col()
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    w.add_panels([_panel("a"), _panel("b")])  # into the selected Col
    col = w.recipe().layout.children[0]
    w.select_node(col.children[0])
    w.delete_selected()
    assert w._selected_node() is col


def test_label_edit_keeps_selection():
    w = _win()
    w.add_col()
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    col = w.recipe().layout.children[0]
    w.set_selected_label("G1")
    assert w._selected_node() is col


def test_render_now_clears_canvas_when_no_panels_left(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    assert render_and_wait(w) is not None and w._canvas is not None
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    w.delete_selected()
    assert render_and_wait(w) is None
    assert w._canvas is None and w._result is None
    assert "add panels" in w._notes_label.text()


def test_outline_marks_suppressed_label():
    w = _win()
    w.add_panels([_panel("a")])
    w.recipe().panels[0].label = ""
    w._rebuild_tree()
    assert "label off" in w._tree.topLevelItem(0).child(0).text(0)


# -- figure-2 acceptance: authored entirely through window methods (T6) -------
def test_figure2_authored_through_window_methods(tmp_path):
    """Spec §C: acceptance figure 2 — ragged 3 columns, shared_x trace stacks
    with group labels, split map/trace scales — authored purely through
    FigureBuilderWindow methods (no JSON editing), rendering with exactly the
    geometry tests/test_compose_acceptance.py's figure-2 test pins."""
    import numpy as np

    from dfxm.common.plotting import measured_box_in
    from dfxm.compose.recipe import PanelSource
    from tests.test_compose_acceptance import _write_profiles_three_fields

    h5 = _write_profiles_three_fields(tmp_path / "obl.h5")
    job_a = {"name": "obl", "offset_um": 0.0, "start_uv": [-8.0, -6.0], "end_uv": [8.0, 6.0]}
    job_b = {"name": "obl", "offset_um": 0.0, "start_uv": [-15.0, -9.0], "end_uv": [15.0, 9.0]}
    len_a = float(np.hypot(16.0, 12.0))  # 20 µm
    len_b = float(np.hypot(30.0, 18.0))  # ~34.99 µm
    fields = ["mosa_com_chi", "strain", "raw_mosa_sum"]
    map_scale, trace_scale, trace_h_cm = 5.0, 2.0, 2.0

    w = _track(
        FigureBuilderWindow(
            lambda: {},
            PlotStyle(
                scale_um_per_cm=map_scale,
                trace_scale_um_per_cm=trace_scale,
                trace_height_cm=trace_h_cm,
                show_title=False,
            ),
        )
    )
    root = w.recipe().layout

    # column 1: the two reference maps, labelled via the inspector
    w.add_col()
    col1 = root.children[0]
    w.select_node(col1)
    w.add_panels(
        [
            PanelDef(
                "A1",
                PanelSource(h5, "profiles_ref", {"job": job_a, "field": "mosa_com_chi"}),
                roi=(10, 50, 10, 60),
            ),
            PanelDef(
                "B1",
                PanelSource(h5, "profiles_ref", {"job": job_b, "field": "mosa_com_chi"}),
                roi=(10, 50, 10, 60),
            ),
        ]
    )
    w.select_node(col1.children[0])
    w._ov_label_mode.setCurrentIndex(2)  # Custom… (the 3-state widget path)
    w._ov_label.setText("A1")
    w._apply_panel_overrides(w.recipe().panel_by_id()["B1"], {"label": "B1"})

    # columns 2/3: shared-x trace stacks with group labels A2/B2
    for tag, job, glabel in (("a", job_a, "A2"), ("b", job_b, "B2")):
        w.select_node(root)
        w.add_col()
        col = root.children[-1]
        w.select_node(col)
        w.add_panels(
            [
                PanelDef(
                    f"t_{tag}_{v}", PanelSource(h5, "profiles_trace", {"job": job, "field": v})
                )
                for v in fields
            ]
        )
        w.select_node(col)
        w._col_shared_x.setChecked(True)  # the NEW inspector authoring path
        if tag == "a":
            w._col_group_mode.setCurrentIndex(2)  # Custom… on the Col page
            w._col_group_label.setText(glabel)
        else:
            w.set_selected_label(glabel)  # the outline Label… path — both must work
        assert col.shared_x is True and col.group_label == glabel

    res = render_and_wait(w)
    assert res is not None and res.n_rendered == 8, w._notes_label.text()
    fig = res.figure

    # geometry pins — identical to test_acceptance_figure_2_ragged_dual_scale
    ext_x, ext_y = 24.5, 19.5
    for pid in ("A1", "B1"):
        bw, bh = measured_box_in(fig, res.axes_by_id[pid])
        assert abs(bw - ext_x / map_scale / 2.54) < 0.005 * bw, pid
        assert abs(bh - ext_y / map_scale / 2.54) < 0.005 * bh, pid
    for tag, length in (("a", len_a), ("b", len_b)):
        for vid in fields:
            bw, bh = measured_box_in(fig, res.axes_by_id[f"t_{tag}_{vid}"])
            assert abs(bw - length / trace_scale / 2.54) < 0.005 * bw, (tag, vid)
            assert abs(bh - trace_h_cm / 2.54) < 0.005 * bh, (tag, vid)

    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "A1" in texts and "B1" in texts and "A2" in texts and "B2" in texts
    for tag in ("a", "b"):
        for vid in fields[:-1]:
            assert res.axes_by_id[f"t_{tag}_{vid}"].get_xlabel() == ""
        assert res.axes_by_id[f"t_{tag}_{fields[-1]}"].get_xlabel() != ""
        xs = {round(res.axes_by_id[f"t_{tag}_{v}"].get_position().x0, 5) for v in fields}
        assert len(xs) == 1, tag
    xa = max(res.axes_by_id[f"t_a_{v}"].get_position().x1 for v in fields)
    xb = {round(res.axes_by_id[f"t_b_{v}"].get_position().x0, 5) for v in fields}
    assert len(xb) == 1 and min(xb) > xa
    assert not any("scale is off" in n for n in res.notes)


# -- Custom label mode is uncommitted until text lands (followup wave) --------
def test_custom_label_mode_uncommitted_until_text():
    w = _win()
    w.add_panels([_panel("a")])
    _select_child(w, 0)
    panel = w.recipe().panels[0]
    assert panel.label is None
    w._ov_label_mode.setCurrentIndex(2)  # Custom… with no text yet
    assert panel.label is None  # pre-fix: clobbered to "" (suppressed)
    w._ov_label.setText("Z1")
    assert panel.label == "Z1"
    w._ov_label.setText("")  # deleting all text is transient too
    assert panel.label == "Z1"


def test_custom_group_mode_uncommitted_until_text():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    w._row_group_mode.setCurrentIndex(1)  # Auto letter
    assert row.group_label == "auto"
    w._row_group_mode.setCurrentIndex(2)  # Custom… with no text yet
    assert row.group_label == "auto"  # pre-fix: clobbered to None (not a group)
    w._row_group_label.setText("M1")
    assert row.group_label == "M1"
    w._row_group_label.setText("")
    assert row.group_label == "M1"


def test_style_controls_cbar_typography_round_trip():
    from gui.widgets.export_dialog import StyleControls

    st = PlotStyle(cbar_label_scale=1.7, cbar_tick_scale=0.8, cbar_labelpad_pt=12.0)
    c = StyleControls(st)
    assert c._w_cbar_label_scale.value() == 1.7
    assert c._w_cbar_tick_scale.value() == 0.8
    assert c._w_cbar_labelpad.text() == "12"
    st2 = PlotStyle()
    c.set_style(st2)  # rebind + sync: defaults displayed, no leak from st
    assert c._w_cbar_label_scale.value() == 1.0
    assert c._w_cbar_labelpad.text() == ""
    c._w_cbar_labelpad.setText("7.5")
    assert st2.cbar_labelpad_pt == 7.5
    c._w_cbar_label_scale.setValue(2.0)
    assert st2.cbar_label_scale == 2.0
    c._w_cbar_labelpad.setText("")
    assert st2.cbar_labelpad_pt is None


# -- panel titles (picker capture + display sites) ----------------------------
def test_panel_picker_slice_leaves_carry_titles(tmp_path):
    import h5py
    import numpy as np
    from PySide6.QtCore import Qt

    from gui.widgets.panel_picker import AddPanelDialog

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0, 1, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 1, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0]))
    dlg = AddPanelDialog({"slices": {"h5": str(h5), "sx": 0.5, "sy": 0.5, "jobs": []}})
    dlg._stage.setCurrentText("slices")
    dlg._reload()
    leaf = dlg._tree.topLevelItem(0).child(0)
    data = leaf.data(0, Qt.ItemDataRole.UserRole)
    assert data["title"] == "strain/obl / plane 0  @ +0.00 µm"
    dlg._check_all()
    panels = dlg._build_panels()
    assert panels[0].title == "strain/obl / plane 0  @ +0.00 µm"


def test_panel_picker_map_titles_from_fake_catalog():
    from PySide6.QtCore import Qt

    from gui.widgets.panel_picker import AddPanelDialog

    class _Grp:
        key = "/chi/Center of mass"
        label = "Mosa χ COM"
        item_labels = ["z=0", "z=1"]

    dlg = AddPanelDialog({})
    dlg._stage.setCurrentText("mosaicity")
    dlg._tree.clear()
    dlg._build_map_tree("mosaicity", [_Grp()])
    leaf = dlg._tree.topLevelItem(0).child(1)
    assert leaf.data(0, Qt.ItemDataRole.UserRole)["title"] == "mosaicity: Mosa χ COM / z=1"


def test_outline_and_scale_bar_combo_show_titles_store_ids():
    w = _win()
    p = _panel("a")
    p.title = "strain: eps / z=0"
    w.add_panels([p])
    assert "strain: eps / z=0" in w._tree.topLevelItem(0).child(0).text(0)
    assert "Panel: a" not in w._tree.topLevelItem(0).child(0).text(0)
    combo = w._compose_scale_bar_panel
    idx = combo.findData("a")
    assert idx > 0 and combo.itemText(idx) == "strain: eps / z=0"
    combo.setCurrentIndex(idx)
    assert w.recipe().compose.scale_bar_panel == "a"  # data (id), not display text


def test_outline_title_fallback_is_id_and_label_off_survives():
    w = _win()
    w.add_panels([_panel("a")])  # no title
    w.recipe().panels[0].label = ""
    w._rebuild_tree()
    text = w._tree.topLevelItem(0).child(0).text(0)
    assert "Panel: a" in text and "label off" in text


# -- two-step Add panels dialog + layout fragments ----------------------------
def _slices_dialog(tmp_path):
    import h5py
    import numpy as np

    from gui.widgets.panel_picker import AddPanelDialog

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((2, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0, 1, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 1, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))
    dlg = AddPanelDialog({"slices": {"h5": str(h5), "sx": 0.5, "sy": 0.5, "jobs": []}})
    dlg._stage.setCurrentText("slices")
    dlg._reload()
    dlg._check_all()
    return dlg


def test_add_dialog_two_step_returns_fragment(tmp_path):
    from dfxm.compose.recipe import Col

    dlg = _slices_dialog(tmp_path)
    dlg._on_next()
    assert dlg._stack.currentIndex() == 1
    assert dlg._arranger.grid() == [[p.id] for p in dlg._staged]  # one-row seed
    item = dlg._arranger._columns[1].list.takeItem(0)  # stack plane 1 under plane 0
    dlg._arranger._columns[0].list.addItem(item)
    dlg.accept()
    assert len(dlg.selected_panels) == 2
    lay = dlg.selected_layout
    assert isinstance(lay, Row) and isinstance(lay.children[0], Col)
    assert [r.panel_id for r in lay.children[0].children] == [p.id for p in dlg.selected_panels]


def test_add_dialog_ok_from_step1_keeps_flat_behaviour(tmp_path):
    dlg = _slices_dialog(tmp_path)
    dlg.accept()
    assert len(dlg.selected_panels) == 2
    assert dlg.selected_layout is None


def test_add_dialog_next_with_nothing_checked_stays_on_step1(tmp_path):
    dlg = _slices_dialog(tmp_path)
    dlg._uncheck_all()
    dlg._on_next()
    assert dlg._stack.currentIndex() == 0
    assert "check" in dlg._status.text()


def test_add_dialog_stale_pick_cleared_on_back_then_ok_from_page0(tmp_path):
    # Next -> corner-click -> Back -> OK-from-page-0 must not carry a pick
    # whose panel id no longer exists once page 0 is left/restaged.
    dlg = _slices_dialog(tmp_path)
    dlg._on_next()
    pid = dlg._staged[0].id
    dlg._on_scale_bar_picked(pid, "upper left")
    assert dlg.scale_bar_pick == (pid, "upper left")
    dlg._goto_page(0)
    dlg.accept()
    assert dlg.scale_bar_pick is None


def test_add_dialog_stale_pick_cleared_by_restage_on_next(tmp_path):
    # Next -> pick -> Back -> Next: the restage must clear the earlier pick
    # even though the dialog never left page 1 via OK.
    dlg = _slices_dialog(tmp_path)
    dlg._on_next()
    pid = dlg._staged[0].id
    dlg._on_scale_bar_picked(pid, "upper left")
    assert dlg.scale_bar_pick == (pid, "upper left")
    dlg._goto_page(0)
    dlg._on_next()
    assert dlg.scale_bar_pick is None


def test_window_add_panels_with_fragment_and_id_collision():
    from dfxm.compose.recipe import Col

    w = _win()
    w.add_panels([_panel("slices_0")])
    frag = Row([Col([PanelRef("slices_0"), PanelRef("slices_1")])])
    renames = w.add_panels([_panel("slices_0"), _panel("slices_1")], layout=frag)
    assert renames == {"slices_0": "slices_0_1"}
    root = w.recipe().layout
    assert root.children[1] is frag  # fragment appended as ONE child
    assert [r.panel_id for r in frag.children[0].children] == ["slices_0_1", "slices_1"]
    assert [p.id for p in w.recipe().panels] == ["slices_0", "slices_0_1", "slices_1"]


def test_apply_scale_bar_pick_translated_through_add_panels_rename():
    # Exercises _on_add_panels's pick-translation logic without exec(): a
    # dialog result whose selected_panels collides with an existing id must
    # have its scale_bar_pick's panel id rewritten through the SAME rename
    # add_panels() applied to the panels themselves — _apply_scale_bar_pick is
    # the exact method _on_add_panels calls, so this tests the shipped code.
    w = _win()
    w.add_panels([_panel("slices_0")])
    renames = w.add_panels([_panel("slices_0")])  # collides -> "slices_0_1"
    assert renames == {"slices_0": "slices_0_1"}
    w._apply_scale_bar_pick(("slices_0", "upper left"), renames)
    assert w.recipe().compose.scale_bar_mode == "one-panel"
    assert w.recipe().compose.scale_bar_panel == "slices_0_1"
    assert w._style.scale_bar_loc == "upper left"


def test_apply_scale_bar_pick_none_is_a_no_op():
    w = _win()
    w.add_panels([_panel("a")])
    w._apply_scale_bar_pick(None, {})
    assert w.recipe().compose.scale_bar_mode != "one-panel"
    assert w.recipe().compose.scale_bar_panel is None


def test_apply_arranged_layout_replaces_purges_and_applies_pick():
    w = _win()
    w.add_panels([_panel("a"), _panel("b")])
    w.apply_arranged_layout(Row([PanelRef("a")]), scale_bar_pick=("a", "upper right"))
    assert [p.id for p in w.recipe().panels] == ["a"]  # "b" purged with the grid
    assert w.recipe().compose.scale_bar_mode == "one-panel"
    assert w.recipe().compose.scale_bar_panel == "a"
    assert w._style.scale_bar_loc == "upper right"
    assert w.recipe().style["scale_bar_loc"] == "upper right"
    assert w.is_dirty()


def test_arrange_button_present():
    w = _win()
    assert w._arrange_btn.text() == "Arrange…"


# -- compose-form colourbar + scale-bar controls ------------------------------
def test_compose_colorbar_controls_write_fields_and_gate_pos():
    w = _win()
    assert w.recipe().compose.colorbar_mode == "per-panel"
    assert not w._compose_cbar_pos.isEnabled()
    w._compose_cbar_mode.setCurrentIndex(w._compose_cbar_mode.findData("united"))
    assert w.recipe().compose.colorbar_mode == "united"
    assert w._compose_cbar_pos.isEnabled()
    w._compose_cbar_pos.setCurrentIndex(w._compose_cbar_pos.findData("bottom"))
    assert w.recipe().compose.colorbar_pos == "bottom"
    w._compose_cbar_mode.setCurrentIndex(w._compose_cbar_mode.findData("per-panel"))
    assert not w._compose_cbar_pos.isEnabled()


def test_compose_colorbar_widgets_reload_from_recipe():
    w = _win()
    w.recipe().compose.colorbar_mode = "united"
    w.recipe().compose.colorbar_pos = "bottom"
    w._load_compose_into_widgets()
    assert w._compose_cbar_mode.currentData() == "united"
    assert w._compose_cbar_pos.currentData() == "bottom"
    assert w._compose_cbar_pos.isEnabled()


def test_compose_trace_autoscale_checkbox_writes_field_and_reloads():
    w = _win()
    assert w.recipe().compose.trace_autoscale is False
    assert not w._compose_trace_autoscale.isChecked()
    w._compose_trace_autoscale.setChecked(True)  # toggled -> _on_compose_edited
    assert w.recipe().compose.trace_autoscale is True
    assert w.is_dirty()
    # reload path: recipe -> widget, signals blocked, no write-back
    w.recipe().compose.trace_autoscale = False
    w._load_compose_into_widgets()
    assert not w._compose_trace_autoscale.isChecked()
    assert w.recipe().compose.trace_autoscale is False


def test_compose_trace_look_widgets_write_fields_and_reload():
    w = _win()
    c = w.recipe().compose
    assert c.trace_linewidth is None and c.trace_color == "" and c.trace_font_scale is None
    assert w._compose_trace_lw.value() == 0.0  # 0 = follow (1.8 pt x font scale)
    assert w._compose_trace_font_scale.value() == 0.0  # 0 = follow Style font scale
    assert w._compose_trace_color.currentText() == ""
    w._compose_trace_lw.setValue(4.0)
    w._compose_trace_font_scale.setValue(1.5)
    w._compose_trace_color.setCurrentText("black")
    c = w.recipe().compose
    assert c.trace_linewidth == 4.0 and c.trace_font_scale == 1.5 and c.trace_color == "black"
    assert w.is_dirty()
    # back to 0 -> None (derived again)
    w._compose_trace_lw.setValue(0.0)
    w._compose_trace_font_scale.setValue(0.0)
    assert w.recipe().compose.trace_linewidth is None
    assert w.recipe().compose.trace_font_scale is None
    # reload path: recipe -> widgets, signals blocked, no write-back
    w.recipe().compose.trace_linewidth = 2.5
    w.recipe().compose.trace_color = "C1"
    w.recipe().compose.trace_font_scale = 0.8
    w._load_compose_into_widgets()
    assert w._compose_trace_lw.value() == 2.5
    assert w._compose_trace_color.currentText() == "C1"
    assert w._compose_trace_font_scale.value() == 0.8


def test_scale_bar_corner_combo_is_style_scale_bar_loc_both_ways():
    w = _win()
    w._compose_scale_bar_loc.setCurrentText("upper left")
    assert w._style.scale_bar_loc == "upper left"
    assert w.recipe().style["scale_bar_loc"] == "upper left"
    assert w._controls._w_bar_loc.currentText() == "upper left"
    w._controls._w_bar_loc.setCurrentText("lower left")  # style-pane edit
    assert w._style.scale_bar_loc == "lower left"
    assert w._compose_scale_bar_loc.currentText() == "lower left"


def test_scale_bar_locs_is_canonical():
    from dfxm.common.plotting import SCALE_BAR_LOCS
    from gui.widgets.export_dialog import _LOCS

    assert (
        list(SCALE_BAR_LOCS)
        == _LOCS
        == [
            "lower right",
            "lower left",
            "upper right",
            "upper left",
        ]
    )


# -- busy indication: async render worker (latest-wins) ----------------------
from tests.qt_helpers import export_and_wait, render_and_wait, wait_builder_idle  # noqa: E402


def test_async_render_shows_overlay_then_clears(tmp_path):
    w = _win()
    w.show()  # BusyOverlay.active reads isVisible(), which needs a shown ancestor chain
    w.add_panels(_obl_recipe_panels(tmp_path))
    w.render_now()
    # synchronous guarantee: the overlay is up and buttons gated BEFORE return
    assert w._overlay.active and not w._refresh_btn.isEnabled()
    assert not w._export_btn.isEnabled()
    wait_builder_idle(w)
    assert not w._overlay.active
    assert w._refresh_btn.isEnabled() and w._export_btn.isEnabled()
    assert w._result is not None and w._canvas is not None


def test_latest_wins_two_rapid_renders_one_canvas(tmp_path, monkeypatch):
    import threading

    import dfxm.compose.render as _render

    real = _render.render_recipe
    release = threading.Event()
    calls: list[str] = []

    def gated(recipe, *a, **k):
        calls.append(recipe.name)
        if len(calls) == 1:
            release.wait(30)  # hold render #1 until #2 has been requested
        return real(recipe, *a, **k)

    monkeypatch.setattr(_render, "render_recipe", gated)
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    # add_panels() arms the 300 ms schedule_preview debounce; without
    # stopping it here, the debounce can fire mid-wait_builder_idle() (its
    # processEvents()/sleep(0.01) loop can easily run past 300 ms) and issue
    # an extra, uncounted render_now() call that this test never asked for —
    # the test only meant to drive render_now() explicitly below. That made
    # this test order-dependent: it passed reliably as part of the full file
    # (earlier tests' overhead pushed timing around) but failed intermittently
    # run alone (fix wave F5).
    w._debounce.stop()
    shows: list = []
    orig_show = w._show_figure
    monkeypatch.setattr(w, "_show_figure", lambda fig: (shows.append(fig), orig_show(fig))[1])
    w.render_now()  # worker 1 (gen 1) — parked in gated()
    w.recipe().name = "second"
    w.render_now()  # gen 2 — queued behind worker 1
    release.set()
    wait_builder_idle(w)
    assert calls == ["untitled", "second"]  # serialized, both ran
    assert len(shows) == 1  # worker 1's stale result was DROPPED, never attached
    assert w._last_outcome is not None and w._canvas.figure is w._result.figure


def test_close_with_live_worker_drops_result_never_attaches(tmp_path, monkeypatch):
    """closeEvent must never join a live worker on the GUI thread (SIGABRT
    risk documented in Task 3's report) — it bumps the generation counter and
    drops any pending request instead, so a result delivered after close is
    discarded by _on_worker_result's generation check on arrival."""
    import threading

    import dfxm.compose.render as _render

    real = _render.render_recipe
    release = threading.Event()

    def gated(recipe, *a, **k):
        release.wait(30)
        return real(recipe, *a, **k)

    monkeypatch.setattr(_render, "render_recipe", gated)
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w.render_now()
    assert w._worker is not None
    w._dirty = False
    assert w.close()  # returns immediately — never joins the thread on the GUI thread
    assert w._pending_render is False and w._pending_export is None
    assert not w._debounce.isActive()
    release.set()
    w._worker.wait(30000)  # test-only join so monkeypatch outlives the worker
    _app.processEvents()
    assert w._canvas is None  # generation was bumped on close: result dropped


# -- fix wave (review F1-F7): dual pending slots + no-panels invalidation ----
def _gate_render_and_export(monkeypatch):
    """Patch render_recipe (gated: blocks on the first call only) and
    export_recipe (recorded, never gated) — shared setup for the F2 tests."""
    import threading

    import dfxm.compose.render as _render

    real_render = _render.render_recipe
    real_export = _render.export_recipe
    release = threading.Event()
    render_calls: list[str] = []
    export_calls: list[str] = []

    def gated_render(recipe, *a, **k):
        render_calls.append(recipe.name)
        if len(render_calls) == 1:
            release.wait(30)  # hold only the first (in-flight) render
        return real_render(recipe, *a, **k)

    def recorded_export(recipe, out_dir, *a, **k):
        export_calls.append(out_dir)
        return real_export(recipe, out_dir, *a, **k)

    monkeypatch.setattr(_render, "render_recipe", gated_render)
    monkeypatch.setattr(_render, "export_recipe", recorded_export)
    return release, render_calls, export_calls


def test_pending_export_survives_a_pending_render_export_then_render(tmp_path, monkeypatch):
    """F2: the old single ``_pending`` slot silently dropped an export queued
    behind a render (or vice versa) — whichever request landed second
    overwrote the first. Order here: render in flight -> export requested ->
    render requested; both a queued export AND a final render must still run."""
    release, render_calls, export_calls = _gate_render_and_export(monkeypatch)
    out = tmp_path / "out"
    monkeypatch.setattr(
        "gui.figure_builder.QFileDialog.getExistingDirectory", lambda *a, **k: str(out)
    )

    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w._debounce.stop()  # drive render_now()/export_now() explicitly below
    w.render_now()  # worker 1 (render) — parked in gated_render()
    assert w._worker is not None
    w.export_now()  # queued in the export slot
    w.render_now()  # queued in the render slot — must NOT clobber the export
    assert w._pending_export == str(out) and w._pending_render is True
    release.set()
    wait_builder_idle(w)
    # 3 render_recipe calls: the initial (gated) render, export_recipe's own
    # internal render_recipe call (export always re-renders, never reuses the
    # preview), and one final render.
    assert render_calls == ["untitled"] * 3
    assert export_calls == [str(out)]  # exactly one export, with the chosen dir
    assert w._pending_render is False and w._pending_export is None


def test_pending_render_survives_a_pending_export_render_then_export(tmp_path, monkeypatch):
    """F2, reverse request order: render in flight -> render requested ->
    export requested. _on_worker_finished always starts a queued export
    before a queued render (it snapshots the CURRENT recipe), so the export
    still runs first regardless of request order, and the render still
    follows it — neither request is dropped."""
    release, render_calls, export_calls = _gate_render_and_export(monkeypatch)
    out = tmp_path / "out"
    monkeypatch.setattr(
        "gui.figure_builder.QFileDialog.getExistingDirectory", lambda *a, **k: str(out)
    )

    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w._debounce.stop()
    w.render_now()  # worker 1 (render) — parked in gated_render()
    assert w._worker is not None
    w.render_now()  # queued in the render slot (requested first this time)
    w.export_now()  # queued in the export slot (requested second)
    release.set()
    wait_builder_idle(w)
    assert render_calls == ["untitled"] * 3  # initial + export's own internal render + final
    assert export_calls == [str(out)]
    assert w._pending_render is False and w._pending_export is None


def test_render_now_no_panels_invalidates_inflight_worker_and_pending(tmp_path, monkeypatch):
    """F3: render_now()'s no-panels branch must bump the generation AND clear
    both pending slots, or a worker already in flight against the
    since-deleted panels could land its result and re-attach a figure the
    user just deleted every panel out of."""
    import threading

    import dfxm.compose.render as _render

    real = _render.render_recipe
    release = threading.Event()

    def gated(recipe, *a, **k):
        release.wait(30)
        return real(recipe, *a, **k)

    monkeypatch.setattr(_render, "render_recipe", gated)
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w._debounce.stop()
    w.render_now()  # worker parked in gated() — snapshotted the recipe with the panel
    assert w._worker is not None
    gen_before = w._generation
    w.recipe().panels.clear()  # simulate "remove all panels"
    w.render_now()  # no-panels branch — must invalidate the in-flight worker too
    assert w._generation > gen_before
    assert w._pending_render is False and w._pending_export is None
    assert w._canvas is None
    assert w._notes_label.text() == "add panels to preview"
    release.set()
    wait_builder_idle(w)
    # the in-flight worker's late (stale-generation) result must never attach
    assert w._canvas is None
    assert w._notes_label.text() == "add panels to preview"


def test_save_as_appends_json_suffix_and_open_offers_all_files(tmp_path, monkeypatch):
    from gui.figure_builder import _ensure_json_suffix

    assert _ensure_json_suffix("/x/recipe") == "/x/recipe.json"
    assert _ensure_json_suffix("/x/recipe.json") == "/x/recipe.json"
    assert _ensure_json_suffix("/x/recipe.txt") == "/x/recipe.txt"  # explicit ext respected
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    target = tmp_path / "myfig"  # user typed no extension
    monkeypatch.setattr(
        "gui.figure_builder.QFileDialog.getSaveFileName", lambda *a, **k: (str(target), "")
    )
    w._on_save_as()
    assert (tmp_path / "myfig.json").exists() and not target.exists()
    assert w._current_path == str(tmp_path / "myfig.json")
    seen = {}

    def fake_open(*a, **k):
        seen["filter"] = a[3] if len(a) > 3 else k.get("filter")
        return ("", "")

    monkeypatch.setattr("gui.figure_builder.QFileDialog.getOpenFileName", fake_open)
    w._on_open()
    assert "All files" in seen["filter"]


def test_add_scale_bar_cell_and_trace_aspect_widget():
    from dfxm.compose.recipe import ScaleBarCell, iter_leaves

    w = _win()
    w.add_scale_bar()
    leaves = list(iter_leaves(w.recipe().layout))
    assert any(isinstance(x, ScaleBarCell) for x in leaves)
    w.select_node(next(x for x in leaves if isinstance(x, ScaleBarCell)))
    assert w._inspector.currentWidget() is w._page_spacer  # shares the size page
    w._spacer_w.setValue(4.0)
    assert next(x for x in leaves if isinstance(x, ScaleBarCell)).w_cm == 4.0
    assert w.recipe().compose.trace_aspect is None
    w._compose_trace_aspect.setValue(3.5)
    assert w.recipe().compose.trace_aspect == 3.5
    w._compose_trace_aspect.setValue(0.0)
    assert w.recipe().compose.trace_aspect is None


def test_row_col_gap_spin_writes_gap_cm_with_follow_sentinel():
    w = _win()
    w.add_col()
    col = w.recipe().layout.children[-1]
    w.select_node(col)
    assert w._col_gap.value() < 0 and col.gap_cm is None  # "follow gutter"
    w._col_gap.setValue(0.0)
    assert col.gap_cm == 0.0  # touching
    w._col_gap.setValue(1.5)
    assert col.gap_cm == 1.5
    w._col_gap.setValue(-0.1)
    assert col.gap_cm is None
    w.add_row()  # appended INTO the selected col
    row = col.children[-1]
    assert isinstance(row, Row)
    w.select_node(row)
    w._row_gap.setValue(0.25)
    assert row.gap_cm == 0.25
    # fill flags
    w.select_node(col)
    assert not w._col_fill.isChecked() and col.fill_height is False
    w._col_fill.setChecked(True)
    assert col.fill_height is True
    w.select_node(row)
    w._row_fill.setChecked(True)
    assert row.fill_width is True


def test_pick_roi_button_writes_roi_from_picker(tmp_path, monkeypatch):
    import gui.figure_builder as fb

    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    panel = w.recipe().panels[0]
    w._select_outline_panel(panel.id)
    seen = {}

    class FakeDlg:
        def __init__(self, previews, initial=None, parent=None):
            seen["previews"] = previews
            seen["initial"] = initial
            self.result = (1, 5, 2, 7)

        def exec(self):
            arr, sx, sy = seen["previews"][0][1]()  # the thunk really loads the full frame
            seen["shape"] = arr.shape
            return 1

    monkeypatch.setattr(fb, "ROIPickerDialog", FakeDlg, raising=False)
    w._ov_roi_pick.click()
    assert panel.roi == (1, 5, 2, 7)
    assert w._ov_roi.text() == "1,5,2,7"
    assert seen["initial"] is None and len(seen["shape"]) == 2
    # second pick is seeded with the current roi
    w._ov_roi_pick.click()
    assert seen["initial"] == (1, 5, 2, 7)


def test_pick_roi_offers_all_image_panels(tmp_path, monkeypatch):
    """Pick… previews every image panel (selected first, traces excluded) so the
    ROI can be checked on each map; the result still lands on the selected panel."""
    import gui.figure_builder as fb

    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    from dfxm.compose.recipe import PanelDef, PanelSource

    h5 = w.recipe().panels[0].source.h5_path
    sel = {"volume_id": "strain", "slice_name": "obl", "plane": 0}
    other = PanelDef("other", PanelSource(h5, "slice_plane", dict(sel)), title="Other map")
    trace = PanelDef("tr", PanelSource(h5, "profiles_trace", {"job": {}, "field": "strain"}))
    w.add_panels([other, trace])
    w._select_outline_panel("other")
    seen = {}

    class FakeDlg:
        def __init__(self, previews, initial=None, parent=None):
            seen["previews"] = previews
            seen["labels"] = [lbl for lbl, _t in previews]
            self.result = (0, 2, 0, 2)

        def exec(self):
            return 1

    monkeypatch.setattr(fb, "ROIPickerDialog", FakeDlg, raising=False)
    w._ov_roi_pick.click()
    assert seen["labels"] == ["Other map", "a"]  # selected first; trace excluded
    arr, _sx, _sy = seen["previews"][1][1]()  # the other map's thunk really loads
    assert arr.shape == (4, 5)
    assert w.recipe().panel_by_id()["other"].roi == (0, 2, 0, 2)
    assert w.recipe().panel_by_id()["a"].roi is None  # only the selected panel is written


def test_copy_roi_to_all_image_panels(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    from dfxm.compose.recipe import PanelDef, PanelSource

    h5 = w.recipe().panels[0].source.h5_path
    sel = {"volume_id": "strain", "slice_name": "obl", "plane": 0}
    other = PanelDef("other", PanelSource(h5, "slice_plane", dict(sel)))
    trace = PanelDef("tr", PanelSource(h5, "profiles_trace", {"job": {}, "field": "strain"}))
    w.add_panels([other, trace])
    src = w.recipe().panels[0]
    src.roi = (1, 5, 2, 7)
    src.crop_to_data = True
    w._select_outline_panel(src.id)
    w._ov_roi_all.click()
    assert other.roi == (1, 5, 2, 7) and other.crop_to_data is True
    assert trace.roi is None  # traces untouched
    assert w.is_dirty() and "copied to 1" in w._notes_label.text()
