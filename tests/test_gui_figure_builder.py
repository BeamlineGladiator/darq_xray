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
    res = w.render_now()
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
    res = w.render_now()
    assert res is not None and res.n_rendered == 1
    assert w._canvas is not None  # a live FigureCanvasQTAgg wrapping res.figure
    assert w._canvas.figure is res.figure


def test_render_error_lands_in_notes_bar_not_crash(tmp_path):
    w = _track(FigureBuilderWindow(lambda: {}, PlotStyle()))  # NO scale anywhere
    w.add_panels(_obl_recipe_panels(tmp_path))
    res = w.render_now()
    assert res is None
    assert "scale" in w._notes_label.text().lower()
    assert "hint" in w._notes_label.text().lower() or "Set Scale" in w._notes_label.text()


def test_cache_survives_file_deletion_until_refresh(tmp_path):
    w = _win()
    panels = _obl_recipe_panels(tmp_path)
    w.add_panels(panels)
    assert w.render_now() is not None
    os.remove(panels[0].source.h5_path)
    res2 = w.render_now()  # served from cache
    assert res2 is not None and res2.n_rendered == 1
    w.refresh_data()  # cache cleared -> placeholder now
    res3 = w.render_now()
    assert res3 is not None and res3.n_rendered == 0
    assert "placeholder" in w._notes_label.text()


def test_click_preview_selects_outline_node(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    res = w.render_now()
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
    w.export_now()
    assert os.path.exists(out / "untitled.png")
    assert "wrote" in w._notes_label.text()


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
    w.export_now()  # must not raise
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
