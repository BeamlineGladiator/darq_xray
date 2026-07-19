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
    # planes-first panel: one row per (slice_name, plane_idx), union across volumes;
    # both raw_sum + strain share slice "plane_a" with 2 offsets -> 2 plane rows.
    assert len(dlg._panel._rows) == 2
    # and one quantity checkbox per volume_id
    assert set(dlg._panel.checked_quantity_keys()) == {"raw_sum", "strain"}
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


def test_slice_replot_defaults_to_all_selected(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path))
    sels = dlg._selections()  # opens with everything ticked → a plain Render remakes all
    assert sels, "dialog should open with all planes selected"
    assert {vid for vid, _s, _p in sels} == {"raw_sum", "strain"}


def test_slice_replot_passes_per_kind_clim(tmp_path, monkeypatch):
    from dfxm.stages import slices as sl
    from gui.widgets.slice_replot import SliceReplotDialog

    captured = {}

    def fake_render_replot(h5, selections, style, clim, out_dir, roi=None, **kw):
        captured["clim"] = clim
        return []

    monkeypatch.setattr(sl, "render_replot", fake_render_replot)
    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))  # volume_ids raw_sum + strain
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path))
    assert set(dlg._clim._edits) == {"raw_sum", "strain"}  # one row per volume_id present
    dlg._clim._edits["strain"][0].setText("-5")
    dlg._clim._edits["strain"][1].setText("5")
    dlg.select_all()
    dlg.render_selection(str(tmp_path))
    assert captured["clim"] == {"strain": (-5.0, 5.0)}  # only the filled kind


def test_slice_replot_clim_survives_reload(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path))
    dlg._clim._edits["strain"][0].setText("-7")
    dlg._clim._edits["strain"][1].setText("7")
    dlg._reload()  # e.g. pressing Load on the same file — must not wipe the limits
    assert dlg._clim._edits["strain"][0].text() == "-7"
    assert dlg._clim._edits["strain"][1].text() == "7"
    assert dlg._clim.clim_by_group() == {"strain": (-7.0, 7.0)}


def test_slice_replot_output_defaults_beside_h5(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default="")  # empty → auto-default
    out = dlg._out_edit.text()
    assert out.startswith(str(tmp_path))  # a subfolder beside the loaded h5
    assert os.path.basename(os.path.dirname(out)) == "replots"


def test_slice_replot_dialog_shows_plane_pixel_size(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))  # planes are (nv, nu) = (7, 9)
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path))
    # the planes-first panel no longer annotates a slice node inline (that hierarchy
    # is gone); the shape is still available on the catalog entry, and visible in
    # the Pick ROI… preview.
    assert dlg._catalog[0].shape == (7, 9)


def test_clim_rows_are_per_volume_id(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    _ = QApplication.instance() or QApplication([])
    h5 = tmp_path / "oblique_slices.h5"
    # two mosa-COM volumes sharing group 'mosa_com'
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    with h5py.File(h5, "w") as f:
        for vid in ("mosa_com_chi", "mosa_com_mu"):
            g = f.create_group(vid)
            g.attrs["kind"] = "mosa_com"
            g.attrs["cmap"] = "magma"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "deg"
            g.attrs["vmin"], g.attrs["vmax"] = -1.0, 1.0
            sg = g.create_group("plane_a")
            sg.create_dataset("slices", data=np.zeros((2, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path / "o"))
    keys = set(dlg._clim._edits.keys())
    assert keys == {"mosa_com_chi", "mosa_com_mu"}  # one row per quantity, not one 'mosa_com'
    dlg.deleteLater()


def test_slice_pick_roi_fills_boxes(tmp_path, monkeypatch):
    from gui.widgets.slice_replot import SliceReplotDialog

    _ = QApplication.instance() or QApplication([])
    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))  # helper already in this file: raw_sum + strain, plane_a
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path / "o"))

    import gui.widgets.slice_replot as SR

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (2, 6, 1, 8)

        def exec(self):
            return 1

    monkeypatch.setattr(SR, "ROIPickerDialog", _FakePicker, raising=False)
    dlg._on_pick_roi()
    assert (dlg._r0.text(), dlg._r1.text(), dlg._c0.text(), dlg._c1.text()) == ("2", "6", "1", "8")
    dlg.deleteLater()


def test_panel_default_all_checked_renders_everything(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))  # 2 volumes (raw_sum, strain) x slice "plane_a" x 2 planes
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path / "out"))
    assert dlg._panel.has_selection()
    written = dlg.render_selection(str(tmp_path / "out"))
    assert len(written) == 4  # 2 volumes x 2 planes


def test_panel_filter_and_check_all_visible_subsets(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path / "out"))
    dlg._panel.set_all_checked(False)
    dlg._panel._filter.setText("1")  # narrows to plane 1 only
    dlg._panel.check_all_visible()
    written = dlg.render_selection(str(tmp_path / "out"))
    assert len(written) == 2  # plane 1 in both volumes


def test_panel_section_header_shows_uniform_shape_px_hint(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))  # raw_sum + strain both store plane_a at (7, 9) px (Y×X)
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path / "out"))
    top = dlg._panel._tree.topLevelItem(0)  # single section: "plane_a"
    assert "7×9 px (Y×X)" in top.text(0)


def test_panel_section_header_shows_mixed_grids_hint(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    h5 = tmp_path / "oblique_slices.h5"
    u9 = np.linspace(-4.0, 4.0, 9)
    v7 = np.linspace(-3.0, 3.0, 7)
    u5 = np.linspace(-2.0, 2.0, 5)
    v3 = np.linspace(-1.0, 1.0, 3)
    with h5py.File(h5, "w") as f:
        for vid, u, v in (("raw_sum", u9, v7), ("strain", u5, v3)):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cmap"] = "gray"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "v"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("plane_a")  # same slice_name, different pixel shape
            sg.create_dataset("slices", data=np.zeros((2, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))
    _app = QApplication.instance() or QApplication([])
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path / "out"))
    top = dlg._panel._tree.topLevelItem(0)  # single section: "plane_a"
    assert "mixed grids — see Pick ROI…" in top.text(0)
