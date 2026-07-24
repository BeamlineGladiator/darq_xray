"""Figure-builder window + panel picker (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.plotting import PlotStyle  # noqa: E402
from dfxm.compose.recipe import PanelDef, PanelSource, Row  # noqa: E402
from gui.figure_builder import FigureBuilderWindow  # noqa: E402


def _win():
    return FigureBuilderWindow(lambda: {}, PlotStyle(scale_um_per_cm=10.0))


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


def test_save_open_round_trip_and_dirty_state(tmp_path):
    w = _win()
    w.add_panels([_panel("a")])
    path = str(tmp_path / "r.json")
    w.save_recipe_file(path)
    assert not w.is_dirty()
    w2 = _win()
    w2.load_recipe_file(path)
    assert [p.id for p in w2.recipe().panels] == ["a"]
    assert not w2.is_dirty()
    assert "r" in w2.windowTitle() or w2.recipe().name


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
