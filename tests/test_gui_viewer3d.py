"""Viewer3DWindow + launcher — construction, scene wiring, close/free (offscreen)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.viewers import LoadedVolume, VolumeSourceSpec  # noqa: E402
from gui.widgets.viewer3d_window import Viewer3DWindow  # noqa: E402
from gui.widgets.volume3d import Volume3DPanel  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _spec(name="vol"):
    lv = LoadedVolume(
        volume=np.ones((2, 3, 4)),
        spacing=(0.15, 0.38, 2.0),
        cmap="magma",
        clim=(0.0, 1.0),
        cbar_label="Intensity",
        group="raw",
    )
    return VolumeSourceSpec(
        name=name, load=lambda: lv, loader={"kind": "h5_dataset", "path": "/x", "dataset": name}
    )


def test_window_builds_scene_from_source():
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    assert w.loaded is not None
    assert w.scene.mode == "volume"
    assert w.scene.clim == (0.0, 1.0)
    assert w.windowTitle() == "visualize — vol (3D)"
    w.close()


def test_window_survives_without_gl():
    # offscreen: PvCanvas.ensure() may fail -> placeholder; nothing raises
    w = Viewer3DWindow(_spec(), "rocking")
    w.load_and_render()
    w.rebuild()  # second rebuild also safe
    w.close()


def test_launcher_opens_windows_and_prunes_closed():
    panel = Volume3DPanel()
    panel.set_sources({"a": _spec("a"), "b": _spec("b")})
    panel._combo.setCurrentText("a")
    panel._open_btn.click()
    assert len(panel._windows) == 1
    panel._windows[0].close()
    _app.processEvents()
    assert len(panel._windows) == 0


def test_launcher_disabled_without_sources():
    panel = Volume3DPanel()
    panel.set_sources({})
    assert not panel._open_btn.isEnabled()


def _broken_spec(name="broken"):
    def _load():
        raise RuntimeError("boom")

    return VolumeSourceSpec(
        name=name, load=_load, loader={"kind": "h5_dataset", "path": "/x", "dataset": name}
    )


def test_launcher_surfaces_load_failure_without_raising():
    panel = Volume3DPanel()
    panel.set_sources({"broken": _broken_spec()})
    panel._combo.setCurrentText("broken")
    panel._open_btn.click()  # must not raise
    assert "open failed" in panel._status.text()
    assert "boom" in panel._status.text()
    assert len(panel._windows) == 0


def test_controls_mutate_scene_and_trigger_rebuild(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    calls = []
    monkeypatch.setattr(w, "rebuild", lambda: calls.append(1))
    w._mode_combo.setCurrentText("isosurface")
    assert w.scene.mode == "isosurface"
    w._clim_min.setValue(0.2)
    w._clim_max.setValue(0.8)
    assert w.scene.clim == (0.2, 0.8)
    w._opacity_slider.setValue(40)
    assert w.scene.opacity == pytest.approx(0.4)
    w._mapping_combo.setCurrentText("sigmoid")
    assert w.scene.opacity_mapping == "sigmoid"
    assert len(calls) >= 4
    w.close()


def test_log_checkbox_guard():
    w = Viewer3DWindow(_spec(), "visualize")  # clim (0.0, 1.0) -> vmin not > 0
    w.load_and_render()
    assert not w._log_check.isEnabled()
    w._clim_min.setValue(0.1)
    assert w._log_check.isEnabled()
    w._log_check.setChecked(True)
    assert w.scene.log_scale is True
    w.close()


def test_auto_clim_button_resets_from_volume():
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    w._clim_min.setValue(0.4)
    w._clim_auto_btn.click()
    lo, hi = w.scene.clim
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)  # all-ones volume
    w.close()
