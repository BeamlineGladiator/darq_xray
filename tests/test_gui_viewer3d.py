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


def _wide_spec():
    """A volume too wide for a small GL 3-D texture limit (the STO2 shape, in
    miniature): 9 px on its longest axis against a limit of 6."""
    lv = LoadedVolume(
        volume=np.ones((2, 8, 9)),
        spacing=(0.15, 0.38, 2.0),
        cmap="magma",
        clim=(0.0, 1.0),
        cbar_label="Intensity",
        group="raw",
    )
    return VolumeSourceSpec(
        name="wide", load=lambda: lv, loader={"kind": "h5_dataset", "path": "/x", "dataset": "wide"}
    )


def test_window_opens_already_fitted_to_the_texture_limit(monkeypatch):
    """A volume over the limit renders blank; the viewer opens it coarsened
    instead, with the Downsample spin showing the factor in force."""
    monkeypatch.setattr("gui.widgets.viewer3d_window.R3.volume_texture_limit", lambda *a, **kw: 6)
    w = Viewer3DWindow(_wide_spec(), "visualize")
    w.load_and_render()
    assert w.scene.downsample == 2  # (2, 4, 4) -> 5 points <= 6
    assert w._downsample_spin.value() == 2
    assert w._autofit_note and "coarsened 2x" in w._autofit_note
    w.close()


def test_a_fitting_volume_opens_untouched(monkeypatch):
    monkeypatch.setattr(
        "gui.widgets.viewer3d_window.R3.volume_texture_limit", lambda *a, **kw: 4096
    )
    w = Viewer3DWindow(_wide_spec(), "visualize")
    w.load_and_render()
    assert w.scene.downsample == 1 and w._autofit_note is None
    w.close()


def _empty_spec():
    """A volume an ROI has cropped to nothing — the `(76, 0, 1832)` shape this
    project has really produced, in miniature."""
    lv = LoadedVolume(
        volume=np.ones((2, 0, 9)),
        spacing=(0.15, 0.38, 2.0),
        cmap="magma",
        clim=(0.0, 1.0),
        cbar_label="Intensity",
        group="raw",
    )
    return VolumeSourceSpec(
        name="empty", load=lambda: lv, loader={"kind": "h5_dataset", "path": "/x", "dataset": "e"}
    )


def test_an_empty_volume_keeps_its_note_even_though_nothing_was_coarsened(monkeypatch):
    """The note filter keys on "a coarsening happened", and an empty volume
    coarsens nothing — so the one note written for this window was thrown away
    here. Nothing else covers it: `oversize_note` is silent (a zero axis is
    under any limit) and `rebuild` falls through to "no finite voxels after
    threshold/clip", blaming the threshold for an ROI that read nothing."""
    monkeypatch.setattr(
        "gui.widgets.viewer3d_window.R3.volume_texture_limit", lambda *a, **kw: 2048
    )
    w = Viewer3DWindow(_empty_spec(), "visualize")
    w.load_and_render()
    assert w.scene.downsample == 1  # precondition: no coarsening to explain
    assert w._autofit_note and "is empty" in w._autofit_note
    w.close()


def test_driving_the_spin_keeps_the_empty_volume_note(monkeypatch):
    """ "The user is driving now" is the right rule for a coarsening they
    overrode, and the wrong one for an emptiness no factor can undo: clearing it
    dropped them back onto "no finite voxels after threshold/clip" at the exact
    moment they reached for a control to fix it."""
    monkeypatch.setattr(
        "gui.widgets.viewer3d_window.R3.volume_texture_limit", lambda *a, **kw: 2048
    )
    w = Viewer3DWindow(_empty_spec(), "visualize")
    w.load_and_render()
    assert w._autofit_note and "is empty" in w._autofit_note  # precondition
    w._downsample_spin.setValue(4)
    assert w._autofit_note and "is empty" in w._autofit_note
    w.close()


def test_driving_the_spin_clears_the_autofit_note(monkeypatch):
    """Spinning back to 1 must reproduce today's blank render, not keep
    explaining a coarsening that is no longer in force."""
    monkeypatch.setattr("gui.widgets.viewer3d_window.R3.volume_texture_limit", lambda *a, **kw: 6)
    w = Viewer3DWindow(_wide_spec(), "visualize")
    w.load_and_render()
    w._downsample_spin.setValue(1)
    assert w.scene.downsample == 1
    assert w._autofit_note is None
    w.close()


def _deep_spec():
    """Too DEEP for the limit: coarsening block-averages Y/X, never Z, so no
    factor can fit this one."""
    lv = LoadedVolume(
        volume=np.ones((40, 4, 4)),
        spacing=(0.15, 0.38, 2.0),
        cmap="magma",
        clim=(0.0, 1.0),
        cbar_label="Intensity",
        group="raw",
    )
    return VolumeSourceSpec(
        name="deep", load=lambda: lv, loader={"kind": "h5_dataset", "path": "/x", "dataset": "deep"}
    )


def test_an_unfittable_volume_gets_one_message_not_two(monkeypatch):
    """Showing the auto-fit note AND the oversize note put two blank-render
    warnings on one line that disagreed about the remedy."""
    monkeypatch.setattr("gui.widgets.viewer3d_window.R3.volume_texture_limit", lambda *a, **kw: 8)
    w = Viewer3DWindow(_deep_spec(), "visualize")
    w.load_and_render()
    text = w._status.text()
    assert text.count("BLANK") <= 1, text
    assert not ("cannot fit it" in text and "downsample to 0" in text), text
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


def test_threshold_and_downsample_controls(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    monkeypatch.setattr(w, "rebuild", lambda: None)
    w._thresh_min.setValue(0.2)
    w._thresh_max.setValue(0.9)
    w._thresh_check.setChecked(True)
    assert w.scene.threshold == (0.2, 0.9)
    w._thresh_check.setChecked(False)
    assert w.scene.threshold is None
    w._downsample_spin.setValue(4)
    assert w.scene.downsample == 4
    w.close()


def test_clip_plane_axis_and_flip(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    monkeypatch.setattr(w, "rebuild", lambda: None)
    w._clip_axis_combo.setCurrentText("Y")
    w._clip_check.setChecked(True)
    origin, normal = w.scene.clip
    # centre of (2,3,4) at spacing (0.15, 0.38, 2.0): y = 3*0.38/2
    assert origin[1] == pytest.approx(0.57)
    assert normal == (0.0, 1.0, 0.0)
    w._clip_flip_btn.click()
    assert w.scene.clip[1] == (0.0, -1.0, 0.0)
    w._clip_check.setChecked(False)
    assert w.scene.clip is None
    w.close()


def test_camera_fields_build_cameraspec(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    seen = {}
    monkeypatch.setattr(
        "gui.widgets.viewer3d_window.R3.apply_camera", lambda pl, cam: seen.update(cam=cam)
    )
    w._az_spin.setValue(30.0)
    w._el_spin.setValue(15.0)
    w._zoom_spin.setValue(1.5)
    w._cam_apply_btn.click()
    if w._canvas.available:  # offscreen may lack GL; the guard itself is under test
        assert seen["cam"].azimuth == 30.0 and seen["cam"].elevation == 15.0
    w.close()


def test_video_job_params_round_trip_jsonable():
    import json

    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    w.scene.mode = "surface"
    w.scene.downsample = 2
    job = w._video_job_params("/tmp/out/orbit", "mp4", 90, 15)
    json.dumps(job)  # JSON-able end to end
    assert job["loader"] == w._spec.loader
    assert job["scene"]["mode"] == "surface" and job["scene"]["downsample"] == 2
    assert job["cbar_label"] == "Intensity" and job["group"] == "raw"
    assert job["n_frames"] == 90 and job["base_path"] == "/tmp/out/orbit"
    # no GL offscreen -> orbit around the default pose
    assert job["base_camera"] is None or len(job["base_camera"]) == 3
    w.close()


def test_save_figure_writes_png(tmp_path, monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    fake_img = np.full((60, 80, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(
        "gui.widgets.viewer3d_window.R3.render_scene_image",
        lambda scene, cam, window_size: (fake_img, 2.0),
    )
    out = tmp_path / "fig.png"
    w._save_figure_to(str(out), window_size=(80, 60))
    assert out.stat().st_size > 0
    w.close()


def test_finish_video_ok_reports_an_empty_scene():
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    w._finish_video_ok({"video": "/tmp/orbit.gif"})
    assert w._status.text() == "rotation video saved to /tmp/orbit.gif"
    w._finish_video_ok({"video": None})  # empty scene -> not "saved to None"
    assert "nothing to export" in w._status.text()
    assert "None" not in w._status.text()
    w.close()


def test_finish_video_failed_shows_the_hint():
    from dfxm.runner import Failed

    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    w._finish_video_failed(Failed("no GL", "", "Install a GPU driver"))
    assert "no GL" in w._status.text() and "Install a GPU driver" in w._status.text()
    w._finish_video_failed(Failed("plain", ""))
    assert w._status.text() == "rotation video failed: plain"
    w.close()


def test_rebuild_hints_at_an_oversize_volume(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    monkeypatch.setattr("gui.widgets.viewer3d_window.R3.volume_texture_limit", lambda *a, **kw: 2)
    w.rebuild()
    if w._canvas.available:  # the status line only carries a scene when GL is up
        assert "texture limit" in w._status.text()
    w.close()


def test_export_buttons_disabled_without_gl():
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    if not w._canvas.available:
        assert not w._fig_btn.isEnabled()
        assert not w._video_btn.isEnabled()
    w.close()
