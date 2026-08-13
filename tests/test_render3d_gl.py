"""Real-GL 3-D rendering checks — skipped wholesale without an off-screen GL context."""

from __future__ import annotations

import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from dfxm.common import render3d as R3  # noqa: E402


def _gl_available() -> bool:
    try:
        pl = pv.Plotter(off_screen=True, window_size=[64, 48])
        pl.add_mesh(pv.Cube())
        img = pl.screenshot(return_img=True)
        pl.close()
        return img is not None and np.asarray(img).size > 0
    except Exception:  # noqa: BLE001 - any GL/driver failure -> skip the module
        return False


pytestmark = pytest.mark.skipif(not _gl_available(), reason="no usable off-screen GL context")


def _scene(mode="volume"):
    # asymmetric ramp so every orbit angle looks different
    z, y, x = np.meshgrid(np.arange(6), np.arange(20), np.arange(30), indexing="ij")
    vol = (x * 2.0 + y * 0.5 + z).astype(float)
    return R3.Scene3D(volume=vol, spacing=(0.15, 0.38, 2.0), mode=mode, clim=(0.0, 70.0))


def test_orbit_frames_actually_rotate():
    got = R3._orbit_frames(
        _scene(), elevation=20.0, zoom=1.2, base_camera=None, window_size=(160, 120)
    )
    assert got is not None
    get_frame, px_per_um = got
    try:
        assert px_per_um > 0
        f0, f90, f180 = (np.asarray(get_frame(i), dtype=float) for i in (0, 45, 90))
        # THE regression check: distinct azimuths must produce distinct images
        assert np.abs(f0 - f90).mean() > 1.0
        assert np.abs(f0 - f180).mean() > 1.0
        assert np.abs(f90 - f180).mean() > 1.0
        # idempotent: re-rendering frame 0 reproduces it exactly (MP4->GIF replay)
        assert np.array_equal(np.asarray(get_frame(0)), np.asarray(get_frame(0)))
    finally:
        get_frame.close()


def test_orbit_frames_closes_the_plotter_when_setup_fails(monkeypatch):
    """A failure during camera setup must not leak the off-screen plotter."""
    seen = []
    real_populate = R3.populate

    def spy(plotter, scene, **kw):
        seen.append(plotter)
        return real_populate(plotter, scene, **kw)

    def boom(*a, **kw):
        raise RuntimeError("camera setup failed")

    monkeypatch.setattr(R3, "populate", spy)
    monkeypatch.setattr(R3, "apply_camera", boom)
    with pytest.raises(RuntimeError, match="camera setup failed"):
        R3._orbit_frames(_scene(), elevation=20.0, zoom=1.2, base_camera=None, window_size=(64, 48))
    assert len(seen) == 1
    assert seen[0].render_window is None  # closed


def test_volume_mode_opacity_changes_the_render():
    """Scene3D.opacity must scale volume-mode transparency, not just meshes.

    The volume branch used to pass the *mapping name* straight to add_volume, so
    the scalar opacity (the stages' ``volume_opacity`` param) was a silent no-op.
    """
    imgs = []
    for opacity in (0.1, 0.9):
        scene = _scene()
        scene.opacity = opacity
        got = R3.render_scene_image(scene, R3.CameraSpec(preset="iso"), window_size=(160, 120))
        assert got is not None
        imgs.append(np.asarray(got[0], dtype=float))
    assert np.abs(imgs[0] - imgs[1]).mean() > 1.0


def test_save_rotation_video_end_to_end(tmp_path):
    out = R3.save_rotation_video(
        _scene(),
        os.path.join(tmp_path, "orbit"),
        "gif",
        cbar_label="I",
        n_frames=6,
        fps=5,
        window_size=(160, 120),
    )
    assert out is not None and os.path.getsize(out) > 0


def test_save_top_view_end_to_end_all_modes(tmp_path):
    for mode in ("volume", "surface", "isosurface"):
        out = R3.save_top_view(
            _scene(mode),
            os.path.join(tmp_path, f"tv_{mode}.png"),
            cbar_label="I",
            window_size=(160, 120),
        )
        assert out is not None and os.path.getsize(out) > 0
