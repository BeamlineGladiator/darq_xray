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


def test_orbit_frames_follow_the_base_camera_zoom():
    """px_per_um must follow the base camera's distance (Save figure… does).

    The base_camera branch never assigned the pose before
    ``enable_parallel_projection()``, so the parallel scale froze at the
    populate-reset default and every video came out at one fixed zoom.
    """
    scale = []
    for dist in (20.0, 80.0):
        got = R3._orbit_frames(
            _scene(),
            elevation=0.0,
            zoom=1.0,
            base_camera=((2.25, 3.8, dist), (2.25, 3.8, 6.0), (0.0, 1.0, 0.0)),
            window_size=(160, 120),
        )
        assert got is not None
        get_frame, px_per_um = got
        get_frame.close()
        scale.append(px_per_um)
    assert scale[0] > 2.0 * scale[1]  # 4x closer eye -> ~4x more px per µm


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


def _half_nan(nx=60, ny=40, nz=8, *, lo=-1.0, hi=1.0):
    """A block whose LEFT half (in X) is NaN padding, values ramping lo..hi in X."""
    x = np.broadcast_to(np.linspace(lo, hi, nx), (nz, ny, nx)).astype(float).copy()
    x[:, :, : nx // 2] = np.nan
    return x


def _white_coverage(img):
    """Per-pixel mask of "something was drawn" against the white background."""
    a = np.asarray(img)[..., :3].astype(int)
    return np.abs(a - 255).max(axis=-1) > 6


def _nan_side_coverage(scene):
    got = R3.render_scene_image(scene, R3.CameraSpec(preset="front"), window_size=(200, 150))
    assert got is not None
    mask = _white_coverage(got[0])
    half = mask.shape[1] // 2  # "front" preset: +X to the right, NaN half on the left
    return mask[:, :half].mean() * 100.0, mask[:, half:].mean() * 100.0


@pytest.mark.parametrize("mapping", R3.OPACITY_MAPPINGS)
def test_nan_padding_is_invisible_for_a_symmetric_clim(mapping):
    """NaN was uploaded as 0.0 — MID-range for a symmetric clim, so the padding
    rendered as a semi-opaque slab of mid-colormap fog (the default for the
    heavily NaN-padded CoM/strain volumes)."""
    scene = R3.Scene3D(
        volume=_half_nan(),
        spacing=(1.0, 1.0, 1.0),
        clim=(-1.0, 1.0),
        opacity_mapping=mapping,
        background="white",
    )
    nan_side, _data_side = _nan_side_coverage(scene)
    assert nan_side < 0.5  # was ~35% (= fully covered) for linear/sigmoid


def test_nan_padding_is_invisible_in_log_space():
    """log10(NaN->0.0) = 0 is mid-range for any log clim with vmin < 1."""
    scene = R3.Scene3D(
        volume=_half_nan(lo=0.1, hi=100.0),
        spacing=(1.0, 1.0, 1.0),
        clim=(0.1, 100.0),
        log_scale=True,
        background="white",
    )
    nan_side, data_side = _nan_side_coverage(scene)
    assert nan_side < 0.5
    assert data_side > 10.0  # the real data still renders


def test_symmetric_clim_still_renders_the_data_half():
    scene = R3.Scene3D(
        volume=_half_nan(), spacing=(1.0, 1.0, 1.0), clim=(-1.0, 1.0), background="white"
    )
    _nan_side, data_side = _nan_side_coverage(scene)
    assert data_side > 10.0


def test_geom_r_keeps_low_scalars_opaque():
    """geom_r is high-alpha at LOW scalars — the NaN sentinel must not take the
    data with it (a naive "sentinel far below vmin" would leave it opaque, and
    clipping too hard would make the lowest real data vanish)."""
    vol = np.full((8, 40, 60), 0.0)
    vol[:, :, :30] = np.nan
    scene = R3.Scene3D(
        volume=vol,
        spacing=(1.0, 1.0, 1.0),
        clim=(0.0, 1.0),
        opacity_mapping="geom_r",
        background="white",
    )
    nan_side, data_side = _nan_side_coverage(scene)
    assert nan_side < 0.5
    assert data_side > 10.0


def test_volume_texture_limit_and_oversize_note():
    """The limit query must answer with a plausible size (or None) and the note
    must fire exactly for volumes that VTK cannot texture."""
    limit = R3.volume_texture_limit()
    assert limit is None or limit >= 256
    small = _scene()
    assert R3.oversize_note(small, limit) is None
    if limit is not None:
        big = R3.Scene3D(volume=np.zeros((2, 2, limit)), spacing=(1.0, 1.0, 1.0))
        note = R3.oversize_note(big, limit)
        assert note and "texture" in note and str(limit) in note
        big.mode = "surface"  # geometry, not a 3-D texture -> no note
        assert R3.oversize_note(big, limit) is None


def test_save_top_view_end_to_end_all_modes(tmp_path):
    for mode in ("volume", "surface", "isosurface"):
        out = R3.save_top_view(
            _scene(mode),
            os.path.join(tmp_path, f"tv_{mode}.png"),
            cbar_label="I",
            window_size=(160, 120),
        )
        assert out is not None and os.path.getsize(out) > 0
