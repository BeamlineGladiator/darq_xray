"""Rotation-video assembly — GL-free frame pipeline and the empty-volume guard."""

from __future__ import annotations

import os

import numpy as np
import pytest

from darq_xray.common import render3d as R3


def _gradient_frame(i):
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :, 0] = (i * 40) % 256
    frame[:, i % 48, 1] = 255
    return frame


def _kw():
    return dict(
        fps=5,
        cbar_label="I",
        group=None,
        clim=(0.0, 1.0),
        log_scale=False,
        cmap="magma",
        px_per_um=2.0,
        style=None,
    )


def test_video_from_frames_writes_gif(tmp_path):
    base = os.path.join(tmp_path, "spin")
    written = R3._video_from_frames(_gradient_frame, 4, base, "gif", **_kw())
    assert written == base + ".gif"
    assert os.path.getsize(written) > 0


def test_video_from_frames_both_prefers_mp4_or_falls_back(tmp_path):
    base = os.path.join(tmp_path, "spin")
    written = R3._video_from_frames(_gradient_frame, 3, base, "both", **_kw())
    assert written in (base + ".mp4", base + ".gif")


def test_video_from_frames_failed_mp4_is_removed(tmp_path, monkeypatch):
    from darq_xray.common import render

    class BoomWriter:
        def __init__(self, *a, **kw):
            raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(render, "FFMpegWriter", BoomWriter)
    base = os.path.join(tmp_path, "spin")
    with open(base + ".mp4", "wb") as fh:
        fh.write(b"partial")
    written = R3._video_from_frames(_gradient_frame, 3, base, "mp4", **_kw())
    assert written == base + ".gif"
    assert not os.path.exists(base + ".mp4")


def test_save_rotation_video_empty_volume_returns_none(tmp_path):
    pytest.importorskip("pyvista")
    scene = R3.Scene3D(volume=np.full((2, 3, 4), np.nan), spacing=(0.15, 0.38, 1.0))
    out = R3.save_rotation_video(scene, os.path.join(tmp_path, "r"), "gif", cbar_label="x")
    assert out is None


def test_save_top_view_empty_volume_returns_none(tmp_path):
    pytest.importorskip("pyvista")
    scene = R3.Scene3D(volume=np.full((2, 3, 4), np.nan), spacing=(0.15, 0.38, 1.0))
    assert R3.save_top_view(scene, os.path.join(tmp_path, "t.png"), cbar_label="x") is None
