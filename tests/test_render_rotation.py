"""Rotation-video rendering — the GL-free assembly path and the empty-grid guard."""

from __future__ import annotations

import os

import numpy as np

from dfxm.common import render


def _gradient_frame(i):
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :, 0] = (i * 40) % 256
    frame[:, i % 48, 1] = 255
    return frame


def test_write_image_video_writes_gif(tmp_path):
    base = os.path.join(tmp_path, "spin")
    written = render._write_image_video(_gradient_frame, 4, base, "gif", fps=5)
    assert written == base + ".gif"
    assert os.path.getsize(written) > 0


def test_write_image_video_both_prefers_mp4_or_falls_back(tmp_path):
    base = os.path.join(tmp_path, "spin")
    written = render._write_image_video(_gradient_frame, 3, base, "both", fps=5)
    # mp4 when ffmpeg is on PATH, else the GIF fallback; either way a file exists
    assert written in (base + ".mp4", base + ".gif")
    assert os.path.getsize(base + ".gif") > 0 or written == base + ".mp4"


def test_write_image_video_failed_mp4_is_removed(tmp_path, monkeypatch):
    class BoomWriter:
        def __init__(self, *a, **kw):
            raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(render, "FFMpegWriter", BoomWriter)
    base = os.path.join(tmp_path, "spin")
    with open(base + ".mp4", "wb") as fh:
        fh.write(b"partial")
    written = render._write_image_video(_gradient_frame, 3, base, "mp4", fps=5)
    assert written == base + ".gif"
    assert not os.path.exists(base + ".mp4")


def test_save_rotation_video_empty_volume_returns_none(tmp_path):
    import pytest

    pytest.importorskip("pyvista")
    volume = np.full((2, 3, 4), np.nan)
    out = render.save_rotation_video(
        volume, 1.0, 0.15, 0.38, 0.0, 1.0, "viridis", 0.85, os.path.join(tmp_path, "r"), "gif"
    )
    assert out is None
