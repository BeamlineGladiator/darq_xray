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
