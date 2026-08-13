"""Shared 3-D volume scene — one description of "what to render" for everything.

:class:`Scene3D` + :func:`populate` are the single 3-D setup used by the
visualize stage's top view and rotation video AND the GUI's pop-out viewer, so
an exported figure is guaranteed to look like the interactive view. ``pyvista``
is imported lazily inside functions (a missing GL stack only disables 3-D);
this module stays Qt-free and figure code uses the explicit Figure/Agg API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRESETS = ("front", "top", "side", "iso")
OPACITY_MAPPINGS = ("linear", "sigmoid", "geom", "geom_r")
RENDER_MODES = ("volume", "surface", "isosurface")


def downsample_volume(vol: np.ndarray, factor: int) -> np.ndarray:
    """Block-average (nanmean) over factor×factor Y/X blocks; Z untouched."""
    if factor <= 1:
        return vol
    z, y, x = vol.shape
    yc, xc = (y // factor) * factor, (x // factor) * factor
    v = vol[:, :yc, :xc].reshape(z, yc // factor, factor, xc // factor, factor)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks -> NaN
        return np.nanmean(v, axis=(2, 4))


def threshold_mask(vol: np.ndarray, window) -> np.ndarray:
    """NaN out voxels outside the (tmin, tmax) value window; None = no-op."""
    if window is None:
        return vol
    tmin, tmax = float(window[0]), float(window[1])
    out = vol.copy()
    out[(out < tmin) | (out > tmax)] = np.nan
    return out


def clip_mask(vol: np.ndarray, spacing, origin, normal) -> np.ndarray:
    """NaN out voxels on the negative side of the plane (world µm, cell centres)."""
    sx, sy, sz = (float(s) for s in spacing)
    z, y, x = vol.shape
    xs = (np.arange(x) + 0.5) * sx
    ys = (np.arange(y) + 0.5) * sy
    zs = (np.arange(z) + 0.5) * sz
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    n = np.asarray(normal, dtype=float)
    o = np.asarray(origin, dtype=float)
    side = (xx - o[0]) * n[0] + (yy - o[1]) * n[1] + (zz - o[2]) * n[2]
    out = vol.copy()
    out[side < 0.0] = np.nan
    return out


def auto_clim(vol: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    valid = vol[np.isfinite(vol)]
    if valid.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(valid, lo)), float(np.percentile(valid, hi)))


def log_valid(clim) -> bool:
    """Log colour mapping is only meaningful for an all-positive colour range."""
    return clim is not None and float(clim[0]) > 0.0 and float(clim[1]) > 0.0


@dataclass
class Scene3D:
    """Everything needed to render one volume in 3-D (Qt-free, JSON-friendly)."""

    volume: np.ndarray  # (Z, Y, X) float
    spacing: tuple  # (sx, sy, sz) µm/px
    mode: str = "volume"  # RENDER_MODES
    n_isosurfaces: int = 10
    cmap: str = "magma"
    clim: tuple | None = None  # None -> auto_clim
    log_scale: bool = False
    opacity: float = 0.85  # surface/isosurface modes
    opacity_mapping: str = "linear"  # volume mode transfer function
    threshold: tuple | None = None  # (tmin, tmax) value window
    clip: tuple | None = None  # ((ox, oy, oz), (nx, ny, nz)) µm
    downsample: int = 1
    background: str = "white"

    def resolved_clim(self):
        return self.clim if self.clim is not None else auto_clim(self.volume)

    def prepared(self):
        """Volume after downsample -> threshold -> clip, plus adjusted spacing."""
        vol = downsample_volume(self.volume, int(self.downsample))
        sx, sy, sz = (float(s) for s in self.spacing)
        if int(self.downsample) > 1:
            sx, sy = sx * int(self.downsample), sy * int(self.downsample)
        vol = threshold_mask(vol, self.threshold)
        if self.clip is not None:
            vol = clip_mask(vol, (sx, sy, sz), self.clip[0], self.clip[1])
        return vol, (sx, sy, sz)


@dataclass
class CameraSpec:
    """A reproducible camera pose: preset base + azimuth/elevation/zoom."""

    preset: str = "front"  # PRESETS
    azimuth: float = 0.0
    elevation: float = 0.0
    zoom: float = 1.0


def _rotate(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation of *vec* about unit *axis* by *angle_deg*."""
    a = np.deg2rad(angle_deg)
    axis = axis / np.linalg.norm(axis)
    return (
        vec * np.cos(a)
        + np.cross(axis, vec) * np.sin(a)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(a))
    )


def orbit_positions(base_camera, elevation_deg: float, n_frames: int):
    """Absolute (eye, focal, up) poses for a 360° orbit — pure numpy.

    Frame *i* = base eye rotated about the view-up axis through the focal
    point by ``i*360/n`` (azimuth), then lifted by *elevation_deg* about the
    horizontal axis. Absolute poses (never incremental camera mutation) are
    what make video frame generation idempotent — the fix for the
    "video doesn't rotate" bug and a requirement of the MP4→GIF replay.
    """
    eye, focal, up = (np.asarray(v, dtype=float) for v in base_camera)
    out = []
    for i in range(n_frames):
        angle = 360.0 * i / n_frames
        e = _rotate(eye - focal, up, angle) + focal
        if elevation_deg:
            horiz = np.cross(up, focal - e)
            if np.linalg.norm(horiz) > 1e-12:
                e = _rotate(e - focal, horiz, float(elevation_deg)) + focal
        out.append((tuple(e), tuple(focal), tuple(np.asarray(base_camera[2], dtype=float))))
    return out
