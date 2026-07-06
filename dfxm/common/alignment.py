"""Shared volume alignment — one implementation, voxel-identical everywhere.

Ported from the alignment used by the PVTI exporter and the aligned-volume
viewer so that visualization, ParaView export, oblique slices and the rocking
volumes all co-register. The fixed pipeline (order matters) is::

    abs (FWHM only) -> ROI -> samy sub-pixel X-shift (canvas auto-expand)
        -> Z interpolation onto a uniform grid -> centre (CoM / strain only)

All geometry constants (pixel scale, samy direction, ROI, darfix origin) are
explicit arguments here rather than module globals, so the same code serves any
experiment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import shift as ndi_shift


def apply_roi_3d(data: np.ndarray, roi_x: tuple | None, roi_y: tuple | None) -> np.ndarray:
    """Crop a (Z, Y, X) volume in pixel coordinates; ``None`` keeps the full axis."""
    _z, y, x = data.shape
    xs, xe = (roi_x[0], roi_x[1]) if roi_x else (0, x)
    ys, ye = (roi_y[0], roi_y[1]) if roi_y else (0, y)
    return data[:, ys:ye, xs:xe]


def _samy_ref(samy: np.ndarray, ref_samy: float | None) -> float:
    """Resolve the samy reference: explicit value, else the first layer."""
    return float(samy[0]) if ref_samy is None else float(ref_samy)


def compute_pad_left(
    samy: np.ndarray, scale_x: float, samy_direction: int = 1, ref_samy: float | None = None
) -> int:
    """Left X-padding (px) that :func:`apply_samy_shifts_to_volume` will add.

    Same formula as the shift, so a world origin can be anchored to the
    reference (un-shifted) column. ``ref_samy`` overrides the default
    first-layer reference (the rocking stage anchors to the mosa frame).
    """
    if samy is None or len(samy) == 0:
        return 0
    offsets_px = samy_direction * (np.asarray(samy) - _samy_ref(samy, ref_samy)) * 1000.0 / scale_x
    return max(0, int(np.ceil(-np.min(offsets_px))))


def compute_pad_right(
    samy: np.ndarray, scale_x: float, samy_direction: int = 1, ref_samy: float | None = None
) -> int:
    """Right X-padding (px) that :func:`apply_samy_shifts_to_volume` will add."""
    if samy is None or len(samy) == 0:
        return 0
    offsets_px = samy_direction * (np.asarray(samy) - _samy_ref(samy, ref_samy)) * 1000.0 / scale_x
    return max(0, int(np.ceil(np.max(offsets_px))))


def apply_samy_shifts_to_volume(
    volume: np.ndarray,
    samy: np.ndarray,
    scale_x: float,
    samy_direction: int = 1,
    ref_samy: float | None = None,
) -> np.ndarray:
    """Shift each Z-layer along image-X by its samy offset, expanding the canvas.

    samy is in mm; offsets are relative to ``ref_samy`` (default: the first
    layer). The canvas grows so nothing is clipped, and exposed regions are
    NaN-padded. Pass ``ref_samy`` to anchor to an external frame (e.g. the
    rocking volume to the mosa reference column).
    """
    n_layers = volume.shape[0]
    n_use = n_layers if len(samy) == n_layers else min(n_layers, len(samy))

    ref = _samy_ref(samy, ref_samy)
    samy_offsets_px = samy_direction * (np.asarray(samy[:n_use]) - ref) * 1000.0 / scale_x
    pad_left = max(0, int(np.ceil(-np.min(samy_offsets_px))))
    pad_right = max(0, int(np.ceil(np.max(samy_offsets_px))))

    ny, nx_orig = volume.shape[1], volume.shape[2]
    nx_new = nx_orig + pad_left + pad_right
    shifted = np.full((n_layers, ny, nx_new), np.nan, dtype=volume.dtype)

    for i in range(n_layers):
        padded = np.full((ny, nx_new), np.nan, dtype=volume.dtype)
        padded[:, pad_left : pad_left + nx_orig] = volume[i]
        if i < n_use and abs(samy_offsets_px[i]) > 0.01:
            padded = ndi_shift(
                padded, shift=(0.0, samy_offsets_px[i]), order=1, mode="constant", cval=np.nan
            )
        shifted[i] = padded
    return shifted


def interpolate_to_uniform_z(
    volume: np.ndarray, samz: np.ndarray, ref_samz: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Resample irregular samz (mm) layers onto a uniform Z grid (µm).

    Z origin is ``ref_samz`` (default: the first layer); pass it to anchor the
    rocking volume to the mosa Z reference. Returns
    ``(interp_volume, z_uniform_um, scale_z_um)``.
    """
    n_use = min(volume.shape[0], len(samz))
    ref = float(samz[0]) if ref_samz is None else float(ref_samz)
    z_um = (np.asarray(samz[:n_use]) - ref) * 1000.0

    median_step = float(np.median(np.abs(np.diff(z_um)))) if n_use > 1 else 1.0
    if median_step < 1e-6:
        median_step = 1.0

    if n_use == 1:
        # A single layer has no Z extent: return it unchanged on a length-1 grid
        # with a nonzero scale (avoids scale_z=0 and the all-NaN interp1d result).
        return volume[:1].astype(volume.dtype), z_um, median_step

    z_min, z_max = float(z_um.min()), float(z_um.max())
    n_uniform = max(2, int(np.round((z_max - z_min) / median_step)) + 1)
    z_uniform = np.linspace(z_min, z_max, n_uniform)
    scale_z = float(z_uniform[1] - z_uniform[0]) if n_uniform > 1 else median_step

    vol_use = volume[:n_use]
    ny, nx = vol_use.shape[1], vol_use.shape[2]
    flat = vol_use.reshape(n_use, -1)
    interp = interp1d(z_um, flat, axis=0, kind="linear", bounds_error=False, fill_value=np.nan)
    vol_interp = interp(z_uniform).reshape(n_uniform, ny, nx)
    return vol_interp, z_uniform, scale_z


def center_around_zero(data: np.ndarray, method: str = "mean") -> tuple[np.ndarray, float]:
    """Subtract a global statistic over finite voxels. Returns ``(data, offset)``."""
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return data, 0.0
    m = method.lower()
    if m == "mean":
        offset = float(np.nanmean(valid))
    elif m == "median":
        offset = float(np.nanmedian(valid))
    else:
        raise ValueError(f"center_around_zero: unknown method {method!r} (expected mean/median)")
    return data - offset, offset


def raw_detector_origin(
    samy: np.ndarray,
    z_positions: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    roi_x: tuple | None = None,
    roi_y: tuple | None = None,
    darfix_origin_xy: tuple = (0, 0),
    samy_direction: int = 1,
) -> tuple[float, float, float]:
    """World origin (µm) placing this volume in the raw-detector-absolute frame.

    A padded voxel at column p, row j maps back to raw pixel
    ``(p - pad_left + roi_x0 + darfix_x0, j + roi_y0 + darfix_y0)``; the origin
    is that mapping times the pixel size, with Z taken from ``z_positions[0]``.
    """
    pad_left = compute_pad_left(samy, scale_x, samy_direction)
    dx0, dy0 = darfix_origin_xy
    roi_x0 = roi_x[0] if roi_x else 0
    roi_y0 = roi_y[0] if roi_y else 0
    ox = (roi_x0 + dx0 - pad_left) * scale_x
    oy = (roi_y0 + dy0) * scale_y
    oz = float(z_positions[0]) if z_positions is not None and len(z_positions) else 0.0
    return (ox, oy, oz)


@dataclass
class AlignedVolume:
    """Output of :func:`align_volume`."""

    data: np.ndarray
    z_uniform_um: np.ndarray
    scale_z_um: float
    pad_left: int
    center_offset: float = 0.0


def align_volume(
    volume: np.ndarray,
    samy: np.ndarray,
    samz: np.ndarray,
    *,
    scale_x: float,
    samy_direction: int = 1,
    roi_x: tuple | None = None,
    roi_y: tuple | None = None,
    take_abs: bool = False,
    center_method: str | None = None,
) -> AlignedVolume:
    """Run the full fixed alignment pipeline on a (Z, Y, X) volume.

    ``take_abs`` is for FWHM maps (drop unphysical negative fits, NaN-safe);
    ``center_method`` ("mean"/"median") centres CoM/strain volumes around zero.
    """
    v = np.abs(volume) if take_abs else volume
    v = apply_roi_3d(v, roi_x, roi_y)
    pad_left = compute_pad_left(samy, scale_x, samy_direction)
    v = apply_samy_shifts_to_volume(v, samy, scale_x, samy_direction)
    v, z_uniform, scale_z = interpolate_to_uniform_z(v, samz)
    offset = 0.0
    if center_method:
        v, offset = center_around_zero(v, center_method)
    return AlignedVolume(v, z_uniform, scale_z, pad_left, offset)
