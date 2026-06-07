"""Rocking-matched layers stage — raw rocking images aligned to strain layers.

Faithful port of ``plot_rocking_matched_layers_v3.py``. For each strain layer it
finds the nearest rocking scan by ``(samy, samz)``, loads one detector frame
(median-background subtracted, negatives→NaN), applies the SAME samy X-shift the
strain/mosaicity layers use, and saves a grayscale PNG pixel-aligned with the
strain/mosaicity layer images.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.ndimage import shift as ndi_shift

from ..common import alignment as A
from ..common import render as Rnd
from ..common.raster import extract_motor_positions, find_h5_file
from ..common.sort import find_matching_folders
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


STAGE = StageSpec(
    name="matched",
    label="Rocking-matched layers",
    description=(
        "For each strain layer, find the nearest rocking scan by (samy, samz), "
        "load a background-subtracted frame, apply the strain samy shift, and "
        "save grayscale PNGs pixel-aligned with the strain/mosaicity layers."
    ),
    params=(
        Param("raw_root", ParamType.DIR, "Raw data root"),
        Param("strain_pattern", ParamType.STR, "Strain pattern", default="*"),
        Param("rocking_pattern", ParamType.STR, "Rocking pattern", default="*"),
        Param("samy_path", ParamType.STR, "samy path", default="1.1/instrument/positioners/samy"),
        Param("samz_path", ParamType.STR, "samz path", default="1.1/instrument/positioners/samz"),
        Param("pco_ff_path", ParamType.STR, "Detector path", default="1.1/measurement/pco_ff"),
        Param(
            "frame_index", ParamType.INT, "Frame index", default=0, help="0-based frame in pco_ff"
        ),
        Param(
            "match_threshold_mm",
            ParamType.FLOAT,
            "Match threshold",
            unit="mm",
            default=0.0004,
            help="max (samy,samz) distance to accept a match",
        ),
        Param(
            "pixel_size_x_um",
            ParamType.FLOAT,
            "Pixel size X",
            unit="µm",
            default=0.152,
            calibration=True,
        ),
        Param(
            "pixel_size_y_um",
            ParamType.FLOAT,
            "Pixel size Y",
            unit="µm",
            default=0.385,
            calibration=True,
        ),
        Param("samy_direction", ParamType.INT, "samy direction", default=-1),
        Param("colormap", ParamType.STR, "Colormap", default="gray"),
        Param("vmin", ParamType.STR, "vmin", default="", help="colour min (blank = auto)"),
        Param("vmax", ParamType.STR, "vmax", default="", help="colour max (blank = auto)"),
        Param("auto_pct_lo", ParamType.FLOAT, "Auto pct low", default=1.0),
        Param("auto_pct_hi", ParamType.FLOAT, "Auto pct high", default=95.0),
        Param("output_dir", ParamType.DIR, "Output dir"),
    ),
)


@dataclass
class MatchedResult:
    output_dir: str = ""
    layers_dir: str | None = None
    n_strain: int = 0
    n_matched: int = 0
    n_saved: int = 0
    frame_index: int = 0
    vmin: float = 0.0
    vmax: float = 0.0
    max_match_dist_um: float = 0.0
    pngs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Core (faithful port)
# -----------------------------------------------------------------------------
def load_pco_ff_frame(h5_path, pco_ff_path, frame_index):
    """Single frame, per-pixel median background subtracted, negatives -> NaN."""
    with h5py.File(h5_path, "r") as f:
        if pco_ff_path not in f:
            return None
        ds = f[pco_ff_path]
        if ds.ndim == 2:
            return ds[:].astype(np.float64)
        if ds.ndim != 3:
            return None
        idx = min(int(frame_index), ds.shape[0] - 1)
        stack = ds[:].astype(np.float64)
    background = np.nanmedian(stack, axis=0)
    corrected = stack[idx] - background
    corrected[corrected < 0] = np.nan
    return corrected


def match_nearest(strain_samy, strain_samz, rock_samy, rock_samz, threshold_mm):
    """For each strain layer, nearest rocking index within threshold, else None.

    Returns ``(matches, max_matched_dist_mm)``.
    """
    matches = []
    dists = []
    for i in range(len(strain_samy)):
        d = np.sqrt((rock_samy - strain_samy[i]) ** 2 + (rock_samz - strain_samz[i]) ** 2)
        best = int(np.argmin(d))
        if d[best] <= threshold_mm:
            matches.append(best)
            dists.append(float(d[best]))
        else:
            matches.append(None)
    return matches, (max(dists) if dists else 0.0)


def _apply_shift_single(image, shift_px, pad_left, nx_new):
    ny, nx_orig = image.shape
    padded = np.full((ny, nx_new), np.nan, dtype=image.dtype)
    padded[:, pad_left : pad_left + nx_orig] = image
    if abs(shift_px) > 0.01:
        padded = ndi_shift(padded, shift=(0.0, shift_px), order=1, mode="constant", cval=np.nan)
    return padded


def _parse_float(text):
    if text is None or str(text).strip() == "":
        return None
    return float(text)


def _rock_h5(raw_root, name):
    return find_h5_file(os.path.join(raw_root, name))


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> MatchedResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    raw_root = (p["raw_root"] or "").rstrip("/")
    if not raw_root:
        raise ValueError("matched requires 'raw_root'")
    scale_x, scale_y = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    frame_index = int(p["frame_index"])

    out_dir = p["output_dir"] or os.path.join(raw_root, "rocking_matched_layers")
    os.makedirs(out_dir, exist_ok=True)
    result = MatchedResult(output_dir=out_dir, frame_index=frame_index)

    progress(0.05, "reading strain motor positions")
    strain_folders = find_matching_folders(raw_root, p["strain_pattern"])
    strain_samy, strain_samz, _ = extract_motor_positions(
        strain_folders, p["samy_path"], p["samz_path"]
    )
    if len(strain_samy) == 0:
        raise ValueError(f"no strain motor positions for {p['strain_pattern']!r}")

    progress(0.1, "reading rocking motor positions")
    rock_folders = find_matching_folders(raw_root, p["rocking_pattern"])
    rock_samy, rock_samz, rock_names = extract_motor_positions(
        rock_folders, p["samy_path"], p["samz_path"]
    )
    if len(rock_samy) == 0:
        raise ValueError(f"no rocking motor positions for {p['rocking_pattern']!r}")

    matches, max_dist_mm = match_nearest(
        strain_samy, strain_samz, rock_samy, rock_samz, float(p["match_threshold_mm"])
    )
    result.n_strain = len(strain_samy)
    result.n_matched = sum(1 for m in matches if m is not None)
    result.max_match_dist_um = max_dist_mm * 1000.0

    # samy shifts derived from the STRAIN positions (same convention as visualize)
    pad_left = A.compute_pad_left(strain_samy, scale_x, samy_dir)
    pad_right = A.compute_pad_right(strain_samy, scale_x, samy_dir)
    shifts_px = samy_dir * (strain_samy - strain_samy[0]) * 1000.0 / scale_x

    # geometry + colour range from the first valid match
    first = None
    for i, m in enumerate(matches):
        if m is not None:
            first = load_pco_ff_frame(
                _rock_h5(raw_root, rock_names[m]), p["pco_ff_path"], frame_index
            )
            if first is not None:
                break
    if first is None:
        result.skipped.append("no rocking image could be loaded")
        progress(1.0, "no matched layers")
        return result

    ny, nx_orig = first.shape
    nx_new = nx_orig + pad_left + pad_right
    ext_x, ext_y = nx_new * scale_x, ny * scale_y
    z_um = (strain_samz - strain_samz[0]) * 1000.0

    vmin, vmax = _parse_float(p["vmin"]), _parse_float(p["vmax"])
    if vmin is None or vmax is None:
        pooled = []
        for m in (m for m in matches if m is not None):
            if len(pooled) >= 10:
                break
            img = load_pco_ff_frame(
                _rock_h5(raw_root, rock_names[m]), p["pco_ff_path"], frame_index
            )
            if img is not None:
                v = img[np.isfinite(img)]
                if v.size:
                    pooled.append(v)
        if pooled:
            allv = np.concatenate(pooled)
            vmin = float(np.percentile(allv, float(p["auto_pct_lo"]))) if vmin is None else vmin
            vmax = float(np.percentile(allv, float(p["auto_pct_hi"]))) if vmax is None else vmax
        else:
            vmin, vmax = (0.0 if vmin is None else vmin), (1.0 if vmax is None else vmax)
    result.vmin, result.vmax = float(vmin), float(vmax)

    layers_dir = os.path.join(out_dir, "rocking_layers")
    os.makedirs(layers_dir, exist_ok=True)
    result.layers_dir = layers_dir

    for i in range(result.n_strain):
        progress(0.2 + 0.78 * i / result.n_strain, f"matched layer {i}")
        m = matches[i]
        if m is None:
            continue
        img = load_pco_ff_frame(_rock_h5(raw_root, rock_names[m]), p["pco_ff_path"], frame_index)
        if img is None:
            result.skipped.append(f"layer {i}: image load failed")
            continue
        shifted = _apply_shift_single(img, shifts_px[i], pad_left, nx_new)
        title = (
            f"Rocking Curve (frame {frame_index}, median-subtracted)\n"
            f"Z = {z_um[i]:.2f} µm (Layer {i}/{result.n_strain - 1})\n{rock_names[m]}"
        )
        fig, _, _ = Rnd.layer_figure(
            shifted,
            vmin,
            vmax,
            p["colormap"],
            ext_x,
            ext_y,
            title,
            "Intensity − background (a.u.)",
        )
        png = os.path.join(layers_dir, f"layer_{i:04d}.png")
        fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
        result.pngs.append(png)
        result.n_saved += 1

    progress(1.0, f"saved {result.n_saved}/{result.n_strain} matched layers -> {layers_dir}")
    return result


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Plot rocking layers matched to strain positions.")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--strain-pattern", default="*")
    ap.add_argument("--rocking-pattern", default="*")
    ap.add_argument("--frame-index", type=int, default=0)
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args(argv)
    res = run(
        dict(
            raw_root=args.raw_root,
            strain_pattern=args.strain_pattern,
            rocking_pattern=args.rocking_pattern,
            frame_index=args.frame_index,
            output_dir=args.output_dir,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\nsaved {res.n_saved}/{res.n_strain} (matched {res.n_matched}) -> {res.layers_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
