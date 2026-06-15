"""Rocking stage — build aligned 3D volumes from RAW rocking scans.

Faithful port of ``build_aligned_raw_rocking_volumes_v3.py``. For every raw
rocking scan whose ``samz`` falls in the UNION of the mosaicity and strain
``samz`` extremes it:

1. loads the rocking detector frames (ROI-cropped at read time);
2. subtracts a per-pixel background (median across that scan's frames);
3. extracts two 2-D images — the background-subtracted ``sum_intensity`` and a
   single ``specific_frame`` — and stacks them by ``samz``;
4. aligns both volumes to the SAME mosa references as the other stages
   (``samy_reference = mosa_samy[0]`` for the X shift, ``samz_reference =
   mosa_samz[0]`` for the Z origin) via :mod:`dfxm.common.alignment`, so they
   overlay the mosaicity/strain/PVTI volumes in absolute coordinates;
5. saves both aligned volumes + alignment metadata to one HDF5 file (consumed
   by the oblique slicer in Phase 3) and renders per-layer PNGs, a layer
   animation, and a 3-D top-view via :mod:`dfxm.common.render`.

The expensive rotating "slice + spin" 3-D video from the legacy script is not
reproduced (pure eye-candy); the data product and the standard renders are.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np

from ..common import alignment as A
from ..common import render as Rnd
from ..common.errors import StageUserError
from ..common.raster import extract_motor_positions, find_h5_file
from ..common.sort import find_matching_folders
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


STAGE = StageSpec(
    name="rocking",
    label="Aligned rocking volumes",
    description=(
        "Builds aligned 3-D volumes directly from the raw rocking scans — a background-subtracted "
        "intensity sum plus one chosen frame — anchored to the mosaicity reference so they overlay "
        "the other volumes. Writes aligned_raw_rocking_volumes.h5 and rendered images."
    ),
    params=(
        Param(
            "raw_root",
            ParamType.DIR,
            "Raw data root",
            must_exist=True,
            help="RAW_DATA root containing the rocking (and mosaicity/strain) scan folders.",
        ),
        Param(
            "rocking_pattern",
            ParamType.STR,
            "Rocking pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help="Glob matching the raw rocking scan folders.",
        ),
        Param(
            "mosa_pattern",
            ParamType.STR,
            "Mosaicity pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob for the mosaicity scan folders — they provide the alignment reference "
                "(samy/samz origin) and part of the Z range."
            ),
        ),
        Param(
            "strain_pattern",
            ParamType.STR,
            "Strain pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob for the strain scan folders; extends the Z range so the rocking volume "
                "covers both (blank = mosaicity range only)."
            ),
        ),
        Param(
            "samy_path",
            ParamType.STR,
            "samy path",
            default="1.1/instrument/positioners/samy",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path to the sample-Y motor position inside each scan file "
                "(under the first BLISS entry). Only change for a different beamline file layout."
            ),
        ),
        Param(
            "samz_path",
            ParamType.STR,
            "samz path",
            default="1.1/instrument/positioners/samz",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path to the sample-Z motor position inside each scan file "
                "(under the first BLISS entry). Only change for a different beamline file layout."
            ),
        ),
        Param(
            "detector_path",
            ParamType.STR,
            "Detector path",
            default="1.1/measurement/pco_ff",
            advanced=True,
            group="Data layout",
            help="HDF5 path to the detector frames inside each rocking scan file.",
        ),
        Param(
            "pixel_size_x_um",
            ParamType.FLOAT,
            "Pixel size X",
            unit="µm",
            default=0.152,
            calibration=True,
            advanced=True,
            group="Calibration",
            help=(
                "Physical size of one detector pixel along X, in µm. "
                "From the beamline optics calibration."
            ),
        ),
        Param(
            "pixel_size_y_um",
            ParamType.FLOAT,
            "Pixel size Y",
            unit="µm",
            default=0.385,
            calibration=True,
            advanced=True,
            group="Calibration",
            help=(
                "Physical size of one detector pixel along Y, in µm. "
                "From the beamline optics calibration."
            ),
        ),
        Param(
            "samy_direction",
            ParamType.INT,
            "samy direction",
            default=-1,
            advanced=True,
            group="Alignment",
            help=(
                "Sign (+1 or −1) relating the samy motor direction to detector X. "
                "If features visibly march the wrong way between layers, flip the sign."
            ),
        ),
        Param(
            "roi_x",
            ParamType.STR,
            "ROI X",
            default="",
            help=(
                "Detector crop 'x0,x1' in pixels applied while reading frames (blank = full). "
                "Match the crop used for the other volumes."
            ),
        ),
        Param(
            "roi_y",
            ParamType.STR,
            "ROI Y",
            default="",
            help=(
                "Detector crop 'y0,y1' in pixels applied while reading frames (blank = full). "
                "Match the crop used for the other volumes."
            ),
        ),
        Param(
            "specific_frame_idx",
            ParamType.STR,
            "Specific frame",
            default="",
            help=(
                "0-based index of the single rocking frame to extract per scan "
                "(blank = the central frame of the first scan). "
                "Lets you look at one angular position instead of the sum."
            ),
        ),
        Param(
            "samz_tol_mm",
            ParamType.FLOAT,
            "samz tolerance",
            unit="mm",
            default=0.0,
            advanced=True,
            group="Alignment",
            help=(
                "Extra tolerance in mm when deciding which rocking scans fall inside "
                "the mosaicity/strain Z range."
            ),
        ),
        Param(
            "normalize_sum",
            ParamType.BOOL,
            "Normalize sum",
            default=False,
            advanced=True,
            group="Alignment",
            help=(
                "Divide each summed image by its frame count so intensities are comparable "
                "across scans with different numbers of frames."
            ),
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help=(
                "Where the aligned volume and rendered media are written "
                "(blank = a folder under the raw root)."
            ),
        ),
        Param(
            "aligned_h5_name",
            ParamType.STR,
            "Aligned filename",
            default="aligned_raw_rocking_volumes.h5",
            advanced=True,
            group="Output",
            help=(
                "Filename of the aligned rocking volume. "
                "The slices stage expects aligned_raw_rocking_volumes.h5."
            ),
        ),
        Param(
            "save_aligned_h5",
            ParamType.BOOL,
            "Save aligned HDF5",
            default=True,
            advanced=True,
            group="Output",
            help="Write the aligned volume file (needed by the slices stage).",
        ),
        Param(
            "save_layers",
            ParamType.BOOL,
            "Save layer PNGs",
            default=True,
            advanced=True,
            group="Output",
            help="Write one PNG per layer.",
        ),
        Param(
            "save_animation",
            ParamType.BOOL,
            "Save animation",
            default=True,
            advanced=True,
            group="Output",
            help="Write the layer-by-layer animation.",
        ),
        Param(
            "save_topview",
            ParamType.BOOL,
            "Save 3D top-view",
            default=True,
            advanced=True,
            group="Output",
            help="Write the static 3-D top-view image.",
        ),
        Param(
            "output_format",
            ParamType.ENUM,
            "Animation format",
            default="mp4",
            choices=("mp4", "gif", "both"),
            advanced=True,
            group="Output",
            help=(
                "Animation container: mp4 needs ffmpeg on PATH; gif always works; both writes both."
            ),
        ),
        Param(
            "volume_opacity",
            ParamType.FLOAT,
            "3D opacity",
            default=0.85,
            advanced=True,
            group="Appearance",
            help="Opacity of the rendered 3-D top view, 0–1.",
        ),
        Param(
            "cbar_pct_lo",
            ParamType.FLOAT,
            "Colorbar pct low",
            default=1.0,
            advanced=True,
            group="Appearance",
            help="Lower intensity percentile for the colour scale of the rendered images.",
        ),
        Param(
            "cbar_pct_hi",
            ParamType.FLOAT,
            "Colorbar pct high",
            default=99.0,
            advanced=True,
            group="Appearance",
            help="Upper intensity percentile for the colour scale of the rendered images.",
        ),
    ),
)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
@dataclass
class RockingProducts:
    name: str
    vmin: float
    vmax: float
    layers_dir: str | None = None
    animation: str | None = None
    top_view: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class RockingResult:
    output_dir: str = ""
    aligned_path: str | None = None
    volume_shape: tuple[int, int, int] | None = None
    n_layers_used: int = 0
    specific_frame_idx: int | None = None
    samy_reference_mm: float | None = None
    samz_reference_mm: float | None = None
    z_span_um: float = 0.0
    datasets: list[RockingProducts] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Raw-scan processing (faithful port)
# -----------------------------------------------------------------------------
def process_raw_scan(
    h5_path: str,
    detector_path: str,
    roi_x: tuple | None,
    roi_y: tuple | None,
    specific_frame_idx: int | None,
    normalize_sum: bool,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Background-subtract one rocking scan; return (sum_2d, specific_2d, n_frames, idx)."""
    with h5py.File(h5_path, "r") as f:
        det = f[detector_path]
        n_frames = det.shape[0]
        h_full, w_full = det.shape[1], det.shape[2]
        ys = roi_y[0] if roi_y else 0
        ye = roi_y[1] if roi_y else h_full
        xs = roi_x[0] if roi_x else 0
        xe = roi_x[1] if roi_x else w_full
        frames = det[:, ys:ye, xs:xe].astype(np.float32)

    # Resolve + copy the specific frame BEFORE the in-place median reorder.
    if specific_frame_idx is None:
        idx = n_frames // 2
    else:
        idx = int(specific_frame_idx)
        if idx < 0 or idx >= n_frames:
            idx = n_frames // 2
    raw_specific = frames[idx].copy()

    # Per-pixel background = median across the rocking dimension (in-place OK:
    # the SUM is order-independent and the specific frame was already copied).
    background = np.median(frames, axis=0, overwrite_input=True).astype(np.float32)
    frames -= background[np.newaxis, :, :]

    sum_2d = frames.sum(axis=0)
    if normalize_sum:
        sum_2d = sum_2d / max(1, n_frames)
    specific_2d = raw_specific - background

    del frames, raw_specific
    return sum_2d, specific_2d, n_frames, idx


def build_raw_volumes(
    paths: list,
    samy_arr: np.ndarray,
    samz_arr: np.ndarray,
    detector_path: str,
    roi_x: tuple | None,
    roi_y: tuple | None,
    specific_frame_idx: int | None,
    normalize_sum: bool,
    progress: ProgressFn = _noop,
):
    """Process each scan (caller pre-sorts by samz) and stack into two 3-D volumes.

    Returns ``(sum_vol, spec_vol, samy_used, samz_used, names_used, spec_idx)`` or
    all-``None`` if nothing processed.
    """
    sum_slices, spec_slices = [], []
    samy_used, samz_used, names_used = [], [], []
    used_spec_idx = None

    n = len(paths)
    for i, folder in enumerate(paths):
        progress(i / max(1, n), f"rocking: {os.path.basename(folder)}")
        h5p = find_h5_file(folder)
        if h5p is None:
            continue
        try:
            sum_2d, spec_2d, _nf, spec_idx = process_raw_scan(
                h5p, detector_path, roi_x, roi_y, specific_frame_idx, normalize_sum
            )
        except (KeyError, OSError, ValueError):
            continue
        sum_slices.append(sum_2d)
        spec_slices.append(spec_2d)
        samy_used.append(samy_arr[i])
        samz_used.append(samz_arr[i])
        names_used.append(os.path.basename(folder))
        if used_spec_idx is None:
            used_spec_idx = spec_idx

    if not sum_slices:
        return None, None, None, None, None, None

    return (
        np.stack(sum_slices, axis=0),
        np.stack(spec_slices, axis=0),
        np.array(samy_used),
        np.array(samz_used),
        names_used,
        used_spec_idx,
    )


def save_aligned_raw_volumes(
    output_path: str,
    sum_aligned: np.ndarray,
    spec_aligned: np.ndarray,
    z_uniform_um: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    samy_direction: int,
    samy_ref_mm: float,
    samz_ref_mm: float,
    samy_used_mm: np.ndarray,
    samz_used_mm: np.ndarray,
    names_used: list,
    roi_x: tuple | None,
    roi_y: tuple | None,
    specific_frame_idx: int,
    normalize_sum: bool,
    pad_left_px: int,
    pad_right_px: int,
) -> None:
    """Save both aligned volumes + alignment metadata (oblique-slicer schema)."""
    with h5py.File(output_path, "w") as f:
        f.create_dataset(
            "sum_intensity",
            data=sum_aligned.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset(
            "specific_frame",
            data=spec_aligned.astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        f.create_dataset("z_uniform_um", data=z_uniform_um.astype(np.float32))
        f.create_dataset("raw_samy_mm", data=samy_used_mm.astype(np.float64))
        f.create_dataset("raw_samz_mm", data=samz_used_mm.astype(np.float64))
        f.create_dataset("folder_names", data=np.array(names_used, dtype="S256"))

        f.attrs["samy_reference_mm"] = float(samy_ref_mm)
        f.attrs["samz_reference_mm"] = float(samz_ref_mm)
        f.attrs["scale_x_um_per_px"] = float(scale_x)
        f.attrs["scale_y_um_per_px"] = float(scale_y)
        f.attrs["scale_z_um_per_px"] = float(scale_z)
        f.attrs["samy_direction"] = int(samy_direction)
        f.attrs["specific_frame_idx"] = int(specific_frame_idx)
        f.attrs["normalize_sum"] = int(bool(normalize_sum))
        f.attrs["pad_left_px"] = int(pad_left_px)
        f.attrs["pad_right_px"] = int(pad_right_px)
        if roi_x is not None:
            f.attrs["roi_x_start"], f.attrs["roi_x_end"] = int(roi_x[0]), int(roi_x[1])
        if roi_y is not None:
            f.attrs["roi_y_start"], f.attrs["roi_y_end"] = int(roi_y[0]), int(roi_y[1])


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_pair(text) -> tuple | None:
    if text is None or str(text).strip() == "":
        return None
    parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'a,b', got {text!r}")
    return tuple(parts)


def _parse_opt_int(text) -> int | None:
    if text is None or str(text).strip() == "":
        return None
    return int(text)


def _colorbar_range(data: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(valid, lo)), float(np.percentile(valid, hi)))


def _motors(raw_root: str, pattern: str, samy_path: str, samz_path: str):
    if not raw_root or not pattern:
        return np.array([]), np.array([]), []
    folders = find_matching_folders(raw_root, pattern)
    if not folders:
        return np.array([]), np.array([]), []
    return extract_motor_positions(folders, samy_path, samz_path)


def _render(result: RockingResult, vol, z_um, scale_z, name, p, out_dir, cmap, title, cbar):
    sx, sy = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    vmin, vmax = _colorbar_range(vol, float(p["cbar_pct_lo"]), float(p["cbar_pct_hi"]))
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)
    prod = RockingProducts(name=name, vmin=vmin, vmax=vmax)
    if p["save_layers"]:
        prod.layers_dir = Rnd.save_layer_pngs(
            vol, z_um, ds_dir, name, vmin, vmax, cmap, title, cbar, sx, sy
        )
    if p["save_animation"]:
        prod.animation = Rnd.save_layer_animation(
            vol,
            z_um,
            os.path.join(ds_dir, f"{name}_layer_anim"),
            name,
            vmin,
            vmax,
            cmap,
            title,
            cbar,
            p["output_format"],
            sx,
            sy,
        )
    if p["save_topview"]:
        try:
            prod.top_view = Rnd.save_top_view(
                vol,
                scale_z,
                sx,
                sy,
                vmin,
                vmax,
                cmap,
                float(p["volume_opacity"]),
                os.path.join(ds_dir, f"{name}_top_view.png"),
            )
        except Exception as exc:  # noqa: BLE001 - no GL -> note + continue
            prod.notes.append(f"3D top-view skipped: {exc}")
    result.datasets.append(prod)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> RockingResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    raw_root = (p["raw_root"] or "").rstrip("/")
    if not raw_root:
        raise StageUserError(
            "rocking requires 'raw_root'",
            hint="Set 'Raw data root' to the RAW_DATA folder that contains the scan folders.",
        )
    scale_x = float(p["pixel_size_x_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["roi_x"]), _parse_pair(p["roi_y"])
    spec_cfg = _parse_opt_int(p["specific_frame_idx"])
    tol = float(p["samz_tol_mm"])

    out_dir = p["output_dir"] or os.path.join(raw_root, "aligned_raw_rocking_volumes")
    result = RockingResult(output_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. mosa reference (anchors X shift + Z origin) + samz union start
    progress(0.02, "reading mosaicity motor positions")
    mosa_samy, mosa_samz, _ = _motors(raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"])
    if len(mosa_samy) == 0:
        raise StageUserError(
            "rocking needs the mosaicity reference; no mosa motor positions found",
            hint=(
                "Check 'Mosaicity pattern' — rocking anchors its alignment "
                "to the mosaicity scans' samy/samz positions."
            ),
        )
    samy_ref, samz_ref = float(mosa_samy[0]), float(mosa_samz[0])
    result.samy_reference_mm, result.samz_reference_mm = samy_ref, samz_ref

    # 2. samz union range (mosa ∪ strain)
    _, strain_samz, _ = _motors(raw_root, p["strain_pattern"], p["samy_path"], p["samz_path"])
    all_samz = np.concatenate([mosa_samz, strain_samz]) if len(strain_samz) else mosa_samz
    z_min, z_max = float(all_samz.min()), float(all_samz.max())

    # 3. rocking scans within the samz union range, sorted by samz
    progress(0.06, "reading rocking motor positions")
    rock_samy, rock_samz, rock_names = _motors(
        raw_root, p["rocking_pattern"], p["samy_path"], p["samz_path"]
    )
    if len(rock_names) == 0:
        raise StageUserError(
            f"no rocking folders matching {p['rocking_pattern']!r} in {raw_root}",
            hint="Check 'Rocking pattern' against the scan folder names under the raw root.",
        )
    rock_paths = [os.path.join(raw_root, n) for n in rock_names]

    mask = (rock_samz >= z_min - tol) & (rock_samz <= z_max + tol)
    keep_paths = [pp for pp, m in zip(rock_paths, mask) if m]
    keep_samy, keep_samz = rock_samy[mask], rock_samz[mask]
    if not keep_paths:
        raise StageUserError(
            f"no rocking scans fall in samz union [{z_min:.6f}, {z_max:.6f}] mm (tol={tol})",
            hint=(
                "Loosen 'samz tolerance' or check that the rocking scans "
                "cover the mosaicity/strain Z range."
            ),
        )
    order = np.argsort(keep_samz)
    keep_paths = [keep_paths[i] for i in order]
    keep_samy, keep_samz = keep_samy[order], keep_samz[order]

    # 4. build the two raw volumes (background-subtracted)
    progress(0.1, f"processing {len(keep_paths)} rocking scans")
    sum_vol, spec_vol, samy_used, samz_used, names_used, spec_idx = build_raw_volumes(
        keep_paths,
        keep_samy,
        keep_samz,
        p["detector_path"],
        roi_x,
        roi_y,
        spec_cfg,
        bool(p["normalize_sum"]),
        progress=lambda fr, m: progress(0.1 + 0.5 * fr, m),
    )
    if sum_vol is None:
        result.skipped.append("no rocking scans processed successfully")
        progress(1.0, "no rocking volumes produced")
        return result
    result.specific_frame_idx = spec_idx
    result.n_layers_used = len(names_used)

    # 5. align (mosa-anchored samy shift + Z interpolation)
    progress(0.65, "aligning (samy shift + Z interpolation)")
    sum_aligned = A.apply_samy_shifts_to_volume(
        sum_vol, samy_used, scale_x, samy_dir, ref_samy=samy_ref
    )
    spec_aligned = A.apply_samy_shifts_to_volume(
        spec_vol, samy_used, scale_x, samy_dir, ref_samy=samy_ref
    )
    pad_left = A.compute_pad_left(samy_used, scale_x, samy_dir, ref_samy=samy_ref)
    pad_right = A.compute_pad_right(samy_used, scale_x, samy_dir, ref_samy=samy_ref)
    del sum_vol, spec_vol
    sum_aligned, z_uniform, scale_z = A.interpolate_to_uniform_z(
        sum_aligned, samz_used, ref_samz=samz_ref
    )
    spec_aligned, _, _ = A.interpolate_to_uniform_z(spec_aligned, samz_used, ref_samz=samz_ref)
    result.volume_shape = tuple(sum_aligned.shape)
    result.z_span_um = float(z_uniform[-1] - z_uniform[0]) if len(z_uniform) else 0.0

    # 6. save aligned HDF5 (feeds the oblique slicer)
    if p["save_aligned_h5"]:
        aligned_path = os.path.join(out_dir, p["aligned_h5_name"])
        save_aligned_raw_volumes(
            aligned_path,
            sum_aligned,
            spec_aligned,
            z_uniform,
            scale_x=scale_x,
            scale_y=float(p["pixel_size_y_um"]),
            scale_z=scale_z,
            samy_direction=samy_dir,
            samy_ref_mm=samy_ref,
            samz_ref_mm=samz_ref,
            samy_used_mm=samy_used,
            samz_used_mm=samz_used,
            names_used=names_used,
            roi_x=roi_x,
            roi_y=roi_y,
            specific_frame_idx=spec_idx,
            normalize_sum=bool(p["normalize_sum"]),
            pad_left_px=pad_left,
            pad_right_px=pad_right,
        )
        result.aligned_path = aligned_path

    # 7. render
    progress(0.8, "rendering volumes")
    sum_tag = "(a.u., normalized)" if p["normalize_sum"] else "(a.u.)"
    _render(
        result,
        sum_aligned,
        z_uniform,
        scale_z,
        "raw_sum_intensity",
        p,
        out_dir,
        "magma",
        "Background-subtracted Sum Intensity",
        f"Sum intensity {sum_tag}",
    )
    _render(
        result,
        spec_aligned,
        z_uniform,
        scale_z,
        f"raw_specific_frame_{spec_idx:03d}",
        p,
        out_dir,
        "magma",
        f"Background-subtracted Frame {spec_idx}",
        "Intensity (a.u.)",
    )

    progress(1.0, f"aligned {result.n_layers_used} rocking layers -> {out_dir}")
    return result


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Build aligned raw rocking volumes.")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--rocking-pattern", default="*")
    ap.add_argument("--mosa-pattern", default="*")
    ap.add_argument("--strain-pattern", default="*")
    ap.add_argument("--roi-x", default="")
    ap.add_argument("--roi-y", default="")
    ap.add_argument("--no-media", action="store_true")
    args = ap.parse_args(argv)
    res = run(
        dict(
            raw_root=args.raw_root,
            rocking_pattern=args.rocking_pattern,
            mosa_pattern=args.mosa_pattern,
            strain_pattern=args.strain_pattern,
            roi_x=args.roi_x,
            roi_y=args.roi_y,
            save_layers=not args.no_media,
            save_animation=not args.no_media,
            save_topview=not args.no_media,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\naligned -> {res.aligned_path}; shape {res.volume_shape}; skipped {len(res.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
