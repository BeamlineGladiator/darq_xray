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
from ..common import progress as _progress_mod
from ..common import render as Rnd
from ..common import render3d as R3
from ..common.errors import StageUserError
from ..common.figures import (
    FigureSpec,
    ReplotGroup,
    register,
    render_volume_layer,
    resolve_clim,
    volume_layer_specs,
)
from ..common.h5io import resolve_input_file
from ..common.plotting import apply_round_clim, resolve_cmap, style_from_params
from ..common.raster import extract_motor_positions, find_h5_file
from ..common.sort import find_matching_folders
from ..config.models import CostEstimate, Param, ParamType, SeeAlso, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


def _sum_title(source: str) -> str:
    return (
        "Mosa-integrated Sum Intensity"
        if source == "mosaicity"
        else ("Background-subtracted Sum Intensity")
    )


def _spec_title(source: str, idx: int) -> str:
    base = "Mosa-integrated Frame" if source == "mosaicity" else "Background-subtracted Frame"
    return f"{base} {idx}"


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
            "source_scan",
            ParamType.ENUM,
            "Source scan",
            default="rocking",
            choices=("rocking", "mosaicity"),
            advanced=True,
            group="Data layout",
            help=(
                "Which scans' detector frames are summed into the raw volume. 'rocking' uses the "
                "rocking scans (within the mosa/strain Z range); 'mosaicity' sums each mosa scan's "
                "frames — one mosa folder per layer — as a DFXM topograph. Run once per source to "
                "build both."
            ),
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
                "Physical size of one detector pixel along X, in µm, from the beamline optics "
                "calibration. This is what converts the sample-Y motor shift (mm) into detector "
                "pixels during alignment, so a wrong value misaligns layers along X as well as "
                "scaling every reported distance."
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
                "Physical size of one detector pixel along Y, in µm, from the beamline optics "
                "calibration. A wrong value skews the vertical physical scale of every rendered "
                "image and volume."
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
            roi_frame="detector",
            help=(
                "Detector crop 'x0,x1' — START and END pixel columns on the raw detector, "
                "applied while reading frames (blank = full). Careful: darfix shows its ROI "
                "as origin+size — end = origin + size, not the size itself. Must cover the "
                "same detector window as the other volumes or they misregister. "
                "Pre-filled from the experiment's darfix + analysis ROIs — normally leave as-is."
            ),
        ),
        Param(
            "roi_y",
            ParamType.STR,
            "ROI Y",
            default="",
            roi_frame="detector",
            help=(
                "Detector crop 'y0,y1' — START and END pixel rows on the raw detector, "
                "applied while reading frames (blank = full). Careful: darfix shows its ROI "
                "as origin+size — end = origin + size, not the size itself. Must cover the "
                "same detector window as the other volumes or they misregister along Y. "
                "Pre-filled from the experiment's darfix + analysis ROIs — normally leave as-is."
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
            "subtract_background",
            ParamType.BOOL,
            "Subtract background",
            default=True,
            advanced=True,
            group="Alignment",
            help=(
                "Before summing, compute each pixel's median across the scan's frames and "
                "subtract it, so only above-background diffraction signal accumulates. "
                "Applies to whichever scan type the run reads (rocking or mosaicity source). "
                "Keep on for the standard rocking sum; turn off for a plain intensity sum, "
                "e.g. a mosa-scan topograph where the background level itself is meaningful."
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
            "volume_downsample",
            ParamType.INT,
            "3D downsample",
            default=0,
            advanced=True,
            group="Appearance",
            advice_key="3d_texture",
            help=(
                "Coarsening applied to the volume before 3-D rendering, as a "
                "block-average over N x N pixels in Y and X (Z is untouched). "
                "0 (the default) picks the smallest factor that fits this "
                "machine's GL 3-D texture limit, which a volume wider than that "
                "limit needs to render at all; 1 never coarsens, and renders "
                "BLANK when the volume is over the limit; N applies exactly N."
            ),
        ),
        Param(
            "cbar_pct_lo",
            ParamType.FLOAT,
            "Colorbar pct low",
            unit="%",
            default=1.0,
            advanced=True,
            group="Appearance",
            help="Lower intensity percentile for the colour scale of the rendered images.",
        ),
        Param(
            "cbar_pct_hi",
            ParamType.FLOAT,
            "Colorbar pct high",
            unit="%",
            default=99.0,
            advanced=True,
            group="Appearance",
            help="Upper intensity percentile for the colour scale of the rendered images.",
        ),
    ),
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” "
            "(left panel); the range fields in Advanced below are this stage's own.",
        ),
    ),
    estimate="dfxm.stages.rocking:estimate",
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
    subtract_background: bool = True,
    progress=None,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Read one scan; return (sum_2d, specific_2d, n_frames, idx).

    With ``subtract_background`` (default) a per-pixel median across the scan's
    frames is removed before summing (rocking behaviour); otherwise a plain sum
    and the raw specific frame are returned (mosa-topograph behaviour).
    """
    report = progress or _progress_mod.noop
    with h5py.File(h5_path, "r") as f:
        det = f[detector_path]
        n_frames = det.shape[0]
        h_full, w_full = det.shape[1], det.shape[2]
        ys = roi_y[0] if roi_y else 0
        ye = roi_y[1] if roi_y else h_full
        xs = roi_x[0] if roi_x else 0
        xe = roi_x[1] if roi_x else w_full
        frames = det[:, ys:ye, xs:xe].astype(np.float32)
    # There is no frame loop to report from — the read, the median and the sum
    # are each one vectorised whole-scan operation — so the boundaries between
    # them are the only honest places to speak, and they are also where the
    # time actually goes.
    report(0.4, "read frames")

    if specific_frame_idx is None:
        idx = n_frames // 2
    else:
        idx = int(specific_frame_idx)
        if idx < 0 or idx >= n_frames:
            idx = n_frames // 2
    raw_specific = frames[idx].copy()

    if subtract_background:
        background = np.median(frames, axis=0, overwrite_input=True).astype(np.float32)
        frames -= background[np.newaxis, :, :]
        specific_2d = raw_specific - background
    else:
        specific_2d = raw_specific
    report(0.75, "background subtracted" if subtract_background else "frames prepared")

    sum_2d = frames.sum(axis=0)
    if normalize_sum:
        sum_2d = sum_2d / max(1, n_frames)

    del frames, raw_specific
    report(1.0, "summed")
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
    subtract_background: bool = True,
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
            scan_lo, scan_hi = _progress_mod.slice_for(i, n, 0.0, 1.0)
            sum_2d, spec_2d, _nf, spec_idx = process_raw_scan(
                h5p,
                detector_path,
                roi_x,
                roi_y,
                specific_frame_idx,
                normalize_sum,
                subtract_background,
                progress=_progress_mod.sub_progress(
                    lambda fr, m: progress(fr, f"{os.path.basename(folder)}: {m}"),
                    scan_lo,
                    scan_hi,
                ),
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


def _roi_span(pair: tuple | None, full: int) -> int:
    """Rows/columns an ROI selects out of *full*, the way ``det[:, a:b]`` would.

    Clamped to ``[0, full]`` and never above *full*, so a stale ROI from another
    dataset cannot make an estimate larger than the detector it is sizing. A
    blank or inverted ROI means the whole axis, matching
    :func:`process_raw_scan`'s ``roi_x[0] if roi_x else 0`` defaults.
    """
    if not pair:
        return int(full)
    lo = max(0, min(int(pair[0]), int(full)))
    hi = max(0, min(int(pair[1]), int(full)))
    return (hi - lo) if hi > lo else int(full)


def _parse_opt_int(text) -> int | None:
    if text is None or str(text).strip() == "":
        return None
    return int(text)


def _colorbar_range(data: np.ndarray, lo: float, hi: float) -> tuple[float, float]:
    """Percentile colour limits for an array the caller already holds.

    Kept in its in-core form on purpose: every caller but the cold replot
    reaches it with the array in hand (`_render` has just built the volume), so
    streaming there would re-read a volume that is already resident. The replot
    path, which held a whole volume purely to take these two numbers, goes
    through :func:`_replot_default_clim` instead — and the two must agree
    exactly, which is what `stream_quantile`'s numpy parity buys.
    """
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(valid, lo)), float(np.percentile(valid, hi)))


# What one block of the volume may cost, per element of it, while
# `volumeio.stream_quantile` runs: the block itself in its stored dtype, plus
# the reductions' own per-element temporaries — `_finite64`'s `isfinite` mask
# (1 B) and float64 copy (8 B), the rank search's `window` (8 B) and its
# `searchsorted` / `- 1` / `clip` index arrays (3 x 8 B). That is the same
# `dtype.itemsize + 8 * (retained + 1) + 1` = 41 B/element accounting
# `alignment.align_volume_streamed` prices its own cached median at, and that
# `slices._aligned_block_budget` uses; it is shared rather than re-invented.
QUANTILE_WORKING_SET_PER_ELEMENT = 41

# The working set `_replot_default_clim` may hold for one block. A fixed
# constant rather than `advice.working_set_budget_bytes`, deliberately: the
# quantile is exact at every budget, so a larger machine would buy nothing but a
# larger block, and a fixed number is one less thing that can differ between two
# machines rendering the same replot. It is not a floor either — the reduction's
# blocking-independent scaffold (`volumeio.centring_scaffold_bytes("median",
# ...)`, ~26 MB at its worst case) sits on top of it and no block size pays it
# off.
REPLOT_CLIM_WORKING_SET_BYTES = 64 * 1024 * 1024


def _clim_block_budget(dataset, budget_bytes: int) -> int:
    """*budget_bytes* of working set, in the block bytes ``iter_blocks`` counts.

    ``iter_blocks`` sizes a block by its bytes **in the stored dtype**, while
    the streamed percentile holds :data:`QUANTILE_WORKING_SET_PER_ELEMENT` more
    per element on top of it. Handing the budget over raw would buy a block
    ~11x too large for a float32 volume. Integer division rounds the budget
    **down**, which is the safe direction: a smaller block can only cost less
    than counted.
    """
    itemsize = max(1, int(np.dtype(dataset.dtype).itemsize))
    return max(1, int(budget_bytes) * itemsize // (itemsize + QUANTILE_WORKING_SET_PER_ELEMENT))


def _fits_in_core(dataset, budget_bytes: int) -> bool:
    """Whether the whole-volume percentile fits the same working-set budget.

    The in-core form costs, per element: the resident volume (``itemsize``), the
    ``data[np.isfinite(data)]`` selection (at most ``itemsize``, less when
    values are non-finite), ``np.percentile``'s internal partition copy (at most
    ``itemsize``), and the boolean ``isfinite`` mask (1 B) — so
    ``3 * itemsize + 1`` bounds it. Measured with ``tracemalloc`` above the
    resident volume: 12.04 B/element for float32 (bound 13), 24.03 for float64
    (bound 25), 6.04 for uint16 (bound 7); the bound holds in all three, with
    about 8% of margin — not the ~17% an earlier, superseded set of figures
    from a smaller fixture implied. The test named after this function
    re-measures them, so they are checked rather than remembered.

    That is ~3.5x cheaper per element than a streamed block, which is the whole
    point of asking: streaming costs ~12 traversals against 1, and on this
    stage's replot path a traversal is a raw HDF5 read.
    """
    itemsize = max(1, int(np.dtype(dataset.dtype).itemsize))
    elements = 1
    for dim in dataset.shape:
        elements *= int(dim)
    return elements * (3 * itemsize + 1) <= int(budget_bytes)


def _motors(raw_root: str, pattern: str, samy_path: str, samz_path: str):
    if not raw_root or not pattern:
        return np.array([]), np.array([]), []
    folders = find_matching_folders(raw_root, pattern)
    if not folders:
        return np.array([]), np.array([]), []
    return extract_motor_positions(folders, samy_path, samz_path)


# --- Recalibrated 2026-08-26 against measured child peak RSS ------------------
#
# The model this replaced was wrong in BOTH directions, which is why no single
# correction factor would have fixed it:
#
# * it charged 20 B per element per scan for the accumulate-stack-align-write
#   chain that measures 48.5, and charged nothing at all for `save_topview`, so
#   on scans with few frames it **under-predicted** — 13.4x on a synthetic
#   512x512, 21-frame, 6-scan run with the default toggles, the dangerous
#   direction;
# * it sized the per-scan read from the whole detector when `process_raw_scan`
#   reads only the ROI, so on scans with many frames that inflated term
#   dominated and it **over-predicted** — 3.2x on the real STO2 form (575-frame
#   mosa scans, a 2048x2048 detector cropped to 700x1832).
#
# Measured on this box with `tests/peak_rss.py`: five scan shapes at six scans,
# scan counts 3/6/12/24 at one shape, frame counts 21/51/101/201 at one shape,
# the render toggles at four volume sizes, and then ten real STO2 layers. A
# two-regressor fit over the first three sweeps reached r2 = 0.997. Tables in
# `docs/Codebase.md`.

# The process image of a rocking child. Measured 154.3 MiB
# (`measure_process_floor`), charged with slack. `save_layers` (+34 MiB, for
# matplotlib) and `save_animation` (+8 MiB) are folded in here rather than made
# conditional: both are small, and over-stating a toggled-off product is safe.
ROCKING_PROCESS_FLOOR_BYTES = 176 * 1024 * 1024

# Per element of the ALIGNED volume: `sum_slices`/`spec_slices` and the
# `np.stack` copies of each, the alignment's padded canvas and uniform-Z
# resample, the aligned-h5 write, and the per-layer render. Measured **48.5 B**
# by least squares over the scan-size and scan-count sweeps (r2 0.997).
#
# **Known looseness, measured rather than suspected.** Those sweeps used 21-frame
# scans, where this term is what the whole peak is made of. On the real STO2
# form the scans carry 575 frames, the per-scan read dominates instead, and the
# volume's own marginal comes out at about **11 B per aligned element** — so the
# model over-predicts that run by 1.9x (11.168 GiB against a measured 5.841).
# 48 is kept because it is what the small-frame regime measures and a model must
# cover both; tightening it means re-measuring across frame counts on real data,
# not splitting the difference. The two regimes and their figures:
#
#   synthetic, 21 frames, 6-24 scans  ->  1.14-1.56x over
#   real STO2, 575 frames, 10 layers  ->  1.14x over
#   real STO2, 575 frames, 76 layers  ->  1.91x over
ROCKING_VOLUME_BYTES_PER_ELEM = 48

# What `save_topview` adds: the pyvista/VTK import and a render context. It is
# **data-independent** — measured at 361.4 / 373.9 / 354.4 / 353.5 / 375.0 MiB
# across a 16x span of volume sizes — and on a small-ROI run it is the single
# largest term, which is exactly what the old model missed. (Measured
# with EGL unavailable on this box, so this is a software-GL figure; a hardware
# context would be no larger.)
ROCKING_TOPVIEW_BYTES = 416 * 1024 * 1024


def estimate(params: dict) -> CostEstimate:
    """Peak memory for a rocking run, from HDF5 shapes only.

    ``run()`` streams scans one at a time: ``build_raw_volumes`` ->
    ``process_raw_scan`` reads one scan's detector stack in its stored dtype and
    immediately ``.astype(np.float32)``\\ s it (source and float32 copy coexist),
    then ``del frames`` drops it before the next scan. What *accumulates* is the
    two lists of 2-D slices, which become the ``(n_layers, H, W)`` volumes, which
    the alignment then pads and resamples. The model is the sum of three
    measured terms plus a conditional fourth:

    * :data:`ROCKING_PROCESS_FLOOR_BYTES`;
    * one scan's read, ``scan_elems * (itemsize + 4)``. Measured: this term only
      bites above ~50 frames per scan, where its local slope is **5.85 B per
      scan element** against the 6 charged — but a real rocking scan can carry
      hundreds of frames, so it stays in the model;
    * :data:`ROCKING_VOLUME_BYTES_PER_ELEM` ``* n_scans * layer_elems``, which is
      what actually dominates at ordinary frame counts;
    * :data:`ROCKING_TOPVIEW_BYTES` when ``save_topview`` is on — **default on,
      ~365 MiB, and entirely absent from the old model**, which is why that
      model under-predicted a few-frame default run by 13.4x.

    ``layer_elems`` and ``scan_elems`` are both sized from the **ROI'd** frame,
    not the detector: ``process_raw_scan`` reads ``det[:, ys:ye, xs:xe]``, so
    the float32 copy, both accumulators and the aligned volumes follow
    ``roi_x``/``roi_y``. On the real STO2 form (``roi_x`` 105,1937 and ``roi_y``
    630,1330 against a 2048x2048 detector) ignoring the ROI over-stated the
    priced work by 3.3x, and on 575-frame scans that inflated read term was
    large enough to hide the two under-charges above. ``input_bytes``
    deliberately keeps the un-cropped figure — that is the data on disk, which
    the ROI does not change.

    ``chunkable=True``.

    The folder glob **follows ``source_scan``**, as ``run()`` does:
    ``"mosaicity"`` processes the mosa folders (every match, no filtering), so
    the estimate globs ``mosa_pattern`` there. Globbing ``rocking_pattern``
    regardless counted 709 folders for a real STO2 run that processes 76.
    On the default ``"rocking"`` path the count remains an upper bound — that
    path additionally masks folders to the mosa/strain samz union, which this
    shape-only estimate cannot evaluate without reading motor positions from
    every folder, and an upper bound is the safe direction for the volume term.

    **Currency**: ``peak_bytes`` is whole-child peak RSS, process image
    included — see :func:`dfxm.stages.strain.estimate`.
    """
    p = {**STAGE.defaults(), **params}
    try:
        root = str(p.get("raw_root") or "").rstrip("/")
        # Mirror run()'s layer choice: `source_scan="mosaicity"` processes the
        # MOSA folders, not the rocking ones. Globbing `rocking_pattern` there
        # regardless counted 709 folders for a run that processes 76 on the real
        # STO2 form — a 9x over-statement of the volume term, in the
        # configuration this project's own saved form actually uses.
        pattern_key = "mosa_pattern" if p.get("source_scan") == "mosaicity" else "rocking_pattern"
        folders = find_matching_folders(root, p.get(pattern_key) or "*") if root else []
        if not folders:
            return CostEstimate(0, 0, None, True, "no scan folders resolved yet")
        first = resolve_input_file(folders[0])
        ds_path = str(p.get("detector_path") or "1.1/measurement/pco_ff")
        with h5py.File(first, "r") as f:
            if ds_path not in f:
                return CostEstimate(0, 0, None, True, f"{ds_path!r} not in {first!r}")
            ds = f[ds_path]
            scan_shape = tuple(int(d) for d in ds.shape)
            itemsize = int(ds.dtype.itemsize)
    except Exception as exc:  # noqa: BLE001 - an estimate is advisory, never fatal
        return CostEstimate(0, 0, None, True, f"cannot size input: {type(exc).__name__}")

    n = len(folders)
    detector_elems = 1
    for dim in scan_shape:
        detector_elems *= dim
    total = n * detector_elems * itemsize  # the whole input on disk, ROI or not

    # `process_raw_scan` reads `det[:, ys:ye, xs:xe]`, so everything downstream
    # — the float32 copy, both accumulators, the aligned volumes — is sized by
    # the ROI, not the detector. On the real STO2 form (roi_x 105,1937 /
    # roi_y 630,1330 against a 2048x2048 detector) ignoring it over-stated the
    # priced work by 3.3x.
    n_frames = scan_shape[0] if scan_shape else 1
    full_h = scan_shape[1] if len(scan_shape) > 1 else 1
    full_w = scan_shape[2] if len(scan_shape) > 2 else 1
    try:
        roi_x, roi_y = _parse_pair(p.get("roi_x")), _parse_pair(p.get("roi_y"))
    except (TypeError, ValueError):
        # A half-typed ROI ("105," while the user is still going) must not make
        # the estimator raise, and must not shrink the estimate either: fall
        # back to the whole detector, which is what run() would read.
        roi_x = roi_y = None
    rows = _roi_span(roi_y, full_h)
    cols = _roi_span(roi_x, full_w)
    read_elems = rows * cols
    scan_elems = n_frames * read_elems

    # The volumes that accumulate are not the frames that were read: the
    # alignment X-pads them by the samy extremes and resamples Z onto a uniform
    # grid, and both inflations compound. `aligned_shape_for_params` is the one
    # implementation of that — a per-caller copy is exactly how the seven peak
    # models drifted apart before — and it returns None on the no-motor path,
    # where run() does no shifting and the read shape IS the volume shape.
    #
    # It is handed the ALREADY-cropped shape, with ROI keys it will not find, so
    # that `_roi_span` above stays the single place this estimator resolves an
    # ROI. Letting it re-parse `roi_x`/`roi_y` itself put two different ROI
    # semantics in one model, which disagreed on exactly the malformed inputs
    # `_roi_span` exists to absorb.
    aligned = A.aligned_shape_for_params(
        p,
        (n, rows, cols),
        pattern_key=pattern_key,
        roi_x_key="__roi_already_applied__",
        roi_y_key="__roi_already_applied__",
    )
    if aligned is not None:
        nz, ny_a, nx_a = aligned
        volume_elems = int(nz) * int(ny_a) * int(nx_a)
    else:
        volume_elems = n * read_elems
    peak = (
        ROCKING_PROCESS_FLOOR_BYTES
        + scan_elems * (itemsize + 4)
        + ROCKING_VOLUME_BYTES_PER_ELEM * volume_elems
    )
    if p["save_topview"]:
        peak += ROCKING_TOPVIEW_BYTES
    return CostEstimate(int(peak), total, (n, *scan_shape), True, None)


def _render(
    result: RockingResult,
    vol,
    z_um,
    scale_z,
    name,
    p,
    out_dir,
    cmap,
    title,
    cbar,
    style=None,
    group=None,
    progress=None,
):
    report = progress or _progress_mod.noop
    sx, sy = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    vmin, vmax = _colorbar_range(vol, float(p["cbar_pct_lo"]), float(p["cbar_pct_hi"]))
    vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)
    prod = RockingProducts(name=name, vmin=vmin, vmax=vmax)
    if clim_note:
        prod.notes.append(clim_note)
    # Product shares of this dataset's slice, over the products actually asked
    # for — the same allocation `visualize._process_dataset` uses, and for the
    # same reason: a fixed fraction reported for a product that is switched off
    # jumps the bar by that product's whole share.
    #
    # `save_layer_pngs` gained a per-layer `progress` in the reporting sweep and
    # `visualize` passes it; this stage took a sub-range, spent it on a single
    # `report(0.0)` and forwarded nothing, so its whole render half was silent.
    # On a 78-layer, two-volume dataset that is minutes with the bar at rest.
    weights = {
        "layers": 6.0 if p["save_layers"] else 0.0,
        "animation": 1.5 if p["save_animation"] else 0.0,
        "top_view": 1.0 if p["save_topview"] else 0.0,
    }
    shares = _progress_mod.weighted_slices(weights)
    starts = {k: v[0] for k, v in shares.items()}
    ends = {k: v[1] for k, v in shares.items()}
    report(0.0, f"{name}: rendering")
    if p["save_layers"]:
        prod.layers_dir = Rnd.save_layer_pngs(
            vol,
            z_um,
            ds_dir,
            name,
            vmin,
            vmax,
            cmap,
            title,
            cbar,
            sx,
            sy,
            style=style,
            group=group,
            progress=_progress_mod.sub_progress(progress, starts["layers"], ends["layers"]),
        )
        report(ends["layers"], f"{name}: layers done")
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
            style=style,
            group=group,
            progress=_progress_mod.sub_progress(progress, starts["animation"], ends["animation"]),
        )
        report(ends["animation"], f"{name}: animation done")
    if p["save_topview"]:
        scene = R3.Scene3D(
            volume=vol,
            spacing=(sx, sy, scale_z),
            cmap=cmap,
            clim=(float(vmin), float(vmax)),
            opacity=float(p["volume_opacity"]),
            mode="volume",
        )
        # A volume wider than the GL 3-D texture limit renders blank without any
        # error. `volume_downsample` 0 (the default) coarsens until it fits, so
        # the top view is a coarser picture rather than an empty one; 1 opts out
        # and gets the warning instead.
        note = R3.apply_texture_fit(scene, R3.volume_texture_limit(), int(p["volume_downsample"]))
        if note:
            prod.notes.append(note)
        # Half the slot for building the scene and probing the GL texture limit
        # (an out-of-process probe), half for the render and save inside
        # `save_top_view` — the same split `visualize` uses.
        report(
            starts["top_view"] + (ends["top_view"] - starts["top_view"]) * 0.5,
            f"{name}: 3-D scene ready",
        )
        try:
            prod.top_view = R3.save_top_view(
                scene,
                os.path.join(ds_dir, f"{name}_top_view.png"),
                cbar_label=cbar,
                group=group,
                style=style,
                progress=_progress_mod.sub_progress(
                    progress,
                    starts["top_view"] + (ends["top_view"] - starts["top_view"]) * 0.5,
                    ends["top_view"],
                ),
            )
        except Exception as exc:  # noqa: BLE001 - no GL -> note + continue
            prod.notes.append(f"3D top-view skipped: {exc}")
        report(ends["top_view"], f"{name}: top view done")
    report(1.0, f"{name}: done")
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

    source = p.get("source_scan", "rocking")
    default_dir = (
        "aligned_raw_mosa_volumes" if source == "mosaicity" else "aligned_raw_rocking_volumes"
    )
    out_dir = p["output_dir"] or os.path.join(raw_root, default_dir)
    result = RockingResult(output_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # 1. mosa reference (anchors X shift + Z origin) + samz union start
    progress(0.02, "reading mosaicity motor positions")
    mosa_samy, mosa_samz, mosa_names = _motors(
        raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"]
    )
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

    # 2/3. choose the layers to process
    if source == "mosaicity":
        # the mosa scans themselves are the layers (no samz-union masking)
        keep_paths = [os.path.join(raw_root, n) for n in mosa_names]
        keep_samy, keep_samz = np.asarray(mosa_samy), np.asarray(mosa_samz)
    else:
        # samz union range (mosa ∪ strain)
        _, strain_samz, _ = _motors(raw_root, p["strain_pattern"], p["samy_path"], p["samz_path"])
        all_samz = np.concatenate([mosa_samz, strain_samz]) if len(strain_samz) else mosa_samz
        z_min, z_max = float(all_samz.min()), float(all_samz.max())

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
        bool(p["subtract_background"]),
        progress=_progress_mod.sub_progress(progress, 0.1, 0.6),
    )
    if sum_vol is None:
        result.skipped.append("no rocking scans processed successfully")
        progress(1.0, "no rocking volumes produced")
        return result
    result.specific_frame_idx = spec_idx
    result.n_layers_used = len(names_used)

    # 5. align (mosa-anchored samy shift + Z interpolation)
    # Four whole-volume operations, each slow on a real dataset and none of them
    # previously reporting: the bar sat at 0.65 through all of them.
    progress(0.60, "aligning: samy shift (sum)")
    sum_aligned = A.apply_samy_shifts_to_volume(
        sum_vol, samy_used, scale_x, samy_dir, ref_samy=samy_ref
    )
    progress(0.64, "aligning: samy shift (specific frame)")
    spec_aligned = A.apply_samy_shifts_to_volume(
        spec_vol, samy_used, scale_x, samy_dir, ref_samy=samy_ref
    )
    pad_left = A.compute_pad_left(samy_used, scale_x, samy_dir, ref_samy=samy_ref)
    pad_right = A.compute_pad_right(samy_used, scale_x, samy_dir, ref_samy=samy_ref)
    del sum_vol, spec_vol
    progress(0.68, "aligning: Z interpolation (sum)")
    sum_aligned, z_uniform, scale_z = A.interpolate_to_uniform_z(
        sum_aligned, samz_used, ref_samz=samz_ref
    )
    progress(0.72, "aligning: Z interpolation (specific frame)")
    spec_aligned, _, _ = A.interpolate_to_uniform_z(spec_aligned, samz_used, ref_samz=samz_ref)
    result.volume_shape = tuple(sum_aligned.shape)
    result.z_span_um = float(z_uniform[-1] - z_uniform[0]) if len(z_uniform) else 0.0

    # 6. save aligned HDF5 (feeds the oblique slicer)
    if p["save_aligned_h5"]:
        aligned_name = p["aligned_h5_name"]
        if source == "mosaicity" and aligned_name == STAGE.defaults()["aligned_h5_name"]:
            aligned_name = "aligned_raw_mosa_volumes.h5"
        aligned_path = os.path.join(out_dir, aligned_name)
        progress(0.76, f"writing {aligned_name}")
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

    # 7. render — two volumes, each owning half of the tail
    progress(0.80, "rendering volumes")
    RENDER_LO, RENDER_HI = 0.80, 0.99
    sum_tag = "(a.u., normalized)" if p["normalize_sum"] else "(a.u.)"
    style = style_from_params(p)
    raw_cmap = resolve_cmap(style, "raw")
    _render(
        result,
        sum_aligned,
        z_uniform,
        scale_z,
        "raw_sum_intensity",
        p,
        out_dir,
        raw_cmap,
        _sum_title(source),
        f"Sum intensity {sum_tag}",
        style=style,
        group="raw",
        progress=_progress_mod.sub_progress(
            progress, *_progress_mod.slice_for(0, 2, RENDER_LO, RENDER_HI)
        ),
    )
    _render(
        result,
        spec_aligned,
        z_uniform,
        scale_z,
        f"raw_specific_frame_{spec_idx:03d}",
        p,
        out_dir,
        raw_cmap,
        _spec_title(source, spec_idx),
        "Intensity (a.u.)",
        style=style,
        group="raw",
        progress=_progress_mod.sub_progress(
            progress, *_progress_mod.slice_for(1, 2, RENDER_LO, RENDER_HI)
        ),
    )

    progress(1.0, f"aligned {result.n_layers_used} {source} layers -> {out_dir}")
    return result


# ---------------------------------------------------------------------------
# Figure catalog
# ---------------------------------------------------------------------------

# Map product name prefix → in-file HDF5 dataset key
_PRODUCT_DATASET: dict[str, str] = {
    "raw_sum_intensity": "sum_intensity",
    "raw_specific_frame": "specific_frame",
}

# Colorbar labels matching _render() calls (static entries; sum_intensity is built dynamically
# in figures() because it depends on the normalize_sum param)
_PRODUCT_CBAR: dict[str, str] = {
    "specific_frame": "Intensity (a.u.)",
}


@register("rocking")
def figures(result: RockingResult, params: dict) -> list[FigureSpec]:
    """One map FigureSpec per Z layer per product in the aligned rocking volume."""
    if not result.aligned_path:
        return []
    if not result.datasets:
        return []

    sx = float(params.get("pixel_size_x_um", STAGE.defaults()["pixel_size_x_um"]))
    sy = float(params.get("pixel_size_y_um", STAGE.defaults()["pixel_size_y_um"]))

    # Read z_uniform_um once from the aligned h5
    with h5py.File(result.aligned_path, "r") as f:
        z_um = f["z_uniform_um"][:].tolist()

    normalize_sum = bool(params.get("normalize_sum", False))
    sum_tag = "(a.u., normalized)" if normalize_sum else "(a.u.)"
    source = str(params.get("source_scan", "rocking"))

    all_specs: list[FigureSpec] = []
    for prod in result.datasets:
        # Resolve in-file dataset key from the product name prefix
        ds_key: str | None = None
        for prefix, key in _PRODUCT_DATASET.items():
            if prod.name.startswith(prefix):
                ds_key = key
                break
        if ds_key is None:
            # Unknown product type: no catalog mapping. Add it to _PRODUCT_DATASET when a
            # new RockingProducts kind is rendered in run().
            continue

        if ds_key == "sum_intensity":
            cbar_label = f"Sum intensity {sum_tag}"
            title = _sum_title(source)
        else:
            cbar_label = _PRODUCT_CBAR.get(ds_key, prod.name)
            if ds_key == "specific_frame":
                if result.specific_frame_idx is not None:
                    title = _spec_title(source, result.specific_frame_idx)
                else:
                    title = (
                        "Mosa-integrated Specific Frame"
                        if source == "mosaicity"
                        else "Background-subtracted Specific Frame"
                    )
            else:
                title = prod.name
        # id_prefix uses the product name so each product gets distinct ids/filenames
        id_prefix = prod.name

        all_specs.extend(
            volume_layer_specs(
                h5_path=result.aligned_path,
                dataset=ds_key,
                id_prefix=id_prefix,
                title=title,
                cbar_label=cbar_label,
                cmap="gray",
                cmap_group="raw",
                sx=sx,
                sy=sy,
                vmin=prod.vmin,
                vmax=prod.vmax,
                z_um=z_um,
            )
        )

    return all_specs


# ---------------------------------------------------------------------------
# Cold replot (re-render from aligned h5 without re-running the stage)
# ---------------------------------------------------------------------------

# in-file dataset key → (default title, default cbar) for cold replot
_DATASET_DISPLAY: dict[str, tuple[str, str]] = {
    "sum_intensity": ("Sum intensity", "Sum intensity (a.u.)"),
    "specific_frame": ("Specific Frame", "Intensity (a.u.)"),
}


def _replot_default_clim(
    dataset, params: dict, style, *, budget_bytes: int | None = None
) -> tuple[float, float]:
    """Compute the default clim for a cold replot the same way the run does.

    **Two rungs, one answer.** When the whole volume fits the working-set budget
    (:func:`_fits_in_core`) the percentiles are taken in one pass by
    :func:`_colorbar_range`, byte for byte what this function always did. Only a
    volume too large for that streams them through
    :func:`~dfxm.common.volumeio.stream_quantile`, which returns what
    ``np.percentile`` returns — not an estimate and not budget-dependent — so
    the colours are identical on either rung and on any machine. That equality
    is the only reason a rung boundary is allowed here at all: which rung runs
    depends on the machine, so a colour that differed between them would be a
    colour that depended on the machine.

    The rung exists because streaming is not free in **time**: an exact
    percentile in bounded memory traverses the volume ~12 times against one, and
    on this path a traversal is a raw HDF5 read with no alignment behind it.
    Measured old-versus-new on the streaming rung: 0.06 s -> 1.25 s on a 9.4 MB
    volume, 1.24 s -> 24.2 s on a 195 MB one. ``compose/adapters.py`` calls this
    per panel while a user waits on a figure-builder preview, which is what
    makes ~20x unacceptable below the boundary and irrelevant above it (there
    the in-core rung is not slower, it is an OOM).

    Then ``apply_round_clim``, as before. An all-NaN volume falls back to
    ``(0.0, 1.0)`` on both rungs — :func:`_colorbar_range`'s empty-input answer
    — and that fallback goes **through** the rounding, which is what the
    original whole-volume form did; short-circuiting the return would both skip
    the rounding and hijack a legitimate percentile pair that happened to be
    exactly ``(0.0, 1.0)``.

    ``budget_bytes`` is the working set this may hold, defaulting to
    :data:`REPLOT_CLIM_WORKING_SET_BYTES`; it is a parameter so tests can force
    either rung on a small volume. ``0`` means zero, not "use the default".
    """
    defaults = STAGE.defaults()
    pct_lo = float(params.get("cbar_pct_lo", defaults["cbar_pct_lo"]))
    pct_hi = float(params.get("cbar_pct_hi", defaults["cbar_pct_hi"]))
    budget = REPLOT_CLIM_WORKING_SET_BYTES if budget_bytes is None else int(budget_bytes)

    if _fits_in_core(dataset, budget):
        vmin, vmax = _colorbar_range(dataset[:], pct_lo, pct_hi)
    else:
        from ..common.volumeio import dataset_blocks, stream_quantile

        block_bytes = _clim_block_budget(dataset, budget)

        def blocks():
            # A factory, not a generator: `stream_quantile` traverses several
            # times and diagnoses a stale one, unlike its stream_mean/minmax
            # siblings.
            return dataset_blocks(dataset, budget_bytes=block_bytes)

        vmin = stream_quantile(blocks, pct_lo)
        vmax = stream_quantile(blocks, pct_hi)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            # `stream_quantile` signals "no finite value anywhere" with NaN;
            # `_colorbar_range` signals it with (0.0, 1.0). Convert, then take
            # the same rounding the other rung takes.
            vmin, vmax = 0.0, 1.0
    vmin, vmax, _ = apply_round_clim(vmin, vmax, style)
    return vmin, vmax


def replot_catalog(h5_path: str) -> list[ReplotGroup]:
    """List every aligned rocking product (3-D dataset) as a replot group."""
    groups: list[ReplotGroup] = []
    with h5py.File(h5_path, "r") as f:
        z_um = f["z_uniform_um"][:].tolist() if "z_uniform_um" in f else None
        for key, (title, _cbar) in _DATASET_DISPLAY.items():
            obj = f.get(key)
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                continue
            n_z = obj.shape[0]
            labels = [
                f"layer {z}" + (f"  (Z={z_um[z]:.2f} µm)" if z_um else "") for z in range(n_z)
            ]
            groups.append(
                ReplotGroup(key=key, label=title, item_labels=labels, shape=tuple(obj.shape[1:]))
            )
    return groups


def render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]:
    """Re-render selected aligned rocking map layers cold from an aligned h5.

    ``selections`` is ``list[(dataset_key, item_idxs | None)]`` where dataset_key
    is ``sum_intensity`` or ``specific_frame``. ``clim`` overrides vmin/vmax and
    may be ``None``, a single ``(vmin, vmax)`` tuple, or a
    ``{dataset_key: (vmin, vmax)}`` mapping (per-product limits); ``roi`` crops
    each layer. PNGs under ``{out_dir}/{key}/``; returns paths.

    When the resolved clim for a product is ``None`` (both boxes blank), defaults
    are computed the same way the run does — percentile-based via
    ``_replot_default_clim`` — so the output is faithful to the original run
    PNGs. Explicit ``clim`` values still win (forwarded to
    ``render_volume_layer``).
    """
    params = params or {}
    px = float(params.get("pixel_size_x_um", STAGE.defaults()["pixel_size_x_um"]))
    py = float(params.get("pixel_size_y_um", STAGE.defaults()["pixel_size_y_um"]))
    # Source-aware title / cbar — mirrors figures() so the replot titles match
    # the originals rather than the generic _DATASET_DISPLAY strings.
    source = str(params.get("source_scan", STAGE.defaults().get("source_scan", "rocking")))
    normalize_sum = bool(params.get("normalize_sum", STAGE.defaults().get("normalize_sum", False)))
    sum_tag = "(a.u., normalized)" if normalize_sum else "(a.u.)"
    written: list[str] = []
    with h5py.File(h5_path, "r") as f:
        z_um = f["z_uniform_um"][:].tolist() if "z_uniform_um" in f else None
        for key, idxs in selections:
            obj = f.get(key)
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                continue
            # Build source-aware title + cbar label (mirrors figures())
            if key == "sum_intensity":
                title = _sum_title(source)
                cbar_label = f"Sum intensity {sum_tag}"
            elif key == "specific_frame":
                title = (
                    "Mosa-integrated Specific Frame"
                    if source == "mosaicity"
                    else "Background-subtracted Specific Frame"
                )
                cbar_label = _PRODUCT_CBAR.get(key, "Intensity (a.u.)")
            else:
                # Unknown product: graceful generic fallback
                _generic_title, _generic_cbar = _DATASET_DISPLAY.get(key, (key, "Intensity (a.u.)"))
                title, cbar_label = _generic_title, _generic_cbar
            # Default clim: percentile-based (mirrors the run), not raw min/max
            # — and computed ONLY when a side of it is actually going to be
            # used. `_apply_clim` lets a half-open override keep the default on
            # its blank side, so the test is "both sides supplied", not
            # "an override exists". It used to run unconditionally, spending a
            # dozen traversals on a number `_apply_clim` then discarded.
            clim_k = resolve_clim(clim, key)
            if clim_k is not None and clim_k[0] is not None and clim_k[1] is not None:
                vmin, vmax = clim_k
            else:
                vmin, vmax = _replot_default_clim(obj, params, style)
            n_z = obj.shape[0]
            layer_list = list(range(n_z)) if idxs is None else list(idxs)
            sub_dir = os.path.join(out_dir, key)
            os.makedirs(sub_dir, exist_ok=True)
            for z in layer_list:
                if z < 0 or z >= n_z:
                    continue
                fig = render_volume_layer(
                    h5_path,
                    key,
                    z,
                    cmap="gray",
                    cmap_group="raw",
                    title=title,
                    cbar_label=cbar_label,
                    sx=px,
                    sy=py,
                    vmin=vmin,
                    vmax=vmax,
                    style=style,
                    clim=clim_k,
                    roi=roi,
                    z_um=z_um,
                )
                if fig is None:
                    continue
                png = os.path.join(sub_dir, f"{key}_layer_{z:04d}.png")
                fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
                written.append(png)
    return written


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
    ap.add_argument(
        "--downsample",
        type=int,
        default=0,
        help="3-D block-average factor; 0 = fit this machine's GL texture limit",
    )
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
            volume_downsample=args.downsample,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\naligned -> {res.aligned_path}; shape {res.volume_shape}; skipped {len(res.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
