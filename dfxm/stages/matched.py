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
from ..common.errors import StageUserError
from ..common.figures import FigureSpec, register
from ..common.h5io import resolve_input_file
from ..common.plotting import CMAP_CHOICES, apply_round_clim, style_from_params
from ..common.raster import extract_motor_positions, find_h5_file
from ..common.sort import find_matching_folders
from ..config.models import CostEstimate, Param, ParamType, SeeAlso, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


STAGE = StageSpec(
    name="matched",
    label="Rocking-matched layers",
    description=(
        "For each strain layer, finds the rocking scan taken at the same (samy, samz) sample "
        "position and saves one background-subtracted detector frame as a grayscale PNG, "
        "pixel-aligned with the strain/mosaicity layer images."
    ),
    params=(
        Param(
            "raw_root",
            ParamType.DIR,
            "Raw data root",
            must_exist=True,
            help="RAW_DATA root containing both the strain and the rocking scan folders.",
        ),
        Param(
            "strain_pattern",
            ParamType.STR,
            "Strain pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help="Glob matching the strain scan folders (one per layer).",
        ),
        Param(
            "rocking_pattern",
            ParamType.STR,
            "Rocking pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help="Glob matching the rocking scan folders to search for position matches.",
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
            "pco_ff_path",
            ParamType.STR,
            "Detector path",
            default="1.1/measurement/pco_ff",
            advanced=True,
            group="Data layout",
            help="HDF5 path to the detector frames inside each rocking scan file.",
        ),
        Param(
            "frame_index",
            ParamType.INT,
            "Frame index",
            default=0,
            help="0-based detector frame to extract from each matched rocking scan.",
        ),
        Param(
            "match_threshold_mm",
            ParamType.FLOAT,
            "Match threshold",
            unit="mm",
            default=0.0004,
            help=(
                "Maximum (samy, samz) distance in mm for a rocking scan to count as matching "
                "a strain layer; layers with no scan inside the threshold are skipped."
            ),
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
                "pixels, so a wrong value shifts the saved frame out of pixel-alignment with the "
                "strain/mosaicity layer images."
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
                "calibration. A wrong value skews the vertical physical scale of the saved frame."
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
                "Sign (+1 or −1) relating the samy motor direction to detector X — "
                "the same shift convention as the strain layers."
            ),
        ),
        Param(
            "colormap",
            ParamType.ENUM,
            "Colormap",
            default="gray",
            choices=CMAP_CHOICES,
            advanced=True,
            group="Appearance",
            help="Colormap for the saved PNGs (default gray).",
        ),
        Param(
            "vmin",
            ParamType.STR,
            "vmin",
            default="",
            advanced=True,
            group="Appearance",
            help="Lower intensity limit (blank = the automatic percentile below).",
        ),
        Param(
            "vmax",
            ParamType.STR,
            "vmax",
            default="",
            advanced=True,
            group="Appearance",
            help="Upper intensity limit (blank = the automatic percentile below).",
        ),
        Param(
            "auto_pct_lo",
            ParamType.FLOAT,
            "Auto pct low",
            unit="%",
            default=1.0,
            advanced=True,
            group="Appearance",
            help="Percentile used for the automatic lower intensity limit.",
        ),
        Param(
            "auto_pct_hi",
            ParamType.FLOAT,
            "Auto pct high",
            unit="%",
            default=95.0,
            advanced=True,
            group="Appearance",
            help="Percentile used for the automatic upper intensity limit.",
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help="Where the matched layer PNGs are written.",
        ),
    ),
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” "
            "(left panel); the range fields in Advanced below are this stage's own.",
        ),
        SeeAlso(
            "param:colormap",
            "“Publication style…” drives the raw-intensity colormap; this is "
            "the fallback for anything without a quantity group.",
        ),
    ),
    estimate="dfxm.stages.matched:estimate",
)


@dataclass
class MatchedLayer:
    """Per-layer recompute record — everything figures() needs to rebuild a shifted frame."""

    raw_h5: str  # absolute path to the rocking scan .h5 file
    pco_ff_path: str  # HDF5 dataset path inside raw_h5
    frame_index: int  # 0-based frame extracted from the detector stack
    shift_px: float  # samy X-shift applied (pixels)
    pad_left: int  # left padding (pixels) from compute_pad_left
    nx_new: int  # total canvas width after padding (pixels)
    ext_x: float  # physical canvas width (µm)
    ext_y: float  # physical canvas height (µm)
    vmin: float  # colour-scale lower bound
    vmax: float  # colour-scale upper bound
    title: str  # axes title used in the saved PNG
    colormap: str  # matplotlib colormap name
    layer_index: int  # strain layer index i (used for figure_id / filename)


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
    vmin_raw: float | None = None
    vmax_raw: float | None = None
    max_match_dist_um: float = 0.0
    pngs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    recorded: list[MatchedLayer] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Core (faithful port)
# -----------------------------------------------------------------------------
# What one in-plane block of the stack costs, per element of it, on top of the
# block's own stored bytes: the `.astype(np.float64)` upcast (8 B) and
# `np.nanmedian`'s internals. Measured with `tracemalloc` over the upcast plus
# the median, invariant in the stored dtype because everything after the upcast
# is float64. The cost is NOT flat: it is a ~35 B/element asymptote plus a
# per-block overhead that the element count divides, so SMALL frame counts are
# the expensive case, not the large ones --
#
#     (41, 160, 160)  35.1     (11, 128, 128)  35.8     (11, 64, 64)  38.2
#     (5, 128, 128)   36.8     (3, 256, 256)   44.7     (3, 64, 64)   50.0
#
# An earlier 36 was fitted to the (11, 128, 128)-and-up shapes alone and was
# therefore under the true cost for every nf <= 4 and for nf = 5-8 on small
# blocks; a short rocking scan is an ordinary input, and one measured run
# exceeded its budget by 1.24x. 48 covers the worst measured case (50.0) only
# barely; 64 is chosen for headroom. The number is a DIVISOR in
# `_median_block_budget`, so charging too little buys a block larger than was
# counted, while charging too much only buys a smaller block -- the median is
# exact on any in-plane block, so a smaller block changes runtime and nothing
# else. Round up, never down.
#
# Version-sensitive: measured on numpy 1.26.4, where `nanmedian` switches to an
# `apply_along_axis` branch at >= 600 frames and the cost drops to ~8.0
# B/element. A rocking scan is tens of frames, so the expensive branch is the
# one to size against -- but that branch boundary, and the costs either side of
# it, can move between numpy versions. `tests/test_stage_matched.py::
# test_matched_median_working_set_constant_covers_the_measured_cost` re-measures
# on the installed numpy and fails if this number stops covering it.
MEDIAN_WORKING_SET_PER_ELEMENT = 64

# The working set one `load_pco_ff_frame` call may hold. A fixed constant rather
# than `advice.working_set_budget_bytes`: the median is exact on any in-plane
# block, so a larger machine would buy nothing but a larger block, and a fixed
# number means two machines run the identical blocking on the identical data.
MEDIAN_BLOCK_WORKING_SET_BYTES = 128 * 1024 * 1024


def _median_block_budget(dset, budget_bytes: int) -> int:
    """*budget_bytes* of working set, in the block bytes ``iter_blocks`` counts.

    ``iter_blocks`` sizes a block by its bytes **in the stored dtype**; this
    stage then holds :data:`MEDIAN_WORKING_SET_PER_ELEMENT` more per element
    while the median runs. Handing the budget over raw would buy a block ~33x
    too large for a uint16 detector stack. Integer division rounds the budget
    **down**, the safe direction.
    """
    itemsize = max(1, int(np.dtype(dset.dtype).itemsize))
    return max(1, int(budget_bytes) * itemsize // (itemsize + MEDIAN_WORKING_SET_PER_ELEMENT))


def load_pco_ff_frame(h5_path, pco_ff_path, frame_index, *, budget_bytes: int | None = None):
    """Single frame, per-pixel median background subtracted, negatives -> NaN.

    The median reduces along the **frame** axis, so it is exact on any in-plane
    sub-block: ``stack[:, y0:y1, :]`` gives those rows the same answer the whole
    stack does, because no pixel's median depends on any other pixel's. Blocking
    in-plane is therefore lossless rather than approximate, which is why this
    stage's :func:`estimate` reports ``chunkable=True``.

    ``budget_bytes`` is the working set one block may cost, defaulting to
    :data:`MEDIAN_BLOCK_WORKING_SET_BYTES`; it is a parameter so tests can force
    the multi-block path on a small stack. ``0`` means zero, not "use the
    default".
    """
    from ..common.volumeio import iter_blocks

    with h5py.File(h5_path, "r") as f:
        if pco_ff_path not in f:
            return None
        ds = f[pco_ff_path]
        if ds.ndim == 2:
            # No median and — as before — no negative clamp on this path.
            return ds[:].astype(np.float64)
        if ds.ndim != 3:
            return None
        idx = min(int(frame_index), ds.shape[0] - 1)
        working_set = MEDIAN_BLOCK_WORKING_SET_BYTES if budget_bytes is None else int(budget_bytes)
        budget = _median_block_budget(ds, working_set)
        out = np.empty((ds.shape[1], ds.shape[2]), dtype=np.float64)
        for ysl, block in iter_blocks(ds, budget_bytes=budget, axis=1):
            sub = block.astype(np.float64)
            out[ysl] = sub[idx] - np.nanmedian(sub, axis=0)
    # Applied once to the finished frame, as the whole-stack form applied it —
    # per block would be equivalent, but keeping it here keeps the diff honest
    # about what changed.
    out[out < 0] = np.nan
    return out


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


def estimate(params: dict) -> CostEstimate:
    """Peak memory for a matched run, from HDF5 shapes only.

    ``run()`` loads **one** scan at a time via ``load_pco_ff_frame`` — the
    matched-layer loop calls it per output layer, and ``stack``/``background``
    are locals that die on return, so nothing accumulates across layers except
    the small pooled-clim list. Within one call: ``ds[:].astype(np.float64)``
    holds the source stack (native dtype) and its float64 copy together, and
    ``np.nanmedian(stack, axis=0)`` needs its own internal float64-sized copy
    of the stack to sort along axis 0 — ``scan_elems * (itemsize + 16)``. On
    top of that, up to ~10 pooled clim arrays plus a couple of frame-sized
    working copies (``background``, ``corrected``) persist at once, bounded as
    ``12 * frame_elems * 8``. Peak is therefore
    ``scan_elems * (itemsize + 16) + 12 * frame_elems * 8`` for one scan,
    independent of the folder count — ``input_bytes`` still sums every
    matching folder's stack (the data that will eventually be read, even
    though only one is resident at a time).

    ``chunkable=True``. The median needs the whole stack along the **frame
    axis** only — no pixel's median depends on any other pixel's — so
    :func:`load_pco_ff_frame` blocks it in-plane and gets the identical frame
    back (see that function). The earlier ``False``, reasoned "an exact median
    needs the whole stack", was a statement the code itself contradicts.

    Two caveats this model carries, both deliberately left for the
    estimator-recalibration pass rather than fixed here. The peak expression
    still describes the *un-budgeted* whole-stack form, so it is now a ceiling
    reached only when a scan fits :data:`MEDIAN_BLOCK_WORKING_SET_BYTES` in one
    block; and its ``+ 16`` under-states ``np.nanmedian``'s real internals,
    measured at 35-50 B/element depending on the frame count (see
    :data:`MEDIAN_WORKING_SET_PER_ELEMENT`, charged at 64).
    """
    p = {**STAGE.defaults(), **params}
    try:
        root = str(p.get("raw_root") or "").rstrip("/")
        folders = find_matching_folders(root, p.get("rocking_pattern") or "*") if root else []
        if not folders:
            return CostEstimate(0, 0, None, True, "no scan folders resolved yet")
        first = resolve_input_file(folders[0])
        ds_path = str(p.get("pco_ff_path") or "1.1/measurement/pco_ff")
        with h5py.File(first, "r") as f:
            if ds_path not in f:
                return CostEstimate(0, 0, None, True, f"{ds_path!r} not in {first!r}")
            ds = f[ds_path]
            scan_shape = tuple(int(d) for d in ds.shape)
            itemsize = int(ds.dtype.itemsize)
    except Exception as exc:  # noqa: BLE001 - an estimate is advisory, never fatal
        return CostEstimate(0, 0, None, True, f"cannot size input: {type(exc).__name__}")

    elems = 1
    for dim in scan_shape:
        elems *= dim
    frame_elems = elems // scan_shape[0] if scan_shape and scan_shape[0] else elems
    input_bytes = len(folders) * elems * itemsize
    peak = elems * (itemsize + 16) + 12 * frame_elems * 8
    # `shape[0]` here counts scan FOLDERS, but what the stage chunks is one
    # scan's detector rows, so it names its own span rather than let
    # `advice.plan_run` read the folder count and call it "layers".
    rows = scan_shape[1] if len(scan_shape) > 1 else 1
    return CostEstimate(
        peak,
        input_bytes,
        (len(folders), *scan_shape),
        True,
        None,
        (rows, "detector rows"),
        confidence="conservative",
    )


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> MatchedResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    style = style_from_params(p)
    raw_root = (p["raw_root"] or "").rstrip("/")
    if not raw_root:
        raise StageUserError(
            "matched requires 'raw_root'",
            hint="Set 'Raw data root' to the RAW_DATA folder that contains the scan folders.",
        )
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
        raise StageUserError(
            f"no strain motor positions for {p['strain_pattern']!r}",
            hint="Check 'Strain pattern' against the scan folder names under the raw root.",
        )

    progress(0.1, "reading rocking motor positions")
    rock_folders = find_matching_folders(raw_root, p["rocking_pattern"])
    rock_samy, rock_samz, rock_names = extract_motor_positions(
        rock_folders, p["samy_path"], p["samz_path"]
    )
    if len(rock_samy) == 0:
        raise StageUserError(
            f"no rocking motor positions for {p['rocking_pattern']!r}",
            hint="Check 'Rocking pattern' against the scan folder names under the raw root.",
        )

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
    vmin_user, vmax_user = _parse_float(p["vmin"]), _parse_float(p["vmax"])
    if vmin_user is None and vmax_user is None:
        rlo, rhi, clim_note = apply_round_clim(result.vmin, result.vmax, style)
        if clim_note:
            result.vmin_raw, result.vmax_raw = result.vmin, result.vmax
            result.vmin, result.vmax = rlo, rhi
            vmin, vmax = rlo, rhi  # the loop below renders with vmin/vmax
            progress(0.2, clim_note)

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
        if img.shape != (ny, nx_orig):
            # a scan with a different detector ROI/shape can't share the canvas
            result.skipped.append(f"layer {i}: frame shape {img.shape} != {(ny, nx_orig)}")
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
            style=style,
        )
        png = os.path.join(layers_dir, f"layer_{i:04d}.png")
        fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
        result.pngs.append(png)
        result.n_saved += 1
        # NOTE: figures().build() in this module mirrors this exact recompute
        # (load_pco_ff_frame → _apply_shift_single → layer_figure) to rebuild
        # the figure at export time; keep the two in sync.
        result.recorded.append(
            MatchedLayer(
                raw_h5=_rock_h5(raw_root, rock_names[m]),
                pco_ff_path=p["pco_ff_path"],
                frame_index=frame_index,
                shift_px=float(shifts_px[i]),
                pad_left=pad_left,
                nx_new=nx_new,
                ext_x=ext_x,
                ext_y=ext_y,
                vmin=float(vmin),
                vmax=float(vmax),
                title=title,
                colormap=p["colormap"],
                layer_index=i,
            )
        )

    progress(1.0, f"saved {result.n_saved}/{result.n_strain} matched layers -> {layers_dir}")
    return result


@register("matched")
def figures(result: MatchedResult, params: dict) -> list[FigureSpec]:
    """Figure catalog for the matched stage — one map spec per saved layer.

    Each spec's ``build(style)`` re-reads the raw rocking frame, re-applies the
    samy X-shift, and calls ``render.layer_figure`` with the same arguments that
    ``run()`` used, so the rebuilt figure matches the saved PNG exactly.

    If the raw ``.h5`` file is missing at build time, ``FileNotFoundError`` is
    raised (the GUI surfaces it with a clear message).
    """
    if not result.recorded:
        return []

    def _make_build(rec: MatchedLayer):
        raw_h5 = rec.raw_h5
        pco_ff_path = rec.pco_ff_path
        frame_index = rec.frame_index
        shift_px = rec.shift_px
        pad_left = rec.pad_left
        nx_new = rec.nx_new
        ext_x = rec.ext_x
        ext_y = rec.ext_y
        vmin = rec.vmin
        vmax = rec.vmax
        title = rec.title
        colormap = rec.colormap
        layer_index = rec.layer_index

        def build(style):
            if not os.path.exists(raw_h5):
                raise FileNotFoundError(
                    f"Raw rocking file for layer {layer_index} not found: {raw_h5!r}"
                )
            img = load_pco_ff_frame(raw_h5, pco_ff_path, frame_index)
            if img is None:
                raise FileNotFoundError(
                    f"Could not load detector data from {raw_h5!r} "
                    f"(path={pco_ff_path!r}, frame={frame_index})"
                )
            shifted = _apply_shift_single(img, shift_px, pad_left, nx_new)
            fig, _, _ = Rnd.layer_figure(
                shifted,
                vmin,
                vmax,
                colormap,
                ext_x,
                ext_y,
                title,
                "Intensity − background (a.u.)",
                style=style,
            )
            return fig

        return build

    return [
        FigureSpec(
            figure_id=f"matched_layer_{rec.layer_index:04d}",
            title=f"Matched layer {rec.layer_index}",
            kind="map",
            filename=f"matched_layer_{rec.layer_index:04d}",
            build=_make_build(rec),
        )
        for rec in result.recorded
    ]


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
