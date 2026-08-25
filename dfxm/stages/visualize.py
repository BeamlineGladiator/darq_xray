"""Visualize stage — aligned mosaicity/strain volumes to images + animation + 3D.

Ported from ``visualize_aligned_volumes_v6``. Per dataset (mosaicity chi/mu
Center-of-mass + FWHM, and strain) it:

1. aligns the stacked volume — ROI -> samy sub-pixel X-shift -> uniform-Z
   interpolation — reusing :mod:`dfxm.common.alignment` (golden-tested,
   voxel-identical to the PVTI exporter);
2. centres CoM volumes (midrange/mean/median) and picks colour limits (strain
   keeps its physical zero, symmetric limits);
3. writes per-layer PNGs, a layer-by-layer animation (MP4 with GIF fallback),
   a 3-D top-view render, and (optionally) a rotating 3-D orbit video (the 3-D
   renders are best-effort — skipped gracefully without a GL context).

Mosaicity uses ``magma``; strain uses ``RdBu_r``. Rendering uses the explicit
Figure/Agg API (no pyplot / matplotlib.use) so the module is import-safe in the
Qt GUI process.
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
from ..common import volumeio
from ..common.figures import FigureSpec, register
from ..common.h5io import sum_dataset_bytes
from ..common.plotting import apply_round_clim, resolve_cmap, style_from_params
from ..common.raster import extract_motor_positions
from ..common.sort import find_matching_folders
from ..config.models import CostEstimate, Param, ParamType, SeeAlso, StageSpec

ProgressFn = Callable[[float, str], None]

# What a child running this stage costs resident before it touches a voxel:
# interpreter, numpy, h5py, matplotlib (and, once a 3-D product is requested,
# pyvista/VTK). `tracemalloc` cannot see any of it, so it is what
# :func:`~dfxm.common.advice.working_set_budget_bytes` must take off the
# machine's headroom before converting the rest into an allocation budget.
#
# Measured with `tests/peak_rss.py` on a 4x6x8 mosaicity-only run — 6 KB of
# data against a ~598 MiB peak — with **every product on**, which is the default
# configuration and by some way the heaviest image. What each stage of it costs,
# measured the same way on the same fixture:
#
#     bare (no PNGs, no animation, no 3-D)   107 MiB
#     + layer PNGs and the animation         267 MiB  (matplotlib, the writers)
#     + pyvista/VTK imported, render failing 445 MiB
#     + the 3-D top view actually rendering  598 MiB  (the GL framebuffer)
#
# The all-on figure is the one that matters, and not merely because it is the
# default: the 3-D import happens *inside* the first field's `_process_dataset`,
# so every field after the first streams its alignment with VTK already
# resident. A floor measured with the 3-D products off would be ~330 MiB short
# for four fields out of five.
#
# It is therefore a **GL-dependent** number, which is the honest state of
# affairs rather than a defect: a headless machine with no usable renderer sits
# at the 445 MiB rung. An independent cross-check landed in the same place —
# eight all-products-on runs at two shapes (24x128x128, 32x160x160) and four
# budgets held RSS between 580.6 and 591.1 MiB while their traced peaks barely
# moved, i.e. the whole of that figure is the process image.
#
# **Not paraview's 300 MB.** That stage imports VTK and never matplotlib; this
# one imports both. Pasting its number here fails the assertion below rather
# than silently over-stating the budget.
#
# The declared value carries ~1.3x slack over the measurement (the same ratio
# paraview's does), because the additive RSS model is not an envelope — see
# `advice.MARGINAL_RSS_PER_TRACED_BYTE` — and the floor is the term with room to
# absorb that: over-stating it only shrinks the budget, under-stating it invites
# an OOM. `test_rss_floor_covers_the_measured_process_image` pins it against a
# live measurement via `tests/peak_rss.py::assert_floor_covers`, which brackets
# from both sides, so the constant can be neither too low nor inflated to
# silence the check.
#
# It is a floor for a *stage child*. `aligned_field` runs in the GUI process,
# where Qt is already resident and the true floor is higher; the budget it
# derives is therefore optimistic there. That costs a coarser blocking than
# ideal, not correctness — `aligned_field` materialises one volume regardless
# (the 3-D viewer needs it whole), and the budget only sizes the blocks feeding
# it.
RSS_FLOOR_BYTES = 768 * 1024 * 1024

# The Z step (µm) the stage falls back to when no raw scan folders were found,
# so there are no samz positions to derive one from. `extract_motor_positions`
# fills samy and samz from the same folders, so they are empty together.
_NO_MOTOR_Z_STEP_UM = 2.0


def _noop(_frac: float, _msg: str) -> None:
    pass


STAGE = StageSpec(
    name="visualize",
    label="Visualize volumes",
    description=(
        "Aligns the stacked mosaicity/strain volumes into the shared sample frame "
        "(samy shift + uniform-Z interpolation) and renders per-layer PNGs, a layer "
        "animation, a 3-D top view, and an optional rotating 3-D video."
    ),
    params=(
        Param(
            "mosa_volume_file",
            ParamType.PATH,
            "Mosaicity volume",
            must_exist=True,
            help=(
                "The stacked mosaicity volume (stacked_volumes.h5) from the mosaicity stage. "
                "Leave blank to skip mosaicity rendering."
            ),
        ),
        Param(
            "strain_volume_file",
            ParamType.PATH,
            "Strain volume",
            must_exist=True,
            help=(
                "The stacked strain volume (stacked_strain_volumes.h5) from the strain stage. "
                "Leave blank to skip strain rendering."
            ),
        ),
        Param(
            "raw_root",
            ParamType.DIR,
            "Raw data root",
            must_exist=True,
            help=(
                "RAW_DATA root with the original scan folders — the samy/samz motor positions "
                "read from there drive the alignment."
            ),
        ),
        Param(
            "mosa_pattern",
            ParamType.STR,
            "Mosaicity raw pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob matching the raw mosaicity scan folders, used to read their "
                "samy/samz positions."
            ),
        ),
        Param(
            "strain_pattern",
            ParamType.STR,
            "Strain raw pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob matching the raw strain scan folders, used to read their samy/samz positions."
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
                "HDF5 path to the sample-Y motor position inside each scan file (under the first "
                "BLISS entry). Only change for a different beamline file layout."
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
                "HDF5 path to the sample-Z motor position inside each scan file (under the first "
                "BLISS entry). Only change for a different beamline file layout."
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
            "Map ROI X",
            default="",
            roi_group="crop",
            roi_axis="x",
            roi_frame="map",
            help=(
                "Crop along map X as 'c0,c1' map pixels — columns of the darfix map, relative "
                "to the darfix window, NOT absolute detector pixels (blank = full width). "
                "Pre-filled from the experiment's analysis window. All volumes must share the "
                "same crop to stay co-registered."
            ),
        ),
        Param(
            "roi_y",
            ParamType.STR,
            "Map ROI Y",
            default="",
            roi_group="crop",
            roi_axis="y",
            roi_frame="map",
            help=(
                "Crop along map Y as 'r0,r1' map pixels — rows of the darfix map, relative to "
                "the darfix window, NOT absolute detector pixels (blank = full height). "
                "Pre-filled from the experiment's analysis window. All volumes must share the "
                "same crop to stay co-registered."
            ),
        ),
        Param(
            "center_method",
            ParamType.ENUM,
            "Centre method",
            default="midrange",
            choices=("midrange", "mean", "median"),
            advanced=True,
            group="Alignment",
            help=(
                "How the colour scale of the misorientation (CoM) maps is centred: "
                "midrange = midpoint of the robust limits, or mean/median of the data. "
                "Display only."
            ),
        ),
        Param(
            "range_pct",
            ParamType.FLOAT,
            "Range percentile",
            unit="%",
            default=99.5,
            advanced=True,
            group="Alignment",
            help=(
                "Robust percentile for colour limits, e.g. 99.5 ignores the most extreme "
                "0.5 % of pixels when setting the scale."
            ),
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help=(
                "Where the rendered PNGs, animation and top view are written "
                "(blank = next to the input volume)."
            ),
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
            "save_layers",
            ParamType.BOOL,
            "Save layer PNGs",
            default=True,
            advanced=True,
            group="Output",
            help="Write one PNG per layer of each volume.",
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
            "save_rotation",
            ParamType.BOOL,
            "Save rotating 3-D video",
            default=False,
            advanced=True,
            group="Output",
            help=(
                "Write a movie of the 3-D volume render spinning once around "
                "(one 360° orbit; frame count from 'Rotation frames'). Uses the "
                "same opacity as the top view and the Animation format container. "
                "Slow — off by default."
            ),
        ),
        Param(
            "volume_opacity",
            ParamType.FLOAT,
            "3D opacity",
            default=0.85,
            advanced=True,
            group="Appearance",
            help=(
                "Opacity of the rendered 3-D top view and rotation video, 0–1. "
                "Scales the render's opacity in every render mode (volume mode: "
                "scales the transfer function)."
            ),
        ),
        Param(
            "render_mode",
            ParamType.ENUM,
            "3D render mode",
            default="volume",
            choices=("volume", "surface", "isosurface"),
            advanced=True,
            group="Appearance",
            advice_key="3d_texture",
            help=(
                "How the 3-D top view and rotation video draw the volume: 'volume' is "
                "true volumetric rendering (shaded, transfer-function opacity), "
                "'surface' the legacy NaN-thresholded mesh, 'isosurface' stacked "
                "contour shells."
            ),
        ),
        Param(
            "opacity_mapping",
            ParamType.ENUM,
            "3D opacity mapping",
            default="linear",
            choices=("linear", "sigmoid", "geom", "geom_r"),
            advanced=True,
            group="Appearance",
            help=(
                "Opacity transfer function for volumetric 3-D rendering: linear, "
                "sigmoid (emphasises mid-range values), geom (high values), geom_r "
                "(low values). Ignored by the surface and isosurface modes."
            ),
        ),
        Param(
            "rotation_frames",
            ParamType.INT,
            "Rotation frames",
            default=180,
            advanced=True,
            group="Output",
            help="Frames in one 360-degree orbit of the rotation video (15 fps).",
        ),
        Param(
            "log_scale",
            ParamType.BOOL,
            "Log colour scale (3D)",
            default=False,
            advanced=True,
            group="Appearance",
            help=(
                "Logarithmic colour mapping for the 3-D top view and rotation video. "
                "Falls back to linear (with a note) when the colour range includes "
                "zero or negative values."
            ),
        ),
    ),
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” (left panel), not here.",
        ),
    ),
    estimate="dfxm.stages.visualize:estimate",
)


@dataclass
class DatasetProducts:
    name: str
    shape: tuple[int, int, int]
    vmin: float
    vmax: float
    layers_dir: str | None = None
    animation: str | None = None
    top_view: str | None = None
    rotation_video: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class VisualizeResult:
    output_dir: str = ""
    datasets: list[DatasetProducts] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Colour / range helpers (faithful port)
# -----------------------------------------------------------------------------
def _symmetric_range(data, pct=99):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return (-1.0, 1.0)
    am = float(np.percentile(np.abs(valid), pct))
    return (-am, am)


def _midrange_clim(data, pct=99.5):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, (-1.0, 1.0)
    if pct >= 100.0:
        lo, hi = float(np.min(valid)), float(np.max(valid))
    else:
        lo, hi = (float(v) for v in np.percentile(valid, [100.0 - pct, pct]))
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) or 1.0
    return center, (-half, half)


def _center_com_and_range(data, method, range_pct):
    method = method.lower()
    if method == "midrange":
        center, (vmin, vmax) = _midrange_clim(data, range_pct)
        return data - center, vmin, vmax
    valid = data[np.isfinite(data)]
    # `volumeio.stream_mean`, not `np.nanmean`, and over the whole array as a
    # single block. The two are not bit-equal — a compensated sum against a
    # pairwise one, differing in ~70% of random samples — and this helper and
    # `_center_com_and_range_streamed` must agree EXACTLY (see the rung-boundary
    # note below `_colorbar_range`). Cost is nil: `valid` is the same full-size
    # finite selection `np.nanmean` was handed, so nothing new is allocated.
    # NaN on an empty selection, exactly as `np.nanmean` of one is.
    sub = (
        float(volumeio.stream_mean([valid]))
        if method == "mean"
        else float(np.nanmedian(valid))
        if valid.size
        else 0.0
    )
    data = data - sub
    vmin, vmax = _symmetric_range(data)
    return data, vmin, vmax


def _colorbar_range(data):
    # `isfinite`, not `~isnan`. This used to keep +/-inf, which no other helper
    # here does and which `_colorbar_range_streamed` cannot reproduce — see the
    # rung-boundary note below. Keeping them was also not a behaviour worth
    # preserving: `np.percentile` over a set containing `inf` returns `inf` for
    # the upper limit, i.e. a colour scale on which every finite voxel renders
    # as one colour.
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(valid, 1)), float(np.percentile(valid, 99)))


# -----------------------------------------------------------------------------
# Streaming siblings of the four helpers above
# -----------------------------------------------------------------------------
# Each takes a *blocks factory* — a zero-argument callable returning a fresh
# iterator of ``(slice, array)`` pairs, i.e. `StreamedAlignment.blocks` — rather
# than an array, and reproduces its in-core original's arithmetic exactly. The
# in-core forms stay: `figures()` and the replot paths hold a whole volume by
# design and have no stream to walk.
#
# -- THE RUNG BOUNDARY IS AN EQUALITY, AND ANY DIVERGENCE ACROSS IT IS A DEFECT -
#
# `_source_and_clim` picks between these helpers and their in-core originals by
# asking how much memory the machine has. So a value that differs between the
# two is a value that depends on the machine — the laptop-versus-workstation
# divergence this whole phase exists to remove, reintroduced at the one seam the
# phase itself created. It is not a tolerable approximation, not even at an ulp,
# and not even on input the fixtures do not contain.
#
# The equality is held by three facts, each of which a future edit can break:
#
# 1. `volumeio.stream_quantile` returns bit-for-bit what `np.percentile`
#    returns (its own contract, pinned in `tests/test_common_volumeio.py`;
#    re-checked here over 240 random cases). Every percentile on both sides is
#    therefore the same number, and `np.nanmedian` and the vector form
#    `np.percentile(v, [a, b])` were checked to agree with the scalar form too.
# 2. The finite selection is `np.isfinite` on BOTH sides. `_colorbar_range` used
#    to select with `~np.isnan`, keeping `±inf` where these reductions drop it;
#    that was the divergence, and it is fixed in the in-core helper rather than
#    papered over here.
# 3. The mean is `volumeio.stream_mean` on BOTH sides. It is a compensated
#    (Neumaier) sum and `np.nanmean` reduces pairwise; they disagreed in 42 of
#    60 random samples, so `_center_com_and_range` calls the streaming reduction
#    over a single block rather than `np.nanmean`.
#
# Pinned by `test_both_rungs_agree_on_clims_with_infinities`, which puts `±inf`
# and NaN in the volume and requires the two rungs to return identical limits
# and identical centred data. Adding a colour convention means extending that
# test, not just the pair of helpers.
def _arrays(blocks):
    """A factory over the bare arrays of a ``(slice, array)`` block factory."""
    return lambda: (block for _sl, block in blocks())


def _shifted(blocks, offset):
    """*blocks* with a constant subtracted — the streaming ``data - center``.

    The ``if offset`` short-circuit is
    :func:`~dfxm.common.alignment.align_volume_streamed`'s own, for the same
    reason: subtracting a zero allocates a second block per block and changes no
    value.
    """
    if not offset:
        return blocks
    return lambda: ((sl, block - offset) for sl, block in blocks())


def _symmetric_range_streamed(blocks, pct=99.0):
    """Streaming :func:`_symmetric_range`.

    ``np.abs`` is applied before the finite filter rather than after it, which
    selects the same values — ``abs`` maps NaN to NaN and ``±inf`` to ``inf``,
    both of which the filter drops either way — while keeping the whole
    computation inside one traversal-driven reduction.
    """
    arrays = _arrays(blocks)
    am = volumeio.stream_quantile(lambda: (np.abs(block) for block in arrays()), pct)
    if not np.isfinite(am):  # no finite voxel anywhere
        return (-1.0, 1.0)
    return (-float(am), float(am))


def _midrange_clim_streamed(blocks, pct=99.5):
    """Streaming :func:`_midrange_clim`."""
    arrays = _arrays(blocks)
    if pct >= 100.0:
        lo, hi = volumeio.stream_minmax(arrays())
    else:
        lo = volumeio.stream_quantile(arrays, 100.0 - pct)
        hi = volumeio.stream_quantile(arrays, pct)
    if not np.isfinite(lo):
        return 0.0, (-1.0, 1.0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) or 1.0
    return center, (-half, half)


def _center_com_and_range_streamed(blocks, method, range_pct):
    """Streaming :func:`_center_com_and_range`: ``(blocks_factory, vmin, vmax)``.

    ``range_pct`` keeps its in-core meaning exactly, which is **not** a scale
    factor on the limits: for ``midrange`` it is the robust percentile pair
    ``[100 - pct, pct]`` that sets the centre and the half-width, and for
    ``mean``/``median`` it is not used at all — those centre on the statistic and
    then take :func:`_symmetric_range`'s fixed 99th percentile of ``|value|``.
    """
    method = method.lower()
    if method == "midrange":
        center, (vmin, vmax) = _midrange_clim_streamed(blocks, range_pct)
        return _shifted(blocks, center), vmin, vmax
    arrays = _arrays(blocks)
    if method == "mean":
        # NaN on an all-non-finite volume, exactly as the in-core helper's own
        # `stream_mean` of an empty selection is — neither guards it, and the
        # resulting all-NaN volume then takes `_symmetric_range`'s empty branch.
        sub = volumeio.stream_mean(arrays())
    else:
        sub = volumeio.stream_quantile(arrays, 50.0)
        if not np.isfinite(sub):
            sub = 0.0  # the in-core `else 0.0` for an empty selection
    shifted = _shifted(blocks, sub)
    vmin, vmax = _symmetric_range_streamed(shifted)
    return shifted, vmin, vmax


def _colorbar_range_streamed(blocks):
    """Streaming :func:`_colorbar_range` — the 1st and 99th percentiles."""
    arrays = _arrays(blocks)
    lo = volumeio.stream_quantile(arrays, 1.0)
    if not np.isfinite(lo):
        return (0.0, 1.0)
    return (float(lo), float(volumeio.stream_quantile(arrays, 99.0)))


def _display_info(dataset_name, is_strain=False):
    """(title, cbar_label, cmap_group) for a dataset; group None = not a std quantity."""
    if is_strain:
        return ("Strain (cot method)", "Strain (ε)", "strain")
    axis = (
        "χ"
        if dataset_name.startswith("chi_")
        else "μ"
        if dataset_name.startswith("mu_")
        else dataset_name
    )
    if "Center_of_mass" in dataset_name:
        return (f"{axis} Misorientation", "Misorientation (°)", "mosa_com")
    if "FWHM" in dataset_name:
        return (f"{axis} Peak Broadening", "Peak broadening (°)", "mosa_fwhm")
    return (dataset_name.replace("_", " "), "(°)", None)


def estimate(params: dict) -> CostEstimate:
    """Peak memory for this run, from HDF5 shapes and motor positions.

    **MEASURED against the real STO2 dataset at master a424b1f** (76x1266x1832,
    both files, ROI 0,1832 / 400,1100, **every 3-D product on** — ``save_topview``
    and ``save_rotation`` both true, which is the shipped default and the
    heaviest configuration):

        estimate 10.51 GiB   measured peak RSS 8.83 GiB   ratio 0.84x

    i.e. it **over-predicts by ~1.2x**, which is the safe direction and is what
    the prose below argues for. Two things that number settles, because both had
    been asserted without measurement:

    * The over-prediction survives the phase-5 rewrite. The earlier worry that
      streaming had turned this into an under-estimate was based on pre-phase
      figures (a 16.36 GiB peak against a 9.79 GiB model); the code got cheaper
      and the model did not, so the sign flipped back.
    * **8.83 GiB does not fit an 8 GB machine.** The stage is bounded, not
      small: with 3-D products on it materialises a whole aligned volume
      (see the ``save_topview``/``save_rotation`` path), so a run on that
      machine relies on ``plan_run`` chunking it — which this over-estimate
      correctly triggers — rather than on the unconstrained peak being modest.

    ``scratch_bytes`` is **zero, on every path** — see the comment at the
    ``return`` below. This stage never hands ``align_volume_streamed`` a
    ``scratch_dir``, so it never caches and never needs disk.

    ``run()``
    streams the alignment (see :func:`_align_streamed`) and materialises a whole
    volume only when a 3-D product is requested, so the arithmetic below
    over-predicts — which is the safe direction: this is what
    ``advice.plan_run`` compares against the machine's headroom, and
    over-predicting there only makes it hand over a *smaller* ``budget_bytes``,
    i.e. block harder, whereas under-predicting would greenlight an in-core run
    that then OOMs. Recalibrating means measuring the streamed peak on the real
    dataset, not editing the terms by inspection; the standing warning at the end
    of this docstring applies unchanged.

    The ``total_input`` term in the arithmetic below models the *old* ``run()``,
    which loaded the mosaicity file with ``load_mosa_datasets`` into a
    ``datasets`` dict (raw, native dtype, every field) that was never deleted or
    reassigned — so it stayed alive through the strain section, and both files'
    raw bytes had to be summed rather than maxed. That dict is gone.
    ``load_mosa_datasets`` was replaced by ``mosa_field_names`` +
    ``load_mosa_field``: ``run()`` reads **one** field at a time, ``del``s the
    raw read once ``_align`` has copied out of it, resets ``data`` before the
    next read, and drops the last field's aligned volume before the strain
    section starts. Only one field's worth of raw data is ever resident, and the
    two sections' peaks are now a max, not a sum.

    The second term is still live: each ``_align`` call (``apply_roi_3d`` ->
    ``apply_samy_shifts_to_volume`` -> ``interpolate_to_uniform_z``, upcasting
    to float64) leaves up to three float64-sized arrays alive for the field
    being processed, bounded as ``3 * largest_elems * 8``. ``chunkable=True``.

    **Recalibration warning — do not just swap the sum for a max.** The current
    figure over-predicts, which is the safe direction, and is deliberately left
    unchanged here. Two hazards for whoever narrows it:

    * The ``3 * largest_elems * 8`` bound is expressed in *unpadded* elements.
      ``apply_samy_shifts_to_volume`` widens the canvas along image-X by the
      extreme samy offsets and ``interpolate_to_uniform_z`` resamples onto a
      grid that exceeds the layer count whenever samz is irregular, and the two
      inflations multiply — so each surviving copy is *larger* than
      ``largest_elems``. Neither extent is derivable from HDF5 shapes alone
      (both depend on motor values); the retired ``total_input`` term was in
      effect accidental headroom masking that. Removing it without replacing it
      turns an over-estimate into an under-estimate, which is the dangerous
      direction (it greenlights a run that then OOMs).
    * One field's raw read (native dtype) is still coexistent with the first
      arrays of its own alignment chain, so the per-field term is not purely
      float64.
    """
    p = {**STAGE.defaults(), **params}
    total = 0
    largest_elems = 0
    largest: tuple[int, ...] | None = None
    for name in ("mosa_volume_file", "strain_volume_file"):
        path = str(p.get(name) or "")
        if not path:
            continue
        nbytes, shape, _itemsize = sum_dataset_bytes(path)
        total += nbytes
        if shape is not None:
            elems = 1
            for dim in shape:
                elems *= dim
            if elems > largest_elems:
                largest_elems = elems
            if largest is None or len(shape) > len(largest):
                largest = shape
    if not total:
        return CostEstimate(0, 0, None, True, "no readable volume files selected yet")
    peak = total + 3 * largest_elems * 8
    # `scratch_bytes` stays at its 0 default, INCLUDING for
    # `center_method="median"`, and that is not an oversight to fix.
    # `_align_streamed` never passes `scratch_dir=` to `align_volume_streamed`
    # (paraview is the only stage that does), and without one the multi-pass
    # statistic re-reads instead of caching — slower, same result, zero disk.
    # Pricing a spill this stage cannot perform would let `advice.plan_run`
    # BLOCK a run that touches no disk at all, which is the one thing this
    # phase promised never to do. Pinned by
    # `test_visualize_never_hands_the_alignment_a_scratch_dir`: if that ever
    # starts passing one, size it here — and not before.
    return CostEstimate(peak, total, largest, True)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def mosa_field_names(path) -> list[str]:
    """Field names in a mosaicity volume file, without reading any data.

    Sorted, so the field order (and therefore ``VisualizeResult.datasets``
    order) is deterministic instead of inheriting h5py's group-key order.
    """
    names = []
    with h5py.File(path, "r") as f:
        for group in ("chi", "mu"):
            if group in f:
                for ds in f[group].keys():
                    names.append(f"{group}_{ds.replace(' ', '_')}")
    return sorted(names)


def _mosa_dataset(f, name):
    """The open HDF5 dataset for *name* in an already-open file, or None.

    The name-matching convention lives here alone: a streaming caller needs the
    dataset (to slice it block by block) rather than its contents, and
    :func:`load_mosa_field` is the read-it-all-now wrapper over the same lookup.
    """
    for group in ("chi", "mu"):
        if group not in f:
            continue
        for ds in f[group].keys():
            if f"{group}_{ds.replace(' ', '_')}" == name:
                return f[group][ds]
    return None


def load_mosa_field(path, name):
    """One field from a mosaicity volume file, or None if absent."""
    with h5py.File(path, "r") as f:
        dset = _mosa_dataset(f, name)
        return None if dset is None else dset[:]


def load_strain_volume(path):
    with h5py.File(path, "r") as f:
        return f["strain"][:] if "strain" in f else None


# -----------------------------------------------------------------------------
# Motor helper (shared by run(), aligned_field(), and figures())
# -----------------------------------------------------------------------------
def _read_motors(raw_root: str, pattern: str, samy_path: str, samz_path: str):
    """samy/samz positions from raw scan folders; empty arrays if unavailable."""
    if not raw_root or not pattern:
        return np.array([]), np.array([])
    folders = find_matching_folders(raw_root, pattern)
    if not folders:
        return np.array([]), np.array([])
    samy, samz, _ = extract_motor_positions(folders, samy_path, samz_path)
    return samy, samz


# -----------------------------------------------------------------------------
# Per-dataset + alignment
# -----------------------------------------------------------------------------
def _parse_pair(text):
    if text is None or str(text).strip() == "":
        return None
    parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'a,b', got {text!r}")
    return tuple(parts)


def _align(volume, samy, samz, *, scale_x, samy_direction, roi_x, roi_y):
    data = A.apply_roi_3d(volume, roi_x, roi_y)
    if len(samy) > 0:
        data = A.apply_samy_shifts_to_volume(data, samy, scale_x, samy_direction)
    if len(samz) > 0:
        data, z_pos, scale_z = A.interpolate_to_uniform_z(data, samz)
    else:
        scale_z = 2.0
        z_pos = np.arange(data.shape[0]) * scale_z
    return data, z_pos, scale_z


def _whole_volume_stream(data):
    """An in-memory array presented as a one-block ``StreamedAlignment``.

    So the rest of the stage has exactly one kind of aligned-volume input. Used
    for the no-motor case in :func:`_align_streamed`, where there is nothing to
    stream in the first place.
    """
    n_z = int(data.shape[0])
    return A.StreamedAlignment(
        shape=tuple(int(d) for d in data.shape),
        dtype=np.dtype(data.dtype),
        z_uniform_um=np.arange(n_z) * _NO_MOTOR_Z_STEP_UM,
        scale_z_um=_NO_MOTOR_Z_STEP_UM,
        pad_left=0,
        pad_right=0,
        center_offset=0.0,
        block_layers=n_z,
        working_set_bytes=int(data.nbytes),
        blocks=lambda: iter([(slice(0, n_z), data)]),
    )


# How many block-sized arrays this stage's heaviest block consumer holds
# *alongside* the block. `align_volume_streamed`'s working-set model prices the
# ALIGNMENT chain and nothing downstream of it, and the colour-limit reductions
# are not free: `volumeio.stream_quantile`'s rank search keeps the finite
# selection and the in-window selection (float64), the `searchsorted` indices and
# `np.clip`'s separate output (int64) and the `isfinite` mask — 41 bytes per
# 8-byte element, which is the very figure `align_volume_streamed` computes for
# its own cached-median pass (`dtype.itemsize + 8 * (retained + 1) + 1`) — and a
# centring pass adds the shifted copy on top. So the alignment gets a *share* of
# the budget and the reductions get the rest.
#
# The count is 41/8 + 1 = 6.125, and this is **rounded UP, to 7**. Rounding is
# not cosmetic here: the constant is a divisor, so a smaller one makes
# `budget // multiple` LARGER and permits more than was counted. 6 would have
# been the under-predicting direction — on a machine whose budget equals its
# headroom the reductions overrun by about 2% — which is the direction that
# ends in an OOM rather than in a slower run.
#
# This divides by the cost of THIS call site's consumers, exactly as
# `paraview._process_mosaicity` divides by the number of concurrent field
# streams. It is not a correction to what `budget_bytes` means. Pinned by
# `test_streamed_run_stays_within_its_working_set_budget`, which measures the
# run's real `tracemalloc` peak against the budget it was handed — an unpinned
# constant in the permissive direction is the defect that cost Task 9 a fix
# round.
REDUCTION_WORKING_SET_MULTIPLE = 7


def _align_streamed(dset, samy, samz, *, scale_x, samy_direction, roi_x, roi_y, budget_bytes):
    """The streaming counterpart of :func:`_align`; same arguments, same order.

    Returns ``(streamed, z_positions_um, scale_z_um)``, all three known before a
    voxel is read. *budget_bytes* is the whole run's working-set budget; the
    alignment is given ``1 / REDUCTION_WORKING_SET_MULTIPLE`` of it so the
    colour-limit reductions that consume its blocks fit in the rest.

    With **no motor positions** it falls back to the old in-core chain and hands
    the result over as a single block, exactly as `paraview._unaligned_field`
    does and for the same reason: `align_volume_streamed` always interpolates,
    and resampling a NaN-bearing volume onto its own Z nodes is not the identity
    (scipy's linear interpolant reads the value *below* each node, spreading
    every NaN one layer down). `extract_motor_positions` empties samy and samz
    together, so this is the misconfigured-run path, not a large production run.
    """
    if len(samz) == 0:
        raw = dset[:]
        data = A.apply_roi_3d(raw, roi_x, roi_y)
        del raw
        if len(samy) > 0:
            data = A.apply_samy_shifts_to_volume(data, samy, scale_x, samy_direction)
        streamed = _whole_volume_stream(data)
    else:
        streamed = A.align_volume_streamed(
            dset,
            samy,
            samz,
            scale_x=scale_x,
            samy_direction=samy_direction,
            roi_x=roi_x,
            roi_y=roi_y,
            budget_bytes=(
                None
                if budget_bytes is None
                else max(1, int(budget_bytes) // REDUCTION_WORKING_SET_MULTIPLE)
            ),
        )
    return streamed, streamed.z_uniform_um, streamed.scale_z_um


class _LayerSource:
    """An aligned volume as ascending per-layer reads, materialised only if asked.

    Deliberately duck-types the ``(Z, Y, X)`` array
    :func:`~dfxm.common.render.save_layer_pngs` and
    :func:`~dfxm.common.render.save_layer_animation` consume: they use ``.shape``
    and ``vol[z]`` and nothing else, and they walk *z* ascending — the animation
    restarting at 0 once per container it writes — which is exactly what a
    Z-block stream can serve. So the renderers need no changes and cannot drift
    from the in-core path: ``source[z]`` **is** ``volume[z]``.

    A read below the current block rewinds and re-walks, which re-runs the whole
    alignment chain. That is the honest cost of a forward-only stream and it is
    why :func:`_process_dataset` materialises up front whenever a 3-D product is
    wanted rather than streaming the 2-D products and then materialising anyway.

    :meth:`whole` is the escape hatch for the one consumer that cannot stream:
    `render3d.Scene3D` uploads the entire grid to VTK. It adopts a stream that
    yields a single covering block instead of copying it into a second array —
    without that, a machine generous enough to leave one block would hold two
    whole volumes and the conversion would make the peak *worse*.
    """

    def __init__(self, blocks, shape, dtype) -> None:
        self._blocks = blocks
        self.shape = tuple(int(d) for d in shape)
        self._dtype = np.dtype(dtype)
        self._array = None
        self._iter = None
        self._block = None
        self._start = self._stop = 0

    def __len__(self) -> int:
        return self.shape[0]

    def _rewind(self) -> None:
        self._iter = self._blocks()
        self._block = None
        self._start = self._stop = 0

    def __getitem__(self, z):
        if self._array is not None:
            return self._array[z]
        z = int(z)
        if z < 0:
            # A forward-only stream has no end to count back from, and left
            # alone this surfaced as `TypeError: 'NoneType' object is not
            # subscriptable` out of the rewind's empty state — which names
            # neither the cause nor the caller. Raise what indexing raises.
            raise IndexError(
                f"layer {z}: a streamed volume cannot be indexed from the end "
                f"(it has {self.shape[0]} layers, readable in ascending order)"
            )
        if self._iter is None or z < self._start:
            self._rewind()
        while z >= self._stop:
            item = next(self._iter, None)
            if item is None:
                raise IndexError(f"layer {z} is past the end of a {self.shape[0]}-layer stream")
            sl, self._block = item
            self._start, self._stop = int(sl.start), int(sl.stop)
        return self._block[z - self._start]

    def whole(self) -> np.ndarray:
        """The volume as one array, walking the stream once. Cached.

        The drain itself — including the adopt-don't-copy of a single covering
        block, which is the part that is expensive to get wrong — is
        :func:`~dfxm.common.alignment.materialise_blocks`, not a fourth copy of
        it. This used to be that fourth copy, so a fix in the shared helper
        would silently not have reached here.
        """
        if self._array is None:
            self._array = A.materialise_blocks(self._blocks, self.shape, self._dtype)
            self._iter = self._block = None
        return self._array


def _materialise(source):
    """*source* as a plain array — a no-op for one that already is."""
    return source.whole() if isinstance(source, _LayerSource) else source


def _fits_in_core(streamed) -> bool:
    """True when the solved blocking is a single block — the whole volume.

    The same question :func:`~dfxm.common.alignment.align_volume_streamed` asks
    of its own cached-median branch (``out_step >= nz``), and asked of the
    *solved* blocking rather than recomputed from the output bytes, so it
    answers "does this fit the budget?" in the budget's own working-set terms.
    """
    return int(streamed.block_layers) >= int(streamed.shape[0])


def _source_and_clim(streamed, *, kind, center_method="", range_pct=99.5):
    """``(source, vmin, vmax)`` for one aligned field.

    *kind* is ``"com"``, ``"fwhm"`` or ``"strain"`` — which of the three colour
    conventions applies.

    **The project's own escalation ladder, one rung of it.** When the budget
    leaves the alignment a single block, the whole aligned volume exists anyway
    and there is nothing to gain by walking it as a stream: the in-core helpers
    run — one traversal, instead of the eight a pair of exact streaming
    percentiles costs. Only when the volume does *not* fit does the streaming
    path take over, where re-reading is the price of running at all.
    ``advice.plan_run`` makes the same in-core-then-chunked choice for the same
    reason.

    **Which rung runs depends on the machine, so the two must return identical
    values** — see the rung-boundary note above the streaming siblings. That is
    an invariant, not an aspiration: a divergence here is the
    laptop-versus-workstation difference this phase exists to remove.
    """

    def make_source(blocks):
        return _LayerSource(blocks, streamed.shape, streamed.dtype)

    if _fits_in_core(streamed):
        data = make_source(streamed.blocks).whole()
        if kind == "com":
            return _center_com_and_range(data, center_method, range_pct)
        if kind == "strain":
            return (data, *_symmetric_range(data))
        return (data, *_colorbar_range(data))
    if kind == "com":
        blocks, vmin, vmax = _center_com_and_range_streamed(
            streamed.blocks, center_method, range_pct
        )
        return make_source(blocks), vmin, vmax
    limits = (
        _symmetric_range_streamed(streamed.blocks)
        if kind == "strain"
        else _colorbar_range_streamed(streamed.blocks)
    )
    return (make_source(streamed.blocks), *limits)


def _process_dataset(
    source,
    z_pos,
    scale_z,
    name,
    vmin,
    vmax,
    cmap,
    title,
    cbar,
    p,
    out_dir,
    style=None,
    group=None,
    progress=None,
):
    """Render one aligned dataset's products.

    *source* is the aligned volume: either a plain ``(Z, Y, X)`` array or a
    :class:`_LayerSource` presenting one as ascending per-layer reads. The
    products are identical either way.

    *progress* takes a **local** 0..1 fraction covering this dataset's products;
    `run` wraps it with `sub_progress` so each dataset owns a slice of the run's
    bar. The internal split is by product, and the layer-PNG loop — much the
    longest of them on a real volume — reports per layer inside its own share.
    """
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)
    sx, sy = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    # The 3-D products upload the entire grid to VTK and so cannot stream at
    # all. When one is wanted, materialise the volume ONCE up front and let the
    # per-layer products read that array too: streaming them as well would
    # re-run the alignment chain two or three more times for a peak that is
    # already one whole volume. When no 3-D product is wanted nothing is ever
    # materialised.
    wants_3d = bool(p["save_topview"] or p["save_rotation"])
    data = _materialise(source) if wants_3d else source
    prod = DatasetProducts(name=name, shape=tuple(data.shape), vmin=float(vmin), vmax=float(vmax))
    if wants_3d and isinstance(source, _LayerSource):
        # The alignment was blocked because the machine asked for it, and then a
        # 3-D product overrode that and took the whole volume anyway. Say so:
        # this is the one place where the stage's memory use is NOT bounded by
        # the budget, and it is the reason a capped STO2 run peaks at 4.8 GiB
        # against an 8 GB machine's 3.6 GiB headroom. Silence here is what makes
        # that look like a mystery rather than a setting.
        prod.notes.append(
            f"3-D products need the whole {data.nbytes / (1 << 30):.2f} GiB volume in memory, "
            "so this dataset ignored the streaming budget — turn off 'Save topview' "
            "and 'Save rotation' to keep the run bounded"
        )

    # Product shares of this dataset's slice. The layer PNGs get the bulk
    # because they are per-layer work on a real volume while the rest are a
    # fixed handful of renders; the exact split matters little, since
    # `EtaEstimator` measures the recent rate rather than trusting these weights.
    report = progress or _progress_mod.noop
    report(0.0, f"{name}: rendering")
    if p["save_layers"]:
        prod.layers_dir = Rnd.save_layer_pngs(
            data,
            z_pos,
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
            progress=_progress_mod.sub_progress(progress, 0.0, 0.6),
        )
    report(0.6, f"{name}: layers done")
    if p["save_animation"]:
        prod.animation = Rnd.save_layer_animation(
            data,
            z_pos,
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
        )
    report(0.75, f"{name}: animation done")
    log_scale = bool(p["log_scale"])
    if log_scale and not R3.log_valid((vmin, vmax)):
        log_scale = False
        prod.notes.append("log scale skipped: colour range includes non-positive values")
    scene = None
    if p["save_topview"] or p["save_rotation"]:
        # `data` is a plain array here by construction (see the materialisation
        # above) — `Scene3D` slices and reshapes its volume, so it is not built
        # at all on the streaming path rather than built and left unused.
        scene = R3.Scene3D(
            volume=data,
            spacing=(sx, sy, scale_z),
            mode=str(p["render_mode"]),
            cmap=cmap,
            clim=(float(vmin), float(vmax)),
            log_scale=log_scale,
            opacity=float(p["volume_opacity"]),
            opacity_mapping=str(p["opacity_mapping"]),
        )
        # A volume wider than the GL 3-D texture limit renders blank without any
        # error — say so instead of shipping empty products (no auto-downsample).
        note = R3.oversize_note(scene, R3.volume_texture_limit())
        if note:
            prod.notes.append(note)
    report(0.85, f"{name}: 3-D scene ready")
    if p["save_topview"]:
        try:
            prod.top_view = R3.save_top_view(
                scene,
                os.path.join(ds_dir, f"{name}_top_view.png"),
                cbar_label=cbar,
                group=group,
                style=style,
            )
        except Exception as exc:  # noqa: BLE001 - no GL / pyvista issue -> note + continue
            prod.notes.append(f"3D top-view skipped: {exc}")
    if p["save_rotation"]:
        try:
            prod.rotation_video = R3.save_rotation_video(
                scene,
                os.path.join(ds_dir, f"{name}_rotation"),
                p["output_format"],
                cbar_label=cbar,
                group=group,
                style=style,
                n_frames=int(p["rotation_frames"]),
            )
            if prod.rotation_video is None:
                prod.notes.append("rotation video skipped: volume has no finite voxels")
        except Exception as exc:  # noqa: BLE001 - no GL / pyvista issue -> note + continue
            prod.notes.append(f"rotation video skipped: {exc}")
    report(1.0, f"{name}: done")
    return prod


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def _run_budget_bytes(p: dict, out_dir: str | None = None) -> int:
    """Working-set budget for this run's alignment streams, in bytes.

    Measured from the machine unless the caller injected ``_budget_bytes``. The
    underscore marks it as not a :class:`StageSpec` parameter: it never appears
    on the form and is not part of the saved config, exactly like the
    ``plot_style`` snapshot ``gui/stage_view.py`` injects at run time. Tests use
    it to pin a blocking that does not depend on the machine they run on.

    The number is in ``tracemalloc``/allocation currency, which is what
    :func:`~dfxm.common.alignment.align_volume_streamed` prices its working set
    in — deliberately *not* RSS, which additionally carries the interpreter,
    matplotlib and h5py's buffers. So the machine's headroom, which *is* RSS,
    goes through :func:`~dfxm.common.advice.working_set_budget_bytes` with this
    stage's own :data:`RSS_FLOOR_BYTES` rather than straight in. An injected
    ``_budget_bytes`` is taken as already being in working-set currency, since a
    caller naming that key is naming the budget itself.
    """
    injected = p.get("_budget_bytes")
    if injected is not None:
        return max(1, int(injected))
    from ..common import advice, machine

    return advice.working_set_budget_bytes(
        machine.profile(output_dir=out_dir), rss_floor_bytes=RSS_FLOOR_BYTES
    )


def run(params: dict, progress: ProgressFn | None = None) -> VisualizeResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    style = style_from_params(p)
    scale_x = float(p["pixel_size_x_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["roi_x"]), _parse_pair(p["roi_y"])
    out_dir = p["output_dir"] or os.path.join(
        os.path.dirname(p["mosa_volume_file"] or p["strain_volume_file"] or "."),
        "aligned_volume_visualizations",
    )
    result = VisualizeResult(output_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)
    raw_root = (p["raw_root"] or "").rstrip("/")
    budget_bytes = _run_budget_bytes(p, out_dir)

    # --- how the bar is divided -------------------------------------------
    # One equal slice per DATASET across both halves, rather than the old fixed
    # breakpoints (0.1-0.5 for all of mosaicity, then a pinned 0.6 for the whole
    # of strain). Strain is one dataset among N+1, not half the bar: with five
    # mosaicity fields it used to get 40% of the bar for ~17% of the work, and
    # reported nothing at all while doing it. Counted before either half runs so
    # both share one denominator.
    mosa_file = p["mosa_volume_file"]
    strain_file = p["strain_volume_file"]
    mosa_names = mosa_field_names(mosa_file) if mosa_file and os.path.exists(mosa_file) else []
    n_datasets = len(mosa_names) + (1 if strain_file and os.path.exists(strain_file) else 0)
    WORK_LO, WORK_HI = 0.05, 0.99

    # --- mosaicity ---
    if mosa_file and os.path.exists(mosa_file):
        progress(0.02, "loading mosaicity volume")
        names = mosa_names
        samy, samz = _read_motors(raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"])
        for i, name in enumerate(names):
            ds_lo, ds_hi = _progress_mod.slice_for(i, n_datasets, WORK_LO, WORK_HI)
            progress(ds_lo, f"mosaicity: {name}")
            # Release the PREVIOUS field's source before the next one is built.
            # `source` is only rebound by `_source_and_clim` below, i.e. *after*
            # the next field's alignment has allocated — and on the in-core rung
            # a source holds a whole aligned volume, so without this reset the
            # two coexist. (It holds one block on the streaming rung, where the
            # reset costs nothing and is still correct.) Rebinding rather than
            # `del` keeps the first iteration safe.
            source = None
            title, cbar, group = _display_info(name)
            cmap = resolve_cmap(style, group)
            # The file stays open for the whole of this field's work: `blocks`
            # slices the dataset, and every traversal — the colour-limit
            # reductions, the layer PNGs, the animation — happens inside here.
            # Nothing from the previous field survives into this one: the
            # aligned volume no longer exists as a local at all (a stream, or an
            # array that lives and dies inside `_process_dataset`).
            with h5py.File(mosa_file, "r") as f:
                dset = _mosa_dataset(f, name)
                if dset is None:
                    continue
                streamed, z_pos, scale_z = _align_streamed(
                    dset,
                    samy,
                    samz,
                    scale_x=scale_x,
                    samy_direction=samy_dir,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    budget_bytes=budget_bytes,
                )
                source, vmin, vmax = _source_and_clim(
                    streamed,
                    kind="com" if "Center_of_mass" in name else "fwhm",
                    center_method=p["center_method"],
                    range_pct=float(p["range_pct"]),
                )
                vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
                if clim_note:
                    progress(ds_lo, f"{name}: {clim_note}")
                prod = _process_dataset(
                    source,
                    z_pos,
                    scale_z,
                    name,
                    vmin,
                    vmax,
                    cmap,
                    title,
                    cbar,
                    p,
                    out_dir,
                    style=style,
                    group=group,
                    progress=_progress_mod.sub_progress(progress, ds_lo, ds_hi),
                )
            if clim_note:
                prod.notes.append(clim_note)
            result.datasets.append(prod)
        # And release the last field's source before the strain section aligns
        # its own, for the same reason: on the in-core rung both would otherwise
        # be live at once and the stage's peak would be the *sum* of the two
        # sections instead of their max. `prod` is already in result.datasets;
        # only the dead local binding goes. Rebinding rather than `del` keeps
        # this safe when the loop body never ran.
        source = prod = None
    elif mosa_file:
        result.skipped.append(f"mosaicity volume not found: {mosa_file}")

    # --- strain --- (the last dataset slice, not half the bar)
    if strain_file and os.path.exists(strain_file):
        st_lo, st_hi = _progress_mod.slice_for(n_datasets - 1, n_datasets, WORK_LO, WORK_HI)
        progress(st_lo, "loading strain volume")
        samy, samz = _read_motors(raw_root, p["strain_pattern"], p["samy_path"], p["samz_path"])
        with h5py.File(strain_file, "r") as f:
            dset = f["strain"] if "strain" in f else None
            if dset is not None:
                title, cbar, group = _display_info("strain", is_strain=True)
                cmap = resolve_cmap(style, group)
                streamed, z_pos, scale_z = _align_streamed(
                    dset,
                    samy,
                    samz,
                    scale_x=scale_x,
                    samy_direction=samy_dir,
                    roi_x=roi_x,
                    roi_y=roi_y,
                    budget_bytes=budget_bytes,
                )
                source, vmin, vmax = _source_and_clim(streamed, kind="strain")
                vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
                if clim_note:
                    progress(st_lo, f"strain: {clim_note}")
                prod = _process_dataset(
                    source,
                    z_pos,
                    scale_z,
                    "strain",
                    vmin,
                    vmax,
                    cmap,
                    title,
                    cbar,
                    p,
                    out_dir,
                    style=style,
                    group=group,
                    progress=_progress_mod.sub_progress(progress, st_lo, st_hi),
                )
                if clim_note:
                    prod.notes.append(clim_note)
                result.datasets.append(prod)
    elif strain_file:
        result.skipped.append(f"strain volume not found: {strain_file}")

    progress(1.0, f"visualized {len(result.datasets)} datasets -> {out_dir}")
    return result


# -----------------------------------------------------------------------------
# Single-field alignment (used by the GUI's lazy 3-D viewer)
# -----------------------------------------------------------------------------
def available_fields(params: dict) -> list[str]:
    """Field ids that can be aligned for 3-D, given the configured volume files."""
    p = {**STAGE.defaults(), **params}
    out: list[str] = []
    if p["mosa_volume_file"] and os.path.exists(p["mosa_volume_file"]):
        out.extend(mosa_field_names(p["mosa_volume_file"]))
    if p["strain_volume_file"] and os.path.exists(p["strain_volume_file"]):
        out.append("strain")
    return out


def aligned_field(params: dict, name: str):
    """Align a single field for display. Returns (volume, spacing_xyz, cmap, clim, meta).

    Reuses the exact alignment + centering the stage applies, so the 3-D view
    matches the rendered PNGs. ``meta`` is
    ``{"cbar_label": str, "group": str | None}``.

    **The return value cannot stream**, and that is a property of the consumer,
    not of this function: `gui/viewers.py::_visualize_load` hands the array to
    the 3-D viewer, which uploads the whole grid to VTK as one object. There is
    no per-layer or reduce-and-discard access to exploit, so a block iterator
    here would only be drained by the caller.

    What *was* streamable is where this function's peak actually sat — inside
    the alignment chain and the centring, not in the returned array. It now
    reads the field as a :class:`~dfxm.common.alignment.StreamedAlignment`,
    derives the colour limits and the centring offset from the streaming
    reductions, and materialises the volume exactly once with the offset already
    applied. So the peak is one volume plus one block, where it used to be the
    aligned volume, the ``data - center`` copy of it and the full-size
    ``data[isfinite(data)]`` selection the percentile made, all live together.
    """
    p = {**STAGE.defaults(), **params}
    scale_x, scale_y = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["roi_x"]), _parse_pair(p["roi_y"])
    raw_root = (p["raw_root"] or "").rstrip("/")
    budget_bytes = _run_budget_bytes(p)

    if name == "strain":
        samy, samz = _read_motors(raw_root, p["strain_pattern"], p["samy_path"], p["samz_path"])
        with h5py.File(p["strain_volume_file"], "r") as f:
            dset = f["strain"] if "strain" in f else None
            if dset is None:
                raise KeyError("strain dataset not found")
            streamed, _z, scale_z = _align_streamed(
                dset,
                samy,
                samz,
                scale_x=scale_x,
                samy_direction=samy_dir,
                roi_x=roi_x,
                roi_y=roi_y,
                budget_bytes=budget_bytes,
            )
            source, vmin, vmax = _source_and_clim(streamed, kind="strain")
            data = _materialise(source)
        cmap = resolve_cmap(None, "strain")
        meta = {"cbar_label": "Strain (ε)", "group": "strain"}
    else:
        samy, samz = _read_motors(raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"])
        with h5py.File(p["mosa_volume_file"], "r") as f:
            dset = _mosa_dataset(f, name)
            if dset is None:
                raise KeyError(name)
            streamed, _z, scale_z = _align_streamed(
                dset,
                samy,
                samz,
                scale_x=scale_x,
                samy_direction=samy_dir,
                roi_x=roi_x,
                roi_y=roi_y,
                budget_bytes=budget_bytes,
            )
            source, vmin, vmax = _source_and_clim(
                streamed,
                kind="com" if "Center_of_mass" in name else "fwhm",
                center_method=p["center_method"],
                range_pct=float(p["range_pct"]),
            )
            data = _materialise(source)
        _t, label, group = _display_info(name)
        cmap = resolve_cmap(None, group)
        meta = {"cbar_label": label, "group": group}
    return data, (scale_x, scale_y, scale_z), cmap, (float(vmin), float(vmax)), meta


def _make_build(loader, z, vn, vx, cmap_group, ex, ey, t, cb):
    """Factory: returns a build(style) closure for one layer of an aligned volume.

    ``loader`` is a zero-arg callable that returns the full aligned 3-D volume
    (cached per dataset by the caller).  ``z`` is captured by value via the
    default-arg trick so late-binding is not an issue. The colormap is resolved
    from *style* at build time via the dataset's quantity group.
    """

    def build(
        style, _loader=loader, _z=z, _vn=vn, _vx=vx, _grp=cmap_group, _ex=ex, _ey=ey, _t=t, _cb=cb
    ):
        vol = _loader()
        layer = vol[_z]
        fig, _ax, _im = Rnd.layer_figure(
            layer,
            _vn,
            _vx,
            resolve_cmap(style, _grp),
            _ex,
            _ey,
            f"{_t} (layer {_z})",
            _cb,
            style=style,
            group=_grp,
        )
        return fig

    return build


@register("visualize")
def figures(result: "VisualizeResult", params: dict) -> list[FigureSpec]:
    """Return one ``map`` FigureSpec per Z layer per dataset in the VisualizeResult.

    Each ``build(style)`` closure reproduces the aligned layer exactly as the
    stage does: load the source volume → ``_align`` (ROI + samy shift + uniform-Z
    interp) → slice layer *z* → ``render.layer_figure``.  Motor data is read from
    ``params`` (``raw_root`` + pattern), so the alignment matches the original run.

    The full alignment is performed AT MOST ONCE per dataset (lazy cache): listing
    specs is cheap; the first ``build()`` call for a dataset loads and aligns its
    volume, and all subsequent layer builds for that same dataset reuse the result.

    Pixel scales fall back to the calibrated beamline defaults (0.152 / 0.385 µm)
    when not supplied in *params*.
    """
    if not result.datasets:
        return []

    p = {**STAGE.defaults(), **params}
    sx = float(p["pixel_size_x_um"])
    sy = float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x = _parse_pair(p["roi_x"])
    roi_y = _parse_pair(p["roi_y"])
    raw_root = (p["raw_root"] or "").rstrip("/")

    specs: list[FigureSpec] = []

    for ds in result.datasets:
        is_strain = ds.name == "strain"
        title, cbar_label, group = _display_info(ds.name, is_strain=is_strain)
        n_z = ds.shape[0]
        ext_x = ds.shape[2] * sx
        ext_y = ds.shape[1] * sy

        # Per-dataset lazy cache: shared across all layer builds for THIS dataset.
        # First build() call fills cache["vol"]; the rest reuse it.
        cache: dict = {}

        if is_strain:
            src_file = p["strain_volume_file"]
            pattern = p["strain_pattern"]

            def _aligned_vol(
                src=src_file,
                pat=pattern,
                _cache=cache,
                _sx=sx,
                _sd=samy_dir,
                _rx=roi_x,
                _ry=roi_y,
                _rr=raw_root,
                _sp=p["samy_path"],
                _szp=p["samz_path"],
            ):
                if "vol" not in _cache:
                    raw = load_strain_volume(src)
                    if raw is None:
                        raise ValueError(f"strain dataset not found in {src!r}")
                    samy, samz = _read_motors(_rr, pat, _sp, _szp)
                    _cache["vol"], _zp, _sz = _align(
                        raw, samy, samz, scale_x=_sx, samy_direction=_sd, roi_x=_rx, roi_y=_ry
                    )
                return _cache["vol"]

        else:
            src_file = p["mosa_volume_file"]
            ds_name = ds.name
            pattern = p["mosa_pattern"]

            def _aligned_vol(
                src=src_file,
                name=ds_name,
                pat=pattern,
                _cache=cache,
                _sx=sx,
                _sd=samy_dir,
                _rx=roi_x,
                _ry=roi_y,
                _rr=raw_root,
                _sp=p["samy_path"],
                _szp=p["samz_path"],
                _cm=p["center_method"],
                _rp=float(p["range_pct"]),
            ):
                if "vol" not in _cache:
                    raw = load_mosa_field(src, name)
                    if raw is None:
                        raise KeyError(f"mosaicity dataset {name!r} not found in {src!r}")
                    samy, samz = _read_motors(_rr, pat, _sp, _szp)
                    vol, _zp, _sz = _align(
                        raw, samy, samz, scale_x=_sx, samy_direction=_sd, roi_x=_rx, roi_y=_ry
                    )
                    del raw
                    # CoM volumes are centred at run() time and ds.vmin/vmax were
                    # derived from the CENTRED data — reproduce that here so the
                    # export matches the saved PNG (and the 3-D viewer).
                    if "Center_of_mass" in name:
                        vol, _vn, _vx = _center_com_and_range(vol, _cm, _rp)
                    _cache["vol"] = vol
                return _cache["vol"]

        # Unique filename stem: sanitise the dataset name (spaces → _, slashes removed)
        stem = ds.name.replace("/", "_").replace(" ", "_")
        vmin_ds, vmax_ds = ds.vmin, ds.vmax

        for z in range(n_z):
            specs.append(
                FigureSpec(
                    figure_id=f"visualize_{stem}_z{z:04d}",
                    title=f"{title} — layer {z}",
                    kind="map",
                    filename=f"{stem}_layer_{z:04d}",
                    build=_make_build(
                        _aligned_vol, z, vmin_ds, vmax_ds, group, ext_x, ext_y, title, cbar_label
                    ),
                )
            )

    return specs


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Visualize aligned mosaicity/strain volumes.")
    ap.add_argument("--mosa-volume-file", default="")
    ap.add_argument("--strain-volume-file", default="")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--mosa-pattern", default="*")
    ap.add_argument("--strain-pattern", default="*")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--no-topview", action="store_true")
    args = ap.parse_args(argv)
    res = run(
        dict(
            mosa_volume_file=args.mosa_volume_file,
            strain_volume_file=args.strain_volume_file,
            raw_root=args.raw_root,
            mosa_pattern=args.mosa_pattern,
            strain_pattern=args.strain_pattern,
            output_dir=args.output_dir,
            save_topview=not args.no_topview,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\n{len(res.datasets)} datasets -> {res.output_dir}; skipped {len(res.skipped)}")
    return 0


def roi_previews(params: dict) -> list:
    """(label, thunk) ROI-picker previews from the stacked mosa/strain volume(s)."""
    from ..common.figures import stacked_volume_previews

    return stacked_volume_previews(params)


if __name__ == "__main__":
    raise SystemExit(_main())
