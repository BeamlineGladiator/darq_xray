"""Oblique slices stage — arbitrary planar cuts through aligned DFXM volumes.

Faithful port of ``extract_oblique_slices_v5.py``. It samples arbitrary planes
(defined by a normal + origin in physical µm, optionally swept along the normal)
from two volume categories and writes a consolidated ``oblique_slices.h5`` (the
file the line-profile stage consumes) plus a PNG per plane:

* ``source = "stacked"`` — mosaicity/strain stacks; this stage runs the SAME
  alignment as the visualize/PVTI stages (reusing :mod:`dfxm.common.alignment`)
  so the slices live in the origin-0 PVTI world frame.
* ``source = "aligned"`` — already-aligned raw rocking volumes; loaded directly
  with their stored spacing/attrs.

The set of volumes to slice is the standard pipeline output (toggled on/off);
the planes themselves are given as JSON (``slices_json``) — natural since
oblique normals/origins are typically copied from ParaView. ``extent="auto"``
fits a plane (and its sweep) to the common data bounding box across all volumes.

One deliberate deviation from the legacy script: the default plane ``up`` is
world Y — the detector-vertical axis (lab-frame X) — not Z, so slice plots are
oriented like the per-layer renders (detector-X-like horizontal, detector-Y
vertical) — see :func:`build_basis`.
"""

from __future__ import annotations

import contextlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.figure import Figure
from scipy.ndimage import map_coordinates

from ..common import alignment as A
from ..common import render as Rnd
from ..common import volumeio as V
from ..common.errors import StageUserError
from ..common.figures import FigureSpec, crop_roi_2d, register, resolve_clim
from ..common.h5io import iter_dataset_sizes, sum_dataset_bytes
from ..common.plotting import (
    GROUP_BY_KIND,
    PlotStyle,
    add_colorbar,
    apply_axes_mode,
    apply_round_clim,
    apply_text_scale,
    draw_scale_bar,
    figure_size,
    fit_axes_to_box,
    fixed_scale_box,
    resolve_cmap,
    style_from_params,
    styled_figure,
)
from ..common.raster import extract_motor_positions
from ..common.sort import find_matching_folders
from ..config.models import CostEstimate, Param, ParamType, SeeAlso, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


# Root-level group holding starred plane offsets (/marks/<slice_name>); every
# enumerator of oblique_slices.h5 root groups must skip it.
MARKS_GROUP = "marks"

# What a child running this stage costs resident before it touches a voxel:
# interpreter, numpy, scipy, h5py and matplotlib. `tracemalloc` cannot see any
# of it, so it is what
# :func:`~dfxm.common.advice.working_set_budget_bytes` must take off the
# machine's headroom before converting the rest into an allocation budget.
#
# Measured with `tests/peak_rss.py::measure_process_floor`, three samples, on a
# 4x6x8 seven-volume run — 9 kB of data — and pinned by
# `test_rss_floor_covers_the_measured_process_image`, which brackets the constant
# from both sides so it can be neither too low nor inflated to silence the check:
#
#     save_png = False (no matplotlib figure ever built)   103.9 MiB
#     save_png = True  (the default)                       193.5 MiB
#
# The **PNGs-on** figure is the right one: it is the default, and the first
# figure a run builds imports the whole of matplotlib's rendering stack, which
# every volume after it then streams with resident.
#
# **Not visualize's 768 MB and not paraview's 300 MB.** Those two import
# pyvista/VTK; this stage never does, so it sits below both, and pasting either
# number fails the assertion rather than silently mis-sizing the budget.
#
# The declared value carries ~1.3x slack over the measurement, the same ratio the
# other two carry, because the additive RSS model is not an envelope (see
# `advice.MARGINAL_RSS_PER_TRACED_BYTE`) and the floor is the term with room to
# absorb that: over-stating it only shrinks the budget, under-stating it invites
# an OOM.
RSS_FLOOR_BYTES = 256 * 1024 * 1024

# The whole run's working-set budget divided among the things that hold a block
# at once. `align_volume_streamed`'s model prices the alignment chain and
# **nothing downstream**, while the colour-limit reductions layered on its blocks
# hold, per float64 element at their peak, `_finite64`, the rank search's window,
# the `searchsorted`/`clip` index arrays and the `isfinite` mask — the
# `dtype.itemsize + 8 * (retained + 1) + 1` = 41 B/element that
# `align_volume_streamed` computes for its own cached median, i.e. 5.125 blocks —
# plus one more for the centred `b - center` the stream yields through. 6.125,
# rounded **up**: the constant is a *divisor*, so a larger number means a smaller
# budget, and rounding down would permit more than was counted. (`visualize`
# shipped 6 for the same count and had to be corrected to 7 for exactly this
# reason.) Dividing by a call site's concurrent consumers is what
# `paraview._process_mosaicity` does for its concurrent field streams; it is not
# a correction to what `budget_bytes` means.
REDUCTION_WORKING_SET_MULTIPLE = 7

# What the Z-blocked gather holds *per plane it is filling*, in units of that
# plane's own float32 bytes. Per element of the coordinate rectangle it builds:
# `coords` is 3 x float64 (6 float32-plane-equivalents), the `local` gather it
# stacks out of them is another 3 x float64 (6) when the whole rectangle is
# selected, `map_coordinates`' float64 result is one more (2) and the boolean
# `sel` mask is 1 byte (0.25) — 14.25, rounded **up**. It multiplies the plane
# size and is then *subtracted* from the budget, so rounding down would leave a
# plane batch larger than what was counted; the in-core sampler holds the same
# rectangle (minus `local`) for the one plane it is on.
GATHER_SCRATCH_PLANE_MULTIPLE = 15

_DEFAULT_SLICES = json.dumps(
    [
        {
            "name": "z_sweep",
            "normal": [0, 0, 1],
            "origin": [0, 0, 0],
            "extent": "auto",
            "sweep_step_um": 5.0,
        },
        {
            "name": "oblique_full",
            "normal": [0.647648, 0, 0.761939],
            "origin": [0, 0, 0],
            "extent": "auto",
            "sweep_step_um": 2.0,
        },
    ],
    indent=2,
)

# Standard sliceable volumes: (toggle param, source, file param, dataset, kind).
# The colormap is resolved per kind via GROUP_BY_KIND + the active PlotStyle.
_STD_VOLUMES = (
    ("include_mosa_com_chi", "stacked", "mosa_volume_file", "chi/Center of mass", "mosa_com"),
    ("include_mosa_fwhm_chi", "stacked", "mosa_volume_file", "chi/FWHM", "mosa_fwhm"),
    ("include_mosa_com_mu", "stacked", "mosa_volume_file", "mu/Center of mass", "mosa_com"),
    ("include_mosa_fwhm_mu", "stacked", "mosa_volume_file", "mu/FWHM", "mosa_fwhm"),
    ("include_strain", "stacked", "strain_volume_file", "strain", "strain"),
    ("include_raw_sum", "aligned", "aligned_rocking_file", "sum_intensity", "raw_sum"),
    ("include_raw_specific", "aligned", "aligned_rocking_file", "specific_frame", "raw_specific"),
    ("include_mosa_sum", "aligned", "aligned_mosa_file", "sum_intensity", "raw_mosa_sum"),
    (
        "include_mosa_specific",
        "aligned",
        "aligned_mosa_file",
        "specific_frame",
        "raw_mosa_specific",
    ),
)


STAGE = StageSpec(
    name="slices",
    label="Oblique slices",
    description=(
        "Cuts arbitrary planes — defined in physical µm, optionally swept along their normal — "
        "through all aligned volumes at once, so every quantity is sampled at identical "
        "positions. Writes oblique_slices.h5 (used by profiles) plus a PNG per plane."
    ),
    params=(
        Param(
            "mosa_volume_file",
            ParamType.PATH,
            "Mosaicity volume",
            must_exist=True,
            help=(
                "The stacked mosaicity volume (stacked_volumes.h5) from the mosaicity stage. "
                "Leave blank to skip mosaicity fields."
            ),
        ),
        Param(
            "strain_volume_file",
            ParamType.PATH,
            "Strain volume",
            must_exist=True,
            help=(
                "The stacked strain volume (stacked_strain_volumes.h5) from the strain stage. "
                "Leave blank to skip strain."
            ),
        ),
        Param(
            "aligned_rocking_file",
            ParamType.PATH,
            "Aligned rocking volume",
            must_exist=True,
            help=(
                "The aligned rocking volume (aligned_raw_rocking_volumes.h5) from the rocking "
                "stage. Leave blank to slice without raw intensity."
            ),
        ),
        Param(
            "aligned_mosa_file",
            ParamType.PATH,
            "Aligned mosa volume",
            must_exist=True,
            help=(
                "The aligned mosa-sum volume (aligned_raw_mosa_volumes.h5) from the rocking stage "
                "run with Source scan = mosaicity. Leave blank to skip the mosa raw fields."
            ),
        ),
        Param(
            "raw_root",
            ParamType.DIR,
            "Raw data root",
            must_exist=True,
            help=(
                "RAW_DATA root with the original scan folders — provides the samy/samz "
                "positions used to align the stacked volumes."
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
                "calibration — sets the physical scale the planes are defined in and, during "
                "alignment, converts the sample-Y motor shift (mm) into detector pixels. A wrong "
                "value misregisters the volumes and puts every plane at the wrong place."
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
                "calibration — sets the physical scale the planes are defined in. A wrong value "
                "skews every plane's vertical extent."
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
            "align_roi_x",
            ParamType.STR,
            "Align ROI X",
            default="",
            advanced=True,
            group="Alignment",
            roi_group="crop",
            roi_axis="x",
            roi_frame="map",
            help=(
                "Map-frame crop 'c0,c1' (map pixels, relative to the darfix window) used "
                "during alignment — must match the crop used when the volumes were "
                "rendered/exported (blank = full width). Pre-filled from the experiment's "
                "analysis window."
            ),
        ),
        Param(
            "align_roi_y",
            ParamType.STR,
            "Align ROI Y",
            default="",
            advanced=True,
            group="Alignment",
            roi_group="crop",
            roi_axis="y",
            roi_frame="map",
            help=(
                "Map-frame crop 'r0,r1' (map pixels, relative to the darfix window) used "
                "during alignment — must match the crop used when the volumes were "
                "rendered/exported (blank = full height). Pre-filled from the experiment's "
                "analysis window."
            ),
        ),
        Param(
            "abs_fwhm",
            ParamType.BOOL,
            "abs() FWHM",
            default=True,
            advanced=True,
            group="Alignment",
            help="Use absolute FWHM values (darfix fits can produce negative widths).",
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
                "How the colour scale of the misorientation (CoM) fields is centred: "
                "midrange = midpoint of the robust limits, or mean/median. Display only."
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
                "Robust percentile for colour limits, e.g. 99.5 ignores the most "
                "extreme 0.5 % of pixels."
            ),
        ),
        Param(
            "include_mosa_com_chi",
            ParamType.BOOL,
            "Slice χ misorientation",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the χ misorientation (centre-of-mass) volume.",
        ),
        Param(
            "include_mosa_fwhm_chi",
            ParamType.BOOL,
            "Slice χ FWHM",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the χ FWHM (rocking-curve width) volume.",
        ),
        Param(
            "include_mosa_com_mu",
            ParamType.BOOL,
            "Slice μ misorientation",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the μ misorientation (centre-of-mass) volume.",
        ),
        Param(
            "include_mosa_fwhm_mu",
            ParamType.BOOL,
            "Slice μ FWHM",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the μ FWHM (curve width) volume.",
        ),
        Param(
            "include_strain",
            ParamType.BOOL,
            "Slice strain",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the axial strain volume.",
        ),
        Param(
            "include_raw_sum",
            ParamType.BOOL,
            "Slice raw sum",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the summed raw rocking intensity volume.",
        ),
        Param(
            "include_raw_specific",
            ParamType.BOOL,
            "Slice raw specific",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the specific-frame raw intensity volume.",
        ),
        Param(
            "include_mosa_sum",
            ParamType.BOOL,
            "Slice mosa sum",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the mosa-scan summed intensity volume (from the aligned mosa file).",
        ),
        Param(
            "include_mosa_specific",
            ParamType.BOOL,
            "Slice mosa specific",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the mosa-scan specific-frame intensity volume.",
        ),
        Param(
            "slices_json",
            ParamType.TEXT,
            "Slices (JSON)",
            default=_DEFAULT_SLICES,
            help=(
                "JSON list of plane definitions. Each needs a name and a normal vector; "
                "'extent': 'auto' fits the plane to the data, and 'sweep_step_um' adds "
                "parallel planes along the normal. The default shows the format."
            ),
        ),
        Param(
            "use_pinned",
            ParamType.BOOL,
            "Run pinned planes only",
            default=False,
            help=(
                "Render only the planes in 'Pinned planes (JSON)' instead of the full sweep — "
                "fast re-computation of a few interesting planes. The sweep in 'Slices (JSON)' "
                "is kept untouched and ignored while this is on; while on, the default output "
                "filename becomes oblique_slices_pinned.h5 so the sweep file is never "
                "overwritten. Untick to run the full sweep again."
            ),
        ),
        Param(
            "pinned_slices_json",
            ParamType.TEXT,
            "Pinned planes (JSON)",
            default="",
            advanced=True,
            group="Pinned planes",
            help=(
                "JSON list of pinned single-plane specs, normally written by the Pin planes… "
                "dialog (exact stored sweep geometry, snapped to stored planes). Only used "
                "when 'Run pinned planes only' is ticked — blank there raises an error asking "
                "you to open Pin planes… or untick it."
            ),
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help="Where oblique_slices.h5 and the per-plane PNGs are written.",
        ),
        Param(
            "output_h5_name",
            ParamType.STR,
            "Output filename",
            default="oblique_slices.h5",
            advanced=True,
            group="Output",
            help=(
                "Filename of the consolidated slices file. "
                "The profiles stage expects oblique_slices.h5. "
                "While 'Run pinned planes only' is on, this default is replaced by "
                "oblique_slices_pinned.h5 (an edited name is respected)."
            ),
        ),
        Param(
            "save_png",
            ParamType.BOOL,
            "Save PNGs",
            default=True,
            advanced=True,
            group="Output",
            help="Write a PNG per plane in addition to the HDF5.",
        ),
    ),
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” (left panel), not here.",
        ),
    ),
    estimate="dfxm.stages.slices:estimate",
)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
@dataclass
class SlicesResult:
    output_dir: str = ""
    output_h5: str | None = None
    volume_ids: list[str] = field(default_factory=list)
    slice_names: list[str] = field(default_factory=list)
    n_planes_total: int = 0
    pngs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class ReplotEntry:
    volume_id: str
    slice_name: str
    n_planes: int
    offsets_um: list[float] = field(default_factory=list)
    shape: tuple[int, int] | None = None  # stored plane (nv, nu) pixel shape (ROI-crop hint)
    group: str = ""  # kind-group (GROUP_BY_KIND[kind]) — keys per-kind clim overrides


# -----------------------------------------------------------------------------
# Centering / colour-range helpers (faithful port)
# -----------------------------------------------------------------------------
# -- ONE DEFINITION PER STATISTIC, FED BY WHICHEVER RUNG IS RUNNING ------------
#
# `prepare_volume` picks between two rungs by asking how much memory the machine
# has: it either materialises the aligned volume or walks it as a stream of
# Z-blocks. A statistic that differed between the two would be a statistic that
# depended on the machine — the laptop-versus-workstation divergence this phase
# exists to remove, reintroduced at the seam the phase created.
#
# `visualize` holds that equality by keeping two implementations in step and
# pinning them with a fixture. Here there is nothing to keep in step: each of the
# four helpers below takes a **blocks factory** and there is exactly one
# implementation, so the rung decides only where the blocks come from — slices of
# a resident array on the in-core rung, the alignment stream on the other. The
# equality is then `volumeio`'s own budget-independence guarantee, which is
# asserted in that module rather than restated here.
#
# What that unification cost, stated because it is a real (if tiny) change of
# product: `_center_offset("mean")` was `np.nanmean` and is now
# `volumeio.stream_mean`, a compensated (Neumaier) sum. The two disagree by about
# an ulp on realistic data — the drift `alignment._center_offset` already took
# project-wide for the same reason, and on the more accurate side. `median` and
# every percentile here are bit-equal to their numpy originals, and
# `center_method` defaults to `midrange`, which uses no mean at all.
def _arrays(blocks):
    """A factory over the bare arrays of a ``(slice, array)`` block factory."""
    return lambda: (block for _sl, block in blocks())


def _shifted(blocks, offset):
    """*blocks* with a constant subtracted — the streaming ``data - center``.

    The ``if not offset`` short-circuit is
    :func:`~dfxm.common.alignment.align_volume_streamed`'s own, for the same
    reason: subtracting a zero allocates a second array per block and changes no
    value.
    """
    if not offset:
        return blocks
    return lambda: ((sl, block - offset) for sl, block in blocks())


def _center_offset(blocks, method):
    """The centring statistic over the finite voxels of *blocks*.

    ``0.0`` when nothing finite is present, which is what the array form
    returned for an empty selection.

    ``±inf`` **voxels are dropped**, not averaged in: these reductions filter on
    ``np.isfinite``, exactly as the array form's ``data[np.isfinite(data)]``
    selection did before them. So the guard keys on ``np.isnan`` and not on
    ``not np.isfinite``: NaN is this reduction's "no finite voxel anywhere"
    signal and must become the array form's ``0.0``, whereas an infinity
    reaching the *result* can only be an overflow of the finite values'
    own sum — a real statistic, which the array form would also have returned
    and which is therefore passed through rather than silently zeroed.
    """
    if method == "mean":
        val = V.stream_mean(_arrays(blocks)())
    else:
        val = V.stream_quantile(_arrays(blocks), 50.0)
    return 0.0 if np.isnan(val) else float(val)


def _symmetric_range(blocks, pct=99):
    """Symmetric limits at the *pct*-th percentile of ``|value|``.

    ``np.abs`` runs before the finite filter rather than after it, which selects
    the same values — ``abs`` maps NaN to NaN and ``±inf`` to ``inf``, both
    dropped either way — while keeping the whole computation inside one
    traversal-driven reduction.
    """
    am = V.stream_quantile(lambda: (np.abs(b) for b in _arrays(blocks)()), float(pct))
    if np.isnan(am):
        return (-1.0, 1.0)
    return (-float(am), float(am))


def _midrange_clim(blocks, pct=99.5):
    """``(center, (vmin, vmax))`` from the robust ``[100 - pct, pct]`` pair.

    Slices is the only stage offering ``midrange``, so the convention stays
    here rather than moving into shared alignment code.
    """
    arrays = _arrays(blocks)
    if pct >= 100.0:
        lo, hi = V.stream_minmax(arrays())
    else:
        lo = V.stream_quantile(arrays, 100.0 - pct)
        hi = V.stream_quantile(arrays, pct)
    if np.isnan(lo):
        return 0.0, (-1.0, 1.0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) or 1.0
    return center, (-half, half)


def _percentile_range(blocks, lo=1, hi=99):
    """The *lo*-th and *hi*-th percentiles of the finite voxels of *blocks*."""
    arrays = _arrays(blocks)
    v_lo = V.stream_quantile(arrays, float(lo))
    if np.isnan(v_lo):
        return (0.0, 1.0)
    return (float(v_lo), float(V.stream_quantile(arrays, float(hi))))


def _parse_pair(text):
    if text is None or str(text).strip() == "":
        return None
    parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'a,b', got {text!r}")
    return tuple(parts)


# -----------------------------------------------------------------------------
# Geometry + sampling (faithful port)
# -----------------------------------------------------------------------------
def build_basis(normal, up=None):
    """Orthonormal (u_hat, v_hat, n_hat) for the plane; vectors are (X, Y, Z).

    u_hat is the plot's horizontal axis and v_hat its vertical axis. The default
    up is world Y — the detector-vertical axis (lab-frame X) — so slices read
    like the per-layer renders (detector-X-like horizontal, detector-Y
    vertical); pass an explicit ``up`` per slice to override.
    """
    n = np.asarray(normal, dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        raise ValueError("normal vector has zero length")
    n_hat = n / nn
    if up is None:
        up_vec = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(n_hat, up_vec)) > 0.99:
            up_vec = np.array([0.0, 0.0, 1.0])
    else:
        up_vec = np.asarray(up, dtype=np.float64)
    v = up_vec - np.dot(up_vec, n_hat) * n_hat
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, n_hat)) > 0.99:
            tmp = np.array([0.0, 1.0, 0.0])
        v = tmp - np.dot(tmp, n_hat) * n_hat
        vn = np.linalg.norm(v)
    v_hat = v / vn
    u_hat = np.cross(v_hat, n_hat)
    u_hat = u_hat / np.linalg.norm(u_hat)
    return u_hat, v_hat, n_hat


def slice_plane_offsets(sl):
    step = sl.get("sweep_step_um")
    if not step or step <= 0:
        return np.array([0.0])
    start = float(sl.get("sweep_start_um", 0.0))
    stop = float(sl.get("sweep_stop_um", start))
    if stop < start:
        start, stop = stop, start
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    return start + np.arange(n) * step


def _plane_axes(half_u, half_v, du, dv):
    """The plane's ``(u_um, v_um)`` sample axes — its grid, and nothing else."""
    nu = max(1, int(np.round(2.0 * half_u / du)) + 1)
    nv = max(1, int(np.round(2.0 * half_v / dv)) + 1)
    return np.linspace(-half_u, half_u, nu), np.linspace(-half_v, half_v, nv)


def _plane_coords(prep, plane_origin, u_hat, v_hat, u_um, v_um):
    """Voxel coordinates ``(3, len(v_um), len(u_um))`` — ``(k, j, i)`` — for one plane.

    Factored out of :func:`sample_plane` so the in-core sampler and the
    Z-blocked gather share it **verbatim**: the coordinates are not what changes
    between the two, and computing them twice in two places is how the two would
    drift apart.

    Every operation is elementwise over ``u_um``/``v_um``, so evaluating this on
    a sub-rectangle of the grid gives bit-identical numbers for that
    sub-rectangle — which is what lets the gather price only the rows and
    columns a Z-block can reach.
    """
    uu, vv = np.meshgrid(u_um, v_um)
    pts = (
        np.asarray(plane_origin, np.float64)[None, None, :]
        + uu[..., None] * u_hat[None, None, :]
        + vv[..., None] * v_hat[None, None, :]
    )
    i = pts[..., 0] / prep["scale_x"] + prep["x_ref_shift_px"]
    j = pts[..., 1] / prep["scale_y"] + prep["y_ref_shift_px"]
    k = (pts[..., 2] - prep.get("z_ref_shift_um", 0.0)) / prep["scale_z"]
    return np.stack([k, j, i], axis=0)


def sample_plane(prep, plane_origin, u_hat, v_hat, half_u, half_v, du, dv):
    """Sample one plane centred at plane_origin (X,Y,Z µm). Returns (slice, u_um, v_um).

    The in-core sampler: it reads ``prep["data"]``, so it needs the whole
    aligned volume resident. :func:`sample_planes_streamed` is the bounded-memory
    counterpart, and :func:`iter_sample_planes` picks between them.
    """
    u_um, v_um = _plane_axes(half_u, half_v, du, dv)
    coords = _plane_coords(prep, plane_origin, u_hat, v_hat, u_um, v_um)
    s = map_coordinates(prep["data"], coords, order=1, mode="constant", cval=np.nan)
    return s.astype(np.float32), u_um, v_um


def _plane_k(prep, plane_origin, u_hat, v_hat, u_um, v_um):
    """Just the ``k`` (Z voxel) coordinate of :func:`_plane_coords`, for probing."""
    uu, vv = np.meshgrid(np.asarray(u_um, np.float64), np.asarray(v_um, np.float64))
    origin = np.asarray(plane_origin, np.float64)
    z = origin[2] + uu * float(u_hat[2]) + vv * float(v_hat[2])
    return (z - prep.get("z_ref_shift_um", 0.0)) / prep["scale_z"]


# How far outside a block's own Z range a probe may look before the plane is
# skipped for that block. `k` is affine in (u, v), so its extremes over the grid
# sit on the grid's edges — but they are computed in floating point, so an
# interior sample can exceed an edge probe by an ulp. One whole voxel of slack
# is many orders of magnitude more than that, and costs at most one extra
# (already cheap) rectangle per block.
_PROBE_SLACK_LAYERS = 1.0


def _span_in_block(low, high, z0: int, z1: int) -> tuple[int, int] | None:
    """The index range of *low*/*high* whose ``k`` interval can meet ``[z0, z1)``.

    *low* and *high* are per-index bounds on ``k`` along one grid axis. Returns
    ``None`` when no index qualifies. NaN bounds compare false and are excluded,
    which is right: a sample with a NaN coordinate belongs to no block and keeps
    the NaN the output starts as, exactly as ``cval=np.nan`` gives it in-core.
    """
    hits = (high >= z0 - _PROBE_SLACK_LAYERS) & (low < z1 + _PROBE_SLACK_LAYERS)
    if not hits.any():
        return None
    idx = np.flatnonzero(hits)
    # `[first, last + 1]` rather than the mask itself: the qualifying set is an
    # interval for an affine `k`, and taking its hull keeps the answer a basic
    # slice (a view) even if rounding were to punch a hole in the mask.
    return int(idx[0]), int(idx[-1]) + 1


def _gather_batch(prep, origins, u_hat, v_hat, u_um, v_um):
    """One walk of ``prep["blocks"]``, filling every plane in *origins*.

    The batch is the memory knob: the walk is shared by however many planes are
    handed to it, so a whole sweep costs one traversal and holds every plane,
    while a sweep split into *b* batches costs *b* traversals and holds only a
    batch. :func:`iter_planes_streamed` chooses *b*; the arithmetic below does
    not depend on it — a plane's samples come from the blocks its ``k`` lands
    in, and which other planes travelled with it is not an input.
    """
    outs = [np.full((len(v_um), len(u_um)), np.nan, dtype=np.float32) for _ in origins]
    # Per plane, `k` along the grid's own edges: rows probed at the two extreme
    # columns and columns probed at the two extreme rows. Two 1-D bounds per
    # plane, so the probes cost O(rows + columns) and are held for the whole
    # traversal without ever approaching the size of an image.
    probes = []
    for origin in origins:
        row_k = _plane_k(prep, origin, u_hat, v_hat, u_um[[0, -1]], v_um)
        col_k = _plane_k(prep, origin, u_hat, v_hat, u_um, v_um[[0, -1]])
        probes.append((row_k.min(axis=1), row_k.max(axis=1), col_k.min(axis=0), col_k.max(axis=0)))

    for interior, window, _within in V.iter_with_context(prep["blocks"](), trailing=1):
        z0, z1 = int(interior.start), int(interior.stop)
        for out, origin, (row_lo, row_hi, col_lo, col_hi) in zip(outs, origins, probes):
            rows = _span_in_block(row_lo, row_hi, z0, z1)
            if rows is None:
                continue
            cols = _span_in_block(col_lo, col_hi, z0, z1)
            if cols is None:
                continue
            r0, r1 = rows
            c0, c1 = cols
            coords = _plane_coords(prep, origin, u_hat, v_hat, u_um[c0:c1], v_um[r0:r1])
            k, j, i = coords[0], coords[1], coords[2]
            sel = (k >= z0) & (k < z1)
            if not sel.any():
                continue
            local = np.stack([k[sel] - z0, j[sel], i[sel]], axis=0)
            out[r0:r1, c0:c1][sel] = map_coordinates(
                window, local, order=1, mode="constant", cval=np.nan
            )
    return outs


def sweep_batch_size(n_planes: int, plane_bytes: int, max_resident_bytes: int | None) -> int:
    """How many planes of a sweep may be gathered in one walk of the Z-blocks.

    ``None`` means "no bound" — the whole sweep in one walk, which is what
    :func:`sample_planes_streamed` and every caller that already holds the
    planes anyway wants. Otherwise it is the number of whole planes that fit in
    *max_resident_bytes*, and never less than one: a single plane is the
    irreducible unit of the gather, so a budget too small for one is reported by
    the measured peak rather than honoured by returning nothing.
    """
    if max_resident_bytes is None:
        return max(1, int(n_planes))
    fit = int(max_resident_bytes) // max(1, int(plane_bytes))
    return max(1, min(int(n_planes), fit))


def iter_planes_streamed(
    prep, plane_origins, u_hat, v_hat, half_u, half_v, du, dv, *, max_resident_bytes=None
):
    """Yield each plane of one sweep, gathered from Z-blocks, in the order given.

    ``max_resident_bytes`` caps the **finished planes** held at once; the sweep
    is split into batches of :func:`sweep_batch_size` planes and each batch costs
    one walk of ``prep["blocks"]``. That is the trade this knob makes and the
    only thing it changes: *b* batches re-read the volume *b* times, and the
    values are identical either way (see :func:`_gather_batch`).

    ``None`` (the default) is one batch — one walk, every plane resident, which
    is what :func:`sample_planes_streamed` returns.
    """
    u_um, v_um = _plane_axes(half_u, half_v, du, dv)
    origins = [np.asarray(o, np.float64) for o in plane_origins]
    plane_bytes = len(u_um) * len(v_um) * 4
    step = sweep_batch_size(len(origins), plane_bytes, max_resident_bytes)
    for start in range(0, len(origins), step):
        outs = _gather_batch(prep, origins[start : start + step], u_hat, v_hat, u_um, v_um)
        for idx in range(len(outs)):
            plane, outs[idx] = outs[idx], None  # drop the batch's own reference
            yield plane


def sample_planes_streamed(prep, plane_origins, u_hat, v_hat, half_u, half_v, du, dv):
    """Every plane of one sweep, gathered from Z-blocks instead of a whole volume.

    Returns ``(planes, u_um, v_um)`` — one ``float32`` image per origin, in the
    order given — and walks ``prep["blocks"]`` **once** for the whole sweep. The
    output of a plane is a small 2-D image whatever the volume's size, so one
    pass over Z can serve every requested plane at the same time; sampling them
    one at a time would re-run the alignment chain once per plane.

    This is the unbounded face of :func:`iter_planes_streamed`: it holds the
    whole sweep. ``run`` uses the iterator instead, because on the shipped
    default geometry the sweep is the largest array the stage touches.

    Why this is bit-identical to :func:`sample_plane` rather than merely close.
    ``map_coordinates`` with ``order=1`` reads layers ``floor(k)`` and
    ``floor(k) + 1`` and weights them by the fraction, so a sample belongs to the
    block whose interior contains ``floor(k)``, and that block needs exactly one
    row of its successor to be self-sufficient — which is what
    :func:`~dfxm.common.volumeio.iter_with_context` supplies. A sample assigned to
    block ``[z0, z1)`` therefore reads the same two rows with the same weights as
    the in-core call, and non-final blocks never touch a boundary condition at
    all. The final block has no successor, so its window ends exactly where the
    volume ends and scipy applies there the identical out-of-bounds rule it
    applies in-core. A sample whose ``floor(k)`` lands in no block is outside the
    volume in Z and keeps the NaN the output starts as, which is what in-core
    ``cval`` gives it.

    ``floor(k) in [z0, z1)`` is spelled ``z0 <= k < z1``: for integer bounds the
    two are the same set, NaN included (both comparisons are false). It is
    **not** an integer cast — ``int(-0.5)`` truncates toward zero and would
    assign a sample above the volume's floor to block 0.

    ``out[…] = <float64>`` writes into a ``float32`` image, which rounds each
    sample exactly once — the same single rounding :func:`sample_plane`'s
    trailing ``.astype(np.float32)`` applies, since each sample is written by
    exactly one block.

    Cost, stated because it is the streaming rung's real price: a plane is
    priced against a block by probing ``k`` along the grid's edges (O(rows +
    columns)) and then computing coordinates only for the bounding rectangle of
    the rows and columns that block can reach. For the two plane families this
    stage ships — a Z-normal sweep (``k`` constant over the plane) and a sweep
    tilted in X–Z with ``up`` = world Y (``k`` varying only along ``u``) — that
    rectangle is tight, and each plane's coordinates are effectively computed
    once across the whole traversal. A plane tilted in *both* in-plane
    directions has a diagonal band whose bounding rectangle is not tight, and
    then coordinates are recomputed per block. That is work, not memory: the
    peak is one block, one window and one rectangle of coordinates however many
    planes the sweep holds.
    """
    u_um, v_um = _plane_axes(half_u, half_v, du, dv)
    planes = list(iter_planes_streamed(prep, plane_origins, u_hat, v_hat, half_u, half_v, du, dv))
    return planes, u_um, v_um


def sample_plane_streamed(prep, plane_origin, u_hat, v_hat, half_u, half_v, du, dv):
    """One plane, gathered from Z-blocks. The single-plane face of
    :func:`sample_planes_streamed`; a sweep should call that directly so the
    stream is walked once rather than once per plane."""
    planes, u_um, v_um = sample_planes_streamed(
        prep, [plane_origin], u_hat, v_hat, half_u, half_v, du, dv
    )
    return planes[0], u_um, v_um


def iter_sample_planes(
    prep, plane_origins, u_hat, v_hat, half_u, half_v, du, dv, *, max_resident_bytes=None
):
    """Yield each plane of one sweep, by whichever rung *prep* was prepared on.

    ``prep["data"]`` is the resident aligned volume on the in-core rung and
    ``None`` on the streaming one; see :func:`prepare_volume` for how the rung
    is chosen. The two yield identical images — pinned by
    ``test_slices_streamed_gather_matches_in_core``.

    A **generator**, not a list, because a sweep's planes are the largest array
    this stage handles: on the shipped default geometry ``oblique_full`` is
    ~172 planes of 2528x1789 float32, i.e. 2.90 GiB, against a 1.15 GiB volume.
    ``run`` writes each plane into its HDF5 dataset as it arrives, so only the
    plane in flight is ever held.

    The in-core rung needs no batching — :func:`sample_plane` produces one plane
    per call from a volume that is already resident — so ``max_resident_bytes``
    applies to the streaming rung only, where planes come out of a shared walk
    of the Z-blocks; see :func:`iter_planes_streamed`.

    ``u_um``/``v_um`` are not returned: they are :func:`_plane_axes` of the same
    four arguments, which any caller can compute for itself without sampling.
    """
    if prep.get("data") is not None:
        for origin in plane_origins:
            yield sample_plane(prep, origin, u_hat, v_hat, half_u, half_v, du, dv)[0]
        return
    yield from iter_planes_streamed(
        prep,
        plane_origins,
        u_hat,
        v_hat,
        half_u,
        half_v,
        du,
        dv,
        max_resident_bytes=max_resident_bytes,
    )


def _world_box(shape, sx, sy, sz, xshift, yshift, zshift):
    nz, ny, nx = shape
    xs = [(0 - xshift) * sx, (nx - 1 - xshift) * sx]
    ys = [(0 - yshift) * sy, (ny - 1 - yshift) * sy]
    zs = [zshift + 0 * sz, zshift + (nz - 1) * sz]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _union_box(boxes):
    a = np.array(boxes)
    return (
        a[:, 0].min(),
        a[:, 1].max(),
        a[:, 2].min(),
        a[:, 3].max(),
        a[:, 4].min(),
        a[:, 5].max(),
    )


def resolve_auto_extent(sl, box, default_du=0.152):
    if sl.get("extent") != "auto":
        return sl
    u_hat, v_hat, n_hat = build_basis(sl["normal"], sl.get("up"))
    origin = np.asarray(sl["origin"], dtype=np.float64)
    x0, x1, y0, y1, z0, z1 = box
    corners = (
        np.array(
            [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float64
        )
        - origin[None, :]
    )
    u_p, v_p, n_p = corners @ u_hat, corners @ v_hat, corners @ n_hat
    margin = float(sl.get("auto_margin_um", 0.0))
    out = dict(sl)
    u_c, v_c = 0.5 * (u_p.min() + u_p.max()), 0.5 * (v_p.min() + v_p.max())
    out["origin"] = (origin + u_c * u_hat + v_c * v_hat).tolist()
    out["half_u"] = 0.5 * (u_p.max() - u_p.min()) + margin
    out["half_v"] = 0.5 * (v_p.max() - v_p.min()) + margin
    out["sweep_start_um"] = float(n_p.min() - margin)
    out["sweep_stop_um"] = float(n_p.max() + margin)
    if not out.get("sweep_step_um"):
        out["sweep_step_um"] = float(out.get("du", default_du))
    return out


# -----------------------------------------------------------------------------
# Loaders + per-volume preparation
# -----------------------------------------------------------------------------
def _axis_suffix(dataset_path):
    p = dataset_path.lower()
    if "chi" in p:
        return "χ", "_chi"
    if "mu" in p:
        return "μ", "_mu"
    return "", ""


def _estimate_box(cfg, p, scale_x, scale_y, samy_dir):
    if cfg["source"] == "aligned":
        with h5py.File(cfg["h5_path"], "r") as f:
            shape = f[cfg["dataset_path"]].shape
            a = dict(f.attrs)
        sx = float(a.get("scale_x_um_per_px", scale_x))
        sy = float(a.get("scale_y_um_per_px", scale_y))
        sz = float(a.get("scale_z_um_per_px", 1.0))
        return _world_box(shape, sx, sy, sz, 0.0, 0.0, 0.0)
    with h5py.File(cfg["h5_path"], "r") as f:
        n_layers, ny0, nx0 = f[cfg["dataset_path"]].shape
    rx, ry = cfg.get("roi_x"), cfg.get("roi_y")
    nx_roi = (rx[1] - rx[0]) if rx else nx0
    ny_roi = (ry[1] - ry[0]) if ry else ny0
    samy, samz, _ = _motors(cfg, p)
    if len(samy) > 0:
        pad_l = A.compute_pad_left(samy, scale_x, samy_dir)
        pad_r = A.compute_pad_right(samy, scale_x, samy_dir)
    else:
        pad_l = pad_r = 0
    nx_pad = nx_roi + pad_l + pad_r
    if len(samz) > 0:
        z_um = (samz[: min(n_layers, len(samz))] - samz[0]) * 1000.0
        med = float(np.median(np.abs(np.diff(z_um)))) if len(z_um) > 1 else 1.0
        med = med if med >= 1e-6 else 1.0
        n_uniform = max(2, int(np.round((z_um.max() - z_um.min()) / med)) + 1)
        sz = (z_um.max() - z_um.min()) / (n_uniform - 1) if n_uniform > 1 else med
    else:
        n_uniform, sz = n_layers, 2.0
    return _world_box((n_uniform, ny_roi, nx_pad), scale_x, scale_y, sz, 0.0, 0.0, 0.0)


def _motors(cfg, p):
    raw_root = cfg.get("raw_root") or ""
    pattern = cfg.get("raw_pattern") or ""
    if not raw_root or not pattern:
        return np.array([]), np.array([]), []
    folders = find_matching_folders(raw_root, pattern)
    if not folders:
        return np.array([]), np.array([]), []
    return extract_motor_positions(folders, p["samy_path"], p["samz_path"])


# Volume kinds whose colour norm is centered on zero (symmetric TwoSlopeNorm).
_CENTERED_KINDS: frozenset[str] = frozenset({"mosa_com"})


class _Float64Dataset:
    """*dset* read as float64 — this stage's historical ``[:].astype(np.float64)``.

    :func:`~dfxm.common.alignment.align_volume_streamed` runs ``abs`` and the
    samy sub-pixel shift in the input's own dtype and upcasts only at the Z
    interpolation. This stage upcast **first**, so on a float32 stacked volume
    ``scipy.ndimage.shift`` ran in float64 and produced different bits from the
    float32 shift the alignment would otherwise do. Presenting the dataset as
    float64 keeps the arithmetic exactly where it was, whatever the file stores,
    at no extra cost: the cast happens on a block instead of on the volume.
    """

    def __init__(self, dset) -> None:
        self._dset = dset
        self.shape = tuple(int(d) for d in dset.shape)
        self.dtype = np.dtype(np.float64)

    def __getitem__(self, key):
        return np.asarray(self._dset[key], dtype=np.float64)


def _materialise(blocks, shape, dtype):
    """This stage's name for :func:`~dfxm.common.alignment.materialise_blocks`.

    Every in-core rung in the project drains a block factory the same way, and
    the adoption of a lone covering block (copying it would hold two whole
    volumes) is the detail worth having in exactly one place.
    """
    return A.materialise_blocks(blocks, shape, dtype)


def _aligned_block_budget(dset, budget_bytes: int) -> int:
    """``budget_bytes`` converted into the block bytes :func:`iter_blocks` counts.

    ``iter_blocks`` sizes a block by its bytes **in the stored dtype**, but this
    stage holds far more than that per block: the float64 upcast, the centred
    copy, and — at their peak — the streaming reductions' own per-element
    temporaries (``_finite64``, the window, the ``searchsorted``/``clip``
    indices and the ``isfinite`` mask), which
    :func:`~dfxm.common.alignment.align_volume_streamed` prices at
    ``dtype.itemsize + 8 * (retained + 1) + 1`` = 41 B per float64 element.
    Handing the budget over raw would buy a block several times too large.

    Integer division rounds the budget **down**, which is the safe direction:
    a smaller budget can only make blocks smaller than counted.
    """
    itemsize = max(1, int(np.dtype(dset.dtype).itemsize))
    per_element = itemsize + 41 + 8  # stored + reductions' peak + the centred copy
    return max(1, int(budget_bytes) * itemsize // per_element)


# The Z step (µm) assumed when there are no samz positions to derive one from.
# `extract_motor_positions` fills samy and samz from the same folders, so they
# are empty together; the value is what this stage has always used here.
_NO_MOTOR_Z_STEP_UM = 2.0


def _unaligned_pad(samy, n_layers: int, scale_x: float, samy_direction: int):
    """The ``(left, right)`` canvas growth the whole-volume samy shift would add.

    Reproduces :func:`~dfxm.common.alignment.apply_samy_shifts_to_volume`'s own
    ``pad=None`` arithmetic, which is **not** ``compute_pad_left(samy)``: the
    canvas comes from ``samy[:n_use]`` with ``n_use`` clipped to the volume's
    layer count. The two agree whenever ``len(samy) == n_layers`` (every real
    run) and differ otherwise, so deriving the pad the other way would silently
    change the output width on the one path that has no motors to trust.
    """
    if samy is None or len(samy) == 0:
        return 0, 0
    n_use = n_layers if len(samy) == n_layers else min(n_layers, len(samy))
    used = np.asarray(samy[:n_use])
    return (
        A.compute_pad_left(used, scale_x, samy_direction),
        A.compute_pad_right(used, scale_x, samy_direction),
    )


def _unaligned_shape(dset, cfg, samy, scale_x: float, samy_direction: int):
    """``(shape, dtype, scale_z)`` of the no-samz chain, from shapes alone."""
    nz, ny0, nx0 = (int(d) for d in dset.shape)
    rx, ry = cfg.get("roi_x"), cfg.get("roi_y")
    nx = (rx[1] - rx[0]) if rx else nx0
    ny = (ry[1] - ry[0]) if ry else ny0
    pad_left, pad_right = _unaligned_pad(samy, nz, scale_x, samy_direction)
    return (nz, ny, nx + pad_left + pad_right), np.dtype(np.float64), _NO_MOTOR_Z_STEP_UM


def _unaligned_blocks(
    dset, cfg, samy, *, scale_x: float, samy_direction: int, take_abs: bool, budget_bytes: int
):
    """The no-samz chain (``abs`` → ROI → samy X-shift) as a Z-block factory.

    Every step here is per layer, so blocking along Z is exact rather than an
    approximation: the ``pad`` and the reference samy are computed once for the
    **whole** volume and passed in explicitly, which is the same trick — and the
    same reason — as :func:`~dfxm.common.alignment.align_volume_streamed`'s. A
    block's own samy would imply a narrower canvas and each block would land on
    a frame of its own.
    """
    n_layers = int(dset.shape[0])
    pad = _unaligned_pad(samy, n_layers, scale_x, samy_direction)
    n_use = 0 if samy is None or len(samy) == 0 else min(n_layers, len(samy))
    ref = float(samy[0]) if n_use else 0.0
    block_budget = _aligned_block_budget(dset, budget_bytes)

    def blocks():
        for sl, block in V.iter_blocks(dset, budget_bytes=block_budget):
            v = np.asarray(block, dtype=np.float64)
            if take_abs:
                v = np.abs(v)
            v = A.apply_roi_3d(v, cfg.get("roi_x"), cfg.get("roi_y"))
            if n_use:
                # `samy[sl]` and not `samy`: the shift is applied per layer and
                # each layer must get its own offset. Layers past `n_use` get an
                # empty slice here and are padded but not shifted, exactly as the
                # whole-volume call leaves them.
                v = A.apply_samy_shifts_to_volume(
                    v, np.asarray(samy)[sl], scale_x, samy_direction, ref_samy=ref, pad=pad
                )
            else:
                v = np.ascontiguousarray(v)
            yield sl, v

    return blocks


def prepare_volume(cfg, p, scale_x, scale_y, samy_dir, style=None, *, stack, budget_bytes):
    """Load and (if stacked) align one volume, resolving render style per kind.

    Returns a dict whose ``blocks`` key is a zero-argument factory yielding
    ``(z_slice, array)`` over the prepared (centred) volume, and whose ``data``
    key is that volume as one array **when the machine can hold it** and ``None``
    otherwise.

    **Two rungs, the project's own escalation.** When ``budget_bytes`` leaves the
    alignment a single block, the whole aligned volume exists anyway: it is
    adopted, and the plane sampling reads it directly — one traversal, and no
    coordinate work beyond what the stage always did. Only when it does *not*
    fit does the plane sampling gather from the Z-block stream, where re-reading
    is the price of running at all. ``advice.plan_run`` makes the same
    in-core-then-chunked choice for the same reason, and `visualize` measured
    that streaming *unconditionally* made its peak worse.

    The colour limits and the centring statistic are computed from ``blocks`` on
    **both** rungs — see the note above :func:`_center_offset` — so which rung
    ran cannot change a number.

    *stack* is an ``ExitStack`` owned by the caller: the HDF5 file must stay
    open for as long as the blocks are walked, and must close when the caller is
    finished with this volume.
    """
    kind, source = cfg["kind"], cfg["source"]
    extra = {}
    f = stack.enter_context(h5py.File(cfg["h5_path"], "r"))
    dset = f[cfg["dataset_path"]]
    if source == "stacked":
        samy, samz, _ = _motors(cfg, p)
        take_abs = kind == "mosa_fwhm" and bool(p["abs_fwhm"])
        if len(samz) > 0:
            streamed = A.align_volume_streamed(
                _Float64Dataset(dset),
                samy,
                samz,
                scale_x=scale_x,
                samy_direction=samy_dir,
                roi_x=cfg.get("roi_x"),
                roi_y=cfg.get("roi_y"),
                take_abs=take_abs,
                # None: slices centres itself below, midrange included, and that
                # convention is not one alignment.py knows.
                center_method=None,
                budget_bytes=max(1, int(budget_bytes) // REDUCTION_WORKING_SET_MULTIPLE),
            )
            raw_blocks = streamed.blocks
            shape, dtype = streamed.shape, streamed.dtype
            scale_z = float(streamed.scale_z_um)
            fits = int(streamed.block_layers) >= int(streamed.shape[0])
        else:
            # No samz means no Z grid to interpolate onto, and re-interpolating
            # a NaN-bearing volume onto its own nodes is not the identity — so
            # `align_volume_streamed`, which always interpolates, cannot serve
            # this case (it would also raise on `samz[0]`). `visualize` and
            # `paraview` answer it by falling back to a whole in-core volume.
            #
            # This stage does **not**, because it does not have to: what is left
            # of the chain once the Z interpolation is gone — `abs`, the ROI
            # crop and the samy X-shift — is entirely **per layer**, so it
            # blocks along Z like everything else. Leaving it in core would have
            # made the one path a run reaches by *misconfiguration* the one path
            # with no memory bound at all, which is a poor place to keep an
            # escape hatch: a user whose scan folders did not match is exactly
            # the user least likely to read a note about it.
            shape, dtype, scale_z = _unaligned_shape(dset, cfg, samy, scale_x, samy_dir)
            raw_blocks = _unaligned_blocks(
                dset,
                cfg,
                samy,
                scale_x=scale_x,
                samy_direction=samy_dir,
                take_abs=take_abs,
                budget_bytes=budget_bytes,
            )
            fits = V.volume_bytes(dset) <= _aligned_block_budget(dset, budget_bytes)
        sx, sy = scale_x, scale_y
        x_ref = y_ref = 0.0
        z_ref = float(cfg.get("z_ref_shift_um", 0.0))
    else:  # aligned — already co-registered, no alignment step
        extra = dict(f.attrs)
        block_budget = _aligned_block_budget(dset, budget_bytes)
        raw_blocks = lambda: (  # noqa: E731 - a factory, deliberately re-callable
            (sl, np.asarray(block, dtype=np.float64))
            for sl, block in V.iter_blocks(dset, budget_bytes=block_budget)
        )
        shape = tuple(int(d) for d in dset.shape)
        dtype = np.dtype(np.float64)
        fits = V.volume_bytes(dset) <= block_budget
        sx = float(extra.get("scale_x_um_per_px", scale_x))
        sy = float(extra.get("scale_y_um_per_px", scale_y))
        scale_z = float(extra.get("scale_z_um_per_px", 1.0))
        x_ref = float(cfg.get("x_ref_shift_px", 0))
        y_ref = float(cfg.get("y_ref_shift_px", 0))
        z_ref = float(cfg.get("z_ref_shift_um", 0.0))

    data = None
    if fits:
        # The volume the machine can hold: build it once, and read the statistics
        # off *slices of it* rather than off the whole array. Basic slicing of an
        # ndarray is a view, so the reductions cost block-sized temporaries here
        # too — where handing them the volume as one block would allocate several
        # full-size ones, which is how `visualize`'s first attempt made its peak
        # worse than the code it replaced.
        data = np.ascontiguousarray(_materialise(raw_blocks, shape, dtype), dtype=np.float64)
        raw_blocks = lambda: V.iter_blocks(  # noqa: E731
            data, budget_bytes=_aligned_block_budget(data, budget_bytes)
        )

    center_method = p["center_method"].lower()
    center = 0.0
    if kind in ("mosa_com", "strain"):
        if center_method == "midrange":
            center, (auto_vmin, auto_vmax) = _midrange_clim(raw_blocks, float(p["range_pct"]))
        else:
            center = _center_offset(raw_blocks, center_method)
            auto_vmin, auto_vmax = _symmetric_range(_shifted(raw_blocks, center))
    else:  # mosa_fwhm / raw_*
        auto_vmin, auto_vmax = _percentile_range(raw_blocks, 1, 99)

    if data is not None:
        if center:
            # In place: `data - center` would hold two whole volumes, and the
            # rounding is the same either way.
            np.subtract(data, center, out=data)
        blocks = raw_blocks  # the factory slices `data`, which is now centred
    else:
        blocks = _shifted(raw_blocks, center)

    sym, suffix = _axis_suffix(cfg["dataset_path"])
    titles = {
        "mosa_com": (f"{sym} Misorientation", "Misorientation (°)", suffix),
        "mosa_fwhm": (f"{sym} Peak Broadening", "Peak broadening (°)", suffix),
        "strain": ("Strain (cot method)", "Strain (ε)", ""),
        "raw_sum": ("Background-subtracted Sum Intensity", "Sum intensity (a.u.)", ""),
        "raw_specific": (
            f"Background-subtracted Frame {int(extra.get('specific_frame_idx', -1))}",
            "Intensity (a.u.)",
            f"_frame{int(extra.get('specific_frame_idx', -1))}",
        ),
        "raw_mosa_sum": ("Mosa-integrated Sum Intensity", "Sum intensity (a.u.)", ""),
        "raw_mosa_specific": (
            f"Mosa-integrated Frame {int(extra.get('specific_frame_idx', -1))}",
            "Intensity (a.u.)",
            f"_frame{int(extra.get('specific_frame_idx', -1))}",
        ),
    }
    title, cbar_label, suffix = titles[kind]
    vmin_f, vmax_f, clim_note = apply_round_clim(float(auto_vmin), float(auto_vmax), style)
    return {
        # The resident aligned volume on the in-core rung, None on the streaming
        # one; `blocks` is the factory both rungs stream from.
        "data": data,
        "blocks": blocks,
        "shape": tuple(int(d) for d in shape),
        "scale_x": float(sx),
        "scale_y": float(sy),
        "scale_z": float(scale_z),
        "x_ref_shift_px": float(x_ref),
        "y_ref_shift_px": float(y_ref),
        "z_ref_shift_um": float(z_ref),
        "vmin": vmin_f,
        "vmax": vmax_f,
        "vmin_raw": float(auto_vmin),
        "vmax_raw": float(auto_vmax),
        "clim_note": clim_note,
        "cmap_name": resolve_cmap(style, GROUP_BY_KIND.get(kind)),
        "group": GROUP_BY_KIND.get(kind),
        "center_zero": kind in _CENTERED_KINDS,
        "title": title,
        "cbar_label": cbar_label,
        "kind": kind,
        "volume_id": f"{kind}{suffix}",
        "h5_path": cfg["h5_path"],
        "dataset_path": cfg["dataset_path"],
    }


# -----------------------------------------------------------------------------
# Rendering + HDF5
# -----------------------------------------------------------------------------
def _make_norm(prep):
    vmin, vmax = prep["vmin"], prep["vmax"]
    if prep.get("center_zero"):
        if vmin < 0.0 < vmax and not np.isclose(abs(vmin), abs(vmax)):
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        v = max(abs(vmin), abs(vmax))
        return mcolors.Normalize(vmin=-v, vmax=v)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


_LEGACY_STYLE = PlotStyle(scale_bar_color="black", colorbar_fraction=0.046)


def draw_slice_axes(
    ax,
    prep,
    sl,
    slice2d,
    u_um,
    v_um,
    *,
    offset_um,
    style: PlotStyle | None = None,
    cax=None,
    colorbar=None,
    scale_bar=None,
    fixed_scale_um_per_cm=None,
):
    """Draw one oblique-slice plane into *ax*; returns the image.

    Extracted verbatim from :func:`build_slice_figure` so the single-figure path
    and the compose adapters share one look. ``colorbar``/``scale_bar`` default
    to the style flags; explicit bools override. ``cax`` routes the colourbar
    into an already-placed axes (steal-free).
    """
    st = style if style is not None else _LEGACY_STYLE
    use_legacy = style is None
    fig = ax.get_figure()
    extent = [float(u_um[0]), float(u_um[-1]), float(v_um[0]), float(v_um[-1])]
    im = ax.imshow(
        slice2d,
        cmap=Rnd.cmap_nan_transparent(prep["cmap_name"]),
        norm=_make_norm(prep),
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    ax.set_xlabel("u (µm)")
    ax.set_ylabel("v (µm)")
    sub = sl["name"] if offset_um is None else f"{sl['name']}  (offset {offset_um:+.2f} µm)"
    ax.set_title(f"{prep['title']}\nslice: {sub}")
    if st.colorbar if colorbar is None else colorbar:
        add_colorbar(fig, im, ax, prep["cbar_label"], st, group=prep.get("group"), cax=cax)
    if st.scale_bar if scale_bar is None else scale_bar:
        draw_scale_bar(
            ax, st.scale_bar_length_um, style=st, fixed_scale_um_per_cm=fixed_scale_um_per_cm
        )
    if not use_legacy:
        apply_text_scale(ax, st)
        apply_axes_mode(ax, st)
    return im


def build_slice_figure(
    prep, sl, slice2d, u_um, v_um, *, offset_um, style: PlotStyle | None = None
) -> Figure:
    """Build and return a slice figure. Does NOT call savefig.

    When *style* is ``None`` the default un-styled appearance is used (black auto
    scale bar, 0.046 colourbar fraction, 12×10 in figsize) via the shared styled
    primitives — close to, but not byte-identical with, the pre-export legacy
    renderer. When a :class:`~dfxm.common.plotting.PlotStyle` is supplied its
    figsize/colourbar/fonts/scale-bar settings are honoured instead.
    """
    st = style if style is not None else _LEGACY_STYLE
    use_legacy = style is None

    if use_legacy:
        figsize = (12, 10)
        box = None
    else:
        ext_u = float(u_um[-1] - u_um[0])
        ext_v = float(v_um[-1] - v_um[0])
        box = fixed_scale_box(st, ext_u, ext_v)
        if box is not None:
            figsize = (box[0] + 1.5, box[1] + 1.5)
        else:
            figsize = figure_size(st, ext_u, ext_v) or (12, 10)

    fig = styled_figure(figsize, styled=not use_legacy)
    ax = fig.add_subplot(111)
    draw_slice_axes(
        ax,
        prep,
        sl,
        slice2d,
        u_um,
        v_um,
        offset_um=offset_um,
        style=style,
        fixed_scale_um_per_cm=(box[2] if box is not None else None),
    )
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])

    return fig


def save_slice_png(prep, sl, slice2d, u_um, v_um, out_png, *, offset_um, dpi=150, style=None):
    """Build a slice figure (legacy look when *style* is None) and save it."""
    build_slice_figure(prep, sl, slice2d, u_um, v_um, offset_um=offset_um, style=style).savefig(
        out_png, dpi=dpi, facecolor="white", bbox_inches="tight"
    )


def open_volume_group(fh, prep):
    """Create one volume's group and write its attrs; the slice subgroups follow.

    Split from the slice writing (it used to be one ``write_volume_group`` call
    taking finished stacks) so a sweep can be written **plane by plane** as it is
    sampled: see :func:`open_slice_dataset` and :class:`PlaneWriter`. Nothing
    here depends on a sampled value, so the whole group is created before the
    first plane exists.
    """
    vg = fh.create_group(prep["volume_id"])
    vg.attrs["kind"] = prep["kind"]
    vg.attrs["dataset_path"] = prep["dataset_path"]
    vg.attrs["source_volume"] = prep["h5_path"]
    vg.attrs["cbar_label"] = prep["cbar_label"]
    vg.attrs["cmap"] = prep["cmap_name"]
    vg.attrs["vmin"] = float(prep["vmin"])
    vg.attrs["vmax"] = float(prep["vmax"])
    if prep.get("clim_note"):
        vg.attrs["vmin_raw"] = float(prep["vmin_raw"])
        vg.attrs["vmax_raw"] = float(prep["vmax_raw"])
    vg.attrs["title"] = prep["title"]
    vg.attrs["scale_x_um_per_px"] = prep["scale_x"]
    vg.attrs["scale_y_um_per_px"] = prep["scale_y"]
    vg.attrs["scale_z_um_per_px"] = prep["scale_z"]
    return vg


def open_slice_dataset(vg, rec):
    """Create one slice subgroup — sized, typed and fully attributed — up front.

    Returns the empty ``slices`` dataset for :class:`PlaneWriter` to fill. Every
    number written here is geometry: ``resolve_auto_extent`` has already fixed
    the extent, :func:`_plane_axes` the ``(nv, nu)`` grid and
    :func:`slice_plane_offsets` the plane count, so the dataset can be sized
    without a single sample being taken.

    Same datasets, same paths, same dtypes, same attrs and the same creation
    order as the old whole-stack write — including the ``gzip``/level-4 filter,
    which is what makes h5py chunk the dataset, and it guesses the same chunk
    shape from ``(shape, dtype)`` whether the data is supplied at creation or
    written afterwards.

    ``fillvalue=nan``, and it is load-bearing. Sizing the dataset up front means
    a run that dies mid-sweep — a `MemoryError`, a kill — leaves a group whose
    unwritten planes are readable, where the old whole-stack write left no group
    at all; :meth:`PlaneWriter.close`'s count check cannot fire on that path,
    because nothing calls it. HDF5's default fill is **zero**, and a plane of
    zeros reads as data: `profiles` will happily average it into a CSV with no
    skip and no note. NaN is what every reader in this pipeline already treats as
    "no sample" (`map_coordinates`' own ``cval``, the colour-limit reductions,
    `profiles`' finite masks), so an interrupted sweep degrades to a plane that
    is visibly absent rather than one that is quietly flat.
    """
    sg = vg.create_group(rec["name"])
    dset = sg.create_dataset(
        "slices",
        shape=(len(rec["offsets"]), len(rec["v_um"]), len(rec["u_um"])),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        fillvalue=np.float32(np.nan),
    )
    sg.create_dataset("u_um", data=rec["u_um"].astype(np.float64))
    sg.create_dataset("v_um", data=rec["v_um"].astype(np.float64))
    sg.create_dataset("offsets_um", data=rec["offsets"].astype(np.float64))
    for key in ("normal", "origin", "up", "u_hat", "v_hat", "n_hat"):
        sg.attrs[key] = np.asarray(rec[key], np.float64)
    for key in ("half_u", "half_v", "du", "dv", "sweep_step_um"):
        sg.attrs[key] = float(rec[key])
    sg.attrs["n_planes"] = int(len(rec["offsets"]))
    return dset


class PlaneWriter:
    """Fills a sized ``slices`` dataset one plane at a time.

    The point of the class is that it buffers **one chunk row** — ``chunks[0]``
    planes — and no more. Writing a single plane into a gzip-compressed dataset
    whose chunks span several planes would make HDF5 read, decompress,
    re-compress and re-write every chunk it touches once per plane in the row;
    buffering to the chunk boundary makes every chunk written exactly once, which
    is the same compression work the old whole-stack write did.

    The buffer is small and its size is not a free parameter. h5py aims each
    chunk at ~1 MiB and shrinks the dimensions roughly evenly to get there, so
    ``chunks[0] * plane_bytes`` grows like ``stack_bytes**(2/3)``: the shipped
    default ``oblique_full`` (172 x 1789 x 2528 float32, 2.90 GiB) chunks at
    ``(6, 56, 158)``, i.e. a 104 MiB buffer for a 2.90 GiB stack, and a stack
    would have to reach ~140 GB before the buffer reached 256 MiB.

    :attr:`depth` is that plane count, exposed so the caller can charge it to the
    same budget it charges the sampler's own batch.
    """

    def __init__(self, dset):
        self._dset = dset
        self._n_planes = int(dset.shape[0])
        self.depth = int(dset.chunks[0]) if dset.chunks else self._n_planes
        self._slab = None
        self._held = 0
        self._written = 0

    def append(self, plane) -> None:
        """Buffer one plane; flush when the buffer reaches a chunk boundary."""
        if self._slab is None:
            # One allocation reused for the whole sweep — `np.stack` per flush
            # would hold the buffer and its copy at once, which is the doubling
            # this class exists to remove.
            self._slab = np.empty((self.depth,) + tuple(plane.shape), dtype=np.float32)
        self._slab[self._held] = plane
        self._held += 1
        if self._held == self.depth:
            self.flush()

    def flush(self) -> None:
        if not self._held:
            return
        self._dset[self._written : self._written + self._held] = self._slab[: self._held]
        self._written += self._held
        self._held = 0

    def close(self) -> None:
        """Flush the tail and release the buffer. Every plane must have arrived."""
        self.flush()
        self._slab = None
        if self._written != self._n_planes:
            raise RuntimeError(
                f"slice dataset {self._dset.name!r} was sized for {self._n_planes} "
                f"planes but {self._written} were written"
            )


# -----------------------------------------------------------------------------
# Pinned planes (shared by tools/pin_slice.py, the Pin planes… dialog and run())
# -----------------------------------------------------------------------------
def _find_slice_group(f, slice_name, volume=None):
    """Return (volume_id, slice_group) for the first volume holding *slice_name*.

    Slice geometry is identical across volumes, so any volume that carries the
    slice works; *volume* forces a specific one.
    """
    vids = [volume] if volume else [k for k in f.keys() if k != MARKS_GROUP]
    for vid in vids:
        if vid not in f or not isinstance(f[vid], h5py.Group):
            continue
        g = f[vid]
        if (
            slice_name in g
            and isinstance(g[slice_name], h5py.Group)
            and "offsets_um" in g[slice_name]
        ):
            return vid, g[slice_name]
    volumes = [k for k in f.keys() if k != MARKS_GROUP and isinstance(f[k], h5py.Group)]
    raise StageUserError(
        f"slice {slice_name!r} not found in any volume group of the file",
        hint=f"volumes present: {', '.join(volumes) or '(none)'} — pick a slice "
        "name from a swept oblique_slices.h5.",
    )


def nearest_plane_index(offsets_um, offset_um) -> int:
    """Index of the stored plane nearest *offset_um* — THE snap rule.

    Single source of the nearest-plane snap shared by pinning, marks, the
    GUI plane lists and ``profiles.resolve_plane_index``.
    """
    stored = np.asarray(offsets_um, np.float64)
    return int(np.argmin(np.abs(stored - float(offset_um))))


def build_pinned_spec(h5_path, slice_name, offsets, *, volume=None) -> list[dict]:
    """Pinned single-plane spec dicts for *slice_name*, snapped to stored planes.

    Geometry (normal/origin/up/half_u/half_v/du/dv) is read byte-exact off the
    stored slice-group attrs, so each pinned plane reproduces the sweep's plane
    exactly. Every requested offset snaps to the nearest stored plane; snaps
    landing on the same plane are collapsed. Raises StageUserError for an
    unreadable file or unknown slice name.
    """
    try:
        fh = h5py.File(h5_path, "r")
    except OSError as exc:
        raise StageUserError(
            f"cannot read {h5_path!r}: {exc}",
            hint="Point at an oblique_slices.h5 written by a slices sweep run.",
        ) from exc
    with fh as f:
        _vid, sg = _find_slice_group(f, slice_name, volume)
        stored = sg["offsets_um"][:].astype(np.float64)
        a = dict(sg.attrs)
    specs, seen = [], set()
    for off in offsets:
        idx = nearest_plane_index(stored, off)
        if idx in seen:
            continue
        seen.add(idx)
        matched = float(stored[idx])
        specs.append(
            {
                "name": f"{slice_name}_pin_{matched:+.2f}um",
                "normal": np.asarray(a["normal"], np.float64).tolist(),
                "origin": np.asarray(a["origin"], np.float64).tolist(),
                "up": np.asarray(a["up"], np.float64).tolist(),
                "half_u": float(a["half_u"]),
                "half_v": float(a["half_v"]),
                "du": float(a["du"]),
                "dv": float(a["dv"]),
                "sweep_step_um": float(a.get("sweep_step_um") or 1.0) or 1.0,
                "sweep_start_um": matched,
                "sweep_stop_um": matched,
            }
        )
    return specs


# -----------------------------------------------------------------------------
# Marks (starred planes; shared by the GUI dialogs and the profiles bridge)
# -----------------------------------------------------------------------------
def read_marks(h5_path_or_file) -> dict[str, list[float]]:
    """All marked offsets, ``{slice_name: [offset_um, ...]}`` (sorted).

    Accepts a path or an open ``h5py.File``. A missing ``/marks`` group means
    no marks; malformed children (non-datasets, non-numeric) are skipped, so a
    hand-edited file degrades to fewer marks, never an error.
    """

    def _read(f):
        out: dict[str, list[float]] = {}
        mg = f.get(MARKS_GROUP)
        if not isinstance(mg, h5py.Group):
            return out
        for sname, ds in mg.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            try:
                offs = np.asarray(ds[()], np.float64).ravel()
            except (TypeError, ValueError):
                continue
            out[str(sname)] = sorted(float(o) for o in offs)
        return out

    if isinstance(h5_path_or_file, h5py.File):
        return _read(h5_path_or_file)
    with h5py.File(h5_path_or_file, "r") as f:
        return _read(f)


def write_marks(h5_path, slice_name, offsets_um) -> list[float]:
    """Replace *slice_name*'s marks with *offsets_um* (snapped to stored planes).

    Offsets snap to the nearest stored plane (the ``resolve_plane_index``
    rule), collapse duplicates, and are stored sorted; an empty list deletes
    the dataset (and the ``/marks`` group once it is empty). Returns the
    snapped offsets actually stored.
    """
    try:
        fh = h5py.File(h5_path, "r+")
    except OSError as exc:
        raise StageUserError(
            f"cannot open {h5_path!r} for writing marks: {exc}",
            hint="Close any dialog/viewer holding the file open, then retry; "
            "the file must be an oblique_slices.h5 from a slices run.",
        ) from exc
    with fh as f:
        _vid, sg = _find_slice_group(f, slice_name)
        stored = sg["offsets_um"][:].astype(np.float64)
        snapped = sorted({float(stored[nearest_plane_index(stored, o)]) for o in offsets_um})
        mg = f.require_group(MARKS_GROUP)
        if slice_name in mg:
            del mg[slice_name]
        if snapped:
            mg.create_dataset(slice_name, data=np.asarray(snapped, np.float64))
        elif len(mg.keys()) == 0:
            del f[MARKS_GROUP]
    return snapped


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def _standard_volumes(p, roi_x, roi_y):
    """Build the VOLUMES config list from the include_* toggles + file paths."""
    file_keys = {
        "mosa_volume_file": (p["mosa_volume_file"], p["raw_root"], p["mosa_pattern"]),
        "strain_volume_file": (p["strain_volume_file"], p["raw_root"], p["strain_pattern"]),
        "aligned_rocking_file": (p["aligned_rocking_file"], "", ""),
        "aligned_mosa_file": (p["aligned_mosa_file"], "", ""),
    }
    out = []
    for toggle, source, file_param, dataset, kind in _STD_VOLUMES:
        if not bool(p.get(toggle)):
            continue
        path, raw_root, pattern = file_keys[file_param]
        if not path or not os.path.exists(path):
            continue
        cfg = {
            "h5_path": path,
            "dataset_path": dataset,
            "kind": kind,
            "source": source,
        }
        if source == "stacked":
            cfg.update(raw_root=raw_root, raw_pattern=pattern, roi_x=roi_x, roi_y=roi_y)
        out.append(cfg)
    return out


_SLICES_VOLUME_PARAMS = (
    "mosa_volume_file",
    "strain_volume_file",
    "aligned_rocking_file",
    "aligned_mosa_file",
)


def estimate(params: dict) -> CostEstimate:
    """Peak memory for a slices run, from HDF5 shapes and motor positions.

    **MEASURED against the real STO2 dataset at master a424b1f** (five stacked
    fields + two aligned, ROI y 400,1100):

        estimate 6.57 GiB   measured peak RSS 7.41 GiB

    **Read that comparison carefully — the two numbers are in different
    currencies, and conflating them is a mistake this project has already made
    once.** ``peak_bytes`` counts Python allocations (``tracemalloc``); a peak
    RSS includes the process image and the allocator's slack, and the two are
    related *additively*, not by a ratio:
    ``RSS ~ RSS_FLOOR_BYTES + MARGINAL_RSS_PER_TRACED_BYTE * traced`` (see the
    note above ``advice.MARGINAL_RSS_PER_TRACED_BYTE``). Against this stage's own
    floor that predicts ``0.25 + 1.3 x 6.57 = 8.79 GiB`` for a measured
    7.41 — so the model **covers** the measurement rather than under-predicting
    it, and the bare ``6.57 < 7.41`` reading is the currency error, not a defect.
    (An earlier revision of this paragraph substituted 1.1 for
    ``MARGINAL_RSS_PER_TRACED_BYTE`` after naming it; the constant is **1.3**.
    Do not re-derive the arithmetic here by hand — read the constant.)

    **The margin is ~19%, i.e. comfortable rather than marginal**, and one known
    term is nonetheless still missing: the output side (a :class:`PlaneWriter`
    chunk-row buffer plus :data:`GATHER_SCRATCH_PLANE_MULTIPLE` planes of
    coordinate scratch) is **~0.35 GiB unmodelled on the shipped default
    geometry**. That residual is real and worth stating — it grows with a
    plane's area, so a much larger ``extent``/finer ``du``/``dv`` eats into the
    margin — but at 19% it is not what stands between this model and an
    under-prediction. It is not folded in here because sizing it needs a plane's
    ``(nv, nu)``, which means resolving ``slices_json`` against the data box —
    work this estimator deliberately does not do — and inventing a bound for an
    oblique plane would be worse than a residual that is documented and
    quantified. The arithmetic once ``(nv, nu)`` is known is
    ``(GATHER_SCRATCH_PLANE_MULTIPLE + chunks[0]) * nv * nu * 4``.


    ``run()`` calls ``prepare_volume`` one selected dataset at a time. The
    model below is the *old* loop, in which ``prep`` was a plain local merely
    *rebound* each iteration, so the previous volume's prepared array stayed
    fully alive while the next one was built — hence a peak pairing the
    current volume's own load peak with the largest OTHER selected volume's
    footprint. ``run()`` now releases the previous ``prep`` before calling
    ``prepare_volume`` again, so that cross-volume term no longer applies.

    Within one ``prepare_volume`` call, ``elems_v * itemsize_v`` is the native
    read (source dtype, coexisting briefly with its ``.astype(np.float64)``
    copy). For a **stacked** source (``mosa_volume_file`` /
    ``strain_volume_file``) the model counts three float64-sized copies of
    that field alive at once: the raw float64 read (kept in its own variable),
    the samy-shifted canvas, and ``interpolate_to_uniform_z``'s float64 output
    — ``load_peak_v = elems_v * itemsize_v + 3 * elems_v * 8``. The raw read
    is now dropped as soon as the samy shift has copied out of it, so only two
    are really coexistent. An **aligned** source (``aligned_rocking_file`` /
    ``aligned_mosa_file``) is already co-registered — no shift/interpolate
    step — so only the one float64 read persists:
    ``load_peak_v = elems_v * itemsize_v + 1 * elems_v * 8``.

    **Recalibration warning — do not simply change the 3 to a 2.** The two
    surviving stacked copies are each *larger* than ``elems_v``, so counting
    them in bare ``elems_v * 8`` units **under**-predicts, and under-prediction
    is the dangerous direction (it greenlights a run that then OOMs). Both
    survivors are inflated, and on the second the inflations compound:

    * ``apply_samy_shifts_to_volume`` expands the canvas along image-X so
      nothing is clipped — ``nx_new = nx_orig + pad_left + pad_right``, the
      pads being the extreme samy offsets in pixels
      (``alignment.py:81-86``). Its output is ``(n_layers, ny, nx_new)``.
    * ``interpolate_to_uniform_z`` resamples onto
      ``n_uniform = max(2, round((z_max - z_min) / median|Δz|) + 1)``
      (``alignment.py:122``), which **exceeds** ``n_use`` whenever samz is
      irregular: one large gap drags the median step down and stretches the
      uniform grid. Its output is ``(n_uniform, ny, nx_new)`` — it inherits
      the X-padding, so both inflations multiply.

    A correct model therefore needs the padded/resampled extents, not
    ``elems_v``; neither is derivable from HDF5 shapes alone (both depend on
    the samy/samz motor values), so recalibration must either read the motors
    or carry an explicit, documented safety factor. The arithmetic is
    deliberately left unchanged here. Net of everything above the current
    figure still **over**-estimates on regular-samz data — the dropped third
    copy was in effect accidental headroom masking the inflation — but that
    margin is not guaranteed on a large sweep or an irregular samz.

    ``total_input`` is unchanged from the file-level ``sum_dataset_bytes``
    total across the four volume-file params (every dataset in each selected
    file, not filtered by the ``include_*`` toggles) — a deliberately
    conservative input figure kept for continuity with the other estimators.

    ``chunkable=True``. It read ``False`` — "alignment is a whole-volume
    operation" — and that was wrong: the alignment runs *along* Z, so blocking it
    along Z is exactly what :func:`~dfxm.common.alignment.align_volume_streamed`
    does, and the sampling that consumes it reads only the eight voxels
    bracketing each sample, two of them in Z. ``run`` now takes the whole stage
    in Z-blocks whenever the machine's budget requires it.

    **The peak model above still describes the old whole-volume loop** and has
    not been recalibrated for the streamed one — the same caveat Task 9 left on
    paraview's and Task 10 on visualize's.

    **The model has never counted the OUTPUT side, and the term it is missing is
    now small but not zero.** A sweep's planes used to be held whole, and on the
    shipped default geometry that was the largest array the stage touched by a
    factor of three (2.90 GiB of ``oblique_full`` against a 1.15 GiB volume,
    transiently doubled by ``np.stack``). ``run`` now sizes the HDF5 dataset up
    front and writes plane by plane, so what survives is a :class:`PlaneWriter`
    chunk-row buffer (~``stack_bytes ** (2/3)``, 104 MiB for that 2.90 GiB
    sweep) plus :data:`GATHER_SCRATCH_PLANE_MULTIPLE` planes of coordinate
    scratch (259 MiB) — both functions of one **plane's** size rather than of
    the sweep's length, but together still about **0.35 GiB unmodelled on the
    shipped default, ~9% of an 8 GB machine's headroom**. That is a residual to
    fold into the next recalibration, not something to ignore. Deriving it from
    ``params`` means resolving ``slices_json`` against the data box here, which
    this estimator does not do; the arithmetic once a plane's ``(nv, nu)`` is
    known is `(GATHER_SCRATCH_PLANE_MULTIPLE + chunks[0]) * nv * nu * 4`.
    """
    p = {**STAGE.defaults(), **params}
    total_input = 0
    largest: tuple[int, ...] | None = None
    for name in _SLICES_VOLUME_PARAMS:
        path = str(p.get(name) or "")
        if not path:
            continue
        nbytes, shape, _itemsize = sum_dataset_bytes(path)
        if not nbytes:
            continue
        total_input += nbytes
        if shape is not None and (largest is None or len(shape) > len(largest)):
            largest = shape
    if not total_input:
        return CostEstimate(0, 0, None, True, "no readable volume files selected yet")

    try:
        # A mid-typed ROI string ("10,", "abc") must not break the estimate —
        # the ROI plays no part in the sizing arithmetic anyway.
        roi_x, roi_y = _parse_pair(p.get("align_roi_x")), _parse_pair(p.get("align_roi_y"))
    except Exception:  # noqa: BLE001 - an estimate is advisory, never fatal
        roi_x = roi_y = None
    try:
        volumes = _standard_volumes(p, roi_x, roi_y)
    except Exception:  # noqa: BLE001 - an estimate is advisory, never fatal
        volumes = []

    # (own_f64_footprint, load_peak) per selected dataset actually resolvable
    load_infos: list[tuple[int, int]] = []
    for cfg in volumes:
        sizes = iter_dataset_sizes(cfg["h5_path"])
        match = next((s for s in sizes if s[0] == cfg["dataset_path"]), None)
        if match is None:
            continue
        _name, shape, itemsize = match
        elems = 1
        for dim in shape:
            elems *= dim
        copies = 3 if cfg["source"] == "stacked" else 1
        load_peak = elems * itemsize + copies * elems * 8
        load_infos.append((elems * 8, load_peak))

    if not load_infos:
        # Per-dataset sizing couldn't resolve anything selected (e.g. the
        # named files don't hold the toggled quantities) — fall back to the
        # coarser file-level figure rather than reporting zero.
        return CostEstimate(total_input, total_input, largest, True)

    peak = 0
    for i, (_own_f64, load_peak_i) in enumerate(load_infos):
        other = max((f64 for j, (f64, _lp) in enumerate(load_infos) if j != i), default=0)
        peak = max(peak, load_peak_i + other)
    # `scratch_bytes` stays at its 0 default on BOTH returns, including for
    # `center_method="median"`, and that is deliberate rather than a dropped
    # term. `prepare_volume` calls `align_volume_streamed` with
    # `center_method=None` and no `scratch_dir=` — this stage centres itself
    # afterwards (midrange included, a convention alignment.py does not know),
    # so nothing here ever caches an aligned volume to disk. Pricing a spill
    # the stage cannot perform would let `advice.plan_run` BLOCK a run that
    # touches no disk at all, which is the one thing this phase promised never
    # to do. Pinned by `test_slices_never_hands_the_alignment_a_scratch_dir`:
    # if that ever starts passing one, size it on both returns — not before.
    return CostEstimate(peak, total_input, largest, True)


# Relative Y-height spread beyond which volumes are flagged as misregistered.
# Legitimate pixel-size differences between runs (calculator vs nominal values)
# are well under 1%; a wrong detector-row crop is tens of percent.
_Y_HEIGHT_RTOL = 0.05


def _y_height_notes(volumes, roi_y, scale_y):
    """Warn when the selected volumes disagree on physical Y height.

    Every volume anchors at Y=0 in the origin-0 world frame, so co-registration
    along Y relies on all volumes covering the same detector-row window. A
    height mismatch beyond pixel-size rounding means one volume was built with
    a different crop — classically a rocking-stage roi_y entered as darfix's
    origin+size instead of start,end detector rows. Reads only shapes/attrs.
    """
    entries = []
    for cfg in volumes:
        try:
            with h5py.File(cfg["h5_path"], "r") as f:
                ny = int(f[cfg["dataset_path"]].shape[1])
                attrs = dict(f.attrs) if cfg["source"] == "aligned" else {}
        except (KeyError, OSError):
            continue  # unreadable volumes get reported as skips by prepare_volume
        label = f"{cfg['kind']} ({os.path.basename(cfg['h5_path'])})"
        if cfg["source"] == "aligned":
            sy = float(attrs.get("scale_y_um_per_px", scale_y))
            ys, ye = attrs.get("roi_y_start"), attrs.get("roi_y_end")
            if ys is not None and ye is not None:
                label += f" [file roi_y {int(ys)},{int(ye)}]"
        else:
            sy = scale_y
            if roi_y:
                ny = roi_y[1] - roi_y[0]
                label += f" [align_roi_y {roi_y[0]},{roi_y[1]}]"
        entries.append((label, ny * sy))
    if len(entries) < 2:
        return []
    heights = [h for _, h in entries]
    lo, hi = min(heights), max(heights)
    if lo <= 0 or (hi - lo) / lo <= _Y_HEIGHT_RTOL:
        return []
    desc = "; ".join(f"{label}: {h:.1f} µm" for label, h in entries)
    return [
        f"volume Y heights differ — {desc}. All volumes anchor at Y=0 in the world "
        "frame, so different heights misregister features along v/Y. Rebuild the "
        "mismatched aligned raw volume with the same detector-row window the map "
        "volumes use (rocking-stage roi_y is 'start,end' in raw-detector pixels; "
        "note darfix displays its ROI as origin+size, not start,end)."
    ]


def _sweep_resident_bytes(budget_bytes: int, plane_bytes: int, reserved_planes: int) -> int:
    """What is left of the budget for a batch of finished planes.

    Everything else alive while a sweep is sampled, subtracted explicitly rather
    than covered by a round fraction:

    * the Z-block stream, which is ``budget_bytes //
      REDUCTION_WORKING_SET_MULTIPLE`` by construction — that is the figure
      :func:`prepare_volume` hands ``align_volume_streamed``, and the chain
      keeps itself inside it;
    * :data:`GATHER_SCRATCH_PLANE_MULTIPLE` planes of coordinate scratch for the
      plane being filled;
    * *reserved_planes* for :class:`PlaneWriter`'s chunk-row buffer.

    Never less than one plane: a plane is the irreducible unit of the sweep, so a
    budget too small for one is reported by the measured peak, not honoured by
    refusing to sample. On the in-core rung the number is unused — that rung
    produces one plane per call.
    """
    plane_bytes = max(1, int(plane_bytes))
    room = (
        int(budget_bytes)
        - int(budget_bytes) // REDUCTION_WORKING_SET_MULTIPLE
        - (GATHER_SCRATCH_PLANE_MULTIPLE + int(reserved_planes)) * plane_bytes
    )
    return max(plane_bytes, room)


def _run_budget_bytes(p: dict, out_dir: str | None = None) -> int:
    """The working-set budget one ``run`` may allocate, in ``tracemalloc`` bytes.

    Measured from the machine unless the caller injected ``_budget_bytes``. The
    injection is the phase-1-4 convention — an underscore-prefixed key placed in
    ``params`` rather than a :class:`StageSpec` parameter, the same way
    ``plot_style`` reaches a stage — and it exists so tests can pin the blocking
    instead of inheriting whatever RAM the runner happens to have.

    The machine's headroom is an **RSS** figure and ``budget_bytes`` is priced in
    Python allocations, so it goes through
    :func:`~dfxm.common.advice.working_set_budget_bytes` with this stage's own
    :data:`RSS_FLOOR_BYTES` rather than straight in. An injected
    ``_budget_bytes`` is taken as already being in working-set currency, since a
    caller naming it is naming the thing the model consumes.
    """
    injected = p.get("_budget_bytes")
    if injected is not None:
        return max(1, int(injected))
    from ..common import advice, machine

    return advice.working_set_budget_bytes(
        machine.profile(output_dir=out_dir), rss_floor_bytes=RSS_FLOOR_BYTES
    )


def run(params: dict, progress: ProgressFn | None = None) -> SlicesResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    style = style_from_params(p)
    if p["center_method"].lower() not in ("mean", "median", "midrange"):
        raise ValueError(f"center_method must be mean/median/midrange (got {p['center_method']!r})")
    scale_x, scale_y = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["align_roi_x"]), _parse_pair(p["align_roi_y"])

    out_dir = p["output_dir"] or os.path.join(
        os.path.dirname(p["mosa_volume_file"] or p["strain_volume_file"] or "."), "oblique_slices"
    )
    result = SlicesResult(output_dir=out_dir)

    volumes = _standard_volumes(p, roi_x, roi_y)
    if not volumes:
        result.skipped.append("no input volumes found / selected")
        progress(1.0, "no volumes to slice")
        return result
    os.makedirs(out_dir, exist_ok=True)  # only once we know there is work to do
    for msg in _y_height_notes(volumes, roi_y, scale_y):
        result.notes.append(msg)
        progress(0.02, msg)
    if bool(p["use_pinned"]):
        raw_pinned = (p["pinned_slices_json"] or "").strip()
        try:
            slices = json.loads(raw_pinned) if raw_pinned else []
        except json.JSONDecodeError as exc:
            raise StageUserError(
                f"Pinned planes JSON is not valid JSON: {exc}",
                hint=(
                    "Open Pin planes… to regenerate the pinned list, or untick "
                    "'Run pinned planes only' to run the full sweep."
                ),
            ) from exc
        if not isinstance(slices, list) or not slices:
            raise StageUserError(
                "'Run pinned planes only' is on but the pinned planes list is empty",
                hint=(
                    "Open Pin planes… to pick planes, or untick "
                    "'Run pinned planes only' to run the full sweep."
                ),
            )
        msg = (
            f"PINNED RUN: rendering {len(slices)} pinned plane(s); "
            "the sweep in slices_json is ignored"
        )
        result.notes.append(msg)
        progress(0.03, msg)
    else:
        slices = json.loads(p["slices_json"])
        if not isinstance(slices, list) or not slices:
            raise StageUserError(
                "slices_json must be a non-empty JSON list of slice specs",
                hint=(
                    "Provide a JSON list of plane specs — the field's default "
                    "shows the format; 'extent': 'auto' fits the plane "
                    "automatically."
                ),
            )

    # Resolve extent='auto' planes against a common data box (shared grid).
    if any(sl.get("extent") == "auto" for sl in slices):
        progress(0.05, "estimating data bounding box")
        boxes = [_estimate_box(cfg, p, scale_x, scale_y, samy_dir) for cfg in volumes]
        box = _union_box(boxes)
        slices = [resolve_auto_extent(sl, box, default_du=scale_x) for sl in slices]

    # Validate each (resolved) slice up front so a bad spec fails clearly here
    # rather than as a ZeroDivisionError / KeyError deep inside sampling.
    for sl in slices:
        name = sl.get("name", "?")
        for key in ("du", "dv"):
            if key in sl and float(sl[key]) <= 0:
                raise ValueError(f"slice {name!r}: {key} must be > 0")
        if "half_u" not in sl or "half_v" not in sl:
            raise ValueError(f"slice {name!r}: needs half_u and half_v (or extent: 'auto')")
        if float(sl["half_u"]) <= 0 or float(sl["half_v"]) <= 0:
            raise ValueError(f"slice {name!r}: half_u and half_v must be > 0")

    h5_name = p["output_h5_name"]
    if bool(p["use_pinned"]) and h5_name == STAGE.defaults()["output_h5_name"]:
        h5_name = "oblique_slices_pinned.h5"
        result.notes.append(
            "pinned run: output filename switched to oblique_slices_pinned.h5 "
            "so the sweep file profiles reads is not overwritten"
        )
    out_h5 = os.path.join(out_dir, h5_name)
    budget_bytes = _run_budget_bytes(p, out_dir)
    save_png = bool(p["save_png"])
    grids_by_slice: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    fh = h5py.File(out_h5, "w")
    fh.attrs["created_by"] = "dfxm.stages.slices"
    fh.attrs["center_method"] = p["center_method"]
    fh.attrs["range_pct"] = float(p["range_pct"])
    fh.attrs["frame"] = "origin-0 PVTI frame: world(X,Y,Z) = (i*scale_x, j*scale_y, k*scale_z)"
    prep = None
    try:
        for vi, cfg in enumerate(volumes):
            progress(0.1 + 0.85 * vi / len(volumes), f"slicing {cfg['kind']} {cfg['dataset_path']}")
            # Before the ExitStack, not inside it: `prep` is a function-local that
            # the next iteration only rebinds *after* `prepare_volume` returns, so
            # without this the previous volume is alive while the next is built.
            # Closing the file does not release it — the array is not the handle.
            prep = None
            with contextlib.ExitStack() as open_files:
                try:
                    prep = prepare_volume(
                        cfg,
                        p,
                        scale_x,
                        scale_y,
                        samy_dir,
                        style=style,
                        stack=open_files,
                        budget_bytes=budget_bytes,
                    )
                except (KeyError, OSError, ValueError) as exc:
                    result.skipped.append(f"{cfg['dataset_path']}: {exc}")
                    continue
                if prep["clim_note"]:
                    msg = f"{prep['volume_id']}: {prep['clim_note']}"
                    progress(0.1 + 0.85 * vi / len(volumes), msg)
                    result.notes.append(msg)
                vg = open_volume_group(fh, prep)
                for sl in slices:
                    du = float(sl.get("du", prep["scale_x"]))
                    dv = float(sl.get("dv", prep["scale_x"]))
                    half_u, half_v = float(sl["half_u"]), float(sl["half_v"])
                    u_hat, v_hat, n_hat = build_basis(sl["normal"], sl.get("up"))
                    origin = np.asarray(sl["origin"], dtype=np.float64)
                    offsets = slice_plane_offsets(sl)
                    # The whole sweep's geometry is known before a single sample
                    # is taken, so the dataset is sized and written into
                    # plane-by-plane: on the shipped default geometry the stack
                    # is 2.90 GiB against a 1.15 GiB volume, and holding it
                    # (never mind `np.stack`'s transient copy of it) was the
                    # largest term in this stage by a factor of three.
                    u_um, v_um = _plane_axes(half_u, half_v, du, dv)
                    up_used = sl.get("up", None)
                    writer = PlaneWriter(
                        open_slice_dataset(
                            vg,
                            {
                                "name": sl["name"],
                                "u_um": u_um,
                                "v_um": v_um,
                                "offsets": offsets,
                                "normal": sl["normal"],
                                "origin": sl["origin"],
                                "up": up_used if up_used is not None else v_hat,
                                "u_hat": u_hat,
                                "v_hat": v_hat,
                                "n_hat": n_hat,
                                "half_u": half_u,
                                "half_v": half_v,
                                "du": du,
                                "dv": dv,
                                "sweep_step_um": float(sl.get("sweep_step_um") or 0.0),
                            },
                        )
                    )
                    planes = iter_sample_planes(
                        prep,
                        [origin + off * n_hat for off in offsets],
                        u_hat,
                        v_hat,
                        half_u,
                        half_v,
                        du,
                        dv,
                        max_resident_bytes=_sweep_resident_bytes(
                            budget_bytes, len(u_um) * len(v_um) * 4, writer.depth
                        ),
                    )
                    slice_dir = os.path.join(out_dir, sl["name"])
                    if save_png:
                        os.makedirs(slice_dir, exist_ok=True)
                    # Driven by the sampler, not by `zip(offsets, planes)`: zip
                    # stops on its *first* argument, which would leave the
                    # generator suspended on its last yield with `prep` — the
                    # whole aligned volume on the in-core rung — alive in its
                    # frame past the end of the loop.
                    for pi, s2d in enumerate(planes):
                        off = float(offsets[pi])
                        if save_png:
                            if len(offsets) == 1:
                                png = os.path.join(slice_dir, f"{prep['volume_id']}.png")
                                save_slice_png(
                                    prep, sl, s2d, u_um, v_um, png, offset_um=None, style=style
                                )
                            else:
                                png = os.path.join(
                                    slice_dir,
                                    f"{prep['volume_id']}__p{pi:03d}_{off:+08.2f}um.png",
                                )
                                save_slice_png(
                                    prep, sl, s2d, u_um, v_um, png, offset_um=off, style=style
                                )
                            result.pngs.append(png)
                        writer.append(s2d)
                        s2d = None  # the writer copied it; do not hold it into the next
                    writer.close()
                    writer = planes = None
                    result.n_planes_total += len(offsets)
                    grids_by_slice.setdefault(sl["name"], []).append(
                        (prep["volume_id"], (len(v_um), len(u_um)))
                    )
                    if sl["name"] not in result.slice_names:
                        result.slice_names.append(sl["name"])
                result.volume_ids.append(prep["volume_id"])
    finally:
        fh.close()

    # Volumes with different pixel scales sample the same plane onto different
    # (nv, nu) grids when the slice spec has no explicit du/dv. Downstream
    # (profiles) can then only mix fields that share a grid — say so up front.
    for sname, entries in grids_by_slice.items():
        by_shape: dict[tuple[int, int], list[str]] = {}
        for vid, shape in entries:
            by_shape.setdefault(shape, []).append(vid)
        if len(by_shape) > 1:
            desc = "; ".join(
                f"{nv}×{nu} px: {', '.join(vids)}" for (nv, nu), vids in sorted(by_shape.items())
            )
            msg = (
                f"slice {sname!r}: plane grids differ across volumes — {desc}. "
                "Profiles jobs can only mix fields on one grid; set explicit "
                "du/dv in slices_json to sample all volumes onto the same grid."
            )
            result.notes.append(msg)
            progress(0.99, msg)

    result.output_h5 = out_h5
    progress(1.0, f"sliced {len(result.volume_ids)} volumes -> {os.path.basename(out_h5)}")
    return result


@register("slices")
def figures(result: SlicesResult, params: dict) -> list[FigureSpec]:
    """One map FigureSpec per plane per slice subgroup per volume in oblique_slices.h5."""
    import h5py

    if not result.output_h5:
        return []
    specs = []
    # attrs are read defensively inside _rebuild_plane_figure, so one group from an
    # older/partial run missing an attr is rendered with fallbacks, not skipped.
    with h5py.File(result.output_h5, "r") as f:
        for vid in f.keys():
            vg = f[vid]
            if vid == MARKS_GROUP or not isinstance(vg, h5py.Group):
                continue
            for sname in vg.keys():
                sub = vg[sname]
                if not (isinstance(sub, h5py.Group) and "slices" in sub):
                    continue
                n_planes = sub["slices"].shape[0]
                for k in range(n_planes):

                    def build(style, vid=vid, sname=sname, k=k):
                        return _rebuild_plane_figure(result.output_h5, vid, sname, k, style)

                    specs.append(
                        FigureSpec(
                            figure_id=f"slice_{vid}_{sname}_{k:03d}",
                            title=f"{vid} / {sname} / plane {k}",
                            kind="map",
                            filename=f"{vid}_{sname}_{k:03d}",
                            build=build,
                        )
                    )
    return specs


def _rebuild_plane_figure(h5_path, vid, sname, k, style, *, clim=None, roi=None) -> Figure | None:
    """Rebuild one plane's slice figure from an oblique_slices.h5 group.

    Shared by :func:`figures` (catalog/export) and :func:`render_replot` so the
    prep-from-attrs reconstruction lives in exactly one place. ``clim`` is an
    optional ``(vmin, vmax)`` override; ``None`` entries keep the stored value.
    ``roi`` is an optional ``(r0, r1, c0, c1)`` pixel-index crop; returns
    ``None`` when the clamped crop is empty.
    """
    with h5py.File(h5_path, "r") as f:
        vg = f[vid]
        kind = str(vg.attrs.get("kind", ""))
        prep = {
            "cmap_name": str(vg.attrs.get("cmap", "magma")),
            "title": str(vg.attrs.get("title", vid)),
            "cbar_label": str(vg.attrs.get("cbar_label", "")),
            "vmin": float(vg.attrs.get("vmin", 0.0)),
            "vmax": float(vg.attrs.get("vmax", 1.0)),
            "center_zero": kind in _CENTERED_KINDS,
        }
        sg = vg[sname]
        s2d = sg["slices"][k]
        u = sg["u_um"][:]
        v = sg["v_um"][:]
        off = float(sg["offsets_um"][k])
    if roi is not None:
        cropped = crop_roi_2d(s2d, roi)
        if cropped is None:
            return None
        r0, r1, c0, c1 = roi
        h, w = s2d.shape[:2]
        r0 = max(0, min(int(r0), h))
        r1 = max(0, min(int(r1), h))
        c0 = max(0, min(int(c0), w))
        c1 = max(0, min(int(c1), w))
        s2d, u, v = cropped, u[c0:c1], v[r0:r1]
    if clim is not None:
        vmin_o, vmax_o = clim
        if vmin_o is not None:
            prep["vmin"] = float(vmin_o)
        if vmax_o is not None:
            prep["vmax"] = float(vmax_o)
    prep["cmap_name"] = resolve_cmap(style, GROUP_BY_KIND.get(kind), fallback=prep["cmap_name"])
    prep["group"] = GROUP_BY_KIND.get(kind)
    return build_slice_figure(prep, {"name": sname}, s2d, u, v, offset_um=off, style=style)


def replot_catalog(h5_path: str) -> list[ReplotEntry]:
    """List every (volume_id, slice_name, n_planes, offsets_um) in an oblique_slices.h5."""
    entries: list[ReplotEntry] = []
    with h5py.File(h5_path, "r") as f:
        for vid in f.keys():
            vg = f[vid]
            if vid == MARKS_GROUP or not isinstance(vg, h5py.Group):
                continue
            kind = str(vg.attrs.get("kind", ""))
            group = GROUP_BY_KIND.get(kind, kind)
            for sname in vg.keys():
                sg = vg[sname]
                if not (isinstance(sg, h5py.Group) and "slices" in sg):
                    continue
                offsets = [float(o) for o in sg["offsets_um"][:]]
                entries.append(
                    ReplotEntry(
                        vid,
                        sname,
                        int(sg["slices"].shape[0]),
                        offsets,
                        shape=tuple(sg["slices"].shape[1:]),
                        group=group,
                    )
                )
    return entries


def plane_preview(h5_path: str, volume_id: str, slice_name: str) -> tuple[np.ndarray, float, float]:
    """Middle plane of a slice group + its (du, dv) µm/px pitch, for the ROI picker.

    Returns ``(array2d, sx, sy)`` where ``sx=du`` (cols/u/X) and ``sy=dv``
    (rows/v/Y) come from the stored ``u_um``/``v_um`` axes — the resampled slice
    pitch, NOT the detector pixel scale.
    """
    with h5py.File(h5_path, "r") as f:
        sg = f[f"{volume_id}/{slice_name}"]
        stack = sg["slices"]
        mid = stack.shape[0] // 2
        arr = stack[mid][...]
        u = sg["u_um"][:]
        v = sg["v_um"][:]
    du = float(abs(u[1] - u[0])) if len(u) > 1 else 1.0
    dv = float(abs(v[1] - v[0])) if len(v) > 1 else 1.0
    return np.asarray(arr, dtype=float), du, dv


def render_replot(
    h5_path: str,
    selections: list[tuple[str, str, list[int] | None]],
    style: PlotStyle | None,
    clim: tuple[float | None, float | None] | dict[str, tuple] | None,
    out_dir: str,
    roi: tuple[int, int, int, int] | None = None,
    *,
    dpi: int = 150,
) -> list[str]:
    """Rebuild + save the selected planes (appearance-only; no resampling).

    ``selections`` is a list of ``(volume_id, slice_name, plane_idxs)`` where
    ``plane_idxs`` is ``None`` for all planes. PNGs are written under
    ``{out_dir}/{slice_name}/`` mirroring the slices run layout; returns the
    written paths. ``clim`` overrides the stored colour limits: ``None`` keeps
    them, a single ``(vmin, vmax)`` applies to every plane, and a
    ``{key: (vmin, vmax)}`` mapping sets them per quantity — keyed by
    ``volume_id`` (e.g. ``mosa_com_chi`` vs ``mosa_com_mu``, each ``raw_*``),
    falling back to the colormap group (``mosa_com``/``mosa_fwhm``/``strain``/
    ``raw``) for keys not found by volume_id. ``roi`` is an optional
    ``(r0, r1, c0, c1)`` pixel-index crop applied to every rebuilt plane; planes
    whose clamped crop is empty are silently skipped.
    """
    catalog = {(e.volume_id, e.slice_name): e for e in replot_catalog(h5_path)}
    written: list[str] = []
    for vid, sname, plane_idxs in selections:
        entry = catalog.get((vid, sname))
        if entry is None:
            continue
        idxs = list(range(entry.n_planes)) if plane_idxs is None else list(plane_idxs)
        clim_k = resolve_clim(clim, entry.volume_id)
        if clim_k is None:
            clim_k = resolve_clim(clim, entry.group)
        slice_dir = os.path.join(out_dir, sname)
        os.makedirs(slice_dir, exist_ok=True)
        for k in idxs:
            if k < 0 or k >= entry.n_planes:
                continue
            fig = _rebuild_plane_figure(h5_path, vid, sname, k, style, clim=clim_k, roi=roi)
            if fig is None:
                continue
            if entry.n_planes == 1:
                png = os.path.join(slice_dir, f"{vid}.png")
            else:
                png = os.path.join(slice_dir, f"{vid}__p{k:03d}_{entry.offsets_um[k]:+08.2f}um.png")
            fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight")
            written.append(png)
    return written


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Extract oblique slices from aligned volumes.")
    ap.add_argument("--mosa-volume-file", default="")
    ap.add_argument("--strain-volume-file", default="")
    ap.add_argument("--aligned-rocking-file", default="")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args(argv)
    res = run(
        dict(
            mosa_volume_file=args.mosa_volume_file,
            strain_volume_file=args.strain_volume_file,
            aligned_rocking_file=args.aligned_rocking_file,
            raw_root=args.raw_root,
            output_dir=args.output_dir,
            save_png=not args.no_png,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\nsliced {len(res.volume_ids)} volumes -> {res.output_h5}; planes {res.n_planes_total}")
    return 0


def roi_previews(params: dict) -> list:
    """(label, thunk) ROI-picker previews from the stacked mosa/strain volume(s)."""
    from ..common.figures import stacked_volume_previews

    return stacked_volume_previews(params)


if __name__ == "__main__":
    raise SystemExit(_main())
