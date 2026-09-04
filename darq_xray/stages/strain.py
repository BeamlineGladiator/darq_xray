"""Strain stage — per-pixel axial strain (cot method, ccmth-only) + 3D stacking.

Port of the legacy ``calc_axial_strain_v7_batch`` calculator:

    ε = cot(ccmth_ref) · Δccmth

Pipeline (steps 1-5 run per layer; order is a physics constraint —
**detrend before ROI**):

1. load the ccmth Center-of-mass map from maps.h5;
2. detrend ccmth on the *full* map (separable 2-D arctan);
3. crop the ROI;
4. compute strain;
5. save per-layer diagnostic plots (when ``save_plots``);
6. stack all layers into a 3-D volume.

Plotting uses the explicit Figure/Agg API (no pyplot) so this module is safe
to import in the Qt GUI process.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from ..common import progress as _progress_mod
from ..common.errors import StageUserError
from ..common.figures import FigureSpec, ReplotGroup, crop_roi_2d, register, resolve_clim
from ..common.h5io import StackedVolumeFile
from ..common.plotting import (
    AGG_RENDER_BYTES_PER_PIXEL,
    PlotStyle,
    add_colorbar,
    apply_axes_mode,
    apply_round_clim,
    apply_text_scale,
    build_histogram,
    canvas_pixels,
    draw_scale_bar,
    figure_size,
    fit_axes_to_box,
    fixed_scale_box,
    physical_extent,
    resolve_cmap,
    style_from_params,
    styled_figure,
    symmetric_limits,
)
from ..common.sort import find_matching_folders, resolve_layer_work
from ..config.models import CostEstimate, Param, ParamType, SeeAlso, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


STAGE = StageSpec(
    name="strain",
    label="Axial strain",
    description=(
        "Turns the darfix maps.h5 centre-of-mass maps into per-layer axial strain maps "
        "(cot method) and stacks them into a 3-D volume. Needs maps.h5 from darfix in each "
        "layer folder; writes per-layer PNGs plus stacked_strain_volumes.h5."
    ),
    params=(
        Param(
            "mode",
            ParamType.ENUM,
            "Mode",
            default="batch",
            choices=("single", "batch"),
            help=(
                "single processes one layer folder ('Input folder'); batch processes every "
                "subfolder of 'Root folder' matching 'Folder pattern'."
            ),
        ),
        Param(
            "input_folder",
            ParamType.DIR,
            "Input folder",
            must_exist=True,
            help="Layer folder containing the darfix maps.h5 (single mode only).",
        ),
        Param(
            "root_folder",
            ParamType.DIR,
            "Root folder",
            must_exist=True,
            help=(
                "Parent of the layer folders (batch mode only); every matching subfolder "
                "with a maps.h5 becomes one layer of the volume."
            ),
        ),
        Param(
            "folder_pattern",
            ParamType.STR,
            "Folder pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help="Glob selecting which subfolders of the root are strain layers in batch mode.",
        ),
        Param(
            "maps_filename",
            ParamType.STR,
            "maps filename",
            default="maps.h5",
            advanced=True,
            group="Data layout",
            help="Filename of the darfix output inside each layer folder (normally maps.h5).",
        ),
        Param(
            "ccmth_com_path",
            ParamType.STR,
            "ccmth COM path",
            default="/entry/ccmth/Center of mass/Center of mass",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path of the ccmth centre-of-mass dataset inside maps.h5, as written by "
                "darfix. Only change for a non-standard darfix export."
            ),
        ),
        Param(
            "ccmth_ref_deg",
            ParamType.FLOAT,
            "ccmth reference",
            unit="deg",
            default=7.144,
            calibration=True,
            advanced=True,
            group="Calibration",
            help=(
                "Reference Bragg angle θ of the unstrained lattice, in degrees. Strain is "
                "cot(θ_ref)·Δccmth per pixel, so a wrong reference silently shifts and scales "
                "every strain value. Confirm against the beamline alignment for your experiment."
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
                "calibration. A wrong value does not change the computed strain (that comes "
                "only from ccmth), but it skews the physical scale of every map, volume, and "
                "scale bar downstream."
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
                "calibration. A wrong value does not change the computed strain (that comes "
                "only from ccmth), but it skews the physical scale of every map, volume, and "
                "scale bar downstream."
            ),
        ),
        Param(
            "roi",
            ParamType.STR,
            "ROI",
            default="",
            roi_group="crop",
            roi_axis="both",
            roi_frame="map",
            help=(
                "Region of interest as 'r0,r1,c0,c1' in map pixels (rows then columns, "
                "relative to the darfix window; blank = full image). Pre-filled from the "
                "experiment's analysis window. Cropped after detrending, so the trend fit "
                "always uses the full map — that order is a physics constraint."
            ),
        ),
        Param(
            "vmin",
            ParamType.STR,
            "vmin",
            default="",
            advanced=True,
            group="Appearance",
            help=(
                "Lower colour limit of the strain plots (blank = symmetric automatic limits). "
                "Display only — does not affect the saved data."
            ),
        ),
        Param(
            "vmax",
            ParamType.STR,
            "vmax",
            default="",
            advanced=True,
            group="Appearance",
            help=(
                "Upper colour limit of the strain plots (blank = symmetric automatic limits). "
                "Display only — does not affect the saved data."
            ),
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help=(
                "Where the per-layer diagnostic PNGs go (default: a strain_maps folder). "
                "The stacked 3-D volume is always written to the input/root folder, not here."
            ),
        ),
        Param(
            "stacked_filename",
            ParamType.STR,
            "Stacked filename",
            default="stacked_strain_volumes.h5",
            advanced=True,
            group="Output",
            help=(
                "Filename of the stacked 3-D strain volume written to the input/root folder. "
                "Downstream stages expect stacked_strain_volumes.h5."
            ),
        ),
        Param(
            "save_plots",
            ParamType.BOOL,
            "Save plots",
            default=True,
            advanced=True,
            group="Appearance",
            help=(
                "Write the per-layer diagnostic PNGs (raw, detrended, strain). Turn off for a "
                "faster volume-only run."
            ),
        ),
    ),
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” "
            "(left panel); the range fields in Advanced below are this stage's own.",
        ),
    ),
    estimate="darq_xray.stages.strain:estimate",
)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
@dataclass
class LayerResult:
    name: str
    shape: tuple[int, int]
    vmin: float
    vmax: float
    mean: float
    std: float
    plots: list[str] = field(default_factory=list)
    # Absolute path to the source maps.h5 this layer was computed from. Stored at
    # run() time so figures() can rebuild the detrend diagnostic without
    # reconstructing the folder from the layer name (which loses nested
    # folder_pattern components). Empty for results predating this field.
    maps_path: str = ""


@dataclass
class StrainResult:
    stacked_path: str | None = None
    volume_shape: tuple[int, int, int] | None = None
    output_dir: str = ""
    layers: list[LayerResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)


# -----------------------------------------------------------------------------
# Numeric core (faithful port)
# -----------------------------------------------------------------------------
def cot(angle_rad):
    return np.cos(angle_rad) / np.sin(angle_rad)


def _arctan_model(x, a, b, c, d):
    return a * np.arctan(b * (x - c)) + d


def _fit_arctan_1d(coords: np.ndarray, profile: np.ndarray):
    mid = coords[len(coords) // 2]
    p0 = [(profile[-1] - profile[0]) / np.pi, 4.0 / len(coords), mid, np.nanmedian(profile)]
    bounds_lo = [-np.inf, 0.0, coords[0], -np.inf]
    bounds_hi = [np.inf, np.inf, coords[-1], np.inf]
    try:
        popt, _ = curve_fit(
            _arctan_model, coords, profile, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=20000
        )
    except RuntimeError:
        popt = np.array(p0)
    return popt


def detrend_arctan_2d(map_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separable 2-D arctan detrend; preserves the original absolute level."""
    ny, nx = map_2d.shape
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    original_median = np.nanmedian(map_2d)

    profile_x = np.nanmedian(map_2d, axis=0)
    finite_x = np.isfinite(profile_x)
    popt_x = _fit_arctan_1d(xs[finite_x], profile_x[finite_x])
    arctan_x_1d_noc = _arctan_model(xs, *popt_x) - popt_x[3]
    residual_1 = map_2d - arctan_x_1d_noc[np.newaxis, :]

    profile_y = np.nanmedian(residual_1, axis=1)
    finite_y = np.isfinite(profile_y)
    popt_y = _fit_arctan_1d(ys[finite_y], profile_y[finite_y])
    arctan_y_1d_noc = _arctan_model(ys, *popt_y) - popt_y[3]
    residual_2 = residual_1 - arctan_y_1d_noc[:, np.newaxis]

    residual_2 += original_median - np.nanmedian(residual_2)
    surface = arctan_x_1d_noc[np.newaxis, :] + arctan_y_1d_noc[:, np.newaxis]
    return residual_2, surface


def apply_roi(map_2d: np.ndarray, roi: list | None) -> np.ndarray:
    if roi is None:
        return map_2d
    r0, r1, c0, c1 = roi
    rows, cols = map_2d.shape
    out_of_bounds = r0 < 0 or c0 < 0 or r0 >= rows or c0 >= cols or r1 > rows or c1 > cols
    empty = r1 <= r0 or c1 <= c0
    if out_of_bounds or empty:
        raise StageUserError(
            f"ROI rows {r0},{r1} / cols {c0},{c1} do not fit this map (shape {rows}x{cols} px)",
            hint=(
                f"this map is {rows}x{cols} px but ROI rows are {r0},{r1} and columns "
                f"{c0},{c1} — the experiment's analysis window may describe a different "
                "dataset; blank the ROI or fix the experiment"
            ),
        )
    return map_2d[r0:r1, c0:c1]


def compute_strain(
    ccmth_map_deg: np.ndarray,
    ccmth_ref_deg: float,
) -> np.ndarray:
    """Per-pixel axial strain (cot method), ccmth-only.

    ``ε = cot(ccmth_ref) · (ccmth − ccmth_ref)`` with angles converted to radians.
    """
    ccmth_rad = np.deg2rad(ccmth_map_deg)
    ccmth_ref_rad = np.deg2rad(ccmth_ref_deg)
    return cot(ccmth_ref_rad) * (ccmth_rad - ccmth_ref_rad)


def load_map(filepath: str, dataset_path: str) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        if dataset_path not in f:
            raise KeyError(f"dataset {dataset_path!r} not found in {filepath!r}")
        return f[dataset_path][:]


# --- Recalibrated 2026-08-26 against measured child peak RSS ------------------
#
# `run()` processes one layer and drops it, so the peak does not scale with
# `n_layers`. Every constant below was measured on this box with
# `tests/peak_rss.py` over six layer shapes spanning 20x (0.262 M to 5.243 M
# elements) at eight layers, plus three fixed-scale styles; the numbers and the
# fits are reproduced in `docs/Codebase.md`.

# The detrend chain's own arrays, per element of a FULL layer: `load_map`'s
# read, its `.copy()`, `residual_1`, `residual_2`, `surface`, `np.nanmedian`'s
# internal copy, and the `strain` map — seven float64. Measured with
# `save_plots` off: 43.0 B/elem by least squares (r2 0.992) with local slopes
# 30.7-52.3, and a 102.0 MiB intercept that lands on the independently measured
# 96.7 MiB process floor. 56 is seven whole copies, above the measured slope.
STRAIN_ARRAY_BYTES_PER_ELEM = 56

# The process image of a `save_plots=False` run: interpreter, numpy, h5py, and
# this module. Measured 96.7 MiB (`measure_process_floor`), charged with slack.
STRAIN_PROCESS_FLOOR_BYTES = 112 * 1024 * 1024

# What importing and using matplotlib's Agg stack adds to the process image.
# Measured as the difference between the two floors, 166.2 - 96.7 = 69.5 MiB.
STRAIN_PLOT_PROCESS_BYTES = 72 * 1024 * 1024

# What rasterising the layer costs per element, on top of the arrays. The four
# `imshow` calls over a full map (one strain map, three detrend panels) each
# normalise to float64 and convert to uint8 RGBA, so this is data-dependent —
# the assumption that a plotting term is data-independent because "an Agg canvas
# is sized by inches and dpi" was only ever half true. Measured as the
# plots-on minus plots-off marginal: 51.7 B/elem. 64 is eight float64-equivalents.
STRAIN_PLOT_BYTES_PER_ELEM = 64


def estimate(params: dict) -> CostEstimate:
    """Peak memory for a strain run, from HDF5 shapes only.

    Reads ``.shape``/``.dtype`` of ONE layer, never touching data, so the GUI
    can call this on every form change. Never raises: an unreadable input
    reports an unknown cost with the reason in ``note``.

    **What this models.** ``run()`` computes one layer, ``append``s it to a
    ``StackedVolumeFile`` and drops it, so the peak is one layer's working set —
    it does **not** scale with ``n_layers``, and the figure this estimator
    returned until 2026-08-26 (``2 * n_layers * H * W * 8``, a model of the
    deleted accumulate-then-``np.stack`` code) over-predicted the real STO2 run
    by **5.2x**. The recalibrated model has four terms, each measured:

    * :data:`STRAIN_PROCESS_FLOOR_BYTES` — the child's process image.
    * :data:`STRAIN_ARRAY_BYTES_PER_ELEM` x the **full** layer, for the detrend
      chain. Full, not ROI-cropped, on purpose: ``_detrend_ccmth`` detrends
      before it crops (the project invariant), and ``apply_roi`` returns a
      *view*, which keeps the whole parent array alive.
    * with ``save_plots`` on, :data:`STRAIN_PLOT_PROCESS_BYTES` plus
      :data:`STRAIN_PLOT_BYTES_PER_ELEM` x the layer, for matplotlib's import
      and its per-element rasterisation.
    * with ``save_plots`` on, the **canvas**:
      :data:`~darq_xray.common.plotting.AGG_RENDER_BYTES_PER_PIXEL` x the pixels of
      the two full-size figures, sized through :func:`strain_map_geometry` so
      the estimate follows the same style the run will use. This term is why a
      style is read here at all: a ``scale_um_per_cm`` of 6.4 grows the strain
      map to the 30-inch clamp, and the measured peak with it, from 465 MiB to
      **1682 MiB**. A model blind to the style under-predicts that by 3.6x, and
      under-prediction is the direction that greenlights a run which then OOMs.

    **Currency.** ``peak_bytes`` here is *whole-child peak RSS*, process image
    included, because that is what :func:`~darq_xray.common.advice.plan_run` compares
    against ``headroom_bytes`` and prints to the user as "needs X RAM". The
    three alignment-chain estimators (visualize/slices/paraview) still report
    allocations only and cover their floor by over-predicting instead; that
    inconsistency is recorded in ``docs/Codebase.md`` and should be closed when
    those are next recalibrated, not papered over here.

    ``total_input`` and the reported ``shape`` still scale with ``n_layers``.
    """
    p = {**STAGE.defaults(), **params}
    try:
        work = resolve_layer_work(p, maps_filename=str(p["maps_filename"] or "maps.h5"))
        if not work:
            return CostEstimate(0, 0, None, True, "no layer folders resolved yet")
        ds_path = str(p["ccmth_com_path"])
        with h5py.File(work[0], "r") as f:
            if ds_path not in f:
                return CostEstimate(0, 0, None, True, f"{ds_path!r} not in {work[0]!r}")
            ds = f[ds_path]
            layer_shape = tuple(int(d) for d in ds.shape)
            itemsize = int(ds.dtype.itemsize)
    except Exception as exc:  # noqa: BLE001 - an estimate is advisory, never fatal
        return CostEstimate(0, 0, None, True, f"cannot size input: {type(exc).__name__}")

    layer_elems = 1
    for dim in layer_shape:
        layer_elems *= dim
    n_layers = len(work)
    input_bytes = n_layers * layer_elems * itemsize

    peak_bytes = STRAIN_PROCESS_FLOOR_BYTES + STRAIN_ARRAY_BYTES_PER_ELEM * layer_elems
    if p["save_plots"]:
        peak_bytes += STRAIN_PLOT_PROCESS_BYTES + STRAIN_PLOT_BYTES_PER_ELEM * layer_elems
        peak_bytes += AGG_RENDER_BYTES_PER_PIXEL * _plot_canvas_pixels(p, layer_shape)
    return CostEstimate(
        int(peak_bytes),
        input_bytes,
        (n_layers, *layer_shape),
        True,
        None,
    )


def _plot_canvas_pixels(p: dict, layer_shape: tuple[int, ...]) -> int:
    """Canvas pixels the two full-size per-layer figures rasterise to.

    Sized from the **ROI-cropped** map, because that is what
    ``process_maps_file`` hands ``build_strain_map`` — and unlike the array
    term, cropping does not merely shrink this one. The legacy and
    ``figure_width`` geometries fix the width and take the height from the
    physical aspect, so a narrow **column** crop makes the figure *taller*: at
    the real STO2 pixel sizes, cropping 1832 columns to 200 takes the canvas
    from 3.9 to 32 Mpx. Charging the un-cropped layer here would under-predict
    that by 8x, and under-prediction is the direction that OOMs.

    The two figures are charged as a sum although they render sequentially, so
    the true peak is the larger of them. That over-charge is deliberate and is
    what covers the third figure — the histogram — which is not modelled at all.

    Degrades to the legacy geometry at the layer's **taller** orientation if the
    style or the pixel sizes are unusable — ``max(ny/nx, nx/ny)``, because the
    aspect is what sets the height and a bare ``ny/nx`` picks the cheaper of the
    two for no reason. That is still a floor rather than a bound: the real
    geometry uses the *physical* aspect ``(ny*py)/(nx*px)``, which without usable
    pixel sizes is not derivable. It is not a hazard, because a form whose pixel
    sizes do not parse cannot run at all — ``run()`` calls
    ``float(p["pixel_size_x_um"])`` on the way into every layer — so this figure
    only ever describes a form the user is still filling in.
    """
    ny, nx = 1, 1
    try:
        ny, nx = int(layer_shape[-2]), int(layer_shape[-1])
        roi = _parse_roi(p.get("roi"))
        if roi is not None:
            r0, r1, c0, c1 = roi
            if r1 > r0 and c1 > c0:
                # Into temporaries: the fallback below sizes the legacy geometry
                # from ny/nx, so an out-of-range ROI must not leave a negative
                # value behind for it to work from.
                cropped_ny = min(ny, r1) - max(0, r0)
                cropped_nx = min(nx, c1) - max(0, c0)
                if cropped_ny <= 0 or cropped_nx <= 0:
                    # The ROI lies outside the layer. `canvas_pixels` clamps a
                    # negative figure size to 0, which would silently delete the
                    # largest plotting term instead of falling back; raise so
                    # the legacy geometry below is used, as for any other
                    # unusable input.
                    raise ValueError("roi does not intersect the layer")
                ny, nx = cropped_ny, cropped_nx
        px, py = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
        figsize, _box = strain_map_geometry(style_from_params(p), nx * px, ny * py)
    except (KeyError, IndexError, TypeError, ValueError, ArithmeticError):
        aspect = max(ny / nx, nx / ny) if (nx and ny) else 1.0
        figsize = (7.0, 7.0 * aspect + 1.5)
    try:
        return canvas_pixels(figsize, STRAIN_MAP_DPI) + canvas_pixels(
            DETREND_DIAG_FIGSIZE, DETREND_DIAG_DPI
        )
    except (TypeError, ValueError, ArithmeticError):
        return canvas_pixels((7.0, 8.5), STRAIN_MAP_DPI) + canvas_pixels(
            DETREND_DIAG_FIGSIZE, DETREND_DIAG_DPI
        )


# -----------------------------------------------------------------------------
# Plotting (Figure/Agg API — no pyplot)
# -----------------------------------------------------------------------------
# The dpi each per-layer plot is saved at, and the detrend diagnostic's fixed
# figure size. Constants rather than literals at the `savefig` calls because
# `estimate()` sizes the same canvases (`plotting.canvas_pixels`) and a dpi
# changed in one place only would silently re-open the under-prediction this
# recalibration closed.
STRAIN_MAP_DPI = 200
STRAIN_HIST_DPI = 150
DETREND_DIAG_DPI = 120
DETREND_DIAG_FIGSIZE = (20, 6)


def strain_map_geometry(
    style: PlotStyle | None, ext_x_um: float, ext_y_um: float
) -> tuple[tuple[float, float], tuple[float, float, float] | None]:
    """``(figsize_inches, fixed_scale_box)`` for the per-layer strain map.

    The one place the strain map's figure size is decided, shared by
    :func:`build_strain_map` (which renders it) and :func:`estimate` (which
    prices the canvas). They drifting apart is exactly how a fixed-scale style
    could grow the canvas 6x with the estimate unmoved — though sharing the
    function is not sufficient on its own: the two call sites passed *different*
    extents (full layer vs ROI-cropped) until a review caught it, so check the
    arguments as well as the callee.

    **One behaviour change from the inline code this replaced**: a zero
    ``ext_x_um`` (a blank pixel size, or a zero-width layer) used to raise
    ``ZeroDivisionError`` out of ``build_strain_map``; it now yields aspect 1.0
    and a ``(7.0, 8.5)`` figure. Rendering something square beats crashing a
    render, and the estimator needs a number rather than an exception.
    """
    legacy = (7.0, 7.0 * (ext_y_um / ext_x_um if ext_x_um else 1.0) + 1.5)
    if style is None:
        return legacy, None
    box = fixed_scale_box(style, ext_x_um, ext_y_um)
    if box is not None:
        # headroom for labels/colourbar; fit_axes_to_box converges regardless
        return (box[0] + 1.5, box[1] + 1.5), box
    return (figure_size(style, ext_x_um, ext_y_um) or legacy), None


def build_strain_map(
    strain: np.ndarray,
    px: float,
    py: float,
    roi: list | None,
    vlim: tuple[float | None, float | None],
    *,
    style: PlotStyle | None = None,
) -> Figure:
    """Build and return a strain-map Figure.

    When *style* is ``None`` the legacy look is reproduced exactly (RdBu_r,
    fig.colorbar at fraction=0.046/pad=0.02, no scale bar). When a
    :class:`~darq_xray.common.plotting.PlotStyle` is supplied, colourbar and fonts
    are routed through the shared helpers and a scale bar is drawn when
    ``style.scale_bar`` is ``True``.

    The caller is responsible for calling ``fig.savefig``.
    """
    extent = physical_extent(strain.shape, px, py, roi)
    if vlim != (None, None):
        vmin, vmax = vlim  # user-specified limits are never rounded
    else:
        vmin, vmax = symmetric_limits(strain)
        vmin, vmax, _ = apply_round_clim(vmin, vmax, style)

    ny, nx = strain.shape
    figsize, box = strain_map_geometry(style, nx * px, ny * py)

    fig = styled_figure(figsize, styled=style is not None)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        strain,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap=resolve_cmap(style, "strain"),
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Strain map (cot method)")

    if style is None:
        # legacy: plain fig.colorbar, no scale bar
        fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046).set_label("Strain (ε)")
    else:
        add_colorbar(fig, im, ax, "Strain (ε)", style, group="strain")
        apply_text_scale(ax, style)
        apply_axes_mode(ax, style)
        if style.scale_bar:
            draw_scale_bar(
                ax,
                style.scale_bar_length_um,
                style=style,
                fixed_scale_um_per_cm=(box[2] if box is not None else None),
            )

    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])
    return fig


def build_strain_histogram(
    data: np.ndarray,
    *,
    title: str = "Strain distribution",
    xlabel: str = "Strain (ε)",
    style: PlotStyle | None = None,
) -> Figure | None:
    """Build and return a strain-histogram Figure, or ``None`` when *data* has no finite values.

    Thin wrapper around :func:`darq_xray.common.plotting.build_histogram` that
    preserves the strain-specific signature and defaults. When *style* is
    ``None`` the legacy look is reproduced exactly. The caller is responsible
    for calling ``fig.savefig``.
    """
    return build_histogram(data, title=title, xlabel=xlabel, style=style)


def build_detrend_diag(
    original: np.ndarray,
    detrended: np.ndarray,
    surface: np.ndarray,
    *,
    style: PlotStyle | None = None,
) -> Figure:
    """Build and return a 3-panel detrend-diagnostic Figure.

    When *style* is ``None`` the legacy look is reproduced exactly. The caller
    is responsible for calling ``fig.savefig``.
    """
    fig = styled_figure(DETREND_DIAG_FIGSIZE, styled=style is not None)
    axes = fig.subplots(1, 3)
    for ax, title, d in zip(
        axes,
        ["Original (after ROI)", "Arctan surface", "Detrended"],
        [original, surface, detrended],
    ):
        valid = d[np.isfinite(d)]
        if valid.size == 0:
            continue
        vlo, vhi = np.percentile(valid, [1, 99])
        im = ax.imshow(
            d, origin="lower", cmap=resolve_cmap(style, "strain"), vmin=vlo, vmax=vhi, aspect="auto"
        )
        ax.set_title(title)
        if style is not None:
            add_colorbar(fig, im, ax, title, style, group="strain")
            apply_text_scale(ax, style)
        else:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


# -----------------------------------------------------------------------------
# Figure catalog
# -----------------------------------------------------------------------------
def _detrend_ccmth(
    maps_path: str, ccmth_com_path: str, roi: list | None, progress=None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ccmth, detrend (arctan surface), then ROI-crop. Returns (original, detrended, surface).

    Detrend BEFORE ROI (project invariant). Used by both process_maps_file (run) and
    figures()._build_detrend (export rebuild) so the two can never drift.

    *progress* takes a local 0..1 fraction. The two slow steps are the HDF5 read
    and the arctan surface fit over the full map — the crops after them are
    views. Together they are the first half of a layer, and before this the
    whole of it passed in silence: a one-layer run (`single` mode, which is one
    layer by definition) said nothing at all until 0.45.
    """
    report = progress or _progress_mod.noop
    ccmth_map = load_map(maps_path, ccmth_com_path)
    report(0.55, "ccmth read")
    ccmth_original = ccmth_map.copy()
    ccmth_map, surface = detrend_arctan_2d(ccmth_map)
    report(1.0, "detrended")
    ccmth_map = apply_roi(ccmth_map, roi)
    surface = apply_roi(surface, roi)
    ccmth_original = apply_roi(ccmth_original, roi)
    return ccmth_original, ccmth_map, surface


def _derive_maps_path(layer_name: str, params: dict) -> str:
    """Reconstruct the maps.h5 path for *layer_name* exactly as run() does.

    In single mode: ``<input_folder>/<maps_filename>``.
    In batch mode:  ``<root_folder>/<layer_name>/<maps_filename>``.
    """
    maps_filename = params.get("maps_filename", "maps.h5")
    mode = params.get("mode", "batch")
    if mode == "single":
        folder = params.get("input_folder") or ""
        return os.path.join(folder, maps_filename)
    else:
        root = (params.get("root_folder", "") or "").rstrip("/")
        return os.path.join(root, layer_name, maps_filename)


def roi_previews(params: dict) -> list:
    """Return ``(label, thunk)`` pairs for the ROI picker — the CoM map + pixel scales.

    Best-effort: returns ``[]`` when the maps file can't be resolved from the form.
    The picker draws on the same 2-D map the run-time ``roi`` crops.

    Each thunk returns ``(array2d, sx_um, sy_um)`` where *array2d* is the raw
    ccmth Center-of-mass map (before detrending/ROI) and *sx*/*sy* are the pixel
    sizes in µm.
    """
    import os

    p = dict(params)
    try:
        if p.get("mode", "batch") == "single":
            maps_path = os.path.join(
                p.get("input_folder", "") or "", p.get("maps_filename", "maps.h5")
            )
        else:
            root = (p.get("root_folder", "") or "").rstrip("/")
            folders = find_matching_folders(root, p.get("folder_pattern", "")) if root else []
            if not folders:
                return []
            maps_path = _derive_maps_path(os.path.basename(folders[0].rstrip("/")), p)
    except Exception:  # noqa: BLE001
        return []
    if not maps_path or not os.path.exists(maps_path):
        return []
    ccmth_com_path = p.get("ccmth_com_path", "/entry/ccmth/Center of mass/Center of mass")
    sx = float(p.get("pixel_size_x_um", 0.152))
    sy = float(p.get("pixel_size_y_um", 0.385))

    def _thunk(_mp=maps_path, _ds=ccmth_com_path, _sx=sx, _sy=sy):
        import numpy as np

        return np.asarray(load_map(_mp, _ds), dtype=float), _sx, _sy

    return [(f"CoM map · {os.path.basename(maps_path)}", _thunk)]


def _build_hist(stacked_path: str, layer_idx: int, layer_name: str, style) -> "Figure":
    """Load one strain layer from the stacked volume and return a histogram Figure."""
    with h5py.File(stacked_path, "r") as fh:
        arr = fh["strain"][layer_idx]
    fig = build_strain_histogram(
        arr,
        title=f"{layer_name} — Strain distribution",
        xlabel="Strain (ε)",
        style=style,
    )
    if fig is None:
        raise ValueError(f"layer {layer_name!r} has no finite strain values to histogram")
    return fig


def _build_detrend(
    layer_name: str,
    maps_path: str,
    ccmth_com_path: str,
    roi_text: str,
    style,
) -> "Figure":
    """Recompute the detrend diagnostic for *layer_name* from the source maps.h5.

    Reproduces exactly the computation in ``process_maps_file()`` so the rebuilt
    diagnostic matches the PNG saved during ``run()``.

    Raises ``FileNotFoundError`` when *maps_path* is absent.
    """
    if not os.path.exists(maps_path):
        raise FileNotFoundError(
            f"source maps.h5 not found; detrend diagnostic needs the original input"
            f" (looked for {maps_path!r})"
        )
    roi = _parse_roi(roi_text)
    orig, detrended, surface = _detrend_ccmth(maps_path, ccmth_com_path, roi)
    return build_detrend_diag(orig, detrended, surface, style=style)


@register("strain")
def figures(result: "StrainResult", params: dict) -> list[FigureSpec]:
    """Return map + histogram + detrend-diagnostic FigureSpecs per layer."""
    if not result.stacked_path:
        return []
    # Merge spec defaults so blank params fall back to the calibrated beamline
    # values (not 1.0), exactly as run() does.
    p = {**STAGE.defaults(), **params}
    px = float(p["pixel_size_x_um"])
    py = float(p["pixel_size_y_um"])
    ccmth_com_path = p["ccmth_com_path"]
    roi_text = str(p["roi"] or "")
    # The map must be rebuilt with the SAME roi + vlim run() used, so the export
    # matches the saved PNG: a blank vlim means symmetric zero-centred limits
    # (white = zero strain on RdBu_r), and the ROI sets the µm axis offset.
    roi = _parse_roi(roi_text)
    vlim = (_parse_float(p["vmin"]), _parse_float(p["vmax"]))
    specs = []
    for i, layer in enumerate(result.layers):

        def build_map(style, i=i, path=result.stacked_path, _roi=roi, _vlim=vlim):
            with h5py.File(path, "r") as fh:
                arr = fh["strain"][i]
            return build_strain_map(arr, px, py, _roi, _vlim, style=style)

        specs.append(
            FigureSpec(
                figure_id=f"strain_map_{i:04d}",
                title=f"Strain map — {layer.name}",
                kind="map",
                # NOTE: layer.name appears verbatim in the export file stem; sanitise
                # path-unsafe chars (spaces/parens/etc.) at the savefig/export site (Task 16),
                # not here — keep the human-readable name in title/filename.
                filename=f"{layer.name}_strain",
                build=build_map,
            )
        )

        # --- histogram (kind="plot") ---
        def build_hist(
            style,
            _path=result.stacked_path,
            _idx=i,
            _name=layer.name,
        ):
            return _build_hist(_path, _idx, _name, style)

        specs.append(
            FigureSpec(
                figure_id=f"strain_hist_{i:04d}",
                title=f"Strain histogram — {layer.name}",
                kind="plot",
                filename=f"{layer.name}_strain_hist",
                build=build_hist,
            )
        )

        # --- detrend diagnostic (kind="plot") ---
        # Prefer the maps.h5 path stored at run() time (correct for nested
        # folder patterns); fall back to reconstructing it for older results.
        _maps_path = layer.maps_path or _derive_maps_path(layer.name, params)

        def build_detrend(
            style,
            _name=layer.name,
            _mp=_maps_path,
            _com=ccmth_com_path,
            _roi=roi_text,
        ):
            return _build_detrend(_name, _mp, _com, _roi, style)

        specs.append(
            FigureSpec(
                figure_id=f"strain_detrend_{i:04d}",
                title=f"Detrend diagnostic — {layer.name}",
                kind="plot",
                filename=f"{layer.name}_strain_detrend",
                build=build_detrend,
            )
        )

    return specs


# -----------------------------------------------------------------------------
# Per-folder processing
# -----------------------------------------------------------------------------
def process_maps_file(
    maps_path: str,
    name: str,
    *,
    ccmth_com_path: str,
    ccmth_ref_deg: float,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    roi: list | None,
    vlim: tuple[float | None, float | None],
    out_dir: str | None,
    save_plots: bool,
    progress=None,
    style: PlotStyle | None = None,
) -> tuple[np.ndarray, LayerResult]:
    """Compute the 2-D strain map for one maps.h5 and (optionally) save plots.

    *progress* takes a local 0..1 fraction over this layer's own work. There is
    no loop here to report from — reading and detrending the map, and then
    rendering each plot, are individually slow steps — so the boundaries between
    them are where it speaks, including the two inside `_detrend_ccmth`. It
    reports its own end rather than relying on the caller to close the layer,
    because a `save_plots=False` run has nothing after `compute_strain`.
    """
    report = progress or _progress_mod.noop
    # detrend ccmth on the FULL map, THEN crop ROI (order matters)
    ccmth_original, ccmth_map, surface = _detrend_ccmth(
        maps_path,
        ccmth_com_path,
        roi,
        progress=_progress_mod.sub_progress(progress, 0.0, 0.45),
    )
    report(0.45, f"{name}: detrended")

    strain = compute_strain(ccmth_map, ccmth_ref_deg)
    report(0.6, f"{name}: strain computed")

    plots: list[str] = []
    if save_plots and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"{name}_strain.png")
        build_strain_map(strain, pixel_size_x_um, pixel_size_y_um, roi, vlim, style=style).savefig(
            p, dpi=STRAIN_MAP_DPI, bbox_inches="tight", facecolor="white"
        )
        plots.append(p)
        report(0.75, f"{name}: strain map saved")
        ph = os.path.join(out_dir, f"{name}_hist.png")
        hist_fig = build_strain_histogram(strain, style=style)
        if hist_fig is not None:
            hist_fig.savefig(ph, dpi=STRAIN_HIST_DPI, bbox_inches="tight", facecolor="white")
            plots.append(ph)
        report(0.85, f"{name}: histogram saved")
        pd = os.path.join(out_dir, f"{name}_detrend_diag.png")
        build_detrend_diag(ccmth_original, ccmth_map, surface, style=style).savefig(
            pd, dpi=DETREND_DIAG_DPI, bbox_inches="tight", facecolor="white"
        )
        plots.append(pd)
    # Reported whether or not plots were saved: the layer is finished either
    # way, and leaving the last word to the plot branch left a `save_plots=False`
    # run reporting 0.6 and then nothing until the caller closed the layer.
    report(1.0, f"{name}: done")

    layer = LayerResult(
        name=name,
        shape=tuple(strain.shape),
        vmin=float(np.nanmin(strain)),
        vmax=float(np.nanmax(strain)),
        mean=float(np.nanmean(strain)),
        std=float(np.nanstd(strain)),
        plots=plots,
        maps_path=maps_path,
    )
    return strain, layer


# -----------------------------------------------------------------------------
# Helpers for string params
# -----------------------------------------------------------------------------
def _parse_roi(text) -> list | None:
    if text is None or str(text).strip() == "":
        return None
    parts = [int(p) for p in str(text).replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must be 'r0,r1,c0,c1', got {text!r}")
    return parts


def _parse_float(text) -> float | None:
    if text is None or str(text).strip() == "":
        return None
    return float(text)


# -----------------------------------------------------------------------------
# Stage entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> StrainResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    style = style_from_params(p)
    roi = _parse_roi(p["roi"])
    vlim = (_parse_float(p["vmin"]), _parse_float(p["vmax"]))
    maps_filename = p["maps_filename"]

    # resolve the (folder_name, maps_path) work list
    if p["mode"] == "single":
        folder = p["input_folder"]
        if not folder:
            raise StageUserError(
                "single mode requires 'input_folder'",
                hint=(
                    "Pick the layer folder holding maps.h5 in 'Input folder', "
                    "or switch Mode to 'batch'."
                ),
            )
        work = [(os.path.basename(folder.rstrip("/")), os.path.join(folder, maps_filename))]
        default_out_root = folder
    else:
        root = (p["root_folder"] or "").rstrip("/")
        if not root:
            raise StageUserError(
                "batch mode requires 'root_folder'",
                hint=(
                    "Pick the parent of the layer folders in 'Root folder', or "
                    "switch Mode to 'single'."
                ),
            )
        folders = find_matching_folders(root, p["folder_pattern"])
        if not folders:
            raise StageUserError(
                f"no folders matching {p['folder_pattern']!r} in {root}",
                hint=(
                    "Check 'Folder pattern' — it matched no subfolders. Each "
                    "matching folder must contain the darfix maps.h5."
                ),
            )
        work = [(os.path.basename(f), os.path.join(f, maps_filename)) for f in folders]
        default_out_root = root

    out_dir = p["output_dir"] or os.path.join(default_out_root, "strain_maps")
    result = StrainResult(output_dir=out_dir)

    stacked_path = os.path.join(default_out_root, p["stacked_filename"])
    LAYERS_HI = 0.95
    names: list[str] = []
    with StackedVolumeFile(stacked_path, compression="gzip") as out:
        # The layer loop owns [0, LAYERS_HI]; what is left is the stacked
        # file's attrs and close, which is real work on a large volume. Each
        # layer reports on entry AND on completion — reporting only on entry
        # left the last layer's own processing, and the whole write tail after
        # it, as a single jump to 1.0.
        for i, (name, maps_path) in enumerate(work):
            lay_lo, lay_hi = _progress_mod.slice_for(i, len(work), 0.0, LAYERS_HI)
            progress(lay_lo, f"strain: {name}")
            if not os.path.exists(maps_path):
                result.skipped.append(f"{name}: {maps_filename} not found")
                continue
            try:
                strain, layer = process_maps_file(
                    maps_path,
                    name,
                    ccmth_com_path=p["ccmth_com_path"],
                    ccmth_ref_deg=float(p["ccmth_ref_deg"]),
                    pixel_size_x_um=float(p["pixel_size_x_um"]),
                    pixel_size_y_um=float(p["pixel_size_y_um"]),
                    roi=roi,
                    vlim=vlim,
                    out_dir=out_dir,
                    save_plots=bool(p["save_plots"]),
                    style=style,
                    progress=_progress_mod.sub_progress(progress, lay_lo, lay_hi),
                )
            except StageUserError:
                # Out-of-bounds ROI etc. is an input problem affecting every layer the
                # same way — stop the run with a clear message rather than skip-and-continue.
                raise
            except (KeyError, OSError, ValueError) as exc:
                result.skipped.append(f"{name}: {exc}")
                continue
            out.append("strain", strain)
            del strain
            names.append(name)
            result.layers.append(layer)
            progress(lay_hi, f"strain: {name} done")

        if not names:
            out.abort()
            progress(1.0, "no strain layers produced")
            return result

        progress(LAYERS_HI, "writing stacked volume")
        out.set_attrs(
            num_layers=len(names),
            source_folders="\n".join(names),
            description="Stacked 3D strain volume (cot, ccmth-only)",
            ccmth_ref_deg=float(p["ccmth_ref_deg"]),
            scale_x_um=float(p["pixel_size_x_um"]),
            scale_y_um=float(p["pixel_size_y_um"]),
        )
        shape = out.shape("strain")

    result.stacked_path = stacked_path
    result.volume_shape = shape
    progress(1.0, f"stacked {len(names)} layers -> {os.path.basename(stacked_path)}")
    return result


# ---------------------------------------------------------------------------
# Cold replot helpers (re-render from stacked strain h5, no re-run needed)
# ---------------------------------------------------------------------------


def _rebuild_strain_map(
    h5_path: str,
    layer_idx: int,
    style,
    *,
    clim: tuple | None = None,
    roi: tuple | None = None,
    params: dict | None = None,
):
    """Rebuild one strain-map Figure from the stacked volume, cold, with clim/ROI.

    ``roi`` (pixel bounds ``(r0, r1, c0, c1)``) crops the stored layer; the
    cropped view uses a zero-origin extent (``roi=None`` passed to
    ``build_strain_map``).  ``clim`` overrides the symmetric vlim.  Returns
    ``None`` when the crop is empty.
    """
    params = params or {}
    with h5py.File(h5_path, "r") as fh:
        arr = fh["strain"][layer_idx]
        px = float(params.get("pixel_size_x_um") or fh.attrs.get("scale_x_um", 0.152))
        py = float(params.get("pixel_size_y_um") or fh.attrs.get("scale_y_um", 0.385))
    if roi is not None:
        cropped = crop_roi_2d(arr, roi)
        if cropped is None:
            return None
        arr = cropped
        extent_roi = None  # cropped view → zero-origin extent
    else:
        extent_roi = _parse_roi(str(params.get("roi", "") or ""))
    if clim is not None:
        vlim = clim
    else:
        vlim = (_parse_float(params.get("vmin", "")), _parse_float(params.get("vmax", "")))
    return build_strain_map(arr, px, py, extent_roi, vlim, style=style)


def replot_catalog(h5_path: str) -> list[ReplotGroup]:
    """Return a single ``ReplotGroup`` with one item per stored layer.

    Layer names come from ``f.attrs["source_folders"]`` (newline-joined).
    """
    with h5py.File(h5_path, "r") as f:
        raw = str(f.attrs.get("source_folders", ""))
        n = int(f.attrs.get("num_layers", f["strain"].shape[0]))
        shape = tuple(f["strain"].shape[1:])
    names = [s for s in raw.split("\n") if s] if raw else [f"layer {i}" for i in range(n)]
    if len(names) != n:
        names = [f"layer {i}" for i in range(n)]
    return [ReplotGroup(key="strain", label="Strain map", item_labels=names, shape=shape)]


def render_replot(
    h5_path: str,
    selections: list,
    style,
    clim: tuple | dict[str, tuple] | None,
    out_dir: str,
    roi: tuple | None = None,
    params: dict | None = None,
) -> list[str]:
    """Re-render selected strain-map layers cold from the stacked strain h5.

    ``selections`` is ``list[("strain", item_idxs | None)]``.  ``clim`` overrides
    the symmetric colour limits: ``None`` keeps them, a ``(vmin, vmax)`` tuple
    applies to every layer, and a ``{group_key: (vmin, vmax)}`` mapping is keyed
    by the ``ReplotGroup.key`` (``"strain"``).  PNGs are saved under
    ``{out_dir}/strain/``; the list of written paths is returned.
    """
    with h5py.File(h5_path, "r") as f:
        n_z = int(f["strain"].shape[0])
        names = str(f.attrs.get("source_folders", "")).split("\n")
    sub_dir = os.path.join(out_dir, "strain")
    os.makedirs(sub_dir, exist_ok=True)
    written: list[str] = []
    for key, idxs in selections:
        if key != "strain":
            continue
        clim_k = resolve_clim(clim, key)
        layer_list = list(range(n_z)) if idxs is None else list(idxs)
        for z in layer_list:
            if z < 0 or z >= n_z:
                continue
            fig = _rebuild_strain_map(h5_path, z, style, clim=clim_k, roi=roi, params=params)
            if fig is None:
                continue
            stem = names[z] if z < len(names) and names[z] else f"layer_{z:04d}"
            png = os.path.join(sub_dir, f"{stem}_strain.png")
            fig.savefig(png, dpi=200, facecolor="white", bbox_inches="tight")
            written.append(png)
    return written


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Axial strain (cot method).")
    ap.add_argument("--mode", choices=("single", "batch"), default="batch")
    ap.add_argument("--input-folder", default="")
    ap.add_argument("--root-folder", default="")
    ap.add_argument("--folder-pattern", default="*")
    ap.add_argument("--ccmth-ref", type=float, default=7.144)
    ap.add_argument("--roi", default="")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    res = run(
        dict(
            mode=args.mode,
            input_folder=args.input_folder,
            root_folder=args.root_folder,
            folder_pattern=args.folder_pattern,
            ccmth_ref_deg=args.ccmth_ref,
            roi=args.roi,
            save_plots=not args.no_plots,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\n{res.n_layers} layers; stacked -> {res.stacked_path}; skipped {len(res.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
