"""Shared plotting helpers, safe to import from both the GUI and the worker.

These deliberately avoid ``pyplot`` and never call ``matplotlib.use(...)``:
touching the global backend from a module that the GUI imports would clobber
the Qt backend that the embedded canvases need. Build figures with the
explicit :class:`~matplotlib.figure.Figure` API instead and save via
``fig.savefig`` (Agg is used implicitly, no global state changed).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, fields

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from .cmaps import register as _register_fast_cmap

_register_fast_cmap()

_log = logging.getLogger(__name__)

# Colormap quantity groups + the curated dropdown list (shared by the GUI).
CMAP_GROUPS: tuple[str, ...] = ("mosa_com", "mosa_fwhm", "strain", "raw")
CMAP_CHOICES: tuple[str, ...] = (
    "fast",
    "magma",
    "viridis",
    "plasma",
    "inferno",
    "cividis",
    "gray",
    "bone",
    "RdBu_r",
    "coolwarm",
    "seismic",
    "turbo",
)

# Volume "kind" (as stored in HDF5 attrs by the map stages) -> quantity group.
# Shared by slices and profiles so the kind->group mapping lives in one place.
GROUP_BY_KIND: dict[str, str] = {
    "mosa_com": "mosa_com",
    "mosa_fwhm": "mosa_fwhm",
    "strain": "strain",
    "raw_sum": "raw",
    "raw_specific": "raw",
    "raw_mosa_sum": "raw",
    "raw_mosa_specific": "raw",
}


@dataclass
class PlotStyle:
    """How to render a figure for export. ``None`` (not this) means 'as today'."""

    # scale bar (map figures only)
    scale_bar: bool = True
    scale_bar_length_um: float | None = None  # None -> auto (~15% of X extent)
    scale_bar_thickness_pt: float = 3.0
    scale_bar_label_scale: float = 1.0  # multiplies font_scale for the bar label
    scale_bar_loc: str = (
        "lower right"  # "lower right" | "lower left" | "upper right" | "upper left"
    )
    scale_bar_color: str = "black"
    scale_bar_box: bool = False
    scale_bar_box_color: str = "black"
    scale_bar_box_alpha: float = 0.45
    scale_bar_box_margin_pt: float = 4.0
    # text
    font_scale: float = 1.0  # multiplies axis labels + ticks (NOT the title)
    title_scale: float = 1.0  # multiplies the title alone (independent of font_scale)
    show_title: bool = True
    center_axis_labels: bool = True
    # colourbar
    colorbar: bool = True
    colorbar_label: str | None = None  # None -> the figure's own label
    colorbar_fraction: float = 0.046  # matplotlib colorbar `fraction` (thickness)
    colorbar_ticks: int = 0  # 0 -> matplotlib default; >=2 -> N evenly spaced incl min/mid/max
    # per-group colourbar number format: "auto" | "scientific" | "arb" | a digit count like "2"
    tickfmt_mosa_com: str = "auto"
    tickfmt_mosa_fwhm: str = "auto"
    tickfmt_strain: str = "auto"
    tickfmt_raw: str = "auto"
    # per-group scientific-notation ×10ⁿ offset text: size multiplier (×font_scale) + placement
    offset_scale_mosa_com: float = 1.0
    offset_scale_mosa_fwhm: float = 1.0
    offset_scale_strain: float = 1.0
    offset_scale_raw: float = 1.0
    offset_pos_mosa_com: str = "top"  # "top" | "bottom"
    offset_pos_mosa_fwhm: str = "top"
    offset_pos_strain: str = "top"
    offset_pos_raw: str = "top"
    round_clim: bool = False  # round auto colour limits outward to nice values
    # figure
    figure_width: str | float = "auto"  # "single" | "double" | "auto" | width in inches
    # fixed physical scale for MAP figures: µm of data per cm of page. None/blank = off.
    # When set (>0), figure_width is ignored for maps (trace figures keep it).
    scale_um_per_cm: float | None = None
    # output
    formats: tuple[str, ...] = ("png",)
    dpi: int = 300
    # per-quantity colormaps (see CMAP_GROUPS)
    cmap_mosa_com: str = "fast"
    cmap_mosa_fwhm: str = "magma"
    cmap_strain: str = "RdBu_r"
    cmap_raw: str = "gray"

    def cmap_for(self, group: str) -> str:
        """Colormap name for a quantity group (KeyError on unknown group)."""
        if group not in CMAP_GROUPS:
            raise KeyError(f"unknown colormap group {group!r}")
        return getattr(self, f"cmap_{group}")

    def tickfmt_for(self, group: str | None) -> str:
        """Tick format for a quantity group; ``group=None`` -> the neutral ``"auto"``."""
        if group is None:
            return "auto"
        if group not in CMAP_GROUPS:
            raise KeyError(f"unknown colormap group {group!r}")
        return getattr(self, f"tickfmt_{group}")

    def offset_scale_for(self, group: str | None) -> float:
        """Scientific-offset size multiplier for a group; ``group=None`` -> ``1.0``."""
        if group is None:
            return 1.0
        if group not in CMAP_GROUPS:
            raise KeyError(f"unknown colormap group {group!r}")
        return getattr(self, f"offset_scale_{group}")

    def offset_pos_for(self, group: str | None) -> str:
        """Scientific-offset placement for a group; ``group=None`` -> ``"top"``."""
        if group is None:
            return "top"
        if group not in CMAP_GROUPS:
            raise KeyError(f"unknown colormap group {group!r}")
        return getattr(self, f"offset_pos_{group}")


PUBLICATION_STYLE = PlotStyle(
    scale_bar=True,
    scale_bar_thickness_pt=4.0,
    scale_bar_label_scale=1.1,
    scale_bar_color="white",
    scale_bar_box=True,
    font_scale=2.2,
    colorbar_fraction=0.07,
    colorbar_ticks=5,
    tickfmt_mosa_com="auto",
    tickfmt_mosa_fwhm="auto",
    tickfmt_strain="scientific",
    tickfmt_raw="arb",
    offset_pos_mosa_com="bottom",
    offset_pos_mosa_fwhm="bottom",
    offset_pos_strain="bottom",
    offset_pos_raw="bottom",
    figure_width="single",
    formats=("png", "pdf", "svg"),
    dpi=300,
)


def resolve_cmap(style: PlotStyle | None, group: str | None, fallback: str = "magma") -> str:
    """Colormap for *group* from *style* (or the PlotStyle defaults when None).

    ``group=None`` means "not one of the four quantity groups" and returns
    *fallback* unchanged.
    """
    if group is None:
        return fallback
    return (style if style is not None else PlotStyle()).cmap_for(group)


def _style_from_dict(data: dict) -> PlotStyle:
    names = {f.name for f in fields(PlotStyle)}
    kwargs = {k: v for k, v in dict(data).items() if k in names}
    if isinstance(kwargs.get("formats"), list):
        kwargs["formats"] = tuple(kwargs["formats"])
    # Migration: a snapshot predating per-group tick formats has none of the
    # tickfmt_* keys. Give it the tuned profile (same as PUBLICATION_STYLE) so
    # old persisted/injected styles gain the sensible defaults. Reached only from
    # the serialized/GUI path — never the bare-PlotStyle style=None code path.
    _tickfmt_keys = ("tickfmt_mosa_com", "tickfmt_mosa_fwhm", "tickfmt_strain", "tickfmt_raw")
    if not any(k in data for k in _tickfmt_keys):
        kwargs.setdefault("tickfmt_strain", "scientific")
        kwargs.setdefault("tickfmt_raw", "arb")
        for grp in CMAP_GROUPS:
            kwargs.setdefault(f"offset_pos_{grp}", "bottom")
    return PlotStyle(**kwargs)


def style_from_params(params: dict) -> PlotStyle | None:
    """Rebuild the GUI-injected style from the reserved ``plot_style`` params key.

    Returns ``None`` when the key is absent/empty (headless CLI ⇒ legacy look).
    Unknown keys are dropped and missing keys defaulted so an older or newer
    GUI snapshot never crashes a stage.
    """
    raw = params.get("plot_style")
    if not raw:
        return None
    return _style_from_dict(raw)


def style_to_json(style: PlotStyle) -> str:
    """Serialize a style for QSettings persistence."""
    from dataclasses import asdict

    return json.dumps(asdict(style))


def style_from_json(text: str) -> PlotStyle | None:
    """Inverse of :func:`style_to_json`; ``None`` on any parse/shape failure."""
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    try:
        return _style_from_dict(data)
    except (TypeError, ValueError):
        return None


def figure_size(style: PlotStyle, ext_x: float, ext_y: float) -> tuple[float, float] | None:
    """Figure (w, h) in inches from the width preset, preserving physical aspect.

    Returns ``None`` for ``figure_width="auto"`` so the builder keeps its own
    figsize (the legacy path). Height follows the physical aspect plus ~1in of
    headroom for the title/colourbar.
    """
    presets = {"single": 3.5, "double": 7.0}
    w = (
        presets.get(style.figure_width)
        if isinstance(style.figure_width, str)
        else style.figure_width
    )
    if w in (None, "auto"):
        return None
    aspect = (ext_y / ext_x) if ext_x else 1.0
    return (float(w), float(w) * aspect + 1.0)


_MAX_FIXED_SIDE_IN = 30.0  # sanity cap: a typo scale must not request a 47k-pixel render


def fixed_scale(style: "PlotStyle | None") -> float | None:
    """Defensively read ``style.scale_um_per_cm``: a positive finite float, else None.

    Stale persisted styles may carry strings or nonsense — degrade to None
    (today's behaviour), matching the other style-field guards. Never raises.
    """
    if style is None:
        return None
    v = getattr(style, "scale_um_per_cm", None)
    if v is None or v == "":
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if (v > 0 and math.isfinite(v)) else None


def fixed_scale_box(
    style: "PlotStyle | None", ext_x_um: float, ext_y_um: float
) -> tuple[float, float, float] | None:
    """Target axes-box (w_in, h_in, effective_um_per_cm) for fixed-scale mode.

    Returns None when the knob is off or the extents are degenerate (skip
    fitting). Sides are clamped to 30 in preserving aspect — the scale is
    effectively raised and a warning logged, never an exception.
    """
    s = fixed_scale(style)
    if s is None:
        return None
    if not (math.isfinite(ext_x_um) and math.isfinite(ext_y_um)) or ext_x_um <= 0 or ext_y_um <= 0:
        return None
    w, h = ext_x_um / s / 2.54, ext_y_um / s / 2.54
    m = max(w, h)
    if m > _MAX_FIXED_SIDE_IN:
        f = _MAX_FIXED_SIDE_IN / m
        w, h, s = w * f, h * f, s / f
        _log.warning(
            "fixed-scale box clamped to %.0f in per side; effective scale raised to %.4g um/cm",
            _MAX_FIXED_SIDE_IN,
            s,
        )
    return (w, h, s)


def fit_axes_to_box(fig, ax, w_in: float, h_in: float, tol_in: float = 0.02, max_iter: int = 3):
    """Resize *fig* until *ax*'s box is (w_in, h_in) inches, within *tol_in*.

    Draws, measures ``ax.get_window_extent()``, and corrects the figure size
    ADDITIVELY by the miss (decorations are constant in inches, so the first
    correction is nearly exact; the loop is insurance). The target box must
    have the data aspect so aspect="equal" does not fight the fit. Returns
    True on convergence; non-convergence keeps the last size, logs, and
    returns False — never fatal.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if fig.canvas is None or not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    for _ in range(max(1, int(max_iter))):
        fig.canvas.draw()
        bb = ax.get_window_extent(fig.canvas.get_renderer())
        cur_w, cur_h = bb.width / fig.dpi, bb.height / fig.dpi
        dw, dh = w_in - cur_w, h_in - cur_h
        if abs(dw) <= tol_in and abs(dh) <= tol_in:
            return True
        fw, fh = fig.get_size_inches()
        fig.set_size_inches(max(fw + dw, 0.5), max(fh + dh, 0.5), forward=False)
    _log.info(
        "fit_axes_to_box: miss > %.3f in after %d iterations (kept last size)", tol_in, max_iter
    )
    return False


def finalize_fixed_scale(
    fig, ax, style: "PlotStyle | None", ext_x_um: float, ext_y_um: float
) -> None:
    """Fit *ax* to the fixed-scale target box when the knob is on; else no-op."""
    box = fixed_scale_box(style, ext_x_um, ext_y_um)
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])


def symmetric_limits(data: np.ndarray, percentile: float | None = None) -> tuple[float, float]:
    """Colour limits symmetric about zero from *data*'s finite values.

    ``percentile=None`` uses the max absolute value; a number (e.g. 99) uses
    that percentile of ``|data|`` to reject outliers.
    """
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return -1e-4, 1e-4
    m = float(
        np.max(np.abs(valid)) if percentile is None else np.percentile(np.abs(valid), percentile)
    )
    if m == 0:
        m = 1e-12
    return -m, m


def round_limits_outward(vmin: float, vmax: float) -> tuple[float, float]:
    """Round colour limits OUTWARD (vmin down, vmax up) to 'nice' values.

    Each non-zero endpoint moves to the next multiple of half its
    leading-digit unit (step = 0.5 * 10**floor(log10(|v|))): ±0.0778 → ±0.08,
    0.11 → 0.15, 0.0432 → 0.045, 1.7e-4 → 2e-4. Results have at most two
    significant digits (last digit 0 or 5), so evenly spaced colourbar ticks
    land on round numbers. Symmetric input stays exactly symmetric; zero
    endpoints, non-finite values and degenerate ranges (vmin >= vmax) are
    returned unchanged.
    """

    def _out(v: float, up: bool) -> float:
        if v == 0.0 or not math.isfinite(v):
            return v
        step = 0.5 * 10.0 ** math.floor(math.log10(abs(v)))
        n = v / step
        # epsilon guard so already-round values do not inflate by a whole step
        n = math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)
        # strip binary float noise scale-relatively — by construction the
        # product has at most 3 significant digits (n in [2, 20], step = 5·10^k)
        return float(f"{n * step:.6g}")

    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmin >= vmax:
        return (vmin, vmax)
    return (_out(vmin, up=False), _out(vmax, up=True))


def apply_round_clim(
    vmin: float, vmax: float, style: "PlotStyle | None"
) -> tuple[float, float, str | None]:
    """Round (vmin, vmax) outward when ``style.round_clim`` is set.

    Returns ``(vmin, vmax, note)``. The note is a user-facing description of
    what changed (``None`` when rounding is off, style is None, or the limits
    were already round) — stages surface it in the run log / results.
    """
    if style is None or not style.round_clim:
        return vmin, vmax, None
    rlo, rhi = round_limits_outward(vmin, vmax)
    if rlo == vmin and rhi == vmax:
        return vmin, vmax, None
    if math.isclose(-vmin, vmax, rel_tol=1e-9) and math.isclose(-rlo, rhi, rel_tol=1e-9):
        note = f"colour limits rounded ±{vmax:.4g} → ±{rhi:.4g} (round_clim)"
    else:
        note = (
            f"colour limits rounded ({vmin:.4g}, {vmax:.4g}) → ({rlo:.4g}, {rhi:.4g}) (round_clim)"
        )
    return rlo, rhi, note


def physical_extent(
    shape: tuple[int, int],
    pixel_size_x: float,
    pixel_size_y: float,
    roi: list | None = None,
) -> list[float]:
    """imshow ``extent`` (µm) for *shape*, offset by *roi* if given."""
    ny, nx = shape
    x_off = roi[2] * pixel_size_x if roi else 0.0
    y_off = roi[0] * pixel_size_y if roi else 0.0
    return [x_off, x_off + nx * pixel_size_x, y_off, y_off + ny * pixel_size_y]


def get_cmap(name: str):
    """Look up a colormap by name (ParaView's ``"fast"`` is registered at import)."""
    registry = matplotlib.colormaps
    if name in registry:
        return registry[name]
    raise KeyError(f"unknown colormap {name!r}")


def new_figure(figsize: tuple[float, float] = (7.0, 5.0)) -> Figure:
    """A white-background :class:`Figure` (no pyplot, GUI-safe)."""
    fig = Figure(figsize=figsize, facecolor="white")
    return fig


def styled_figure(figsize: tuple[float, float], *, styled: bool) -> Figure:
    """A white-background Figure for the shared figure builders.

    ``styled=True`` (a PlotStyle is in play) uses matplotlib's constrained
    layout, which measures every text element at its final font size and
    reserves space so title, axis labels, colorbar and offset text can never
    overlap — the figure keeps its exact width and the axes shrink instead.
    ``styled=False`` is the legacy path: plain fixed margins, byte-identical
    with the pre-export renderers.
    """
    if styled:
        return Figure(figsize=figsize, facecolor="white", layout="constrained")
    return Figure(figsize=figsize, facecolor="white")


def auto_scale_bar_length_um(ext_x: float) -> float:
    """A 'nice' bar length ~15% of the X extent (1-2-5-10 series)."""
    target = ext_x * 0.15
    if target <= 0:
        return target
    exp = math.floor(math.log10(target))
    frac = target / (10**exp)
    nice = 1.0 if frac < 1.5 else (2.0 if frac < 3.5 else (5.0 if frac < 7.5 else 10.0))
    return nice * (10**exp)


# Points of extra title clearance per unit of font_scale on constrained-layout
# figures.  A colourbar's scientific-notation offset text (the "×10ⁿ" above its
# ticks) sits just above the axes and grows with font_scale; constrained layout
# does not account for that neighbouring artist when budgeting room for the title,
# so the two can collide at large font_scale.  We compensate by expanding the
# title's own pad proportionally — but only on constrained-layout figures.
# Plain figures must keep matplotlib's default (6.0 pt) to avoid legacy-output drift.
_TITLE_PAD_PER_FONT_SCALE = 12.0


def apply_text_scale(ax, style: "PlotStyle") -> None:
    """Scale axis-label/tick fonts by ``style.font_scale`` and the title by the
    independent ``style.title_scale``; apply title/centre options."""
    fs = style.font_scale
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontsize(label.get_fontsize() * fs)
        if style.center_axis_labels:
            label.set_ha("center")
    ax.tick_params(
        axis="x",
        labelsize=(
            ax.xaxis.get_ticklabels()[0].get_fontsize() * fs
            if ax.xaxis.get_ticklabels()
            else 10 * fs  # matplotlib default; only reachable after set_xticks([])
        ),
    )
    ax.tick_params(
        axis="y",
        labelsize=(
            ax.yaxis.get_ticklabels()[0].get_fontsize() * fs
            if ax.yaxis.get_ticklabels()
            else 10 * fs
        ),
    )
    title = ax.title
    if not style.show_title:
        ax.set_title("")
    else:
        # Only grow the title pad on constrained-layout figures — that is where the
        # ×10ⁿ offset-text clearance is needed.  Plain (no layout engine) figures
        # must use pad=None so matplotlib's default (6.0 pt) is preserved; passing
        # an explicit pad would silently drift legacy per-layer PNG output.
        pad = (
            _TITLE_PAD_PER_FONT_SCALE * fs
            if ax.get_figure().get_layout_engine() is not None
            else None
        )
        ax.set_title(
            title.get_text(),
            fontsize=title.get_fontsize() * style.title_scale,
            pad=pad,
        )


def draw_scale_bar(
    ax,
    length_um: float | None = None,
    *,
    style: "PlotStyle",
    fixed_scale_um_per_cm: float | None = None,
) -> None:
    """Draw a µm scale bar (label centred over the bar, optional background box).

    *ax* must use data coordinates in microns. ``length_um=None`` auto-sizes.

    Built on matplotlib's offsetbox machinery: the ``AnchoredOffsetbox`` frame
    is laid out at draw time around the *rendered* label + bar, so the
    background box hugs its content at any ``font_scale`` (exact even under
    constrained layout, whose axes positions are only final at first draw),
    and the ``VPacker`` centres label and bar on each other by construction.

    ``fixed_scale_um_per_cm`` is opt-in per call: when given a positive value,
    the bar height is pinned to ``scale_bar_thickness_pt`` in TRUE printed
    points at that known µm-per-cm scale, instead of a fraction of the axes'
    Y extent. Only callers that have actually fit the axes to a physical page
    size (so the axes' data-to-page scale really is that value) may pass it —
    passing it for an un-fitted axes draws a bar with the wrong thickness.
    Leave it ``None`` (the default) for the legacy auto-scaled geometry.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, TextArea, VPacker
    from matplotlib.patches import Rectangle

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = (x1 - x0), (y1 - y0)
    sl = length_um if length_um is not None else auto_scale_bar_length_um(abs(xr))
    if fixed_scale_um_per_cm is not None and fixed_scale_um_per_cm > 0:
        # Fixed-scale mode: bar height = thickness in TRUE points at the known scale
        # (1 pt = 2.54/72 cm of page = that many cm x um-per-cm of data).
        bh = style.scale_bar_thickness_pt * (2.54 / 72.0) * float(fixed_scale_um_per_cm)
    else:
        # Bar height in data coords: 0.004·thickness_pt·|yr| (≈0.012·|yr| at the
        # default thickness_pt=3.0) — unchanged from the previous hand-rolled
        # geometry; close to, but NOT byte-identical to, the pre-export legacy bar
        # (which used 0.01·|yr| and 50/10/1 length rounding).
        bh = abs(yr) * 0.004 * style.scale_bar_thickness_pt
    # Floor guards the pad division below (and FontProperties) against
    # font_scale/label_scale = 0 from hand-written or stale persisted styles.
    label_size = max(10.0 * style.font_scale * style.scale_bar_label_scale, 0.1)
    # AnchoredOffsetbox rejects non-canonical loc strings with ValueError; keep
    # the old substring tolerance for hand-written styles ("bottom right" -> a
    # sensible corner) instead of crashing the stage.
    loc = style.scale_bar_loc
    if loc not in {
        "upper right",
        "upper left",
        "lower left",
        "lower right",
        "right",
        "center left",
        "center right",
        "lower center",
        "upper center",
        "center",
    }:
        loc = f"{'upper' if 'upper' in loc else 'lower'} {'right' if 'right' in loc else 'left'}"

    # Deliberately NOT mpl_toolkits' AnchoredSizeBar (which assembles the same
    # offsetbox tree): it draws the bar Rectangle with a point-based edge — the
    # doubled-thickness look the pinned no-edge geometry forbids — and we need
    # direct access to the frame patch for colour/alpha/rounding anyway.
    bar = AuxTransformBox(ax.transData)  # width stays true to data µm
    bar.add_artist(
        Rectangle((0, 0), sl, bh, facecolor=style.scale_bar_color, edgecolor="none", linewidth=0)
    )
    label = TextArea(
        f"{sl:g} µm",
        textprops={"color": style.scale_bar_color, "fontsize": label_size, "fontweight": "bold"},
    )
    box = AnchoredOffsetbox(
        loc=loc,
        child=VPacker(children=[label, bar], align="center", pad=0, sep=0.25 * label_size),
        # pad/borderpad are in font-size units of *prop*; pinning prop to the
        # label size makes box_margin_pt mean real points. No box -> no pad,
        # so the Box-margin control cannot inset the bar by a phantom frame.
        prop=FontProperties(size=label_size),
        pad=(style.scale_bar_box_margin_pt / label_size) if style.scale_bar_box else 0.0,
        borderpad=1.5,
        frameon=style.scale_bar_box,
    )
    if style.scale_bar_box:
        box.patch.set(
            facecolor=style.scale_bar_box_color,
            edgecolor="none",
            alpha=style.scale_bar_box_alpha,
        )
        # Rounded corners without extra growth: all padding comes from the
        # offsetbox pad above; rounding_size is in mutation-scale (font) units.
        box.patch.set_boxstyle("round", pad=0, rounding_size=0.4)
    # In-axes decoration: keep the constrained-layout solver from budgeting
    # figure margin for it (the old code clipped the label for the same reason).
    box.set_in_layout(False)
    ax.add_artist(box)
    # AnchoredOffsetbox ignores its own clip settings, so clip the frame patch
    # and every packed artist to the axes instead: at extreme font scales the
    # assembly must truncate at the axes edge — like the old clipped label —
    # not overdraw tick labels or run past the figure border.
    to_clip, stack = [box.patch], [box.get_child()]
    while stack:
        a = stack.pop()
        to_clip.append(a)
        if hasattr(a, "get_children"):
            stack.extend(a.get_children())
    for a in to_clip:
        a.set_clip_on(True)
        a.set_clip_box(ax.bbox)


def colorbar_tick_values(vmin: float, vmax: float, n: int) -> list[float]:
    """``n`` evenly-spaced tick values from vmin..vmax (always includes both ends)."""
    return list(np.linspace(vmin, vmax, max(2, n)))


def _tick_formatter(fmt: str):
    """Formatter for plain/digit formats. ``"auto"``, ``"scientific"`` and ``"arb"``
    return ``None`` here — scientific/arb are handled directly in ``add_colorbar``
    because they need the colour limits / axis, and ``auto`` means matplotlib default.
    """
    if fmt in ("auto", "scientific", "arb"):
        return None
    try:
        d = int(fmt)
        if d >= 0:
            return FuncFormatter(lambda v, _pos: f"{v:.{d}f}")
    except ValueError:
        pass
    return None  # matplotlib default


def build_histogram(
    data: np.ndarray,
    *,
    title: str,
    xlabel: str,
    style: "PlotStyle | None" = None,
) -> "Figure | None":
    """Histogram of finite values in *data* (Figure), or None when there are none.

    *title* and *xlabel* are required keyword arguments (no defaults — callers
    supply stage-specific strings). When *style* is ``None`` the legacy look is
    reproduced exactly. The caller is responsible for calling ``fig.savefig``.
    """
    valid = data[np.isfinite(data)].ravel()
    if valid.size == 0:
        return None
    figsize = (8.0, 5.0)  # legacy default (style is None or figure_width="auto")
    if style is not None:
        presets = {"single": 3.5, "double": 7.0}
        w = (
            presets.get(style.figure_width)
            if isinstance(style.figure_width, str)
            else style.figure_width
        )
        if w not in (None, "auto"):
            figsize = (float(w), float(w) * 5.0 / 8.0)  # keep the histogram's ~8:5 aspect
    fig = styled_figure(figsize, styled=style is not None)
    ax = fig.add_subplot(111)
    ax.hist(valid, bins=200, color="steelblue", alpha=0.85)
    mean_val = float(valid.mean())
    median_val = float(np.median(valid))
    ax.axvline(mean_val, color="red", ls="--", lw=1.5, label=f"mean = {mean_val:.3e}")
    ax.axvline(
        median_val,
        color="orange",
        ls="--",
        lw=1.5,
        label=f"median = {median_val:.3e}",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count")
    ax.set_title(title)
    ax.legend()
    if style is not None:
        apply_text_scale(ax, style)
    return fig


def _apply_scientific(cb, im, style, group) -> None:
    """Render scientific notation on *cb* with a custom, styleable exponent label.

    Computes one common order of magnitude from the colour limits, formats the
    ticks as mantissas, hides matplotlib's built-in (un-styleable, top-only)
    offset text, and draws our own ``×10ⁿ`` label at the group's chosen
    top/bottom position and size. Deterministic and redraw-safe (a static Text
    artist), unlike the built-in offset whose position matplotlib re-derives on
    every draw.
    """
    vmin, vmax = im.norm.vmin, im.norm.vmax
    maxabs = max(abs(vmin), abs(vmax)) if (vmin is not None and vmax is not None) else 0.0
    oom = int(math.floor(math.log10(maxabs))) if maxabs > 0 and math.isfinite(maxabs) else 0

    if oom == 0:
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    else:
        scale = 10.0**oom
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos, s=scale: f"{v / s:.2f}"))

    # Silence matplotlib's built-in offset text; we draw our own below.
    cb.ax.yaxis.get_offset_text().set_visible(False)
    if oom == 0:
        return  # mantissas are the values themselves — no exponent label needed

    size = max(9 * style.font_scale * style.offset_scale_for(group), 0.1)
    exp = r"$\times\mathdefault{10^{%d}}$" % oom
    if style.offset_pos_for(group) == "bottom":
        cb.ax.text(0.5, -0.02, exp, transform=cb.ax.transAxes, ha="center", va="top", fontsize=size)
    else:  # top
        cb.ax.text(
            0.5, 1.02, exp, transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=size
        )


def add_colorbar(fig, im, ax, label: str, style: "PlotStyle", *, group: str | None = None):
    """Add a colourbar honouring thickness, label, tick count and per-group number format.

    *group* (one of :data:`CMAP_GROUPS`, or ``None`` for the neutral default)
    selects the tick format via ``style.tickfmt_for(group)``:
    ``"auto"``/digit as before; ``"scientific"`` renders a custom, styleable
    ``×10ⁿ`` exponent (see :func:`_apply_scientific`); ``"arb"`` drops all
    numeric ticks and marks the label "arbitrary units".
    """
    cb = fig.colorbar(im, ax=ax, fraction=style.colorbar_fraction, pad=0.04)
    text = style.colorbar_label if style.colorbar_label is not None else label
    fmt = style.tickfmt_for(group)

    if fmt == "arb":
        cb.set_ticks([])  # no numeric scale for arbitrary units
        if style.colorbar_label is None and not ("a.u." in text.lower() or "arb" in text.lower()):
            text = f"{text} (arb. units)"
    else:
        if style.colorbar_ticks and style.colorbar_ticks >= 2:
            cb.set_ticks(colorbar_tick_values(im.norm.vmin, im.norm.vmax, style.colorbar_ticks))
        if fmt == "scientific":
            _apply_scientific(cb, im, style, group)
        else:
            f = _tick_formatter(fmt)
            if f is not None:
                cb.ax.yaxis.set_major_formatter(f)

    cb.set_label(text, fontsize=10 * style.font_scale)
    cb.ax.tick_params(labelsize=9 * style.font_scale)
    return cb
