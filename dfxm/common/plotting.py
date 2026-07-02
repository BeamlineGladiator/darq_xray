"""Shared plotting helpers, safe to import from both the GUI and the worker.

These deliberately avoid ``pyplot`` and never call ``matplotlib.use(...)``:
touching the global backend from a module that the GUI imports would clobber
the Qt backend that the embedded canvases need. Build figures with the
explicit :class:`~matplotlib.figure.Figure` API instead and save via
``fig.savefig`` (Agg is used implicitly, no global state changed).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields

import matplotlib
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter, ScalarFormatter

from .cmaps import register as _register_fast_cmap

_register_fast_cmap()

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
    colorbar_tick_format: str = "auto"  # "auto" | "scientific" | a digit count like "2"
    round_clim: bool = False  # round auto colour limits outward to nice values
    # figure
    figure_width: str | float = "auto"  # "single" | "double" | "auto" | width in inches
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


PUBLICATION_STYLE = PlotStyle(
    scale_bar=True,
    scale_bar_thickness_pt=4.0,
    scale_bar_label_scale=1.1,
    scale_bar_color="white",
    scale_bar_box=True,
    font_scale=2.2,
    colorbar_fraction=0.07,
    colorbar_ticks=5,
    colorbar_tick_format="scientific",
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
        # A colourbar's scientific-notation offset text (the "×10ⁿ" above its
        # ticks) sits just above the axes, at a height that grows with
        # ``font_scale``. matplotlib's constrained layout does not account
        # for that neighbouring artist when it reserves room for the title,
        # so at large font_scale the two can collide. Grow the title's own
        # pad (the gap between it and the axes) proportionally to font_scale
        # so there is always clearance, regardless of colourbar settings.
        ax.set_title(
            title.get_text(),
            fontsize=title.get_fontsize() * style.title_scale,
            pad=12.0 * fs,
        )


def draw_scale_bar(ax, length_um: float | None = None, *, style: "PlotStyle") -> None:
    """Draw a µm scale bar (and optional background box) per *style*.

    *ax* must use data coordinates in microns. ``length_um=None`` auto-sizes.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = (x1 - x0), (y1 - y0)
    sl = length_um if length_um is not None else auto_scale_bar_length_um(abs(xr))
    # Bar height in data coords: 0.004·thickness_pt·|yr| (≈0.012·|yr| at the
    # default thickness_pt=3.0). This is the styled renderer's own geometry; it
    # is close to, but NOT byte-identical to, the pre-export legacy bar (which
    # used 0.01·|yr| and 50/10/1 length rounding).
    bh = abs(yr) * 0.004 * style.scale_bar_thickness_pt
    pad_x, pad_y = 0.05 * abs(xr), 0.05 * abs(yr)
    bx = (x1 - pad_x - sl) if "right" in style.scale_bar_loc else (x0 + pad_x)
    by = (y1 - pad_y - bh) if "upper" in style.scale_bar_loc else (y0 + pad_y)
    label = f"{sl:g} µm"
    label_size = 10.0 * style.font_scale * style.scale_bar_label_scale

    if style.scale_bar_box:
        # Box geometry entirely in data/µm coordinates so it snugly encloses
        # the bar Rectangle (height bh) plus the label (va="bottom" at by+bh*2.5).
        # A small data-coord allowance for the label text (~0.06·|yr|) is used
        # instead of the old label_size*0.02*|yr| mix that ballooned the box.
        label_allowance = 0.06 * abs(yr)
        box_h = bh * 2.5 + label_allowance
        # Padding in data units, scaled by the configurable box margin. The
        # default margin (4.0) reproduces the original 0.015·|yr| padding, so the
        # 'Box margin' control now visibly grows/shrinks the box (it was inert).
        pad_data = 0.015 * abs(yr) * (style.scale_bar_box_margin_pt / 4.0)
        box = FancyBboxPatch(
            (bx, by),
            sl,
            box_h,
            boxstyle=f"round,pad={pad_data}",
            transform=ax.transData,
            facecolor=style.scale_bar_box_color,
            edgecolor="none",
            alpha=style.scale_bar_box_alpha,
            zorder=4,
        )
        ax.add_patch(box)

    # Bar thickness is expressed once in data coords (bh), with no doubled
    # point-based edge.  edgecolor="none" + linewidth=0 ensures the visible
    # thickness is exactly bh.
    ax.add_patch(
        Rectangle(
            (bx, by),
            sl,
            bh,
            facecolor=style.scale_bar_color,
            edgecolor="none",
            linewidth=0,
            zorder=5,
        )
    )
    ax.text(
        bx + sl / 2.0,
        by + bh * 2.5,
        label,
        color=style.scale_bar_color,
        fontsize=label_size,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=6,
        # Text defaults to clip_on=False (unlike patches, which clip to the
        # axes automatically). At large font_scale the label can otherwise
        # spill past the axes edge, which (a) looks wrong and (b) confuses
        # matplotlib's constrained-layout solver into reserving unbounded
        # margin for an artist it thinks is unclippable, collapsing the axes
        # to zero size. Clip it like every other in-axes decoration.
        clip_on=True,
    )


def colorbar_tick_values(vmin: float, vmax: float, n: int) -> list[float]:
    """``n`` evenly-spaced tick values from vmin..vmax (always includes both ends)."""
    return list(np.linspace(vmin, vmax, max(2, n)))


def _tick_formatter(fmt: str):
    if fmt == "scientific":
        f = ScalarFormatter(useMathText=True)
        f.set_powerlimits((0, 0))
        return f
    if fmt != "auto":
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


def add_colorbar(fig, im, ax, label: str, style: "PlotStyle"):
    """Add a colourbar honouring thickness, label, tick count and number format."""
    cb = fig.colorbar(im, ax=ax, fraction=style.colorbar_fraction, pad=0.04)
    text = style.colorbar_label if style.colorbar_label is not None else label
    cb.set_label(text, fontsize=10 * style.font_scale)
    if style.colorbar_ticks and style.colorbar_ticks >= 2:
        cb.set_ticks(colorbar_tick_values(im.norm.vmin, im.norm.vmax, style.colorbar_ticks))
    fmt = _tick_formatter(style.colorbar_tick_format)
    if fmt is not None:
        cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.tick_params(labelsize=9 * style.font_scale)
    return cb
