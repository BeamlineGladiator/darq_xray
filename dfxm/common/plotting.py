"""Shared plotting helpers, safe to import from both the GUI and the worker.

These deliberately avoid ``pyplot`` and never call ``matplotlib.use(...)``:
touching the global backend from a module that the GUI imports would clobber
the Qt backend that the embedded canvases need. Build figures with the
explicit :class:`~matplotlib.figure.Figure` API instead and save via
``fig.savefig`` (Agg is used implicitly, no global state changed).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
from matplotlib.figure import Figure


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
    font_scale: float = 1.0  # multiplies axis labels, ticks, title
    show_title: bool = True
    center_axis_labels: bool = True
    # colourbar
    colorbar: bool = True
    colorbar_label: str | None = None  # None -> the figure's own label
    colorbar_fraction: float = 0.046  # matplotlib colorbar `fraction` (thickness)
    colorbar_ticks: int = 0  # 0 -> matplotlib default; >=2 -> N evenly spaced incl min/mid/max
    colorbar_tick_format: str = "auto"  # "auto" | "scientific" | a digit count like "2"
    # figure
    figure_width: str | float = "auto"  # "single" | "double" | "auto" | width in inches
    # output
    formats: tuple[str, ...] = ("png",)
    dpi: int = 300


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
    """Look up a colormap by name.

    Supports the ParaView ``"fast"`` map by falling back to ``coolwarm`` when
    it is not registered with matplotlib.
    """
    registry = matplotlib.colormaps
    if name in registry:
        return registry[name]
    if name == "fast" and "coolwarm" in registry:
        return registry["coolwarm"]
    raise KeyError(f"unknown colormap {name!r}")


def new_figure(figsize: tuple[float, float] = (7.0, 5.0)) -> Figure:
    """A white-background :class:`Figure` (no pyplot, GUI-safe)."""
    fig = Figure(figsize=figsize, facecolor="white")
    return fig


def add_scale_bar(
    ax,
    length_um: float,
    *,
    loc: str = "lower right",
    color: str = "white",
    label: str | None = None,
) -> None:
    """Draw a horizontal µm scale bar in axes-fraction coordinates.

    Assumes *ax* uses data coordinates in microns (see :func:`physical_extent`).
    """
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = x1 - x0
    span_y = y1 - y0
    pad_x = 0.05 * span_x
    pad_y = 0.06 * span_y
    if "right" in loc:
        x_end = x1 - pad_x
        x_start = x_end - length_um
    else:
        x_start = x0 + pad_x
        x_end = x_start + length_um
    y = y0 + pad_y if "lower" in loc else y1 - pad_y
    ax.plot([x_start, x_end], [y, y], color=color, lw=3, solid_capstyle="butt")
    ax.text(
        (x_start + x_end) / 2,
        y + 0.02 * span_y,
        label if label is not None else f"{length_um:g} µm",
        color=color,
        ha="center",
        va="bottom",
        fontsize=10,
    )
