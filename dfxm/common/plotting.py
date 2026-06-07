"""Shared plotting helpers, safe to import from both the GUI and the worker.

These deliberately avoid ``pyplot`` and never call ``matplotlib.use(...)``:
touching the global backend from a module that the GUI imports would clobber
the Qt backend that the embedded canvases need. Build figures with the
explicit :class:`~matplotlib.figure.Figure` API instead and save via
``fig.savefig`` (Agg is used implicitly, no global state changed).
"""

from __future__ import annotations

import matplotlib
import numpy as np
from matplotlib.figure import Figure


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
