"""Separable 2-D arctan detrend — shared by strain (ccmth) and mosaicity (χ/μ CoM).

A slowly varying background across the field of view (sample bending, a
curved lattice) is fitted as one arctan along X and one along Y and
subtracted, keeping the map's original median level. Strain detrends its ccmth
map this way before computing strain; mosaicity optionally does the same to
its χ and μ centre-of-mass maps before stacking them. Both always detrend the
**full** map — strain crops its ROI afterwards.

Plotting uses the explicit Figure/Agg API (no pyplot) so this module is safe
to import in the Qt GUI process.
"""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure
from scipy.optimize import curve_fit

from .plotting import PlotStyle, add_colorbar, apply_text_scale, resolve_cmap, styled_figure

# The diagnostic's fixed figure size and the dpi it is saved at. Constants
# rather than literals at the `savefig` calls because the stages' `estimate()`
# functions size the same canvas (`plotting.canvas_pixels`).
DETREND_DIAG_DPI = 120
DETREND_DIAG_FIGSIZE = (20, 6)

# The arctan model has four parameters, so a profile needs at least this many
# finite points before `curve_fit` can run on it at all.
_MIN_FIT_POINTS = 4


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


def can_detrend(map_2d: np.ndarray) -> bool:
    """Whether :func:`detrend_arctan_2d` can fit *map_2d* at all.

    Only a 2-D map, and the fit runs on the median profile along each axis, so
    it needs at least four columns and four rows holding a finite value. Below
    that (an all-NaN map, a sliver of data) ``curve_fit`` refuses the input
    outright. Necessary, not sufficient: a column of mostly-inf values counts
    as finite here yet has an inf median, so a caller that must not fail still
    guards the fit itself.
    """
    if np.ndim(map_2d) != 2:
        return False
    finite = np.isfinite(map_2d)
    return (
        int(finite.any(axis=0).sum()) >= _MIN_FIT_POINTS
        and int(finite.any(axis=1).sum()) >= _MIN_FIT_POINTS
    )


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


def build_detrend_diag(
    original: np.ndarray,
    detrended: np.ndarray,
    surface: np.ndarray,
    *,
    style: PlotStyle | None = None,
    group: str = "strain",
    original_title: str = "Original (after ROI)",
) -> Figure:
    """Build and return a 3-panel detrend-diagnostic Figure.

    *group* is the colormap group (a ``CMAP_GROUPS`` name) every panel is
    coloured with — ``"strain"`` for strain's ccmth, ``"mosa_com"`` for
    mosaicity's χ/μ centre of mass. *original_title* names the first panel;
    strain's default says the ROI has been applied, mosaicity has no ROI.

    When *style* is ``None`` the legacy look is reproduced exactly. The caller
    is responsible for calling ``fig.savefig``.
    """
    fig = styled_figure(DETREND_DIAG_FIGSIZE, styled=style is not None)
    axes = fig.subplots(1, 3)
    for ax, title, d in zip(
        axes,
        [original_title, "Arctan surface", "Detrended"],
        [original, surface, detrended],
    ):
        valid = d[np.isfinite(d)]
        if valid.size == 0:
            continue
        vlo, vhi = np.percentile(valid, [1, 99])
        im = ax.imshow(
            d, origin="lower", cmap=resolve_cmap(style, group), vmin=vlo, vmax=vhi, aspect="auto"
        )
        ax.set_title(title)
        if style is not None:
            add_colorbar(fig, im, ax, title, style, group=group)
            apply_text_scale(ax, style)
        else:
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig
