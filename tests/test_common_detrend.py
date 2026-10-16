"""Tests for darq_xray.common.detrend — the separable arctan detrend shared by
strain (ccmth) and mosaicity (χ/μ centre of mass)."""

from __future__ import annotations

import numpy as np

from darq_xray.common import detrend as D
from darq_xray.common.plotting import PlotStyle, resolve_cmap


def _trended_map(ny=40, nx=60, seed=0):
    """A flat map plus a separable arctan ramp along X and along Y."""
    rng = np.random.default_rng(seed)
    xs, ys = np.arange(nx, dtype=float), np.arange(ny, dtype=float)
    trend = (
        0.4 * np.arctan(0.15 * (xs - nx / 2))[None, :]
        + 0.2 * np.arctan(0.2 * (ys - ny / 2))[:, None]
    )
    return 3.0 + trend + 0.001 * rng.standard_normal((ny, nx))


def test_detrend_removes_a_separable_arctan_trend():
    m = _trended_map()
    detrended, surface = D.detrend_arctan_2d(m.copy())
    assert np.nanstd(detrended) < 0.05 * np.nanstd(m)
    assert surface.shape == m.shape


def test_detrend_preserves_the_original_median_level():
    m = _trended_map()
    detrended, _ = D.detrend_arctan_2d(m.copy())
    assert np.isclose(np.nanmedian(detrended), np.nanmedian(m))


def test_strain_uses_the_shared_detrend():
    """One implementation: strain must not keep a private copy that can drift."""
    from darq_xray.stages import strain

    assert strain.detrend_arctan_2d is D.detrend_arctan_2d
    assert strain.build_detrend_diag is D.build_detrend_diag


def test_can_detrend_needs_four_finite_rows_and_columns():
    m = _trended_map(ny=10, nx=10)
    assert D.can_detrend(m)
    assert not D.can_detrend(np.full((10, 10), np.nan))
    sparse = np.full((10, 10), np.nan)
    sparse[:, :3] = 1.0  # every row finite, only three columns
    assert not D.can_detrend(sparse)
    sparse[:, :4] = 1.0
    assert D.can_detrend(sparse)


def test_can_detrend_refuses_anything_but_a_2d_map():
    assert not D.can_detrend(np.ones((1, 10, 10)))
    assert not D.can_detrend(np.ones(10))


def test_diag_colours_with_the_requested_group():
    m = _trended_map()
    detrended, surface = D.detrend_arctan_2d(m.copy())
    style = PlotStyle()
    fig = D.build_detrend_diag(m, detrended, surface, style=style, group="mosa_com")
    cmaps = {im.get_cmap().name for ax in fig.axes for im in ax.get_images()}
    assert cmaps == {resolve_cmap(style, "mosa_com")}
    assert resolve_cmap(style, "mosa_com") != resolve_cmap(style, "strain")


def test_diag_defaults_to_the_strain_group():
    m = _trended_map()
    detrended, surface = D.detrend_arctan_2d(m.copy())
    style = PlotStyle()
    fig = D.build_detrend_diag(m, detrended, surface, style=style)
    cmaps = {im.get_cmap().name for ax in fig.axes for im in ax.get_images()}
    assert cmaps == {resolve_cmap(style, "strain")}
