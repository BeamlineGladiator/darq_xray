"""ParaView 'Fast' colormap: registration + fidelity to the official control points."""

from __future__ import annotations

import matplotlib
import numpy as np

import dfxm.common.plotting  # noqa: F401 — importing registers "fast"
from dfxm.common.cmaps import _FAST_POINTS, fast_colormap, register


def test_fast_is_registered_with_matplotlib():
    assert "fast" in matplotlib.colormaps
    # and resolvable through the shared lookup helper
    from dfxm.common.plotting import get_cmap

    assert get_cmap("fast").name == "fast"


def test_register_is_idempotent():
    register()
    register()  # second call must not raise "already registered"
    assert "fast" in matplotlib.colormaps


def test_fast_endpoints_match_paraview_control_points():
    cmap = fast_colormap()
    # x=0 -> first control point, x=1 -> last (Lab round-trip ~exact at nodes)
    np.testing.assert_allclose(cmap(0.0)[:3], _FAST_POINTS[0, 1:], atol=2e-3)
    np.testing.assert_allclose(cmap(1.0)[:3], _FAST_POINTS[-1, 1:], atol=2e-3)
    # the 0.5 node is a control point too (1e-2: 256-entry LUT quantization,
    # 0.5 falls between two adjacent LUT entries)
    np.testing.assert_allclose(cmap(0.5)[:3], [0.89950, 0.94465, 0.76866], atol=1e-2)


def test_fast_is_not_coolwarm():
    """The old silent coolwarm fallback must be gone."""
    fast = fast_colormap()
    coolwarm = matplotlib.colormaps["coolwarm"]
    # coolwarm(0) is a blue ~(0.23, 0.30, 0.75); fast(0) is a much darker navy
    assert not np.allclose(fast(0.0)[:3], coolwarm(0.0)[:3], atol=0.05)
