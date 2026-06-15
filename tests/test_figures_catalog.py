import numpy as np

from dfxm.common import render
from dfxm.common.plotting import PlotStyle


def _layer():
    return np.random.default_rng(0).normal(size=(20, 40))


def test_layer_figure_legacy_has_scale_bar_and_equal_aspect():
    fig, ax, im = render.layer_figure(_layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb")
    assert ax.get_aspect() == 1.0  # physical equal aspect
    assert len(ax.patches) >= 1  # legacy scale bar present


def test_layer_figure_style_off_drops_scale_bar():
    fig, ax, im = render.layer_figure(
        _layer(),
        -1,
        1,
        "viridis",
        40.0,
        20.0,
        "t",
        "cb",
        style=PlotStyle(scale_bar=False),
    )
    assert len(ax.patches) == 0
