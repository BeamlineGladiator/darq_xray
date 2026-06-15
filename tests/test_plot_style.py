from dataclasses import replace

import numpy as np
from matplotlib.figure import Figure

from dfxm.common.plotting import (
    PUBLICATION_STYLE,
    PlotStyle,
    auto_scale_bar_length_um,
    draw_scale_bar,
    figure_size,
)


def test_default_style_is_conservative():
    s = PlotStyle()
    assert s.font_scale == 1.0
    assert s.formats == ("png",)
    assert s.colorbar_ticks == 0  # 0 => matplotlib default ticks


def test_publication_style_is_bigger():
    assert PUBLICATION_STYLE.font_scale >= 2.0
    assert PUBLICATION_STYLE.scale_bar is True
    assert PUBLICATION_STYLE.scale_bar_box is True
    assert set(PUBLICATION_STYLE.formats) == {"png", "pdf", "svg"}


def test_replace_makes_independent_copy():
    s = replace(PUBLICATION_STYLE, font_scale=1.5)
    assert s.font_scale == 1.5
    assert PUBLICATION_STYLE.font_scale != 1.5


def test_figure_size_auto_returns_none():
    assert figure_size(PlotStyle(figure_width="auto"), 50.0, 30.0) is None


def test_figure_size_single_preserves_aspect():
    w, h = figure_size(PlotStyle(figure_width="single"), 50.0, 25.0)
    assert w == 3.5 and abs(h - (3.5 * 0.5 + 1.0)) < 1e-9


def _ax(ext_x=50.0, ext_y=30.0):
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((10, 10)), extent=[0, ext_x, 0, ext_y], origin="lower")
    return fig, ax


def test_auto_length_rounds_to_nice_value():
    assert auto_scale_bar_length_um(50.0) == 10  # ~15% of 50 = 7.5 -> rounds to 10


def test_auto_length_small_extent_is_nice():
    # X extents in the old dead zone now snap to nice 1-2-5 values, never raw floats
    assert auto_scale_bar_length_um(10.0) == 2  # target 1.5 -> 2
    assert auto_scale_bar_length_um(7.0) == 1  # target 1.05 -> 1
    assert auto_scale_bar_length_um(40.0) == 5  # target 6.0 -> 5


def test_draw_scale_bar_adds_patch_and_text():
    fig, ax = _ax()
    n_before = len(ax.patches)
    draw_scale_bar(ax, length_um=10.0, style=PlotStyle(scale_bar_color="white"))
    assert len(ax.patches) == n_before + 1  # the bar
    assert any("µm" in t.get_text() for t in ax.texts)


def test_draw_scale_bar_box_adds_a_second_patch():
    fig, ax = _ax()
    draw_scale_bar(ax, length_um=10.0, style=PlotStyle(scale_bar_box=True))
    # one patch for the bar, one for the background box
    assert len(ax.patches) == 2
