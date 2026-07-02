"""Regression tests for styled-figure layout: no text may overlap at large font scales."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfxm.common.plotting import PlotStyle, styled_figure
from dfxm.stages.slices import build_slice_figure

# The exact style family that produced the overlapping export in the bug report
_BIG = PlotStyle(
    font_scale=2.2,
    figure_width="single",
    colorbar_ticks=5,
    colorbar_tick_format="scientific",
    scale_bar=True,
    scale_bar_box=True,
    scale_bar_color="white",
)


def _slice_fixture():
    u = np.linspace(-200.0, 200.0, 80)
    v = np.linspace(-120.0, 120.0, 50)
    data = np.outer(np.linspace(-0.0778, 0.0778, 50), np.ones(80))
    prep = {
        "cmap_name": "RdBu_r",
        "vmin": -0.0778,
        "vmax": 0.0778,
        "center_zero": True,
        "title": "χ Misorientation",
        "cbar_label": "Misorientation (°)",
    }
    return prep, {"name": "oblique_full"}, data, u, v


def _drawn(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def test_styled_figure_layout_flag():
    assert styled_figure((4, 3), styled=True).get_layout_engine() is not None
    assert styled_figure((4, 3), styled=False).get_layout_engine() is None


def test_slice_figure_legacy_path_has_no_layout_engine():
    prep, sl, data, u, v = _slice_fixture()
    fig = build_slice_figure(prep, sl, data, u, v, offset_um=None, style=None)
    assert fig.get_layout_engine() is None


def test_slice_figure_texts_do_not_overlap_at_publication_scale():
    prep, sl, data, u, v = _slice_fixture()
    fig = build_slice_figure(prep, sl, data, u, v, offset_um=194.0, style=_BIG)
    renderer = _drawn(fig)
    ax, cax = fig.axes[0], fig.axes[1]

    title_bb = ax.title.get_window_extent(renderer)
    cbar_bb = cax.get_tightbbox(renderer)  # includes ticks, label AND the ×10ⁿ offset text
    image_bb = ax.bbox

    # The three failures visible in the bug report:
    assert not title_bb.overlaps(cbar_bb), "title collides with colorbar"
    assert not title_bb.overlaps(image_bb), "title collides with the map"
    cb_label_bb = cax.yaxis.label.get_window_extent(renderer)
    for tick in cax.yaxis.get_ticklabels():
        assert not cb_label_bb.overlaps(tick.get_window_extent(renderer)), (
            "colorbar label collides with its tick labels"
        )


def test_slice_figure_keeps_exact_single_column_width():
    prep, sl, data, u, v = _slice_fixture()
    fig = build_slice_figure(prep, sl, data, u, v, offset_um=194.0, style=_BIG)
    assert fig.get_size_inches()[0] == 3.5
