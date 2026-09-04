"""Pin layer_figure's single-figure output across the draw_map_layer extraction."""

import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox

from darq_xray.common.plotting import PlotStyle
from darq_xray.common.render import layer_figure

LAYER = np.linspace(0.0, 1.0, 24 * 30).reshape(24, 30)


def _bar_boxes(ax):
    return [a for a in ax.artists if isinstance(a, AnchoredOffsetbox)]


def test_layer_figure_unstyled_pinned():
    fig, ax, im = layer_figure(LAYER, 0.0, 1.0, "magma", 3.0, 2.4, "T", "C")
    assert tuple(fig.get_size_inches()) == (12.0, 10.0)
    assert fig.get_layout_engine() is None  # legacy: plain figure
    assert list(im.get_extent()) == [0.0, 3.0, 0.0, 2.4]
    assert (im.norm.vmin, im.norm.vmax) == (0.0, 1.0)
    assert ax.get_xlabel() == "X (µm)" and ax.get_ylabel() == "Y (µm)"
    assert ax.get_title() == "T"
    assert len(fig.axes) == 2  # main + stolen colorbar
    assert fig.axes[1].get_ylabel() == "C"
    assert len(_bar_boxes(ax)) == 1  # scale bar drawn


def test_layer_figure_styled_flags_and_fixed_scale():
    style = PlotStyle(scale_um_per_cm=10.0, colorbar=False, scale_bar=False, show_title=False)
    fig, ax, im = layer_figure(LAYER, 0.0, 1.0, "magma", 30.0, 24.0, "T", "C", style=style)
    assert len(fig.axes) == 1  # colorbar honoured off
    assert _bar_boxes(ax) == []  # scale bar honoured off
    assert ax.get_title() == ""  # show_title off via apply_text_scale
    from darq_xray.common.plotting import measured_box_in

    w, h = measured_box_in(fig, ax)
    assert abs(w - 30.0 / 10.0 / 2.54) < 0.02 and abs(h - 24.0 / 10.0 / 2.54) < 0.02


def test_layer_figure_styled_group_cmap_resolution():
    style = PlotStyle(cmap_strain="coolwarm")
    fig, ax, im = layer_figure(
        LAYER, -1.0, 1.0, "coolwarm", 3.0, 2.4, "T", "C", style=style, group="strain"
    )
    assert im.get_cmap().name == "coolwarm"
