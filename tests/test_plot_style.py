from dataclasses import replace

import numpy as np
from matplotlib.figure import Figure

from dfxm.common.plotting import (
    PUBLICATION_STYLE,
    PlotStyle,
    _tick_formatter,
    add_colorbar,
    apply_text_scale,
    auto_scale_bar_length_um,
    build_histogram,
    colorbar_tick_values,
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


def test_draw_scale_bar_box_geometry_is_sane():
    """Box must be a snug corner element, not half the figure (the deferred bug)."""
    from matplotlib.patches import FancyBboxPatch, Rectangle

    ext_y = 30.0
    fig, ax = _ax(ext_x=50.0, ext_y=ext_y)
    style = PlotStyle(
        scale_bar_box=True,
        font_scale=2.2,
        scale_bar_label_scale=1.1,
        scale_bar_thickness_pt=4.0,
    )
    draw_scale_bar(ax, length_um=10.0, style=style)

    boxes = [p for p in ax.patches if isinstance(p, FancyBboxPatch)]
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(boxes) == 1, "expected exactly one background FancyBboxPatch"
    assert len(rects) == 1, "expected exactly one bar Rectangle"

    box = boxes[0]
    rect = rects[0]

    # Box height must be a snug fraction of the y-extent (bug produced ~14.9, i.e. ~50%).
    assert box.get_height() > 0, "box height must be positive"
    assert box.get_height() < 0.25 * ext_y, (
        f"box height {box.get_height():.2f} is too large (>25% of {ext_y} µm) — "
        "geometry bug still present"
    )

    # Box must be wide enough to cover the scale bar.
    assert box.get_width() >= 10.0, "box width should cover the bar length"

    # Bar rectangle must have no doubled point-based edge.
    # edgecolor is returned as an RGBA tuple; alpha=0 or "none" both mean invisible.
    ec = rect.get_edgecolor()  # (R, G, B, A)
    assert ec[3] == 0 or rect.get_linewidth() == 0, (
        "bar Rectangle should have no visible edge (edgecolor='none' or linewidth=0)"
    )


def test_apply_text_scale_grows_label_fonts():
    fig, ax = _ax()
    ax.set_xlabel("X (µm)")
    base = ax.xaxis.label.get_fontsize()
    apply_text_scale(ax, PlotStyle(font_scale=2.0))
    assert ax.xaxis.label.get_fontsize() == base * 2.0


def test_apply_text_scale_hides_title_when_asked():
    fig, ax = _ax()
    ax.set_title("keep me?")
    apply_text_scale(ax, PlotStyle(show_title=False))
    assert ax.get_title() == ""


def test_apply_text_scale_noop_at_font_scale_1():
    fig, ax = _ax()
    ax.set_xlabel("X (µm)")
    ax.set_title("keep me")
    xb = ax.xaxis.label.get_fontsize()
    tb = ax.title.get_fontsize()
    apply_text_scale(ax, PlotStyle())  # default font_scale=1.0
    assert ax.xaxis.label.get_fontsize() == xb
    assert ax.title.get_fontsize() == tb
    assert ax.get_title() == "keep me"  # title kept when show_title=True


def test_colorbar_tick_values_includes_extremes_and_mid():
    vals = colorbar_tick_values(-2e-3, 2e-3, n=5)
    assert len(vals) == 5
    assert vals[0] == -2e-3 and vals[-1] == 2e-3
    assert abs(vals[2]) < 1e-12  # midpoint of a symmetric range is 0


def test_add_colorbar_sets_label_and_tick_count():
    fig, ax = _ax()
    im = ax.imshow(
        np.linspace(-2e-3, 2e-3, 100).reshape(10, 10),
        extent=[0, 50, 0, 30],
        origin="lower",
    )
    cb = add_colorbar(
        fig, im, ax, "Strain (ε)", PlotStyle(colorbar_ticks=5, colorbar_tick_format="scientific")
    )
    assert cb.ax.get_ylabel() == "Strain (ε)"
    assert len(cb.get_ticks()) == 5


def test_tick_formatter_digit_count_and_negative_guard():
    f = _tick_formatter("2")
    assert f is not None
    assert f(3.14159, 0) == "3.14"  # fixed 2-decimal formatting
    assert _tick_formatter("-1") is None  # negative digit count -> matplotlib default
    assert _tick_formatter("auto") is None  # auto -> matplotlib default
    assert _tick_formatter("nonsense") is None  # unparseable -> matplotlib default


def test_build_histogram_respects_figure_width():
    data = np.random.default_rng(0).normal(size=(20, 20))
    assert round(build_histogram(data, title="t", xlabel="x").get_size_inches()[0]) == 8
    fig = build_histogram(data, title="t", xlabel="x", style=PlotStyle(figure_width="single"))
    assert abs(fig.get_size_inches()[0] - 3.5) < 1e-6
