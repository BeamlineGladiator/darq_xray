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


def _box_patch_pad(margin_pt):
    from matplotlib.patches import FancyBboxPatch

    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((10, 20)), extent=[0, 20, 0, 10], origin="lower")
    draw_scale_bar(
        ax,
        5.0,
        style=PlotStyle(scale_bar_box=True, scale_bar_box_margin_pt=margin_pt),
    )
    box = next(p for p in ax.patches if isinstance(p, FancyBboxPatch))
    return box.get_boxstyle().pad


def test_box_margin_control_affects_box_padding():
    # The 'Box margin' control must actually change the background-box padding
    # (it was previously hardcoded and inert).
    assert _box_patch_pad(12.0) > _box_patch_pad(2.0)


def test_box_margin_default_preserves_padding():
    # The default margin (4.0) must reproduce the original 0.015*|yr| padding so
    # existing default-style boxes are unchanged.
    assert np.isclose(_box_patch_pad(4.0), 0.015 * 10.0)


def test_scale_bar_auto_length_is_1_2_5_10_series():
    # Pin the styled renderer's actual auto-length behaviour (1-2-5-10), so it
    # can't drift silently again.
    assert auto_scale_bar_length_um(200.0) == 20.0  # 0.15*200=30 -> 20
    assert auto_scale_bar_length_um(500.0) == 100.0  # 0.15*500=75 -> 100


def test_scale_bar_thickness_pins_styled_geometry():
    # Pin the styled bar thickness (0.004*thickness_pt*|yr|, =0.012*|yr| at the
    # default 3.0 pt) so the kept-as-new geometry can't drift unnoticed.
    from matplotlib.patches import Rectangle

    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((10, 20)), extent=[0, 20, 0, 10], origin="lower")
    draw_scale_bar(ax, 5.0, style=PlotStyle())
    bar = next(p for p in ax.patches if isinstance(p, Rectangle))
    assert np.isclose(bar.get_height(), 0.012 * 10.0)


def test_add_colorbar_default_tick_fontsize_is_nine():
    # Pin add_colorbar's default tick label size (9 pt) — a deliberate, tested
    # value rather than the matplotlib default.
    fig = Figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(np.zeros((4, 4)))
    cb = add_colorbar(fig, im, ax, "label", PlotStyle())
    sizes = {round(t.get_fontsize(), 3) for t in cb.ax.get_yticklabels()}
    assert sizes == {9.0}, sizes


def test_cmap_groups_defaults_and_lookup():
    import pytest

    from dfxm.common.plotting import CMAP_CHOICES, CMAP_GROUPS, resolve_cmap

    s = PlotStyle()
    assert s.cmap_for("mosa_com") == "fast"
    assert s.cmap_for("mosa_fwhm") == "magma"
    assert s.cmap_for("strain") == "RdBu_r"
    assert s.cmap_for("raw") == "gray"
    assert CMAP_GROUPS == ("mosa_com", "mosa_fwhm", "strain", "raw")
    for g in CMAP_GROUPS:
        assert s.cmap_for(g) in CMAP_CHOICES
    with pytest.raises(KeyError):
        s.cmap_for("nope")
    # resolve_cmap: None style -> defaults; None group -> fallback
    assert resolve_cmap(None, "raw") == "gray"
    assert resolve_cmap(replace(s, cmap_raw="viridis"), "raw") == "viridis"
    assert resolve_cmap(None, None, fallback="bone") == "bone"


def test_style_from_params_roundtrip_and_tolerance():
    from dataclasses import asdict

    from dfxm.common.plotting import style_from_params

    src = replace(PUBLICATION_STYLE, cmap_strain="seismic", font_scale=3.0)
    p = {"plot_style": asdict(src)}
    got = style_from_params(p)
    assert got == src
    assert style_from_params({}) is None
    # unknown keys dropped, missing keys defaulted, formats list -> tuple
    got = style_from_params({"plot_style": {"font_scale": 2.0, "formats": ["png"], "bogus": 1}})
    assert got.font_scale == 2.0 and got.formats == ("png",) and got.cmap_mosa_com == "fast"


def test_style_json_roundtrip_and_bad_blob():
    from dfxm.common.plotting import style_from_json, style_to_json

    src = replace(PUBLICATION_STYLE, cmap_raw="turbo")
    assert style_from_json(style_to_json(src)) == src
    assert style_from_json("{not json") is None
    assert style_from_json("[1,2]") is None


def test_title_scale_is_independent_of_font_scale():
    fig, ax = _ax()
    ax.set_xlabel("X (µm)")
    ax.set_title("χ Misorientation")
    label_base = ax.xaxis.label.get_fontsize()
    title_base = ax.title.get_fontsize()
    apply_text_scale(ax, PlotStyle(font_scale=3.0, title_scale=0.5))
    assert ax.xaxis.label.get_fontsize() == label_base * 3.0
    assert ax.title.get_fontsize() == title_base * 0.5  # font_scale must NOT touch the title


def test_title_scale_default_leaves_title_at_base_size():
    fig, ax = _ax()
    ax.set_title("t")
    title_base = ax.title.get_fontsize()
    apply_text_scale(ax, PlotStyle(font_scale=2.2))  # title_scale defaults to 1.0
    assert ax.title.get_fontsize() == title_base


def test_title_scale_survives_json_roundtrip():
    from dfxm.common.plotting import style_from_json, style_to_json

    s = PlotStyle(title_scale=0.4)
    assert style_from_json(style_to_json(s)).title_scale == 0.4
    # Old persisted blobs (no title_scale key) default to 1.0
    assert style_from_json(style_to_json(PlotStyle())).title_scale == 1.0
