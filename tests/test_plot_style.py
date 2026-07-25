import json
from dataclasses import replace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnchoredOffsetbox

from dfxm.common.plotting import (
    PUBLICATION_STYLE,
    PlotStyle,
    _tick_formatter,
    add_colorbar,
    apply_axes_mode,
    apply_text_scale,
    auto_scale_bar_length_um,
    build_histogram,
    colorbar_tick_values,
    draw_scale_bar,
    figure_size,
    fixed_scale,
    fixed_scale_box,
    style_from_json,
    style_to_json,
    styled_figure,
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


def _drawn_renderer(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def _scale_bar_artist(ax):
    return next(a for a in ax.artists if isinstance(a, AnchoredOffsetbox))


def _offsetbox_children(artist):
    """Flatten every artist nested under an offsetbox tree (Texts, Rectangles, ...)."""
    out, stack = [], [artist]
    while stack:
        a = stack.pop()
        out.append(a)
        if hasattr(a, "get_children"):
            stack.extend(a.get_children())
    return out


def test_draw_scale_bar_adds_anchored_box_with_label_and_bar():
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    fig, ax = _ax()
    draw_scale_bar(ax, length_um=10.0, style=PlotStyle(scale_bar_color="white"))
    abox = _scale_bar_artist(ax)  # raises StopIteration if missing
    kids = _offsetbox_children(abox)
    assert any(isinstance(t, Text) and "µm" in t.get_text() for t in kids)
    bar = next(p for p in kids if isinstance(p, Rectangle))
    ec = bar.get_edgecolor()  # (R, G, B, A) — no doubled point-based edge
    assert ec[3] == 0 or bar.get_linewidth() == 0
    assert len(ax.patches) == 0  # nothing leaks into ax.patches any more


def test_draw_scale_bar_box_toggles_frame():
    fig, ax = _ax()
    draw_scale_bar(ax, length_um=10.0, style=PlotStyle(scale_bar_box=True))
    assert _scale_bar_artist(ax).patch.get_visible()

    fig2, ax2 = _ax()
    draw_scale_bar(ax2, length_um=10.0, style=PlotStyle(scale_bar_box=False))
    assert not _scale_bar_artist(ax2).patch.get_visible()


def test_scale_bar_box_hugs_label_at_large_font_scale():
    from matplotlib.text import Text

    fig, ax = _ax(ext_x=50.0, ext_y=30.0)
    style = PlotStyle(
        scale_bar_box=True,
        font_scale=2.2,
        scale_bar_label_scale=1.1,
        scale_bar_thickness_pt=4.0,
    )
    draw_scale_bar(ax, length_um=10.0, style=style)
    renderer = _drawn_renderer(fig)
    abox = _scale_bar_artist(ax)
    frame_bb = abox.patch.get_window_extent(renderer)
    label = next(
        t for t in _offsetbox_children(abox) if isinstance(t, Text) and "µm" in t.get_text()
    )
    text_bb = label.get_window_extent(renderer)
    # The reported bug: at large font scale the label spilled out of the box.
    assert frame_bb.contains(text_bb.x0, text_bb.y0), "label bottom-left outside box"
    assert frame_bb.contains(text_bb.x1, text_bb.y1), "label top-right outside box"
    # Snug corner element, not half the figure.
    ax_bb = ax.get_window_extent(renderer)
    assert frame_bb.height < 0.35 * ax_bb.height


def test_scale_bar_label_and_bar_are_centred():
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    fig, ax = _ax()
    # Short bar + big font -> label wider than the bar; the two must share a centre.
    draw_scale_bar(ax, length_um=2.0, style=PlotStyle(scale_bar_box=True, font_scale=2.2))
    renderer = _drawn_renderer(fig)
    kids = _offsetbox_children(_scale_bar_artist(ax))
    label = next(t for t in kids if isinstance(t, Text) and "µm" in t.get_text())
    bar = next(p for p in kids if isinstance(p, Rectangle))
    text_bb = label.get_window_extent(renderer)
    bar_bb = bar.get_window_extent(renderer)
    text_cx = (text_bb.x0 + text_bb.x1) / 2
    bar_cx = (bar_bb.x0 + bar_bb.x1) / 2
    assert abs(text_cx - bar_cx) < 2.0  # px


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
        fig,
        im,
        ax,
        "Strain (ε)",
        PlotStyle(colorbar_ticks=5, tickfmt_strain="scientific"),
        group="strain",
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


def _box_frame_width(margin_pt):
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((10, 20)), extent=[0, 20, 0, 10], origin="lower")
    draw_scale_bar(
        ax,
        5.0,
        style=PlotStyle(scale_bar_box=True, scale_bar_box_margin_pt=margin_pt),
    )
    renderer = _drawn_renderer(fig)
    return _scale_bar_artist(ax).patch.get_window_extent(renderer).width


def test_box_margin_control_affects_box_padding():
    # The 'Box margin' control must actually change the background-box padding
    # (it was previously hardcoded and inert).
    assert _box_frame_width(12.0) > _box_frame_width(2.0)


def test_box_margin_is_real_points():
    # Box margin semantics: real points. The AnchoredOffsetbox pad is expressed
    # in units of the label font size (10 pt at default style), so margin_pt=4.0
    # must land as pad=0.4 font units — pinned so the point semantics can't drift.
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((10, 20)), extent=[0, 20, 0, 10], origin="lower")
    draw_scale_bar(ax, 5.0, style=PlotStyle(scale_bar_box=True, scale_bar_box_margin_pt=4.0))
    abox = _scale_bar_artist(ax)
    assert np.isclose(abox.pad, 0.4)
    assert np.isclose(abox.borderpad, 1.5)  # default 15 pt inset / 10 pt label = 1.5 font units


def test_scale_bar_edge_inset_is_real_points():
    # Edge-inset semantics: real points from the axes corner. borderpad is in
    # label-font units, so inset_pt must be divided by the label size — the
    # inset must NOT grow with font scale (that pushed the bar into the data).
    def borderpad(**style_kw):
        fig, ax = _ax()
        draw_scale_bar(ax, 5.0, style=PlotStyle(**style_kw))
        return _scale_bar_artist(ax).borderpad

    assert np.isclose(borderpad(scale_bar_inset_pt=5.0), 0.5)  # 5 pt / 10 pt label
    assert np.isclose(borderpad(scale_bar_inset_pt=15.0, font_scale=2.0), 0.75)  # 15 / 20
    assert borderpad(scale_bar_inset_pt=0.0) == 0.0  # flush with the corner
    assert borderpad(scale_bar_inset_pt=-3.0) == 0.0  # negative clamps, never overhangs


def test_box_margin_ignored_when_box_disabled():
    # With the background box off there is no frame to pad — the Box-margin
    # control must not inset the bar by an invisible phantom frame.
    def bar_x1(margin_pt):
        fig, ax = _ax()
        draw_scale_bar(
            ax,
            5.0,
            style=PlotStyle(scale_bar_box=False, scale_bar_box_margin_pt=margin_pt),
        )
        renderer = _drawn_renderer(fig)
        from matplotlib.patches import Rectangle

        kids = _offsetbox_children(_scale_bar_artist(ax))
        bar = next(p for p in kids if isinstance(p, Rectangle))
        return bar.get_window_extent(renderer).x1

    assert np.isclose(bar_x1(4.0), bar_x1(20.0))


def test_draw_scale_bar_tolerates_nonstandard_loc_and_zero_font_scale():
    # Old code parsed loc by substring and used label_size only as a fontsize;
    # hand-written/stale persisted styles must keep rendering, not crash.
    fig, ax = _ax()
    draw_scale_bar(ax, 5.0, style=PlotStyle(scale_bar_loc="bottom right"))  # no ValueError
    fig2, ax2 = _ax()
    draw_scale_bar(ax2, 5.0, style=PlotStyle(font_scale=0.0))  # no ZeroDivisionError
    assert _scale_bar_artist(ax) is not None and _scale_bar_artist(ax2) is not None


def test_scale_bar_artists_are_clipped_to_axes():
    # AnchoredOffsetbox ignores its own clip settings, so every packed artist
    # (and the frame patch) must be clipped individually — at extreme font
    # scales the assembly truncates at the axes edge instead of overdrawing
    # tick labels (the old code clipped its label for the same reason).
    fig, ax = _ax()
    draw_scale_bar(ax, 5.0, style=PlotStyle(scale_bar_box=True))
    abox = _scale_bar_artist(ax)
    assert abox.patch.get_clip_on()
    # skip the AnchoredOffsetbox container itself — matplotlib ignores its own
    # clip flag (which is exactly why every packed artist is clipped instead)
    for a in _offsetbox_children(abox.get_child()):
        assert a.get_clip_on(), f"{type(a).__name__} is not clipped"


def test_scale_bar_auto_length_is_1_2_5_10_series():
    # Pin the styled renderer's actual auto-length behaviour (1-2-5-10), so it
    # can't drift silently again.
    assert auto_scale_bar_length_um(200.0) == 20.0  # 0.15*200=30 -> 20
    assert auto_scale_bar_length_um(500.0) == 100.0  # 0.15*500=75 -> 100


def test_scale_bar_thickness_pins_styled_geometry():
    # Pin the styled bar thickness (0.004*thickness_pt*|yr|, =0.012*|yr| at the
    # default 3.0 pt) so the kept-as-new geometry can't drift unnoticed. The bar
    # Rectangle lives in an AuxTransformBox(transData), so its height is still
    # expressed in data (µm) coordinates.
    from matplotlib.patches import Rectangle

    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((10, 20)), extent=[0, 20, 0, 10], origin="lower")
    draw_scale_bar(ax, 5.0, style=PlotStyle())
    kids = _offsetbox_children(_scale_bar_artist(ax))
    bar = next(p for p in kids if isinstance(p, Rectangle))
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


def test_round_limits_outward_symmetric_stays_symmetric():
    from dfxm.common.plotting import round_limits_outward

    lo, hi = round_limits_outward(-0.0778, 0.0778)
    assert (lo, hi) == (-0.08, 0.08)


def test_round_limits_outward_examples():
    from dfxm.common.plotting import round_limits_outward

    assert round_limits_outward(0.0, 0.11)[1] == 0.15
    assert round_limits_outward(0.0, 0.0432)[1] == 0.045
    assert abs(round_limits_outward(0.0, 1.7e-4)[1] - 2e-4) < 1e-12
    # asymmetric: vmin floors, vmax ceils
    lo, hi = round_limits_outward(-5.3, -1.2)
    assert (lo, hi) == (-5.5, -1.0)


def test_round_limits_outward_already_round_is_unchanged():
    from dfxm.common.plotting import round_limits_outward

    assert round_limits_outward(-0.08, 0.08) == (-0.08, 0.08)  # no float-epsilon inflation
    assert round_limits_outward(0.0, 0.1) == (0.0, 0.1)


def test_round_limits_outward_degenerate_and_zero():
    import math

    from dfxm.common.plotting import round_limits_outward

    assert round_limits_outward(0.5, 0.5) == (0.5, 0.5)  # degenerate: unchanged
    assert round_limits_outward(0.0, 0.0778) == (0.0, 0.08)  # zero endpoint stays 0
    lo, hi = round_limits_outward(float("nan"), 1.0)  # non-finite: passthrough
    assert math.isnan(lo) and hi == 1.0


def test_round_limits_outward_preserves_tiny_symmetric_fallback():
    from dfxm.common.plotting import round_limits_outward

    # symmetric_limits() returns ±1e-12 for all-zero data; rounding must not
    # collapse it to a degenerate (0.0, 0.0) range
    assert round_limits_outward(-1e-12, 1e-12) == (-1e-12, 1e-12)


def test_apply_round_clim_notes_and_gating():
    from dfxm.common.plotting import apply_round_clim

    # disabled (default style) and style=None: passthrough, no note
    assert apply_round_clim(-0.0778, 0.0778, PlotStyle()) == (-0.0778, 0.0778, None)
    assert apply_round_clim(-0.0778, 0.0778, None) == (-0.0778, 0.0778, None)
    # enabled: rounded + symmetric note
    lo, hi, note = apply_round_clim(-0.0778, 0.0778, PlotStyle(round_clim=True))
    assert (lo, hi) == (-0.08, 0.08)
    assert note == "colour limits rounded ±0.0778 → ±0.08 (round_clim)"
    # enabled but already round: no note
    assert apply_round_clim(-0.08, 0.08, PlotStyle(round_clim=True))[2] is None
    # asymmetric note shows both pairs
    _, _, note = apply_round_clim(0.0, 0.11, PlotStyle(round_clim=True))
    assert note == "colour limits rounded (0, 0.11) → (0, 0.15) (round_clim)"


def _title_y0(fig, ax):
    """Draw fig on Agg and return the title's bottom window-coord (y0)."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return ax.title.get_window_extent(canvas.get_renderer()).y0


def test_apply_text_scale_does_not_change_title_pad_on_plain_figure():
    """apply_text_scale must not alter the title pad on a non-constrained figure.

    Before the fix, apply_text_scale unconditionally set pad=12.0*fs, doubling
    matplotlib's default (6.0 pt) even on plain figures produced by layer_figure
    with style=None — a silent legacy-output drift.
    """
    # Reference: untouched plain Figure with the same title (matplotlib default pad)
    fig_ref = Figure(figsize=(6, 4))
    ax_ref = fig_ref.add_subplot(111)
    ax_ref.set_title("test title")
    ref_y = _title_y0(fig_ref, ax_ref)

    # Treated: plain Figure after apply_text_scale with a default PlotStyle
    fig_treated = Figure(figsize=(6, 4))
    ax_treated = fig_treated.add_subplot(111)
    ax_treated.set_title("test title")
    apply_text_scale(ax_treated, PlotStyle())  # font_scale=1.0, title_scale=1.0
    treated_y = _title_y0(fig_treated, ax_treated)

    assert treated_y == ref_y, (
        f"apply_text_scale altered the title pad on a plain (non-constrained) figure "
        f"(ref y0={ref_y:.3f}, treated y0={treated_y:.3f}) — legacy output drift"
    )


def test_apply_text_scale_increases_title_pad_on_constrained_figure():
    """On a constrained-layout figure, apply_text_scale must increase the title pad.

    The pad compensates for the ×10ⁿ colorbar-offset text that constrained layout
    does not account for when reserving room for the title.  A colorbar with
    scientific notation is needed here so the constrained-layout engine has actual
    competing artists and the pad effect is visible in window coordinates.
    """
    style = PlotStyle(font_scale=2.2, colorbar_ticks=5, tickfmt_strain="scientific")
    data = np.linspace(-2e-3, 2e-3, 100).reshape(10, 10)

    # Reference: constrained figure WITH colorbar but WITHOUT apply_text_scale
    fig_ref = styled_figure((3.5, 3.0), styled=True)
    ax_ref = fig_ref.add_subplot(111)
    im_ref = ax_ref.imshow(data)
    ax_ref.set_title("test title")
    add_colorbar(fig_ref, im_ref, ax_ref, "label", style, group="strain")
    ref_y = _title_y0(fig_ref, ax_ref)

    # Treated: same setup WITH apply_text_scale (pad should push title up)
    fig_treated = styled_figure((3.5, 3.0), styled=True)
    ax_treated = fig_treated.add_subplot(111)
    im_treated = ax_treated.imshow(data)
    ax_treated.set_title("test title")
    add_colorbar(fig_treated, im_treated, ax_treated, "label", style, group="strain")
    apply_text_scale(ax_treated, style)
    treated_y = _title_y0(fig_treated, ax_treated)

    assert treated_y > ref_y, (
        f"apply_text_scale should push the title up on a constrained-layout figure "
        f"at large font_scale, but ref y0={ref_y:.3f} >= treated y0={treated_y:.3f}"
    )


def test_per_group_tickfmt_defaults_and_lookup():
    from dfxm.common.plotting import GROUP_BY_KIND

    s = PlotStyle()
    # bare defaults preserve the legacy look
    assert s.tickfmt_for("strain") == "auto"
    assert s.tickfmt_for("raw") == "auto"
    assert s.offset_scale_for("mosa_com") == 1.0
    assert s.offset_pos_for("mosa_fwhm") == "top"
    # group=None is the neutral fallback (callers that don't know their group)
    assert s.tickfmt_for(None) == "auto"
    assert s.offset_scale_for(None) == 1.0
    assert s.offset_pos_for(None) == "top"
    # unknown non-None group raises, like cmap_for
    import pytest

    with pytest.raises(KeyError):
        s.tickfmt_for("bogus")
    # explicit per-group values round-trip through the lookups
    s2 = PlotStyle(tickfmt_strain="scientific", offset_scale_strain=1.5, offset_pos_strain="bottom")
    assert s2.tickfmt_for("strain") == "scientific"
    assert s2.offset_scale_for("strain") == 1.5
    assert s2.offset_pos_for("strain") == "bottom"
    # GROUP_BY_KIND collapses raw_sum / raw_specific onto the raw group
    assert GROUP_BY_KIND["raw_sum"] == "raw"
    assert GROUP_BY_KIND["raw_specific"] == "raw"
    assert GROUP_BY_KIND["strain"] == "strain"


def test_tick_formatter_scientific_and_arb_are_deferred():
    # scientific + arb are handled inside add_colorbar, not by _tick_formatter
    assert _tick_formatter("scientific") is None
    assert _tick_formatter("arb") is None
    # digit + auto behaviour unchanged
    assert _tick_formatter("2") is not None
    assert _tick_formatter("auto") is None


def test_add_colorbar_arbitrary_units_drops_ticks_and_marks_label():
    fig, ax = _ax()
    im = ax.imshow(np.arange(100).reshape(10, 10), extent=[0, 50, 0, 30], origin="lower")
    style = PlotStyle(tickfmt_raw="arb")
    cb = add_colorbar(fig, im, ax, "Intensity", style, group="raw")
    assert list(cb.get_ticks()) == []  # no numeric ticks
    assert cb.ax.get_ylabel() == "Intensity (arb. units)"


def test_add_colorbar_arbitrary_units_does_not_double_up_existing_au():
    fig, ax = _ax()
    im = ax.imshow(np.arange(100).reshape(10, 10), extent=[0, 50, 0, 30], origin="lower")
    style = PlotStyle(tickfmt_raw="arb")
    cb = add_colorbar(fig, im, ax, "Sum intensity (a.u.)", style, group="raw")
    assert cb.ax.get_ylabel() == "Sum intensity (a.u.)"  # already mentions a.u. -> no suffix


def test_style_from_dict_migrates_old_snapshot_to_tuned_defaults():
    from dfxm.common.plotting import _style_from_dict

    # An old GUI snapshot: no per-group tickfmt_* keys at all.
    old = {"font_scale": 2.2, "colorbar_ticks": 5}
    s = _style_from_dict(old)
    assert s.tickfmt_for("strain") == "scientific"
    assert s.tickfmt_for("raw") == "arb"
    assert s.tickfmt_for("mosa_com") == "auto"
    assert s.offset_pos_for("strain") == "bottom"

    # A current snapshot carrying the keys is left exactly as-is (no migration).
    new = {"tickfmt_strain": "auto", "tickfmt_raw": "auto", "font_scale": 1.0}
    s2 = _style_from_dict(new)
    assert s2.tickfmt_for("strain") == "auto"
    assert s2.tickfmt_for("raw") == "auto"


def test_publication_style_is_tuned_per_group():
    from dfxm.common.plotting import PUBLICATION_STYLE

    assert PUBLICATION_STYLE.tickfmt_for("strain") == "scientific"
    assert PUBLICATION_STYLE.tickfmt_for("raw") == "arb"
    assert PUBLICATION_STYLE.tickfmt_for("mosa_com") == "auto"
    assert PUBLICATION_STYLE.offset_pos_for("strain") == "bottom"


def test_add_colorbar_scientific_hides_builtin_offset_and_draws_custom():
    from matplotlib.text import Text

    fig, ax = _ax()
    im = ax.imshow(
        np.linspace(-2e-3, 2e-3, 100).reshape(10, 10), extent=[0, 50, 0, 30], origin="lower"
    )
    style = PlotStyle(
        tickfmt_strain="scientific",
        offset_pos_strain="bottom",
        offset_scale_strain=2.0,
        font_scale=1.0,
    )
    cb = add_colorbar(fig, im, ax, "Strain (ε)", style, group="strain")
    # matplotlib's built-in offset text is hidden
    assert cb.ax.yaxis.get_offset_text().get_visible() is False
    # exactly one custom exponent label exists, sized by font_scale*offset_scale
    exps = [
        t
        for t in cb.ax.texts
        if isinstance(t, Text) and "10" in t.get_text() and "times" in t.get_text()
    ]
    assert len(exps) == 1
    assert abs(exps[0].get_fontsize() - 9 * 1.0 * 2.0) < 1e-6
    # placed below the axes (va="top", y < 0)
    assert exps[0].get_position()[1] < 0.0


def test_fixed_scale_defensive_parse():
    assert fixed_scale(None) is None
    assert fixed_scale(PlotStyle()) is None
    assert fixed_scale(PlotStyle(scale_um_per_cm=50.0)) == 50.0
    assert fixed_scale(PlotStyle(scale_um_per_cm="50")) == 50.0  # stale persisted string
    assert fixed_scale(PlotStyle(scale_um_per_cm="junk")) is None
    assert fixed_scale(PlotStyle(scale_um_per_cm=-3)) is None
    assert fixed_scale(PlotStyle(scale_um_per_cm=0)) is None


def test_trace_fixed_scale_precedence_and_fallback():
    from dfxm.common.plotting import trace_fixed_scale

    assert trace_fixed_scale(None) is None
    assert trace_fixed_scale(PlotStyle()) is None
    # blank trace field inherits the map scale
    assert trace_fixed_scale(PlotStyle(scale_um_per_cm=50.0)) == 50.0
    # a valid trace value wins over the map scale
    assert trace_fixed_scale(PlotStyle(scale_um_per_cm=50.0, trace_scale_um_per_cm=25.0)) == 25.0
    # trace value alone works without a map scale
    assert trace_fixed_scale(PlotStyle(trace_scale_um_per_cm=25.0)) == 25.0
    assert trace_fixed_scale(PlotStyle(trace_scale_um_per_cm="25")) == 25.0  # persisted string
    # junk / non-positive trace values degrade to the map-scale fallback
    assert trace_fixed_scale(PlotStyle(scale_um_per_cm=50.0, trace_scale_um_per_cm="junk")) == 50.0
    assert trace_fixed_scale(PlotStyle(scale_um_per_cm=50.0, trace_scale_um_per_cm=-3)) == 50.0
    assert trace_fixed_scale(PlotStyle(trace_scale_um_per_cm=0)) is None


def test_fixed_scale_box_explicit_scale_override():
    from dfxm.common.plotting import fixed_scale_box

    # explicit scale wins over the style's own scale_um_per_cm
    box = fixed_scale_box(PlotStyle(scale_um_per_cm=50.0), 100.0, 50.0, scale=25.0)
    assert abs(box[0] - 100.0 / 25.0 / 2.54) < 1e-9
    assert abs(box[1] - 50.0 / 25.0 / 2.54) < 1e-9
    # scale=None keeps the style read (backward compatible)
    box2 = fixed_scale_box(PlotStyle(scale_um_per_cm=50.0), 100.0, 50.0)
    assert abs(box2[0] - 100.0 / 50.0 / 2.54) < 1e-9
    # explicit scale with no style scale still works
    assert fixed_scale_box(PlotStyle(), 100.0, 50.0, scale=25.0) is not None
    assert fixed_scale_box(PlotStyle(), 100.0, 50.0) is None


def test_fixed_scale_box_geometry_clamp_and_degenerate():
    box = fixed_scale_box(PlotStyle(scale_um_per_cm=50.0), 200.0, 100.0)
    assert box is not None
    w, h, eff = box
    assert w == pytest.approx(200.0 / 50.0 / 2.54)
    assert h == pytest.approx(100.0 / 50.0 / 2.54)
    assert eff == 50.0
    # typo scale (0.1 µm/cm on 200 µm ≈ 787 in): clamped to 30 in, aspect kept, scale raised
    w, h, eff = fixed_scale_box(PlotStyle(scale_um_per_cm=0.1), 200.0, 100.0)
    assert max(w, h) == pytest.approx(30.0)
    assert h / w == pytest.approx(0.5)
    assert eff > 0.1
    # degenerate extents / knob off -> None
    assert fixed_scale_box(PlotStyle(scale_um_per_cm=50.0), 0.0, 100.0) is None
    assert fixed_scale_box(PlotStyle(), 200.0, 100.0) is None
    assert fixed_scale_box(None, 200.0, 100.0) is None


def test_scale_um_per_cm_json_roundtrip_and_old_snapshots():
    s2 = style_from_json(style_to_json(PlotStyle(scale_um_per_cm=75.0)))
    assert s2 is not None and s2.scale_um_per_cm == 75.0
    old = style_from_json('{"font_scale": 2.0}')  # snapshot predating the knob
    assert old is not None and old.scale_um_per_cm is None


def _bar_rect(ax):
    """The scale-bar Rectangle inside the AnchoredOffsetbox assembly."""
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox

    box = next(a for a in ax.artists if isinstance(a, AnchoredOffsetbox))
    stack = [box.get_child()]
    while stack:
        a = stack.pop()
        if isinstance(a, AuxTransformBox):
            return a.get_children()[0]
        if hasattr(a, "get_children"):
            stack.extend(a.get_children())
    raise AssertionError("no bar rectangle found")


def _bar_axes(xr=200.0, yr=100.0):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, xr)
    ax.set_ylim(0, yr)
    return ax


def test_draw_scale_bar_fixed_mode_height_is_point_exact():
    style = PlotStyle(scale_bar_thickness_pt=4.0)
    ax = _bar_axes()
    draw_scale_bar(ax, 50.0, style=style, fixed_scale_um_per_cm=100.0)
    assert _bar_rect(ax).get_height() == pytest.approx(4.0 * (2.54 / 72.0) * 100.0)
    assert _bar_rect(ax).get_width() == pytest.approx(50.0)


def test_draw_scale_bar_default_mode_geometry_unchanged():
    style = PlotStyle(scale_bar_thickness_pt=3.0)
    ax = _bar_axes(yr=100.0)
    draw_scale_bar(ax, 50.0, style=style)  # no kwarg -> today's geometry
    assert _bar_rect(ax).get_height() == pytest.approx(100.0 * 0.004 * 3.0)


def test_draw_scale_bar_fixed_mode_rejects_non_positive_scale():
    style = PlotStyle(scale_bar_thickness_pt=3.0)
    legacy = 100.0 * 0.004 * 3.0

    ax = _bar_axes(yr=100.0)
    draw_scale_bar(ax, 50.0, style=style, fixed_scale_um_per_cm=0.0)
    assert _bar_rect(ax).get_height() == pytest.approx(legacy)

    ax2 = _bar_axes(yr=100.0)
    draw_scale_bar(ax2, 50.0, style=style, fixed_scale_um_per_cm=-5.0)
    assert _bar_rect(ax2).get_height() == pytest.approx(legacy)


def _bare_ax():
    fig = Figure()
    return fig.add_subplot(111)


def test_axes_mode_default_is_full():
    assert PlotStyle().axes_mode == "full"
    assert PUBLICATION_STYLE.axes_mode == "full"


def test_apply_axes_mode_no_frame_hides_spines_keeps_ticks():
    ax = _bare_ax()
    apply_axes_mode(ax, PlotStyle(axes_mode="no_frame"))
    assert all(not sp.get_visible() for sp in ax.spines.values())
    assert ax.axison  # ticks and labels survive


def test_apply_axes_mode_none_removes_axes():
    ax = _bare_ax()
    apply_axes_mode(ax, PlotStyle(axes_mode="none"))
    assert not ax.axison


def test_apply_axes_mode_full_and_stale_values_are_noops():
    for mode in ("full", "boxless", "", 0, None):
        ax = _bare_ax()
        apply_axes_mode(ax, replace(PlotStyle(), axes_mode=mode))
        assert ax.axison
        assert all(sp.get_visible() for sp in ax.spines.values())


def test_axes_mode_json_roundtrip_and_legacy_snapshot_default():
    assert style_from_json(style_to_json(PlotStyle(axes_mode="none"))).axes_mode == "none"
    # a persisted snapshot from before this feature has no axes_mode key
    assert style_from_json(json.dumps({"font_scale": 2.0})).axes_mode == "full"


def test_histogram_keeps_axes_under_axes_mode_none():
    fig = build_histogram(
        np.linspace(-1.0, 1.0, 100),
        title="t",
        xlabel="x",
        style=PlotStyle(axes_mode="none"),
    )
    assert fig.axes[0].axison


def test_trace_height_cm_roundtrips_and_defaults():
    from dfxm.common.plotting import PlotStyle, style_from_json, style_to_json, trace_height_cm

    st = PlotStyle(trace_height_cm=4.2)
    assert style_from_json(style_to_json(st)).trace_height_cm == 4.2
    # old persisted styles (no field) load with the default
    st_old = style_from_json(style_to_json(PlotStyle()))
    assert trace_height_cm(st_old) == 3.0


def _cbar_fig():
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    im = ax.imshow(np.linspace(0.0, 1.0, 20).reshape(4, 5))
    return fig, ax, im


def test_cbar_typography_knobs_applied():
    fig, ax, im = _cbar_fig()
    st = PlotStyle(cbar_label_scale=2.0, cbar_tick_scale=0.5, cbar_labelpad_pt=17.0)
    cb = add_colorbar(fig, im, ax, "value", st)
    assert cb.ax.yaxis.label.get_fontsize() == 20.0  # 10 * font_scale * cbar_label_scale
    assert cb.ax.yaxis.labelpad == 17.0
    fig.canvas.draw()
    ticklabs = [t for t in cb.ax.get_yticklabels() if t.get_text()]
    assert ticklabs and all(t.get_fontsize() == 4.5 for t in ticklabs)  # 9 * 1.0 * 0.5


def test_cbar_typography_defaults_byte_compatible():
    fig, ax, im = _cbar_fig()
    default_pad = None
    cb = add_colorbar(fig, im, ax, "value", PlotStyle())
    assert cb.ax.yaxis.label.get_fontsize() == 10.0
    fig.canvas.draw()
    ticklabs = [t for t in cb.ax.get_yticklabels() if t.get_text()]
    assert ticklabs and all(t.get_fontsize() == 9.0 for t in ticklabs)
    default_pad = cb.ax.yaxis.labelpad
    fig2, ax2, im2 = _cbar_fig()
    cb2 = add_colorbar(fig2, im2, ax2, "value", PlotStyle(cbar_labelpad_pt=None))
    assert cb2.ax.yaxis.labelpad == default_pad  # None -> matplotlib default, unchanged


def test_apply_axis_tickfmt_scientific_digits_and_auto():
    from dfxm.common.plotting import apply_axis_tickfmt

    # scientific: mantissa ticks + our own x10^n exponent artist (the built-in
    # scilimits offset silently vanishes under constrained layout, mpl 3.6)
    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.plot([0.0, 10.0], [0.0, 50.0])
    apply_axis_tickfmt(ax, PlotStyle(tickfmt_strain="scientific"), "strain")
    fig.canvas.draw()
    assert any("10^{1}" in t.get_text() for t in ax.texts)  # exponent for 0..50
    assert not ax.yaxis.get_offset_text().get_visible()
    labels = [t.get_text() for t in ax.get_yticklabels() if t.get_text()]
    assert labels and all(float(x) <= 6 for x in labels)  # mantissas, not raw values
    # digits: fixed decimal count
    fig2 = Figure()
    FigureCanvasAgg(fig2)
    ax2 = fig2.add_subplot(111)
    ax2.plot([0.0, 10.0], [0.0, 50.0])
    apply_axis_tickfmt(ax2, PlotStyle(tickfmt_mosa_com="2"), "mosa_com")
    fig2.canvas.draw()
    texts = [t.get_text() for t in ax2.get_yticklabels() if t.get_text()]
    assert texts and all("." in t and len(t.split(".")[1]) == 2 for t in texts)
    # auto: untouched (no offset for plain 0..50)
    fig3 = Figure()
    FigureCanvasAgg(fig3)
    ax3 = fig3.add_subplot(111)
    ax3.plot([0.0, 10.0], [0.0, 50.0])
    apply_axis_tickfmt(ax3, PlotStyle(), "strain")
    fig3.canvas.draw()
    assert not any("10^{" in t.get_text() for t in ax3.texts)
