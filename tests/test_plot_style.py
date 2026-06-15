from dataclasses import replace

from dfxm.common.plotting import PUBLICATION_STYLE, PlotStyle, figure_size


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
