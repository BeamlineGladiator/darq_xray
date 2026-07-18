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
    tickfmt_mosa_com="scientific",
    # bottom placement is the shipped publication default; top+scientific at this
    # font scale has a known ~2px title graze, so we pin the default the regression guards.
    offset_pos_mosa_com="bottom",
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
        "group": "mosa_com",
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


def _box_inches(fig, ax):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    return bb.width / fig.dpi, bb.height / fig.dpi


def test_fit_axes_to_box_reaches_target_under_two_decoration_loads():
    import numpy as np

    from dfxm.common.plotting import fit_axes_to_box, styled_figure

    for title, with_cbar in (
        ("A long two-line title\nwith even more text on the second line", True),
        ("t", False),
    ):
        fig = styled_figure((6.0, 5.0), styled=True)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            np.random.default_rng(0).random((10, 20)),
            extent=[0, 200, 0, 100],
            origin="lower",
            aspect="equal",
        )
        ax.set_title(title)
        if with_cbar:
            fig.colorbar(im, ax=ax, fraction=0.07)
        assert fit_axes_to_box(fig, ax, 3.0, 1.5) is True
        w, h = _box_inches(fig, ax)
        assert abs(w - 3.0) <= 0.05 and abs(h - 1.5) <= 0.05


def test_fit_axes_to_box_nonconvergence_is_nonfatal():
    import numpy as np

    from dfxm.common.plotting import fit_axes_to_box, styled_figure

    fig = styled_figure((2.0, 2.0), styled=True)
    ax = fig.add_subplot(111)
    ax.imshow([[0.0, 1.0]], extent=[0, 2, 0, 1], origin="lower", aspect="equal")
    ok = fit_axes_to_box(fig, ax, 5.0, 2.5, tol_in=1e-9, max_iter=1)
    assert ok is False  # kept the last size, did not raise
    assert np.all(np.isfinite(fig.get_size_inches()))


def test_finalize_fixed_scale_noop_when_knob_off():
    from dfxm.common.plotting import PlotStyle, finalize_fixed_scale, styled_figure

    fig = styled_figure((6.0, 5.0), styled=True)
    ax = fig.add_subplot(111)
    finalize_fixed_scale(fig, ax, PlotStyle(), 200.0, 100.0)
    finalize_fixed_scale(fig, ax, None, 200.0, 100.0)
    assert tuple(fig.get_size_inches()) == (6.0, 5.0)


def test_layer_figure_fixed_scale_equal_boxes_across_decoration_loads():
    import numpy as np

    from dfxm.common import render
    from dfxm.common.plotting import PlotStyle

    layer = np.random.default_rng(1).random((10, 20))
    style = PlotStyle(scale_um_per_cm=50.0, figure_width="single", tickfmt_raw="scientific")
    boxes = []
    for vmax, title in ((1.0e-4, "short"), (123456.0, "a much longer two-line\ntitle text here")):
        fig, ax, _ = render.layer_figure(
            layer * vmax,
            0.0,
            vmax,
            "gray",
            200.0,
            100.0,
            title,
            "I (a.u.)",
            style=style,
            group="raw",
        )
        boxes.append(_box_inches(fig, ax))
    target_w, target_h = 200.0 / 50.0 / 2.54, 100.0 / 50.0 / 2.54
    for w, h in boxes:
        assert abs(w - target_w) <= 0.05 and abs(h - target_h) <= 0.05


def test_build_strain_map_fixed_scale_box():
    import numpy as np

    from dfxm.common.plotting import PlotStyle
    from dfxm.stages.strain import build_strain_map

    strain = np.random.default_rng(2).standard_normal((50, 100)) * 1e-4
    style = PlotStyle(scale_um_per_cm=10.0)
    fig = build_strain_map(strain, 1.0, 1.0, None, (None, None), style=style)
    w, h = _box_inches(fig, fig.axes[0])
    assert abs(w - 100.0 / 10.0 / 2.54) <= 0.05
    assert abs(h - 50.0 / 10.0 / 2.54) <= 0.05
