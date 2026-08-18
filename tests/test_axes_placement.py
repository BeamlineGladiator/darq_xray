"""Deterministic axes placement (trace fixed-scale engine) — dfxm.common.plotting."""

import numpy as np
from matplotlib.figure import Figure

from dfxm.common.plotting import (
    AxesMargins,
    PlotStyle,
    box_drift_note,
    measure_axes_margins,
    measured_box_in,
    place_axes_box,
    trace_fixed_box,
    trace_height_cm,
)


def _plot_fig(ylabel="value", title="a title", font=10.0):
    fig = Figure(figsize=(6, 4), facecolor="white")
    ax = fig.add_subplot(111)
    x = np.linspace(0, 29.668647, 200)
    ax.plot(x, np.sin(x / 5.0) * 1e-4)
    ax.set_xlim(0, x[-1])
    ax.set_xlabel("distance along line (µm)", fontsize=font * 1.2)
    ax.set_ylabel(ylabel, fontsize=font)
    ax.set_title(title, loc="left", fontsize=font)
    ax.tick_params(labelsize=font)
    return fig, ax


def test_place_axes_box_exact_small_box_large_fonts():
    # the exact configuration that defeated fit_axes_to_box (L=29.67, 10 um/cm)
    for font in (10.0, 14.0, 20.0):
        fig, ax = _plot_fig(ylabel="COM mu (deg)", font=font)
        place_axes_box(fig, ax, 29.668647 / 10.0 / 2.54, 3.0 / 2.54)
        w, h = measured_box_in(fig, ax)
        assert abs(w - 29.668647 / 10.0 / 2.54) < 0.005 * w, (font, w)
        assert abs(h - 3.0 / 2.54) < 0.005 * h, (font, h)


def test_place_axes_box_with_shared_margins_keeps_box_and_margins():
    fig1, ax1 = _plot_fig(ylabel="short")
    fig2, ax2 = _plot_fig(ylabel="a much longer y-axis label (units)")
    m1 = place_axes_box(fig1, ax1, 2.0, 1.2)
    m2 = place_axes_box(fig2, ax2, 2.0, 1.2)
    shared = m1.max_with(m2)
    place_axes_box(fig1, ax1, 2.0, 1.2, margins=shared)
    place_axes_box(fig2, ax2, 2.0, 1.2, margins=shared)
    # identical canvas sizes and identical box positions
    assert np.allclose(fig1.get_size_inches(), fig2.get_size_inches())
    assert np.allclose(list(ax1.get_position().bounds), list(ax2.get_position().bounds))
    for fig, ax in ((fig1, ax1), (fig2, ax2)):
        w, h = measured_box_in(fig, ax)
        assert abs(w - 2.0) < 0.01 and abs(h - 1.2) < 0.01


def test_measure_axes_margins_covers_decorations():
    fig, ax = _plot_fig()
    place_axes_box(fig, ax, 2.5, 1.5)
    m = measure_axes_margins(fig, ax)
    assert m.left > 0.2 and m.bottom > 0.2 and m.top > 0.05  # labels/ticks/title exist
    fw, fh = fig.get_size_inches()
    assert fw >= m.left + 2.5 and fh >= m.bottom + 1.5  # canvas holds box+margins


def test_measure_axes_margins_includes_axis_label_overhang():
    """A very wide x label on a narrow axes must widen the LEFT/RIGHT margins:
    matplotlib's ``get_tightbbox`` squashes axis-label widths to 1 px
    (`for_layout_only`), which let long labels overflow into neighbours."""
    fig = Figure(figsize=(6, 4), facecolor="white")
    ax = fig.add_subplot(111)
    ax.plot([0, 1], [0, 1])
    ax.set_yticks([])
    ax.set_xlabel("a very very very long distance-along-line label (µm)", fontsize=14)
    place_axes_box(fig, ax, 0.6, 1.0)  # 0.6 in wide box, label ~ 5 in wide
    m = measure_axes_margins(fig, ax)
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    lab = ax.xaxis.label.get_window_extent(ren)
    box = ax.get_window_extent(ren)
    d = fig.dpi
    assert m.right >= (lab.x1 - box.x1) / d - 1e-6
    assert m.left >= (box.x0 - lab.x0) / d - 1e-6


def test_measure_axes_margins_ignores_labels_of_axis_off_axes():
    fig = Figure(figsize=(6, 4), facecolor="white")
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((4, 4)))
    ax.set_xlabel("a very very very long label that would be huge")
    ax.set_ylabel("another very long label")
    ax.set_axis_off()
    place_axes_box(fig, ax, 1.0, 1.0)
    m = measure_axes_margins(fig, ax, pad_in=0.0)
    assert max(m.left, m.right, m.top, m.bottom) < 0.05


def test_axes_margins_max_with():
    a = AxesMargins(1.0, 0.1, 0.2, 0.5)
    b = AxesMargins(0.5, 0.4, 0.1, 0.9)
    m = a.max_with(b)
    assert (m.left, m.right, m.top, m.bottom) == (1.0, 0.4, 0.2, 0.9)


def test_trace_height_cm_defensive():
    assert trace_height_cm(PlotStyle()) == 3.0  # default when unset
    assert trace_height_cm(PlotStyle(trace_height_cm=4.5)) == 4.5
    assert trace_height_cm(PlotStyle(trace_height_cm=-1)) == 3.0
    assert trace_height_cm(PlotStyle(trace_height_cm="junk")) == 3.0
    assert trace_height_cm(None) == 3.0


def test_trace_fixed_box_geometry_and_clamp():
    st = PlotStyle(trace_scale_um_per_cm=10.0, trace_height_cm=3.0)
    box = trace_fixed_box(st, 44.941256)
    assert box is not None
    w, h, s = box
    assert abs(w - 44.941256 / 10.0 / 2.54) < 1e-9
    assert abs(h - 3.0 / 2.54) < 1e-9
    assert s == 10.0
    assert trace_fixed_box(PlotStyle(), 40.0) is None  # knob off
    assert trace_fixed_box(st, 0.0) is None  # degenerate line
    w, h, s = trace_fixed_box(PlotStyle(trace_scale_um_per_cm=0.1, trace_height_cm=3.0), 500.0)
    assert w == 30.0 and s > 0.1  # width clamped, effective scale raised


def test_trace_fixed_box_height_clamp(caplog):
    # An extreme trace_height_cm clamps the height side to 30 in, leaving the
    # width/scale untouched (height has no bearing on the effective µm/cm) and
    # logs a warning — the height clamp must not be silent, mirroring the
    # width clamp above.
    import logging

    st = PlotStyle(trace_scale_um_per_cm=10.0, trace_height_cm=100.0)  # 100/2.54 = 39.4 in
    with caplog.at_level(logging.WARNING, logger="dfxm.common.plotting"):
        w, h, s = trace_fixed_box(st, 44.941256)
    assert h == 30.0
    assert abs(w - 44.941256 / 10.0 / 2.54) < 1e-9  # width unchanged
    assert s == 10.0  # scale unchanged
    assert any("height clamped" in rec.message for rec in caplog.records)


def test_box_drift_note_fires_only_on_miss():
    fig, ax = _plot_fig()
    place_axes_box(fig, ax, 2.0, 1.2)
    assert box_drift_note("t", fig, ax, 2.0, 1.2) is None
    note = box_drift_note("t", fig, ax, 3.0, 1.2)  # deliberately wrong target
    assert note is not None and "t" in note and "cm" in note


def test_place_axes_stack_left_aligned_exact_boxes():
    fig = Figure(figsize=(8, 10), facecolor="white")
    axs = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["short", "a very very long y label (units)", "mid label"]
    for ax, lab in zip(axs, labels):
        ax.plot([0, 1], [0, 1])
        ax.set_ylabel(lab)
    from dfxm.common.plotting import place_axes_stack

    boxes = [(2.5, 1.6), (1.4, 1.0), (2.0, 1.0)]
    place_axes_stack(fig, [(ax, w, h, (), None) for ax, (w, h) in zip(axs, boxes)])
    x0 = {round(ax.get_position().x0, 4) for ax in axs}
    assert len(x0) == 1  # shared left edge
    for ax, (w, h) in zip(axs, boxes):
        bw, bh = measured_box_in(fig, ax)
        assert abs(bw - w) < 0.01 and abs(bh - h) < 0.01
    # panels must not overlap: y-intervals strictly descending
    ys = [ax.get_position() for ax in axs]
    assert ys[0].y0 > ys[1].y1 - 1e-6 and ys[1].y0 > ys[2].y1 - 1e-6
