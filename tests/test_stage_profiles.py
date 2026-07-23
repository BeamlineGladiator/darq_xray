"""Tests for dfxm.stages.profiles — profiling core (the legacy self-test, as
pytest) and end-to-end figure/CSV generation from a consolidated slice file.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from dfxm.common.errors import StageUserError
from dfxm.stages import profiles as PR

A, B = 0.7, -1.3  # linear field coefficients


def _linear_plane():
    u = np.linspace(-10.0, 10.0, 201)
    v = np.linspace(-8.0, 8.0, 161)
    uu, vv = np.meshgrid(u, v)
    return u, v, A * uu + B * vv


# -- profiling core (mirrors the legacy self-test) ----------------------------
def test_single_line_on_linear_field():
    u, v, plane = _linear_plane()
    geom = PR.line_geometry(u, v, (-5.0, -3.0), (5.0, 3.0), 80, 1, PR.grid_pitch(u, v))
    vm, vs, _ = PR.profile_plane(plane, geom)
    assert vs is None
    pts = np.asarray((-5.0, -3.0))[None, :] + geom["distance"][:, None] * geom["dhat"][None, :]
    expect = A * pts[:, 0] + B * pts[:, 1]
    assert float(np.nanmax(np.abs(vm - expect))) < 1e-4


def test_nan_propagates_but_tails_finite():
    u, v, plane = _linear_plane()
    plane = plane.copy()
    plane[60:100, 90:110] = np.nan
    geom = PR.line_geometry(u, v, (-5.0, -3.0), (5.0, 3.0), 80, 1, PR.grid_pitch(u, v))
    vm, _, _ = PR.profile_plane(plane, geom)
    assert np.any(np.isnan(vm))
    assert np.isfinite(vm[0]) and np.isfinite(vm[-1])


def test_out_of_plane_line_all_nan():
    u, v, plane = _linear_plane()
    geom = PR.line_geometry(u, v, (100.0, 100.0), (120.0, 100.0), 20, 1, PR.grid_pitch(u, v))
    vm, _, _ = PR.profile_plane(plane, geom)
    assert np.all(np.isnan(vm))


def test_band_on_constant_field():
    u, v, plane = _linear_plane()
    const = np.full_like(plane, 3.14159)
    geom = PR.line_geometry(u, v, (-5.0, -3.0), (5.0, 3.0), 40, 7, PR.grid_pitch(u, v))
    cm, csd, _ = PR.profile_plane(const, geom)
    assert float(np.nanmax(np.abs(cm - 3.14159))) < 1e-6
    assert float(np.nanmax(np.abs(csd))) < 1e-9  # zero spread across the band


# -- end to end ---------------------------------------------------------------
def _write_consolidated(path):
    """Minimal oblique_slices.h5: two fields sharing one slice's (u,v) grid."""
    u = np.linspace(-10.0, 10.0, 81)
    v = np.linspace(-8.0, 8.0, 65)
    uu, vv = np.meshgrid(u, v)
    offsets = np.array([-1.0, 0.0, 1.0])
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (("raw_sum", "raw_sum", "gray"), ("strain", "strain", "RdBu_r")):
            g = f.create_group(vid)
            g.attrs["kind"] = kind
            g.attrs["cbar_label"] = "value"
            g.attrs["cmap"] = cmap
            g.attrs["title"] = vid
            g.attrs["vmin"] = -10.0
            g.attrs["vmax"] = 10.0
            sg = g.create_group("oblique_full")
            stack = np.stack([A * uu + B * vv + o for o in offsets], axis=0).astype(np.float32)
            sg.create_dataset("slices", data=stack)
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offsets)


def test_run_parameter_writes_figure_and_csv(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"prof0"}]'
    )
    res = PR.run(
        {"consolidated_h5": str(h5), "mode": "parameter", "jobs_json": jobs, "output_dir": str(out)}
    )
    assert len(res.jobs) == 1
    jr = res.jobs[0]
    assert jr.figure and os.path.exists(jr.figure)
    assert set(jr.fields) == {"raw_sum", "strain"}  # all fields profiled
    assert jr.csvs and all(os.path.exists(c) for c in jr.csvs)
    # CSV content matches the linear profile
    data = np.loadtxt(
        next(c for c in jr.csvs if c.endswith("strain.csv")), delimiter=",", skiprows=1
    )
    assert data.shape[1] == 2 and data.shape[0] == 40


def test_run_preview(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    jobs = '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3]}]'
    res = PR.run(
        {
            "consolidated_h5": str(h5),
            "mode": "preview",
            "jobs_json": jobs,
            "output_dir": str(tmp_path / "prev"),
        }
    )
    assert len(res.jobs) == 1 and os.path.exists(res.jobs[0].figure)


def test_run_with_injected_style_produces_figures(tmp_path):
    """A run carrying the GUI plot_style snapshot renders through the styled path."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof_styled"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"prof0"}]'
    )
    res = PR.run(
        {
            "consolidated_h5": str(h5),
            "mode": "parameter",
            "jobs_json": jobs,
            "output_dir": str(out),
            "save_overview": True,
            "plot_style": {"font_scale": 2.0, "colorbar": True},
        }
    )
    assert len(res.jobs) == 1 and os.path.exists(res.jobs[0].figure)


def test_render_single_accepts_style(tmp_path):
    from dfxm.common.plotting import PlotStyle

    plane = np.linspace(0, 1, 12).reshape(3, 4)
    attrs = {"cmap": "gray", "cbar_label": "c", "title": "t", "vmin": 0.0, "vmax": 1.0}
    ref = (plane, np.arange(4.0), np.arange(3.0), attrs, "lbl")
    out = str(tmp_path / "single.png")
    PR.render_single(ref, None, "cyan", out, "hdr", 100, style=PlotStyle(font_scale=2.0))
    assert os.path.exists(out)


def test_run_per_job_fields_restricts_profiled_fields(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"only_strain","fields":["strain"]}]'
    )
    res = PR.run(
        {"consolidated_h5": str(h5), "mode": "parameter", "jobs_json": jobs, "output_dir": str(out)}
    )
    assert res.jobs[0].fields == ["strain"]  # raw_sum excluded for this job


# -- trace-figure primitives (Task 1) -----------------------------------------
def _fake_field(vid="strain", *, std=True):
    n = 20
    dist = np.linspace(0.0, 10.0, n)
    vm = np.sin(dist)
    vs = np.full(n, 0.1) if std else None
    fld = {
        "vid": vid,
        "attrs": {
            "cbar_label": "c",
            "kind": vid,
            "source_volume": "",
            "title": vid,
            "cmap": "gray",
        },
        "value_mean": vm,
        "value_std": vs,
    }
    geom = {"distance": dist, "L": 10.0}
    return fld, geom


def test_parse_aspect_valid():
    assert PR.parse_aspect("4:3") == (4.0, 3.0)
    assert PR.parse_aspect("1:1") == (1.0, 1.0)
    assert PR.parse_aspect("1.5:1") == (1.5, 1.0)


@pytest.mark.parametrize("bad", ["", "4", "4:0", "a:b", "1:2:3", "-4:3"])
def test_parse_aspect_invalid_raises(bad):
    with pytest.raises(StageUserError):
        PR.parse_aspect(bad)


def test_build_trace_figure_aspect_linewidth_color():
    from matplotlib.colors import to_rgba

    fld, geom = _fake_field(std=True)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(2.0, 1.0), width_in=6.0, linewidth=3.5, color="red", font_scale=1.0
    )
    w, h = fig.get_size_inches()
    assert abs(w - 6.0) < 1e-6 and abs(h - 3.0) < 1e-6
    line = fig.axes[0].lines[0]
    assert abs(line.get_linewidth() - 3.5) < 1e-9
    assert to_rgba(line.get_color()) == to_rgba("red")


def test_build_trace_figure_blank_color_defaults_c0():
    from matplotlib.colors import to_rgba

    fld, geom = _fake_field(std=True)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="", font_scale=1.0
    )
    line = fig.axes[0].lines[0]
    assert to_rgba(line.get_color()) == to_rgba("C0")


def test_build_trace_figure_no_std_no_band():
    fld, geom = _fake_field(std=False)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="", font_scale=1.0
    )
    assert len(fig.axes[0].collections) == 0  # no fill_between std band


def test_build_trace_figure_scales_offset_text():
    # the scientific ×10ⁿ offset text must scale with font_scale like the rest
    fld, geom = _fake_field(std=True)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="", font_scale=2.0
    )
    ax = fig.axes[0]
    assert ax.yaxis.get_offset_text().get_fontsize() == 20.0
    assert ax.xaxis.get_offset_text().get_fontsize() == 20.0


def test_build_trace_figure_show_title_false_omits_title():
    from dfxm.common.plotting import PlotStyle

    fld, geom = _fake_field(std=True)
    kwargs = dict(aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="", font_scale=1.0)
    off = PR.build_trace_figure(fld, geom, style=PlotStyle(show_title=False), **kwargs)
    assert off.axes[0].get_title(loc="left") == ""
    on = PR.build_trace_figure(fld, geom, style=PlotStyle(show_title=True), **kwargs)
    assert on.axes[0].get_title(loc="left") != ""
    legacy = PR.build_trace_figure(fld, geom, style=None, **kwargs)  # unstyled keeps the title
    assert legacy.axes[0].get_title(loc="left") != ""


def test_build_trace_figure_title_scale_scales_title_only():
    from matplotlib import text as mtext

    from dfxm.common.plotting import PlotStyle

    fld, geom = _fake_field(std=True)
    kwargs = dict(aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="", font_scale=2.0)
    fig = PR.build_trace_figure(fld, geom, style=PlotStyle(title_scale=1.5), **kwargs)
    ax = fig.axes[0]
    label = ax.get_title(loc="left")
    ttl = next(t for t in ax.findobj(mtext.Text) if t.get_text() == label and t is not ax.title)
    assert ttl.get_fontsize() == 10 * 2.0 * 1.5  # title follows title_scale
    assert ax.xaxis.label.get_fontsize() == 12 * 2.0  # labels don't


def test_build_trace_figure_pins_plot_box_aspect():
    # trace_aspect pins the PLOT BOX (data rectangle) to exactly W:H via
    # set_box_aspect(h/w), independent of the label/title/font margins.
    fld, geom = _fake_field(std=True)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="", font_scale=3.0
    )
    assert abs(fig.axes[0].get_box_aspect() - 3.0 / 4.0) < 1e-9  # height/width
    sq = PR.build_trace_figure(
        fld, geom, aspect_wh=(1.0, 1.0), width_in=5.0, linewidth=2.0, color="", font_scale=1.0
    )
    assert abs(sq.axes[0].get_box_aspect() - 1.0) < 1e-9


# -- run() trace/companion toggles (Task 2) -----------------------------------
def _base_params(h5, out, **extra):
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"prof0"}]'
    )
    return {
        "consolidated_h5": str(h5),
        "mode": "parameter",
        "jobs_json": jobs,
        "output_dir": str(out),
        **extra,
    }


def test_run_writes_traces_by_default(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out))
    jr = res.jobs[0]
    assert jr.figure and os.path.exists(jr.figure)  # companion on by default
    assert len(jr.traces) == 2 and all(os.path.exists(t) for t in jr.traces)
    assert all("__trace__" in t for t in jr.traces)


def test_run_companion_off_yields_no_companion(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out, save_companion=False))
    jr = res.jobs[0]
    assert jr.figure is None
    assert not os.path.exists(os.path.join(str(out), "prof0.png"))
    assert len(jr.traces) == 2


def test_run_traces_off_keeps_companion(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out, save_traces=False))
    jr = res.jobs[0]
    assert jr.figure and os.path.exists(jr.figure)
    assert jr.traces == []


def test_run_bad_aspect_raises(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    with pytest.raises(StageUserError):
        PR.run(_base_params(h5, out, trace_aspect="oops"))


@pytest.mark.parametrize(
    "bad", [{"trace_width_in": -6.0}, {"trace_linewidth": 0.0}, {"trace_font_scale": -1.0}]
)
def test_run_bad_trace_dimensions_raise(tmp_path, bad):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    with pytest.raises(StageUserError):
        PR.run(_base_params(h5, out, **bad))


def _png_size(path):
    import matplotlib.image as mpimg

    img = mpimg.imread(path)
    return img.shape[1], img.shape[0]  # (w_px, h_px)


def _write_consolidated_mixed_scale(path):
    """Like `_write_consolidated`, but the two fields differ in magnitude by
    ~1e8: "raw_sum" is O(100) (plain tick labels, no scientific offset text),
    "strain" is O(1e-6) (triggers a scientific offset annotation, which adds
    TOP margin). Own-margin placement (no sharing across the run) therefore
    gives every "strain" trace a taller canvas than every "raw_sum" trace —
    a real discriminator for the uniform-margins flush. The original
    same-magnitude fixture cannot discriminate: both fields' natural margins
    coincide by construction, so per-figure and shared-max placement produce
    the same heights either way (empirically confirmed — see task-3 report).
    """
    u = np.linspace(-10.0, 10.0, 81)
    v = np.linspace(-8.0, 8.0, 65)
    uu, vv = np.meshgrid(u, v)
    offsets = np.array([-1.0, 0.0, 1.0])
    with h5py.File(path, "w") as f:
        for vid, kind, cmap, scale, base in (
            ("raw_sum", "raw_sum", "gray", 1.0, 100.0),
            ("strain", "strain", "RdBu_r", 1e-6, 0.0),
        ):
            g = f.create_group(vid)
            g.attrs["kind"] = kind
            g.attrs["cbar_label"] = "value"
            g.attrs["cmap"] = cmap
            g.attrs["title"] = vid
            g.attrs["vmin"] = -10.0
            g.attrs["vmax"] = 10.0
            sg = g.create_group("oblique_full")
            stack = np.stack(
                [scale * (A * uu + B * vv + o) + base for o in offsets], axis=0
            ).astype(np.float32)
            sg.create_dataset("slices", data=stack)
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offsets)


def _own_margin_flush(deferred, dpi, notes):
    """Test-only stand-in for `_flush_deferred_traces` that places each
    figure at its OWN measured margins instead of the shared max — simulates
    the bug the uniform-margins flush fixes, to prove the test below actually
    discriminates (rather than passing by fixture coincidence)."""
    import os as _os

    from dfxm.common.plotting import apply_axes_margins, box_drift_note, measure_axes_margins

    if not deferred:
        return
    for fig, w_in, h_in, png in deferred:
        m = measure_axes_margins(fig, fig.axes[0])
        apply_axes_margins(fig, fig.axes[0], w_in, h_in, m)
        note = box_drift_note(_os.path.basename(png), fig, fig.axes[0], w_in, h_in)
        if note:
            notes.append(note)
        fig.savefig(png, dpi=dpi, facecolor="white", edgecolor="none")


def test_run_fixed_scale_traces_share_height_and_margins(tmp_path, monkeypatch):
    h5 = tmp_path / "c.h5"
    _write_consolidated_mixed_scale(str(h5))
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"jobA"},'
        '{"name":"oblique_full","offset_um":0.0,"start_uv":[-2,-1],"end_uv":[2,1],'
        '"n_samples":40,"width_pixels":1,"fig_name":"jobB"}]'
    )
    style = {"trace_scale_um_per_cm": 2.0, "trace_height_cm": 3.0}

    # Discrimination check (verification contract, item 1): with the fields'
    # natural margins forced apart, an own-margins-only flush must NOT
    # satisfy "same height" — proving this fixture (unlike the same-scale
    # original) actually exercises the sharing behaviour.
    monkeypatch.setattr(PR, "_flush_deferred_traces", _own_margin_flush)
    bug_out = tmp_path / "prof_bug"
    bug_res = PR.run(_base_params(h5, bug_out, jobs_json=jobs, plot_style=style))
    bug_heights = {h for _, h in (_png_size(t) for jr in bug_res.jobs for t in jr.traces)}
    assert len(bug_heights) > 1, bug_heights  # own-margins-only: heights DO differ
    monkeypatch.undo()

    # Real code: the shared-max flush makes every trace PNG of the run the
    # same height.
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out, jobs_json=jobs, plot_style=style))
    assert len(res.jobs) == 2
    sizes = [_png_size(t) for jr in res.jobs for t in jr.traces]
    heights = {h for _, h in sizes}
    assert len(heights) == 1, sizes  # every trace PNG of the run: same pixel height
    # widths track line length: jobA line is ~2.5x jobB's
    wA = _png_size(res.jobs[0].traces[0])[0]
    wB = _png_size(res.jobs[1].traces[0])[0]
    assert wA > wB
    # and no drift notes were emitted
    assert not any("physical scale is off" in n for n in res.notes)


def test_run_fixed_scale_clamp_appends_note(tmp_path):
    h5 = tmp_path / "c.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    res = PR.run(
        _base_params(
            h5,
            out,
            plot_style={"trace_scale_um_per_cm": 0.001, "trace_height_cm": 3.0},
        )
    )
    assert any("clamped to 30 in" in n for n in res.notes), res.notes


def _trace_box_inches(fig):
    """Draw *fig* on Agg and return the axes-box (w_in, h_in)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)
    fig.canvas.draw()
    bbox = fig.axes[0].get_window_extent()
    return bbox.width / fig.dpi, bbox.height / fig.dpi


def test_build_trace_figure_fixed_scale_box_is_length_by_height():
    # fixed-scale mode: box width = L/scale, box height = trace_height_cm —
    # trace_aspect no longer shapes the box (it only governs the legacy mode).
    from dfxm.common.plotting import PlotStyle

    fld, geom = _fake_field(std=True)  # L = 10 um
    st = PlotStyle(trace_scale_um_per_cm=2.0, trace_height_cm=3.0)
    fig = PR.build_trace_figure(
        fld,
        geom,
        aspect_wh=(2.0, 1.0),
        width_in=6.0,
        linewidth=2.0,
        color="",
        font_scale=1.0,
        style=st,
    )
    w_in, h_in = _trace_box_inches(fig)
    assert abs(w_in - 10.0 / 2.0 / 2.54) < 0.005 * w_in
    assert abs(h_in - 3.0 / 2.54) < 0.005 * h_in
    assert fig.axes[0].get_box_aspect() is None  # no aspect pin in fixed mode


def test_build_trace_figure_fixed_scale_ignores_width_in():
    from dfxm.common.plotting import PlotStyle

    fld, geom = _fake_field()
    widths = []
    for width_in in (4.0, 9.0):
        fig = PR.build_trace_figure(
            fld,
            geom,
            aspect_wh=(4.0, 3.0),
            width_in=width_in,
            linewidth=1.5,
            color="",
            font_scale=1.0,
            style=PlotStyle(scale_um_per_cm=2.0),
        )
        widths.append(_trace_box_inches(fig)[0])
    assert abs(widths[0] - widths[1]) < 0.03  # width_in is a no-op in fixed mode


def test_build_trace_figure_fixed_scale_exact_for_short_line_real_repro():
    # regression: L=29.668647 at 10 um/cm rendered at ~5.7 um/cm on real data
    # (set_box_aspect defeated fit_axes_to_box, which silently kept the miss).
    from dfxm.common.plotting import PlotStyle

    n = 200
    dist = np.linspace(0.0, 29.668647, n)
    fld = {
        "vid": "mosa_com_mu",
        "attrs": {
            "cbar_label": "COM mu (deg)",
            "kind": "mosa_com",
            "source_volume": "aligned_raw_mosa_volumes.h5",
            "title": "t",
            "cmap": "viridis",
        },
        "value_mean": np.sin(dist / 5.0) * 1e-4,
        "value_std": None,
    }
    geom = {"distance": dist, "L": 29.668647}
    for fs in (1.0, 1.4, 2.0):
        st = PlotStyle(trace_scale_um_per_cm=10.0, trace_height_cm=3.0)
        fig = PR.build_trace_figure(
            fld,
            geom,
            aspect_wh=(4.0, 3.0),
            width_in=2.0,
            linewidth=2.0,
            color="",
            font_scale=fs,
            style=st,
        )
        w_in, _ = _trace_box_inches(fig)
        implied = 29.668647 / (w_in * 2.54)
        assert abs(implied - 10.0) < 0.05, (fs, implied)


def test_build_trace_figure_fixed_scale_clamps_width_only():
    from dfxm.common.plotting import PlotStyle

    fld, geom = _fake_field(std=True)
    geom = {**geom, "L": 10.0}
    st = PlotStyle(trace_scale_um_per_cm=0.01, trace_height_cm=3.0)  # 10um/0.01 -> 393 in
    fig = PR.build_trace_figure(
        fld,
        geom,
        aspect_wh=(4.0, 3.0),
        width_in=6.0,
        linewidth=2.0,
        color="",
        font_scale=1.0,
        style=st,
    )
    w_in, h_in = _trace_box_inches(fig)
    assert w_in <= 30.0 + 0.2  # clamped width
    assert abs(h_in - 3.0 / 2.54) < 0.02  # height keeps trace_height_cm


def test_build_trace_figure_trace_scale_overrides_map_scale():
    # trace_scale_um_per_cm wins over scale_um_per_cm for the trace box width;
    # a smaller µm/cm value prints the same line physically larger.
    from dfxm.common.plotting import PlotStyle

    fld, geom = _fake_field()  # geom["L"] == 10.0
    fig = PR.build_trace_figure(
        fld,
        geom,
        aspect_wh=(2.0, 1.0),
        width_in=6.0,
        linewidth=1.5,
        color="",
        font_scale=1.0,
        style=PlotStyle(scale_um_per_cm=2.0, trace_scale_um_per_cm=1.0),
    )
    w_in, _ = _trace_box_inches(fig)
    assert abs(w_in - (10.0 / 1.0) / 2.54) < 0.03  # 10 cm, from the TRACE scale


def test_run_tolerates_stale_trace_file_aspect_param(tmp_path):
    # trace_file_aspect was removed; persisted forms may still carry it — the
    # stray key must ride along ignored, not crash the run.
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    res = PR.run(_base_params(h5, tmp_path / "o", trace_file_aspect="1:1"))
    assert len(res.jobs) == 1 and res.jobs[0].traces


# -- two jobs on the same slice -----------------------------------------------
def test_run_same_name_jobs_write_distinct_outputs(tmp_path):
    """Two jobs on one slice at the same offset must not overwrite each other."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1},'
        '{"name":"oblique_full","offset_um":0.0,"start_uv":[-2,0],"end_uv":[2,0],'
        '"n_samples":40,"width_pixels":1}]'
    )
    res = PR.run(_base_params(h5, out, jobs_json=jobs))
    assert len(res.jobs) == 2
    assert [jr.job_index for jr in res.jobs] == [0, 1]
    figs = [jr.figure for jr in res.jobs]
    assert figs[0] != figs[1]
    assert all(f and os.path.exists(f) for f in figs)
    all_traces = res.jobs[0].traces + res.jobs[1].traces
    assert len(set(all_traces)) == len(all_traces)  # no trace overwrote another


def test_run_preview_same_name_jobs_distinct_pngs(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3]},'
        '{"name":"oblique_full","offset_um":1.0,"start_uv":[-2,0],"end_uv":[2,0]}]'
    )
    res = PR.run(_base_params(h5, tmp_path / "prev", mode="preview", jobs_json=jobs))
    figs = [jr.figure for jr in res.jobs]
    assert len(figs) == 2 and figs[0] != figs[1]
    assert all(os.path.exists(f) for f in figs)


# -- per-field drop on grid mismatch ------------------------------------------
def _add_mismatched_field(path, vid="raw_mosa_sum"):
    """Add a field whose slice sits on a slightly different (u, v) grid."""
    u = np.linspace(-10.0, 10.0, 79)
    v = np.linspace(-8.0, 8.0, 63)
    uu, vv = np.meshgrid(u, v)
    offsets = np.array([-1.0, 0.0, 1.0])
    with h5py.File(path, "a") as f:
        g = f.create_group(vid)
        g.attrs["kind"] = "raw_mosa_sum"
        g.attrs["cbar_label"] = "value"
        g.attrs["cmap"] = "gray"
        g.attrs["title"] = vid
        g.attrs["vmin"] = -10.0
        g.attrs["vmax"] = 10.0
        sg = g.create_group("oblique_full")
        stack = np.stack([A * uu + B * vv + o for o in offsets], axis=0).astype(np.float32)
        sg.create_dataset("slices", data=stack)
        sg.create_dataset("u_um", data=u)
        sg.create_dataset("v_um", data=v)
        sg.create_dataset("offsets_um", data=offsets)


def test_run_drops_mismatched_field_keeps_job(tmp_path):
    """A field on a different grid is dropped with a note; the job still runs."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    _add_mismatched_field(str(h5))
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out))
    assert len(res.jobs) == 1
    jr = res.jobs[0]
    assert set(jr.fields) == {"raw_sum", "strain"}  # raw_mosa_sum dropped, rest profiled
    assert jr.figure and os.path.exists(jr.figure)
    assert any("raw_mosa_sum" in n for n in res.notes)  # drop reason surfaced
    assert res.skipped == []  # the job ran — drop notes must not count as skips


def test_run_skips_job_when_no_usable_fields(tmp_path):
    """When every requested field mismatches the reference grid, the job skips."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    _add_mismatched_field(str(h5))
    out = tmp_path / "prof"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fields":["raw_mosa_sum"]}]'
    )
    res = PR.run(_base_params(h5, out, jobs_json=jobs))
    assert res.jobs == []
    assert any("raw_mosa_sum" in s for s in res.skipped)


def test_run_absent_only_fields_job_runs_reference_only(tmp_path):
    """A job whose fields name only absent ids keeps the old behaviour: it runs
    (reference-only companion), with no dangling 'no usable fields:' skip."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fields":["no_such_field"]}]'
    )
    res = PR.run(_base_params(h5, out, jobs_json=jobs))
    assert len(res.jobs) == 1 and res.jobs[0].fields == []
    assert res.jobs[0].figure and os.path.exists(res.jobs[0].figure)
    assert res.skipped == []


def _write_pinned(path):
    """Minimal oblique_slices_pinned.h5 as a pinned slices run writes it: each
    plane is its own single-offset group named {slice}_pin_{offset:+.2f}um."""
    u = np.linspace(-10.0, 10.0, 81)
    v = np.linspace(-8.0, 8.0, 65)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (("raw_sum", "raw_sum", "gray"), ("strain", "strain", "RdBu_r")):
            g = f.create_group(vid)
            g.attrs["kind"] = kind
            g.attrs["cbar_label"] = "value"
            g.attrs["cmap"] = cmap
            g.attrs["title"] = vid
            g.attrs["vmin"] = -10.0
            g.attrs["vmax"] = 10.0
            for off in (-1.0, 1.0):
                sg = g.create_group(f"oblique_full_pin_{off:+.2f}um")
                sg.create_dataset(
                    "slices", data=(A * uu + B * vv + off)[None, ...].astype(np.float32)
                )
                sg.create_dataset("u_um", data=u)
                sg.create_dataset("v_um", data=v)
                sg.create_dataset("offsets_um", data=np.array([off]))


def test_run_resolves_pinned_slice_names(tmp_path):
    """Sweep-era jobs must run against a pinned file: the job's slice name falls
    back to the {name}_pin_* group nearest the job's offset_um (the same
    nearest-plane snap a sweep applies), with the substitution surfaced in notes."""
    h5 = tmp_path / "oblique_slices_pinned.h5"
    _write_pinned(str(h5))
    out = tmp_path / "prof"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.8,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"prof0"}]'
    )
    res = PR.run(_base_params(h5, out, jobs_json=jobs))
    assert len(res.jobs) == 1
    jr = res.jobs[0]
    assert jr.name == "oblique_full_pin_+1.00um"  # nearest pin to offset 0.8
    assert jr.offset_used_um == 1.0
    assert set(jr.fields) == {"raw_sum", "strain"}
    assert jr.figure and os.path.exists(jr.figure)
    assert any("oblique_full_pin_+1.00um" in n for n in res.notes)
    assert res.skipped == []


def test_run_preview_resolves_pinned_slice_names(tmp_path):
    h5 = tmp_path / "oblique_slices_pinned.h5"
    _write_pinned(str(h5))
    jobs = '[{"name":"oblique_full","offset_um":-0.7,"start_uv":[-5,-3],"end_uv":[5,3]}]'
    res = PR.run(_base_params(h5, tmp_path / "prev", jobs_json=jobs, mode="preview"))
    assert len(res.jobs) == 1
    assert res.jobs[0].name == "oblique_full_pin_-1.00um"
    assert os.path.exists(res.jobs[0].figure)


def test_run_exact_slice_name_wins_over_pins(tmp_path):
    """When the plain slice group exists, pinned groups must not hijack the job."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    with h5py.File(str(h5), "a") as f:
        for vid in ("raw_sum", "strain"):
            src = f[f"{vid}/oblique_full"]
            f.copy(src, f[vid], name="oblique_full_pin_+1.00um")
    res = PR.run(_base_params(h5, tmp_path / "prof"))
    assert len(res.jobs) == 1 and res.jobs[0].name == "oblique_full"
    assert res.notes == []


def test_run_unknown_slice_still_skips_with_pins_absent(tmp_path):
    h5 = tmp_path / "oblique_slices_pinned.h5"
    _write_pinned(str(h5))
    jobs = '[{"name":"no_such_slice","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3]}]'
    res = PR.run(_base_params(h5, tmp_path / "prof", jobs_json=jobs))
    assert res.jobs == [] and any("no_such_slice" in s for s in res.skipped)


def test_run_drops_field_with_shorter_sweep(tmp_path):
    """A field whose slice stores fewer planes than the reference is dropped
    with a note — never an uncaught IndexError."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    u = np.linspace(-10.0, 10.0, 81)
    v = np.linspace(-8.0, 8.0, 65)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(str(h5), "a") as f:
        g = f.create_group("raw_short")
        g.attrs["kind"] = "raw_sum"
        g.attrs["cbar_label"] = "value"
        g.attrs["cmap"] = "gray"
        g.attrs["title"] = "raw_short"
        g.attrs["vmin"] = -10.0
        g.attrs["vmax"] = 10.0
        sg = g.create_group("oblique_full")
        sg.create_dataset("slices", data=(A * uu + B * vv)[None, ...].astype(np.float32))
        sg.create_dataset("u_um", data=u)
        sg.create_dataset("v_um", data=v)
        sg.create_dataset("offsets_um", data=np.array([-1.0]))  # 1 plane vs ref's 3
    out = tmp_path / "prof"
    res = PR.run(_base_params(h5, out))  # ref offset 0.0 -> ref idx 1 >= len(short)
    assert len(res.jobs) == 1
    assert "raw_short" not in res.jobs[0].fields
    assert any("raw_short" in n and "sweep shorter" in n for n in res.notes)


def test_unique_name_registers_generated_suffixes():
    """A user-supplied name equal to a generated one must not collide."""
    used: dict[str, int] = {}
    assert PR._unique_name(used, "line") == "line"
    assert PR._unique_name(used, "line") == "line_2"
    assert PR._unique_name(used, "line_2") == "line_2_2"  # not a second 'line_2'
    assert PR._unique_name(used, "line") == "line_3"


# -- companion-map scale bar --------------------------------------------------
def _companion_inputs():
    """Minimal (ref, fields, geom) for calling build_companion_figure directly."""
    u, v, plane = _linear_plane()
    geom = PR.line_geometry(u, v, (-5.0, -3.0), (5.0, 3.0), 40, 1, PR.grid_pitch(u, v))
    vm, vs, _ = PR.profile_plane(plane, geom)
    attrs = {
        "cmap": "gray",
        "cbar_label": "value",
        "title": "t",
        "vmin": -20.0,
        "vmax": 20.0,
        "kind": "raw_sum",
        "source_volume": "vol.h5",
    }
    ref = (plane, u, v, attrs, "lbl")
    fields = [{"vid": "raw_sum", "attrs": attrs, "value_mean": vm, "value_std": vs}]
    return ref, fields, geom


def _offsetbox_artists(ax):
    from matplotlib.offsetbox import AnchoredOffsetbox

    return [a for a in ax.artists if isinstance(a, AnchoredOffsetbox)]


def _offsetbox_children(artist):
    out, stack = [], [artist]
    while stack:
        a = stack.pop()
        out.append(a)
        if hasattr(a, "get_children"):
            stack.extend(a.get_children())
    return out


def test_companion_map_styled_scale_bar_honours_style():
    """Styled companion maps must draw the shared styled scale bar (exact length),
    not the hard-coded legacy one."""
    from matplotlib.patches import Rectangle

    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_inputs()
    style = PlotStyle(scale_bar_length_um=5.0)
    fig = PR.build_companion_figure(ref, fields, geom, "cyan", style=style)
    ax_img = fig.axes[0]
    boxes = _offsetbox_artists(ax_img)
    assert len(boxes) == 1  # styled offsetbox bar present
    bar = next(p for p in _offsetbox_children(boxes[0]) if isinstance(p, Rectangle))
    assert bar.get_width() == 5.0  # the requested length, not the legacy auto length
    assert len(ax_img.patches) == 0  # legacy hand-drawn bar is gone


def test_companion_map_scale_bar_off_draws_no_bar():
    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_inputs()
    fig = PR.build_companion_figure(ref, fields, geom, "cyan", style=PlotStyle(scale_bar=False))
    ax_img = fig.axes[0]
    assert not _offsetbox_artists(ax_img)
    assert len(ax_img.patches) == 0


def test_companion_map_legacy_scale_bar_unchanged():
    """style=None keeps the legacy hand-drawn bar (Rectangle in ax.patches)."""
    from matplotlib.patches import Rectangle

    ref, fields, geom = _companion_inputs()
    fig = PR.build_companion_figure(ref, fields, geom, "cyan", style=None)
    ax_img = fig.axes[0]
    assert any(isinstance(p, Rectangle) for p in ax_img.patches)
    assert not _offsetbox_artists(ax_img)


# -- fixed-scale overview vs. pinned companion --------------------------------
def test_render_single_overview_fits_fixed_scale(tmp_path, monkeypatch):
    """The standalone overview map is fitted to the fixed physical scale.

    A PNG-existence check alone can't fail if the fit call is later deleted, so
    this spies on profiles.fit_axes_to_box (patched where profiles.py looks it
    up, i.e. dfxm.stages.profiles.fit_axes_to_box) to assert it is called
    exactly once with the expected target box when the knob is set, and not at
    all when it's unset — pinning the companion-path fidelity from the run side
    too.
    """
    from dfxm.common.plotting import PlotStyle

    ref, _fields, geom = _companion_inputs()
    u_um, v_um = ref[1], ref[2]
    ext_u = float(u_um[-1] - u_um[0])
    ext_v = float(v_um[-1] - v_um[0])
    scale = 50.0
    expected_w = ext_u / scale / 2.54
    expected_h = ext_v / scale / 2.54

    calls = []
    real_fit = PR.fit_axes_to_box

    def spy(fig, ax, w_in, h_in, *args, **kwargs):
        calls.append((w_in, h_in))
        return real_fit(fig, ax, w_in, h_in, *args, **kwargs)

    monkeypatch.setattr(PR, "fit_axes_to_box", spy)

    style = PlotStyle(scale_um_per_cm=scale)
    out = str(tmp_path / "ov.png")
    PR.render_single(ref, geom, "red", out, "hdr", 100, style=style)
    assert os.path.exists(out)
    assert len(calls) == 1
    w_in, h_in = calls[0]
    assert w_in == pytest.approx(expected_w, abs=1e-6)
    assert h_in == pytest.approx(expected_h, abs=1e-6)

    # knob off: fit_axes_to_box must NOT be called (legacy path stays unfitted)
    calls.clear()
    out2 = str(tmp_path / "ov_unfit.png")
    PR.render_single(ref, geom, "red", out2, "hdr", 100, style=PlotStyle())
    assert os.path.exists(out2)
    assert calls == []


def test_render_single_appends_drift_note_on_forced_miss(tmp_path, monkeypatch):
    """render_single's overview drift guard: when fit_axes_to_box fails to place
    the axes at the target box (forced here via monkeypatch to simulate a miss),
    box_drift_note must catch the discrepancy and append a user-visible note."""
    from dfxm.common.plotting import PlotStyle

    # force fit_axes_to_box to do nothing so the guard must catch the miss
    monkeypatch.setattr(PR, "fit_axes_to_box", lambda *a, **k: False)
    ref, _fields, geom = _companion_inputs()
    notes = []
    PR.render_single(
        ref,
        geom,
        "white",
        str(tmp_path / "ov.png"),
        "hdr",
        100,
        style=PlotStyle(scale_um_per_cm=20.0),
        notes=notes,
    )
    assert notes and "physical scale is off" in notes[0]


def test_companion_map_panel_bar_geometry_unchanged_without_scale_knob():
    """Without a fixed scale, the multi-panel companion is NOT fitted: the map
    panel's scale-bar keeps today's data-fraction thickness (the legacy path,
    :func:`_build_companion_legacy`, never forwards fixed_scale_um_per_cm to
    _draw_reference_image)."""
    from matplotlib.patches import Rectangle

    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_inputs()
    style = PlotStyle(scale_bar_thickness_pt=3.0)  # no scale_um_per_cm -> legacy path
    fig = PR.build_companion_figure(ref, fields, geom, "cyan", style=style)
    ax_img = fig.axes[0]
    yr = ax_img.get_ylim()[1] - ax_img.get_ylim()[0]
    box = _offsetbox_artists(ax_img)[0]
    bar = next(p for p in _offsetbox_children(box) if isinstance(p, Rectangle))
    # companion is NOT fitted: bar height stays the data-fraction geometry
    assert bar.get_height() == pytest.approx(abs(yr) * 0.004 * 3.0)


def test_companion_map_panel_bar_geometry_matches_fixed_scale():
    """With a fixed scale, the companion routes to the deterministic stack
    (:func:`_build_companion_fixed`) and the map panel's scale bar uses the
    fixed-scale point-based thickness — the same geometry the standalone map
    figures use (e.g. render_single) — not the data-fraction fallback."""
    from matplotlib.patches import Rectangle

    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_inputs()
    style = PlotStyle(scale_um_per_cm=50.0, scale_bar_thickness_pt=3.0)
    fig = PR.build_companion_figure(ref, fields, geom, "cyan", style=style)
    ax_img = fig.axes[0]
    box = _offsetbox_artists(ax_img)[0]
    bar = next(p for p in _offsetbox_children(box) if isinstance(p, Rectangle))
    expected = 3.0 * (2.54 / 72.0) * 50.0  # thickness_pt in TRUE points at 50 um/cm
    assert bar.get_height() == pytest.approx(expected)


# -- companion on the deterministic stack layout (fixed scale) ----------------
def test_companion_fixed_scale_panel_boxes_and_trace_style():
    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_inputs()
    st = PlotStyle(scale_um_per_cm=20.0, trace_scale_um_per_cm=2.0, trace_height_cm=3.0)
    topts = {"linewidth": 2.5, "color": "red", "font_scale": 1.4}
    fig = PR.build_companion_figure(ref, fields, geom, "white", style=st, trace_opts=topts)
    from dfxm.common.plotting import measured_box_in

    # trace axes carry the plotted lines; the manual colorbar axes has none
    ax_map, ax_traces = fig.axes[0], [a for a in fig.axes[1:] if a.lines]
    u, v = ref[1], ref[2]
    ext_u, ext_v = float(u[-1] - u[0]), float(v[-1] - v[0])
    mw, mh = measured_box_in(fig, ax_map)
    assert abs(mw - ext_u / 20.0 / 2.54) < 0.01 * max(1.0, mw)
    assert abs(mh - ext_v / 20.0 / 2.54) < 0.01 * max(1.0, mh)
    for ax in ax_traces:
        tw, th = measured_box_in(fig, ax)
        assert abs(tw - geom["L"] / 2.0 / 2.54) < 0.01 * tw
        assert abs(th - 3.0 / 2.54) < 0.01 * th
        assert abs(ax.lines[0].get_linewidth() - 2.5) < 1e-9  # trace_opts, not 1.8
        assert ax.yaxis.label.get_fontsize() == 10 * 1.4  # trace font scale, not map


def test_companion_fixed_scale_show_title_false_no_panel_titles():
    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_inputs()
    st = PlotStyle(
        scale_um_per_cm=20.0, trace_scale_um_per_cm=2.0, trace_height_cm=3.0, show_title=False
    )
    fig = PR.build_companion_figure(ref, fields, geom, "white", style=st)
    for ax in fig.axes:
        assert ax.get_title() == "" and ax.get_title(loc="left") == ""


def test_companion_without_fixed_scale_keeps_legacy_layout():
    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_inputs()
    fig_none = PR.build_companion_figure(ref, fields, geom, "white", style=None)
    w, h = fig_none.get_size_inches()
    assert abs(w - 9.0) < 1e-6  # legacy canvas untouched
    fig_styled = PR.build_companion_figure(ref, fields, geom, "white", style=PlotStyle())
    w2, _ = fig_styled.get_size_inches()
    assert abs(w2 - 9.0) < 1e-6  # styled-but-no-scale also legacy


def test_companion_fixed_scale_degenerate_map_extent_falls_back_to_legacy():
    """A fixed TRACE scale is set (so the dispatcher enters the fixed-scale
    path) but the reference plane's own U extent is degenerate (zero-width —
    a plausible pinned edge-of-ROI plane), so fixed_scale_box can't fit a
    physical map box. Must degrade to the legacy layout, never raise."""
    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_inputs()
    plane, u, v, attrs, label = ref
    degenerate_ref = (plane, np.array([u[0]]), v, attrs, label)  # ext_u == 0
    st = PlotStyle(scale_um_per_cm=20.0, trace_scale_um_per_cm=2.0, trace_height_cm=3.0)
    fig = PR.build_companion_figure(degenerate_ref, fields, geom, "white", style=st)
    w, h = fig.get_size_inches()
    assert abs(w - 9.0) < 1e-6  # legacy canvas, not a crash
    assert abs(h - (4.8 + 1.85 * len(fields))) < 1e-6  # legacy height formula


def test_clim_attrs_field_id_beats_group_fallback():
    attrs = {"kind": "strain", "vmin": -10.0, "vmax": 10.0}
    out = PR._clim_attrs(dict(attrs), "strain", {"strain": (-1.0, 1.0)})
    assert (out["vmin"], out["vmax"]) == (-1.0, 1.0)
    # group fallback: vid not in mapping, kind's colormap group is
    attrs2 = {"kind": "raw_sum", "vmin": 0.0, "vmax": 100.0}
    out2 = PR._clim_attrs(dict(attrs2), "raw_sum", {"raw": (5.0, None)})
    assert (out2["vmin"], out2["vmax"]) == (5.0, 100.0)  # half-open keeps stored vmax
    # vid key wins over group key when both present
    out3 = PR._clim_attrs(dict(attrs2), "raw_sum", {"raw": (5.0, 50.0), "raw_sum": (7.0, 70.0)})
    assert (out3["vmin"], out3["vmax"]) == (7.0, 70.0)
    # no matching key / clim None -> untouched
    out4 = PR._clim_attrs(dict(attrs), "strain", {"mosa_com": (0.0, 1.0)})
    assert (out4["vmin"], out4["vmax"]) == (-10.0, 10.0)
    assert PR._clim_attrs(dict(attrs), "strain", None)["vmin"] == -10.0


def test_collect_applies_clim_to_ref_and_fields(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    job = {"name": "oblique_full", "offset_um": 0.0, "start_uv": [-5, -3], "end_uv": [5, 3]}
    p = PR.STAGE.defaults()
    with h5py.File(str(h5), "r") as f:
        ref, fields, _geom, _off, _dropped = PR._collect(
            f, job, p, "", None, clim={"strain": (-2.0, 2.0)}
        )
    by_vid = {fl["vid"]: fl["attrs"] for fl in fields}
    assert (by_vid["strain"]["vmin"], by_vid["strain"]["vmax"]) == (-2.0, 2.0)
    assert (by_vid["raw_sum"]["vmin"], by_vid["raw_sum"]["vmax"]) == (-10.0, 10.0)  # stored
    assert (ref[3]["vmin"], ref[3]["vmax"]) == (-10.0, 10.0)  # ref is raw_sum -> stored


def test_render_replot_writes_figures_no_csvs(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "replots"
    jobs = [
        {
            "name": "oblique_full",
            "offset_um": 0.0,
            "start_uv": [-5, -3],
            "end_uv": [5, 3],
            "n_samples": 40,
            "width_pixels": 1,
            "fig_name": "rp0",
        }
    ]
    res = PR.render_replot(str(h5), jobs, None, {"strain": (-2.0, 2.0)}, str(out))
    assert len(res.jobs) == 1
    jr = res.jobs[0]
    assert jr.figure and os.path.exists(jr.figure)  # companion
    assert len(jr.overviews) == 2 and all(os.path.exists(p) for p in jr.overviews)
    assert len(jr.traces) == 2 and all(os.path.exists(p) for p in jr.traces)
    assert jr.csvs == []  # replots never write CSVs
    assert not any(fn.endswith(".csv") for fn in os.listdir(out))


def test_render_replot_resolves_pinned_names(tmp_path):
    h5 = tmp_path / "oblique_slices_pinned.h5"
    _write_pinned(str(h5))
    jobs = [{"name": "oblique_full", "offset_um": 0.8, "start_uv": [-5, -3], "end_uv": [5, 3]}]
    res = PR.render_replot(str(h5), jobs, None, None, str(tmp_path / "rp"))
    assert len(res.jobs) == 1 and res.jobs[0].name == "oblique_full_pin_+1.00um"
    assert any("pinned" in n for n in res.notes)


def test_render_replot_bad_inputs_raise_stageusererror(tmp_path):
    with pytest.raises(PR.StageUserError):
        PR.render_replot(str(tmp_path / "missing.h5"), [{"name": "x"}], None, None, str(tmp_path))
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    with pytest.raises(PR.StageUserError):
        PR.render_replot(str(h5), [], None, None, str(tmp_path))


def test_replot_catalog_lists_jobs_and_fields(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    jobs = [
        {"name": "oblique_full", "offset_um": 0.0, "fig_name": "rp0"},
        {"name": "no_such_slice", "offset_um": 0.0},
    ]
    cat = PR.replot_catalog(str(h5), jobs)
    assert len(cat) == 1  # jobs whose slice is absent (plain or pinned) are omitted
    e = cat[0]
    assert e.job_index == 0 and e.name == "oblique_full"
    assert e.fields == ["raw_sum", "strain"]
    assert "rp0" in e.label and e.note is None


# -- F1: render_replot honours the form's appearance params -------------------
def _replot_job():
    return [
        {
            "name": "oblique_full",
            "offset_um": 0.0,
            "start_uv": [-5, -3],
            "end_uv": [5, 3],
            "n_samples": 40,
            "width_pixels": 1,
            "fig_name": "rp0",
        }
    ]


def test_render_replot_default_params_write_overviews(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    res = PR.render_replot(str(h5), _replot_job(), None, None, str(tmp_path / "rp"))
    assert len(res.jobs) == 1
    assert res.jobs[0].overviews  # save_overview defaults True via STAGE.defaults()


def test_render_replot_params_save_overview_false_suppresses_overviews(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    res = PR.render_replot(
        str(h5),
        _replot_job(),
        None,
        None,
        str(tmp_path / "rp"),
        params={"save_overview": False},
    )
    assert len(res.jobs) == 1
    assert res.jobs[0].overviews == []


def test_render_replot_params_reference_override_changes_field_order(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    res = PR.render_replot(
        str(h5),
        _replot_job(),
        None,
        None,
        str(tmp_path / "rp"),
        params={"reference_volume_id": "strain"},
    )
    assert len(res.jobs) == 1
    assert res.jobs[0].fields[0] == "strain"


def test_render_replot_params_fig_dpi_used_when_no_explicit_dpi(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "rp"
    res = PR.render_replot(str(h5), _replot_job(), None, None, str(out), params={"fig_dpi": 72})
    assert len(res.jobs) == 1 and os.path.exists(res.jobs[0].figure)


def test_render_replot_explicit_dpi_wins_over_params(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "rp"
    # If dpi didn't win, fig_dpi=1 would produce a near-empty/degenerate figure;
    # this just exercises the precedence path without asserting on pixel dims.
    res = PR.render_replot(
        str(h5),
        _replot_job(),
        None,
        None,
        str(out),
        dpi=150,
        params={"fig_dpi": 1},
    )
    assert len(res.jobs) == 1 and os.path.exists(res.jobs[0].figure)


# -- F4: render_replot malformed-job guard ------------------------------------
def test_render_replot_skips_malformed_job_renders_good_one(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    jobs = [{}] + _replot_job()
    res = PR.render_replot(str(h5), jobs, None, None, str(tmp_path / "rp"))
    assert len(res.jobs) == 1
    assert any("malformed job spec" in s for s in res.skipped)
