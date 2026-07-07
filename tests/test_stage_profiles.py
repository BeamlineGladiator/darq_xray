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
