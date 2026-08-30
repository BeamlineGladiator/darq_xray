"""Tests for dfxm.stages.slices — plane geometry/sampling and end-to-end output."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from matplotlib.offsetbox import AnchoredOffsetbox

from dfxm.common.errors import StageUserError
from dfxm.common.plotting import PlotStyle
from dfxm.stages import slices as SL

L, NY, NX = 4, 6, 8


# -- geometry / sampling ------------------------------------------------------
def test_build_basis_orthonormal_right_handed():
    u, v, n = SL.build_basis((0.3, 0.0, 0.95))
    for a in (u, v, n):
        assert np.isclose(np.linalg.norm(a), 1.0)
    assert np.isclose(np.dot(u, v), 0, atol=1e-9)
    assert np.isclose(np.dot(u, n), 0, atol=1e-9)
    np.testing.assert_allclose(np.cross(u, v), n, atol=1e-9)  # right-handed


def test_build_basis_matches_layer_plot_orientation():
    """Plots must read like the per-layer renders: X-like horizontal, Y-like vertical.

    u_hat is the plot's horizontal axis and v_hat its vertical axis, so for a
    z-normal plane (u, v) must be exactly (X, Y), and for a plane tilted in the
    X-Z plane the vertical axis must stay world-Y with the horizontal axis the
    +X-ish in-plane direction — not the 90°-rotated (Y-horizontal) layout.
    """
    u, v, _ = SL.build_basis((0, 0, 1))
    np.testing.assert_allclose(u, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(v, [0, 1, 0], atol=1e-12)

    u, v, _ = SL.build_basis((0.647648, 0, 0.761939))  # default oblique_full normal
    np.testing.assert_allclose(v, [0, 1, 0], atol=1e-9)  # vertical axis = world Y
    assert u[0] > 0.5  # horizontal axis points +X-ish
    assert abs(u[1]) < 1e-9


def test_slice_plane_offsets():
    np.testing.assert_allclose(SL.slice_plane_offsets({"sweep_step_um": None}), [0.0])
    off = SL.slice_plane_offsets(
        {"sweep_step_um": 2.0, "sweep_start_um": 0.0, "sweep_stop_um": 6.0}
    )
    np.testing.assert_allclose(off, [0, 2, 4, 6])


def test_sample_plane_on_index_field():
    """Volume where value == X index -> an XY plane samples value == world X."""
    data = np.broadcast_to(np.arange(NX, dtype=float), (L, NY, NX)).copy()
    prep = {
        "data": np.ascontiguousarray(data),
        "scale_x": 1.0,
        "scale_y": 1.0,
        "scale_z": 1.0,
        "x_ref_shift_px": 0.0,
        "y_ref_shift_px": 0.0,
        "z_ref_shift_um": 0.0,
    }
    u_hat, v_hat, _ = SL.build_basis((0, 0, 1))  # u->X, v->Y
    s, u_um, v_um = SL.sample_plane(prep, (3.0, 2.0, 1.0), u_hat, v_hat, 2.0, 1.0, 1.0, 1.0)
    # world X at column c = origin_x + u_um[c]; value == that X
    for c, u in enumerate(u_um):
        col = s[:, c]
        col = col[np.isfinite(col)]
        if col.size:
            np.testing.assert_allclose(col, 3.0 + u, atol=1e-6)


def test_resolve_auto_extent_fits_box():
    box = (0.0, 10.0, 0.0, 8.0, 0.0, 6.0)
    out = SL.resolve_auto_extent(
        {"name": "z", "normal": [0, 0, 1], "origin": [0, 0, 0], "extent": "auto", "du": 1.0}, box
    )
    assert out["half_u"] > 0 and out["half_v"] > 0
    assert out["sweep_start_um"] <= 0.0 <= out["sweep_stop_um"]
    assert out["sweep_step_um"] == 1.0


def test_resolve_auto_extent_default_step_uses_pixel_scale():
    """No du / no sweep_step_um -> step defaults to the configured pixel scale."""
    box = (0.0, 10.0, 0.0, 8.0, 0.0, 6.0)
    sl = {"name": "z", "normal": [0, 0, 1], "origin": [0, 0, 0], "extent": "auto"}
    out = SL.resolve_auto_extent(sl, box, default_du=0.385)
    assert out["sweep_step_um"] == 0.385


# -- end to end ---------------------------------------------------------------
def _setup(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    rng = np.random.default_rng(0)
    with h5py.File(proc / "stacked_volumes.h5", "w") as f:
        for grp in ("chi", "mu"):
            g = f.create_group(grp)
            g.create_dataset("Center of mass", data=rng.standard_normal((L, NY, NX)))
            g.create_dataset("FWHM", data=np.abs(rng.standard_normal((L, NY, NX))))
    with h5py.File(proc / "stacked_strain_volumes.h5", "w") as f:
        f.create_dataset("strain", data=rng.standard_normal((L, NY, NX)) * 1e-4)
    # an already-aligned rocking volume with stored spacing
    with h5py.File(proc / "aligned_raw_rocking_volumes.h5", "w") as f:
        f.create_dataset("sum_intensity", data=rng.standard_normal((L, NY, NX)).astype(np.float32))
        f.create_dataset("specific_frame", data=rng.standard_normal((L, NY, NX)).astype(np.float32))
        f.create_dataset("z_uniform_um", data=np.arange(L, dtype=np.float32))
        f.attrs["scale_x_um_per_px"] = 0.152
        f.attrs["scale_y_um_per_px"] = 0.385
        f.attrs["scale_z_um_per_px"] = 1.0
        f.attrs["specific_frame_idx"] = 2
    raw = tmp_path / "raw"
    raw.mkdir()
    # samy is SUB-PIXEL on purpose. It used to run 0 → 4 µm against a 0.152 µm
    # pixel, i.e. 0 → 26 px of X-shift on an 8 px-wide volume, which slid the
    # data clean past itself: the aligned volume was **6 finite voxels out of
    # 840** (0.7%). Nothing failed and no assertion was wrong — every colour
    # limit, centring and equality check in this module was simply a statement
    # about NaN, and could not have failed on real data either. That is the same
    # defect Task 10 found in `test_stage_visualize.py::_setup`; found here the
    # same way, by a precondition that turned out to be unsatisfiable.
    # `_assert_mostly_finite` below is what stops it coming back.
    samy = np.array([0.0, 0.00001, 0.000025, 0.00004])
    samz = np.array([0.0, 0.001, 0.0021, 0.0035])
    for base in ("mosa", "strain"):
        for i in range(L):
            folder = raw / f"{base}__{i + 1}"
            folder.mkdir()
            with h5py.File(folder / f"{base}__{i + 1}.h5", "w") as f:
                f.create_dataset("1.1/instrument/positioners/samy", data=samy[i])
                f.create_dataset("1.1/instrument/positioners/samz", data=samz[i])
    _assert_mostly_finite(proc, raw, tmp_path)
    return proc, raw


# A NaN rim is expected (the samy canvas grows and the Z grid is resampled); a
# volume that is *mostly* NaN is a broken fixture, not a hard test case.
_MIN_FINITE_FRACTION = 0.5


def _assert_mostly_finite(proc, raw, tmp_path):
    """The aligned χ CoM volume this fixture produces must be mostly finite.

    Asserted where the fixture is *defined*, not in one test that happens to
    care, because every equality and colour-limit assertion in this module is
    only as strong as the data behind it.
    """
    _fits, aligned, _limits = _prepare_at_budget(
        proc, raw, tmp_path, 1 << 30, "midrange", cfg_kind="mosa_fwhm", abs_fwhm=False
    )
    fraction = float(np.isfinite(aligned).mean()) if aligned.size else 0.0
    assert fraction >= _MIN_FINITE_FRACTION, (
        f"the aligned fixture volume is only {100 * fraction:.1f}% finite "
        f"(shape {aligned.shape}) — the samy shifts have slid it past itself, so "
        "every assertion built on it is a statement about NaN"
    )


def test_run_writes_consolidated_h5_and_pngs(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    # one explicit single plane (controlled shape) + one auto z-sweep
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null},'
        '{"name":"zsweep","normal":[0,0,1],"origin":[0,0,0],"extent":"auto","sweep_step_um":1.0}]'
    )
    res = SL.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "aligned_rocking_file": str(proc / "aligned_raw_rocking_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "slices_json": slices_json,
            "output_dir": str(out),
        }
    )
    assert res.output_h5 and os.path.exists(res.output_h5)
    # five mosa/strain ids + two raw ids
    assert set(res.volume_ids) == {
        "mosa_com_chi",
        "mosa_fwhm_chi",
        "mosa_com_mu",
        "mosa_fwhm_mu",
        "strain",
        "raw_sum",
        "raw_specific_frame2",
    }
    assert set(res.slice_names) == {"mid", "zsweep"}
    with h5py.File(res.output_h5, "r") as f:
        sg = f["strain"]["mid"]
        assert sg["slices"].ndim == 3 and sg["slices"].shape[0] == 1  # single plane
        for key in ("u_um", "v_um", "offsets_um"):
            assert key in sg
        for attr in ("normal", "u_hat", "v_hat", "n_hat", "half_u", "sweep_step_um"):
            assert attr in sg.attrs
        # auto sweep produced several planes
        assert f["strain"]["zsweep"]["slices"].shape[0] >= 2
    assert res.pngs and all(os.path.exists(p) for p in res.pngs)


def test_run_no_volumes_selected(tmp_path):
    res = SL.run({"mosa_volume_file": "", "strain_volume_file": "", "aligned_rocking_file": ""})
    assert any("no input volumes" in s for s in res.skipped)


def test_run_rejects_nonpositive_du(tmp_path):
    proc, raw = _setup(tmp_path)
    bad = '[{"name":"mid","normal":[0,0,1],"origin":[0,0,0],"half_u":1,"half_v":1,"du":0,"dv":0.2}]'
    with pytest.raises(ValueError, match="du must be > 0"):
        SL.run(
            {
                "mosa_volume_file": str(proc / "stacked_volumes.h5"),
                "raw_root": str(raw),
                "mosa_pattern": "mosa__*",
                "slices_json": bad,
                "output_dir": str(tmp_path / "sl"),
            }
        )


def test_run_rejects_missing_half(tmp_path):
    proc, raw = _setup(tmp_path)
    bad = '[{"name":"mid","normal":[0,0,1],"origin":[0,0,0],"du":0.2,"dv":0.2}]'
    with pytest.raises(ValueError, match="half_u and half_v"):
        SL.run(
            {
                "mosa_volume_file": str(proc / "stacked_volumes.h5"),
                "raw_root": str(raw),
                "mosa_pattern": "mosa__*",
                "slices_json": bad,
                "output_dir": str(tmp_path / "sl"),
            }
        )


# -- build_slice_figure -------------------------------------------------------


def _prep():
    return {
        "cmap_name": "viridis",
        "title": "t",
        "cbar_label": "cb",
        "vmin": -1.0,
        "vmax": 1.0,
        "center_zero": False,
    }


def test_build_slice_figure_returns_figure_with_equal_aspect():
    sl = {"name": "p0"}
    s2d = np.random.rand(10, 12)
    fig = SL.build_slice_figure(
        _prep(),
        sl,
        s2d,
        np.linspace(0, 12, 12),
        np.linspace(0, 10, 10),
        offset_um=None,
        style=PlotStyle(scale_bar=False),
    )
    assert fig.axes[0].get_aspect() == 1.0
    assert not any(isinstance(a, AnchoredOffsetbox) for a in fig.axes[0].artists)


def test_build_slice_figure_legacy_figsize_and_colorbar():
    sl = {"name": "p0"}
    s2d = np.random.rand(10, 12)
    fig = SL.build_slice_figure(
        _prep(),
        sl,
        s2d,
        np.linspace(0, 12, 12),
        np.linspace(0, 10, 10),
        offset_um=None,
        style=None,
    )
    # legacy figsize is the hardcoded 12x10
    w, h = fig.get_size_inches()
    assert (round(w), round(h)) == (12, 10)
    # main axes + colourbar axes
    assert len(fig.axes) == 2
    # legacy draws the (black) scale bar -> the AnchoredOffsetbox is present
    assert any(isinstance(a, AnchoredOffsetbox) for a in fig.axes[0].artists)


def test_build_slice_figure_offset_annotation_in_title():
    sl = {"name": "p0"}
    s2d = np.random.rand(10, 12)
    fig = SL.build_slice_figure(
        _prep(),
        sl,
        s2d,
        np.linspace(0, 12, 12),
        np.linspace(0, 10, 10),
        offset_um=3.5,
        style=None,
    )
    title = fig.axes[0].get_title()
    assert "3.50" in title  # the offset annotation appears as "+3.50" in the title


def _box_inches(fig, ax):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    return bb.width / fig.dpi, bb.height / fig.dpi


def test_build_slice_figure_fixed_scale_equal_boxes_across_colorbar_text():
    u = np.linspace(0.0, 200.0, 21)
    v = np.linspace(0.0, 100.0, 11)
    s2d = np.random.default_rng(3).random((11, 21))
    style = PlotStyle(scale_um_per_cm=50.0, figure_width="single", tickfmt_strain="scientific")
    boxes = []
    for vmin, vmax, group in ((-1.0e-4, 1.0e-4, "strain"), (-1.0, 1.0, None)):
        prep = dict(_prep(), vmin=vmin, vmax=vmax, group=group)
        fig = SL.build_slice_figure(prep, {"name": "p"}, s2d, u, v, offset_um=None, style=style)
        boxes.append(_box_inches(fig, fig.axes[0]))
    tw, th = 200.0 / 50.0 / 2.54, 100.0 / 50.0 / 2.54
    for w, h in boxes:
        assert abs(w - tw) <= 0.05 and abs(h - th) <= 0.05


def _minimal_params(proc, raw, out):
    return {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "aligned_rocking_file": str(proc / "aligned_raw_rocking_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "slices_json": (
            '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
            '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
        ),
        "output_dir": str(out),
        "save_png": False,
    }


def test_run_warns_on_mismatched_plane_grids(tmp_path):
    """Volumes with different pixel scales land on different (u, v) grids when
    the slice spec has no explicit du/dv — the run must say so in notes."""
    proc, raw = _setup(tmp_path)
    with h5py.File(proc / "aligned_raw_rocking_volumes.h5", "a") as f:
        f.attrs["scale_x_um_per_px"] = 0.2  # differs from the stage pixel size
        f.attrs["scale_y_um_per_px"] = 0.5
    params = _minimal_params(proc, raw, tmp_path / "sl_warn")
    spec = json.loads(params["slices_json"])
    for sl in spec:
        del sl["du"], sl["dv"]  # per-volume default step -> mismatched grids
    params["slices_json"] = json.dumps(spec)
    res = SL.run(params)
    warn = [n for n in res.notes if "grid" in n]
    assert warn and "mid" in warn[0]
    assert "du" in warn[0]  # points at the explicit du/dv remedy
    assert "raw_sum" in warn[0]


def test_run_matching_grids_no_grid_note(tmp_path):
    """Explicit du/dv puts every volume on one grid — no grid note."""
    proc, raw = _setup(tmp_path)
    res = SL.run(_minimal_params(proc, raw, tmp_path / "sl_nogrid"))
    assert not any("grid" in n for n in res.notes)


def test_slices_never_hands_the_alignment_a_scratch_dir(tmp_path, monkeypatch):
    """No `scratch_dir=`, therefore no disk, therefore `scratch_bytes == 0`.

    This is the run-side half of
    `test_stage_estimates.py::test_slices_never_prices_a_spill_it_cannot_perform`,
    and the reason that zero is the truth rather than a dropped term.
    `prepare_volume` also passes `center_method=None` — the stage centres itself
    afterwards, midrange included — so the alignment never computes a multi-pass
    statistic and would have nothing to cache even if it were handed somewhere
    to put it. An estimator that priced a spill here would let
    `advice.plan_run` BLOCK a run on a full disk that the run would never use.

    Run with `center_method="median"` (the only setting that could cache) at a
    budget small enough to force the blocked rung, where the caching would
    happen if it happened anywhere.
    """
    proc, raw = _setup(tmp_path)
    seen: list = []
    real = SL.A.align_volume_streamed

    def spy(*a, **kw):
        seen.append((kw.get("scratch_dir", "NOT PASSED"), kw.get("center_method", "NOT PASSED")))
        return real(*a, **kw)

    monkeypatch.setattr(SL.A, "align_volume_streamed", spy)
    out = tmp_path / "sl_median"
    SL.run(
        {
            **_minimal_params(proc, raw, out),
            "center_method": "median",
            "_budget_bytes": 1 << 20,
        }
    )
    assert seen, "no volume went through the streaming alignment"
    assert {s for s, _c in seen} == {"NOT PASSED"}, seen
    assert {c for _s, c in seen} == {None}, seen
    assert not list(out.rglob("*scratch*")), "the run left a scratch directory behind"


def _two_volume_slices_params(tmp_path):
    """Synthetic params that select several volumes (both map files + rocking)."""
    proc, raw = _setup(tmp_path)
    return _minimal_params(proc, raw, tmp_path / "sl_lifetime")


def test_slices_releases_previous_volume(tmp_path, monkeypatch):
    """The previous prepared volume is dead before the next one is built."""
    import weakref

    seen: list = []
    real = SL.prepare_volume

    def spy(cfg, p, *args, **kwargs):
        # Every previously prepared volume must already be collectable.
        assert all(ref() is None for ref in seen), "previous volume still alive"
        prep = real(cfg, p, *args, **kwargs)
        seen.append(weakref.ref(prep["data"]))
        return prep

    monkeypatch.setattr(SL, "prepare_volume", spy)
    SL.run(_two_volume_slices_params(tmp_path))
    assert len(seen) >= 2, "test needs at least two volumes to be meaningful"


def test_run_warns_on_mismatched_y_heights(tmp_path):
    """An aligned raw volume built with a different detector-row crop is taller
    than the map volumes — the run must flag the Y misregistration in notes."""
    proc, raw = _setup(tmp_path)
    rng = np.random.default_rng(1)
    with h5py.File(proc / "aligned_raw_rocking_volumes.h5", "a") as f:
        for key in ("sum_intensity", "specific_frame"):
            del f[key]
            f.create_dataset(key, data=rng.standard_normal((L, NY * 2, NX)).astype(np.float32))
        f.attrs["roi_y_start"] = 230
        f.attrs["roi_y_end"] = 230 + NY * 2
    res = SL.run(_minimal_params(proc, raw, tmp_path / "sl_h"))
    warn = [n for n in res.notes if "Y heights differ" in n]
    assert warn and "raw_sum" in warn[0]
    assert "roi_y 230,242" in warn[0]  # the aligned file's recorded detector crop
    assert "origin+size" in warn[0]  # points at the darfix origin/size trap


def test_run_resolves_cmaps_from_injected_style(tmp_path):
    proc, raw = _setup(tmp_path)
    params = _minimal_params(proc, raw, tmp_path / "sl_styled")
    params["plot_style"] = {"cmap_mosa_com": "viridis", "cmap_raw": "bone", "font_scale": 1.0}
    res = SL.run(params)
    with h5py.File(res.output_h5, "r") as f:
        assert f["mosa_com_chi"].attrs["cmap"] == "viridis"
        assert f["raw_sum"].attrs["cmap"] == "bone"
        assert f["strain"].attrs["cmap"] == "RdBu_r"  # default from PlotStyle


def test_run_without_style_uses_group_defaults(tmp_path):
    proc, raw = _setup(tmp_path)
    res = SL.run(_minimal_params(proc, raw, tmp_path / "sl_default"))
    with h5py.File(res.output_h5, "r") as f:
        assert f["mosa_com_chi"].attrs["cmap"] == "fast"  # real fast, no coolwarm fallback
        assert f["raw_sum"].attrs["cmap"] == "gray"
        assert f["mosa_fwhm_chi"].attrs["cmap"] == "magma"


def test_figures_re_resolve_cmap_by_kind(tmp_path):
    from dfxm.common.plotting import PlotStyle

    proc, raw = _setup(tmp_path)
    res = SL.run(_minimal_params(proc, raw, tmp_path / "sl_figs"))
    specs = SL.figures(res, {})
    spec = next(s for s in specs if "mosa_com_chi" in s.figure_id)
    fig = spec.build(PlotStyle(cmap_mosa_com="plasma"))
    assert fig.axes[0].images[0].cmap.name == "plasma"
    assert spec.build(None).axes[0].images[0].cmap.name == "fast"


def test_run_round_clim_rounds_notes_and_h5_attrs(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "slices_json": slices_json,
        "output_dir": str(out),
        "plot_style": {"round_clim": True},
    }
    res = SL.run(params)
    assert res.notes and all("rounded" in n for n in res.notes)
    with h5py.File(res.output_h5, "r") as f:
        for note in res.notes:
            vid = note.split(":")[0]
            vg = f[vid]
            assert "vmin_raw" in vg.attrs and "vmax_raw" in vg.attrs
            # final limits enclose the raw ones (outward rounding never clips)
            assert vg.attrs["vmin"] <= vg.attrs["vmin_raw"]
            assert vg.attrs["vmax"] >= vg.attrs["vmax_raw"]


def test_build_slice_figure_raw_arbitrary_units_drops_ticks():
    import numpy as np

    from dfxm.common.plotting import PlotStyle
    from dfxm.stages.slices import build_slice_figure

    u = np.linspace(0.0, 40.0, 40)
    v = np.linspace(0.0, 30.0, 30)
    data = np.arange(30 * 40, dtype=float).reshape(30, 40)
    prep = {
        "cmap_name": "gray",
        "vmin": 0.0,
        "vmax": float(data.max()),
        "center_zero": False,
        "title": "Sum intensity",
        "cbar_label": "Sum intensity (a.u.)",
        "group": "raw",
    }
    style = PlotStyle(tickfmt_raw="arb")
    fig = build_slice_figure(prep, {"name": "s"}, data, u, v, offset_um=None, style=style)
    cbar_ax = fig.axes[1]
    assert list(cbar_ax.get_yticks()) == []
    assert cbar_ax.get_ylabel() == "Sum intensity (a.u.)"  # already a.u. -> unchanged


def test_run_writes_pngs_under_per_slice_subfolders(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    res = SL.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "slices_json": slices_json,
            "output_dir": str(out),
        }
    )
    assert res.pngs and all(os.path.exists(p) for p in res.pngs)
    # every PNG lives under {out_dir}/mid/, not flat in {out_dir}
    for p in res.pngs:
        assert os.path.basename(os.path.dirname(p)) == "mid"
        assert not os.path.exists(os.path.join(str(out), os.path.basename(p)))


def test_run_without_round_clim_has_no_notes_or_raw_attrs(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    res = SL.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "slices_json": slices_json,
            "output_dir": str(out),
        }
    )
    assert res.notes == []
    with h5py.File(res.output_h5, "r") as f:
        for vid in res.volume_ids:
            assert "vmin_raw" not in f[vid].attrs


def _write_mini_consolidated(path):
    """Two fields sharing one slice with 3 planes; raw-group + strain-group attrs."""
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offsets = np.array([-1.0, 0.0, 1.0])
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (("raw_sum", "raw_sum", "gray"), ("strain", "strain", "RdBu_r")):
            g = f.create_group(vid)
            g.attrs["kind"] = kind
            g.attrs["cmap"] = cmap
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "value"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("plane_a")
            sg.create_dataset("slices", data=np.zeros((3, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offsets)


def test_replot_catalog_enumerates_volumes_slices_planes(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    cat = SL.replot_catalog(str(h5))
    by_vid = {(e.volume_id, e.slice_name): e for e in cat}
    assert set(by_vid) == {("raw_sum", "plane_a"), ("strain", "plane_a")}
    assert by_vid[("strain", "plane_a")].n_planes == 3
    assert by_vid[("strain", "plane_a")].offsets_um == [-1.0, 0.0, 1.0]
    assert by_vid[("strain", "plane_a")].shape == (7, 9)  # (nv, nu) plane pixels — ROI hint
    # kind-group keys the per-kind clim overrides (raw_sum → "raw")
    assert by_vid[("strain", "plane_a")].group == "strain"
    assert by_vid[("raw_sum", "plane_a")].group == "raw"


def test_render_replot_per_group_clim_maps_by_kind(tmp_path, monkeypatch):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    seen: dict[str, tuple] = {}

    def fake_rebuild(h5_path, vid, sname, k, style, *, clim=None, roi=None):
        seen[vid] = clim
        return None  # skip the actual figure build / save

    monkeypatch.setattr(SL, "_rebuild_plane_figure", fake_rebuild)
    SL.render_replot(
        str(h5),
        [("strain", "plane_a", [0]), ("raw_sum", "plane_a", [0])],
        style=None,
        clim={"raw": (0.0, 10.0), "strain": (-5.0, 5.0)},
        out_dir=str(tmp_path / "r"),
    )
    # each volume's plane gets the clim for its kind-group, not a single global one
    assert seen["strain"] == (-5.0, 5.0)
    assert seen["raw_sum"] == (0.0, 10.0)


def test_render_replot_writes_selected_planes_under_subfolders(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    out = tmp_path / "replots"
    # strain: only planes 0 and 2; raw_sum: all planes (None)
    written = SL.render_replot(
        str(h5),
        [("strain", "plane_a", [0, 2]), ("raw_sum", "plane_a", None)],
        style=None,
        clim=None,
        out_dir=str(out),
    )
    assert len(written) == 2 + 3
    assert all(os.path.exists(p) for p in written)
    # per-slice subfolder layout
    assert all(os.path.basename(os.path.dirname(p)) == "plane_a" for p in written)


def test_render_replot_clim_override_changes_norm(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    fig = SL._rebuild_plane_figure(str(h5), "strain", "plane_a", 1, style=None, clim=(-5.0, 5.0))
    im = fig.axes[0].images[0]
    assert im.norm.vmin == -5.0 and im.norm.vmax == 5.0


def test_render_replot_roi_crops_slice(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    # crop to a sub-window; the rebuilt image must have the cropped shape
    fig = SL._rebuild_plane_figure(str(h5), "strain", "plane_a", 1, style=None, roi=(0, 2, 0, 2))
    im = fig.axes[0].images[0]
    assert im.get_array().shape == (2, 2)


def test_render_replot_roi_empty_crop_skipped(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    written = SL.render_replot(
        str(h5),
        [("strain", "plane_a", None)],
        style=None,
        clim=None,
        out_dir=str(tmp_path / "r"),
        roi=(2, 2, 0, 3),
    )
    assert written == []


def _write_mosa_consolidated(path):
    """Two mosa-COM volumes (chi, mu) sharing group 'mosa_com', one slice, 2 planes."""
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offsets = np.array([0.0, 1.0])
    with h5py.File(path, "w") as f:
        for vid in ("mosa_com_chi", "mosa_com_mu"):
            g = f.create_group(vid)
            g.attrs["kind"] = "mosa_com"
            g.attrs["cmap"] = "magma"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "deg"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("plane_a")
            sg.create_dataset("slices", data=np.zeros((2, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offsets)


def test_render_replot_clim_keyed_by_volume_id(tmp_path, monkeypatch):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mosa_consolidated(str(h5))
    seen: dict[str, tuple] = {}

    def fake_rebuild(h5_path, vid, sname, k, style, *, clim=None, roi=None):
        seen[vid] = clim
        return None

    monkeypatch.setattr(SL, "_rebuild_plane_figure", fake_rebuild)
    SL.render_replot(
        str(h5),
        [("mosa_com_chi", "plane_a", [0]), ("mosa_com_mu", "plane_a", [0])],
        style=None,
        clim={"mosa_com_chi": (-2.0, 2.0), "mosa_com_mu": (-9.0, 9.0)},
        out_dir=str(tmp_path / "r"),
    )
    assert seen["mosa_com_chi"] == (-2.0, 2.0)
    assert seen["mosa_com_mu"] == (-9.0, 9.0)  # chi and mu NO LONGER share a limit


def test_render_replot_clim_group_key_still_works(tmp_path, monkeypatch):
    """Back-compat: a group-keyed dict still applies via the fallback."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_mosa_consolidated(str(h5))
    seen: dict[str, tuple] = {}
    monkeypatch.setattr(
        SL,
        "_rebuild_plane_figure",
        lambda h5_path, vid, sname, k, style, *, clim=None, roi=None: seen.__setitem__(vid, clim),
    )
    SL.render_replot(
        str(h5),
        [("mosa_com_chi", "plane_a", [0]), ("mosa_com_mu", "plane_a", [0])],
        style=None,
        clim={"mosa_com": (-3.0, 3.0)},  # group key
        out_dir=str(tmp_path / "r"),
    )
    assert seen["mosa_com_chi"] == (-3.0, 3.0)
    assert seen["mosa_com_mu"] == (-3.0, 3.0)


def test_run_includes_mosa_raw_field(tmp_path):
    proc, raw = _setup(tmp_path)
    rng = np.random.default_rng(1)
    with h5py.File(proc / "aligned_raw_mosa_volumes.h5", "w") as f:
        f.create_dataset("sum_intensity", data=rng.standard_normal((L, NY, NX)).astype(np.float32))
        f.create_dataset("specific_frame", data=rng.standard_normal((L, NY, NX)).astype(np.float32))
        f.create_dataset("z_uniform_um", data=np.arange(L, dtype=np.float32))
        f.attrs["scale_x_um_per_px"] = 0.152
        f.attrs["scale_y_um_per_px"] = 0.385
        f.attrs["scale_z_um_per_px"] = 1.0
        f.attrs["specific_frame_idx"] = 2
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    res = SL.run(
        {
            "aligned_mosa_file": str(proc / "aligned_raw_mosa_volumes.h5"),
            "include_mosa_sum": True,
            "include_mosa_specific": False,
            # keep the run small: turn the standard volumes off
            "include_mosa_com_chi": False,
            "include_mosa_fwhm_chi": False,
            "include_mosa_com_mu": False,
            "include_mosa_fwhm_mu": False,
            "include_strain": False,
            "include_raw_sum": False,
            "include_raw_specific": False,
            "slices_json": slices_json,
            "output_dir": str(tmp_path / "sl"),
        }
    )
    assert "raw_mosa_sum" in res.volume_ids
    with h5py.File(res.output_h5, "r") as f:
        assert f["raw_mosa_sum"].attrs["kind"] == "raw_mosa_sum"
        assert f["raw_mosa_sum"].attrs["title"] == "Mosa-integrated Sum Intensity"


def test_plane_preview_returns_middle_plane_and_du_dv(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(
        str(h5)
    )  # u: 9 pts over [-4,4] -> du=1.0; v: 7 pts over [-3,3] -> dv=1.0
    arr, sx, sy = SL.plane_preview(str(h5), "strain", "plane_a")
    assert arr.shape == (7, 9)  # (nv, nu)
    assert sx == pytest.approx(1.0)  # du (cols/X)
    assert sy == pytest.approx(1.0)  # dv (rows/Y)


def test_plane_preview_du_dv_not_swapped(tmp_path):
    """Asymmetric pitch (du != dv) catches an sx/sy swap: sx=du(X/cols), sy=dv(Y/rows)."""
    h5 = tmp_path / "oblique_slices_asym.h5"
    u = np.linspace(-4.0, 4.0, 5)  # 5 pts → du = 2.0 (cols/X)
    v = np.linspace(-3.0, 3.0, 7)  # 7 pts → dv = 1.0 (rows/Y)
    offsets = np.array([0.0, 1.0])
    with h5py.File(str(h5), "w") as f:
        g = f.create_group("strain")
        g.attrs["kind"] = "strain"
        g.attrs["cmap"] = "RdBu_r"
        g.attrs["title"] = "strain"
        g.attrs["cbar_label"] = "value"
        g.attrs["vmin"] = -1.0
        g.attrs["vmax"] = 1.0
        sg = g.create_group("slice0")
        sg.create_dataset("slices", data=np.zeros((2, v.size, u.size), dtype=np.float32))
        sg.create_dataset("u_um", data=u)
        sg.create_dataset("v_um", data=v)
        sg.create_dataset("offsets_um", data=offsets)
    arr, sx, sy = SL.plane_preview(str(h5), "strain", "slice0")
    assert arr.shape == (7, 5)  # (nv, nu) = (Y rows, X cols)
    assert sx == pytest.approx(2.0)  # du / X / cols
    assert sy == pytest.approx(1.0)  # dv / Y / rows


# -- pinned planes ------------------------------------------------------------
def test_build_pinned_spec_snaps_and_reproduces_sweep_plane(tmp_path):
    proc, raw = _setup(tmp_path)
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "slices_json": (
            '[{"name":"zsweep","normal":[0,0,1],"origin":[0,0,0],'
            '"extent":"auto","sweep_step_um":1.0}]'
        ),
        "output_dir": str(tmp_path / "sl"),
        "save_png": False,
    }
    res = SL.run(dict(params))
    with h5py.File(res.output_h5, "r") as f:
        sweep_offsets = f["strain/zsweep/offsets_um"][:]
        sweep_stack = f["strain/zsweep/slices"][:]
        attrs = dict(f["strain/zsweep"].attrs)
    target = float(sweep_offsets[1]) + 0.2  # off-grid: must snap to plane 1
    specs = SL.build_pinned_spec(res.output_h5, "zsweep", [target, target])
    assert len(specs) == 1  # duplicate snap collapsed
    spec = specs[0]
    assert spec["sweep_start_um"] == spec["sweep_stop_um"]
    assert spec["sweep_start_um"] == pytest.approx(float(sweep_offsets[1]))
    for key in ("normal", "origin", "up"):
        np.testing.assert_allclose(spec[key], np.asarray(attrs[key], float))
    for key in ("half_u", "half_v", "du", "dv"):
        assert spec[key] == pytest.approx(float(attrs[key]))
    # golden: a run on the pinned spec reproduces the sweep's plane 1 exactly
    res2 = SL.run({**params, "slices_json": json.dumps(specs), "output_dir": str(tmp_path / "pin")})
    with h5py.File(res2.output_h5, "r") as f:
        pinned = f[f"strain/{spec['name']}/slices"][:]
    np.testing.assert_array_equal(pinned[0], sweep_stack[1])


def test_build_pinned_spec_unknown_slice_raises_user_error(tmp_path):
    p = str(tmp_path / "s.h5")
    with h5py.File(p, "w") as f:
        f.create_group("strain")
    with pytest.raises(StageUserError):
        SL.build_pinned_spec(p, "nope", [0.0])


def test_pin_slice_cli_delegates_to_core(tmp_path):
    p = str(tmp_path / "s.h5")
    with h5py.File(p, "w") as f:
        sg = f.create_group("strain").create_group("oblique")
        sg.create_dataset("offsets_um", data=np.array([0.0, 1.0, 2.0]))
        sg.attrs["normal"] = [0.0, 0.0, 1.0]
        sg.attrs["origin"] = [0.0, 0.0, 0.0]
        sg.attrs["up"] = [0.0, 1.0, 0.0]
        for k, v in (
            ("half_u", 4.0),
            ("half_v", 3.0),
            ("du", 0.2),
            ("dv", 0.2),
            ("sweep_step_um", 1.0),
        ):
            sg.attrs[k] = v
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [
            sys.executable,
            os.path.join(root, "tools", "pin_slice.py"),
            p,
            "oblique",
            "--offset",
            "1.4",
        ],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert len(out) == 1 and out[0]["sweep_start_um"] == pytest.approx(1.0)


def _pinned_params(proc, raw, out):
    return {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "slices_json": "[]",  # would raise on the sweep path — proves routing skips it
        "use_pinned": True,
        "pinned_slices_json": (
            '[{"name":"pin","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
            '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
        ),
        "output_dir": str(out),
        "save_png": False,
    }


def test_run_use_pinned_routes_and_renames_output(tmp_path):
    proc, raw = _setup(tmp_path)
    res = SL.run(_pinned_params(proc, raw, tmp_path / "sl"))
    assert res.output_h5 and res.output_h5.endswith("oblique_slices_pinned.h5")
    assert any("PINNED RUN" in n for n in res.notes)
    assert res.slice_names == ["pin"]


def test_run_use_pinned_respects_user_edited_name(tmp_path):
    proc, raw = _setup(tmp_path)
    p = _pinned_params(proc, raw, tmp_path / "sl")
    p["output_h5_name"] = "custom.h5"
    res = SL.run(p)
    assert res.output_h5.endswith("custom.h5")


def test_run_use_pinned_empty_or_invalid_raises_user_error(tmp_path):
    proc, raw = _setup(tmp_path)
    for bad in ("", "   ", "{not json", "[]"):
        p = _pinned_params(proc, raw, tmp_path / "sl")
        p["pinned_slices_json"] = bad
        with pytest.raises(StageUserError, match="[Pp]inned"):
            SL.run(p)


# -- draw_slice_axes extraction pins ------------------------------------------
def _prep(cmap="magma", center=False):
    return {
        "cmap_name": cmap,
        "title": "χ CoM",
        "cbar_label": "deg",
        "vmin": -1.0,
        "vmax": 3.0,
        "center_zero": center,
        "group": "mosa_com",
    }


def test_build_slice_figure_unstyled_pinned_shape_and_decor():
    import numpy as np

    from dfxm.stages.slices import build_slice_figure

    u = np.linspace(-5.0, 5.0, 21)
    v = np.linspace(-4.0, 4.0, 17)
    fig = build_slice_figure(_prep(), {"name": "obl"}, np.zeros((17, 21)), u, v, offset_um=1.0)
    assert tuple(fig.get_size_inches()) == (12.0, 10.0)
    ax = fig.axes[0]
    im = ax.images[0]
    assert list(im.get_extent()) == [-5.0, 5.0, -4.0, 4.0]
    assert ax.get_xlabel() == "u (µm)" and ax.get_ylabel() == "v (µm)"
    assert "offset +1.00" in ax.get_title()
    assert len(fig.axes) == 2  # stolen colorbar present


def test_build_slice_figure_centered_norm_pinned():
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    from dfxm.stages.slices import build_slice_figure

    u = np.linspace(0.0, 2.0, 5)
    v = np.linspace(0.0, 2.0, 5)
    fig = build_slice_figure(
        _prep(center=True), {"name": "obl"}, np.zeros((5, 5)), u, v, offset_um=None
    )
    assert isinstance(fig.axes[0].images[0].norm, TwoSlopeNorm)


# -- Z-blocked gather ---------------------------------------------------------
# The stage samples planes from either a resident aligned volume (`sample_plane`)
# or a stream of Z-blocks (`sample_planes_streamed`), and which one runs is
# decided by how much memory the machine has. The two must therefore agree
# bit-for-bit, or a figure would depend on the machine it was made on.
_GATHER_GEOMETRY = {
    "scale_x": 0.7,
    "scale_y": 1.3,
    "scale_z": 0.9,
    "x_ref_shift_px": 2.0,
    "y_ref_shift_px": 1.0,
    "z_ref_shift_um": 0.5,
}

# Tilted in BOTH in-plane directions on purpose: that is the case the gather's
# row/column probes cannot prune tightly, so it exercises the general path rather
# than the two shapes the stage ships defaults for.
_GATHER_PLANE = {
    "plane_origin": (5.0, 12.0, 11.0),
    "half_u": 12.0,
    "half_v": 12.0,
    "du": 0.5,
    "dv": 0.5,
}


def _gather_volume(nz=24, ny=20, nx=20, seed=5):
    """A volume with structure in all three axes, and a NaN region."""
    rng = np.random.default_rng(seed)
    z, y, x = np.mgrid[0:nz, 0:ny, 0:nx]
    vol = (z * 100.0 + y * 10.0 + x).astype(np.float64) + rng.standard_normal((nz, ny, nx))
    vol[nz // 3, ny // 2 :, :3] = np.nan  # NaN must travel with the sample, not the block
    return vol


def _gather_basis():
    return SL.build_basis((0.6, 0.2, 0.77))


def _in_core_prep(volume):
    return {**_GATHER_GEOMETRY, "data": np.ascontiguousarray(volume, dtype=np.float64)}


def _streamed_prep(volume, budget_bytes):
    from dfxm.common import volumeio

    volume = np.ascontiguousarray(volume, dtype=np.float64)
    return {
        **_GATHER_GEOMETRY,
        "data": None,
        "shape": volume.shape,
        "blocks": lambda: volumeio.iter_blocks(volume, budget_bytes=int(budget_bytes)),
    }


def _n_blocks(volume, budget_bytes):
    from dfxm.common import volumeio

    return sum(1 for _ in volumeio.iter_blocks(volume, budget_bytes=int(budget_bytes)))


def _gather_args(plane=None):
    plane = plane or _GATHER_PLANE
    u_hat, v_hat, _n = _gather_basis()
    return (
        plane["plane_origin"],
        u_hat,
        v_hat,
        plane["half_u"],
        plane["half_v"],
        plane["du"],
        plane["dv"],
    )


def test_slices_streamed_gather_matches_in_core():
    """The Z-blocked gather returns the same image as sampling the whole volume."""
    volume = _gather_volume()
    budget = volume.nbytes // 6
    reference = SL.sample_plane(_in_core_prep(volume), *_gather_args())[0]
    streamed = SL.sample_plane_streamed(_streamed_prep(volume, budget), *_gather_args())[0]

    # Assert the preconditions, or this passes while measuring nothing: the
    # budget must really split the volume, the plane must really straddle its
    # edges (so `cval=np.nan` is exercised), and it must really sample data.
    assert _n_blocks(volume, budget) >= 5, "the budget must actually block the volume"
    assert np.isnan(reference).any(), "the plane must run off the volume somewhere"
    assert np.isfinite(reference).sum() > 100, "the plane must sample the volume somewhere"

    assert np.array_equal(streamed, reference, equal_nan=True)


def test_slices_gather_is_budget_independent():
    from tests.equivalence import assert_budget_independent

    volume = _gather_volume(seed=6)
    assert_budget_independent(
        lambda vol, budget_bytes: SL.sample_plane_streamed(
            _streamed_prep(vol, budget_bytes), *_gather_args()
        )[0],
        volume,
    )


def test_slices_gather_walks_the_stream_once_for_a_whole_sweep():
    """One pass over Z serves every plane — not one alignment per plane.

    The output of a plane is a small 2-D image whatever the volume's size, so a
    sweep can be gathered in a single traversal. Sampling plane by plane would
    re-run the alignment chain once per plane, which on a real sweep is the
    difference between one read of the volume and two hundred.
    """
    volume = _gather_volume(seed=7)
    prep = _streamed_prep(volume, volume.nbytes // 6)
    walks = []
    inner = prep["blocks"]

    def counting():
        walks.append(1)
        return inner()

    prep["blocks"] = counting
    _origin, u_hat, v_hat, half_u, half_v, du, dv = _gather_args()
    _n_hat = _gather_basis()[2]
    origins = [np.asarray(_origin, float) + off * _n_hat for off in (-4.0, -2.0, 0.0, 2.0, 4.0)]
    planes, _u, _v = SL.sample_planes_streamed(prep, origins, u_hat, v_hat, half_u, half_v, du, dv)

    assert len(planes) == 5
    assert sum(walks) == 1, f"the sweep walked the Z stream {sum(walks)} times"
    # …and each plane is what sampling it alone would have given.
    for origin, plane in zip(origins, planes):
        alone = SL.sample_plane(_in_core_prep(volume), origin, u_hat, v_hat, half_u, half_v, du, dv)
        assert np.array_equal(plane, alone[0], equal_nan=True)


def test_slices_gather_batches_a_sweep_without_changing_a_value():
    """`max_resident_bytes` trades walks for resident planes and nothing else.

    Splitting the sweep is what stops it holding its whole stack, and the price
    is one extra traversal per batch. Both halves of that are asserted: the walk
    count must really rise (or the batching is not happening and the value check
    below compares a run against itself), and the planes must be unchanged.
    """
    volume = _gather_volume(seed=9)
    _origin, u_hat, v_hat, half_u, half_v, du, dv = _gather_args()
    n_hat = _gather_basis()[2]
    origins = [np.asarray(_origin, float) + off * n_hat for off in (-4.0, -2.0, 0.0, 2.0, 4.0)]

    def gather(max_resident_bytes):
        prep = _streamed_prep(volume, volume.nbytes // 6)
        walks = []
        inner = prep["blocks"]
        prep["blocks"] = lambda: (walks.append(1), inner())[1]
        planes = list(
            SL.iter_planes_streamed(
                prep,
                origins,
                u_hat,
                v_hat,
                half_u,
                half_v,
                du,
                dv,
                max_resident_bytes=max_resident_bytes,
            )
        )
        return planes, sum(walks)

    axis_u, axis_v = SL._plane_axes(half_u, half_v, du, dv)
    plane_bytes = len(axis_u) * len(axis_v) * 4
    whole, whole_walks = gather(None)
    batched, batched_walks = gather(2 * plane_bytes)

    assert whole_walks == 1, "the unbounded gather must still be one walk"
    assert batched_walks == 3, f"5 planes at 2 per batch is 3 walks, got {batched_walks}"
    assert len(batched) == len(whole) == 5
    for a, b in zip(batched, whole):
        assert np.array_equal(a, b, equal_nan=True)
    # …and the planes really carry data, or "equal" would be a statement about NaN.
    assert np.isfinite(whole[2]).sum() > 100


def test_sweep_batch_size_never_returns_zero_planes():
    """A budget under one plane still yields one: a plane is indivisible here."""
    assert SL.sweep_batch_size(10, 1000, None) == 10
    assert SL.sweep_batch_size(10, 1000, 10_000) == 10  # …and never more than the sweep
    assert SL.sweep_batch_size(10, 1000, 2500) == 2
    assert SL.sweep_batch_size(10, 1000, 999) == 1
    assert SL.sweep_batch_size(10, 1000, 0) == 1


def test_gather_scratch_plane_multiple_is_rounded_the_safe_way():
    """The per-plane scratch charge must cover what the gather holds, rounded UP.

    It is *subtracted* from the budget before the plane batch is sized, so
    rounding down leaves a batch bigger than what was counted — the permissive
    direction, and the one this project has already had to correct twice.
    Nothing else pins the number.
    """
    import math

    # Per element of the coordinate rectangle, in float32-plane units:
    coords = 3 * 8 / 4  # `_plane_coords` — (k, j, i) as float64
    local = 3 * 8 / 4  # `np.stack([k[sel], j[sel], i[sel]])` at full selection
    sampled = 8 / 4  # `map_coordinates`' float64 return
    mask = 1 / 4  # the boolean `sel`
    counted = coords + local + sampled + mask
    assert counted == 14.25
    assert SL.GATHER_SCRATCH_PLANE_MULTIPLE == math.ceil(counted)


def test_sweep_resident_bytes_subtracts_the_stream_and_the_scratch():
    """The plane batch gets what is left, never the whole budget, never nothing."""
    plane = 1 << 20
    budget = 512 << 20
    reserved = 6
    room = SL._sweep_resident_bytes(budget, plane, reserved)
    assert (
        room
        == budget
        - budget // SL.REDUCTION_WORKING_SET_MULTIPLE
        - (SL.GATHER_SCRATCH_PLANE_MULTIPLE + reserved) * plane
    )
    assert 0 < room < budget
    # A budget the scratch alone exhausts still buys one plane, not a negative one.
    assert SL._sweep_resident_bytes(plane, plane, reserved) == plane


def test_slices_gather_matches_in_core_for_the_shipped_plane_families():
    """The two plane shapes the stage ships defaults for, which the probes prune.

    A Z-normal plane has a constant `k` (its whole image lives in one block) and
    the default oblique normal varies `k` along `u` only. Both take the pruning
    branches that the deliberately-tilted fixture above does not, so a pruning
    bug that dropped a band would show here and nowhere else.
    """
    volume = _gather_volume(seed=8)
    budget = volume.nbytes // 6
    for normal in ((0, 0, 1), (0.647648, 0, 0.761939)):
        u_hat, v_hat, _n = SL.build_basis(normal)
        args = ((5.0, 12.0, 11.0), u_hat, v_hat, 12.0, 12.0, 0.5, 0.5)
        reference = SL.sample_plane(_in_core_prep(volume), *args)[0]
        assert np.isfinite(reference).sum() > 100, f"normal {normal} sampled nothing"
        streamed = SL.sample_plane_streamed(_streamed_prep(volume, budget), *args)[0]
        assert np.array_equal(streamed, reference, equal_nan=True), f"normal {normal}"


# -- the two rungs, end to end ------------------------------------------------
def _rung_params(proc, raw, out, budget_bytes):
    params = _minimal_params(proc, raw, out)
    params["save_png"] = True
    params["_budget_bytes"] = budget_bytes
    return params


def _h5_contents(path):
    """Every dataset and attribute of an oblique_slices.h5, for exact comparison."""
    out = {}

    def visit(name, obj):
        for key, value in obj.attrs.items():
            out[f"{name}@{key}"] = np.asarray(value).tobytes()
        if isinstance(obj, h5py.Dataset):
            out[name] = obj[()].tobytes()

    with h5py.File(path, "r") as f:
        for key, value in f.attrs.items():
            out[f"/@{key}"] = np.asarray(value).tobytes()
        f.visititems(visit)
    return out


@pytest.mark.parametrize("center_method", ["midrange", "mean", "median"])
def test_slices_both_rungs_produce_identical_products(tmp_path, center_method):
    """Which rung runs depends on the machine, so neither product may differ.

    Compared at the byte level: every dataset and attribute of
    `oblique_slices.h5` and every PNG. The `_budget_bytes` injection is what
    forces the rungs apart, and the run under it is asserted to have actually
    taken the streaming one — without that this compares a run against itself.
    """
    proc, raw = _setup(tmp_path)
    big = _rung_params(proc, raw, tmp_path / "in_core", 1 << 30)
    small = _rung_params(proc, raw, tmp_path / "streamed", 4096)
    big["center_method"] = small["center_method"] = center_method

    rungs = []
    real = SL.prepare_volume

    def spy(*args, **kwargs):
        prep = real(*args, **kwargs)
        rungs.append(prep["data"] is not None)
        return prep

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(SL, "prepare_volume", spy)
        in_core = SL.run(big)
        assert rungs and all(rungs), "the generous budget must take the in-core rung"
        rungs.clear()
        streamed = SL.run(small)
        assert rungs and not any(rungs), "the tiny budget must take the streaming rung"
    finally:
        monkeypatch.undo()

    assert _h5_contents(in_core.output_h5) == _h5_contents(streamed.output_h5)
    assert len(in_core.pngs) == len(streamed.pngs) > 0
    for a, b in zip(sorted(in_core.pngs), sorted(streamed.pngs)):
        assert os.path.basename(a) == os.path.basename(b)
        with open(a, "rb") as fa, open(b, "rb") as fb:
            assert fa.read() == fb.read(), os.path.basename(a)


def test_slices_streaming_rung_never_materialises_a_volume(tmp_path):
    """A small budget must not quietly assemble the aligned volume anyway."""
    proc, raw = _setup(tmp_path)
    params = _rung_params(proc, raw, tmp_path / "sl_nomat", 4096)
    seen = []
    real = SL.prepare_volume

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            SL, "prepare_volume", lambda *a, **k: seen.append(real(*a, **k)) or seen[-1]
        )
        SL.run(params)
    finally:
        monkeypatch.undo()
    assert len(seen) >= 2, "the test needs several volumes to be meaningful"
    assert all(prep["data"] is None for prep in seen)
    assert all(callable(prep["blocks"]) for prep in seen)


def _prepare_at_budget(
    proc, raw, tmp_path, budget_bytes, center_method, cfg_kind="mosa_com", abs_fwhm=True
):
    """`prepare_volume` for the χ CoM volume at a chosen budget, drained to an array."""
    import contextlib

    p = {**SL.STAGE.defaults(), **_minimal_params(proc, raw, tmp_path / "unused")}
    p["center_method"] = center_method
    p["abs_fwhm"] = abs_fwhm
    cfg = {
        "h5_path": str(proc / "stacked_volumes.h5"),
        "dataset_path": "chi/Center of mass",
        "kind": cfg_kind,
        "source": "stacked",
        "raw_root": str(raw),
        "raw_pattern": "mosa__*",
        "roi_x": None,
        "roi_y": None,
    }
    with contextlib.ExitStack() as opened:
        prep = SL.prepare_volume(
            cfg, p, 0.152, 0.385, -1, style=None, stack=opened, budget_bytes=budget_bytes
        )
        fits = prep["data"] is not None
        blocks = [b for _sl, b in prep["blocks"]()]
        data = np.concatenate(blocks, axis=0) if blocks else np.empty((0,))
        limits = (prep["vmin_raw"], prep["vmax_raw"])
    return fits, data, limits


@pytest.mark.parametrize("center_method", ["midrange", "mean", "median"])
def test_slices_both_rungs_agree_on_the_centred_volume_and_its_limits(tmp_path, center_method):
    """The rungs must agree on VALUES, which the byte-level product check cannot see.

    `test_slices_both_rungs_produce_identical_products` compares stored planes
    and rendered PNGs, and that comparison is structurally blind to the change
    this test exists for: the centring offset moves by about one ulp between
    `np.nanmean` and `volumeio.stream_mean`, and a one-ulp shift in a float64
    offset neither moves a percentile of `|value|` nor survives the float32 cast
    the stored planes take. It was live and invisible on a real fixture — checked,
    not assumed.

    **This test used to demand that `np.nanmean` and `volumeio.stream_mean`
    disagree on this fixture, and that precondition has been deliberately
    removed — do not restore it.** It rested on a floating-point accident, not a
    property: over the aligned volume's 168 values the two reductions differed
    by exactly *one ulp* under numpy 1.26, and under numpy 2.5 — whose summation
    is markedly more accurate — they agree exactly, so the assert fired. It
    cannot be repaired by choosing better data, which was measured rather than
    assumed: sweeping pedestal magnitudes 1e8…1e16 through the alignment chain,
    only 1e12 separated them, and only by two ulps; and outside the chain the
    two numpy generations disagree with the compensated sum in *uncorrelated*
    regimes (a 2 000 000-value pedestal separates 1.26 by 7 ulps and 2.5 by 0,
    while plain normals at n=200 000 separate 2.5 by 3 ulps and 1.26 by 0).
    There is no array that separates them on both, because numpy 2's `nanmean`
    is itself near-correctly-rounded almost everywhere.

    What actually keeps this test honest is structural and version-stable: the
    two budgets are asserted to take *different rungs* (`in_core_fits and not
    streamed_fits` below), so the comparison is always across the in-core and
    streamed paths rather than the same path twice. And the property the
    compensated mean is really there for — that the answer does not move with
    the block size — is pinned directly, and robustly, by
    `test_common_volumeio.py::test_stream_mean_is_budget_independent_on_adversarial_data`,
    which passes on both numpy generations.
    """
    proc, raw = _setup(tmp_path)
    from dfxm.common import volumeio

    # `mosa_fwhm` with `abs_fwhm=False` runs the same alignment on the same
    # dataset and skips the centring, so this is the uncentred aligned volume.
    _fits, aligned, _limits = _prepare_at_budget(
        proc, raw, tmp_path, 1 << 30, "midrange", cfg_kind="mosa_fwhm", abs_fwhm=False
    )
    finite = aligned[np.isfinite(aligned)]
    assert finite.size, "the aligned volume must hold finite voxels"
    # Centring must be a real transformation on this fixture, or the rungs would
    # agree trivially: an offset of zero centres nothing and the comparison
    # below would hold however the offset was computed.
    offset = float(volumeio.stream_mean([finite]))
    assert np.isfinite(offset) and offset != 0.0, (
        f"the fixture's centring offset is {offset!r} — centring is a no-op here, "
        "so this test would pass however the offset was reduced"
    )

    in_core_fits, in_core, in_core_limits = _prepare_at_budget(
        proc, raw, tmp_path, 1 << 30, center_method
    )
    streamed_fits, streamed, streamed_limits = _prepare_at_budget(
        proc, raw, tmp_path, 4096, center_method
    )
    assert in_core_fits and not streamed_fits, "the two budgets must take different rungs"
    assert in_core_limits == streamed_limits
    assert np.array_equal(in_core, streamed, equal_nan=True)


# -- measured memory ----------------------------------------------------------
def _peak_setup(tmp_path, L, NY, NX):
    """A slices fixture at a chosen volume size — seven volumes, no PNGs."""
    proc, raw = tmp_path / "proc", tmp_path / "raw"
    proc.mkdir()
    raw.mkdir()
    rng = np.random.default_rng(0)
    with h5py.File(proc / "stacked_volumes.h5", "w") as f:
        for grp in ("chi", "mu"):
            g = f.create_group(grp)
            g.create_dataset("Center of mass", data=rng.standard_normal((L, NY, NX)))
            g.create_dataset("FWHM", data=np.abs(rng.standard_normal((L, NY, NX))))
    with h5py.File(proc / "stacked_strain_volumes.h5", "w") as f:
        f.create_dataset("strain", data=rng.standard_normal((L, NY, NX)) * 1e-4)
    with h5py.File(proc / "aligned_raw_rocking_volumes.h5", "w") as f:
        f.create_dataset("sum_intensity", data=rng.standard_normal((L, NY, NX)).astype(np.float32))
        f.create_dataset("specific_frame", data=rng.standard_normal((L, NY, NX)).astype(np.float32))
        f.attrs["scale_x_um_per_px"] = 0.152
        f.attrs["scale_y_um_per_px"] = 0.385
        f.attrs["scale_z_um_per_px"] = 1.0
        f.attrs["specific_frame_idx"] = 2
    samy = np.linspace(0.0, 0.004, L)
    samz = np.linspace(0.0, 0.006, L)
    for base in ("mosa", "strain"):
        for i in range(L):
            folder = raw / f"{base}__{i + 1}"
            folder.mkdir()
            with h5py.File(folder / f"{base}__{i + 1}.h5", "w") as f:
                f.create_dataset("1.1/instrument/positioners/samy", data=samy[i])
                f.create_dataset("1.1/instrument/positioners/samz", data=samz[i])
    data_bytes = 5 * L * NY * NX * 8 + 2 * L * NY * NX * 4
    return proc, raw, data_bytes


def _peak_params(proc, raw, out):
    sj = json.dumps(
        [
            {
                "name": "mid",
                "normal": [0, 0, 1],
                "origin": [1.0, 10.0, 3.0],
                "half_u": 3.0,
                "half_v": 3.0,
                "du": 0.15,
                "dv": 0.15,
                "sweep_step_um": None,
            },
            {
                "name": "obl",
                "normal": [0.647648, 0, 0.761939],
                "origin": [1.0, 10.0, 3.0],
                "half_u": 3.0,
                "half_v": 3.0,
                "du": 0.15,
                "dv": 0.15,
                "sweep_step_um": 2.0,
                "sweep_start_um": -2.0,
                "sweep_stop_um": 2.0,
            },
        ]
    )
    return {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "aligned_rocking_file": str(proc / "aligned_raw_rocking_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "slices_json": sj,
        "output_dir": str(out),
        "save_png": False,
    }


def test_slices_peak_stays_under_budget(tmp_path):
    """The streaming rung really lowers peak RSS, not just the code path taken.

    `_budget_bytes` alone proves only that the blocks got smaller; a stage can
    block its read and then materialise a float64 copy anyway. Measured in the
    real child, seven 64x256x256 volumes (32 MiB a volume) against a ~104 MiB
    child floor (PNGs off):

        previous commit (whole volumes)   314.7 MiB   (~6.6 volumes of data)
        this commit, machine budget       315.9 MiB   (the in-core rung, i.e.
                                                       unchanged, as intended)
        this commit, 64 MiB budget        179.2 MiB
        this commit, 16 MiB budget        120.2 MiB   (~0.5 volumes)
        this commit, 4 MiB budget         112.7 MiB

    The 200 MiB limit sits between the two, so this fails against the previous
    commit — checked by running it there, not assumed — and has ~1.66x of margin
    on the passing side for a machine whose process image differs from this one's.
    `_budget_bytes` is pinned rather than measured from the machine, so the
    figure does not depend on how much RAM the runner happens to have.
    """
    from tests.peak_rss import assert_peak_under

    proc, raw, _bytes = _peak_setup(tmp_path, 64, 256, 256)
    params = {**_peak_params(proc, raw, tmp_path / "sl_peak"), "_budget_bytes": 16 << 20}
    result = assert_peak_under("dfxm.stages.slices:run", params, 200 << 20, timeout=900)
    assert len(result.volume_ids) == 7, "the run must actually have sliced the volumes"
    assert result.n_planes_total == 28


# The sweep fixture for the two peak tests below. The volume is 8x256x128 —
# world box 19.4 x 98.6 x 6.0 µm — and the sweep is 201 planes of 801x801
# float32 (2.45 MiB each, 492 MiB of stack) at 0.1 µm pitch, centred so that:
#
# * **every** plane's Z lies inside the volume (origin Z 3.0 µm, sweep ±2.9 µm
#   against a 0-6 µm box), so no plane is skipped for being out of range, and
# * ~23% of each plane's samples land in the volume — the plane is 80 µm wide
#   against a 19.4 µm X extent, and covers Y entirely.
#
# That last number is the point of the geometry. The first version of this
# fixture put an 80 µm plane over a 3.6 x 9.2 µm volume and swept ±100 µm, which
# was **99.97% NaN** — 0.027% of the sweep finite. It measured the right byte
# counts, but it is the exact fixture shape this project has already had to
# repair twice, and a sampler that had stopped sampling would have passed it.
# `_assert_sweep_samples_the_volume` is what stops that coming back.
_SWEEP_ORIGIN = (9.7, 49.0, 3.0)
_SWEEP_SHAPE = (8, 256, 128)
_MIN_SWEEP_FINITE_FRACTION = 0.15


def _sweep_peak_params(proc, raw, out, *, half=40.0, d=0.1, step=0.029, span=2.9):
    """One volume, PNGs off, and a sweep whose PLANE STACK dwarfs the volume.

    The volume is deliberately tiny (2 MB) so nothing but the sampled planes can
    account for the peak: whatever this run costs above its process image is the
    sweep. Only `include_strain` is on — seven volumes would multiply the
    runtime without changing what is being measured, since the stage releases
    each volume before preparing the next.

    Returns `(params, n_planes, (nv, nu))`. The plane count comes from
    `SL.slice_plane_offsets` on the actual spec rather than from arithmetic
    here, so a floating-point edge in the sweep window cannot silently make the
    two disagree.
    """
    spec = {
        "name": "wide",
        "normal": [0, 0, 1],
        "origin": list(_SWEEP_ORIGIN),
        "half_u": half,
        "half_v": half,
        "du": d,
        "dv": d,
        "sweep_step_um": step,
        "sweep_start_um": -span,
        "sweep_stop_um": span,
    }
    params = {
        **_peak_params(proc, raw, out),
        "slices_json": json.dumps([spec]),
        "save_png": False,
    }
    for key in (
        "include_mosa_com_chi",
        "include_mosa_fwhm_chi",
        "include_mosa_com_mu",
        "include_mosa_fwhm_mu",
        "include_raw_sum",
        "include_raw_specific",
    ):
        params[key] = False
    params["include_strain"] = True
    u_um, v_um = SL._plane_axes(half, half, d, d)
    return params, len(SL.slice_plane_offsets(spec)), (len(v_um), len(u_um))


def _sweep_strain_cfg(proc, raw):
    """The one volume `_sweep_peak_params` selects, as a `prepare_volume` cfg."""
    return {
        "h5_path": str(proc / "stacked_strain_volumes.h5"),
        "dataset_path": "strain",
        "kind": "strain",
        "source": "stacked",
        "raw_root": str(raw),
        "raw_pattern": "strain__*",
        "roi_x": None,
        "roi_y": None,
    }


def _sweep_rung_is_in_core(proc, raw, tmp_path, budget_bytes):
    """Whether `budget_bytes` puts the sweep fixture's volume on the in-core rung.

    Run in-process against the same 2 MB volume the child gets, because the
    child cannot be spied on — and the rung a peak test claims to measure has to
    be asserted, not assumed. It was assumed once here: `_budget_bytes = 64 MiB`
    was documented as the in-core rung and in fact streamed, because
    `align_volume_streamed` prices a layer at ~41 B per float64 element and 8 of
    them did not fit in the 9.1 MiB alignment share.
    """
    import contextlib

    p = {**SL.STAGE.defaults(), **_peak_params(proc, raw, tmp_path / "unused")}
    with contextlib.ExitStack() as opened:
        prep = SL.prepare_volume(
            _sweep_strain_cfg(proc, raw),
            p,
            0.152,
            0.385,
            -1,
            style=None,
            stack=opened,
            budget_bytes=budget_bytes,
        )
        return prep["data"] is not None


def _assert_sweep_samples_the_volume(h5_path, n_planes):
    """The stored sweep must really carry samples, on every plane, everywhere.

    Three separate ways this fixture could go quietly degenerate, each checked:
    the aligned volume could be mostly NaN (the defect repaired in `_setup` and
    in `test_stage_visualize.py`), the sweep could miss the volume in Z so most
    planes are empty, or a pruning bug could drop a band out of some planes. A
    single `> 100 finite somewhere` guard sees none of the three.
    """
    sampled = range(0, n_planes, 10)
    with h5py.File(h5_path, "r") as f:
        stored = f["strain"]["wide"]["slices"]
        assert stored.shape[0] == n_planes
        per_plane = [int(np.isfinite(stored[k]).sum()) for k in sampled]
        plane_size = stored.shape[1] * stored.shape[2]
    fraction = sum(per_plane) / (len(per_plane) * plane_size)
    assert fraction >= _MIN_SWEEP_FINITE_FRACTION, (
        f"the sweep is only {100 * fraction:.2f}% finite — the planes have slid "
        "off the volume, so this fixture measures byte counts over NaN"
    )
    assert min(per_plane) >= 0.5 * _MIN_SWEEP_FINITE_FRACTION * plane_size, (
        f"the emptiest sampled plane carries only {min(per_plane)} finite samples "
        f"of {plane_size} — the sweep is meant to lie wholly inside the volume "
        "in Z, so every plane should sample it"
    )


def _slice_rec(n_planes, nv, nu):
    return {
        "name": "wide",
        "u_um": np.linspace(-1.0, 1.0, nu),
        "v_um": np.linspace(-2.0, 2.0, nv),
        "offsets": np.arange(n_planes, dtype=np.float64) * 0.5,
        "normal": [0.0, 0.0, 1.0],
        "origin": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "u_hat": [1.0, 0.0, 0.0],
        "v_hat": [0.0, 1.0, 0.0],
        "n_hat": [0.0, 0.0, 1.0],
        "half_u": 1.0,
        "half_v": 2.0,
        "du": 0.1,
        "dv": 0.2,
        "sweep_step_um": 0.5,
    }


def test_plane_by_plane_write_is_byte_identical_to_a_whole_stack_write(tmp_path):
    """The new writer must reproduce the old `create_dataset(data=stack)` exactly.

    This is the direct check that the products did not move: the same planes are
    written both ways and the stored dataset compared **raw**, together with the
    layout parameters a downstream reader sees. `slices` planes are raw float32
    samples, so this comparison does see a value change of even one ulp — unlike
    a percentile or an 8-bit PNG, which cannot. What it does *not* cover is the
    sampling itself (identical code on both sides here) or the h5 file's byte
    image, which is not a product contract: `profiles` and the viewers read
    `sg["slices"][k]`, never the file's bytes.

    The chunk shape is asserted equal rather than merely present, because the
    writer's buffer is sized from it — a silent change there would change how
    much memory the stage holds without changing a stored value.
    """
    n_planes, nv, nu = 11, 20, 30
    rng = np.random.default_rng(3)
    planes = [rng.standard_normal((nv, nu)).astype(np.float32) for _ in range(n_planes)]
    planes[4][3:7, 5:9] = np.nan  # NaN must survive the slab copy unchanged
    rec = _slice_rec(n_planes, nv, nu)

    with h5py.File(tmp_path / "whole.h5", "w") as f:
        sg = f.create_group("vol").create_group("wide")
        old = sg.create_dataset(
            "slices",
            data=np.stack(planes, axis=0).astype(np.float32),
            compression="gzip",
            compression_opts=4,
        )
        old_chunks, old_shape, old_dtype = old.chunks, old.shape, old.dtype

    with h5py.File(tmp_path / "streamed.h5", "w") as f:
        dset = SL.open_slice_dataset(f.create_group("vol"), rec)
        writer = SL.PlaneWriter(dset)
        assert writer.depth == dset.chunks[0], "the buffer must be one chunk row"
        assert 1 < writer.depth < n_planes, (
            f"the fixture must span several chunk rows (depth {writer.depth} of "
            f"{n_planes} planes) or the buffered write path is never exercised"
        )
        for plane in planes:
            writer.append(plane)
        writer.close()
        new_chunks, new_shape, new_dtype = dset.chunks, dset.shape, dset.dtype
        stored = dset[()]

    assert (new_shape, new_dtype, new_chunks) == (old_shape, old_dtype, old_chunks)
    assert stored.tobytes() == np.stack(planes, axis=0).tobytes()


def test_plane_writer_refuses_a_short_sweep(tmp_path):
    """A sized dataset left partly unwritten is an unfinished product."""
    with h5py.File(tmp_path / "short.h5", "w") as f:
        dset = SL.open_slice_dataset(f.create_group("vol"), _slice_rec(11, 20, 30))
        writer = SL.PlaneWriter(dset)
        writer.append(np.zeros((20, 30), np.float32))
        with pytest.raises(RuntimeError, match="sized for 11 planes"):
            writer.close()


def test_unwritten_planes_read_as_nan_not_zeros(tmp_path):
    """An aborted sweep must leave NaN where it never sampled, never zeros.

    Sizing the dataset up front means a run killed mid-sweep leaves a group whose
    unwritten planes are readable, where the old whole-stack write left no group
    at all — and `PlaneWriter.close()`'s count check cannot fire on that path,
    because nothing calls it. HDF5's default fill is zero, and a plane of zeros
    reads as *data*: `profiles` averages it into a CSV with no skip and no note.
    NaN is what every reader here already treats as "no sample".

    Simulated the way it really happens — the writer is abandoned mid-sweep with
    no `close()` — rather than by trusting the creation-property list.
    """
    with h5py.File(tmp_path / "aborted.h5", "w") as f:
        dset = SL.open_slice_dataset(f.create_group("vol"), _slice_rec(11, 20, 30))
        assert np.isnan(dset.fillvalue), "the dataset must be NaN-filled, not zero-filled"
        writer = SL.PlaneWriter(dset)
        for _ in range(3):
            writer.append(np.full((20, 30), 7.0, np.float32))
        writer.flush()  # 3 of 11 planes on disk; the run then dies
        stored = dset[()]
    assert np.all(stored[:3] == 7.0)
    assert np.isnan(stored[3:]).all(), "unwritten planes came back as data, not as NaN"


# The rungs the two peak tests below pin, and the budgets that reach them. Both
# are asserted at run time by `_sweep_rung_is_in_core`, never assumed: 64 MiB was
# once documented here as the in-core rung and in fact streamed.
_SWEEP_RUNGS = {"in_core": 1 << 30, "streamed": 4096}

# Holding the sweep's stack ONCE must already exceed the limit by this much, or
# the test would pass on code that holds it. The pre-change peak was about twice
# the stack again, since `np.stack(planes)` held the list and the stack together.
_STACK_OVER_LIMIT = 1.4
_SWEEP_PEAK_LIMIT = 350 << 20


@pytest.mark.parametrize("rung", sorted(_SWEEP_RUNGS))
def test_slices_never_holds_the_whole_plane_stack(tmp_path, rung):
    """A sweep's planes must be written as they are sampled, not accumulated.

    The Z-blocked gather bounded the stage's *input*; on the shipped default
    geometry the *output* is the larger term — ~172 planes of 2528x1789 float32
    is 2.90 GiB for `oblique_full` alone against a 1.15 GiB volume, and
    `np.stack(planes)` doubled it transiently. This fixture reproduces that
    shape at 1/6th the size and asserts the peak no longer carries it.

    **Both rungs, because both matter.** An earlier version measured only what
    it called the in-core rung, on the argument that a 1.15 GiB STO2 volume fits
    an 8 GB machine's 1.65-2.58 GiB budget. That was wrong twice over: measured,
    `align_volume_streamed` returns `block_layers == 1` for STO2 at that budget
    (it prices the alignment chain at ~41 B per float64 element and gets a 254 MB
    share, against a 285 MB minimum), so the target machine takes the
    **streaming** rung — and the budget that test pinned took the streaming rung
    too, so its own claim about which code it measured was false. Both rungs are
    now measured, and which one each budget reaches is asserted rather than
    described.

    Measured in the real child, one 8x256x128 volume (2 MB) and a 201-plane
    sweep of 801x801 float32 (2.45 MiB a plane, 492 MiB of stack), PNGs off:

        rung        before      after
        in-core     1117.7      205.0 MiB
        streamed    1105.8      239.3 MiB

    The 350 MiB limit sits between them with ~3.2x of failing margin and
    1.5-1.7x of passing margin. `_budget_bytes` is pinned so the figures do not
    depend on the runner's RAM, and the stack-vs-limit precondition below is
    what stops the test going insensitive if the fixture is ever shrunk.
    """
    from tests.peak_rss import assert_peak_under

    budget = _SWEEP_RUNGS[rung]
    proc, raw, _bytes = _peak_setup(tmp_path, *_SWEEP_SHAPE)
    params, n_planes, (nv, nu) = _sweep_peak_params(proc, raw, tmp_path / f"sl_{rung}")
    params["_budget_bytes"] = budget

    assert _sweep_rung_is_in_core(proc, raw, tmp_path, budget) == (rung == "in_core"), (
        f"budget {budget} does not reach the {rung} rung — this test would then "
        "measure the same code twice and claim to have covered both"
    )
    stack_bytes = n_planes * nv * nu * 4
    assert stack_bytes > _STACK_OVER_LIMIT * _SWEEP_PEAK_LIMIT, (
        f"the sweep's stack is only {stack_bytes / (1 << 20):.1f} MiB against a "
        f"{_SWEEP_PEAK_LIMIT / (1 << 20):.0f} MiB limit — holding it whole would "
        "still pass, so this test would measure nothing"
    )

    result = assert_peak_under("dfxm.stages.slices:run", params, _SWEEP_PEAK_LIMIT, timeout=900)
    assert result.volume_ids == ["strain"], "the run must actually have sliced a volume"
    assert result.n_planes_total == n_planes
    with h5py.File(result.output_h5, "r") as f:
        assert f["strain"]["wide"]["slices"].dtype == np.float32
    _assert_sweep_samples_the_volume(result.output_h5, n_planes)


def test_slices_run_bounds_the_streamed_sweep(tmp_path):
    """`run` must hand the gather a real `max_resident_bytes`, not `None`.

    Everything under `iter_planes_streamed` is bounded and pinned, and none of it
    matters if the one call site stops asking for a bound: dropping
    `max_resident_bytes=_sweep_resident_bytes(...)` in `run` puts the whole sweep
    back in memory on the streaming rung — 172 resident planes, 2.90 GiB, on the
    shipped default — and leaves every other test in this module green. So the
    call site is pinned here, by counting the walks of the Z-block stream that
    `run` actually causes.

    Not by asserting the argument was passed, which a refactor renames away, and
    not by asserting the run merely finished. `n_planes` walks means every plane
    got its own traversal, i.e. the budget really sized the batch; `None` would
    be exactly one walk however long the sweep.
    """
    proc, raw, _bytes = _peak_setup(tmp_path, 4, 24, 24)
    params, n_planes, (nv, nu) = _sweep_peak_params(
        proc, raw, tmp_path / "sl_bound", half=2.0, d=0.2, step=0.5, span=1.0
    )
    params["_budget_bytes"] = 4096
    assert n_planes >= 4, "a one-plane sweep cannot tell the two batchings apart"

    rungs, walks = [], []
    real = SL.prepare_volume

    def spy(*args, **kwargs):
        prep = real(*args, **kwargs)
        rungs.append(prep["data"] is not None)
        inner = prep["blocks"]
        # Wrapped AFTER prepare_volume returns, so the colour-limit reductions
        # it ran internally are not counted — only the sampling walks.
        prep["blocks"] = lambda: (walks.append(1), inner())[1]
        return prep

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(SL, "prepare_volume", spy)
        result = SL.run(params)
    finally:
        monkeypatch.undo()

    assert rungs == [False], "the tiny budget must take the streaming rung"
    assert result.n_planes_total == n_planes
    plane_bytes = nv * nu * 4
    assert (
        SL.sweep_batch_size(n_planes, plane_bytes, SL._sweep_resident_bytes(4096, plane_bytes, 1))
        == 1
    )
    assert sum(walks) == n_planes, (
        f"the sweep walked the Z stream {sum(walks)} times for {n_planes} planes; "
        "1 means `run` passed no `max_resident_bytes` and held the whole sweep"
    )


def test_rss_floor_covers_the_measured_process_image(tmp_path):
    """`RSS_FLOOR_BYTES` must not sit below what this stage's child costs.

    The formula in `advice.working_set_budget_bytes` is pinned by
    `tests/test_common_advice.py`, but a *value* is not a formula: setting
    `RSS_FLOOR_BYTES = 1` passes every one of those and the budget it derives is
    then far too large. Only a live measurement catches that, and
    `assert_floor_covers` brackets from both sides so an inflated constant fails
    too.

    **PNGs on**, which is the stage default: the first figure a run builds
    imports the whole of matplotlib's rendering stack (measured 103.9 MiB
    without it against 193.5 MiB with it), and every volume after the first
    streams with that resident.
    """
    from tests.peak_rss import assert_floor_covers

    proc, raw, data_bytes = _peak_setup(tmp_path, *(4, 6, 8))
    params = {**_peak_params(proc, raw, tmp_path / "sl_floor"), "save_png": True}
    assert_floor_covers(
        SL.RSS_FLOOR_BYTES,
        "dfxm.stages.slices:run",
        params,
        data_bytes=data_bytes,
    )


def test_reduction_working_set_multiple_is_rounded_the_safe_way():
    """The budget divisor must cover what really holds a block, rounded UP.

    `align_volume_streamed` prices the alignment chain and **nothing
    downstream**. The colour-limit reductions layered on its blocks hold, per
    float64 element at their peak, `dtype.itemsize + 8 * (retained + 1) + 1` =
    41 B — the same figure `align_volume_streamed` computes for its own cached
    median — plus one more block for the centred `b - center` the stream yields
    through. The constant is a **divisor**, so a larger number means a smaller
    budget and rounding *down* permits more than was counted; that is exactly the
    defect `visualize` shipped (6 against the same 6.125) and had to correct.

    Nothing else pins this number, and an unpinned constant in the permissive
    direction is the failure mode this project has now met a dozen times. The
    outcome is checked separately by `test_slices_peak_stays_under_budget`.
    """
    import math

    per_element = 8 + 8 * (3 + 1) + 1  # the block, the rank search's arrays, the mask
    counted = per_element / 8 + 1  # …expressed in blocks, plus the centred copy
    assert counted == 6.125
    assert SL.REDUCTION_WORKING_SET_MULTIPLE == math.ceil(counted)


def test_aligned_block_budget_never_buys_more_than_the_budget():
    """An `aligned` source's block budget converts currency and rounds down.

    `iter_blocks` sizes a block by its bytes **in the stored dtype**, but the
    stage holds far more per block than that: the float64 upcast, the centred
    copy and the reductions' 41 B per float64 element. Handing the working-set
    budget over raw would buy a block several times too large — and the stored
    dtype here is float32, where the error is largest.
    """

    class _Dset:
        dtype = np.dtype(np.float32)
        shape = (8, 16, 16)

    budget = 1 << 20
    block_bytes = SL._aligned_block_budget(_Dset(), budget)
    assert block_bytes == budget * 4 // (4 + 41 + 8)
    # What that block actually costs, in the currency the budget is priced in.
    elements = block_bytes // 4
    assert elements * (4 + 41 + 8) <= budget


def test_probe_slack_covers_the_rounding_it_guards_and_is_load_bearing(tmp_path):
    """`_PROBE_SLACK_LAYERS` must cover the edge-probe rounding — and be needed.

    `k` is affine in `(u, v)`, so its extremes over the grid sit on the grid's
    edges and the row/column probes bound the interior exactly — *in exact
    arithmetic*. They are computed in floating point, and the slack is what
    stops a sample at a block boundary being pruned away into a NaN.

    Two halves, because either alone would be satisfied by a constant that does
    nothing. First, measure the discrepancy the slack guards over a randomised
    family of planes and require it to be negligible against one layer. Second,
    require that a **negative** slack actually breaks gather/in-core equality —
    without that, `0.0` and `1.0` and any other value would pass alike and the
    constant would be pinned by nothing.
    """
    volume = _gather_volume(seed=11)
    prep_in_core = _in_core_prep(volume)
    rng = np.random.default_rng(4)

    worst = 0.0
    for _ in range(200):
        normal = rng.standard_normal(3)
        if np.linalg.norm(normal) < 1e-6:
            continue
        u_hat, v_hat, _n = SL.build_basis(normal)
        origin = rng.uniform(-2.0, 14.0, 3)
        u_um, v_um = SL._plane_axes(8.0, 8.0, 0.5, 0.5)
        interior = SL._plane_coords(prep_in_core, origin, u_hat, v_hat, u_um, v_um)[0]
        row_k = SL._plane_k(prep_in_core, origin, u_hat, v_hat, u_um[[0, -1]], v_um)
        col_k = SL._plane_k(prep_in_core, origin, u_hat, v_hat, u_um, v_um[[0, -1]])
        # How far outside the probes' hull any interior sample lands, in layers.
        for lo, hi in (
            (row_k.min(axis=1)[:, None], row_k.max(axis=1)[:, None]),
            (col_k.min(axis=0)[None, :], col_k.max(axis=0)[None, :]),
        ):
            worst = max(worst, float(np.max(lo - interior)), float(np.max(interior - hi)))
    assert worst < 1e-9, f"edge probes miss interior samples by {worst} layers"
    assert SL._PROBE_SLACK_LAYERS >= 1.0 > worst

    # …and the slack is load-bearing: a negative one prunes real samples away.
    budget = volume.nbytes // 6
    args = _gather_args()
    reference = SL.sample_plane(prep_in_core, *args)[0]
    original = SL._PROBE_SLACK_LAYERS
    try:
        SL._PROBE_SLACK_LAYERS = -0.5
        starved = SL.sample_plane_streamed(_streamed_prep(volume, budget), *args)[0]
    finally:
        SL._PROBE_SLACK_LAYERS = original
    assert not np.array_equal(starved, reference, equal_nan=True), (
        "a negative slack changed nothing — the pruning is not being exercised, "
        "so this test cannot see a slack that is too small"
    )
    assert np.array_equal(
        SL.sample_plane_streamed(_streamed_prep(volume, budget), *args)[0],
        reference,
        equal_nan=True,
    ), "restoring the slack must restore equality"


# -- the no-samz chain --------------------------------------------------------
def _no_motor_dataset(nz=9, ny=7, nx=11, seed=12):
    return np.random.default_rng(seed).standard_normal((nz, ny, nx)).astype(np.float32)


@pytest.mark.parametrize("with_samy", [False, True])
@pytest.mark.parametrize("roi", [None, (2, 9)])
def test_no_samz_chain_blocks_and_matches_the_whole_volume_form(with_samy, roi):
    """With no samz the remaining chain is per-layer, so it blocks exactly.

    `align_volume_streamed` cannot serve this case (it always interpolates, and
    would raise on `samz[0]`), and `visualize`/`paraview` answer it with a whole
    in-core volume. This stage blocks it instead, so the path a run reaches by
    *misconfiguration* is not the one path with no memory bound. The reference
    here is the whole-volume chain this replaced, computed inline.
    """
    from dfxm.common import alignment as A

    dset = _no_motor_dataset()
    cfg = {"roi_x": roi, "roi_y": None}
    samy = np.array([0.0, 0.00001, 0.000025, 0.00004, 0.00006, 0.00007, 0.00009, 0.0001, 0.00012])
    samy = samy if with_samy else np.array([])
    kwargs = dict(scale_x=0.152, samy_direction=-1, take_abs=True)

    reference = np.abs(np.asarray(dset, dtype=np.float64))
    reference = A.apply_roi_3d(reference, roi, None)
    if with_samy:
        reference = A.apply_samy_shifts_to_volume(reference, samy, 0.152, -1)

    shape, dtype, scale_z = SL._unaligned_shape(dset, cfg, samy, 0.152, -1)
    assert shape == reference.shape and dtype == np.float64 and scale_z == 2.0

    counts = []
    for budget in (1 << 30, dset.nbytes // 4, 1):
        blocks = list(SL._unaligned_blocks(dset, cfg, samy, budget_bytes=budget, **kwargs)())
        counts.append(len(blocks))
        got = np.concatenate([b for _sl, b in blocks], axis=0)
        assert np.array_equal(got, reference, equal_nan=True), f"budget {budget}"
    assert counts[0] == 1 and counts[-1] == dset.shape[0], (
        f"the budgets must actually change the blocking, got {counts}"
    )


def test_no_samz_prepare_volume_is_not_forced_in_core(tmp_path):
    """The no-samz branch must obey the budget like every other path.

    It used to set `fits = True` unconditionally, which made a misconfigured run
    the only unbounded one. Both rungs must be reachable, and must agree.
    """
    import contextlib

    proc = tmp_path / "proc"
    proc.mkdir()
    with h5py.File(proc / "stacked_volumes.h5", "w") as f:
        f.create_dataset("chi/Center of mass", data=_no_motor_dataset(nz=12, ny=8, nx=10))
    p = {**SL.STAGE.defaults(), "center_method": "midrange"}
    cfg = {
        "h5_path": str(proc / "stacked_volumes.h5"),
        "dataset_path": "chi/Center of mass",
        "kind": "mosa_com",
        "source": "stacked",
        "raw_root": "",  # no folders -> extract_motor_positions returns empty samy AND samz
        "raw_pattern": "",
        "roi_x": None,
        "roi_y": None,
    }

    def prepared(budget):
        with contextlib.ExitStack() as opened:
            prep = SL.prepare_volume(
                cfg, p, 0.152, 0.385, -1, style=None, stack=opened, budget_bytes=budget
            )
            return (
                prep["data"] is not None,
                np.concatenate([b for _sl, b in prep["blocks"]()], axis=0),
                (prep["vmin_raw"], prep["vmax_raw"]),
                prep["scale_z"],
            )

    in_core_fits, in_core, in_core_limits, sz = prepared(1 << 30)
    streamed_fits, streamed, streamed_limits, sz2 = prepared(2048)
    assert in_core_fits and not streamed_fits, "both rungs must be reachable here"
    assert in_core_limits == streamed_limits
    assert sz == sz2 == SL._NO_MOTOR_Z_STEP_UM
    assert np.array_equal(in_core, streamed, equal_nan=True)


def test_run_stops_on_an_impossible_roi_instead_of_skipping_every_volume(tmp_path):
    """The per-volume handler catches `ValueError`, which `StageUserError`
    subclasses — so without an explicit re-raise an impossible ROI is reported
    as one skip per volume behind a run that "succeeded" and wrote nothing,
    with the hint dropped."""
    from dfxm.common.errors import StageUserError

    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "slices_json": slices_json,
        "output_dir": str(out),
        "align_roi_y": "9,3",  # inverted: crops every volume to nothing
    }
    with pytest.raises(StageUserError) as exc:
        SL.run(params)
    assert "9,3" in str(exc.value) and exc.value.hint
