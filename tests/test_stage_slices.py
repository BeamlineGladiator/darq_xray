"""Tests for dfxm.stages.slices — plane geometry/sampling and end-to-end output."""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest
from matplotlib.offsetbox import AnchoredOffsetbox

from dfxm.common.plotting import PlotStyle
from dfxm.stages import slices as S

L, NY, NX = 4, 6, 8


# -- geometry / sampling ------------------------------------------------------
def test_build_basis_orthonormal_right_handed():
    u, v, n = S.build_basis((0.3, 0.0, 0.95))
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
    u, v, _ = S.build_basis((0, 0, 1))
    np.testing.assert_allclose(u, [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(v, [0, 1, 0], atol=1e-12)

    u, v, _ = S.build_basis((0.647648, 0, 0.761939))  # default oblique_full normal
    np.testing.assert_allclose(v, [0, 1, 0], atol=1e-9)  # vertical axis = world Y
    assert u[0] > 0.5  # horizontal axis points +X-ish
    assert abs(u[1]) < 1e-9


def test_slice_plane_offsets():
    np.testing.assert_allclose(S.slice_plane_offsets({"sweep_step_um": None}), [0.0])
    off = S.slice_plane_offsets({"sweep_step_um": 2.0, "sweep_start_um": 0.0, "sweep_stop_um": 6.0})
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
    u_hat, v_hat, _ = S.build_basis((0, 0, 1))  # u->X, v->Y
    s, u_um, v_um = S.sample_plane(prep, (3.0, 2.0, 1.0), u_hat, v_hat, 2.0, 1.0, 1.0, 1.0)
    # world X at column c = origin_x + u_um[c]; value == that X
    for c, u in enumerate(u_um):
        col = s[:, c]
        col = col[np.isfinite(col)]
        if col.size:
            np.testing.assert_allclose(col, 3.0 + u, atol=1e-6)


def test_resolve_auto_extent_fits_box():
    box = (0.0, 10.0, 0.0, 8.0, 0.0, 6.0)
    out = S.resolve_auto_extent(
        {"name": "z", "normal": [0, 0, 1], "origin": [0, 0, 0], "extent": "auto", "du": 1.0}, box
    )
    assert out["half_u"] > 0 and out["half_v"] > 0
    assert out["sweep_start_um"] <= 0.0 <= out["sweep_stop_um"]
    assert out["sweep_step_um"] == 1.0


def test_resolve_auto_extent_default_step_uses_pixel_scale():
    """No du / no sweep_step_um -> step defaults to the configured pixel scale."""
    box = (0.0, 10.0, 0.0, 8.0, 0.0, 6.0)
    sl = {"name": "z", "normal": [0, 0, 1], "origin": [0, 0, 0], "extent": "auto"}
    out = S.resolve_auto_extent(sl, box, default_du=0.385)
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
    samy = np.array([0.0, 0.001, 0.0025, 0.004])
    samz = np.array([0.0, 0.001, 0.0021, 0.0035])
    for base in ("mosa", "strain"):
        for i in range(L):
            folder = raw / f"{base}__{i + 1}"
            folder.mkdir()
            with h5py.File(folder / f"{base}__{i + 1}.h5", "w") as f:
                f.create_dataset("1.1/instrument/positioners/samy", data=samy[i])
                f.create_dataset("1.1/instrument/positioners/samz", data=samz[i])
    return proc, raw


def test_run_writes_consolidated_h5_and_pngs(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    # one explicit single plane (controlled shape) + one auto z-sweep
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null},'
        '{"name":"zsweep","normal":[0,0,1],"origin":[0,0,0],"extent":"auto","sweep_step_um":1.0}]'
    )
    res = S.run(
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
    res = S.run({"mosa_volume_file": "", "strain_volume_file": "", "aligned_rocking_file": ""})
    assert any("no input volumes" in s for s in res.skipped)


def test_run_rejects_nonpositive_du(tmp_path):
    proc, raw = _setup(tmp_path)
    bad = '[{"name":"mid","normal":[0,0,1],"origin":[0,0,0],"half_u":1,"half_v":1,"du":0,"dv":0.2}]'
    with pytest.raises(ValueError, match="du must be > 0"):
        S.run(
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
        S.run(
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
    fig = S.build_slice_figure(
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
    fig = S.build_slice_figure(
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
    fig = S.build_slice_figure(
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


def test_run_resolves_cmaps_from_injected_style(tmp_path):
    proc, raw = _setup(tmp_path)
    params = _minimal_params(proc, raw, tmp_path / "sl_styled")
    params["plot_style"] = {"cmap_mosa_com": "viridis", "cmap_raw": "bone", "font_scale": 1.0}
    res = S.run(params)
    with h5py.File(res.output_h5, "r") as f:
        assert f["mosa_com_chi"].attrs["cmap"] == "viridis"
        assert f["raw_sum"].attrs["cmap"] == "bone"
        assert f["strain"].attrs["cmap"] == "RdBu_r"  # default from PlotStyle


def test_run_without_style_uses_group_defaults(tmp_path):
    proc, raw = _setup(tmp_path)
    res = S.run(_minimal_params(proc, raw, tmp_path / "sl_default"))
    with h5py.File(res.output_h5, "r") as f:
        assert f["mosa_com_chi"].attrs["cmap"] == "fast"  # real fast, no coolwarm fallback
        assert f["raw_sum"].attrs["cmap"] == "gray"
        assert f["mosa_fwhm_chi"].attrs["cmap"] == "magma"


def test_figures_re_resolve_cmap_by_kind(tmp_path):
    from dfxm.common.plotting import PlotStyle

    proc, raw = _setup(tmp_path)
    res = S.run(_minimal_params(proc, raw, tmp_path / "sl_figs"))
    specs = S.figures(res, {})
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
    res = S.run(params)
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
    res = S.run(
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
    res = S.run(
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
    cat = S.replot_catalog(str(h5))
    by_vid = {(e.volume_id, e.slice_name): e for e in cat}
    assert set(by_vid) == {("raw_sum", "plane_a"), ("strain", "plane_a")}
    assert by_vid[("strain", "plane_a")].n_planes == 3
    assert by_vid[("strain", "plane_a")].offsets_um == [-1.0, 0.0, 1.0]


def test_render_replot_writes_selected_planes_under_subfolders(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    out = tmp_path / "replots"
    # strain: only planes 0 and 2; raw_sum: all planes (None)
    written = S.render_replot(
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
    fig = S._rebuild_plane_figure(str(h5), "strain", "plane_a", 1, style=None, clim=(-5.0, 5.0))
    im = fig.axes[0].images[0]
    assert im.norm.vmin == -5.0 and im.norm.vmax == 5.0


def test_render_replot_roi_crops_slice(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    # crop to a sub-window; the rebuilt image must have the cropped shape
    fig = S._rebuild_plane_figure(str(h5), "strain", "plane_a", 1, style=None, roi=(0, 2, 0, 2))
    im = fig.axes[0].images[0]
    assert im.get_array().shape == (2, 2)


def test_render_replot_roi_empty_crop_skipped(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))
    written = S.render_replot(
        str(h5),
        [("strain", "plane_a", None)],
        style=None,
        clim=None,
        out_dir=str(tmp_path / "r"),
        roi=(2, 2, 0, 3),
    )
    assert written == []


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
    res = S.run(
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
