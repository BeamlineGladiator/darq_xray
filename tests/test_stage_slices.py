"""Tests for dfxm.stages.slices — plane geometry/sampling and end-to-end output."""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

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
    assert len(fig.axes[0].patches) == 0


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
    # legacy draws the (black) scale bar -> at least one patch
    assert len(fig.axes[0].patches) >= 1


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
