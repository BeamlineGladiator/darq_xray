"""Qt-free 3-D scene core — pure numpy parts (no pyvista, no GL)."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm.common import render3d as R3


def _vol():
    # (Z=2, Y=4, X=6) ramp with one NaN
    v = np.arange(48, dtype=float).reshape(2, 4, 6)
    v[0, 0, 0] = np.nan
    return v


def test_downsample_volume_block_means_yx_only():
    v = _vol()
    d = R3.downsample_volume(v, 2)
    assert d.shape == (2, 2, 3)  # Z untouched, Y/X halved
    # block (z=1, rows 0-1, cols 0-1) mean
    assert d[1, 0, 0] == pytest.approx(np.nanmean(v[1, 0:2, 0:2]))
    assert np.array_equal(R3.downsample_volume(v, 1), v, equal_nan=True)


def test_threshold_mask_nans_outside_window():
    v = _vol()
    t = R3.threshold_mask(v, (10.0, 20.0))
    assert np.isnan(t[0, 0, 1])  # value 1 < 10 -> NaN
    assert t[0, 2, 3] == 15.0  # inside window kept
    assert np.array_equal(R3.threshold_mask(v, None), v, equal_nan=True)


def test_clip_mask_halves_volume_on_plane():
    v = np.ones((2, 4, 6))
    # plane through x=3 µm (spacing sx=1), normal +x: keep x >= 3 µm side
    c = R3.clip_mask(v, (1.0, 1.0, 1.0), (3.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert np.isnan(c[:, :, :3]).all() and (c[:, :, 3:] == 1.0).all()


def test_auto_clim_and_resolved_clim():
    v = _vol()
    lo, hi = R3.auto_clim(v)
    assert lo < hi
    s = R3.Scene3D(volume=v, spacing=(1, 1, 1))
    assert s.resolved_clim() == pytest.approx((lo, hi))
    s2 = R3.Scene3D(volume=v, spacing=(1, 1, 1), clim=(0.0, 5.0))
    assert s2.resolved_clim() == (0.0, 5.0)


def test_log_valid():
    assert R3.log_valid((0.1, 2.0))
    assert not R3.log_valid((0.0, 2.0))
    assert not R3.log_valid((-1.0, 2.0))
    assert not R3.log_valid(None)


def test_scene_prepared_applies_downsample_threshold_clip():
    v = _vol()
    s = R3.Scene3D(volume=v, spacing=(1.0, 2.0, 3.0), downsample=2, threshold=(10.0, 40.0))
    out, spacing = s.prepared()
    assert out.shape == (2, 2, 3)
    assert spacing == (2.0, 4.0, 3.0)  # sx, sy scaled; sz untouched
    assert np.isnan(out[0, 0, 0])  # block mean 3.75 < 10 -> thresholded


def test_orbit_positions_absolute_and_equidistant():
    base = ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    poses = R3.orbit_positions(base, 0.0, 4)
    assert len(poses) == 4
    eyes = [np.array(p[0]) for p in poses]
    # frame 0 with no elevation reproduces the base eye
    assert eyes[0] == pytest.approx(np.array(base[0]))
    # all eyes stay on the orbit sphere around the focal point
    for e in eyes:
        assert np.linalg.norm(e) == pytest.approx(10.0)
    # 90° steps about +y: eye moves into the x-z plane
    assert abs(eyes[1][0]) == pytest.approx(10.0, abs=1e-6)
    # focal + up unchanged
    for p in poses:
        assert p[1] == (0.0, 0.0, 0.0) and p[2] == (0.0, 1.0, 0.0)


def test_orbit_positions_elevation_tilts_eye():
    base = ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    poses = R3.orbit_positions(base, 20.0, 2)
    # elevation lifts the eye along +y (view-up side), distance preserved
    assert poses[0][0][1] > 0.0
    assert np.linalg.norm(np.array(poses[0][0])) == pytest.approx(10.0)


# --- pyvista-dependent (no GL needed: grid building only) -----------------


def test_grid_for_scene_surface_thresholds_nans():
    pytest.importorskip("pyvista")
    v = _vol()
    s = R3.Scene3D(volume=v, spacing=(1, 1, 1), mode="surface")
    kind, mesh = R3._grid_for_scene(s)
    assert kind == "mesh"
    assert mesh.n_cells == 47  # 48 voxels, 1 NaN thresholded out


def test_grid_for_scene_volume_keeps_grid_shape():
    pytest.importorskip("pyvista")
    v = _vol()
    s = R3.Scene3D(volume=v, spacing=(1.0, 2.0, 3.0), mode="volume")
    kind, grid = R3._grid_for_scene(s)
    assert kind == "volume"
    assert tuple(grid.dimensions) == (7, 5, 3)  # cells+1 in (X, Y, Z)
    assert grid.spacing == (1.0, 2.0, 3.0)
    # NaN voxel uploaded as 0 (transparent under the transfer function)
    assert float(grid.cell_data["values"].min()) == 0.0


def test_grid_for_scene_empty_returns_none():
    pytest.importorskip("pyvista")
    s = R3.Scene3D(volume=np.full((2, 3, 4), np.nan), spacing=(1, 1, 1))
    assert R3._grid_for_scene(s) is None


def test_contour_meshes_convert_cell_data_to_point_data():
    # regression: VTK's contour filter rejects cell data ("Contour filter only
    # works on point data"), so isosurface mode must interpolate to points.
    pytest.importorskip("pyvista")
    s = R3.Scene3D(volume=_vol(), spacing=(1, 1, 1), mode="isosurface", clim=(5.0, 40.0))
    kind, grid = R3._grid_for_scene(s)
    assert kind == "mesh"
    assert list(grid.point_data.keys()) == []  # scalars live on the cells
    contours = R3._contour_meshes(grid, (5.0, 40.0), 3)
    assert len(contours) == 3  # interior levels only
    assert all(mesh.n_points > 0 for _level, mesh in contours)


def test_contour_meshes_no_levels_when_none_requested():
    pytest.importorskip("pyvista")
    s = R3.Scene3D(volume=_vol(), spacing=(1, 1, 1), mode="isosurface")
    _kind, grid = R3._grid_for_scene(s)
    assert R3._contour_meshes(grid, (5.0, 40.0), 0) == []


def test_grid_for_scene_log_uploads_log10_values():
    pytest.importorskip("pyvista")
    v = np.full((1, 2, 2), 100.0)
    s = R3.Scene3D(volume=v, spacing=(1, 1, 1), mode="volume", clim=(1.0, 100.0), log_scale=True)
    kind, grid = R3._grid_for_scene(s)
    assert float(grid.cell_data["values"].max()) == pytest.approx(2.0)  # log10(100)


# --- compositor (Agg, no pyvista) -----------------------------------------


def _fake_render():
    return np.full((120, 200, 3), 255, dtype=np.uint8)


def test_scene_figure_extent_is_micron_true():
    fig, ax, im = R3.scene_figure(
        _fake_render(), px_per_um=2.0, cbar_label="Misorientation (°)", clim=(0.0, 1.0)
    )
    # 200 px wide at 2 px/µm -> 100 µm x-extent (exact scale-bar basis)
    assert ax.get_xlim() == (0.0, 100.0)
    assert ax.get_ylim() == (0.0, 60.0)
    assert len(fig.axes) == 2  # image + colorbar


def test_scene_figure_log_uses_lognorm():
    from matplotlib.colors import LogNorm

    fig, ax, im = R3.scene_figure(
        _fake_render(), px_per_um=2.0, cbar_label="I", clim=(1.0, 100.0), log_scale=True
    )
    # the colorbar was built from a ScalarMappable with LogNorm
    assert isinstance(fig._scene_mappable.norm, LogNorm)


def test_scene_figure_saves_png(tmp_path):
    fig, ax, im = R3.scene_figure(
        _fake_render(), px_per_um=2.0, cbar_label="ε", group="strain", clim=(-1e-3, 1e-3)
    )
    out = tmp_path / "f.png"
    fig.savefig(out, dpi=100)
    assert out.stat().st_size > 0
