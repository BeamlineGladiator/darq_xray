"""Tests for darq_xray.stages.mosaicity, incl. golden equivalence vs the legacy
stack_h5_darfix_volumes script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from darq_xray.stages import mosaicity as M

PATHS = {
    "/entry/chi/Center of mass/Center of mass": "chi_com",
    "/entry/chi/FWHM/FWHM": "chi_fwhm",
    "/entry/mu/Center of mass/Center of mass": "mu_com",
    "/entry/mu/FWHM/FWHM": "mu_fwhm",
}


def _write_mosa(folder, layer_idx, ny=8, nx=10):
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(layer_idx)
    with h5py.File(os.path.join(folder, "maps.h5"), "w") as f:
        for path in PATHS:
            f.create_dataset(path, data=rng.standard_normal((ny, nx)) + layer_idx)


def _make_root(tmp_path, names=("layer__1", "layer__2", "layer__10")):
    root = tmp_path / "root"
    for i, name in enumerate(names):
        _write_mosa(str(root / name), i)
    return root


def test_batch_stacks_four_volumes_in_layer_order(tmp_path):
    root = _make_root(tmp_path)
    res = M.run({"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"})
    assert res.n_layers == 3
    assert res.layers == ["layer__1", "layer__2", "layer__10"]  # natural sort
    with h5py.File(res.stacked_path, "r") as f:
        for grp, ds in [
            ("chi", "Center of mass"),
            ("chi", "FWHM"),
            ("mu", "Center of mass"),
            ("mu", "FWHM"),
        ]:
            assert f[f"/{grp}/{ds}"].shape == (3, 8, 10)
        assert f.attrs["num_layers"] == 3


def test_mosaicity_volume_matches_layerwise_stack(tmp_path):
    """The incremental writer's volumes equal np.stack of the per-layer maps."""
    root = _make_root(tmp_path)
    res = M.run({"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"})
    assert res.layers == ["layer__1", "layer__2", "layer__10"]

    src = "/entry/chi/Center of mass/Center of mass"
    per_layer = []
    for name in res.layers:
        with h5py.File(str(root / name / "maps.h5"), "r") as f:
            per_layer.append(f[src][:])
    expected = np.stack(per_layer, axis=0)
    with h5py.File(res.stacked_path, "r") as f:
        stored = f["/chi/Center of mass"][:]
        assert f.attrs["source_folders"].split("\n") == res.layers
    assert stored.dtype == expected.dtype
    assert np.array_equal(stored, expected, equal_nan=True)
    assert res.datasets["/chi/Center of mass"] == (3, 8, 10)


def test_mosaicity_no_layers_leaves_no_stacked_file(tmp_path):
    """Every folder skipped -> abort inside the with-block; no file, no .part."""
    root = tmp_path / "root"
    os.makedirs(root / "layer__1")  # matches the pattern but has no maps.h5
    res = M.run({"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"})
    assert res.n_layers == 0 and res.stacked_path is None
    assert list(root.glob("*.h5")) == []
    assert list(root.glob("*.part")) == []


def test_mosaicity_absent_dataset_creates_no_group(tmp_path):
    """A layer set missing every mu map yields no /mu group at all — the old
    code called require_group for all four routings up front."""
    root = tmp_path / "root"
    folder = root / "layer__1"
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(0)
    with h5py.File(os.path.join(folder, "maps.h5"), "w") as f:
        f.create_dataset("/entry/chi/Center of mass/Center of mass", data=rng.random((4, 5)))
    res = M.run({"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"})
    assert set(res.datasets) == {"/chi/Center of mass"}
    with h5py.File(res.stacked_path, "r") as f:
        assert list(f.keys()) == ["chi"]
        assert list(f["chi"].keys()) == ["Center of mass"]


def test_single_mode_missing_folder_skips_rather_than_raising(tmp_path):
    """A stale 'Input folder' must come back as a skip banner, not a raw h5py
    FileNotFoundError from the writer eagerly creating its part file."""
    missing = tmp_path / "nope"
    res = M.run({"mode": "single", "input_folder": str(missing)})
    assert res.n_layers == 0 and res.stacked_path is None
    assert res.skipped == ["nope: maps.h5 not found"]
    assert not missing.exists()


def test_batch_missing_maps_file_records_reason(tmp_path):
    root = tmp_path / "root"
    _write_mosa(str(root / "layer__1"), 0)
    os.makedirs(root / "layer__2")  # matches the pattern but has no maps.h5
    res = M.run({"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"})
    assert res.n_layers == 1
    assert res.skipped == ["layer__2: maps.h5 not found"]


def test_single_mode(tmp_path):
    folder = tmp_path / "layer__1"
    _write_mosa(str(folder), 0)
    res = M.run({"mode": "single", "input_folder": str(folder)})
    assert res.n_layers == 1
    assert set(res.datasets) == {
        "/chi/Center of mass",
        "/chi/FWHM",
        "/mu/Center of mass",
        "/mu/FWHM",
    }


def test_matches_legacy_stack_h5_darfix_volumes(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "stack_h5_darfix_volumes.py").exists():
        pytest.skip("legacy stack_h5_darfix_volumes.py not found")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import stack_h5_darfix_volumes as legacy

    root = _make_root(tmp_path)
    folders = legacy.find_matching_folders(str(root), "layer__*")
    cfg = {
        "chi_Center_of_mass": "/entry/chi/Center of mass/Center of mass",
        "chi_FWHM": "/entry/chi/FWHM/FWHM",
        "mu_Center_of_mass": "/entry/mu/Center of mass/Center of mass",
        "mu_FWHM": "/entry/mu/FWHM/FWHM",
    }
    stacked, names = legacy.stack_datasets(folders, "maps.h5", cfg)
    gold = str(tmp_path / "gold.h5")
    legacy.save_stacked_volumes(gold, stacked, names, "gzip")

    res = M.run({"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"})

    with h5py.File(gold, "r") as g, h5py.File(res.stacked_path, "r") as m:
        for grp, ds in [
            ("chi", "Center of mass"),
            ("chi", "FWHM"),
            ("mu", "Center of mass"),
            ("mu", "FWHM"),
        ]:
            np.testing.assert_array_equal(g[f"/{grp}/{ds}"][:], m[f"/{grp}/{ds}"][:])


# -----------------------------------------------------------------------------
# Optional χ/μ COM detrend
# -----------------------------------------------------------------------------
CHI_COM = "/entry/chi/Center of mass/Center of mass"
MU_COM = "/entry/mu/Center of mass/Center of mass"


def _trend(ny, nx, amp_x, amp_y):
    xs, ys = np.arange(nx, dtype=float), np.arange(ny, dtype=float)
    return (
        amp_x * np.arctan(0.3 * (xs - nx / 2))[None, :]
        + amp_y * np.arctan(0.3 * (ys - ny / 2))[:, None]
    )


def _write_trended(folder, layer_idx, ny=16, nx=20, dtype="float32", chi=None):
    """A maps.h5 whose χ and μ CoM maps carry a separable arctan trend."""
    os.makedirs(folder, exist_ok=True)
    rng = np.random.default_rng(layer_idx)
    noise = lambda: 0.001 * rng.standard_normal((ny, nx))  # noqa: E731
    maps = {
        CHI_COM: 1.0 + _trend(ny, nx, 0.5, 0.2) + noise() if chi is None else chi,
        "/entry/chi/FWHM/FWHM": 0.1 + _trend(ny, nx, 0.05, 0.05) + noise(),
        MU_COM: -2.0 + _trend(ny, nx, 0.3, 0.4) + noise(),
        "/entry/mu/FWHM/FWHM": 0.2 + _trend(ny, nx, 0.05, 0.05) + noise(),
    }
    with h5py.File(os.path.join(folder, "maps.h5"), "w") as f:
        for path, arr in maps.items():
            f.create_dataset(path, data=np.asarray(arr, dtype=dtype))


def _trended_root(tmp_path, n=2, **kw):
    root = tmp_path / "root"
    for i in range(n):
        _write_trended(str(root / f"layer__{i + 1}"), i, **kw)
    return root


def _run(root, **over):
    return M.run({"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*", **over})


def _expected_detrended(maps_path, src):
    from darq_xray.common.detrend import detrend_arctan_2d

    with h5py.File(maps_path, "r") as f:
        raw = f[src][:]
    return detrend_arctan_2d(raw.astype(np.float64))[0].astype(raw.dtype)


def test_detrend_is_off_by_default(tmp_path):
    """No box ticked: maps stored as read, no attribute, no plot folder."""
    root = _trended_root(tmp_path)
    res = _run(root)
    assert res.detrended == [] and res.detrend_plots == []
    with h5py.File(res.stacked_path, "r") as f, h5py.File(root / "layer__1" / "maps.h5") as src:
        assert np.array_equal(f["/chi/Center of mass"][0], src[CHI_COM][:])
        assert "detrend" not in f["/chi/Center of mass"].attrs
    assert not (root / "mosaicity_detrend").exists()


def test_detrend_chi_replaces_chi_com_and_leaves_the_rest_alone(tmp_path):
    root = _trended_root(tmp_path)
    res = _run(root, detrend_chi_com=True)
    assert res.detrended == ["/chi/Center of mass"]
    with h5py.File(res.stacked_path, "r") as f:
        chi = f["/chi/Center of mass"]
        assert chi.attrs["detrend"] == "arctan_2d"
        for z, name in enumerate(res.layers):
            maps_path = str(root / name / "maps.h5")
            assert np.array_equal(chi[z], _expected_detrended(maps_path, CHI_COM))
            with h5py.File(maps_path, "r") as src:
                assert np.array_equal(f["/mu/Center of mass"][z], src[MU_COM][:])
                assert np.array_equal(f["/chi/FWHM"][z], src["/entry/chi/FWHM/FWHM"][:])
                assert np.array_equal(f["/mu/FWHM"][z], src["/entry/mu/FWHM/FWHM"][:])
        for key in ("/mu/Center of mass", "/chi/FWHM", "/mu/FWHM"):
            assert "detrend" not in f[key].attrs
        # the trend is actually gone from what was stored, not merely recomputed
        with h5py.File(str(root / "layer__1" / "maps.h5"), "r") as src:
            assert np.nanstd(chi[0]) < 0.05 * np.nanstd(src[CHI_COM][:])


def test_detrend_mu_only(tmp_path):
    root = _trended_root(tmp_path)
    res = _run(root, detrend_mu_com=True)
    assert res.detrended == ["/mu/Center of mass"]
    with h5py.File(res.stacked_path, "r") as f:
        assert f["/mu/Center of mass"].attrs["detrend"] == "arctan_2d"
        assert "detrend" not in f["/chi/Center of mass"].attrs
        maps_path = str(root / "layer__1" / "maps.h5")
        assert np.array_equal(f["/mu/Center of mass"][0], _expected_detrended(maps_path, MU_COM))


def test_detrend_keeps_the_source_dtype(tmp_path):
    for dtype in ("float32", "float64"):
        root = _trended_root(tmp_path / dtype, dtype=dtype)
        res = _run(root, detrend_chi_com=True, detrend_mu_com=True)
        with h5py.File(res.stacked_path, "r") as f:
            assert f["/chi/Center of mass"].dtype == np.dtype(dtype)
            assert f["/mu/Center of mass"].dtype == np.dtype(dtype)


def test_detrend_writes_a_diag_png_per_layer_and_quantity(tmp_path):
    root = _trended_root(tmp_path)
    res = _run(root, detrend_chi_com=True, detrend_mu_com=True)
    out = root / "mosaicity_detrend"
    expected = sorted(
        str(out / f"{name}_{stem}_detrend_diag.png")
        for name in ("layer__1", "layer__2")
        for stem in ("chi_com", "mu_com")
    )
    assert sorted(res.detrend_plots) == expected
    assert all(os.path.getsize(p) > 0 for p in expected)
    assert res.detrend_dir == str(out)
    assert res.stacked_path == str(root / "stacked_volumes.h5")


def test_output_dir_takes_the_plots_but_the_stack_stays_in_the_root(tmp_path):
    root = _trended_root(tmp_path)
    out = tmp_path / "elsewhere"
    res = _run(root, detrend_chi_com=True, output_dir=str(out))
    assert res.stacked_path == str(root / "stacked_volumes.h5")
    assert {os.path.dirname(p) for p in res.detrend_plots} == {str(out)}


def test_detrend_stores_an_unfittable_map_as_read_and_says_so(tmp_path):
    root = tmp_path / "root"
    _write_trended(str(root / "layer__1"), 0)
    _write_trended(str(root / "layer__2"), 1, chi=np.full((16, 20), np.nan))
    res = _run(root, detrend_chi_com=True)
    assert res.layers == ["layer__1", "layer__2"]
    assert any("layer__2" in n and "χ" in n for n in res.notes), res.notes
    with h5py.File(res.stacked_path, "r") as f:
        assert np.isnan(f["/chi/Center of mass"][1]).all()
    assert [os.path.basename(p) for p in res.detrend_plots] == ["layer__1_chi_com_detrend_diag.png"]


def test_figures_list_a_detrend_diag_per_detrended_layer(tmp_path):
    from matplotlib.figure import Figure

    root = _trended_root(tmp_path)
    params = {"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"}
    res = M.run({**params, "detrend_chi_com": True})
    diag = [s for s in M.figures(res, params) if "detrend" in s.figure_id]
    assert [s.figure_id for s in diag] == ["chi_com_detrend_z0000", "chi_com_detrend_z0001"]
    assert all(s.kind == "plot" for s in diag)
    assert diag[0].filename == "layer__1_chi_com_detrend_diag"
    assert isinstance(diag[0].build(None), Figure)

    plain = M.run(params)
    assert not [s for s in M.figures(plain, params) if "detrend" in s.figure_id]


def test_the_attribute_names_exactly_the_detrended_z_slices(tmp_path):
    """A volume that is only partly detrended must say which slices are."""
    root = tmp_path / "root"
    _write_trended(str(root / "layer__1"), 0)
    _write_trended(str(root / "layer__2"), 1, chi=np.full((16, 20), np.nan))
    _write_trended(str(root / "layer__3"), 2)
    res = _run(root, detrend_chi_com=True)
    with h5py.File(res.stacked_path, "r") as f:
        assert list(f["/chi/Center of mass"].attrs["detrended_z"]) == [0, 2]


def test_detrend_z_counts_only_layers_that_carry_the_map(tmp_path):
    """Each volume stacks only the layers holding its map, so a layer without a
    χ CoM shifts every later χ slice down — the figure ids and the attribute
    must use the dataset's own Z, as the map and histogram specs do."""
    root = tmp_path / "root"
    _write_trended(str(root / "layer__1"), 0)
    _write_trended(str(root / "layer__2"), 1)
    with h5py.File(root / "layer__2" / "maps.h5", "a") as f:
        del f[CHI_COM]
    _write_trended(str(root / "layer__3"), 2)
    params = {"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"}
    res = M.run({**params, "detrend_chi_com": True})
    with h5py.File(res.stacked_path, "r") as f:
        assert f["/chi/Center of mass"].shape[0] == 2
        assert list(f["/chi/Center of mass"].attrs["detrended_z"]) == [0, 1]
    diag = [s for s in M.figures(res, params) if "detrend" in s.figure_id]
    assert [(s.figure_id, s.filename) for s in diag] == [
        ("chi_com_detrend_z0000", "layer__1_chi_com_detrend_diag"),
        ("chi_com_detrend_z0001", "layer__3_chi_com_detrend_diag"),
    ]


def test_a_map_the_fit_chokes_on_is_stored_as_read_and_the_run_survives(tmp_path):
    """inf rows pass the finite-pixel count but leave every column median inf;
    the fit then fails, which must cost that map its detrend, not the volume."""
    chi = 1.0 + _trend(16, 20, 0.5, 0.2)
    chi[:10] = np.inf
    root = tmp_path / "root"
    _write_trended(str(root / "layer__1"), 0, chi=chi)
    _write_trended(str(root / "layer__2"), 1)
    res = _run(root, detrend_chi_com=True)
    assert res.layers == ["layer__1", "layer__2"]
    assert any("layer__1" in n and "χ COM" in n for n in res.notes), res.notes
    with h5py.File(res.stacked_path, "r") as f, h5py.File(root / "layer__1" / "maps.h5") as src:
        assert np.array_equal(f["/chi/Center of mass"][0], src[CHI_COM][:])
        assert list(f["/chi/Center of mass"].attrs["detrended_z"]) == [1]


def test_an_unwritable_plot_folder_costs_the_plots_not_the_volume(tmp_path):
    root = _trended_root(tmp_path)
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory")
    res = _run(root, detrend_chi_com=True, output_dir=str(blocker / "plots"))
    assert res.stacked_path == str(root / "stacked_volumes.h5")
    assert res.detrend_plots == []
    assert any("diagnostic" in n for n in res.notes), res.notes
    with h5py.File(res.stacked_path, "r") as f:
        assert list(f["/chi/Center of mass"].attrs["detrended_z"]) == [0, 1]


def test_the_plot_folder_is_never_taken_for_a_layer(tmp_path):
    """The stage's default pattern '*' matches every subfolder of the root,
    including the mosaicity_detrend/ a previous run left there."""
    root = _trended_root(tmp_path)
    first = _run(root, detrend_chi_com=True, folder_pattern="*")
    assert (root / "mosaicity_detrend").is_dir()
    again = _run(root, detrend_chi_com=True, folder_pattern="*")
    assert again.layers == first.layers == ["layer__1", "layer__2"]
    assert again.skipped == []
    est = M.estimate({"mode": "batch", "root_folder": str(root), "folder_pattern": "*"})
    assert est.shape == (2, 16, 20), est.note


def test_a_custom_plot_folder_inside_the_root_is_never_taken_for_a_layer(tmp_path):
    root = _trended_root(tmp_path)
    plots = str(root / "my_plots")
    _run(root, detrend_chi_com=True, folder_pattern="*", output_dir=plots)
    again = _run(root, folder_pattern="*", output_dir=plots)
    assert again.skipped == []
    est = M.estimate(
        {"mode": "batch", "root_folder": str(root), "folder_pattern": "*", "output_dir": plots}
    )
    assert est.shape == (2, 16, 20), est.note


def test_an_earlier_runs_plot_folder_is_never_taken_for_a_layer(tmp_path):
    """The folder a PREVIOUS run wrote to stays out of the layer list after
    'Output dir' has been changed or cleared — it is recognised by what it
    holds, not by the current form value."""
    root = _trended_root(tmp_path)
    _run(root, detrend_chi_com=True, folder_pattern="*", output_dir=str(root / "a_plots"))
    again = _run(root, folder_pattern="*")
    assert again.layers == ["layer__1", "layer__2"]
    assert again.skipped == []
    est = M.estimate({"mode": "batch", "root_folder": str(root), "folder_pattern": "*"})
    assert est.shape == (2, 16, 20), est.note


# Every folder the app writes into the processed root, which is the root strain
# and mosaicity search for layers: each stage's own PNG folder, the replot
# dialogs' default, and visualize / slices / paraview beside the stacked files.
APP_FOLDERS = (
    "strain_maps",
    "mosaicity_detrend",
    "replots",
    "aligned_volume_visualizations",
    "oblique_slices",
    "paraview_exports",
)


def test_no_folder_the_app_writes_is_taken_for_a_layer(tmp_path):
    """'aligned_volume_visualizations' sorts before 'layer__1', so the estimate
    sized it, as well as the run listing every one of these as a layer without
    a maps.h5."""
    root = _trended_root(tmp_path)
    for name in APP_FOLDERS:
        (root / name).mkdir()
        (root / name / "something.dat").write_text("")
    res = _run(root, folder_pattern="*")
    assert res.layers == ["layer__1", "layer__2"]
    assert res.skipped == []
    est = M.estimate({"mode": "batch", "root_folder": str(root), "folder_pattern": "*"})
    assert est.shape == (2, 16, 20), est.note


def test_a_map_that_is_not_2d_says_so(tmp_path):
    root = tmp_path / "root"
    _write_trended(str(root / "layer__1"), 0, chi=np.ones((1, 16, 20)))
    res = _run(root, detrend_chi_com=True)
    assert res.layers == ["layer__1"]
    assert any("not a 2-D map" in n and "(1, 16, 20)" in n for n in res.notes), res.notes
    assert not any("finite pixels" in n for n in res.notes), res.notes


def test_cli_can_switch_the_detrend_on(tmp_path):
    root = _trended_root(tmp_path)
    plots = tmp_path / "plots"
    rc = M._main(
        [
            "--root-folder",
            str(root),
            "--folder-pattern",
            "layer__*",
            "--detrend-chi-com",
            "--detrend-mu-com",
            "--output-dir",
            str(plots),
        ]
    )
    assert rc == 0
    with h5py.File(root / "stacked_volumes.h5", "r") as f:
        assert f["/chi/Center of mass"].attrs["detrend"] == "arctan_2d"
        assert f["/mu/Center of mass"].attrs["detrend"] == "arctan_2d"
    assert len(list(plots.glob("*_detrend_diag.png"))) == 4


def _write_stacked(path):
    import h5py
    import numpy as np

    rng = np.random.default_rng(3)
    with h5py.File(path, "w") as f:
        for key in ("/chi/Center of mass", "/chi/FWHM"):
            f.create_dataset(key, data=rng.standard_normal((2, 4, 5)).astype(np.float32))
        f.attrs["num_layers"] = 2
    return path


def test_mosaicity_replot_catalog_lists_datasets(tmp_path):
    h5 = str(tmp_path / "stacked.h5")
    _write_stacked(h5)
    cat = M.replot_catalog(h5)
    by_key = {g.key: g for g in cat}
    assert set(by_key) == {"/chi/Center of mass", "/chi/FWHM"}
    assert len(by_key["/chi/FWHM"].item_labels) == 2
    assert by_key["/chi/FWHM"].shape == (4, 5)  # (Y, X) of the stored layer — ROI hint


def test_mosaicity_render_replot_writes_pngs_with_crop(tmp_path):
    import os

    h5 = str(tmp_path / "stacked.h5")
    _write_stacked(h5)
    out = str(tmp_path / "replots")
    written = M.render_replot(
        h5,
        [("/chi/Center of mass", [0]), ("/chi/FWHM", None)],
        style=None,
        clim=None,
        out_dir=out,
        roi=(0, 2, 0, 3),
    )
    assert len(written) == 1 + 2
    assert all(os.path.exists(p) for p in written)


def test_mosaicity_render_replot_per_group_clim(tmp_path, monkeypatch):
    h5 = str(tmp_path / "stacked.h5")
    _write_stacked(h5)
    seen: dict[str, tuple] = {}

    def fake_layer(h5_path, key, z, **kw):
        seen[key] = kw.get("clim")
        return None  # skip figure build / save

    monkeypatch.setattr(M, "render_volume_layer", fake_layer)
    M.render_replot(
        h5,
        [("/chi/Center of mass", [0]), ("/chi/FWHM", [0])],
        style=None,
        clim={"/chi/Center of mass": (-0.5, 0.5), "/chi/FWHM": (0.0, 0.2)},
        out_dir=str(tmp_path / "r"),
    )
    # each dataset gets its own limits — the whole point of per-kind clim
    assert seen["/chi/Center of mass"] == (-0.5, 0.5)
    assert seen["/chi/FWHM"] == (0.0, 0.2)
