"""Tests for dfxm.stages.mosaicity, incl. golden equivalence vs the legacy
stack_h5_darfix_volumes script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm.stages import mosaicity as M

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
