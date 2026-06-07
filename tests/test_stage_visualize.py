"""Tests for dfxm.stages.visualize — produces aligned 2D products and records
datasets; alignment reuses the golden-tested common.alignment primitives.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from dfxm.stages import visualize as V

L, NY, NX = 4, 6, 8


def _write_mosa(path):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for grp in ("chi", "mu"):
            g = f.create_group(grp)
            g.create_dataset("Center of mass", data=rng.standard_normal((L, NY, NX)))
            g.create_dataset("FWHM", data=np.abs(rng.standard_normal((L, NY, NX))))


def _write_strain(path):
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        f.create_dataset("strain", data=rng.standard_normal((L, NY, NX)) * 1e-4)


def _write_raw(root, pattern_base, samy, samz):
    for i in range(L):
        folder = os.path.join(root, f"{pattern_base}__{i + 1}")
        os.makedirs(folder)
        name = os.path.basename(folder)
        with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
            f.create_dataset("1.1/instrument/positioners/samy", data=samy[i])
            f.create_dataset("1.1/instrument/positioners/samz", data=samz[i])


def _setup(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_mosa(str(proc / "stacked_volumes.h5"))
    _write_strain(str(proc / "stacked_strain_volumes.h5"))
    raw = tmp_path / "raw"
    raw.mkdir()
    samy = np.array([0.0, 0.001, 0.0025, 0.004])
    samz = np.array([0.0, 0.001, 0.0021, 0.0035])
    _write_raw(str(raw), "mosa", samy, samz)
    _write_raw(str(raw), "strain", samy, samz)
    return proc, raw


def test_run_produces_layers_and_animation(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "viz"
    res = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "output_dir": str(out),
            "save_topview": False,  # GL not guaranteed in CI
        }
    )
    names = {d.name for d in res.datasets}
    assert names == {
        "chi_Center_of_mass",
        "chi_FWHM",
        "mu_Center_of_mass",
        "mu_FWHM",
        "strain",
    }
    for d in res.datasets:
        assert d.layers_dir and os.path.isdir(d.layers_dir)
        pngs = [p for p in os.listdir(d.layers_dir) if p.endswith(".png")]
        assert len(pngs) == d.shape[0]  # one PNG per aligned Z layer
        assert d.animation and os.path.exists(d.animation)


def test_com_is_centered_strain_is_symmetric(tmp_path):
    proc, raw = _setup(tmp_path)
    res = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "output_dir": str(tmp_path / "viz"),
            "center_method": "midrange",
            "save_layers": False,
            "save_animation": False,
            "save_topview": False,
        }
    )
    by_name = {d.name: d for d in res.datasets}
    com = by_name["chi_Center_of_mass"]
    assert com.vmin == pytest.approx(-com.vmax)  # midrange -> symmetric
    strain = by_name["strain"]
    assert strain.vmin == pytest.approx(-strain.vmax)  # strain symmetric range


def test_alignment_shape_matches_primitives(tmp_path):
    proc, raw = _setup(tmp_path)
    res = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "output_dir": str(tmp_path / "viz"),
            "roi_y": "1,5",
            "save_layers": False,
            "save_animation": False,
            "save_topview": False,
        }
    )
    # ROI in Y -> 4 rows; X expanded by samy padding -> >= NX
    for d in res.datasets:
        assert d.shape[1] == 4
        assert d.shape[2] >= NX


def test_missing_inputs_recorded(tmp_path):
    res = V.run({"mosa_volume_file": str(tmp_path / "nope.h5"), "strain_volume_file": ""})
    assert any("not found" in s for s in res.skipped)


def test_parse_pair_and_display_info():
    assert V._parse_pair("") is None
    assert V._parse_pair("3, 7") == (3, 7)
    with pytest.raises(ValueError):
        V._parse_pair("1,2,3")
    assert V._display_info("strain", is_strain=True)[2] == "RdBu_r"
    assert V._display_info("chi_Center_of_mass")[2] == "magma"
