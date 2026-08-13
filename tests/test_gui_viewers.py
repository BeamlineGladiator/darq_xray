"""Tests for the interactive-viewer glue (headless parts only):
visualize.aligned_field, gui.viewers.volume_sources, and inject_line_into_jobs.
The Qt/GL widgets themselves are exercised by tests/gui_smoke.py.
"""

from __future__ import annotations

import json
import types

import h5py
import numpy as np
import pytest

from dfxm.stages import visualize as V
from gui import viewers

L, NY, NX = 4, 6, 8


def _setup_volumes(tmp_path):
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


def _params(proc, raw):
    return {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
    }


# -- visualize.aligned_field --------------------------------------------------
def test_available_fields_and_aligned_field(tmp_path):
    proc, raw = _setup_volumes(tmp_path)
    p = _params(proc, raw)
    fields = V.available_fields(p)
    assert set(fields) == {
        "chi_Center_of_mass",
        "chi_FWHM",
        "mu_Center_of_mass",
        "mu_FWHM",
        "strain",
    }

    vol, spacing, cmap, clim, meta = V.aligned_field(p, "chi_Center_of_mass")
    assert vol.ndim == 3 and vol.shape[2] >= NX  # X canvas expanded by samy shift
    assert spacing[0] == pytest.approx(0.152) and spacing[2] > 0
    assert cmap == "fast" and clim[0] == pytest.approx(-clim[1])  # midrange -> symmetric
    assert meta == {"cbar_label": "Misorientation (°)", "group": "mosa_com"}

    sv, _sp, scmap, _sc, smeta = V.aligned_field(p, "strain")
    assert scmap == "RdBu_r" and sv.ndim == 3
    assert smeta == {"cbar_label": "Strain (ε)", "group": "strain"}


def test_aligned_field_unknown_raises(tmp_path):
    proc, raw = _setup_volumes(tmp_path)
    with pytest.raises(KeyError):
        V.aligned_field(_params(proc, raw), "nope")


# -- gui.viewers.volume_sources ----------------------------------------------
def _make_aligned_h5(tmp_path):
    """Build a synthetic aligned_raw_rocking_volumes.h5 and return its path."""
    aligned = tmp_path / "aligned_raw_rocking_volumes.h5"
    with h5py.File(aligned, "w") as f:
        f.create_dataset(
            "sum_intensity", data=np.arange(L * NY * NX).reshape(L, NY, NX).astype(float)
        )
        f.create_dataset("specific_frame", data=np.zeros((L, NY, NX)))
        f.attrs["scale_x_um_per_px"] = 0.152
        f.attrs["scale_y_um_per_px"] = 0.385
        f.attrs["scale_z_um_per_px"] = 1.5
    return str(aligned)


def _result_with(**kw):
    return types.SimpleNamespace(**kw)


def test_volume_sources_visualize_lazy(tmp_path):
    proc, raw = _setup_volumes(tmp_path)
    p = _params(proc, raw)
    sources = viewers.volume_sources("visualize", None, p)
    assert set(sources) == set(V.available_fields(p))
    loaded = sources["strain"].load()  # invoking the loader does the work
    assert loaded.volume.ndim == 3 and loaded.cmap == "RdBu_r"
    assert loaded.cbar_label == "Strain (ε)" and loaded.group == "strain"


def test_visualize_source_spec_loader_is_jsonable(monkeypatch):
    monkeypatch.setattr("dfxm.stages.visualize.available_fields", lambda p: ["chi_Center_of_mass"])
    params = {"mosa_volume_file": "/x/maps.h5"}
    sources = viewers.volume_sources("visualize", object(), params)
    spec = sources["chi_Center_of_mass"]
    assert spec.loader["kind"] == "visualize_field"
    assert spec.loader["field"] == "chi_Center_of_mass"
    assert spec.loader["stage_params"] == params
    json.dumps(spec.loader)  # must not raise


def test_volume_sources_rocking_reads_attrs(tmp_path):
    aligned = _make_aligned_h5(tmp_path)
    result = _result_with(aligned_path=aligned)
    sources = viewers.volume_sources("rocking", result, {})
    assert set(sources) == {"sum_intensity", "specific_frame"}
    loaded = sources["sum_intensity"].load()
    assert loaded.spacing == (0.152, 0.385, 1.5) and loaded.cmap == "gray"
    assert loaded.clim is not None and loaded.clim[0] < loaded.clim[1]


def test_rocking_source_spec_carries_meta_and_loader(tmp_path):
    path = _make_aligned_h5(tmp_path)
    sources = viewers.volume_sources("rocking", _result_with(aligned_path=path), {})
    spec = sources["sum_intensity"]
    assert spec.loader == {"kind": "h5_dataset", "path": path, "dataset": "sum_intensity"}
    loaded = spec.load()
    assert loaded.group == "raw"
    assert loaded.cbar_label == "Intensity"
    assert loaded.volume.ndim == 3


def test_volume_sources_empty_for_other_stages():
    assert viewers.volume_sources("concat", None, {}) == {}
    assert (
        viewers.volume_sources("rocking", types.SimpleNamespace(aligned_path="/no/such.h5"), {})
        == {}
    )


# -- inject_line_into_jobs ----------------------------------------------------
def test_inject_updates_matching_job():
    jobs = json.dumps([{"name": "a", "offset_um": 0}, {"name": "oblique_full", "offset_um": 9}])
    out = viewers.inject_line_into_jobs(jobs, "oblique_full", (1.1, 2.2), (3.3, 4.4), -5.0)
    j = json.loads(out)
    assert j[0] == {"name": "a", "offset_um": 0}  # untouched
    assert j[1]["start_uv"] == [1.1, 2.2] and j[1]["end_uv"] == [3.3, 4.4]
    assert j[1]["offset_um"] == -5.0


def test_inject_creates_job_when_empty():
    out = viewers.inject_line_into_jobs("[]", "oblique_full", (0, 0), (1, 1), 2.0)
    j = json.loads(out)
    assert len(j) == 1 and j[0]["name"] == "oblique_full" and j[0]["end_uv"] == [1.0, 1.0]


def test_inject_handles_garbage():
    out = viewers.inject_line_into_jobs("not json", "s", (0, 0), (1, 1), 0.0)
    assert json.loads(out)[0]["name"] == "s"


def test_inject_clears_stale_fields_when_none():
    """A stale 'fields' key must be removed when fields=None is passed (FIX 1)."""
    base = json.dumps([{"name": "oblique_full", "offset_um": 0.0}])
    # First inject with a field restriction
    restricted = viewers.inject_line_into_jobs(
        base, "oblique_full", (0.0, 0.0), (1.0, 0.0), 0.0, fields=["strain"]
    )
    assert json.loads(restricted)[0]["fields"] == ["strain"]
    # Second inject with fields=None — stale key must be removed
    unrestricted = viewers.inject_line_into_jobs(
        restricted, "oblique_full", (0.0, 0.0), (1.0, 0.0), 0.0, fields=None
    )
    assert "fields" not in json.loads(unrestricted)[0]
