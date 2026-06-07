"""Tests for dfxm.stages.paraview — PVTI partitioning (parity with the legacy
exporter), end-to-end file writing, and a VTK round-trip of values + valid_mask.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm.stages import paraview as PV

L, NY, NX = 4, 6, 8


def _legacy_export():
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "export_aligned_volumes_to_paraview_v6_pvti.py").exists():
        pytest.skip("legacy PVTI exporter not found")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return __import__("export_aligned_volumes_to_paraview_v6_pvti")


# -- partitioning parity ------------------------------------------------------
@pytest.mark.parametrize("nz,n", [(5, 2), (10, 4), (17, 16), (3, 8), (2, 2)])
def test_piece_extents_match_legacy(nz, n):
    legacy = _legacy_export()
    assert PV.compute_piece_extents_z(nz, n) == legacy.compute_piece_extents_z(nz, n)


def test_piece_extents_share_boundary():
    # adjacent pieces share exactly one Z index; coverage is gap-free
    ext = PV.compute_piece_extents_z(9, 3)
    assert ext[0][0] == 0 and ext[-1][1] == 8
    for (a0, a1), (b0, b1) in zip(ext, ext[1:]):
        assert a1 == b0  # shared boundary index


def test_numpy_to_vtk_type_str():
    assert PV._numpy_to_vtk_type_str(np.float32) == "Float32"
    with pytest.raises(ValueError):
        PV._numpy_to_vtk_type_str(np.complex64)


# -- end-to-end ---------------------------------------------------------------
def _write_raw(root, base, samy, samz):
    for i in range(L):
        folder = os.path.join(root, f"{base}__{i + 1}")
        os.makedirs(folder)
        name = os.path.basename(folder)
        with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
            f.create_dataset("1.1/instrument/positioners/samy", data=samy[i])
            f.create_dataset("1.1/instrument/positioners/samz", data=samz[i])


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
    raw = tmp_path / "raw"
    raw.mkdir()
    samy = np.array([0.0, 0.001, 0.0025, 0.004])
    samz = np.array([0.0, 0.001, 0.0021, 0.0035])
    _write_raw(str(raw), "mosa", samy, samz)
    _write_raw(str(raw), "strain", samy, samz)
    return proc, raw


def test_run_writes_pvti_and_info(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "pv"
    res = PV.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "output_dir": str(out),
            "num_pieces_z": 2,
        }
    )
    names = {e.name for e in res.exports}
    assert names == {"mosaicity", "strain"}
    assert os.path.exists(out / "mosaicity_volume.pvti")
    assert os.path.exists(out / "strain_volume.pvti")
    assert os.path.isdir(out / "mosaicity_volume_pieces")
    assert res.info_path and os.path.exists(res.info_path)

    mosa = next(e for e in res.exports if e.name == "mosaicity")
    # all four mosaicity scalars + the valid_mask travel in one PVTI
    assert "valid_mask" in mosa.fields
    assert {"chi_Center_of_mass", "mu_FWHM"}.issubset(mosa.fields)
    vti = [p for p in os.listdir(out / "mosaicity_volume_pieces") if p.endswith(".vti")]
    assert len(vti) == mosa.n_pieces == 2


def test_pvti_roundtrip_values_and_mask(tmp_path):
    """Write a single-field volume with a NaN, read it back via VTK, and confirm
    the NaN became a finite sentinel and valid_mask marks it."""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    vol = np.arange(2 * NY * NX, dtype=np.float64).reshape(2, NY, NX)
    vol[0, 0, 0] = np.nan
    out = tmp_path / "rt.pvti"
    info = PV.save_volumes_as_pvti({"strain": vol}, (0.152, 0.385, 1.0), str(out), n_pieces=1)
    assert info["nan_sentinel"] is not None

    reader = vtk.vtkXMLPImageDataReader()
    reader.SetFileName(str(out))
    reader.Update()
    img = reader.GetOutput()
    dims = img.GetDimensions()  # (nx, ny, nz)
    assert dims == (NX, NY, 2)
    pd = img.GetPointData()
    strain = vtk_to_numpy(pd.GetArray("strain")).reshape(2, NY, NX)
    mask = vtk_to_numpy(pd.GetArray("valid_mask")).reshape(2, NY, NX)
    assert np.isfinite(strain).all()  # NaN replaced by sentinel
    assert strain[0, 0, 0] == pytest.approx(info["nan_sentinel"])
    assert mask[0, 0, 0] == 0.0 and mask[1, 1, 1] == 1.0
