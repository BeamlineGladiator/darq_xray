"""Tests for dfxm.stages.paraview — PVTI partitioning (parity with the legacy
exporter), end-to-end file writing, and a VTK round-trip of values + valid_mask.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm.common import alignment as A
from dfxm.stages import paraview as PV

L, NY, NX = 4, 6, 8

# -- streaming fixtures -------------------------------------------------------
# One shape and one pair of motor arrays shared by every streamed field:
# `save_volumes_streamed` serves one Z range of every field per piece, which is
# only meaningful for providers on the same Z axis and the same grid.
S_L, S_NY, S_NX = 24, 20, 32
_S_SAMY = np.linspace(0.0, 0.0012, S_L)
# Deliberately irregular, so `interpolate_to_uniform_z` resamples onto a grid
# that is NOT the input layers — the streamed and in-core paths then have real
# interpolation to agree about rather than a pass-through.
_S_SAMZ = np.cumsum(np.linspace(0.0008, 0.0013, S_L)) - 0.0008
_S_ALIGN = dict(scale_x=0.152, samy_direction=-1)


def _synthetic_volume(seed: int) -> np.ndarray:
    """A float64 volume with scattered NaNs, so the sentinel/valid_mask branches run."""
    rng = np.random.default_rng(seed)
    vol = rng.standard_normal((S_L, S_NY, S_NX))
    vol[vol > 1.6] = np.nan
    return vol


def _as_streamed(vol, *, budget_bytes):
    """Wrap an in-memory volume as the `StreamedAlignment` the writer consumes."""
    return A.align_volume_streamed(vol, _S_SAMY, _S_SAMZ, budget_bytes=budget_bytes, **_S_ALIGN)


def _as_in_core(vol) -> np.ndarray:
    """The same alignment, materialised — what the old stage handed the writer."""
    return A.align_volume(vol, _S_SAMY, _S_SAMZ, **_S_ALIGN).data


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
    for i in range(len(samy)):
        folder = os.path.join(root, f"{base}__{i + 1}")
        os.makedirs(folder)
        name = os.path.basename(folder)
        with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
            f.create_dataset("1.1/instrument/positioners/samy", data=samy[i])
            f.create_dataset("1.1/instrument/positioners/samz", data=samz[i])


def _setup(tmp_path, layers=L, ny=NY, nx=NX):
    proc = tmp_path / "proc"
    proc.mkdir()
    rng = np.random.default_rng(0)
    with h5py.File(proc / "stacked_volumes.h5", "w") as f:
        for grp in ("chi", "mu"):
            g = f.create_group(grp)
            g.create_dataset("Center of mass", data=rng.standard_normal((layers, ny, nx)))
            g.create_dataset("FWHM", data=np.abs(rng.standard_normal((layers, ny, nx))))
    with h5py.File(proc / "stacked_strain_volumes.h5", "w") as f:
        f.create_dataset("strain", data=rng.standard_normal((layers, ny, nx)) * 1e-4)
    raw = tmp_path / "raw"
    raw.mkdir()
    if layers == L:
        samy = np.array([0.0, 0.001, 0.0025, 0.004])
        samz = np.array([0.0, 0.001, 0.0021, 0.0035])
    else:
        samy = np.linspace(0.0, 0.0012, layers)
        # Irregular, so the Z interpolation resamples rather than passing through.
        samz = np.cumsum(np.linspace(0.0008, 0.0013, layers)) - 0.0008
    _write_raw(str(raw), "mosa", samy, samz)
    _write_raw(str(raw), "strain", samy, samz)
    return proc, raw


def test_mosa_field_names_and_load_mosa_field(tmp_path):
    """The lazy API enumerates exactly what the old eager dict contained."""
    proc, _raw = _setup(tmp_path)
    path = str(proc / "stacked_volumes.h5")

    names = PV.mosa_field_names(path)
    assert names == sorted(names), "names must be deterministic across runs"
    assert names == ["chi_Center_of_mass", "chi_FWHM", "mu_Center_of_mass", "mu_FWHM"]

    with h5py.File(path, "r") as f:
        eager = {
            f"{grp}_{ds.replace(' ', '_')}": f[grp][ds][:] for grp in ("chi", "mu") for ds in f[grp]
        }
    assert sorted(eager) == names
    for name in names:
        field = PV.load_mosa_field(path, name)
        assert field is not None and field.ndim == 3
        np.testing.assert_array_equal(field, eager[name])

    assert PV.load_mosa_field(path, "not_a_field") is None


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


# -- streamed piece writer ----------------------------------------------------
def test_streamed_pvti_matches_in_core(tmp_path):
    """Streaming the piece writer changes nothing about the pieces."""
    fields = {
        "chi_Center_of_mass": _synthetic_volume(seed=1),
        "mu_FWHM": _synthetic_volume(seed=2),
    }
    # Deliberately different budgets: in the real stage a centred CoM field and
    # a plain FWHM field are priced differently and end up with different block
    # sizes, so the writer must never assume the fields block alike.
    budgets = {"chi_Center_of_mass": 300 << 10, "mu_FWHM": 350 << 10}
    providers = {
        name: _as_streamed(vol, budget_bytes=budgets[name]) for name, vol in fields.items()
    }

    # Preconditions, asserted rather than assumed. A sizing change elsewhere
    # could quietly turn this into a one-block stream whose piece slabs each sit
    # inside a single block — the test would stay green while covering none of
    # the slab bookkeeping it exists to pin. So: several blocks, block
    # boundaries that do not line up with piece boundaries (which forces
    # `_FieldStream` to both split a block and hold one across two pieces), and
    # two fields that disagree about where their blocks end.
    piece_starts = {z0 for z0, _z1 in PV.compute_piece_extents_z(S_L, 4)}
    starts_by_field = {}
    for name, prov in providers.items():
        assert prov.block_layers < prov.shape[0], f"{name} streams as one block"
        starts_by_field[name] = {sl.start for sl, _b in prov.blocks()}
        assert len(starts_by_field[name]) >= 4, f"{name} streams as too few blocks"
        assert piece_starts - starts_by_field[name], f"{name} blocks up with the pieces"
    assert len({frozenset(s) for s in starts_by_field.values()}) == 2, starts_by_field
    in_core = {name: _as_in_core(vol) for name, vol in fields.items()}
    assert not np.isfinite(next(iter(in_core.values()))).all(), "no NaN to sentinel"

    reference_dir = tmp_path / "ref"
    streamed_dir = tmp_path / "str"
    reference_dir.mkdir()
    streamed_dir.mkdir()

    ref_info = PV.save_volumes_as_pvti(
        in_core, (0.1, 0.2, 0.3), str(reference_dir / "v.pvti"), n_pieces=4
    )
    str_info = PV.save_volumes_streamed(
        providers, (0.1, 0.2, 0.3), str(streamed_dir / "v.pvti"), n_pieces=4
    )

    # The two datasets live in different directories, so the two path keys are
    # expected to differ; everything else must be identical.
    assert str_info["path_pvti"] == str(streamed_dir / "v.pvti")
    assert str_info["pieces_dir"] == str(streamed_dir / "v_pieces")
    drop = ("path_pvti", "pieces_dir")
    assert {k: v for k, v in str_info.items() if k not in drop} == {
        k: v for k, v in ref_info.items() if k not in drop
    }

    ref_pieces = sorted(reference_dir.rglob("*.vti"))
    assert len(ref_pieces) == ref_info["n_pieces"] == 4
    for ref_piece in ref_pieces:
        rel = ref_piece.relative_to(reference_dir)
        assert (streamed_dir / rel).read_bytes() == ref_piece.read_bytes(), rel
    assert (streamed_dir / "v.pvti").read_text() == (reference_dir / "v.pvti").read_text()


def test_run_pieces_are_budget_independent(tmp_path, monkeypatch):
    """The exported bytes do not depend on how the streamed run was blocked."""
    # Big enough that the frugal budget below is a budget the run can actually
    # meet — on the 4-layer fixture no budget above `align_volume_streamed`'s
    # own floor blocks at all, so the "frugal" run would warn and fall back to
    # one layer instead of demonstrating anything.
    proc, raw = _setup(tmp_path, layers=S_L, ny=S_NY, nx=S_NX)
    base = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "num_pieces_z": 2,
    }

    seen: dict[int, list[tuple[int, int]]] = {}

    def _record(budget):
        real = A.align_volume_streamed
        blockings: list[tuple[int, int]] = []

        def spy(*args, **kwargs):
            streamed = real(*args, **kwargs)
            blockings.append((streamed.block_layers, streamed.shape[0]))
            return streamed

        monkeypatch.setattr(PV.A, "align_volume_streamed", spy)
        out = tmp_path / f"out_{budget}"
        PV.run({**base, "output_dir": str(out), "_budget_bytes": budget})
        monkeypatch.undo()
        seen[budget] = blockings
        return out

    generous_budget, frugal_budget = 1 << 30, 4 << 20
    generous = _record(generous_budget)
    frugal = _record(frugal_budget)

    # The measurement has to be live: five fields aligned each time (four
    # mosaicity, one strain), the generous run in one block apiece and the
    # frugal one in several. Without this the two runs could be identically
    # blocked and the comparison below would compare a run with itself. The
    # frugal budget is split across the four concurrent mosaicity streams, so
    # only those four are forced to block; the lone strain stream keeps the
    # whole budget and does not have to.
    assert len(seen[generous_budget]) == len(seen[frugal_budget]) == 5
    assert all(layers >= nz for layers, nz in seen[generous_budget]), seen[generous_budget]
    chunked = [(layers, nz) for layers, nz in seen[frugal_budget] if layers < nz]
    assert len(chunked) == 4, seen[frugal_budget]

    written = sorted(p.relative_to(generous) for p in generous.rglob("*.vti"))
    assert len(written) == 4  # two exports x two pieces
    for rel in written:
        assert (frugal / rel).read_bytes() == (generous / rel).read_bytes(), rel
    for name in ("mosaicity_volume.pvti", "strain_volume.pvti"):
        assert (frugal / name).read_text() == (generous / name).read_text()


def test_no_motor_export_skips_both_motor_steps(tmp_path):
    """With no raw folders the export is unaligned — and its NaNs do not spread.

    Pinning the whole no-motor answer, because the obvious way to route it
    through the streamed alignment (a uniform stand-in samz) is wrong in a way
    only the data shows: resampling onto its own Z nodes reads the value below
    each node, so every NaN leaks one layer down and `valid_mask` changes.
    """
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy

    proc = tmp_path / "proc"
    proc.mkdir()
    vol = _synthetic_volume(seed=5)
    assert not np.isfinite(vol).all(), "fixture has no NaN to spread"
    with h5py.File(proc / "stacked_volumes.h5", "w") as f:
        f.create_group("chi").create_dataset("Center of mass", data=vol)
    empty_raw = tmp_path / "raw"
    empty_raw.mkdir()

    out = tmp_path / "pv"
    res = PV.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": "",
            "raw_root": str(empty_raw),
            "mosa_pattern": "nothing__*",
            "export_strain": False,
            "output_dir": str(out),
            "num_pieces_z": 3,
            "center_mosa_com": False,
        }
    )
    (mosa,) = res.exports
    # No samy padding, no Z resampling: the exported grid IS the input grid.
    assert mosa.dimensions_xyz == (S_NX, S_NY, S_L)
    assert mosa.spacing_um_xyz[2] == 2.0

    reader = vtk.vtkXMLPImageDataReader()
    reader.SetFileName(str(out / "mosaicity_volume.pvti"))
    reader.Update()
    mask = vtk_to_numpy(reader.GetOutput().GetPointData().GetArray("valid_mask"))
    mask = mask.reshape(S_L, S_NY, S_NX).astype(bool)
    np.testing.assert_array_equal(mask, np.isfinite(vol))


def test_paraview_peak_stays_under_budget(tmp_path):
    """A budgeted export does not need the whole aligned volume set resident.

    One float64 volume here is 36 MiB and the export has four fields; the
    in-core writer this replaced peaked at 586-593 MiB on exactly this input (a
    ~229 MiB floor for the spawned child's interpreter, numpy, h5py and VTK,
    plus ~10 volumes), while the streamed one measures ~261 MiB. The 480 MiB
    limit sits between them with ~110 MiB of margin on the failing side and
    ~220 MiB on the passing side — deliberately not tighter, because the floor
    is environment-dependent (a heavier VTK build raises it for everyone) and a
    peak test that flakes on an unrelated upgrade stops being read. Checked
    against the previous commit, where it fails.
    """
    from tests.peak_rss import assert_peak_under

    proc, raw = _setup(tmp_path, layers=128, ny=192, nx=192)
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": "",
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "export_strain": False,
        "output_dir": str(tmp_path / "pv"),
        "num_pieces_z": 16,
        # Pinned rather than measured from the machine, so the blocking under
        # test is the same on a laptop and on a workstation.
        "_budget_bytes": 64 << 20,
    }
    result = assert_peak_under(
        "dfxm.stages.paraview:run", params, limit_bytes=480 * (1 << 20), timeout=600
    )
    # ... and the bounded run still produced the export, rather than skipping it.
    assert [e.name for e in result.exports] == ["mosaicity"]
    assert result.exports[0].n_pieces == 16


def test_streamed_writer_rejects_mismatched_z_grids(tmp_path):
    """Per-field slabs pair by absolute Z index — a shared Z grid is the invariant."""
    vol = _synthetic_volume(seed=3)
    a = _as_streamed(vol, budget_bytes=1 << 30)
    b = A.align_volume_streamed(
        vol,
        _S_SAMY,
        _S_SAMZ * 1.5,  # same shape, different Z grid
        budget_bytes=1 << 30,
        **_S_ALIGN,
    )
    # The precondition the check exists for: same shape, different grid. Without
    # this the test could pass because the shapes differ, which is a different
    # guard entirely.
    assert tuple(a.shape) == tuple(b.shape)
    assert a.scale_z_um != b.scale_z_um

    with pytest.raises(ValueError, match="Z grid"):
        PV.save_volumes_streamed(
            {"one": a, "two": b}, (0.1, 0.2, 0.3), str(tmp_path / "v.pvti"), n_pieces=2
        )


def test_advisory_n_pieces_scales_with_budget_and_fields():
    """The floor is a real function of the inputs, not a constant."""
    shape = (128, 192, 192)
    per_piece_layer = 192 * 192 * 4 * PV.PIECE_BYTES_PER_VOXEL_PER_FIELD

    # A budget that fits the whole volume needs one piece; halving it needs two.
    assert PV.advisory_n_pieces(shape, 4, per_piece_layer * 128) == 1
    assert PV.advisory_n_pieces(shape, 4, per_piece_layer * 64) == 2
    # More fields at the same budget means more pieces, in proportion.
    assert PV.advisory_n_pieces(shape, 8, per_piece_layer * 64) == 4
    # Never above the layer count, never below one.
    assert PV.advisory_n_pieces(shape, 4, 1) == 128
    assert PV.advisory_n_pieces(shape, 4, 1 << 40) == 1


def test_run_warns_when_z_pieces_are_too_few_for_the_budget(tmp_path):
    """A low Z-piece count can peak above an in-core export — say so, do not just do it."""
    # Sized so the two budgets below straddle ONE piece's residency (~9 MB for
    # five fields) while both stay above the alignment chain's own floor — so
    # the difference under test is the piece advice and nothing else. A smaller
    # fixture cannot separate the two: the centring scaffold puts the alignment
    # floor above a whole piece.
    proc, raw = _setup(tmp_path, layers=24, ny=96, nx=96)
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": "",
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "export_strain": False,
        "num_pieces_z": 1,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no alignment-floor warning may fire here
        tight = PV.run({**params, "output_dir": str(tmp_path / "tight"), "_budget_bytes": 8 << 20})
        roomy = PV.run({**params, "output_dir": str(tmp_path / "roomy"), "_budget_bytes": 32 << 20})

    assert tight.exports, "the run must still produce its export — advice, not a refusal"
    assert any("Z pieces" in n for n in tight.notes), tight.notes
    assert "raise Z pieces to" in " ".join(tight.notes)
    # The note reaches the on-disk record too, not just the in-memory result.
    assert "Z pieces" in (tmp_path / "tight" / "export_info.txt").read_text()

    # A budget that comfortably holds a whole piece -> silence.
    assert roomy.exports
    assert not [n for n in roomy.notes if "Z pieces" in n], roomy.notes


def test_scratch_dir_is_a_subdirectory_and_only_for_median(tmp_path, monkeypatch):
    """Only a multi-pass centring caches, and never beside the user's PVTI."""
    seen: list = []
    real = A.align_volume_streamed

    def spy(*args, **kwargs):
        seen.append(kwargs.get("scratch_dir"))
        return real(*args, **kwargs)

    monkeypatch.setattr(PV.A, "align_volume_streamed", spy)

    proc, raw = _setup(tmp_path)
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": "",
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "export_strain": False,
        "num_pieces_z": 2,
    }

    PV.run({**params, "output_dir": str(tmp_path / "mean"), "center_method": "mean"})
    assert seen and set(seen) == {None}, seen
    assert not (tmp_path / "mean" / PV.SCRATCH_SUBDIR).exists()

    seen.clear()
    out = tmp_path / "median"
    PV.run({**params, "output_dir": str(out), "center_method": "median"})
    # The CoM fields centre (median -> multi-pass); the FWHM fields do not.
    cached = [d for d in seen if d is not None]
    assert len(cached) == 2, seen
    assert all(d == str(out / PV.SCRATCH_SUBDIR) for d in cached), cached
    # A subdirectory, and nothing left loose beside the products — and the
    # empty subdirectory itself is cleaned up when the run ends.
    assert not list(out.glob("dfxm_scratch*"))
    assert not (out / PV.SCRATCH_SUBDIR).exists()


def test_rss_floor_covers_the_measured_process_image(tmp_path):
    """`RSS_FLOOR_BYTES` must not sit below what this stage's child actually costs.

    The formula in `advice.working_set_budget_bytes` is pinned by the tests in
    `tests/test_common_advice.py`, but a *value* is not a formula: setting
    `RSS_FLOOR_BYTES = 1` passes every one of those, and the budget it derives
    is then far too large. Only a live measurement can catch that, and only a
    live measurement fails on the first VTK build where the constant stops
    travelling — which is the whole reason it is written down as a constant.

    **This is the recipe for Tasks 10-12.** Copy the call, not the number: the
    floor is per stage by construction, so each stage measures its own.
    """
    from tests.peak_rss import assert_floor_covers

    # The smallest real export, so what is measured is the process image and
    # not an export: four 8x8x4 float64 fields is 8 KB of data against a
    # ~229 MiB peak.
    layers, ny, nx = 4, 8, 8
    proc, raw = _setup(tmp_path, layers=layers, ny=ny, nx=nx)
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": "",
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "export_strain": False,
        "output_dir": str(tmp_path / "pv"),
        "num_pieces_z": 2,
    }
    measured = assert_floor_covers(
        PV.RSS_FLOOR_BYTES,
        "dfxm.stages.paraview:run",
        params,
        data_bytes=4 * layers * ny * nx * 8,
        samples=2,
        timeout=300,
    )
    # Not an equality: the constant carries deliberate slack for a heavier VTK
    # and for the model's blocking-dependent variance. What must hold is that
    # the slack is upward.
    assert measured <= PV.RSS_FLOOR_BYTES
