"""Tests for dfxm.stages.paraview — PVTI partitioning (parity with the legacy
exporter), end-to-end file writing, and a VTK round-trip of values + valid_mask.
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import replace
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


# -- the two rungs ------------------------------------------------------------
def _rung_spies(monkeypatch):
    """Record which rung each field took and how it blocked, changing neither.

    ``_drained`` is called by the **in-core rung and by nothing else**, so
    counting it pins which rung ran rather than merely observing that the run
    finished. A test that watched only the blocking would still pass with the
    ladder deleted and the stage streaming unconditionally, which is the whole
    thing under test here.
    """
    drained: list[tuple] = []
    real_drain = PV._drained

    def drain_spy(provider):
        drained.append(tuple(provider.shape))
        return real_drain(provider)

    blockings: list[tuple[int, int]] = []
    real_align = A.align_volume_streamed

    def align_spy(*args, **kwargs):
        streamed = real_align(*args, **kwargs)
        blockings.append((int(streamed.block_layers), int(streamed.shape[0])))
        return streamed

    monkeypatch.setattr(PV, "_drained", drain_spy)
    monkeypatch.setattr(PV.A, "align_volume_streamed", align_spy)
    return drained, blockings


def test_fits_in_core_asks_the_blocking_and_answers_for_the_whole_set():
    """The rung selector is a live function of the budget, not a constant."""
    vol = _synthetic_volume(seed=4)
    roomy = _as_streamed(vol, budget_bytes=1 << 30)
    tight = _as_streamed(vol, budget_bytes=512 << 10)
    # Preconditions: one really is a single block and the other really is not,
    # so neither branch below can pass for the wrong reason after a sizing change.
    assert roomy.block_layers >= roomy.shape[0], roomy.block_layers
    assert tight.block_layers < tight.shape[0], tight.block_layers

    assert PV._fits_in_core({"a": roomy})
    assert not PV._fits_in_core({"a": tight})
    # All-or-nothing across the set: one piece of EVERY field is resident at
    # once, so a single field short of its share sends the whole export
    # streaming. The fields do not block alike (a centred CoM field carries the
    # centring statistic's working set and a plain FWHM field does not), so this
    # is a case that really occurs.
    assert not PV._fits_in_core({"a": roomy, "b": tight})
    # No fields at all is not "it fits" — there is nothing to materialise and
    # nothing to write.
    assert not PV._fits_in_core({})


@pytest.mark.parametrize("center_method", ["mean", "median"])
def test_both_rungs_write_identical_bytes(tmp_path, monkeypatch, center_method):
    """Which rung runs depends on the machine, so the products may not differ.

    The strictest form of the burden in this project: the stage advertises
    byte-for-byte pieces, so any behavioural difference between the rungs would
    be a **machine-dependent difference in the exported file**. `visualize`
    shipped two of those (``~isnan`` vs ``isfinite``, ``np.nanmean`` vs
    ``stream_mean``); this stage avoids them the way `slices` does, by having one
    implementation of every statistic and letting the rung decide only where the
    blocks come from.

    The two exports are budgeted **separately** because they cannot cross the
    rung at one budget: the mosaicity budget is divided by its four concurrent
    field streams while the lone strain stream keeps the whole of it, so a budget
    small enough to block strain leaves each mosaicity field under the alignment
    chain's own one-layer floor.
    """
    proc, raw = _setup(tmp_path, layers=S_L, ny=S_NY, nx=S_NX)
    common = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "num_pieces_z": 3,
        "center_method": center_method,
        "center_strain": True,
        "anchor_origin_to_reference": True,
    }
    # The frugal budget is per centring, because a median traverses more than
    # once and `align_volume_streamed` prices that scaffolding into the working
    # set: the budget that blocks a mean-centred field leaves a median-centred
    # one under its own one-layer floor.
    frugal_mosa, frugal_strain = (
        (4 << 20, 1 << 20) if center_method == "mean" else (12 << 20, 3 << 20)
    )
    cases = (
        ("mosaicity", {"export_strain": False}, frugal_mosa, 4),
        ("strain", {"export_mosaicity": False}, frugal_strain, 1),
    )
    for label, only, frugal, n_fields in cases:
        outs = {}
        for tag, budget, want_in_core in (("core", 1 << 30, True), ("stream", frugal, False)):
            drained, blockings = _rung_spies(monkeypatch)
            out = tmp_path / f"{label}_{tag}"
            with warnings.catch_warnings():
                # A "budget too small" warning would mean the frugal run fell
                # back to one layer per block instead of demonstrating a
                # blocking, so it is an error here rather than noise.
                warnings.simplefilter("error")
                PV.run({**common, **only, "output_dir": str(out), "_budget_bytes": budget})
            monkeypatch.undo()
            assert len(blockings) == n_fields, (label, tag, blockings)
            if want_in_core:
                assert len(drained) == n_fields, f"{label}/{tag} did not take the in-core rung"
                assert all(b >= nz for b, nz in blockings), blockings
            else:
                assert not drained, f"{label}/{tag} materialised a volume anyway"
                # `any`, not `all`: the fields do not block alike, and the rung
                # is all-or-nothing, so a set where only the centred field is
                # short of its share still streams. What must hold is that the
                # budget really bit somewhere.
                assert any(b < nz for b, nz in blockings), blockings
            outs[tag] = out

        pieces = sorted(p.relative_to(outs["core"]) for p in outs["core"].rglob("*.vti"))
        assert len(pieces) == 3, pieces
        for rel in pieces:
            assert (outs["stream"] / rel).read_bytes() == (outs["core"] / rel).read_bytes(), rel
        master = f"{label}_volume.pvti"
        assert (outs["stream"] / master).read_text() == (outs["core"] / master).read_text()


def test_the_in_core_rung_matches_the_pre_ladder_in_core_writer(tmp_path):
    """The in-core rung reproduces the writer the streaming conversion replaced.

    `save_volumes_as_pvti` is a **structurally different** implementation — it
    builds the combined mask over whole arrays, reads the finite range straight
    off them and cleans in float64 before `write_piece_vti` casts — so agreeing
    with it is evidence the ladder did not quietly redefine the sentinel, the
    mask or the rounding. It is also the only in-repo expression of "the same
    bytes as before this phase": the numbers pinned in the docs came from a
    commit this test cannot reach, and a claim no test can see is a claim that
    stops being checked.
    """
    proc, raw = _setup(tmp_path, layers=S_L, ny=S_NY, nx=S_NX)
    out = tmp_path / "run"
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": "",
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "export_strain": False,
        "output_dir": str(out),
        "num_pieces_z": 3,
        "_budget_bytes": 1 << 30,  # the in-core rung
    }
    result = PV.run(params)
    assert [e.name for e in result.exports] == ["mosaicity"]

    # The pre-ladder stage, rebuilt: align every field whole, hand the arrays to
    # the in-core writer.
    path = str(proc / "stacked_volumes.h5")
    defaults = PV.STAGE.defaults()
    samy, samz = PV._motors(str(raw), "mosa__*", defaults["samy_path"], defaults["samz_path"])
    aligned = {}
    for name in PV.mosa_field_names(path):
        aligned[name] = A.align_volume(
            PV.load_mosa_field(path, name),
            samy,
            samz,
            scale_x=float(defaults["pixel_size_x_um"]),
            samy_direction=int(defaults["samy_direction"]),
            take_abs="FWHM" in name,
            center_method=defaults["center_method"] if "Center_of_mass" in name else None,
        )
    first = next(iter(aligned.values()))
    ref_dir = tmp_path / "ref"
    ref_dir.mkdir()
    ref = PV.save_volumes_as_pvti(
        {name: av.data for name, av in aligned.items()},
        (float(defaults["pixel_size_x_um"]), float(defaults["pixel_size_y_um"]), first.scale_z_um),
        str(ref_dir / "mosaicity_volume.pvti"),
        n_pieces=3,
    )
    # The fixture must reach the branches whose definitions could drift: NaN
    # padding (sentinel + valid_mask) and a real interpolation.
    assert ref["nan_sentinel"] is not None and ref["padded_fraction"] > 0
    assert first.data.shape[2] != S_NX, "no samy X-pad — the shift did nothing to agree about"
    assert not np.allclose(first.z_uniform_um, A._z_positions_um(samz, S_L, None)), (
        "the uniform Z grid IS the input grid — there is no interpolation to agree about"
    )

    written = sorted(p.relative_to(out) for p in out.rglob("*.vti"))
    assert len(written) == 3, written
    for rel in written:
        assert (out / rel).read_bytes() == (ref_dir / rel).read_bytes(), rel
    assert (out / "mosaicity_volume.pvti").read_text() == (
        ref_dir / "mosaicity_volume.pvti"
    ).read_text()


@pytest.mark.parametrize("budget,want_alignments", [(1 << 30, 1), (4 << 20, 2)])
def test_the_in_core_rung_aligns_each_field_once(tmp_path, monkeypatch, budget, want_alignments):
    """`_survey` is not skipped in-core — it stops being a second alignment.

    `blocks` is a factory and every call re-runs the whole alignment chain, so
    the streaming rung's two passes (`_survey`, then the pieces) align each field
    **twice**; that second traversal is where the 1.2-1.7x went. The in-core rung
    drains the stream once and both passes then walk resident memory.

    Counting the traversals rather than timing them, because a wall-clock
    assertion in a test suite is a flake. This is the mechanism the measurement
    in the report attributes the recovery to, and it fails if the in-core rung is
    removed *or* if it is made to survey the stream instead of the array.
    """
    proc, raw = _setup(tmp_path, layers=S_L, ny=S_NY, nx=S_NX)
    counts: list[list[int]] = []
    real = A.align_volume_streamed

    def spy(*args, **kwargs):
        streamed = real(*args, **kwargs)
        tally = [0]
        counts.append(tally)
        inner = streamed.blocks

        def counting():
            tally[0] += 1
            return inner()

        return replace(streamed, blocks=counting)

    monkeypatch.setattr(PV.A, "align_volume_streamed", spy)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        PV.run(
            {
                "mosa_volume_file": str(proc / "stacked_volumes.h5"),
                "strain_volume_file": "",
                "raw_root": str(raw),
                "mosa_pattern": "mosa__*",
                "export_strain": False,
                "output_dir": str(tmp_path / f"out_{budget}"),
                "num_pieces_z": 3,
                "_budget_bytes": budget,
            }
        )
    assert len(counts) == 4, counts  # four mosaicity fields
    assert [t[0] for t in counts] == [want_alignments] * 4, counts


def test_run_pieces_are_budget_independent(tmp_path, monkeypatch):
    """The exported bytes do not depend on how the *streamed* run was blocked.

    Two **streaming** budgets, not a streamed run against an in-core one — that
    comparison is `test_both_rungs_write_identical_bytes`'s, and letting one side
    of this one drift onto the in-core rung would leave the streamed writer's
    slab bookkeeping compared against nothing. Mosaicity only: the strain export
    keeps the whole budget where the four concurrent mosaicity streams each get a
    quarter, so no single budget blocks both.
    """
    # Big enough that the budgets below are budgets the run can actually meet —
    # on the 4-layer fixture no budget above `align_volume_streamed`'s own floor
    # blocks at all, so both runs would warn and fall back to one layer instead
    # of demonstrating anything.
    proc, raw = _setup(tmp_path, layers=S_L, ny=S_NY, nx=S_NX)
    base = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": "",
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "export_strain": False,
        "num_pieces_z": 2,
    }

    seen: dict[int, list[tuple[int, int]]] = {}

    def _record(budget):
        drained, blockings = _rung_spies(monkeypatch)
        out = tmp_path / f"out_{budget}"
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            PV.run({**base, "output_dir": str(out), "_budget_bytes": budget})
        monkeypatch.undo()
        assert not drained, f"budget {budget} took the in-core rung — nothing streamed"
        seen[budget] = blockings
        return out

    coarse_budget, fine_budget = 4 << 20, 3 << 20
    coarse = _record(coarse_budget)
    fine = _record(fine_budget)

    # The measurement has to be live: four fields aligned each time, every one of
    # them blocked, and the two runs blocked *differently*. Without this the two
    # could be identically blocked and the comparison below would compare a run
    # with itself.
    assert len(seen[coarse_budget]) == len(seen[fine_budget]) == 4
    for budget in (coarse_budget, fine_budget):
        assert all(layers < nz for layers, nz in seen[budget]), seen[budget]
    assert seen[coarse_budget] != seen[fine_budget], seen

    written = sorted(p.relative_to(coarse) for p in coarse.rglob("*.vti"))
    assert len(written) == 2
    for rel in written:
        assert (fine / rel).read_bytes() == (coarse / rel).read_bytes(), rel
    assert (fine / "mosaicity_volume.pvti").read_text() == (
        coarse / "mosaicity_volume.pvti"
    ).read_text()


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


def test_in_core_rung_keeps_the_streamed_writers_saving(tmp_path, monkeypatch):
    """Restoring an in-core rung must not restore the in-core *writer*.

    The ladder's rung is about the **alignment**, not the piece writer: on both
    rungs each `.vti` piece is built from that piece's Z-slab and dropped, which
    is a strict improvement independent of how much memory the machine has.
    Routing the in-core rung through `save_volumes_as_pvti` instead — the obvious
    reconstruction, since that function is still here as the reference
    implementation — would build a `np.where`-cleaned float64 copy of every
    field and a whole-volume boolean mask before the first piece is written, and
    hand back the ~586-593 MiB this stage used to cost.

    Same 128x192x192 four-field export as the streamed peak test. Measured
    **409-421 MiB** on the in-core rung at 16 pieces over four runs, against the
    576 MiB limit.

    **The failing side is the regression itself, measured.** The variant this
    guards against — the in-core rung routed through `save_volumes_as_pvti` —
    was built and run twice: **594.9 and 595.1 MiB**. So the limit sits between
    421 (passing, 155 MiB of margin) and 595 (failing by 19 MiB). The failing
    margin is the thinner one, and it holds because the regression is a
    deterministic +174 MiB of float64 cleaned copies, an order of magnitude
    above the ~12 MiB run-to-run spread. The limit is kept high rather than
    centred so a heavier VTK build — the process image is ~229 MiB here and
    `RSS_FLOOR_BYTES` declares 300 MB — does not flake it.

    Deliberately *not* calibrated against the same run at `num_pieces_z = 1`
    (566.4 MiB): that is under the limit, so it witnesses nothing here. It is
    evidence for a different claim — see `_writable_providers`.

    **The budget must be 2 GiB, not 1.** `align_volume_streamed`'s working set
    for a whole block of this fixture is ~430 MB per field, and the mosaicity
    budget is divided by the four concurrent streams — so a 1 GiB budget leaves
    each field 256 MB, blocks at 75/128 layers, and takes the **streaming** rung.
    This test was written that way and measured the streaming rung while its
    docstring claimed the in-core one; it then passed with `_fits_in_core`
    forced to `False`, i.e. with the whole ladder deleted. `slices` has the
    identical defect on record (`docs/Codebase.md`, `_budget_bytes = 64 MiB`
    "streamed too"), one wave earlier. Hence the precondition below, which is
    not optional decoration: the measurement runs in a spawn child where nothing
    can be spied, so without it this test cannot see its own subject disappear.
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
        # Pinned rather than measured from the machine: a laptop would take the
        # streaming rung here and measure something else entirely.
        "_budget_bytes": 2 << 30,
    }

    # Precondition, in process and on the REAL decision: `_writable_providers`
    # is handed the providers the stage actually built, so this reads the rung
    # off the same objects the child will, and aborts before a voxel is read
    # (the blocking is solved from shapes and motor arrays alone).
    class _Stop(Exception):
        pass

    decisions: list[bool] = []

    def probe(providers, **_kwargs):
        decisions.append(PV._fits_in_core(providers))
        raise _Stop

    monkeypatch.setattr(PV, "_writable_providers", probe)
    with pytest.raises(_Stop):
        PV.run(params)
    monkeypatch.undo()
    assert decisions == [True], (
        f"the budget does not select the in-core rung ({decisions}) — this test would "
        "measure the streaming rung and pass with the ladder deleted"
    )

    result = assert_peak_under(
        "dfxm.stages.paraview:run", params, limit_bytes=576 * (1 << 20), timeout=600
    )
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


def test_run_warns_when_z_pieces_are_too_few_for_the_budget(tmp_path, monkeypatch):
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
    runs = {}
    for tag, budget in (("tight", 8 << 20), ("roomy", 32 << 20), ("in_core", 1 << 30)):
        drained, _blockings = _rung_spies(monkeypatch)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no alignment-floor warning may fire here
            runs[tag] = PV.run(
                {**params, "output_dir": str(tmp_path / tag), "_budget_bytes": budget}
            )
        monkeypatch.undo()
        runs[tag + "_in_core"] = bool(drained)

    tight, roomy = runs["tight"], runs["roomy"]
    assert tight.exports, "the run must still produce its export — advice, not a refusal"
    assert any("Z pieces" in n for n in tight.notes), tight.notes
    assert "raise Z pieces to" in " ".join(tight.notes)
    # The note reaches the on-disk record too, not just the in-memory result.
    assert "Z pieces" in (tmp_path / "tight" / "export_info.txt").read_text()

    # A budget that comfortably holds a whole piece -> silence. Both budgets must
    # still be on the STREAMING rung, or the two runs would differ in the rung as
    # well as the budget and this would have stopped isolating the advice.
    assert not runs["tight_in_core"] and not runs["roomy_in_core"], runs
    assert roomy.exports
    assert not [n for n in roomy.notes if "Z pieces" in n], roomy.notes

    # The advice is a property of the PIECE pass, which both rungs share, so it
    # is deliberately NOT gated on the rung — a piece of every field is held on
    # top of whatever the alignment left resident either way (measured: a
    # 128x192x192 four-field export at a 1 GiB budget peaks at 492.8 MiB at 16
    # pieces and 632.5 MiB at one). On a real in-core run it nonetheless has
    # nothing to say, and that is arithmetic rather than suppression: taking the
    # in-core rung requires each field's share of the budget to hold a whole
    # alignment working set (~5x the volume), which already leaves room for one
    # piece of every field.
    assert runs["in_core_in_core"], "the generous budget did not take the in-core rung"
    assert runs["in_core"].exports
    assert not [n for n in runs["in_core"].notes if "Z pieces" in n], runs["in_core"].notes
    # ... so the check that the path is not gated has to be made directly, or
    # "silent in-core" and "suppressed in-core" would be indistinguishable here.
    in_core_notes: list[str] = []
    PV._writable_providers(
        {"one": _as_streamed(_synthetic_volume(seed=6), budget_bytes=1 << 30)},
        budget_bytes=1,
        n_pieces=1,
        write_valid_mask=True,
        label="probe",
        notes=in_core_notes,
    )
    assert any("Z pieces" in n for n in in_core_notes), in_core_notes


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
