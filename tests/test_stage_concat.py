"""Tests for darq_xray.stages.concat — structure, lengths, modes, and a golden
equivalence check against the legacy concatenate_h5_scans_v3 script.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from darq_xray.common import h5io
from darq_xray.stages import concat


# -- single mode --------------------------------------------------------------
def test_single_mode_structure_and_lengths(bliss_factory):
    folder = bliss_factory(specs=(("1.1", 3), ("2.1", 2), ("3.1", 4)))
    result = concat.run({"mode": "single", "input_folder": folder})

    assert result.n_ok == 1 and result.n_failed == 0
    fr = result.files[0]
    assert fr.total_frames == 9 and fr.n_entries == 3 and not fr.copied

    out = h5io.make_output_path(h5io.resolve_input_file(folder))
    assert os.path.exists(out)
    with h5py.File(out, "r") as f:
        det = f["entry_0000/measurement/pco_ff"]
        assert det.is_virtual and det.shape == (9, 2, 3)
        pos = f["entry_0000/instrument/positioners"]
        # varying scalar ccmth -> per-frame array; constant obpitch -> scalar
        assert pos["ccmth"].shape == (9,)
        assert pos["mu"].shape == (9,)
        assert pos["obpitch"].shape == ()
        assert f["entry_0000"].attrs["num_scans"] == 3


def test_copy_data_writes_self_contained_dataset(bliss_factory):
    folder = bliss_factory(specs=(("1.1", 2), ("2.1", 3)))
    # reference frames via a VDS run
    concat.run({"mode": "single", "input_folder": folder})
    out = h5io.make_output_path(h5io.resolve_input_file(folder))
    with h5py.File(out, "r") as f:
        vds_frames = f["entry_0000/measurement/pco_ff"][()]

    fr = concat.run({"mode": "single", "input_folder": folder, "copy_data": True}).files[0]
    assert fr.copied
    with h5py.File(out, "r") as f:
        det = f["entry_0000/measurement/pco_ff"]
        assert not det.is_virtual
        np.testing.assert_array_equal(det[()], vds_frames)


def test_single_mode_requires_input_folder():
    with pytest.raises(ValueError):
        concat.run({"mode": "single"})


# -- batch mode ---------------------------------------------------------------
def test_batch_mode_processes_all_folders(batch_root):
    result = concat.run({"mode": "batch", "root_folder": batch_root, "folder_pattern": "layer__*"})
    assert result.n_ok == 3 and result.n_failed == 0
    for out in result.outputs:
        assert os.path.exists(out) and out.endswith("_concat.h5")


def test_batch_skip_existing(batch_root):
    concat.run({"mode": "batch", "root_folder": batch_root, "folder_pattern": "layer__*"})
    again = concat.run(
        {
            "mode": "batch",
            "root_folder": batch_root,
            "folder_pattern": "layer__*",
            "skip_existing": True,
        }
    )
    assert again.n_skipped == 3 and again.n_ok == 0
    assert all(f.error == "output exists (skip_existing)" for f in again.files)


def test_batch_no_matching_folders_raises(tmp_path):
    with pytest.raises(ValueError):
        concat.run({"mode": "batch", "root_folder": str(tmp_path), "folder_pattern": "none__*"})


# -- golden equivalence vs the legacy script ---------------------------------
def _import_legacy():
    """Import concatenate_h5_scans_v3 from the parent repo, or skip."""
    repo_root = Path(__file__).resolve().parents[2]  # worktree root, above the darq_xray checkout
    candidate = repo_root / "concatenate_h5_scans_v3.py"
    if not candidate.exists():
        pytest.skip("legacy concatenate_h5_scans_v3.py not found alongside the package")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import concatenate_h5_scans_v3 as legacy

    return legacy


def test_matches_legacy_concatenate_h5_scans_v3(bliss_factory, tmp_path):
    legacy = _import_legacy()
    folder = bliss_factory(specs=(("1.1", 3), ("2.1", 4), ("3.1", 2)))
    raw = h5io.resolve_input_file(folder)

    gold = str(tmp_path / "gold.h5")
    with h5py.File(raw, "r") as h5f:
        entries = legacy.get_filtered_entries(h5f, legacy.ENTRY_SUFFIX)
        legacy.write_output(
            output_path=gold,
            input_path=raw,
            entries=entries,
            h5f_in=h5f,
            overwrite=True,
            vds_policy="relative",
            output_entry="entry_0000",
        )

    mine = str(tmp_path / "mine.h5")
    assert concat.concatenate_single_file(raw, mine, vds_policy="relative").ok

    def dump(path):
        with h5py.File(path, "r") as f:
            det = f["entry_0000/measurement/pco_ff"][()]
            pg = f["entry_0000/instrument/positioners"]
            pos = {k: pg[k][()] for k in pg.keys()}
        return det, pos

    gdet, gpos = dump(gold)
    mdet, mpos = dump(mine)
    np.testing.assert_array_equal(gdet, mdet)
    assert set(gpos) == set(mpos)
    for k in gpos:
        np.testing.assert_array_equal(np.atleast_1d(gpos[k]), np.atleast_1d(mpos[k]))
