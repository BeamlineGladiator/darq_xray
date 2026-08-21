"""Tests for dfxm.common.h5io."""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from dfxm.common import h5io
from dfxm.common.h5io import StackedVolumeFile


def test_resolve_input_file_matches_folder_name():
    assert h5io.resolve_input_file("/data/scan_26") == "/data/scan_26/scan_26.h5"
    assert h5io.resolve_input_file("/data/scan_26/") == "/data/scan_26/scan_26.h5"


def test_resolve_input_file_override():
    assert h5io.resolve_input_file("/data/x", "other.h5") == "/data/x/other.h5"


def test_make_output_path():
    assert h5io.make_output_path("/d/a.h5") == "/d/a_concat.h5"
    assert h5io.make_output_path("/d/a.h5", "_stacked") == "/d/a_stacked.h5"


def test_get_filtered_entries_filters_and_sorts(bliss_factory):
    folder = bliss_factory(specs=(("2.1", 1), ("1.1", 1), ("10.1", 1), ("bad.2", 1)))
    with h5py.File(h5io.resolve_input_file(folder), "r") as f:
        assert h5io.get_filtered_entries(f, ".1") == ["1.1", "2.1", "10.1"]


def test_detector_info_rejects_non_3d(tmp_path):
    p = tmp_path / "x.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("a", data=np.zeros((3, 4)))
        f.create_dataset("b", data=np.zeros((3, 4, 5), dtype="uint16"))
        assert h5io.detector_info(f["b"]) == (3, (4, 5), np.dtype("uint16"))
        with pytest.raises(ValueError):
            h5io.detector_info(f["a"])


def test_vds_round_trip_resolves_source_frames(bliss_factory, tmp_path):
    folder = bliss_factory(specs=(("1.1", 3), ("2.1", 2)))
    raw = h5io.resolve_input_file(folder)
    out = str(tmp_path / "out.h5")
    with h5py.File(raw, "r") as f:
        entries = h5io.get_filtered_entries(f, ".1")
        sources, counts, fshape, dtype = [], [], None, None
        for e in entries:
            ds = f[f"{e}/instrument/pco_ff/image"]
            n, fshape, dtype = h5io.detector_info(ds)
            counts.append(n)
            sources.append(h5io.make_virtual_source(ds, out, "relative"))
        layout = h5io.build_virtual_layout(sources, counts, fshape, dtype)
        with h5py.File(out, "w") as o:
            o.create_virtual_dataset("entry_0000/measurement/pco_ff", layout)

        with h5py.File(out, "r") as o:
            vds = o["entry_0000/measurement/pco_ff"]
            assert vds.is_virtual and vds.shape == (5, 2, 3)
            # frame 3 (= first frame of entry 2.1) matches the source
            np.testing.assert_array_equal(vds[3], f["2.1/instrument/pco_ff/image"][0])


def test_make_virtual_source_rejects_bad_policy(bliss_factory):
    folder = bliss_factory(specs=(("1.1", 1),))
    with h5py.File(h5io.resolve_input_file(folder), "r") as f:
        ds = f["1.1/instrument/pco_ff/image"]
        with pytest.raises(ValueError):
            h5io.make_virtual_source(ds, "/tmp/o.h5", "bogus")


def test_read_positioners_and_samy_samz(bliss_factory):
    folder = bliss_factory(specs=(("1.1", 4),))
    with h5py.File(h5io.resolve_input_file(folder), "r") as f:
        pos = h5io.read_positioners(f, "1.1/instrument/positioners")
        assert set(pos) == {"mu", "ccmth", "obpitch", "samy", "samz"}
        samy, samz = h5io.read_samy_samz(f, "1.1/instrument/positioners")
        np.testing.assert_array_equal(samy, [0, 1, 2, 3])
        np.testing.assert_array_equal(samz, [5, 5, 5, 5])


def test_read_positioners_missing_group_raises(bliss_factory):
    folder = bliss_factory(specs=(("1.1", 1),))
    with h5py.File(h5io.resolve_input_file(folder), "r") as f:
        with pytest.raises(KeyError):
            h5io.read_positioners(f, "nope/positioners")


def test_validate_maps_file(tmp_path):
    p = str(tmp_path / "maps.h5")
    req = [
        "/entry/ccmth/Center of mass/Center of mass",
        "/entry/mu/Center of mass/Center of mass",
    ]
    with h5py.File(p, "w") as f:
        f.create_dataset(req[0], data=np.zeros((4, 4)))

    v = h5io.validate_maps_file(p, req)
    assert not v.ok and v.missing == [req[1]] and v.shapes[req[0]] == (4, 4)
    assert bool(v) is False

    with h5py.File(p, "a") as f:
        f.create_dataset(req[1], data=np.zeros((4, 4)))
    assert h5io.validate_maps_file(p, req).ok


def test_validate_maps_file_missing_file(tmp_path):
    v = h5io.validate_maps_file(str(tmp_path / "nope.h5"), ["/a"])
    assert not v.ok and v.error == "file not found"
    assert not os.path.exists(str(tmp_path / "nope.h5"))


# ---------------------------------------------------------------------------
# StackedVolumeFile
# ---------------------------------------------------------------------------


def test_stacked_volume_file_appends_layers(tmp_path):
    path = str(tmp_path / "stacked.h5")
    layers = [np.full((3, 4), i, dtype=np.float64) for i in range(5)]
    with StackedVolumeFile(path, compression=None) as out:
        for layer in layers:
            out.append("strain", layer)
        out.set_attrs(num_layers=len(layers))
        assert out.shape("strain") == (5, 3, 4)
    with h5py.File(path, "r") as f:
        assert np.array_equal(f["strain"][:], np.stack(layers, axis=0))
        assert f.attrs["num_layers"] == 5


def test_stacked_volume_file_rejects_shape_change(tmp_path):
    path = str(tmp_path / "stacked.h5")
    with pytest.raises(ValueError, match="differing shapes"):
        with StackedVolumeFile(path, compression=None) as out:
            out.append("strain", np.zeros((3, 4)))
            out.append("strain", np.zeros((3, 5)))


def test_stacked_volume_file_leaves_no_file_on_failure(tmp_path):
    path = str(tmp_path / "stacked.h5")
    with pytest.raises(ValueError):
        with StackedVolumeFile(path, compression=None) as out:
            out.append("strain", np.zeros((3, 4)))
            raise ValueError("boom")
    assert not (tmp_path / "stacked.h5").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_stacked_volume_file_nested_group_paths(tmp_path):
    path = str(tmp_path / "stacked.h5")
    with StackedVolumeFile(path, compression=None) as out:
        out.append("chi/Center_of_mass", np.zeros((2, 2)))
        out.append("mu/FWHM", np.zeros((2, 2)))
        out.append("chi/Center_of_mass", np.ones((2, 2)))
    with h5py.File(path, "r") as f:
        assert f["chi/Center_of_mass"].shape == (2, 2, 2)
        assert f["mu/FWHM"].shape == (1, 2, 2)


def test_stacked_volume_file_abort_then_clean_exit_is_a_noop(tmp_path):
    """The stages' "no layers produced" path: abort() and return from *inside*
    the with-block, so __exit__ sees no exception and calls close(). That close
    must not trip over the part file abort() already unlinked."""
    path = str(tmp_path / "stacked.h5")
    with StackedVolumeFile(path, compression=None) as out:
        out.abort()
    assert not (tmp_path / "stacked.h5").exists()
    assert list(tmp_path.glob("*.part")) == []


def test_stacked_volume_file_opens_lazily(tmp_path):
    """Nothing touches the filesystem until the first append, so a stage may
    construct the writer before a loop that turns out to produce no layers —
    even when the output directory does not exist (single mode pointed at a
    missing folder must stay a skip, not a raw FileNotFoundError)."""
    missing = tmp_path / "nope"
    out = StackedVolumeFile(str(missing / "stacked.h5"), compression=None)
    assert not missing.exists()
    out.close()  # clean exit with nothing appended commits nothing
    assert not missing.exists()


def test_stacked_volume_file_set_attrs_before_append_raises(tmp_path):
    """Attributes must never be silently dropped for want of an open file."""
    out = StackedVolumeFile(str(tmp_path / "stacked.h5"), compression=None)
    with pytest.raises(ValueError, match="no layer has been appended"):
        out.set_attrs(num_layers=0)
    with pytest.raises(ValueError, match="no layer has been appended"):
        out.shape("strain")


def test_stacked_volume_file_rejects_dtype_change(tmp_path):
    """The dataset dtype is fixed by the first layer, so a wider later layer
    would be truncated on write where np.stack promoted. Refuse instead."""
    path = str(tmp_path / "stacked.h5")
    with pytest.raises(ValueError, match="differing dtypes"):
        with StackedVolumeFile(path, compression=None) as out:
            out.append("strain", np.zeros((3, 4), dtype=np.float32))
            out.append("strain", np.zeros((3, 4), dtype=np.float64))
    assert not (tmp_path / "stacked.h5").exists()


def test_stacked_volume_file_removes_orphaned_part_file_on_construction(tmp_path):
    """A cancelled run is SIGKILLed, so its .part survives — gigabytes, in the
    experiment root. Reclaiming it must not depend on the re-run producing any
    layers, because a run that produces none never opens the part file at all.
    So: construct, append nothing, and the orphan is already gone."""
    orphan = tmp_path / "stacked.h5.part"
    orphan.write_bytes(b"not even valid HDF5")
    out = StackedVolumeFile(str(tmp_path / "stacked.h5"), compression=None)
    assert not orphan.exists()
    out.close()  # nothing appended -> commits nothing
    assert not (tmp_path / "stacked.h5").exists()


def test_stacked_volume_file_construction_creates_nothing(tmp_path):
    """The orphan reclaim must not itself touch the filesystem when there is
    no orphan — in particular it must not create the missing directory of a
    mistyped single-mode input folder."""
    missing = tmp_path / "nope"
    StackedVolumeFile(str(missing / "stacked.h5"), compression=None)
    assert not missing.exists()
    assert list(tmp_path.iterdir()) == []


def test_stacked_volume_file_first_close_failure_still_raises(tmp_path):
    """The _closed flag makes *repeat* calls no-ops; it must never swallow a
    genuinely failing first commit. Here the destination is a directory, so
    os.replace cannot succeed."""
    path = tmp_path / "stacked.h5"
    path.mkdir()  # os.replace(file, dir) fails
    out = StackedVolumeFile(str(path), compression=None)
    out.append("strain", np.zeros((2, 2)))
    with pytest.raises(OSError):
        out.close()
    # the failure left the part file in place — nothing was silently discarded
    assert (tmp_path / "stacked.h5.part").exists()
    out.abort()  # and it is still cleanable afterwards
    assert not (tmp_path / "stacked.h5.part").exists()


def test_stacked_volume_file_abort_after_close_is_a_noop(tmp_path):
    """The mirror case: a committed file is never unlinked by a late abort()."""
    path = str(tmp_path / "stacked.h5")
    out = StackedVolumeFile(path, compression=None)
    out.append("strain", np.zeros((2, 2)))
    out.close()
    out.abort()
    out.close()
    assert os.path.exists(path)
    assert list(tmp_path.glob("*.part")) == []
