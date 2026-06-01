"""Tests for dfxm.common.h5io."""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from dfxm.common import h5io


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
