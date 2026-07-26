"""Marks storage in oblique_slices.h5: round-trip, snapping, reader hardening."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dfxm.common.errors import StageUserError
from dfxm.stages import profiles as pr
from dfxm.stages import slices as sl


def _mini(path, offsets=(-2.0, 0.0, 2.0)):
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offs = np.asarray(offsets, np.float64)
    with h5py.File(path, "w") as f:
        for vid in ("raw_sum", "strain"):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cmap"] = "gray"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "v"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("oblique_full")
            sg.create_dataset(
                "slices", data=np.zeros((offs.size, v.size, u.size), dtype=np.float32)
            )
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offs)
            for key, val in (
                ("normal", (0.0, 0.0, 1.0)),
                ("origin", (0.0, 0.0, 0.0)),
                ("up", (0.0, 1.0, 0.0)),
                ("u_hat", (1.0, 0.0, 0.0)),
                ("v_hat", (0.0, 1.0, 0.0)),
                ("n_hat", (0.0, 0.0, 1.0)),
            ):
                sg.attrs[key] = np.asarray(val, np.float64)
            for key, val in (
                ("half_u", 4.0),
                ("half_v", 3.0),
                ("du", 1.0),
                ("dv", 1.0),
                ("sweep_step_um", 2.0),
            ):
                sg.attrs[key] = float(val)
            sg.attrs["n_planes"] = int(offs.size)
    return str(path)


def test_write_and_read_marks_roundtrip(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    snapped = sl.write_marks(h5, "oblique_full", [0.3, -1.7, 0.4])
    assert snapped == [-2.0, 0.0]  # snapped to stored planes, deduped, sorted
    assert sl.read_marks(h5) == {"oblique_full": [-2.0, 0.0]}
    with h5py.File(h5, "r") as f:  # open-file variant
        assert sl.read_marks(f) == {"oblique_full": [-2.0, 0.0]}


def test_write_marks_replaces_and_deletes(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    sl.write_marks(h5, "oblique_full", [0.0])
    sl.write_marks(h5, "oblique_full", [2.0])  # replace, not append
    assert sl.read_marks(h5) == {"oblique_full": [2.0]}
    sl.write_marks(h5, "oblique_full", [])  # empty -> dataset and group gone
    assert sl.read_marks(h5) == {}
    with h5py.File(h5, "r") as f:
        assert sl.MARKS_GROUP not in f


def test_write_marks_unknown_slice_raises(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    sl.write_marks(h5, "oblique_full", [0.0])  # creates /marks alongside the volumes
    with pytest.raises(StageUserError) as ei:
        sl.write_marks(h5, "nope", [0.0])
    assert "marks" not in ei.value.hint
    assert "raw_sum" in ei.value.hint


def test_read_marks_absent_and_malformed(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    assert sl.read_marks(h5) == {}  # no /marks group at all
    with h5py.File(h5, "a") as f:
        mg = f.require_group(sl.MARKS_GROUP)
        mg.create_group("weird_subgroup")  # non-dataset child: skipped
        mg.create_dataset("strs", data=np.bytes_([b"x"]))  # non-numeric: skipped
        mg.create_dataset("oblique_full", data=np.asarray([2.0], np.float64))
    assert sl.read_marks(h5) == {"oblique_full": [2.0]}


def test_readers_skip_marks_group(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    sl.write_marks(h5, "oblique_full", [0.0])
    # replot_catalog: volume groups only
    cat = sl.replot_catalog(h5)
    assert sorted({e.volume_id for e in cat}) == ["raw_sum", "strain"]
    # build_pinned_spec: still resolves geometry (would crash on /marks before)
    specs = sl.build_pinned_spec(h5, "oblique_full", [0.0])
    assert len(specs) == 1
    # figures catalog: one spec per (vid, slice, plane) — none for /marks
    result = SimpleNamespace(output_h5=h5)
    specs = sl.figures(result, {})
    assert len(specs) == 6  # 2 vids x 1 slice x 3 planes
    # profiles enumerators
    with h5py.File(h5, "r") as f:
        assert pr.list_volume_ids(f) == ["raw_sum", "strain"]
        assert pr.volume_ids_with_slice(f, "oblique_full") == ["raw_sum", "strain"]
