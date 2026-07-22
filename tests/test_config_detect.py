"""Tests for dfxm.config.detect (data-driven experiment initialization)."""

from __future__ import annotations

import h5py

from dfxm.config.detect import (
    Detection,
    detect_entry_suffix,
    detect_patterns,
    folder_families,
    select_scan_file,
)


def _mkdirs(root, *names):
    for n in names:
        (root / n).mkdir(parents=True)


def _by_field(rows: list[Detection]) -> dict[str, Detection]:
    return {d.field: d for d in rows}


# -- folder families / patterns -----------------------------------------------


def test_folder_families_groups_and_counts(tmp_path):
    _mkdirs(tmp_path, "s_strain__0", "s_strain__1", "s_mosa__0", "loose_folder")
    (tmp_path / "s_strain__2").write_text("")  # a FILE — must be ignored
    fams = folder_families(str(tmp_path))
    assert fams == {"s_strain": 2, "s_mosa": 1}


def test_folder_families_missing_root_is_empty():
    assert folder_families("/nonexistent/nowhere") == {}


def test_detect_patterns_classifies_families(tmp_path):
    _mkdirs(
        tmp_path,
        "s_energy_strain__0",
        "s_energy_strain__1",
        "s_energy_strain__2",
        "s_mosa__0",
        "s_mosa__1",
        "s_rocking__0",
    )
    rows = _by_field(detect_patterns(str(tmp_path)))
    assert rows["folder_pattern"].value == "s_energy_strain__*"
    assert rows["mosa_pattern"].value == "s_mosa__*"
    assert rows["rocking_pattern"].value == "s_rocking__*"
    assert "3 folders" in rows["folder_pattern"].note


def test_detect_patterns_missing_family_skips_with_reason(tmp_path):
    _mkdirs(tmp_path, "s_strain__0")
    rows = _by_field(detect_patterns(str(tmp_path)))
    assert rows["folder_pattern"].value == "s_strain__*"
    assert rows["mosa_pattern"].value is None and "mosa" in rows["mosa_pattern"].error
    assert rows["rocking_pattern"].value is None


def test_detect_patterns_no_families_at_all(tmp_path):
    _mkdirs(tmp_path, "no_numeric_suffix")
    rows = detect_patterns(str(tmp_path))
    assert len(rows) == 1
    assert rows[0].field == "folder_pattern" and rows[0].error


def test_detect_patterns_largest_family_wins(tmp_path):
    _mkdirs(tmp_path, "a__0", "b__0", "b__1")
    rows = _by_field(detect_patterns(str(tmp_path)))
    assert rows["folder_pattern"].value == "b__*"


# -- scan file selection ------------------------------------------------------


def test_select_scan_file_prefers_folder_name(tmp_path):
    d = tmp_path / "layer__0"
    d.mkdir()
    (d / "aaa_first_alphabetically.h5").write_text("")
    (d / "layer__0.h5").write_text("")
    assert select_scan_file(str(d)).endswith("layer__0.h5")


def test_select_scan_file_excludes_concat(tmp_path):
    d = tmp_path / "layer__0"
    d.mkdir()
    (d / "layer__0_concat.h5").write_text("")
    (d / "other_scan.h5").write_text("")
    assert select_scan_file(str(d)).endswith("other_scan.h5")


def test_select_scan_file_none_when_only_concat(tmp_path):
    d = tmp_path / "layer__0"
    d.mkdir()
    (d / "layer__0_concat.h5").write_text("")
    assert select_scan_file(str(d)) is None


# -- entry suffix -------------------------------------------------------------


def _write_entries(path, *entries):
    with h5py.File(path, "w") as f:
        for e in entries:
            f.create_group(e)
    return str(path)


def test_detect_entry_suffix_majority(tmp_path):
    p = _write_entries(tmp_path / "s.h5", "1.1", "2.1", "3.1", "2.2")
    d = detect_entry_suffix(p)
    assert d.value == ".1"
    assert "mixed" in d.note  # the minority .2 is called out


def test_detect_entry_suffix_clean(tmp_path):
    p = _write_entries(tmp_path / "s.h5", "1.1", "2.1")
    d = detect_entry_suffix(p)
    assert d.value == ".1" and "mixed" not in d.note


def test_detect_entry_suffix_no_entries(tmp_path):
    p = _write_entries(tmp_path / "s.h5", "not_an_entry")
    d = detect_entry_suffix(p)
    assert d.value is None and d.error


def test_detect_entry_suffix_unreadable_file(tmp_path):
    p = tmp_path / "junk.h5"
    p.write_text("this is not hdf5")
    d = detect_entry_suffix(str(p))
    assert d.value is None and d.error


# -- calibration detectors ----------------------------------------------------

import numpy as np  # noqa: E402  (test-section import, keeps diffs local)

from dfxm.config.detect import (  # noqa: E402
    detect_ccmth_from_maps,
    detect_ccmth_from_positioners,
    detect_darfix_roi,
    detect_pixel_sizes,
    find_strain_maps,
)

CCMTH_COM = "/entry/ccmth/Center of mass/Center of mass"


def _write_scan(path, *, entry="1.1", ccmth=7.1, **extra):
    """Minimal BLISS scan: the five pixel-size motors + ccmth."""
    motors = dict(mainx=-5000.0, obx=273.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0)
    motors.update(extra)
    if ccmth is not None:
        motors["ccmth"] = ccmth
    with h5py.File(path, "w") as f:
        pos = f.create_group(f"{entry}/instrument/positioners")
        for name, val in motors.items():
            pos.create_dataset(name, data=val)
    return str(path)


def _write_maps(path, *, with_ccmth=True, shape=(6, 8), fill=7.5):
    with h5py.File(path, "w") as f:
        if with_ccmth:
            data = np.full(shape, fill)
            data[0, 0] = np.nan  # nanmedian must survive NaNs
            f.create_dataset(CCMTH_COM, data=data)
        else:
            f.create_dataset("/entry/chi/Center of mass/Center of mass", data=np.zeros(shape))
    return str(path)


def test_detect_pixel_sizes_success(tmp_path):
    p = _write_scan(tmp_path / "s.h5")
    rows = detect_pixel_sizes(p, "instrument/positioners", ".1")
    by = _by_field(rows)
    m = 5000.0 / 273.0 - 1.0
    assert by["pixel_size_x_um"].value == round(3.25 / m, 6)
    assert by["pixel_size_y_um"].value > by["pixel_size_x_um"].value  # sin(2θ) division
    assert "2x" in by["pixel_size_x_um"].note and "M=" in by["pixel_size_x_um"].note


def test_detect_pixel_sizes_user_error_becomes_rows(tmp_path):
    p = _write_scan(tmp_path / "s.h5", ffsel=-30.0)  # unrecognized objective
    rows = detect_pixel_sizes(p, "instrument/positioners", ".1")
    assert len(rows) == 2
    assert all(d.value is None and d.error for d in rows)
    assert "ffsel" in rows[0].error


def test_detect_pixel_sizes_unreadable_file(tmp_path):
    p = tmp_path / "junk.h5"
    p.write_text("nope")
    rows = detect_pixel_sizes(str(p), "instrument/positioners", ".1")
    assert all(d.error for d in rows)


def test_find_strain_maps_skips_mosa_style(tmp_path):
    proc = tmp_path / "proc"
    for name, ccm in (("s__0", False), ("s__1", True)):
        d = proc / name
        d.mkdir(parents=True)
        _write_maps(d / "maps.h5", with_ccmth=ccm)
    found = find_strain_maps(str(proc), "s__*", "maps.h5", CCMTH_COM)
    assert found is not None
    maps_path, folder = found
    assert folder == "s__1" and maps_path.endswith("s__1/maps.h5")


def test_find_strain_maps_none_when_absent(tmp_path):
    assert find_strain_maps(str(tmp_path), "s__*", "maps.h5", CCMTH_COM) is None
    assert find_strain_maps("", "s__*", "maps.h5", CCMTH_COM) is None


def test_detect_ccmth_from_maps_nanmedian(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", fill=7.1442)
    d = detect_ccmth_from_maps(p, "s__0", CCMTH_COM)
    assert d.field == "ccmth_ref_deg" and d.value == 7.1442
    assert "median" in d.note and "s__0" in d.note


def test_detect_ccmth_from_maps_malformed_dataset(tmp_path):
    p = tmp_path / "maps.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset(CCMTH_COM, data="oops")  # non-numeric -> nanmedian TypeErrors
    d = detect_ccmth_from_maps(str(p), "s__0", CCMTH_COM)
    assert d.value is None and d.error


def test_detect_ccmth_from_positioners(tmp_path):
    p = _write_scan(tmp_path / "s.h5", ccmth=7.144236)
    d = detect_ccmth_from_positioners(p, "instrument/positioners", ".1")
    assert d.value == 7.1442
    assert "confirm" in d.note  # flags itself as a snapshot needing confirmation


def test_detect_ccmth_from_positioners_missing_motor(tmp_path):
    p = _write_scan(tmp_path / "s.h5", ccmth=None)
    d = detect_ccmth_from_positioners(p, "instrument/positioners", ".1")
    assert d.value is None and "ccmth" in d.error


def test_detect_darfix_roi_blank_current_gives_partial(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", shape=(1266, 1832))
    d = detect_darfix_roi(p, "s__0", CCMTH_COM, "")
    assert d.value == "?,?,1832,1266"
    assert "origin" in d.note


def test_detect_darfix_roi_consistent_is_info_row(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", shape=(1266, 1832))
    d = detect_darfix_roi(p, "s__0", CCMTH_COM, "105,230,1832,1266")
    assert d.value is None and d.error is None
    assert "matches" in d.note


def test_detect_darfix_roi_mismatch_keeps_origin(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", shape=(1266, 1832))
    d = detect_darfix_roi(p, "s__0", CCMTH_COM, "105,230,999,999")
    assert d.value == "105,230,1832,1266"
    assert "not 999×999" in d.note


def test_detect_darfix_roi_malformed_current_treated_as_blank(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", shape=(6, 8))
    d = detect_darfix_roi(p, "s__0", CCMTH_COM, "banana")
    assert d.value == "?,?,8,6"


def test_detect_darfix_roi_malformed_dataset(tmp_path):
    p = tmp_path / "maps.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset(CCMTH_COM, data=np.arange(3.0))  # 1-D -> shape[:2] unpack fails
    d = detect_darfix_roi(str(p), "s__0", CCMTH_COM, "")
    assert d.value is None and d.error


# -- orchestrator + CLI -------------------------------------------------------

from dfxm.config.detect import detect_experiment, main  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402


def _make_tree(tmp_path, *, with_maps=True):
    """Full synthetic experiment: raw families + scan, optional processed maps."""
    raw = tmp_path / "RAW"
    _mkdirs(raw, "s_strain__0", "s_strain__1", "s_mosa__0", "s_rocking__0")
    _write_scan(raw / "s_strain__0" / "s_strain__0.h5", ccmth=7.10)
    proc = tmp_path / "PROC"
    if with_maps:
        d = proc / "s_strain__0"
        d.mkdir(parents=True)
        _write_maps(d / "maps.h5", shape=(1266, 1832), fill=7.1442)
    else:
        proc.mkdir()
    return str(raw), str(proc)


def test_detect_experiment_full_pass(tmp_path):
    raw, proc = _make_tree(tmp_path)
    rows = _by_field(detect_experiment(Experiment(raw_root=raw, processed_root=proc)))
    assert rows["folder_pattern"].value == "s_strain__*"
    assert rows["entry_suffix"].value == ".1"
    assert rows["pixel_size_x_um"].value and rows["pixel_size_x_um"].error is None
    assert rows["ccmth_ref_deg"].value == 7.1442  # maps median wins over positioner 7.10
    assert "median" in rows["ccmth_ref_deg"].note
    assert rows["darfix_roi"].value == "?,?,1832,1266"


def test_detect_experiment_pre_darfix_falls_back(tmp_path):
    raw, proc = _make_tree(tmp_path, with_maps=False)
    rows = _by_field(detect_experiment(Experiment(raw_root=raw, processed_root=proc)))
    assert rows["ccmth_ref_deg"].value == 7.1  # positioner snapshot fallback
    assert "confirm" in rows["ccmth_ref_deg"].note
    assert rows["darfix_roi"].error and "re-run" in rows["darfix_roi"].error


def test_detect_experiment_explicit_pattern_wins(tmp_path):
    raw, proc = _make_tree(tmp_path)
    # user set a pattern that matches nothing -> scan-dependent rows skip
    exp = Experiment(raw_root=raw, processed_root=proc, folder_pattern="zzz__*")
    rows = _by_field(detect_experiment(exp))
    assert rows["folder_pattern"].value == "s_strain__*"  # suggestion still shown
    assert rows["pixel_size_x_um"].error  # but zzz__* found no scan


def test_detect_experiment_no_families_reasons(tmp_path):
    raw = tmp_path / "RAW"
    _mkdirs(raw, "loose")
    rows = _by_field(detect_experiment(Experiment(raw_root=str(raw))))
    for field in ("entry_suffix", "pixel_size_x_um", "pixel_size_y_um", "darfix_roi"):
        assert "folder pattern" in rows[field].error.lower()
    assert not any("re-run after darfix" in (d.error or "") for d in rows.values())


def test_detect_experiment_no_raw_root():
    rows = detect_experiment(Experiment(raw_root=""))
    assert len(rows) == 1 and rows[0].field == "raw_root" and rows[0].error


def test_detect_experiment_survives_bad_scan_file(tmp_path):
    raw = tmp_path / "RAW"
    _mkdirs(raw, "s_strain__0")
    (raw / "s_strain__0" / "s_strain__0.h5").write_text("not hdf5")
    rows = _by_field(detect_experiment(Experiment(raw_root=str(raw))))
    assert rows["folder_pattern"].value == "s_strain__*"  # patterns still detected
    assert rows["entry_suffix"].error and rows["pixel_size_x_um"].error


def test_cli_main_prints_table(tmp_path, capsys):
    raw, proc = _make_tree(tmp_path)
    assert main([raw, "--processed-root", proc]) == 0
    out = capsys.readouterr().out
    assert "folder_pattern" in out and "s_strain__*" in out
    assert "ccmth_ref_deg" in out and "7.1442" in out
    assert "SKIP" not in out.split("darfix_roi")[0]  # detected rows are not skips
