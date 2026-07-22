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
