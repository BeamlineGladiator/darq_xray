"""Unit tests for the per-experiment form-state store (QSettings-backed)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QSettings  # noqa: E402

from darq_xray.gui.form_state import FormStateStore, _slug  # noqa: E402


def _store(tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    return FormStateStore(settings)


def test_round_trip_per_experiment_and_stage(tmp_path):
    st = _store(tmp_path)
    st.save(
        "STO2_overnight",
        "strain",
        {"root_folder": "/data/a", "range_pct": 99.5, "flag": True, "n": 3},
    )
    st.save("STO2_overnight", "slices", {"output_h5_name": "x.h5"})
    st.save("other_exp", "strain", {"root_folder": "/data/b"})
    # values round-trip with their types intact (float/bool/int survive JSON)
    assert st.load("STO2_overnight", "strain") == {
        "root_folder": "/data/a",
        "range_pct": 99.5,
        "flag": True,
        "n": 3,
    }
    # keyed independently per (experiment, stage)
    assert st.load("STO2_overnight", "slices") == {"output_h5_name": "x.h5"}
    assert st.load("other_exp", "strain") == {"root_folder": "/data/b"}
    assert st.load("STO2_overnight", "mosaicity") is None  # unseen stage


def test_load_missing_returns_none(tmp_path):
    assert _store(tmp_path).load("nope", "strain") is None


def test_load_corrupt_returns_none(tmp_path):
    settings = QSettings(str(tmp_path / "s.ini"), QSettings.Format.IniFormat)
    settings.setValue("formState/exp/strain", "{not valid json")
    assert FormStateStore(settings).load("exp", "strain") is None


def test_slug_sanitizes_and_defaults():
    assert _slug("STO2/overnight") == "STO2_overnight"  # no stray QSettings groups
    assert _slug("") == "default"
    assert _slug("  a b  ") == "a_b"
