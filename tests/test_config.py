"""Tests for dfxm.config (models + presets)."""

from __future__ import annotations

from dataclasses import fields

import pytest

from dfxm.config import presets
from dfxm.config.models import (
    EXPERIMENT_SCHEMA,
    Experiment,
    Param,
    ParamType,
    StageSpec,
)


def test_experiment_schema_matches_dataclass_fields():
    """The GUI schema must stay in lock-step with the Experiment dataclass."""
    dataclass_names = [f.name for f in fields(Experiment)]
    schema_names = [p.name for p in EXPERIMENT_SCHEMA]
    assert dataclass_names == schema_names


def test_calibration_fields_flagged():
    flagged = {p.name for p in EXPERIMENT_SCHEMA if p.calibration}
    assert flagged == {"ccmth_ref_deg", "pixel_size_x_um", "pixel_size_y_um"}


def test_sto2_preset_ships_expected_values():
    exp = presets.load_experiment_by_name("STO2_overnight")
    assert exp.name == "STO2_overnight"
    assert exp.ccmth_ref_deg == 7.144
    assert exp.pixel_size_x_um == 0.152 and exp.pixel_size_y_um == 0.385


def test_preset_round_trip(tmp_path):
    exp = presets.load_experiment_by_name("STO2_overnight")
    path = tmp_path / "rt.yaml"
    presets.save_experiment(exp, path)
    again = presets.load_experiment(path)
    assert again.to_dict() == exp.to_dict()


def test_from_dict_ignores_unknown_keys_with_warning():
    with pytest.warns(UserWarning):
        exp = Experiment.from_dict({"name": "x", "bogus": 1})
    assert exp.name == "x"


def test_param_coercion():
    assert Param("a", ParamType.INT, "A").coerce("5") == 5
    assert Param("a", ParamType.FLOAT, "A").coerce("1.5") == 1.5
    pb = Param("b", ParamType.BOOL, "B")
    assert pb.coerce("true") is True and pb.coerce("0") is False and pb.coerce("yes") is True


def test_enum_param_validates_choices():
    p = Param("m", ParamType.ENUM, "M", default="a", choices=("a", "b"))
    assert p.coerce("b") == "b"
    with pytest.raises(ValueError):
        p.coerce("c")


def test_enum_param_requires_choices():
    with pytest.raises(ValueError):
        Param("m", ParamType.ENUM, "M")


def test_stagespec_helpers():
    spec = StageSpec(
        "s",
        "S",
        "desc",
        (
            Param("n", ParamType.INT, "N", default=1),
            Param("flag", ParamType.BOOL, "F", default=False),
        ),
    )
    assert spec.defaults() == {"n": 1, "flag": False}
    assert spec.get("n").label == "N"
    assert spec.coerce_all({"flag": "true"}) == {"n": 1, "flag": True}
    with pytest.raises(KeyError):
        spec.get("missing")
