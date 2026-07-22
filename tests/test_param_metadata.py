"""Schema metadata enforcement: every param must carry first-timer help.

Field-existence tests run from Task 1 of the GUI-overhaul plan; the
per-stage enforcement tests below them go green stage-by-stage as the
specs gain metadata (Tasks 4-12).
"""

import importlib

import pytest

from dfxm.config.models import EXPERIMENT_SCHEMA, Param, ParamType
from dfxm.stages.registry import STAGE_TARGETS


def test_param_metadata_fields_default_off():
    p = Param("x", ParamType.STR, "X")
    assert p.advanced is False
    assert p.group == ""
    assert p.must_exist is False


def test_param_metadata_fields_settable():
    p = Param(
        "x",
        ParamType.DIR,
        "X",
        advanced=True,
        group="Data layout",
        must_exist=True,
    )
    assert p.advanced is True
    assert p.group == "Data layout"
    assert p.must_exist is True


_STAGES = sorted(STAGE_TARGETS)


def _spec(stage_name: str):
    module_name = STAGE_TARGETS[stage_name].split(":")[0]
    return importlib.import_module(module_name).STAGE


@pytest.mark.parametrize("stage_name", _STAGES)
def test_every_param_has_help(stage_name):
    missing = [p.name for p in _spec(stage_name).params if not (p.help or "").strip()]
    assert not missing, f"{stage_name}: params without help text: {missing}"


@pytest.mark.parametrize("stage_name", _STAGES)
def test_advanced_params_have_group(stage_name):
    bad = [p.name for p in _spec(stage_name).params if p.advanced and not p.group.strip()]
    assert not bad, f"{stage_name}: advanced params without a group: {bad}"


@pytest.mark.parametrize("stage_name", _STAGES)
def test_essential_param_count(stage_name):
    essentials = [p.name for p in _spec(stage_name).params if not p.advanced]
    assert 1 <= len(essentials) <= 8, (
        f"{stage_name}: {len(essentials)} essentials (want 1-8): {essentials}"
    )


@pytest.mark.parametrize("stage_name", _STAGES)
def test_must_exist_only_on_input_paths(stage_name):
    bad = [
        p.name
        for p in _spec(stage_name).params
        if p.must_exist and p.type not in (ParamType.PATH, ParamType.DIR)
    ]
    assert not bad, f"{stage_name}: must_exist on non-path params: {bad}"


def test_experiment_schema_has_help():
    missing = [p.name for p in EXPERIMENT_SCHEMA if not (p.help or "").strip()]
    assert not missing, f"experiment schema params without help: {missing}"


def test_roi_fields_default_empty():
    from dfxm.config.models import Param, ParamType

    p = Param("x", ParamType.STR, "X")
    assert p.roi_group == ""
    assert p.roi_axis == ""
    assert p.roi_frame == ""


def test_roi_axis_requires_group_and_valid_value():
    from dfxm.config.models import Param, ParamType

    Param("roi_x", ParamType.STR, "ROI x", roi_group="align", roi_axis="x")  # ok
    with pytest.raises(ValueError):
        Param("roi_x", ParamType.STR, "ROI x", roi_axis="x")  # axis without group
    with pytest.raises(ValueError):
        Param("roi_x", ParamType.STR, "ROI x", roi_group="align", roi_axis="diagonal")  # bad value


def test_roi_frame_validated():
    Param("roi_x", ParamType.STR, "ROI x", roi_frame="detector")  # ok, no group needed
    with pytest.raises(ValueError):
        Param("roi_x", ParamType.STR, "ROI x", roi_frame="galactic")


def test_roi_params_declare_frame():
    """Every ROI param states its coordinate frame — and says so in its help."""
    from gui.bindings import STAGE_SPECS

    for stage_name, spec in STAGE_SPECS.items():
        for p in spec.params:
            if not (p.roi_group or p.roi_frame):
                continue
            assert p.roi_frame in ("detector", "map"), f"{stage_name}.{p.name}: no roi_frame"
            assert p.roi_frame in (p.help or "").lower(), (
                f"{stage_name}.{p.name}: help must state its '{p.roi_frame}' frame"
            )


def test_rocking_roi_params_are_detector_frame():
    from dfxm.stages import rocking

    assert rocking.STAGE.get("roi_x").roi_frame == "detector"
    assert rocking.STAGE.get("roi_y").roi_frame == "detector"
