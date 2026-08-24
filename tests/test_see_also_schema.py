"""The SeeAlso pointer carrier on a stage schema (Qt-free)."""

import pytest

from dfxm.config.models import Param, ParamType, SeeAlso, StageSpec


def _spec(*see_also):
    return StageSpec(
        name="demo",
        label="Demo",
        description="A demo stage.",
        params=(
            Param("colormap", ParamType.STR, "Colormap"),
            Param("vmin", ParamType.FLOAT, "vmin"),
        ),
        see_also=see_also,
    )


def test_a_spec_has_no_pointers_by_default():
    spec = StageSpec(name="d", label="D", description="d", params=())
    assert spec.see_also == ()
    assert spec.see_also_problems() == []


def test_a_stage_level_pointer_is_valid():
    spec = _spec(SeeAlso("", "Colormaps live in Publication style…"))
    assert spec.see_also_problems() == []


def test_a_param_pointer_naming_a_real_param_is_valid():
    spec = _spec(SeeAlso("param:colormap", "Publication style wins here."))
    assert spec.see_also_problems() == []


def test_a_param_pointer_naming_a_missing_param_is_reported():
    spec = _spec(SeeAlso("param:nope", "text"))
    problems = spec.see_also_problems()
    assert len(problems) == 1
    assert "nope" in problems[0]


def test_every_bad_pointer_is_reported_not_just_the_first():
    spec = _spec(SeeAlso("param:nope", "a"), SeeAlso("param:alsonope", "b"))
    assert len(spec.see_also_problems()) == 2


def test_an_unknown_anchor_prefix_is_rejected_at_construction():
    with pytest.raises(ValueError, match="anchor"):
        SeeAlso("group:Appearance", "text")


def test_an_empty_text_is_rejected_at_construction():
    with pytest.raises(ValueError, match="text"):
        SeeAlso("", "   ")
