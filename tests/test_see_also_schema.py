"""The SeeAlso pointer carrier on a stage schema (Qt-free)."""

import pytest

from dfxm.config.models import Param, ParamType, SeeAlso, StageSpec
from gui.bindings import STAGE_ORDER, STAGE_SPECS


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
    # The registry-wide sweep below reports every stage at once, so the
    # message has to say WHICH stage carries the bad anchor.
    assert "demo" in problems[0]


def test_every_bad_pointer_is_reported_not_just_the_first():
    spec = _spec(SeeAlso("param:nope", "a"), SeeAlso("param:alsonope", "b"))
    assert len(spec.see_also_problems()) == 2


def test_an_unknown_anchor_prefix_is_rejected_at_construction():
    with pytest.raises(ValueError, match="anchor"):
        SeeAlso("group:Appearance", "text")


def test_an_anchor_naming_no_parameter_is_rejected_at_construction():
    with pytest.raises(ValueError, match="anchor"):
        SeeAlso("param:", "text")


def test_an_empty_text_is_rejected_at_construction():
    with pytest.raises(ValueError, match="text"):
        SeeAlso("", "   ")


_FIGURE_STAGES = ("strain", "mosaicity", "rocking", "visualize", "slices", "profiles", "matched")


def test_every_real_stage_spec_has_valid_see_also_anchors():
    # Precondition: this walk is worthless if no stage declares a pointer.
    assert sum(len(STAGE_SPECS[n].see_also) for n in STAGE_ORDER) > 0
    for name in STAGE_ORDER:
        assert STAGE_SPECS[name].see_also_problems() == []


def test_every_figure_producing_stage_points_at_the_style_dialog():
    for name in _FIGURE_STAGES:
        texts = [s.text for s in STAGE_SPECS[name].see_also if not s.param_name]
        assert texts, f"{name} has no stage-level pointer"
        assert any("Publication style" in t for t in texts), name


def test_stages_that_produce_no_figures_have_no_pointer():
    # concat writes .h5 only; paraview writes VTI whose colormap is chosen in
    # ParaView itself. A pointer there would be a lie.
    for name in ("concat", "paraview"):
        assert STAGE_SPECS[name].see_also == ()


def test_matched_additionally_annotates_its_colormap_dropdown():
    entries = {s.param_name: s.text for s in STAGE_SPECS["matched"].see_also if s.param_name}
    assert "colormap" in entries
    assert "Publication style" in entries["colormap"]
