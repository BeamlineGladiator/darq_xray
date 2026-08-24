"""The SeeAlso pointer carrier on a stage schema (Qt-free)."""

import pytest

from dfxm.config.models import Param, ParamType, SeeAlso, StageSpec
from gui.bindings import STAGE_SPECS


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
    # message has to open by saying WHICH stage carries the bad anchor.
    assert problems[0].startswith("stage 'demo':")


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
    assert sum(len(spec.see_also) for spec in STAGE_SPECS.values()) > 0
    for name, spec in STAGE_SPECS.items():
        assert spec.see_also_problems() == [], name


def test_no_spec_declares_two_stage_level_pointers():
    # ParamForm keys stage-level rows under "" in a single dict, so a second
    # anchor="" entry would overwrite the first in `_see_also_labels` — both
    # rows render, but only the last is findable. One per spec, by design.
    for name, spec in STAGE_SPECS.items():
        assert len([s for s in spec.see_also if not s.param_name]) <= 1, name


def test_every_figure_producing_stage_points_at_the_style_dialog():
    for name in _FIGURE_STAGES:
        texts = [s.text for s in STAGE_SPECS[name].see_also if not s.param_name]
        assert texts, f"{name} has no stage-level pointer"
        assert any("Publication style" in t for t in texts), name


_RANGE_TOKENS = ("vmin", "vmax", "pct")
_EXTENDED = "this stage's own"


def _owns_appearance_range_field(spec) -> bool:
    """True when the stage has a colour-RANGE field in the same Advanced group
    the colormap discussion belongs to.

    This is the invariant behind the two wordings: `visualize`/`slices` also
    have a `range_pct`, but theirs sits in "Alignment", where it does not read
    as competing with the pointer.
    """
    return any(
        p.group == "Appearance" and any(t in p.name for t in _RANGE_TOKENS) for p in spec.params
    )


def test_the_extended_wording_is_used_exactly_where_the_stage_owns_a_range_field():
    # Precondition: the split is only meaningful if the registry has both kinds.
    assert {_owns_appearance_range_field(STAGE_SPECS[n]) for n in _FIGURE_STAGES} == {True, False}
    for name in _FIGURE_STAGES:
        spec = STAGE_SPECS[name]
        text = " ".join(s.text for s in spec.see_also if not s.param_name)
        assert (_EXTENDED in text) == _owns_appearance_range_field(spec), name
        if _EXTENDED in text:
            # Those range fields are all advanced=True while the stage-level
            # row renders ABOVE the collapsed expander, so a bare "below"
            # points at nothing the newcomer can see.
            assert "Advanced" in text, name


def test_stages_that_produce_no_figures_have_no_pointer():
    # concat writes .h5 only; paraview writes VTI whose colormap is chosen in
    # ParaView itself. A pointer there would be a lie.
    for name in ("concat", "paraview"):
        assert STAGE_SPECS[name].see_also == ()


def test_matched_additionally_annotates_its_colormap_dropdown():
    entries = {s.param_name: s.text for s in STAGE_SPECS["matched"].see_also if s.param_name}
    assert "colormap" in entries
    assert "Publication style" in entries["colormap"]
