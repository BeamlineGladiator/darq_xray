"""The SeeAlso pointer carrier on a stage schema (Qt-free)."""

import pytest

from darq_xray.config.models import Param, ParamType, SeeAlso, StageSpec
from darq_xray.gui.bindings import STAGE_SPECS


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


def test_two_pointers_on_the_same_param_are_reported():
    # ParamForm keys its pointer rows by param name in one dict, so a second
    # "param:colormap" entry renders only the LAST one's text and the first
    # silently disappears — the quietest possible failure. Nobody means it, so
    # the sweep must catch it.
    spec = _spec(SeeAlso("param:colormap", "a"), SeeAlso("param:colormap", "b"))
    problems = spec.see_also_problems()
    assert len(problems) == 1
    assert "twice" in problems[0]
    assert "param:colormap" in problems[0]


def test_two_stage_level_pointers_are_reported():
    # The other half: two anchor="" entries render both rows but collide in
    # `_see_also_labels[""]`, so only the last is findable. The two anchor kinds
    # fail differently and are both reported here, in one place.
    spec = _spec(SeeAlso("", "a"), SeeAlso("", "b"))
    problems = spec.see_also_problems()
    assert len(problems) == 1
    assert "twice" in problems[0]


def test_two_pointers_on_different_params_are_fine():
    # precondition for the two above: it is the DUPLICATION that is reported,
    # not merely having more than one pointer.
    spec = _spec(SeeAlso("param:colormap", "a"), SeeAlso("param:vmin", "b"))
    assert spec.see_also_problems() == []


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
    # `see_also_problems` reports unknown anchors AND duplicates of either kind,
    # so this one sweep is also what pins "no spec declares two stage-level
    # pointers" (which used to be a separate walk of its own).
    for name, spec in STAGE_SPECS.items():
        assert spec.see_also_problems() == [], name


def test_every_figure_producing_stage_points_at_the_style_dialog():
    for name in _FIGURE_STAGES:
        texts = [s.text for s in STAGE_SPECS[name].see_also if not s.param_name]
        assert texts, f"{name} has no stage-level pointer"
        assert any("Publication style" in t for t in texts), name


# Any of these in a param NAME means the field decides something about colour:
# a range ("vmin"/"vmax"/"pct"/"clim") or a colour outright ("color", which also
# covers "colormap"). Matching meaning rather than just range names is what the
# earlier `("vmin", "vmax", "pct")` list missed: `profiles` owns `line_color`
# and `trace_color` in Appearance and yet carried the short "…, not here."
# wording, so expanding Advanced put "not here" directly above **Line colour**.
_COLOUR_TOKENS = ("color", "vmin", "vmax", "pct", "clim")


def _owns_appearance_colour_field(spec) -> bool:
    """True when the stage has a colour-deciding field in the same Advanced
    group the colormap discussion belongs to.

    Group membership is half the invariant: `visualize`/`slices` also have a
    `range_pct`, but theirs sits in "Alignment", where it does not read as
    competing with the pointer.
    """
    return any(
        p.group == "Appearance" and any(t in p.name for t in _COLOUR_TOKENS) for p in spec.params
    )


def test_a_stage_owning_colour_fields_points_at_them_instead_of_saying_not_here():
    # Precondition: the split is only meaningful if the registry has both kinds.
    assert {_owns_appearance_colour_field(STAGE_SPECS[n]) for n in _FIGURE_STAGES} == {True, False}
    # ...and `profiles` must land on the OWNING side. It is the stage a
    # range-names-only predicate skips, so without this the whole check goes
    # quietly vacuous for the one stage that was actually broken.
    assert _owns_appearance_colour_field(STAGE_SPECS["profiles"]), "predicate dropped profiles"
    for name in _FIGURE_STAGES:
        spec = STAGE_SPECS[name]
        text = " ".join(s.text for s in spec.see_also if not s.param_name)
        if not _owns_appearance_colour_field(spec):
            continue
        # Those fields are all advanced=True while the stage-level row renders
        # ABOVE the collapsed expander, so a bare "below" points at nothing the
        # newcomer can see: the sentence has to name *Advanced*.
        assert "Advanced" in text, name
        # ...and it must not ALSO claim the stage owns no colour control, which
        # is what the user reads right above those very fields.
        assert "not here" not in text, name


def test_stages_that_produce_no_figures_have_no_pointer():
    # concat writes .h5 only; paraview writes VTI whose colormap is chosen in
    # ParaView itself. A pointer there would be a lie.
    for name in ("concat", "paraview"):
        assert STAGE_SPECS[name].see_also == ()


def test_matched_additionally_annotates_its_colormap_dropdown():
    entries = {s.param_name: s.text for s in STAGE_SPECS["matched"].see_also if s.param_name}
    assert "colormap" in entries
    assert "Publication style" in entries["colormap"]


def test_a_param_pointer_and_that_params_help_do_not_state_the_same_rule():
    """help = what the field is; pointer = where the real control lives.

    `help_panel.param_help_html` concatenates the two into ONE label (which is
    also the editor's tooltip), so a rule written into both is a rule the user
    reads twice in one box. matched's `colormap` is where the two were allowed
    to collapse into each other: both said "headless CLI → this dropdown, in-app
    → the style's raw group", in that order, in the same box.
    """
    spec = STAGE_SPECS["matched"]
    help_text = spec.get("colormap").help
    pointer = {s.param_name: s.text for s in spec.see_also if s.param_name}["colormap"]
    # precondition: the pointer really is the one that names the dialog
    assert "Publication style" in pointer
    # case-folded on purpose: the text this replaced said "publication style"
    # in lower case, and a case-sensitive check waved it straight through.
    assert "publication style" not in help_text.lower(), help_text
    # ...and the help is the one that says what the dropdown itself is for
    assert "headless" in help_text.lower()
    assert "headless" not in pointer.lower(), pointer


def test_no_stage_declares_one_advice_key_on_two_fields():
    """`ParamForm.apply_hints` writes a hint under EVERY param declaring its
    key, so two fields sharing one key render the same paragraph twice."""
    for name, spec in STAGE_SPECS.items():
        keys = [p.advice_key for p in spec.params if p.advice_key]
        assert len(keys) == len(set(keys)), f"{name} repeats an advice_key: {keys}"
