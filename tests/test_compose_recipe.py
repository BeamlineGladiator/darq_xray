"""Recipe schema + JSON round-trip — dfxm.compose.recipe."""

import pytest

from dfxm.common.errors import StageUserError
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    Row,
    Spacer,
    iter_leaves,
    recipe_from_json,
    recipe_to_json,
    validate_recipe,
)


def _mini_recipe():
    p1 = PanelDef("m0", __src("/data/stack.h5", "map_layer", {"stage": "mosaicity"}))
    p2 = PanelDef("t0", __src("/data/obl.h5", "profiles_trace", {"field": "strain"}))
    layout = Row([PanelRef("m0"), Col([PanelRef("t0"), Spacer(1.0, 1.0)], shared_x=True)])
    return FigureRecipe(
        name="demo",
        style={"scale_um_per_cm": 10.0},
        compose=ComposeStyle(label_template="(A)"),
        layout=layout,
        panels=[p1, p2],
    )


def __src(path, kind, sel):
    from dfxm.compose.recipe import PanelSource

    return PanelSource(path, kind, sel)


def test_round_trip_preserves_everything():
    r = _mini_recipe()
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.name == "demo" and r2.version == r.version
    assert r2.style == {"scale_um_per_cm": 10.0}
    assert r2.compose.label_template == "(A)"
    assert [p.id for p in r2.panels] == ["m0", "t0"]
    assert isinstance(r2.layout, Row)
    col = r2.layout.children[1]
    assert isinstance(col, Col) and col.shared_x is True
    assert isinstance(col.children[1], Spacer)
    leaves = list(iter_leaves(r2.layout))
    assert [type(x).__name__ for x in leaves] == ["PanelRef", "PanelRef", "Spacer"]


def test_relative_h5_paths_round_trip(tmp_path):
    r = _mini_recipe()
    r.panels[0].source.h5_path = str(tmp_path / "sub" / "stack.h5")
    text = recipe_to_json(r, base_dir=str(tmp_path))
    assert str(tmp_path) not in text  # stored relative
    r2 = recipe_from_json(text, base_dir=str(tmp_path))
    assert r2.panels[0].source.h5_path == str(tmp_path / "sub" / "stack.h5")


def test_unknown_version_raises_stageusererror():
    text = recipe_to_json(_mini_recipe()).replace('"version": 1', '"version": 99')
    with pytest.raises(StageUserError) as e:
        recipe_from_json(text)
    assert "version" in str(e.value) and e.value.hint


@pytest.mark.parametrize(
    "mutate,frag",
    [
        (lambda r: r.panels.append(PanelDef("m0", __src("x", "map_layer", {}))), "duplicate"),
        (lambda r: r.layout.children.append(PanelRef("ghost")), "ghost"),
        (lambda r: setattr(r.panels[0].source, "kind", "hologram"), "hologram"),
        (lambda r: setattr(r.compose, "scale_bar_mode", "everywhere"), "scale_bar_mode"),
        (lambda r: setattr(r.compose, "label_template", "xx"), "label_template"),
        (lambda r: setattr(r.compose, "colorbar_mode", "rainbow"), "colorbar_mode"),
        (lambda r: setattr(r.compose, "colorbar_pos", "left"), "colorbar_pos"),
    ],
)
def test_validate_refuses_bad_recipes(mutate, frag):
    r = _mini_recipe()
    mutate(r)
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert frag in str(e.value)


def test_bad_json_raises_stageusererror():
    with pytest.raises(StageUserError):
        recipe_from_json("{not json")
    with pytest.raises(StageUserError):
        recipe_from_json('{"no": "layout"}')


def _as_dict(r=None):
    import json

    return json.loads(recipe_to_json(r or _mini_recipe()))


def test_malformed_recipe_unknown_compose_key_wrapped_not_raw_typeerror():
    import json

    d = _as_dict()
    d["compose"]["no_such_knob"] = 3
    with pytest.raises(StageUserError) as e:
        recipe_from_json(json.dumps(d))
    assert "malformed" in str(e.value) and e.value.hint


def test_malformed_recipe_missing_panel_id_wrapped_not_raw_keyerror():
    import json

    d = _as_dict()
    del d["panels"][0]["id"]
    with pytest.raises(StageUserError) as e:
        recipe_from_json(json.dumps(d))
    assert "malformed" in str(e.value) and e.value.hint


def test_malformed_recipe_missing_panel_source_wrapped():
    import json

    d = _as_dict()
    del d["panels"][0]["source"]
    with pytest.raises(StageUserError) as e:
        recipe_from_json(json.dumps(d))
    assert "malformed" in str(e.value)


def test_not_a_recipe_json_gets_dedicated_message():
    with pytest.raises(StageUserError) as e:
        recipe_from_json('{"no": "layout"}')
    assert "not a figure recipe" in str(e.value)
    assert "version" not in str(e.value)  # not the old "unsupported recipe version None"


def test_duplicate_panel_ref_refused():
    r = _mini_recipe()
    r.layout.children.append(PanelRef("m0"))  # m0 already referenced once
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert "more than once" in str(e.value) and e.value.hint


def test_blank_group_label_normalized_to_none_on_load():
    import json

    d = _as_dict()
    d["layout"]["children"][1]["group_label"] = ""  # the nested Col
    r = recipe_from_json(json.dumps(d))
    assert r.layout.children[1].group_label is None


def test_nested_stage_user_error_surfaces_unwrapped():
    """A StageUserError raised inside node parsing must pass through the
    malformed-v1 wrapper untouched (never double-wrapped as "malformed")."""
    import json

    d = _as_dict()
    d["layout"]["children"][0] = {"type": "hologram"}
    with pytest.raises(StageUserError) as e:
        recipe_from_json(json.dumps(d))
    assert "unknown layout node type" in str(e.value)
    assert "malformed" not in str(e.value)


def test_duplicate_panel_ref_nested_refused():
    """Duplicate detection counts across nested containers, not just siblings."""
    r = _mini_recipe()
    r.layout.children.append(Col([PanelRef("m0")]))  # m0 already referenced at top level
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert "more than once" in str(e.value)


def test_panel_title_round_trips_and_old_recipes_load_none():
    import json

    r = _mini_recipe()
    r.panels[0].title = "strain: layer / z=3"
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.panels[0].title == "strain: layer / z=3"
    assert r2.panels[1].title is None
    # an old (pre-title) recipe JSON still loads, title=None everywhere
    d = json.loads(recipe_to_json(_mini_recipe()))
    for pd in d["panels"]:
        pd.pop("title", None)
    r3 = recipe_from_json(json.dumps(d))
    assert all(p.title is None for p in r3.panels)
    assert r3.version == 1


def test_colorbar_mode_fields_round_trip_and_old_recipe_defaults():
    import json

    r = _mini_recipe()
    r.compose.colorbar_mode = "united"
    r.compose.colorbar_pos = "bottom"
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.compose.colorbar_mode == "united" and r2.compose.colorbar_pos == "bottom"
    d = json.loads(recipe_to_json(_mini_recipe()))
    d["compose"].pop("colorbar_mode")
    d["compose"].pop("colorbar_pos")
    r3 = recipe_from_json(json.dumps(d))
    assert r3.compose.colorbar_mode == "per-panel" and r3.compose.colorbar_pos == "right"
