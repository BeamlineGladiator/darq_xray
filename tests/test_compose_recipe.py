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
    sanitize_stem,
    validate_recipe,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("figure 3", "figure_3"),  # spaces are not filename material
        ("strain/mosaicity", "strain_mosaicity"),  # a separator can never survive
        ("../../etc/passwd", "etc_passwd"),  # ...so a typed name cannot escape out_dir
        (".", "figure"),  # nor become a path component
        ("..", "figure"),
        (".hidden", "hidden"),  # a typed name must not make a hidden file
        ("fig.v2", "fig.v2"),  # a dotted name is kept as-is
        ("", "figure"),
        (None, "figure"),
        ("   ", "figure"),
    ],
)
def test_sanitize_stem(name, expected):
    assert sanitize_stem(name) == expected
    assert sanitize_stem(sanitize_stem(name)) == expected  # idempotent


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


def test_trace_autoscale_round_trips_and_old_recipe_defaults():
    import json

    r = _mini_recipe()
    r.compose.trace_autoscale = True
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.compose.trace_autoscale is True
    # an old (pre-trace_autoscale) recipe JSON still loads, defaulting False
    d = json.loads(recipe_to_json(_mini_recipe()))
    d["compose"].pop("trace_autoscale")
    r3 = recipe_from_json(json.dumps(d))
    assert r3.compose.trace_autoscale is False
    assert r3.version == 1


def test_trace_look_knobs_round_trip_and_old_recipe_defaults():
    import json

    r = _mini_recipe()
    r.compose.trace_linewidth = 4.0
    r.compose.trace_color = "k"
    r.compose.trace_font_scale = 2.0
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.compose.trace_linewidth == 4.0
    assert r2.compose.trace_color == "k"
    assert r2.compose.trace_font_scale == 2.0
    # an old recipe JSON (no trace look keys) loads with the "follow style" defaults
    d = json.loads(recipe_to_json(_mini_recipe()))
    for k in ("trace_linewidth", "trace_color", "trace_font_scale"):
        d["compose"].pop(k)
    r3 = recipe_from_json(json.dumps(d))
    assert r3.compose.trace_linewidth is None
    assert r3.compose.trace_color == ""
    assert r3.compose.trace_font_scale is None


def test_trace_look_knobs_validated():
    r = _mini_recipe()
    r.compose.trace_linewidth = 0.0
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert "trace_linewidth" in str(e.value) and e.value.hint
    r = _mini_recipe()
    r.compose.trace_font_scale = -1.0
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert "trace_font_scale" in str(e.value) and e.value.hint


def test_panel_crop_to_data_round_trips_and_defaults_false():
    import json

    r = _mini_recipe()
    r.panels[0].crop_to_data = True
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.panels[0].crop_to_data is True and r2.panels[1].crop_to_data is False
    d = json.loads(recipe_to_json(_mini_recipe()))
    for p in d["panels"]:
        p.pop("crop_to_data")
    r3 = recipe_from_json(json.dumps(d))
    assert all(p.crop_to_data is False for p in r3.panels)


def test_scale_bar_cell_round_trips():
    from dfxm.compose.recipe import ScaleBarCell

    r = _mini_recipe()
    r.layout.children.append(ScaleBarCell(3.5, 1.25))
    r2 = recipe_from_json(recipe_to_json(r))
    leaf = r2.layout.children[-1]
    assert isinstance(leaf, ScaleBarCell) and (leaf.w_cm, leaf.h_cm) == (3.5, 1.25)
    assert [type(x).__name__ for x in iter_leaves(r2.layout)][-1] == "ScaleBarCell"


def test_y_tick_labels_round_trips_and_old_recipe_defaults_true():
    import json

    r = _mini_recipe()
    r.panels[1].y_tick_labels = False  # the trace panel
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.panels[1].y_tick_labels is False and r2.panels[0].y_tick_labels is True
    d = json.loads(recipe_to_json(_mini_recipe()))
    for p in d["panels"]:
        p.pop("y_tick_labels")  # a recipe written before the field existed
    r3 = recipe_from_json(json.dumps(d))
    assert all(p.y_tick_labels is True for p in r3.panels)


def test_image_panel_width_round_trips_relative_path_and_defaults():
    import json
    import os

    r = _mini_recipe()
    r.panels.append(PanelDef("i0", __src("/data/figs/ref.png", "image", {}), width_cm=4.5))
    r.layout.children.append(PanelRef("i0"))
    txt = recipe_to_json(r, base_dir="/data")
    d = json.loads(txt)
    assert d["panels"][2]["source"]["h5_path"] == os.path.join("figs", "ref.png")
    r2 = recipe_from_json(txt, base_dir="/data")
    img = r2.panels[2]
    assert img.source.kind == "image" and img.source.h5_path == "/data/figs/ref.png"
    assert img.width_cm == 4.5 and r2.panels[0].width_cm is None
    validate_recipe(r2)  # "image" is a legal kind
    for p in d["panels"]:
        p.pop("width_cm")
    r3 = recipe_from_json(json.dumps(d), base_dir="/data")
    assert all(p.width_cm is None for p in r3.panels)


@pytest.mark.parametrize("bad", [0.0, -1.0, "abc", float("inf"), float("nan")])
def test_image_panel_width_validated(bad):
    # a hand-edited recipe must never produce a bare TypeError/ValueError: a
    # non-number and a non-finite value are refused the same way as a
    # non-positive one
    r = _mini_recipe()
    r.panels.append(PanelDef("i0", __src("/data/ref.png", "image", {}), width_cm=bad))
    r.layout.children.append(PanelRef("i0"))
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert "width_cm" in str(e.value) and e.value.hint


def test_image_panel_width_numeric_string_is_accepted():
    # float-castable, like layout._validate_scale; layout casts it when sizing
    r = _mini_recipe()
    r.panels.append(PanelDef("i0", __src("/data/ref.png", "image", {}), width_cm="3"))
    r.layout.children.append(PanelRef("i0"))
    validate_recipe(r)
