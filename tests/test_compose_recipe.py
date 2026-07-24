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
