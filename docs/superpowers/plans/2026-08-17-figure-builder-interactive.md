# Figure builder: panel titles, drag-grid arranger, united bars — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Spec: `docs/superpowers/specs/2026-08-17-figure-builder-interactive-design.md` (approved 2026-08-17).

## Goal

Give the figure builder human-readable panel titles everywhere, a mouse-driven drag-grid layout arranger (two-step Add-panels dialog + Arrange… dialog), and a "one colorbar per quantity" united-bars render mode with first-class scale-bar/colorbar position controls and an arranger schematic.

## Architecture

All model/renderer work lands Qt-free in `dfxm/compose/` first: `PanelDef.title` and `ComposeStyle.colorbar_mode`/`colorbar_pos` extend `recipe.py`, a new `gridmap.py` owns the grid ↔ layout-tree mapping with a tested round-trip law, and `render.py` gains a `_apply_united_colorbars` pass that reuses the existing shared-bar machinery (pre-measure bar drawing, post-placement stretch generalized to scattered members). The GUI layer then consumes it: a new `gui/widgets/layout_arranger.py` widget (plus `ArrangeDialog`), a two-step `AddPanelDialog`, and new compose-form controls in `gui/figure_builder.py`.

## Tech stack

Python 3, dataclasses + JSON for the recipe model, matplotlib **explicit `Figure` API only** for rendering, PySide6 for the GUI, pytest (+ offscreen Qt) for tests, h5py/numpy for fixtures.

## Global constraints

- **`dfxm/` stays Qt-free.** Never import PySide6 (or anything from `gui/`) under `dfxm/`. `gridmap.py` imports only `dfxm.compose.recipe`.
- **Matplotlib:** explicit `matplotlib.figure.Figure` API only — never `pyplot`, never `matplotlib.use(...)`.
- **Lint/format:** `ruff check .` must pass; `ruff format` runs automatically on Write/Edit via the repo hook — do not fight its output.
- **Docs contract:** any change to stage/GUI behaviour or module structure updates `docs/Usage.md` (user-visible) and/or `docs/Codebase.md` (code structure) **in the same task/commit**, never as a follow-up. Read the exact target region of the doc before editing (em-dashes and reflowed prose make reconstructed `old_string`s fail).
- **Tests:** `python3 -m pytest -q` must be green at the end of every task. The GUI smoke test is `tests/gui_smoke.py` (run `python3 tests/gui_smoke.py`; it is not a pytest file).
- **Read before first Edit** for any file not created this session. Line numbers in this plan refer to the tree at plan time — always locate edit sites by the quoted anchor text, not by counting lines.
- **No git remote** — never pull/push/PR; commits are local only.
- Existing per-panel-mode rendering must stay byte-for-byte: never touch the `_apply_shared_colorbars` behaviour path except where a task explicitly refactors an internal helper with its signature preserved.

---

## Task 1 — `PanelDef.title` in the Qt-free model

**Files:**
- Modify: `dfxm/compose/recipe.py` — `PanelDef` dataclass (anchor: `class PanelDef:` — fields end at `colorbar: bool | None = None`), `_panel_def_to_dict` (anchor: `def _panel_def_to_dict(p, rel):`), `_panel_def_from_dict` (anchor: `def _panel_def_from_dict(d, base_dir):`).
- Modify: `docs/Codebase.md` — the `dfxm/compose/recipe.py` bullet (search for `PanelDef`).
- Test: `tests/test_compose_recipe.py` (append).

**Interfaces:**
- Consumes: existing `PanelDef`, `recipe_to_json`, `recipe_from_json`.
- Produces: `PanelDef.title: str | None = None` — serialized as `"title"`; absent key loads as `None`. `RECIPE_VERSION` stays `1`.

**Steps:**

- [ ] Write the failing test (append to `tests/test_compose_recipe.py`):

```python
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
```

- [ ] Run `python3 -m pytest -q tests/test_compose_recipe.py` — expect `AttributeError: 'PanelDef' object has no attribute 'title'` (dataclass rejects the assignment via missing field → the `r.panels[0].title = ...` line fails only on frozen classes; here it sets an instance attr, so the actual failure is the round-trip assert: `r2.panels[0].title` raises `AttributeError`).
- [ ] Implement in `dfxm/compose/recipe.py`. Add to `PanelDef` after the `colorbar` field:

```python
    colorbar: bool | None = None  # None = follow style; False when a shared bar covers it
    title: str | None = None  # human-readable data name (display only); None = show the id
```

In `_panel_def_to_dict`, add to the returned dict after `"colorbar": p.colorbar,`:

```python
        "title": p.title,
```

In `_panel_def_from_dict`, add to the `PanelDef(...)` call after `colorbar=d.get("colorbar"),`:

```python
        title=d.get("title"),
```

- [ ] Run `python3 -m pytest -q tests/test_compose_recipe.py` — green; then `python3 -m pytest -q` — green; `ruff check .`.
- [ ] Update `docs/Codebase.md`: in the `dfxm/compose/recipe.py` description, extend the `PanelDef` field list with: `title` — optional human-readable data name captured by the panel picker at pick time (e.g. `"strain: Strain map / z=3"`); display-only (outline tree, scale-bar combo, arranger tiles show `title or id`), never part of identity; absent in old recipes → `None`; `RECIPE_VERSION` stays 1 (purely additive).
- [ ] Commit: `git add dfxm/compose/recipe.py tests/test_compose_recipe.py docs/Codebase.md && git commit -m "feat(compose): PanelDef.title optional display name (JSON round-trip, docs synced)"`

---

## Task 2 — `dfxm/compose/gridmap.py`: grid ↔ layout mapping

**Files:**
- Create: `dfxm/compose/gridmap.py`
- Modify: `docs/Codebase.md` — add a `gridmap.py` bullet in the `dfxm/compose/` section (after the `recipe.py` bullet).
- Test: `tests/test_compose_gridmap.py` (new).

**Interfaces:**
- Consumes: `dfxm.compose.recipe` (`Row`, `Col`, `PanelRef`, `iter_leaves`).
- Produces:
  - `GridModel = list` (documented as `list[list[str]]` — columns left→right, tiles top→bottom)
  - `layout_to_grid(layout, panels_by_id) -> GridModel | None`
  - `flatten_panel_ids(layout) -> list[str]`
  - `grid_to_layout(grid) -> Row`
- Round-trip law: `layout_to_grid(grid_to_layout(g), panels) == g` for any normalized grid (no empty columns, all ids known).
- Unmappable (→ `None`): spacers/text cells anywhere; nested `Row`; `Col` inside `Col`; ghost panel ids; group flags on the root `Row` (`group_label`/`shared_colorbar`/`shared_clim`/`pinned_height_cm`); a flagged `Col` with fewer than 2 members (the rebuild would silently drop it).

**Steps:**

- [ ] Write the failing test file `tests/test_compose_gridmap.py`:

```python
"""layout_to_grid / flatten_panel_ids / grid_to_layout — dfxm.compose.gridmap."""

import pytest

from dfxm.compose.gridmap import flatten_panel_ids, grid_to_layout, layout_to_grid
from dfxm.compose.recipe import Col, PanelDef, PanelRef, PanelSource, Row, Spacer, TextCell


def _panels(*pids):
    return {
        pid: PanelDef(pid, PanelSource("/x.h5", "map_layer", {"stage": "strain", "z": 0}))
        for pid in pids
    }


@pytest.mark.parametrize(
    "grid",
    [
        [],
        [["a"]],
        [["a", "b"]],
        [["a"], ["b", "c"], ["d"]],
    ],
)
def test_round_trip_law(grid):
    pids = [p for col in grid for p in col]
    assert layout_to_grid(grid_to_layout(grid), _panels(*pids)) == grid


def test_grid_to_layout_shapes():
    lay = grid_to_layout([["a"], ["b", "c"], []])
    assert isinstance(lay, Row)
    assert len(lay.children) == 2  # empty column dropped
    assert isinstance(lay.children[0], PanelRef) and lay.children[0].panel_id == "a"
    col = lay.children[1]
    assert isinstance(col, Col) and [c.panel_id for c in col.children] == ["b", "c"]
    assert isinstance(grid_to_layout([]), Row) and grid_to_layout([]).children == []


def test_layout_to_grid_recognized_shapes():
    p = _panels("a", "b", "c")
    assert layout_to_grid(PanelRef("a"), p) == [["a"]]
    assert layout_to_grid(Col([PanelRef("a"), PanelRef("b")]), p) == [["a", "b"]]
    assert layout_to_grid(Row([PanelRef("a"), Col([PanelRef("b"), PanelRef("c")])]), p) == [
        ["a"],
        ["b", "c"],
    ]
    assert layout_to_grid(Row([]), p) == []


@pytest.mark.parametrize(
    "layout",
    [
        Row([Spacer(1.0, 1.0)]),
        Row([TextCell("t")]),
        Row([Row([PanelRef("a")])]),
        Row([Col([Col([PanelRef("a")])])]),
        Row([PanelRef("ghost")]),
        Row([PanelRef("a")], group_label="auto"),
        Row([PanelRef("a")], shared_colorbar=True),
        Row([PanelRef("a")], pinned_height_cm=3.0),
        Row([Col([PanelRef("a")], shared_colorbar=True)]),  # flagged single-member Col
        TextCell("t"),
    ],
)
def test_unmappable_layouts_return_none(layout):
    assert layout_to_grid(layout, _panels("a")) is None


def test_flagged_multi_member_col_is_mappable():
    p = _panels("a", "b")
    lay = Row([Col([PanelRef("a"), PanelRef("b")], shared_colorbar=True, group_label="auto")])
    assert layout_to_grid(lay, p) == [["a", "b"]]


def test_flatten_panel_ids_dfs_order():
    lay = Row([PanelRef("a"), Col([PanelRef("b"), Spacer(1, 1), PanelRef("c")]), TextCell("t")])
    assert flatten_panel_ids(lay) == ["a", "b", "c"]
```

- [ ] Run `python3 -m pytest -q tests/test_compose_gridmap.py` — expect `ModuleNotFoundError: No module named 'dfxm.compose.gridmap'`.
- [ ] Create `dfxm/compose/gridmap.py`:

```python
"""Grid <-> layout-tree mapping for the drag arranger (Qt-free).

GridModel = list of columns (left->right), each a list of panel ids
(top->bottom). Only "plain grid" layouts map cleanly: a root Row of PanelRefs
and/or single-level Cols of PanelRefs. A Col's group/shared flags are allowed
only where the Col survives the rebuild (>= 2 members) — the caller
(ArrangeDialog) re-applies them to rebuilt Cols by member-id set. Everything
else — spacers, text cells, nested containers, ghost refs, flags the rebuilt
grid cannot represent — makes ``layout_to_grid`` return ``None``; callers then
offer the flatten-with-warning path seeded from :func:`flatten_panel_ids`.

Round-trip law (tested): ``layout_to_grid(grid_to_layout(g), panels) == g``
for any normalized grid (no empty columns, every id known)."""

from __future__ import annotations

from .recipe import Col, PanelRef, Row, iter_leaves

GridModel = list  # list[list[str]] — columns of panel ids


def _row_has_flags(row: Row) -> bool:
    return (
        row.group_label is not None
        or row.shared_colorbar
        or row.shared_clim is not None
        or row.pinned_height_cm is not None
    )


def _col_has_flags(col: Col) -> bool:
    return (
        col.group_label is not None
        or col.shared_x
        or col.shared_colorbar
        or col.shared_clim is not None
        or col.pinned_width_cm is not None
    )


def _col_ids(col: Col, panels_by_id) -> list[str] | None:
    ids: list[str] = []
    for child in col.children:
        if not isinstance(child, PanelRef) or child.panel_id not in panels_by_id:
            return None
        ids.append(child.panel_id)
    if _col_has_flags(col) and len(ids) < 2:
        return None  # the rebuild would silently drop this Col (and its flags)
    return ids


def layout_to_grid(layout, panels_by_id) -> GridModel | None:
    """Map *layout* to columns of panel ids, or ``None`` when it isn't a plain grid."""
    if isinstance(layout, PanelRef):
        return [[layout.panel_id]] if layout.panel_id in panels_by_id else None
    if isinstance(layout, Col):
        ids = _col_ids(layout, panels_by_id)
        return None if ids is None else ([ids] if ids else [])
    if not isinstance(layout, Row) or _row_has_flags(layout):
        return None
    grid: GridModel = []
    for child in layout.children:
        if isinstance(child, PanelRef):
            if child.panel_id not in panels_by_id:
                return None
            grid.append([child.panel_id])
        elif isinstance(child, Col):
            ids = _col_ids(child, panels_by_id)
            if ids is None:
                return None
            if ids:
                grid.append(ids)
        else:
            return None
    return grid


def flatten_panel_ids(layout) -> list[str]:
    """Every PanelRef id under *layout*, depth-first — the flatten-path seed."""
    return [leaf.panel_id for leaf in iter_leaves(layout) if isinstance(leaf, PanelRef)]


def grid_to_layout(grid) -> Row:
    """Build ``Row([...])`` from *grid*: a ``Col([PanelRef...])`` per multi-tile
    column, a bare ``PanelRef`` per single-tile column; empty columns dropped;
    an empty grid yields ``Row([])``."""
    children = []
    for column in grid:
        if not column:
            continue
        if len(column) == 1:
            children.append(PanelRef(column[0]))
        else:
            children.append(Col([PanelRef(pid) for pid in column]))
    return Row(children)
```

- [ ] Run `python3 -m pytest -q tests/test_compose_gridmap.py` — green; `python3 -m pytest -q` — green; `ruff check .`.
- [ ] Update `docs/Codebase.md` `dfxm/compose/` section with a new bullet for `gridmap.py`: the `GridModel` definition, all three function signatures, the round-trip law, the exact unmappable conditions (list them), and that it is Qt-free and imported by `gui/widgets/layout_arranger.py` + the two arranger dialogs.
- [ ] Commit: `git add dfxm/compose/gridmap.py tests/test_compose_gridmap.py docs/Codebase.md && git commit -m "feat(compose): gridmap grid<->layout mapping with round-trip law (docs synced)"`

---

## Task 3 — `ComposeStyle.colorbar_mode` / `colorbar_pos` + validation

**Files:**
- Modify: `dfxm/compose/recipe.py` — constants block (anchor: `SCALE_BAR_MODES = (`), `ComposeStyle` dataclass, `validate_recipe` (anchor: `if recipe.compose.scale_bar_mode not in SCALE_BAR_MODES:`).
- Modify: `docs/Codebase.md` — the `recipe.py` bullet (`ComposeStyle` fields).
- Test: `tests/test_compose_recipe.py` (append + extend the parametrized validate test).

**Interfaces:**
- Produces: `COLORBAR_MODES = ("per-panel", "united")`, `COLORBAR_POSITIONS = ("right", "bottom")`, `ComposeStyle.colorbar_mode: str = "per-panel"`, `ComposeStyle.colorbar_pos: str = "right"`; `validate_recipe` raises `StageUserError` (message contains the field name, hint present) on unknown values. Old recipes (no keys) load with the defaults via `ComposeStyle(**d.get("compose", {}))` — no loader change needed.

**Steps:**

- [ ] Write failing tests. In `tests/test_compose_recipe.py`, add two rows to the existing `test_validate_refuses_bad_recipes` parametrize list (after the `label_template` row):

```python
        (lambda r: setattr(r.compose, "colorbar_mode", "rainbow"), "colorbar_mode"),
        (lambda r: setattr(r.compose, "colorbar_pos", "left"), "colorbar_pos"),
```

and append:

```python
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
```

- [ ] Run `python3 -m pytest -q tests/test_compose_recipe.py` — expect the new parametrized cases to fail with `DID NOT RAISE` and the round-trip test to fail with `KeyError: 'colorbar_mode'` on the `pop`.
- [ ] Implement in `dfxm/compose/recipe.py`. After `SCALE_BAR_MODES = (...)` add:

```python
COLORBAR_MODES = ("per-panel", "united")
COLORBAR_POSITIONS = ("right", "bottom")
```

In `ComposeStyle`, after `pinned_width_cm: float | None = None` add:

```python
    colorbar_mode: str = "per-panel"  # one of COLORBAR_MODES
    colorbar_pos: str = "right"  # one of COLORBAR_POSITIONS (united mode only)
```

In `validate_recipe`, right after the `scale_bar_mode` check block add:

```python
    if recipe.compose.colorbar_mode not in COLORBAR_MODES:
        raise StageUserError(
            f"invalid compose.colorbar_mode {recipe.compose.colorbar_mode!r}",
            hint=f"colorbar_mode must be one of {COLORBAR_MODES}.",
        )
    if recipe.compose.colorbar_pos not in COLORBAR_POSITIONS:
        raise StageUserError(
            f"invalid compose.colorbar_pos {recipe.compose.colorbar_pos!r}",
            hint=f"colorbar_pos must be one of {COLORBAR_POSITIONS}.",
        )
```

- [ ] Run `python3 -m pytest -q tests/test_compose_recipe.py` then `python3 -m pytest -q` — green; `ruff check .`.
- [ ] Update `docs/Codebase.md` `recipe.py` bullet: document the two new `ComposeStyle` fields, the two constants, and the validation rules.
- [ ] Commit: `git add dfxm/compose/recipe.py tests/test_compose_recipe.py docs/Codebase.md && git commit -m "feat(compose): ComposeStyle colorbar_mode/colorbar_pos with validation (docs synced)"`

---

## Task 4 — renderer united-colorbars core pass

**Files:**
- Modify: `dfxm/compose/render.py` —
  - new function `_apply_united_colorbars` (place directly after `_apply_shared_colorbars`, anchor: `def _stretch_shared_bar(`);
  - `render_recipe`: the shared-colorbars call site (anchor: `no_colorbar_pids, bar_specs = _apply_shared_colorbars(`), the working-layout build (anchor: `working_layout = _build_working_layout(recipe.layout, bar_map)`), the map-cell branch of the draw loop (anchor: `if pid in no_colorbar_pids:`), and the shared-bar pre-measure draw loop (anchor: `for _node, grp, pids, _bar_leaf, bar_ax in bar_specs:`).
- Modify: `docs/Codebase.md` (`render.py` bullet) and `docs/Usage.md` (Figure builder → **Concepts**/Recipe description: mention the recipe-level `colorbar_mode`/`colorbar_pos` united option, reachable from saved recipes and the CLI).
- Test: `tests/test_compose_render.py` (append).

**Interfaces:**
- Consumes: `PanelData.group`, `_panel_leaves`, `_cbar_label`, `add_colorbar`, `SizedCell`, `Spacer`, `dc_replace`, Task 3's `ComposeStyle.colorbar_mode`/`colorbar_pos`.
- Produces:
  - `_apply_united_colorbars(recipe, style, panels_by_id, data_by_id, cells, fig, notes) -> tuple[set[str], list, set[str]]` returning `(no_colorbar_pids, united_specs, forced_pids)` where each united spec is `(grp: str, pids: list[str], bar_leaf: Spacer, bar_ax)`.
  - In united mode: node-level `shared_colorbar` flags ignored with note `"united colorbars override {n} group flag(s)"`; zero groupable panels → note `"united colorbars: no eligible panels — nothing to unite"`, no wrap, no error; `panel.colorbar is True` excludes that panel from grouping and forces its own bar; working layout wrapped once — `Row([root, Col(bars)])` for `"right"`, `Col([root, Row(bars)])` for `"bottom"`; united bars drawn pre-measure like shared bars.
  - `"per-panel"` mode: byte-for-byte today's behaviour.
- Note for Task 5: this task leaves united bars at their provisional placed size — the post-placement stretch is Task 5.

**Steps:**

- [ ] Write failing tests (append to `tests/test_compose_render.py`):

```python
# -- united colorbars (colorbar_mode="united") --------------------------------
def _united_recipe(h5, *, pos="right"):
    def mk(pid, vid):
        return PanelDef(
            pid,
            PanelSource(h5, "slice_plane", {"volume_id": vid, "slice_name": "obl", "plane": 0}),
        )

    return FigureRecipe(
        "united",
        {"scale_um_per_cm": 10.0, "show_title": False},
        ComposeStyle(colorbar_mode="united", colorbar_pos=pos),
        Row([PanelRef("a"), PanelRef("b"), PanelRef("c")]),
        [mk("a", "strain"), mk("b", "raw_sum"), mk("c", "strain")],
    )


def test_united_one_bar_per_quantity_and_clims_unified(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _united_recipe(h5)
    r.panels[2].clim = (-20.0, 5.0)  # "c" widens the strain union
    res = render_recipe(r)
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(extra) == 2  # one strain bar + one raw bar, no per-panel bars
    for pid in ("a", "c"):
        im = res.axes_by_id[pid].images[0]
        assert (im.norm.vmin, im.norm.vmax) == (-20.0, 10.0)
    imb = res.axes_by_id["b"].images[0]
    assert (imb.norm.vmin, imb.norm.vmax) == (-10.0, 10.0)  # raw group untouched


def test_united_right_and_bottom_wrapping(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    res_r = render_recipe(_united_recipe(h5, pos="right"))
    panels = list(res_r.axes_by_id.values())
    max_x1 = max(ax.get_position().x1 for ax in panels)
    extra = [ax for ax in res_r.figure.axes if ax not in panels]
    assert extra and all(ax.get_position().x0 >= max_x1 - 1e-6 for ax in extra)
    res_b = render_recipe(_united_recipe(h5, pos="bottom"))
    panels_b = list(res_b.axes_by_id.values())
    min_y0 = min(ax.get_position().y0 for ax in panels_b)
    extra_b = [ax for ax in res_b.figure.axes if ax not in panels_b]
    assert extra_b and all(ax.get_position().y1 <= min_y0 + 1e-6 for ax in extra_b)


def test_united_ignores_group_flags_with_note(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _united_recipe(h5)
    r.layout = Row([Col([PanelRef("a"), PanelRef("c")], shared_colorbar=True), PanelRef("b")])
    res = render_recipe(r)
    assert any("override 1 group flag" in n for n in res.notes)
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(extra) == 2  # united bars only — the flagged Col added no bar


def test_united_panel_colorbar_true_forces_own_bar(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _united_recipe(h5)
    r.panels[0].colorbar = True  # "a" keeps its own bar, excluded from the union
    r.panels[2].clim = (-20.0, 5.0)
    res = render_recipe(r)
    ima = res.axes_by_id["a"].images[0]
    assert (ima.norm.vmin, ima.norm.vmax) == (-10.0, 10.0)  # NOT unified with "c"
    imc = res.axes_by_id["c"].images[0]
    assert (imc.norm.vmin, imc.norm.vmax) == (-20.0, 5.0)  # union of {c} alone
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(extra) == 3  # a's own cax + strain united bar + raw united bar


def test_united_trace_panels_keep_per_panel_behaviour(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p1 = PanelDef(
        "a",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
    )
    p2 = PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    r = FigureRecipe(
        "mix",
        {"scale_um_per_cm": 10.0, "trace_scale_um_per_cm": 5.0, "show_title": False},
        ComposeStyle(colorbar_mode="united"),
        Row([PanelRef("a"), PanelRef("t")]),
        [p1, p2],
    )
    res = render_recipe(r)
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(extra) == 1  # one united strain bar; the trace contributes nothing


def test_united_zero_groupable_panels_note_no_error(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    pt = [
        PanelDef(f"t{i}", PanelSource(h5, "profiles_trace", {"job": JOB, "field": vid}))
        for i, vid in enumerate(["raw_sum", "strain"])
    ]
    r = FigureRecipe(
        "tunited",
        {"trace_scale_um_per_cm": 5.0, "trace_height_cm": 2.0, "show_title": False},
        ComposeStyle(colorbar_mode="united"),
        Col([PanelRef("t0"), PanelRef("t1")]),
        pt,
    )
    res = render_recipe(r)
    assert res.n_rendered == 2
    assert any("nothing to unite" in n for n in res.notes)
```

- [ ] Run `python3 -m pytest -q tests/test_compose_render.py -k united` — expect assertion failures (`len(extra) == 2` etc. — today united mode changes nothing; every panel draws its own bar).
- [ ] Implement in `dfxm/compose/render.py`. Add after `_apply_shared_colorbars` (before `_stretch_shared_bar`):

```python
def _apply_united_colorbars(recipe, style, panels_by_id, data_by_id, cells, fig, notes):
    """One bar per quantity group (``colorbar_mode == "united"``).

    Partitions live, non-placeholder map/slice/ref panels by ``data.group``
    (first-seen DFS order). ``group=None`` panels and traces keep their
    per-panel behaviour; a ``panel.colorbar is True`` override excludes the
    panel from grouping (it keeps its own bar — explicit override outranks
    the mode). Per group: clim unified as the union of member effective
    ranges (per-panel ``clim`` overrides respected, like the shared path),
    members rewritten via ``dc_replace`` and their per-panel bars suppressed,
    and one provisional bar leaf built (cross-dimension corrected after
    placement, see ``_stretch_bar_to_span``). Returns
    ``(no_colorbar_pids, united_specs, forced_pids)`` with each spec
    ``(group, member_pids, bar_leaf, bar_ax)``.
    """
    no_colorbar_pids: set[str] = set()
    forced_pids: set[str] = set()
    groups: dict[str, list] = {}
    for leaf in _panel_leaves(recipe.layout):
        pid = leaf.panel_id
        d = data_by_id[pid]
        if d.kind not in ("map_layer", "slice_plane", "profiles_ref"):
            continue
        if panels_by_id[pid].colorbar is True:
            forced_pids.add(pid)
            continue
        if d.group is None:
            continue
        groups.setdefault(d.group, []).append(leaf)

    united_specs = []
    for grp, members in groups.items():
        pids = [m.panel_id for m in members]
        vmins, vmaxs = [], []
        for pid in pids:
            d = data_by_id[pid]
            lo, hi = panels_by_id[pid].clim if panels_by_id[pid].clim is not None else (None, None)
            vmins.append(lo if lo is not None else d.vmin)
            vmaxs.append(hi if hi is not None else d.vmax)
        unified = (min(vmins), max(vmaxs))
        for pid in pids:
            panels_by_id[pid] = dc_replace(panels_by_id[pid], clim=unified)
            no_colorbar_pids.add(pid)
        first = cells[id(members[0])]
        if recipe.compose.colorbar_pos == "right":
            bar_w_in = style.colorbar_fraction * first.w_in + 0.1
            bar_h_in = first.h_in
        else:
            bar_w_in = first.w_in
            bar_h_in = style.colorbar_fraction * first.h_in + 0.1
        bar_leaf = Spacer(bar_w_in / _IN_PER_CM, bar_h_in / _IN_PER_CM)
        bar_ax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
        cells[id(bar_leaf)] = SizedCell(bar_leaf, None, "spacer", bar_w_in, bar_h_in, ax=bar_ax)
        united_specs.append((grp, pids, bar_leaf, bar_ax))

    if not united_specs:
        notes.append("united colorbars: no eligible panels — nothing to unite")
    return no_colorbar_pids, united_specs, forced_pids
```

In `render_recipe`, replace the two lines

```python
    no_colorbar_pids, bar_specs = _apply_shared_colorbars(
        recipe, style, panels_by_id, data_by_id, cells, fig
    )
    bar_map = {id(node): bar_leaf for node, _grp, _pids, bar_leaf, _ax in bar_specs}
```

with:

```python
    united = recipe.compose.colorbar_mode == "united"
    if united:
        n_flagged = sum(1 for _ in _find_shared_bar_nodes(recipe.layout))
        if n_flagged:
            notes.append(f"united colorbars override {n_flagged} group flag(s)")
        no_colorbar_pids, united_specs, forced_pids = _apply_united_colorbars(
            recipe, style, panels_by_id, data_by_id, cells, fig, notes
        )
        bar_specs = []
    else:
        no_colorbar_pids, bar_specs = _apply_shared_colorbars(
            recipe, style, panels_by_id, data_by_id, cells, fig
        )
        united_specs, forced_pids = [], set()
    bar_map = {id(node): bar_leaf for node, _grp, _pids, bar_leaf, _ax in bar_specs}
```

Right after `working_layout = _build_working_layout(recipe.layout, bar_map)` insert (BEFORE the gutter-leaf wrap so the gutter stays outermost-bottom):

```python
    if united and united_specs:
        united_bars = [spec[2] for spec in united_specs]
        if recipe.compose.colorbar_pos == "right":
            working_layout = Row([working_layout, Col(united_bars)])
        else:
            working_layout = Col([working_layout, Row(united_bars)])
```

In the draw loop's map branch, replace

```python
            cax = None
            if pid in no_colorbar_pids:
                colorbar_kw = False
```

with:

```python
            cax = None
            if pid in forced_pids:
                # explicit panel-level override outranks united mode
                colorbar_kw = True
                cax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
                cell.extras = (cax,)
                cell.sync = _make_cax_sync(cax, style, cell)
            elif pid in no_colorbar_pids:
                colorbar_kw = False
```

Right after the existing shared-bar pre-measure draw loop (`for _node, grp, pids, _bar_leaf, bar_ax in bar_specs:` … `add_colorbar(...)`) add:

```python
    # United bars are drawn pre-measure too — same reserved-margin rule.
    for grp, pids, _bar_leaf, bar_ax in united_specs:
        rep_pid = next((pid for pid in pids if im_by_pid.get(pid) is not None), None)
        if rep_pid is None:
            bar_ax.set_axis_off()
            continue
        add_colorbar(
            fig,
            im_by_pid[rep_pid],
            cell_by_pid[rep_pid].ax,
            _cbar_label(data_by_id[rep_pid]),
            style,
            group=grp,
            cax=bar_ax,
        )
```

- [ ] Run `python3 -m pytest -q tests/test_compose_render.py` — all green (including every pre-existing per-panel/shared test, proving per-panel mode is untouched); `python3 -m pytest -q`; `ruff check .`.
- [ ] Docs: `docs/Codebase.md` `render.py` bullet — describe `_apply_united_colorbars` (signature, grouping/exclusion rules, override, the root-wrap `Row([root, Col([bars])])`/`Col([root, Row([bars])])`, flags-ignored note, provisional bar size corrected in Task 5's stretch). `docs/Usage.md` Figure builder **Concepts** → extend the Recipe bullet: composer-level settings now include a colourbar mode (`per-panel` — today's behaviour — or `united`: one bar per quantity, placed on the right or bottom edge; group flags on rows/columns are ignored in united mode with a note; a panel's own Colourbar=On override still forces its private bar).
- [ ] Commit: `git add dfxm/compose/render.py tests/test_compose_render.py docs/Codebase.md docs/Usage.md && git commit -m "feat(compose): united per-quantity colorbars render pass (docs synced)"`

---

## Task 5 — stretch united bars to scattered members' union span

**Files:**
- Modify: `dfxm/compose/render.py` — `_stretch_shared_bar` (anchor: `def _stretch_shared_bar(node, pids, bar_ax, axes_by_id, data_by_id):`) and the post-placement stretch loop in `render_recipe` (anchor: `for node, _grp, pids, _bar_leaf, bar_ax in bar_specs:` near the end).
- Modify: `docs/Codebase.md` — `render.py` bullet.
- Test: `tests/test_compose_render.py` (append).

**Interfaces:**
- Produces: `_stretch_bar_to_span(bar_ax, member_axes, vertical: bool) -> None` — extracted body of `_stretch_shared_bar` with the Col/Row test replaced by the `vertical` flag (same end-inset clamp, same collapsed-bar degradation). `_stretch_shared_bar(node, pids, bar_ax, axes_by_id, data_by_id)` keeps its exact signature and delegates with `vertical=isinstance(node, Col)`. United bars stretched with `vertical=(recipe.compose.colorbar_pos == "right")` over ALL member axes (scattered members included).

**Steps:**

- [ ] Write the failing test (append to `tests/test_compose_render.py`):

```python
def test_united_bar_stretches_to_scattered_members_union_span(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")

    def mk(pid, vid):
        return PanelDef(
            pid,
            PanelSource(h5, "slice_plane", {"volume_id": vid, "slice_name": "obl", "plane": 0}),
        )

    # vertical stack: strain (top), raw (middle), strain (bottom) — the strain
    # bar must span from the top panel's top to the bottom panel's bottom,
    # bridging the raw panel between them.
    r = FigureRecipe(
        "span",
        {"scale_um_per_cm": 10.0, "show_title": False},
        ComposeStyle(colorbar_mode="united", colorbar_pos="right"),
        Col([PanelRef("a"), PanelRef("b"), PanelRef("c")]),
        [mk("a", "strain"), mk("b", "raw_sum"), mk("c", "strain")],
    )
    res = render_recipe(r)
    top = res.axes_by_id["a"].get_position().y1
    bottom = res.axes_by_id["c"].get_position().y0
    extra = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    strain_bar = max(extra, key=lambda ax: ax.get_position().height)
    bp = strain_bar.get_position()
    eps = 1e-6
    assert bottom - eps <= bp.y0 and bp.y1 <= top + eps  # inside the union span
    assert bp.height > 0.8 * (top - bottom)  # and covering most of it
```

- [ ] Run `python3 -m pytest -q tests/test_compose_render.py -k scattered` — expect the `bp.height > 0.8 * (top - bottom)` assert to fail (the bar still has its provisional single-panel height from Task 4).
- [ ] Implement in `dfxm/compose/render.py`: rename the body of `_stretch_shared_bar` into a new function placed directly above it, and make the old name a thin delegate:

```python
def _stretch_bar_to_span(bar_ax, member_axes, vertical: bool):
    """Stretch/reposition *bar_ax* to the union span of *member_axes* along the
    bar's long axis (``vertical=True`` -> y span, else x span), then inset the
    ends so the bar's own decorations (end tick labels) stay inside the span —
    the same end-inset clamp as the original shared-bar fix (2026-07-25).
    Members may be scattered anywhere in the figure (united mode), not just a
    contiguous Row/Col group."""
    if not member_axes:
        bar_ax.set_axis_off()
        return
    fig = bar_ax.figure
    bpos = bar_ax.get_position()
    if vertical:
        top = max(ax.get_position().y1 for ax in member_axes)
        bottom = min(ax.get_position().y0 for ax in member_axes)
        bar_ax.set_position([bpos.x0, bottom, bpos.width, top - bottom])
    else:
        left = min(ax.get_position().x0 for ax in member_axes)
        right = max(ax.get_position().x1 for ax in member_axes)
        bar_ax.set_position([left, bpos.y0, right - left, bpos.height])
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    bb = bar_ax.get_tightbbox(ren)
    pos = bar_ax.get_position()
    if vertical:
        span_lo, span_hi = (
            fig.transFigure.transform((0.0, bottom))[1],
            fig.transFigure.transform((0.0, top))[1],
        )
        under = max(0.0, span_lo - bb.y0)
        over = max(0.0, bb.y1 - span_hi)
        if under or over:
            h_px = fig.get_size_inches()[1] * fig.dpi
            # clamp: a span smaller than the bar's own decorations degrades to
            # a collapsed bar, never an inverted (negative-height) axes
            new_h = max(0.0, pos.height - (under + over) / h_px)
            bar_ax.set_position([pos.x0, pos.y0 + under / h_px, pos.width, new_h])
    else:
        span_lo, span_hi = (
            fig.transFigure.transform((left, 0.0))[0],
            fig.transFigure.transform((right, 0.0))[0],
        )
        under = max(0.0, span_lo - bb.x0)
        over = max(0.0, bb.x1 - span_hi)
        if under or over:
            w_px = fig.get_size_inches()[0] * fig.dpi
            new_w = max(0.0, pos.width - (under + over) / w_px)
            bar_ax.set_position([pos.x0 + under / w_px, pos.y0, new_w, pos.height])


def _stretch_shared_bar(node, pids, bar_ax, axes_by_id, data_by_id):
    """Shared-group form of :func:`_stretch_bar_to_span`: a Col group's bar is
    vertical, a Row group's horizontal; placeholder members are skipped."""
    member_axes = [axes_by_id[pid] for pid in pids if data_by_id[pid].kind != "placeholder"]
    _stretch_bar_to_span(bar_ax, member_axes, isinstance(node, Col))
```

(Keep the original explanatory docstring content of `_stretch_shared_bar` merged into `_stretch_bar_to_span`'s docstring — do not lose the 2026-07-25 provenance notes.) Then, in `render_recipe`, right after the existing shared-bar stretch loop, add:

```python
    for _grp, pids, _bar_leaf, bar_ax in united_specs:
        member_axes = [axes_by_id[pid] for pid in pids]
        _stretch_bar_to_span(bar_ax, member_axes, recipe.compose.colorbar_pos == "right")
```

- [ ] Run `python3 -m pytest -q tests/test_compose_render.py` — green, INCLUDING `test_shared_colorbar_unified_clim_and_single_bar` and `test_shared_colorbar_decorations_reserved_not_overlapping` (the refactor must be behaviour-preserving); `python3 -m pytest -q`; `ruff check .`.
- [ ] Docs: `docs/Codebase.md` `render.py` bullet — note the `_stretch_bar_to_span(bar_ax, member_axes, vertical)` extraction, that `_stretch_shared_bar` delegates to it, and that united bars stretch to the scattered members' union span after placement.
- [ ] Commit: `git add dfxm/compose/render.py tests/test_compose_render.py docs/Codebase.md && git commit -m "feat(compose): stretch united bars to scattered members' union span (docs synced)"`

---

## Task 6 — titles captured by the picker + shown in outline and scale-bar combo

**Files:**
- Modify: `gui/widgets/panel_picker.py` — `_build_map_tree` (anchor: `selector: dict = {"stage": stage, "z": z}`), `_build_slice_tree` (anchor: `"kind": "slice_plane",`), `_build_profiles_tree` (both `setData` calls), `_build_panels` (anchor: `panels.append(PanelDef(id=pid, source=src))`).
- Modify: `gui/figure_builder.py` — `_node_label` (anchor: `if isinstance(node, PanelRef):` inside `_node_label`), `_refresh_compose_panel_combo` (whole method), `_on_compose_edited` (anchor: `c.scale_bar_panel = self._compose_scale_bar_panel.currentText() or None`).
- Modify: `docs/Usage.md` (Figure builder — outline + compose pane paragraphs) and `docs/Codebase.md` (`panel_picker.py` + `figure_builder.py` rows).
- Test: `tests/test_gui_figure_builder.py` (append).

**Interfaces:**
- Consumes: Task 1's `PanelDef.title`.
- Produces: picker leaf `UserRole` dicts gain `"title"`; `_build_panels` sets `PanelDef.title` from it. Title formats: map `f"{stage}: {grp.label} / {layer label}"`; slices `f"{volume_id}/{slice_name} / {plane label}"`; profiles `f"{entry.name} / reference"` / `f"{entry.name} / {vid}"`. Display sites show `title or id`: outline `Panel: {title or id}` (with the existing `(label off)` / `({label})` suffixes preserved), scale-bar combo shows the title with the **id stored as item data** (`addItem(text, id)` / `findData` / `currentData`). Ids remain the unique key; render/export notes keep ids.

**Steps:**

- [ ] Write failing tests (append to `tests/test_gui_figure_builder.py`):

```python
# -- panel titles (picker capture + display sites) ----------------------------
def test_panel_picker_slice_leaves_carry_titles(tmp_path):
    import h5py
    import numpy as np

    from PySide6.QtCore import Qt

    from gui.widgets.panel_picker import AddPanelDialog

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0, 1, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 1, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0]))
    dlg = AddPanelDialog({"slices": {"h5": str(h5), "sx": 0.5, "sy": 0.5, "jobs": []}})
    dlg._stage.setCurrentText("slices")
    dlg._reload()
    leaf = dlg._tree.topLevelItem(0).child(0)
    data = leaf.data(0, Qt.ItemDataRole.UserRole)
    assert data["title"] == "strain/obl / plane 0  @ +0.00 µm"
    dlg._check_all()
    panels = dlg._build_panels()
    assert panels[0].title == "strain/obl / plane 0  @ +0.00 µm"


def test_panel_picker_map_titles_from_fake_catalog():
    from PySide6.QtCore import Qt

    from gui.widgets.panel_picker import AddPanelDialog

    class _Grp:
        key = "/chi/Center of mass"
        label = "Mosa χ COM"
        item_labels = ["z=0", "z=1"]

    dlg = AddPanelDialog({})
    dlg._stage.setCurrentText("mosaicity")
    dlg._tree.clear()
    dlg._build_map_tree("mosaicity", [_Grp()])
    leaf = dlg._tree.topLevelItem(0).child(1)
    assert leaf.data(0, Qt.ItemDataRole.UserRole)["title"] == "mosaicity: Mosa χ COM / z=1"


def test_outline_and_scale_bar_combo_show_titles_store_ids():
    w = _win()
    p = _panel("a")
    p.title = "strain: eps / z=0"
    w.add_panels([p])
    assert "strain: eps / z=0" in w._tree.topLevelItem(0).child(0).text(0)
    assert "Panel: a" not in w._tree.topLevelItem(0).child(0).text(0)
    combo = w._compose_scale_bar_panel
    idx = combo.findData("a")
    assert idx > 0 and combo.itemText(idx) == "strain: eps / z=0"
    combo.setCurrentIndex(idx)
    assert w.recipe().compose.scale_bar_panel == "a"  # data (id), not display text


def test_outline_title_fallback_is_id_and_label_off_survives():
    w = _win()
    w.add_panels([_panel("a")])  # no title
    w.recipe().panels[0].label = ""
    w._rebuild_tree()
    text = w._tree.topLevelItem(0).child(0).text(0)
    assert "Panel: a" in text and "label off" in text
```

- [ ] Run `python3 -m pytest -q tests/test_gui_figure_builder.py -k "title or combo_show"` — expect `KeyError: 'title'` on the picker tests and `AssertionError` on the combo test (`findData("a")` returns -1 today: the combo stores text only).
- [ ] Implement `gui/widgets/panel_picker.py`. In `_build_map_tree`, replace the leaf `setData` call with:

```python
                child.setData(
                    0,
                    Qt.ItemDataRole.UserRole,
                    {
                        "kind": "map_layer",
                        "selector": selector,
                        "title": f"{stage}: {grp.label} / {label}",
                    },
                )
```

In `_build_slice_tree`, add to the leaf's UserRole dict (after the `"selector": {...}` entry):

```python
                        "title": f"{e.volume_id}/{e.slice_name} / {label}",
```

In `_build_profiles_tree`, add to the reference leaf's dict: `"title": f"{e.name} / reference",` and to the field leaf's dict: `"title": f"{e.name} / {vid}",`. In `_build_panels`, replace `panels.append(PanelDef(id=pid, source=src))` with:

```python
                        panels.append(PanelDef(id=pid, source=src, title=data.get("title")))
```

- [ ] Implement `gui/figure_builder.py`. Replace the `PanelRef` branch of `_node_label` with:

```python
        if isinstance(node, PanelRef):
            panel = self._recipe.panel_by_id().get(node.panel_id)
            shown = (panel.title if panel and panel.title else None) or node.panel_id
            if panel is not None and panel.label == "":
                return f"Panel: {shown} (label off)"
            suffix = f" ({panel.label})" if panel and panel.label else ""
            return f"Panel: {shown}{suffix}"
```

Replace the body of `_refresh_compose_panel_combo` with:

```python
        combo = self._compose_scale_bar_panel
        target = self._recipe.compose.scale_bar_panel or ""
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("", "")  # "" = no single panel designated
        for p in self._recipe.panels:
            combo.addItem(p.title or p.id, p.id)
        idx = combo.findData(target)
        combo.setCurrentIndex(idx if idx >= 0 else 0)
        combo.blockSignals(False)
```

In `_on_compose_edited`, replace `c.scale_bar_panel = self._compose_scale_bar_panel.currentText() or None` with:

```python
        c.scale_bar_panel = self._compose_scale_bar_panel.currentData() or None
```

- [ ] Run `python3 -m pytest -q tests/test_gui_figure_builder.py` then `python3 -m pytest -q` — green; `ruff check .`.
- [ ] Docs same change: `docs/Usage.md` Figure builder — in the live-preview/outline paragraph state that panels now show their **data name** (stage/group/layer, slice plane, or job/field, captured when picked) in the outline and the Scale-bar panel dropdown, falling back to the internal id for old recipes; render/export notes keep ids. `docs/Codebase.md` — `panel_picker.py` row: leaves carry `"title"` in UserRole, `_build_panels` threads it into `PanelDef.title` (formats listed); `figure_builder.py` row: `_node_label` shows `title or id`, scale-bar combo stores the id as item data (`findData`/`currentData`).
- [ ] Commit: `git add gui/widgets/panel_picker.py gui/figure_builder.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md && git commit -m "feat(gui): panel titles captured at pick time, shown in outline + scale-bar combo (docs synced)"`

---

## Task 7 — `LayoutArranger` drag-grid widget

**Files:**
- Create: `gui/widgets/layout_arranger.py`
- Modify: `docs/Codebase.md` — new `layout_arranger.py` row in the `gui/widgets/` table.
- Test: `tests/test_gui_layout_arranger.py` (new).

**Interfaces:**
- Consumes: `dfxm.compose.gridmap.GridModel` shape (list of columns of pids) — the widget itself has **no recipe knowledge**.
- Produces (module `gui/widgets/layout_arranger.py`):
  - `GROUP_COLORS: dict[str | None, str]` — chip colours for `mosa_com`/`mosa_fwhm`/`strain`/`raw`/`trace`/`None`.
  - `_corner_at(pos, rect) -> str | None` — corner hot-zone mapping (10 px) to `"upper left" | "upper right" | "lower left" | "lower right"`.
  - `class LayoutArranger(QWidget)` — signals `gridChanged = Signal()`, `scaleBarPicked = Signal(str, str)`; methods `set_grid(grid, tile_info_by_id)` (tile info: `{pid: {"title": str, "group": str | None}}`), `grid() -> list[list[str]]` (empty columns dropped), `set_bar_schematic(colorbar_mode, colorbar_pos, flagged_member_sets=None)`, `set_scale_bar(panel_id, loc)`; internals `_columns: list[_ArrangerColumn]`, `_on_add_column()`, `_move_column(col, delta)`, `_remove_column(col)` (tiles merge into the neighbouring column; no-op with 1 column), `_on_corner_clicked(pid, loc)`, `_emit_grid_changed()`, `_column_ids(col)`, `_right_strip`/`_bottom_strip` (united schematic), per-column `flag_strip` (group-mode schematic).
- Tasks 8/9 import `LayoutArranger` from this module; Task 9 adds `ArrangeDialog` beside it.

**Steps:**

- [ ] Write the failing test file `tests/test_gui_layout_arranger.py`:

```python
"""LayoutArranger drag-grid widget (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QRect  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui.widgets.layout_arranger import LayoutArranger, _corner_at  # noqa: E402

_INFO = {
    "a": {"title": "strain / z=0", "group": "strain"},
    "b": {"title": "raw / z=0", "group": "raw"},
    "c": {"title": "job / strain", "group": "trace"},
}


def _arr(grid):
    w = LayoutArranger()
    w.set_grid(grid, _INFO)
    return w


def test_set_grid_grid_round_trip():
    assert _arr([["a"], ["b", "c"]]).grid() == [["a"], ["b", "c"]]


def test_item_move_between_columns_updates_grid_and_signals():
    w = _arr([["a"], ["b", "c"]])
    fired = []
    w.gridChanged.connect(lambda: fired.append(1))
    item = w._columns[1].list.takeItem(0)  # the model move a drag performs
    w._columns[0].list.addItem(item)
    assert w.grid() == [["a", "b"], ["c"]]
    assert fired


def test_add_move_remove_column():
    w = _arr([["a"], ["b"]])
    w._on_add_column()
    assert len(w._columns) == 3
    assert w.grid() == [["a"], ["b"]]  # empty column normalized away in grid()
    w._move_column(w._columns[1], -1)
    assert w.grid() == [["b"], ["a"]]
    w._remove_column(w._columns[0])  # "b" merges into its right neighbour
    assert w.grid() == [["a", "b"]]
    only = w._columns[0]
    n = len(w._columns)
    w._remove_column(only) if n == 1 else None
    assert w.grid() == [["a", "b"]]  # single remaining column: ✕ is a no-op


def test_corner_hotzone_mapping():
    r = QRect(0, 0, 100, 40)
    assert _corner_at(QPoint(3, 3), r) == "upper left"
    assert _corner_at(QPoint(97, 3), r) == "upper right"
    assert _corner_at(QPoint(3, 37), r) == "lower left"
    assert _corner_at(QPoint(97, 37), r) == "lower right"
    assert _corner_at(QPoint(50, 20), r) is None


def test_corner_click_emits_scale_bar_pick_and_marks_tile():
    w = _arr([["a"]])
    picks = []
    w.scaleBarPicked.connect(lambda pid, loc: picks.append((pid, loc)))
    w._on_corner_clicked("a", "lower left")
    assert picks == [("a", "lower left")]
    assert w._scale_bar_panel == "a" and w._scale_bar_loc == "lower left"


def test_schematic_strips_follow_mode_and_pos():
    w = _arr([["a"], ["b"]])
    w.set_bar_schematic("united", "right")
    assert not w._right_strip.isHidden() and w._bottom_strip.isHidden()
    w.set_bar_schematic("united", "bottom")
    assert w._right_strip.isHidden() and not w._bottom_strip.isHidden()
    w.set_bar_schematic("per-panel", "right", {frozenset(["a"])})
    assert w._right_strip.isHidden() and w._bottom_strip.isHidden()
    assert not w._columns[0].flag_strip.isHidden()  # flagged column strip
    assert w._columns[1].flag_strip.isHidden()
```

- [ ] Run `python3 -m pytest -q tests/test_gui_layout_arranger.py` — expect `ModuleNotFoundError: No module named 'gui.widgets.layout_arranger'`.
- [ ] Create `gui/widgets/layout_arranger.py`:

```python
"""Drag-grid layout arranger for the figure builder.

A horizontal strip of columns over a GridModel (``dfxm.compose.gridmap``):
each column is a drag-enabled tile list (internal move + drag between
columns), with ◀/▶ reorder, ✕ merge-remove, and an "+ Add column" button.
Pure view over the grid — no recipe knowledge lives here. The bar schematic
(quantity chips, united/group colorbar strips, scale-bar corner dot) is
schematic only; the real preview stays the source of truth."""

from __future__ import annotations

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QStyle,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)

GROUP_COLORS: dict[str | None, str] = {
    "mosa_com": "#7b6ff0",
    "mosa_fwhm": "#2fa4a9",
    "strain": "#d1495b",
    "raw": "#8d99ae",
    "trace": "#e9a03b",
    None: "#c7c7c7",
}
_CORNER_PX = 10  # corner hot-zone size for scale-bar picking


def _corner_at(pos, rect) -> str | None:
    """Map *pos* inside *rect* to a scale-bar corner name, or None off-corner."""
    near_l = pos.x() - rect.left() <= _CORNER_PX
    near_r = rect.right() - pos.x() <= _CORNER_PX
    near_t = pos.y() - rect.top() <= _CORNER_PX
    near_b = rect.bottom() - pos.y() <= _CORNER_PX
    if near_t and near_l:
        return "upper left"
    if near_t and near_r:
        return "upper right"
    if near_b and near_l:
        return "lower left"
    if near_b and near_r:
        return "lower right"
    return None


class _TileDelegate(QStyledItemDelegate):
    """Paints a tile: group chip + title + optional scale-bar corner dot."""

    def __init__(self, arranger):
        super().__init__(arranger)
        self._arranger = arranger

    def sizeHint(self, option, index):  # noqa: N802 — Qt override
        return QSize(150, 34)

    def paint(self, painter: QPainter, option, index):  # noqa: N802 — Qt override
        pid = index.data(Qt.ItemDataRole.UserRole)
        info = self._arranger._tile_info.get(pid, {})
        r = option.rect.adjusted(2, 2, -2, -2)
        painter.save()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)
        painter.setPen(QColor("#666666"))
        painter.setBrush(QColor("#dce8e4") if selected else QColor("#f4f4f4"))
        painter.drawRoundedRect(r, 4, 4)
        chip = QRect(r.left() + 4, r.top() + 4, 10, r.height() - 8)
        painter.fillRect(chip, QColor(GROUP_COLORS.get(info.get("group"), GROUP_COLORS[None])))
        painter.setPen(QColor("#222222"))
        text = info.get("title") or pid or ""
        painter.drawText(
            r.adjusted(20, 0, -4, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            painter.fontMetrics().elidedText(text, Qt.TextElideMode.ElideRight, r.width() - 26),
        )
        if pid and pid == self._arranger._scale_bar_panel:
            loc = self._arranger._scale_bar_loc
            cx = r.right() - 7 if "right" in loc else r.left() + 20
            cy = r.top() + 7 if "upper" in loc else r.bottom() - 7
            painter.setBrush(QColor("#009682"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QRect(cx - 3, cy - 3, 6, 6))
        painter.restore()


class _TileList(QListWidget):
    """One column's tile list; drags move tiles within and between columns."""

    def __init__(self, arranger):
        super().__init__()
        self._arranger = arranger
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setItemDelegate(_TileDelegate(arranger))
        self.model().rowsInserted.connect(arranger._emit_grid_changed)
        self.model().rowsRemoved.connect(arranger._emit_grid_changed)

    def mousePressEvent(self, event):  # noqa: N802 — Qt override
        item = self.itemAt(event.position().toPoint())
        if item is not None:
            loc = _corner_at(event.position().toPoint(), self.visualItemRect(item))
            if loc is not None:
                self._arranger._on_corner_clicked(item.data(Qt.ItemDataRole.UserRole), loc)
                return
        super().mousePressEvent(event)


class _ArrangerColumn(QWidget):
    """Header (◀ ▶ … ✕) + tile list + group-mode schematic strip."""

    def __init__(self, arranger):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        head = QHBoxLayout()
        self.left_btn = QPushButton("◀")
        self.right_btn = QPushButton("▶")
        self.close_btn = QPushButton("✕")
        for b in (self.left_btn, self.right_btn, self.close_btn):
            b.setFixedWidth(28)
        head.addWidget(self.left_btn)
        head.addWidget(self.right_btn)
        head.addStretch(1)
        head.addWidget(self.close_btn)
        lay.addLayout(head)
        self.list = _TileList(arranger)
        lay.addWidget(self.list, 1)
        self.flag_strip = QLabel("")
        self.flag_strip.setFixedHeight(6)
        self.flag_strip.setVisible(False)
        lay.addWidget(self.flag_strip)
        self.left_btn.clicked.connect(lambda: arranger._move_column(self, -1))
        self.right_btn.clicked.connect(lambda: arranger._move_column(self, +1))
        self.close_btn.clicked.connect(lambda: arranger._remove_column(self))


class LayoutArranger(QWidget):
    gridChanged = Signal()
    scaleBarPicked = Signal(str, str)  # (panel_id, corner loc)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tile_info: dict[str, dict] = {}
        self._columns: list[_ArrangerColumn] = []
        self._loading = False
        self._cbar_mode = "per-panel"
        self._cbar_pos = "right"
        self._flagged_sets: set[frozenset] = set()
        self._scale_bar_panel: str | None = None
        self._scale_bar_loc = "lower right"

        outer = QVBoxLayout(self)
        strip_row = QHBoxLayout()
        self._strip_host = QHBoxLayout()
        strip_row.addLayout(self._strip_host, 1)
        self._add_btn = QPushButton("+ Add column")
        self._add_btn.clicked.connect(self._on_add_column)
        strip_row.addWidget(self._add_btn)
        self._right_strip = QLabel("")
        self._right_strip.setFixedWidth(8)
        self._right_strip.setVisible(False)
        strip_row.addWidget(self._right_strip)
        outer.addLayout(strip_row, 1)
        self._bottom_strip = QLabel("")
        self._bottom_strip.setFixedHeight(8)
        self._bottom_strip.setVisible(False)
        outer.addWidget(self._bottom_strip)

    # -- grid API --------------------------------------------------------------
    def set_grid(self, grid, tile_info_by_id) -> None:
        self._loading = True
        try:
            for col in self._columns:
                col.setParent(None)
                col.deleteLater()
            self._columns = []
            self._tile_info = dict(tile_info_by_id)
            for column in grid:
                col = self._new_column()
                for pid in column:
                    col.list.addItem(self._make_item(pid))
        finally:
            self._loading = False
        self._refresh_schematic()

    def grid(self) -> list[list[str]]:
        return [ids for col in self._columns if (ids := self._column_ids(col))]

    def _make_item(self, pid: str) -> QListWidgetItem:
        item = QListWidgetItem(self._tile_info.get(pid, {}).get("title") or pid)
        item.setData(Qt.ItemDataRole.UserRole, pid)
        item.setFlags(
            Qt.ItemFlag.ItemIsEnabled
            | Qt.ItemFlag.ItemIsSelectable
            | Qt.ItemFlag.ItemIsDragEnabled
        )
        return item

    def _new_column(self) -> _ArrangerColumn:
        col = _ArrangerColumn(self)
        self._columns.append(col)
        self._strip_host.addWidget(col)
        return col

    def _column_ids(self, col) -> list[str]:
        return [col.list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(col.list.count())]

    # -- column ops ------------------------------------------------------------
    def _on_add_column(self) -> None:
        self._new_column()
        self._emit_grid_changed()

    def _move_column(self, col, delta: int) -> None:
        i = self._columns.index(col)
        j = i + delta
        if j < 0 or j >= len(self._columns):
            return
        self._columns[i], self._columns[j] = self._columns[j], self._columns[i]
        for c in self._columns:
            self._strip_host.removeWidget(c)
        for c in self._columns:
            self._strip_host.addWidget(c)
        self._emit_grid_changed()

    def _remove_column(self, col) -> None:
        if len(self._columns) <= 1:
            return  # never remove the last column
        i = self._columns.index(col)
        target = self._columns[i - 1] if i > 0 else self._columns[1]
        self._loading = True
        try:
            while col.list.count():
                item = col.list.takeItem(0)
                target.list.addItem(self._make_item(item.data(Qt.ItemDataRole.UserRole)))
            self._columns.remove(col)
            col.setParent(None)
            col.deleteLater()
        finally:
            self._loading = False
        self._emit_grid_changed()

    def _emit_grid_changed(self, *_args) -> None:
        if self._loading:
            return
        self._refresh_schematic()
        self.gridChanged.emit()

    # -- bar schematic ---------------------------------------------------------
    def set_bar_schematic(
        self, colorbar_mode: str, colorbar_pos: str, flagged_member_sets=None
    ) -> None:
        self._cbar_mode = colorbar_mode
        self._cbar_pos = colorbar_pos
        self._flagged_sets = set(flagged_member_sets or ())
        self._refresh_schematic()

    def set_scale_bar(self, panel_id: str | None, loc: str) -> None:
        self._scale_bar_panel = panel_id
        self._scale_bar_loc = loc or "lower right"
        for col in self._columns:
            col.list.viewport().update()

    def _on_corner_clicked(self, pid: str, loc: str) -> None:
        self.set_scale_bar(pid, loc)
        self.scaleBarPicked.emit(pid, loc)

    def _strip_css(self, horizontal: bool) -> str:
        groups: list[str] = []
        for col in self._columns:
            for pid in self._column_ids(col):
                g = self._tile_info.get(pid, {}).get("group")
                if g is not None and g != "trace" and g not in groups:
                    groups.append(g)
        if not groups:
            return "background: #c7c7c7;"
        n = len(groups)
        stops = []
        for i, g in enumerate(groups):
            c = GROUP_COLORS[g]
            stops.append(f"stop:{i / n:.3f} {c}, stop:{(i + 1) / n - 0.001:.3f} {c}")
        coords = "x1:0, y1:0, x2:1, y2:0" if horizontal else "x1:0, y1:0, x2:0, y2:1"
        return f"background: qlineargradient({coords}, {', '.join(stops)});"

    def _refresh_schematic(self) -> None:
        united = self._cbar_mode == "united"
        self._right_strip.setVisible(united and self._cbar_pos == "right")
        self._bottom_strip.setVisible(united and self._cbar_pos == "bottom")
        if united:
            css = self._strip_css(horizontal=self._cbar_pos == "bottom")
            self._right_strip.setStyleSheet(css)
            self._bottom_strip.setStyleSheet(css)
        for col in self._columns:
            members = frozenset(self._column_ids(col))
            flagged = (not united) and bool(members) and members in self._flagged_sets
            col.flag_strip.setVisible(bool(flagged))
            if flagged:
                col.flag_strip.setStyleSheet("background: #009682;")
```

- [ ] Run `python3 -m pytest -q tests/test_gui_layout_arranger.py` — green; `python3 -m pytest -q`; `ruff check .`.
- [ ] Docs: `docs/Codebase.md` `gui/widgets/` table — add a `layout_arranger.py` row documenting `LayoutArranger` (signals, `set_grid`/`grid`/`set_bar_schematic`/`set_scale_bar`, column ops semantics — ✕ merges into the neighbour, never removes the last column; `grid()` drops empty columns), `GROUP_COLORS`, `_corner_at`, and that the schematic is display-only.
- [ ] Commit: `git add gui/widgets/layout_arranger.py tests/test_gui_layout_arranger.py docs/Codebase.md && git commit -m "feat(gui): LayoutArranger drag-grid widget with bar schematic (docs synced)"`

---

## Task 8 — two-step Add-panels dialog + `add_panels(panels, layout=None)`

**Files:**
- Modify: `dfxm/compose/gridmap.py` — add `panel_group_hint` at module end.
- Modify: `gui/widgets/panel_picker.py` — `__init__` (restructure into a `QStackedWidget`), new `_goto_page`/`_on_next`, `accept`.
- Modify: `gui/figure_builder.py` — `add_panels` (anchor: `def add_panels(self, panels: list[PanelDef]) -> None:`), `_on_add_panels` (anchor: `self.add_panels(dlg.selected_panels)`).
- Modify: `docs/Usage.md` (Add panels… flow) + `docs/Codebase.md` (`gridmap.py`, `panel_picker.py`, `figure_builder.py`).
- Test: `tests/test_gui_figure_builder.py` (append), `tests/test_compose_gridmap.py` (append).

**Interfaces:**
- Produces:
  - `dfxm.compose.gridmap.panel_group_hint(panel) -> str | None` — cheap, file-free chip-group guess (schematic only, never used for rendering).
  - `AddPanelDialog(defaults, schematic=("per-panel", "right"), parent=None)` — page 0 = existing picker, page 1 = `LayoutArranger` seeded one tile per column; attrs `selected_panels: list[PanelDef]`, `selected_layout: Row | None` (a `grid_to_layout` fragment; `None` when OK'd from page 0), `scale_bar_pick: tuple[str, str] | None`; internals `_stack`, `_staged`, `_arranger`, `_back_btn`, `_next_btn`, `_goto_page(i)`, `_on_next()`. Step 1 error handling (`no such file` / `cannot read`) unchanged.
  - `FigureBuilderWindow.add_panels(panels, layout=None) -> dict[str, str]` — returns the `{old_id: new_id}` rename map; with a `layout` fragment it rewrites the fragment's `PanelRef`s through the renames and appends the fragment as ONE child of the current container.

**Steps:**

- [ ] Write failing tests. Append to `tests/test_compose_gridmap.py`:

```python
def test_panel_group_hint_covers_kinds():
    from dfxm.compose.gridmap import panel_group_hint

    def p(kind, sel):
        return PanelDef("x", PanelSource("/x.h5", kind, sel))

    assert panel_group_hint(p("map_layer", {"stage": "strain", "z": 0})) == "strain"
    assert panel_group_hint(p("map_layer", {"stage": "rocking", "dataset": "d"})) == "raw"
    assert panel_group_hint(p("map_layer", {"stage": "mosaicity", "dataset": "/chi/FWHM"})) == (
        "mosa_fwhm"
    )
    assert panel_group_hint(p("map_layer", {"stage": "mosaicity", "dataset": "/chi/Center"})) == (
        "mosa_com"
    )
    assert panel_group_hint(p("profiles_trace", {"job": {}, "field": "strain"})) == "trace"
    assert panel_group_hint(p("slice_plane", {"volume_id": "raw_mosa_sum"})) == "raw"
    assert panel_group_hint(p("slice_plane", {"volume_id": "strain"})) == "strain"
    assert panel_group_hint(p("slice_plane", {"volume_id": "mosa_com_chi"})) == "mosa_com"
    assert panel_group_hint(p("profiles_ref", {"job": {}, "field": None})) is None
```

Append to `tests/test_gui_figure_builder.py`:

```python
# -- two-step Add panels dialog + layout fragments ----------------------------
def _slices_dialog(tmp_path):
    import h5py
    import numpy as np

    from gui.widgets.panel_picker import AddPanelDialog

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((2, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0, 1, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 1, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))
    dlg = AddPanelDialog({"slices": {"h5": str(h5), "sx": 0.5, "sy": 0.5, "jobs": []}})
    dlg._stage.setCurrentText("slices")
    dlg._reload()
    dlg._check_all()
    return dlg


def test_add_dialog_two_step_returns_fragment(tmp_path):
    from dfxm.compose.recipe import Col

    dlg = _slices_dialog(tmp_path)
    dlg._on_next()
    assert dlg._stack.currentIndex() == 1
    assert dlg._arranger.grid() == [[p.id] for p in dlg._staged]  # one-row seed
    item = dlg._arranger._columns[1].list.takeItem(0)  # stack plane 1 under plane 0
    dlg._arranger._columns[0].list.addItem(item)
    dlg.accept()
    assert len(dlg.selected_panels) == 2
    lay = dlg.selected_layout
    assert isinstance(lay, Row) and isinstance(lay.children[0], Col)
    assert [r.panel_id for r in lay.children[0].children] == [p.id for p in dlg.selected_panels]


def test_add_dialog_ok_from_step1_keeps_flat_behaviour(tmp_path):
    dlg = _slices_dialog(tmp_path)
    dlg.accept()
    assert len(dlg.selected_panels) == 2
    assert dlg.selected_layout is None


def test_add_dialog_next_with_nothing_checked_stays_on_step1(tmp_path):
    dlg = _slices_dialog(tmp_path)
    dlg._uncheck_all()
    dlg._on_next()
    assert dlg._stack.currentIndex() == 0
    assert "check" in dlg._status.text()


def test_window_add_panels_with_fragment_and_id_collision():
    from dfxm.compose.recipe import Col

    w = _win()
    w.add_panels([_panel("slices_0")])
    frag = Row([Col([PanelRef("slices_0"), PanelRef("slices_1")])])
    renames = w.add_panels([_panel("slices_0"), _panel("slices_1")], layout=frag)
    assert renames == {"slices_0": "slices_0_1"}
    root = w.recipe().layout
    assert root.children[1] is frag  # fragment appended as ONE child
    assert [r.panel_id for r in frag.children[0].children] == ["slices_0_1", "slices_1"]
    assert [p.id for p in w.recipe().panels] == ["slices_0", "slices_0_1", "slices_1"]
```

- [ ] Run `python3 -m pytest -q tests/test_compose_gridmap.py tests/test_gui_figure_builder.py -k "hint or two_step or step1 or nothing_checked or fragment"` — expect `ImportError: cannot import name 'panel_group_hint'` and `AttributeError: 'AddPanelDialog' object has no attribute '_on_next'` / `TypeError: add_panels() got an unexpected keyword argument 'layout'`.
- [ ] Implement `panel_group_hint` at the end of `dfxm/compose/gridmap.py`:

```python
def panel_group_hint(panel) -> str | None:
    """Cheap, file-free quantity-group guess for schematic tile chips ONLY —
    rendering always uses the loaded ``PanelData.group``, never this."""
    src = panel.source
    if src.kind == "profiles_trace":
        return "trace"
    if src.kind == "map_layer":
        stage = src.selector.get("stage")
        if stage == "strain":
            return "strain"
        if stage == "rocking":
            return "raw"
        if stage == "mosaicity":
            return "mosa_fwhm" if "FWHM" in str(src.selector.get("dataset", "")) else "mosa_com"
        return None
    key = "volume_id" if src.kind == "slice_plane" else "field"
    v = str(src.selector.get(key) or "").lower()
    if "raw" in v:
        return "raw"
    if "fwhm" in v:
        return "mosa_fwhm"
    if "strain" in v:
        return "strain"
    if "mosa" in v or "com" in v or "chi" in v or "mu" in v:
        return "mosa_com"
    return None
```

- [ ] Implement the two-step dialog in `gui/widgets/panel_picker.py`. Add imports: `QStackedWidget`, `QWidget` to the PySide6 import list; `from dfxm.compose.gridmap import grid_to_layout, panel_group_hint`; `from .layout_arranger import LayoutArranger`. In `__init__`: add state after `self.selected_panels = []`:

```python
        self.selected_layout = None
        self.scale_bar_pick: tuple[str, str] | None = None
        self._staged: list[PanelDef] = []
        self._schematic = schematic
```

(and extend the signature to `def __init__(self, defaults: dict[str, dict], schematic=("per-panel", "right"), parent=None) -> None:`). Replace the `buttons = QDialogButtonBox(...)` block and the final `layout = QVBoxLayout(self)` block with:

```python
        self._page0 = QWidget()
        layout0 = QVBoxLayout(self._page0)
        layout0.addLayout(top_row)
        layout0.addLayout(check_row)
        layout0.addWidget(self._tree, 1)
        layout0.addWidget(self._status)

        self._page1 = QWidget()
        layout1 = QVBoxLayout(self._page1)
        layout1.addWidget(QLabel("Drag the new panels into rows/columns (optional):"))
        self._arranger = LayoutArranger()
        self._arranger.scaleBarPicked.connect(self._on_scale_bar_picked)
        layout1.addWidget(self._arranger, 1)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._page0)
        self._stack.addWidget(self._page1)

        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(lambda: self._goto_page(0))
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(self._on_next)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._back_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self._next_btn)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self._stack, 1)
        layout.addLayout(btn_row)
        self._goto_page(0)
```

(the `QDialogButtonBox` import can then be dropped). Add the new methods before `accept`:

```python
    # -- two-step flow ---------------------------------------------------------
    def _goto_page(self, i: int) -> None:
        self._stack.setCurrentIndex(i)
        self._back_btn.setEnabled(i == 1)
        self._next_btn.setEnabled(i == 0)

    def _on_next(self) -> None:
        self._staged = self._build_panels()
        if not self._staged:
            self._status.setText("check at least one item first")
            return
        info = {
            p.id: {"title": p.title or p.id, "group": panel_group_hint(p)} for p in self._staged
        }
        self._arranger.set_grid([[p.id] for p in self._staged], info)
        self._arranger.set_bar_schematic(self._schematic[0], self._schematic[1])
        self._goto_page(1)

    def _on_scale_bar_picked(self, pid: str, loc: str) -> None:
        self.scale_bar_pick = (pid, loc)
```

Replace `accept` with:

```python
    def accept(self) -> None:
        if self._stack.currentIndex() == 1:
            self.selected_panels = list(self._staged)
            self.selected_layout = grid_to_layout(self._arranger.grid())
        else:
            self.selected_panels = self._build_panels()
            self.selected_layout = None
        super().accept()
```

- [ ] Implement in `gui/figure_builder.py`. Replace `add_panels` with:

```python
    def add_panels(self, panels: list[PanelDef], layout=None) -> dict[str, str]:
        """Append *panels* (ids uniquified) and either flat PanelRefs (default)
        or *layout* — a gridmap fragment appended as ONE child of the current
        container, its refs rewritten through the same renames. Returns the
        ``{old_id: new_id}`` rename map."""
        container = self._current_container()
        existing_ids = {p.id for p in self._recipe.panels}
        renames: dict[str, str] = {}
        stored: list[PanelDef] = []
        for p in panels:
            pid = p.id
            if pid in existing_ids:
                n = 1
                while f"{pid}_{n}" in existing_ids:
                    n += 1
                renames[pid] = f"{pid}_{n}"
                p = dc_replace(p, id=f"{pid}_{n}")
            existing_ids.add(p.id)
            stored.append(p)
        self._recipe.panels.extend(stored)
        if layout is None:
            for p in stored:
                container.children.append(PanelRef(p.id))
        else:
            for leaf in iter_leaves(layout):
                if isinstance(leaf, PanelRef) and leaf.panel_id in renames:
                    leaf.panel_id = renames[leaf.panel_id]
            container.children.append(layout)
        self._after_mutation()
        return renames
```

and in `_on_add_panels` replace `self.add_panels(dlg.selected_panels)` with `self.add_panels(dlg.selected_panels, dlg.selected_layout)`.

- [ ] Run `python3 -m pytest -q tests/test_gui_figure_builder.py tests/test_compose_gridmap.py` then `python3 -m pytest -q` — green; `ruff check .`.
- [ ] Docs: `docs/Usage.md` Figure builder — rewrite the Add panels… description as the two-step flow (step 1 unchanged picker; **Next** stages the checked panels on a drag grid seeded one panel per column; drag tiles between columns, ◀/▶/✕/+ Add column; **OK from step 1 skips arrangement** and appends flat as before; OK from step 2 inserts the arranged rows/columns as one block). `docs/Codebase.md` — `gridmap.py` bullet gains `panel_group_hint`; `panel_picker.py` row rewritten for the `QStackedWidget` two-step structure and new attrs; `figure_builder.py` row documents `add_panels(panels, layout=None) -> dict[str, str]`.
- [ ] Commit: `git add dfxm/compose/gridmap.py gui/widgets/panel_picker.py gui/figure_builder.py tests/test_compose_gridmap.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md && git commit -m "feat(gui): two-step Add-panels dialog with drag-grid arrangement (docs synced)"`

---

## Task 9 — Arrange… dialog with flatten warning + Col-flag preservation

**Files:**
- Modify: `gui/widgets/layout_arranger.py` — add `QDialog` to the imports, then `_collect_col_flags` + `ArrangeDialog` at module end.
- Modify: `gui/figure_builder.py` — `_build_left_pane` (anchor: `add_btn = QPushButton("Add panels…")`), new `_on_arrange` + `apply_arranged_layout` (place next to `_on_add_panels`).
- Modify: `docs/Usage.md` + `docs/Codebase.md`.
- Test: `tests/test_gui_layout_arranger.py` + `tests/test_gui_figure_builder.py` (append).

**Interfaces:**
- Produces:
  - `_collect_col_flags(layout) -> dict[frozenset, dict]` — member-id set → `{"pinned_width_cm", "group_label", "shared_x", "shared_colorbar", "shared_clim"}` for every `Col` in the tree.
  - `ArrangeDialog(recipe, style, parent=None)` — seeds from `layout_to_grid(recipe.layout, recipe.panel_by_id())`; on `None` seeds `[flatten_panel_ids(recipe.layout)]` (one column) and shows the persistent `self._warning`; shows the always-on purge note; schematic from `recipe.compose` + `style.scale_bar_loc`; attrs `result_layout: Row | None`, `scale_bar_pick: tuple[str, str] | None`, `_arranger`, `_warning`; `_on_apply()` rebuilds via `grid_to_layout`, re-applies Col flags by unchanged member-id set, accepts.
  - `FigureBuilderWindow.apply_arranged_layout(new_root, scale_bar_pick=None) -> None` — replaces `recipe.layout`, purges orphans, applies an optional scale-bar pick (`compose.scale_bar_mode="one-panel"`, `compose.scale_bar_panel`, `style.scale_bar_loc` + style sync), refreshes compose widgets, `_after_mutation()`. New left-pane button `Arrange…` → `_on_arrange()`.

**Steps:**

- [ ] Write failing tests. Append to `tests/test_gui_layout_arranger.py` (add `from PySide6.QtWidgets import QApplication` is already there; add recipe imports inside the tests):

```python
def _fb_win():
    from dfxm.common.plotting import PlotStyle
    from gui.figure_builder import FigureBuilderWindow

    return FigureBuilderWindow(lambda: {}, PlotStyle(scale_um_per_cm=10.0))


def _mk_panel(pid):
    from dfxm.compose.recipe import PanelDef, PanelSource

    return PanelDef(pid, PanelSource("/x.h5", "map_layer", {"stage": "strain", "z": 0}))


def test_arrange_dialog_clean_grid_preserves_col_flags():
    from dfxm.compose.recipe import Col, PanelRef

    from gui.widgets.layout_arranger import ArrangeDialog

    w = _fb_win()
    w.add_panels([_mk_panel("a"), _mk_panel("b"), _mk_panel("c")])
    root = w.recipe().layout
    root.children = [
        PanelRef("a"),
        Col(
            [PanelRef("b"), PanelRef("c")],
            pinned_width_cm=4.0,
            group_label="G",
            shared_x=True,
            shared_colorbar=True,
            shared_clim=(-1.0, 1.0),
        ),
    ]
    w._rebuild_tree()
    dlg = ArrangeDialog(w.recipe(), w._style)
    assert dlg._warning.isHidden()
    assert dlg._arranger.grid() == [["a"], ["b", "c"]]
    dlg._on_apply()
    col = dlg.result_layout.children[1]
    assert isinstance(col, Col)
    assert col.shared_x and col.shared_colorbar and col.group_label == "G"
    assert col.pinned_width_cm == 4.0 and col.shared_clim == (-1.0, 1.0)
    w._debounce.stop()


def test_arrange_dialog_flatten_path_warns_and_seeds_one_column():
    from dfxm.compose.recipe import Col, Spacer

    from gui.widgets.layout_arranger import ArrangeDialog

    w = _fb_win()
    w.add_panels([_mk_panel("a"), _mk_panel("b")])
    w.recipe().layout.children.append(Spacer(1.0, 1.0))  # unmappable
    dlg = ArrangeDialog(w.recipe(), w._style)
    assert not dlg._warning.isHidden()
    assert dlg._arranger.grid() == [["a", "b"]]
    dlg._on_apply()
    assert isinstance(dlg.result_layout.children[0], Col)  # one two-tile column
    w._debounce.stop()
```

Append to `tests/test_gui_figure_builder.py`:

```python
def test_apply_arranged_layout_replaces_purges_and_applies_pick():
    w = _win()
    w.add_panels([_panel("a"), _panel("b")])
    w.apply_arranged_layout(Row([PanelRef("a")]), scale_bar_pick=("a", "upper right"))
    assert [p.id for p in w.recipe().panels] == ["a"]  # "b" purged with the grid
    assert w.recipe().compose.scale_bar_mode == "one-panel"
    assert w.recipe().compose.scale_bar_panel == "a"
    assert w._style.scale_bar_loc == "upper right"
    assert w.recipe().style["scale_bar_loc"] == "upper right"
    assert w.is_dirty()


def test_arrange_button_present():
    w = _win()
    assert w._arrange_btn.text() == "Arrange…"
```

- [ ] Run `python3 -m pytest -q tests/test_gui_layout_arranger.py tests/test_gui_figure_builder.py -k "arrange or arranged"` — expect `ImportError: cannot import name 'ArrangeDialog'` and `AttributeError: ... has no attribute 'apply_arranged_layout'`.
- [ ] Implement in `gui/widgets/layout_arranger.py` (append; add `QDialog` to the widget imports):

```python
def _collect_col_flags(layout) -> dict[frozenset, dict]:
    """member-id set -> the Col's group/shared settings, over the whole tree."""
    from dfxm.compose.recipe import Col, PanelRef, Row

    out: dict[frozenset, dict] = {}

    def walk(node):
        if isinstance(node, Col):
            ids = frozenset(c.panel_id for c in node.children if isinstance(c, PanelRef))
            if ids:
                out[ids] = {
                    "pinned_width_cm": node.pinned_width_cm,
                    "group_label": node.group_label,
                    "shared_x": node.shared_x,
                    "shared_colorbar": node.shared_colorbar,
                    "shared_clim": node.shared_clim,
                }
        if isinstance(node, (Row, Col)):
            for c in node.children:
                walk(c)

    walk(layout)
    return out


class ArrangeDialog(QDialog):
    """Arrange the recipe's panels on a drag grid; Apply yields a new root Row.

    Clean-grid case: a Col whose member-id set is unchanged keeps its
    group/shared flags. Unmappable layouts seed from flatten_panel_ids (one
    column) behind a persistent warning — applying then drops spacers, text
    cells and nested groups."""

    def __init__(self, recipe, style, parent=None) -> None:
        super().__init__(parent)
        from dfxm.compose.gridmap import flatten_panel_ids, layout_to_grid, panel_group_hint

        self.setWindowTitle("Arrange panels")
        self.result_layout = None
        self.scale_bar_pick: tuple[str, str] | None = None
        self._old_flags = _collect_col_flags(recipe.layout)

        self._arranger = LayoutArranger()
        self._warning = QLabel(
            "⚠ This layout is not a plain grid — applying will rebuild it as one: "
            "spacers, text cells and nested groups will be dropped."
        )
        self._warning.setWordWrap(True)
        grid = layout_to_grid(recipe.layout, recipe.panel_by_id())
        if grid is None:
            grid = [flatten_panel_ids(recipe.layout)]
        else:
            self._warning.setVisible(False)
        info = {
            p.id: {"title": p.title or p.id, "group": panel_group_hint(p)}
            for p in recipe.panels
        }
        self._arranger.set_grid(grid, info)
        self._arranger.set_bar_schematic(
            recipe.compose.colorbar_mode,
            recipe.compose.colorbar_pos,
            {ids for ids, f in self._old_flags.items() if f["shared_colorbar"]},
        )
        self._arranger.set_scale_bar(
            recipe.compose.scale_bar_panel
            if recipe.compose.scale_bar_mode == "one-panel"
            else None,
            style.scale_bar_loc,
        )
        self._arranger.scaleBarPicked.connect(self._on_scale_bar_picked)

        purge_note = QLabel("Panels removed from the grid are removed from the recipe on Apply.")
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self._on_apply)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns = QHBoxLayout()
        btns.addStretch(1)
        btns.addWidget(apply_btn)
        btns.addWidget(cancel_btn)

        lay = QVBoxLayout(self)
        lay.addWidget(self._warning)
        lay.addWidget(self._arranger, 1)
        lay.addWidget(purge_note)
        lay.addLayout(btns)
        self.resize(720, 420)

    def _on_scale_bar_picked(self, pid: str, loc: str) -> None:
        self.scale_bar_pick = (pid, loc)

    def _on_apply(self) -> None:
        from dfxm.compose.gridmap import grid_to_layout
        from dfxm.compose.recipe import Col

        new_root = grid_to_layout(self._arranger.grid())
        for child in new_root.children:
            if isinstance(child, Col):
                flags = self._old_flags.get(frozenset(c.panel_id for c in child.children))
                if flags:
                    child.pinned_width_cm = flags["pinned_width_cm"]
                    child.group_label = flags["group_label"]
                    child.shared_x = flags["shared_x"]
                    child.shared_colorbar = flags["shared_colorbar"]
                    child.shared_clim = flags["shared_clim"]
        self.result_layout = new_root
        self.accept()
```

- [ ] Implement in `gui/figure_builder.py`. In `_build_left_pane`, right after `layout.addWidget(add_btn)`:

```python
        self._arrange_btn = QPushButton("Arrange…")
        self._arrange_btn.clicked.connect(self._on_arrange)
        layout.addWidget(self._arrange_btn)
```

Next to `_on_add_panels` add:

```python
    def _on_arrange(self) -> None:
        from .widgets.layout_arranger import ArrangeDialog

        dlg = ArrangeDialog(self._recipe, self._style, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_layout is not None:
            self.apply_arranged_layout(dlg.result_layout, dlg.scale_bar_pick)

    def apply_arranged_layout(self, new_root, scale_bar_pick=None) -> None:
        """Replace the layout with an arranged grid, purge orphans, and apply
        an optional (panel_id, corner) scale-bar pick from the arranger."""
        self._recipe.layout = new_root
        self._purge_orphaned_panels()
        if scale_bar_pick is not None:
            pid, loc = scale_bar_pick
            self._recipe.compose.scale_bar_mode = "one-panel"
            self._recipe.compose.scale_bar_panel = pid
            self._style.scale_bar_loc = loc
            self._controls.sync_from_style()
            self._recipe.style = asdict(self._style)
        self._load_compose_into_widgets()
        self._after_mutation()
```

- [ ] Run `python3 -m pytest -q tests/test_gui_layout_arranger.py tests/test_gui_figure_builder.py` then `python3 -m pytest -q` — green; `ruff check .`.
- [ ] Docs: `docs/Usage.md` Figure builder — new **Arrange…** paragraph (opens the drag grid seeded from the current layout; non-grid layouts show the flatten warning; Apply rebuilds the layout as a plain grid, keeping a column's group/shared settings when its member set didn't change; panels removed from the grid are removed from the recipe like Delete; clicking a tile corner sets the one-panel scale-bar target + corner). `docs/Codebase.md` — `layout_arranger.py` row gains `_collect_col_flags` + `ArrangeDialog`; `figure_builder.py` row gains `_on_arrange`/`apply_arranged_layout`/`_arrange_btn`.
- [ ] Commit: `git add gui/widgets/layout_arranger.py gui/figure_builder.py tests/test_gui_layout_arranger.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md && git commit -m "feat(gui): Arrange… drag-grid dialog with flatten warning + Col-flag preservation (docs synced)"`

---

## Task 10 — compose-form Colourbars/Scale-bar controls + schematic settings wiring

**Files:**
- Modify: `dfxm/common/plotting.py` — add `SCALE_BAR_LOCS` next to `AXES_MODES` (anchor: `AXES_MODES: tuple[str, ...] = ("full", "no_frame", "none")`).
- Modify: `gui/widgets/export_dialog.py` — `_LOCS` (anchor: `_LOCS = ["lower right", "lower left", "upper right", "upper left"]`).
- Modify: `gui/figure_builder.py` — imports, `_build_compose_form`, `_load_compose_into_widgets`, `_on_compose_edited`, `_sync_style_to_recipe`, `_on_add_panels`; new `_on_scale_bar_loc_edited`.
- Modify: `docs/Usage.md` + `docs/Codebase.md`.
- Test: `tests/test_gui_figure_builder.py` (append).

**Interfaces:**
- Produces: `dfxm.common.plotting.SCALE_BAR_LOCS: tuple[str, ...] = ("lower right", "lower left", "upper right", "upper left")` (export_dialog's `_LOCS = list(SCALE_BAR_LOCS)`). Compose form: **Colourbars** heading with `_compose_cbar_mode` (Per panel / One per quantity → itemData `"per-panel"`/`"united"`) and `_compose_cbar_pos` (Right/Bottom → `"right"`/`"bottom"`, enabled only in united mode); **Scale bar** heading regrouping the existing mode + target-panel combos plus `_compose_scale_bar_loc` (over `SCALE_BAR_LOCS`) bound to `style.scale_bar_loc` — the SAME setting as the Style pane's Bar location, synced both ways. `_on_add_panels` passes `schematic=(compose.colorbar_mode, compose.colorbar_pos)` to `AddPanelDialog` and applies its `scale_bar_pick` through the rename map.

**Steps:**

- [ ] Write failing tests (append to `tests/test_gui_figure_builder.py`):

```python
# -- compose-form colourbar + scale-bar controls ------------------------------
def test_compose_colorbar_controls_write_fields_and_gate_pos():
    w = _win()
    assert w.recipe().compose.colorbar_mode == "per-panel"
    assert not w._compose_cbar_pos.isEnabled()
    w._compose_cbar_mode.setCurrentIndex(w._compose_cbar_mode.findData("united"))
    assert w.recipe().compose.colorbar_mode == "united"
    assert w._compose_cbar_pos.isEnabled()
    w._compose_cbar_pos.setCurrentIndex(w._compose_cbar_pos.findData("bottom"))
    assert w.recipe().compose.colorbar_pos == "bottom"
    w._compose_cbar_mode.setCurrentIndex(w._compose_cbar_mode.findData("per-panel"))
    assert not w._compose_cbar_pos.isEnabled()


def test_compose_colorbar_widgets_reload_from_recipe():
    w = _win()
    w.recipe().compose.colorbar_mode = "united"
    w.recipe().compose.colorbar_pos = "bottom"
    w._load_compose_into_widgets()
    assert w._compose_cbar_mode.currentData() == "united"
    assert w._compose_cbar_pos.currentData() == "bottom"
    assert w._compose_cbar_pos.isEnabled()


def test_scale_bar_corner_combo_is_style_scale_bar_loc_both_ways():
    w = _win()
    w._compose_scale_bar_loc.setCurrentText("upper left")
    assert w._style.scale_bar_loc == "upper left"
    assert w.recipe().style["scale_bar_loc"] == "upper left"
    assert w._controls._w_bar_loc.currentText() == "upper left"
    w._controls._w_bar_loc.setCurrentText("lower left")  # style-pane edit
    assert w._style.scale_bar_loc == "lower left"
    assert w._compose_scale_bar_loc.currentText() == "lower left"


def test_scale_bar_locs_is_canonical():
    from dfxm.common.plotting import SCALE_BAR_LOCS

    from gui.widgets.export_dialog import _LOCS

    assert list(SCALE_BAR_LOCS) == _LOCS == [
        "lower right",
        "lower left",
        "upper right",
        "upper left",
    ]
```

- [ ] Run `python3 -m pytest -q tests/test_gui_figure_builder.py -k "cbar or corner_combo or canonical or colorbar_controls or reload_from"` — expect `AttributeError: ... '_compose_cbar_mode'` and `ImportError: cannot import name 'SCALE_BAR_LOCS'`.
- [ ] Implement. `dfxm/common/plotting.py`, after the `AXES_MODES` line:

```python
# Scale-bar corner locations (canonical order — GUI combos derive from this).
SCALE_BAR_LOCS: tuple[str, ...] = ("lower right", "lower left", "upper right", "upper left")
```

`gui/widgets/export_dialog.py`: add `SCALE_BAR_LOCS` to the `from dfxm.common.plotting import ...` line and replace the `_LOCS = [...]` literal with `_LOCS = list(SCALE_BAR_LOCS)`. `gui/figure_builder.py`: extend the plotting import to `from dfxm.common.plotting import CMAP_CHOICES, SCALE_BAR_LOCS, PlotStyle, style_from_params` (let ruff order it). In `_build_compose_form`, after the padding row and before the existing scale-bar-mode row insert:

```python
        form.addRow(QLabel("<b>Colourbars</b>"))
        self._compose_cbar_mode = QComboBox()
        for text, value in (("Per panel", "per-panel"), ("One per quantity", "united")):
            self._compose_cbar_mode.addItem(text, value)
        self._compose_cbar_mode.currentIndexChanged.connect(self._on_compose_edited)
        form.addRow("Colourbar mode", self._compose_cbar_mode)

        self._compose_cbar_pos = QComboBox()
        for text, value in (("Right", "right"), ("Bottom", "bottom")):
            self._compose_cbar_pos.addItem(text, value)
        self._compose_cbar_pos.setEnabled(False)
        self._compose_cbar_pos.currentIndexChanged.connect(self._on_compose_edited)
        form.addRow("Colourbar position (united)", self._compose_cbar_pos)

        form.addRow(QLabel("<b>Scale bar</b>"))
```

(keeping the existing mode + panel combos directly under the new heading), then after the scale-bar-panel row:

```python
        self._compose_scale_bar_loc = QComboBox()
        self._compose_scale_bar_loc.addItems(list(SCALE_BAR_LOCS))
        self._compose_scale_bar_loc.setCurrentText(self._style.scale_bar_loc)
        self._compose_scale_bar_loc.currentTextChanged.connect(self._on_scale_bar_loc_edited)
        form.addRow("Corner", self._compose_scale_bar_loc)
```

Add next to `_on_compose_edited`:

```python
    def _on_scale_bar_loc_edited(self, loc: str) -> None:
        """The corner combo IS style.scale_bar_loc — one setting, two widgets."""
        if self._style.scale_bar_loc == loc:
            return
        self._style.scale_bar_loc = loc
        self._controls.sync_from_style()
        self._sync_style_to_recipe()
```

In `_on_compose_edited`, after `c.scale_bar_panel = ...` add:

```python
        c.colorbar_mode = self._compose_cbar_mode.currentData()
        c.colorbar_pos = self._compose_cbar_pos.currentData()
        self._compose_cbar_pos.setEnabled(c.colorbar_mode == "united")
```

In `_load_compose_into_widgets`: add `self._compose_cbar_mode`, `self._compose_cbar_pos`, `self._compose_scale_bar_loc` to the blocked-widgets tuple and, in the value-setting section, add:

```python
        self._compose_cbar_mode.setCurrentIndex(
            max(0, self._compose_cbar_mode.findData(c.colorbar_mode))
        )
        self._compose_cbar_pos.setCurrentIndex(
            max(0, self._compose_cbar_pos.findData(c.colorbar_pos))
        )
        self._compose_cbar_pos.setEnabled(c.colorbar_mode == "united")
        self._compose_scale_bar_loc.setCurrentText(self._style.scale_bar_loc)
```

In `_sync_style_to_recipe`, after `self._recipe.style = asdict(self._style)` add:

```python
        if hasattr(self, "_compose_scale_bar_loc"):
            self._compose_scale_bar_loc.blockSignals(True)
            self._compose_scale_bar_loc.setCurrentText(self._style.scale_bar_loc)
            self._compose_scale_bar_loc.blockSignals(False)
```

Replace `_on_add_panels` with:

```python
    def _on_add_panels(self) -> None:
        defaults = self._defaults_provider()
        c = self._recipe.compose
        dlg = AddPanelDialog(defaults, schematic=(c.colorbar_mode, c.colorbar_pos), parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            renames = self.add_panels(dlg.selected_panels, dlg.selected_layout)
            if dlg.scale_bar_pick is not None:
                pid, loc = dlg.scale_bar_pick
                self._recipe.compose.scale_bar_mode = "one-panel"
                self._recipe.compose.scale_bar_panel = renames.get(pid, pid)
                self._style.scale_bar_loc = loc
                self._controls.sync_from_style()
                self._sync_style_to_recipe()
                self._load_compose_into_widgets()
```

- [ ] Run `python3 -m pytest -q tests/test_gui_figure_builder.py` then `python3 -m pytest -q` — green; `ruff check .`.
- [ ] Docs: `docs/Usage.md` Figure builder right-pane *Compose* bullet — document the **Colourbars** controls (Per panel vs One per quantity; position right/bottom, united only; group flags ignored in united mode with a note; panel Colourbar=On still forces its own bar) and the regrouped **Scale bar** controls (mode, target panel by title, corner — explicitly "the same setting as the Style pane's Bar location", plus the arranger corner-click route). `docs/Codebase.md` — `plotting.py` bullet gains `SCALE_BAR_LOCS`; `export_dialog.py` row notes `_LOCS` now derives from it; `figure_builder.py` row documents the new widgets, `_on_scale_bar_loc_edited`, the two-way sync, and the schematic/pick pass-through in `_on_add_panels`.
- [ ] Commit: `git add dfxm/common/plotting.py gui/widgets/export_dialog.py gui/figure_builder.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md && git commit -m "feat(gui): compose-form united-colorbar + first-class scale-bar controls (docs synced)"`

---

## Task 11 — GUI smoke coverage + final verification

**Files:**
- Modify: `tests/gui_smoke.py` — append a `[41]` block after `[40]` (anchor: `print("[40] 3-D viewer: launcher opens window, controls live, close frees")`).
- Test: the smoke script itself + full suite. No behaviour change → no docs.

**Steps:**

- [ ] Append to `tests/gui_smoke.py` before `print("\nGUI SMOKE PASSED")`:

```python
    # [41] figure builder interactive: arranger grid, Arrange… apply, united
    # colorbar mode, two-step Add-panels dialog.
    from gui.widgets.layout_arranger import ArrangeDialog, LayoutArranger
    from gui.widgets.panel_picker import AddPanelDialog as _APD41

    _la41 = LayoutArranger()
    _la41.set_grid([["s0"]], {"s0": {"title": "strain slice", "group": "strain"}})
    assert _la41.grid() == [["s0"]]
    fb.recipe().compose.colorbar_mode = "united"
    fb._load_compose_into_widgets()
    assert fb._compose_cbar_mode.currentData() == "united"
    _adlg41 = ArrangeDialog(fb.recipe(), fb._style)
    assert _adlg41._arranger.grid() == [["s0"]]
    _adlg41._on_apply()
    fb.apply_arranged_layout(_adlg41.result_layout)
    _res41 = fb.render_now()
    assert _res41 is not None and _res41.n_rendered == 1, fb._notes_label.text()
    _pdlg41 = _APD41({"slices": {"h5": _bh5, "sx": 0.5, "sy": 0.5, "jobs": []}})
    _pdlg41._stage.setCurrentText("slices")
    _pdlg41._reload()
    _pdlg41._check_all()
    _pdlg41._on_next()
    assert _pdlg41._stack.currentIndex() == 1
    _pdlg41.accept()
    assert _pdlg41.selected_panels and _pdlg41.selected_layout is not None
    assert _pdlg41.selected_panels[0].title  # picker captured a data name
    print("[41] figure builder: arranger + Arrange… + united mode + two-step Add panels")
```

- [ ] Run `python3 tests/gui_smoke.py` — all steps `[1]`–`[41]` print, `GUI SMOKE PASSED`. (If `[41]` fails before the implementation tasks are all merged, this task is out of order — it must run last.)
- [ ] Full verification: `python3 -m pytest -q` (expect ~0 failures over the grown suite), `ruff check .`, `ruff format .` (no diffs).
- [ ] Commit: `git add tests/gui_smoke.py && git commit -m "test(smoke): figure-builder arranger/united/two-step dialog checks [41]"`

---

## Spec coverage

| Spec requirement | Task |
|---|---|
| §1 `PanelDef.title` field, serialized, old recipes → `None`, `RECIPE_VERSION` stays 1 | 1 |
| §1 picker derives titles at pick time (map / slices / profiles formats) | 6 |
| §1 outline `Panel: {title}` (+ "(label off)" preserved), title-or-id fallback | 6 |
| §1 scale-bar combo shows titles, stores id as item data | 6 |
| §1 arranger tiles show titles | 7 (widget), 8/9 (info dicts) |
| §1 ids stay the unique key; render/export notes keep ids; dedup unchanged | 6, 8 (`add_panels` renames) |
| §2 `gridmap.py`: `layout_to_grid` recognized shapes + `None` cases | 2 |
| §2 `flatten_panel_ids` DFS order, one-column seed | 2 (function), 9 (seed) |
| §2 `grid_to_layout` Row/Col/bare-ref shapes, empty columns dropped, empty grid → `Row([])` | 2 |
| §2 round-trip law tested | 2 |
| §2 `LayoutArranger`: drag lists, ◀/▶/✕ merge, + Add column, rows = tiles per column | 7 |
| §2 tiles: title + quantity chip + bar badges | 7 (paint), 8 (`panel_group_hint`) |
| §2 API `set_grid`/`grid`/`gridChanged`, pure view | 7 |
| §2 bar schematic: united strip / group-mode strips / corner dot / corner click sets target+corner, schematic-only | 7 (drawing + signal), 9/10 (writes) |
| §2 Add-panels two-step stack, Back/Next/OK/Cancel, OK-from-step-1 flat behaviour | 8 |
| §2 `add_panels(panels, layout=None)` — fragment as one child, ids uniquified incl. refs | 8 |
| §2 Arrange… button + dialog, flatten warning text, apply→replace+purge+rebuild+preview | 9 |
| §2 Col flags preserved by member-id set in the clean case | 9 |
| §3 `colorbar_mode`/`colorbar_pos` + `COLORBAR_MODES`/`COLORBAR_POSITIONS` + validation | 3 |
| §3 scale bar: no new model fields — first-class UI + schematic over existing fields | 10, 7/9 |
| §3 `_apply_united_colorbars`: per-group partition, None-group/trace excluded | 4 |
| §3 clim union with per-panel overrides, `dc_replace`, per-panel bars suppressed | 4 |
| §3 root wrap `Row([root, Col([bars…])])` / `Col([root, Row([bars…])])`, bars pre-measure | 4 |
| §3 post-placement stretch generalized to scattered members (union span, end-inset clamp) | 5 |
| §3 group flags ignored with note; `panel.colorbar=True` override; per-panel mode byte-for-byte | 4 |
| §3 compose-form mode/position dropdowns (position gated on united) | 10 |
| §3 Scale-bar heading: mode, target by title, corner = `style.scale_bar_loc` (one setting) | 10 |
| §3 arranger schematic reflects current settings; corner click writes mode/panel/loc | 9 (Arrange), 10 (Add + form) |
| Error handling: enum validation `StageUserError`+hint; united-with-zero-groupable note; empty-grid orphan purge with dialog note; step-1 errors unchanged | 3, 4, 9, 8 |
| Testing: Qt-free (title round-trip, gridmap, enums, renderer united suite on Agg) | 1–5 |
| Testing: Qt (arranger ops, two-step accept, Arrange flag preservation + flatten, titles in outline/combo) | 6–10 |
| Testing: GUI smoke — window still opens, new dialogs open/close | 11 |
| Docs: Usage.md + Codebase.md updated in the same change as each behaviour change | folded into Tasks 1–10 |
