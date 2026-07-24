# Figure-builder node inspector + hardening sweep — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> Every task is TDD: implementers follow superpowers:test-driven-development.
> Before claiming any task green, run the checks via the repo's `verify-suite` skill.

**Date:** 2026-07-24
**Spec (the contract):** `docs/superpowers/specs/2026-07-24-node-inspector-hardening-design.md` — covers spec sections A–C and hardening items 1–15.
**Base:** master `59bbf47` (all cited line numbers are at this commit).
**Branch:** `node-inspector-hardening`, in place (this session works without worktrees). No remote — no push/PR.

## Goal

Close the figure-builder authoring gap (the GUI cannot yet author `shared_x`, `shared_colorbar`/`shared_clim`, `pinned_height_cm`/`pinned_width_cm`, `Spacer`/`TextCell` box sizes — spec's acceptance figure 2 is only reachable by hand-editing JSON), make outline selection survive mutations, and land the whole-branch review's 15-item hardening backlog. The headline gate is a GUI-side end-to-end test that authors acceptance figure 2 purely through `FigureBuilderWindow` methods and reproduces the exact box geometry pinned by `tests/test_compose_acceptance.py::test_acceptance_figure_2_ragged_dual_scale`.

## Architecture

- **`dfxm/compose/` (Qt-free)** gets the correctness half: `recipe.py` (malformed-recipe wrapping, duplicate-`PanelRef` refusal, not-a-recipe message, blank-`group_label` normalization — items 1, 2, 12, 8), `layout.py` (nested-pin division math, double-pin trace fix, zero-length-trace placeholder, type hints — items 5, 6, 15), `render.py`/`adapters.py`/`__main__.py` (one-panel scale-bar target checks, orphan-def skip, export dir + `style_overrides` fidelity, UV-crop dedup, formats message — items 3, 4, 7, 10, 11, 13).
- **`gui/figure_builder.py`** gets the inspector: the right pane's panel-only override box becomes a `QStackedWidget` with one page per node type (Panel/Row/Col/Spacer/TextCell/none-hint), all following the established per-key submission discipline; plus outline selection persistence, stale-canvas clearing, and the suppressed-label affordance.
- Inspector edits **do not rebuild the tree** — they refresh the selected item's text in place (a rebuild would tear down the widget being typed into). Structural mutators keep rebuilding, and `_rebuild_tree` re-selects by node identity.
- **Nested-pin semantics (binding, from the spec):** a pinned dimension propagating through a container that *stacks along that dimension* (a height pin through a `Col`; a width pin through a `Row`) is divided equally among the children **after subtracting inter-child gutters**, so the container's total equals the pin. Implied-scale/split notes are emitted; nothing is ever silent.

## Tech stack

Python 3.10, PySide6 (only under `gui/`), matplotlib via the explicit `Figure` API (never pyplot), h5py+numpy synthetic fixtures, pytest (offscreen Qt: `QT_QPA_PLATFORM=offscreen`), ruff (line length 100, double quotes, E/F/I; `ruff format` runs on Write/Edit via hook).

## Global constraints (binding)

1. **Per-key submission discipline:** an inspector edit submits only the touched field; no cross-field clobber (the 3-state label lesson). A key absent from a submitted values dict is never touched.
2. **No-op edits must not dirty:** writing a value equal to the current one neither sets `_dirty` nor schedules a preview.
3. **Parse failure → notes bar + no mutation:** malformed text (clim, shared clim, ROI) reports to `self._notes_label` and mutates nothing — never crashes, never partially applies.
4. **Refusals are `StageUserError(message, hint)`** from `dfxm.common.errors` — GUI paths surface them via the notes bar; the CLI prints message + hint to stderr and exits 2.
5. **Never silent:** every degradation (placeholder, pin division, dropped pin, skipped orphan, suppressed bar) appends a note.
6. **Suite baseline must hold throughout:** `python3 -m pytest -q` = 735 passed / 13 skipped / **0 warnings** at base; every task ends ≥ baseline + its new tests, still 13 skipped / 0 warnings.
7. **Docs same-change:** `docs/Usage.md` + `docs/Codebase.md` updated in the same commit as any behaviour/structure change. Treat a code-only commit as incomplete.
8. **TDD failing-first:** write the test, run it, observe the expected failure, then implement.
9. **Qt only under `gui/`** — nothing in `dfxm/` may import PySide6.
10. **Read before first Edit** — especially `hint=` strings (em-dashes, 12- or 16-space indents) and the giant single-line table rows in `docs/Codebase.md` (row for `figure_builder.py` at line 898; entry at line 444). Never reconstruct `old_string` from memory.
11. Commit messages use `feat:`/`fix:`/`test:`/`docs:` with scope and end with the standard `Co-Authored-By: Claude …` / `Claude-Session:` trailers per the harness convention.

---

## Task 1 — recipe.py core hardening (items 1, 2, 12, 8-recipe-side)

**Files**
- `dfxm/compose/recipe.py` — `Row`/`Col` field comments (lines 66–81), `_node_from_dict` (137–165), `recipe_from_json` (249–286), `validate_recipe` (289–337)
- `dfxm/compose/render.py` — `_assign_labels` only (lines 58–94; one-line guard change, no overlap with Task 3's regions)
- `tests/test_compose_recipe.py`, `tests/test_compose_render.py`
- `docs/Codebase.md` — `#### recipe.py` bullets (lines ~496–516); `docs/Usage.md` — CLI exit-code paragraph (~1386–1396) and the Rows/Cols concept bullet (~1289–1293)

**Interfaces**
- Consumes: `StageUserError(message: str, hint: str = "")` (`dfxm/common/errors.py`); `RECIPE_VERSION = 1`.
- Produces (unchanged signatures, new behaviour): `recipe_from_json(text: str, *, base_dir: str | None = None) -> FigureRecipe` now raises `StageUserError` for (a) non-recipe JSON — message contains `"not a figure recipe"`, (b) structurally malformed v1 recipes — message contains `"malformed"`; `validate_recipe(recipe) -> None` now refuses duplicate `PanelRef`s — message contains `"more than once"`; `_node_from_dict` normalizes `group_label` `""` → `None`.

**Steps**

- [ ] 1.1 Read `dfxm/compose/recipe.py` and `tests/test_compose_recipe.py` in full (already-cited line numbers assume `59bbf47`).
- [ ] 1.2 Append failing tests to `tests/test_compose_recipe.py`:

```python
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
```

   And to `tests/test_compose_render.py` (after `test_label_template_and_manual_override`):

```python
def test_blank_group_label_is_not_a_group_slot(tmp_path):
    """Item 8: '' group_label = "not a group" (each member gets its own letter),
    distinct from PanelDef.label where '' = suppressed."""
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.layout.group_label = ""  # programmatic edge — JSON load already normalizes
    res = render_recipe(r)
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert "A" in texts and "B" in texts  # two per-panel letters, not one group slot
```

- [ ] 1.3 Run and confirm the failures:
  `python3 -m pytest tests/test_compose_recipe.py tests/test_compose_render.py -q`
  Expected: `test_malformed_*` fail with raw `TypeError`/`KeyError` escaping; `test_not_a_recipe_*` fails on the "unsupported recipe version None" message; `test_duplicate_panel_ref_refused` fails with `DID NOT RAISE`; `test_blank_group_label_normalized*` fails (`"" != None`); `test_blank_group_label_is_not_a_group_slot` fails (`"B" not in texts`).
- [ ] 1.4 Implement in `dfxm/compose/recipe.py`:
  - In `_node_from_dict`, both `Row` and `Col` branches: `group_label=d.get("group_label") or None`.
  - On the `Row.group_label` field (line 69) replace the trailing comment with:

```python
    # None = not a group; "auto" = auto-lettered slot; text = manual label.
    # "" is normalized to None on load — a blank GROUP label means "not a
    # group", unlike PanelDef.label where "" means "suppress the label".
    group_label: str | None = None
```

  (same comment on `Col.group_label`).
  - Rewrite the body of `recipe_from_json` after the JSON-decode `try` block:

```python
    if not isinstance(d, dict) or not any(k in d for k in ("version", "layout", "panels")):
        raise StageUserError(
            "this JSON file is not a figure recipe",
            hint="Pick a recipe .json saved by the figure builder "
            "(it has version/layout/panels keys).",
        )

    version = d.get("version")
    if version != RECIPE_VERSION:
        raise StageUserError(
            f"unsupported recipe version {version!r}",
            hint="This app writes version 1 recipes — re-save the recipe with this version of the app.",
        )

    if "layout" not in d or "panels" not in d:
        raise StageUserError(
            "recipe is missing 'layout' or 'panels'",
            hint="This does not look like a figure-recipe file.",
        )

    try:
        compose = ComposeStyle(**d.get("compose", {}))
        panels = [_panel_def_from_dict(pd, base_dir) for pd in d["panels"]]
        layout = _node_from_dict(d["layout"])
    except StageUserError:
        raise
    except (TypeError, KeyError) as exc:
        raise StageUserError(
            f"recipe is malformed ({type(exc).__name__}: {exc})",
            hint="A hand-edited or corrupted field is the likely cause — "
            "re-save the recipe from the figure builder.",
        ) from exc

    return FigureRecipe(
        name=d.get("name", ""),
        style=d.get("style", {}),
        compose=compose,
        layout=layout,
        panels=panels,
        version=version,
    )
```

  (update the docstring to mention "not a figure recipe" and malformed-v1 wrapping).
  - In `validate_recipe`, replace the leaf loop (lines 300–306) with:

```python
    by_id = recipe.panel_by_id()
    ref_counts: dict[str, int] = {}
    for leaf in iter_leaves(recipe.layout):
        if isinstance(leaf, PanelRef):
            if leaf.panel_id not in by_id:
                raise StageUserError(
                    f"layout refers to unknown panel id {leaf.panel_id!r} (ghost reference)",
                    hint="Every panel referenced by the layout must exist in the recipe's panel list.",
                )
            ref_counts[leaf.panel_id] = ref_counts.get(leaf.panel_id, 0) + 1
    dupes = sorted(pid for pid, n in ref_counts.items() if n > 1)
    if dupes:
        raise StageUserError(
            f"layout references panel(s) {', '.join(repr(p) for p in dupes)} more than once",
            hint="Each panel can appear in the layout once — add a second PanelDef "
            "(new id, same source) to show the same data twice.",
        )
```

  - In `dfxm/compose/render.py` `_assign_labels` (line 74): change `if node.group_label is not None:` → `if node.group_label:` with a trailing comment `# "" = not a group (item 8) — falls through to per-child labelling`.
- [ ] 1.5 `python3 -m pytest tests/test_compose_recipe.py tests/test_compose_render.py tests/test_gui_figure_builder.py -q` → all pass (the GUI file guards against regressions in `toggle_group_selected`/`set_selected_label`, which are untouched).
- [ ] 1.6 Docs (same commit):
  - `docs/Codebase.md` `#### recipe.py`: extend the `recipe_from_json` bullet (~505–510) with the not-a-recipe and malformed-v1 refusals; extend the `validate_recipe` bullet (~511–516) with "a panel referenced by the layout more than once"; extend the `Row`/`Col` bullet (~496–498) with the `group_label` tri-state semantics and the `""`≠suppressed distinction vs `PanelDef.label`.
  - `docs/Usage.md`: in the CLI exit-code paragraph (~1386–1396) add "a file that isn't a figure recipe at all" to the exit-2 list; in the Rows/Cols concept bullet note that a blank group label means "no group label" (panel labels use a separate explicit "no label" state).
- [ ] 1.7 `ruff check . && ruff format .` then commit:
  `git add dfxm/compose/recipe.py dfxm/compose/render.py tests/test_compose_recipe.py tests/test_compose_render.py docs/Usage.md docs/Codebase.md`
  `git commit -m "fix(compose): recipe hardening — malformed-v1 wrapping, duplicate PanelRef refusal, not-a-recipe message, blank group_label semantics"`

---

## Task 2 — layout.py nested pins, zero-length traces, type hints (items 5, 6, 15)

**Files**
- `dfxm/compose/layout.py` — `size_cells` (78–206: `_map_cell` 109–150, `_trace_cell` 152–191, `walk` 193–205), `measure_cells` (209–227), `place_tree` (235–318)
- `tests/test_compose_layout.py`, `tests/test_compose_render.py`
- `docs/Codebase.md` `#### layout.py — sizing pass` (~584–603+); `docs/Usage.md` Rows/Cols bullet (~1289–1293)

**Interfaces**
- Consumes: `recipe.compose.gutter_cm` (positive, validated by `validate_recipe`); `PLACEHOLDER_CM = (4.0, 3.0)`; `_finite_positive(v)`; `trace_height_cm(style)`.
- Produces: `size_cells(recipe, style, data_by_id, notes) -> dict[int, SizedCell]` (signature unchanged) with the new pin-division/lockstep behaviour; `measure_cells(fig: "Figure", cells: list[SizedCell], pad_in: float = 0.02) -> None`; `place_tree(fig: "Figure", layout: "Row | Col | PanelRef | Spacer | TextCell", cells: dict[int, SizedCell], *, gutter_in: float, pad_in: float) -> tuple[float, float]`.
- Note wording contracts (tests grep for these fragments): `"split over {n} stacked children"`, `"width pin ignored"`, `"degenerate trace length"`, `"too small"`.

**Steps**

- [ ] 2.1 Read `dfxm/compose/layout.py` and `tests/test_compose_layout.py` in full.
- [ ] 2.2 Append failing tests to `tests/test_compose_layout.py`:

```python
def test_nested_col_under_pinned_row_divides_height_after_gutters():
    """Item 5(b): Row(pinned_height) > Col([a, b]) — each stacked child gets
    (pin − gutter)/2, not the full pin (which overflowed the container)."""
    style = PlotStyle(scale_um_per_cm=10.0)
    a, b = PanelRef("a"), PanelRef("b")
    layout = Row([Col([a, b])], pinned_height_cm=4.0)
    recipe = _recipe(layout, [_panel("a"), _panel("b")])
    recipe.compose.gutter_cm = 0.5
    cells = size_cells(recipe, style, {"a": _map_data(), "b": _map_data()}, notes := [])
    each_in = ((4.0 - 0.5) / 2) / 2.54
    assert abs(cells[id(a)].h_in - each_in) < 1e-9
    assert abs(cells[id(b)].h_in - each_in) < 1e-9
    assert any("split over 2 stacked children" in n for n in notes)


def test_trace_under_both_pins_honours_row_height_too():
    """Item 5(a): Col(pinned_width) inside Row(pinned_height) — the width-pin
    early return used to keep the cosmetic trace height, silently dropping the
    row's height pin."""
    style = PlotStyle(trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    t = PanelRef("t")
    layout = Row([Col([t], pinned_width_cm=6.0)], pinned_height_cm=4.0)
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]), style, {"t": _trace_data()}, notes := []
    )
    c = cells[id(t)]
    assert abs(c.w_in - 6.0 / 2.54) < 1e-9
    assert abs(c.h_in - 4.0 / 2.54) < 1e-9  # NOT trace_height_cm's 2.0
    assert any("pinned row height" in n for n in notes)


def test_map_double_pin_height_wins_with_note():
    style = PlotStyle(scale_um_per_cm=10.0)
    a = PanelRef("a")
    layout = Row([Col([a], pinned_width_cm=6.0)], pinned_height_cm=2.0)
    cells = size_cells(_recipe(layout, [_panel("a")]), style, {"a": _map_data()}, notes := [])
    c = cells[id(a)]
    assert abs(c.h_in - 2.0 / 2.54) < 1e-9
    assert abs(c.w_in - 2.0 * (20.0 / 10.0) / 2.54) < 1e-9  # aspect from the height pin
    assert any("width pin ignored" in n for n in notes)


def test_pin_too_small_for_children_refused():
    layout = Row([Col([PanelRef("a"), PanelRef("b")])], pinned_height_cm=0.4)
    recipe = _recipe(layout, [_panel("a"), _panel("b")])
    recipe.compose.gutter_cm = 0.5  # the gutter alone exceeds the pin
    with pytest.raises(StageUserError) as e:
        size_cells(
            recipe, PlotStyle(scale_um_per_cm=10.0), {"a": _map_data(), "b": _map_data()}, []
        )
    assert "too small" in str(e.value) and e.value.hint


def test_zero_length_trace_becomes_placeholder_with_note():
    style = PlotStyle(trace_scale_um_per_cm=5.0)
    layout = PanelRef("t")
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]), style, {"t": _trace_data(0.0)}, notes := []
    )
    c = cells[id(layout)]
    assert c.kind == "placeholder"
    assert (c.w_in, c.h_in) == (4.0 / 2.54, 3.0 / 2.54)
    assert any("degenerate trace length" in n for n in notes)


def test_zero_length_trace_under_width_pin_still_placeholder():
    t = PanelRef("t")
    layout = Col([t], pinned_width_cm=4.0)
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]),
        PlotStyle(),
        {"t": _trace_data(0.0)},
        notes := [],
    )
    assert cells[id(t)].kind == "placeholder"
    assert any("degenerate trace length" in n for n in notes)
```

   And to `tests/test_compose_render.py`:

```python
def test_zero_length_trace_renders_placeholder_lockstep(tmp_path, monkeypatch):
    """Item 6: length_um == 0 joins the degenerate-extent placeholder lockstep —
    placeholder draw + note, never a zero-width trace axes (no mpl warnings)."""
    import warnings

    from dfxm.compose.adapters import PanelData

    monkeypatch.setattr(
        "dfxm.compose.render.load_panel",
        lambda p, cache=None: PanelData(kind="profiles_trace", length_um=0.0, payload={}),
    )
    p = PanelDef("t", PanelSource("/x.h5", "profiles_trace", {"job": JOB, "field": "strain"}))
    r = FigureRecipe("z", {"trace_scale_um_per_cm": 5.0}, ComposeStyle(), PanelRef("t"), [p])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = render_recipe(r)
    assert res.n_rendered == 0
    assert any("degenerate trace length" in n for n in res.notes)
```

- [ ] 2.3 Run and confirm failures: `python3 -m pytest tests/test_compose_layout.py tests/test_compose_render.py -q` — nested-pin test fails with `h_in == 4/2.54` (full pin), double-pin trace fails with `h_in == 2/2.54`, map-double-pin fails on the missing note, too-small does not raise, zero-length tests fail with `StageUserError: trace panel t has no trace scale to size from` (unit) / same error propagating (render).
- [ ] 2.4 Implement in `dfxm/compose/layout.py`:
  - At the top of `size_cells` (after `cells = {}`), add:

```python
    gutter_in = recipe.compose.gutter_cm * _IN_PER_CM

    def _split_pin(pin_in, n, what):
        """Divide a pin among n stacked children, subtracting inter-child
        gutters, so the container's total equals the pin. Never silent."""
        if pin_in is None or n <= 1:
            return pin_in
        each = (pin_in - gutter_in * (n - 1)) / n
        if each <= 0:
            raise StageUserError(
                f"pinned {what} {pin_in / _IN_PER_CM:.4g} cm is too small for "
                f"{n} stacked children plus {n - 1} gutter(s)",
                hint="Increase the pinned size, reduce compose.gutter_cm, or remove the pin.",
            )
        notes.append(
            f"pinned {what} {pin_in / _IN_PER_CM:.4g} cm split over {n} stacked "
            f"children — {each / _IN_PER_CM:.4g} cm each after gutters"
        )
        return each
```

  - Replace `walk` (193–205) with:

```python
    def walk(node, pinned_h_in, pinned_w_in):
        if isinstance(node, Row):
            ph = node.pinned_height_cm * _IN_PER_CM if node.pinned_height_cm else pinned_h_in
            # a width pin crossing a Row is shared by its side-by-side children
            pw = _split_pin(pinned_w_in, len(node.children), "column width")
            for child in node.children:
                walk(child, ph, pw)
        elif isinstance(node, Col):
            pw = node.pinned_width_cm * _IN_PER_CM if node.pinned_width_cm else pinned_w_in
            # a height pin crossing a Col is shared by its stacked children
            ph = _split_pin(pinned_h_in, len(node.children), "row height")
            for child in node.children:
                walk(child, ph, pw)
        else:
            cells[id(node)] = leaf_cell(node, pinned_h_in, pinned_w_in)
```

  - In `_map_cell`, inside the `if pinned_h_in is not None:` branch (before `return`), add:

```python
            if pinned_w_in is not None:
                notes.append(
                    f"panel {panel.id}: both row height and column width pinned — "
                    "height pin wins (map aspect is fixed); width pin ignored"
                )
```

  - In `_trace_cell`: at the very top (before the `pinned_w_in` early return) add the degenerate guard, and honour the height pin in the width-pin branch:

```python
    def _trace_cell(leaf, panel, data, pinned_h_in, pinned_w_in):
        length = data.length_um
        if not _finite_positive(length):
            notes.append(
                f"panel {panel.id}: degenerate trace length — rendered as placeholder"
            )
            return SizedCell(
                leaf,
                panel,
                "placeholder",
                PLACEHOLDER_CM[0] * _IN_PER_CM,
                PLACEHOLDER_CM[1] * _IN_PER_CM,
            )
        if pinned_w_in is not None:
            w = pinned_w_in
            h = pinned_h_in if pinned_h_in is not None else trace_height_cm(style) * _IN_PER_CM
            implied = length / (w / _IN_PER_CM) if w > 0 else 0.0
            notes.append(
                f"panel {panel.id}: pinned column width — implied trace scale {implied:.4g} µm/cm"
            )
            if pinned_h_in is not None:
                notes.append(
                    f"panel {panel.id}: pinned row height — trace height {h / _IN_PER_CM:.4g} cm"
                )
            return SizedCell(leaf, panel, "trace", w, h)
```

    (the rest of `_trace_cell` — the scale-resolution/clamp path and the height-pin-only tail — is unchanged; drop the now-dead `length = data.length_um or 0.0` line).
  - Item 15: annotate `measure_cells`/`place_tree` per the Interfaces block above, adding at module top:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure
```

    (string annotations only — layout.py must stay matplotlib-free at import time).
  - No change needed in `render.py` for the lockstep: the existing `cell.kind == "placeholder" and data.kind != "placeholder"` conversion in `render_recipe` (lines 398–401) already covers trace cells — the new render test proves it.
- [ ] 2.5 `python3 -m pytest tests/test_compose_layout.py tests/test_compose_render.py tests/test_compose_acceptance.py -q` → all pass (acceptance figures 1/2 must be byte-identical in geometry — they use no nested cross-axis pins).
- [ ] 2.6 Docs (same commit):
  - `docs/Codebase.md` `#### layout.py`: in the `size_cells` bullet document the pin-propagation rule (division across stacked children after gutters, equal split, `StageUserError` when gutters exceed the pin, the split note), the trace double-pin resolution (width pin sets width, row pin overrides the cosmetic height), the map double-pin note, and the zero-length-trace placeholder; note the new `measure_cells`/`place_tree` annotations.
  - `docs/Usage.md` Rows/Cols bullet: one sentence — "a pinned height crossing a nested column (or a pinned width crossing a nested row) is divided equally among the stacked panels after gutters, so the container matches the pin exactly; the split is reported in the notes bar."
- [ ] 2.7 `ruff check . && ruff format .`; commit:
  `git add dfxm/compose/layout.py tests/test_compose_layout.py tests/test_compose_render.py docs/Usage.md docs/Codebase.md`
  `git commit -m "fix(compose): nested-pin division + double-pin traces + zero-length trace placeholder + solver type hints"`

---

## Task 3 — render/CLI/adapters hardening (items 3, 4, 7, 10, 11, 13)

**Files**
- `dfxm/compose/render.py` — `render_recipe` head (360–373), `_resolve_scale_bar_kwargs` (301–357) and its call site (419–421), `export_recipe` (609–629)
- `dfxm/compose/__main__.py` — formats message (line 43)
- `dfxm/compose/adapters.py` — `_load_slice_plane` ROI block (176–186), `_crop_profiles_uv` (201–213), `_load_profiles_ref` call (243)
- `tests/test_compose_render.py`, `tests/test_compose_cli.py`, `tests/test_compose_adapters.py`
- `docs/Codebase.md` — render numbered-steps region (~763–811), `export_recipe` entry, `#### adapters.py`, CLI entry (~837); `docs/Usage.md` — CLI section (~1368–1396), Export bullet (~1361–1366)

**Interfaces**
- Produces: `_resolve_scale_bar_kwargs(recipe, panels_by_id, data_by_id, cell_by_pid, notes)` — **new `notes: list[str]` parameter** (module-private; single call site in `render_recipe`). One-panel mode now raises `StageUserError` for a target that is not placed in the layout (`"not placed"`) or is a trace (`"trace panel"`), and appends a `"no scale bar drawn"` note (no raise) when the target degraded to a placeholder.
- `render_recipe` loads data **only for layout-referenced pids**; orphaned defs get a note `"skipped without loading"`.
- `export_recipe(recipe, out_dir, *, formats=None, dpi=None, style_overrides=None, loader_cache=None)` — unchanged signature; now (a) `os.makedirs` moved to the top and wrapped into `StageUserError` (message contains `"output directory"`), (b) formats/dpi fall back to `{**recipe.style, **(style_overrides or {})}`.
- `adapters._crop_uv(plane, u, v, roi)` — rename of `_crop_profiles_uv` (verified: no users outside `adapters.py`), now also called by `_load_slice_plane`.

**Steps**

- [ ] 3.1 Read the three source files in the cited regions plus `tests/test_compose_cli.py` and `tests/test_compose_adapters.py` headers.
- [ ] 3.2 Append failing tests. To `tests/test_compose_render.py`:

```python
def test_orphaned_panel_def_not_loaded_at_all(tmp_path):
    """Item 7: an orphaned PanelDef is skipped WITHOUT an h5 read and reported."""
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.layout = Row([PanelRef("a")])  # "b" orphaned
    cache: dict = {}
    res = render_recipe(r, loader_cache=cache)
    assert res.n_panels == 1 and res.n_rendered == 1
    assert len(cache) == 1  # only "a" was loaded
    assert any("skipped without loading" in n and "b" in n for n in res.notes)


def test_one_panel_scale_bar_unplaced_target_refused(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.layout = Row([PanelRef("a")])  # "b" exists as a def but is not placed
    r.compose.scale_bar_mode = "one-panel"
    r.compose.scale_bar_panel = "b"
    with pytest.raises(StageUserError) as e:
        render_recipe(r)
    assert "not placed" in str(e.value) and e.value.hint


def test_one_panel_scale_bar_trace_target_refused(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels.append(
        PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    )
    r.layout.children.append(PanelRef("t"))
    r.style["trace_scale_um_per_cm"] = 5.0
    r.compose.scale_bar_mode = "one-panel"
    r.compose.scale_bar_panel = "t"
    with pytest.raises(StageUserError) as e:
        render_recipe(r)
    assert "trace panel" in str(e.value) and e.value.hint


def test_one_panel_scale_bar_placeholder_target_degrades_with_note(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels[1].source.h5_path = str(tmp_path / "gone.h5")  # "b" -> placeholder
    r.compose.scale_bar_mode = "one-panel"
    r.compose.scale_bar_panel = "b"
    res = render_recipe(r)
    assert _scale_bar_box(res.axes_by_id["a"]) is None  # no bar leaks elsewhere
    assert any("no scale bar drawn" in n for n in res.notes)


def test_export_dir_uncreatable_raises_user_error(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    blocker = tmp_path / "blocker"
    blocker.write_text("a file standing where the out dir should be")
    with pytest.raises(StageUserError) as e:
        export_recipe(_two_panel_recipe(h5), str(blocker / "out"))
    assert "output directory" in str(e.value) and e.value.hint


def test_export_honours_style_overrides_formats_and_dpi(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    h5 = _write_obl(tmp_path / "obl.h5")
    recorded = {}
    orig = Figure.savefig

    def rec(self, path, **kw):
        recorded[os.path.basename(path)] = kw.get("dpi")
        return orig(self, path, **kw)

    monkeypatch.setattr(Figure, "savefig", rec)
    paths, _res = export_recipe(
        _two_panel_recipe(h5),
        str(tmp_path / "out"),
        style_overrides={"formats": ["svg"], "dpi": 72},
    )
    assert [os.path.splitext(p)[1] for p in paths] == [".svg"]
    assert recorded == {"demo.svg": 72}
```

   To `tests/test_compose_cli.py`:

```python
def test_cli_uncreatable_out_dir_exits_two(tmp_path, capsys):
    h5 = _write_obl(tmp_path / "obl.h5")
    rp = tmp_path / "r.json"
    rp.write_text(recipe_to_json(_two_panel_recipe(h5)))
    blocker = tmp_path / "blocker"
    blocker.write_text("file, not a directory")
    rc = _main(["render", str(rp), "-o", str(blocker / "out")])
    assert rc == 2
    err = capsys.readouterr().err.lower()
    assert "output directory" in err and "hint" in err


def test_cli_multi_bad_formats_quoted_individually(tmp_path, capsys):
    h5 = _write_obl(tmp_path / "obl.h5")
    rp = tmp_path / "r.json"
    rp.write_text(recipe_to_json(_two_panel_recipe(h5)))
    rc = _main(["render", str(rp), "-o", str(tmp_path / "out"), "--formats", "png,jpg,tiff"])
    assert rc == 2
    assert "'jpg', 'tiff'" in capsys.readouterr().err
```

   To `tests/test_compose_adapters.py` (refactor guard for item 11):

```python
def test_slice_plane_roi_clamps_out_of_range_indices(tmp_path):
    """Out-of-range ROI indices clamp to the plane bounds (parity guard for the
    _crop_uv dedup — same behaviour as the pre-refactor inline clamp)."""
    import h5py
    import numpy as np

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0.0, 2.0, 5))
        sg.create_dataset("v_um", data=np.linspace(0.0, 1.5, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0]))
    p = PanelDef(
        "s",
        PanelSource(
            str(h5), "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}
        ),
        roi=(0, 999, 0, 999),
    )
    d = load_panel(p)
    assert d.kind == "slice_plane"
    assert d.ext_x_um == 2.0 and d.ext_y_um == 1.5  # full extents survive the clamp
```

- [ ] 3.3 Run and confirm failures: `python3 -m pytest tests/test_compose_render.py tests/test_compose_cli.py tests/test_compose_adapters.py -q` — orphan test fails on `len(cache) == 1` and the missing note; unplaced/trace targets fail `DID NOT RAISE`; placeholder-target fails on the missing note; export-dir test fails with a raw `OSError`/`NotADirectoryError`; overrides test fails writing `demo.png` at style dpi; CLI dir test fails with an escaping traceback (exit ≠ 2); formats test fails (`'jpg, tiff'` joined form); the adapters clamp test passes already (pure guard) — that is expected, note it in the task log.
- [ ] 3.4 Implement:
  - **`render.py` `render_recipe` head** (replace lines 369–373):

```python
    panels_by_id = dict(recipe.panel_by_id())
    notes: list[str] = []
    live_pids = {
        leaf.panel_id for leaf in iter_leaves(recipe.layout) if isinstance(leaf, PanelRef)
    }
    orphans = sorted(set(panels_by_id) - live_pids)
    if orphans:
        notes.append(
            "panel def(s) not referenced by the layout — skipped without loading: "
            + ", ".join(orphans)
        )
    data_by_id = {pid: load_panel(panels_by_id[pid], cache=loader_cache) for pid in live_pids}

    cells = size_cells(recipe, style, data_by_id, notes)
```

    (delete the old separate `notes: list[str] = []` line).
  - **`_resolve_scale_bar_kwargs`**: add the `notes` parameter and replace the `one-panel` branch:

```python
def _resolve_scale_bar_kwargs(recipe, panels_by_id, data_by_id, cell_by_pid, notes):
    ...
    if mode == "one-panel":
        target = recipe.compose.scale_bar_panel
        if target not in panels_by_id:
            raise StageUserError(
                f"compose.scale_bar_panel {target!r} is not a known panel id",
                hint=_NO_SCALE_BAR_PANEL_HINT,
            )
        if target not in cell_by_pid:
            raise StageUserError(
                f"compose.scale_bar_panel {target!r} is not placed in the layout",
                hint=_NO_SCALE_BAR_PANEL_HINT,
            )
        target_kind = data_by_id[target].kind
        if target_kind == "profiles_trace":
            raise StageUserError(
                f"compose.scale_bar_panel {target!r} is a trace panel — "
                "a scale bar needs a map panel",
                hint=_NO_SCALE_BAR_PANEL_HINT,
            )
        if target_kind == "placeholder":
            # data-availability, not authoring: degrade with a note, no bar anywhere
            notes.append(
                f"scale-bar panel {target}: data unavailable (placeholder) — "
                "no scale bar drawn"
            )
            for pid in map_pids:
                scale_bar_by_pid[pid] = False
        else:
            for pid in map_pids:
                scale_bar_by_pid[pid] = pid == target
```

    Update the call site: `_resolve_scale_bar_kwargs(recipe, panels_by_id, data_by_id, cell_by_pid, notes)`.
  - **`export_recipe`**: move `os.makedirs` to the top, wrapped, and merge overrides into the style read:

```python
def export_recipe(
    recipe,
    out_dir,
    *,
    formats=None,
    dpi=None,
    style_overrides: dict | None = None,
    loader_cache: dict | None = None,
):
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        raise StageUserError(
            f"cannot create output directory {out_dir!r}: {exc}",
            hint="Check the path — it must be creatable and writable (permissions, "
            "read-only media, or a file standing where the directory should be).",
        ) from exc
    res = render_recipe(recipe, style_overrides, loader_cache=loader_cache)
    style = (
        style_from_params({"plot_style": {**recipe.style, **(style_overrides or {})}})
        or PlotStyle()
    )
    fmts = tuple(formats) if formats else tuple(style.formats)
    the_dpi = int(dpi) if dpi else int(style.dpi)
    ...  # rest unchanged (stem/savefig loop)
```

  - **`__main__.py` line 43**: `print(f"error: unknown format(s) {', '.join(repr(b) for b in bad)}", file=sys.stderr)`.
  - **`adapters.py`**: rename `_crop_profiles_uv` → `_crop_uv` (update the `_load_profiles_ref` call at 243) and replace `_load_slice_plane`'s inline block (176–186) with:

```python
    s2d, u, v = _crop_uv(s2d, u, v, roi)
```

    (delete the now-duplicated clamp lines; `_crop_uv` handles `roi is None` and raises the same `ValueError` for an empty crop).
- [ ] 3.5 `python3 -m pytest tests/test_compose_render.py tests/test_compose_cli.py tests/test_compose_adapters.py tests/test_compose_acceptance.py tests/test_gui_figure_builder.py -q` → all pass. Pay attention to the pre-existing `test_orphaned_panel_def_tolerated_in_gutter_mode` and `test_one_panel_scale_bar_only_designated_panel` — both must still pass unmodified.
- [ ] 3.6 Docs (same commit):
  - `docs/Codebase.md`: render steps — orphan-skip note in the data-loading step; extend the one-panel scale-bar step (~763–765) with the placed/map-kind checks and the placeholder-degrade note; `export_recipe` entry gains the makedirs-first `StageUserError` and the `style_overrides` merge for formats/dpi; `#### adapters.py` documents `_crop_uv` as the shared UV-crop/clamp helper used by `_load_slice_plane` and `_load_profiles_ref`; CLI entry notes the exit-2 unwritable-out-dir case and per-value format quoting.
  - `docs/Usage.md`: CLI section — add "an output directory that cannot be created (e.g. a file stands in its way)" to the exit-2 list; Export bullet — note the export honours the builder's current style for formats/DPI exactly as previewed.
- [ ] 3.7 `ruff check . && ruff format .`; commit:
  `git add dfxm/compose/render.py dfxm/compose/__main__.py dfxm/compose/adapters.py tests/test_compose_render.py tests/test_compose_cli.py tests/test_compose_adapters.py docs/Usage.md docs/Codebase.md`
  `git commit -m "fix(compose): render/CLI/adapters hardening — scale-bar target checks, orphan skip, export dir+overrides, UV-crop dedup, formats message"`

---

## Task 4 — node inspector: QStackedWidget pages, shared-colorbar controls, 3-state labels, "auto" sentinel fix (spec §A)

**Files**
- `gui/figure_builder.py` — imports (19–39: add `QCheckBox`, `QStackedWidget`), `_build_right_pane` (194–217), `_build_override_editor` (332–384), `_on_tree_selection_changed` (386–426), `_on_override_field_edited` (428–452), `_apply_panel_overrides` (461–536), `_on_label_selected` (842–856)
- `tests/test_gui_figure_builder.py`
- `docs/Usage.md` right-pane section (~1326–1366); `docs/Codebase.md` `figure_builder.py` table row (single line 898 — edit via small unique substrings)

**Interfaces (Produces — Task 5 and Task 6 consume these)**
- `self._inspector: QStackedWidget`; pages `self._page_hint/_page_panel/_page_row/_page_col/_page_spacer/_page_text`; `self._inspector_node` (the selected layout node or `None`).
- `_apply_node_field(self, node, field: str, value) -> None` — writes ONE field on a layout node; returns without side effects when `node is None`, the field is absent, or the value is unchanged (no-op ⇒ not dirty); otherwise `_after_inspector_mutation()`.
- `_apply_pin_spin(self, node, field: str, value: float) -> None` — `0 → None`.
- `_apply_shared_clim_text(self, node, text: str) -> None` — `"lo,hi"` → tuple, blank → `None`, parse failure → notes bar + no mutation.
- `_after_inspector_mutation(self) -> None` — dirty + refresh the current tree item's text in place (NO `_rebuild_tree`) + retitle + `schedule_preview()`.
- `_label_override_value(self) -> str | None` — resolves the panel-label 3-state combo (`None` = auto / `""` = suppressed / text = manual).
- Widgets: `_row_group_mode/_row_group_label/_row_pin_h/_row_shared_cb/_row_shared_clim`, `_col_group_mode/_col_group_label/_col_pin_w/_col_shared_x/_col_shared_cb/_col_shared_clim`, `_spacer_w/_spacer_h`, `_text_edit/_text_w/_text_h`, `_ov_label_mode` (+ existing `_ov_*`).
- **Semantics change:** `_apply_panel_overrides`'s `"label"` key is now assigned verbatim (tri-state), no `or None` coercion; the method no-ops (no dirty) when every submitted key equals its current value; it calls `_after_inspector_mutation()` instead of rebuild+reselect.

**Steps**

- [ ] 4.1 Read `gui/figure_builder.py` in full (last full read this session; later tasks re-read only target regions).
- [ ] 4.2 Append failing tests to `tests/test_gui_figure_builder.py`:

```python
# -- node inspector (T4) ------------------------------------------------------
def _select_child(w, i):
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(i))


def test_inspector_switches_page_per_node_type():
    w = _win()
    w.add_panels([_panel("a")])
    w._tree.setCurrentItem(None)
    w.add_row()
    w._tree.setCurrentItem(None)
    w.add_col()
    w._tree.setCurrentItem(None)
    w.add_spacer()
    w._tree.setCurrentItem(None)
    w.add_text()
    w._tree.setCurrentItem(None)
    assert w._inspector.currentWidget() is w._page_hint
    _select_child(w, 0)
    assert w._inspector.currentWidget() is w._page_panel
    _select_child(w, 1)
    assert w._inspector.currentWidget() is w._page_row
    _select_child(w, 2)
    assert w._inspector.currentWidget() is w._page_col
    _select_child(w, 3)
    assert w._inspector.currentWidget() is w._page_spacer
    _select_child(w, 4)
    assert w._inspector.currentWidget() is w._page_text


def test_row_page_pin_and_shared_controls_write_fields():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    w._row_pin_h.setValue(3.5)
    assert row.pinned_height_cm == 3.5
    w._row_pin_h.setValue(0.0)
    assert row.pinned_height_cm is None
    w._row_shared_cb.setChecked(True)
    assert row.shared_colorbar is True
    w._row_shared_clim.setText("-1,2")
    assert row.shared_clim == (-1.0, 2.0)
    w._row_shared_clim.setText("")
    assert row.shared_clim is None


def test_col_page_shared_x_and_pin_width():
    w = _win()
    w.add_col()
    _select_child(w, 0)
    col = w.recipe().layout.children[0]
    w._col_shared_x.setChecked(True)
    assert col.shared_x is True
    w._col_pin_w.setValue(6.0)
    assert col.pinned_width_cm == 6.0


def test_shared_clim_parse_failure_notes_no_mutation():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    row.shared_clim = (0.0, 1.0)
    w._dirty = False
    w._row_shared_clim.setText("nope")
    assert row.shared_clim == (0.0, 1.0)
    assert "shared clim" in w._notes_label.text()
    assert not w.is_dirty()


def test_spacer_and_text_pages_edit_boxes():
    w = _win()
    w.add_spacer()
    _select_child(w, 0)
    sp = w.recipe().layout.children[0]
    w._spacer_w.setValue(1.25)
    assert sp.w_cm == 1.25
    w._tree.setCurrentItem(None)
    w.add_text()
    _select_child(w, 1)
    tx = w.recipe().layout.children[1]
    w._text_edit.setText("Header")
    assert tx.text == "Header"
    w._text_h.setValue(0.75)
    assert tx.h_cm == 0.75


def test_panel_label_three_state_control():
    w = _win()
    w.add_panels([_panel("a")])
    _select_child(w, 0)
    panel = w.recipe().panels[0]
    assert panel.label is None
    w._ov_label_mode.setCurrentIndex(1)  # "No label"
    assert panel.label == ""
    w._ov_label_mode.setCurrentIndex(2)  # "Custom…"
    w._ov_label.setText("Z9")
    assert panel.label == "Z9"
    w._ov_label_mode.setCurrentIndex(0)  # back to auto
    assert panel.label is None


def test_group_mode_control_and_no_auto_sentinel_leak(monkeypatch):
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    w._row_group_mode.setCurrentIndex(1)  # Auto letter
    assert row.group_label == "auto"
    w._tree.setCurrentItem(None)
    w.add_spacer()  # a second node to bounce selection off
    _select_child(w, 1)
    _select_child(w, 0)  # reload the Row page
    assert w._row_group_label.text() == ""  # sentinel never shown
    assert w._row_group_mode.currentData() == "auto"
    captured = {}

    def fake_get_text(*_a, **kw):
        captured["prefill"] = kw.get("text", "")
        return ("", False)

    monkeypatch.setattr("gui.figure_builder.QInputDialog.getText", fake_get_text)
    w._on_label_selected()
    assert captured["prefill"] == ""  # the Label… dialog leak, fixed
    w._row_group_mode.setCurrentIndex(2)  # Custom…
    w._row_group_label.setText("M1")
    assert row.group_label == "M1"


def test_noop_inspector_edit_does_not_dirty():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    row = w.recipe().layout.children[0]
    w._dirty = False
    w._apply_node_field(row, "shared_colorbar", False)  # already False
    assert not w.is_dirty()
    w._tree.setCurrentItem(None)
    w.add_panels([_panel("a")])
    panel = w.recipe().panels[0]
    w._dirty = False
    w._apply_panel_overrides(panel, {"cmap": ""})  # "" maps to None == current
    assert not w.is_dirty()


def test_inspector_edit_updates_item_text_in_place_without_rebuild():
    w = _win()
    w.add_row()
    _select_child(w, 0)
    item_before = w._tree.currentItem()
    w._row_group_mode.setCurrentIndex(1)  # -> "[group]" marker
    assert w._tree.currentItem() is item_before  # same item object: no rebuild
    assert "[group]" in item_before.text(0)
```

- [ ] 4.3 `python3 -m pytest tests/test_gui_figure_builder.py -q` — all new tests fail with `AttributeError` (`_inspector`, `_row_pin_h`, `_ov_label_mode`, …) or on the sentinel/dirty assertions.
- [ ] 4.4 Implement in `gui/figure_builder.py` (per the Interfaces block; key code):
  - Imports: add `QCheckBox`, `QStackedWidget` to the PySide6 import list.
  - `_build_right_pane`: replace the two override-editor lines with `layout.addWidget(QLabel("<b>Selected node</b>"))` + `layout.addWidget(self._build_inspector())`.
  - New builder:

```python
    def _build_inspector(self) -> QStackedWidget:
        self._inspector_node = None
        self._inspector = QStackedWidget()
        self._page_hint = QLabel("select a node in the outline to edit it")
        self._page_hint.setWordWrap(True)
        self._page_panel = self._build_override_editor()
        self._page_row = self._build_row_page()
        self._page_col = self._build_col_page()
        self._page_spacer = self._build_spacer_page()
        self._page_text = self._build_text_page()
        for page in (
            self._page_hint,
            self._page_panel,
            self._page_row,
            self._page_col,
            self._page_spacer,
            self._page_text,
        ):
            self._inspector.addWidget(page)
        self._inspector.setCurrentWidget(self._page_hint)
        return self._inspector
```

  - Core appliers:

```python
    def _after_inspector_mutation(self) -> None:
        """Dirty + retitle + preview + refresh the selected item's text IN PLACE.

        Inspector edits never rebuild the tree: the structure didn't change,
        and a rebuild would tear down the very widget being typed into."""
        self._dirty = True
        item = self._tree.currentItem()
        if item is not None:
            item.setText(0, self._node_label(item.data(0, Qt.ItemDataRole.UserRole)))
        self._update_title()
        self.schedule_preview()

    def _apply_node_field(self, node, field: str, value) -> None:
        """Per-key submission for layout-node fields; no-op edits do not dirty."""
        if node is None or not hasattr(node, field):
            return
        if getattr(node, field) == value:
            return
        setattr(node, field, value)
        self._after_inspector_mutation()

    def _apply_pin_spin(self, node, field: str, value: float) -> None:
        self._apply_node_field(node, field, float(value) if value > 0 else None)

    def _apply_shared_clim_text(self, node, text: str) -> None:
        t = text.strip()
        if not t:
            self._apply_node_field(node, "shared_clim", None)
            return
        parts = t.split(",")
        try:
            if len(parts) != 2:
                raise ValueError(t)
            lo, hi = (float(p) for p in parts)
        except ValueError:
            self._notes_label.setText(
                f"invalid shared clim text {t!r} — expected 'lo,hi' "
                "(blank = union of member ranges)"
            )
            return
        self._apply_node_field(node, "shared_clim", (lo, hi))
```

  - Group-label control (shared by Row/Col pages):

```python
    def _make_group_label_row(self, form: QFormLayout):
        mode = QComboBox()
        for text, value in (("Not a group", "none"), ("Auto letter", "auto"), ("Custom…", "custom")):
            mode.addItem(text, value)
        edit = QLineEdit()
        edit.setEnabled(False)

        def apply(*_a):
            edit.setEnabled(mode.currentData() == "custom")
            value = {"none": None, "auto": "auto"}.get(mode.currentData(), edit.text() or None)
            self._apply_node_field(self._inspector_node, "group_label", value)

        mode.currentIndexChanged.connect(apply)
        edit.textChanged.connect(apply)
        form.addRow("Group label", mode)
        form.addRow("", edit)
        return mode, edit

    @staticmethod
    def _load_group_label(mode: QComboBox, edit: QLineEdit, value) -> None:
        # the "auto" sentinel stays internal — always display blank for it
        idx = 0 if value is None else 1 if value == "auto" else 2
        mode.setCurrentIndex(idx)
        edit.setText(value if idx == 2 else "")
        edit.setEnabled(idx == 2)
```

  - Row page (`_build_row_page`): `_make_group_label_row` + `_row_pin_h` (`QDoubleSpinBox`, range 0–1000, 2 decimals, suffix `" cm"`, special value text `"off"`, `valueChanged → lambda v: self._apply_pin_spin(self._inspector_node, "pinned_height_cm", v)`, form row label `"Pinned height (0 = off)"`) + `_row_shared_cb` (`QCheckBox("One colorbar for this group")`, `toggled → lambda on: self._apply_node_field(self._inspector_node, "shared_colorbar", bool(on))`) + `_row_shared_clim` (`QLineEdit`, placeholder `"lo,hi (blank = union of member ranges)"`, `textChanged → lambda t: self._apply_shared_clim_text(self._inspector_node, t)`).
  - Col page (`_build_col_page`): same set with `_col_pin_w → "pinned_width_cm"` (`"Pinned width (0 = off)"`), plus `_col_shared_x` (`QCheckBox("Shared x axis (bottom labels only)")` → field `"shared_x"`).
  - Spacer page: `_spacer_w`/`_spacer_h` (`QDoubleSpinBox`, range 0.1–100, 2 decimals, suffix `" cm"`) → fields `"w_cm"`/`"h_cm"` via `lambda v: self._apply_node_field(self._inspector_node, "w_cm", float(v))` etc.
  - Text page: `_text_edit` (`QLineEdit` → `"text"`), `_text_w`/`_text_h` as for Spacer.
  - Loaders `_load_row_page/_load_col_page/_load_spacer_page/_load_text_page`: block signals on the page's widgets, set values (`pin or 0.0`; shared clim as `f"{lo:g},{hi:g}"` or `""`; `_load_group_label`), unblock.
  - `_on_tree_selection_changed` → dispatcher:

```python
    def _on_tree_selection_changed(self, *_args) -> None:
        node = self._selected_node()
        self._inspector_node = node
        panel = None
        if isinstance(node, PanelRef):
            panel = self._recipe.panel_by_id().get(node.panel_id)
        self._override_panel = panel
        if panel is not None:
            self._load_panel_page(panel)
            self._inspector.setCurrentWidget(self._page_panel)
        elif isinstance(node, Row):
            self._load_row_page(node)
            self._inspector.setCurrentWidget(self._page_row)
        elif isinstance(node, Col):
            self._load_col_page(node)
            self._inspector.setCurrentWidget(self._page_col)
        elif isinstance(node, Spacer):
            self._load_spacer_page(node)
            self._inspector.setCurrentWidget(self._page_spacer)
        elif isinstance(node, TextCell):
            self._load_text_page(node)
            self._inspector.setCurrentWidget(self._page_text)
        else:
            self._inspector.setCurrentWidget(self._page_hint)
```

    `_load_panel_page(panel)` is the existing widget-loading block moved out of the old method verbatim (blockSignals list now includes `_ov_label_mode`), with the label load replaced by the 3-state form: `None` → mode 0/text ""/disabled; `""` → mode 1/text ""/disabled; text → mode 2/text/enabled. Drop the `_override_group.setEnabled(...)` calls (page switching replaces them; the `self._override_panel is None` guard in `_on_override_field_edited` stays).
  - Panel label control in `_build_override_editor`: insert `self._ov_label_mode` (`QComboBox` with `("Auto letter", "auto"), ("No label", "none"), ("Custom…", "custom")` as `addItem(text, value)`) above the existing `_ov_label` edit; wire `currentIndexChanged` to (a) `self._ov_label.setEnabled(self._ov_label_mode.currentData() == "custom")` and (b) `self._on_override_field_edited("label")`; keep `_ov_label.textChanged → _on_override_field_edited("label")`. Replace the `"label"` getter with `self._label_override_value`:

```python
    def _label_override_value(self):
        mode = self._ov_label_mode.currentData()
        if mode == "auto":
            return None
        if mode == "none":
            return ""
        return self._ov_label.text()
```

  - `_apply_panel_overrides`: replace the assignment tail (lines 516–536) with a changes-dict + no-op guard, and switch to `_after_inspector_mutation`:

```python
        changes: dict = {}
        if "roi" in values:
            changes["roi"] = new_roi
        if "clim" in values:
            changes["clim"] = new_clim
        if "cmap" in values:
            changes["cmap"] = values["cmap"] or None
        if "label" in values:
            changes["label"] = values["label"]  # tri-state verbatim: None/""/text
        if "show_title" in values:
            changes["show_title"] = values["show_title"]
        if "scale_um_per_cm" in values:
            scale = values["scale_um_per_cm"]
            changes["scale_um_per_cm"] = float(scale) if scale else None
        if "colorbar" in values:
            changes["colorbar"] = values["colorbar"]
        if all(getattr(panel, k) == v for k, v in changes.items()):
            return  # no-op edit — nothing changed, nothing dirties
        for k, v in changes.items():
            setattr(panel, k, v)
        self._after_inspector_mutation()
```

    Update its docstring (tri-state label, no-op guard, in-place item refresh instead of rebuild+reselect).
  - `_on_label_selected` sentinel fix: `current = "" if node.group_label in (None, "auto") else node.group_label` in the Row/Col branch.
- [ ] 4.5 `python3 -m pytest tests/test_gui_figure_builder.py -q` → all pass, including every pre-existing test (notably `test_override_widget_edit_preserves_suppressed_label_and_precise_clim` and the two malformed-no-mutation tests). Then `python3 tests/gui_smoke.py` → `[1]`–`[37]` pass (step [37] does not touch the override widgets, but proves the window still builds/renders/exports).
- [ ] 4.6 Docs (same commit):
  - `docs/Usage.md`: rewrite the "*Selected panel overrides*" bullet (~1347–1360) as "*Selected node*" — one sub-paragraph per page (Panel: existing fields + the new three-state Label control Auto letter / No label / Custom; Row: group-label control [Not a group / Auto letter / Custom], pinned height (0 = off), one-colorbar-for-group checkbox + shared colour limits `lo,hi` (blank = union of member ranges); Col: as Row plus shared-x checkbox and pinned width; Spacer/Text: box sizes and text). State the editing rules: each field applies independently, malformed text reports to the notes bar and changes nothing, and re-entering the current value changes nothing at all.
  - `docs/Codebase.md` row 898: update the override-editor portion — `QStackedWidget` pages, `_apply_node_field`/`_apply_pin_spin`/`_apply_shared_clim_text`/`_after_inspector_mutation`/`_label_override_value`, the tri-state label semantics, the no-op guard, and the in-place item-text refresh replacing rebuild+reselect for inspector edits. Also record the `_on_label_selected` sentinel fix. (Edit with several small unique substrings — the row is one enormous line.)
- [ ] 4.7 `ruff check . && ruff format .`; commit:
  `git add gui/figure_builder.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md`
  `git commit -m "feat(gui): figure-builder node inspector — per-type pages, shared colorbar/x/pin controls, 3-state labels, auto-sentinel fix"`

---

## Task 5 — outline selection persistence, stale canvas clear, suppressed-label affordance (spec §B, item 9)

**Files**
- `gui/figure_builder.py` — `_node_label` (572–585), `_rebuild_tree` (595–599), `delete_selected` (692–702), `render_now` (755–781), plus a new `select_node` near `_select_outline_panel` (815–833)
- `tests/test_gui_figure_builder.py`
- `docs/Usage.md` preview/outline paragraphs (~1303–1324); `docs/Codebase.md` `figure_builder.py` row 898

**Interfaces (Produces — Task 6 consumes)**
- `select_node(self, node) -> None` — **public**: selects the outline item whose stored `UserRole` node `is` the given object; no-op if absent/`None`.
- `_rebuild_tree` captures the selected node before teardown and re-selects it by identity after rebuild.
- `delete_selected` selects the deleted node's parent container afterwards.
- `render_now` with an empty panel list clears the canvas (`self._canvas is None`, `self._result is None`) via a new `_clear_canvas(self) -> None`, then sets the `"add panels to preview"` note.
- `_node_label` marks a `PanelDef.label == ""` panel as `"Panel: {id} (label off)"`.

**Steps**

- [ ] 5.1 Append failing tests to `tests/test_gui_figure_builder.py`:

```python
# -- selection persistence + stale canvas (T5) --------------------------------
def test_move_selected_is_repeatable_selection_persists():
    w = _win()
    w.add_panels([_panel("a"), _panel("b"), _panel("c")])
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(2))  # select "c"
    w.move_selected(-1)
    assert [getattr(x, "panel_id", None) for x in _row(w).children] == ["a", "c", "b"]
    assert getattr(w._selected_node(), "panel_id", None) == "c"  # survived the rebuild
    w.move_selected(-1)  # pressing ↑ again must keep working
    assert [getattr(x, "panel_id", None) for x in _row(w).children] == ["c", "a", "b"]


def test_delete_selects_parent_container():
    w = _win()
    w.add_col()
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    w.add_panels([_panel("a"), _panel("b")])  # into the selected Col
    col = w.recipe().layout.children[0]
    w.select_node(col.children[0])
    w.delete_selected()
    assert w._selected_node() is col


def test_label_edit_keeps_selection():
    w = _win()
    w.add_col()
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    col = w.recipe().layout.children[0]
    w.set_selected_label("G1")
    assert w._selected_node() is col


def test_render_now_clears_canvas_when_no_panels_left(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    assert w.render_now() is not None and w._canvas is not None
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    w.delete_selected()
    assert w.render_now() is None
    assert w._canvas is None and w._result is None
    assert "add panels" in w._notes_label.text()


def test_outline_marks_suppressed_label():
    w = _win()
    w.add_panels([_panel("a")])
    w.recipe().panels[0].label = ""
    w._rebuild_tree()
    assert "label off" in w._tree.topLevelItem(0).child(0).text(0)
```

- [ ] 5.2 `python3 -m pytest tests/test_gui_figure_builder.py -q` — the new tests fail: repeatable-move fails on the second move (selection lost → no-op), `select_node` is missing (`AttributeError`), canvas test fails on `w._canvas is None`, suppressed-label fails on the missing marker.
- [ ] 5.3 Implement in `gui/figure_builder.py`:

```python
    def select_node(self, node) -> None:
        """Select the outline item holding exactly *node* (identity), if present."""
        if node is None:
            return

        def walk(item):
            if item.data(0, Qt.ItemDataRole.UserRole) is node:
                return item
            for i in range(item.childCount()):
                found = walk(item.child(i))
                if found is not None:
                    return found
            return None

        root = self._tree.topLevelItem(0)
        if root is None:
            return
        item = walk(root)
        if item is not None:
            self._tree.setCurrentItem(item)

    def _rebuild_tree(self) -> None:
        selected = self._selected_node()  # captured BEFORE teardown (spec §B)
        self._tree.clear()
        root_item = self._build_item(self._recipe.layout)
        self._tree.addTopLevelItem(root_item)
        self._tree.expandAll()
        if selected is not None:
            self.select_node(selected)
```

  - `delete_selected`: after `self._after_mutation()`, add `self.select_node(container)` (the `(container, idx)` pair is already in scope).
  - `_clear_canvas` + `render_now` head:

```python
    def _clear_canvas(self) -> None:
        """Item 9: drop the previous preview so a stale figure never lingers
        behind the 'add panels to preview' note."""
        if self._canvas is not None:
            self._preview_layout.removeWidget(self._canvas)
            self._canvas.deleteLater()
            self._canvas = None
        self._result = None
```

    and in `render_now`, the empty-recipe branch becomes:

```python
        if not self._recipe.panels:
            self._clear_canvas()
            self._notes_label.setText("add panels to preview")
            return None
```

  - `_node_label` `PanelRef` branch:

```python
        if isinstance(node, PanelRef):
            panel = self._recipe.panel_by_id().get(node.panel_id)
            if panel is not None and panel.label == "":
                return f"Panel: {node.panel_id} (label off)"
            suffix = f" ({panel.label})" if panel and panel.label else ""
            return f"Panel: {node.panel_id}{suffix}"
```

- [ ] 5.4 `python3 -m pytest tests/test_gui_figure_builder.py -q` → all pass (watch `test_move_and_delete_edit_the_recipe` and `test_delete_row_purges_nested_panel_defs_and_gutter_renders` — delete now re-selects the parent, which those tests tolerate). `python3 tests/gui_smoke.py` → `[1]`–`[37]`.
- [ ] 5.5 Docs (same commit): `docs/Usage.md` — in the live-preview/outline paragraphs, state that the outline keeps the edited node selected across move/group/label edits (↑/↓ can be pressed repeatedly), that deleting a node selects its parent container, that a panel with its label switched off shows "(label off)" in the outline, and that deleting the last panel clears the preview canvas. `docs/Codebase.md` row 898 — document `select_node`, the `_rebuild_tree` capture/re-select, `delete_selected`'s parent selection, `_clear_canvas`, and the `_node_label` marker.
- [ ] 5.6 `ruff check . && ruff format .`; commit:
  `git add gui/figure_builder.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md`
  `git commit -m "feat(gui): outline selection persistence + stale preview clear + suppressed-label marker"`

---

## Task 6 — figure-2 GUI acceptance test, docs finalization, Codebase run-on split (spec §C, item 14)

**Files**
- `tests/test_gui_figure_builder.py` (new test at end; reuses `tests/test_compose_acceptance.py::_write_profiles_three_fields` — `tests/` is a package, `tests/__init__.py` exists)
- `docs/Codebase.md` — line 444 run-on entry (item 14) + coherence pass over `### dfxm/compose` (474–840) and row 898
- `docs/Usage.md` — coherence pass over `## Figure builder` (1259–1396)

**Interfaces**
- Consumes: `FigureBuilderWindow` public surface only — `add_panels`, `add_col`, `select_node` (T5), `set_selected_label`, the T4 inspector widgets (`_ov_label_mode`/`_ov_label`, `_col_shared_x`, `_col_group_mode`/`_col_group_label`), `_apply_panel_overrides`, `render_now`; `measured_box_in(fig, ax)` from `dfxm.common.plotting`; the acceptance fixture `_write_profiles_three_fields(path)`.
- Geometry pins are **identical** to `test_acceptance_figure_2_ragged_dual_scale`: `ext_x, ext_y = 24.5, 19.5`; `map_scale, trace_scale, trace_h_cm = 5.0, 2.0, 2.0`; `len_a = hypot(16, 12)`, `len_b = hypot(30, 18)`.

**Steps**

- [ ] 6.1 Append the acceptance test to `tests/test_gui_figure_builder.py`:

```python
# -- figure-2 acceptance: authored entirely through window methods (T6) -------
def test_figure2_authored_through_window_methods(tmp_path):
    """Spec §C: acceptance figure 2 — ragged 3 columns, shared_x trace stacks
    with group labels, split map/trace scales — authored purely through
    FigureBuilderWindow methods (no JSON editing), rendering with exactly the
    geometry tests/test_compose_acceptance.py's figure-2 test pins."""
    import numpy as np

    from dfxm.common.plotting import measured_box_in
    from dfxm.compose.recipe import PanelSource
    from tests.test_compose_acceptance import _write_profiles_three_fields

    h5 = _write_profiles_three_fields(tmp_path / "obl.h5")
    job_a = {"name": "obl", "offset_um": 0.0, "start_uv": [-8.0, -6.0], "end_uv": [8.0, 6.0]}
    job_b = {"name": "obl", "offset_um": 0.0, "start_uv": [-15.0, -9.0], "end_uv": [15.0, 9.0]}
    len_a = float(np.hypot(16.0, 12.0))  # 20 µm
    len_b = float(np.hypot(30.0, 18.0))  # ~34.99 µm
    fields = ["mosa_com_chi", "strain", "raw_mosa_sum"]
    map_scale, trace_scale, trace_h_cm = 5.0, 2.0, 2.0

    w = _track(
        FigureBuilderWindow(
            lambda: {},
            PlotStyle(
                scale_um_per_cm=map_scale,
                trace_scale_um_per_cm=trace_scale,
                trace_height_cm=trace_h_cm,
                show_title=False,
            ),
        )
    )
    root = w.recipe().layout

    # column 1: the two reference maps, labelled via the inspector
    w.add_col()
    col1 = root.children[0]
    w.select_node(col1)
    w.add_panels(
        [
            PanelDef(
                "A1",
                PanelSource(h5, "profiles_ref", {"job": job_a, "field": "mosa_com_chi"}),
                roi=(10, 50, 10, 60),
            ),
            PanelDef(
                "B1",
                PanelSource(h5, "profiles_ref", {"job": job_b, "field": "mosa_com_chi"}),
                roi=(10, 50, 10, 60),
            ),
        ]
    )
    w.select_node(col1.children[0])
    w._ov_label_mode.setCurrentIndex(2)  # Custom… (the 3-state widget path)
    w._ov_label.setText("A1")
    w._apply_panel_overrides(w.recipe().panel_by_id()["B1"], {"label": "B1"})

    # columns 2/3: shared-x trace stacks with group labels A2/B2
    for tag, job, glabel in (("a", job_a, "A2"), ("b", job_b, "B2")):
        w.select_node(root)
        w.add_col()
        col = root.children[-1]
        w.select_node(col)
        w.add_panels(
            [
                PanelDef(
                    f"t_{tag}_{v}", PanelSource(h5, "profiles_trace", {"job": job, "field": v})
                )
                for v in fields
            ]
        )
        w.select_node(col)
        w._col_shared_x.setChecked(True)  # the NEW inspector authoring path
        if tag == "a":
            w._col_group_mode.setCurrentIndex(2)  # Custom… on the Col page
            w._col_group_label.setText(glabel)
        else:
            w.set_selected_label(glabel)  # the outline Label… path — both must work
        assert col.shared_x is True and col.group_label == glabel

    res = w.render_now()
    assert res is not None and res.n_rendered == 8, w._notes_label.text()
    fig = res.figure

    # geometry pins — identical to test_acceptance_figure_2_ragged_dual_scale
    ext_x, ext_y = 24.5, 19.5
    for pid in ("A1", "B1"):
        bw, bh = measured_box_in(fig, res.axes_by_id[pid])
        assert abs(bw - ext_x / map_scale / 2.54) < 0.005 * bw, pid
        assert abs(bh - ext_y / map_scale / 2.54) < 0.005 * bh, pid
    for tag, length in (("a", len_a), ("b", len_b)):
        for vid in fields:
            bw, bh = measured_box_in(fig, res.axes_by_id[f"t_{tag}_{vid}"])
            assert abs(bw - length / trace_scale / 2.54) < 0.005 * bw, (tag, vid)
            assert abs(bh - trace_h_cm / 2.54) < 0.005 * bh, (tag, vid)

    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "A1" in texts and "B1" in texts and "A2" in texts and "B2" in texts
    for tag in ("a", "b"):
        for vid in fields[:-1]:
            assert res.axes_by_id[f"t_{tag}_{vid}"].get_xlabel() == ""
        assert res.axes_by_id[f"t_{tag}_{fields[-1]}"].get_xlabel() != ""
        xs = {round(res.axes_by_id[f"t_{tag}_{v}"].get_position().x0, 5) for v in fields}
        assert len(xs) == 1, tag
    xa = max(res.axes_by_id[f"t_a_{v}"].get_position().x1 for v in fields)
    xb = {round(res.axes_by_id[f"t_b_{v}"].get_position().x0, 5) for v in fields}
    assert len(xb) == 1 and min(xb) > xa
    assert not any("scale is off" in n for n in res.notes)
```

- [ ] 6.2 `python3 -m pytest tests/test_gui_figure_builder.py::test_figure2_authored_through_window_methods -q` — must pass first try if T1–T5 are correct; if it fails, the failure IS a finding: fix the window/solver, not the test's geometry expectations (they are the merged acceptance contract).
- [ ] 6.3 Item 14 — split the `_draw_reference_image` run-on entry, `docs/Codebase.md` line 444. Read the exact line first. Replace the single bullet with a lead line + four sub-bullets, preserving all existing facts and sentences (reflow, don't rewrite): (1) signature + "shared map-panel renderer used by both `build_companion_figure` and `render_single`"; (2) the `fixed_scale_um_per_cm` keyword-only pass-through sentences; (3) the `scale_bar` convention sentences incl. the `dfxm.compose.adapters.draw_panel` caller rationale; (4) the extent-pinning/autoscale-off rationale sentences (task 9 finding).
- [ ] 6.4 Docs finalization pass: re-read `docs/Usage.md` 1259–1396 and `docs/Codebase.md` 474–840 + row 898 once each; fix any stale statement the sweep obsoleted (e.g. "the first axes is silently left empty", "reads `recipe.style` only", per-panel-only override editor phrasing) and confirm every T1–T5 doc addition landed coherently.
- [ ] 6.5 **Final verify (the branch gate):**
  - `python3 -m pytest -q` → expect **735 + all new tests passed / 13 skipped / 0 warnings** (record the exact new-test count from the run; any warning is a failure).
  - `ruff check . && ruff format .` → clean, no reformats.
  - `python3 tests/gui_smoke.py` → `[1]`–`[37]` all print, ends `GUI SMOKE PASSED` (no new smoke step: inspector interactions are covered by the pytest GUI tests, per spec §G).
  - `git status` → clean tree; `git log --oneline master..HEAD` → 6 commits.
- [ ] 6.6 Commit:
  `git add tests/test_gui_figure_builder.py docs/Codebase.md docs/Usage.md`
  `git commit -m "test(gui): figure-2 authored end-to-end through the builder + docs finalization (Codebase run-on split)"`

---

## Risks & edge cases

- **Pin-division equal split** assumes homogeneous stacked children (the spec's rule). Heterogeneous stacks get equal slices with per-panel implied-scale notes — visible, not silent; accepted by the spec.
- **`_resolve_scale_bar_kwargs` placeholder-target degrade vs raise:** the split follows the codebase's authored-vs-data-availability convention (adapters docstring): wrong-kind/unplaced = authoring → raise; placeholder = data → note. Reviewers should check this against spec item 3's wording ("non-map or unplaced target → StageUserError") — a missing-file target is neither authoring error nor silently ignorable, so the note path is the deliberate interpretation.
- **Item 7 orphan skip** changes `loader_cache` contents for recipes with orphans — the GUI's `_purge_orphaned_panels` already prevents orphans from the builder itself; only hand-edited recipes are affected, and the note makes it visible.
- **`_apply_panel_overrides` label semantics change** (`values["label"]` verbatim): any out-of-tree caller relying on `"" → None` would change behaviour; a repo-wide grep in T4 (`grep -rn "_apply_panel_overrides" gui tests`) must show only `figure_builder.py` and the GUI tests.
- **Inspector signals vs Qt offscreen:** `QDoubleSpinBox.setValue` with an equal value emits no signal, `QLineEdit.setText` emits once, `QComboBox.setCurrentIndex` with the same index emits nothing — the tests above rely on these Qt guarantees; the no-op guard in `_apply_node_field` is the backstop either way.
- **Per-keystroke container edits**: safe because inspector edits refresh the item text in place instead of rebuilding the tree (T4 design); the T4 `test_inspector_edit_updates_item_text_in_place_without_rebuild` pins it.
- **Cross-test import** (`from tests.test_compose_acceptance import _write_profiles_three_fields`) relies on `tests/__init__.py` (present) — run pytest from the repo root as always.
- **Docs edits in `Codebase.md`** are single-line table rows and a single-line 444 entry — always Read the exact bytes first and edit via short unique substrings; em-dash `hint=` strings in `dfxm/stages/*.py` are not touched by this project.
