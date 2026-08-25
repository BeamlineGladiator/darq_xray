# Trace autoscale + text-collision warnings — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Spec: `docs/superpowers/specs/2026-08-17-trace-autoscale-collisions-design.md` (approved 2026-08-17). Base: master `4878b47`.

## Goal

Two additive figure-builder/composer features:

1. **`ComposeStyle.trace_autoscale`** (default `False`): a sizing pass that rescales each composed trace cell (both dimensions, box ratio kept) so its width matches the widest map cell in its innermost enclosing `Col` (fallback: widest map in the whole figure; no maps → untouched; pin-sized traces → untouched; placeholders → untouched). One implied-scale note per rescaled trace.
2. **Post-render text-collision advisory**: at the very end of `render_recipe` (final geometry), detect visible text artists from *different* axes whose window extents overlap, and append at most ONE advisory note naming the colliding panels plus condition-driven suggestions. Cost-guarded (>400 text artists → skip note). Never an error.

## Architecture

- `dfxm/compose/recipe.py` — new `ComposeStyle.trace_autoscale: bool = False`. Serialization is automatic (`asdict` / `ComposeStyle(**d.get("compose", {}))`); old recipes load `False`; `RECIPE_VERSION` stays 1; no validation (bool).
- `dfxm/compose/layout.py` — `SizedCell` gains `pinned: bool = False`, set in **both** pin branches of `_trace_cell` (width-pin early return and the height-pin return). Two new Qt-free functions:
  - `trace_column_targets(recipe, cells) -> dict[int, float | None]` — per non-pinned trace leaf, the target width in inches (innermost enclosing `Col` containing a `kind == "map"` cell, walking outward; else figure-wide widest map; else `None`). Shared by the autoscale pass and the render-side suggestion check.
  - `autoscale_traces(recipe, cells, data_by_id, notes) -> None` — the match-column-width pass. **Deliberate deviation from the spec's literal signature** (`(recipe, cells, notes)`): the spec's mandated note text is `implied scale {length/(w_in*2.54):.4g} µm/cm`, and `length_um` lives only in `PanelData` — so `data_by_id` is added, mirroring `size_cells(recipe, style, data_by_id, notes)`. Record this in the commit message body if a reviewer asks.
- `dfxm/compose/render.py` — `render_recipe` calls `autoscale_traces` immediately after `size_cells`, gated on `recipe.compose.trace_autoscale` (the pass also early-returns on a flag-off call, so both the spec's "called only when true" and its "flag off: no-op" error-handling line hold). At the very end (after `_align_axis_labels`, the gutter scale-bar draw, and the drift-note loop — geometry final), `notes` is extended with `_detect_text_collisions(...)`. Export runs through `render_recipe`, so exports get both features for free. New module constants `_MAX_COLLISION_TEXTS = 400`, `_TRACE_TINY_FRACTION = 0.4`; new helpers `_axes_texts`, `_overlap_area`, `_detect_text_collisions(fig, axes_by_owner, pre_suggestions=())`, `_collision_presuggestions(recipe, cells)`. **Deliberate deviation**: the detector takes an owner-name→axes map (panel `title or id`, uniquified; bar axes named `"colorbar (<group>)"`) plus a `pre_suggestions` list, because the note must print display names and the trace-tiny suggestion needs recipe/cell context the bare `(fig, axes_by_id)` signature cannot supply.
- `gui/figure_builder.py` — `self._compose_trace_autoscale` (`QCheckBox`, already-imported class) between the padding row and the **Colourbars** heading; wired through `_on_compose_edited` / `_load_compose_into_widgets` exactly like every other compose widget.
- `tests/gui_smoke.py` — extend step **[40]** with the toggle + re-render; step **[41]** (3-D) stays LAST and untouched.

## Tech stack

Python 3, dataclasses, matplotlib **explicit `Figure` API only** (never pyplot), h5py + numpy in test fixtures, PySide6 only under `gui/` and `tests/`, pytest.

## Global constraints

- **Qt-free `dfxm/`** — no PySide6/pyvista imports anywhere under `dfxm/`.
- **Explicit-Figure matplotlib only**; never `pyplot`, never `matplotlib.use(...)`.
- **Docs-in-same-change contract**: every behaviour/structure change commits with its matching `docs/Usage.md` (user-visible) and/or `docs/Codebase.md` (code-structure) update in the SAME commit.
- **Full suite**: `DISPLAY= python3 -m pytest -q` — this shell's DISPLAY has a broken GL stack that kills the vtk tests; the blank `DISPLAY=` is mandatory. Baseline at master `4878b47`: **949 passed / 13 skipped**. Every task must end with `failed=0` and only ever grow the passed count.
- **Lint**: `ruff check . && ruff format .` before every commit (a Write/Edit hook also auto-formats; still run the check).
- **Smoke**: `python3 tests/gui_smoke.py` — the 3-D step [41] must remain the last step.
- **This repo has no git remote** — no pull/push/PR anywhere.
- **Never reconstruct `old_string` from memory** — Read (or grep the exact bytes of) every edit target first; markdown prose reflows and the docs lines are long. Line numbers below are as of master `4878b47` — always locate by content.
- Work on a feature branch (e.g. `trace-autoscale-collisions`), one commit per task.

---

## Task 1 — `ComposeStyle.trace_autoscale` model field

**Files**
- `dfxm/compose/recipe.py` — `ComposeStyle` dataclass (lines 18–28; last field is `colorbar_pos` at line 28).
- `tests/test_compose_recipe.py` — append after `test_colorbar_mode_fields_round_trip_and_old_recipe_defaults` (ends line 209).
- `docs/Codebase.md` — `ComposeStyle` bullet (lines 532–538).

**Interfaces**
- Produces: `ComposeStyle.trace_autoscale: bool = False`. Round-trips through `recipe_to_json`/`recipe_from_json` automatically (`asdict` + `ComposeStyle(**d.get("compose", {}))` — no serializer change needed). `RECIPE_VERSION` stays 1. `validate_recipe` untouched (bool needs no validation).
- Consumed by: Task 2's `autoscale_traces`, Task 3's `render_recipe` gate + `_collision_presuggestions`, Task 4's checkbox.

**Steps**

- [ ] Append this failing test at the end of `tests/test_compose_recipe.py`:

```python
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
```

- [ ] Run `DISPLAY= python3 -m pytest -q tests/test_compose_recipe.py` — expect **1 failed**: `AttributeError: 'ComposeStyle' object has no attribute 'trace_autoscale'` (on the `r2` read-back; setting the attribute on the un-slotted dataclass succeeds, but `asdict` never serializes it) or `KeyError: 'trace_autoscale'` at the `pop`.
- [ ] In `dfxm/compose/recipe.py`, add one field to `ComposeStyle`, directly after the `colorbar_pos` line (line 28):

```python
    colorbar_pos: str = "right"  # one of COLORBAR_POSITIONS (united mode only)
    trace_autoscale: bool = False  # autoscale trace cells to their column's map width
```

- [ ] Run `DISPLAY= python3 -m pytest -q tests/test_compose_recipe.py` — expect **all pass** (23 passed).
- [ ] Docs (same commit): in `docs/Codebase.md`, Read the `ComposeStyle` bullet (grep `composer-level look knobs` for its location; lines 532–538 at base). Replace its final two lines

```
  Both are additive fields: absent in old recipe JSON → the defaults above via
  `ComposeStyle(**d.get("compose", {}))`, no loader change needed.
```

with:

```
  and `trace_autoscale` (bool, default `False` — when true, `render_recipe`
  runs `layout.autoscale_traces` right after `size_cells`, rescaling each
  trace cell to its column's widest map width; a bool needs no validation).
  All three are additive fields: absent in old recipe JSON → the defaults
  above via `ComposeStyle(**d.get("compose", {}))`, no loader change needed.
```

  (Read the surrounding bullet first and splice so the `colorbar_pos` clause flows into "and `trace_autoscale` …" — adjust the connecting comma/period to keep the sentence grammatical.)
- [ ] `ruff check . && ruff format .` — clean.
- [ ] `DISPLAY= python3 -m pytest -q` — expect **950 passed / 13 skipped / 0 failed**.
- [ ] Commit:

```bash
git add dfxm/compose/recipe.py tests/test_compose_recipe.py docs/Codebase.md
git commit -m "feat(compose): ComposeStyle.trace_autoscale recipe field (docs synced)"
```

**Verification** — round-trip + old-recipe default pinned by the new test; suite green.

**Risks/edge cases** — none beyond serialization (covered). `asdict` handles bools natively; JSON `true/false` round-trips exactly.

---

## Task 2 — layout pass: `SizedCell.pinned` + `trace_column_targets` + `autoscale_traces`

**Files**
- `dfxm/compose/layout.py` — `SizedCell` (lines 67–79), `_trace_cell` pin branches (width-pin return at line 206, height-pin return at line 232), import line 26, new functions inserted after `size_cells` (after line 252, before `measure_cells`).
- `tests/test_compose_layout.py` — import line 10; new tests appended after `test_zero_length_trace_under_width_pin_still_placeholder` (line 252), before the `# -- measure/align/place` section.
- `docs/Codebase.md` — `SizedCell` bullet (lines 697–703) and new bullets before the placement-half paragraph (grep `The layout solver's placement half`, line ~805).

**Interfaces**
- Consumes: `FigureRecipe` (layout tree + `compose.trace_autoscale`), `dict[int, SizedCell]` from `size_cells`, `data_by_id: dict[str, PanelData]` (`length_um` for traces), `notes: list[str]`.
- Produces:
  - `SizedCell.pinned: bool = False` — new field inserted **after `h_in`**, before the placement-pass fields (safe: every call site passes at most 5 positional args; `ax` is always keyword).
  - `trace_column_targets(recipe, cells) -> dict[int, float | None]` — keyed by `id(leaf)`; only non-pinned, non-placeholder trace leaves appear.
  - `autoscale_traces(recipe, cells, data_by_id, notes) -> None` — mutates `cells` in place; appends the spec's note per rescaled trace; strict no-op when the flag is off / no traces / no maps.

**Steps**

- [ ] In `tests/test_compose_layout.py`, change the import at line 10 from

```python
from dfxm.compose.layout import SizedCell, measure_cells, place_tree, size_cells
```

to

```python
from dfxm.compose.layout import (
    SizedCell,
    autoscale_traces,
    measure_cells,
    place_tree,
    size_cells,
    trace_column_targets,
)
```

- [ ] Append these failing tests after `test_zero_length_trace_under_width_pin_still_placeholder` (immediately before the `# -- measure/align/place` comment block):

```python
# -- trace autoscale (match column width) -------------------------------------


def _autoscale_recipe(layout, panels):
    r = _recipe(layout, panels)
    r.compose.trace_autoscale = True
    return r


def test_autoscale_matches_trace_to_column_map_width_ratio_kept_with_note():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, t])
    r = _autoscale_recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])
    data = {"m": _map_data(), "t": _trace_data(30.0)}
    cells = size_cells(r, style, data, notes := [])
    ct = cells[id(t)]
    w0, h0 = ct.w_in, ct.h_in  # natural: 30/5 = 6 cm wide, 2 cm tall
    autoscale_traces(r, cells, data, notes)
    assert abs(ct.w_in - cells[id(m)].w_in) < 1e-9  # matched (downscale: 6 cm -> 2 cm)
    assert abs(ct.h_in / ct.w_in - h0 / w0) < 1e-9  # box ratio kept
    # implied scale = 30 µm / 2 cm = 15 µm/cm
    assert any("autoscaled to column width" in n and "15" in n for n in notes)


def test_autoscale_falls_back_to_figure_widest_map_and_upscales():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Row([m, t])  # trace has NO enclosing Col
    r = _autoscale_recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])
    data = {"m": _map_data(40.0, 10.0), "t": _trace_data(10.0)}  # trace 2 cm, map 4 cm
    cells = size_cells(r, style, data, notes := [])
    autoscale_traces(r, cells, data, notes)
    assert abs(cells[id(t)].w_in - cells[id(m)].w_in) < 1e-9  # upscaled: match, not grow-only
    assert any("autoscaled to column width" in n for n in notes)


def test_autoscale_prefers_innermost_enclosing_col_with_a_map():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    mo, mi, t = PanelRef("mo"), PanelRef("mi"), PanelRef("t")
    layout = Col([mo, Col([mi, t])])
    r = _autoscale_recipe(layout, [_panel("mo"), _panel("mi"), _panel("t", "profiles_trace")])
    data = {"mo": _map_data(40.0, 10.0), "mi": _map_data(20.0, 10.0), "t": _trace_data(30.0)}
    cells = size_cells(r, style, data, notes := [])
    autoscale_traces(r, cells, data, notes)
    # the inner Col's 2 cm map wins over the outer Col's 4 cm one
    assert abs(cells[id(t)].w_in - cells[id(mi)].w_in) < 1e-9


def test_autoscale_all_trace_figure_untouched_no_note():
    style = PlotStyle(trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    t1, t2 = PanelRef("t1"), PanelRef("t2")
    layout = Col([t1, t2])
    r = _autoscale_recipe(
        layout, [_panel("t1", "profiles_trace"), _panel("t2", "profiles_trace")]
    )
    data = {"t1": _trace_data(30.0), "t2": _trace_data(10.0)}
    cells = size_cells(r, style, data, notes := [])
    w1, w2 = cells[id(t1)].w_in, cells[id(t2)].w_in
    autoscale_traces(r, cells, data, notes)
    assert (cells[id(t1)].w_in, cells[id(t2)].w_in) == (w1, w2)
    assert not any("autoscaled" in n for n in notes)


def test_autoscale_skips_pin_sized_trace():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, Col([t], pinned_width_cm=6.0)])
    r = _autoscale_recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])
    data = {"m": _map_data(), "t": _trace_data(30.0)}
    cells = size_cells(r, style, data, notes := [])
    ct = cells[id(t)]
    assert ct.pinned is True
    assert id(t) not in trace_column_targets(r, cells)  # pinned cells never targeted
    w0 = ct.w_in
    autoscale_traces(r, cells, data, notes)
    assert ct.w_in == w0  # the pin wins
    assert not any("autoscaled" in n for n in notes)


def test_size_cells_marks_pin_sized_traces_and_not_free_ones():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    t = PanelRef("t")
    layout = Row([t], pinned_height_cm=4.0)  # the height-pin branch
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]), style, {"t": _trace_data(30.0)}, []
    )
    assert cells[id(t)].pinned is True
    t2 = PanelRef("t")
    cells2 = size_cells(
        _recipe(t2, [_panel("t", "profiles_trace")]), style, {"t": _trace_data(30.0)}, []
    )
    assert cells2[id(t2)].pinned is False


def test_autoscale_leaves_placeholder_trace_untouched():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, t])
    r = _autoscale_recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])
    data = {"m": _map_data(), "t": _trace_data(0.0)}  # degenerate -> placeholder
    cells = size_cells(r, style, data, notes := [])
    autoscale_traces(r, cells, data, notes)
    assert cells[id(t)].kind == "placeholder"
    assert (cells[id(t)].w_in, cells[id(t)].h_in) == (4.0 / 2.54, 3.0 / 2.54)


def test_autoscale_flag_off_is_strict_noop():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, t])
    r = _recipe(layout, [_panel("m"), _panel("t", "profiles_trace")])  # flag stays False
    data = {"m": _map_data(), "t": _trace_data(30.0)}
    cells = size_cells(r, style, data, notes := [])
    w0 = cells[id(t)].w_in
    autoscale_traces(r, cells, data, notes)
    assert cells[id(t)].w_in == w0
    assert not any("autoscaled" in n for n in notes)
```

- [ ] Run `DISPLAY= python3 -m pytest -q tests/test_compose_layout.py` — expect **collection error**: `ImportError: cannot import name 'autoscale_traces' from 'dfxm.compose.layout'`.
- [ ] Implement in `dfxm/compose/layout.py`:

  1. Change the import at line 26 from `from .recipe import Col, PanelRef, Row, Spacer, TextCell` to:

```python
from .recipe import Col, PanelRef, Row, Spacer, TextCell, iter_leaves
```

  2. In `SizedCell`, insert after the `h_in: float` line (line 74):

```python
    # True when size_cells sized this cell from a pinned row height / column
    # width — autoscale_traces never touches pinned cells (pins win).
    pinned: bool = False
```

  3. In `_trace_cell`, change the width-pin return (line 206) `return SizedCell(leaf, panel, "trace", w, h)` to:

```python
            return SizedCell(leaf, panel, "trace", w, h, pinned=True)
```

  and the height-pin return (line 232) `return SizedCell(leaf, panel, "trace", box[0], pinned_h_in)` to:

```python
            return SizedCell(leaf, panel, "trace", box[0], pinned_h_in, pinned=True)
```

  (Read both sites first; the map-cell pin branches are deliberately left unmarked — a map's pin fixes its aspect and autoscale never reads map `pinned`.)

  4. Insert after `size_cells` (after the `walk(recipe.layout, None, None); return cells` block, before `measure_cells`):

```python
def trace_column_targets(recipe, cells):
    """Target width (inches) per autoscalable trace leaf, keyed by ``id(leaf)``.

    The target is the widest ``kind == "map"`` cell under the trace's
    innermost enclosing ``Col`` that contains one (walking outward through
    enclosing ``Col``s), falling back to the widest map cell in the whole
    figure, or ``None`` when the figure has no map cells at all. Pinned
    (``SizedCell.pinned``) and placeholder trace cells are excluded entirely.
    Shared by :func:`autoscale_traces` and render.py's collision-note
    suggestion check.
    """

    def widest_map(node) -> float:
        best = 0.0
        for leaf in iter_leaves(node):
            cell = cells.get(id(leaf))
            if cell is not None and cell.kind == "map":
                best = max(best, cell.w_in)
        return best

    figure_widest = widest_map(recipe.layout)
    targets: dict[int, float | None] = {}

    def walk(node, col_stack):
        if isinstance(node, Col):
            for child in node.children:
                walk(child, col_stack + [node])
        elif isinstance(node, Row):
            for child in node.children:
                walk(child, col_stack)
        else:
            cell = cells.get(id(node))
            if cell is None or cell.kind != "trace" or cell.pinned:
                return
            target = None
            for col in reversed(col_stack):  # innermost enclosing Col first
                w = widest_map(col)
                if w > 0.0:
                    target = w
                    break
            if target is None and figure_widest > 0.0:
                target = figure_widest
            targets[id(node)] = target

    walk(recipe.layout, [])
    return targets


def autoscale_traces(recipe, cells, data_by_id, notes) -> None:
    """Match each trace cell's width to its column's widest map cell.

    Run by ``render_recipe`` immediately after :func:`size_cells`, only when
    ``recipe.compose.trace_autoscale`` is true (the early return below also
    makes a flag-off call a strict no-op). Scales BOTH dimensions by the same
    factor (box ratio kept; up- and down-scaling both apply — the option
    means "match", not "grow only"). Pinned trace cells, placeholders, and
    figures without any map cell are never touched. Appends one
    implied-scale note per rescaled trace; ``data_by_id`` supplies each
    trace's ``length_um`` for that note's arithmetic.
    """
    if not recipe.compose.trace_autoscale:
        return
    targets = trace_column_targets(recipe, cells)
    for leaf in iter_leaves(recipe.layout):
        target = targets.get(id(leaf))
        if target is None:
            continue
        cell = cells[id(leaf)]
        if cell.w_in <= 0.0 or abs(target - cell.w_in) <= 1e-12:
            continue  # degenerate width can't scale; equal width needs no work
        f = target / cell.w_in
        cell.w_in *= f
        cell.h_in *= f
        pid = cell.panel.id
        length = data_by_id[pid].length_um
        notes.append(
            f"panel {pid}: trace autoscaled to column width — "
            f"implied scale {length / (cell.w_in * 2.54):.4g} µm/cm"
        )
```

- [ ] Run `DISPLAY= python3 -m pytest -q tests/test_compose_layout.py` — expect **all pass** (30 passed).
- [ ] Docs (same commit), `docs/Codebase.md` (Read each target region first):
  - In the `SizedCell` bullet (lines 697–703), after the `w_in`/`h_in` mention, insert: `` `pinned` (bool, default `False` — set by `size_cells` when a TRACE cell's size came from a pinned row height / column width, i.e. both pin branches of `_trace_cell`; `autoscale_traces` skips pinned cells — pins outrank autoscale; map cells never set it). `` Splice grammatically into the existing sentence.
  - Immediately before the placement-half paragraph (grep `The layout solver's placement half`), insert:

```
- `trace_column_targets(recipe, cells) -> dict[int, float | None]` — for
  every non-pinned, non-placeholder trace leaf, the autoscale target width in
  inches: the widest `kind == "map"` cell under the trace's innermost
  enclosing `Col` that contains one (walking outward through enclosing
  `Col`s), else the widest map cell in the whole figure, else `None` (no map
  cells at all). Shared by `autoscale_traces` and `render.py`'s
  collision-note suggestion check (`_collision_presuggestions`).
- `autoscale_traces(recipe, cells, data_by_id, notes) -> None` — the
  match-column-width pass behind `ComposeStyle.trace_autoscale`, run by
  `render_recipe` immediately after `size_cells` when the flag is true (an
  early return also makes a flag-off call a strict no-op). Each targeted
  trace cell gets `f = target_w / w_in` applied to BOTH `w_in` and `h_in`
  (box ratio kept; both up- and down-scaling), with one note per rescale:
  `"panel {pid}: trace autoscaled to column width — implied scale {…} µm/cm"`
  (`data_by_id` is read only for the trace's `length_um` in that note).
  Pinned trace cells and placeholders are never touched.
```

- [ ] `ruff check . && ruff format .` — clean.
- [ ] `DISPLAY= python3 -m pytest -q` — expect **958 passed / 13 skipped / 0 failed**.
- [ ] Commit:

```bash
git add dfxm/compose/layout.py tests/test_compose_layout.py docs/Codebase.md
git commit -m "feat(compose): trace autoscale sizing pass + SizedCell.pinned (docs synced)"
```

**Verification** — the eight new tests pin: match+ratio+note text, upscale fallback, innermost-Col precedence, all-trace no-op, pin precedence (both pin branches), placeholder immunity, flag-off no-op.

**Risks/edge cases**
- Dataclass field ordering: `pinned` sits between `h_in` and `ax`; verified that no call site passes more than 5 positional arguments (`layout.py`, `render.py`, tests all use `ax=` keyword) — grep `SizedCell(` across the repo before editing to confirm nothing changed since base.
- `cell.panel.id` is safe: trace cells always carry their `PanelDef` (set in `_trace_cell`).
- Equal-width traces (target == w) are skipped without a note — "scaled" per spec means actually rescaled.

---

## Task 3 — renderer wiring + text-collision detector

**Files**
- `dfxm/compose/render.py` — import at line 24; autoscale call after `cells = size_cells(...)` (line 568); new module constants + 4 helpers (place them after `_align_axis_labels`, before `_wrap_bar_node`); detector wiring after the `box_drift_note` loop (lines 872–876), immediately before `return ComposeResult(...)` (line 878).
- `tests/test_compose_render.py` — new tests appended at end of file (line 740).
- `docs/Codebase.md` — `render_recipe` pipeline list (starts line 866; step 4 ends line 892; last step is the `box_drift_note` item near line 1079).
- `docs/Usage.md` — notes-bar paragraph (grep `A **notes bar** under the preview`, lines 1479–1485).

**Interfaces**
- Consumes: `autoscale_traces`, `trace_column_targets` from `.layout`; `iter_leaves` (already imported in render.py); the final `fig`, `axes_by_id`, `bar_specs`, `united_specs`, `panels_by_id`, `cells`.
- Produces:
  - `_MAX_COLLISION_TEXTS = 400`, `_TRACE_TINY_FRACTION = 0.4` (module constants).
  - `_axes_texts(ax) -> list[Text]` — visible, non-empty title/axis-label/tick-label/annotation artists.
  - `_overlap_area(a, b) -> float` — positive-area bbox intersection, 0 otherwise.
  - `_detect_text_collisions(fig, axes_by_owner, pre_suggestions=()) -> list[str]` — `[]` when clean; one summary note otherwise; one skip note past the cost guard. Never raises for layout content.
  - `_collision_presuggestions(recipe, cells) -> list[str]` — `["enable trace autoscale"]` iff flag off AND some trace `< 0.4 ×` its column target; else `[]`.
  - `render_recipe` behaviour: autoscale pass gated on the flag; notes extended by the detector on every render **and export** (export shares `render_recipe`).

**Steps**

- [ ] Append these failing tests at the end of `tests/test_compose_render.py`:

```python
# -- trace autoscale + text-collision advisory (2026-08-17 spec) ---------------


def test_render_trace_autoscale_matches_column_map_width_and_notes(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    pm = PanelDef(
        "m",
        PanelSource(h5, "slice_plane", {"volume_id": "raw_sum", "slice_name": "obl", "plane": 0}),
    )
    pt = PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    r = FigureRecipe(
        "auto",
        {"scale_um_per_cm": 10.0, "trace_scale_um_per_cm": 2.0, "show_title": False},
        ComposeStyle(trace_autoscale=True),
        Col([PanelRef("m"), PanelRef("t")]),
        [pm, pt],
    )
    res = render_recipe(r)
    from dfxm.common.plotting import measured_box_in

    wm, _hm = measured_box_in(res.figure, res.axes_by_id["m"])
    wt, _ht = measured_box_in(res.figure, res.axes_by_id["t"])
    assert abs(wt - wm) < 0.01 * wm  # trace matched to the column's map width
    assert any("autoscaled to column width" in n for n in res.notes)

    r.compose.trace_autoscale = False
    res_off = render_recipe(r)
    wt_off, _ = measured_box_in(res_off.figure, res_off.axes_by_id["t"])
    # discriminates: flag off keeps the physical trace box (~5.8 cm vs 2 cm)
    assert abs(wt_off - wm) > 0.05 * wm
    assert not any("autoscaled" in n for n in res_off.notes)


def _fig_with_two_axes():
    from matplotlib.figure import Figure

    fig = Figure(figsize=(4.0, 2.0), facecolor="white")
    ax1 = fig.add_axes([0.1, 0.2, 0.35, 0.6])
    ax2 = fig.add_axes([0.55, 0.2, 0.35, 0.6])
    return fig, ax1, ax2


def test_collision_detector_flags_cross_axes_overlap_with_suggestions():
    from dfxm.compose.render import _detect_text_collisions

    fig, ax1, ax2 = _fig_with_two_axes()
    # two texts from DIFFERENT axes pinned to the same figure spot -> collide
    ax1.text(0.5, 0.5, "left panel text", transform=fig.transFigure)
    ax2.text(0.5, 0.5, "right panel text", transform=fig.transFigure)
    notes = _detect_text_collisions(fig, {"A": ax1, "B": ax2}, ["enable trace autoscale"])
    assert len(notes) == 1
    assert "text overlaps between panels A and B" in notes[0]
    assert "collision(s)" in notes[0]
    assert "enable trace autoscale" in notes[0]
    assert "increase gutter" in notes[0] and "reduce font scale" in notes[0]


def test_collision_detector_ignores_same_axes_overlaps_and_clean_figures():
    from dfxm.compose.render import _detect_text_collisions

    fig, ax1, ax2 = _fig_with_two_axes()
    # overlapping texts on the SAME axes: matplotlib's business, not ours
    ax1.text(0.2, 0.5, "one", transform=fig.transFigure)
    ax1.text(0.2, 0.5, "two", transform=fig.transFigure)
    assert _detect_text_collisions(fig, {"A": ax1, "B": ax2}) == []


def test_collision_detector_cost_guard_skips_past_400_texts():
    from dfxm.compose.render import _detect_text_collisions

    fig, ax1, ax2 = _fig_with_two_axes()
    for i in range(401):
        ax1.text(0.1, 0.1, f"t{i}")
    notes = _detect_text_collisions(fig, {"A": ax1, "B": ax2})
    assert len(notes) == 1
    assert "text-collision check skipped (" in notes[0] and "text artists" in notes[0]


def test_collision_presuggestions_trace_tiny_only_when_flag_off():
    from dfxm.common.plotting import PlotStyle
    from dfxm.compose.adapters import PanelData
    from dfxm.compose.layout import size_cells
    from dfxm.compose.render import _collision_presuggestions

    m, t = PanelRef("m"), PanelRef("t")
    layout = Col([m, t])
    pm = PanelDef("m", PanelSource("/x.h5", "map_layer", {}))
    pt = PanelDef("t", PanelSource("/x.h5", "profiles_trace", {}))
    r = FigureRecipe("s", {}, ComposeStyle(), layout, [pm, pt])
    data = {
        "m": PanelData(kind="map_layer", ext_x_um=20.0, ext_y_um=10.0),
        "t": PanelData(kind="profiles_trace", length_um=1.0),
    }
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    cells = size_cells(r, style, data, [])
    # trace box 0.2 cm wide < 40% of the column's 2 cm map -> suggest autoscale
    assert _collision_presuggestions(r, cells) == ["enable trace autoscale"]
    r.compose.trace_autoscale = True
    assert _collision_presuggestions(r, cells) == []


def test_render_runs_collision_check_at_end_and_clean_figure_has_no_note(
    tmp_path, monkeypatch
):
    h5 = _write_obl(tmp_path / "obl.h5")
    res = render_recipe(_two_panel_recipe(h5))
    assert not any("text overlaps" in n for n in res.notes)  # spacious -> clean

    import dfxm.compose.render as render_mod

    seen = {}

    def _spy(fig, axes_by_owner, pre_suggestions=()):
        seen["owners"] = dict(axes_by_owner)
        return ["SENTINEL-COLLISION-NOTE"]

    monkeypatch.setattr(render_mod, "_detect_text_collisions", _spy)
    res2 = render_recipe(_two_panel_recipe(h5))
    assert "SENTINEL-COLLISION-NOTE" in res2.notes
    assert set(seen["owners"]) >= {"a", "b"}  # owners keyed by panel title-or-id
```

- [ ] Run `DISPLAY= python3 -m pytest -q tests/test_compose_render.py` — expect **failures**: the autoscale test fails on `abs(wt - wm) < 0.01 * wm` (no pass wired yet), the detector tests fail with `ImportError: cannot import name '_detect_text_collisions'`, the presuggestion test with `ImportError: cannot import name '_collision_presuggestions'`, the spy test with `AttributeError` from `monkeypatch.setattr` (no such attribute).
- [ ] Implement in `dfxm/compose/render.py`:

  1. Change the layout import (line 24) to:

```python
from .layout import SizedCell, autoscale_traces, measure_cells, place_tree, size_cells, trace_column_targets
```

  (ruff format will wrap it.)

  2. Add module constants next to `_IN_PER_CM` (line 27):

```python
_MAX_COLLISION_TEXTS = 400  # cost guard: skip the pairwise pass beyond this
_TRACE_TINY_FRACTION = 0.4  # "microscopic trace" threshold for the advisory
```

  3. After `cells = size_cells(recipe, style, data_by_id, notes)` (line 568), insert:

```python
    if recipe.compose.trace_autoscale:
        autoscale_traces(recipe, cells, data_by_id, notes)
```

  4. Insert the helpers after `_align_axis_labels` (after line 444, before `_wrap_bar_node`):

```python
def _axes_texts(ax):
    """Visible, non-empty text artists on *ax*: title, x/y axis labels, tick
    labels, and annotations/free texts (panel letters are annotations on the
    panel axes; a colorbar axes' label + tick numbers are its own axis
    artists, so bar axes need no special casing)."""
    arts = [ax.title, ax.xaxis.label, ax.yaxis.label]
    arts += list(ax.get_xticklabels()) + list(ax.get_yticklabels())
    arts += list(ax.texts)
    return [t for t in arts if t.get_visible() and t.get_text().strip()]


def _overlap_area(a, b) -> float:
    """Positive intersection area of two bboxes, else 0 (inverted/degenerate
    boxes — e.g. a tiny text shrunk past nothing by the 1 pt inset — never
    count)."""
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return w * h if (w > 0.0 and h > 0.0) else 0.0


def _detect_text_collisions(fig, axes_by_owner, pre_suggestions=()) -> list[str]:
    """Advisory cross-panel text-overlap check on the FINAL geometry.

    *axes_by_owner* maps a display name (panel ``title or id``, or a bar
    name) to its axes. A collision is two text artists with DIFFERENT parent
    axes whose window extents, each shrunk by 1 pt (hairline touches
    ignored), intersect with positive area; same-axes overlaps are
    matplotlib's own tick-layout business and are ignored. Prefilter: only
    text pairs whose parent axes' ``get_tightbbox`` boxes (expanded by 2 pt)
    intersect are compared. Returns at most ONE summary note (``[]`` when
    clean); a figure with more than ``_MAX_COLLISION_TEXTS`` text artists
    returns a skip note instead of running the O(n²) pass. Never an error.
    """
    collected = []
    n_texts = 0
    for owner, ax in axes_by_owner.items():
        texts = _axes_texts(ax)
        n_texts += len(texts)
        collected.append((owner, ax, texts))
    if n_texts > _MAX_COLLISION_TEXTS:
        return [f"text-collision check skipped ({n_texts} text artists)"]

    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    pad = fig.dpi / 72.0  # 1 pt in pixels

    groups = []
    for owner, ax, texts in collected:
        if not texts:
            continue
        bb = ax.get_tightbbox(ren)
        if bb is None:
            continue
        exts = [t.get_window_extent(ren).padded(-pad) for t in texts]
        groups.append((owner, bb.padded(2.0 * pad), exts))

    n_collisions = 0
    involved: list[str] = []
    for i, (owner_a, bb_a, exts_a) in enumerate(groups):
        for owner_b, bb_b, exts_b in groups[i + 1 :]:
            if _overlap_area(bb_a, bb_b) <= 0.0:
                continue  # prefilter: distant axes never compared text-by-text
            hits = sum(1 for ea in exts_a for eb in exts_b if _overlap_area(ea, eb) > 0.0)
            if hits:
                n_collisions += hits
                for owner in (owner_a, owner_b):
                    if owner not in involved:
                        involved.append(owner)
    if not n_collisions:
        return []
    if len(involved) == 1:  # unreachable in practice, defensive
        names = involved[0]
    else:
        names = ", ".join(involved[:-1]) + " and " + involved[-1]
    suggestions = [*pre_suggestions, "increase gutter", "reduce font scale"]
    return [
        f"text overlaps between panels {names} ({n_collisions} collision(s)) — "
        + "; ".join(suggestions)
    ]


def _collision_presuggestions(recipe, cells) -> list[str]:
    """Conditional lead suggestion for the collision note: recommend enabling
    trace autoscale when it is off and some trace rendered below
    ``_TRACE_TINY_FRACTION`` of its column's widest map width."""
    if recipe.compose.trace_autoscale:
        return []
    targets = trace_column_targets(recipe, cells)
    for leaf in iter_leaves(recipe.layout):
        target = targets.get(id(leaf))
        if target is None:
            continue
        if cells[id(leaf)].w_in < _TRACE_TINY_FRACTION * target:
            return ["enable trace autoscale"]
    return []
```

  5. At the end of `render_recipe`, after the `box_drift_note` loop (lines 872–876) and immediately before `return ComposeResult(...)`, insert:

```python
    # Text-collision advisory — very last, on the FINAL geometry (after
    # _align_axis_labels and the gutter/scale-bar draws). Runs on export too
    # via this shared path. Owners are display names: panel title-or-id
    # (uniquified — two panels sharing a title must not shadow each other in
    # the check) plus each shared/united bar axes.
    owners: dict[str, object] = {}

    def _owner_key(base):
        name, k = base, 2
        while name in owners:
            name = f"{base} #{k}"
            k += 1
        return name

    for pid, ax in axes_by_id.items():
        owners[_owner_key(panels_by_id[pid].title or pid)] = ax
    for _node, grp, _pids, _bar_leaf, bar_ax in bar_specs:
        owners[_owner_key(f"colorbar ({grp})" if grp else "colorbar")] = bar_ax
    for grp, _pids, _bar_leaf, bar_ax in united_specs:
        owners[_owner_key(f"colorbar ({grp})" if grp else "colorbar")] = bar_ax
    notes.extend(_detect_text_collisions(fig, owners, _collision_presuggestions(recipe, cells)))
```

- [ ] Run `DISPLAY= python3 -m pytest -q tests/test_compose_render.py` — expect **all pass** (existing 26 + 6 new).
- [ ] Docs (same commit), Read each region first:
  - `docs/Codebase.md`, `render_recipe` pipeline list: after step 4's paragraph (ends `…what size_cells actually decided.`, line 892), insert an item that keeps the existing numbering intact:

```
  4b. `autoscale_traces(recipe, cells, data_by_id, notes)` — only when
     `recipe.compose.trace_autoscale` is true — rescales every eligible
     trace cell to its column's widest map width (see `layout.py`) BEFORE
     any axes exist, so every downstream consumer (bar provisional boxes,
     margins, placement, per-panel scale bars, drift notes) sees the final
     trace boxes; a later `compose.pinned_width_cm` rescale multiplies all
     cells uniformly, preserving the match.
```

  - Same file, after the final pipeline item (the `box_drift_note` bullet near line 1079, ending `appended to notes (never raised).`), insert:

```
  - **Text-collision advisory** (very last, after `_align_axis_labels` and
    the gutter/scale-bar draws — the geometry is final): `render_recipe`
    builds an owner→axes map (each panel keyed by `title or id`, uniquified
    with ` #2`/` #3` on duplicate titles; each shared/united bar axes keyed
    `"colorbar (<group>)"`) and extends `notes` with
    `_detect_text_collisions(fig, owners, _collision_presuggestions(recipe, cells))`.
    Exports inherit the check via the shared `render_recipe` path.
- `_detect_text_collisions(fig, axes_by_owner, pre_suggestions=()) -> list[str]` —
  final-geometry cross-panel text-overlap check. Per axes it collects the
  visible, non-empty text artists (`_axes_texts`: title, x/y axis labels,
  tick labels, annotations/free texts — panel letters are annotations; a bar
  axes' label + ticks are its own axis artists). Cost guard first: more than
  `_MAX_COLLISION_TEXTS` (400) text artists total returns
  `"text-collision check skipped ({n} text artists)"` without drawing.
  Otherwise one draw, then a prefilter (per-axes `get_tightbbox` padded by
  2 pt — only text pairs whose parent axes' boxes intersect are compared)
  and the pairwise test: two texts with DIFFERENT parent axes whose
  `get_window_extent` rectangles, each shrunk by 1 pt, intersect with
  positive area (`_overlap_area`). Same-axes overlaps are ignored. Clean →
  `[]`; else exactly one note, `"text overlaps between panels {names}
  ({n} collision(s)) — {suggestions}"`, suggestions = `pre_suggestions` +
  `"increase gutter"` + `"reduce font scale"`. Never an error.
- `_collision_presuggestions(recipe, cells) -> list[str]` —
  `["enable trace autoscale"]` when `compose.trace_autoscale` is off and some
  non-pinned trace cell rendered below `_TRACE_TINY_FRACTION` (40%) of its
  column's widest map width (via `layout.trace_column_targets`), else `[]`.
```

  - `docs/Usage.md`: in the notes-bar paragraph (grep `A **notes bar** under the preview`), after the sentence ending `— the note always describes exactly what's on screen.` (line 1485), insert:

```
The notes bar can also carry a **text-overlap advisory**: after every render
(and every export — same code path) the composer checks whether visible text
from different panels overlaps in the final figure (titles, axis and tick
labels, panel letters, colorbar text) and, if so, appends one note naming the
colliding panels with suggested fixes — *enable trace autoscale* (offered
when a trace panel rendered far narrower than its column's maps and that
option is off), *increase gutter*, and *reduce font scale*. It is advisory
only, never an error, and on a figure with an unusually large number of text
artists (over 400) the check skips itself with a note rather than slow the
render down.
```

- [ ] `ruff check . && ruff format .` — clean.
- [ ] `DISPLAY= python3 -m pytest -q` — expect **964 passed / 13 skipped / 0 failed**.
- [ ] Commit:

```bash
git add dfxm/compose/render.py tests/test_compose_render.py docs/Codebase.md docs/Usage.md
git commit -m "feat(compose): wire trace autoscale into render + text-collision advisory (docs synced)"
```

**Verification** — integration test proves the trace box physically matches the map box on the rendered figure and that the flag-off render differs (discrimination); detector unit tests prove cross-axes flagging, same-axes immunity, clean-figure silence, and the cost guard; the spy test proves the detector runs at the end of every `render_recipe` with title-or-id owner keys.

**Risks/edge cases**
- The detector's `fig.canvas.draw()` on a bare `Figure` is already proven safe in this codebase (`_align_axis_labels`/`_stretch_bar_to_span` do the same on every render).
- `Bbox.padded(-pad)` can invert a sub-1pt text box; `_overlap_area` returns 0 for inverted boxes by construction.
- Placeholder-only figures / no-text figures: `groups` ends empty → `[]` (spec's error-handling case).
- Duplicate panel titles: uniquified owner keys prevent silent axes shadowing in the check.
- Autoscale runs **before** `_apply_shared_colorbars`/`_apply_united_colorbars` build provisional bar boxes and before any axes exist, so no stale geometry anywhere; the `pinned_width_cm` whole-figure rescale multiplies every cell by one factor, preserving the width match (documented, exercised implicitly by existing pinned-width tests remaining green).

---

## Task 4 — GUI checkbox "Autoscale traces to column width"

**Files**
- `gui/figure_builder.py` — `_build_compose_form` (insert between the Padding row, line 256, and the Colourbars heading, line 258); `_load_compose_into_widgets` (widgets tuple lines 317–327 + value loads); `_on_compose_edited` (lines 348–363). `QCheckBox` is already imported (line 21).
- `tests/test_gui_figure_builder.py` — append after `test_compose_colorbar_widgets_reload_from_recipe` (ends ~line 1065).
- `docs/Usage.md` — compose-pane bullet (lines 1518–1521 intro sentence).
- `docs/Codebase.md` — the `figure_builder.py` table row (line 1183): two short-fragment edits.

**Interfaces**
- Produces: `self._compose_trace_autoscale: QCheckBox`, checked ⇔ `recipe.compose.trace_autoscale`; edits route through `_on_compose_edited` (dirty + retitle + `schedule_preview`), reloads through `_load_compose_into_widgets` (signals blocked, never writes back).

**Steps**

- [ ] Append this failing test to `tests/test_gui_figure_builder.py` (after `test_compose_colorbar_widgets_reload_from_recipe`):

```python
def test_compose_trace_autoscale_checkbox_writes_field_and_reloads():
    w = _win()
    assert w.recipe().compose.trace_autoscale is False
    assert not w._compose_trace_autoscale.isChecked()
    w._compose_trace_autoscale.setChecked(True)  # toggled -> _on_compose_edited
    assert w.recipe().compose.trace_autoscale is True
    assert w.is_dirty()
    # reload path: recipe -> widget, signals blocked, no write-back
    w.recipe().compose.trace_autoscale = False
    w._load_compose_into_widgets()
    assert not w._compose_trace_autoscale.isChecked()
    assert w.recipe().compose.trace_autoscale is False
```

- [ ] Run `DISPLAY= python3 -m pytest -q tests/test_gui_figure_builder.py -k trace_autoscale` — expect **1 failed**: `AttributeError: 'FigureBuilderWindow' object has no attribute '_compose_trace_autoscale'`.
- [ ] Implement in `gui/figure_builder.py`:

  1. In `_build_compose_form`, between the Padding `form.addRow` (line 256) and `form.addRow(QLabel("<b>Colourbars</b>"))` (line 258), insert:

```python
        self._compose_trace_autoscale = QCheckBox("Autoscale traces to column width")
        self._compose_trace_autoscale.setChecked(c.trace_autoscale)
        self._compose_trace_autoscale.toggled.connect(self._on_compose_edited)
        form.addRow(self._compose_trace_autoscale)
```

  2. In `_load_compose_into_widgets`, add `self._compose_trace_autoscale,` to the `widgets` tuple (after `self._compose_padding,`, line 321) and, with the other value loads (after `self._compose_padding.setValue(c.padding_cm)`, line 333), add:

```python
        self._compose_trace_autoscale.setChecked(c.trace_autoscale)
```

  3. In `_on_compose_edited`, after `c.padding_cm = self._compose_padding.value()` (line 353), add:

```python
        c.trace_autoscale = self._compose_trace_autoscale.isChecked()
```

- [ ] Run `DISPLAY= python3 -m pytest -q tests/test_gui_figure_builder.py` — expect **all pass**.
- [ ] Docs (same commit), Read each target first:
  - `docs/Usage.md`, compose-pane intro bullet (lines 1518–1521): replace `the gutter and padding (cm), then two headed groups, then a pinned total width in cm (0 = auto-sized from the layout):` with:

```
  the gutter and padding (cm), an **Autoscale traces to column width**
  checkbox (off by default — when on, every trace panel is rescaled with its
  box ratio kept so its width matches the widest map panel in its own
  column, falling back to the widest map anywhere in the figure when its
  column has none; a figure with no maps leaves traces untouched, and a
  trace sized by a pinned row height or column width keeps its pin — pins
  always win over autoscale; each rescale is reported in the notes bar with
  the implied µm/cm), then two headed groups, then a pinned total width in
  cm (0 = auto-sized from the layout):
```

  - `docs/Codebase.md`, `figure_builder.py` row (one very long table line — grep the exact fragments, edit in place, never retype the row):
    - Replace the fragment `` pinned width's special value `0` reads back as `None` = auto), a **Colourbars** heading `` with `` pinned width's special value `0` reads back as `None` = auto), `self._compose_trace_autoscale` (`QCheckBox` "Autoscale traces to column width", bound to `compose.trace_autoscale`), a **Colourbars** heading ``
    - Replace the fragment `` `colorbar_mode`/`colorbar_pos`/`pinned_width_cm` into `self._recipe.compose` `` with `` `colorbar_mode`/`colorbar_pos`/`pinned_width_cm`/`trace_autoscale` into `self._recipe.compose` ``
- [ ] `ruff check . && ruff format .` — clean.
- [ ] `DISPLAY= python3 -m pytest -q` — expect **965 passed / 13 skipped / 0 failed**.
- [ ] Commit:

```bash
git add gui/figure_builder.py tests/test_gui_figure_builder.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): figure-builder trace-autoscale checkbox (docs synced)"
```

**Verification** — the Qt test covers write (toggle → recipe + dirty) and reload (recipe → widget, no write-back). `_load_compose_into_widgets` blocks signals over the whole tuple, so adding the checkbox there automatically prevents a reload from re-triggering `_on_compose_edited`.

**Risks/edge cases** — `setChecked` on an already-matching state emits no `toggled`, so reload cannot ping-pong. `FormStateStore`/QSettings persistence is not involved (the recipe file is the store) — no extra wiring.

---

## Task 5 — smoke step [40] extension + final whole-branch verification

**Files**
- `tests/gui_smoke.py` — step [40] block (lines 1175–1201; the `print("[40] …")` is line 1201). Step **[41]** (lines 1203–1223) must remain the file's last step — do not touch it.

**Interfaces**
- Consumes: the live `fb` window from step [37]/[40], `_compose_trace_autoscale` from Task 4, `render_now()`.

**Steps**

- [ ] In `tests/gui_smoke.py`, Read the [40] block, then insert immediately **before** the `print("[40] …")` line (line 1201):

```python
    # trace-autoscale toggle: recipe -> widget -> recipe, and a re-render each way
    fb.recipe().compose.trace_autoscale = True
    fb._load_compose_into_widgets()
    assert fb._compose_trace_autoscale.isChecked()
    _res40b = fb.render_now()
    assert _res40b is not None and _res40b.n_rendered == 1, fb._notes_label.text()
    fb._compose_trace_autoscale.setChecked(False)  # widget -> recipe via _on_compose_edited
    assert fb.recipe().compose.trace_autoscale is False
    _res40c = fb.render_now()
    assert _res40c is not None and _res40c.n_rendered == 1, fb._notes_label.text()
```

  and update the step's print line from

```python
    print("[40] figure builder: arranger + Arrange… + united mode + two-step Add panels")
```

  to

```python
    print(
        "[40] figure builder: arranger + Arrange… + united mode + two-step Add panels"
        " + trace-autoscale toggle"
    )
```

- [ ] Run `python3 tests/gui_smoke.py` — expect all steps `[1]`–`[41]` to print and the final line `GUI SMOKE PASSED`, with `[41]` still the last step before it.
- [ ] `ruff check . && ruff format .` — clean.
- [ ] Full-suite final check: `DISPLAY= python3 -m pytest -q` — expect **965 passed / 13 skipped / 0 failed** (baseline 949 + 16 new tests across Tasks 1–4).
- [ ] Commit:

```bash
git add tests/gui_smoke.py
git commit -m "test(smoke): trace-autoscale toggle in figure-builder step [40]"
```

- [ ] Branch done — hand back for the final whole-branch review/merge flow (`finish-and-record` skill; no remote, so no push).

**Verification** — smoke exercises the real window end-to-end offscreen; the [40] recipe holds one slice panel and no traces, so the toggled render is a wiring/no-crash check (the geometry semantics are pinned by Tasks 2–3's tests).

**Risks/edge cases** — the [40] recipe is in `united` colorbar mode at this point; `render_now` already passes there, and `trace_autoscale` is orthogonal. Keep the insertion strictly inside the [40] block so [41] stays last.

---

## Spec coverage

| Spec requirement | Task |
|---|---|
| `ComposeStyle.trace_autoscale: bool = False`, serialized like other compose fields | 1 |
| Old recipes load `False`; no validation; `RECIPE_VERSION` stays 1 | 1 |
| Qt-free `autoscale_traces(...)` called by `render_recipe` right after `size_cells`, only when flag true | 2 (function) + 3 (call site) |
| Target = widest map cell in nearest (innermost) enclosing `Col` | 2 (`trace_column_targets` + test) |
| Fallback: widest map in whole figure (trace in root Row / all-trace column) | 2 |
| No map cells at all → trace untouched | 2 (all-trace test) |
| `f = target_w / w`; `w *= f; h *= f` (ratio kept); up- AND down-scaling | 2 (match + upscale tests) |
| Note per scaled trace: `"panel {pid}: trace autoscaled to column width — implied scale {length/(w_in*2.54):.4g} µm/cm"` | 2 |
| Pinned sizes win — `SizedCell.pinned` set in `_trace_cell` pin branches; autoscale skips pinned | 2 |
| Placeholder traces untouched | 2 |
| GUI checkbox "Autoscale traces to column width" wired via `_on_compose_edited` / `_load_compose_into_widgets` | 4 |
| `_detect_text_collisions(fig, …) -> list[str]` at the very end of `render_recipe` (after `_align_axis_labels` + gutter/scale-bar draws), result extended into `notes` | 3 |
| Runs on export via shared `render_recipe` path | 3 (export calls `render_recipe`; documented) |
| Collect title/axis-labels/tick-labels/annotations per panel axes + shared/united bar axes (bar label + ticks) | 3 (`_axes_texts`, owners map) |
| Prefilter: per-axes `get_tightbbox` once, compare only intersecting axes boxes (expanded 2 pt) | 3 |
| Collision = different parent axes, extents shrunk 1 pt, positive-area intersection; same-axes ignored | 3 |
| At most ONE note: `"text overlaps between panels {A} and {B}{, …} ({n} collision(s)) — {suggestions}"`, names = `title or id` | 3 |
| Suggestion order: trace <40% of column map width AND flag off → "enable trace autoscale"; always "increase gutter"; "reduce font scale" | 3 (`_collision_presuggestions` + note assembly) |
| Cost guard: >400 text artists → `"text-collision check skipped ({n} text artists)"` | 3 |
| Never an error; empty list when clean; placeholders/no-text → empty | 3 |
| `autoscale_traces` with zero traces or flag off: no-op, no note | 2 |
| Tests: recipe round-trip + old default; autoscale pass (match/ratio/fallback/all-trace/pinned/note); detector (cramped→note+suggestion, spacious→none, >400 skip); Qt checkbox write+reload; smoke toggle in [40] | 1–5 |
| Docs same-change: Usage.md (checkbox, autoscale semantics, pins win, collision note + reactions) / Codebase.md (`trace_autoscale`, `SizedCell.pinned`, `autoscale_traces`, `_detect_text_collisions`) | 1–4 |

**Recorded spec deviations** (both signature-only, behaviour identical): `autoscale_traces` takes `data_by_id` (the mandated note text needs `length_um`); `_detect_text_collisions` takes an owner-name→axes map plus `pre_suggestions` (the note needs display names, and the trace-tiny condition needs recipe/cell context computed by the caller). Plus one helper the spec implies but doesn't name: `trace_column_targets`, shared by the pass and the suggestion check so the "column's widest map" rule exists exactly once.
