# Figure builder: trace autoscale + text-collision warnings — design

Date: 2026-08-17. Approved by Albert (semantics chosen via numbered options:
match-column-width autoscale; composed-figures-only collision check).
Builds on the figure-builder interactive branch merged at master 1c94e8f.

## Problem

1. Composed trace panels are sized physically (line length ÷ trace scale,
   ~3 cm tall) while map panels at the figure's µm/cm scale can be tens of
   cm tall — traces render microscopic next to their maps and their text
   piles up (real screenshot, 2026-08-17).
2. Nothing warns when text from different panels overlaps in a composed
   figure; the user discovers it by eyeballing the export.

## 1. Trace autoscale (match column width)

### Model — `dfxm/compose/recipe.py`

- `ComposeStyle.trace_autoscale: bool = False`. Serialized like the other
  compose fields; old recipes load `False` (today's behaviour). No
  validation needed (bool). `RECIPE_VERSION` stays 1.

### Sizing pass — `dfxm/compose/layout.py`

- New Qt-free `autoscale_traces(recipe, cells, notes) -> None`, called by
  `render_recipe` immediately after `size_cells` **only when
  `recipe.compose.trace_autoscale`** is true.
- Rule, per trace cell (kind == "trace"):
  - Target width = the widest `kind == "map"` cell within the trace's
    nearest enclosing `Col` (walk the recipe layout tree; "nearest
    enclosing" = innermost Col ancestor of the trace's leaf).
  - If no map sibling exists in any enclosing Col (e.g. the trace sits
    directly in the root Row, or an all-trace column): target = the widest
    map cell in the whole figure.
  - If the figure has no map cells at all: leave the trace untouched.
  - Scale factor `f = target_w / w`; apply `w *= f; h *= f` (box ratio
    kept). Both up- and down-scaling apply (the option means "match", not
    "grow only").
  - Append one note per scaled trace:
    `"panel {pid}: trace autoscaled to column width — implied scale {length/(w_in*2.54):.4g} µm/cm"`.
- Precedence: pinned sizes win — a trace cell whose size came from a
  pinned row height / column width (the pin paths in `_trace_cell`) is
  NOT autoscaled. Detection: the pass re-derives nothing; `size_cells`
  marks trace cells it sized via a pin (new `SizedCell.pinned: bool =
  False` field set in the pin branches) and `autoscale_traces` skips
  `pinned` cells.
- Placeholder/degenerate traces (kind == "placeholder") are untouched.

### GUI — `gui/figure_builder.py`

- Compose form gains a checkbox "Autoscale traces to column width" bound
  to `compose.trace_autoscale` (in `_build_compose_form` under the Figure
  fields, wired through `_on_compose_edited` / `_load_compose_into_widgets`
  like every other compose widget).

## 2. Text-collision check (composed figures only)

### Detector — `dfxm/compose/render.py`

- New `_detect_text_collisions(fig, axes_by_id) -> list[str]`, called at
  the very end of `render_recipe` (after `_align_axis_labels` and the
  gutter/scale-bar draws — the geometry is final), its return extended
  into `notes`. Also runs on export via the shared `render_recipe` path
  (no extra work).
- Method:
  - For each panel axes (and each shared/united bar axes), collect its
    visible text artists with non-empty text: title, x/y axis labels,
    tick labels, annotations (panel letter labels are annotations on the
    axes), and for bar axes the colorbar label + ticks.
  - Prefilter: compute each axes' `get_tightbbox` once; only compare text
    pairs whose parent axes' boxes intersect (expanded by 2 pt).
  - A collision = two text artists with **different parent axes** whose
    `get_window_extent` rectangles intersect with positive area (after
    shrinking each by 1 pt to ignore hairline touches).
  - Same-axes overlaps are ignored (matplotlib's own tick layout is not
    our warning's business).
- Reporting: at most ONE note summarizing everything:
  `"text overlaps between panels {A} and {B}{, …} ({n} collision(s)) — {suggestions}"`
  where panel names use `title or id` and the suggestion list is chosen
  from actual conditions, in order:
  - any trace cell < 40% of its column's widest map width AND
    `trace_autoscale` off → "enable trace autoscale";
  - always → "increase gutter"; "reduce font scale".
- Cost guard: if the figure has more than 400 text artists total, skip the
  pairwise pass and append
  `"text-collision check skipped ({n} text artists)"` instead — never let
  the check dominate render time.
- Never an error; empty list when clean.

## Error handling

- `autoscale_traces` with zero traces or flag off: no-op, no note.
- Collision detector on a figure with placeholders/no text: empty result.

## Testing

- Qt-free: recipe round-trip of `trace_autoscale` + old-recipe default;
  autoscale pass (trace matched to its column's map width with ratio kept,
  fallback to figure-wide widest map, all-trace figure untouched, pinned
  trace skipped, note text); collision detector (a deliberately cramped
  two-panel figure with overlapping labels → note lists both panels and a
  suggestion; a spacious figure → no note; >400-artist skip note).
- Qt: compose checkbox writes the field, reloads from recipe.
- Smoke: one line in the existing figure-builder step ([40]) toggling
  `trace_autoscale` and re-rendering.

## Docs (same change)

- `docs/Usage.md`: the new checkbox + what autoscale does (and that pins
  win); the collision note and how to react to it.
- `docs/Codebase.md`: `autoscale_traces`, `SizedCell.pinned`,
  `_detect_text_collisions`, new compose field.
