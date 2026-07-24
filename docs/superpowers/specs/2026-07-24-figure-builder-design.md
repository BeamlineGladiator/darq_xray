# Publication figure builder (grid-based composer) — design

**Date:** 2026-07-24
**Status:** approved (brainstormed with Albert; Phase 2 of the trace fixed-scale
work — realizes the `figure-builder-wish`)

## Problem

Publication figures are currently composed outside the app (PowerPoint) from
per-stage PNGs. That path cannot produce vector output, and it only looks
uniform if every source PNG happens to be exported at exactly the same µm/cm,
font size, and margin geometry. The pipeline already re-renders every figure
cold from HDF5 (replot catalogs) and places axes at exact physical size
(`place_axes_box` / `place_axes_stack`); what is missing is a tool that
composes panels **from multiple stages and files into one canvas** with global
style applied once and PNG/PDF/**SVG** export at exact physical size.

## Requirements (agreed feature list)

- Pick panels across stages/files via the existing replot catalogs
  (strain / mosaicity / rocking map layers, slices planes, profiles
  reference maps and traces).
- Structured layout: nested rows/columns (ragged allowed — e.g. a 2-panel
  column beside 3-panel columns), **not** free-form dragging. Spanning-like
  layouts fall out of nesting.
- Mixed panel sizes with per-row/column consistency; sizing is
  **physical-scale-first** with optional pinned row-height / column-width
  ("hybrid" model, see below).
- Separate µm/cm for maps (`scale_um_per_cm`) and traces
  (`trace_scale_um_per_cm`, box height `trace_height_cm`) — the existing
  split — both global with per-group override.
- Subplot letter labels: template-driven (`A`, `a`, `A)`, `a)`, `(A)`, `A1`)
  with auto-increment in layout order and per-node manual override; labels can
  attach to a single panel **or a group** (a whole stack).
- Margin compensation: within a row/column, shorter panels get trailing
  padding so outer cell edges align and tick fonts/positions are identical
  across panels.
- Trace stacks with shared distance axis (numbers + axis label on the bottom
  panel only).
- Shared colorbars: one bar serving a group of same-quantity panels with a
  unified clim.
- Shared scale bars for maps at the same µm/cm: on every map, on one
  designated panel, or in a **gutter cell** between panels.
- Per-panel overrides on top of global style: ROI crop, clim, cmap, label,
  title (titles are **off by default** in composed figures).
- Spacer cells and free-text cells (row headers etc.).
- Figure **recipes**: the whole composition saved/loaded as JSON; re-render
  any time, including headless.
- One canvas → PNG/PDF/SVG export at exact physical size, no tight-crop.

### Acceptance figures (Albert's two real targets)

1. **2×2 grid** — columns = mosaicity COM | strain, rows = two different
   oblique slices; labels A B / C D (row-major); all maps share one µm/cm.
2. **Ragged 3-column figure** —
   - column 1: two ROI-zoomed profiles reference maps (mosaicity, with the
     picked line overlay), labels **A1**/**B1**, same µm/cm and same scale
     bar treatment;
   - columns 2 & 3: stacks of three trace panels each
     (mosaicity, strain, raw sum-mosa) from the A1 line and B1 line
     respectively, group-labelled **A2** and **B2**; identical box height and
     width within each stack; shared distance axis (bottom only); no titles
     anywhere; identical font size on all panels; if the A line is shorter,
     the A2 stack is padded on the right so column edges align. The trace
     stacks use a **different µm/cm than the maps**
     (`trace_scale_um_per_cm` ≠ `scale_um_per_cm`) — the acceptance test
     must assert both scales are honoured exactly in the same canvas.

## Approach decision

Chosen: **composer core + box-tree layout + dedicated window** (over
GridSpec-based layout, which hands final sizes back to matplotlib auto-layout
and cannot guarantee µm/cm — the exact failure the trace project removed —
and over pasting rendered PNGs, which is raster-only). All panels draw into
**one matplotlib Figure**; placement is deterministic (measure-then-place).

## Design

### A. Package layout

New Qt-free package `dfxm/compose/`:

- `recipe.py` — dataclasses + JSON (de)serialization + validation.
- `layout.py` — the solver (sizing, alignment, placement).
- `adapters.py` — panel-kind registry mapping a `PanelRef` to a
  `draw_<kind>(ax, data, style, ...)` call + data loader.
- `render.py` — `render_recipe(recipe, style_overrides=None) -> (Figure,
  notes)` and file export.
- `__main__.py` — headless CLI: `python3 -m dfxm.compose render recipe.json
  -o outdir [--formats png,pdf,svg]`.

GUI on top (Phase B): `gui/figure_builder.py` + widgets.

### B. Recipe data model

`FigureRecipe`:

- `version` (int), `name`.
- `style`: the existing `PlotStyle` fields (fonts, colorbar knobs, tick
  formats, `scale_um_per_cm`, `trace_scale_um_per_cm`, `trace_height_cm`,
  `axes_mode`, formats/dpi, cmaps) + `ComposeStyle`: label template, label
  font scale, gutter width (cm), canvas padding (cm), scale-bar sharing mode
  (`per-panel` | `one-panel` | `gutter`) + designated panel/gutter position,
  pinned total width (cm, optional).
- `layout`: a tree of nodes —
  - `Row(children, pinned_height_cm=None)` — children side by side;
  - `Col(children, pinned_width_cm=None, group_label=None, shared_x=False)`
    — children stacked; `group_label`/`shared_x` make it a labelled group
    (trace stacks); `Row` accepts `group_label` too;
  - leaves: `PanelRef`, `Spacer(w_cm, h_cm)`, `TextCell(text)`.
- `panels`: list of `PanelRef` —
  - `id`;
  - `source`: h5 path (stored relative to the recipe file when possible,
    absolute fallback) + stage name + the stage's replot-catalog selection
    key (slices `volume_id` + plane, profiles job + field + figure kind
    (reference map | trace), strain/mosaicity/rocking/visualize group +
    layer index);
  - overrides (all optional; global style is the default): `roi` (pixel
    crop, same `r0,r1,c0,c1` convention as replot), `clim`, `cmap`,
    `label` (manual override of the auto sequence), `show_title`,
    `scale_um_per_cm` (per-group override), `colorbar` (on/off — off when a
    shared bar covers it).

Labels: the template is applied by auto-increment in depth-first layout
order; a node with a `group_label` consumes one sequence slot for the whole
group (figure 2: panels A1, B1 and groups A2, B2 share one A/B row-letter
scheme via manual overrides — auto numbering covers the plain `A B C D`
case, manual override covers compound schemes).

### C. Sizing semantics (hybrid model)

- Map panel intrinsic box = data extent (µm) ÷ map µm/cm (per-panel override
  wins over global).
- Trace panel intrinsic box = line length ÷ trace µm/cm (fallback: map
  scale), height = `trace_height_cm` (fallback 3.0 cm) — same rules as
  `trace_fixed_box` today, including the 30-in clamps.
- `TextCell`/`Spacer` (and future non-physical panels) = explicit cm box.
- Pinned `Row.pinned_height_cm` / `Col.pinned_width_cm`: physical panels in
  that row/col rescale to fit; the implied µm/cm is reported as a note —
  never silent. Pinned total canvas width scales everything and reports all
  implied scales.
- No global scale set and no pinned sizes → `StageUserError` with a hint
  (the composer never guesses a scale).

### D. Layout solver (`layout.py`)

Deterministic, generalizing `place_axes_stack`; no matplotlib auto-layout.

1. **Intrinsic sizes** — walk the tree, compute every panel box in cm (§C).
2. **Pinned dimensions** — apply rescales, record implied-scale notes.
3. **Measure pass** — place each panel's axes provisionally at final box
   size in the real Figure, draw decorations (ticks, labels, colorbar,
   panel label), measure `AxesMargins`. Measuring at final size is mandatory
   (tick density depends on size) — same rule as `place_axes_box`.
4. **Alignment pass** — within each `Row`: shared max top/bottom margins;
   within each `Col`: shared max left/right margins; sibling cells that are
   narrower/shorter than the row/col envelope get trailing padding (margin
   compensation). Gutter cells (shared scale bar / shared colorbar) are
   solver cells with fixed sizes.
5. **Place** — absolute `ax.set_position` for every axes in one Figure;
   total canvas size is an output of the tree (unless pinned).
6. **Drift guard** — `measured_box_in` + `box_drift_note` (0.5% tol) on
   every panel post-render; drift and clamp notes surface in the GUI notes
   bar and CLI output.

Shared-x stacks: upper panels keep tick marks, drop tick labels and the axis
label; the group's left margins are unified so y-labels align.

### E. Panel adapters (`adapters.py` + per-stage extractions)

The one real refactor: each supported panel kind gets a
`draw_<kind>(ax, data, style, ...)` function **extracted from the existing
figure builders** (the `_draw_trace_axes` pattern — traces are nearly free).
The existing single-figure paths re-call the extractions and are pinned by
regression tests (byte-stable where the suite already pins byte-stability).

v1 adapters:

- map layers for strain / mosaicity / rocking (shared
  `render_volume_layer` path);
- slices planes (per-quantity clim/cmap groups as in the slices replot);
- profiles reference maps (with the picked-line overlay);
- profiles traces (`_draw_trace_axes`).

Excluded from v1 (later project): visualize (no replot catalog today — add
one first if wanted), matched, histograms, companion-style map-with-marker
composites beyond the profiles reference map.

Each adapter has a data loader that reads only its panel's arrays from h5;
loaders are pure functions of (source, roi) so the GUI can cache results.

### F. Shared colorbars & scale bars

- **Shared colorbar**: attached to a group node; member panels render with
  `colorbar: off`; group clim = explicit value or the max-range across
  members; the bar renders into its own solver cell via `add_colorbar(cax=)`
  (steal-free). Tick format follows the per-quantity `tickfmt_*` knobs.
  Mixing quantity groups under one shared bar is refused with a
  `StageUserError` hint.
- **Shared scale bar**: modes `per-panel` (today's look), `one-panel`
  (designated panel only), `gutter` (its own cell between panels, drawn with
  the existing offsetbox scale-bar machinery). A shared bar across panels
  with differing µm/cm is refused with a hint.

### G. Export

`render.py` saves the single Figure at exact physical size in any of
PNG/PDF/SVG (style `formats` + `dpi`). **No tight-crop** — the solver owns
all margins; tight-crop is what breaks grid alignment in the export dialog's
single-figure path today.

### H. GUI (Phase B, `gui/figure_builder.py`)

Dedicated **non-modal** window opened from the main-window toolbar.

- **Left — sources & outline.** Source list pre-filled from the current
  experiment's stage outputs (the `bindings.py` auto-chaining paths) +
  Browse…; adding a panel opens the stage's existing replot catalog picker.
  Below: the layout outline tree mirroring the recipe (rows / cols / groups /
  panels / spacers / text) with add, move up/down, delete, group toggle,
  per-node label override. Structured editing only — no canvas dragging.
- **Center — preview.** `MplCanvas` at true aspect with a page outline.
  Panel data cached in memory on first load; style/layout edits redraw from
  cache behind a short debounce; a Refresh button re-reads h5. Clicking a
  panel selects its outline node.
- **Right — style.** Global knobs (reusing `StyleControls` sections where
  they fit), compose knobs (label template, gutters, scale-bar sharing,
  pinned sizes), and the selected node's overrides. Notes bar under the
  preview shows implied-scale / drift / degraded-panel notes.
- Recipes: Save / Save as… / Open; window title tracks dirty state.

### I. Error handling

- Recipe schema problems and refused configurations (mixed-quantity shared
  bar, no scale anywhere) → `StageUserError(message, hint)`.
- Missing h5 file / missing dataset key at render time → the panel renders
  as a hatched placeholder cell with the reason in the notes (the
  composition survives partial data — important after re-analysis); the CLI
  prints the same notes and exits non-zero only if *no* panel rendered.
- Degenerate extents → per-panel fallback consistent with the companion's
  degenerate-extent behaviour (never a crash).

### J. Testing

- Solver units: intrinsic sizes exact at given scales; margin compensation
  equalizes cell edges; pinned-row implied-scale math; drift guard fires on
  injected drift; shared-x stack label suppression.
- Adapter regressions: existing single-figure outputs unchanged after
  extraction (byte-stable where already pinned by the suite).
- Recipe JSON round-trip (+ unknown-version / bad-schema errors).
- **Acceptance**: build figure 1 and figure 2 from synthetic h5 fixtures;
  assert panel box sizes/positions, label sequence, shared-bar placement,
  and column-edge alignment under unequal line lengths. The figure-2
  fixture sets `trace_scale_um_per_cm` different from `scale_um_per_cm`
  and asserts map boxes honour the map scale while trace boxes honour the
  trace scale, exactly, within one canvas.
- CLI smoke (render a recipe headless); gui_smoke additions (open window,
  load recipe, preview, export).

### K. Phasing (one branch, two phases)

- **Phase A — core**: recipe schema, solver, adapters, CLI render; proven by
  the two acceptance figures rendered headless.
- **Phase B — GUI**: the window on top of the proven core.

Deferred to a later project: histogram/matched panels, rich text-cell
styling, journal width presets (pinned total width covers the need).

### L. Documentation

Usage.md gains a "Figure builder" chapter (workflow: sources → layout →
labels → export; recipe files; CLI); Codebase.md gains `dfxm/compose` and
`gui/figure_builder` sections + data-flow updates. Same-change contract.

## What this builds on

- `place_axes_box` / `place_axes_stack` / `AxesMargins` /
  `measure_axes_margins` (trace fixed-scale project).
- Replot catalogs + `render_replot` on strain / mosaicity / rocking /
  slices / profiles; `resolve_clim`; per-quantity clim/cmap groups.
- `add_colorbar(cax=)` steal-free path; offsetbox scale bar;
  `trace_fixed_box` sizing rules; drift/clamp note machinery.
