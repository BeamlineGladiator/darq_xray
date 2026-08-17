# Figure builder: panel titles, drag-grid arranger, united bars — design

Date: 2026-08-17. Approved by Albert (all four scoping questions answered with
the recommended options; overall design approved as-is).

## Problem

Three usability gaps in the figure-builder window (`gui/figure_builder.py`):

1. **Panels lose their data names.** The Add-panels dialog shows readable
   labels (job / field / plane), but on OK each panel becomes an opaque id
   (`profiles_0`, `profiles_1`, …) — that id is all the Outline tree and the
   "Scale-bar panel" dropdown ever show.
2. **No mouse-based layout.** Arranging panels means clicking Row/Col/↑/↓
   buttons against an outline tree. There is no way to drag panels into a
   grid or choose how many rows each column has.
3. **Bars are blind and per-group only.** Scale-bar/colorbar placement is
   invisible until the debounced render lands; shared colorbars exist only as
   checkboxes on Row/Col nodes buried in the node inspector; there is no
   "one bar per quantity" option and no first-class position selection.

## Scope

All changes live in `dfxm/compose/` (Qt-free model + renderer) and `gui/`
(figure-builder window + new arranger widget). No stage modules change.

## 1. Human-readable panel titles

- `PanelDef` (`dfxm/compose/recipe.py`) gains `title: str | None = None`,
  serialized like every other optional field. Old recipes load with
  `title=None`; every display site falls back to the id. `RECIPE_VERSION`
  stays 1 (purely additive field).
- `AddPanelDialog._build_panels` (`gui/widgets/panel_picker.py`) derives the
  title from the tree it already built, at pick time:
  - map stages: `"{stage}: {group label} / {layer label}"` (e.g.
    `"strain / z=3"`; the group label is the catalog group's `label`),
  - slices: `"{volume_id}/{slice_name} / plane {k} @ {off} µm"`,
  - profiles: `"{job name} / {field}"` and `"{job name} / reference"`.
- Display sites switch to `title or id`:
  - Outline tree `_node_label` → `Panel: {title}` (id no longer shown; a
    suppressed label still appends "(label off)").
  - "Scale-bar panel" combo → shows titles, stores the id as item data.
  - Arranger tiles (below) → titles.
  - Render/export notes keep ids (debugging identity) — unchanged.
- Ids remain the unique key (`add_panels` dedup unchanged); titles may repeat.

## 2. Drag-grid arranger

### Qt-free grid ↔ tree mapping — `dfxm/compose/gridmap.py` (new)

- `layout_to_grid(layout, panels_by_id) -> GridModel | None`:
  `GridModel = list[list[str]]` — columns of panel ids, outer list ordered
  left→right, inner top→bottom. Recognized shapes: `Row([Col…])`,
  `Row([PanelRef…])`, `Col([PanelRef…])`, a single `PanelRef`, and mixes of
  `PanelRef`/`Col` children under the root Row. Returns `None` when the
  layout does not map cleanly (spacers, text cells, nested groups deeper
  than one Col, group flags on inner nodes) — the caller then offers the
  flatten-with-warning path.
- `flatten_panel_ids(layout) -> list[str]`: DFS PanelRef order, used to seed
  the grid (one column) when `layout_to_grid` returns `None`.
- `grid_to_layout(grid) -> Row`: builds `Row([...])` whose children are
  `Col([PanelRef…])` for multi-tile columns and a bare `PanelRef` for
  single-tile columns. Empty columns are dropped. An empty grid yields
  `Row([])`.
- Round-trip law (tested): `layout_to_grid(grid_to_layout(g)) == g` for any
  normalized grid (no empty columns).

### Widget — `gui/widgets/layout_arranger.py` (new)

- `LayoutArranger(QWidget)`: a horizontal strip of columns; each column is a
  drag-enabled `QListWidget` (internal move + drag between columns), a
  header with ◀/▶ (reorder column) and ✕ (remove — tiles merge into the
  neighbouring column), plus an "+ Add column" button at the strip's end.
  Rows-per-column is simply how many tiles a column holds.
- Tiles show the panel **title**, a quantity chip (colour keyed by the
  panel's group: misorientation / FWHM / strain / raw / trace / none), and
  the bar badges from §3.
- API: `set_grid(grid, tile_info_by_id)`, `grid() -> GridModel`, signal
  `gridChanged`. Pure view over the grid model — no recipe knowledge.
- Bar schematic (see §3): the arranger draws a colorbar strip along the
  chosen edge (united mode) or beside/below flagged groups (group mode),
  and a corner dot on the scale-bar target panel. Clicking a tile corner
  sets the scale-bar target (one-panel mode) + corner. These are schematic
  markers only; the real preview stays the source of truth.

### Integration

- **Add-panels dialog**: becomes a two-step `QStackedWidget` — step 1 is the
  existing stage/file/tree picker, step 2 shows the *new* panels in a
  `LayoutArranger` seeded one-row (each panel its own column). Buttons:
  Back / Next / OK / Cancel; OK from step 2 returns `selected_panels` plus
  `selected_layout` (the grid's `Row` fragment). OK from step 1 (skipping
  arrangement) keeps today's append-flat behaviour.
- `FigureBuilderWindow.add_panels(panels, layout=None)`: with a layout
  fragment, appends that fragment as one child of the current container
  (ids uniquified before both the defs and the fragment's refs).
- **Arrange… button** (left pane, next to "Add panels…"): opens
  `ArrangeDialog` — a `LayoutArranger` seeded from
  `layout_to_grid(recipe.layout)`; when that is `None`, seeds from
  `flatten_panel_ids` and shows a persistent warning label: "applying will
  rebuild the layout as a plain grid — spacers, text cells and nested
  groups will be dropped". Apply replaces `recipe.layout` with
  `grid_to_layout(...)`, purges orphans, rebuilds the tree, previews.
  Group/shared settings on dropped nodes are lost only in the warned case;
  in the clean grid case columns are rebuilt as plain Cols (a Col that had
  `shared_colorbar`/`group_label`/`shared_x`/pins keeps them when its
  member set is unchanged — matched by member-id set).

## 3. United bars + position selection

### Model — `ComposeStyle` (`dfxm/compose/recipe.py`)

- `colorbar_mode: str = "per-panel"` — one of `("per-panel", "united")`
  (new tuple `COLORBAR_MODES`, validated in `validate_recipe`).
- `colorbar_pos: str = "right"` — one of `("right", "bottom")`
  (`COLORBAR_POSITIONS`, validated).
- Scale bar: no new model fields — `scale_bar_mode`/`scale_bar_panel`
  (compose) and `scale_bar_loc` (style) already express mode, target and
  corner. They get first-class UI + schematic instead.

### Renderer — `dfxm/compose/render.py`

- New `_apply_united_colorbars(...)`, active when
  `colorbar_mode == "united"`:
  - Partition live, non-placeholder map/slice/ref panels by `data.group`
    (quantity). Panels with `group=None` and trace panels keep their
    per-panel behaviour.
  - Per group: unify clim as the union of member effective ranges
    (per-panel `clim` overrides respected as today's shared path does),
    rewrite members via `dc_replace(..., clim=unified)`, suppress their
    per-panel bars.
  - One bar leaf per group, appended on the chosen edge by wrapping the
    working layout once: `Row([root, Col([bars…])])` for `right`,
    `Col([root, Row([bars…])])` for `bottom`. Bars are drawn pre-measure
    (same reserved-margin rule as `_apply_shared_colorbars`) and stretched
    post-placement to the union span of their member axes (generalizing
    `_stretch_shared_bar` to scattered members: min/max over member
    positions along the bar's long axis, same end-inset clamp).
  - Node-level `shared_colorbar` flags are ignored in united mode with a
    note ("united colorbars override N group flag(s)").
  - A `panel.colorbar is True` override still forces that panel's own bar
    (explicit override outranks the mode); `False`/`None` follow the mode.
- `per-panel` mode: byte-for-byte today's behaviour (per-panel bars +
  Row/Col shared groups).

### GUI — compose form + arranger

- Compose form gains "Colourbars" controls: mode dropdown
  (Per panel / One per quantity), position dropdown (right / bottom,
  enabled only in united mode).
- Scale-bar controls regrouped under a "Scale bar" heading: mode,
  target-panel dropdown (titles), corner dropdown (bound to
  `style.scale_bar_loc`, same choices as the Style pane's Bar location —
  the two are one setting).
- The arranger renders the schematic for whatever the current settings are
  and lets the mouse set scale-bar target + corner (writes
  `compose.scale_bar_mode = "one-panel"`, `compose.scale_bar_panel`, and
  `style.scale_bar_loc`).

## Error handling

- `validate_recipe` rejects unknown `colorbar_mode`/`colorbar_pos` with the
  usual `StageUserError` + hint.
- United mode with zero groupable panels (all traces/placeholders) renders
  with a note, no error.
- ArrangeDialog never applies an empty grid over a non-empty figure without
  the same orphan-purge path used by Delete (panels removed from the grid
  are removed from the recipe — the dialog says so).
- The Add-dialog step 2 is skippable; nothing about step 1's error handling
  ("no such file", "cannot read") changes.

## Testing

- Qt-free (`tests/`): `PanelDef.title` JSON round-trip + old-recipe load;
  `gridmap` round-trip + unmappable-layout cases + flatten order;
  `validate_recipe` on the new enums; renderer united-mode tests on Agg
  (bar count = quantity-group count, member clims unified, per-panel bars
  suppressed, `right` vs `bottom` wrapping, trace/None-group exclusion,
  group-flag-ignored note, panel-level `colorbar=True` override).
- Qt (existing figure-builder test patterns): arranger `set_grid/grid`
  round-trip, drag-model mutations via item moves, column add/remove/merge;
  Add-dialog two-step accept returns panels + fragment; Arrange dialog
  clean-grid preservation of Col flags and the warned flatten path; titles
  shown in outline + scale-bar combo (ids in item data).
- GUI smoke: the builder window still opens; new dialogs open/close.

## Docs (same change)

- `docs/Usage.md`: Add-panels two-step flow, Arrange…, united colorbars,
  scale-bar target/corner via the arranger, titles in the outline.
- `docs/Codebase.md`: `gridmap.py`, `layout_arranger.py`, new
  `ComposeStyle` fields, `PanelDef.title`, renderer united-bar pass.
