# Figure builder: trace y-axis numbers + external image panels — design

Date: 2026-09-02
Status: APPROVED DESIGN — not yet implemented.
Baseline: master `fc0204b`

## Problem

Two gaps in the figure builder (`dfxm/compose` + `gui/figure_builder.py`):

1. **A line-shape (trace) panel always prints its y-axis numbers.** The sum
   intensity of a profile is an arbitrary-unit quantity, and a publication
   figure often wants the y-label without the numbers. Today nothing can hide
   them: `PlotStyle.axes_mode` ("Full / No frame / None") is map-only by
   contract (`plotting.py:804` — *callers must not apply this to trace axes*),
   and `apply_axis_tickfmt` deliberately ignores the `"arb"` tick format on a
   line axis (`plotting.py:1104`). The `shared_x` plumbing hides *x* tick
   labels on all-but-last traces in a Col, but there is no y counterpart.
2. **An image from another source cannot be a panel.** The only workaround
   (documented in `docs/Usage.md` §Spacer, commit `fc0204b`) is to reserve a
   Spacer, export SVG/PDF, and overlay the image in a vector editor. Every
   loader in `adapters._LOADERS` opens HDF5; nothing in `dfxm/` or `gui/` reads
   a raster file.

## Decisions taken with the user

| Question | Decision |
|---|---|
| Y-axis numbers: scope | **Per panel**, saved in the recipe (not a figure-wide trace knob). |
| Y-axis numbers: what hides | **Numbers only** — tick labels and the ×10ⁿ offset text. Tick marks, grid and the y-label stay. |
| Image: how referenced | **File path, stored relative** to the recipe (like `h5_path`), not embedded. |
| Image: sizing | **Width in cm; height from the image's pixel aspect.** |
| Image: labels | **Lettered like any panel, no title.** |
| Image: architecture | **A panel kind** (`"image"` in `PANEL_KINDS`, ordinary `PanelRef`), not a layout leaf. Reuse `PanelSource.h5_path` as the source file path. |

The panel-kind choice is what makes lettering, the drag arranger, per-panel
overrides and relative-path round-tripping work with no special-casing:
`_assign_labels` walks `PanelRef`s only (`render.py:78`), and
`gridmap.layout_to_grid` returns `None` for any non-`PanelRef` leaf, so a
layout-leaf image would have needed both patched.

## Section 1 — per-panel y-axis numbers on trace panels

### Schema (`dfxm/compose/recipe.py`)

- `PanelDef.y_tick_labels: bool = True`. Comment: *"trace panels only —
  False hides the y tick labels and offset text; ignored by other kinds"*
  (precedent: `crop_to_data` is "ignored by trace panels").
- `_panel_def_to_dict` writes `"y_tick_labels"`; `_panel_def_from_dict` reads
  `bool(d.get("y_tick_labels", True))`. Absent key in an old recipe → `True`.
  `RECIPE_VERSION` stays 1 (additive change, same as every field so far).
- No new validation: a bool has no invalid values.

### Render (`dfxm/compose/adapters.py`)

In the `profiles_trace` branch of `draw_panel`, after `draw_trace_axes` and
the existing `labelbottom` handling:

```python
if not panel.y_tick_labels:
    ax.tick_params(labelleft=False)
    ax.yaxis.get_offset_text().set_visible(False)
```

`draw_trace_axes` itself (`dfxm/stages/profiles.py:943`) is **not** changed —
the profiles stage's own trace figures keep their numbers. Nothing else in the
pipeline is affected. `measure_cells` (`layout.py:373`) measures the tight bbox
of what was drawn, so the panel's left margin shrinks by itself; no sizing
change is needed. `_align_axis_labels` still aligns the surviving y-labels
across a row.

### GUI (`gui/figure_builder.py`)

One checkbox `self._ov_ynums = QCheckBox("Y-axis numbers")` in
`_build_override_editor`, tooltip *"Untick to drop the tick numbers (and any
×10ⁿ offset) from this trace's y-axis; the tick marks and the y-label stay.
Trace panels only."* Wired through the existing trio:

- `_on_override_field_edited` getter `"y_tick_labels": self._ov_ynums.isChecked`
- `_apply_panel_overrides`: `changes["y_tick_labels"] = bool(values["y_tick_labels"])`
- `_load_panel_page`: set checked from the panel, inside the existing
  blockSignals loop; **enabled only when `panel.source.kind == "profiles_trace"`**.

### Docs

- `docs/Usage.md` §Figure builder → *Selected node* → **Panel** page: add the
  checkbox, say what stays and what goes.
- `docs/Codebase.md` `#### recipe.py` `PanelDef` field list; `#### adapters.py`
  `draw_panel` trace branch; the `figure_builder.py` row.

### Tests

- `tests/test_compose_recipe.py`: `test_y_tick_labels_round_trips_and_old_recipe_defaults_true`
  (the established pair-of-asserts pattern, cf. line 261).
- `tests/test_compose_adapters.py`: draw a trace with `y_tick_labels=False`;
  assert every `ax.get_yticklabels()` is invisible, the offset text is
  invisible, the y-label text is non-empty and visible, and the bottom tick
  labels are untouched. Draw a map panel with the flag False and assert nothing
  changes (ignored by kind).
- `tests/test_gui_figure_builder.py`: select a trace panel, toggle the box,
  assert the `PanelDef` flips, the window is dirty, and the recipe JSON carries
  the key; select a map panel and assert the box is disabled.

## Section 2 — external image panel

### Schema (`dfxm/compose/recipe.py`)

- `PANEL_KINDS = ("map_layer", "slice_plane", "profiles_ref", "profiles_trace", "image")`.
- Source: `PanelSource(h5_path=<image file>, kind="image", selector={})`.
  The `h5_path` field is documented (dataclass comment, `adapters.py` module
  docstring, Codebase.md) as *"source file path — an `.h5` for the data
  kinds, a PNG/JPEG/TIFF for `image`"*. No rename: the JSON key stays
  `"h5_path"` so v1 recipes stay readable, and the existing
  `_rel_path`/`_resolve_path` handling on save/load applies unchanged.
- `PanelDef.width_cm: float | None = None` — *"image panels only: printed
  width; None = `IMAGE_DEFAULT_WIDTH_CM` (6.0)"*. Written/read explicitly in
  the panel-dict helpers; `validate_recipe` rejects a non-positive value with
  a `StageUserError` (same loop shape as the trace knobs, `recipe.py:451`).
- `RECIPE_VERSION` stays 1.

### Loader (`dfxm/compose/adapters.py`)

`_load_image(path, sel, roi, *, crop_to_data=False) -> PanelData`, registered
in `_LOADERS["image"]`:

- `matplotlib.image.imread(path)` — PNG natively, JPEG/TIFF via Pillow, which
  matplotlib already requires. **No new dependency**, so
  `tests/test_docs_dependencies.py` and the two docs lists are untouched.
- Normalises to a float RGB(A) array in 0–1 (uint8 → /255; 2-D greyscale kept
  as 2-D and drawn with `cmap="gray"`).
- `roi` is applied as a plain pixel crop `arr[r0:r1, c0:c1]` when set (same
  `(r0, r1, c0, c1)` convention as the maps); an empty crop raises → placeholder.
  `crop_to_data` is ignored.
- Returns `PanelData(kind="image", ext_x_um=float(w_px), ext_y_um=float(h_px),
  group=None, payload={"image": arr})`. `ext_*` carry **pixels**, used only
  for the aspect — the image cell never consults the µm/cm scale.
- Any read failure (missing file, undecodable bytes, zero-size crop) goes
  through the existing `except Exception` in `load_panel` and becomes a
  placeholder whose reason names the path — the composition survives.
- `_cache_key` already includes `h5_path`, `kind`, `selector`, `roi`, so the
  per-render cache works without change.

`panel_preview` raises `StageUserError` for `"image"` the way it does for
traces (*"an image panel's ROI is a pixel crop — type it in the ROI box"*).
The ROI picker preview for images is **out of scope**.

### Sizing (`dfxm/compose/layout.py`)

`leaf_cell` dispatches `data.kind == "image"` to a new `_image_cell`:

```
pinned row height h  → w = h · w_px / h_px   (note: "pinned row height — image width follows")
pinned col width w   → h = w · h_px / w_px
both pinned          → height wins, width ignored (same rule as _map_cell)
neither              → w = (panel.width_cm or IMAGE_DEFAULT_WIDTH_CM) / 2.54 in, h from aspect
```

Cell kind is `"image"` (a new `SizedCell.kind` value). Degenerate pixel
dimensions → placeholder, mirroring `_map_cell`'s degenerate-extent branch.
`trace_column_targets.widest_map` (`layout.py:278`) keys on `cell.kind ==
"map"`, so an `"image"` cell is never a trace-autoscale target — correct, an
image has no physical scale to autoscale a trace to. A test pins this.

### Draw (`dfxm/compose/adapters.py` + `render.py`)

- `draw_panel` branch `data.kind == "image"`: `ax.imshow(arr, interpolation="none",
  aspect="auto")` (the cell already has the right aspect, so `"auto"` fills
  it edge to edge), `ax.set_axis_off()`, return `None`. `titled` is ignored —
  image panels have no title by decision.
- `render_recipe` draw loop: a new `elif cell.kind == "image"` calling
  `draw_panel(ax, panel, data, style, show_title=False)`. The image leaf gets
  an axes in the leaf loop like any `PanelRef` (it already does — the loop
  keys on `PanelRef`, not on kind).
- **Excluded from colourbar and scale-bar bookkeeping.** `group=None` means
  united colourbars never adopt it; the `("map_layer", "slice_plane",
  "profiles_ref")` tuples at `render.py:713` and `render.py:1115` are left as
  they are (an image is not a map). `_cbar_label` is never reached. A
  `colorbar=True` override on an image panel is ignored silently.
- Lettering: no change — `_assign_labels` sees the `PanelRef`.
- `gridmap.panel_group_hint` returns `"image"` for the kind so the arranger
  chip gets a neutral colour (add the key to whatever colour table the
  arranger uses; grey).

### GUI (`gui/figure_builder.py`, `gui/widgets/layout_arranger.py`)

- **"Add image…" button** next to "Add panels…" (`figure_builder.py:256`),
  tooltip *"Place a PNG/JPEG/TIFF — e.g. a panel reproduced from another
  paper — as a lettered panel. Set its printed width in the inspector."*
  Handler `add_image(path: str)` (testable, no `exec()`): builds
  `PanelDef(id="image_<n>", source=PanelSource(h5_path=path, kind="image",
  selector={}), title=<basename>)` and hands it to the existing
  `add_panels([panel])`, which uniquifies the id, appends the `PanelRef` to
  the current container, and refreshes. The button's slot opens a
  `QFileDialog.getOpenFileName` with an image filter, starting in the
  recipe's directory when a recipe is loaded.
- **Inspector, panel page:** a `QDoubleSpinBox` "Width (cm)" (`_ov_width`,
  range 0.1–30, `specialValueText` "default (6 cm)" at 0 → `None`), wired
  through the getter/changes/load trio. `_load_panel_page` enables widgets by
  kind:

  | widget | map/slice/ref | trace | image |
  |---|---|---|---|
  | ROI text | on | off | on (pixel crop) |
  | Pick… / → all maps | on | off | off |
  | Crop to data | on | off | off |
  | Colour limits, Colormap, Colourbar | on | off | off |
  | Label | on | on | on |
  | Show title | on | on | off |
  | Panel scale | on | off | off |
  | Width (cm) | off | off | on |
  | Y-axis numbers | off | on | off |

  (Today every widget is always enabled; this table is the new rule. The
  existing `"profiles_trace"` special cases at `figure_builder.py:1110`,
  `1124`, `1160` become *"kind not in image kinds"* checks so `→ all maps`
  and the picker skip image panels too.)
- Outline label: `_node_label` shows the panel title (the file name) as for
  any panel — no change needed.
- Arranger: image panels are ordinary `PanelRef`s and map to grid cells
  without change.

### Docs

- `docs/Usage.md` §Figure builder:
  - **Add panels…** subsection gains an **Add image…** paragraph (what it
    accepts, width in the inspector, lettered, no title, pixel ROI crop by
    typing, missing file → placeholder, relative path travels with the
    recipe).
  - **Spacer** bullet (2545–2555): replace *"external images cannot be added
    as panels"* and the SVG-overlay workflow with a pointer to Add image…;
    keep the Spacer's "exact-size empty spot" purpose and its two caveats.
  - **Panel** page: the Width (cm) spin and the per-kind enable table above.
  - **Concepts → Panels**: list the fifth kind.
- `docs/Codebase.md` `### dfxm/compose`: `PANEL_KINDS`, `IMAGE_DEFAULT_WIDTH_CM`,
  `PanelDef.width_cm`, `_load_image`, `_image_cell` / cell kind `"image"`,
  `draw_panel` branch, `panel_group_hint`; `figure_builder.py` row gains
  `add_image` and the enable-by-kind rule.

### Tests

- `tests/test_compose_adapters.py` (fixture: a 40×20 RGB PNG written with
  matplotlib's `imsave` into `tmp_path`): loads with the right `ext_*`;
  missing file → placeholder naming the path; `roi` crops to the expected
  shape; empty crop → placeholder; `draw_panel` dispatch draws one
  `AxesImage` and turns the axis off; `panel_preview` raises.
- `tests/test_compose_layout.py`: `_image_cell` under no pin (6 cm default,
  aspect 2:1), `width_cm=3`, pinned row height, pinned col width, both pinned
  (height wins + note).
- `tests/test_compose_recipe.py`: `width_cm` round-trip + old-recipe default
  `None`; `validate_recipe` rejects `width_cm=0`; an `"image"` panel with a
  relative `h5_path` resolves against `base_dir` on load.
- `tests/test_compose_render.py`: map – image – map in a Row letters A, B, C;
  the image panel gets no colourbar axes in `style.colorbar=True` mode; a
  united colourbar over two maps is unaffected by an image between them.
- `tests/test_compose_cli.py`: a recipe with an image panel renders headless.
- `tests/test_gui_figure_builder.py`: `add_image(path)` appends a panel and a
  `PanelRef`; selecting it enables Width and disables clim/cmap/colourbar/
  scale/crop/picker; editing Width writes `width_cm` and dirties; `→ all
  maps` skips it.

## Out of scope

Embedding image bytes in the recipe; image titles; DPI- or physical-size-based
sizing from image metadata; ROI picker previews for images; SVG input.

## Implementation order

1. Section 1 end to end (schema → render → GUI → docs → tests): one small
   commit, independently useful.
2. Section 2 core (`recipe`, `adapters`, `layout`, `render`, `gridmap`) with
   core tests and the CLI test.
3. Section 2 GUI + docs + builder tests.
