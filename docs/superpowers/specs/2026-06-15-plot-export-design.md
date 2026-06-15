# Design spec: publication-quality plot export

Date: 2026-06-15
Status: approved (brainstorm), ready for implementation planning
Provenance: "project 2" deferred from the 2026-06-11 GUI-overhaul brainstorm; the
GUI overhaul (its own spec/plan) is now merged to `master`. This is its own spec.

## Goal

Let a beamline scientist turn any figure the pipeline saves into a
publication-ready image — larger text and colourbars sized for use as a paper
**subfigure**, a micron scale bar, and vector output — with a **live preview**
before exporting. The defining use case is shrinking a figure into a multi-panel
paper figure where default Matplotlib text becomes unreadable.

## Scope

**In scope:** publication export for **every figure the pipeline saves** —
strain maps, the strain histogram and detrend diagnostic, mosaicity maps,
visualize/rocking per-layer images, oblique slices, line-profile plots, and
matched detector frames.

Figures split into two kinds:
- **`map`** — has a physical (µm) extent: gets a scale bar + the full styling.
- **`plot`** — no µm extent (histograms, line traces): gets the font/colourbar
  styling and format export, but **no** scale bar.

**Non-goals:** changing the science/computation; redesigning the Results/Output
tabs beyond adding the export entry points; a separate batch CLI beyond what the
core API enables for free; per-paper template management.

## The control set (locked during brainstorm)

A figure's appearance is fully described by a `PlotStyle`. Controls:

- **Scale bar** (map figures only): show; length (`auto` or fixed µm);
  thickness (pt); label size (multiplier, independent of the rest of the text);
  position (corner); colour; **background box** (show, colour, opacity, margin)
  — a rounded rectangle enclosing bar + label so it stays legible over light
  regions.
- **Text** (axes labels, ticks, title): overall size multiplier (~1–3×, default
  ~2.2× for publication); hide-title toggle; centre-X/Y-axis-labels toggle.
- **Colourbar:** show; **label** (editable text, defaults to the figure's own,
  e.g. "Strain (ε)"); thickness; **ticks** = a count that always includes
  min / mid / max plus evenly-spaced in-between values; tick number **format**
  (`auto` | `scientific` | fixed decimals).
- **Figure:** width preset (single-column | double-column | custom inches).
  **Physical aspect is always equal** (µm extent + `aspect="equal"`), so a
  0.1 × 0.3 µm/pixel field renders a true 1 : 3 — never stretched. This is
  enforced by the builders, not a user knob.
- **Output:** formats — **PNG** (with DPI) + **PDF** + **SVG**; the user ticks
  which to write.

A background box is provided **only** behind the scale bar (not the colourbar).

## Workflow (hybrid)

- **Global publication style** — a session-level `PlotStyle` (seeded from a
  `PUBLICATION_STYLE` preset) set via a "Publication style…" editor. Used by
  "export all".
- **Per-figure dialog** — starts from the global style, allows local overrides
  with a live preview, exports the single figure.
- **Export all** — re-renders every figure of a stage's result at the global
  style in one action.

## Architecture (Approach A: rebuild from data)

Figures are **rebuilt from the persisted data**, not restyled as rasters. This
is feasible because the data behind nearly every figure is already saved:
strain/mosaicity/rocking write their volumes to HDF5, slices writes
`oblique_slices.h5`, paraview writes `.pvti`; visualize/paraview render from the
volume files; only `matched` re-reads raw frames (its inputs remain on disk).

All rendering and data-loading live in the **Qt-free, headless-testable core**
(`dfxm/`); the GUI is a thin shell. Three layers:

### 1. Core styling layer — `dfxm/common/plotting.py`

- **`PlotStyle`** dataclass with the fields above. Two named instances:
  - **`DEFAULT_STYLE`** — reproduces today's look exactly, so a normal stage run
    is unchanged when its builders are called with the default.
  - **`PUBLICATION_STYLE`** — the larger-text/scale-bar/colourbar starting point.
- **Styled primitives**, each taking a `PlotStyle`:
  - `draw_scale_bar(ax, length_um, style)` — bar + label + optional background
    box. **Unifies the two scale-bar implementations that exist today**
    (`plotting.add_scale_bar` and `render.add_scale_bar`) into one.
  - `apply_text_scale(ax, style)` — scales tick/label/title fonts; handles the
    centre-axis-labels toggle.
  - `add_colorbar(fig, im, ax, label, style)` — thickness, the configurable
    N-tick set (linspace over vmin..vmax including the midpoint), and the tick
    number format.
- Each stage's existing figure functions (e.g. `strain._save_strain_map`,
  `render.layer_figure`, `slices.render_slice_png`) are refactored to take
  `style=DEFAULT_STYLE` and route through these primitives — reusable at any
  style, current behaviour preserved as the default.

### 2. Figure catalog — `dfxm/common/figures.py`

- **`FigureSpec`** — describes one rebuildable figure: `figure_id`, human
  `title`, `kind` (`"map"` | `"plot"`), a suggested export filename, and a
  `build(style) -> matplotlib.figure.Figure` callable. The callable is a closure
  that loads the figure's data from the persisted output (HDF5 volume,
  `oblique_slices.h5`, `.pvti`, or raw frame for matched) and calls the stage's
  refactored style-aware builder.
- **Per-stage ownership:** each stage module exposes a module-level
  `figures(result, params) -> list[FigureSpec]` next to its `run()` (the stage
  already owns its plotting). Volume stages enumerate one `FigureSpec` per layer;
  strain returns map + histogram + detrend diagnostic; etc.
- **Dispatch + enforcement:** `figures.py` maps `stage_name → that function`
  (same pattern as `STAGE_TARGETS` / the GUI `_SUMMARIZERS`) and exposes
  `figures_for(stage_name, result, params)`. A registry-sync test asserts every
  stage in `STAGE_TARGETS` has a catalog entry, so a new stage can't silently
  miss export support.

The GUI never touches stage internals — only `figures_for(...)` and
`spec.build(style)`.

### 3. GUI — thin shell

- **`gui/widgets/export_dialog.py`** — per-figure dialog: an `MplCanvas` preview
  (reuse the existing widget) of `spec.build(working_style)`; a controls panel
  bound to a working `PlotStyle` (copy of the global style); live re-render on
  edit, debounced via a short `QTimer`; a figure selector to switch between the
  result's figures; **Export** and **Reset to global style** buttons. For a
  `plot`-kind figure the scale-bar group is hidden (no µm extent to scale to).
- **Global publication-style editor** — a "Publication style…" button on the
  Output tab opens the same controls (minus per-figure scale-bar
  length/position) and sets the session-level global `PlotStyle`.
- **Output-tab entry points** (`gui/stage_view.py`): **Export…** (opens the
  dialog) and **Export all…** (batch).

## Data flow

1. A stage runs → result carrying saved data paths.
2. Output tab **Export…** → `figures_for(stage, result, params)` → dialog.
3. Dialog renders `spec.build(working_style)` into the preview; edits re-render.
4. **Export** → `fig.savefig` for each ticked format into an `exports/`
   (publication) subfolder of the stage's output dir, named from the
   `FigureSpec` + extension.
5. **Export all…** → loop `figures_for(...)`, build each at the global style,
   save all formats.

## Error handling

- If a figure's data file is missing/moved, `spec.build` raises and the dialog
  shows a message (no crash). **Export all** collects per-figure failures and
  reports them in a summary — skip-and-report, not a batch-aborting exception
  (consistent with the pipeline's existing empty-result reporting).
- matched-without-raw → a clear message.
- Style values are clamped (ticks ≥ 2, dpi within a sane range).
- Everything renders headless (Agg), so `build`/export is testable — and usable
  from a CLI — without a display. The core stays Qt-free; no `pyplot`, no
  `matplotlib.use`.

## Module layout

- `dfxm/common/plotting.py` — `PlotStyle`, `DEFAULT_STYLE`, `PUBLICATION_STYLE`,
  styled primitives (extends the existing file).
- `dfxm/common/figures.py` — `FigureSpec`, catalog dispatch (`figures_for`).
- `dfxm/stages/*.py` — refactored style-aware builders + `figures(result, params)`.
- `gui/widgets/export_dialog.py` (+ a small style-form widget if warranted),
  wired into the `gui/stage_view.py` Output tab.

## Testing

- Core unit tests: `draw_scale_bar` (geometry + background box), `add_colorbar`
  (tick count incl. midpoint + each number format), `apply_text_scale`; and
  `figures_for`/`spec.build` per stage from synthetic persisted data (assert the
  Figure's artists and that physical aspect is equal).
- **`DEFAULT_STYLE` regression:** a figure built at `DEFAULT_STYLE` matches the
  current output, proving the refactor doesn't change normal runs.
- Catalog registry-sync test (every stage has a catalog entry).
- `tests/gui_smoke.py` extension: open the export dialog offscreen, tweak the
  style, export to a tmp dir, and assert the PNG/PDF/SVG files exist and are
  non-empty; a small "export all" smoke.

## Documentation (repo contract)

Update in the same change as the code: `docs/Usage.md` (the export workflow and
each control) and `docs/Codebase.md` (`PlotStyle` + primitives, the `figures`
catalog, the export dialog).

## Suggested phasing (to be detailed in the implementation plan)

1. Core: `PlotStyle` + styled primitives + the `DEFAULT_STYLE` regression
   (no behaviour change yet).
2. Catalog + **map** figures + the per-figure export dialog with live preview
   (strain/mosaicity/slices/volume layers/matched).
3. **plot** figures (histogram, detrend diagnostic, profile traces) + the global
   style editor + "Export all".
4. Documentation sweep.
