# Profiles stage — separate per-field trace figures

Date: 2026-07-07
Stage: `dfxm/stages/profiles.py` (Line profiles)

## Problem

The profiles stage emits **one stacked "companion" figure** per job
(`build_companion_figure`): a large reference-image panel on top
(height ratio 3.0, with the profile line drawn on it) and **N trace panels**
beneath it (one per profiled field — intensity, strain, misorientation —
each height ratio 1.0). The whole figure is `9 × (4.8 + 1.85·N)` inches.

Two consequences make the traces hard to read, especially for publication:

1. **Structural** — each trace is squeezed into a ~1.85-inch band under a
   dominant image, so the curve itself has almost no vertical room.
2. **Text** — the trace labels are hard-coded (`ylabel=10 pt`, panel
   `title=10 pt`, `xlabel=12 pt`). On the headless/CLI path (`style is None`)
   nothing scales them. Through the GUI, the publication `font_scale` *does*
   multiply them (`apply_text_scale`), but enlarging text inside a 1.85-inch
   panel just makes labels collide with ticks.

The user wants each profiled field exported as its **own full-size figure**
with **legible, user-controllable labels** and a **settable aspect ratio**
(e.g. 4:3, 1:1) so it drops cleanly into a paper.

## Goals

- Export each profiled field as its own standalone line-profile ("trace")
  figure, with room for readable labels.
- Let the user set the trace figure's **aspect ratio** (free-text `W:H`) and
  **width**.
- Let the user set the **line thickness** of the plotted profile curve, so a
  single trace reads clearly as a subfigure.
- Let the user **override the curve colour** (line + std band), blank = the
  current default.
- Give the trace figures a **dedicated font scale**, independent of the map
  figures' `font_scale`.
- Keep the existing stacked companion figure available via a toggle, with its
  pinned appearance unchanged.

## Non-goals

- No change to the companion figure's rendering or its regression-pinned look
  (`tests/test_figures_catalog.py`).
- No change to the overview images (plane + line), which already export
  separately via `save_overview` / `_save_overviews`.
- Aspect ratio applies to the 1-D **trace** figures only, not to the
  map-like overview/reference images (whose aspect follows physical extent).
- No per-job aspect/width/font override in `jobs_json` for now (global stage
  params only; can be added later if needed).
- Profiles keeps its existing publication-export (`figures()`) integration; it
  gains no cold "Replot…" dialog.

## Decisions (from brainstorming)

- **Primary output** = separate per-field trace figures.
- **Companion figure** = kept, gated behind a toggle (default ON).
- **Label sizing** = a trace-specific font control, independent of the map
  `font_scale`.
- **Aspect entry** = free-text `W:H` string (validated), not a fixed dropdown.

## New stage parameters

Added to `profiles.STAGE.params` (schema-driven — the GUI auto-builds the
form; `tests/test_param_metadata.py` enforces `help`/`group`/etc.):

| Param | Type | Default | Group | Help (user-facing intent) |
|---|---|---|---|---|
| `save_traces` | BOOL | `True` | Output | Write each profiled field as its own line-profile figure (separate from the stacked companion). |
| `save_companion` | BOOL | `True` | Output | Also write the stacked companion figure (overview image + all traces in one). Turn off to export only the separate traces. |
| `trace_aspect` | STR | `"4:3"` | Appearance | Aspect ratio (width:height) of each separate trace figure, e.g. `4:3`, `1:1`, `16:9`. |
| `trace_width_in` | FLOAT (unit `in`) | `6.0` | Appearance | Width of each separate trace figure in inches; height follows the aspect ratio. |
| `trace_linewidth` | FLOAT (unit `pt`) | `2.0` | Appearance | Line thickness of the plotted profile curve on the separate trace figures. |
| `trace_color` | STR | `""` | Appearance | Colour of the profile curve and its std band on the separate trace figures (blank = default matplotlib blue, `C0`). |
| `trace_font_scale` | FLOAT | `1.4` | Appearance | Multiplies the label/tick/title font size of the separate trace figures (independent of the map figures' font scale). |

All are `advanced=True` and grouped, consistent with the stage's other
appearance/output params.

## Rendering (new code in `profiles.py`)

### `parse_aspect(s: str) -> tuple[float, float]`

Parse a `"W:H"` string into `(w, h)` floats. Raise
`StageUserError(message, hint=...)` when the string is not exactly two
positive, finite numbers separated by `:` (e.g. `"4:3"`, `"1.618:1"`).
Hint guides the user to the `W:H` form.

### `build_trace_figure(fld, geom, *, aspect_wh, width_in, linewidth, color, font_scale, style) -> Figure`

Build one standalone trace figure for a single field. Does **not** savefig.

- Figure size `(width_in, width_in * h / w)` where `(w, h) = aspect_wh`.
- `color` is the curve colour; an empty/`None` value falls back to `"C0"`
  (the companion default), so blank reproduces today's look.
- Single axes: `ax.plot(distance, value_mean, lw=linewidth, color=color)`; if
  `value_std is not None`, `ax.fill_between(distance, mean±std, color=color)`
  band at the same alpha/z-order as the companion's trace panels. Only the
  curve's `lw` is user-set; the band stays a fill (no line width).
- `ax.set_xlabel("distance along line (µm)")`, `ax.set_ylabel(cbar_label)`,
  `ax.set_title(f"{kind}  |  {vid}  |  {src}", loc="left")`,
  `ax.grid(...)`, `ax.set_xlim(0.0, geom["L"])`.
- Base font sizes match the companion trace panels (10/10/12), then **all
  trace text is multiplied by `font_scale`** (xlabel, ylabel, ticks, title).
  This is the trace figures' own scale — it does **not** compound with the map
  `style.font_scale`.
- Uses `styled_figure(size, styled=style is not None)` so the background/layout
  engine matches the rest of the stage's output. No colorbar (1-D plot).

### `_save_traces(out_dir, stem, fields, geom, *, aspect, width_in, linewidth, color, font_scale, dpi, style) -> list[str]`

Loop the fields, build each trace figure, savefig to
`{stem}__trace__{vid}.png` (white facecolor, `bbox_inches="tight"`,
`dpi`). Return the list of written paths.

## `run()` changes

- Parse the new params: `save_traces`, `save_companion`,
  `parse_aspect(trace_aspect)`, `trace_width_in`, `trace_linewidth`,
  `trace_color` (blank → `"C0"`), `trace_font_scale`.
- Parameter mode, per job:
  - Build/save the companion **only if** `save_companion` (else
    `jr.figure = None`).
  - If `save_traces`, call `_save_traces(...)` and store the paths in
    `jr.traces`.
  - CSVs and overviews behave exactly as today.
- Preview mode is unchanged (it renders a single overview via `render_single`).

### Result dataclass

`ProfileJobResult` gains:

```python
traces: list[str] = field(default_factory=list)
```

`figure` stays typed `str | None` (already is) and may now be `None` when
`save_companion` is off.

## Publication export — `figures()`

`figures()` currently registers one companion `FigureSpec` (kind `"plot"`) per
parameter-mode job. It will mirror the toggles:

- **Companion spec** (if `save_companion`): unchanged — one per job.
- **Trace specs** (if `save_traces`): one `FigureSpec` per (job × field),
  iterating `jr.fields`:
  - `figure_id = f"profiles_{name}__trace__{vid}"`
  - `title = f"Profile trace: {name} · {vid}"`
  - `filename = f"{fig_name}__trace__{vid}"` (share the companion's stem
    disambiguation so repeated `fig_name`s don't collide)
  - `kind = "plot"`
  - `build(style)` re-`_collect`s the job, selects the field by `vid`, and
    calls `build_trace_figure` with
    `aspect`/`width_in`/`linewidth`/`color`/`font_scale` **captured from
    `params`** in the closure (same closure pattern the companion build already
    uses for `h5`/`job`/`ref`/`restrict`).

This keeps parity: whatever formats the publication export offers (PNG/PDF/SVG,
styled), the separate traces are included too.

## GUI wiring (`gui/stage_view.py`)

- `_summarize_profiles`: append `traces=N` to each job line and render a
  `None` companion as `-> (no companion)` instead of `-> None`.
- `_image_profiles`: return `j.figure` when present, else `j.traces[0]` when
  present, else `None` — so the Results preview still shows something when the
  companion is toggled off.

No `bindings.py` change is required (no new experiment pre-fill/chaining; the
new params default sensibly). The interactive line-picker path is untouched.

## Tests

- `tests/test_param_metadata.py` — already enforces the new params carry
  `help`/`group`; they must comply.
- `parse_aspect`: valid (`"4:3"`, `"1:1"`, `"1.5:1"`) → correct tuple;
  invalid (`""`, `"4"`, `"4:0"`, `"a:b"`, `"1:2:3"`) → `StageUserError`.
- `build_trace_figure`: returned figure size ratio matches the requested
  aspect within tolerance; the plotted curve's `Line2D` linewidth equals
  `trace_linewidth`; a non-blank `trace_color` sets the curve colour and blank
  falls back to `"C0"`; a `value_std=None` field renders without a band.
- `run()` toggles: `save_traces=True` populates `jr.traces` and writes the
  files; `save_companion=False` yields `jr.figure is None` and writes no
  companion PNG; both on writes both.
- `figures()`: with both toggles on, emits `1 + len(fields)` specs per job
  (companion + one per field); `save_companion=False` drops the companion spec;
  `save_traces=False` drops the trace specs. Each trace `build(style)` returns a
  `Figure`.
- `tests/test_figures_catalog.py` — update for the new spec set; keep the
  companion-look regression assertions intact.

## Docs (same change)

- `docs/Usage.md` (profiles Stage-reference section): document `save_traces` /
  `save_companion`, `trace_aspect` / `trace_width_in` / `trace_linewidth` /
  `trace_color` / `trace_font_scale`; explain that each field now exports as its
  own figure and how the aspect, line-thickness, and colour knobs shape it for a
  paper.
- `docs/Codebase.md` (`dfxm/stages` → profiles): add `parse_aspect`,
  `build_trace_figure` (incl. its `linewidth` and `color` args), `_save_traces`,
  the `ProfileJobResult.traces` field, and the expanded `figures()` behaviour.

## Rollout risk

Low. The companion path and its regression pins are untouched; new behaviour is
additive and gated by defaulted toggles. The only outward change with defaults
as shipped is that a parameter-mode run now *also* writes `__trace__` PNGs and
the publication export lists extra trace specs.
