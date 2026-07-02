# Styled-figure layout overhaul: constrained layout, title scale, round colour limits

**Date:** 2026-07-02
**Status:** approved (user Q&A in session)

## Problem

Styled map exports (e.g. slices `oblique_full__mosa_com_chi__*.png` at
`figure_width="single"`, `font_scale=2.2`) render with overlapping text: the
two-line title collides with the colorbar top and its ×10⁻² offset text, the
colorbar label collides with its ticks, and axis labels crowd the ticks. This is
structural, not a tuning problem:

1. Styled figures use matplotlib's default fixed margins; nothing reserves space
   for text, so fonts scaled 2.2× on a 3.5-inch figure overflow.
2. `figure_size()` adds a flat +1 in of headroom regardless of font size.
3. `savefig(bbox_inches="tight")` only grows the canvas outward; it cannot fix
   elements overlapping each other.

Additionally the user needs (a) a title size independent of the main font scale
(title is only an identification aid, to be set very small), and (b) round
colour-limit extremes (±0.08 rather than the auto-found ±0.0778) with plain
decimal ticks instead of ×10⁻ⁿ offset notation.

## Decisions (from user Q&A)

- Layout fix: **constrained layout** (approach A), applied to **all styled
  figures**, legacy `style=None` path untouched.
- Title scale: **independent** multiplier on the base title size — `font_scale`
  no longer affects the title at all.
- Round limits: **auto-round toggle** (no manual vmin/vmax in the style), with
  the rounding made **very explicit** to the user (log + results summary + HDF5).
- ×10⁻ⁿ: no behaviour change needed — the existing `colorbar_tick_format`
  already controls it (`scientific` forces the offset; digit counts give plain
  decimals). Make the GUI wording self-explanatory and document it.

## Design

### 1. Constrained layout for styled figures

New helper in `dfxm/common/plotting.py`:

```python
def styled_figure(figsize, *, styled: bool) -> Figure:
    """White-face Figure; constrained layout when styled (reserves space for
    title/labels/colorbar at their final font sizes), plain margins otherwise."""
```

Call sites switch to it, passing `styled=(style is not None)`:

- `dfxm/stages/slices.py` `build_slice_figure` (line ~751)
- `dfxm/common/render.py` `layer_figure` (line ~49) — feeds visualize, rocking
  AND matched (matched builds all its images through `Rnd.layer_figure`)
- `dfxm/stages/profiles.py` map figure (~505); the composite figure (~446) is
  already constrained
- `dfxm/stages/strain.py` styled diagnostic figures (~380, ~437, via `new_figure`)
- `plotting.build_histogram` (~370) when `style is not None`

`style=None` (headless CLI / legacy look) keeps today's plain `Figure` and
byte-similar output. `savefig(..., bbox_inches="tight")` calls stay — tight
bbox and constrained layout compose fine. `apply_text_scale` ordering is safe:
constrained layout resolves at draw/save time, after all font changes.

`figure_size()` keeps its +1 in height headroom as the initial guess; the
layout engine does the real work inside that canvas. Column widths
(`single`=3.5 in, `double`=7 in) remain exact — big fonts shrink the axes, not
the canvas.

### 2. Independent title scale

- `PlotStyle` gains `title_scale: float = 1.0`.
- `apply_text_scale()` sets the title size to `base × title_scale` (previously
  `base × font_scale`). `show_title=False` behaviour unchanged.
- Serialization (`style_to_json`/`style_from_json`/`_style_from_dict`) picks the
  field up automatically; old persisted styles default to 1.0.
- Intentional behaviour change: with `font_scale=2.2` and default
  `title_scale=1.0`, titles come out smaller than before (base size). This is
  the requested semantics.
- GUI (`gui/widgets/export_dialog.py`): "Title scale" `QDoubleSpinBox`
  (0.1–5.0, step 0.1, 2 decimals) placed next to "Show title", wired like
  `font_scale` (live preview + both dialog variants, lines ~100 and ~304
  regions).

### 3. Round colour limits (`round_clim`)

- `PlotStyle` gains `round_clim: bool = False`.
- New pure helper in `dfxm/common/plotting.py`:

```python
def round_limits_outward(vmin: float, vmax: float, sig: int = 2) -> tuple[float, float]:
    """Round limits OUTWARD (floor vmin, ceil vmax) to `sig` significant
    figures. Symmetric input stays exactly symmetric; endpoints equal to 0
    stay 0; degenerate ranges (vmin == vmax) are returned unchanged."""
```

- Applied only to **auto-computed** limits in stages that produce styled map
  exports, when `style.round_clim` is true: slices (`_prep`, ~line 672),
  visualize (`~303/314/342`), rocking (`~531`), matched (auto-percentile path
  at ~399 only — explicit user vmin/vmax params are never rounded), strain
  diagnostic maps. Rounding happens where the stage
  computes its limits so PNGs, animations, colorbars and HDF5 all agree.
- Explicit reporting, three places:
  - **Run log** (per volume): `chi: colour limits rounded ±0.0778 → ±0.0800 (round_clim)`.
  - **Results summary** (stage Results tab): limits row shows
    `clim ±0.08 (rounded from ±0.0778)` — via the `_summarize_*` formatters in
    `gui/stage_view.py` reading the raw values from the results dict.
  - **HDF5 attrs**: stages that store `vmin`/`vmax` (slices `write_volume_group`)
    additionally store `vmin_raw`/`vmax_raw` when rounding changed them.
- GUI: checkbox "Round colour limits (outward, 2 s.f.)" in the export dialog's
  colorbar group.

### 4. Tick-format wording

`_TICK_FMTS` values stay the same (persisted format strings); the combo shows
descriptive labels instead: `auto (matplotlib default)`,
`scientific (×10ⁿ offset)`, `0–3 (plain decimals, N dp)` — display text maps to
the unchanged stored values.

## Non-goals

- No change to the legacy (`style=None`) render path or golden outputs.
- No manual vmin/vmax fields in the style (matched's per-stage params remain the
  manual escape hatch).
- No layout change to non-map figures beyond routing through `styled_figure`.

## Testing

- Style round-trip: `title_scale` + `round_clim` survive
  `style_to_json`/`style_from_json` and `style_from_params`.
- **No-overlap regression** (the bug in the user's image): build a slice figure
  at `font_scale=2.2`, `figure_width="single"`, two-line title, colorbar with
  `scientific` format; draw on an Agg canvas and assert the title, colorbar and
  axes tick/label bounding boxes are pairwise non-overlapping.
- `round_limits_outward` unit tests: symmetric stays symmetric
  (±0.0778 → ±0.08), asymmetric floors/ceils outward, zero endpoints, degenerate
  vmin == vmax, negative-only ranges, sig-figure count.
- Title independence: `apply_text_scale` with `font_scale=3, title_scale=0.5`
  gives title `12 × 0.5` pt while labels scale by 3.
- Summary formatting shows the `(rounded from …)` text when raw ≠ final.
- Existing suite (281 passed / 13 skipped) stays green; gui_smoke unchanged.

## Documentation (same change)

- `docs/Usage.md`: export-dialog reference gains Title scale + Round colour
  limits; a short "Why is there ×10⁻² on my colorbar?" note explaining the tick
  format options; note that styled exports no longer overlap at large font
  scales.
- `docs/Codebase.md`: new `PlotStyle` fields, `styled_figure`,
  `round_limits_outward`, and the touched stage functions.
