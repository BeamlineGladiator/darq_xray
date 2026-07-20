# Fixed-scale trace figures + box-only ratio — design

**Date:** 2026-07-20 · **Status:** approved (design + inline execution)
**Requested by:** Albert — "change the line profile line plots so that the
micron per cm applies to them as well, so that the horizontal axis is always
the same. Also make the ratio of the plot apply to the box inside the plot,
not the total png."

## Goal

1. The publication style's **Scale (µm/cm)** applies to the per-field trace
   (line-profile) figures: the distance axis prints at the same physical scale
   as the maps, so horizontal axes line up across traces and against maps.
2. The aspect ratio's single job is the **plot box** (it already pins the box
   via `set_box_aspect`); the `trace_file_aspect` knob that pinned the total
   PNG is **removed entirely** — the saved PNG is always a tight crop around
   box + labels.

## Behaviour

- **Fixed-scale mode** (style's `scale_um_per_cm` set, read via the defensive
  `plotting.fixed_scale`): trace plot-box width = `geom["L"] / scale` cm;
  box height = width × H/W from `trace_aspect`. Implemented with the shared
  primitives: `fixed_scale_box(style, L, L·H/W)` (inherits the 30-inch clamp
  with warning, aspect preserved, degenerate-L guard → legacy) and
  `fit_axes_to_box` run last inside `build_trace_figure`, so the box is
  point-exact regardless of label/font sizes. `trace_width_in` is **ignored**
  in this mode (same rule as `figure_width` for maps).
- **No scale set:** today's behaviour unchanged (canvas `width_in × width_in·H/W`,
  box ratio pinned, tight crop).
- **Coverage:** run, replot (`render_replot`), and publication export all flow
  through `build_trace_figure` — all three get fixed scale automatically. The
  companion figure stays excluded (look pinned). Overview/companion map panels
  already had fixed scale.
- **`trace_file_aspect` removal:** Param leaves the profiles `STAGE` spec (the
  schema-built form drops the field by itself); `_save_traces` loses the
  file-ratio branch (always `bbox_inches="tight"`); `run()` loses its
  fail-fast `parse_aspect(trace_file_aspect)`; the export `figures()` path and
  `stage_view._replot_profiles`'s params whitelist drop the key. Stale
  persisted form values / stray params keys are ignored harmlessly (the
  `{**defaults, **params}` merge tolerates extras; verified by test).

## Tests

- Box width exactly `L/scale` inches (`ax.get_window_extent` after draw), box
  ratio = trace_aspect, at a chosen scale.
- `trace_width_in` change is a no-op in fixed mode; governs size as today
  without a scale.
- Clamp path (tiny scale → 30 in max, no exception).
- No-scale path unchanged (existing trace tests keep passing untouched except
  the two `trace_file_aspect` tests, which are removed).
- `run()` with a stray `trace_file_aspect` key in params succeeds (stale
  persisted forms).

## Docs (same change)

- `Usage.md`: trace-figure table — drop the File aspect row, note the
  fixed-scale rule (width from µm/cm, Trace width ignored, PNG always tight
  crop); update the fixed-scale section's exclusion note ("companion and
  traces excluded" → companion excluded, **traces included**).
- `Codebase.md`: `build_trace_figure`/`_save_traces` entries + param list.

## Out of scope

- Companion figure fixed scale (pinned look).
- Y-axis physical scaling (trace y is a value axis, not spatial).
