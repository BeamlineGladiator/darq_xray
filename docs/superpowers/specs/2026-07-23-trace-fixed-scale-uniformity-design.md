# Trace fixed-scale fidelity + uniform trace figures — design

**Date:** 2026-07-23
**Status:** approved (brainstormed with Albert; Phase 1 of 2 — the Phase-2
grid-based figure builder is recorded as a separate wish, not designed here)

## Problem

Profiles trace figures (`<fig>__trace__<vid>.png`) do not reliably honour the
publication µm-per-cm scale, and traces from different jobs come out with
different physical box heights, so composing them into one final figure
(PowerPoint, posters, papers) requires rescaling — which also breaks font
parity between panels. Companion figures are overcrowded and use different
fonts/line styles from the standalone traces.

### Evidence (real data, `/mnt/data/ESRF/STO2_overnight_plots/profiles`, run of 2026-07-20)

Measured spine-to-spine boxes at the files' 300 dpi, against line lengths from
the exported CSVs, with the trace scale set to 10 µm/cm:

| job | L (µm) | box (cm) | implied µm/cm |
|---|---|---|---|
| `oblique_full_3_um` traces | 29.67 | 5.15–5.19 wide | **5.72–5.76 (wrong)** |
| `oblique_full_parallel_26_um` traces | 44.94 | 4.50 wide | 9.98–10.00 (right) |

Overview/reference **map** figures measured exact (42.46 × 39.91 cm for a
424.0 × 398.5 µm plane → 9.99 µm/cm both axes). The defect is confined to the
trace path.

### Root cause (reproduced synthetically)

`build_trace_figure` pins the plot-box shape with `ax.set_box_aspect(h/w)` and
*then* runs the iterative `fit_axes_to_box` to reach the physical target box.
`set_box_aspect` makes the box fill whichever figure dimension is limiting, so
the fitter's core assumption — grow figure width by Δ ⇒ box width grows by Δ —
does not hold. The fit oscillates, gives up after 3 iterations, and silently
keeps whatever size it landed on (INFO-level log only). Whether it converges
depends erratically on line length × font scale × label text: with the real
label strings, L=29.67 µm at 10 µm/cm lands at 5.06–5.19 µm/cm depending on
`trace_font_scale`, while other combinations converge exactly.

Even when exact, each trace box is `trace_aspect`-shaped, so different line
lengths give different box *heights* — the uniformity gap.

## Design

### A. Deterministic layout engine (`dfxm/common/plotting.py`)

Replace the iterative fit for the trace/companion path with placement that is
exact by construction:

- `measure_margins(fig, axes) -> margins`: one draw pass at final font sizes;
  measure each axes' decoration extents (y-label + ticks left, x-label + ticks
  bottom, title top, offset text right) in inches.
- `place_axes(fig, placements, margins)`: set the figure size to exactly
  margins + boxes and pin each axes' position in figure coordinates
  (`ax.set_position`). No iteration, no `set_box_aspect` on placed axes.
  Supports several axes on one canvas (companion).
- `fit_axes_to_box` **stays** for the map builders (verified exact on real
  data); its non-convergence log is upgraded INFO → WARNING. Migrating map
  builders to the new engine is out of scope.

### B. Trace figures

- New `PlotStyle` knob **`trace_height_cm`** (default 3.0), shown in the GUI
  style controls next to "Trace scale (µm/cm)", persisted with the other style
  knobs (QSettings), honoured by run and replot alike.
- Fixed-scale mode (`trace_fixed_scale(style)` is set): the plot box is exactly
  `(L / µm-per-cm)` cm wide × `trace_height_cm` tall. `trace_aspect` and
  `trace_width_in` are **ignored in this mode**; they keep governing the
  non-fixed-scale legacy path, which stays byte-identical (including
  `set_box_aspect` and tight-crop there).
- **Uniform margins per invocation:** the run and replot drivers render all
  trace figures of the invocation in two passes — measure all, take the max
  margin per side, place every figure with those shared margins. Every trace
  PNG of a run then has the same height, same fonts (points), and the same
  left/bottom decoration offsets → aligns in rows, columns, or grids with no
  rescaling. Margins depend on the set rendered together (re-rendering a single
  field alone may shift them slightly) — documented in Usage.md.
- **No `bbox_inches="tight"` in fixed-scale mode** — tight-crop would trim the
  reserved uniform margins differently per figure. The canvas is exactly
  margins + box, so there is nothing to crop. The 30-inch safety clamp stays
  and now appends a visible result note when it fires.

### C. Companion restyle (fixed-scale mode)

One canvas, N+1 panels placed deterministically with the Section-A engine:

- **Map panel** at the *map* scale (`scale_um_per_cm`): box = plane extent /
  scale, same as the standalone overview maps. Scale bar, colorbar, axes mode
  per style, as today.
- **Trace panels** below, each `(L / trace-scale)` cm × `trace_height_cm`,
  visually identical to the standalone trace figures: `trace_font_scale` (not
  the map `font_scale`), `trace_linewidth`, and the job's line colour for the
  curve (currently hard-coded C0 / 1.8 lw / map fonts).
- **De-crowding:** per-panel titles only when `style.show_title`; otherwise
  panels carry just their y-label. Fixed real-inch spacing between panels
  replaces gridspec height ratios. Shared x-limits (0..L) across trace panels.
- Fixed scale off → legacy companion layout pinned unchanged.

### D. Guards — no silent drift

After rendering any fixed-scale figure (traces, companion panels, and the same
cheap check on the overview maps), measure the axes box on the canvas; if it
misses the target by more than 0.5 %, append a human-readable warning to
`ProfilesResult.notes` (surfaced in the GUI Results tab) and log a WARNING.
Never an exception.

### E. Tests

- Box exact for short/long lines × font scales 1.0/2.0 × long labels — direct
  regression of the reproduced failure (L=29.67, scale=10).
- All traces of a rendered set share identical margins and pixel heights.
- Companion panel boxes exact at both scales (map panel vs trace panels).
- Legacy (no fixed scale) trace + companion paths byte-stable against the
  pinned regression figures.
- 30-in clamp appends a note; drift guard fires on an artificial miss.
- New style knob covered by the style-controls metadata/persistence tests and
  touched in `tests/gui_smoke.py`.

### F. Docs (same change, per contract)

- `docs/Usage.md`: profiles stage (trace figures + companion behaviour under a
  fixed scale, the new "Trace height (cm)" knob, uniform-margin semantics).
- `docs/Codebase.md`: new plotting helpers, changed profiles functions.

## Out of scope

- Migrating map builders (`render.layer_figure`, strain/slices/`render_single`)
  onto the deterministic engine — they measure exact today.
- The interactive/grid figure builder ("compose elements from different
  widgets into one publication-ready figure") — Phase 2, separate wish +
  brainstorm; builds on the replot catalogs and this engine.
- CSV outputs, non-styled CLI paths: unchanged.
