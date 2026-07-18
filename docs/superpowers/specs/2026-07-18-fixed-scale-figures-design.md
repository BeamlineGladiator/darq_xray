# Fixed-scale map figures (µm-per-cm) + slice selection — design

Date: 2026-07-18 (Part II added 2026-07-19). Status: approved by Albert
(brainstorm session). Part I: fixed-scale figures. Part II: pin-planes fast
re-runs + planes-first replot selection.

## Problem

Styled map figures size the **outer canvas** from the data aspect
(`figure_size`), then constrained layout carves the title, axis labels, and
colorbar out of that fixed canvas. The colorbar column width varies per field
(scientific ×10ⁿ offset text, tick label widths), and title lines vary — so the
equal-aspect data box gets whatever is left over. **Each figure ends up with its
own µm-per-inch scale**, and the scale bar drifts with it: its bar is drawn in
true data µm (`AuxTransformBox(ax.transData)`) so its on-page size tracks the
varying scale, while its label is in points (constant) — the bar-to-label
proportion visibly shifts between figures of the same slice.

Requirement (Albert): within a publication set, the **data** must render at an
identical physical scale and the scale bar must be identical; the **outer**
figure ratio may vary with the amount of text.

## Decisions (from brainstorm)

- **Scale anchor: physical µm-per-cm.** Every figure prints at the same
  magnification — a 400 µm slice draws twice as wide as a 200 µm crop, and
  50 µm is the same number of millimetres everywhere, including ROI crops.
- **Scope: all styled map figures.** Slices + slices replot, the shared
  per-layer figures (visualize/paraview/rocking/mosaicity) + their replots, the
  strain diagnostic, and the profiles per-field **overview** maps (styled map
  figures since f2d7a61 threaded PlotStyle into `_draw_reference_image`). The
  multi-panel profiles companion and the trace/profile line figures are
  untouched (traces have their own aspect knobs; the companion's map panel is
  not fitted — see §4).
- **Opt-in override.** Blank knob = today's behaviour byte-for-byte; a value
  activates fixed-scale mode and `figure_width` is ignored for map figures
  (it still governs trace figures).
- **Approach A: draw–measure–resize.** Keep constrained layout and all current
  styling; iterate the figure size until the axes box hits the target inches.
  (Rejected: analytic fixed-size layout via axes_grid1 dividers — reimplements
  the pinned constrained-layout behaviour, higher risk.)

## Design

### 1. Style knob

`PlotStyle.scale_um_per_cm: float | None = None`.

- `None`/blank → current behaviour everywhere.
- `> 0` → fixed-scale mode for map figures: target data box is
  `ext_u/scale/2.54 × ext_v/scale/2.54` inches.
- Defensive parsing: ≤ 0 or non-numeric (stale persisted styles) → treated as
  `None`, matching the other style-field guards.
- Sanity clamp: target box capped at 30 in per side (scale effectively raised),
  with a logged note — a typo like `0.1` must not request a 47k-pixel render.

### 2. Fitting helper

`fit_axes_to_box(fig, ax, w_in, h_in, tol_in=0.02, max_iter=3)` in
`dfxm/common/plotting.py`:

1. Attach a `FigureCanvasAgg` if the figure has no canvas.
2. Draw; measure the axes box via `ax.get_window_extent(renderer)` / `fig.dpi`.
3. Grow/shrink the figure size **additively** by the miss (decorations are
   constant in inches, so the first correction is nearly exact; the loop is
   insurance) and redraw until both dimensions are within `tol_in`.

Equal-aspect note: the target box has exactly the data aspect (both dimensions
derive from extent ÷ scale), so aspect="equal" does not fight the fit.

### 3. Builder wiring

When `style is not None and style.scale_um_per_cm` is set, the three styled
builders call the helper after fully assembling the figure (title, colorbar,
scale bar):

- shared layer renderer in `dfxm/common/render.py` (visualize/paraview/
  rocking/mosaicity runs, their exports and replots via
  `figures.render_volume_layer`),
- `dfxm/stages/slices.py` `build_slice_figure` (run, replot, export),
- the strain diagnostic figure (`dfxm/stages/strain.py`),
- the profiles per-field overview map (`dfxm/stages/profiles.py`
  `_draw_reference_image` standalone-overview path; the multi-panel companion
  figure is NOT fitted).

Initial figsize stays the current `figure_size(...)` guess (or box + ~1–2 in
headroom in fixed mode) — the helper converges regardless. Replot dialogs and
publication export inherit automatically. Export tight-crop is unaffected:
`bbox_inches="tight"` trims outer whitespace and never rescales axes, so the
printed µm-per-cm survives every format (PNG/PDF/SVG).

### 4. Scale bar

- Fixed-scale mode: bar height in data units =
  `thickness_pt × (2.54/72) × scale_um_per_cm` — true points, computed exactly
  from the known scale (no draw-time measurement). The bar length is already
  true data µm, so the whole assembly is physically identical across figures.
- The fixed geometry is opt-in per call: builders that actually fitted the
  axes pass the scale explicitly (`draw_scale_bar(..., fixed_scale_um_per_cm=…)`);
  the bar never infers it from the style alone. Un-fitted figures that share
  the bar (e.g. the profiles companion's map panel) keep today's geometry even
  when the style knob is set.
- Non-fixed mode: today's geometry (`0.004·thickness_pt·|yr|`) byte-for-byte.
- `scale_bar_length_um` on auto still picks ~15% of each figure's extent — for
  identical bars across *different crops*, set an explicit length (e.g. 50).
  Document this in Usage.md.

### 5. GUI, tests, docs

- GUI: "Scale (µm/cm)" float field (blank = off) next to Figure width in the
  publication-style panel; persisted via QSettings like its neighbours; tooltip
  states it overrides figure width for map figures.
- Tests:
  - helper reaches the target box (within tol) under two decoration loads
    (long two-line title + scientific-offset colorbar vs bare);
  - two slices figures with different colorbar text yield equal axes-box
    inches within tolerance at the same scale;
  - scale-bar height equals the point-exact value in fixed mode;
  - `None` regression: existing suite covers unchanged behaviour.
- Docs: Usage.md publication-style section (knob semantics, identical-bar
  recipe) + Codebase.md (`plotting.py` helper + builder notes), same change.

## Error handling (Part I)

- Invalid knob values degrade to `None` (no exceptions from style parsing).
- Non-convergence after `max_iter` (pathological decorations): keep the last
  size — the miss is bounded and logged, never fatal.
- Empty/degenerate extents (single-pixel crops): skip fitting, fall back to
  current sizing.

---

# Part II — slice selection & fast partial re-runs (added 2026-07-19)

Motivation: a 195-plane sweep × 8 volumes makes (a) full re-runs slow when only
a few planes matter, and (b) the replot dialogs' per-volume tree (~3000 rows)
painful — a plane must be re-found and re-ticked under every volume branch.

## 6. Pin planes… (slices stage form)

Fast recompute of chosen planes without losing the sweep definition.

- **Two fields + a toggle.** The existing `slices_json` (the sweep) is never
  written to by anything. New params: `pinned_slices_json` (multi-line JSON,
  normally machine-written) and `use_pinned` (BOOL, "Run pinned planes only",
  default off). `run()` reads the pinned list iff the toggle is on, and logs
  loudly that a pinned run is active. Toggle off → full sweep, byte-identical
  behaviour.
- **Pin planes… dialog** (button on the slices form): loads an
  `oblique_slices.h5` (pre-filled from the chained/last output), lists slice
  group → planes with the same number/offset filter as §7, and on OK writes
  one-plane-per-entry pinned specs (exact stored geometry — `normal`, `origin`,
  `up`, `half_u/v`, snapped to stored planes) and ticks `use_pinned`.
- **Shared core.** The snap-and-emit logic of `tools/pin_slice.py` moves into
  `dfxm/stages/slices.py` (e.g. `build_pinned_spec(h5_path, slice_name,
  offsets) -> list[dict]`); the CLI tool and the dialog both call it.
- **Clobber guard.** While `use_pinned` is on, the default `output_h5_name`
  becomes `oblique_slices_pinned.h5` (unless user-edited), so a pinned run
  never overwrites the sweep file that profiles reads.
- Both fields persist per experiment via the existing form-state store;
  calibration-style exclusions do not apply.

## 7. Planes-first replot selection (shared widget, slices + generic dialogs)

One selection widget replaces the current per-volume tree in **both** replot
dialogs (slices; strain/mosaicity/rocking generic).

- **Left panel: planes/layers, listed once.** Slices dialog: slice group →
  plane rows (`p118  −3.72 µm`) — offsets are identical across volumes within
  a group, so planes appear once, not per volume (union across volumes keyed
  by `(slice_name, plane_idx)`). Generic dialogs: layer rows (index + z µm
  when available), likewise listed once.
- **Filter box (narrows, never selects).** Comma-separated tokens; an integer
  token filters by plane/layer number, a decimal or signed token filters by
  offset/z within half a sweep step (nearest plane). Filtering only controls
  visibility — clicking selects, so equal plane numbers across slice groups
  stay unambiguous. Clearing restores the full list. A **"check all visible"**
  button makes filter → select one action.
- **Right panel: quantities.** Slices: volume ids (mosa_com_mu, strain,
  raw_mosa_sum, …). Generic: the dialog's quantity groups. Short flat checkbox
  lists.
- **Render = checked planes × checked quantities.** A quantity that lacks a
  selected plane index is skipped with a reason (existing skip reporting),
  never an error.
- **Defaults preserved.** Everything checked on load (a plain Render still
  remakes the whole file). Per-kind clim sections (`ClimGroupSection`), the
  ROI picker, and the timestamped `replots/<stamp>/` output-dir behaviour
  carry over unchanged.

## Part II testing & docs

- Pin: `build_pinned_spec` snaps to stored planes and reproduces sweep
  geometry (golden vs a real sweep h5 fixture); `use_pinned` routing in
  `run()` (on → pinned list + log note + default name suffix; off →
  untouched); `tools/pin_slice.py` delegates to the shared core.
- Replot: filter tokenising (int vs decimal vs signed), visibility narrowing,
  check-all-visible, cartesian render selection, missing-plane skip, defaults
  all-checked; GUI smoke steps extended for both dialogs + the Pin planes…
  dialog.
- Docs: Usage.md (slices stage: pin workflow; replot sections for slices and
  generic stages) + Codebase.md (`slices.py` new params/functions,
  `gui/widgets/` new selection widget, replot dialog changes) in the same
  change.

## Error handling (Part II)

- Pin dialog with an unreadable/empty h5: dialog shows the error inline,
  writes nothing, toggle untouched.
- `use_pinned` on with empty/invalid `pinned_slices_json`:
  `StageUserError` with a hint to open Pin planes… or untick the toggle.
- Filter tokens that match nothing: empty plane list + a small "no match"
  hint label; Render disabled until at least one plane × quantity is checked.
