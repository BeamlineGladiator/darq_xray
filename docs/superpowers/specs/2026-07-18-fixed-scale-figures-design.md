# Fixed-scale map figures (µm-per-cm) — design

Date: 2026-07-18. Status: approved by Albert (brainstorm session).

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
  per-layer figures (visualize/paraview/rocking/mosaicity) + their replots, and
  the strain diagnostic. Trace/profile line figures are untouched (they have
  their own aspect knobs).
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
- the strain diagnostic figure (`dfxm/stages/strain.py`).

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

## Error handling

- Invalid knob values degrade to `None` (no exceptions from style parsing).
- Non-convergence after `max_iter` (pathological decorations): keep the last
  size — the miss is bounded and logged, never fatal.
- Empty/degenerate extents (single-pixel crops): skip fitting, fall back to
  current sizing.
