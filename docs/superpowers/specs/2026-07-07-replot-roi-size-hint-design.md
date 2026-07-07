# Replot ROI size hint — Design

- **Date:** 2026-07-07
- **Status:** Approved design (pre-implementation)
- **Scope:** Show each replot group's stored pixel size in the replot dialogs so a
  user filling the ROI-crop boxes knows the bounds they're cropping within.
- **Follows:** the just-merged replot-map-stages feature (master HEAD `f8939f1`).

## Motivation

The replot dialogs (`ReplotDialog`, `SliceReplotDialog`) take a pixel-index ROI crop
`(r0, r1, c0, c1)` — rows(Y) then cols(X) — but the boxes are blank fields, so the user
types bounds blind, with no idea whether a layer is 120×256 or 512×512. Show the stored
layer size per group so the ROI boxes have a visible reference.

The size shown is the **stored** layer shape — which for strain/rocking is the run's
already-ROI-cropped size (exactly the maximum a replot crop can reach), and for
mosaicity/slices the full stored layer/plane. So it is precisely the crop's upper bound.

## Design

**Size is per-group, not one global number.** mosaicity `χ`/`μ` datasets and (especially)
slices — where different volumes/planes have different u/v grids from per-job
`half_u`/`du` — can each differ in shape. Strain/rocking layers within a stage are
uniform. So the size is displayed on each group/slice node in the tree.

### Core (`dfxm/`)

- `dfxm/common/figures.py::ReplotGroup` gains `shape: tuple[int, int] | None = None`
  (the layer `(Y, X)` = `(rows, cols)`). Backward-compatible default.
- `mosaicity.replot_catalog` / `rocking.replot_catalog`: set `shape = tuple(obj.shape[1:])`
  from the 3-D dataset already in hand.
- `strain.replot_catalog`: set `shape = tuple(f["strain"].shape[1:])`.
- `slices.ReplotEntry` gains `shape: tuple[int, int] | None = None`; `slices.replot_catalog`
  sets it from `sg["slices"].shape[1:]` (= `(nv, nu)`).

### GUI (`gui/`)

- `ReplotDialog._reload`: when building each group node, append the size to the label —
  e.g. `f"{grp.label}   ·   {Y}×{X} px (Y×X)"` when `grp.shape` is present.
- `SliceReplotDialog._reload`: append the size to each **slice** node (the plane owner),
  e.g. `f"{sname}   ·   {nv}×{nu} px (Y×X)"`.
- Clarify the ROI caption in both dialogs: `ROI crop — rows r0:r1 (Y, 0–H) · cols c0:c1
  (X, 0–W); blank = full`.

Leaf (layer/plane) nodes are NOT annotated — every layer in a group shares the group's
`(Y, X)`, so the group-level hint suffices (YAGNI).

## Non-goals

- No dynamic/selection-driven size label (per-group-in-tree is always-visible and needs
  no signal wiring).
- No change to ROI semantics, clamping, or the all-four-or-none contract.
- No µm-size display (the crop is pixel-index; µm axes already render on the figure).

## Testing

- Core: `replot_catalog` for each of strain/mosaicity/rocking/slices returns groups whose
  `shape` matches the fixture's `(Y, X)` (`(nv, nu)` for slices).
- GUI: `ReplotDialog` populated from a catalog with `shape` shows the `Y×X` text in the
  group node; `SliceReplotDialog` shows it on the slice node. (Headless offscreen Qt.)

## Docs (same change)

- `docs/Usage.md`: note the size hint in the Replot section.
- `docs/Codebase.md`: `ReplotGroup.shape` / `ReplotEntry.shape` fields + the dialog labels.

## Execution

Small, single subsystem (the replot code) → inline execution with one end review, not SDD.
