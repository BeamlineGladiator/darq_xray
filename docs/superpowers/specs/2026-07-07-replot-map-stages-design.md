# Extend Replot… to strain, mosaicity, and rocking — Design

- **Date:** 2026-07-07
- **Status:** Approved design (pre-plan)
- **Scope:** Add the cold, clim-override, layer-subset **Replot…** capability
  (today slices-only) to the **strain**, **mosaicity**, and **rocking** stages,
  and add a **pixel-index ROI crop** override to replot on **all four** stages
  (the three above *and* the existing slices dialog).

## Motivation

The slices stage ships a **Replot…** button that re-renders selected planes
straight from `oblique_slices.h5` — cold (no prior run needed, works in a fresh
session), with a colour-limit override and a per-plane subset selection. This is
strictly more than the **Export…** path, which re-renders *all* figures from the
last run's in-memory result with the current publication style but offers no
clim override, no subset, and requires a fresh run to have populated the result.

Users want the same "tweak clim and re-render a subset without re-running the
slow stage" workflow on the other map-producing stages. This design determines
**where that is cleanly valid** and how to build it.

## Validity audit (why exactly these three)

Replot is valid only where a stage **persists its own figure-level array data**
on disk, so re-rendering is a pure re-plot (read stored layer → draw). Audit:

| Stage | Persists its figures' data? | Replot valid? |
|---|---|---|
| slices | `oblique_slices.h5` (2-D slice arrays) | already shipped |
| **strain** | strain volume h5 (`strain[i]` per layer) | **yes — clean** |
| **mosaicity** | stacked h5, per-layer via `volume_layer_specs` | **yes — clean** |
| **rocking** | aligned volumes h5, per-layer via `volume_layer_specs` | **yes — clean** |
| visualize | nothing — re-loads source volume + **re-aligns** from `raw_root` | no — would silently recompute; fragile to moved files |
| matched | nothing — **re-reads raw detector `.h5`** frames | no — depends on raw files still present |
| profiles | data in slices' h5, but figures are **line plots** | n/a — clim/subset don't apply (has *Pick line…*) |
| concat, paraview | produce no figures | n/a |

Only strain, mosaicity, and rocking store their own re-renderable volume and
render maps where a clim override is meaningful. This design covers exactly those
three. visualize/matched are explicitly out of scope (they re-derive from
upstream files and cannot offer a faithful cold re-plot).

## Scope decisions (locked during brainstorming)

1. **Stages:** strain + mosaicity + rocking (the clean tier). Not visualize/matched.
2. **Figure kinds:** **map layers only** — the clim-relevant figures, exact parity
   with the slices Replot dialog. Per-layer histograms (strain, mosaicity) and the
   strain detrend-diagnostic stay on the Export path; they have no clim knob and
   Export already re-renders them with the current style.
3. **ROI crop override:** replot gains a **pixel-index ROI crop** (row/col ranges)
   alongside the clim override, on all four stages including the existing slices
   dialog. It is a *reframe/zoom of the stored map* (a re-plot), **not** a
   re-analysis — see "ROI crop semantics" below.

## Current architecture (what we mirror)

The slices replot is three cooperating pieces (all figure work in the Qt-free core):

- `slices.replot_catalog(h5_path) -> list[ReplotEntry]` — enumerates
  `(volume_id, slice_name, n_planes, offsets_um)` by reading the h5 structure. Cold;
  no run result needed.
- `slices._rebuild_plane_figure(h5, vid, sname, k, style, clim=None)` — rebuilds one
  figure from stored group attrs (cmap, title, cbar, vmin/vmax, kind) + the stored
  2-D array, applying an optional clim override. **Shared** by `figures()` (export)
  and `render_replot`.
- `slices.render_replot(h5, selections, style, clim, out_dir)` — writes the selected
  figures as PNGs under `{out_dir}/{slice_name}/`, returns written paths.
- GUI `SliceReplotDialog` (3-level checkable tree volume→slice→plane + clim boxes +
  output dir), opened by `StageView._on_replot`, which is gated on
  `stage_name == "slices"`.

Mosaicity and rocking already build their map figures through one shared helper,
`dfxm/common/figures.py::volume_layer_specs(...)`, whose per-layer `build(style)`
reads one layer from the h5 and calls `render.layer_figure`. Strain builds maps
with its own zero-centred `build_strain_map` (RdBu_r, symmetric limits).

## Design

### Core (`dfxm/`)

Give each of the three stages the same public pair slices already exposes:

- `replot_catalog(h5_path) -> list[ReplotEntry]`
- `render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]`

where `roi` is an optional `(r0, r1, c0, c1)` pixel-index crop applied to each
stored 2-D layer/plane before rendering (`None` = full stored extent). The crop
is array slicing plus a recomputed axis extent — no recompute of the underlying
analysis. **slices' existing `render_replot` gains the same `roi` parameter**, and
`_rebuild_plane_figure` gains the crop, so the slices dialog gets ROI too (the
one behaviour change to the shipped slices feature — additive, parallel to its
existing clim override).

Each `render_replot` reuses a per-figure rebuild helper factored out of the
stage's existing `figures()` build-closures (the same refactor slices already
made with `_rebuild_plane_figure`); the helper is where the crop + clim override
are applied in one place. Concretely:

- **mosaicity + rocking** share a generic helper in `dfxm/common/figures.py`
  (e.g. `volume_replot_catalog` / `render_volume_replot`) built on the existing
  `render.layer_figure`, since both already render through `volume_layer_specs`.
  Each stage supplies its dataset-key list + display mapping (`_KEY_DISPLAY`/
  `_KEY_STEM` for mosaicity; `_PRODUCT_DATASET`/`_PRODUCT_CBAR` for rocking) and a
  thin `replot_catalog`/`render_replot` wrapper.
- **strain** gets its own `replot_catalog`/`render_replot` using `build_strain_map`
  (its renderer differs — zero-centred RdBu_r), reading `strain[i]` from the
  volume h5 and layer names from `f.attrs["source_folders"]`.

The catalog is a generic 2-level shape: a list of groups, each with an opaque
`group_key`, a display `label`, and a list of per-item labels (layer indices /
z-offsets). `selections` is a list of `(group_key, item_idxs | None)` where `None`
means all items in the group.

### GUI (`gui/`)

- New generic **2-level** `ReplotDialog(catalog, render_fn, style, out_default)` in
  `gui/widgets/` — a group→item checkable tree (tristate) + clim boxes + **ROI-crop
  boxes** (`r0, r1, c0, c1`, blank = full) + output-dir field + Render/Close,
  reusing the interaction patterns of `SliceReplotDialog`.
- `SliceReplotDialog`: add the same **ROI-crop boxes** (a small additive change,
  parallel to its existing clim boxes) and pass `roi` through to
  `slices.render_replot`. Otherwise keep it as-is (do not migrate slices onto the
  generic dialog — its 3-level tree keeps working). Both dialogs share the same
  clim/ROI-parse helpers where clean.
- `StageView._on_replot`: generalise from slices-only to a per-stage dispatch that
  resolves the stage's h5 path from the form values and hands the correct
  `catalog` + `render_fn` (+ current `params`, style, and a timestamped
  `replots/<ts>/` output dir) to `ReplotDialog`.
- `StageView.__init__`: add the `Replot…` button for `strain`, `mosaicity`,
  `rocking` (currently gated on `stage_name == "slices"`).

### Axis fidelity (params, not h5-format change)

`render_replot` accepts the current form `params` for exact pixel scale / ROI /
titles — the same contract as `figures(result, params)` — falling back to the
calibrated beamline defaults (0.152 / 0.385 µm) when a value is absent. In the GUI
the form always has these, so replot axes are exact; a CLI cold-replot with no
params degrades to the calibrated defaults (identical to how `figures()` already
behaves). Strain additionally has `scale_x_um`/`scale_y_um` in its h5 attrs as a
fallback. **No stage's h5 output format changes.**

### ROI crop semantics

The ROI override is a **pixel-index crop of the stored 2-D layer/plane** — `arr[r0:r1,
c0:c1]` with the axis extent recomputed from the crop (and, for strain, the
`build_strain_map` extent ROI shifted to match). It is a re-plot, never a
re-analysis, so it can only *tighten* the frame within stored data:

| Stage | Stored extent | ROI crop can reach |
|---|---|---|
| slices | full oblique plane (u/v µm) | anywhere in the plane — fully flexible |
| mosaicity | full map layer (no run ROI today) | anywhere in the layer — fully flexible |
| strain | already cropped to the run's `roi` | sub-region of the run ROI only |
| rocking | already cropped to the run's `roi_x`/`roi_y` | sub-region of the run ROI only |

For strain, a sub-crop is faithful because detrend is computed on the full map
(ROI-independent) — only the frame changes, not the values. For a *wider* region
than the run stored, the user re-runs the stage with a new ROI (that is what the
stage's Run + ROI field already does; replot cannot manufacture missing data).

Like clim, a single ROI applies to the whole render batch. Groups can differ in
shape (e.g. mosaicity COM vs FWHM, or different slice volumes), so the crop is
**clamped per-array** to valid bounds; a user wanting an exact crop selects one
group at a time. An out-of-range crop that leaves an empty array is skipped with
a status note rather than raising.

### Output layout

Mirror slices: PNGs under `{output_dir}/replots/<timestamp>/…`, grouped by the
stage's natural sub-directory (dataset key / product / layer), returning written
paths for the Log line.

## Data flow

```
click Replot…  →  _on_replot resolves stage h5 + params
                →  <stage>.replot_catalog(h5)      # cold enumerate groups/items
                →  ReplotDialog(catalog, render_fn) # user picks subset + clim + ROI
                →  <stage>.render_replot(h5, selections, style, clim, out, roi, params)
                →  per item: read layer → crop to ROI → rebuild figure (clim) → savefig
                →  written PNG paths → Log
```

## Selection unit per stage

| Stage | h5 source | Group (tree top) | Item (leaf) | clim renderer |
|---|---|---|---|---|
| strain | strain volume h5 | `strain` | layer 0…N (source-folder name) | `build_strain_map` (RdBu_r, symmetric) |
| mosaicity | stacked h5 | dataset key (`/chi/Center of mass`, `/chi/FWHM`, …) | layer z | `render.layer_figure` (magma) |
| rocking | aligned volumes h5 | product (`sum`, `specific_frame`, …) | layer z (z-offset µm) | `render.layer_figure` (gray) |

## Caveats (documented, not blockers)

1. **One clim per render batch** (as in slices): a single vmin/vmax applies to all
   selected items. Because mosaicity COM vs FWHM datasets have very different
   scales, a user overriding clim there should select one dataset group at a time.
   Note this in the dialog help text and `Usage.md`.
2. **rocking titles** depend on `source_scan` / `specific_frame_idx`. With params
   present (the GUI case) titles are exact; a bare CLI cold-replot falls back to
   generic titles.
3. **strain ROI axis origin**: ROI is not persisted in the strain h5, so a bare
   cold-replot without params starts axes at 0 rather than the ROI origin. GUI
   replot (params present) is exact. Acceptable; note in `Usage.md`.
4. **ROI crop is bounded by stored data** (see "ROI crop semantics"): strain/rocking
   can only sub-crop within the run ROI; a wider frame needs a re-run. The dialog
   help text and `Usage.md` state this so users don't expect replot to expand a
   crop.

## Testing

- Per-stage core round-trip tests (synthetic h5 fixtures): `replot_catalog`
  enumerates the expected groups/items; `render_replot` writes the expected PNGs
  for a subset selection and honours a clim override (assert the override reaches
  the rendered figure, e.g. via the rebuild helper's returned `Figure` image clim
  in a unit call).
- ROI-crop tests (all four stages incl. slices): `render_replot` with an `roi`
  produces a figure whose data extent matches the crop; an out-of-bounds ROI is
  clamped per-array; an empty crop is skipped, not raised.
- A `stage_view` test asserting the `Replot…` button now exists for the three new
  stages and still for slices, and not for the others.
- GUI smoke coverage: opening `ReplotDialog` on a synthetic h5 populates the tree
  and Render writes files (headless-safe, matching existing dialog smoke checks).

## Docs (same-change contract)

- `docs/Usage.md` — extend the Replot section to strain/mosaicity/rocking; document
  the new ROI-crop boxes (all four stages, incl. slices) and the crop-not-recompute /
  bounded-by-stored-data behaviour; note the one-clim-per-batch and cold-vs-GUI axis
  caveats.
- `docs/Codebase.md` — add the new core functions (`replot_catalog`/`render_replot`
  per stage, now taking `roi` + the shared `dfxm/common/figures.py` helpers) and the
  generic `ReplotDialog`; note the `roi` addition to slices' `render_replot` /
  `_rebuild_plane_figure`.

## Non-goals

- visualize / matched replot (they re-derive from upstream files; no faithful cold
  re-plot). Explicitly out of scope.
- Histogram / detrend-diagnostic replot (no clim knob; Export covers them).
- Migrating slices onto the new generic dialog (the slices dialog keeps its
  3-level tree; it only *gains* the ROI-crop boxes + `roi` pass-through).
- Interactive rubber-band ROI selection (typed pixel boxes only for now; a
  drag-to-crop preview is a possible future enhancement).
- ROI as a re-analysis / expanding a crop beyond stored data (that is a stage
  re-run, not replot).
- Any change to stage h5 output formats.
