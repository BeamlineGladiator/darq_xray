# Slices replot + mosa-sum raw source + per-job profiles — design spec

Date: 2026-07-06
Status: approved (brainstorming) → ready for implementation plan

## Motivation

Three related pain points in the oblique-slices → line-profiles workflow, all
about avoiding wasteful recomputation and giving finer control over the raw
intensity products:

1. **Replotting oblique slices is wasteful.** Remaking slice PNGs with different
   *appearance* (colour limits, colormap, fonts, scale bar) currently means
   re-running the whole `slices` stage — alignment + `map_coordinates`
   resampling of every volume across every plane — even though the sampled slice
   arrays are already sitting in `oblique_slices.h5`.
2. **Line profiles are all-or-(one-global-)nothing.** Every field present for a
   slice is profiled and shown, or a single global `volume_ids` list restricts
   *all* jobs the same way. There is no per-line control over which fields get a
   profile.
3. **Raw intensity only comes from rocking scans.** The raw `sum_intensity` /
   `specific_frame` volumes are built exclusively from *rocking* scans
   (`rocking.py`). The *mosaicity* scans — a 2-D χ/μ grid of detector frames,
   one folder per layer — also carry raw intensity that, when summed, is a
   distinct and useful product (a DFXM topograph), but nothing reconstructs it.

## Scope

Five features across `slices`, `profiles`, `rocking`, one new GUI subwidget, and
the shared plotting/alignment cores. All decisions below were settled in the
brainstorming Q&A; this spec records them.

Out of scope: changing plane geometry on replot (that always requires a real
re-slice from the source volumes); wiring the mosa-sum source into `matched.py`;
any change to the darfix / concat / strain stages.

---

## A. Slices — per-direction PNG subfolders (original runs)

**What.** In `slices.run()`, the per-plane PNGs are written into one subfolder
per slice direction instead of a flat directory:

- Before: `{out_dir}/{name}__{volume_id}.png`
  and `{out_dir}/{name}__{volume_id}__p{NNN}_{off}um.png`
- After: `{out_dir}/{name}/{volume_id}.png`
  and `{out_dir}/{name}/{volume_id}__p{NNN}_{off}um.png`

`os.makedirs({out_dir}/{name}, exist_ok=True)` is created lazily the first time a
plane for that slice is saved. `SlicesResult.pngs` keeps **full paths**, so
`_image_first_png` (GUI preview) and `_summarize_slices` are unaffected.

**Why.** Multi-direction runs (e.g. `z_sweep` + `oblique_full`) otherwise pile
dozens of PNGs into one folder. This mirrors the replot output layout (B) so the
two are consistent.

**Touch points.** `dfxm/stages/slices.py` (`run()` PNG path construction only).
Docs: Usage.md (slices output description).

---

## B. Slices — "Replot…" subwidget (appearance-only, works from cold start)

**Goal.** Re-render slice PNGs from an existing `oblique_slices.h5` with
different *appearance* only (no resampling), choosing any subset of
volume/slice/plane, **without having run the slices stage in the current
session**.

### B.1 Qt-free core (in `dfxm/stages/slices.py`)

Two new public functions so all heavy logic stays in the importable, testable
core (honours the "keep `dfxm/` Qt-free" rule):

```python
def replot_catalog(h5_path: str) -> list[ReplotEntry]:
    """List every (volume_id, slice_name, n_planes, offsets_um) in the file."""

def render_replot(
    h5_path: str,
    selections: list[tuple[str, str, list[int] | None]],  # (vid, slice, plane_idxs|None=all)
    style: PlotStyle | None,
    clim: tuple[float | None, float | None] | None,       # (vmin, vmax) override; None entries = stored
    out_dir: str,
    *,
    dpi: int = 150,
) -> list[str]:
    """Rebuild + save the selected planes; return written PNG paths."""
```

- `render_replot` reconstructs the `prep` dict from the stored volume-group
  attrs exactly as `slices.figures()` already does (`cmap`, `title`,
  `cbar_label`, `vmin`, `vmax`, `center_zero` from `kind ∈ _CENTERED_KINDS`,
  `group` from `GROUP_BY_KIND`), applies the clim override when provided, then
  calls the existing `build_slice_figure(...)` and saves via the existing
  `save_slice_png` path.
- Output layout mirrors A: `{out_dir}/{slice_name}/{vid}[__pNNN_{off}um].png`.
- **Refactor for DRY:** factor the per-plane rebuild that `figures()` currently
  inlines into a shared helper (e.g. `_rebuild_plane_figure(h5, vid, sname, k,
  style, prep, kind, clim=None)`) that both `figures()` and `render_replot` call,
  so the reconstruction logic lives in exactly one place.

### B.2 GUI dialog (`gui/widgets/slice_replot.py`)

A `SliceReplotDialog(QDialog)` built lazily on click, mirroring
`LinePickerDialog`:

- **File field** at top: path to an `oblique_slices.h5`, with a Browse button and
  a "Load" action. Default value = the slices form's `{output_dir}/{output_h5_name}`
  (or the last-run `output_h5` if present) — but the dialog **reads from disk**,
  never from `_last_result`, so it works after a program restart.
- **Checkable tree** (`QTreeWidget`): top level = `volume_id`, children =
  `slice_name`, leaves = planes (labelled by offset). A tri-state parent checkbox
  selects "all planes" for a slice. Populated from `replot_catalog`.
- **clim override**: two line edits (vmin / vmax); blank = use the stored clim
  for each volume. Applies to the whole render batch.
- **Output field**: defaults to `{h5_dir}/replots/{timestamp}/`, editable via a
  folder picker. Per-slice subfolders are created inside (layout as A/B.1).
- **Render** button → calls `render_replot(...)` with the style resolved from the
  **current slices param-form values** (passed into the dialog by `StageView`)
  plus the clim override; logs the written paths back to the stage log.

### B.3 Wiring (`gui/stage_view.py`)

- Add a **"Replot…"** `QPushButton` to the slices button row (guarded by
  `if stage_name == "slices"`, exactly like the profiles `Pick line…` guard).
  Always enabled (does not depend on a successful run).
- `_on_replot()` slot: import `SliceReplotDialog` on demand, seed it with the
  form's output path + `self._form.values()` (for the style), open it.

**Touch points.** `dfxm/stages/slices.py` (2 public fns + shared helper),
`gui/widgets/slice_replot.py` (new), `gui/stage_view.py` (button + slot).
Docs: Usage.md (new subwidget section), Codebase.md (new functions + widget).

---

## C. Rocking — mosa-sum as a new parallel raw volume

**Goal.** Reconstruct raw intensity from the *mosaicity* scans (sum of each mosa
folder's detector frames) as a **separate** aligned raw volume, so it can be
sliced/profiled alongside — or instead of — the rocking-sum volume.

### C.1 New params on the `rocking` stage

- **`source_scan`** — `ParamType.ENUM`, choices `("rocking", "mosaicity")`,
  default `"rocking"`, group "Data layout". Selects which folders' frames are
  summed into the volume.
- **`subtract_background`** — `ParamType.BOOL`, default `True`, group "Alignment".
  When true, per-pixel median across the scan's frames is subtracted before
  summing (current rocking behaviour); when false, a plain sum (topograph with
  background). Applies to both sources; default preserves existing rocking output.

### C.2 Behaviour when `source_scan == "mosaicity"`

- The **`mosa_pattern` folders are the layers** (they already provide samy/samz).
  Sorted by samz, each folder's detector frames (at `detector_path`) are read,
  optionally median-background-subtracted (per `subtract_background`), and summed
  — reusing `process_raw_scan` with a `subtract_background` argument threaded
  through.
- No samz-union masking (that logic is rocking-specific: for the mosaicity source
  every matched mosa folder is a layer).
- Alignment is the **same** `dfxm/common/alignment.py` anchored to the mosa
  reference (`ref_samy = mosa_samy[0]`, `ref_samz = mosa_samz[0]`) — for this
  source the reference *is* the first layer, so shifts are self-relative.
- `specific_frame` still applies (one frame of the mosa grid; default central).

### C.3 Output naming (no clobber)

When `source_scan == "mosaicity"` and the output filename/dir are still the
rocking defaults, `run()` substitutes mosa defaults:

- `aligned_h5_name`: `aligned_raw_rocking_volumes.h5` → `aligned_raw_mosa_volumes.h5`
- `output_dir` default folder: `aligned_raw_rocking_volumes/` → `aligned_raw_mosa_volumes/`

If the user set explicit non-default values, those are respected as-is. Product
titles branch on source (e.g. "Mosa-integrated Sum Intensity"); the colormap
`group` stays `"raw"`. `figures()` mirrors the source-aware titles.

**Workflow.** Run rocking once with `source_scan="rocking"` and once with
`source_scan="mosaicity"` to produce both files.

**Touch points.** `dfxm/stages/rocking.py` (`STAGE` params; `process_raw_scan`
gains `subtract_background`; `run()` source branch + filename branch + titles;
`figures()` titles). Docs: Usage.md (rocking source selector + toggle),
Codebase.md (rocking params/functions).

---

## D. Slices + profiles — consume the mosa-sum volume

### D.1 Slices

- New param **`aligned_mosa_file`** (`ParamType.PATH`, `must_exist=True`,
  "Aligned mosa volume", may be blank), placed beside `aligned_rocking_file`.
- New toggles **`include_mosa_sum`** / **`include_mosa_specific`**
  (`ParamType.BOOL`, group "Quantities").
- Two new `_STD_VOLUMES` rows:
  - `("include_mosa_sum", "aligned", "aligned_mosa_file", "sum_intensity", "raw_mosa_sum")`
  - `("include_mosa_specific", "aligned", "aligned_mosa_file", "specific_frame", "raw_mosa_specific")`
- `_standard_volumes` `file_keys` gains the `aligned_mosa_file` entry.
- New kinds **`raw_mosa_sum` / `raw_mosa_specific`**:
  - `prepare_volume` titles dict: e.g.
    `"raw_mosa_sum": ("Mosa-integrated Sum Intensity", "Sum intensity (a.u.)", "")`
    and a frame-indexed title for `raw_mosa_specific` (mirroring `raw_specific`).
  - Add both to `GROUP_BY_KIND` → `"raw"` (in `dfxm/common/plotting.py`) so
    `resolve_cmap` and the per-group colourbar formatting resolve. They use the
    same percentile-range clim branch as the other raw kinds.

Because the h5 group/schema is identical, the replot dialog (B) and profiles (E)
pick these up automatically once they are written to `oblique_slices.h5`.

### D.2 Profiles

No new plumbing needed for D specifically — the new `raw_mosa_sum` field appears
as another volume group in `oblique_slices.h5` and is profiled like any other
field (subject to the per-job selection in E).

**Touch points.** `dfxm/stages/slices.py` (params, `_STD_VOLUMES`,
`_standard_volumes`, titles), `dfxm/common/plotting.py` (`GROUP_BY_KIND`). Docs:
Usage.md (slices quantities), Codebase.md.

---

## E. Profiles — per-job field selection

**Goal.** Choose, per profile line, which fields get profiled/shown — instead of
one global list for all jobs.

### E.1 Job schema

Each job in `jobs_json` gains two optional keys:

- **`fields`** — list of `volume_id`s to profile for this job, in order. Absent /
  null → fall back to the global `volume_ids` restrict (then to "all present").
- **`reference`** — `volume_id` for the top reference image. Absent → fall back
  to global `reference_volume_id`, then the `raw_sum`/first default.

### E.2 Core change

The override is read **inside `_collect`** (from the `job` dict), so both the
`run()` parameter-mode path and the `figures()` catalog rebuild honour it with no
extra threading:

- `restrict = job.get("fields") or global_restrict`
- `ref_pref = job.get("reference") or global_ref_pref`

`_ordered_field_ids` / `_pick_reference_id` already accept these. Fields listed in
a job's `fields` that are not present for that slice are silently dropped (same
as the global restrict today).

### E.3 Pick-line dialog checkboxes

`LinePickerDialog` gains a **field checklist** (one checkbox per `volume_id`
present for the slice, all checked by default). On "Use line" the accepted result
includes the ticked `fields` list, which `stage_view` writes into the job it
appends to `jobs_json`. Manual JSON editing of `fields` remains supported.

**Touch points.** `dfxm/stages/profiles.py` (`_collect` override read; the
`figures()` closure already passes the job dict, so it inherits the change),
`gui/widgets/line_picker.py` (checklist + result), `gui/stage_view.py`
(`_on_pick_line` writes `fields`). Docs: Usage.md (profiles per-job fields +
picker), Codebase.md.

---

## Testing

New / extended tests under `tests/` (synthetic HDF5 fixtures, following the
existing slices/profiles/rocking test style):

- **A** — a slices run writes each direction's PNGs under `{out_dir}/{name}/`;
  `result.pngs` are valid full paths.
- **B** — `replot_catalog` enumerates the volumes/slices/planes of a synthetic
  `oblique_slices.h5`; `render_replot` writes the expected files under per-slice
  subfolders, and a vmin/vmax override changes the saved image's norm (assert via
  the rebuilt figure, not pixels). Headless (no Qt).
- **C** — with a synthetic set of mosa folders, `source_scan="mosaicity"` builds a
  volume of the right shape; `subtract_background=False` differs from `True`; the
  output filename/dir branch to the mosa defaults; explicit names are respected.
- **D** — a slices run with `aligned_mosa_file` + `include_mosa_sum` produces a
  `raw_mosa_sum` group with the raw group's cmap/clim behaviour.
- **E** — a job with `fields=[...]` profiles only those fields (and orders them);
  a job `reference` picks that top image; `figures()` reflects the same subset.

Existing guardrail tests (`test_param_metadata`, registry/summarizer sync) cover
the new params automatically — every new `Param` needs `help`, advanced params a
`group`, input paths `must_exist=True`.

## Documentation (same change, per the docs contract)

- **Usage.md** — slices: per-direction subfolders + the "Replot…" subwidget;
  rocking: `source_scan` + `subtract_background`; slices: mosa file + toggles;
  profiles: per-job `fields`/`reference` + picker checkboxes. Update the stage
  reference sections; no pipeline-diagram change (no new stage).
- **Codebase.md** — new public functions (`replot_catalog`, `render_replot`,
  shared rebuild helper), the new widget module, the new params, and
  `GROUP_BY_KIND` additions.

## Rollout / sequencing (for the implementation plan)

Independent-ish slices that can be ordered as:

1. A (subfolders) — tiny, unblocks the shared output layout.
2. C (rocking mosa source) — self-contained; produces the new file.
3. D (slices consumes mosa file) — depends on the schema from C existing (but not
   on C's code; schema is identical to rocking's).
4. B (replot core + dialog) — depends on A's layout; benefits from D's new kinds
   but not blocked by them.
5. E (profiles per-job fields + picker) — independent of B/C/D.

Each slice ships its matching Usage.md + Codebase.md edits in the same commit.
