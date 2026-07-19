# Profiles Replot — design

**Date:** 2026-07-20 · **Status:** approved (design), spec pending user review
**Requested by:** Albert — "replot should also be allowed in the line profile
part, as I want to replot it with different edges of the colourmap. Maybe make
it very similar to what oblique slices allows me to do."

## Goal

A **Replot…** button on the profiles (line profiles) stage that re-renders
profile figures cold from disk with per-quantity colour-limit overrides on the
map panels — the same workflow the slices stage already offers, adapted to the
job-shaped outputs of profiles.

Scope decisions (confirmed):

- **Figures rendered:** everything a run produces except CSVs — per-field
  overview maps (line drawn on the plane), the stacked companion (its map
  panel honours the clim override), and the per-field trace figures (clim
  cannot affect them; they render so a replot folder is a complete set).
- **Jobs source:** the profiles form's current **Slices file** + **Jobs
  (JSON)** contents (same sourcing as Pick line…). No prior run needed.
- **Clim controls:** per quantity via the shared `ClimGroupSection`, keyed by
  field id with colormap-group fallback — identical semantics to the slices
  replot dialog. Blank fields keep the stored limits.

## Core changes (`dfxm/stages/profiles.py`, Qt-free)

1. **Extract `_render_job(...)`** from the parameter-mode body of `run()`
   (collect → companion → overviews → traces → CSV). `run()` keeps identical
   behaviour by calling the helper with `clim=None, save_csv=<param>`. One
   code path for run and replot — they cannot drift.
2. **Clim threading.** `_render_job` accepts
   `clim: dict[str, tuple[float | None, float | None]] | None`. Resolution per
   field: exact field id first (`strain`, `mosa_com_chi`, `raw_sum`, …), then
   the field's colormap group (`mosa_com` / `mosa_fwhm` / `strain` / `raw`)
   via the existing kind→group mapping, else `None` (keep stored
   `vmin`/`vmax` attrs). A half-open pair keeps the stored value on the blank
   side (same as `figures._apply_clim`). The override is applied to the
   resolved per-field `attrs["vmin"]/["vmax"]` in exactly one place, so the
   companion map panel and the per-field overviews both honour it and traces
   are untouched.
3. **New public `render_replot(h5_path, jobs, style, clim, out_dir, *,
   dpi=150) -> ProfilesResult`.** Runs the given (already filtered) jobs with
   CSVs off, writing into `out_dir`. Reuses `resolve_job_slice_name` (pinned
   files replot transparently), `_unique_name` stem dedup, and the existing
   skip/notes reporting. Raises `StageUserError` for an unreadable file /
   invalid jobs, mirroring `run()`.

## GUI changes

1. **New `gui/widgets/profiles_replot.py`** modeled on the slices replot
   dialog (`slice_replot.py`):
   - Left: checkbox tree **job → fields present for that job's slice**
     (fields listed from the h5 for the job's — possibly pin-resolved — slice
     group). Initial check state: the job's own `"fields"` list when it has
     one, else all fields checked (session-state convention). On Render the
     checked set becomes the job's `"fields"` override; fully unchecked jobs
     are dropped.
   - Right: one `ClimGroupSection` per quantity present (χ/μ CoM split, FWHM,
     strain, raw…, discovery identical to the slices dialog), **Output dir**
     defaulting to a timestamped `replots/<stamp>/` beside the loaded h5
     (follows Browse/Load unless the user edited it), and a **DPI** spin.
   - **Render** runs in-process (profiles rendering is light; no volume
     resampling), then shows the written-file count and any skip / pin-
     substitution notes inline.
2. **Replot… button** on the profiles stage view, wired the same way as the
   slices one (always enabled; opens cold). Pre-fills from the form's current
   Slices file + Jobs (JSON).
3. Style: the dialog renders with the current publication style snapshot,
   like the other replot dialogs.

## Error handling

- Unreadable h5 / invalid JSON → `StageUserError` message + hint shown in the
  dialog banner; dialog stays open.
- Job slice missing (no plain or pinned match) → rendered set skips it and
  the note is listed, same wording as the stage run.
- Empty selection → the Render button is disabled while nothing is checked.

## Testing

- **Core (`tests/test_stage_profiles.py`):** `render_replot` writes
  companion + overviews + traces and **no CSVs**; a per-quantity clim changes
  the overview/companion map norm (assert via the figure's `im.get_clim()` or
  rendered output), field-id key beats group fallback, blank keeps stored
  attrs; jobs against a pinned file replot with the substitution noted.
- **GUI (`tests/gui_smoke.py`):** new step — open the dialog on a synthetic
  h5 + jobs, tree populated all-checked, set a clim, render to a temp dir,
  assert files written (mirrors the slices replot smoke step).
- Existing run-path tests guard the `_render_job` extraction (behaviour must
  be identical for `clim=None`).

## Docs (same change as code)

- `docs/Usage.md`: "Replotting line profiles" subsection under the profiles
  stage (mirrors the slices replot section; notes clim-per-quantity, traces
  unaffected by clim, CSVs never rewritten, pinned interop).
- `docs/Codebase.md`: `render_replot` + `_render_job` under `dfxm/stages`
  profiles entry; new widget under `gui/widgets`.

## Out of scope

- Re-picking the line inside the replot dialog (use Pick line… on the form).
- ROI cropping in this dialog (the map panels follow the job's plane extent;
  slices replot already covers plane cropping upstream).
- CSV regeneration (replots are appearance-only, per the replot convention).
