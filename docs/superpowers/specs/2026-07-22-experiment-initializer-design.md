# Experiment initializer — design

**Date:** 2026-07-22
**Status:** approved (brainstorm session, section-by-section)
**Realizes:** the remaining half of the experiment-initializer wish (the darfix-ROI
half was built by the ROI unification project, merged 2026-07-22).

## Goal

Bootstrap and enrich an experiment preset **from the data itself** instead of
hand-filling the Edit dialog: read pixel sizes and the ccmth reference from a raw
scan's positioners, detect the layer folder patterns and entry suffix from disk,
and (once darfix has run) pull the darfix-ROI size and a refined ccmth reference
out of `maps.h5`. End the flow with an offer to persist to a preset YAML, so
derived calibration cannot silently evaporate at restart.

The class of error this must prevent: calibration typed wrong or never saved
(pixel sizes recomputed every session), and ROI numbers hand-copied between
frames (the 2026-07-18 STO2 misregistration incident).

## Decisions made in the brainstorm

- **Auto-derive all four:** pixel sizes from positioners; folder patterns +
  entry suffix; ccmth reference; darfix-ROI size from `maps.h5`.
- **Re-runnable enrichment** (not a one-shot wizard): one "Initialize from
  data…" flow fills whatever is currently derivable; run it again after darfix
  and it adds what now exists. It never overwrites a user-set field without
  asking.
- **Entry point:** a button inside the existing Edit-experiment dialog. A
  brand-new experiment is just Edit on a blank experiment; no separate wizard.
- **Apply UX:** a review table (field | current | detected | note | apply
  checkbox), one OK applies checked rows into the form.
- **Persistence scope:** only the flow-end save prompt. Remember-last-preset
  and warn-on-close-if-unsaved were explicitly declined.

## Feasibility facts (verified on STO2 data, 2026-07-22)

- Raw scan positioners contain all five pixel-size motors (`mainx`, `obx`,
  `ffsel`, `ffz`, `lenssel`) **and** `ccmth`.
- The strain-family `maps.h5` ccmth COM map's median is 7.1442 vs the
  calibrated `ccmth_ref_deg` 7.144 — a good suggestion source. Mosa-family
  maps have only chi/mu, so ccmth enrichment **requires a strain-family**
  `maps.h5`.
- darfix records **no ROI metadata** — no attrs in `maps.h5`, no sidecar file.
  The map shape (1266×1832) recovers the crop **size** only; the **origin**
  (105, 230) must still be typed from the darfix widget.
- A raw layer folder holds both the scan (`<folder>.h5`) and `*_concat.h5`;
  scan-file selection must exclude concat output.

## Architecture

Chosen approach: **Qt-free detection core + thin GUI review dialog.**
(Rejected: GUI-only logic — untestable, violates the Qt-free-core rule; a
pipeline stage — wrong shape, it edits the experiment rather than producing
data products.)

### §1 Detection core — `dfxm/config/detect.py`

New Qt-free module beside `models.py` (schema-coupled, not stage code).

```python
@dataclass(frozen=True)
class Detection:
    field: str          # Experiment field name
    value: Any | None   # coerced detected value; None = nothing to apply
    note: str           # human-readable source, e.g. "M=20.4, 2× objective"
    error: str | None   # failure/skip reason; None = success
    # value None + error None = info-only row (e.g. "darfix ROI ✓ consistent")
```

Per-detector functions plus an orchestrator
`detect_experiment(current: Experiment) -> list[Detection]` running in
dependency order:

1. **Patterns** (`detect_patterns(raw_root)`): list subfolders, strip a
   trailing `__<digits>` to group families; a family whose stem contains
   `mosa` / `rocking` → `mosa_pattern` / `rocking_pattern`; the largest
   remaining family → `folder_pattern`. Each value is `<stem>__*`.
2. **Entry suffix** (`detect_entry_suffix(scan_h5)`): read entry names
   (`1.1`, `2.1`, …) from the chosen scan file; report the majority suffix,
   noting mixed suffixes when present.
3. **Pixel sizes**: call the existing `compute_pixel_size` (untouched) on the
   chosen scan file; note carries magnification / objective / 2θ.
4. **ccmth reference** (`detect_ccmth_ref(...)`): preferred source = median
   (`nanmedian`) of the ccmth COM map in a strain-family `maps.h5` under
   `processed_root`; fallback = the raw `ccmth` positioner value, with the
   note "single motor snapshot — confirm against beamline alignment".
5. **Darfix ROI size** (`detect_darfix_roi_size(maps_h5)`): map shape →
   `w,h`. Returns a partial value; origin handling is GUI-side (§2).

**Scan-file selection rule:** within the first folder-pattern family folder,
prefer `<folder>.h5` exactly; fallback = first `*.h5` not ending in
`_concat.h5`; none → skip with reason.

**maps.h5 selection rule:** first strain-family folder (matching the current
`folder_pattern` if set, else the largest family — the note says which) that
actually contains `maps_filename`.

**Skip/error semantics:** detectors never raise for "not there yet" — they
return a `Detection` with `error` set (skip-with-reason, same reporting style
as the stages). `StageUserError` from `compute_pixel_size` is caught and
surfaced as the row's error text (message + hint). One bad file never blocks
the other detections.

**CLI:** `python3 -m dfxm.config.detect <raw_root> [--processed-root DIR]`
prints the detection table headless.

### §2 GUI flow — `ExperimentDialog` + `gui/widgets/detect_review.py`

- **Button** "Initialize from data…" in the existing Edit-dialog button row
  (beside "Compute pixel size…" / "Pick analysis ROI…"). It reads the
  **current form values**, not the saved experiment, so the first-run flow is:
  New → set `raw_root` (+ `processed_root` if it exists) → click Initialize.
  Blank/missing `raw_root` → message box (same precondition style as the
  pixel-size button).
- **Run:** synchronous under a wait cursor. No thread, no progress plumbing
  (positioner reads are milliseconds; the single COM-map read is ~18 MB).
- **Review dialog** (small QDialog, table *Field | Current | Detected | Note |
  Apply*):
  - Apply pre-**checked** when the current value is blank or equals the schema
    default; pre-**unchecked** when it would overwrite a user-set value (row
    marked "differs").
  - Skipped/errored detectors appear as greyed, non-checkable rows with the
    reason — a first pass shows what a post-darfix re-run will add.
  - **Darfix-ROI special row:** Detected cell is editable, pre-filled
    `?,?,<w>,<h>`; its checkbox stays disabled until the origin digits replace
    `?,?`. If `darfix_roi` is already filled the row flips to validation:
    size matches the map shape → "✓ consistent" (nothing to apply); mismatch →
    warning offering the corrected `w,h` with the existing origin kept.
- **Apply** writes checked rows through the normal form setters, so the
  existing dirty-tracking and ⚠ deviation-marker machinery works unchanged.
- **Save prompt:** if any detection was applied and the dialog is closed with
  OK (in-memory only), one prompt offers the existing Save-as… path, or Skip.
  No prompt when nothing was applied.

### §3 Testing and docs

**Tests** (synthetic tmp-dir fixtures, following the existing pixel-size test
pattern):

- folder families with `__N` suffixes → pattern/classification cases,
  including no-family and multiple-family disambiguation;
- a small HDF5 with the five motors + `ccmth` + entries → suffix, pixel-size
  and ccmth-fallback detectors, scan-file selection (concat exclusion);
- a stub strain-family `maps.h5` (known COM median, known shape) → ccmth
  preferred source and ROI-size detector;
- orchestrator skip/error containment (one bad file, others still detect);
- Qt tests for the review dialog: pre-check rules, differs marking, ROI-row
  checkbox gating, apply-through-form-setters;
- one new `tests/gui_smoke.py` step: button → review → apply.

**Docs (same change, per contract):** `docs/Usage.md` — new "Initializing an
experiment from data" subsection (including the two-pass before/after-darfix
story); `docs/Codebase.md` — `dfxm/config/detect.py` + the review widget.

## Out of scope

- Remember-last-selected-preset and warn-on-close-if-unsaved (declined).
- Any change to `compute_pixel_size` or `dfxm/common/roi.py` — consumed as-is.
- paraview `mosa/strain_darfix_origin_xy` (standing decision: untouched).
- Persistent oblique-plane marking and other unrelated wishes.
