# Design: slice marks + three UX features

**Date:** 2026-07-26
**Status:** approved (Albert, 2026-07-26 — all sections)
**Scope:** four requested features in one branch: (1) scroll-wheel guard on
param forms, (2) line-picker background switch, (3) persistent oblique-slice
marks with a visual marking dialog, (4) marks → profiles-jobs bridge, plus
(5) a tooltip precision sweep. Realizes the `scroll-wheel-spinbox-wish`,
`line-picker-reference-switch-wish`, and `oblique-slice-selection-wish`
memories and the `tooltip-precision-pass` request.

## Decisions taken during brainstorm

- Slice-selection wish scope: **both** persistent marks in the h5 **and** a
  bridge that turns marks into profiles jobs (Pin planes… and Pick line…
  already cover pinned re-runs and single-job line injection).
- Marking UI: **visual scroller dialog** (image-based, like Pick line…), not a
  checkbox tree.
- Bridge flow: **guided line picking** — one Pick-line dialog per checked
  mark; complete jobs only, no stubs.
- Line-picker dropdown choice **is written** into the job's per-job
  `"reference"` on accept.
- Tooltip sweep: **full sweep, Claude's judgment**, with the ROI coordinate
  frames and mosaicity `subtract_background` as named priorities; Albert
  reviews the diff.

## 1. Scroll-wheel guard

**Problem:** `QSpinBox`/`QDoubleSpinBox`/`QComboBox` inside the stage-form
scroll areas grab wheel events while scrolling the page, silently changing
values — and form persistence now saves those stray edits.

**Design:** new module `gui/widgets/wheel_guard.py`:

- `install_wheel_guard(widget)` — sets `Qt.FocusPolicy.StrongFocus` and
  installs a module-level `QObject` event filter that, on `QEvent.Type.Wheel`,
  ignores the event and returns `True` (consumed → propagates to the scroll
  area) unless the widget has focus. Focused widgets behave exactly as today.
- `ParamForm` calls it on every `QSpinBox`, `QDoubleSpinBox`, and `QComboBox`
  it builds (`_enum_editor`, `_int_editor`, `_float_editor`).

Other widgets (clim sections, replot dialogs) may adopt the helper later; this
change wires only `ParamForm` (the complaint's site). `QPlainTextEdit`/
`QLineEdit` are untouched (no wheel-edit behaviour).

**Tests:** offscreen Qt test — build a `ParamForm` with an INT param; send a
`QWheelEvent` to the unfocused spin box → value unchanged; give it focus and
send again → value changes.

## 2. Line-picker background switch

**Problem:** `LinePickerDialog` shows exactly one background image (the
resolved reference field); with ≥2 field groups Albert wants to draw the line
while looking at a different one.

**Design:**

- A `Background:` `QComboBox` in the dialog listing `self._present`
  (the groups that contain this slice), initialized to the resolved
  `self._ref_id`.
- On change: re-resolve `self._sg`, `self._attrs` (and axes) for the chosen
  group, keep the current plane index and picked points (all groups share the
  (u, v) grid — `geom_tol_um` guards registration at profile time), redraw.
  Window title updates to the chosen group.
- On accept, `result` grows a 5th element: the chosen group id.
  `_on_pick_line` passes it to `inject_line_into_jobs`, which gains a
  `reference=None` keyword and writes `target["reference"]` when given
  (always given from the dialog — the group you drew against becomes the
  job's reference, so companion/replot backgrounds match).

**Tests:** extend the JSON-level tests for `inject_line_into_jobs`
(reference written / omitted); offscreen dialog test on a synthetic
`oblique_slices.h5` with two groups — switch background, assert the displayed
attrs changed and picked points survive.

## 3. Persistent marks + visual marking dialog

### 3.1 Storage (Qt-free core, `dfxm/stages/slices.py`)

- Layout: `/marks/<slice_name>` — one float64 dataset of offsets (µm) per
  slice group, at the file root of `oblique_slices.h5`. Volume-independent;
  absent group ⇒ no marks; old files unaffected (readers treat missing
  `/marks` as empty).
- `read_marks(h5_file_or_path) -> dict[str, list[float]]` — all marks, sorted.
- `write_marks(path, slice_name, offsets_um) -> list[float]` — snaps each
  offset to the nearest stored plane of that slice (same rule as
  `resolve_plane_index`), dedupes, sorts, replaces the dataset; empty list
  deletes it. Opens the file `r+` briefly (open–write–close; no long-lived
  writable handles). Returns the snapped list. Raises `StageUserError` with a
  hint if the slice name is absent.
- Known trade-off (accepted): re-running the slices stage rewrites the h5, so
  marks die with the sweep they belong to — offsets are only meaningful
  within one sweep anyway.
- **Reader hardening (required):** `/marks` is a root Group, and several
  readers enumerate root groups as volume ids. Audited sites and their fate:
  `profiles.list_volume_ids` would list it (its `volume_ids_with_slice`
  filter happens to exclude it, but the exclusion must become explicit);
  the slices figure-catalog builder (`dfxm/stages/slices.py` ≈ line 1316)
  would **crash** (`vg[sname]["slices"]` on a dataset); `build_pinned_spec`
  (≈ line 963) would mis-treat it; `replot_catalog` is already structurally
  guarded. A module-level `MARKS_GROUP = "marks"` constant is added and every
  root-enumeration site over `oblique_slices.h5` skips it (plus cheap
  isinstance guards where indexing currently assumes structure). A dedicated
  test runs a marked synthetic file through every catalog/enumeration path.

### 3.2 Shared plane browser

The plane-rendering guts currently inside `LinePickerDialog` (open file, list
present groups, resolve reference, read axes/attrs, draw one plane with
`cmap_nan_transparent` + stored vmin/vmax, ◀ plane ▶ stepping) are factored
into a shared helper widget `gui/widgets/plane_browser.py`
(`PlaneBrowserCanvas`), used by `LinePickerDialog`, the new marking dialog,
and the background dropdown. One renderer, not two. `LinePickerDialog`
behaviour is otherwise unchanged (regression-guarded by existing smoke).

### 3.3 Marking dialog (`gui/widgets/mark_planes.py`)

`MarkPlanesDialog` — visual scroller:

- Slice-group dropdown (marks span slice names), background-field dropdown,
  ◀ plane ▶, a checkable "★ Mark" button reflecting the current plane's
  state, info line (`plane k/N  offset ±x.xxx µm  ★ M marked`), Save + Close.
- Loads existing marks on open; Save writes all slice groups' mark sets via
  `write_marks` and reports the snapped counts. Close without Save discards
  edits (dialog tracks dirty state and confirms discard).
- New "Mark planes…" button on the slices stage (`gui/stage_view.py`, beside
  Pin planes…), resolving the h5 path exactly like `_on_pin_planes`.

### 3.4 ★ in existing dialogs

- `PlaneRow` (`gui/widgets/plane_selection_model.py`) gains `marked: bool`.
- Row-label builders show `★ ` on marked planes; `PlaneSelectionPanel` gains a
  "Marked only" filter checkbox, visible only when at least one row is marked.
- Wired where marks exist: Pin planes… and the slices Replot… dialog. The
  generic (strain/mosaicity/rocking/profiles) replot dialogs are untouched —
  marks are a slices concept.
- `LinePickerDialog` shows ★ in its info line when the current plane is
  marked.

**Tests:** synthetic-h5 pytest for `read_marks`/`write_marks` (round-trip,
snapping, dedupe, replace, delete-on-empty, missing slice → `StageUserError`,
old file without `/marks`); `plane_selection_model` unit test for the
`marked` flag + marked-only filtering; offscreen dialog construction + mark
toggle + save; gui_smoke entry for the button.

## 4. Jobs from marks (profiles stage)

- New "Jobs from marks…" button beside Pick line… (`gui/stage_view.py`,
  profiles only). Reads `consolidated_h5` from the form (same guard/log
  message pattern as Pick line…), calls `read_marks`; if empty, logs a hint
  pointing at Mark planes… on the slices stage.
- A small checklist dialog lists all marks (`slice_name @ offset µm`), all
  checked by default. OK → for each checked mark in order, open
  `LinePickerDialog` pre-navigated to that slice + offset, window-titled
  `Pick line (k/N) — <slice> @ <offset>`.
- Each accepted line **appends** one complete job via a new
  `append_line_job(jobs_json, slice_name, start, end, offset, fields=None,
  reference=None) -> str` in `gui/viewers.py` — append semantics, never
  updates an existing job (unlike `inject_line_into_jobs`), so multiple marks
  on one slice and pre-existing same-name jobs are safe (the profiles stage
  already de-duplicates output stems for same-named jobs).
- Cancelling one picker skips that mark and continues with the next; the log
  reports `added J job(s), skipped S` at the end and the form's `jobs_json`
  is updated once, after the loop.

**Tests:** JSON-level tests for `append_line_job` (append to empty/existing,
same-name jobs, fields/reference threading); offscreen test for the checklist
dialog; gui_smoke entry.

## 5. Tooltip precision sweep

- Full pass over every `Param.help` in `dfxm/stages/*.py` and the experiment
  schema (`dfxm/config`). Tighten vague units, blank-value semantics,
  defaults, and cross-references.
- Named priorities: every ROI-ish param's help states its **coordinate frame
  in the first sentence** — absolute detector pixels (rocking `roi_x/y`) vs
  darfix-map pixels (slices `align_roi_x/y`, strain `roi`, analysis ROIs) —
  and `subtract_background`'s mosaicity-mode behaviour is described from the
  code, not from memory.
- `docs/Usage.md` updated wherever a help-string change alters documented
  behaviour wording; `tests/test_param_metadata.py` must stay green.
- Deliverable is a reviewable diff — no behaviour changes, strings only.

## Error handling

- `write_marks` on a read-locked/missing file → `StageUserError` with a hint
  (close viewers holding the file / run slices first); dialogs surface it in
  the existing log-console pattern, never a traceback dialog.
- All new buttons follow the existing lazy-import + guard-then-log pattern
  (`_on_pick_line` / `_on_pin_planes` precedents).
- Marks whose offsets no longer match any stored plane after a re-run cannot
  occur (file is rewritten ⇒ marks gone); `read_marks` still tolerates
  malformed `/marks` content by skipping bad datasets.

## Documentation (same change, both docs)

- `docs/Usage.md`: slices stage — "Marking interesting planes" (replaces the
  tool-based "Pinning one plane from a sweep" note as the primary flow;
  `tools/pin_slice.py` stays documented); profiles stage — "Jobs from
  marks…" + the Pick-line background dropdown; a line on the new wheel
  behaviour in the forms section.
- `docs/Codebase.md`: new modules (`wheel_guard`, `plane_browser`,
  `mark_planes`), new functions (`read_marks`, `write_marks`,
  `append_line_job`), changed signatures (`inject_line_into_jobs`,
  `LinePickerDialog.result`), `PlaneRow.marked`.

## Out of scope

- Marks in the generic replot dialogs (strain/mosaicity/rocking).
- A CLI for marks (GUI + Python API only; `tools/pin_slice.py` unchanged).
- Wheel-guarding widgets outside `ParamForm` (helper is reusable; adoption
  elsewhere is a follow-up).
- Any change to profiles CSV/figure output formats.
