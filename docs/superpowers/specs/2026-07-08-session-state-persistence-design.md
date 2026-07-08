# Session state: per-experiment form persistence + replot default-checked

**Date:** 2026-07-08
**Scope:** GUI only. Two independent parts on one branch.

## Part 1 — Replot tree defaults to all-checked

**Problem:** Both replot dialogs (`gui/widgets/slice_replot.py`,
`gui/widgets/replot_dialog.py`) populate their checkable tree with every node
**unchecked**. There is a "Select all" button, but a user who doesn't spot it
ticks one node and Render remakes only that one (the "only remade the COM/chi
map" report).

**Fix:** After building the tree in `_reload()`, default every node to
**checked** (call `select_all()` at the end of `_reload`). "Deselect all" and
per-node ticking still let the user choose a subset. Applies on Browse/Load too.

**Test:** `_selections()` is non-empty immediately after the dialog opens (both
dialogs).

## Part 2 — Per-experiment form-state persistence (QSettings)

**Goal:** Restarting the app resumes each stage's form exactly as left, per
experiment, so analysis continues without re-entering paths/folders/numbers.

### Store — `gui/form_state.py` (new)

`FormStateStore`, backed by the app-wide `QSettings` (org `dfxm`, app
`pipeline`), sibling to `window_state.py`:

- key: `formState/<slug(experiment-name)>/<stage>` (the experiment name is
  slugged so `/` etc. can't create stray QSettings groups).
- value: `json.dumps(form_values)` — JSON is type-safe across QSettings
  backends; `set_values` + `Param.coerce` re-hydrate on load.
- `load(exp, stage) -> dict | None` (None on missing/corrupt — swallow JSON
  errors, matching `window_state`'s defensive restore).
- `save(exp, stage, values)`.

### StageView ownership

`StageView.__init__(..., store: FormStateStore | None = None)` — **optional**;
`None` disables persistence entirely (keeps existing unit-test construction and
`set_experiment` semantics unchanged).

When a store is present:
- `self._calib_names = {p.name for p in spec.params if p.calibration}`.
- `_persistable_values()` = `form.values()` minus the calibration keys.
- Debounced **save-on-edit**: a single-shot `QTimer` (~400 ms) restarted on the
  form's `changed` signal; on timeout `store.save(exp.name, stage,
  _persistable_values())`. Wired *after* the initial restore so construction
  doesn't self-save.
- `flush()`: stop the timer and save now (called by `MainWindow.closeEvent`).
- `_restore_state()`: overlay `store.load(exp.name, stage)` onto the form
  (saved never contains calibration keys, so calibration is untouched).
- `__init__` calls `_restore_state()` after building the form (startup restore
  for the initial experiment).
- `set_experiment(new)` with a store: `flush()` (snapshot outgoing under the old
  name) → `self._experiment = new` → rebuild `defaults → experiment_overrides` →
  `_restore_state()` (overlay new experiment's saved values). Without a store it
  keeps today's behavior (apply overrides only).

### MainWindow wiring

- Construct `self._form_state = FormStateStore()`; pass it to every `StageView`.
- `closeEvent`: `for v in self._views.values(): v.flush()` (also flushes any
  pending debounce).
- `_on_experiment_changed` is unchanged — the outgoing-save happens inside
  `set_experiment`.

### Precedence (confirmed with user)

`defaults` → `experiment_overrides(stage, exp)` → **saved overlay (wins)**, with
**calibration-flagged params excluded from the overlay** — those always follow
the experiment ("calibration is physical", per CLAUDE.md). Non-calibration
experiment-derived fields (e.g. paths from `processed_root`) are covered by the
overlay, so **known limitation**: editing an experiment's non-calibration fields
mid-session won't re-propagate into a stage that already has saved values for
that experiment (the saved value wins). Acceptable for the resume-first goal.

## Non-goals

- Dialog-internal state (replot tree ticks, clim/outdir/ROI, pick-line jobs,
  export choices) is **not** persisted — forms only, per the user.
- No new UI to clear/reset saved state (could be a later "reset stage to
  experiment defaults" button).

## Tests

- `FormStateStore` round-trip (save/load, per-experiment keying, type
  preservation, corrupt-JSON → None, name slugging).
- `StageView` persistence: save-on-edit writes; `flush()` writes; calibration
  keys never saved; `_restore_state` overlays saved but not calibration;
  `set_experiment` saves outgoing + loads incoming (use an in-memory
  `QSettings` or a temp `.ini`).
- Replot default-checked assertions (both dialogs).
- `gui_smoke`: a step that edits a field, rebuilds the window against the same
  QSettings, and asserts the value restored.

## Docs (same change)

- `docs/Usage.md` — replot default-checked; a "Resuming a session" note
  (per-experiment, auto-saved, calibration follows the experiment).
- `docs/Codebase.md` — `gui/form_state.py`; the `StageView` store/debounce/flush
  and `MainWindow` wiring rows.
