# App-wide busy indication — design

Date: 2026-08-17. Approved by Albert (hybrid depth chosen via numbered
options: threaded renders with real spinner; wait-cursor elsewhere; ETA on
stage runs). Runs AFTER the trace-autoscale/collisions plan.

## Problem

The user cannot tell when the app is working. Stage runs already stream a
progress bar (dfxm/runner.py → StageView), but everything else that takes
time runs synchronously on the GUI thread with zero indication: the
figure-builder preview render and export, the five replot dialogs' batch
renders, line-picker volume loads, viewer loads, ROI-picker loads,
Mark-planes reads/writes, experiment detection. Requirement: something
visible whenever work is in flight — progress bar + time estimate where a
count exists, an animated spinner for renders, at minimum a wait cursor.

## Principles

- A spinner only animates if the GUI thread is free → the two heavy render
  paths move to worker threads. Everything else keeps its synchronous shape
  and gets an honest wait-cursor + status text (no risky threading of
  h5/GL code).
- One shared vocabulary: new `gui/widgets/busy.py` owns all of it.

## 1. Shared widgets/helpers — `gui/widgets/busy.py` (new)

- `BusyOverlay(QWidget)`: a translucent child overlay covering a host
  widget, showing an animated indeterminate spinner (a rotating arc drawn
  in `paintEvent`, driven by a QTimer, KIT-green on the theme background)
  plus a one-line text label; optional determinate mode showing a
  QProgressBar + `"{done}/{total} — {eta}"` text. API:
  `start(text)`, `set_progress(done, total, eta_text)`, `stop()`. The
  overlay resizes with its host (eventFilter on host resize), swallows
  input to the host while active.
- `busy_cursor(text="", widget=None)`: context manager — sets the
  application override cursor (WaitCursor), optionally writes `text` to
  `widget`'s status label if given, forces one `processEvents()` repaint
  so the cursor/text appear BEFORE the block, always restores in
  `finally`.
- `format_eta(elapsed_s, frac) -> str`: `"~{m} min left"` / `"~{s} s
  left"`; empty string when `frac < 0.05` or elapsed < 2 s (too noisy).
  Pure function (testable Qt-free? it has no Qt — put the pure helpers
  `format_eta` and an `EtaEstimator` (monotonic smoothing over (t, frac)
  samples) in `dfxm/common/eta.py` so they're Qt-free; busy.py imports).

## 2. Threaded figure-builder renders — `gui/figure_builder.py`

- `render_now` becomes non-blocking: it snapshots the recipe (deep copy via
  recipe_to_json/from_json — cheap, and guarantees the worker never shares
  mutable state with the GUI), then starts a `QThread` worker running
  `render_recipe(recipe_copy, loader_cache=self._cache)`.
  - `render_recipe` builds canvas-less matplotlib figures (already runs
    headless in pytest) — safe off the GUI thread. The `FigureCanvasQTAgg`
    is attached on the GUI thread in the result slot (`_show_figure`,
    unchanged).
  - The loader cache is handed to the worker; the GUI thread never touches
    it while a worker runs (renders are serialized — see latest-wins).
- Latest-wins serialization: at most one worker at a time; if a render is
  requested while one runs, remember the request and start it when the
  worker finishes; the superseded worker's result is dropped (its
  generation counter != current).
- UI while running: `BusyOverlay.start("Rendering…")` over the preview
  host; Refresh/Export buttons disabled. On finish: overlay stops, notes
  bar as today. On worker exception: overlay stops, error to the notes bar
  (same text as today's except paths).
- `export_now` runs its `export_recipe` through the same worker mechanism
  (spinner text "Exporting…"), reusing the just-rendered result is NOT
  attempted (export re-renders as today — simplest correct thing).
- The 300 ms debounce stays; `closeEvent` stops the timer AND asks a live
  worker to finish/discard (worker holds no Qt objects, so letting it run
  to completion and dropping the result is safe; block close only until
  the drop flag is set, never join the thread on the GUI thread).

## 3. Threaded replot batches — the five replot dialogs

- The replot render loops (`gui/widgets/replot_dialog.py`,
  `slice_replot.py`, `profiles_replot.py`) move their per-item loop into a
  shared worker (`busy.py`: `run_batch_in_thread(items, fn, on_item_done,
  on_finished)` or a small `BatchWorker(QThread)` yielding
  `itemDone(i, total)` signals).
- Dialog UI while running: determinate `BusyOverlay` over the dialog's
  body — progress `{i}/{N}`, ETA from `EtaEstimator`; Render/Close buttons
  disabled (Cancel requests stop after the current item).
- The per-item render functions are the existing Qt-free
  `render_replot`-family calls writing PNGs via Agg — thread-safe as used
  (one item at a time, no shared figure objects).

## 4. Wait-cursor sweep — short synchronous blocks

Wrap with `busy_cursor(...)` (text into the nearest status label where one
exists): line-picker volume load, viewer h5 loads (gui/viewers.py load
paths), ROI-picker load, Mark-planes read/write, Jobs-from-marks load,
experiment detect ("Initialize from data…"), figure-builder recipe open.
(3-D viewer window loads keep their existing placeholder/label behaviour —
GL code stays untouched; add the wait cursor only around the h5 read.)

## 5. Stage-run ETA — `gui/stage_view.py`

- `EtaEstimator` fed from the existing progress messages
  (`_on_progress`): progress text becomes
  `"{stage text} — ~{eta} left"` once estimable (frac ≥ 5%, ≥ 2 s
  elapsed); resets on run start. Progress bar itself unchanged.

## Error handling

- Worker exceptions are caught in the worker, delivered as a signal, and
  rendered exactly where today's synchronous exception text lands (notes
  bar / dialog status). No dialog/window may be left with a stuck overlay:
  `stop()` in every finish path (success, error, cancel, close).
- `busy_cursor` always restores the cursor (finally), including on raise.

## Testing

- Qt-free: `EtaEstimator`/`format_eta` behaviour (monotonic, quiet under
  5%, smoothing).
- Qt (offscreen): BusyOverlay start/stop/visibility + determinate text;
  figure-builder worker — render completes and attaches a canvas, error
  path writes notes and clears the overlay, latest-wins drops the stale
  result (two rapid renders → exactly one final canvas, generation
  checked); replot batch worker itemDone counts + cancel-after-item;
  StageView progress text gains ETA (fed synthetic progress messages);
  busy_cursor restores cursor on exception.
- Smoke: figure-builder step extended to wait on the async render
  (processEvents loop with timeout asserting the overlay appears then
  clears and n_rendered is set); one replot-dialog batch with the overlay.
- All existing figure-builder tests that call `render_now()` synchronously
  are adapted via a test helper `render_and_wait(window)` (drives the
  event loop until the worker delivers) so their assertions stay.

## Docs (same change)

- `docs/Usage.md`: what the spinner/overlay means, replot progress + ETA,
  stage-run ETA.
- `docs/Codebase.md`: busy.py, dfxm/common/eta.py, the worker architecture
  in figure_builder.py + replot dialogs, StageView ETA.
