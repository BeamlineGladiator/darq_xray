# Surfacing the cost advisory in the GUI — design

**Date:** 2026-08-23
**Status:** approved for implementation planning.
**Scope:** phase 6 of the machine-aware robustness project — making the cost
estimate, the machine profile and the run plan visible to the user. No change to
what any stage actually does.

## Goal

Phases 1–5 built a machine-aware cost model and rewrote every volume stage onto
a bounded-memory ladder. **None of it is visible in the app.** `plan_run` has no
production caller, `StageSpec.estimator()` is never called from `gui/`, and
`gui/` imports nothing from `machine.py` or `advice.py`. The user learns what a
run costs by watching it succeed or die.

This phase closes that gap and nothing else: before you commit to a run, the app
tells you what it will cost, what the machine can afford, and what it therefore
expects to do.

Out of scope, deliberately, and each for a stated reason below: recalibrating the
four stale estimators, bounding `visualize`'s 3-D overrun, and injecting the
`RunPlan` into stage parameters.

## Starting point

What exists and is verified (master `f0a0f2d`):

- `dfxm/common/machine.py` — `profile(output_dir=…, probe_gl_now=…)`, never
  raises, GL probed only on request and disk-cached (`gl_cache_path`).
- `dfxm/common/advice.py` — `headroom_bytes`, `working_set_budget_bytes`,
  `plan_run` → `RunPlan(strategy, budget_bytes, chunk_layers, downsample,
  scratch_dir, reasons, blocked)`, and `advise_3d` → `Advice(downsample,
  render_mode, reasons)`. Pure, no IO, tested against synthetic machines in
  `tests/machine_fixtures.py`.
- `dfxm/config/models.py` — `CostEstimate(peak_bytes, input_bytes, shape,
  chunkable, note, chunk_span, scratch_bytes)` and `StageSpec.estimate` /
  `.estimator()`.
- Seven stages declare an estimator: `strain`, `mosaicity`, `rocking`,
  `visualize`, `paraview`, `slices`, `matched`.
- `dfxm/common/raster.motor_positions_for_estimate` memoises the ~76-folder
  motor read **specifically so `estimate()` can rerun on every GUI form change**.
  Nothing has ever exercised that promise.

The gaps this phase fills:

- No GUI surface reads a `MachineProfile`, a `CostEstimate` or a `RunPlan`.
- `RunPlan.blocked` — insufficient scratch disk, the project's one genuine
  blocker — has no consumer, so the failure it exists to prevent still happens
  mid-run.
- `gui/theme.py` has `banner-error` and `banner-success` roles only; there is no
  neutral role for informational text.
- `MainWindow` is a `QMainWindow` with no status bar.

## Decisions taken during brainstorming

1. **All four surfaces** from the original spec are in scope: pre-flight banner,
   System check dialog, live status bar, inline per-param hints.
2. **The GUI is advisory only. It injects nothing.** The 2026-08-20 spec said
   the GUI would compute the `RunPlan` and inject it into `run_params` the way
   `plot_style` is injected, with the stage falling back to its own. Phase 5 went
   the other way: `visualize`, `slices` and `paraview` each call
   `advice.working_set_budget_bytes(machine.profile(...), rss_floor_bytes=…)`
   internally with a **per-stage measured RSS floor** the GUI has no basis to
   guess. Injection would now override a decision the stage makes better than its
   caller. That part of the old spec is superseded.
3. **A blocked run asks, it does not refuse.** When `plan_run` reports
   insufficient scratch disk, the GUI shows the measured numbers in a
   confirmation dialog and runs if the user says so. Consistent with "nothing
   refuses to run", and appropriate for a prediction the stage itself never
   consults.
4. **Cost is shown live *and* at Run.** One computation, two renderings — a
   persistent line under the stage form that follows the parameters, and a fuller
   banner when the run starts.
5. **Estimators known to over-predict are marked as such.** `CostEstimate` gains
   a `confidence` field; the four stale estimators set it; the GUI renders
   "at most ~N (conservative estimate)".
6. **`chunk_layers` is not displayed.** See "What the messages may claim" below.

## Architecture

One new Qt-free module composes; one new GUI module caches and debounces; four
surfaces render. No surface computes policy.

```
dfxm/common/advisory.py     compose: (stage, params) -> Advisory      [Qt-free]
gui/advisor.py              profile cache + debounce + worker thread
gui/widgets/system_check.py the probe table dialog
```

### `dfxm/common/advisory.py`

```python
@dataclass(frozen=True)
class Advisory:
    profile: MachineProfile
    estimate: CostEstimate | None    # None: no estimator, or estimation failed
    plan: RunPlan | None             # None whenever estimate is None
    headline: str                    # one line, for the live label
    details: tuple[str, ...]         # composed here, NOT plan.reasons verbatim
    blocked: str | None              # mirrors plan.blocked, or None
    conservative: bool               # estimate.confidence != "measured"
    hints: dict[str, str]            # advice_key -> note, for inline hints
```

Two public functions:

- **`disk_probe_dir(spec, params) -> str`** — which directory's filesystem to
  measure for free space. `output_dir` when set; else the directory of the first
  `must_exist` input path present in `params`; else `os.getcwd()`. This rule
  earns its own tested function: `output_dir` is optional on all seven stages
  and each `run()` computes its own fallback internally (e.g.
  `paraview.py:1630`), so a naive `params["output_dir"] or "."` reports the free
  space on the repository's filesystem while the data sits on an external SSD —
  precisely the machine where the disk answer matters.

- **`advise_stage(stage_name, params, *, profile=None) -> Advisory`** — resolve
  the spec from the registry, resolve and call its estimator, call
  `advice.plan_run`, compose the strings. The optional *profile* lets a caller
  (and every test) supply a measured or synthetic machine instead of this one.

**`advise_stage` never raises.** Estimators open user-supplied HDF5 paths and
read raw scan folders; a missing file, a corrupt file or a half-typed path is
the normal case while a form is being filled in, not an error. Any exception
becomes an `Advisory` with `estimate=None` and a headline naming the failure
(`"cannot estimate: FileNotFoundError"`). A traceback escaping into a
form-change handler would take the window with it.

A stage with no estimator (`concat`, `profiles`) returns an `Advisory` with
`estimate=None` and an empty headline; its surfaces render nothing.

### What the messages may claim

The headline states **cost and headroom**, and the strategy only as an
expectation:

```
needs ~10.5 GB, 3.6 GB safely available — expected to stream
at most ~6.6 GB (conservative estimate), 3.6 GB available — expected to stream
needs 0.2 GB, 3.6 GB available — runs in memory
select input files to estimate cost
```

**`chunk_layers` is deliberately not shown, and this constrains `details`.**
`plan_run` writes the group count into `RunPlan.reasons` itself ("chunking into
groups of 8 of 76 layers"), so `details` **must not** be `plan.reasons` verbatim
— `advisory.py` composes it, substituting a group-count-free sentence for that
one reason and passing the rest through. A test pins that no rendered string
contains the chunk count.
 `RunPlan.chunk_layers` is
display-only and is *not* the blocking a stage picks: `visualize`, `slices` and
`paraview` each derive their own from `working_set_budget_bytes` with a per-stage
RSS floor. Printing "groups of 8 of 76 layers" would state as fact a number
nothing acts on. `advice.py`'s own module docstring makes the general point — a
wrong unit in an advisory message is how advice stops being read — and an
unowned number is the same failure.

For the same reason the strategy word is hedged ("expected to"). The prediction
and the run share a headroom figure, so they agree in direction, but they are
computed independently and the GUI must not imply it is issuing an instruction.

### `CostEstimate.confidence`

```python
confidence: str = "measured"     # "measured" | "conservative"
```

Set to `"conservative"` by `strain`, `mosaicity`, `rocking` and `matched`, whose
models still describe the pre-phase-5 accumulate-then-`np.stack` code. Measured
over-prediction on the real STO2 dataset: **strain 5.2×** (2.627 GiB estimated
against 0.508 actual), **mosaicity 36×** (6.566 against 0.181).

The direction is safe — over-predicting only makes a stage stream harder — but a
banner that says "mosaicity needs 6.6 GB" for a run that needs 0.18 GB is
stating something false, and false-but-safe is exactly how a user learns to
ignore the banner. The field is the honest way to ship the surfaces without
first doing the recalibration.

A defaulted single-word field: the three verified estimators and every existing
test are untouched, and the marker disappears per stage as each is recalibrated —
no second cleanup pass, no flag to remember to remove.

### `gui/advisor.py`

- A module-level `MachineProfile` cache keyed by probe directory with a short
  TTL, so four surfaces and a 5 s status-bar timer cost one probe. **Never
  probes GL** (`probe_gl_now=False`).
- `StageAdvisor(QObject)`, one per `StageView`: a single-shot debounce timer on
  the existing `ParamForm.changed` signal, a worker thread following the
  `BatchWorker` pattern in `gui/widgets/busy.py`, latest-wins with stale results
  discarded (the pattern `gui/figure_builder.py` already uses), emitting one
  `advisoryReady(Advisory)` signal.

Off the GUI thread because the estimators do real IO: `sum_dataset_bytes` opens
every candidate HDF5 file and is **not** memoised (only the motor read is), so a
synchronous call per keystroke would stutter the form on network or external
storage.

### GL, and when it is probed

The valuable inline hints come from `advise_3d`, which needs `GLInfo`, which
needs the probe child. Policy:

- The status bar and the cost surfaces **never** trigger a GL probe; they render
  GL only on a cache hit.
- The first time a stage carrying 3-D parameters is opened, `gui/advisor.py`
  probes GL **once per session on a background thread**. Results are already
  disk-cached by `machine.py`, so every later launch is instant and the hint
  simply appears when the probe lands.
- The System check dialog probes on demand, and its **Re-probe** button forces
  `use_cache=False`.

The old spec's hazard stands unchanged: the probe child must not re-enter the
GUI. It is a leaf module (`dfxm/common/_glprobe.py`) run as a subprocess, which
is what makes this safe under `spawn`.

## The four surfaces

**1 · Live cost line.** A `QLabel` in the left column below the button row
(`gui/stage_view.py:205`), `headline` as text, `details` as tooltip. Hidden for
stages with no estimator.

**2 · Pre-flight banner.** `_on_run` (`gui/stage_view.py:389`) uses the last
advisory when it is fresh, else computes inline under `busy_cursor` — a click
may block briefly, a keystroke may not. If `blocked`, a `QMessageBox` naming the
measured numbers ("needs 12.4 GB of scratch disk but only 3.1 GB is free — run
anyway?"); on decline, nothing starts. Otherwise `_show_banner` with the new
`banner-info` role, showing `headline` and `details`, then the runner starts
exactly as it does today. **The banner is never a gate on memory** — only the
disk confirmation can stop a run, and only with the user's assent.

**3 · Status bar.** `MainWindow` gains `self.statusBar()` with a label refreshed
on a 5 s timer from the cached profile: `36 cores · 12.4 GB free · 24.6/64 GB
RAM`. GL appended only when already probed.

**4 · Inline hints.** `Param` gains `advice_key: str = ""`, keeping the GUI
schema-driven per CLAUDE.md; `gui/widgets/param_form.py` renders a note under any
widget whose key appears in `Advisory.hints`. Initially populated from
`advise_3d` for the render-mode and downsample parameters of `visualize` and
`paraview`.

### `gui/theme.py`

A third `QLabel[role="banner-info"]` rule beside `banner-error` and
`banner-success` (`gui/theme.py:178`), neutral in both light and dark themes.
Neither red nor green is right for "here is what this will cost".

## Data flow

```
form values ──> advise_stage ──> spec.estimator() ──> CostEstimate ──┐
                     │                                              ├──> plan_run ──> RunPlan
                     └──> machine.profile(disk_probe_dir(...)) ──────┘        │
                                                                              v
                                                                          Advisory
                                                                              │
              live line · pre-flight banner · status bar · inline hints ──────┘

stage run() ──> its own machine.profile() + working_set_budget_bytes(own RSS floor)
                (unchanged by this phase, and not fed by the Advisory)
```

The second line is the point of decision 2: the advisory path and the execution
path are parallel, not sequential.

## Error handling

- `advise_stage` never raises; every failure becomes a rendered headline.
- A profile whose probes failed already carries `probe_errors`; the System check
  dialog renders those as "unknown" with the stated reason, and the status bar
  omits the field rather than showing a zero.
- `disk_free == 0` (unmeasurable) disables the blocked check rather than
  blocking every run — `plan_run` already guards on `profile.disk_free` being
  truthy.
- A GL probe crash or timeout is `gl_status="crashed"`, rendered as such in the
  dialog; no hint is shown and nothing else changes.

## Testing

Qt-free (`tests/test_common_advisory.py`), against the synthetic machines in
`tests/machine_fixtures.py`:

- headline and details for in-core, streaming and blocked plans;
- no rendered string carries `chunk_layers`, on a plan whose `reasons` does;
- an estimator that raises → `estimate is None`, headline names the exception,
  no propagation;
- a stage declaring no estimator;
- the conservative marker present for the four marked stages and absent for the
  three measured ones;
- `disk_probe_dir` on all three branches, including the one that matters — an
  unset `output_dir` resolving to the *input file's* filesystem, not cwd.

Offscreen Qt tests: the live line updating on a form change; the banner text at
run; the blocked path with `QMessageBox` monkeypatched, asserting both that
declining starts no runner and that accepting does; the status-bar text; a hint
rendered under the right widget.

`tests/gui_smoke.py` gains a step covering the live line and the System check
dialog.

**Mutation discipline is a requirement of this phase, not a suggestion.** Twenty
checks in this project have been found to have stopped checking what they name,
and the nineteenth and twentieth were each authored by the fix for the one
before. For every test added here: run the mutation that should break it and
confirm it does, and assert the precondition that keeps the fixture inside the
region the test claims to cover — in particular, the blocked test must assert
that its fixture machine really is short of disk relative to the estimate, or it
silently becomes a test of the unblocked path.

## Documentation

Both docs, in the same change, per the CLAUDE.md contract:

- `docs/Usage.md` — reading the cost line and the pre-flight banner, what
  "conservative estimate" means and why some stages say it, the System check
  dialog, and what the disk confirmation is asking.
- `docs/Codebase.md` — `dfxm/common/advisory.py`, `gui/advisor.py`,
  `gui/widgets/system_check.py`, `Param.advice_key`, `CostEstimate.confidence`,
  the `banner-info` theme role.

## Explicitly out of scope

- **Recalibrating `strain`/`mosaicity`/`rocking`/`matched`.** A real measurement
  job: the plotting term must be measured in isolation (plots on/off across two
  or three dpi and figure-size combinations) before remodelling, because
  `strain.estimate`'s docstring warns that a naive per-layer model
  *under*-predicts — the dangerous direction, and an error this project has
  already made once. `confidence="conservative"` is what makes shipping the
  surfaces before that work honest rather than misleading.
- **Bounding `visualize`'s 3-D overrun** (4.80 GiB against an 8 GB machine's
  3.60 GiB budget when `save_topview`/`save_rotation` are on). Bounding it means
  decimating a saved product, which is a separate decision. Note that this phase
  delivers most of that item's value anyway: the overrun becomes visible
  *before* the run rather than announced during it.
- **Injecting the `RunPlan` into `run_params`.** Superseded — see decision 2.
- **Acting on CPU count.** Measured and displayed, never acted on; parallelism
  remains sub-project D.
