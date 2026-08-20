# Machine-aware robustness — design

**Date:** 2026-08-20
**Status:** approved. Phases 1–4 approved for implementation planning;
phases 5–8 deferred pending the estimator's real numbers.
**Scope:** sub-projects A (machine profile + advisory) and B (memory-safe data
path), brainstormed and specced together.

## Goal

The app must run to completion on any machine it is placed on — workstation or
laptop, Linux or Windows, hardware GL or software GL, 500 GB of RAM or 8 —
degrading to *slower*, never to *failed* and never to *silently wrong*. Where a
machine's limits force a compromise, the app measures the machine, says what it
is doing and why, and records the decision in the data product.

Out of scope here (separate later specs): sub-project C, on-hardware Windows
verification; sub-project D, speed optimisation of hot loops.

## Starting point

The repository is already better positioned than it looks:

- `dfxm/runner.py:123` runs stages under the `spawn` start method with
  `"module:function"` targets — the hardest Windows blocker is solved by design.
- No POSIX-only imports (`fcntl`/`resource`/`pwd`), no hardcoded `/` paths, no
  `os.sep` assumptions in `dfxm/` or `gui/`.
- `pyvista`/`vtk` are imported lazily; `PvCanvas.ensure()`
  (`gui/widgets/pv_canvas.py:31`) degrades to a placeholder label on any GL
  failure.
- One real hardware probe already exists: `volume_texture_limit()` /
  `oversize_note()` (`dfxm/common/render3d.py:356`) read
  `GL_MAX_3D_TEXTURE_SIZE` and warn when a volume exceeds it. This project
  generalises that seed.
- `mosaicity.py:223` already streams layer-by-layer and is the template for
  chunked IO.

The gaps:

- **No memory awareness at all.** No `psutil`, no pre-flight estimate. Twelve
  sites read a whole volume with `[:]` and are the out-of-memory kills on a
  laptop: `viewer_jobs.py:20`, `visualize.py:457`, `visualize.py:463`,
  `slices.py:735`, `slices.py:751`, `paraview.py:595`, `paraview.py:601`,
  `strain.py:368`, `mosaicity.py:398`, `rocking.py:985`, `matched.py:268`,
  `matched.py:272`. (A further fifteen `[:]` reads are small 1-D coordinate
  arrays — `u_um`, `v_um`, `z_um`, `offsets_um` — and stay as they are.)
- **Four of those twelve read and convert in one expression.**
  `slices.py:735`, `slices.py:751`, `matched.py:268` and `matched.py:272` all do
  `[:].astype(np.float64)`, holding the raw array and its float64 copy live at
  the same time — a 3x transient spike from a float32 source, not 1x. Peak
  estimates and conversions must both account for this; it is worst precisely
  where it hurts most.
- **No CPU awareness.** Nothing consults core count. This spec measures and
  reports it but does not act on it — parallelism is sub-project D.
- **`ffmpeg` handled ad hoc** at each call site (`dfxm/common/render.py:188`)
  rather than probed once.
- **Windows never exercised.** Spawn-safe is not the same as verified.

The development workstation is itself an instructive case: 36 cores, 502 GB RAM,
and **`llvmpipe` software rendering with no GPU**. GPU capability and RAM are
independent axes, not one fast/slow dial, and the design treats them so.

## Decisions taken during brainstorming

1. Sub-projects A and B are specced and implemented together; C and D follow.
2. When a setting is too heavy for the machine, the app **auto-degrades and says
   so**, with the knob left available to override.
3. For analysis stages the automatic degradation is **slower, never coarser**:
   chunked or disk-backed execution producing a bit-identical product.
   Coarsening exists only behind an explicit opt-in parameter.
4. The profile and its recommendations surface in **all four** places: a System
   check dialog, inline hints on affected params, a pre-flight banner at Run
   time, and a status-bar indicator.
5. **Every** volume-loading stage is converted, not just the worst offenders.
6. Windows code paths are written now, defensively and stdlib-first, and marked
   unverified until a machine is available.
7. Nothing refuses to run for lack of RAM. The escalation is
   **in-core → chunked → disk-backed**; the only genuine blocker is disk space.

## Architecture

Three new Qt-free modules in `dfxm/common/`, one probe child, and four GUI
surfaces that read them without computing policy.

```
dfxm/common/machine.py     probe hardware once, cache it
dfxm/common/_glprobe.py    leaf module run as a child process; prints JSON
dfxm/common/advice.py      pure: (profile, estimate, params) -> RunPlan
dfxm/common/volumeio.py    chunked / two-pass / disk-backed readers
```

Rejected alternatives:

- **Per-stage self-management** (each stage handles its own memory, no central
  model): no pre-flight prediction to show the user, and eight divergent
  chunking implementations.
- **Out-of-core arrays via dask/zarr**: solves memory as a category, but a heavy
  dependency plus a broad rewrite of numerically delicate code
  (`alignment.py` ordering, strain detrending), and h5py+dask on Windows is its
  own project. Wrong risk for a stability push.

### `machine.py` — the probe layer

```python
@dataclass(frozen=True)
class GLInfo:
    renderer: str            # e.g. "llvmpipe (LLVM 20.1.2, 256 bits)"
    vendor: str
    version: str
    max_3d_texture: int | None
    software: bool           # llvmpipe/swrast/softpipe/"Microsoft Basic Render"/"GDI Generic"

@dataclass(frozen=True)
class MachineProfile:
    os_name: str                    # "Linux" | "Windows" | "Darwin"
    cpu_logical: int
    cpu_physical: int | None
    ram_total: int                  # bytes
    ram_available: int              # bytes, always live, never cached
    disk_free: int                  # bytes, measured against the output directory
    gl: GLInfo | None               # None -> 3-D disabled, with a stated reason
    gl_status: str                  # "ok" | "no-gl" | "crashed" | "no-vtk" | "unprobed"
    ffmpeg: str | None              # resolved path
    probe_errors: tuple[str, ...]   # every failure, human-readable
```

**Probing rules.** Each probe is individually wrapped: a failure appends to
`probe_errors` and leaves its field `None`. **There is no code path in which
building a profile raises.** A machine we cannot measure is one we describe as
unmeasured, never a crash.

- **CPU / RAM** — `psutil` when importable; otherwise `os.sysconf` on POSIX and
  `ctypes.GlobalMemoryStatusEx` on Windows. `psutil` is added to
  `pyproject.toml` dependencies *and* the fallback is kept, so its absence
  degrades one feature rather than breaking launch.
- **Disk** — `shutil.disk_usage` against the stage's output directory.
- **ffmpeg** — `shutil.which` once, replacing the ad-hoc handling at each call
  site.
- **GL — out of process.** `machine.py` spawns
  `python -m dfxm.common._glprobe`, which builds the throwaway off-screen
  plotter, prints one JSON line and exits. The parent reads it with a timeout.

**Why the GL probe must not run in this process.** `_probe_texture_limit()`
(`render3d.py:374`) builds a real off-screen plotter to read the texture limit,
and a bad or missing driver does not raise — it segfaults. The project memory
records `viewer3d` GL tests segfaulting in this very environment. Today that
risk is confined to 3-D paths the user opts into; a startup profile probing GL
in-process would let a broken laptop driver take down the app on launch, the
exact opposite of this project's goal. Child died or timed out →
`gl_status="crashed"` → 3-D disabled with a clear reason. The existing
in-process `_probe_texture_limit()` body moves into the child, so there is one
implementation rather than two.

**Caching.** In-process memo, plus the GL result cached to JSON at
`~/.cache/dfxm/` (POSIX) or `%LOCALAPPDATA%\dfxm\` (Windows), keyed on
`(os, python version, vtk version, hostname)`. GL probing costs a process spawn
and a context creation, and its answer cannot change mid-run. RAM and disk
figures are always live and never cached. The System check dialog offers
**Re-probe** for driver updates.

**Startup cost is zero.** `MachineProfile` is built lazily on first request. The
status-bar indicator asks only for the cheap fields; the GL field is filled by a
background probe after the window is up, so a broken driver can neither delay
nor crash launch.

### `advice.py` — the cost model and policy

`StageSpec` gains one field:

```python
estimate: str | None = None    # "module:function" target, resolved lazily
```

A string rather than a callable, so `dfxm/stages/registry.py` keeps its property
that importing the registry never drags in h5py or matplotlib. The estimator is
imported only when a prediction is requested, through the existing
`registry.resolve`.

```python
@dataclass(frozen=True)
class CostEstimate:
    peak_bytes: int                 # in-core high-water mark, whole-volume strategy
    input_bytes: int                # on-disk volume(s) touched
    shape: tuple[int, ...] | None
    chunkable: bool                 # can this be streamed losslessly?
    note: str | None
```

Estimators open the HDF5 file and read **`.shape` and `.dtype` only** — never
data. That costs microseconds, so an estimate can be recomputed on every form
change and drive the live status bar and the pre-flight banner.

**Memory is predicted; time is not.** Peak bytes from shapes is arithmetic and
defensible. Runtime depends on disk, cache and CPU in ways a first version would
get badly wrong. Elapsed-time feedback already exists in `dfxm/common/eta.py`
and remains the source of truth once a run is moving.

Pure functions, no IO, no Qt:

```python
@dataclass(frozen=True)
class RunPlan:
    strategy: str                   # "in-core" | "chunked" | "disk-backed"
    budget_bytes: int               # what volumeio may hold at once
    chunk_layers: int               # 0 when strategy is "in-core"
    downsample: int                 # 1 unless allow_downsample opted in
    scratch_dir: str | None         # set only for "disk-backed"
    reasons: tuple[str, ...]        # plain-language, shown verbatim
    blocked: str | None             # set only for insufficient disk

@dataclass(frozen=True)
class Advice:
    downsample: int                 # smallest factor fitting max_3d_texture
    render_mode: str | None         # suggested mode, None = keep current
    reasons: tuple[str, ...]

def plan_run(profile, estimate, params) -> RunPlan
def advise_3d(profile, shape, mode) -> Advice
```

Note that `RunPlan` carries no worker count. Nothing in the pipeline is
parallelised today, and adding a field no consumer reads would be speculative;
CPU-count-driven parallelism belongs to sub-project D, which adds the field when
something needs it. `cpu_logical` / `cpu_physical` are still probed and
displayed, because the System check dialog reports them.

`advise_3d` returns the smallest `downsample` that brings every dimension within
`max_3d_texture`, and nudges toward `surface`/`isosurface` on software GL —
those upload geometry rather than one large 3-D texture, the failure
`render3d.py:356` already documents.

Headroom policy lives in one named constant: target peak
`<= min(0.6 * ram_available, 0.5 * ram_total)`, leaving room for Qt, matplotlib
and the OS. Every decision carries a plain-language `reason` string, and those
strings are what the banner, the log and the stage result notes all display, so
each explanation is written once.

**Three buckets.** "Chunk it and the answer is bit-identical" is not universally
true. Some operations are global — strain's detrend-before-ROI fit, the
percentile-based `auto_clim`, the centring in `alignment.py`. Every one of the
twelve full-load sites is classified as:

1. **Trivially streamable** — layer-independent work. Direct conversion;
   `mosaicity.py:223` is the template.
2. **Two-pass streamable** — needs a global statistic first: pass 1 accumulates,
   pass 2 applies, at the cost of one extra read.
3. **Genuinely in-core** — irreducibly whole-array. Not chunked; instead run
   **disk-backed** (below) with an accurate `peak_bytes` and `chunkable=False`.

Classifying all sites into these buckets is part of the implementation plan.
`strain.py`'s detrend-before-ROI and the percentile `auto_clim` calls are the
likely bucket-3 candidates. A site landing in bucket 3 is a correct outcome, not
a failure.

### `volumeio.py` — chunked, two-pass and disk-backed IO

One shared module so eight stages do not invent eight chunking schemes.

```python
def iter_blocks(dset, *, budget_bytes, axis=0) -> Iterator[tuple[slice, np.ndarray]]
def load_or_stream(dset, *, budget_bytes) -> np.ndarray | BlockReader
def block_reduce(dset, fn, *, budget_bytes, init)        # one-pass accumulator
def two_pass(dset, stat_fn, apply_fn, *, budget_bytes)   # global statistic, then apply
def scratch_array(shape, dtype, *, dirpath)              # disk-backed working array
def volume_bytes(dset) -> int                            # shape * itemsize
```

`budget_bytes` comes from `RunPlan`, so identical stage code runs whole-volume on
a 500 GB box and block-wise on a 16 GB laptop with no branch in the stage.

**Disk-backed execution (bucket 3).** The working array is allocated as a
`np.memmap` over a scratch file rather than in RAM. `np.memmap` is an `ndarray`
subclass, so the numerical code is unchanged; pages spill to disk and the OS
pages them back. Slow, but it runs. Two requirements, or this is a false
promise:

- **Temporaries still allocate in RAM.** A memmapped `a` does not help if the
  code writes `out = a * b + c`; numpy materialises a full-size temporary.
  Bucket-3 sites therefore also perform their element-wise passes in slabs with
  in-place operations. Memmap fixes the storage; slabbing fixes the temporaries.
  Both are required.
- **Scratch space is a measured resource.** `scratch_array` is a context manager
  that deletes on exit *including on crash*, with an explicit Windows path for
  "a mapped file that is still open cannot be unlinked".

Consequently `RunPlan.blocked` is never about RAM. The escalation is
**in-core → chunked → disk-backed**, and the only genuine blocker is running out
of disk, which is measured and stated in advance.

**Two invariants, enforced by tests rather than asserted in prose:**

1. **Chunking never changes the answer.** Every converted site has a test
   running one synthetic volume through both strategies and asserting
   bit-identical output. For float accumulation this requires a fixed,
   size-independent summation order — pairwise or Kahan within a block, then a
   deterministic combine. Without it, "identical" silently weakens to "close",
   and differing strain maps between laptop and workstation is precisely the
   failure this project exists to prevent.
2. **Coarsening is opt-in and recorded.** The `allow_downsample` parameter
   defaults off. When it fires, the factor is written into the output HDF5
   attributes and the result notes, so a coarse product can never be mistaken
   for a full-fidelity one months later.

`alignment.py` is explicitly out of scope for reordering. CLAUDE.md fixes the
pipeline order (`abs(FWHM) -> ROI -> samy X-shift -> uniform-Z interp ->
centre`); streaming must preserve it exactly, not optimise it.

### GUI surfaces

All four read the same `MachineProfile` and `RunPlan`; none computes policy.

1. **System check dialog** — `gui/widgets/system_check.py`, opened from a
   `System check…` button in the left rail beside `Publication style…`
   (`gui/main_window.py:116`). A table of probes (CPU, RAM, disk, GL, ffmpeg,
   VTK), each with measured value, verdict and setting implications; a
   **Re-probe** button and **Copy as text** for support. Auto-offered once per
   new machine fingerprint.
2. **Inline hints** — `Param` gains `advice_key: str = ""`, keeping the GUI
   schema-driven per CLAUDE.md. `param_form` renders a note under any widget
   whose key has an active advisory, e.g. *"software GL — 2048 px texture limit;
   this volume is 2891 px wide, downsample >= 2 suggested."*
3. **Pre-flight banner** — `StageView._on_run` (`gui/stage_view.py:400`)
   computes the estimate and plan before constructing `StageRunner`, and shows
   it through the existing `_show_banner`. This needs a third neutral
   `banner-info` role in `gui/theme.py` alongside `banner-error` and
   `banner-success`. The banner states estimated peak, the strategy chosen
   (in-core / chunked / disk-backed) and any coarsening, before work starts.
4. **Status bar** — `MainWindow` is a `QMainWindow` with no status bar today, so
   `self.statusBar()` provides the surface. A compact live readout
   (`SW GL · 12.4 GB free · 36 cores`) refreshed on a 5 s timer from cheap
   fields only; it never touches GL.

**CLI parity.** The GUI computes the `RunPlan` and injects it into `run_params`
exactly as `plot_style` is injected at `gui/stage_view.py:405`. A stage run
headless via `python -m dfxm.stages.strain` receives no plan, so `run()` falls
back to computing its own from `machine.profile()` and `advice.plan_run()`. The
protection is identical either way; the GUI merely gets to show the decision
first.

## Data flow

```
h5 shapes ──> stage estimator ──> CostEstimate ──┐
                                                 ├──> advice.plan_run ──> RunPlan
machine.profile() ──> MachineProfile ────────────┘                          │
                                                                            v
                                     GUI banner / hints / status bar   stage run()
                                                                            │
                                                    volumeio (budget_bytes) ┘
                                                                            │
                                              reasons ──> log + result notes + h5 attrs
```

## Error handling

- Probe failure never raises; it is recorded in `probe_errors` and rendered as
  "unknown" with a stated reason.
- GL child crash or timeout is a first-class result (`gl_status="crashed"`) that
  disables 3-D with an explanation, not an exception.
- Insufficient disk for a disk-backed run is the one genuine blocker: it raises
  `StageUserError(message, hint)` per the existing convention, stating bytes
  needed versus bytes free.
- Degrade decisions are *not* errors: they go to the log, the result notes and
  the output attributes.

## Testing

The suite currently collects 1064 tests. This work adds four kinds:

- **Synthetic profile fixtures** — `laptop_hw_gl` (16 GB, hardware GL),
  `workstation_sw_gl` (500 GB, llvmpipe — the development box),
  `windows_no_vtk`, `tiny_ram`. Every `advice.py` decision is tested against all
  of them. This is how the Windows code paths get genuine coverage before a
  Windows machine exists.
- **Equivalence tests** — per converted site: one synthetic volume through
  in-core, chunked and disk-backed paths, asserting bit-identical output.
- **The requirement as a test** — every stage runs on a small synthetic volume
  with an absurdly small `budget_bytes` and must *complete*. This encodes "runs
  no matter the setup, just slower" as something CI enforces rather than
  something the design claims.
- **Fast-path guard** — a volume that comfortably fits must still take the
  in-core path, so the workstation does not pay for the laptop's safety.

Plus offscreen GUI tests for the four surfaces via the existing
`tests/qt_helpers.py`, and a new step in `tests/gui_smoke.py`.

## Implementation sequencing

Eight phases, each independently reviewable. Phases 1–4 are infrastructure with
no user-visible change; phase 5 is where the promise lands; phase 6 is where it
becomes visible.

**Approved split (2026-08-20): phases 1–4 are implemented first as one run.**
Phase 5 is reconsidered afterwards, once the phase-3 estimators report real peak
figures for the STO2 dataset — those numbers are expected to redirect the
conversion order, and possibly to show that some of the twelve sites never
exceed the headroom budget on any target machine and need no conversion at all.
Deciding the conversion order before measuring would be guesswork.

**Equivalence policy (confirmed): bit-identical.** Every chunked path
accumulates in a fixed, size-independent order — pairwise or Kahan within a
block, then a deterministic combine — so a laptop and the workstation produce
byte-identical products, and diffing two outputs remains a valid proof of
equivalence. This constrains how each phase-5 conversion is written, and the
phase-4 harness asserts exact equality rather than a tolerance.

1. `machine.py` + out-of-process `_glprobe.py` + disk cache
2. `advice.py` + `CostEstimate` + the `StageSpec.estimate` field
3. Per-stage shape-only estimators
4. `volumeio.py` + the equivalence-test harness
5. Stage conversions — **one task per stage**, eight tasks, the bulk of the work
6. The four GUI surfaces
7. 3-D advisory wiring, extending `oversize_note` / `volume_texture_limit` into
   the new layer
8. Windows defensive paths + a verification checklist for when a machine is
   available

## Risks

- **The GL probe child must not re-enter the GUI.** Under `spawn` the child
  re-imports its module; `_glprobe.py` must be a leaf importing only `pyvista`.
  Getting this wrong produces a fork bomb of windows.
- **Bit-identical float summation** is the subtlest requirement. Fixed-order
  pairwise accumulation guarded by tests, or the guarantee silently weakens to
  "approximately equal".
- **Windows memmap deletion** — a mapped file still open cannot be unlinked; the
  scratch context manager needs an explicit Windows path.
- **Scope.** Phase 5 is eight stages, each requiring classification, conversion
  and an equivalence test. By task count this is the largest project in the
  repository's history, larger than the figure builder. This is stated so it is
  known before starting rather than discovered midway.
- **`psutil` becomes a declared dependency**, with stdlib fallbacks so its
  absence degrades the profile rather than breaking launch.
- **Windows remains unverified** at the end of this spec. Sub-project C closes
  that; until then the docs must say so plainly rather than implying tested
  support.

## Documentation contract

Per CLAUDE.md, in the same change and not as follow-up:

- `docs/Usage.md` — the System check dialog, how to read the pre-flight banner,
  and the `allow_downsample` opt-in.
- `docs/Codebase.md` — `machine.py`, `_glprobe.py`, `advice.py`, `volumeio.py`,
  and the `StageSpec.estimate` / `Param.advice_key` fields.
