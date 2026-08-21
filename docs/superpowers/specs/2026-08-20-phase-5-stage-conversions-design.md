# Phase 5 — stage conversions to bounded memory — design

**Date:** 2026-08-20
**Status:** approved for implementation planning.
**Scope:** phase 5 of [[2026-08-20-machine-aware-robustness-design]] — converting
the thirteen whole-volume load sites (the twelve named in the parent spec plus
`gui/viewers.py:53`, found here) to bounded-memory execution. Phases 6–8 (GUI
surfaces, 3-D advisory wiring, Windows verification) remain out of scope.

## Goal

The twelve `[:]` sites named in the parent spec — and the thirteenth,
`gui/viewers.py:53`, found here — stop being the out-of-memory
kills on a small machine. After this phase every volume stage completes the real
STO2 dataset with an enforced memory ceiling, producing output bit-identical to
an unconstrained run.

The parent spec deferred this phase deliberately, "once the phase-3 estimators
report real peak figures … those numbers are expected to redirect the conversion
order". They did, and they redirected more than the order.

## Starting point

Phases 1–4 shipped (master `1937cc5`) and are unchanged by this phase:
`machine.py` profiling, `advice.py` policy with `plan_run`, `CostEstimate` plus
seven verified stage estimators, `volumeio.py` with budget-independent
compensated reductions and `scratch_array`, and `tests/equivalence.py`.

The measured STO2 peaks that drive this phase (from commit `08513f5`):

| stage | peak | shape of the problem |
| --- | --- | --- |
| paraview | 17.07 GB | all fields raw + two float64 copies |
| visualize | 10.51 GB | both files' raw bytes never freed + 3 float64 temporaries |
| mosaicity | 6.57 GB | every layer of four datasets collected, then `np.stack` |
| slices | 6.40 GB | previous prepared volume held live across the loop |
| strain | 2.63 GB | per-layer maps collected, then `np.stack` |
| rocking, matched | small | already stream per scan |

Headroom is `min(0.6 · ram_available, 0.5 · ram_total)`. On the 8 GB floor the
parent spec commits to, that is roughly 3.6 GB, so every stage in the table
except rocking and matched exceeds it on STO2 — and STO2 is one dataset, not a
worst case. Spec decision #5, "every volume-loading stage is converted", is
confirmed by the numbers rather than softened by them.

## Findings that correct the parent spec

Four claims in the parent spec were made before the code was read closely. Each
is corrected here, at the site, with the reasoning that overturns it.

**1. The three-bucket classification assumed axis-0 blocking.** Read that way,
several sites look irreducible. Blocked along the axis the operation actually
runs over, they are not. The fixed alignment chain decomposes:

| step | coupling | blockable |
| --- | --- | --- |
| `abs` (FWHM only) | elementwise | Z |
| `apply_roi_3d` | crops Y/X only | Z |
| `apply_samy_shifts_to_volume` | independent per Z-layer; canvas width comes from the 1-D `samy` array | Z, given a globally computed pad |
| `interpolate_to_uniform_z` | `interp1d(kind="linear")` — each output Z reads only its two bracketing input layers | Z, with one row of forward context |
| `center_around_zero` | global `nanmean` / `nanmedian` | no — a genuine reduction |

**2. `slices` is chunkable; its `chunkable=False` is wrong.** The estimator says
"alignment is a whole-volume operation". Sampling is
`map_coordinates(prep["data"], coords, order=1, ...)` (`slices.py:622`), so each
output sample depends on the eight voxels bracketing it — two in Z. A Z-block
plus one row of forward context computes exactly those samples whose Z falls
inside it and scatters them into the output image, which is a small 2-D array.
One pass over Z serves every requested plane at once.

**3. `matched` is chunkable; its `chunkable=False` is wrong.** The estimator says
"an exact median needs the whole stack". It needs the whole stack *along the
frame axis only*. `np.nanmedian(stack, axis=0)` over a `(frames, Y, X)` stack is
exact on any in-plane sub-block: `stack[:, y0:y1, :]` yields the identical
answer for those rows. `matched` blocks in-plane rather than along axis 0.

**4. `strain`'s detrend is not a whole-volume operation.** The parent spec named
it, with the percentile `auto_clim` calls, as the likely bucket-3 candidate.
`_detrend_ccmth` (`strain.py:540`) loads one layer's `ccmth` map, detrends that
2-D map, and ROI-crops it. "Detrend before ROI" is a per-layer ordering
constraint, not a volume-wide one. `strain` has no irreducible step at all.

**Consequence:** no *transform* in phase 5 is irreducibly whole-array. The only
irreducible work left is the centring statistic, and only for one of its three
settings — see "The centring statistic" below, which is where `scratch_array`
earns its use. The escalation ladder in `advice.py` is unchanged.

## Decisions taken during brainstorming

1. **Done means a memory-capped STO2 run**, not merely converted code. Every
   converted stage completes the real dataset under an enforced ceiling with
   bit-identical output. The known failure mode of phases 1–4 was estimators
   that modelled peaks wrongly and needed three fix waves; a `budget_bytes`
   parameter alone would not have caught them.
2. **The cap is verified by measuring the child's peak RSS.** `dfxm/runner.py`
   already spawns each stage in a child process and `psutil` is already a
   dependency, so the harness samples the child's RSS and asserts the observed
   peak. This measures the thing itself rather than the model of it.
3. **Compensated summation everywhere, accepting the drift.** `center_around_zero`
   uses `np.nanmean`, whose pairwise order is budget-dependent; the compensated
   replacement differs by roughly 1 ulp, and the offset is subtracted from every
   voxel. Products made before this change therefore reproduce to ~1 ulp, not
   bit-for-bit. One definition of the answer on every machine is the promise the
   project exists to make; the alternative — numpy in-core, compensated when
   chunked — reintroduces exactly the laptop-versus-workstation divergence the
   project set out to eliminate. Recorded in the docs and the output attributes.
4. **Percentile colour limits stay exact.** A deterministic strided subsample
   would be budget-independent and one-pass, but it shifts clim values, so
   re-running an old plot would give visibly different colour scaling. The
   streaming quantile is exact and returns the number `np.percentile` returns
   today.
5. **The 3-D viewer decimates on read and says so.** `gui/viewers.py:53` is a
   display path, not a data product; VTK uploads the whole array, so streaming
   cannot help it and a disk-backed array only trades an out-of-memory kill for
   a very slow one. Spec decision #3's "never coarser" rule governs analysis
   stages, and `advise_3d` already coarsens here for GL reasons.
6. **Lifecycle fixes land before the streaming machinery**, and STO2 is
   re-measured between them. A site brought under the floor by its lifecycle fix
   alone does not get a conversion built on faith.

## Architecture

### `volumeio.py` — block context, a second axis, three reductions

```python
def iter_blocks(dset, *, budget_bytes, axis=0)        # axis=1 is new
def iter_with_context(blocks, *, trailing=1)
    # -> Iterator[tuple[slice, np.ndarray, slice]]

def stream_mean(blocks) -> float                      # Neumaier sum / finite count
def stream_minmax(blocks) -> tuple[float, float]
def stream_quantile(blocks_factory, q) -> float       # exact, three passes
```

`iter_with_context` re-yields a stream of blocks with each block carrying the
first *trailing* rows of its successor, plus the slice identifying the block's
own interior within the enlarged window — so a consumer writes `window[interior]`
and never redoes the arithmetic. The final block gets no context, which is
correct: it ends where the source ends, so its edge behaviour must match the
source's.

This is deliberately a function over a *block stream* rather than a widened read
window on `iter_blocks`. The operations needing context look **forward** — both
linear interpolation and `map_coordinates(order=1)` read the row after the one
they land on — so a symmetric halo would read a row nothing uses. More
importantly, the blocks needing context are frequently *generated* rather than
stored: an aligned volume that is never materialised cannot be re-indexed to
widen a read. One implementation serves an HDF5 dataset and a generator alike.

`axis` gains support for `axis=1`, which `matched` needs. The existing
`ValueError("only axis=0 blocking is supported")` is replaced, not worked around.

The three reductions take **an iterable of arrays**, not a dataset. This matters:
the statistics phase 5 actually needs are over *aligned* blocks, which are
generated and never materialised, not over anything on disk. A dataset is just
one thing you can iterate, so the dataset case stays a one-liner at the call
site.

`stream_quantile` is exact in three passes: min/max and finite count; a histogram
locating the bin that straddles the target rank; then exact selection among the
values in that one bin, which is a small array. It takes a *factory* rather than
an iterator because it must re-traverse. Exactness keeps every existing figure's
colour limits numerically identical, and it is budget-independent by
construction — which bin a value falls into does not depend on how the data was
blocked.

`stream_mean` layers on the existing `neumaier_sum` continuable state.

### `alignment.py` — one streaming entry point, two compatible parameters

CLAUDE.md requires one alignment. The streaming path therefore calls the
existing step functions rather than reimplementing them — which needs two small,
backward-compatible parameter additions, because two of the steps currently
derive global quantities from whatever slice of the motor arrays they are given:

- `apply_samy_shifts_to_volume(..., pad=None)` — `None` computes `pad_left` /
  `pad_right` from `samy` exactly as today. The streamed path passes the
  globally computed pad, because a per-block `samy` slice would otherwise
  produce a different canvas width for every block. This is the sharpest trap in
  the conversion and is closed by construction.
- `interpolate_to_uniform_z(..., z_uniform=None)` — `None` derives the grid from
  `samz` exactly as today. The streamed path passes the global grid, for the
  same reason.

Both defaults preserve current behaviour byte-for-byte, so no existing caller
changes.

```python
@dataclass
class StreamedAlignment:
    shape: tuple[int, int, int]     # aligned output shape, known before any read
    dtype: np.dtype
    z_uniform_um: np.ndarray
    scale_z_um: float
    pad_left: int
    center_offset: float            # 0.0 when center_method is None
    blocks: Iterator[tuple[slice, np.ndarray]]

def align_volume_streamed(dset, samy, samz, *, scale_x, samy_direction=1,
                          roi_x=None, roi_y=None, take_abs=False,
                          center_method=None, budget_bytes) -> StreamedAlignment
```

Every field except `blocks` is derivable from the small 1-D motor arrays and the
dataset's shape, so a writer can size its output before a voxel is read. The one
exception is `center_offset`, which is computed eagerly when `center_method` is
set — see below.

`align_volume` remains, reimplemented as the in-core façade that drains the
generator into one array. Every existing caller keeps working unchanged, and
there is still exactly one implementation of the arithmetic.

### The centring statistic

`center_method` is a user-facing parameter, not an internal constant, and it
offers **mean**, **median**, and — in `slices` only — **midrange**. The three
have very different streaming costs, and the statistic must be taken over the
*aligned* volume (NaN-padded canvas, interpolated Z grid), not over the source,
so it cannot be precomputed from the file:

| method | reduction | cost |
| --- | --- | --- |
| `mean` | `stream_mean` | one extra alignment pass |
| `midrange` | `stream_minmax` | one extra alignment pass |
| `median` | `stream_quantile(q=50)` | three extra alignment passes |

Three extra passes through the full alignment chain is the one place in phase 5
where the honest cost is unacceptable. So when `center_method` is `median` **and**
the plan is not in-core, the aligned blocks are written once to a `scratch_array`
and the quantile's three passes read from that instead — one alignment pass plus
three cheap disk reads. This is the genuine use for the disk-backed path built in
Task 11 of phases 1–4: not because the work is irreducibly whole-array, but
because re-deriving it is far more expensive than caching it.

The `midrange` path exists only in `slices` and is handled where `slices` handles
it today (`slices.py:764`), not promoted into `alignment.py`.

### Per-site conversion

| site | family | conversion |
| --- | --- | --- |
| `strain.py:369`, `:792` | collect + `np.stack` | preallocate the output dataset, write `out[z]` per layer |
| `mosaicity.py:399`, `:521` | collect + `np.stack` | same, across the four chi/mu × com/fwhm datasets |
| `visualize.py:497`, `:503` | all-fields dict, then align | per-field lifecycle, then `align_volume_streamed` |
| `paraview.py:643`, `:649` | all-fields dict, then align | same; `write_piece_vti` consumes Z-blocks directly |
| `slices.py:737`, `:753` | previous `prep` held live | free before rebinding, then Z-blocked gather |
| `matched.py:270`, `:274` | median over frames | in-plane blocking, `chunkable=True` |
| `rocking.py:1039` | global percentile | `stream_quantile` |
| `dfxm/viewer_jobs.py:20` | render load (video export child) | decimate on read, by the same policy as the viewer |
| `gui/viewers.py:53` | render load (on screen) | decimate on read; drop the two redundant full-size copies |

**Thirteen sites in all:** the twelve `[:]` reads named in the parent spec —
`viewer_jobs.py:20`, `visualize.py:457`/`:463`, `slices.py:735`/`:751`,
`paraview.py:595`/`:601`, `strain.py:368`, `mosaicity.py:398`,
`rocking.py:985`, `matched.py:268`/`:272`, at their current line numbers above —
plus `gui/viewers.py:53`, which this design found. The extra line numbers on the
`strain` and `mosaicity` rows (`:792`, `:521`) are the `np.stack` that the same
conversion removes, not additional sites. A later re-audit should check all
thirteen rows against this table.

### Recorded exemption — the no-motor path in `visualize` and `paraview`

Added during the wave-3 review, so that an inconsistency between three stages is
a decision with a reason rather than an accident of which reviewer noticed it.

Each of the three alignment stages has a **no-motor** fallback, taken when
`find_matching_folders` matches nothing and `extract_motor_positions` therefore
returns `samy` and `samz` empty together. It cannot go through
`align_volume_streamed`, which always interpolates: resampling a NaN-bearing
volume onto its own Z nodes is not the identity, because scipy's linear
interpolant reads the value *below* each node and spreads every NaN one layer
down (measured in `paraview`: 1299 of 9360 voxels changed their `valid_mask`).

The three do not treat it alike, and deliberately:

| stage | no-motor site | bounded? |
| --- | --- | --- |
| `slices` | `_unaligned_blocks` | **yes** — the chain (`abs` → ROI → samy X-shift) as a Z-block factory, pad and reference samy computed once for the whole volume, verified over 12 combinations of pad sign, `samy_direction` and `len(samy)` against `nz` |
| `visualize` | `_align_streamed`, `len(samz) == 0` | no — `dset[:]`, handed on as one covering block |
| `paraview` | `_unaligned_field` | no — `dset[:]`, plus the whole-volume `center_around_zero` |

`slices` is bounded because its Z-blocked plane gather needed a block factory on
*every* path anyway; giving the no-motor chain one cost nothing extra. In
`visualize` and `paraview` the same conversion would be new machinery built for
this path alone — and in `paraview` it also needs the centring statistic
streamed over the unaligned blocks, which is a second reduction with its own
budget accounting.

**The cost of the exemption, stated plainly.** The trigger is a typo'd
`mosa_pattern` or `raw_root`, so on the 8 GB target machine a typo becomes an
OOM kill rather than an unaligned result — a *failure*, which sits against this
phase's "slower, never failed". Two things make that acceptable rather than
merely tolerated:

- A run that reaches this path produces an **unaligned** volume, which is not a
  usable product. The path exists to keep a misconfigured run from crashing, not
  to serve a legitimate large run; nothing is lost by it that was worth having.
- The cost is still *predicted*. Both stages' `estimate()` deliberately
  over-predict by pricing the pre-phase whole-array form, so `advice.plan_run`
  compares the whole-volume cost against the machine's headroom and reports it
  before the run starts. The user is warned; they simply are not saved.

**Bounding them is available if wanted**, and `visualize` is the cheap half: its
no-motor chain is `ROI → samy X-shift`, the same per-layer steps `slices`
already blocks, so `_unaligned_blocks` is close to liftable as-is. `paraview`
additionally wants `stream_mean` / `stream_quantile` over the unaligned blocks
for its centring. Neither was done here because the phase's budget is better
spent on paths a correct run can reach.

Three families, and they want different things. `strain` and `mosaicity` are not
alignment stages at all — they *build* volumes from per-layer maps, and their fix
is structural: preallocate and write incrementally, no blocking machinery
involved. `visualize`, `paraview` and `slices` hold more than they need before
they align, and their fix is a lifecycle change followed by streamed alignment.
`rocking`, `matched` and the viewer are each a single localised change.

The consumers cooperate with Z-blocking rather than fighting it: paraview's
`compute_piece_extents_z` already splits its `.vti` output along Z, and
visualize renders per-layer figures.

`gui/viewers.py:53` and `dfxm/viewer_jobs.py:20` are the two sites outside the
stage machinery — the viewer load runs in the GUI process and the rotation-video
export runs in a child job, and neither sees `StageSpec.estimate` or `plan_run`.
They are one conversion, not two: the stride policy lives in `dfxm/common/` so
both call it (`dfxm/` may not import `gui/`), and the video is coarsened by the
same rule as the view — though each measures headroom in its own process, so the
factor can differ. `gui/viewers.py:53` reads the
volume, upcasts with `.astype(float)`, then builds `vol[np.isfinite(vol)]` for
the percentile clim, a second full-size copy. It reads with a stride when the
volume exceeds headroom, using the larger of the RAM-derived factor and the
`advise_3d` GL factor, labels the decimation in the viewer, and computes its
clim without the redundant copy.

## Data flow

Unchanged from the parent spec. Phase 5 populates the last edge of it — the one
from `RunPlan.budget_bytes` into the stage bodies:

```
RunPlan.budget_bytes ──> align_volume_streamed ──> (z_slice, block) ──> writer
                    └──> stream_quantile / stream_mean ──> clim, centre offset
```

Stages drive `volumeio` with `budget_bytes` alone. `RunPlan.chunk_layers` stays
display-only, as phases 1–4 established.

## Error handling

No new failure modes, and nothing in this phase refuses to run for lack of RAM.
One existing path becomes reachable that was not before: a chunked run with
`center_method="median"` caches its aligned blocks to scratch disk, so the
insufficient-disk `StageUserError` added in phases 1–4 can now fire for a stage
that is not otherwise disk-backed. It must be predicted by the estimator and
reported before work starts, per the existing convention — bytes needed versus
bytes free, with a hint.

Degrade decisions remain not-errors: they go to the log, the result notes and the
output attributes. The viewer's decimation factor is shown in the viewer.

## Testing

- **Budget-independence per converted site**, through the existing
  `tests/equivalence.py` harness: one synthetic volume through in-core and
  chunked paths, asserting bit-identical output.
- **`tests/peak_rss.py`** — a new harness, not a pytest file, sampling the
  `runner.py` child's RSS and returning the observed peak. Converted stages
  assert the peak stayed under the budget on a synthetic volume sized to force
  streaming.
- **Fast-path guard** — a volume that comfortably fits still takes the in-core
  path, so a large machine does not pay for a small one's safety.
- **Alignment parity** — `align_volume` and `align_volume_streamed` agree
  bit-for-bit on a synthetic volume at several budgets, including a budget below
  one layer.
- **The `pad` and `z_uniform` traps, tested directly** — a streamed run whose
  per-block `samy` slice would imply a different canvas must still produce the
  global canvas.
- **All three centring methods**, since each takes a different route: `mean`
  through `stream_mean`, `midrange` through `stream_minmax`, `median` through
  `stream_quantile` and its scratch cache. The median path is the only one that
  touches disk, so it is also where scratch cleanup on an exception is asserted.
- **The STO2 canary** — the real dataset, memory-capped, output diffed against
  an unconstrained run. Hand-run and recorded, per wave 4.

## Implementation sequencing

Thirteen tasks in four waves. Docs are not a separate task: CLAUDE.md's
same-change contract means each task carries its own `Usage.md` and
`Codebase.md` edits.

**Wave 1 — lifecycle fixes. No new machinery, no numeric change.**

1. `strain` and `mosaicity` incremental layer writes (one task; identical pattern)
2. `visualize` and `paraview` per-field lifecycle — stop holding all-fields dicts
3. `slices` — free the previous prepared volume before rebinding
4. `gui/viewers.py` — decimate on read, drop the redundant copies

**Re-measure gate.** Re-run the STO2 figures. A site now under the floor does not
get a wave-3 conversion.

**Wave 2 — machinery.**

5. `volumeio` — `iter_with_context`, `axis=1` support, `stream_mean`,
   `stream_minmax`
6. `volumeio.stream_quantile` — exact streaming order statistic
7. `alignment` — `pad` / `z_uniform` parameters, `align_volume_streamed`,
   `align_volume` as façade
8. `tests/peak_rss.py`

**Wave 3 — conversions.**

9. `paraview` — streamed alignment into the Z-piece writer
10. `visualize` — streamed alignment
11. `slices` — Z-blocked `map_coordinates` gather, `chunkable=True`
12. `rocking` — `stream_quantile`; `matched` — in-plane blocked median,
    `chunkable=True`

**Wave 4 — verification.**

13. Re-verify all seven estimators against the rewritten `run()` bodies; run the
    memory-capped STO2 canary; record the figures.

Task 13 is not a formality. Every peak model in the table at the top of this
document describes code that waves 1–3 rewrite, so every one of them is stale by
the time wave 4 starts. Phases 1–4 shipped estimators whose peak models were
wrong and needed three fix waves to correct; the same mistake is available here,
and this task is where it gets caught.

## Risks

- **The `samy` pad trap.** A per-block `samy` slice implies a different canvas
  width. Closed by the explicit `pad` parameter, guarded by a test.
- **Halo arithmetic off by one.** The mapping from an output Z range to the input
  layers its interpolation brackets is where a subtle wrongness would hide, and
  it would show up as NaN seams at block boundaries. The parity test at several
  budgets, including one below a single layer, is the guard.
- **`ndi_shift` row independence is assumed.** `order=1` does no prefiltering, so
  a shift purely along X should leave each row depending only on itself. The
  conversion depends on this; it is asserted by test rather than by argument.
- **Estimator staleness.** Addressed by task 13, above.
- **The median-centring scratch cache adds a disk dependency** to a path that had
  none, which means `RunPlan.blocked` — the insufficient-disk error — becomes
  reachable from a stage that merely chose `center_method="median"`. The
  estimator must account for the cache so the block is predicted before the run
  starts rather than raised halfway through it. This is the phase's one widening
  of a phase-1–4 interface: `CostEstimate` gains `scratch_bytes: int = 0`, which
  `plan_run` folds into the check it already makes against `profile.disk_free`.
  Defaulting to zero leaves every other estimator unaffected.
- **Scope.** Thirteen tasks touching seven stage modules and the shared
  alignment. Smaller than the parent spec's "eight stages, the bulk of the work"
  estimate for the conversions themselves, but wider, because it also rewrites
  the lifecycle of five of them.
- **The ~1 ulp drift** against previously produced products, accepted in decision
  3, is a one-way door. It is recorded in the output attributes so a file's
  convention is discoverable from the file.

## Documentation contract

`Usage.md` gains a note per affected stage on what a memory-capped run does
differently — it is slower, never coarser, and the product is identical — plus
the viewer's decimation behaviour, which *is* user-visible coarsening and must
say so. `Codebase.md` gains the new `volumeio` and `alignment` functions and the
`StreamedAlignment` dataclass. Both are updated in the same change as the code,
per CLAUDE.md.
