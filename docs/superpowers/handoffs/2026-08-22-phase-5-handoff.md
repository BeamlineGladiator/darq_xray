# Phase 5 handoff — bounded-memory stage conversions

**Written** 2026-08-22 · **master HEAD** `753a1c4` · **no remote** (skip pull/push/PR)

> Companion docs: spec `docs/superpowers/specs/2026-08-20-phase-5-stage-conversions-design.md`,
> plan `docs/superpowers/plans/2026-08-20-phase-5-stage-conversions.md`.
> The full audit ledger is `.superpowers/sdd/2026-08-20-phase-5-stage-conversions/progress.md`
> — **git-ignored, so `git clean -fdx` destroys it**; this file is the durable summary.

---

## 1. State right now

| | |
|---|---|
| master HEAD | `753a1c4` |
| Verified | pytest **1435 passed / 13 skipped / 17 deselected**, ruff + format clean, `gui_smoke` **41/41 on two consecutive runs** |
| Working tree | clean (except 4 pre-existing untracked `test_figure_builder*` files, which predate this project — ask before deleting) |
| Rollback for the **whole** phase | `git reset --hard d59afbd` |
| Branch `phase-5-stage-conversions` | merged at `a424b1f`, kept as a safety net; deletable |

Commit trail:

```
753a1c4  fix: report the scratch a run actually uses, and test the one that does
a7be104  feat: say when a 3-D product overrides the streaming budget
9605d15  fix: re-verify the estimators against the rewritten stages, and price the spill
a424b1f  Merge phase 5 waves 1-3: bounded-memory stage conversions   <- 43 commits
d59afbd  (phase-5 base — the rollback point)
```

**Verify before trusting any of the above:**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py   # the deselect is mandatory on this box
ruff check . && ruff format --check .
DISPLAY= python3 -u tests/gui_smoke.py                       # see the [41] warning in §5
```

---

## 2. What the phase delivered

Stages now run on a **two-rung ladder** — in-core when the volume fits the budget,
streaming when it does not — with products **bit-identical either way**, so a laptop
and a workstation emit the same publishable figures.

- **Waves 1–3** (in `a424b1f`): lifecycle fixes (incremental stacked-volume writes,
  per-field mosaicity loads, prepared-volume release, viewer decimation); the streaming
  machinery (`iter_with_context`, `axis=1` blocking, exact `stream_quantile`,
  `align_volume_streamed`, vectorised Neumaier, working-set-bounded budgets,
  `tests/peak_rss.py`); and the conversions of paraview, visualize, slices, rocking, matched.
- **Task 13** (`9605d15`, `a7be104`, `753a1c4`): estimator re-verification against measurement,
  `CostEstimate.scratch_bytes` + `plan_run`'s chunked-path disk check,
  `alignment.roi_shape` / `aligned_extent` / `aligned_elems_for_params`,
  `raster.motor_positions_for_estimate`.

**The headline result, measured not argued:** capped at an 8 GB machine's 3.60 GiB budget
on the real STO2 dataset, **445 files came out bit-identical** across visualize, slices and paraview.

### Measured figures (real STO2, master, child peak RSS)

| stage | estimate | measured | capped peak (cap 3.60 GiB) | products identical |
|---|---|---|---|---|
| visualize | 10.51 GiB | 8.83 | **4.80 — over cap** | yes (395 files) |
| slices | 6.57 | 7.41 (see currency note) | 2.33 | yes (15) |
| paraview | 17.07 | 10.49 | 2.18 | yes (35) |
| strain | 2.627 | 0.508 | not run capped | — |
| mosaicity | 6.566 | 0.181 | not run capped | — |

**Currency note — do not misread the slices row.** `peak_bytes` counts Python allocations
(tracemalloc); peak RSS adds the process image and allocator slack, related *additively* as
`RSS ≈ RSS_FLOOR_BYTES + MARGINAL_RSS_PER_TRACED_BYTE × traced`. Against slices' own 0.25 GiB
floor and the real constant **1.3**, the model predicts `0.25 + 1.3 × 6.57 = 8.79 GiB` for a
measured 7.41 — a **19% margin**. Comparing the two numbers directly is a conflation this
project has now made twice. **No stage under-predicts.**

---

## 3. What remains — pick up here

### 3a. Four estimators still say "pending recalibration" *(safe; tightening, not fixing)*

`strain`, `mosaicity`, `rocking`, `matched`. Measured over-prediction: **strain 5.2×**
(2.627 est vs 0.508 actual), **mosaicity 36×** (6.566 vs 0.181). Their models still describe
the deleted accumulate-then-`np.stack` code.

**Deliberately not rewritten, and the reason matters:** `strain.estimate`'s own docstring warns
that a naive per-layer model **under**-predicts, because with `save_plots` on each layer
rasterises figures whose Agg canvases are sized by *figure inches × dpi*, not by the data — so
the high-water mark is not in `H*W*8` units at all. Flipping a safe over-estimate into the
dangerous direction is the exact error already made once in this project (a model computing
4.75 GiB against a 10.14 GiB reality).

**To do it properly:** measure the plotting term in isolation (plots on vs off, across two or
three dpi/figure-size combinations — it is data-independent, so a synthetic fixture transfers),
then model `k × layer_elems × 8 + plot_term + one-layer gzip buffer`. Validate that the result
still over-predicts on STO2 before committing.

### 3b. visualize exceeds an 8 GB machine's cap — **your decision**

**4.80 GiB against a 3.60 GiB cap**, whenever `save_topview` or `save_rotation` is on. Both
default on, and **both are on in your saved STO2 form**. This is *not* the streaming failing —
the products are bit-identical. `_process_dataset` materialises the whole aligned volume because
the 3-D products upload the grid to VTK in one piece.

The run now **says so** (a per-dataset note naming the volume size and the two toggles). It was
not bounded, because bounding means decimating a *saved product*, and you scoped the phase's one
user-visible coarsening to the interactive viewer. The reviewer agreed the trade is correct.

Options if you want it bounded: decimate the saved 3-D products the way the viewer does
(`volumeio.display_decimation`), or leave it and rely on the note.

### 3c. Smaller carried items

- **`plan_run` has no production caller.** `RunPlan.blocked` has no consumer and
  `StageSpec.estimator()` is never called from `gui/`, so the cost estimate and the new
  scratch-disk check are computed but never shown to a user. Surfacing them was always
  "phase 6" — but it means the estimator work is currently invisible in the app.
- **`plan_run` compares currencies.** It checks a traced-currency `peak_bytes` against an
  RSS-currency `headroom_bytes`, which `working_set_budget_bytes`' own docstring warns against.
  Ruled *not* to change: the models over-predict enough to absorb it, and converting would make
  every stage stream more and run slower. Documented and pinned instead. Revisit only with
  measurement.
- **The no-motor path** stays whole-array in `visualize` and `paraview` (slices bounds it) —
  a **recorded exemption** in the design spec, not an oversight. Trigger is a typo'd
  `mosa_pattern`, i.e. a misconfiguration.
- **paraview over-predicts 1.6×** — safe, but the loosest of the three; biases toward streaming.

---

## 4. Hazards — read before running anything on real data

1. **STO2 lives on the external SSD**, not `/mnt/data`:
   `/path/to/data/ESRF/ma6778/id03/20251029/` (`PROCESSED_DATA/STO2_overnight` =
   76 mosa + 76 strain layer folders; `RAW_DATA` alongside). `/mnt/data` holds only derived products.
2. **`strain` and `mosaicity` write their stacked volume into the BATCH ROOT.** Pointing
   `root_folder` at the real data directory **overwrites** `stacked_volumes.h5` /
   `stacked_strain_volumes.h5`. Always use a symlink tree of layer folders — there is one at
   `/mnt/data/task13_canary/{in_strain,in_mosa}`.
3. **Calibration is not persisted.** `pixel_size_x_um` / `pixel_size_y_um` are excluded from
   saved form state. STO2's are **0.151733 / 0.387584**. Real params otherwise live in
   `~/.config/dfxm/pipeline.conf`, section `[formState]`, keys `sto2_overnight\<stage>`.
4. **`paraview` rejects `center_method="midrange"`** (visualize/slices only), so visualize's
   saved form cannot be reused for paraview verbatim.
5. **Any script driving `StageRunner` needs `if __name__ == "__main__":`** — without it, spawn
   re-imports the module and re-runs everything. This cost a wasted measurement run.

---

## 5. Environment quirks

- **`--deselect tests/test_gui_viewer3d.py` is mandatory** — in-process Qt GL segfaults on this box.
- **`gui_smoke` step `[41]` is genuinely intermittent** — it aborted in roughly half of all runs
  here **and aborts on an unmodified tree too**. Run the smoke 2–3 times before attributing a
  failure to your diff. (I concluded a regression from two data points once this session and was
  wrong.) Run it as `DISPLAY= python3 -u tests/gui_smoke.py`; ambient `:10` causes X BadWindow in
  the runner child.
- Known flake: `tests/test_gui_figure_builder.py::test_export_now_writes_files` (timing).

**Scratch left behind:** `/mnt/data/task13_canary*` — **16 GB**, the evidence behind the
bit-identity comparison. Delete whenever you want the space.

**Measurement harnesses** (not committed): `~/.claude/jobs/277b2e92/tmp/` —
`canary.py` (estimate vs measured, uncapped), `canary_capped.py` (capped + product comparison),
`canary_folders.py` (strain/mosaicity via the symlink trees).

---

## 6. The one process lesson worth carrying

**Twenty checks in this project have been found to have stopped checking what they name** — and
the 19th and 20th were each *authored by the fix for the one before*. None was visible by reading;
every one needed a mutation. Two shapes recur:

- a test whose fixture drifts out of the region it claims to cover (a measurement test sampling
  only the flat part of a cost curve; a scratch test landing on the fallback return instead of
  the main one);
- a constant or branch that nothing pins, so deleting the feature leaves the suite green.

**So: for every test you add or change, run the mutation that should break it and confirm it
does — and assert the precondition that keeps the fixture inside the region the test names.**
