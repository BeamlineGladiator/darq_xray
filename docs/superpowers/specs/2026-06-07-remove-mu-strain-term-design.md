# Remove the mu-term strain method (full purge)

Date: 2026-06-07
Status: Approved (design); pending implementation plan
Branch: `remove-mu-strain-term`

## Summary

Make the `strain` stage compute axial strain **unconditionally from `ccmth`
only**:

```
ε = cot(ccmth_ref) · Δccmth
```

Delete the alternative `ccmth_mu` method and the `−cot(mu_ref)·Δmu` term it
adds, and purge the now-orphaned `mu_ref_deg` / `mu_com_path` calibration fields
from the strain stage, the `Experiment` schema, the shipped preset, the GUI
bindings, the tests, and the docs.

## Motivation (physics)

The `ccmth_mu` method computed `ε = cot(ccmth_ref)·Δccmth − cot(mu_ref)·Δmu`.
The second term tried to "correct" axial strain using the `mu`
(incidence/goniometer) angle. That is physically wrong for DFXM: axial strain
(Δd/d) is read from the Bragg angle (`ccmth`), whereas `mu`/`chi` variation is
**misorientation/mosaicity** — a different physical quantity. Folding the mu
term into ε contaminates the strain map with rigid crystal tilt. Misorientation
already has its correct home in the **mosaicity stage** (which reads the `mu`
Center-of-mass map) and the misorientation display (which centres from the data
via `center_method` ∈ {midrange, mean, median}).

Removing the mu term also dissolves the long-standing `mu_ref_deg` calibration
ambiguity (11.5015 vs the legacy 11.2491): once `mu` never enters strain, the
reference value is irrelevant to every product the pipeline makes.

## Scope

In scope: the `ccmth_mu` method, the mu strain term, and the orphaned
`mu_ref_deg` + `mu_com_path` Experiment-level calibration fields.

### The `mu_com_path` boundary (important)

`mu_com_path` appears in two unrelated roles:

- **Strain stage** read of the mu CoM map — REMOVE.
- **Mosaicity stage**'s own `mu_com_path` param (`mosaicity.py`) — the
  legitimate misorientation read — **KEEP, untouched.**

`bindings.py` feeds `exp.mu_com_path` **only** to the strain override; the
mosaicity override uses the mosaicity stage's own default. Therefore removing
`Experiment.mu_com_path` is safe and does not affect mosaicity.

### Out of scope (left untouched)

`dfxm/stages/mosaicity.py`, `dfxm/common/alignment.py::center_around_zero`, and
the `visualize` / `paraview` / `slices` data-driven centering — all the
legitimate misorientation/centering machinery.

## Design decision: drop the `method` param

With a single behavior remaining, the `method` enum param and the `--method`
CLI flag are deleted rather than kept as a one-choice enum. `StrainResult.method`
and the GUI `method:` summary line are removed too. Strain is unconditionally
ccmth-only.

## Detailed change list

### Core — `dfxm/stages/strain.py`
- `compute_strain(ccmth_map_deg, ccmth_ref_deg) -> np.ndarray` — drop `mu_map`,
  `mu_ref`, and the `(strain, ccmth_term, mu_term)` 3-tuple return.
- `process_maps_file` — drop `method` / `mu_com_path` / `mu_ref_deg` params,
  the mu load + mu ROI crop, and the `_save_contributions` plot. Delete
  `_save_contributions` and its call (a two-term contributions panel is
  meaningless with one term).
- `run()` and stacked-volume attrs — drop `method` and `mu_ref_deg`;
  description → `"Stacked 3D strain volume (cot, ccmth-only)"`.
- `STAGE` spec — remove `method`, `mu_com_path`, `mu_ref_deg` params.
- `_main` — drop `--method` and `--mu-ref`.
- Module docstring — rewrite (no "two methods").

### Config / GUI
- `config/models.py` — remove `mu_ref_deg` and `mu_com_path` from the
  `Experiment` dataclass **and** `EXPERIMENT_SCHEMA`, preserving lock-step field
  order (enforced by `test_schema_matches_dataclass`).
- `gui/bindings.py` — drop `mu_com_path=` / `mu_ref_deg=` from the strain
  override.
- `gui/stage_view.py` — drop the `method:` summary line in `_summarize`.
- `gui/experiment_panel.py` and `config/presets.py` docstrings — replace the
  "mu_ref discrepancy" example with a neutral one.

### Preset — `experiments/STO2_overnight.yaml`
- Delete the `mu_ref_deg` line, the `mu_com_path` line, the
  `CALIBRATION DISCREPANCY — mu_ref_deg` header block, and the `notes:` caveat.

### Tests
- `test_stage_strain.py` — delete the `ccmth_mu`-vs-legacy test; update the
  `ccmth_only` test to the new single-array return; convert the `run` and
  detrend-before-ROI tests to ccmth-only (drop `method` / `mu_ref_deg` /
  `contributions` assertions). `test_detrend_matches_legacy` stays (self-skips).
- `test_config.py` — drop `mu_ref_deg` from the flagged-calibration set; delete
  the `== 11.5015` and `notes contain 11.2491` assertions.
- `gui_smoke.py` — drop the mu_ref / caveat assertions; remove `method` /
  `mu_ref_deg` from the strain form values.

### Docs (same-change contract)
- `docs/Usage.md` — remove the mu_ref caveat callout; strain section → ccmth-only
  formula; drop the `method` / `mu` / `mu_ref` rows.
- `docs/Codebase.md` — drop `mu_ref_deg` from the `Experiment` calibration list;
  fix the `compute_strain(...)` signature line; update the `experiment_panel`,
  `test_config`, and preset descriptions that cite the mu_ref caveat.

## Verification

- `python3 -m pytest -q` passes.
- `ruff check .` clean.
- `python3 -m dfxm.stages.strain -h` shows no `--method` / `--mu-ref`.
- A scan for residual references: `grep -rn "mu_ref\|ccmth_mu\|11.5015\|11.2491"`
  returns nothing outside intentional history/spec docs.

## Behavioural impact

Strain maps produced via the (former) default `ccmth_mu` method will change:
the mu contribution is gone. For datasets that were already run `ccmth_only`,
output is unchanged. This is the intended correction, not a regression.
