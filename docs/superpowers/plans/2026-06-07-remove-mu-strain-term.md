# Remove the mu-term strain method (full purge) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `strain` stage compute axial strain unconditionally from `ccmth` only (`ε = cot(ccmth_ref)·Δccmth`), deleting the `ccmth_mu` method, its mu term, and the orphaned `mu_ref_deg` / `mu_com_path` calibration fields — while leaving the mosaicity misorientation read of `mu` untouched.

**Architecture:** A removal refactor across four areas, each its own commit kept green: (1) the strain stage + its GUI glue + tests, (2) the `Experiment` config + preset + config tests, (3) the two Obsidian docs, (4) final verification. The `ccmth_only` path already exists, so the numeric core barely changes — most work is deleting the mu branch and its parameters.

**Tech Stack:** Python 3.10, numpy, h5py, scipy, PySide6 (GUI, not exercised by `pytest -q`), ruff, pytest. Run from repo root `dfxm_pipeline` on branch `remove-mu-strain-term`.

**Reference spec:** `docs/superpowers/specs/2026-06-07-remove-mu-strain-term-design.md`

---

## File structure / blast radius

- `dfxm/stages/strain.py` — core math + stage (heaviest change).
- `gui/stage_view.py` — drop `method:` summary line; fix the `StrainResult` discriminator.
- `gui/bindings.py` — drop two lines from the strain override.
- `dfxm/config/models.py` — drop two `Experiment` fields + two schema entries.
- `experiments/STO2_overnight.yaml` — drop mu lines + discrepancy block + notes.
- `gui/experiment_panel.py`, `dfxm/config/presets.py` — neutralise a docstring example.
- `tests/test_stage_strain.py`, `tests/test_config.py`, `tests/gui_smoke.py` — update.
- `docs/Usage.md`, `docs/Codebase.md` — same-change doc contract.

**Untouched (intentionally):** `dfxm/stages/mosaicity.py`, `dfxm/common/alignment.py`, `dfxm/stages/visualize.py|paraview.py|slices.py`.

---

## Task 1: Strain stage → ccmth-only

**Files:**
- Modify: `dfxm/stages/strain.py`
- Modify: `gui/stage_view.py:281-283`
- Modify: `gui/bindings.py:81-83`
- Test: `tests/test_stage_strain.py` (rewrite)

- [ ] **Step 1: Rewrite the strain tests to expect ccmth-only**

Replace the entire contents of `tests/test_stage_strain.py` with:

```python
"""Tests for dfxm.stages.strain — ccmth-only axial strain (cot method): numeric
golden equivalence vs the legacy calc_axial_strain_v7_batch script (self-skips
when the legacy file is absent), an independent cot-formula check, and the
detrend-before-ROI ordering constraint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from dfxm.stages import strain as S

CCMTH_PATH = "/entry/ccmth/Center of mass/Center of mass"


def _legacy(modname: str):
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / f"{modname}.py").exists():
        pytest.skip(f"legacy {modname}.py not found")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return __import__(modname)


def _synthetic_ccmth(ny=40, nx=60, seed=1):
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(np.linspace(-3, 3, nx), np.linspace(-2, 2, ny))
    return (
        7.144
        + 0.002 * np.arctan(2 * X)
        + 0.001 * np.arctan(1.5 * Y)
        + 0.0001 * rng.standard_normal((ny, nx))
    )


def _write_maps(folder, ccmth):
    os.makedirs(folder, exist_ok=True)
    with h5py.File(os.path.join(folder, "maps.h5"), "w") as f:
        f.create_dataset(CCMTH_PATH, data=ccmth)


def test_detrend_matches_legacy():
    legacy = _legacy("calc_axial_strain_v7_batch")
    ccmth = _synthetic_ccmth()
    mine_dt, mine_surf = S.detrend_arctan_2d(ccmth.copy())
    leg_dt, leg_surf = legacy.detrend_arctan_2d(ccmth.copy())
    np.testing.assert_allclose(mine_dt, leg_dt, atol=1e-12)
    np.testing.assert_allclose(mine_surf, leg_surf, atol=1e-12)


def test_compute_strain_ccmth_only_matches_legacy_v7():
    legacy = _legacy("calc_axial_strain_v7_batch")
    ccmth = _synthetic_ccmth()
    dt, _ = S.detrend_arctan_2d(ccmth.copy())
    np.testing.assert_allclose(S.compute_strain(dt, 7.144), legacy.compute_strain(dt, 7.144), atol=1e-15)


def test_compute_strain_is_cot_ccmth():
    """compute_strain returns a single array = cot(ref)·(Δ in radians)."""
    ccmth = _synthetic_ccmth()
    dt, _ = S.detrend_arctan_2d(ccmth.copy())
    expected = S.cot(np.deg2rad(7.144)) * (np.deg2rad(dt) - np.deg2rad(7.144))
    out = S.compute_strain(dt, 7.144)
    assert out.shape == dt.shape
    np.testing.assert_allclose(out, expected, atol=1e-15)


def test_run_single_stacks_volume_and_writes_plots(tmp_path):
    ccmth = _synthetic_ccmth()
    folder = tmp_path / "layer__1"
    _write_maps(str(folder), ccmth)
    res = S.run(
        {
            "mode": "single",
            "input_folder": str(folder),
            "ccmth_ref_deg": 7.144,
            "output_dir": str(tmp_path / "out"),
        }
    )
    assert res.n_layers == 1 and res.volume_shape == (1, 40, 60)
    with h5py.File(res.stacked_path, "r") as f:
        vol = f["strain"][:]
        dt, _ = S.detrend_arctan_2d(ccmth.copy())
        np.testing.assert_allclose(vol[0], S.compute_strain(dt, 7.144), atol=1e-15)
    pngs = list((tmp_path / "out").glob("*.png"))
    assert any("strain" in p.name for p in pngs)
    assert not any("contributions" in p.name for p in pngs)


def test_run_batch_over_multiple_layers(tmp_path):
    ccmth = _synthetic_ccmth()
    root = tmp_path / "root"
    for name in ["layer__1", "layer__2"]:
        _write_maps(str(root / name), ccmth)
    res = S.run(
        {
            "mode": "batch",
            "root_folder": str(root),
            "folder_pattern": "layer__*",
            "ccmth_ref_deg": 7.144,
            "save_plots": False,
        }
    )
    assert res.n_layers == 2 and res.volume_shape == (2, 40, 60)


def test_detrend_runs_before_roi(tmp_path):
    """ROI must crop the detrended map, not detrend a pre-cropped map."""
    ccmth = _synthetic_ccmth()
    folder = tmp_path / "layer__1"
    _write_maps(str(folder), ccmth)
    roi = [5, 25, 10, 40]
    res = S.run(
        {
            "mode": "single",
            "input_folder": str(folder),
            "ccmth_ref_deg": 7.144,
            "roi": "5,25,10,40",
            "save_plots": False,
            "output_dir": str(tmp_path / "out"),
        }
    )
    dt_full, _ = S.detrend_arctan_2d(ccmth.copy())
    dt_crop = dt_full[roi[0] : roi[1], roi[2] : roi[3]]
    expected = S.compute_strain(dt_crop, 7.144)
    with h5py.File(res.stacked_path, "r") as f:
        np.testing.assert_allclose(f["strain"][0], expected, atol=1e-15)
        assert f["strain"].shape == (1, 20, 30)


def test_parse_helpers():
    assert S._parse_roi("") is None
    assert S._parse_roi("1,2,3,4") == [1, 2, 3, 4]
    with pytest.raises(ValueError):
        S._parse_roi("1,2,3")
    assert S._parse_float("") is None
    assert S._parse_float("0.5") == 0.5
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `cd dfxm_pipeline && python3 -m pytest tests/test_stage_strain.py -q`
Expected: FAIL — `compute_strain` still requires/returns the old 4-arg / 3-tuple form, so `test_compute_strain_is_cot_ccmth` and the `run` tests error (e.g. `KeyError: 'method'` inside `run`, or tuple/shape mismatch).

- [ ] **Step 3: Make `compute_strain` ccmth-only**

In `dfxm/stages/strain.py`, replace the whole `compute_strain` function (currently lines ~216-234) with:

```python
def compute_strain(
    ccmth_map_deg: np.ndarray,
    ccmth_ref_deg: float,
) -> np.ndarray:
    """Per-pixel axial strain (cot method), ccmth-only.

    ``ε = cot(ccmth_ref) · (ccmth − ccmth_ref)`` with angles converted to radians.
    """
    ccmth_rad = np.deg2rad(ccmth_map_deg)
    ccmth_ref_rad = np.deg2rad(ccmth_ref_deg)
    return cot(ccmth_ref_rad) * (ccmth_rad - ccmth_ref_rad)
```

- [ ] **Step 4: Delete the contributions plot helper**

In `dfxm/stages/strain.py`, delete the entire `_save_contributions(...)` function (currently lines ~305-339, from `def _save_contributions(` through its final `fig.savefig(...)`).

- [ ] **Step 5: Drop the `method: str` field from `StrainResult`**

In `dfxm/stages/strain.py`, change the `StrainResult` dataclass (lines ~146-153) from:

```python
@dataclass
class StrainResult:
    method: str
    stacked_path: str | None = None
```

to:

```python
@dataclass
class StrainResult:
    stacked_path: str | None = None
```

- [ ] **Step 6: Rewrite `process_maps_file` to ccmth-only**

In `dfxm/stages/strain.py`, replace the `process_maps_file` signature + body down to the `return strain, layer` (currently lines ~345-406) with:

```python
def process_maps_file(
    maps_path: str,
    name: str,
    *,
    ccmth_com_path: str,
    ccmth_ref_deg: float,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    roi: list | None,
    vlim: tuple[float | None, float | None],
    out_dir: str | None,
    save_plots: bool,
) -> tuple[np.ndarray, LayerResult]:
    """Compute the 2-D strain map for one maps.h5 and (optionally) save plots."""
    ccmth_map = load_map(maps_path, ccmth_com_path)

    # detrend ccmth on the FULL map, THEN crop ROI (order matters)
    ccmth_original = ccmth_map.copy()
    ccmth_map, surface = detrend_arctan_2d(ccmth_map)
    ccmth_map = apply_roi(ccmth_map, roi)
    surface = apply_roi(surface, roi)
    ccmth_original = apply_roi(ccmth_original, roi)

    strain = compute_strain(ccmth_map, ccmth_ref_deg)

    plots: list[str] = []
    if save_plots and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"{name}_strain.png")
        _save_strain_map(strain, pixel_size_x_um, pixel_size_y_um, roi, vlim, p)
        plots.append(p)
        ph = os.path.join(out_dir, f"{name}_hist.png")
        _save_histogram(strain, ph)
        plots.append(ph)
        pd = os.path.join(out_dir, f"{name}_detrend_diag.png")
        _save_detrend_diag(ccmth_original, ccmth_map, surface, pd)
        plots.append(pd)

    layer = LayerResult(
        name=name,
        shape=tuple(strain.shape),
        vmin=float(np.nanmin(strain)),
        vmax=float(np.nanmax(strain)),
        mean=float(np.nanmean(strain)),
        std=float(np.nanstd(strain)),
        plots=plots,
    )
    return strain, layer
```

- [ ] **Step 7: Update `run()` — drop method + mu plumbing**

In `dfxm/stages/strain.py::run` (lines ~446-525): delete the line `method = p["method"]`; change `result = StrainResult(method=method, output_dir=out_dir)` to `result = StrainResult(output_dir=out_dir)`; change the `process_maps_file(...)` call so it passes only the new kwargs (remove `method=method,`, `mu_com_path=p["mu_com_path"],`, `mu_ref_deg=float(p["mu_ref_deg"]),`):

```python
            strain, layer = process_maps_file(
                maps_path,
                name,
                ccmth_com_path=p["ccmth_com_path"],
                ccmth_ref_deg=float(p["ccmth_ref_deg"]),
                pixel_size_x_um=float(p["pixel_size_x_um"]),
                pixel_size_y_um=float(p["pixel_size_y_um"]),
                roi=roi,
                vlim=vlim,
                out_dir=out_dir,
                save_plots=bool(p["save_plots"]),
            )
```

and change the stacked-volume `attrs` dict to drop `mu_ref_deg` and `method` and reword the description:

```python
    attrs = dict(
        description="Stacked 3D strain volume (cot, ccmth-only)",
        ccmth_ref_deg=float(p["ccmth_ref_deg"]),
        scale_x_um=float(p["pixel_size_x_um"]),
        scale_y_um=float(p["pixel_size_y_um"]),
    )
```

- [ ] **Step 8: Trim the `STAGE` param spec**

In `dfxm/stages/strain.py` `STAGE = StageSpec(...)` `params=(...)` (lines ~52-128): delete the `method` `Param(...)` block (the first one, lines ~53-60), the `mu_com_path` `Param(...)` block (lines ~79-84), and the `mu_ref_deg` `Param(...)` block (lines ~93-100). Leave `ccmth_com_path`, `ccmth_ref_deg`, the pixel sizes, and everything else.

- [ ] **Step 9: Rewrite the module docstring and `_main`**

In `dfxm/stages/strain.py`, replace the module docstring header (lines 1-16, through the numbered pipeline list) with a single-method description:

```python
"""Strain stage — per-pixel axial strain (cot method, ccmth-only) + 3D stacking.

Port of the legacy ``calc_axial_strain_v7_batch`` calculator:

    ε = cot(ccmth_ref) · Δccmth

Pipeline per layer (order is a physics constraint — **detrend before ROI**):

1. load the ccmth Center-of-mass map from maps.h5;
2. detrend ccmth on the *full* map (separable 2-D arctan);
3. crop the ROI;
4. compute strain;
5. stack all layers into a 3-D volume.
"""
```

Then in `_main` (lines ~528-558) delete `ap.add_argument("--method", ...)` and `ap.add_argument("--mu-ref", ...)`, and remove `method=args.method,` and `mu_ref_deg=args.mu_ref,` from the `run(dict(...))` call.

- [ ] **Step 10: Fix the GUI `StrainResult` discriminator + summary**

In `gui/stage_view.py` lines 281-283, replace:

```python
    if hasattr(result, "method") and hasattr(result, "volume_shape"):  # StrainResult
        lines = [
            f"method: {result.method}",
            f"layers: {result.n_layers}   volume: {result.volume_shape}",
```

with (drop the `method:` line; switch the discriminator to `volume_shape` + `layers`, which is unique to `StrainResult` — `MosaicityResult` has no `volume_shape`, `RockingResult` has no `layers`):

```python
    if hasattr(result, "volume_shape") and hasattr(result, "layers"):  # StrainResult
        lines = [
            f"layers: {result.n_layers}   volume: {result.volume_shape}",
```

- [ ] **Step 11: Drop the mu lines from the strain override in bindings**

In `gui/bindings.py` `experiment_overrides`, the `if stage_name == "strain":` block (lines 75-86), delete `mu_com_path=exp.mu_com_path,` and `mu_ref_deg=exp.mu_ref_deg,` so it reads:

```python
    if stage_name == "strain":
        return dict(
            root_folder=exp.processed_root,
            folder_pattern=exp.folder_pattern,
            maps_filename=exp.maps_filename,
            ccmth_com_path=exp.ccmth_com_path,
            ccmth_ref_deg=exp.ccmth_ref_deg,
            pixel_size_x_um=exp.pixel_size_x_um,
            pixel_size_y_um=exp.pixel_size_y_um,
        )
```

- [ ] **Step 12: Run strain tests + the full suite**

Run: `cd dfxm_pipeline && python3 -m pytest tests/test_stage_strain.py -q && python3 -m pytest -q`
Expected: `test_stage_strain.py` passes (the two `_legacy` tests skip — legacy not vendored); the full suite still passes (config/gui tests untouched yet — `Experiment.mu_ref_deg` still exists, so they remain green).

- [ ] **Step 13: Lint**

Run: `cd dfxm_pipeline && ~/.local/bin/ruff check dfxm/stages/strain.py gui/stage_view.py gui/bindings.py tests/test_stage_strain.py`
Expected: clean (no unused imports — e.g. confirm `cot` is still used, `mu`-only helpers are gone).

- [ ] **Step 14: Commit**

```bash
cd dfxm_pipeline
git add dfxm/stages/strain.py gui/stage_view.py gui/bindings.py tests/test_stage_strain.py
git commit -m "strain: make ccmth-only; remove the mu-term method

Drop the ccmth_mu method, the -cot(mu_ref)·Δmu term, the method/mu params,
the contributions plot, and StrainResult.method. compute_strain is now a
single-array cot(ccmth_ref)·Δccmth. Fix the GUI StrainResult discriminator.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: Purge the orphaned `Experiment` calibration fields + preset

**Files:**
- Modify: `dfxm/config/models.py:133,145,200,210`
- Modify: `experiments/STO2_overnight.yaml`
- Modify: `gui/experiment_panel.py:5`, `dfxm/config/presets.py:8`
- Test: `tests/test_config.py`, `tests/gui_smoke.py`

- [ ] **Step 1: Update the config tests to expect no mu fields**

In `tests/test_config.py`:

(a) In `test_calibration_fields_flagged`, change the expected set (line 28) to drop `mu_ref_deg`:

```python
    assert flagged == {"ccmth_ref_deg", "pixel_size_x_um", "pixel_size_y_um"}
```

(b) In `test_sto2_preset_ships_expected_values`, delete the `mu_ref_deg` assertion line:

```python
    assert exp.mu_ref_deg == 11.5015  # CLAUDE.local.md value, not the 11.2491 in the scripts
```

(c) Delete the entire `test_sto2_preset_notes_record_mu_discrepancy` function (lines 39-41).

- [ ] **Step 2: Run config tests to verify they fail**

Run: `cd dfxm_pipeline && python3 -m pytest tests/test_config.py -q`
Expected: FAIL — `test_schema_matches_dataclass` and `test_calibration_fields_flagged` still see `mu_ref_deg`/`mu_com_path` in the live dataclass+schema, so the flagged-set assertion now mismatches.

- [ ] **Step 3: Remove the two fields from the `Experiment` dataclass**

In `dfxm/config/models.py`, delete line 133 (`mu_ref_deg: float = 0.0  # reference mu / sample Bragg angle (theta_s)`) and line 145 (`mu_com_path: str = "/entry/mu/Center of mass/Center of mass"`).

- [ ] **Step 4: Remove the two matching `EXPERIMENT_SCHEMA` entries**

In `dfxm/config/models.py`, delete the schema `Param` lines (must stay lock-step with the dataclass): line 200 (`Param("mu_ref_deg", ParamType.FLOAT, "mu reference", unit="deg", calibration=True),`) and line 210 (`Param("mu_com_path", ParamType.STR, "mu COM path"),`). `CALIBRATION_FIELDS` (derived from the schema) updates automatically.

- [ ] **Step 5: Clean the STO2 preset YAML**

In `experiments/STO2_overnight.yaml`:
- Delete the header block lines 9-15 (the `>>> CALIBRATION DISCREPANCY — mu_ref_deg <<<` comment through `round-trip.`).
- Replace the `notes:` block (lines 20-23) with a neutral one-liner:

```yaml
notes: STO2 overnight reference dataset.
```

- Delete the `mu_ref_deg: 11.5015 ...` line (35) and the `mu_com_path: ...` line (47).

- [ ] **Step 6: Neutralise the docstring examples**

In `gui/experiment_panel.py` line 5, change `notes (e.g. the ``mu_ref`` discrepancy), and Save-as a new preset. Emits` to:

```
notes (shown in red when present), and Save-as a new preset. Emits
```

In `dfxm/config/presets.py` line 8, change `Comments in a hand-written preset (e.g. the ``mu_ref`` discrepancy note) are` to:

```
Comments in a hand-written preset (e.g. per-field unit notes) are
```

- [ ] **Step 7: Update the GUI smoke script**

In `tests/gui_smoke.py`:
- Lines 83-86: delete `assert exp.mu_ref_deg == 11.5015 and exp.ccmth_ref_deg == 7.144` and replace with `assert exp.ccmth_ref_deg == 7.144`; delete the `assert "11.2491" in win._experiment_panel._notes.text()` line; change the `print("[2] ...")` to drop the mu_ref caveat wording, e.g. `print("[2] STO2 preset loaded")`. (Keep the `_notes.isVisible()` assertion only if notes remain non-empty — since `notes:` is now `STO2 overnight reference dataset.`, it stays visible, so keep it.)
- Lines 120-128: in the strain `set_values({...})`, delete `"method": "ccmth_mu",` and `"mu_ref_deg": 11.2491,`. The `_make_maps` helper (lines 50-60) may keep writing a `mu` dataset — harmless; leave it.

- [ ] **Step 8: Run config tests + full suite**

Run: `cd dfxm_pipeline && python3 -m pytest tests/test_config.py -q && python3 -m pytest -q`
Expected: PASS. `test_schema_matches_dataclass` confirms dataclass↔schema still lock-step; `test_preset_round_trip` passes (preset no longer carries unknown keys, so no warning).

- [ ] **Step 9: Lint**

Run: `cd dfxm_pipeline && ~/.local/bin/ruff check dfxm/config/models.py dfxm/config/presets.py gui/experiment_panel.py tests/test_config.py`
Expected: clean.

- [ ] **Step 10: Commit**

```bash
cd dfxm_pipeline
git add dfxm/config/models.py dfxm/config/presets.py gui/experiment_panel.py experiments/STO2_overnight.yaml tests/test_config.py tests/gui_smoke.py
git commit -m "config: drop orphaned mu_ref_deg/mu_com_path from Experiment + preset

Remove the now-unused mu calibration fields from the Experiment dataclass,
EXPERIMENT_SCHEMA, and the STO2 preset (incl. the mu_ref discrepancy note).
Mosaicity keeps its own mu_com_path param.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Documentation (same-change contract)

**Files:**
- Modify: `docs/Usage.md:59-65,167-170`
- Modify: `docs/Codebase.md:115,218,299,325,370`

- [ ] **Step 1: `Usage.md` — calibration warning callout**

Replace the warning block (lines 59-65) with:

```markdown
> [!warning] Calibration values are physical
> `ccmth reference` and the pixel scales (µm/px) are flagged with **⚠ calibration**
> in red. Wrong values produce *meaningless* strain maps — confirm them against the
> beamline calibration for your experiment.
```

- [ ] **Step 2: `Usage.md` — strain param table**

In the `### 2. Axial strain (strain)` section, delete the `method` row (line 169) and reword the formula line. The param table (lines 167-172) becomes:

```markdown
| Param | Meaning |
|---|---|
| `ccmth reference` | calibration angle (deg) ⚠ — strain is `cot(ccmth_ref)·Δccmth` |
| `roi` | `r0,r1,c0,c1` (blank = full image) |
| `vmin` / `vmax` | colour limits (blank = symmetric auto) |
```

- [ ] **Step 3: `Codebase.md` — Experiment row**

Line 115: change `calibration (`ccmth_ref_deg`, `mu_ref_deg`, pixel scales)` to `calibration (`ccmth_ref_deg`, pixel scales)`.

- [ ] **Step 4: `Codebase.md` — strain stage description**

Lines 214-218: update the port note and the `compute_strain` signature line. Replace lines 214-218 with:

```markdown
Port of `calc_axial_strain_v7_batch`. Per-pixel axial strain (cot method,
ccmth-only) → stacked 3-D volume.
- `LayerResult` / `StrainResult` — per-layer stats + the stacked path/shape.
- `cot`, `_arctan_model`, `_fit_arctan_1d`, `detrend_arctan_2d` — the separable arctan **detrend** (run on the full map, **before** ROI).
- `compute_strain(ccmth, ccmth_ref)` — single-array `cot(ccmth_ref)·Δccmth`.
```

- [ ] **Step 5: `Codebase.md` — remaining mu_ref mentions**

- Line 299 (`experiment_panel.py` row): change `the preset's notes (red, e.g. the `mu_ref` caveat)` to `the preset's notes (red when present)`.
- Line 325 (`test_config.py` row): change `preset round-trip, the shipped `mu_ref` value/notes.` to `preset round-trip, the shipped calibration values.`
- Line 370 (`STO2_overnight.yaml` row): change `pixel scales, and the `mu_ref` discrepancy note).` to `pixel scales).`

- [ ] **Step 6: Verify no stray mu_ref left in docs**

Run: `cd dfxm_pipeline && grep -rni "mu_ref\|11.5015\|11.2491\|ccmth_mu" docs/Usage.md docs/Codebase.md`
Expected: no matches.

- [ ] **Step 7: Commit**

```bash
cd dfxm_pipeline
git add docs/Usage.md docs/Codebase.md
git commit -m "docs: update Usage + Codebase for ccmth-only strain

Drop the method param, the mu term, and the mu_ref calibration caveat from
both the user guide and the code reference.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `cd dfxm_pipeline && python3 -m pytest -q`
Expected: all pass; the two `_legacy` strain tests report as skipped.

- [ ] **Step 2: GUI smoke (not auto-collected by pytest)**

Run: `cd dfxm_pipeline && QT_QPA_PLATFORM=offscreen python3 tests/gui_smoke.py`
Expected: exits 0; prints the `[2] STO2 preset loaded` line and runs the strain stage through the UI without referencing mu. (If PySide6/offscreen is unavailable in this environment, note it as skipped rather than failing the task.)

- [ ] **Step 3: Lint the whole repo**

Run: `cd dfxm_pipeline && ~/.local/bin/ruff check .`
Expected: clean.

- [ ] **Step 4: CLI no longer advertises mu/method**

Run: `cd dfxm_pipeline && python3 -m dfxm.stages.strain -h`
Expected: help text shows no `--method` and no `--mu-ref`.

- [ ] **Step 5: Residual-reference scan**

Run: `cd dfxm_pipeline && grep -rni "mu_ref\|ccmth_mu\|11.5015\|11.2491" dfxm/ gui/ experiments/ tests/ docs/Usage.md docs/Codebase.md | grep -v "superpowers/specs\|superpowers/plans"`
Expected: no matches (the only surviving mentions are in this plan + the design spec, which intentionally record the history).

- [ ] **Step 6: Confirm mosaicity mu read is intact**

Run: `cd dfxm_pipeline && grep -n "mu_com_path" dfxm/stages/mosaicity.py`
Expected: still present (the legitimate misorientation read was not touched).

- [ ] **Step 7: Hand off to finishing-a-development-branch**

After all green, use `superpowers:finishing-a-development-branch` to choose how to integrate `remove-mu-strain-term` (merge / PR / keep).

---

## Self-review notes

- **Spec coverage:** every change-list item in the spec maps to a task — strain core/stage (T1), config+preset+GUI bindings/summary/docstrings (T1+T2), tests (T1+T2), docs (T3), verification (T4). The `mu_com_path` keep/remove boundary is enforced (removed from Experiment in T2 step 3-4; mosaicity verified intact in T4 step 6).
- **Discriminator change** (T1 step 10) is required because `StrainResult.method` — the old type tag — is deleted; `volume_shape`+`layers` is unique to `StrainResult`.
- **Green at every commit:** T1 leaves `Experiment.mu_ref_deg` in place so config/GUI tests stay green; T2 then removes it together with its tests and the preset key (no unknown-key warning window).
- **Line numbers** are from the pre-change files and will drift as edits apply; match on the quoted code, not the numbers.
