# dfxm_pipeline

A single Qt desktop application that drives the whole ESRF ID03 **Dark-Field
X-ray Microscopy (DFXM)** analysis flow — from raw/darfix output to finished data
products (strain/mosaicity maps, aligned volumes, ParaView PVTI, oblique slices,
line profiles, rocking-matched layers).

Parameters are entered in forms or chosen from dropdowns; calibration constants
and paths are saved as reusable, named **experiment presets**. The app reproduces
the behaviour of the existing standalone scripts — those scripts are left
untouched in the parent repo and serve as reference oracles for the ports here.

> **Nothing existing is modified or deleted.** This is a new, self-contained
> subfolder. The legacy flat scripts keep working exactly as before.

## Architecture

- **`dfxm/` — the Qt-free core library.** Pure Python implementations of every
  analysis stage, plus shared helpers. *Nothing in this package imports Qt*, so
  every stage stays runnable headless (CLI / `python3 -m dfxm.stages.<name>`) and
  unit-testable.
- **`gui/` — the PySide6 desktop app.** Builds parameter forms from each stage's
  typed schema, runs stages, and embeds matplotlib + pyvista viewers. It only
  *calls* `dfxm/`.

This split is deliberate: keeping `gui/` a sibling of `dfxm/` (rather than nesting
it) makes the "core never imports Qt" invariant easy to enforce and test.

## Layout

```
dfxm_pipeline/
  pyproject.toml            # metadata + ruff/pytest config (inherits repo ruff rules)
  experiments/              # named presets (YAML): paths, patterns, angles, scales
    STO2_overnight.yaml
  dfxm/                     # Qt-FREE core
    config/                 # Experiment model + per-stage param schemas; preset I/O
    common/                 # natural sort, HDF5 I/O, alignment, raster, plotting helpers
    stages/                 # one module per script family; each exposes run(params, progress)
    runner.py               # child-process execution + progress/log/cancel protocol
  gui/                      # PySide6 app (entry point: python3 -m gui.app)
    widgets/                # param-form, matplotlib canvas, pyvista canvas, log console
  tests/                    # synthetic-fixture unit tests + golden reproduction tests
```

## Running

The app is designed to **run in place** — no `pip install` step, matching the
parent repo's `python3 script.py` workflow.

```bash
cd dfxm_pipeline
python3 -m gui.app
```

Stages are also runnable headless, without the GUI:

```bash
cd dfxm_pipeline
python3 -m dfxm.stages.concat --help
```

### Dependencies

```bash
pip install --user numpy h5py scipy matplotlib pyyaml PySide6 pyvista pyvistaqt vtk
# plus: pytest (to run the test suite); ffmpeg on PATH for MP4 export
```

(On an externally-managed/PEP 668 Python, add `--break-system-packages`.)

## Pipeline / stage map

The app **brackets darfix**: it prepares input (concat), you run darfix yourself,
then the app resumes from darfix's `maps.h5`. It validates `maps.h5` before the
downstream stages; it does **not** launch darfix.

| Stage | Status | Ported from |
|---|---|---|
| `concat` | implemented (Phase 0) | `concatenate_h5_scans_v3` + `batch_concatenate_h5_scans_v1` |
| *(darfix)* | external — run yourself | → produces `maps.h5` |
| `strain` | Phase 1 | `y_calc_axial_strain_v6_batch` (ccmth+mu) / `calc_axial_strain_v7_batch` (ccmth-only) |
| `mosaicity` | Phase 1 | `stack_h5_darfix_volumes` |
| `visualize` | Phase 1 | `visualize_aligned_volumes_v6` |
| `rocking` | Phase 2 | `build_aligned_raw_rocking_volumes_v3` |
| `paraview` | Phase 2 | `export_aligned_volumes_to_paraview_v6_pvti` |
| `slices` | Phase 3 | `extract_oblique_slices_v5` |
| `profiles` | Phase 3 | `line_profile_oblique_slices_v2` |
| `matched` | Phase 3 | `plot_rocking_matched_layers_v3` |

## Domain constraints carried into the core

- **Reference angles are experiment-specific.** Wrong angles → meaningless strain
  maps. The shipped `STO2_overnight` preset flags its calibration fields.
- **Detrend before ROI.** Strain stages always polynomial-detrend the full map,
  then crop the ROI. This order is fixed and not reorderable in the UI.
- **Fixed alignment order:** `abs (FWHM only) → ROI → samy sub-pixel shift
  (X-canvas expand) → Z interpolation onto a uniform grid → centre (CoM/strain
  only)`. One implementation in `common/alignment.py`, shared by visualization,
  PVTI export, slices, and rocking, so ParaView coordinates stay interchangeable.
- **VDS fragility.** Concat defaults to virtual datasets that reference the
  original `.h5` files; a `copy_data` toggle makes a self-contained copy.
- **Motor paths.** `samy`/`samz` are read from `…/instrument/positioners/`; the
  detector lives at `…/measurement/pco_ff`. These are overridable constants.
