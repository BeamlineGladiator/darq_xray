# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Overview

GUI-based data-analysis pipeline for **Dark-Field X-ray Microscopy (DFXM)** data
from the ESRF ID03 beamline. A PySide6 desktop app drives the whole flow from
raw/darfix output to finished strain & mosaicity products. It was extracted from
the `Scripts2` collection of standalone analysis scripts and reproduces them as a
single 9-stage pipeline. (DFXM — never call it XRD / X-ray diffraction.)

## Architecture

- **`dfxm/` — Qt-free core library** (importable, testable, and CLI-runnable
  without Qt). The GUI depends on the core, never the reverse.
  - `dfxm/config/` — `Experiment` dataclass + the `Param`/`StageSpec` schema; YAML
    presets live in `experiments/` (ships `STO2_overnight.yaml`).
  - `dfxm/common/` — shared primitives: `sort`, `h5io`, `alignment` (the single
    voxel-identical samy-shift + Z-interpolation pipeline), `raster` (samy/samz),
    `plotting`, `render` (per-layer PNGs / animation / pyvista top-view).
  - `dfxm/stages/` — one module per stage, each exposing
    `run(params: dict, progress=None) -> result` plus a `__main__` CLI.
    `registry.py` maps stage name → `"module:function"`.
  - `dfxm/runner.py` — runs a stage in a child process with progress/log/cancel.
- **`gui/` — PySide6 app**: `app` (entry), `main_window`, `experiment_panel`,
  `stage_view` (param form + run/cancel + Log/Results/Output[/3D] tabs),
  `bindings` (stage registry + experiment pre-fill / output auto-chaining),
  `viewers` (lazy 3-D + line-pick glue), `widgets/` (`param_form`, `mpl_canvas`,
  `pv_canvas`, `volume3d`, `line_picker`, `log_console`).

## Pipeline (stage order)

```
concat → (darfix, external) → strain → mosaicity → rocking → visualize → paraview → slices → profiles → matched
```

darfix (the ESRF tool that turns concatenated `.h5` into `maps.h5`) runs outside
the app, between `concat` and the map stages.

## Running

```bash
python3 -m gui.app                  # launch the GUI
python3 -m dfxm.stages.strain -h    # run any stage headless (each has a CLI)
python3 -m pytest -q                # tests
ruff check . && ruff format .       # lint + format (config in pyproject.toml)
```

`ruff format` runs automatically on Write/Edit via the `.claude/settings.json`
hook. Ruff config: line length 100, double quotes, target py310, rules E/F/I.

Dependencies: `numpy h5py scipy matplotlib PySide6 pyvista pyvistaqt vtk`
(`pytest` for tests; `ffmpeg` on PATH for MP4 export, else GIF fallback).

## Conventions & gotchas

- **Keep `dfxm/` Qt-free.** Never import PySide6/pyvista there.
- **Lazy heavy deps.** `pyvista`/`vtk` are imported only inside the functions
  that render/write 3-D, so GUI startup stays light and headless-safe. The 3-D
  viewer (`pv_canvas`/`volume3d`) and the profiles line picker build nothing —
  no import, no GL context, no volume load — until the user clicks.
- **Plotting:** build figures with the explicit `matplotlib.figure.Figure` API;
  never `pyplot` or `matplotlib.use(...)` (that clobbers the Qt backend the
  embedded canvases need). Shared volume renderers live in `dfxm/common/render.py`.
- **One alignment.** Every volume stage reuses `dfxm/common/alignment.py` so they
  co-register in the origin-0 PVTI world frame. Don't reimplement the
  samy-shift / Z-interpolation. The fixed order is
  `abs(FWHM) → ROI → samy X-shift → uniform-Z interp → centre`; don't reorder.
  Strain always **detrends before ROI**.
- **Calibration is physical.** `ccmth_ref_deg`, `mu_ref_deg`, and the pixel
  scales are flagged `calibration=True`. The STO2 preset ships
  `mu_ref_deg = 11.5015` (per the user) while the legacy scripts used `11.2491` —
  flag this and confirm which is canonical before trusting absolute strain.
- **Versioned, schema-driven config.** A stage declares its parameters as a
  `StageSpec`; the GUI auto-builds the form (enum→dropdown, path→file picker,
  number→spin, multi-line JSON→`ParamType.TEXT`). Don't hard-code stage fields in
  the GUI.

## Documentation (keep it in sync)

`docs/Usage.md` is the user-facing guide (Obsidian-flavoured). **It is part of
the contract: whenever you change a stage's parameters, behaviour,
inputs/outputs, add or remove a stage, or change how a viewer works, update the
matching section of `docs/Usage.md` in the SAME change** — not as a follow-up.
A PostToolUse hook (`.claude/settings.json`) prints a reminder whenever you edit
`dfxm/stages/` or `gui/`. Treat a code change that alters user-visible behaviour
without a `docs/Usage.md` update as incomplete.

## Adding a stage

1. New module in `dfxm/stages/` with a module-level `STAGE: StageSpec` and
   `run(params, progress=None)` (+ a small `__main__`).
2. Register it in `dfxm/stages/registry.py` (`STAGE_TARGETS`).
3. Wire it in `gui/bindings.py`: `STAGE_ORDER`, `STAGE_SPECS`, and an
   `experiment_overrides` branch (pre-fill from the experiment, chain prior
   outputs).
4. Add a result summary branch in `gui/stage_view.py::_summarize`.
5. Add tests under `tests/` (synthetic HDF5 fixtures; golden comparison where a
   reference output exists).
6. Document it: add a section to `docs/Usage.md` ([[#Stage reference]]) and
   update the pipeline diagram.

## Provenance

Extracted from the `Scripts2` repo (the original ESRF analysis scripts) via
`git subtree split`, preserving the phase-by-phase history. The "vs-legacy"
parity tests self-skip here because those original scripts are not vendored into
this repo.
