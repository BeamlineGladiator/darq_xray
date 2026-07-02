# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Overview

GUI-based data-analysis pipeline for **Dark-Field X-ray Microscopy (DFXM)** data
from the ESRF ID03 beamline. A PySide6 desktop app drives the whole flow from
raw/darfix output to finished strain & mosaicity products. It was extracted from
the `Scripts2` collection of standalone analysis scripts and reproduces them as a
single 9-stage pipeline. (DFXM — never call it XRD / X-ray diffraction.)

## Interaction preferences

- **Confirmations as numbered choices.** When you need the user to confirm or
  authorise something (proceed/abort, pick between approaches, approve a change),
  ask via the `AskUserQuestion` tool with concrete options rather than a free-text
  "should I proceed?" prompt, so the user can answer by choosing an option number.

## Autonomous execution kickoff

- **Subagent model tier:** dispatch mid-tier implementer/reviewer subagents as the
  custom `sonnet-4-6` agent type, never Sonnet 5; reserve fable for final
  whole-branch reviews.
- **Kickoff confirmation:** before starting any autonomous multi-task run
  (subagent-driven-development, executing-plans), confirm via AskUserQuestion:
  (1) the stop point — a named task/phase boundary or "run to completion",
  (2) anything about subagent tiering that isn't already settled above, and
  (3) execution mode — full SDD (per-task implementer + dual reviewers) vs
  inline execution with a single end review; recommend inline when the plan is
  <= ~4 small tasks in one subsystem.
- **Bare "resume":** on a bare resume request, first state the resume point and
  the next action, then wait for confirmation before dispatching any agent —
  the no-check-in rule applies *after* scope is confirmed, not to scoping itself.
- **SDD commit hygiene:** when cleaning a subagent's commit, never strip `docs/`
  changes that accompany `dfxm/stages/` or `gui/` changes — the same-change docs
  contract outranks code-only commit aesthetics.

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
- **Calibration is physical.** `ccmth_ref_deg` and the pixel scales are flagged
  `calibration=True`; wrong values produce meaningless strain maps — confirm them
  against the beamline calibration for your experiment.
- **Versioned, schema-driven config.** A stage declares its parameters as a
  `StageSpec`; the GUI auto-builds the form (enum→dropdown, path→file picker,
  number→spin, multi-line JSON→`ParamType.TEXT`). Don't hard-code stage fields in
  the GUI.
- **User-facing errors carry hints.** Input-validation failures raise
  `StageUserError(message, hint)`; the GUI banner shows both. Don't convert
  the skip-based reporting (empty results list reasons) into exceptions.
- **Read before first Edit.** Any file not created this session — especially
  `memory/MEMORY.md` and `.superpowers/sdd/progress.md` — must be Read once
  before its first Edit.
- **Never reconstruct `old_string` from memory or sibling files.** Known hazards
  here: `hint=` strings in `dfxm/stages/*.py` contain em-dashes and sit at 12
  *or* 16 spaces depending on nesting; markdown prose reflows. Read (or grep the
  exact bytes of) every target site first — batch one Read covering all sites
  before a multi-file edit sweep.
- **Read big docs once.** Read plan/spec files (`docs/superpowers/plans/*`) in
  full at most once per session; for later per-task slices use `Read` with
  `offset`/`limit` on the task's section, or quote from context. Same for stage
  modules during review→fix spans: re-read only the target function region.
- **Bash hygiene:** for equality checks use `cmp -s A B && echo IDENTICAL || echo DIFFERS`
  (never `diff` inside an `&&` chain — it exits 1 on difference). Never
  `pkill -f <pattern>` with a pattern matching your own command line; collect PIDs
  first (`pgrep -f <pattern> | grep -v $$`). The GUI smoke test is
  `tests/gui_smoke.py` (no `test_` prefix; it is not a pytest file).
- `stage_view.py` and all Qt code live under `gui/`, never `dfxm/` — grep for a
  filename before Read if unsure which tree it's in.
- **This repo has no git remote** — skip pull/push/PR in any branch flow.
- `~/.claude/projects/.../memory/` is not git-tracked; writing the file is the save.
- Custom agent-type files (`~/.claude/agents/*.md`) load at **session start**
  only — don't test-dispatch a just-written type; tell the user it needs a restart.
- Subagent resume via SendMessage works in this harness — prefer SendMessage to
  resume a reviewer for re-review (preserves context, avoids re-priming) and an
  implementer for small fixes; dispatch a fresh fix agent only when the original
  agent has exited or the fix needs a clean context. (Fact dated 2026-07;
  date-stamp harness facts so stale ones get retired.)
- If a dispatched subagent runs well past the typical duration for its task class
  (e.g. a read-only review outlasting an implementation), check on it instead of
  waiting indefinitely; before re-dispatching, verify the worktree/git state is
  untouched.
- Verify downloads complete (exit status + size/tail) before parsing; prefer
  WebFetch for remote JSON — never pipe curl straight into `json.load`.
- Match review effort to change size: full xhigh sweeps for feature branches;
  medium/high scoped to changed files for follow-up commits.

## Documentation (keep it in sync)

Two docs live under `docs/` (both Obsidian-flavoured):
- `docs/Usage.md` — the **user** guide (how to operate each stage/viewer).
- `docs/Codebase.md` — the **code** reference (every module/class/function).

**It is part of the contract: whenever you change a stage's parameters,
behaviour, inputs/outputs, add or remove a stage/module/public function, or
change how a viewer works, update the matching sections of BOTH docs in the SAME
change** — `Usage.md` for user-visible behaviour, `Codebase.md` for the code
structure — not as a follow-up. A PostToolUse hook (`.claude/settings.json`)
prints a reminder whenever you edit `dfxm/stages/` or `gui/`. Treat a code change
that alters behaviour or structure without the matching `docs/` update as
incomplete.

## Adding a stage

1. New module in `dfxm/stages/` with a module-level `STAGE: StageSpec` and
   `run(params, progress=None)` (+ a small `__main__`). Every param needs
   `help` (written for a first-time beamline user); advanced params need
   `group`; input paths set `must_exist=True`. Give the spec a
   newcomer-friendly `description` (it feeds the Overview page and help
   panel). `tests/test_param_metadata.py` enforces all of this. Raise
   `StageUserError(message, hint=...)` from `dfxm.common.errors` for input
   problems the user can fix.
2. Register it in `dfxm/stages/registry.py` (`STAGE_TARGETS`).
3. Wire it in `gui/bindings.py`: `STAGE_ORDER`, `STAGE_SPECS`, and an
   `experiment_overrides` branch (pre-fill from the experiment, chain prior
   outputs).
4. Add a `_summarize_<stage>` formatter in `gui/stage_view.py` and register it
   in `_SUMMARIZERS` (plus `_IMAGE_PICKERS` if the stage produces a preview
   image). A test asserts the tables stay in sync with the registry.
5. Add tests under `tests/` (synthetic HDF5 fixtures; golden comparison where a
   reference output exists).
6. Document it: add a section to `docs/Usage.md` (Stage reference) and update the
   pipeline diagram, and add the module/functions to `docs/Codebase.md`
   (`dfxm/stages` + the data-flow table).

## Provenance

Extracted from the `Scripts2` repo (the original ESRF analysis scripts) via
`git subtree split`, preserving the phase-by-phase history. The "vs-legacy"
parity tests self-skip here because those original scripts are not vendored into
this repo.
