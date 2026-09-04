# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Overview

**DARQ** (distribution / import name `darq_xray`) — a GUI-based data-analysis
pipeline for **Dark-Field X-ray Microscopy (DFXM)** data from the ESRF ID03
beamline. A PySide6 desktop app drives the whole flow from raw/darfix output to
finished strain & mosaicity products. It was extracted from
the `Scripts2` collection of standalone analysis scripts and reproduces them as a
single 9-stage pipeline. (DFXM — never call it XRD / X-ray diffraction.)

## Interaction preferences

- **Confirmations as numbered choices.** When you need the user to confirm or
  authorise something (proceed/abort, pick between approaches, approve a change),
  ask via the `AskUserQuestion` tool with concrete options rather than a free-text
  "should I proceed?" prompt, so the user can answer by choosing an option number.

## Autonomous execution kickoff

- **Subagent model tier:** no pinned tiers — dispatch subagents with no `model`
  override so they inherit the session model. Reach for a different tier only
  when a particular dispatch clearly warrants it, and say which and why when you
  do. The custom `planner`, `supervisor` (both Fable 5, `effort: high`,
  read-only) and `sonnet-4-6` agent types still exist but are opt-in: use them
  when asked for by name. A Fable review is not a required step of any flow.
- **Kickoff confirmation:** before starting any autonomous multi-task run
  (subagent-driven-development, executing-plans), confirm via AskUserQuestion:
  (1) the stop point — a named task/phase boundary or "run to completion", and
  (2) execution mode — full SDD (per-task implementer + dual reviewers) vs
  inline execution with a single end review; recommend inline when the plan is
  <= ~4 small tasks in one subsystem.
- **Bare "resume":** on a bare resume request, first state the resume point and
  the next action, then wait for confirmation before dispatching any agent —
  the no-check-in rule applies *after* scope is confirmed, not to scoping itself.
- **SDD commit hygiene:** when cleaning a subagent's commit, never strip `docs/`
  changes that accompany `darq_xray/stages/` or `darq_xray/gui/` changes — the same-change docs
  contract outranks code-only commit aesthetics.

## Architecture

- **`darq_xray/` — Qt-free core library** (importable, testable, and CLI-runnable
  without Qt). The GUI depends on the core, never the reverse.
  - `darq_xray/stages/` — one module per stage, each exposing
    `run(params: dict, progress=None) -> result` plus a `__main__` CLI.
    `registry.py` maps stage name → `"module:function"`.
  - `darq_xray/runner.py` — runs a stage in a child process with progress/log/cancel.
- **`darq_xray/gui/` — the PySide6 desktop app** on top of the core; stage wiring
  (order, pre-fill, output auto-chaining) lives in `darq_xray/gui/bindings.py`.

## Pipeline (stage order)

```
concat → (darfix, external) → strain → mosaicity → rocking → visualize → paraview → slices → profiles → matched
```

darfix (the ESRF tool that turns concatenated `.h5` into `maps.h5`) runs outside
the app, between `concat` and the map stages.

## Running

```bash
pip install -e ".[test]"            # install deps (once; editable, run-in-place kept)
python3 -m darq_xray.gui.app                  # launch the GUI
python3 -m darq_xray.stages.strain -h    # run any stage headless (each has a CLI)
python3 -m pytest -q                # tests
ruff check . && ruff format .       # lint + format (config in pyproject.toml)
```

`ruff format` runs automatically on Write/Edit via the `.claude/settings.json`
hook.

`pyproject.toml` is the **single source of truth** for dependencies. The prose
lists in `README.md` and `docs/Usage.md` are checked against it by
`tests/test_docs_dependencies.py`, so a new dependency goes in `pyproject.toml`
*and* both marked (`<!-- deps:start -->`) blocks in the same change. That test
also guards `[build-system]` and `[tool.setuptools.packages.find]`, without
which `pip install -e .` fails outright on this flat layout. `ffmpeg` on PATH
enables MP4 export (GIF fallback without it).

## Conventions & gotchas

- **Keep the core Qt-free.** Never import PySide6/pyvista anywhere in
  `darq_xray/` outside the `darq_xray/gui/` subpackage.
- **Lazy heavy deps.** `pyvista`/`vtk` are imported only inside the functions
  that render/write 3-D, so GUI startup stays light and headless-safe. The 3-D
  viewer (`pv_canvas`/`volume3d`) and the profiles line picker build nothing —
  no import, no GL context, no volume load — until the user clicks.
- **Plotting:** build figures with the explicit `matplotlib.figure.Figure` API;
  never `pyplot` or `matplotlib.use(...)` (that clobbers the Qt backend the
  embedded canvases need). Shared volume renderers live in `darq_xray/common/render.py`.
- **One alignment.** Every volume stage reuses `darq_xray/common/alignment.py` so they
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
  here: `hint=` strings in `darq_xray/stages/*.py` contain em-dashes and sit at 12
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
- `stage_view.py` and all Qt code live under `darq_xray/gui/`, never elsewhere in
  `darq_xray/` — grep for a filename before Read if unsure which tree it's in.
- **Check for a git remote before any push/PR step.** The repo was developed
  entirely locally; if `git remote -v` is empty, skip pull/push/PR in any
  branch flow rather than trying to create one.
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
prints a reminder whenever you edit `darq_xray/stages/` or `darq_xray/gui/`. Treat a code change
that alters behaviour or structure without the matching `docs/` update as
incomplete.

## Adding a stage

Follow the `add-stage` skill (`.claude/skills/add-stage/SKILL.md`) — the 6-step
checklist covering the stage module, registry, GUI bindings, summarizer, tests,
and the docs contract.

## Provenance

Extracted from the `Scripts2` repo (the original ESRF analysis scripts) via
`git subtree split`, preserving the phase-by-phase history. The "vs-legacy"
parity tests self-skip here because those original scripts are not vendored into
this repo.
