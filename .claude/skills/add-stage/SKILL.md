---
name: add-stage
description: Use when adding a new pipeline stage to darq_xray — the 6-step checklist (StageSpec module, registry, GUI bindings, summarizer, tests, docs) that keeps the enforcement tests and both docs in sync.
---

# Adding a stage

1. New module in `darq_xray/stages/` with a module-level `STAGE: StageSpec` and
   `run(params, progress=None)` (+ a small `__main__`). Every param needs
   `help` (written for a first-time beamline user); advanced params need
   `group`; input paths set `must_exist=True`. Give the spec a
   newcomer-friendly `description` (it feeds the Overview page and help
   panel). `tests/test_param_metadata.py` enforces all of this. Raise
   `StageUserError(message, hint=...)` from `darq_xray.common.errors` for input
   problems the user can fix.
2. Register it in `darq_xray/stages/registry.py` (`STAGE_TARGETS`).
3. Wire it in `darq_xray/gui/bindings.py`: `STAGE_ORDER`, `STAGE_SPECS`, and an
   `experiment_overrides` branch (pre-fill from the experiment, chain prior
   outputs).
4. Add a `_summarize_<stage>` formatter in `darq_xray/gui/stage_view.py` and register it
   in `_SUMMARIZERS` (plus `_IMAGE_PICKERS` if the stage produces a preview
   image). A test asserts the tables stay in sync with the registry.
5. Add tests under `tests/` (synthetic HDF5 fixtures; golden comparison where a
   reference output exists).
6. Document it: add a section to `docs/Usage.md` (Stage reference) and update the
   pipeline diagram, and add the module/functions to `docs/Codebase.md`
   (`darq_xray/stages` + the data-flow table). **Every param must be named in
   both docs** — `tests/test_docs_stage_params.py` fails otherwise. Essentials
   belong in the stage's `| Param | Meaning |` table; `advanced` fields can go
   in the collapsed "Advanced fields" table that every stage section carries.
