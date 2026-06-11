# GUI Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the DFXM pipeline GUI usable by first-time beamline users: essentials-first forms with a grouped Advanced expander, full help coverage in a focus-following help panel, a pipeline-rail window with an Overview landing page, and friendly run feedback.

**Architecture:** All declutter/help metadata lives in the Qt-free stage schemas (`Param` gains `advanced`/`group`/`must_exist`); the GUI keeps auto-generating forms from them. A `StageUserError(message, hint)` exception carries actionable hints from stages through the runner (`Failed.hint`) to a new banner in `StageView`. The spec is `docs/superpowers/specs/2026-06-11-gui-overhaul-design.md` — consult it for rationale; this plan is self-contained for execution.

**Tech Stack:** Python 3.10+, PySide6 (GUI only — `dfxm/` stays Qt-free), pytest, ruff (line length 100, double quotes).

**Verification commands (used throughout):**

```bash
python3 -m pytest -q                     # full suite; expect "114 passed, 13 skipped" before this plan
ruff check . && ruff format --check .    # lint + format
python3 tests/gui_smoke.py               # offscreen GUI harness (NOT pytest-collected; keep it that way)
```

**Repo rules that bind every task** (from CLAUDE.md):
- Never import PySide6/pyvista inside `dfxm/`.
- Never use `matplotlib.pyplot` or `matplotlib.use(...)` in GUI code.
- A PostToolUse hook runs `ruff format` automatically on edits; don't fight it.
- Docs (`docs/Usage.md`, `docs/Codebase.md`) must be updated with behaviour changes — Task 20 does this; the branch must not merge without it.
- Never change calibration values (`ccmth_ref_deg`, pixel sizes) — this plan only adds *metadata* to params, never touches `default=`.

---

## File structure

| File | Action | Responsibility |
|---|---|---|
| `dfxm/config/models.py` | modify | `Param` gains `advanced`, `group`, `must_exist`; `EXPERIMENT_SCHEMA` gains help texts |
| `dfxm/common/errors.py` | create | `StageUserError(message, hint)` — Qt-free user-facing error |
| `dfxm/runner.py` | modify | `Failed` gains `hint`; `_worker` forwards it |
| `dfxm/stages/{concat,strain,mosaicity,visualize,rocking,paraview,slices,profiles,matched}.py` | modify | param metadata + help texts + newcomer descriptions; later: `StageUserError` adoption |
| `gui/widgets/param_form.py` | modify | essentials + grouped Advanced expander; `focusedParamChanged`; `focus_param()` |
| `gui/widgets/help_panel.py` | create | focus-following help box |
| `gui/experiment_panel.py` | modify | compact header + modal Edit dialog |
| `gui/overview_page.py` | create | landing page: chips + stage descriptions + status recap |
| `gui/main_window.py` | modify | pipeline rail (single list, darfix row, optional concat), Overview in stack |
| `gui/stage_view.py` | modify | banner, pre-run validation, progress bar, `runStarted` signal |
| `tests/test_param_metadata.py` | create | enforcement: help coverage, group-on-advanced, ≤8 essentials, `must_exist` sanity |
| `tests/test_runner_hints.py` | create | `StageUserError` → `Failed.hint` round-trip |
| `tests/test_stage_user_errors.py` | create | stages raise `StageUserError` with hints on broken inputs |
| `tests/gui_smoke.py` | modify | assertions for forms, help panel, rail, overview, banner, dialog |
| `docs/Usage.md`, `docs/Codebase.md`, `CLAUDE.md` | modify | Task 20 |

**Branch:** create `gui-overhaul` from `master` before Task 1 (`git checkout -b gui-overhaul`). Commit after every task. Do not merge before Task 20 is done.

**Ordering note (deliberate):** `tests/test_param_metadata.py` is *written* in Task 3 but only *committed* in Task 13, after Tasks 4–12 make it pass stage by stage. Every commit keeps the committed suite green; the uncommitted test file is the red→green driver across those tasks. Do not `git add` it early.

---

### Task 1: `Param` schema fields (`advanced`, `group`, `must_exist`)

**Files:**
- Modify: `dfxm/config/models.py:40-60`
- Test: `tests/test_param_metadata.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_param_metadata.py`:

```python
"""Schema metadata enforcement: every param must carry first-timer help.

Field-existence tests run from Task 1 of the GUI-overhaul plan; the
per-stage enforcement tests below them go green stage-by-stage as the
specs gain metadata (Tasks 4-12).
"""

from dfxm.config.models import Param, ParamType


def test_param_metadata_fields_default_off():
    p = Param("x", ParamType.STR, "X")
    assert p.advanced is False
    assert p.group == ""
    assert p.must_exist is False


def test_param_metadata_fields_settable():
    p = Param(
        "x",
        ParamType.DIR,
        "X",
        advanced=True,
        group="Data layout",
        must_exist=True,
    )
    assert p.advanced is True
    assert p.group == "Data layout"
    assert p.must_exist is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m pytest tests/test_param_metadata.py -v`
Expected: 2 FAILED with `TypeError: __init__() got an unexpected keyword argument 'advanced'` (second test) and `AttributeError: 'Param' object has no attribute 'advanced'` (first).

- [ ] **Step 3: Add the fields**

In `dfxm/config/models.py`, the `Param` dataclass currently ends:

```python
    help: str | None = None
    calibration: bool = False
```

Change to:

```python
    help: str | None = None
    calibration: bool = False
    advanced: bool = False  # True -> rendered inside the collapsed Advanced expander
    group: str = ""  # themed section header inside Advanced (required when advanced)
    must_exist: bool = False  # input path/dir: GUI checks existence before a run
```

Also extend the class docstring's last paragraph to:

```python
    """One declarative parameter in a stage schema.

    ``choices`` is required for :attr:`ParamType.ENUM` and renders as a
    dropdown. ``unit`` and ``help`` are advisory text for the form. Mark
    physically-meaningful calibration constants with ``calibration=True``.
    ``advanced`` params collapse into the form's Advanced expander under
    their ``group`` header; ``must_exist`` marks input paths the GUI
    verifies on disk before launching a run (never set it on outputs).
    """
```

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_param_metadata.py tests/test_config.py -v`
Expected: all PASS (test_config.py guards the experiment schema sync — must stay green).

- [ ] **Step 5: Lint and commit (NOT the new test file yet — it grows in Task 3)**

```bash
ruff check . && ruff format --check .
git add dfxm/config/models.py
git commit -m "config: add Param.advanced/.group/.must_exist metadata fields"
```

(Yes, the test file stays uncommitted until Task 13 — see Ordering note.)

---

### Task 2: `StageUserError` + `Failed.hint` round-trip

**Files:**
- Create: `dfxm/common/errors.py`
- Modify: `dfxm/runner.py:52-57` (Failed), `dfxm/runner.py:96-101` (_worker except)
- Test: `tests/test_runner_hints.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_runner_hints.py`:

```python
"""StageUserError hints must survive the child-process boundary."""

import time

from dfxm.runner import Failed, StageRunner


def _fail_with_hint(params, progress=None):
    from dfxm.common.errors import StageUserError

    raise StageUserError("maps.h5 not found in /nowhere", hint="Run darfix first.")


def _fail_plain(params, progress=None):
    raise RuntimeError("boom")


def _run_to_failure(fn) -> Failed:
    runner = StageRunner(fn, {}, start_method="fork")
    runner.start()
    t0 = time.time()
    while not runner.finished and time.time() - t0 < 30:
        runner.poll()
        time.sleep(0.02)
    assert runner.failure is not None
    return runner.failure


def test_stage_user_error_attrs():
    from dfxm.common.errors import StageUserError

    exc = StageUserError("bad input", hint="fix it like so")
    assert isinstance(exc, ValueError)  # existing pytest.raises(ValueError) keep working
    assert str(exc) == "bad input"
    assert exc.hint == "fix it like so"


def test_hint_round_trips_through_runner():
    failure = _run_to_failure(_fail_with_hint)
    assert failure.error == "maps.h5 not found in /nowhere"
    assert failure.hint == "Run darfix first."


def test_plain_exception_has_empty_hint():
    failure = _run_to_failure(_fail_plain)
    assert failure.error == "boom"
    assert failure.hint == ""
```

- [ ] **Step 2: Run it to verify it fails**

Run: `python3 -m pytest tests/test_runner_hints.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfxm.common.errors'` and/or `TypeError` on `Failed.hint`.

- [ ] **Step 3: Implement**

Create `dfxm/common/errors.py`:

```python
"""User-facing stage errors (Qt-free).

:class:`StageUserError` marks a failure caused by the stage's *inputs*
rather than a bug: the message says what is wrong, ``hint`` says what the
user should do about it. It subclasses :class:`ValueError` so existing
callers (and tests) that treat input validation as ValueError keep working.
The runner forwards ``hint`` to the GUI via ``Failed.hint``.
"""

from __future__ import annotations


class StageUserError(ValueError):
    """An input problem the user can fix, with an actionable hint."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint
```

In `dfxm/runner.py`, change the `Failed` dataclass:

```python
@dataclass
class Failed:
    error: str
    traceback: str
    hint: str = ""  # actionable advice from StageUserError, "" otherwise
```

and in `_worker`, change the except clause's last line from
`q.put(Failed(str(exc), traceback.format_exc()))` to:

```python
        q.put(Failed(str(exc), traceback.format_exc(), str(getattr(exc, "hint", "") or "")))
```

(`getattr` keeps the child free of an import that could itself fail mid-error.)

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_runner_hints.py -q && python3 -m pytest -q`
Expected: new tests PASS; full suite stays green (114+3 passed, 13 skipped).

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && ruff format --check .
git add dfxm/common/errors.py dfxm/runner.py tests/test_runner_hints.py
git commit -m "core: StageUserError with hint, forwarded through Failed.hint"
```

---

### Task 3: Enforcement tests (written now, committed in Task 13)

**Files:**
- Modify: `tests/test_param_metadata.py` (append)

- [ ] **Step 1: Append the enforcement tests**

Append to `tests/test_param_metadata.py`:

```python
import importlib

import pytest

from dfxm.config.models import EXPERIMENT_SCHEMA, ParamType
from dfxm.stages.registry import STAGE_TARGETS

_STAGES = sorted(STAGE_TARGETS)


def _spec(stage_name: str):
    module_name = STAGE_TARGETS[stage_name].split(":")[0]
    return importlib.import_module(module_name).STAGE


@pytest.mark.parametrize("stage_name", _STAGES)
def test_every_param_has_help(stage_name):
    missing = [p.name for p in _spec(stage_name).params if not (p.help or "").strip()]
    assert not missing, f"{stage_name}: params without help text: {missing}"


@pytest.mark.parametrize("stage_name", _STAGES)
def test_advanced_params_have_group(stage_name):
    bad = [p.name for p in _spec(stage_name).params if p.advanced and not p.group.strip()]
    assert not bad, f"{stage_name}: advanced params without a group: {bad}"


@pytest.mark.parametrize("stage_name", _STAGES)
def test_at_most_eight_essentials(stage_name):
    essentials = [p.name for p in _spec(stage_name).params if not p.advanced]
    assert 1 <= len(essentials) <= 8, f"{stage_name}: essentials = {essentials}"


@pytest.mark.parametrize("stage_name", _STAGES)
def test_must_exist_only_on_input_paths(stage_name):
    bad = [
        p.name
        for p in _spec(stage_name).params
        if p.must_exist and p.type not in (ParamType.PATH, ParamType.DIR)
    ]
    assert not bad, f"{stage_name}: must_exist on non-path params: {bad}"


def test_experiment_schema_has_help():
    missing = [p.name for p in EXPERIMENT_SCHEMA if not (p.help or "").strip()]
    assert not missing, f"experiment schema params without help: {missing}"
```

- [ ] **Step 2: Run it — verify it fails for every stage (this is the red state Tasks 4-12 burn down)**

Run: `python3 -m pytest tests/test_param_metadata.py -q`
Expected: `test_every_param_has_help` FAILS for all 9 stages, `test_at_most_eight_essentials` FAILS for all 9 (every param is currently essential), `test_experiment_schema_has_help` FAILS. `test_advanced_params_have_group` and `test_must_exist_only_on_input_paths` pass vacuously. **Do not commit.**

---

### Tasks 4–12: per-stage metadata + help + description

These nine tasks share one mechanical pattern, stated once here; each task below
supplies only its stage's *content table* and *description*. **The pattern (read
once, apply in every task):**

1. Open the stage module's `STAGE: StageSpec` block (`grep -n "STAGE = StageSpec" dfxm/stages/<stage>.py`).
2. Replace `description=` with the task's **New description** string (exact text).
3. For every `Param(...)` in `params`, set the keyword arguments from the task's
   table: add `advanced=True, group="<Group>"` where the table says so (omit
   both for essentials — the defaults are fine), add `must_exist=True` where
   marked, and set `help="<text>"` to the table's exact text (replacing any
   existing `help=`). **Never touch `name`, `type`, `label`, `default`, `unit`,
   `choices`, `calibration`.**
4. Keep the params in their existing order in the tuple — display order within
   essentials and within groups follows tuple order; groups appear in
   first-appearance order. The tables below list params in the current tuple
   order, so apply them top to bottom.
5. Run the scoped enforcement test red→green:
   `python3 -m pytest tests/test_param_metadata.py -q -k <stage>` (FAILs before the edit, PASSes after).
6. Run that stage's own tests: `python3 -m pytest tests/test_stage_<stage>.py -q` — must stay green (metadata must not change behaviour).
7. `ruff check . && ruff format --check .`
8. Commit **only the stage module**:
   `git add dfxm/stages/<stage>.py && git commit -m "<stage>: first-timer help, essentials/advanced groups, must_exist inputs"`

µ/χ/µm characters are intentional in help texts (the codebase already uses µm).

---

### Task 4: concat metadata

**Files:** Modify: `dfxm/stages/concat.py` (STAGE block). Steps: the 8-step pattern above with this content.

**New description:**
> Merges the separate BLISS scan entries of each raw layer folder into one darfix-ready .h5 file (detector frames + motor positions). Optional — skip it if your scans are already concatenated. Writes <folder>_concat.h5 next to each input.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `mode` | essential | | single processes one scan folder ('Input folder'); batch processes every subfolder of 'Root folder' whose name matches 'Folder pattern'. |
| `input_folder` | essential | yes | The raw scan folder containing the .h5 file to concatenate (single mode only). |
| `h5_filename_override` | Data layout | | Name of the .h5 file inside the input folder, if it is not '<folder name>.h5'. Leave blank to auto-detect (single mode). |
| `root_folder` | essential | yes | Parent folder holding one subfolder per layer (batch mode only). Each matching subfolder is concatenated separately. |
| `folder_pattern` | essential | | Glob pattern selecting which subfolders of the root to process in batch mode, e.g. '*' for all or 'layer_*' for a subset. |
| `skip_existing` | essential | | Skip folders that already contain a _concat.h5 output — useful when re-running after adding new layers. |
| `entry_suffix` | Data layout | | Only BLISS entries ending in this suffix are merged (e.g. '.1' keeps 1.1, 2.1, …); other entries such as alignment scans are ignored. |
| `detector_read_path` | Data layout | | HDF5 path to the detector frames inside each scan entry. Only change if your beamline files use a different detector or layout. |
| `detector_write_path` | Data layout | | HDF5 path where the merged detector data is written inside the output entry (darfix reads this location). |
| `positioners_path` | Data layout | | HDF5 path to the motor-position group inside each scan entry; positions are merged across scans. |
| `output_entry` | Data layout | | Name of the single merged entry in the output file. darfix expects 'entry_0000'. |
| `vds_policy` | Output | | How the virtual dataset stores references to the source files: relative paths survive moving the whole tree together; absolute paths break when anything moves. Ignored when 'Copy data' is on. |
| `copy_data` | Output | | Off = write a virtual dataset (fast and small, but it breaks if the source files move). On = copy the frames into a self-contained archival file (slower, larger). |
| `overwrite` | Output | | Replace an existing output file. If off, folders with an existing output fail instead. |

(Essentials: 5 — mode, input_folder, root_folder, folder_pattern, skip_existing.)

---

### Task 5: strain metadata

**Files:** Modify: `dfxm/stages/strain.py` (STAGE block). Steps: the shared 8-step pattern.

**New description:**
> Turns the darfix maps.h5 centre-of-mass maps into per-layer axial strain maps (cot method) and stacks them into a 3-D volume. Needs maps.h5 from darfix in each layer folder; writes per-layer PNGs plus stacked_strain_volumes.h5.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `mode` | essential | | single processes one layer folder ('Input folder'); batch processes every subfolder of 'Root folder' matching 'Folder pattern'. |
| `input_folder` | essential | yes | Layer folder containing the darfix maps.h5 (single mode only). |
| `root_folder` | essential | yes | Parent of the layer folders (batch mode only); every matching subfolder with a maps.h5 becomes one layer of the volume. |
| `folder_pattern` | Data layout | | Glob selecting which subfolders of the root are strain layers in batch mode. |
| `maps_filename` | Data layout | | Filename of the darfix output inside each layer folder (normally maps.h5). |
| `ccmth_com_path` | Data layout | | HDF5 path of the ccmth centre-of-mass dataset inside maps.h5, as written by darfix. Only change for a non-standard darfix export. |
| `ccmth_ref_deg` | Calibration | | Reference Bragg angle θ of the unstrained lattice, in degrees. Strain is cot(θ_ref)·Δccmth per pixel, so a wrong reference silently shifts and scales every strain value. Confirm against the beamline alignment for your experiment. |
| `pixel_size_x_um` | Calibration | | Physical size of one detector pixel along X, in µm — sets the lateral scale of every map and volume. From the beamline optics calibration. |
| `pixel_size_y_um` | Calibration | | Physical size of one detector pixel along Y, in µm — sets the vertical scale of every map and volume. From the beamline optics calibration. |
| `roi` | essential | | Region of interest as 'r0,r1,c0,c1' in pixels (blank = full image). Cropped after detrending, so the trend fit always uses the full map — that order is a physics constraint. |
| `vmin` | Appearance | | Lower colour limit of the strain plots (blank = symmetric automatic limits). Display only — does not affect the saved data. |
| `vmax` | Appearance | | Upper colour limit of the strain plots (blank = symmetric automatic limits). Display only — does not affect the saved data. |
| `output_dir` | essential | | Where the per-layer diagnostic PNGs go (default: a strain_maps folder). The stacked 3-D volume is always written to the input/root folder, not here. |
| `stacked_filename` | Output | | Filename of the stacked 3-D strain volume written to the input/root folder. Downstream stages expect stacked_strain_volumes.h5. |
| `save_plots` | Appearance | | Write the per-layer diagnostic PNGs (raw, detrended, strain). Turn off for a faster volume-only run. |

(Essentials: 5 — mode, input_folder, root_folder, roi, output_dir.)

---

### Task 6: mosaicity metadata

**Files:** Modify: `dfxm/stages/mosaicity.py` (STAGE block). Steps: the shared 8-step pattern.

**New description:**
> Stacks the darfix χ (chi) and μ (mu) centre-of-mass and width (FWHM) maps of each layer into one 3-D mosaicity volume. Needs maps.h5 from darfix in each layer folder; writes stacked_volumes.h5.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `mode` | essential | | single processes one layer folder ('Input folder'); batch processes every subfolder of 'Root folder' matching 'Folder pattern'. |
| `input_folder` | essential | yes | Layer folder containing the darfix maps.h5 (single mode only). |
| `root_folder` | essential | yes | Parent of the mosaicity layer folders (batch mode only). |
| `folder_pattern` | Data layout | | Glob selecting the mosaicity layer subfolders, usually the *_mosa__* naming pattern. |
| `maps_filename` | Data layout | | Filename of the darfix output inside each layer folder (normally maps.h5). |
| `chi_com_path` | Data layout | | HDF5 path of the χ centre-of-mass map inside maps.h5 (darfix layout). χ CoM is the local lattice tilt about the rocking axis. |
| `chi_fwhm_path` | Data layout | | HDF5 path of the χ FWHM map — the local rocking-curve width, a measure of mosaic spread. |
| `mu_com_path` | Data layout | | HDF5 path of the μ centre-of-mass map — the local lattice tilt about the second tilt axis. |
| `mu_fwhm_path` | Data layout | | HDF5 path of the μ FWHM map — the local curve width about the second tilt axis. |
| `output_dir` | essential | | Where stacked_volumes.h5 is written (blank = the input/root folder). |
| `stacked_filename` | Output | | Filename of the stacked mosaicity volume. Downstream stages expect stacked_volumes.h5. |
| `compression` | Output | | HDF5 compression for the volume: gzip (small, slower), lzf (fast, larger), none. |

(Essentials: 4 — mode, input_folder, root_folder, output_dir.)

---

### Task 7: visualize metadata

**Files:** Modify: `dfxm/stages/visualize.py` (STAGE block). Steps: the shared 8-step pattern.

**New description:**
> Aligns the stacked mosaicity/strain volumes into the shared sample frame (samy shift + uniform-Z interpolation) and renders per-layer PNGs, a layer animation, and a 3-D top view.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `mosa_volume_file` | essential | yes | The stacked mosaicity volume (stacked_volumes.h5) from the mosaicity stage. Leave blank to skip mosaicity rendering. |
| `strain_volume_file` | essential | yes | The stacked strain volume (stacked_strain_volumes.h5) from the strain stage. Leave blank to skip strain rendering. |
| `raw_root` | essential | yes | RAW_DATA root with the original scan folders — the samy/samz motor positions read from there drive the alignment. |
| `mosa_pattern` | Data layout | | Glob matching the raw mosaicity scan folders, used to read their samy/samz positions. |
| `strain_pattern` | Data layout | | Glob matching the raw strain scan folders, used to read their samy/samz positions. |
| `samy_path` | Data layout | | HDF5 path to the sample-Y motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `samz_path` | Data layout | | HDF5 path to the sample-Z motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `pixel_size_x_um` | Calibration | | Physical size of one detector pixel along X, in µm — sets the lateral scale of the volumes. From the beamline optics calibration. |
| `pixel_size_y_um` | Calibration | | Physical size of one detector pixel along Y, in µm — sets the vertical scale of the volumes. From the beamline optics calibration. |
| `samy_direction` | Alignment | | Sign (+1 or −1) relating the samy motor direction to detector X. If features visibly march the wrong way between layers, flip the sign. |
| `roi_x` | essential | | Crop along detector X as 'x0,x1' in pixels (blank = full width). All volumes must share the same crop to stay co-registered. |
| `roi_y` | essential | | Crop along detector Y as 'y0,y1' in pixels (blank = full height). All volumes must share the same crop to stay co-registered. |
| `center_method` | Alignment | | How the colour scale of the misorientation (CoM) maps is centred: midrange = midpoint of the robust limits, or mean/median of the data. Display only. |
| `range_pct` | Alignment | | Robust percentile for colour limits, e.g. 99.5 ignores the most extreme 0.5 % of pixels when setting the scale. |
| `output_dir` | essential | | Where the rendered PNGs, animation and top view are written (blank = next to the input volume). |
| `output_format` | Output | | Animation container: mp4 needs ffmpeg on PATH; gif always works; both writes both. |
| `save_layers` | Output | | Write one PNG per layer of each volume. |
| `save_animation` | Output | | Write the layer-by-layer animation. |
| `save_topview` | Output | | Write the static 3-D top-view image. |
| `volume_opacity` | Appearance | | Opacity of the rendered 3-D top view, 0–1. |

(Essentials: 6 — mosa_volume_file, strain_volume_file, raw_root, roi_x, roi_y, output_dir.)

---

### Task 8: rocking metadata

**Files:** Modify: `dfxm/stages/rocking.py` (STAGE block). Steps: the shared 8-step pattern.

**New description:**
> Builds aligned 3-D volumes directly from the raw rocking scans — a background-subtracted intensity sum plus one chosen frame — anchored to the mosaicity reference so they overlay the other volumes. Writes aligned_raw_rocking_volumes.h5 and rendered images.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `raw_root` | essential | yes | RAW_DATA root containing the rocking (and mosaicity/strain) scan folders. |
| `rocking_pattern` | Data layout | | Glob matching the raw rocking scan folders. |
| `mosa_pattern` | Data layout | | Glob for the mosaicity scan folders — they provide the alignment reference (samy/samz origin) and part of the Z range. |
| `strain_pattern` | Data layout | | Glob for the strain scan folders; extends the Z range so the rocking volume covers both (blank = mosaicity range only). |
| `samy_path` | Data layout | | HDF5 path to the sample-Y motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `samz_path` | Data layout | | HDF5 path to the sample-Z motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `detector_path` | Data layout | | HDF5 path to the detector frames inside each rocking scan file. |
| `pixel_size_x_um` | Calibration | | Physical size of one detector pixel along X, in µm. From the beamline optics calibration. |
| `pixel_size_y_um` | Calibration | | Physical size of one detector pixel along Y, in µm. From the beamline optics calibration. |
| `samy_direction` | Alignment | | Sign (+1 or −1) relating the samy motor direction to detector X. If features visibly march the wrong way between layers, flip the sign. |
| `roi_x` | essential | | Detector crop 'x0,x1' in pixels applied while reading frames (blank = full). Match the crop used for the other volumes. |
| `roi_y` | essential | | Detector crop 'y0,y1' in pixels applied while reading frames (blank = full). Match the crop used for the other volumes. |
| `specific_frame_idx` | essential | | 0-based index of the single rocking frame to extract per scan (blank = the central frame of the first scan). Lets you look at one angular position instead of the sum. |
| `samz_tol_mm` | Alignment | | Extra tolerance in mm when deciding which rocking scans fall inside the mosaicity/strain Z range. |
| `normalize_sum` | Alignment | | Divide each summed image by its frame count so intensities are comparable across scans with different numbers of frames. |
| `output_dir` | essential | | Where the aligned volume and rendered media are written (blank = a folder under the raw root). |
| `aligned_h5_name` | Output | | Filename of the aligned rocking volume. The slices stage expects aligned_raw_rocking_volumes.h5. |
| `save_aligned_h5` | Output | | Write the aligned volume file (needed by the slices stage). |
| `save_layers` | Output | | Write one PNG per layer. |
| `save_animation` | Output | | Write the layer-by-layer animation. |
| `save_topview` | Output | | Write the static 3-D top-view image. |
| `output_format` | Output | | Animation container: mp4 needs ffmpeg on PATH; gif always works; both writes both. |
| `volume_opacity` | Appearance | | Opacity of the rendered 3-D top view, 0–1. |
| `cbar_pct_lo` | Appearance | | Lower intensity percentile for the colour scale of the rendered images. |
| `cbar_pct_hi` | Appearance | | Upper intensity percentile for the colour scale of the rendered images. |

(Essentials: 5 — raw_root, roi_x, roi_y, specific_frame_idx, output_dir.)

---

### Task 9: paraview metadata

**Files:** Modify: `dfxm/stages/paraview.py` (STAGE block). Steps: the shared 8-step pattern.

**New description:**
> Aligns the mosaicity/strain volumes and exports them as partitioned .pvti datasets (with a validity mask) for 3-D volume rendering in ParaView, outside this app.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `mosa_volume_file` | essential | yes | The stacked mosaicity volume (stacked_volumes.h5) from the mosaicity stage. Leave blank to skip the mosaicity export. |
| `strain_volume_file` | essential | yes | The stacked strain volume (stacked_strain_volumes.h5) from the strain stage. Leave blank to skip the strain export. |
| `raw_root` | essential | yes | RAW_DATA root with the original scan folders — the samy/samz motor positions read from there drive the alignment. |
| `mosa_pattern` | Data layout | | Glob matching the raw mosaicity scan folders, used to read their samy/samz positions. |
| `strain_pattern` | Data layout | | Glob matching the raw strain scan folders, used to read their samy/samz positions. |
| `samy_path` | Data layout | | HDF5 path to the sample-Y motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `samz_path` | Data layout | | HDF5 path to the sample-Z motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `pixel_size_x_um` | Calibration | | Physical size of one detector pixel along X, in µm — sets the voxel spacing of the export. From the beamline optics calibration. |
| `pixel_size_y_um` | Calibration | | Physical size of one detector pixel along Y, in µm — sets the voxel spacing of the export. From the beamline optics calibration. |
| `samy_direction` | Alignment | | Sign (+1 or −1) relating the samy motor direction to detector X. If features visibly march the wrong way between layers, flip the sign. |
| `roi_x` | essential | | Crop along detector X as 'x0,x1' in pixels (blank = full width). All volumes must share the same crop to stay co-registered. |
| `roi_y` | essential | | Crop along detector Y as 'y0,y1' in pixels (blank = full height). All volumes must share the same crop to stay co-registered. |
| `center_method` | Alignment | | Statistic used to centre the misorientation values before export: mean or median. |
| `center_mosa_com` | Alignment | | Subtract the centre statistic from the χ/μ CoM volumes so misorientation is relative to the bulk orientation. |
| `center_strain` | Alignment | | Also centre the strain volume (usually off — strain is already relative to the reference angle). |
| `abs_mosa_fwhm` | Alignment | | Export FWHM as absolute values (darfix fits can produce negative widths). |
| `anchor_origin_to_reference` | Alignment | | Place the world origin in the raw-detector frame shared with the rocking volume, so everything co-registers in ParaView. |
| `mosa_darfix_origin_xy` | Alignment | | Pixel position 'x,y' of the darfix crop origin for the mosaicity maps, used when anchoring to the reference frame. |
| `strain_darfix_origin_xy` | Alignment | | Pixel position 'x,y' of the darfix crop origin for the strain maps, used when anchoring to the reference frame. |
| `num_pieces_z` | Export | | Number of Z chunks the dataset is split into — match the MPI rank count of your pvserver for parallel rendering. |
| `piece_compression` | Export | | Compress the .vti pieces (smaller files, slower write). |
| `replace_nan` | Export | | Replace NaN padding with a sentinel value so ParaView's volume renderer behaves. |
| `write_valid_mask` | Export | | Write a 0/1 valid_mask field — threshold on it in ParaView to hide the padding. |
| `export_mosaicity` | Export | | Export the mosaicity (χ/μ) volumes. |
| `export_strain` | Export | | Export the strain volume. |
| `output_dir` | essential | | Where the .pvti files and their piece folders are written. |

(Essentials: 6 — mosa_volume_file, strain_volume_file, raw_root, roi_x, roi_y, output_dir.)

---

### Task 10: slices metadata

**Files:** Modify: `dfxm/stages/slices.py` (STAGE block). Steps: the shared 8-step pattern.

**New description:**
> Cuts arbitrary planes — defined in physical µm, optionally swept along their normal — through all aligned volumes at once, so every quantity is sampled at identical positions. Writes oblique_slices.h5 (used by profiles) plus a PNG per plane.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `mosa_volume_file` | essential | yes | The stacked mosaicity volume (stacked_volumes.h5) from the mosaicity stage. Leave blank to skip mosaicity fields. |
| `strain_volume_file` | essential | yes | The stacked strain volume (stacked_strain_volumes.h5) from the strain stage. Leave blank to skip strain. |
| `aligned_rocking_file` | essential | yes | The aligned rocking volume (aligned_raw_rocking_volumes.h5) from the rocking stage. Leave blank to slice without raw intensity. |
| `raw_root` | essential | yes | RAW_DATA root with the original scan folders — provides the samy/samz positions used to align the stacked volumes. |
| `mosa_pattern` | Data layout | | Glob matching the raw mosaicity scan folders, used to read their samy/samz positions. |
| `strain_pattern` | Data layout | | Glob matching the raw strain scan folders, used to read their samy/samz positions. |
| `samy_path` | Data layout | | HDF5 path to the sample-Y motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `samz_path` | Data layout | | HDF5 path to the sample-Z motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `pixel_size_x_um` | Calibration | | Physical size of one detector pixel along X, in µm — sets the physical scale the planes are defined in. From the beamline optics calibration. |
| `pixel_size_y_um` | Calibration | | Physical size of one detector pixel along Y, in µm — sets the physical scale the planes are defined in. From the beamline optics calibration. |
| `samy_direction` | Alignment | | Sign (+1 or −1) relating the samy motor direction to detector X. If features visibly march the wrong way between layers, flip the sign. |
| `align_roi_x` | Alignment | | Detector crop 'x0,x1' used during alignment — must match the crop used when the volumes were rendered/exported. |
| `align_roi_y` | Alignment | | Detector crop 'y0,y1' used during alignment — must match the crop used when the volumes were rendered/exported. |
| `abs_fwhm` | Alignment | | Use absolute FWHM values (darfix fits can produce negative widths). |
| `center_method` | Alignment | | How the colour scale of the misorientation (CoM) fields is centred: midrange = midpoint of the robust limits, or mean/median. Display only. |
| `range_pct` | Alignment | | Robust percentile for colour limits, e.g. 99.5 ignores the most extreme 0.5 % of pixels. |
| `include_mosa_com_chi` | Quantities | | Slice the χ misorientation (centre-of-mass) volume. |
| `include_mosa_fwhm_chi` | Quantities | | Slice the χ FWHM (rocking-curve width) volume. |
| `include_mosa_com_mu` | Quantities | | Slice the μ misorientation (centre-of-mass) volume. |
| `include_mosa_fwhm_mu` | Quantities | | Slice the μ FWHM (curve width) volume. |
| `include_strain` | Quantities | | Slice the axial strain volume. |
| `include_raw_sum` | Quantities | | Slice the summed raw rocking intensity volume. |
| `include_raw_specific` | Quantities | | Slice the specific-frame raw intensity volume. |
| `slices_json` | essential | | JSON list of plane definitions. Each needs a name and a normal vector; 'extent': 'auto' fits the plane to the data, and 'sweep_step_um' adds parallel planes along the normal. The default shows the format. |
| `output_dir` | essential | | Where oblique_slices.h5 and the per-plane PNGs are written. |
| `output_h5_name` | Output | | Filename of the consolidated slices file. The profiles stage expects oblique_slices.h5. |
| `save_png` | Output | | Write a PNG per plane in addition to the HDF5. |

(Essentials: 6 — mosa_volume_file, strain_volume_file, aligned_rocking_file, raw_root, slices_json, output_dir.)

---

### Task 11: profiles metadata

**Files:** Modify: `dfxm/stages/profiles.py` (STAGE block). Steps: the shared 8-step pattern.

**New description:**
> Draws 1-D line profiles across a slice plane — every field is sampled at the same in-plane positions, so intensity, strain and misorientation line up point by point. Writes a stacked figure plus CSVs. Use 'Pick line…' to choose the line by clicking on the plane.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `consolidated_h5` | essential | yes | The oblique_slices.h5 file written by the slices stage. |
| `mode` | essential | | 'parameter' runs the jobs below and saves figures/CSVs (reproducible); 'preview' just displays the plane so you can inspect it. |
| `reference_volume_id` | Selection | | Which field is shown as the top image of the figure (blank = raw_sum if present, else the first field). |
| `volume_ids` | Selection | | Comma-separated field ids to profile, in this order (blank = all fields). |
| `jobs_json` | essential | | JSON list of profile jobs: slice name, plane offset, line start/end in µm ('start_uv'/'end_uv'), and band width in pixels. Easiest filled by 'Pick line…'. |
| `save_csv` | Output | | Write one CSV per profiled field. |
| `save_overview` | Output | | Write per-field overview images with the profile line drawn on the plane. |
| `line_color` | Appearance | | Colour of the profile line drawn on the overview images (blank = automatic per colormap). |
| `geom_tol_um` | Matching | | Maximum allowed geometry mismatch between fields sharing a plane, in µm — guards against profiling mis-registered slices. |
| `offset_tol_um` | Matching | | Tolerance when matching the requested plane offset to the stored planes, in µm. |
| `fig_dpi` | Appearance | | Resolution of the saved figures, in dots per inch. |
| `output_dir` | essential | | Where the figures and CSVs are written (blank = next to the slices file). |

(Essentials: 4 — consolidated_h5, mode, jobs_json, output_dir.)

---

### Task 12: matched metadata

**Files:** Modify: `dfxm/stages/matched.py` (STAGE block). Steps: the shared 8-step pattern.

**New description:**
> For each strain layer, finds the rocking scan taken at the same (samy, samz) sample position and saves one background-subtracted detector frame as a grayscale PNG, pixel-aligned with the strain/mosaicity layer images.

| param | advanced/group | must_exist | help |
|---|---|---|---|
| `raw_root` | essential | yes | RAW_DATA root containing both the strain and the rocking scan folders. |
| `strain_pattern` | Data layout | | Glob matching the strain scan folders (one per layer). |
| `rocking_pattern` | Data layout | | Glob matching the rocking scan folders to search for position matches. |
| `samy_path` | Data layout | | HDF5 path to the sample-Y motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `samz_path` | Data layout | | HDF5 path to the sample-Z motor position inside each scan file (under the first BLISS entry). Only change for a different beamline file layout. |
| `pco_ff_path` | Data layout | | HDF5 path to the detector frames inside each rocking scan file. |
| `frame_index` | essential | | 0-based detector frame to extract from each matched rocking scan. |
| `match_threshold_mm` | essential | | Maximum (samy, samz) distance in mm for a rocking scan to count as matching a strain layer; layers with no scan inside the threshold are skipped. |
| `pixel_size_x_um` | Calibration | | Physical size of one detector pixel along X, in µm. From the beamline optics calibration. |
| `pixel_size_y_um` | Calibration | | Physical size of one detector pixel along Y, in µm. From the beamline optics calibration. |
| `samy_direction` | Alignment | | Sign (+1 or −1) relating the samy motor direction to detector X — the same shift convention as the strain layers. |
| `colormap` | Appearance | | Matplotlib colormap for the saved PNGs (default gray). |
| `vmin` | Appearance | | Lower intensity limit (blank = the automatic percentile below). |
| `vmax` | Appearance | | Upper intensity limit (blank = the automatic percentile below). |
| `auto_pct_lo` | Appearance | | Percentile used for the automatic lower intensity limit. |
| `auto_pct_hi` | Appearance | | Percentile used for the automatic upper intensity limit. |
| `output_dir` | essential | | Where the matched layer PNGs are written. |

(Essentials: 4 — raw_root, frame_index, match_threshold_mm, output_dir.)

---

### Task 13: experiment-schema help + commit the enforcement tests

**Files:**
- Modify: `dfxm/config/models.py:167-207` (`EXPERIMENT_SCHEMA`)
- Commit: `tests/test_param_metadata.py`

- [ ] **Step 1: Confirm the only remaining red is the experiment schema**

Run: `python3 -m pytest tests/test_param_metadata.py -q`
Expected: only `test_experiment_schema_has_help` FAILS (all 9 stages green from Tasks 4-12).

- [ ] **Step 2: Set `help` on every `EXPERIMENT_SCHEMA` param**

Keep every existing `name`/`type`/`label`/`unit`/`calibration` argument; set these `help` texts (replacing the existing shorter ones):

| param | help |
|---|---|
| `name` | Short name of the preset (used as the filename when saving). |
| `description` | One line describing the experiment/sample. |
| `notes` | Free-text caveats shown in red in the GUI — e.g. calibration warnings for whoever loads this preset. |
| `raw_root` | RAW_DATA root: the folder with the original beamline scan folders (input to concat, rocking and matched). |
| `processed_root` | PROCESSED_DATA root: where darfix wrote maps.h5 per layer (input to strain/mosaicity; the stacked volumes land here too). |
| `folder_pattern` | Glob for the concat/strain layer subfolders. |
| `mosa_pattern` | Glob for the mosaicity layer subfolders (often *_mosa__*). |
| `rocking_pattern` | Glob for the rocking scan subfolders (often *_rocking__*). |
| `entry_suffix` | BLISS entry filter for concat — only entries ending in this suffix are merged (e.g. '.1'). |
| `ccmth_ref_deg` | Reference Bragg angle of the unstrained lattice, in degrees. Strain is computed from deviations from this angle — a wrong value silently shifts every strain map. From the beamline alignment. |
| `pixel_size_x_um` | Detector pixel size along X in µm, from the beamline optics calibration. Sets the physical scale of every map. |
| `pixel_size_y_um` | Detector pixel size along Y in µm, from the beamline optics calibration. Sets the physical scale of every map. |
| `maps_filename` | Filename darfix writes inside each layer folder (normally maps.h5). |
| `positioners_path` | HDF5 path of the motor-position group inside each scan entry. |
| `detector_read_path` | HDF5 path of the raw detector frames inside each scan entry. |
| `detector_write_path` | HDF5 path where concat writes the merged detector data. |
| `samy_key` | Name of the sample-Y translation motor under the positioners group. |
| `samz_key` | Name of the sample-Z translation motor under the positioners group. |
| `ccmth_com_path` | HDF5 path of the ccmth centre-of-mass dataset inside maps.h5. |

- [ ] **Step 3: Full verification**

Run: `python3 -m pytest -q && ruff check . && ruff format --check .`
Expected: everything green, including all of `test_param_metadata.py` and the pre-existing `test_config.py` schema-sync test.

- [ ] **Step 4: Commit (the enforcement test file finally lands)**

```bash
git add dfxm/config/models.py tests/test_param_metadata.py
git commit -m "config: experiment-schema help texts; enforce help/group/essentials metadata"
```

---

### Task 14: `ParamForm` — essentials + grouped Advanced expander + focus signal

GUI tasks are verified through `tests/gui_smoke.py` (offscreen, run as a script —
**never** rename it into pytest collection). TDD shape: add the smoke assertions
first, watch the script fail, implement, watch it pass.

**Files:**
- Modify: `gui/widgets/param_form.py`
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add failing smoke assertions**

In `tests/gui_smoke.py`, after the current step `[6]` block (the cancel test), insert:

```python
    # Forms: essentials visible, Advanced expander collapsed, values round-trip.
    for name, view in win._views.items():
        form = view._form
        spec = view._spec
        assert set(form.values()) == {p.name for p in spec.params}, name
        n_adv = sum(1 for p in spec.params if p.advanced)
        if n_adv:
            assert form._adv_toggle is not None, name
            assert f"({n_adv} settings)" in form._adv_toggle.text(), name
            assert not form._adv_box.isVisible(), name
    # focus_param reveals an advanced field
    sform = win._views["strain"]._form
    sform._adv_toggle.setChecked(False)
    sform.focus_param("ccmth_ref_deg")
    assert sform._adv_toggle.isChecked()
    print("[7] grouped forms: essentials/advanced split + value round-trip OK")
```

(Existing prints `[6]` stay; new steps number onward from 7.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL with `AttributeError: 'ParamForm' object has no attribute '_adv_toggle'`.

- [ ] **Step 3: Rewrite `ParamForm.__init__` and add the focus machinery**

In `gui/widgets/param_form.py`:

Imports — add to the `PySide6.QtCore` import: `QEvent, QObject, Qt, Signal`; add to the
`PySide6.QtWidgets` import: `QToolButton, QVBoxLayout`.

Replace the class-level signal block and `__init__` with:

```python
class ParamForm(QWidget):
    """A form whose rows are generated from a parameter schema.

    Essentials (``advanced=False``) render first, in spec order; advanced
    params collapse into one "Advanced (N settings)" expander, grouped under
    their ``group`` headers. Use :meth:`values` to read coerced values and
    :meth:`set_values` to load a dict back into the widgets.
    """

    changed = Signal()
    focusedParamChanged = Signal(object)  # the focused Param

    def __init__(
        self,
        params: Sequence[Param],
        values: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._params = list(params)
        self._getters: dict[str, Callable[[], Any]] = {}
        self._setters: dict[str, Callable[[Any], None]] = {}
        self._editors: dict[str, QWidget] = {}
        self._param_for_widget: dict[QObject, Param] = {}
        self._param_by_name: dict[str, Param] = {p.name: p for p in self._params}
        self._adv_toggle: QToolButton | None = None
        self._adv_box: QWidget | None = None

        initial = values or {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        essentials = [p for p in self._params if not p.advanced]
        advanced = [p for p in self._params if p.advanced]

        ess_form = QFormLayout()
        for p in essentials:
            ess_form.addRow(self._label_for(p), self._make_editor(p, initial))
        outer.addLayout(ess_form)

        if advanced:
            self._adv_toggle = QToolButton()
            self._adv_toggle.setText(f"Advanced ({len(advanced)} settings)")
            self._adv_toggle.setCheckable(True)
            self._adv_toggle.setArrowType(Qt.ArrowType.RightArrow)
            self._adv_toggle.setToolButtonStyle(
                Qt.ToolButtonStyle.ToolButtonTextBesideIcon
            )
            self._adv_box = QWidget()
            adv_layout = QVBoxLayout(self._adv_box)
            adv_layout.setContentsMargins(12, 0, 0, 0)
            group_forms: dict[str, QFormLayout] = {}
            for p in advanced:
                form = group_forms.get(p.group)
                if form is None:
                    header = QLabel(p.group)
                    header.setStyleSheet("font-weight: bold; margin-top: 6px;")
                    adv_layout.addWidget(header)
                    form = QFormLayout()
                    adv_layout.addLayout(form)
                    group_forms[p.group] = form
                form.addRow(self._label_for(p), self._make_editor(p, initial))
            self._adv_box.setVisible(False)
            self._adv_toggle.toggled.connect(self._on_adv_toggled)
            outer.addWidget(self._adv_toggle)
            outer.addWidget(self._adv_box)

    def _make_editor(self, p: Param, initial: dict[str, Any]) -> QWidget:
        editor = self._build_editor(p, initial.get(p.name, p.default))
        self._editors[p.name] = editor
        for w in (editor, *editor.findChildren(QWidget)):
            w.installEventFilter(self)
            self._param_for_widget[w] = p
        return editor

    def _on_adv_toggled(self, checked: bool) -> None:
        assert self._adv_toggle is not None and self._adv_box is not None
        self._adv_box.setVisible(checked)
        self._adv_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def eventFilter(self, obj: QObject, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.FocusIn and obj in self._param_for_widget:
            self.focusedParamChanged.emit(self._param_for_widget[obj])
        return super().eventFilter(obj, event)

    def focus_param(self, name: str) -> None:
        """Reveal (if advanced) and focus the editor for *name*."""
        editor = self._editors.get(name)
        param = self._param_by_name.get(name)
        if editor is None or param is None:
            return
        if param.advanced and self._adv_toggle is not None:
            self._adv_toggle.setChecked(True)
        target = editor.findChild(QLineEdit) or editor
        target.setFocus()
```

Everything else in the file (`values`, `set_values`, `_label_for`, `_build_editor`,
`_register`, the per-type editor builders) stays exactly as it is.

- [ ] **Step 4: Run smoke + suite**

Run: `python3 tests/gui_smoke.py && python3 -m pytest -q && ruff check .`
Expected: smoke prints through `[7] grouped forms…` and `GUI SMOKE PASSED`; pytest green.

- [ ] **Step 5: Commit**

```bash
git add gui/widgets/param_form.py tests/gui_smoke.py
git commit -m "gui: ParamForm essentials + grouped Advanced expander, focus tracking"
```

---

### Task 15: `HelpPanel` + wiring into `StageView`

**Files:**
- Create: `gui/widgets/help_panel.py`
- Modify: `gui/stage_view.py` (left column)
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add failing smoke assertions**

After the `[7]` block in `tests/gui_smoke.py`:

```python
    # Help panel follows focus and idles on the stage description.
    sview = win._views["strain"]
    assert "strain" in sview._help._label.text().lower()
    sview._form.focus_param("ccmth_ref_deg")
    app.processEvents()
    help_text = sview._help._label.text()
    assert "Bragg" in help_text and "calibration" in help_text.lower()
    print("[8] help panel idles on description and follows focus")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL with `AttributeError: 'StageView' object has no attribute '_help'`.

- [ ] **Step 3: Create `gui/widgets/help_panel.py`**

```python
"""Focus-following help panel for parameter forms.

Shows the focused :class:`~dfxm.config.models.Param`'s label, unit,
calibration warning and full help text; idles on the stage description when
nothing is focused. Connect :attr:`ParamForm.focusedParamChanged` to
:meth:`show_param`.
"""

from __future__ import annotations

import html

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from dfxm.config.models import Param

_CAL_WARNING = (
    '<span style="color:#b00020;">⚠ calibration — physically meaningful; '
    "confirm against the beamline calibration for your experiment.</span>"
)


class HelpPanel(QFrame):
    """Styled read-only box explaining the focused parameter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(
            "HelpPanel { background: #eef2fb; border-left: 3px solid #4a6fd0; }"
        )
        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._idle_html = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self._label)

    def set_idle(self, title: str, description: str) -> None:
        """Set (and show) the text used when no field is focused."""
        self._idle_html = f"<b>{html.escape(title)}</b> — {html.escape(description)}"
        self._label.setText(self._idle_html)

    def show_idle(self) -> None:
        self._label.setText(self._idle_html)

    def show_param(self, p: Param) -> None:
        head = f"<b>{html.escape(p.label)}</b>"
        if p.unit:
            head += f" ({html.escape(p.unit)})"
        parts = [head]
        if p.calibration:
            parts.append(_CAL_WARNING)
        if p.help:
            parts.append(html.escape(p.help))
        self._label.setText("<br>".join(parts))
```

- [ ] **Step 4: Wire it into `StageView`**

In `gui/stage_view.py`:

- Import: `from .widgets.help_panel import HelpPanel`.
- In `__init__`, the left column currently does:

```python
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._form)
        left_layout.addLayout(btn_row)
        left_layout.addStretch(1)
```

Change to:

```python
        self._help = HelpPanel()
        self._help.set_idle(spec.label, spec.description)
        self._form.focusedParamChanged.connect(self._help.show_param)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._form)
        left_layout.addLayout(btn_row)
        left_layout.addWidget(self._help)
        left_layout.addStretch(1)
```

- The form column can now exceed the window height for big stages: wrap the left
  widget in a scroll area when adding it to the splitter. Replace
  `splitter.addWidget(left)` with:

```python
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)
```

(`QScrollArea` is already imported in this file.)

- [ ] **Step 5: Run smoke + suite, commit**

Run: `python3 tests/gui_smoke.py && python3 -m pytest -q && ruff check .`
Expected: prints through `[8] …` and `GUI SMOKE PASSED`.

```bash
git add gui/widgets/help_panel.py gui/stage_view.py tests/gui_smoke.py
git commit -m "gui: focus-following HelpPanel under every stage form"
```

---

### Task 16: compact experiment header + Edit dialog

**Files:**
- Modify: `gui/experiment_panel.py` (full rework)
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add failing smoke assertions**

After the `[8]` block:

```python
    # Compact experiment header: summary line + notes + Edit dialog.
    panel = win._experiment_panel
    assert "7.144" in panel._summary.text()
    assert panel._notes.isVisible()
    dlg = panel._make_dialog()
    dlg.show()
    app.processEvents()
    vals = dlg._form.values()
    assert vals["ccmth_ref_deg"] == 7.144
    dlg._form.set_values({"description": "smoke-edited"})
    dlg.accept()
    panel._set_experiment(dlg.experiment())  # what _on_edit does after exec()
    app.processEvents()
    assert panel.current_experiment().description == "smoke-edited"
    print("[9] compact experiment header + edit dialog round-trip")
```

(The smoke test drives the dialog non-modally — `exec()` would block offscreen —
so it applies the result the same way `_on_edit` does.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL with `AttributeError: 'ExperimentPanel' object has no attribute '_summary'`.

- [ ] **Step 3: Rework `gui/experiment_panel.py`**

Replace the module with:

```python
"""Compact experiment header + modal editor dialog.

The header shows: preset dropdown, a one-line calibration summary, the
preset's notes (red, when present), and an **Edit…** button that opens
:class:`ExperimentDialog` — the full schema-driven field form (with help
panel) plus **Save as…**. Emits :attr:`experimentChanged` whenever the
active experiment changes so the stage views re-pull their defaults.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from dfxm.config import presets
from dfxm.config.models import EXPERIMENT_SCHEMA, Experiment

from .widgets.help_panel import HelpPanel
from .widgets.param_form import ParamForm


class ExperimentDialog(QDialog):
    """Modal editor for every experiment field (schema-driven)."""

    def __init__(self, experiment: Experiment, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit experiment")
        self.resize(560, 640)

        self._form = ParamForm(EXPERIMENT_SCHEMA, experiment.to_dict())
        help_panel = HelpPanel()
        help_panel.set_idle(
            "Experiment",
            "Shared state every stage inherits: data roots, folder patterns, "
            "calibration constants and beamline HDF5 paths.",
        )
        self._form.focusedParamChanged.connect(help_panel.show_param)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._form)

        save_btn = QPushButton("Save as…")
        save_btn.setToolTip("Write the edited fields to a new preset YAML")
        save_btn.clicked.connect(self._on_save_as)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.addButton(save_btn, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        layout.addWidget(help_panel)
        layout.addWidget(buttons)

    def experiment(self) -> Experiment:
        return Experiment.from_dict(self._form.values())

    def _on_save_as(self) -> None:
        exp = self.experiment()
        start = os.fspath(presets.experiments_dir() / f"{exp.name or 'experiment'}.yaml")
        path, _ = QFileDialog.getSaveFileName(self, "Save preset", start, "YAML (*.yaml)")
        if path:
            presets.save_experiment(exp, path)


class ExperimentPanel(QWidget):
    """Preset dropdown + calibration summary + notes + Edit… dialog."""

    experimentChanged = Signal(Experiment)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._presets: dict[str, object] = {}
        self._experiment = Experiment()

        self._combo = QComboBox()
        self._combo.currentTextChanged.connect(self._on_select)
        reload_btn = QPushButton("↻")
        reload_btn.setToolTip("Rescan the experiments/ folder")
        reload_btn.setFixedWidth(32)
        reload_btn.clicked.connect(self._reload_presets)
        edit_btn = QPushButton("Edit…")
        edit_btn.setToolTip("Open the full experiment editor")
        edit_btn.clicked.connect(self._on_edit)

        top = QHBoxLayout()
        top.addWidget(QLabel("Experiment:"))
        top.addWidget(self._combo, 1)
        top.addWidget(reload_btn)
        top.addWidget(edit_btn)

        self._summary = QLabel("")
        self._summary.setStyleSheet("color: #666;")
        self._summary.setWordWrap(True)

        self._notes = QLabel("")
        self._notes.setWordWrap(True)
        self._notes.setStyleSheet("color: #b00020; font-style: italic;")
        self._notes.setVisible(False)

        layout = QVBoxLayout(self)
        layout.addLayout(top)
        layout.addWidget(self._summary)
        layout.addWidget(self._notes)

        self._reload_presets()

    # -- presets ----------------------------------------------------------
    def _reload_presets(self) -> None:
        self._presets = presets.discover_experiments()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(list(self._presets.keys()))
        self._combo.blockSignals(False)
        if self._presets:
            # Loads the first preset and emits experimentChanged.
            self._on_select(self._combo.currentText())

    def _on_select(self, name: str) -> None:
        if not name or name not in self._presets:
            return
        self._set_experiment(presets.load_experiment(self._presets[name]))

    # -- editing ----------------------------------------------------------
    def current_experiment(self) -> Experiment:
        return self._experiment

    def _make_dialog(self) -> ExperimentDialog:
        return ExperimentDialog(self._experiment, self)

    def _on_edit(self) -> None:
        dlg = self._make_dialog()
        if dlg.exec():
            self._set_experiment(dlg.experiment())
            self._reload_combo_keep_selection()

    def _reload_combo_keep_selection(self) -> None:
        """Pick up presets Save-as may have written, without re-emitting."""
        current = self._combo.currentText()
        self._presets = presets.discover_experiments()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(list(self._presets.keys()))
        if current in self._presets:
            self._combo.setCurrentText(current)
        self._combo.blockSignals(False)

    def _set_experiment(self, exp: Experiment) -> None:
        self._experiment = exp
        self._summary.setText(
            f"ccmth {exp.ccmth_ref_deg:g}° · "
            f"{exp.pixel_size_x_um:g}×{exp.pixel_size_y_um:g} µm/px"
        )
        notes = (exp.notes or "").strip()
        self._notes.setText(f"⚠ {notes}" if notes else "")
        self._notes.setVisible(bool(notes))
        self.experimentChanged.emit(exp)
```

- [ ] **Step 4: Run smoke + suite, commit**

Run: `python3 tests/gui_smoke.py && python3 -m pytest -q && ruff check .`
Expected: prints through `[9] …` and `GUI SMOKE PASSED`. (The existing `[2]` assertion on `panel._notes.isVisible()` must still pass.)

```bash
git add gui/experiment_panel.py tests/gui_smoke.py
git commit -m "gui: compact experiment header with modal schema-driven Edit dialog"
```

---

### Task 17: pipeline rail + Overview page

**Files:**
- Create: `gui/overview_page.py`
- Modify: `gui/main_window.py`
- Modify: `gui/stage_view.py` (add `runStarted` signal)
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add failing smoke assertions**

After the `[9]` block:

```python
    # Pipeline rail: Overview first, darfix row disabled, concat optional.
    from PySide6.QtCore import Qt as _Qt

    nav = win._nav
    assert nav.item(0).text().endswith("Overview")
    texts = [nav.item(i).text() for i in range(nav.count())]
    darfix_rows = [i for i, t in enumerate(texts) if "darfix" in t]
    assert len(darfix_rows) == 1
    assert nav.item(darfix_rows[0]).flags() == _Qt.ItemFlag.NoItemFlags
    concat_row = next(i for i, t in enumerate(texts) if "Concatenate" in t)
    assert "(optional)" in texts[concat_row]
    assert darfix_rows[0] == concat_row + 1
    # Overview is the landing page; chips navigate.
    assert win._stack.currentWidget() is win._overview
    win._overview.stageSelected.emit("strain")
    app.processEvents()
    assert win._stack.currentWidget() is win._views["strain"]
    # Status glyphs survived the runs from steps [3]/[4].
    assert win._status_items["concat"].text().startswith("✓")
    assert win._status_items["strain"].text().startswith("✓")
    print("[10] pipeline rail + overview page wired")
```

Also update the **existing** step `[3]`/`[4]` assertions: they already use
`win._status_items[...]` — keep them working by preserving that attribute
(the rail keeps a name→item dict under the same name).

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL with `AttributeError: 'MainWindow' object has no attribute '_overview'` (or the Overview-row assert).

- [ ] **Step 3: Create `gui/overview_page.py`**

```python
"""Landing page: the pipeline drawn as clickable chips + stage descriptions.

Default screen on launch. Shows the stage order left-to-right (darfix as a
dashed external step, concat marked optional), one newcomer-friendly
sentence per stage from each :class:`StageSpec.description`, and a
per-session status recap. Clicking a chip (or a row) jumps to that stage.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from dfxm.config.models import StageSpec

_CHIP_STYLE = (
    "QPushButton { border: 1px solid #aab; border-radius: 10px; "
    "padding: 3px 10px; background: #eef; }"
)
_EXTERNAL_STYLE = "border: 1px dashed #999; border-radius: 10px; padding: 3px 10px; color: #666;"


class OverviewPage(QWidget):
    """Pipeline chips + per-stage description list + status recap."""

    stageSelected = Signal(str)

    def __init__(
        self,
        order: tuple[str, ...],
        specs: dict[str, StageSpec],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._status_labels: dict[str, QLabel] = {}

        title = QLabel("<h2>DFXM pipeline — overview</h2>")
        intro = QLabel(
            "Run the stages top to bottom. <b>Concat is optional</b> — skip it if "
            "your scans are already concatenated. <b>darfix runs outside this "
            "app</b>, between concat and the map stages: it fits the rocking "
            "curves into the maps.h5 files that strain and mosaicity consume."
        )
        intro.setWordWrap(True)

        chips = QHBoxLayout()
        for i, name in enumerate(order):
            if i:
                chips.addWidget(QLabel("→"))
            label = specs[name].label + (" (optional)" if name == "concat" else "")
            btn = QPushButton(label)
            btn.setStyleSheet(_CHIP_STYLE)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.clicked.connect(lambda _=False, n=name: self.stageSelected.emit(n))
            chips.addWidget(btn)
            if name == "concat":
                chips.addWidget(QLabel("→"))
                ext = QLabel("darfix (external)")
                ext.setStyleSheet(_EXTERNAL_STYLE)
                ext.setToolTip(
                    "Run darfix outside this app: it turns the concatenated .h5 "
                    "into the maps.h5 files used by strain and mosaicity."
                )
                chips.addWidget(ext)
        chips.addStretch(1)
        chips_host = QWidget()
        chips_host.setLayout(chips)
        chips_scroll = QScrollArea()
        chips_scroll.setWidgetResizable(True)
        chips_scroll.setWidget(chips_host)
        chips_scroll.setFixedHeight(64)

        rows = QVBoxLayout()
        for name in order:
            status = QLabel("—")
            status.setFixedWidth(18)
            self._status_labels[name] = status
            text = QLabel(f"<b>{specs[name].label}</b> — {specs[name].description}")
            text.setWordWrap(True)
            row = QHBoxLayout()
            row.addWidget(status)
            row.addWidget(text, 1)
            rows.addLayout(row)
        rows.addStretch(1)
        rows_host = QWidget()
        rows_host.setLayout(rows)
        rows_scroll = QScrollArea()
        rows_scroll.setWidgetResizable(True)
        rows_scroll.setWidget(rows_host)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(intro)
        layout.addWidget(chips_scroll)
        layout.addWidget(rows_scroll, 1)

    def set_status(self, name: str, glyph: str) -> None:
        label = self._status_labels.get(name)
        if label is not None:
            label.setText(glyph)
```

- [ ] **Step 4: Add `runStarted` to `StageView`**

In `gui/stage_view.py`, next to the existing signal:

```python
    runFinished = Signal(str, bool)  # (stage_name, ok)
    runStarted = Signal(str)  # stage_name
```

and in `_on_run`, immediately after `self._set_running(True)`:

```python
        self.runStarted.emit(self._stage_name)
```

- [ ] **Step 5: Rework `gui/main_window.py`**

Replace the module body (keep the docstring spirit, update it) with:

```python
"""Main window: compact experiment header + pipeline rail + stage stack.

The left column holds the :class:`~gui.experiment_panel.ExperimentPanel`
(compact header) above a single *pipeline rail*: Overview first, then the
stages in pipeline order with a status glyph each. darfix appears as a
disabled external row after concat; concat is marked optional. The right
side is a stacked set of :class:`~gui.stage_view.StageView` panels behind
an :class:`~gui.overview_page.OverviewPage` landing page.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from dfxm.config.models import Experiment

from .bindings import STAGE_ORDER, STAGE_SPECS
from .experiment_panel import ExperimentPanel
from .overview_page import OverviewPage
from .stage_view import StageView

_GLYPH_IDLE = "—"
_GLYPH_RUNNING = "▶"
_GLYPH_OK = "✓"
_GLYPH_FAIL = "✗"


class MainWindow(QMainWindow):
    """Top-level window wiring experiment + rail + overview + stages."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DFXM pipeline")
        self.resize(1100, 720)

        self._experiment_panel = ExperimentPanel()
        experiment = self._experiment_panel.current_experiment()

        # Stage views + overview page (stacked).
        self._stack = QStackedWidget()
        self._overview = OverviewPage(STAGE_ORDER, STAGE_SPECS)
        self._overview.stageSelected.connect(self._show_stage)
        self._stack.addWidget(self._overview)
        self._views: dict[str, StageView] = {}
        for name in STAGE_ORDER:
            view = StageView(name, STAGE_SPECS[name], experiment)
            view.runStarted.connect(self._on_run_started)
            view.runFinished.connect(self._on_run_finished)
            self._views[name] = view
            self._stack.addWidget(view)

        # Pipeline rail: one list = navigation + status.
        self._nav = QListWidget()
        self._status_items: dict[str, QListWidgetItem] = {}
        self._item_base: dict[str, str] = {}
        self._row_target: list[str | None] = []

        overview_item = QListWidgetItem("☰  Overview")
        self._nav.addItem(overview_item)
        self._row_target.append("__overview__")
        muted = QBrush(QColor("#888888"))
        for i, name in enumerate(STAGE_ORDER, start=1):
            label = STAGE_SPECS[name].label
            base = f"{i} {label}" + (" (optional)" if name == "concat" else "")
            item = QListWidgetItem(f"{_GLYPH_IDLE}  {base}")
            if name == "concat":
                item.setForeground(muted)
            self._nav.addItem(item)
            self._row_target.append(name)
            self._status_items[name] = item
            self._item_base[name] = base
            if name == "concat":
                darfix = QListWidgetItem("    ⤷ darfix (external)")
                darfix.setFlags(Qt.ItemFlag.NoItemFlags)
                darfix.setToolTip(
                    "Run darfix outside this app: it turns the concatenated .h5 "
                    "into the maps.h5 files used by strain and mosaicity."
                )
                self._nav.addItem(darfix)
                self._row_target.append(None)

        self._nav.currentRowChanged.connect(self._on_row_changed)
        self._nav.setCurrentRow(0)  # land on Overview

        self._experiment_panel.experimentChanged.connect(self._on_experiment_changed)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._experiment_panel)
        left_layout.addWidget(self._nav, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(self._stack)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 720])
        self.setCentralWidget(splitter)

    # -- navigation ---------------------------------------------------------
    def _on_row_changed(self, row: int) -> None:
        if not 0 <= row < len(self._row_target):
            return
        target = self._row_target[row]
        if target == "__overview__":
            self._stack.setCurrentWidget(self._overview)
        elif target is not None:
            self._stack.setCurrentWidget(self._views[target])

    def _show_stage(self, name: str) -> None:
        view = self._views.get(name)
        if view is None:
            return
        self._nav.setCurrentRow(self._row_target.index(name))
        self._stack.setCurrentWidget(view)

    # -- slots ----------------------------------------------------------------
    def _on_experiment_changed(self, experiment: Experiment) -> None:
        for view in self._views.values():
            view.set_experiment(experiment)

    def _set_glyph(self, stage_name: str, glyph: str) -> None:
        item = self._status_items.get(stage_name)
        if item is not None:
            item.setText(f"{glyph}  {self._item_base[stage_name]}")
        self._overview.set_status(stage_name, glyph)

    def _on_run_started(self, stage_name: str) -> None:
        self._set_glyph(stage_name, _GLYPH_RUNNING)

    def _on_run_finished(self, stage_name: str, ok: bool) -> None:
        self._set_glyph(stage_name, _GLYPH_OK if ok else _GLYPH_FAIL)
```

- [ ] **Step 6: Run smoke + suite, commit**

Run: `python3 tests/gui_smoke.py && python3 -m pytest -q && ruff check .`
Expected: prints through `[10] …` and `GUI SMOKE PASSED`. The pre-existing
assertions `[2]`–`[5]` must still hold (`_status_items` glyph text still starts
with ✓ after a run; `_views` unchanged).

```bash
git add gui/overview_page.py gui/main_window.py gui/stage_view.py tests/gui_smoke.py
git commit -m "gui: pipeline rail with darfix/optional-concat rows + Overview landing page"
```

---

### Task 18: status banner, pre-run validation, progress bar

**Files:**
- Modify: `gui/stage_view.py`
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add failing smoke assertions**

After the `[10]` block in `tests/gui_smoke.py`:

```python
    # Success banner from the earlier strain run; progress completed.
    assert sview._banner.isVisible() and sview._banner.text().startswith("✓")
    assert sview._progress.value() == 100
    # Pre-run validation blocks on a missing must_exist path (no child process).
    mview = win._views["mosaicity"]
    mview._form.set_values({"mode": "batch", "root_folder": "/nonexistent/nowhere"})
    mview._on_run()
    app.processEvents()
    assert mview._banner.isVisible()
    assert "/nonexistent/nowhere" in mview._banner.text()
    assert mview._runner is None  # blocked before launch
    # A real failing run shows the red banner with the error text.
    mview._form.set_values({"root_folder": ""})  # empty: passes must_exist, fails in-stage
    mdone: list[bool] = []
    mview.runFinished.connect(lambda name, ok: mdone.append(ok))
    mview._on_run()
    t0 = time.time()
    while not mdone and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.02)
    assert mdone == [False]
    assert mview._banner.isVisible()
    assert "root_folder" in mview._banner.text()
    assert win._status_items["mosaicity"].text().startswith("✗")
    print("[11] banner + pre-run validation + progress bar")
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL with `AttributeError: 'StageView' object has no attribute '_banner'`.

- [ ] **Step 3: Implement in `gui/stage_view.py`**

Imports: add `import html` next to `import os`; add `QProgressBar` to the
`PySide6.QtWidgets` import list.

In `__init__`, after the `btn_row` block and before the help panel (Task 15),
add the progress row:

```python
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress_text = QLabel("")
        self._progress_text.setWordWrap(True)
        progress_row = QHBoxLayout()
        progress_row.addWidget(self._progress, 1)
        progress_row.addWidget(self._progress_text, 2)
```

and in the left column layout, insert it between the buttons and the help panel:

```python
        left_layout.addWidget(self._form)
        left_layout.addLayout(btn_row)
        left_layout.addLayout(progress_row)
        left_layout.addWidget(self._help)
        left_layout.addStretch(1)
```

Create the banner and wrap the right side. Where `__init__` currently does
`splitter.addWidget(left_scroll)` followed by `splitter.addWidget(self._tabs)`,
build a container instead:

```python
        self._banner = QLabel("")
        self._banner.setWordWrap(True)
        self._banner.setTextFormat(Qt.TextFormat.RichText)
        self._banner.setVisible(False)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._banner)
        right_layout.addWidget(self._tabs, 1)

        splitter = QSplitter()
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
```

Add the banner helpers and validation (new methods on `StageView`):

```python
    # -- banner / validation ------------------------------------------------
    def _show_banner(self, html_text: str, *, error: bool) -> None:
        style = (
            "QLabel { background: #fdecea; border: 1px solid #f5c6cb; "
            "border-radius: 4px; padding: 6px; }"
            if error
            else "QLabel { background: #e6f4ea; border: 1px solid #b7e1c0; "
            "border-radius: 4px; padding: 6px; }"
        )
        self._banner.setStyleSheet(style)
        self._banner.setText(html_text)
        self._banner.setVisible(True)

    def _hide_banner(self) -> None:
        self._banner.setVisible(False)

    def _validate_inputs(self, params: dict) -> tuple[str, str] | None:
        """First (param_name, message) whose must_exist path is set but absent."""
        for p in self._spec.params:
            if not p.must_exist:
                continue
            value = params.get(p.name)
            if value and not os.path.exists(str(value)):
                return p.name, f"{p.label}: path does not exist: {value}"
        return None
```

Rework `_on_run` to validate first and reset progress:

```python
    def _on_run(self) -> None:
        if self._runner is not None and self._runner.is_alive():
            return
        params = self._form.values()
        problem = self._validate_inputs(params)
        if problem is not None:
            name, message = problem
            self._show_banner(f"✗ {html.escape(message)}", error=True)
            self._form.focus_param(name)
            return
        self._hide_banner()
        self._last_params = dict(params)
        target = STAGE_TARGETS[self._stage_name]
        self._log.clear()
        self._results.clear()
        self._progress.setValue(0)
        self._progress_text.setText("")
        self._log.append(f"Running stage '{self._stage_name}'…")
        self._set_running(True)
        self.runStarted.emit(self._stage_name)
        self._runner = StageRunner(target, params, start_method="spawn")
        self._runner.start()
        self._timer.start()
```

In `_handle`, extend the `Progress` branch:

```python
        if isinstance(msg, Progress):
            self._log.set_progress(msg.frac, msg.text)
            self._progress.setValue(max(0, min(100, int(round(msg.frac * 100)))))
            if msg.text:
                self._progress_text.setText(msg.text)
                self._log.append(f"  [{msg.frac * 100:5.1f}%] {msg.text}")
```

In `_finish_ok`, set the success banner and complete the bar (first lines become):

```python
    def _finish_ok(self, result) -> None:
        self._timer.stop()
        self._log.set_progress(1.0, "Done.")
        self._progress.setValue(100)
        summary = _summarize(self._stage_name, result)
        first_line = summary.splitlines()[0] if summary else "done"
        self._show_banner(f"✓ {html.escape(first_line)}", error=False)
        self._results.setPlainText(summary)
```

(the rest of `_finish_ok` is unchanged). In `_finish_failed`, show error + hint:

```python
    def _finish_failed(self, failure: Failed) -> None:
        self._timer.stop()
        self._log.set_status(f"Failed: {failure.error}", error=True)
        self._log.append(failure.traceback)
        text = f"✗ {html.escape(failure.error)}"
        hint = getattr(failure, "hint", "")
        if hint:
            text += f"<br><i>{html.escape(hint)}</i>"
        self._show_banner(text, error=True)
        self._set_running(False)
        self.runFinished.emit(self._stage_name, False)
```

- [ ] **Step 4: Run smoke + suite, commit**

Run: `python3 tests/gui_smoke.py && python3 -m pytest -q && ruff check .`
Expected: prints through `[11] …` and `GUI SMOKE PASSED`.

```bash
git add gui/stage_view.py tests/gui_smoke.py
git commit -m "gui: status banner with hints, pre-run must_exist validation, progress bar"
```

---

### Task 19: stages adopt `StageUserError` at their input choke points

**Files:**
- Modify: `dfxm/stages/concat.py`, `strain.py`, `mosaicity.py`, `rocking.py`, `matched.py`, `profiles.py`, `slices.py`
- Test: `tests/test_stage_user_errors.py` (new)

Only the listed raises change — same messages, new type + hint. **No computation
or skip-reporting changes** (per the spec, the all-layers-skipped paths stay
result-based; do not convert them to exceptions).

- [ ] **Step 1: Write the failing test**

Create `tests/test_stage_user_errors.py`:

```python
"""Stages raise StageUserError (with actionable hints) on bad user inputs."""

import pytest

from dfxm.common.errors import StageUserError
from dfxm.stages import concat, mosaicity, profiles, strain


def test_concat_single_requires_input_folder():
    with pytest.raises(StageUserError) as exc_info:
        concat.run({"mode": "single", "input_folder": ""})
    assert exc_info.value.hint


def test_strain_batch_no_matching_folders(tmp_path):
    with pytest.raises(StageUserError) as exc_info:
        strain.run({"mode": "batch", "root_folder": str(tmp_path), "folder_pattern": "zzz*"})
    assert "zzz*" in str(exc_info.value)
    assert "Folder pattern" in exc_info.value.hint


def test_mosaicity_batch_requires_root_folder():
    with pytest.raises(StageUserError) as exc_info:
        mosaicity.run({"mode": "batch", "root_folder": ""})
    assert exc_info.value.hint


def test_profiles_missing_slices_file(tmp_path):
    with pytest.raises(StageUserError) as exc_info:
        profiles.run({"consolidated_h5": str(tmp_path / "nope.h5")})
    assert "slices" in exc_info.value.hint
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_stage_user_errors.py -v`
Expected: 4 FAILED — the stages raise plain `ValueError`/`FileNotFoundError`, which `pytest.raises(StageUserError)` rejects.

- [ ] **Step 3: Convert the raises**

In each listed file add `from dfxm.common.errors import StageUserError` to the
imports, then convert exactly these raises (line numbers as of `216178a` —
re-locate by message text if drifted). **Keep every message identical**; only
the exception type and `hint=` are new.

`dfxm/stages/concat.py` (in `run`):
- `raise ValueError("single mode requires 'input_folder'")` →
  `raise StageUserError("single mode requires 'input_folder'", hint="Pick the scan folder in 'Input folder', or switch Mode to 'batch' and set 'Root folder'.")`
- `raise ValueError("batch mode requires 'root_folder'")` →
  `raise StageUserError("batch mode requires 'root_folder'", hint="Pick the parent of the layer folders in 'Root folder', or switch Mode to 'single'.")`
- `raise ValueError(f"no folders matching {p['folder_pattern']!r} in {root}")` →
  `raise StageUserError(f"no folders matching {p['folder_pattern']!r} in {root}", hint="Check 'Root folder' and 'Folder pattern' — the pattern matched no subfolders.")`

`dfxm/stages/strain.py` (in `run`):
- `"single mode requires 'input_folder'"` → hint: `"Pick the layer folder holding maps.h5 in 'Input folder', or switch Mode to 'batch'."`
- `"batch mode requires 'root_folder'"` → hint: `"Pick the parent of the layer folders in 'Root folder', or switch Mode to 'single'."`
- `f"no folders matching {p['folder_pattern']!r} in {root}"` → hint: `"Check 'Folder pattern' — it matched no subfolders. Each matching folder must contain the darfix maps.h5."`

`dfxm/stages/mosaicity.py` (in `run`): same three messages as strain, hints:
- single → `"Pick the layer folder holding maps.h5 in 'Input folder', or switch Mode to 'batch'."`
- batch → `"Pick the parent of the mosaicity layer folders in 'Root folder', or switch Mode to 'single'."`
- no folders → `"Check 'Folder pattern' — usually the *_mosa__* naming pattern of the mosaicity layers."`

`dfxm/stages/rocking.py` (in `run`):
- `"rocking requires 'raw_root'"` → hint: `"Set 'Raw data root' to the RAW_DATA folder that contains the scan folders."`
- `"rocking needs the mosaicity reference; no mosa motor positions found"` → hint: `"Check 'Mosaicity pattern' — rocking anchors its alignment to the mosaicity scans' samy/samz positions."`
- `f"no rocking folders matching {p['rocking_pattern']!r} in {raw_root}"` → hint: `"Check 'Rocking pattern' against the scan folder names under the raw root."`
- `f"no rocking scans fall in samz union [{z_min:.6f}, {z_max:.6f}] mm (tol={tol})"` → hint: `"Loosen 'samz tolerance' or check that the rocking scans cover the mosaicity/strain Z range."`

`dfxm/stages/matched.py` (in `run`):
- `"matched requires 'raw_root'"` → hint: `"Set 'Raw data root' to the RAW_DATA folder that contains the scan folders."`
- `f"no strain motor positions for {p['strain_pattern']!r}"` → hint: `"Check 'Strain pattern' against the scan folder names under the raw root."`
- `f"no rocking motor positions for {p['rocking_pattern']!r}"` → hint: `"Check 'Rocking pattern' against the scan folder names under the raw root."`

`dfxm/stages/profiles.py` (in `run`):
- `raise FileNotFoundError(f"consolidated slice file not found: {h5_path!r}")` →
  `raise StageUserError(f"consolidated slice file not found: {h5_path!r}", hint="Run the slices stage first — it writes oblique_slices.h5, which this stage profiles.")`
- `raise ValueError("jobs_json must be a non-empty JSON list of jobs")` → hint: `"Define at least one job, or use 'Pick line…' to click a line on a slice plane."`

`dfxm/stages/slices.py` (in `run`):
- `raise ValueError("slices_json must be a non-empty JSON list of slice specs")` → hint: `"Provide a JSON list of plane specs — the field's default shows the format; 'extent': 'auto' fits the plane automatically."`

(visualize and paraview keep their skip-based reporting — nothing to convert.)

- [ ] **Step 4: Run the tests — new file green, nothing else broke**

Run: `python3 -m pytest tests/test_stage_user_errors.py -q && python3 -m pytest -q`
Expected: all green. The pre-existing `pytest.raises(ValueError)` tests
(`test_stage_concat.py:58,85`, `test_stage_strain.py:169`,
`test_stage_matched.py:113`, `test_stage_rocking.py:123`) still pass because
`StageUserError` subclasses `ValueError`.

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && ruff format --check .
git add dfxm/stages/*.py tests/test_stage_user_errors.py
git commit -m "stages: raise StageUserError with actionable hints at input choke points"
```

---

### Task 20: documentation sweep (Usage.md, Codebase.md, CLAUDE.md)

This task is the repo's docs contract for everything above. The branch must
not merge without it.

**Files:**
- Modify: `docs/Usage.md`, `docs/Codebase.md`, `CLAUDE.md`

- [ ] **Step 1: `docs/Usage.md` — Quick start**

Replace the four "Typical first run" steps with:

```markdown
**Typical first run:**

1. The app opens on the **Overview** page — the pipeline drawn left-to-right
   with one sentence per stage. Concat is **optional** (skip it if your scans
   are already concatenated) and **darfix runs outside the app**, between
   concat and the map stages.
2. Pick an **experiment preset** from the dropdown (ships with `STO2_overnight`).
   The one-line summary shows its calibration; **Edit…** opens the full editor.
3. Click a stage in the pipeline rail (or its chip on the Overview page). The
   form shows the stage's **essentials**; everything else is under
   **Advanced (N settings)**, grouped by theme.
4. Click into any field — the **help panel** under the form explains it. Press
   **Run**. Progress shows next to the buttons; results land in **Results**, a
   preview in **Output**, and a green/red **banner** summarises the outcome
   (with a fix-it hint when an input was wrong).
```

- [ ] **Step 2: `docs/Usage.md` — "The stage panel" table**

Replace the table under "### The stage panel" with:

```markdown
| Area | What it does |
|---|---|
| **Parameter form** (left) | Auto-generated from the stage's schema. The few **essential** fields show first; the rest collapse under **Advanced (N settings)**, grouped by theme (Calibration, Data layout, Alignment, Appearance, Output, …). Hover any label for a tooltip. |
| **Help panel** (under the form) | Explains whichever field has focus — what it does, its unit, and the calibration warning where relevant. Idles on a description of the stage. |
| **Run / Cancel + progress** | Runs the stage in a **separate process**; the bar and step text track progress; **Cancel** truly kills it. Before launching, input paths are checked on disk — a missing one blocks the run and focuses the offending field. |
| **Status banner** (above the tabs) | Green one-liner on success; on failure, the error in plain language plus an actionable hint (the full traceback stays in **Log**). |
| **Log** tab | Live progress + streamed messages. |
| **Results** tab | A text summary of what was produced — including every skipped layer/input and the reason. |
| **Output** tab | A representative image preview. |
| **3D** tab | (visualize & rocking only) interactive volume viewer — see [[#Interactive viewers]]. |
```

- [ ] **Step 3: `docs/Usage.md` — main window + stage essentials**

Under "## Core concepts", before "### Experiment presets", insert:

```markdown
### The main window

The left column is a **pipeline rail**: *Overview* first, then the stages in
pipeline order, each with a status glyph (— idle, ▶ running, ✓ ok, ✗ failed).
**Concat is marked (optional)** — skip it when your scans are already
concatenated — and **darfix** appears as a greyed, non-clickable row right
after concat because it runs outside the app. Above the rail, the experiment
header shows the active preset and its calibration in one line; **Edit…**
opens the full schema-driven editor (every field explained in its help panel).
```

In the "Stage reference" intro, replace the callout body with:

```markdown
> [!info] Every parameter is explained in the app
> Tables below list the **key** parameters only. In the app, click into any
> field and the help panel under the form explains it (hover tooltips work
> too). Each stage shows its essentials first; the rest live under *Advanced*.
```

Then add one `**Essentials:** …` line at the top of each stage's parameter
table section, matching the spec:

```markdown
1. concat — **Essentials:** mode, input/root folder, folder pattern, skip existing
2. strain — **Essentials:** mode, input/root folder, ROI, output dir
3. mosaicity — **Essentials:** mode, input/root folder, output dir
4. rocking — **Essentials:** raw root, ROI X/Y, specific frame, output dir
5. visualize — **Essentials:** both volume files, raw root, ROI X/Y, output dir
6. paraview — **Essentials:** both volume files, raw root, ROI X/Y, output dir
7. slices — **Essentials:** three volume files, raw root, slices JSON, output dir
8. profiles — **Essentials:** slices file, mode, jobs JSON, output dir
9. matched — **Essentials:** raw root, frame index, match threshold, output dir
```

(Write each line into its own stage section, not as one block.)

- [ ] **Step 4: `docs/Codebase.md` updates**

- `#### models.py` (dfxm/config): append to the entry: *"`Param` also carries
  GUI metadata: `advanced` (collapse into the Advanced expander), `group`
  (themed header inside it), `must_exist` (input path the GUI verifies before a
  run) — `tests/test_param_metadata.py` enforces help coverage, group-on-advanced
  and ≤ 8 essentials per stage."*
- New `#### errors.py` under `dfxm/common`: *"`StageUserError(message, hint)` —
  ValueError subclass marking input problems the user can fix; stages raise it
  at their validation choke points and the runner forwards `hint` to the GUI."*
- `### dfxm/runner.py` entry: note *"`Failed` carries `error`, `traceback` and
  `hint` (from `StageUserError`, else empty)."*
- Layer 2 `gui/` section: update `main_window.py` (pipeline rail, Overview in
  the stack, darfix/optional-concat rows), `experiment_panel.py` (compact
  header + `ExperimentDialog`), `stage_view.py` (banner, `_validate_inputs`,
  progress row, `runStarted`); add `overview_page.py` (`OverviewPage`,
  `stageSelected`, `set_status`).
- `### gui/widgets/`: update `param_form.py` (essentials + grouped Advanced
  expander, `focusedParamChanged`, `focus_param`); add `help_panel.py`
  (`HelpPanel.set_idle/show_param`).
- Layer 3 `tests/` section: add `test_param_metadata.py`,
  `test_runner_hints.py`, `test_stage_user_errors.py`, and note the extended
  `gui_smoke.py` steps.

- [ ] **Step 5: `CLAUDE.md` updates**

In "Adding a stage", replace step 1 with:

```markdown
1. New module in `dfxm/stages/` with a module-level `STAGE: StageSpec` and
   `run(params, progress=None)` (+ a small `__main__`). Every param needs
   `help` (written for a first-time beamline user); advanced params need
   `group`; input paths set `must_exist=True`. Give the spec a
   newcomer-friendly `description` (it feeds the Overview page and help
   panel). `tests/test_param_metadata.py` enforces all of this. Raise
   `StageUserError(message, hint=...)` from `dfxm.common.errors` for input
   problems the user can fix.
```

In "Conventions & gotchas", add a bullet:

```markdown
- **User-facing errors carry hints.** Input-validation failures raise
  `StageUserError(message, hint)`; the GUI banner shows both. Don't convert
  the skip-based reporting (empty results list reasons) into exceptions.
```

- [ ] **Step 6: Final full verification**

```bash
python3 -m pytest -q                  # everything green
ruff check . && ruff format --check .
python3 tests/gui_smoke.py            # prints through [11] and GUI SMOKE PASSED
```

Optionally, on a machine with a display: `python3 -m gui.app` and click through
Overview → strain → Advanced → help panel.

- [ ] **Step 7: Commit**

```bash
git add docs/Usage.md docs/Codebase.md CLAUDE.md
git commit -m "docs: GUI tour, schema metadata and StageUserError reference for the overhaul"
```

---

## Completion

After Task 20, the branch `gui-overhaul` holds 17 commits, all green. Use the
**superpowers:finishing-a-development-branch** skill to merge/clean up. Reminder
of execution-time constraints:

- `tests/gui_smoke.py` must never be renamed into pytest collection.
- No `default=` of any calibration param may change.
- `dfxm/` must still import without PySide6 installed (`python3 -c "import dfxm.stages.strain"` in an env without Qt is the acid test; at minimum confirm no `PySide6` import appears in `dfxm/`).

