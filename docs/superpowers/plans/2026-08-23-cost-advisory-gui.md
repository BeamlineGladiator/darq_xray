# Cost Advisory GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the phase 1–5 cost model visible — before a run starts, the app says what it will cost, what the machine can afford, and what it therefore expects to do.

**Architecture:** One Qt-free module (`dfxm/common/advisory.py`) composes a `MachineProfile`, a `CostEstimate` and a `RunPlan` into a single `Advisory` carrying pre-rendered text. One GUI module (`gui/advisor.py`) caches the profile and computes advisories off the GUI thread on a debounce. Four surfaces render that one object and compute no policy of their own.

**Tech Stack:** Python 3.11+, PySide6, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-23-cost-advisory-gui-design.md`

## Global Constraints

- **`dfxm/` stays Qt-free.** Never import PySide6 or pyvista under `dfxm/`.
- **The GUI is advisory only.** It never injects a `RunPlan`, a budget or a
  strategy into `run_params`. Stages keep computing their own budget from their
  own measured `RSS_FLOOR_BYTES`.
- **`chunk_layers` is never rendered.** No user-visible string may contain the
  chunk group count. It is display-only in `RunPlan` and is not the blocking a
  stage actually picks.
- **Nothing refuses to run for lack of RAM.** The only stop is the scratch-disk
  confirmation, and the user can always say yes.
- **`advise_stage` never raises.** Estimators read user-supplied paths; a bad
  path is the normal case while a form is being filled in.
- **Docs contract:** any change under `dfxm/stages/` or `gui/` updates
  `docs/Usage.md` (user-visible behaviour) and `docs/Codebase.md` (code
  structure) **in the same commit**. A PostToolUse hook reminds you.
- **Test suite command:** `python3 -m pytest -q --deselect tests/test_gui_viewer3d.py`
  — the deselect is mandatory on this box (in-process Qt GL segfaults).
- **Mutation discipline:** for every test you add, run the mutation that should
  break it and confirm it does. Assert the precondition that keeps the fixture
  inside the region the test names. Twenty checks in this project have been
  found to have stopped checking what they name; two of them were authored by
  the fix for the one before.
- **Read before first Edit.** Never reconstruct an `old_string` from memory.
- `ruff check . && ruff format .` clean before every commit.

---

### Task 1: `CostEstimate.confidence` and the four stale estimators

Four estimators still model the pre-phase-5 accumulate-then-`np.stack` code and
over-predict badly on real data: **strain 5.2×** (2.627 GiB estimated against
0.508 measured) and **mosaicity 36×** (6.566 against 0.181). Over-predicting is
the safe direction, but a banner stating 6.6 GB for a 0.18 GB run is false, and
false-but-safe is how a user learns to ignore the banner. This field is what
lets the surfaces ship before the recalibration work.

**Files:**
- Modify: `dfxm/config/models.py` (the `CostEstimate` dataclass, ~line 97-148)
- Modify: `dfxm/stages/strain.py` (the `return CostEstimate(...)` at ~line 431)
- Modify: `dfxm/stages/mosaicity.py` (the `return CostEstimate(...)` at ~line 462)
- Modify: `dfxm/stages/rocking.py` (the `return CostEstimate(...)` at ~line 740)
- Modify: `dfxm/stages/matched.py` (the `return CostEstimate(...)` at ~line 448)
- Modify: `docs/Codebase.md`
- Test: `tests/test_stage_estimates.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `CostEstimate.confidence: str = "measured"`, the only other legal
  value being `"conservative"`. Task 3 reads it.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_stage_estimates.py`:

```python
# --- confidence marking -------------------------------------------------
# The four stages whose estimators still model the pre-phase-5
# accumulate-then-np.stack code. Measured over-prediction on real STO2:
# strain 5.2x (2.627 GiB est vs 0.508 actual), mosaicity 36x (6.566 vs 0.181).
_CONSERVATIVE = ("strain", "mosaicity", "rocking", "matched")
_MEASURED = ("visualize", "paraview", "slices")


def test_cost_estimate_confidence_defaults_to_measured():
    est = CostEstimate(1, 1, None, True)
    assert est.confidence == "measured"


@pytest.mark.parametrize("stage_name", _CONSERVATIVE)
def test_stale_estimators_mark_themselves_conservative(stage_name, tmp_path):
    est = _estimate_for(stage_name, tmp_path)
    # Precondition: this test is meaningless on the empty-input early return,
    # which never reaches the marked `return` statement.
    assert est.peak_bytes > 0, "fixture did not reach the priced return"
    assert est.confidence == "conservative"


@pytest.mark.parametrize("stage_name", _MEASURED)
def test_verified_estimators_are_not_marked(stage_name, tmp_path):
    est = _estimate_for(stage_name, tmp_path)
    assert est.peak_bytes > 0, "fixture did not reach the priced return"
    assert est.confidence == "measured"
```

Read `tests/test_stage_estimates.py` first and reuse whatever helper it already
has for building a priced estimate per stage; name it `_estimate_for(stage_name,
tmp_path)` if no such helper exists yet, and build it from the fixtures that
file already uses. Add `from dfxm.config.models import CostEstimate` if absent.

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_stage_estimates.py -k confidence -v`
Expected: FAIL — `AttributeError: 'CostEstimate' object has no attribute 'confidence'`.

- [ ] **Step 3: Add the field**

In `dfxm/config/models.py`, after `scratch_bytes: int = 0`:

```python
    confidence: str = "measured"
```

And extend the `CostEstimate` docstring with a paragraph:

```
    ``confidence`` is ``"measured"`` when the model has been checked against a
    real run, and ``"conservative"`` when it has not been recalibrated since the
    phase-5 streaming rewrite and is known to over-predict. Over-predicting is
    the safe direction — it only makes a stage stream harder — but a surface
    that states 6.6 GB for a run that needs 0.18 GB teaches the user to ignore
    it, so the GUI renders a marked estimate as "at most ~N". The marker is
    removed per stage as each model is measured and rewritten; there is no
    separate cleanup to remember.
```

- [ ] **Step 4: Mark the four stale estimators**

In each of the four files, add `confidence="conservative"` as a keyword argument
to the **priced** `return CostEstimate(...)` — the one at the end of the
function, never the early `CostEstimate(0, 0, None, True, "…")` guards. Read
each site before editing; the calls use positional arguments and differ in
arity. For example `dfxm/stages/mosaicity.py` becomes:

```python
    return CostEstimate(
        peak_bytes,
        input_bytes,
        (n_layers, *layer_shape),
        True,
        f"{present} datasets stacked together",
        confidence="conservative",
    )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_stage_estimates.py -v`
Expected: PASS.

- [ ] **Step 6: Run the mutation**

Remove `confidence="conservative"` from `dfxm/stages/rocking.py` only and rerun.
Expected: `test_stale_estimators_mark_themselves_conservative[rocking]` FAILS.
Restore it. If the test still passes, the fixture is not reaching the priced
return and the precondition assertion needs fixing before you continue.

- [ ] **Step 7: Update `docs/Codebase.md`**

In the `CostEstimate` entry, add `confidence` to the field list with one line:
`"measured" | "conservative" — "conservative" marks a model not yet recalibrated
since the streaming rewrite; it over-predicts and the GUI renders it as
"at most ~N".`

- [ ] **Step 8: Lint and commit**

```bash
ruff check . && ruff format .
git add dfxm/config/models.py dfxm/stages/strain.py dfxm/stages/mosaicity.py \
        dfxm/stages/rocking.py dfxm/stages/matched.py \
        tests/test_stage_estimates.py docs/Codebase.md
git commit -m "feat: mark the estimators that have not been recalibrated"
```

---

### Task 2: `advisory.disk_probe_dir`

Which directory's filesystem to measure for free space. This earns its own
tested function: `output_dir` is optional on all seven estimating stages and
each `run()` computes its own fallback internally (e.g. `paraview.py:1630`), so
the naive `params["output_dir"] or "."` reports the repository disk's free space
while the data sits on an external SSD — exactly the machine where the disk
answer decides whether a run is blocked.

**Files:**
- Create: `dfxm/common/advisory.py`
- Modify: `docs/Codebase.md`
- Test: `tests/test_common_advisory.py` (create)

**Interfaces:**
- Consumes: `StageSpec` and `Param.must_exist` from `dfxm/config/models.py`.
- Produces: `disk_probe_dir(spec: StageSpec, params: dict) -> str`. Tasks 3 and 4
  call it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_common_advisory.py`:

```python
"""Composition of profile + estimate + plan into one Advisory (Qt-free)."""

from __future__ import annotations

import os

from dfxm.common.advisory import disk_probe_dir
from dfxm.config.models import Param, ParamType, StageSpec

_SPEC = StageSpec(
    name="demo",
    label="Demo",
    description="",
    params=(
        Param("mosa_volume_file", ParamType.PATH, "Volume", must_exist=True),
        Param("root_folder", ParamType.DIR, "Root", must_exist=True),
        Param("output_dir", ParamType.DIR, "Out"),
    ),
)


def test_output_dir_wins_when_set(tmp_path):
    out = str(tmp_path / "out")
    assert disk_probe_dir(_SPEC, {"output_dir": out, "root_folder": "/elsewhere"}) == out


def test_falls_back_to_the_input_files_directory(tmp_path):
    """The branch that matters: an unset output_dir must NOT land on cwd while
    the data lives on another filesystem."""
    vol = tmp_path / "data" / "volumes.h5"
    vol.parent.mkdir(parents=True)
    vol.write_bytes(b"")
    got = disk_probe_dir(_SPEC, {"output_dir": "", "mosa_volume_file": str(vol)})
    assert got == str(vol.parent)
    assert got != os.getcwd()  # precondition: the fixture really is elsewhere


def test_falls_back_to_an_input_directory_unchanged(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    assert disk_probe_dir(_SPEC, {"output_dir": "", "root_folder": str(root)}) == str(root)


def test_falls_back_to_cwd_when_nothing_is_filled_in():
    assert disk_probe_dir(_SPEC, {}) == os.getcwd()


def test_ignores_params_that_are_not_inputs(tmp_path):
    """A non-must_exist path must never be chosen as the probe target."""
    spec = StageSpec(
        name="demo",
        label="Demo",
        description="",
        params=(Param("some_output", ParamType.SAVE_PATH, "Out"),),
    )
    assert disk_probe_dir(spec, {"some_output": str(tmp_path / "x.h5")}) == os.getcwd()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_common_advisory.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfxm.common.advisory'`.

- [ ] **Step 3: Write the implementation**

Create `dfxm/common/advisory.py`:

```python
"""Compose a machine profile, a cost estimate and a run plan into one Advisory.

Qt-free and side-effect-free apart from the probes it delegates to. This is the
single place that decides *what the user is told* about a run's cost; the four
GUI surfaces render an :class:`Advisory` and compute no policy of their own.

Nothing here influences what a stage does. Since phase 5 each volume stage
derives its own streaming budget from
:func:`~dfxm.common.advice.working_set_budget_bytes` with its own **measured**
``RSS_FLOOR_BYTES``, which a caller cannot guess; the advisory path and the
execution path are parallel, not sequential.
"""

from __future__ import annotations

import os

from ..config.models import StageSpec


def disk_probe_dir(spec: StageSpec, params: dict) -> str:
    """Which directory's filesystem to measure for free space.

    ``output_dir`` when the user set one; otherwise the directory of the first
    filled-in ``must_exist`` input; otherwise the working directory.

    The fallback is not cosmetic. ``output_dir`` is optional on every estimating
    stage — each ``run()`` computes its own default internally — so reading it
    alone would measure the filesystem the *app* was started from while the data
    sits on an external drive, and the scratch-disk check that decides whether a
    run is blocked would be answered about the wrong disk.
    """
    out = str(params.get("output_dir") or "").strip()
    if out:
        return out
    for p in spec.params:
        if not p.must_exist:
            continue
        value = str(params.get(p.name) or "").strip()
        if not value:
            continue
        if os.path.isdir(value):
            return value
        return os.path.dirname(value) or os.getcwd()
    return os.getcwd()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_common_advisory.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the mutation**

Change the fallback loop's `if not p.must_exist: continue` to `if False: continue`
and rerun. Expected: `test_ignores_params_that_are_not_inputs` FAILS. Restore.

- [ ] **Step 6: Update `docs/Codebase.md`**

Add a `dfxm/common/advisory.py` section under the other `dfxm/common/` modules,
documenting `disk_probe_dir` and stating that the module composes rather than
decides.

- [ ] **Step 7: Lint and commit**

```bash
ruff check . && ruff format .
git add dfxm/common/advisory.py tests/test_common_advisory.py docs/Codebase.md
git commit -m "feat: resolve which disk a run's free space should be measured on"
```

---

### Task 3: `Advisory` and `advise_stage`

**Files:**
- Modify: `dfxm/common/advice.py` (promote `_human`, add `CHUNK_REASON_PREFIX`)
- Modify: `dfxm/common/advisory.py`
- Modify: `docs/Codebase.md`
- Test: `tests/test_common_advisory.py`, `tests/test_common_advice.py`

**Interfaces:**
- Consumes: `disk_probe_dir` (Task 2); `CostEstimate.confidence` (Task 1);
  `advice.plan_run`, `machine.profile`.
- Produces:
  - `advice.human_bytes(nbytes: float) -> str` (was the private `_human`)
  - `advice.CHUNK_REASON_PREFIX: str`
  - `Advisory(profile, estimate, plan, headline, details, blocked, conservative, hints)`
  - `advise_stage(spec: StageSpec, params: dict, *, profile: MachineProfile | None = None) -> Advisory`

  Tasks 4–10 consume `Advisory` and `advise_stage`.

- [ ] **Step 1: Promote `_human` and pin the chunk reason**

`advisory.py` needs the same byte formatting, and `_human` is private to
`advice.py`. Rename it and pin the one reason string that carries the chunk
count, so the filter in Step 3 cannot silently stop matching.

First: `grep -rn "_human" --include=*.py .` and update every call site.

In `dfxm/common/advice.py`, rename `def _human(` to `def human_bytes(` and add
above `plan_run`:

```python
# The one `plan_run` reason that carries `chunk_layers`. `advisory.py` replaces
# any reason starting with this prefix, because the group count is display-only
# and is NOT the blocking a stage picks — visualize/slices/paraview each derive
# their own from `working_set_budget_bytes` with a per-stage RSS floor. Pinned
# as a constant so a reworded message fails a test instead of silently escaping
# the filter and reaching the user.
CHUNK_REASON_PREFIX = "chunking into groups of"
```

and build that reason from it:

```python
        reasons.append(
            f"{CHUNK_REASON_PREFIX} {chunk_layers} of {n_layers} {unit}"
            " — slower, same result"
        )
```

- [ ] **Step 2: Write the failing tests**

Append to `tests/test_common_advice.py`:

```python
def test_the_chunking_reason_starts_with_the_pinned_prefix():
    """advisory.py filters on this prefix; a reworded message must fail here,
    not leak a chunk count into the GUI."""
    from dfxm.common import advice
    from dfxm.common.machine import MachineProfile

    prof = MachineProfile("Linux", 4, 2, 8 * 1024**3, 1 * 1024**3, 40 * 1024**3,
                          None, "unprobed", None, ())
    est = CostEstimate(200 * 1024**3, 100 * 1024**3, (76, 1200, 1800), True)
    plan = advice.plan_run(prof, est)
    assert plan.strategy == "chunked"  # precondition for the reason to exist
    assert any(r.startswith(advice.CHUNK_REASON_PREFIX) for r in plan.reasons)
```

Append to `tests/test_common_advisory.py`:

```python
import pytest

from dfxm.common import advice
from dfxm.common.advisory import Advisory, advise_stage
from dfxm.config.models import CostEstimate
from tests.machine_fixtures import tiny_ram, workstation_sw_gl

GB = 1024**3


def _spec_with(estimator_target: str) -> StageSpec:
    return StageSpec(
        name="demo", label="Demo", description="",
        params=_SPEC.params, estimate=estimator_target,
    )


def test_in_core_headline_names_cost_and_headroom(monkeypatch):
    spec = _spec_with("tests.test_common_advisory:_cheap_estimate")
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.plan.strategy == "in-core"
    assert "runs in memory" in adv.headline
    assert "1.0 GB" in adv.headline


def test_streaming_headline_says_expected_and_hides_the_chunk_count():
    spec = _spec_with("tests.test_common_advisory:_huge_estimate")
    adv = advise_stage(spec, {}, profile=tiny_ram())
    assert adv.plan.strategy == "chunked"       # precondition
    assert adv.plan.chunk_layers > 0            # precondition: there IS a count
    assert "expected to stream" in adv.headline
    rendered = " ".join((adv.headline, *adv.details))
    assert advice.CHUNK_REASON_PREFIX not in rendered
    assert str(adv.plan.chunk_layers) not in rendered


def test_conservative_estimate_is_marked_in_the_headline_and_details():
    spec = _spec_with("tests.test_common_advisory:_conservative_estimate")
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.conservative is True
    assert "at most" in adv.headline
    assert any("over-predict" in d for d in adv.details)


def test_measured_estimate_is_not_marked():
    spec = _spec_with("tests.test_common_advisory:_cheap_estimate")
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.conservative is False
    assert "at most" not in adv.headline


def test_a_raising_estimator_becomes_a_headline_not_an_exception():
    spec = _spec_with("tests.test_common_advisory:_boom")
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.estimate is None and adv.plan is None
    assert "FileNotFoundError" in adv.headline


def test_a_stage_without_an_estimator_says_nothing():
    spec = StageSpec(name="demo", label="Demo", description="", params=())
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.headline == "" and adv.estimate is None


def test_an_unpriced_estimate_shows_its_note():
    spec = _spec_with("tests.test_common_advisory:_unpriced")
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.headline == "no readable volume files selected yet"
    assert adv.plan is None


def test_blocked_on_scratch_disk_is_carried_through():
    spec = _spec_with("tests.test_common_advisory:_spilling_estimate")
    adv = advise_stage(spec, {}, profile=tiny_ram())
    # Precondition: this machine really is short of disk for this estimate,
    # or the test silently becomes a test of the unblocked path.
    assert adv.estimate.scratch_bytes > tiny_ram().disk_free
    assert adv.blocked and "scratch disk" in adv.blocked


# -- estimator stand-ins, resolved by StageSpec.estimator() -------------------
def _cheap_estimate(params):
    return CostEstimate(1 * GB, 1 * GB, (10, 100, 100), True)


def _huge_estimate(params):
    return CostEstimate(200 * GB, 100 * GB, (76, 1200, 1800), True)


def _conservative_estimate(params):
    return CostEstimate(1 * GB, 1 * GB, (10, 100, 100), True, confidence="conservative")


def _unpriced(params):
    return CostEstimate(0, 0, None, True, "no readable volume files selected yet")


def _spilling_estimate(params):
    return CostEstimate(200 * GB, 100 * GB, (76, 1200, 1800), True,
                        scratch_bytes=100 * GB)


def _boom(params):
    raise FileNotFoundError("no such file")
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_common_advisory.py tests/test_common_advice.py -v`
Expected: FAIL — `ImportError: cannot import name 'Advisory'`.

- [ ] **Step 4: Write the implementation**

Append to `dfxm/common/advisory.py` (and extend its imports):

```python
from dataclasses import dataclass, field

from . import advice, machine
from ..config.models import CostEstimate, StageSpec
from .advice import RunPlan
from .machine import MachineProfile

# How each RunPlan.strategy is phrased. Hedged deliberately: the prediction and
# the run share a headroom figure so they agree in direction, but they are
# computed independently and this must not read as an instruction.
_STRATEGY_WORDS = {
    "in-core": "runs in memory",
    "chunked": "expected to stream",
    "disk-backed": "expected to run disk-backed",
}

_CONSERVATIVE_NOTE = (
    "this stage's estimate has not been recalibrated since the streaming "
    "rewrite — it over-predicts, so the run may be much lighter than shown"
)

_CHUNK_REPLACEMENT = "blocking the work into groups — slower, same result"


@dataclass(frozen=True)
class Advisory:
    """What to tell the user about a run, and the raw facts behind it."""

    profile: MachineProfile
    estimate: CostEstimate | None
    plan: RunPlan | None
    headline: str
    details: tuple[str, ...] = ()
    blocked: str | None = None
    conservative: bool = False
    hints: dict[str, str] = field(default_factory=dict)


def _headline(estimate: CostEstimate, plan: RunPlan, conservative: bool) -> str:
    need = advice.human_bytes(estimate.peak_bytes)
    lead = f"at most ~{need} (conservative estimate)" if conservative else f"needs ~{need}"
    strategy = _STRATEGY_WORDS.get(plan.strategy, plan.strategy)
    return f"{lead}, {advice.human_bytes(plan.budget_bytes)} safely available — {strategy}"


def _details(plan: RunPlan, conservative: bool) -> tuple[str, ...]:
    """`plan.reasons`, with the chunk-count sentence replaced.

    NOT `plan.reasons` verbatim: `plan_run` writes the group count into its own
    reasons, and that number is display-only — it is not the blocking a stage
    picks. Pinned by `advice.CHUNK_REASON_PREFIX` so a reworded reason fails a
    test rather than leaking the count.
    """
    out = [
        _CHUNK_REPLACEMENT if r.startswith(advice.CHUNK_REASON_PREFIX) else r
        for r in plan.reasons
    ]
    if conservative:
        out.append(_CONSERVATIVE_NOTE)
    return tuple(out)


def advise_stage(
    spec: StageSpec, params: dict, *, profile: MachineProfile | None = None
) -> Advisory:
    """What this run will cost on this machine, ready to render.

    **Never raises.** Estimators open user-supplied HDF5 paths and read raw scan
    folders; a missing file, a corrupt file or a half-typed path is the ordinary
    state of a form being filled in, not an error. Any failure becomes a
    headline naming it — an exception escaping into a form-change handler would
    take the window with it.

    *profile* lets a caller supply an already-measured (or synthetic) machine
    instead of probing again; the GUI passes its cached one.
    """
    probe_dir = disk_probe_dir(spec, params)
    prof = profile if profile is not None else machine.profile(output_dir=probe_dir)
    try:
        estimator = spec.estimator()
        if estimator is None:
            return Advisory(prof, None, None, "")
        estimate = estimator(dict(params))
    except Exception as exc:  # noqa: BLE001 — delivered to the user as text
        return Advisory(prof, None, None, f"cannot estimate cost: {type(exc).__name__}")

    if not estimate.peak_bytes:
        return Advisory(prof, estimate, None, estimate.note or "not enough input to estimate cost")

    plan = advice.plan_run(prof, estimate, scratch_dir=probe_dir)
    conservative = estimate.confidence != "measured"
    return Advisory(
        profile=prof,
        estimate=estimate,
        plan=plan,
        headline=_headline(estimate, plan, conservative),
        details=_details(plan, conservative),
        blocked=plan.blocked,
        conservative=conservative,
    )
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_common_advisory.py tests/test_common_advice.py -v`
Expected: PASS.

- [ ] **Step 6: Run the mutations**

1. Change `_details` to `return tuple(plan.reasons)`. Expected:
   `test_streaming_headline_says_expected_and_hides_the_chunk_count` FAILS.
2. Reword `plan_run`'s chunking reason to start with `"splitting into"`,
   leaving `CHUNK_REASON_PREFIX` alone. Expected:
   `test_the_chunking_reason_starts_with_the_pinned_prefix` FAILS. That is the
   whole point of the constant: a reworded reason must fail loudly rather than
   silently escape the filter and leak a chunk count to the user. Restore.
3. Remove the `try`/`except` around the estimator call. Expected:
   `test_a_raising_estimator_becomes_a_headline_not_an_exception` FAILS with
   `FileNotFoundError`.

- [ ] **Step 7: Full suite, docs, commit**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
```

Update `docs/Codebase.md`: `Advisory`, `advise_stage`, `advice.human_bytes`
(renamed from `_human`) and `advice.CHUNK_REASON_PREFIX`.

```bash
git add dfxm/common/advisory.py dfxm/common/advice.py tests/test_common_advisory.py \
        tests/test_common_advice.py docs/Codebase.md
git commit -m "feat: compose profile, estimate and plan into one advisory"
```

---

### Task 4: `gui/advisor.py` — cached profile and the debounced worker

**Files:**
- Create: `gui/advisor.py`
- Modify: `docs/Codebase.md`
- Test: `tests/test_gui_advisor.py` (create)

**Interfaces:**
- Consumes: `advise_stage`, `disk_probe_dir`, `Advisory` (Task 3);
  `keep_alive` from `gui/widgets/busy.py`.
- Produces:
  - `cached_profile(output_dir: str) -> MachineProfile`
  - `clear_profile_cache() -> None`
  - `StageAdvisor(spec, values_fn, parent=None, debounce_ms=400)` with
    `request()`, `compute_blocking() -> Advisory`, `latest: Advisory | None`,
    and signal `advisoryReady(object)`.

  Tasks 5–10 consume `StageAdvisor` and `cached_profile`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_advisor.py`:

```python
"""Profile caching and the debounced advisory worker (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time  # noqa: E402

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import CostEstimate, Param, ParamType, StageSpec  # noqa: E402
from gui import advisor as A  # noqa: E402

GB = 1024**3

_SPEC = StageSpec(
    name="demo", label="Demo", description="",
    params=(Param("output_dir", ParamType.DIR, "Out"),),
    estimate="tests.test_gui_advisor:_cheap_estimate",
)


def _cheap_estimate(params):
    return CostEstimate(1 * GB, 1 * GB, (10, 100, 100), True)


def _drain(timeout_s=10.0):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _app.processEvents()
        time.sleep(0.01)


def test_profile_is_cached_within_the_ttl(monkeypatch):
    calls = []
    real = A.machine.profile
    monkeypatch.setattr(
        A.machine, "profile",
        lambda **kw: (calls.append(kw), real(**kw))[1],
    )
    A.clear_profile_cache()
    A.cached_profile(os.getcwd())
    A.cached_profile(os.getcwd())
    assert len(calls) == 1


def test_cache_is_per_directory(monkeypatch, tmp_path):
    calls = []
    real = A.machine.profile
    monkeypatch.setattr(
        A.machine, "profile",
        lambda **kw: (calls.append(kw), real(**kw))[1],
    )
    A.clear_profile_cache()
    A.cached_profile(os.getcwd())
    A.cached_profile(str(tmp_path))
    assert len(calls) == 2


def test_cached_profile_never_probes_gl(monkeypatch):
    monkeypatch.setattr(
        A.machine, "probe_gl",
        lambda **kw: pytest.fail("cached_profile must never probe GL"),
    )
    A.clear_profile_cache()
    prof = A.cached_profile(os.getcwd())
    assert prof.gl_status == "unprobed"


def test_request_debounces_and_emits_once():
    seen = []
    adv = A.StageAdvisor(_SPEC, lambda: {}, debounce_ms=50)
    adv.advisoryReady.connect(seen.append)
    for _ in range(5):
        adv.request()
    _drain(5.0)
    assert len(seen) == 1
    assert "runs in memory" in seen[0].headline


def test_compute_blocking_returns_and_stores_latest():
    adv = A.StageAdvisor(_SPEC, lambda: {}, debounce_ms=50)
    got = adv.compute_blocking()
    assert got.plan is not None
    assert adv.latest is got
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_gui_advisor.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.advisor'`.

- [ ] **Step 3: Write the implementation**

Create `gui/advisor.py`:

```python
"""GUI-side plumbing for the cost advisory: a cached profile and a worker.

Computes no policy. `dfxm/common/advisory.py` decides what the user is told;
this module only decides *when* and *on which thread*.
"""

from __future__ import annotations

import time

from PySide6.QtCore import QObject, QThread, QTimer, Signal

from dfxm.common import machine
from dfxm.common.advisory import Advisory, advise_stage, disk_probe_dir
from dfxm.config.models import StageSpec

from .widgets.busy import keep_alive

# Short enough that the status bar tracks a filling disk, long enough that four
# surfaces plus a 5 s timer cost one probe rather than five.
_PROFILE_TTL_S = 5.0

_cache: dict[str, tuple[float, machine.MachineProfile]] = {}


def cached_profile(output_dir: str) -> machine.MachineProfile:
    """A recent :class:`MachineProfile` for *output_dir*'s filesystem.

    **Never probes GL.** The probe costs a child process; only the System check
    dialog and the one-shot background probe may pay for it.
    """
    now = time.monotonic()
    hit = _cache.get(output_dir)
    if hit is not None and now - hit[0] < _PROFILE_TTL_S:
        return hit[1]
    prof = machine.profile(output_dir=output_dir)
    _cache[output_dir] = (now, prof)
    return prof


def clear_profile_cache() -> None:
    """Drop every cached profile (tests, and a forced re-probe)."""
    _cache.clear()


class _AdvisoryWorker(QThread):
    """One `advise_stage` call off the GUI thread. Emits `done(Advisory|None)`."""

    done = Signal(object)

    def __init__(self, spec: StageSpec, params: dict) -> None:
        super().__init__()
        self._spec = spec
        self._params = params

    def run(self) -> None:  # worker thread — no Qt widgets in here
        try:
            probe_dir = disk_probe_dir(self._spec, self._params)
            result = advise_stage(
                self._spec, self._params, profile=cached_profile(probe_dir)
            )
        except Exception:  # noqa: BLE001 — advise_stage promises not to, but a
            result = None  # dead worker must not take the window with it
        self.done.emit(result)


class StageAdvisor(QObject):
    """Debounced, latest-wins advisories for one stage form.

    Off the GUI thread because the estimators do real IO: `sum_dataset_bytes`
    opens every candidate HDF5 file and is not memoised (only the motor read
    is), so a synchronous call per keystroke stutters the form on network or
    external storage.
    """

    advisoryReady = Signal(object)  # Advisory

    def __init__(self, spec: StageSpec, values_fn, parent=None, debounce_ms: int = 400) -> None:
        super().__init__(parent)
        self._spec = spec
        self._values_fn = values_fn
        self._worker: _AdvisoryWorker | None = None
        self._pending = False
        self.latest: Advisory | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(debounce_ms)
        self._timer.timeout.connect(self._start)

    def request(self) -> None:
        """Ask for a fresh advisory once the form stops changing."""
        self._timer.start()

    def compute_blocking(self) -> Advisory:
        """Compute one now, on the calling thread. For the Run click only."""
        params = self._values_fn()
        result = advise_stage(
            self._spec, params, profile=cached_profile(disk_probe_dir(self._spec, params))
        )
        self.latest = result
        return result

    def _start(self) -> None:
        if self._worker is not None:
            self._pending = True  # latest-wins: one re-run after this one lands
            return
        worker = _AdvisoryWorker(self._spec, self._values_fn())
        worker.done.connect(self._on_done)
        self._worker = worker
        keep_alive(worker)
        worker.start()

    def _on_done(self, result) -> None:
        self._worker = None
        if result is not None:
            self.latest = result
            self.advisoryReady.emit(result)
        if self._pending:
            self._pending = False
            self._start()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_gui_advisor.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Run the mutation**

Set `_PROFILE_TTL_S = 0.0` and rerun. Expected:
`test_profile_is_cached_within_the_ttl` FAILS. Restore. Then change
`self._timer.setSingleShot(True)` to `False` — expected:
`test_request_debounces_and_emits_once` FAILS (repeated emissions). Restore.

- [ ] **Step 6: Docs, lint, commit**

Add `gui/advisor.py` to `docs/Codebase.md` (cached profile, `StageAdvisor`,
why the work is off-thread).

```bash
ruff check . && ruff format .
git add gui/advisor.py tests/test_gui_advisor.py docs/Codebase.md
git commit -m "feat: compute stage cost advisories off the GUI thread"
```

---

### Task 5: The live cost line in `StageView`

**Files:**
- Modify: `gui/stage_view.py` (constructor around lines 183-208; `showEvent` at 274)
- Modify: `docs/Usage.md`, `docs/Codebase.md`
- Test: `tests/test_gui_stage_advice.py` (create)

**Interfaces:**
- Consumes: `StageAdvisor` (Task 4).
- Produces: `StageView._advice_label` (QLabel), `StageView._advisor`
  (`StageAdvisor`), `StageView._show_advisory(advisory)`. Task 6 reuses
  `self._advisor`; Task 9 reuses `_show_advisory`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_stage_advice.py`:

```python
"""The live cost line under a stage form (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.advisory import Advisory  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import StageView  # noqa: E402
from tests.machine_fixtures import workstation_sw_gl  # noqa: E402


def _advisory(headline="needs ~1.0 GB, 4.0 GB safely available — runs in memory",
              details=("a reason",)):
    return Advisory(workstation_sw_gl(), None, None, headline, details)


def test_advice_line_starts_hidden():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    assert view._advice_label.isVisibleTo(view) is False


def test_advice_line_shows_headline_and_tooltips_details():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    view._show_advisory(_advisory())
    assert "runs in memory" in view._advice_label.text()
    assert "a reason" in view._advice_label.toolTip()
    assert view._advice_label.isVisibleTo(view) is True


def test_an_empty_headline_hides_the_line_again():
    """A stage with no estimator, or a cleared form, must not leave stale text."""
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    view._show_advisory(_advisory())
    assert view._advice_label.text()  # precondition: it really was shown
    view._show_advisory(_advisory(headline="", details=()))
    assert view._advice_label.text() == ""
    assert view._advice_label.isVisibleTo(view) is False


def test_form_changes_ask_the_advisor_for_a_refresh():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    asked = []
    view._advisor.request = lambda: asked.append(1)
    view._form.changed.emit()
    assert asked == [1]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_gui_stage_advice.py -v`
Expected: FAIL — `AttributeError: 'StageView' object has no attribute '_advice_label'`.

- [ ] **Step 3: Add the widget and the advisor**

In `gui/stage_view.py`, add the import near the other `gui` imports:

```python
from .advisor import StageAdvisor
```

In `__init__`, after the `btn_row.addStretch(1)` block and before the progress
widgets:

```python
        # Live cost line: what this run is expected to cost on this machine.
        # Advisory only — it never changes what the stage does.
        self._advice_label = QLabel("")
        self._advice_label.setWordWrap(True)
        self._advice_label.setProperty("role", "muted")
        self._advice_label.setVisible(False)
        self._advisor = StageAdvisor(spec, self._form.values, parent=self)
        self._advisor.advisoryReady.connect(self._show_advisory)
        # Connected unconditionally: the save-on-edit hookup above is gated on a
        # store (absent in unit tests), and the cost line must follow the form
        # either way.
        self._form.changed.connect(self._advisor.request)
```

In the layout, insert the label between the button row and the progress row:

```python
        left_layout.addLayout(btn_row)
        left_layout.addWidget(self._advice_label)
        left_layout.addLayout(progress_row)
```

Add the slot next to `_hide_banner`:

```python
    def _show_advisory(self, advisory) -> None:
        """Render the live cost line. Empty headline -> hidden, never stale."""
        text = advisory.headline if advisory is not None else ""
        self._advice_label.setText(text)
        self._advice_label.setToolTip("\n".join(advisory.details) if advisory else "")
        self._advice_label.setVisible(bool(text))
```

In `showEvent`, ask for the first advisory — not in `__init__`, so opening the
app does not stat nine stages' worth of HDF5 files:

```python
    def showEvent(self, event) -> None:  # Qt hook
        super().showEvent(event)
        self._help.show_idle()  # every stage opens on its description
        self._advisor.request()
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_gui_stage_advice.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the mutation**

Delete the `self._advice_label.setVisible(bool(text))` line. Expected:
`test_an_empty_headline_hides_the_line_again` FAILS. Restore.

- [ ] **Step 6: Docs, suite, commit**

`docs/Usage.md`: a short subsection under the stage-view description explaining
the cost line — what "needs ~N" and "at most ~N (conservative estimate)" mean,
and that it never changes what the run does. `docs/Codebase.md`: the new
`StageView` attributes and `_show_advisory`.

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add gui/stage_view.py tests/test_gui_stage_advice.py docs/Usage.md docs/Codebase.md
git commit -m "feat: show what a stage run will cost while the form is filled in"
```

---

### Task 6: Pre-flight banner, `banner-info` role, and the scratch-disk confirmation

**Files:**
- Modify: `gui/theme.py` (banner rules around lines 86-87 and 178-185)
- Modify: `gui/stage_view.py` (`_show_banner` at 357; `_on_run` at 389)
- Modify: `docs/Usage.md`, `docs/Codebase.md`
- Test: `tests/test_gui_stage_advice.py`

**Interfaces:**
- Consumes: `StageAdvisor.compute_blocking` (Task 4).
- Produces: `StageView._show_banner(html_text, *, error=False, role="")` (the
  `role` keyword is new; existing calls are unaffected) and
  `StageView._confirm_blocked(reason) -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_gui_stage_advice.py`:

```python
from PySide6.QtWidgets import QMessageBox  # noqa: E402

from dfxm.common.advice import RunPlan  # noqa: E402
from dfxm.config.models import CostEstimate  # noqa: E402

GB = 1024**3


def _blocked_advisory():
    plan = RunPlan("chunked", 4 * GB, 8, 1, "/scratch", ("a reason",),
                   "needs 100.0 GB of scratch disk but only 40.0 GB is free")
    return Advisory(
        workstation_sw_gl(),
        CostEstimate(200 * GB, 100 * GB, (76, 1200, 1800), True, scratch_bytes=100 * GB),
        plan,
        "needs ~200.0 GB, 4.0 GB safely available — expected to stream",
        ("a reason",),
        plan.blocked,
    )


def _view_with_advisory(advisory, monkeypatch):
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    monkeypatch.setattr(view._advisor, "compute_blocking", lambda: advisory)
    started = []
    monkeypatch.setattr(view, "_start_runner", lambda params: started.append(params))
    return view, started


def test_run_shows_the_cost_in_an_info_banner(monkeypatch):
    view, started = _view_with_advisory(_advisory(), monkeypatch)
    view._on_run()
    assert started, "the run must still start"
    assert view._banner.isVisibleTo(view)
    assert view._banner.property("role") == "banner-info"
    assert "runs in memory" in view._banner.text()


def test_a_blocked_run_asks_and_starts_when_accepted(monkeypatch):
    adv = _blocked_advisory()
    assert adv.blocked  # precondition: this fixture really is blocked
    view, started = _view_with_advisory(adv, monkeypatch)
    monkeypatch.setattr(view, "_confirm_blocked", lambda reason: True)
    view._on_run()
    assert started


def test_a_blocked_run_starts_nothing_when_declined(monkeypatch):
    adv = _blocked_advisory()
    assert adv.blocked  # precondition
    view, started = _view_with_advisory(adv, monkeypatch)
    monkeypatch.setattr(view, "_confirm_blocked", lambda reason: False)
    view._on_run()
    assert started == []


def test_the_confirmation_defaults_to_cancel(monkeypatch):
    """A stray Enter must not launch a run the machine cannot finish."""
    seen = {}

    def fake_exec(self):
        seen["default"] = self.defaultButton()
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    assert view._confirm_blocked("needs more disk") is False
    assert seen["default"].text().endswith("Cancel")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_stage_advice.py -v`
Expected: FAIL — `AttributeError: 'StageView' object has no attribute '_start_runner'`.

- [ ] **Step 3: Add the theme role**

In `gui/theme.py`, beside `_BANNER_ERROR` / `_BANNER_SUCCESS`:

```python
_BANNER_INFO = "#37474f"
```

and beside the two existing banner rules:

```python
    QLabel[role="banner-info"] {{
        background: {_BANNER_INFO}; color: #ffffff;
        border-radius: 7px; padding: 6px 10px;
    }}
```

- [ ] **Step 4: Extract `_start_runner` and widen `_show_banner`**

In `gui/stage_view.py`, change the banner helper's signature (the body's
property assignment is the only line that changes):

```python
    def _show_banner(self, html_text: str, *, error: bool = False, role: str = "") -> None:
        kind = role or ("banner-error" if error else "banner-success")
        self._banner.setProperty("role", kind)
```

Read the existing `_on_run` before editing. Move its tail — everything from
`self._log.clear()` to `self._timer.start()` — into a new method, so a test can
observe a launch without spawning a child process:

```python
    def _start_runner(self, run_params: dict) -> None:
        """Launch the stage child. Split out of `_on_run` so the pre-flight
        checks above it can be tested without spawning a process."""
        target = STAGE_TARGETS[self._stage_name]
        self._log.clear()
        ...  # the existing body, unchanged
        self._runner = StageRunner(target, run_params, start_method="spawn")
        self._runner.start()
        self._timer.start()
```

Then, in `_on_run`, between building `run_params` and launching:

```python
        # Pre-flight: what will this cost, and can the disk take it? Computed
        # fresh on the click (cheap, and never stale). Advisory only — it never
        # changes what the stage does, and only the disk question can stop it.
        advisory = self._advisor.compute_blocking()
        if advisory.blocked and not self._confirm_blocked(advisory.blocked):
            return
        self._show_advisory(advisory)
        if advisory.headline:
            lines = [html.escape(advisory.headline)]
            lines += [html.escape(d) for d in advisory.details]
            self._show_banner("<br>".join(lines), role="banner-info")
        self._start_runner(run_params)
```

and the confirmation:

```python
    def _confirm_blocked(self, reason: str) -> bool:
        """Ask before a run the machine may not have the disk to finish.

        The project's rule is that nothing refuses to run for lack of RAM; disk
        is the one genuine blocker, and even then the answer is the user's. The
        default button is Cancel so a stray Enter cannot launch a long run that
        dies part-way.
        """
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Not enough scratch disk")
        box.setText(f"This run {reason}.")
        box.setInformativeText("It may fail part-way through. Run anyway?")
        box.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Cancel)
        return box.exec() == QMessageBox.StandardButton.Ok
```

Add `QMessageBox` to the PySide6 imports at the top of the file.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_stage_advice.py -v`
Expected: PASS (8 tests).

- [ ] **Step 6: Run the mutation**

Change the guard to `if advisory.blocked and self._confirm_blocked(advisory.blocked): return`.
Expected: BOTH blocked tests FAIL (accepted starts nothing, declined starts).
Restore. Then change `role="banner-info"` to `error=True` — expected:
`test_run_shows_the_cost_in_an_info_banner` FAILS on the role assertion.

- [ ] **Step 7: Docs, suite, commit**

`docs/Usage.md`: what the banner says at Run, and what the scratch-disk
confirmation is asking — that declining is safe and accepting may fail part-way.
`docs/Codebase.md`: `_start_runner`, `_confirm_blocked`, the `role` keyword and
the `banner-info` theme role.

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add gui/stage_view.py gui/theme.py tests/test_gui_stage_advice.py \
        docs/Usage.md docs/Codebase.md
git commit -m "feat: state a run's cost before it starts, and ask before spending disk we lack"
```

---

### Task 7: The machine status bar

**Files:**
- Modify: `gui/main_window.py` (constructor, after the splitter at ~line 149)
- Modify: `docs/Usage.md`, `docs/Codebase.md`
- Test: `tests/test_gui_status_bar.py` (create)

**Interfaces:**
- Consumes: `cached_profile` (Task 4), `advice.human_bytes` (Task 3).
- Produces: `MainWindow._machine_label`, `MainWindow._refresh_machine_status()`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_status_bar.py`:

```python
"""The ambient machine readout in the status bar (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui import advisor as A  # noqa: E402
from gui.main_window import MainWindow  # noqa: E402
from tests.machine_fixtures import tiny_ram, windows_no_vtk, workstation_sw_gl  # noqa: E402


def test_status_bar_names_cores_disk_and_ram(monkeypatch):
    monkeypatch.setattr(A, "cached_profile", lambda d: workstation_sw_gl())
    win = MainWindow()
    win._refresh_machine_status()
    text = win._machine_label.text()
    assert "36 cores" in text
    assert "RAM" in text
    assert "free" in text


def test_software_gl_is_called_out(monkeypatch):
    monkeypatch.setattr(A, "cached_profile", lambda d: workstation_sw_gl())
    win = MainWindow()
    win._refresh_machine_status()
    # Precondition: this fixture really is a software renderer.
    assert workstation_sw_gl().gl.software is True
    assert "software GL" in win._machine_label.text()


def test_unmeasured_fields_are_omitted_not_shown_as_zero(monkeypatch):
    monkeypatch.setattr(A, "cached_profile", lambda d: windows_no_vtk())
    win = MainWindow()
    win._refresh_machine_status()
    text = win._machine_label.text()
    assert "GL" not in text  # gl is None on this fixture
    assert "0.0 B" not in text


def test_the_readout_never_probes_gl(monkeypatch):
    monkeypatch.setattr(A, "cached_profile", lambda d: tiny_ram())
    monkeypatch.setattr(
        A.machine, "probe_gl",
        lambda **kw: pytest.fail("the status bar must never probe GL"),
    )
    win = MainWindow()
    win._refresh_machine_status()
    assert win._machine_label.text()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_gui_status_bar.py -v`
Expected: FAIL — `AttributeError: 'MainWindow' object has no attribute '_machine_label'`.

- [ ] **Step 3: Write the implementation**

In `gui/main_window.py`, add the imports:

```python
import os

from PySide6.QtCore import QTimer

from dfxm.common.advice import human_bytes

from . import advisor
```

After `self.setCentralWidget(splitter)`:

```python
        # Ambient machine readout. Cheap fields only, on a timer — it must never
        # probe GL (that costs a child process) and never block the UI.
        self._machine_label = QLabel("")
        self.statusBar().addPermanentWidget(self._machine_label)
        self._machine_timer = QTimer(self)
        self._machine_timer.setInterval(5000)
        self._machine_timer.timeout.connect(self._refresh_machine_status)
        self._machine_timer.start()
        self._refresh_machine_status()
```

and the slot:

```python
    def _refresh_machine_status(self) -> None:
        """Cores, free disk, RAM and (only if already probed) the GL stack.

        Unmeasured fields are omitted rather than shown as zero: a probe that
        failed is recorded in `probe_errors`, and "0.0 B free" would read as a
        full disk.
        """
        prof = advisor.cached_profile(os.getcwd())
        parts = [f"{prof.cpu_logical} cores"]
        if prof.disk_free:
            parts.append(f"{human_bytes(prof.disk_free)} free")
        if prof.ram_total:
            parts.append(
                f"{human_bytes(prof.ram_available)}/{human_bytes(prof.ram_total)} RAM"
            )
        if prof.gl is not None:
            parts.append("software GL" if prof.gl.software else "hardware GL")
        self._machine_label.setText(" · ".join(parts))
```

Note the module-level `advisor` import and `advisor.cached_profile(...)` call —
not `from .advisor import cached_profile` — so the tests' `monkeypatch.setattr(A,
"cached_profile", ...)` is seen.

- [ ] **Step 4: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_gui_status_bar.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Run the mutation**

Drop the `if prof.disk_free:` guard so the disk part is always appended.
Expected: `test_unmeasured_fields_are_omitted_not_shown_as_zero` FAILS — but
only if `windows_no_vtk` has a zero field; check, and if its `disk_free` is
non-zero, mutate the `if prof.gl is not None` guard instead and confirm the
`"GL" not in text` assertion fails. Restore.

- [ ] **Step 6: Docs, suite, commit**

`docs/Usage.md`: one line on the status bar and what it shows. `docs/Codebase.md`:
`MainWindow._machine_label` / `_refresh_machine_status`.

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add gui/main_window.py tests/test_gui_status_bar.py docs/Usage.md docs/Codebase.md
git commit -m "feat: keep the machine's limits visible in the status bar"
```

---

### Task 8: `Param.advice_key` and per-field notes

Schema plumbing only — no advisory is wired in yet. Keeps the GUI
schema-driven per CLAUDE.md rather than hard-coding which fields can carry a
note.

**Files:**
- Modify: `dfxm/config/models.py` (the `Param` dataclass, lines 40-75)
- Modify: `gui/widgets/param_form.py` (constructor loops at 84-115; public API)
- Modify: `docs/Codebase.md`
- Test: `tests/test_param_metadata.py`, `tests/test_gui_param_notes.py` (create)

**Interfaces:**
- Consumes: nothing.
- Produces: `Param.advice_key: str = ""`;
  `ParamForm.set_field_note(name: str, text: str) -> None`;
  `ParamForm.apply_hints(hints: dict[str, str]) -> None`. Task 9 calls
  `apply_hints`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_param_metadata.py`:

```python
def test_advice_key_defaults_empty_and_is_settable():
    assert Param("x", ParamType.STR, "X").advice_key == ""
    assert Param("x", ParamType.STR, "X", advice_key="3d_texture").advice_key == "3d_texture"
```

Create `tests/test_gui_param_notes.py`:

```python
"""Per-field advisory notes rendered under a form widget (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import Param, ParamType  # noqa: E402
from gui.widgets.param_form import ParamForm  # noqa: E402

_PARAMS = (
    Param("mode", ParamType.ENUM, "Mode", default="volume",
          choices=("volume", "surface"), advice_key="3d_texture"),
    Param("plain", ParamType.STR, "Plain"),
)


def test_a_note_row_exists_only_for_params_that_declare_a_key():
    form = ParamForm(_PARAMS)
    assert "mode" in form._notes
    assert "plain" not in form._notes


def test_setting_a_note_shows_it_and_clearing_hides_it():
    form = ParamForm(_PARAMS)
    assert form._notes["mode"].isVisibleTo(form) is False  # precondition
    form.set_field_note("mode", "this GL stack caps 3-D textures at 2048 px")
    assert "2048" in form._notes["mode"].text()
    assert form._notes["mode"].isVisibleTo(form) is True
    form.set_field_note("mode", "")
    assert form._notes["mode"].text() == ""
    assert form._notes["mode"].isVisibleTo(form) is False


def test_apply_hints_routes_by_advice_key_and_clears_the_rest():
    form = ParamForm(_PARAMS)
    form.apply_hints({"3d_texture": "downsample 2x"})
    assert "downsample 2x" in form._notes["mode"].text()
    form.apply_hints({})
    assert form._notes["mode"].text() == ""


def test_setting_a_note_on_a_keyless_param_is_a_no_op():
    form = ParamForm(_PARAMS)
    form.set_field_note("plain", "ignored")  # must not raise


def test_the_editor_dict_still_holds_the_real_widget():
    """Other code and tests reach into _editors expecting the editor itself
    (tests/gui_smoke.py:255, tests/test_gui_wheel_guard.py) — a note row must
    not wrap it."""
    from PySide6.QtWidgets import QComboBox

    form = ParamForm(_PARAMS)
    assert isinstance(form._editors["mode"], QComboBox)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_param_notes.py tests/test_param_metadata.py -k "note or advice_key" -v`
Expected: FAIL — `TypeError: Param.__init__() got an unexpected keyword argument 'advice_key'`.

- [ ] **Step 3: Add the schema field**

In `dfxm/config/models.py`, after `roi_frame`:

```python
    advice_key: str = ""  # key into Advisory.hints -> a note under this field
```

and one line in the `Param` docstring:

```
    ``advice_key`` names the advisory this parameter can carry: when
    ``Advisory.hints`` has that key, the form renders its text as a note under
    the field. Declaring it here rather than in the GUI keeps the form
    schema-driven.
```

- [ ] **Step 4: Render the note rows**

In `gui/widgets/param_form.py`, add `self._notes: dict[str, QLabel] = {}` beside
`self._labels`, then add the note row after each `addRow` in **both** loops:

```python
        for p in essentials:
            ess_form.addRow(self._label_for(p), self._make_editor(p, initial))
            self._add_note_row(ess_form, p)
```

```python
                form.addRow(self._label_for(p), self._make_editor(p, initial))
                self._add_note_row(form, p)
```

and the three methods:

```python
    def _add_note_row(self, form: QFormLayout, p: Param) -> None:
        """A hidden, full-width note row under *p*'s editor.

        Only for params that declare an ``advice_key`` — a hidden row per field
        would be dead weight on every form. The editor itself is NOT wrapped:
        `self._editors[name]` must stay the real widget, which `gui_smoke` and
        the wheel-guard tests reach into directly.
        """
        if not p.advice_key:
            return
        note = QLabel("")
        note.setWordWrap(True)
        note.setProperty("role", "warning")
        note.setVisible(False)
        form.addRow(note)
        self._notes[p.name] = note

    def set_field_note(self, name: str, text: str) -> None:
        """Show *text* under *name*'s editor; empty text hides the row."""
        note = self._notes.get(name)
        if note is None:
            return
        note.setText(text)
        note.setVisible(bool(text))

    def apply_hints(self, hints: dict) -> None:
        """Route an advisory's hints to their fields, clearing every other note.

        Clearing matters: a hint that no longer applies (the user picked a
        lighter render mode) must disappear rather than linger as advice about
        a setting they already changed.
        """
        for p in self._params:
            if p.advice_key:
                self.set_field_note(p.name, hints.get(p.advice_key, ""))
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_param_notes.py tests/test_param_metadata.py -v`
Expected: PASS.

- [ ] **Step 6: Run the mutation**

Change `apply_hints` to `if p.advice_key and p.advice_key in hints:` (i.e. stop
clearing). Expected: `test_apply_hints_routes_by_advice_key_and_clears_the_rest`
FAILS on the second assertion. Restore.

- [ ] **Step 7: Docs, suite, commit**

`docs/Codebase.md`: `Param.advice_key`, `ParamForm._notes` / `set_field_note` /
`apply_hints`.

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add dfxm/config/models.py gui/widgets/param_form.py \
        tests/test_gui_param_notes.py tests/test_param_metadata.py docs/Codebase.md
git commit -m "feat: let a parameter declare the advisory it can carry"
```

---

### Task 9: 3-D hints and the one-shot background GL probe

`advise_3d` already knows that volume mode uploads the grid as one 3-D texture
and renders **blank** past `GL_MAX_3D_TEXTURE_SIZE` — a silently empty product.
Nothing has ever shown that to the user. It needs `GLInfo`, which needs the
probe child, so this task also adds the one-shot background probe.

**Files:**
- Modify: `dfxm/common/advisory.py` (populate `Advisory.hints`)
- Modify: `dfxm/stages/visualize.py` (the `render_mode` Param at ~line 363)
- Modify: `gui/advisor.py` (background GL probe)
- Modify: `gui/stage_view.py` (`_show_advisory` applies hints)
- Modify: `docs/Usage.md`, `docs/Codebase.md`
- Test: `tests/test_common_advisory.py`, `tests/test_gui_advisor.py`,
  `tests/test_gui_stage_advice.py`

**Interfaces:**
- Consumes: `advice.advise_3d`; `ParamForm.apply_hints` (Task 8);
  `Advisory.hints` (Task 3).
- Produces: `advisory.HINT_3D_TEXTURE = "3d_texture"`;
  `gui.advisor.probe_gl_async()`; `gui.advisor.gl_ready() -> bool`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_common_advisory.py`:

```python
from dfxm.common.advisory import HINT_3D_TEXTURE
from tests.machine_fixtures import laptop_hw_gl


def test_an_oversized_volume_on_software_gl_gets_a_texture_hint():
    spec = _spec_with("tests.test_common_advisory:_wide_estimate")
    prof = workstation_sw_gl()
    # Precondition: the fixture volume really does exceed this GL stack's cap,
    # or the hint under test is not the one being exercised.
    assert prof.gl.max_3d_texture == 2048
    adv = advise_stage(spec, {"render_mode": "volume"}, profile=prof)
    assert HINT_3D_TEXTURE in adv.hints
    assert "2048" in adv.hints[HINT_3D_TEXTURE]


def test_no_texture_hint_when_the_volume_fits():
    spec = _spec_with("tests.test_common_advisory:_wide_estimate")
    adv = advise_stage(spec, {"render_mode": "volume"}, profile=laptop_hw_gl())
    assert HINT_3D_TEXTURE not in adv.hints


def test_no_texture_hint_for_geometry_render_modes():
    """Surface/isosurface upload geometry, not one big texture."""
    spec = _spec_with("tests.test_common_advisory:_wide_estimate")
    adv = advise_stage(spec, {"render_mode": "surface"}, profile=workstation_sw_gl())
    assert HINT_3D_TEXTURE not in adv.hints


def test_no_texture_hint_when_gl_is_unprobed():
    spec = _spec_with("tests.test_common_advisory:_wide_estimate")
    adv = advise_stage(spec, {"render_mode": "volume"}, profile=windows_no_vtk())
    assert adv.hints == {}


def _wide_estimate(params):
    return CostEstimate(1 * GB, 1 * GB, (76, 1200, 2891), True)
```

Add `windows_no_vtk` to that file's `machine_fixtures` import.

Append to `tests/test_gui_advisor.py`:

```python
def test_gl_is_not_probed_until_asked(monkeypatch):
    monkeypatch.setattr(
        A.machine, "probe_gl",
        lambda **kw: pytest.fail("GL must not be probed by the cost path"),
    )
    A.clear_profile_cache()
    A._set_gl_ready(False)
    A.cached_profile(os.getcwd())


def test_once_probed_the_cached_profile_carries_gl(monkeypatch):
    from dfxm.common.machine import GLInfo

    monkeypatch.setattr(
        A.machine, "probe_gl",
        lambda **kw: (GLInfo("llvmpipe", "Mesa", "4.5", 2048, True), "ok"),
    )
    A.clear_profile_cache()
    A._set_gl_ready(False)
    A.probe_gl_async()
    _drain(10.0)
    assert A.gl_ready() is True
    A.clear_profile_cache()
    assert A.cached_profile(os.getcwd()).gl is not None
```

Append to `tests/test_gui_stage_advice.py`:

```python
def test_hints_reach_the_form():
    view = StageView("visualize", STAGE_SPECS["visualize"], Experiment())
    view._show_advisory(
        Advisory(workstation_sw_gl(), None, None, "a headline", (),
                 hints={"3d_texture": "downsample 2x or volume mode renders blank"})
    )
    assert "renders blank" in view._form._notes["render_mode"].text()
    view._show_advisory(Advisory(workstation_sw_gl(), None, None, "a headline", ()))
    assert view._form._notes["render_mode"].text() == ""
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_common_advisory.py tests/test_gui_advisor.py tests/test_gui_stage_advice.py -v`
Expected: FAIL — `ImportError: cannot import name 'HINT_3D_TEXTURE'`.

- [ ] **Step 3: Populate the hints**

In `dfxm/common/advisory.py`:

```python
# Advisory key for the 3-D texture ceiling. Declared on the parameter it belongs
# to (`visualize`'s `render_mode`) rather than matched by name in the GUI.
HINT_3D_TEXTURE = "3d_texture"


def _hints(profile: MachineProfile, estimate: CostEstimate, params: dict) -> dict[str, str]:
    """Per-parameter advisories. Empty until GL has actually been probed.

    The texture ceiling is not cosmetic: volume mode uploads the grid as ONE
    3-D texture, and past `GL_MAX_3D_TEXTURE_SIZE` VTK renders nothing at all —
    a silently blank product rather than an error.
    """
    mode = str(params.get("render_mode") or "")
    if not mode or profile.gl is None or estimate.shape is None:
        return {}
    result = advice.advise_3d(profile, estimate.shape, mode)
    if not result.reasons:
        return {}
    return {HINT_3D_TEXTURE: " ".join(result.reasons)}
```

and pass it into the returned `Advisory`:

```python
        hints=_hints(prof, estimate, params),
```

- [ ] **Step 4: Declare the key on the parameter**

In `dfxm/stages/visualize.py`, add to the `render_mode` `Param(...)` call:

```python
            advice_key="3d_texture",
```

- [ ] **Step 5: Add the background GL probe**

In `gui/advisor.py`:

```python
_gl_ready = False


def gl_ready() -> bool:
    """True once a GL probe has succeeded in this session."""
    return _gl_ready


def _set_gl_ready(value: bool) -> None:
    """Test seam; production code sets this through `probe_gl_async`."""
    global _gl_ready
    _gl_ready = value


class _GlProbeWorker(QThread):
    """The one GL probe this session pays for. Result is cached on disk by
    `machine.probe_gl`, so every later launch is a file read."""

    finished_ok = Signal(bool)

    def run(self) -> None:  # worker thread
        try:
            _info, status = machine.probe_gl()
        except Exception:  # noqa: BLE001 — a dead probe is a result, not a crash
            status = "crashed"
        self.finished_ok.emit(status == "ok")


_gl_worker: _GlProbeWorker | None = None


def probe_gl_async() -> None:
    """Probe GL once per session, off the GUI thread.

    Costs a child process, so it is never triggered by the cost path. Once it
    lands, `cached_profile` starts including GL — `machine.probe_gl` memoises
    in-process and caches on disk, so that is a lookup, not another child.
    """
    global _gl_worker
    if _gl_ready or _gl_worker is not None:
        return

    def _done(ok: bool) -> None:
        global _gl_worker
        _gl_worker = None
        if ok:
            _set_gl_ready(True)
            clear_profile_cache()  # so the next profile carries GL

    worker = _GlProbeWorker()
    worker.finished_ok.connect(_done)
    _gl_worker = worker
    keep_alive(worker)
    worker.start()
```

and in `cached_profile`, replace the probe line with:

```python
    prof = machine.profile(output_dir=output_dir, probe_gl_now=_gl_ready)
```

- [ ] **Step 6: Kick the probe and apply the hints in `StageView`**

In `gui/stage_view.py`, extend `_show_advisory`:

```python
        self._form.apply_hints(advisory.hints if advisory is not None else {})
```

and in `showEvent`, after `self._advisor.request()`:

```python
        # A stage with a 3-D setting is the only reason to pay for a GL probe.
        if any(p.advice_key == "3d_texture" for p in self._spec.params):
            probe_gl_async()
```

with `from .advisor import StageAdvisor, probe_gl_async` at the top.

- [ ] **Step 7: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_common_advisory.py tests/test_gui_advisor.py tests/test_gui_stage_advice.py -v`
Expected: PASS.

- [ ] **Step 8: Run the mutations**

1. Drop the `mode != "volume"` protection by passing `"volume"` unconditionally
   into `advise_3d`. Expected: `test_no_texture_hint_for_geometry_render_modes`
   FAILS.
2. Change `probe_gl_now=_gl_ready` back to `probe_gl_now=False`. Expected:
   `test_once_probed_the_cached_profile_carries_gl` FAILS.
3. Remove the `apply_hints` call from `_show_advisory`. Expected:
   `test_hints_reach_the_form` FAILS.

- [ ] **Step 9: Docs, suite, commit**

`docs/Usage.md`: the 3-D note under the render-mode field, what the texture cap
means, and that volume mode past the cap renders blank rather than erroring.
`docs/Codebase.md`: `HINT_3D_TEXTURE`, `_hints`, `probe_gl_async`, `gl_ready`,
`visualize`'s `render_mode.advice_key`.

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add dfxm/common/advisory.py dfxm/stages/visualize.py gui/advisor.py \
        gui/stage_view.py tests/ docs/Usage.md docs/Codebase.md
git commit -m "feat: warn on the form when volume mode would render blank"
```

---

### Task 10: The System check dialog

**Files:**
- Create: `gui/widgets/system_check.py`
- Modify: `gui/main_window.py` (rail buttons at 115-141)
- Modify: `docs/Usage.md`, `docs/Codebase.md`
- Test: `tests/test_gui_system_check.py` (create)

**Interfaces:**
- Consumes: `machine.profile`, `advice.headroom_bytes`, `advice.human_bytes`,
  `gui.advisor.clear_profile_cache`.
- Produces: `SystemCheckDialog(parent=None, profile=None)` with `rows() ->
  list[tuple[str, str, str]]` (label, value, implication) and `as_text() -> str`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_system_check.py`:

```python
"""The System check probe table (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui.widgets.system_check import SystemCheckDialog  # noqa: E402
from tests.machine_fixtures import tiny_ram, windows_no_vtk, workstation_sw_gl  # noqa: E402


def _labels(dlg):
    return [label for label, _value, _implication in dlg.rows()]


def test_every_probe_gets_a_row():
    dlg = SystemCheckDialog(profile=workstation_sw_gl())
    for expected in ("CPU", "RAM", "Headroom", "Disk", "OpenGL", "ffmpeg"):
        assert expected in _labels(dlg)


def test_software_gl_row_explains_the_consequence():
    prof = workstation_sw_gl()
    assert prof.gl.software is True  # precondition
    dlg = SystemCheckDialog(profile=prof)
    gl_row = next(r for r in dlg.rows() if r[0] == "OpenGL")
    assert "2048" in gl_row[1]
    assert "surface" in gl_row[2].lower()


def test_a_failed_probe_reads_as_unknown_with_its_reason():
    prof = tiny_ram()
    assert prof.gl_status == "crashed" and prof.probe_errors  # precondition
    dlg = SystemCheckDialog(profile=prof)
    gl_row = next(r for r in dlg.rows() if r[0] == "OpenGL")
    assert "unknown" in gl_row[1].lower() or "crashed" in gl_row[1].lower()
    assert "child exited" in dlg.as_text()


def test_missing_ffmpeg_says_what_is_lost():
    prof = windows_no_vtk()
    assert prof.ffmpeg is None  # precondition
    dlg = SystemCheckDialog(profile=prof)
    row = next(r for r in dlg.rows() if r[0] == "ffmpeg")
    assert "GIF" in row[2]


def test_as_text_is_copyable_plain_text():
    dlg = SystemCheckDialog(profile=workstation_sw_gl())
    text = dlg.as_text()
    assert "CPU" in text and "<" not in text
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_gui_system_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.widgets.system_check'`.

- [ ] **Step 3: Write the dialog**

Create `gui/widgets/system_check.py`:

```python
"""What this machine is, and what it implies for settings.

The only surface that pays for a GL probe on demand. Renders a
:class:`~dfxm.common.machine.MachineProfile`; decides nothing.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from dfxm.common import advice, machine
from dfxm.common.advice import human_bytes

from ..advisor import clear_profile_cache
from .busy import busy_cursor

_UNKNOWN = "unknown"


class SystemCheckDialog(QDialog):
    """A probe table: measured value and what it means, one row per probe."""

    def __init__(self, parent=None, profile=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("System check")
        self.resize(760, 420)
        self._profile = profile if profile is not None else self._measure()

        self._table = QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Probe", "Measured", "What it means"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        self._errors = QLabel("")
        self._errors.setWordWrap(True)
        self._errors.setProperty("role", "warning")

        self._reprobe_btn = QPushButton("Re-probe")
        self._reprobe_btn.clicked.connect(self._on_reprobe)
        self._copy_btn = QPushButton("Copy as text")
        self._copy_btn.clicked.connect(self._on_copy)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self._reprobe_btn)
        btn_row.addWidget(self._copy_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(buttons)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addWidget(self._errors)
        layout.addLayout(btn_row)
        self._rebuild()

    # -- probing ----------------------------------------------------------
    @staticmethod
    def _measure(*, use_cache: bool = True):
        """Measure this machine, GL included. The one place that pays for it."""
        with busy_cursor("Probing this machine…"):
            if not use_cache:
                machine.probe_gl(use_cache=False)
                clear_profile_cache()
            return machine.profile(probe_gl_now=True)

    def _on_reprobe(self) -> None:
        self._profile = self._measure(use_cache=False)
        self._rebuild()

    def _on_copy(self) -> None:
        QApplication.clipboard().setText(self.as_text())

    # -- content ----------------------------------------------------------
    def rows(self) -> list[tuple[str, str, str]]:
        """(label, measured value, what it means) for every probe."""
        p = self._profile
        out: list[tuple[str, str, str]] = [
            (
                "CPU",
                f"{p.cpu_logical} logical / {p.cpu_physical or _UNKNOWN} physical",
                "Measured only — the pipeline does not parallelise yet.",
            ),
            (
                "RAM",
                f"{human_bytes(p.ram_available)} available of {human_bytes(p.ram_total)}"
                if p.ram_total
                else _UNKNOWN,
                "Stages stream when a run does not fit; slower, same result.",
            ),
            (
                "Headroom",
                human_bytes(advice.headroom_bytes(p)),
                "The most a run will plan to use, leaving room for Qt and the OS.",
            ),
            (
                "Disk",
                f"{human_bytes(p.disk_free)} free" if p.disk_free else _UNKNOWN,
                "The one genuine blocker: a run that must spill to scratch needs this.",
            ),
        ]
        if p.gl is not None:
            cap = p.gl.max_3d_texture or _UNKNOWN
            out.append(
                (
                    "OpenGL",
                    f"{p.gl.renderer} · 3-D texture cap {cap} px",
                    "Software renderer — prefer surface mode; volume mode renders "
                    "blank past the cap."
                    if p.gl.software
                    else "Hardware accelerated.",
                )
            )
        else:
            out.append(
                (
                    "OpenGL",
                    f"{_UNKNOWN} ({p.gl_status})",
                    "3-D products may render blank; re-probe, or use surface mode.",
                )
            )
        out.append(
            (
                "ffmpeg",
                p.ffmpeg or "not found",
                "MP4 export needs it; without it exports fall back to GIF.",
            )
        )
        return out

    def as_text(self) -> str:
        """Plain text for the clipboard — no markup, safe to paste anywhere."""
        lines = [f"{label}: {value} — {implication}" for label, value, implication in self.rows()]
        lines.extend(self._profile.probe_errors)
        return "\n".join(lines)

    def _rebuild(self) -> None:
        rows = self.rows()
        self._table.setRowCount(len(rows))
        for r, (label, value, implication) in enumerate(rows):
            for c, text in enumerate((label, value, implication)):
                self._table.setItem(r, c, QTableWidgetItem(text))
        self._table.resizeColumnsToContents()
        errors = self._profile.probe_errors
        self._errors.setText("\n".join(errors))
        self._errors.setVisible(bool(errors))
```

Check `busy_cursor`'s signature in `gui/widgets/busy.py:96` before using it —
it takes `(text="", widget=None)` and is a context manager.

- [ ] **Step 4: Wire the rail button**

In `gui/main_window.py`, beside `self._pub_style_btn`:

```python
        # "System check…" — what this machine is and what it implies for
        # settings. The only surface that pays for a GL probe on demand.
        self._system_check_btn = QPushButton("System check…")
        self._system_check_btn.clicked.connect(self._on_system_check)
```

```python
        left_layout.addWidget(self._system_check_btn)
```
(placed after `self._pub_style_btn`), and:

```python
    def _on_system_check(self) -> None:
        from .widgets.system_check import SystemCheckDialog

        SystemCheckDialog(self).exec()
```

Lazy-imported to match how `Figure builder…` is handled in the same file.

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_gui_system_check.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Run the mutation**

Make the OpenGL row's implication a constant `"Hardware accelerated."`
regardless of `prof.gl.software`. Expected:
`test_software_gl_row_explains_the_consequence` FAILS. Restore.

- [ ] **Step 7: Docs, suite, commit**

`docs/Usage.md`: a subsection on the System check dialog — how to open it, what
each row means, when to press Re-probe, and that Copy as text is for support.
`docs/Codebase.md`: `gui/widgets/system_check.py` and `MainWindow._on_system_check`.

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format .
git add gui/widgets/system_check.py gui/main_window.py \
        tests/test_gui_system_check.py docs/Usage.md docs/Codebase.md
git commit -m "feat: show what this machine is and what it means for settings"
```

---

### Task 11: Smoke coverage and final verification

**Files:**
- Modify: `tests/gui_smoke.py`
- Modify: `docs/Usage.md` (final read-through only if something is missing)

**Interfaces:**
- Consumes: everything above.
- Produces: no new API.

- [ ] **Step 1: Read `tests/gui_smoke.py` and add two steps**

Follow the file's existing numbered-step convention exactly (read it first — it
is not a pytest file and has its own harness). Add:

- a step that opens a stage view, drives the event loop until
  `view._advisor.latest` is not None, and asserts the cost line is populated;
- a step that opens `SystemCheckDialog`, asserts `rows()` is non-empty and
  `as_text()` is non-empty, and closes it.

- [ ] **Step 2: Run the smoke test twice**

```bash
DISPLAY= python3 -u tests/gui_smoke.py
DISPLAY= python3 -u tests/gui_smoke.py
```

Expected: all steps pass. **Step `[41]` is genuinely intermittent** — it aborts
in roughly half of all runs on an unmodified tree. Run 2–3 times before
attributing any `[41]` failure to this work; never conclude a regression from a
single failure.

- [ ] **Step 3: Full suite and lint**

```bash
python3 -m pytest -q --deselect tests/test_gui_viewer3d.py
ruff check . && ruff format --check .
```

Expected: pass counts at or above the 1435 passed / 13 skipped / 17 deselected
baseline, plus the ~35 tests this plan adds. Known flake:
`tests/test_gui_figure_builder.py::test_export_now_writes_files` (timing).

- [ ] **Step 4: Audit the mutation log**

Confirm every task above recorded a mutation that actually failed. Any test
whose mutation did **not** fail is not testing what it names — fix it now, not
later. This is the recurring defect in this project: twenty checks have been
found to have stopped checking what they name, and two of those were introduced
by the fix for the previous one.

- [ ] **Step 5: Verify the docs contract**

```bash
git log --oneline <base>..HEAD --stat | grep -c "docs/"
```

Every commit touching `dfxm/stages/` or `gui/` must also touch `docs/`. Fix any
that do not before finishing.

- [ ] **Step 6: Commit**

```bash
git add tests/gui_smoke.py docs/
git commit -m "test: cover the cost line and the system check in the smoke run"
```

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: `advisory.py` → Tasks 2–3;
`CostEstimate.confidence` → Task 1; `gui/advisor.py` → Task 4; live line →
Task 5; pre-flight banner, `banner-info` and the blocked confirmation → Task 6;
status bar → Task 7; `Param.advice_key` and the form note → Task 8; GL policy
and `advise_3d` hints → Task 9; System check dialog → Task 10; testing and smoke
→ every task plus Task 11; docs → folded into each task per the contract.

**Type consistency.** `Advisory` is constructed in Task 3 and consumed unchanged
by Tasks 5, 6, 9. `advise_stage(spec, params, *, profile=None)` keeps that
signature everywhere. `advice.human_bytes` is renamed once in Task 3 and used by
Tasks 3, 7, 10. `ParamForm.apply_hints` is defined in Task 8 and called in
Task 9. `StageAdvisor.compute_blocking` is defined in Task 4 and called in
Task 6.

**Known deviation to watch.** Task 6 extracts `_start_runner` out of `_on_run`.
That is a refactor of existing behaviour, not new behaviour: move the body
verbatim and change nothing inside it, or the ETA reset and progress-clearing
sequence will drift.
