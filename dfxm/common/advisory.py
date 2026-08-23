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
from dataclasses import dataclass, field

from ..config.models import CostEstimate, StageSpec
from . import advice, alignment, machine
from .advice import RunPlan
from .machine import MachineProfile

# How each RunPlan.strategy is phrased. Hedged deliberately: the prediction and
# the run share a headroom figure so they agree in direction, but they are
# computed independently and this must not read as an instruction. All three
# say "expected to" — the advisory computes this from `headroom_bytes` while
# the stage itself budgets from `working_set_budget_bytes` (its own measured
# RSS floor), a genuinely different number, so "runs in memory" stated as fact
# would be no more certain than the other two strategies.
_STRATEGY_WORDS = {
    "in-core": "expected to run in memory",
    "chunked": "expected to stream",
    "disk-backed": "expected to run disk-backed",
}

_CONSERVATIVE_NOTE = (
    "this stage's estimate has not been recalibrated since the streaming "
    "rewrite — it over-predicts, so the run may be much lighter than shown"
)

_CHUNK_REPLACEMENT = "blocking the work into groups — slower, same result"

# Advisory key for the 3-D texture ceiling. Declared on the parameter it belongs
# to (`visualize`'s `render_mode`) rather than matched by name in the GUI.
HINT_3D_TEXTURE = "3d_texture"


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
    """One line naming the cost, the headroom, and the expected strategy.

    Both byte counts are RAM, and the lead says so: the same banner can also
    carry `plan.reasons` lines measured in scratch *disk*, and an unlabelled
    "N safely available" beside them reads as whichever resource the eye
    reached for last.
    """
    need = advice.human_bytes(estimate.peak_bytes)
    lead = f"at most ~{need} RAM (conservative estimate)" if conservative else f"needs ~{need} RAM"
    strategy = _STRATEGY_WORDS.get(plan.strategy, plan.strategy)
    return f"{lead}, {advice.human_bytes(plan.budget_bytes)} safely available — {strategy}"


def _details(plan: RunPlan, conservative: bool) -> tuple[str, ...]:
    """`plan.reasons`, with the chunk-count sentence replaced and the
    headline-duplicating in-core sentence dropped.

    NOT `plan.reasons` verbatim, for two reasons:

    * `plan_run` writes the group count into its own reasons, and that number is
      display-only — it is not the blocking a stage picks. Pinned by
      `advice.CHUNK_REASON_PREFIX` so a reworded reason fails a test rather than
      leaking the count.
    * An in-core plan's headroom reason states the same three facts as
      `_headline` (peak, budget, strategy), and every surface renders the two
      together — `StageView._on_run` stacks them in the pre-flight banner, and
      the cost line puts the details in the tooltip under that same headline.
      Rendered as-is it said the same thing twice. Dropped by
      `advice.INCORE_REASON_SUFFIX`; see that constant for why it is keyed on
      the sentence rather than on `plan.strategy`.
    """
    out = [
        _CHUNK_REPLACEMENT if r.startswith(advice.CHUNK_REASON_PREFIX) else r
        for r in plan.reasons
        if not r.endswith(advice.INCORE_REASON_SUFFIX)
    ]
    if conservative:
        out.append(_CONSERVATIVE_NOTE)
    return tuple(out)


# Which of a stage's folder-pattern params to try when widening a raw shape to
# its aligned extent (see `_aligned_shape_for_hint`). Only `visualize` wires a
# `render_mode` param today, and its two candidate volumes (the mosaicity file
# and the strain file) are selected by these two params respectively.
_ALIGNMENT_PATTERN_KEYS = ("mosa_pattern", "strain_pattern")


def _aligned_shape_for_hint(raw_shape: tuple[int, ...], params: dict) -> tuple[int, ...]:
    """Widen *raw_shape* to what the alignment chain will actually upload.

    `estimate.shape` is the on-disk shape read straight out of the volume
    file's HDF5 dataset — see e.g. `visualize.estimate()`'s docstring. What VTK
    uploads for volume mode is the *aligned* array
    (`apply_roi_3d -> apply_samy_shifts_to_volume -> interpolate_to_uniform_z`),
    and the last two steps each widen the canvas — `aligned_extent`'s docstring
    spells out why the two inflations multiply. Comparing the raw shape against
    the GL cap therefore stays silent in exactly the dangerous direction: a
    volume that would render blank can still read as under the cap.

    Tries each of `_ALIGNMENT_PATTERN_KEYS` (a stage may have more than one
    candidate volume, e.g. visualize's mosaicity file and strain file) via
    `alignment.aligned_shape_for_params` and keeps whichever widens the shape
    the most, so the hint never goes quiet just because the wrong volume's
    motors were read. Falls back to *raw_shape* unchanged when no pattern
    resolves (no `raw_root` yet, motors unreadable, ...) — same as before this
    widening existed, never worse. No new file IO: `aligned_shape_for_params`
    only reads motor positions, which `raster.motor_positions_for_estimate`
    memoises, and *raw_shape* is already paid for by the estimator call that
    produced it.
    """
    best = raw_shape
    for pattern_key in _ALIGNMENT_PATTERN_KEYS:
        candidate = alignment.aligned_shape_for_params(params, raw_shape, pattern_key=pattern_key)
        if candidate is not None and max(candidate) > max(best):
            best = candidate
    return best


def _hints(profile: MachineProfile, estimate: CostEstimate, params: dict) -> dict[str, str]:
    """Per-parameter advisories. Empty until GL has actually been probed.

    The texture ceiling is not cosmetic: volume mode uploads the grid as ONE
    3-D texture, and past `GL_MAX_3D_TEXTURE_SIZE` VTK renders nothing at all —
    a silently blank product rather than an error. Checked against the
    *aligned* shape (`_aligned_shape_for_hint`), not `estimate.shape` as read
    off disk — see that helper's docstring for why the raw shape under-states
    the risk.

    Wrapped in its own try/except (rather than relying on `advise_stage`'s
    outer one, which only covers the estimator call) because this function is
    handed `params["render_mode"]` straight from the user-facing form: the
    enum constrains it in production, but `advise_stage`'s never-raises
    contract is absolute and this guard is cheap. Any failure here degrades to
    "no hint" rather than taking the caller's window down.
    """
    try:
        mode = str(params.get("render_mode") or "")
        if not mode or profile.gl_status != "ok" or profile.gl is None or estimate.shape is None:
            return {}
        shape = _aligned_shape_for_hint(estimate.shape, params)
        result = advice.advise_3d(profile, shape, mode)
        if not result.reasons:
            return {}
        return {HINT_3D_TEXTURE: " ".join(result.reasons)}
    except Exception:  # noqa: BLE001 — advise_stage must never raise
        return {}


def advise_stage(
    spec: StageSpec, params: dict, *, profile: MachineProfile | None = None
) -> Advisory:
    """What this run will cost on this machine, ready to render.

    **Never raises.** Estimators open user-supplied HDF5 paths and read raw scan
    folders; a missing file, a corrupt file or a half-typed path is the ordinary
    state of a form being filled in, not an error. Any failure becomes a
    headline naming it — an exception escaping into a form-change handler would
    take the window with it. The whole body from the estimator call onward
    sits inside one `try`: an estimator returning something that is not a
    genuine `CostEstimate` (wrong type, `None`, ...) would otherwise raise
    `AttributeError` on `estimate.peak_bytes` *outside* the guard, and
    `StageView._on_run`'s `compute_blocking()` call (the GUI thread, on the
    Run click) does not catch it — the same never-raises promise this
    docstring makes, just broken one call later.

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

        if not estimate.peak_bytes:
            return Advisory(
                prof, estimate, None, estimate.note or "not enough input to estimate cost"
            )

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
            hints=_hints(prof, estimate, params),
        )
    except Exception as exc:  # noqa: BLE001 — delivered to the user as text
        return Advisory(prof, None, None, f"cannot estimate cost: {type(exc).__name__}")
