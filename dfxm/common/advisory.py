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
from . import advice, machine
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
        _CHUNK_REPLACEMENT if r.startswith(advice.CHUNK_REASON_PREFIX) else r for r in plan.reasons
    ]
    if conservative:
        out.append(_CONSERVATIVE_NOTE)
    return tuple(out)


def _hints(profile: MachineProfile, estimate: CostEstimate, params: dict) -> dict[str, str]:
    """Per-parameter advisories. Empty until GL has actually been probed.

    The texture ceiling is not cosmetic: volume mode uploads the grid as ONE
    3-D texture, and past `GL_MAX_3D_TEXTURE_SIZE` VTK renders nothing at all —
    a silently blank product rather than an error.

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
        result = advice.advise_3d(profile, estimate.shape, mode)
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
        hints=_hints(prof, estimate, params),
    )
