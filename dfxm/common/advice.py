"""Machine-aware policy: what should this run actually do? (Qt-free, no IO.)

Pure functions over a :class:`~dfxm.common.machine.MachineProfile` and a
:class:`~dfxm.config.models.CostEstimate`. Everything here is deterministic and
side-effect-free, so every decision is testable against synthetic machines we
do not own (see ``tests/machine_fixtures.py``).

The governing rule: **nothing refuses to run for lack of RAM.** The escalation
is in-core -> chunked -> disk-backed, and the only genuine blocker is running
out of *disk*, which is measured and reported before work starts.

Every decision carries a plain-language reason. Those strings are what the log,
the GUI banner and the stage result notes all display, so each explanation is
written exactly once — here.
"""

from __future__ import annotations

from dataclasses import dataclass

# Headroom: never plan to use more than this share of memory, leaving room for
# Qt, matplotlib, h5py buffers and the OS. Two limits, the tighter one wins:
# available RAM guards against other processes, total RAM guards against a
# misleadingly large "available" on a machine with a huge page cache.
AVAILABLE_FRACTION = 0.6
TOTAL_FRACTION = 0.5

# Below this there is no point chunking — the bookkeeping costs more than the
# memory it saves.
MIN_BUDGET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class RunPlan:
    """How a stage should execute on this machine."""

    strategy: str  # "in-core" | "chunked" | "disk-backed"
    budget_bytes: int
    chunk_layers: int  # 0 when in-core
    downsample: int  # 1 unless allow_downsample was opted into
    scratch_dir: str | None  # set only for "disk-backed"
    reasons: tuple[str, ...]
    blocked: str | None  # set only for insufficient disk


@dataclass(frozen=True)
class Advice:
    """Recommended 3-D settings for this machine and volume."""

    downsample: int
    render_mode: str | None  # None = keep whatever the user chose
    reasons: tuple[str, ...]


def _human(nbytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(nbytes) < 1024 or unit == "TB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def headroom_bytes(profile) -> int:
    """How much memory a run may plan to use on *profile*."""
    if profile.ram_total <= 0:
        return MIN_BUDGET_BYTES  # unmeasurable machine: assume the worst
    return int(min(AVAILABLE_FRACTION * profile.ram_available, TOTAL_FRACTION * profile.ram_total))


def plan_run(profile, estimate, *, allow_downsample: bool = False, scratch_dir=None) -> RunPlan:
    """Decide the execution strategy for *estimate* on *profile*."""
    budget = headroom_bytes(profile)
    reasons: list[str] = []

    if estimate.peak_bytes <= budget:
        reasons.append(
            f"needs {_human(estimate.peak_bytes)}, {_human(budget)} available — running in memory"
        )
        return RunPlan("in-core", budget, 0, 1, None, tuple(reasons), None)

    reasons.append(
        f"needs {_human(estimate.peak_bytes)} but only {_human(budget)} is safely available"
    )

    downsample = 1
    if allow_downsample:
        # Each doubling of the factor quarters the in-plane element count.
        while downsample < 8 and estimate.peak_bytes / (downsample**2) > budget:
            downsample *= 2
        if downsample > 1:
            reasons.append(
                f"'allow downsample' is on — coarsening by {downsample}x, recorded in the output"
            )

    effective_peak = estimate.peak_bytes / (downsample**2)

    if estimate.chunkable:
        n_layers = estimate.shape[0] if estimate.shape else 1
        per_layer = max(1, int(effective_peak / max(1, n_layers)))
        chunk_layers = max(1, min(n_layers, int(max(budget, MIN_BUDGET_BYTES) / per_layer)))
        reasons.append(
            f"chunking into groups of {chunk_layers} of {n_layers} layers — slower, same result"
        )
        return RunPlan("chunked", budget, chunk_layers, downsample, None, tuple(reasons), None)

    needed = int(effective_peak)
    reasons.append("this step needs the whole array addressable — running disk-backed")
    blocked = None
    if profile.disk_free and needed > profile.disk_free:
        blocked = (
            f"needs {_human(needed)} of scratch disk but only {_human(profile.disk_free)} is free"
        )
    return RunPlan("disk-backed", budget, 0, downsample, scratch_dir, tuple(reasons), blocked)


def advise_3d(profile, shape, mode: str) -> Advice:
    """Recommended downsample and render mode for a volume of *shape*.

    Volume mode uploads the grid as ONE 3-D texture; exceeding
    ``GL_MAX_3D_TEXTURE_SIZE`` makes VTK render nothing at all — a silently
    blank product. Geometry modes (surface/isosurface) upload geometry instead,
    so they are unaffected and get no advice.
    """
    reasons: list[str] = []
    if mode != "volume" or profile.gl is None or not profile.gl.max_3d_texture:
        return Advice(1, None, ())
    if not shape:
        return Advice(1, None, ())
    limit = int(profile.gl.max_3d_texture)
    # The texture is sized in POINTS — one more than the voxel count per axis.
    longest = max(int(d) + 1 for d in shape)
    if longest <= limit:
        return Advice(1, None, ())

    downsample = 1
    while downsample < 16 and (longest // downsample) + 1 > limit:
        downsample *= 2
    reasons.append(
        f"volume is {longest - 1} px on its longest axis but this GL stack caps 3-D "
        f"textures at {limit} px — downsample {downsample}x, or volume mode renders blank"
    )
    if profile.gl.software:
        reasons.append(
            f"software renderer ({profile.gl.renderer}) — surface mode uploads geometry "
            "instead of one large texture and will be far faster here"
        )
    return Advice(downsample, "surface", tuple(reasons))
