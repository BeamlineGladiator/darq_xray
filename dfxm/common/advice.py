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

# -- RSS headroom -> working-set (tracemalloc) budget -------------------------
# `headroom_bytes` is RSS. `alignment.align_volume_streamed`'s `budget_bytes` is
# `tracemalloc` — Python-level allocations, which carry neither the interpreter,
# the extension modules, h5py's chunk cache, a memmap's resident pages nor
# allocator fragmentation. Handing one straight to the other is a category
# error, and this is where the two are related.
#
# **The relationship is additive, not a ratio.** Measured for the `paraview`
# stage over 18 runs (two shapes x budgets 16/64/512 MB x 1/8/16 Z pieces), peak
# RSS against peak traced bytes:
#
#     RSS = 230 MB + 1.18 x traced        (least squares, r^2 = 0.98)
#
# while the *ratio* RSS/traced over those same runs ranged from **2.06 to
# 23.43** — it is not a constant and no single multiplier can stand in for it.
# The intercept is the process image (interpreter, numpy, h5py, VTK) that
# `tracemalloc` never sees; it dominates every small run, which is why the ratio
# explodes as the traced peak shrinks. The slope is the part that scales:
# allocations made inside extension libraries — chiefly VTK's deep copy of each
# piece — which are resident but untraced.
#
# The constants below sit **above** the least-squares line, so the model
# normally over-states RSS. It is not an envelope and does not promise to be
# one: at a *fixed* traced peak (276.6-276.9 MB) measured RSS ranged over 45 MB
# (566-613 MB), so there is untraced, blocking-dependent variance wider than any
# margin worth claiming, and a sweep large enough will find a case the model
# under-states. Both terms are therefore set with deliberate slack rather than
# fitted tight — over-stating costs a slightly smaller budget, under-stating
# costs an OOM — and the *floor* is where that slack is put, because it is the
# term a stage can measure and pin (see `tests/peak_rss.py::assert_floor_covers`
# and the test that calls it).
#
# They are empirical, and specific to this machine and this class of stage:
#
# * `MARGINAL_RSS_PER_TRACED_BYTE` is the slope. It is shared across stages on
#   the argument that it is a property of how numpy and VTK allocate rather than
#   of what is imported, and it errs safe: paraview's per-piece VTK deep copy is
#   probably the heaviest slope in the pipeline, so a stage measuring its own
#   should come in *below* this. **Escalation trigger for whoever measures the
#   next stage: if a stage measures a marginal above 1.3, this constant stops
#   being shared and becomes a per-stage argument like the floor.** Shared until
#   contradicted, not shared on principle.
# * The **intercept does not travel**, which is why it is a required argument
#   rather than a constant here: it is set by which extension modules a stage
#   imports, and differs per stage by hundreds of MB (a VTK-importing stage
#   against one that only needs matplotlib). The project's estimator gate
#   reached the same verdict for its own per-stage error and said so plainly:
#   not one universal factor.
#
# Explicitly NOT related to the ~1.66x by which `estimate().peak_bytes`
# under-predicted real RSS on the real dataset. That is *estimator model error*
# — a wrong guess at how much a stage allocates — and this is a *currency
# conversion*. They are different quantities that happen to be dimensionless;
# see `tests/peak_rss.py`'s module docstring, which keeps them apart.
MARGINAL_RSS_PER_TRACED_BYTE = 1.3

# A budget below this buys nothing. The alignment chain floors its blocking at
# one output layer, so once the budget is under one layer's working set, making
# it smaller cannot make the blocks smaller — it only turns every call into a
# "budget too small" warning. Machines whose headroom does not cover the process
# image land here; the run proceeds at its finest blocking, which is the best
# answer available and the project's rule (nothing refuses to run for lack of
# RAM).
MIN_STREAM_BUDGET_BYTES = 4 * 1024 * 1024


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


def working_set_budget_bytes(profile, *, rss_floor_bytes: int) -> int:
    """:func:`headroom_bytes` converted into working-set (``tracemalloc``) currency.

    What a stage should hand to :func:`~dfxm.common.alignment.align_volume_streamed`
    as ``budget_bytes``. Never pass ``headroom_bytes`` there directly: it is an
    RSS figure and ``budget_bytes`` is priced in Python allocations.

    Solves ``rss_floor_bytes + MARGINAL_RSS_PER_TRACED_BYTE * traced <=
    headroom`` for ``traced`` — see the note above the constants for why the
    relationship is additive and why a single ratio cannot express it.

    *rss_floor_bytes* is **required and per stage**: the resident cost of the
    stage's own process image before it touches data, which `tracemalloc` cannot
    see and which is set by what the stage imports. A stage measures its own (a
    child running it on trivial input) and names it. There is no default,
    because a wrong floor here is silent and a missing one is not.

    Returns at least :data:`MIN_STREAM_BUDGET_BYTES`, including when the floor
    alone exceeds the headroom — that machine cannot host the run inside its
    headroom at all, and the honest answer is the finest blocking available
    rather than a refusal or a warning storm.

    Deliberately not floored at :data:`MIN_BUDGET_BYTES` — that floor says "do
    not bother chunking below this", and applying it here would hand back an
    unconverted number on exactly the small machine the conversion protects.
    """
    affordable = (headroom_bytes(profile) - int(rss_floor_bytes)) / MARGINAL_RSS_PER_TRACED_BYTE
    return max(MIN_STREAM_BUDGET_BYTES, int(affordable))


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
