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
#
# These are a *budget cap*, not a measurement, and that distinction is the whole
# reason the surfaces call the result a "budget" rather than what is "available"
# — on the 502 GiB development box the cap sat at 251 GiB while the status bar
# truthfully read 467 GiB free, and the two numbers side by side read as a
# contradiction rather than as a policy.
#
# What they trade: headroom only chooses in-core vs streaming and never blocks a
# run, so a cap set too high lets a run go in-core and OOM, while one set too low
# only makes it stream (slower, same result). That asymmetry is why these stay
# well below 1.0.
#
# Raised 0.6/0.5 -> 0.75/0.65 on 2026-08-25, then 0.75/0.65 -> 0.85/0.80 on
# 2026-08-26, both at Albert's request. The second raise waited on the estimator
# recalibration, and it is what unblocked it: an under-prediction judged against
# a looser cap is exactly the OOM direction, and until 2026-08-26 `matched`
# under-predicted the real dataset by 2.3x while `rocking` under-predicted a
# short-scan run by 13x. **Every stage now over-predicts every configuration it
# has been measured against**, by 1.10x at the tightest, so the cap can be
# spent. If a future estimator is rewritten from reasoning rather than
# measurement, that margin is what this raise consumed — revisit here.
#
# **Which limit binds changed with the second raise.** AVAILABLE binds whenever
# `ram_available / ram_total < TOTAL_FRACTION / AVAILABLE_FRACTION` = 0.941, so
# on any machine with less than ~94% of its RAM free — including the 502 GiB
# development box at 93% — it is now the available side that decides, where
# before it was the total side. TOTAL_FRACTION is not vestigial: it still guards
# the case it was added for, a page cache making `available` misleadingly large
# on a machine whose real working set is much bigger. Both branches are pinned
# by `test_headroom_is_the_tighter_of_the_two_limits`, which builds its own
# profile for the TOTAL case precisely because no shared fixture reaches 94%.
#
# One thing these fractions do NOT model: the GUI **parent** process is alive
# while the stage child runs, and its Qt/matplotlib footprint comes out of the
# same `ram_available`. Every estimator prices the child alone. That is part of
# why the available side keeps 15% rather than being pushed to 0.95 — on an 8 GB
# laptop 15% is 0.8 GiB, which is roughly one GUI parent.
AVAILABLE_FRACTION = 0.85
TOTAL_FRACTION = 0.80

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
#
#   **First independent test, `visualize` (2026-08-21): not contradicted, but
#   AT the trigger rather than comfortably below it.** Four sweeps of twelve
#   runs each (three shapes x budgets 4/16/64/512 MB) fitted slopes of **1.249,
#   1.270, 1.295 and 1.216** — every one under 1.3, none by much, with local
#   slopes between consecutive points spanning 1.10 to 1.59. So the constant
#   survives on evidence and should be read as "measured 1.2-1.3", not as a
#   margin anyone may spend. The fitted intercepts landed on that stage's
#   independently measured bare process image (107.6 MB), which is what an
#   additive model should do and a ratio cannot.
#
#   **Second independent test, `slices` (2026-08-21): 1.053, comfortably under.**
#   Thirteen runs, three shapes x budgets 2/8/32/128/512 MB and the machine's own,
#   giving 12 distinct traced levels spanning **36x** (5.9 to 213.2 MB):
#
#       RSS = 106.9 MB + 1.053 x traced        (least squares, r^2 = 0.995)
#
#   with local slopes between consecutive levels running -0.97 to 2.30 — the
#   wild ones all at the low end, where the process image dominates and the
#   points come from different shapes; over the well-conditioned upper half
#   (62 MB traced and above) they run 0.76 to 1.21. The fitted intercept lands on
#   that stage's independently measured PNGs-off process image (103.9 MB), the
#   same corroboration `visualize` produced. Ratios over the same runs ranged
#   1.52 to 18.53, which is again the point about a ratio not being a constant.
#   So: three stages measured, none above the trigger, and the constant stays
#   shared.
#
#   **How to measure this properly — the point estimate is not enough.** A
#   sweep must give at least FIVE distinct traced levels spanning at least 5x,
#   and the slope must be checked LOCALLY (between consecutive levels) as well
#   as by least squares. Two ways the naive version misleads, both met here:
#
#   * Vary only the shape and the traced peak barely moves. A first `visualize`
#     sweep did that with the 3-D render on: traced went 82.8 -> 86.5 MB while
#     RSS sat near 586 MB, and the "slope" was 1.53 at r^2 = 0.64 — fitting
#     noise, and it would have tripped this trigger for nothing. Vary the
#     *budget* on a streaming path; that is what moves traced bytes.
#   * Vary the budget carelessly and the points pile into two clusters (runs
#     pinned at the alignment's one-layer floor, plus a few in-core ones). A
#     two-cluster fit reports a high r^2 while conditioning the slope on almost
#     nothing — the 0.995 in the first `visualize` sweep was largely that.
#     Spread the levels; report the local slopes.
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


def human_bytes(nbytes: float) -> str:
    """Format a byte count in **binary** units, labelled as such.

    The divisor was always 1024 while the labels read "KB"/"GB", which name the
    SI powers of 1000 — so every figure the app showed was understated by its
    own label: 7% at GiB scale, and enough that the status bar's "502.4 GB RAM"
    did not match the 539 GB a machine with 502.4 **GiB** is sold as. The
    numbers were right; only the unit was. Changed on 2026-08-25 at Albert's
    request, and this is the **one** formatter for byte counts in the app — the
    cost line, the pre-flight banner, every `plan_run` reason, and the status
    bar (`gui/main_window.py::_refresh_machine_status`) all route through it, so
    the units cannot drift apart between surfaces.

    Anything comparing against this output should call it rather than hardcode a
    unit, the way `tests/test_gui_status_bar.py` does.
    """
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(nbytes) < 1024 or unit == "TiB":
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TiB"


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

    **The recipe, so this does not have to be reinvented per stage:**
    :func:`tests.peak_rss.measure_process_floor` takes the measurement and
    :func:`tests.peak_rss.assert_floor_covers` pins it, in one test the stage
    copies verbatim (the template is the section comment above that function).
    Copy the *call*, never another stage's number — the assertion fails in both
    directions, so a floor pasted from a heavier stage is caught as surely as one
    set too low. And when measuring, note what
    :data:`MARGINAL_RSS_PER_TRACED_BYTE` says above it: that constant is shared
    across stages only until a stage measures its own marginal above 1.3, at
    which point it becomes a per-stage argument like this one. Measure and report
    it either way — over at least five traced levels spanning at least 5x, with
    the local slopes reported alongside the fit, for the reasons recorded there.

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


# The one `plan_run` reason that carries `chunk_layers`. `advisory.py` replaces
# any reason starting with this prefix, because the group count is display-only
# and is NOT the blocking a stage picks — visualize/slices/paraview each derive
# their own from `working_set_budget_bytes` with a per-stage RSS floor. Pinned
# as a constant so a reworded message fails a test instead of silently escaping
# the filter and reaching the user.
CHUNK_REASON_PREFIX = "chunking into groups of"

# The in-core reason restates the advisory headline word for word in substance
# — same peak, same budget, same strategy — so `advisory._details` drops it and
# the pre-flight banner prints the sentence once. Keyed on this constant rather
# than on `strategy == "in-core"` so that a *second*, genuinely new in-core
# reason would still reach the user, and a reword here fails
# `test_the_in_core_detail_does_not_restate_the_headline` rather than silently
# reintroducing the duplicate.
INCORE_REASON_SUFFIX = " — running in memory"


def plan_run(profile, estimate, *, allow_downsample: bool = False, scratch_dir=None) -> RunPlan:
    """Decide the execution strategy for *estimate* on *profile*."""
    budget = headroom_bytes(profile)
    reasons: list[str] = []

    if estimate.peak_bytes <= budget:
        reasons.append(
            f"needs {human_bytes(estimate.peak_bytes)} RAM of a {human_bytes(budget)} budget"
            f"{INCORE_REASON_SUFFIX}"
        )
        return RunPlan("in-core", budget, 0, 1, None, tuple(reasons), None)

    reasons.append(
        f"needs {human_bytes(estimate.peak_bytes)} RAM but the budget is only {human_bytes(budget)}"
    )

    downsample = 1
    if allow_downsample:
        # Each doubling of the factor quarters the in-plane element count.
        #
        # **Before wiring `allow_downsample` up to anything**: since 2026-08-26
        # four estimators (strain, mosaicity, rocking, matched) report peaks
        # that INCLUDE their child's process image — 96-176 MiB, plus 416 for
        # rocking's top view. A process image does not shrink when you coarsen
        # the data, so dividing the whole figure here under-states a downsampled
        # run's real cost for those stages. The chunked path below divides the
        # same way and errs the safe direction (it over-states per-layer cost and
        # so chunks harder); this one errs the other way. No caller passes
        # `allow_downsample=True` today, which is the only reason it is a note
        # rather than a fix — the fix is to subtract the floor before dividing,
        # which needs the floor to be a field on `CostEstimate`.
        while downsample < 8 and estimate.peak_bytes / (downsample**2) > budget:
            downsample *= 2
        if downsample > 1:
            reasons.append(
                f"'allow downsample' is on — coarsening by {downsample}x, recorded in the output"
            )

    effective_peak = estimate.peak_bytes / (downsample**2)

    if estimate.chunkable:
        # `shape[0]` layers is right for a stage that chunks its volume along Z,
        # and wrong for one that chunks something else (`matched` divides one
        # scan's detector rows, while its `shape[0]` counts scan folders), so an
        # estimate may name its own span and unit.
        if estimate.chunk_span:
            n_layers, unit = int(estimate.chunk_span[0]), str(estimate.chunk_span[1])
            n_layers = max(1, n_layers)
        else:
            n_layers, unit = (estimate.shape[0] if estimate.shape else 1), "layers"
        per_layer = max(1, int(effective_peak / max(1, n_layers)))
        chunk_layers = max(1, min(n_layers, int(max(budget, MIN_BUDGET_BYTES) / per_layer)))
        reasons.append(
            f"{CHUNK_REASON_PREFIX} {chunk_layers} of {n_layers} {unit} — slower, same result"
        )
        # A chunked run that spills to scratch (the median centring statistic is
        # the one irreducibly whole-array step) must be checked against free disk
        # here, not discovered mid-run. `scratch_dir` is where it would land.
        scratch_needed = int(getattr(estimate, "scratch_bytes", 0) or 0)
        chunk_blocked = None
        if scratch_needed and profile.disk_free and scratch_needed > profile.disk_free:
            chunk_blocked = (
                f"needs {human_bytes(scratch_needed)} of scratch disk but only "
                f"{human_bytes(profile.disk_free)} is free"
            )
        elif scratch_needed:
            reasons.append(
                f"caching aligned blocks to {human_bytes(scratch_needed)} of scratch disk"
            )
        return RunPlan(
            "chunked",
            budget,
            chunk_layers,
            downsample,
            scratch_dir if scratch_needed else None,
            tuple(reasons),
            chunk_blocked,
        )

    needed = int(effective_peak)
    reasons.append("this step needs the whole array addressable — running disk-backed")
    blocked = None
    if profile.disk_free and needed > profile.disk_free:
        blocked = (
            f"needs {human_bytes(needed)} of scratch disk but only "
            f"{human_bytes(profile.disk_free)} is free"
        )
    return RunPlan("disk-backed", budget, 0, downsample, scratch_dir, tuple(reasons), blocked)


def advise_3d(profile, shape, mode: str, requested: int = 0) -> Advice:
    """Recommended downsample and render mode for a volume of *shape*.

    Volume mode uploads the grid as ONE 3-D texture; exceeding
    ``GL_MAX_3D_TEXTURE_SIZE`` makes VTK render nothing at all — a silently
    blank product. Geometry modes (surface/isosurface) upload geometry instead,
    so they are unaffected and get no advice.

    *requested* is the stage's ``volume_downsample``: ``0`` (the default) means
    the run fits the volume itself, so the reason says what will happen rather
    than asking for an action; ``>= 1`` is the user's own factor and produces a
    reason only when that factor leaves the volume over the limit.

    The factor named here comes from :func:`dfxm.common.render3d.fit_factor_for_shape`
    — the same function the run uses — because the remedy is "set it to 0", and
    a hint that names a factor auto-fit would not pick states the wrong
    resolution loss, or promises a fit past the 16x cap that never arrives.

    *shape* must be ``(Z, Y, X)``. Anything else returns no advice rather than
    guessing which axis is Z: ``rocking.estimate`` reports a four-element
    detector shape ``(n_folders, n_frames, H, W)``, and reading that as a volume
    produced a hint about the detector.
    """
    reasons: list[str] = []
    if mode != "volume" or profile.gl is None or not profile.gl.max_3d_texture:
        return Advice(1, None, ())
    dims = tuple(int(d) for d in shape) if shape else ()
    if len(dims) != 3 or min(dims) < 1:
        return Advice(1, None, ())

    from . import render3d  # local: keeps the figure stack off a bare estimate's import path

    limit = int(profile.gl.max_3d_texture)
    requested = max(0, int(requested or 0))
    if render3d.points_at_factor(dims, max(1, requested)) <= limit:
        return Advice(1, None, ())  # fits as configured — nothing to say

    longest = max(dims)
    downsample = render3d.fit_factor_for_shape(dims, limit)
    fits = downsample > 1  # 1 means no factor helps: Z is never block-averaged
    cap = f"volume is {longest} px on its longest axis but this GL stack caps 3-D textures at {limit} px"
    if requested >= 1:
        remedy = (
            f"set it to 0 to coarsen {downsample}x and fit"
            if fits
            else "use surface mode or crop the ROI"
        )
        reasons.append(f"{cap} — at 3-D downsample {requested} volume mode renders BLANK; {remedy}")
    elif fits:
        reasons.append(
            f"{cap} — the run coarsens it {downsample}x to fit; set 3-D downsample to 1 "
            f"for full resolution, which renders blank here"
        )
    else:
        reasons.append(
            f"{cap}, and {render3d.unfittable_reason(dims, limit)} — volume mode renders "
            f"BLANK; use surface mode or crop the ROI"
        )
    if profile.gl.software:
        # Restored after `56c23c3` dropped it while rewriting this function. On a
        # software renderer this is the more actionable half of the advice — the
        # texture cap is only the reason volume mode fails, this is the way out —
        # and `test_advise_3d_recommends_surface_mode_on_a_software_renderer`
        # now exists so it cannot be deleted silently a second time.
        reasons.append(
            f"software renderer ({profile.gl.renderer}) — surface mode uploads geometry "
            "instead of one large texture and will be far faster here"
        )
    return Advice(downsample, "surface", tuple(reasons))
