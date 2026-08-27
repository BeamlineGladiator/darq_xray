"""Machine-aware policy (dfxm/common/advice.py). Pure functions, no IO."""

from __future__ import annotations

import dataclasses

from dfxm.common.advice import (
    AVAILABLE_FRACTION,
    MARGINAL_RSS_PER_TRACED_BYTE,
    MIN_STREAM_BUDGET_BYTES,
    TOTAL_FRACTION,
    advise_3d,
    headroom_bytes,
    human_bytes,
    plan_run,
    working_set_budget_bytes,
)
from dfxm.config.models import CostEstimate
from tests.machine_fixtures import laptop_hw_gl, tiny_ram, windows_no_vtk, workstation_sw_gl

GB = 1024**3


def _est(peak_gb, *, chunkable=True, shape=(100, 700, 2891)):
    return CostEstimate(
        peak_bytes=int(peak_gb * GB),
        input_bytes=int(peak_gb * GB / 3),
        shape=shape,
        chunkable=chunkable,
        note=None,
    )


def test_headroom_is_the_tighter_of_the_two_limits():
    """Each case is chosen so a *different* one of the two limits binds.

    Asserted against the constants rather than their literal values, so tuning
    the fractions (0.6/0.5 -> 0.75/0.65 on 2026-08-25, -> 0.85/0.80 on
    2026-08-26) does not turn this into a busywork edit — while still failing
    loudly if a change ever flips which limit binds, which is the property the
    test actually names. Substituting one fraction for the other in either line
    is the mutation: both fail.

    The TOTAL case builds its own profile rather than using a shared fixture.
    The 2026-08-26 raise moved the crossover to `available/total >= 0.941`, and
    no shared fixture is that idle — `workstation_sw_gl` sits at 460/502 = 0.916
    and now binds on AVAILABLE like everything else. Left on that fixture this
    test would still have passed, silently covering one branch twice.
    """
    # The crossover itself, stated rather than implied: TOTAL binds only above it.
    crossover = TOTAL_FRACTION / AVAILABLE_FRACTION
    idle = dataclasses.replace(workstation_sw_gl(), ram_available=int(0.98 * 502 * GB))
    assert idle.ram_available / idle.ram_total > crossover, "fixture is not idle enough"
    assert headroom_bytes(idle) == int(TOTAL_FRACTION * 502 * GB)

    # tiny_ram: almost nothing free, so the available side binds by a wide margin.
    assert tiny_ram().ram_available / tiny_ram().ram_total < crossover
    assert headroom_bytes(tiny_ram()) == int(AVAILABLE_FRACTION * 1 * GB)

    # And the shared workstation fixture, which the raise moved across the line —
    # asserted so a future change that moves it back is visible here.
    ws = workstation_sw_gl()
    assert ws.ram_available / ws.ram_total < crossover
    assert headroom_bytes(ws) == int(AVAILABLE_FRACTION * 460 * GB)


FLOOR = 250 * 1024 * 1024  # a VTK-importing stage's process image


def test_working_set_budget_solves_the_additive_rss_model():
    """RSS = floor + marginal x traced, solved for traced against the headroom.

    Both constants have to be doing work, so both are asserted non-degenerate: a
    marginal of 1.0 or a floor that never reaches the arithmetic would leave the
    equalities below true while converting nothing.
    """
    assert MARGINAL_RSS_PER_TRACED_BYTE > 1.0
    for profile in (workstation_sw_gl(), laptop_hw_gl()):
        rss = headroom_bytes(profile)
        assert rss > FLOOR, "fixture must be big enough that the floor is not the binding term"
        budget = working_set_budget_bytes(profile, rss_floor_bytes=FLOOR)
        assert budget == int((rss - FLOOR) / MARGINAL_RSS_PER_TRACED_BYTE)
        # Feeding the answer back through the model must land inside headroom —
        # the property the number exists to have.
        assert FLOOR + MARGINAL_RSS_PER_TRACED_BYTE * budget <= rss


def test_working_set_budget_is_per_stage_through_the_floor():
    """The floor is the per-stage term, so it must change the answer.

    A stage that imports VTK and one that does not differ by hundreds of MB of
    process image, and the whole reason `rss_floor_bytes` is a required argument
    rather than a module constant is that the difference has to reach the
    budget. A shared constant would make these two equal.
    """
    profile = workstation_sw_gl()
    light_floor = 32 * 1024 * 1024
    light = working_set_budget_bytes(profile, rss_floor_bytes=light_floor)
    heavy = working_set_budget_bytes(profile, rss_floor_bytes=FLOOR)
    assert light > heavy
    expected = (FLOOR - light_floor) / MARGINAL_RSS_PER_TRACED_BYTE
    assert abs((light - heavy) - expected) <= 1  # int() truncation only


def test_working_set_budget_survives_a_machine_smaller_than_the_process_image():
    """Headroom under the floor means the run cannot fit — it still has to run.

    The subtraction goes negative there, and the answer must be the finest
    blocking available: not a negative budget, not a refusal. `tiny_ram` is not
    small enough for this (0.6 GB of headroom, above a VTK stage's floor), which
    is why the fixture is built explicitly — and why the precondition is
    asserted rather than assumed.
    """
    import dataclasses

    cramped = dataclasses.replace(
        tiny_ram(), ram_total=512 * 1024 * 1024, ram_available=256 * 1024 * 1024
    )
    assert 0 < headroom_bytes(cramped) < FLOOR
    assert working_set_budget_bytes(cramped, rss_floor_bytes=FLOOR) == MIN_STREAM_BUDGET_BYTES


def test_working_set_budget_is_not_floored_back_up_to_the_min_budget():
    """MIN_BUDGET_BYTES ("do not bother chunking") must not leak in here.

    `tiny_ram` has 0.6 GB of headroom, well above MIN_BUDGET_BYTES, so a naive
    `max(MIN_BUDGET_BYTES, ...)` would be invisible there — hence a light floor,
    where the converted answer legitimately lands below MIN_BUDGET_BYTES and a
    stray `max` would show up.
    """
    import dataclasses

    from dfxm.common.advice import MIN_BUDGET_BYTES

    unmeasurable = dataclasses.replace(laptop_hw_gl(), ram_total=0, ram_available=0)
    assert headroom_bytes(unmeasurable) == MIN_BUDGET_BYTES
    budget = working_set_budget_bytes(unmeasurable, rss_floor_bytes=8 * 1024 * 1024)
    assert MIN_STREAM_BUDGET_BYTES < budget < MIN_BUDGET_BYTES


def test_small_job_on_a_big_machine_stays_in_core():
    """The fast path must not pay for the slow path's safety."""
    plan = plan_run(workstation_sw_gl(), _est(4))
    assert plan.strategy == "in-core"
    assert plan.chunk_layers == 0
    assert plan.downsample == 1
    assert plan.blocked is None


def test_big_job_on_a_small_machine_chunks():
    plan = plan_run(tiny_ram(), _est(20))
    assert plan.strategy == "chunked"
    assert plan.chunk_layers >= 1
    assert plan.blocked is None
    assert any("chunk" in r.lower() for r in plan.reasons)


def test_chunked_reason_names_the_unit_the_estimate_declares():
    """`shape[0]` layers is right for six stages and wrong for `matched`.

    `matched`'s `shape[0]` counts scan FOLDERS while what it chunks is one
    scan's detector rows, so an estimate may name its own span. Both halves
    asserted: the default must stay "layers", or the override would be the only
    thing tested and the six correct stages could regress unnoticed.
    """
    default = plan_run(tiny_ram(), _est(20))
    assert any("of 100 layers" in r for r in default.reasons), default.reasons

    named = plan_run(
        tiny_ram(),
        CostEstimate(
            peak_bytes=20 * GB,
            input_bytes=GB,
            shape=(37, 21, 2048, 2048),  # folders, frames, rows, columns
            chunkable=True,
            note=None,
            chunk_span=(2048, "detector rows"),
        ),
    )
    assert any("of 2048 detector rows" in r for r in named.reasons), named.reasons
    assert not any("layers" in r for r in named.reasons), named.reasons


def test_unchunkable_big_job_goes_disk_backed_not_blocked(tmp_path):
    """Nothing refuses for lack of RAM — the escalation ends at disk-backed."""
    plan = plan_run(tiny_ram(), _est(20, chunkable=False), scratch_dir=str(tmp_path))
    assert plan.strategy == "disk-backed"
    assert plan.scratch_dir == str(tmp_path)
    assert plan.blocked is None


def test_disk_backed_blocks_only_when_disk_is_short(tmp_path):
    """The one genuine blocker, and it is stated in advance."""
    profile = tiny_ram()
    huge = _est(100, chunkable=False)  # 100 GB needed, 40 GB free
    plan = plan_run(profile, huge, scratch_dir=str(tmp_path))
    assert plan.blocked is not None
    assert "disk" in plan.blocked.lower()


def test_downsample_stays_off_unless_opted_in():
    assert plan_run(tiny_ram(), _est(20)).downsample == 1
    opted = plan_run(tiny_ram(), _est(20), allow_downsample=True)
    assert opted.downsample >= 1


def test_every_plan_explains_itself():
    for profile in (laptop_hw_gl(), workstation_sw_gl(), windows_no_vtk(), tiny_ram()):
        plan = plan_run(profile, _est(20))
        assert plan.reasons, f"{profile.os_name} plan gave no reason"
        assert all(isinstance(r, str) and r for r in plan.reasons)


def test_advise_3d_downsamples_to_fit_the_texture_limit():
    """STO2 is 2891 px wide; llvmpipe caps 3-D textures at 2048."""
    advice = advise_3d(workstation_sw_gl(), (100, 700, 2891), "volume")
    assert advice.downsample >= 2
    assert advice.render_mode in ("surface", "isosurface")
    assert any("2048" in r for r in advice.reasons)


def test_advise_3d_says_what_the_run_will_do_rather_than_asking():
    """`volume_downsample` 0 is the shipped default: the run fits the volume
    itself, so the hint reports that instead of demanding an action."""
    advice = advise_3d(workstation_sw_gl(), (100, 700, 2891), "volume", 0)
    reason = " ".join(advice.reasons)
    assert "coarsens it 2x" in reason and "2048" in reason
    assert "BLANK" not in reason


def test_advise_3d_warns_when_the_user_opted_out_of_the_fit():
    advice = advise_3d(workstation_sw_gl(), (100, 700, 2891), "volume", 1)
    reason = " ".join(advice.reasons)
    assert "BLANK" in reason and "set it to 0" in reason


def test_advise_3d_is_silent_when_the_users_own_factor_already_fits():
    """2891 // 4 = 722: over the cap at 1, under it at 4, so there is nothing
    left to warn about."""
    assert advise_3d(workstation_sw_gl(), (100, 700, 2891), "volume", 4).reasons == ()


def test_advise_3d_does_not_promise_a_fit_coarsening_cannot_deliver():
    """Z is never block-averaged, so a volume too DEEP is not rescued by any
    factor — the hint must not offer one (it would be a lie the run then
    contradicts)."""
    reason = " ".join(advise_3d(workstation_sw_gl(), (3000, 8, 8), "volume", 0).reasons)
    assert "cannot fit it" in reason and "surface mode" in reason


def test_advise_3d_names_the_factor_the_actor_will_actually_use():
    """The hint's remedy is "set it to 0", so the factor it names must be the
    one auto-fit picks — the same 1/2/4/8/16 ladder, not `requested` doubled."""
    from dfxm.common import render3d as R3

    shape = (100, 700, 12000)
    actor = R3.fit_factor_for_shape(shape, 2048)
    assert actor == 8  # precondition: 12000 // 8 = 1500 fits, 4x does not
    reason = " ".join(advise_3d(workstation_sw_gl(), shape, "volume", 3).reasons)
    assert "coarsen 8x" in reason and "coarsen 6x" not in reason


def test_advise_3d_does_not_promise_a_fit_beyond_the_actors_cap():
    """Past 16x auto-fit declines and the render stays blank; the hint must not
    say "set it to 0 and fit"."""
    reason = " ".join(advise_3d(workstation_sw_gl(), (100, 700, 40960), "volume", 3).reasons)
    assert "and fit" not in reason
    assert "surface mode" in reason or "crop the ROI" in reason


def test_advise_3d_ignores_a_shape_whose_axes_it_cannot_identify():
    """`rocking.estimate` reports a four-element detector shape
    `(n_folders, n_frames, H, W)`. Guessing which axis is Z produces a hint
    about the detector, not the volume."""
    assert advise_3d(workstation_sw_gl(), (76, 575, 2048, 2048), "volume", 0).reasons == ()
    assert advise_3d(workstation_sw_gl(), (76, 0, 1832), "volume", 0).reasons == ()


def test_advise_3d_recommends_surface_mode_on_a_software_renderer():
    """The texture cap says why volume mode fails; THIS says what to do about
    it, and on a software renderer it is the more actionable half. It was
    deleted silently in 56c23c3 because nothing covered it."""
    reasons = advise_3d(workstation_sw_gl(), (100, 700, 2891), "volume", 0).reasons
    assert any("surface mode uploads geometry" in r for r in reasons)
    assert any("llvmpipe" in r for r in reasons)  # names the renderer it saw

    # Hardware GL with the same small cap gets the cap reason and NOT this one.
    import dataclasses

    prof = workstation_sw_gl()
    prof = dataclasses.replace(prof, gl=dataclasses.replace(prof.gl, software=False))
    reasons = advise_3d(prof, (100, 700, 2891), "volume", 0).reasons
    assert reasons and not any("surface mode uploads geometry" in r for r in reasons)


def test_advise_3d_leaves_a_fitting_volume_alone():
    advice = advise_3d(laptop_hw_gl(), (100, 700, 2891), "volume")
    assert advice.downsample == 1
    assert advice.render_mode is None


def test_advise_3d_is_silent_for_geometry_modes():
    """surface/isosurface upload geometry, not one big 3-D texture."""
    advice = advise_3d(workstation_sw_gl(), (100, 700, 2891), "surface")
    assert advice.downsample == 1


def test_advise_3d_without_gl_recommends_nothing():
    advice = advise_3d(windows_no_vtk(), (100, 700, 2891), "volume")
    assert advice.downsample == 1
    assert advice.render_mode is None


def test_advise_3d_survives_an_empty_shape():
    """An estimate with no resolvable shape must not crash `max()` on empty."""
    advice = advise_3d(workstation_sw_gl(), (), "volume")
    assert advice.downsample == 1
    assert advice.render_mode is None


# -----------------------------------------------------------------------------
# CostEstimate.scratch_bytes — the chunked path's disk check
# -----------------------------------------------------------------------------
def _chunked_estimate(peak_gb: float, scratch_gb: float = 0.0):
    """An estimate large enough that plan_run must chunk it on the given profile."""
    GB = 1024**3
    return CostEstimate(
        peak_bytes=int(peak_gb * GB),
        input_bytes=int(peak_gb * GB),
        shape=(100, 500, 500),
        chunkable=True,
        scratch_bytes=int(scratch_gb * GB),
    )


def test_a_chunked_run_that_spills_is_blocked_when_the_disk_is_too_small():
    """The median centring statistic caches the aligned volume to scratch.

    Before this, `plan_run`'s chunked return had NO disk check at all, so a
    machine short of disk discovered the problem halfway through a long run —
    the failure this phase exists to prevent.
    """
    GB = 1024**3
    profile = dataclasses.replace(laptop_hw_gl(), disk_free=2 * GB)

    plan = plan_run(profile, _chunked_estimate(peak_gb=40, scratch_gb=20))
    assert plan.strategy == "chunked"  # precondition: we are on the path under test
    assert plan.blocked, "a 20 GB spill onto 2 GB of free disk must be blocked"
    assert "scratch disk" in plan.blocked


def test_a_chunked_run_that_spills_is_allowed_when_the_disk_is_big_enough():
    GB = 1024**3
    profile = dataclasses.replace(laptop_hw_gl(), disk_free=200 * GB)

    plan = plan_run(profile, _chunked_estimate(peak_gb=40, scratch_gb=20), scratch_dir="/tmp/x")
    assert plan.strategy == "chunked"
    assert plan.blocked is None
    assert plan.scratch_dir == "/tmp/x"
    assert any("scratch disk" in r for r in plan.reasons)


def test_a_chunked_run_that_does_not_spill_is_never_disk_blocked():
    """scratch_bytes defaults to 0, so mean/midrange runs must be unaffected.

    Mutation guard: if the disk check ever stops consulting scratch_bytes and
    starts using peak_bytes, this fails — a 40 GB peak on 2 GB of disk would
    block a run that never touches the disk at all.
    """
    GB = 1024**3
    profile = dataclasses.replace(laptop_hw_gl(), disk_free=2 * GB)

    plan = plan_run(profile, _chunked_estimate(peak_gb=40, scratch_gb=0))
    assert plan.strategy == "chunked"
    assert plan.blocked is None
    assert plan.scratch_dir is None
    assert not any("scratch disk" in r for r in plan.reasons)


def test_the_chunking_reason_starts_with_the_pinned_prefix():
    """advisory.py filters on this prefix; a reworded message must fail here,
    not leak a chunk count into the GUI."""
    from dfxm.common import advice
    from dfxm.common.machine import MachineProfile

    prof = MachineProfile(
        "Linux", 4, 2, 8 * 1024**3, 1 * 1024**3, 40 * 1024**3, None, "unprobed", None, ()
    )
    est = CostEstimate(200 * 1024**3, 100 * 1024**3, (76, 1200, 1800), True)
    plan = advice.plan_run(prof, est)
    assert plan.strategy == "chunked"  # precondition for the reason to exist
    assert any(r.startswith(advice.CHUNK_REASON_PREFIX) for r in plan.reasons)


def test_human_bytes_labels_the_binary_units_it_actually_divides_by():
    """The divisor is 1024, so the label has to be the binary one.

    The units read "KB"/"MB"/"GB" while the arithmetic divided by 1024 — every
    figure the app displayed was understated by its own label, by 7% at GiB
    scale, enough that a machine sold as 539 GB read as "502.4 GB RAM" in the
    status bar. The numbers were right; only the unit was wrong. Relabelled
    2026-08-25. Restoring any SI spelling is the mutation.
    """
    assert human_bytes(512) == "512.0 B"
    assert human_bytes(1024) == "1.0 KiB"
    assert human_bytes(1024**2) == "1.0 MiB"
    assert human_bytes(1024**3) == "1.0 GiB"
    assert human_bytes(1024**4) == "1.0 TiB"
    # The arithmetic really is binary, which is what those labels now claim: an
    # SI gigabyte is not one GiB and must not print as one.
    assert human_bytes(1000**3) != "1.0 GiB"
    # Past the end of the table it saturates rather than inventing a unit.
    assert human_bytes(5 * 1024**5).endswith(" TiB")
