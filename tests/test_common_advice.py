"""Machine-aware policy (dfxm/common/advice.py). Pure functions, no IO."""

from __future__ import annotations

from dfxm.common.advice import (
    MARGINAL_RSS_PER_TRACED_BYTE,
    MIN_STREAM_BUDGET_BYTES,
    advise_3d,
    headroom_bytes,
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
    # workstation: 0.6*460 = 276 GB vs 0.5*502 = 251 GB -> total-based wins
    assert headroom_bytes(workstation_sw_gl()) == int(0.5 * 502 * GB)
    # tiny_ram: 0.6*1 = 0.6 GB vs 0.5*8 = 4 GB -> available-based wins
    assert headroom_bytes(tiny_ram()) == int(0.6 * 1 * GB)


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
