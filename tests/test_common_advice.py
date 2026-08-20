"""Machine-aware policy (dfxm/common/advice.py). Pure functions, no IO."""

from __future__ import annotations

from dfxm.common.advice import advise_3d, headroom_bytes, plan_run
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
