"""Composition of profile + estimate + plan into one Advisory (Qt-free)."""

from __future__ import annotations

import os

from dfxm.common import advice
from dfxm.common.advisory import HINT_3D_TEXTURE, advise_stage, disk_probe_dir
from dfxm.config.models import CostEstimate, Param, ParamType, StageSpec
from tests.machine_fixtures import laptop_hw_gl, tiny_ram, windows_no_vtk, workstation_sw_gl

GB = 1024**3

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


def _spec_with(estimator_target: str) -> StageSpec:
    return StageSpec(
        name="demo",
        label="Demo",
        description="",
        params=_SPEC.params,
        estimate=estimator_target,
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
    assert adv.plan.strategy == "chunked"  # precondition
    assert adv.plan.chunk_layers > 0  # precondition: there IS a count
    assert "expected to stream" in adv.headline
    rendered = " ".join((adv.headline, *adv.details))
    assert advice.CHUNK_REASON_PREFIX not in rendered
    # Not "no digit anywhere" — the headroom figure itself may contain digits
    # that happen to include the chunk count (e.g. "614.4 MB" contains "1").
    # The actual requirement is that the chunk-count SENTENCE is gone,
    # replaced by the generic streaming-groups sentence.
    assert any("blocking the work into groups" in d for d in adv.details)


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
    return CostEstimate(200 * GB, 100 * GB, (76, 1200, 1800), True, scratch_bytes=100 * GB)


def _boom(params):
    raise FileNotFoundError("no such file")


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
