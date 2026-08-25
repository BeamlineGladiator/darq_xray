"""Composition of profile + estimate + plan into one Advisory (Qt-free)."""

from __future__ import annotations

import os

from dfxm.common import advice, advisory, alignment
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
    assert "expected to run in memory" in adv.headline
    assert "1.0 GB" in adv.headline


def test_the_in_core_detail_does_not_restate_the_headline():
    """An in-core run's only `plan.reason` says what the headline already says.

    `StageView._on_run` stacks the headline over the details, so the pre-flight
    banner printed the same sentence twice — caught on the first real STO2 run
    (`needs ~10.5 GB RAM, 251.2 GB safely available — expected to run in
    memory` / `needs 10.5 GB RAM, 251.2 GB available — running in memory`).
    Every fixture until then carried extra reasons that hid the repeat.
    """
    spec = _spec_with("tests.test_common_advisory:_cheap_estimate")
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.plan.strategy == "in-core"  # precondition
    assert adv.conservative is False  # precondition: no extra note to keep
    # precondition: the duplicate really is what `plan_run` emits upstream.
    assert any(r.endswith(advice.INCORE_REASON_SUFFIX) for r in adv.plan.reasons)
    assert adv.details == ()


def test_a_conservative_in_core_run_still_keeps_its_note():
    """Dropping the in-core reason must not take the conservative note with it.

    Same branch of `_details`; only the duplicated headroom sentence goes.
    """
    spec = _spec_with("tests.test_common_advisory:_conservative_estimate")
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.plan.strategy == "in-core"  # precondition
    assert adv.details == (advisory._CONSERVATIVE_NOTE,)


def test_the_headline_says_its_figures_are_ram():
    """The cost line's two byte counts are memory, and must say so.

    They can sit in the same banner as `plan.reasons` lines measured in
    scratch *disk*; an unlabelled second figure beside those reads as
    whichever resource the eye reached for last.
    """
    spec = _spec_with("tests.test_common_advisory:_cheap_estimate")
    plain = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert plain.conservative is False  # precondition: the normal lead
    assert "RAM" in plain.headline, plain.headline

    cons = advise_stage(
        _spec_with("tests.test_common_advisory:_conservative_estimate"),
        {},
        profile=workstation_sw_gl(),
    )
    # The conservative lead is a separate branch of `_headline` and needs its
    # own label — the plain assertion above cannot reach it.
    assert cons.conservative is True  # precondition
    assert "RAM" in cons.headline, cons.headline


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


def test_estimator_returning_a_non_cost_estimate_does_not_raise():
    """advise_stage's never-raises contract must hold even when the estimator
    returns something that is not a genuine CostEstimate. The earlier version
    wrapped only the `estimator(dict(params))` call in a try/except, so
    `estimate.peak_bytes` a few lines later raised `AttributeError` past the
    guard -- and `StageView._on_run`'s `compute_blocking()` call (the GUI
    thread, on the Run click) does not catch it."""
    spec = _spec_with("tests.test_common_advisory:_returns_none")
    adv = advise_stage(spec, {}, profile=workstation_sw_gl())
    assert adv.estimate is None and adv.plan is None
    assert "cannot estimate cost" in adv.headline


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


def _returns_none(params):
    return None


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
    prof = laptop_hw_gl()
    # Precondition: GL really was probed (not merely unprobed, which would also
    # produce no hint, but for the wrong reason) and its cap genuinely exceeds
    # the fixture volume's longest axis, or a no-hint assertion below would
    # pass vacuously even if the fixture were ever changed to gl=None.
    assert prof.gl is not None
    assert prof.gl.max_3d_texture > max(_wide_estimate({}).shape)
    adv = advise_stage(spec, {"render_mode": "volume"}, profile=prof)
    assert HINT_3D_TEXTURE not in adv.hints


def test_no_texture_hint_for_geometry_render_modes():
    """Surface/isosurface upload geometry, not one big texture."""
    spec = _spec_with("tests.test_common_advisory:_wide_estimate")
    prof = workstation_sw_gl()
    # Precondition: this fixture volume really would trigger the hint on this
    # profile in volume mode (see
    # test_an_oversized_volume_on_software_gl_gets_a_texture_hint) — otherwise
    # "surface" suppressing the hint below would be untested, and this
    # assertion would pass vacuously even if the mode check were deleted.
    assert prof.gl.max_3d_texture < max(_wide_estimate({}).shape)
    adv = advise_stage(spec, {"render_mode": "surface"}, profile=prof)
    assert HINT_3D_TEXTURE not in adv.hints


def test_no_texture_hint_when_gl_is_unprobed():
    spec = _spec_with("tests.test_common_advisory:_wide_estimate")
    prof = windows_no_vtk()
    # Precondition: this fixture's no-hint outcome really is because GL was
    # never probed successfully (gl_status != "ok"), not because gl happens to
    # be None for some other reason, or this test would pass vacuously.
    assert prof.gl_status != "ok"
    assert prof.gl is None
    adv = advise_stage(spec, {"render_mode": "volume"}, profile=prof)
    assert adv.hints == {}


def _wide_estimate(params):
    return CostEstimate(1 * GB, 1 * GB, (76, 1200, 2891), True)


_RAW_SHAPE = (3, 50, 100)  # comfortably under workstation_sw_gl's 2048 px cap


def _raw_shape_estimate(params):
    return CostEstimate(1 * GB, 1 * GB, _RAW_SHAPE, True)


def test_texture_hint_uses_the_aligned_shape_not_the_raw_on_disk_one(tmp_path):
    """I2: the pre-fix `_hints` compared the GL cap against `estimate.shape` —
    the raw on-disk shape a volume-producing stage's estimator reads straight
    out of HDF5 — but VTK actually uploads the ALIGNED volume
    (`apply_roi_3d -> apply_samy_shifts_to_volume -> interpolate_to_uniform_z`),
    which the samy X-pad widens well past the raw shape. Comparing the raw
    shape therefore stays silent in the dangerous direction: a volume that
    will actually render blank can still read as comfortably under the cap.

    Build a real raw_root with samy motors spanning enough millimetres, at
    this fixture's pixel scale, to push the ALIGNED X extent from comfortably
    under the cap to comfortably over it, while the RAW shape stays under it
    throughout.
    """
    import h5py

    raw_root = tmp_path / "raw"
    samy_mm = [0.0, 0.15, 0.30]  # -> offsets [0, 1000, 2000] px at 0.15 um/px
    samz_mm = [0.0, 0.01, 0.02]
    for i, (sy, sz) in enumerate(zip(samy_mm, samz_mm)):
        folder = raw_root / f"mosa__{i}"
        folder.mkdir(parents=True)
        with h5py.File(folder / f"mosa__{i}.h5", "w") as f:
            f["1.1/instrument/positioners/samy"] = sy
            f["1.1/instrument/positioners/samz"] = sz

    params = {
        "render_mode": "volume",
        "raw_root": str(raw_root),
        "mosa_pattern": "mosa__*",
        "pixel_size_x_um": 0.15,
        "samy_direction": 1,
        "samy_path": "1.1/instrument/positioners/samy",
        "samz_path": "1.1/instrument/positioners/samz",
    }
    prof = workstation_sw_gl()

    # Preconditions pinning the fixture in the region this test claims to
    # cover: the RAW shape must genuinely read as fitting the cap (else the
    # pre-fix bug would already have warned, and this test would not
    # distinguish the fix from the bug it replaces), and the widened ALIGNED
    # shape resolved by the same machinery `_hints` now uses must genuinely
    # exceed it.
    assert max(_RAW_SHAPE) + 1 <= prof.gl.max_3d_texture
    aligned = alignment.aligned_shape_for_params(params, _RAW_SHAPE, pattern_key="mosa_pattern")
    assert aligned is not None
    assert max(aligned) + 1 > prof.gl.max_3d_texture

    spec = _spec_with("tests.test_common_advisory:_raw_shape_estimate")
    adv = advise_stage(spec, params, profile=prof)
    assert HINT_3D_TEXTURE in adv.hints
    assert str(prof.gl.max_3d_texture) in adv.hints[HINT_3D_TEXTURE]


def test_advise_stage_never_raises_when_hint_computation_blows_up(monkeypatch):
    """advise_stage's never-raises contract must hold even when the hint
    computation itself blows up (a real estimate/plan must still come back)."""

    def _boom_advise_3d(*a, **k):
        raise RuntimeError("GL query exploded")

    monkeypatch.setattr(advice, "advise_3d", _boom_advise_3d)
    spec = _spec_with("tests.test_common_advisory:_wide_estimate")
    adv = advise_stage(spec, {"render_mode": "volume"}, profile=workstation_sw_gl())
    assert adv.estimate is not None and adv.plan is not None
    assert adv.hints == {}


def test_the_headline_calls_its_second_figure_a_budget():
    """The second figure is a self-imposed cap, and must not read as free RAM.

    `advice.headroom_bytes` is a share of total and of available RAM, so it is
    always *smaller* than what the machine reports free — and the status bar
    reports exactly that. Worded "safely available" the pair contradicted
    itself on screen: the cost line said 251.2 GB while the status bar said
    466.7 GB free, which is what Albert hit on the first real STO2 `visualize`
    run. Naming it a budget is what makes the smaller number self-explaining,
    so the word is behaviour, not decoration.

    Restoring "safely available" is the mutation; it fails the last assertion.
    """
    profile = workstation_sw_gl()
    adv = advise_stage(
        _spec_with("tests.test_common_advisory:_cheap_estimate"), {}, profile=profile
    )

    # precondition: the two figures really do disagree, which is the whole
    # reason the wording has to explain itself.
    assert adv.plan is not None
    assert adv.plan.budget_bytes < profile.ram_available

    assert "budget" in adv.headline, adv.headline
    assert "available" not in adv.headline, adv.headline
