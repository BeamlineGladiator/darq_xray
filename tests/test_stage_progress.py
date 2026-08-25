"""Every stage must report progress smoothly enough to extrapolate.

One test per stage, each reusing that stage's own fixture helpers so this
module owns no synthetic data of its own and cannot drift away from what the
stage tests already exercise. The invariants and the reasoning behind
`MAX_FRAC_GAP` live in `tests/progress_trace.py`.

The stages are covered here rather than in their own modules deliberately: the
point is that *all nine* meet one shared bar, and a per-module version of this
check is how five of them ended up with per-item loops while three reported four
fractions for an entire run.
"""

from __future__ import annotations

import pytest

from tests.progress_trace import assert_progress_wellformed, trace


def test_visualize_progress_is_smooth(tmp_path):
    import tests.test_stage_visualize as T
    from dfxm.stages import visualize

    proc, raw = T._setup(tmp_path)
    params = T._stream_params(proc, raw, tmp_path / "viz")
    assert_progress_wellformed(trace(visualize.run, params), label="visualize")


def test_visualize_progress_is_smooth_with_a_product_switched_off(tmp_path):
    """Turning a product off must not jump the bar, nor narrate work not done.

    The per-product fractions were fixed constants reported unconditionally, so
    `save_layers=False` still reported 0.6 "layers done" with no layers written:
    a 0.56 step in one go, and a step line asserting work that never happened.
    Both configurations below were silent stretches before the shares were
    allocated over the enabled products only.
    """
    import tests.test_stage_visualize as T
    from dfxm.stages import visualize

    proc, raw = T._setup(tmp_path)
    base = T._stream_params(proc, raw, tmp_path / "viz_no_layers")

    seen = trace(visualize.run, {**base, "save_layers": False})
    assert_progress_wellformed(seen, label="visualize without layer PNGs")
    assert "layers done" not in " ".join(t for _, t in seen)

    flat = {
        **base,
        "output_dir": str(tmp_path / "viz_layers_only"),
        "save_animation": False,
        "save_topview": False,
        "save_rotation": False,
    }
    seen = trace(visualize.run, flat)
    assert_progress_wellformed(seen, label="visualize with layer PNGs only")
    text = " ".join(t for _, t in seen)
    assert "animation done" not in text, text
    # `scene` is None on this path; it used to say otherwise.
    assert "3-D scene ready" not in text, text


def test_paraview_progress_is_smooth(tmp_path):
    import tests.test_stage_paraview as T
    from dfxm.stages import paraview

    proc, raw = T._setup(tmp_path)
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "output_dir": str(tmp_path / "pv"),
        "num_pieces_z": 2,
    }
    assert_progress_wellformed(trace(paraview.run, params), label="paraview")


def test_paraview_progress_is_smooth_with_a_single_export(tmp_path):
    """One export must fill the working range, not sit at the top of it.

    `Export mosaicity` is a default-on checkbox in the GUI, so unchecking it is
    an ordinary user action. The range used to be split at
    `EXPORT_LO + span * 1`, which for one export *is* `EXPORT_HI`: strain got
    the empty slot [0.97, 0.97] and reported 0.97 for every piece of the whole
    export — worse than the milestone reporting the sweep replaced.

    The fixture is deeper than the other paraview ones (12 layers, the stage's
    default `num_pieces_z`) because piece count is what sets the bar's
    resolution once the slots are right, and a 4-layer volume yields three
    pieces however the arithmetic goes. That is a property of the fixture, not
    of the stage, so the assertion this test really exists for is the one below
    — where the export's range starts and ends — which holds at any piece count.
    """
    import tests.test_stage_paraview as T
    from dfxm.stages import paraview

    proc, raw = T._setup(tmp_path, layers=12)
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "output_dir": str(tmp_path / "pv_strain_only"),
        "export_mosaicity": False,
    }
    seen = trace(paraview.run, params)
    assert_progress_wellformed(seen, label="paraview strain-only")
    _assert_export_spans_the_working_range(seen, "strain_volume")


def test_paraview_progress_skips_the_slot_of_a_missing_volume(tmp_path):
    """An enabled volume whose file is absent must not reserve half the bar.

    Running against a dataset that only has a strain volume, with
    `mosa_volume_file` left at its pre-filled default, is a normal
    configuration; the export is skipped but its *flag* used to buy it a slot,
    so the run's very first report was 0.495.
    """
    import tests.test_stage_paraview as T
    from dfxm.stages import paraview

    proc, raw = T._setup(tmp_path, layers=12)
    missing = proc / "no_such_mosa_volumes.h5"
    params = {
        "mosa_volume_file": str(missing),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "output_dir": str(tmp_path / "pv_missing_mosa"),
    }
    result = paraview.run(params)
    assert any("not found" in note for note in result.skipped), result.skipped

    seen = trace(paraview.run, params)
    assert_progress_wellformed(seen, label="paraview missing mosaicity")
    _assert_export_spans_the_working_range(seen, "strain_volume")


def _assert_export_spans_the_working_range(seen, label):
    """The one export's reports must run from EXPORT_LO to EXPORT_HI.

    Independent of `num_pieces_z`, which is what makes it the real pin: the
    defect was a slot of zero width, and a slot of zero width is invisible to a
    gap ceiling — every report lands on the same number, so no two of them are
    far apart.
    """
    mine = [f for f, t in seen if t.startswith(label)]
    assert mine, f"no {label} reports in {seen}"
    assert min(mine) > 0.0
    # 0.97 is paraview's EXPORT_HI; the export's last report must land on it.
    assert max(mine) == pytest.approx(0.97), f"{label} never reaches EXPORT_HI: {mine}"
    assert len(set(mine)) > 1, f"{label} reported one fraction for the whole export: {mine}"


def test_rocking_progress_is_smooth(tmp_path):
    import tests.test_stage_rocking as T
    from dfxm.stages import rocking

    raw = T._setup(tmp_path)
    raw = raw[0] if isinstance(raw, tuple) else raw
    params = {
        "raw_root": str(raw),
        "rocking_pattern": "rock__*",
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "pixel_size_x_um": 0.152,
        "pixel_size_y_um": 0.385,
        "output_dir": str(tmp_path / "rk"),
        "save_layers": True,
        "save_animation": False,
        "save_topview": False,
    }
    assert_progress_wellformed(trace(rocking.run, params), label="rocking")


def test_slices_progress_is_smooth(tmp_path):
    import tests.test_stage_slices as T
    from dfxm.stages import slices

    proc, raw = T._setup(tmp_path)
    params = T._minimal_params(proc, raw, tmp_path / "sl")
    assert_progress_wellformed(trace(slices.run, params), label="slices")


def test_strain_progress_is_smooth(tmp_path):
    import tests.test_stage_strain as T
    from dfxm.stages import strain

    ccmth = T._synthetic_ccmth()
    root = tmp_path / "root"
    for name in ("layer__1", "layer__2", "layer__3"):
        T._write_maps(str(root / name), ccmth)
    params = {
        "mode": "batch",
        "root_folder": str(root),
        "folder_pattern": "layer__*",
        "ccmth_ref_deg": 7.144,
        "output_dir": str(tmp_path / "out"),
    }
    assert_progress_wellformed(trace(strain.run, params), label="strain")


def test_mosaicity_progress_is_smooth(tmp_path):
    import tests.test_stage_mosaicity as T
    from dfxm.stages import mosaicity

    root = T._make_root(tmp_path)
    params = {"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"}
    assert_progress_wellformed(trace(mosaicity.run, params), label="mosaicity")


def test_matched_progress_is_smooth(tmp_path):
    import numpy as np

    import tests.test_stage_matched as T
    from dfxm.stages import matched

    raw = tmp_path / "raw"
    samy = [0.0, 0.0005, 0.001]
    samz = [0.0, 0.001, 0.002]
    for i in range(3):
        T._write_strain(str(raw), f"strain__{i + 1}", samy[i], samz[i])
        frames = np.random.default_rng(i).standard_normal((T.NF, T.H, T.W)) + 10.0
        T._write_rocking(str(raw), f"rock__{i + 1}", samy[i], samz[i], frames)
    params = {
        "raw_root": str(raw),
        "strain_pattern": "strain__*",
        "rocking_pattern": "rock__*",
        "frame_index": 0,
        "match_threshold_mm": 0.001,
        "output_dir": str(tmp_path / "matched_out"),
    }
    assert_progress_wellformed(trace(matched.run, params), label="matched")


def test_profiles_progress_is_smooth(tmp_path):
    import tests.test_stage_profiles as T
    from dfxm.stages import profiles

    h5 = tmp_path / "consolidated.h5"
    T._write_consolidated(str(h5))
    params = T._base_params(h5, tmp_path / "prof")
    assert_progress_wellformed(trace(profiles.run, params), label="profiles")


def test_slices_reports_a_plane_only_once_it_is_on_disk(tmp_path):
    """The fraction reports work *done*, so the PNG must exist when it fires.

    The per-plane report sat at the top of the loop body, ahead of both
    `save_slice_png` and `writer.append`, so on the last plane it reported the
    slice's whole share before that plane was written at all. `render.py`
    documents the opposite rule for the identical situation in
    `save_layer_pngs`, and `tests/test_common_progress.py` pins it there.
    Cancelling mid-plane left the bar one plane ahead of the disk.
    """
    import tests.test_stage_slices as T
    from dfxm.stages import slices

    proc, raw = T._setup(tmp_path)
    out = tmp_path / "sl"
    params = {
        **T._minimal_params(proc, raw, out),
        "save_png": True,
        "slices_json": (
            '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
            '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,'
            '"sweep_step_um":0.5,"sweep_start_um":0.0,"sweep_stop_um":1.5}]'
        ),
    }

    seen: list[tuple[str, int]] = []

    def record(frac, text=""):
        seen.append((str(text), len(list(out.rglob("*.png")))))

    slices.run(params, record)

    plane_reports = [(t, n) for t, n in seen if ": plane " in t]
    assert plane_reports, [t for t, _ in seen]
    for text, n_png in plane_reports:
        done = int(text.rsplit("plane ", 1)[1].split("/")[0])
        assert n_png >= done, f"{text!r} fired with only {n_png} PNG(s) written"


def test_strain_speaks_before_it_has_finished_reading(tmp_path):
    """A one-layer strain run must not do half a layer before its first report.

    `single` mode is one layer by definition, and `process_maps_file` reported
    nothing until after `_detrend_ccmth` — an HDF5 read plus a full-map surface
    fit — so the trace opened at 0.4275 and the three-layer fixture passed the
    gap ceiling only because 0.45/3 happens to be 0.1425.

    The gap ceiling is not asserted for one layer: the read, the fit and each of
    the three `savefig`s are single indivisible calls, so six atomic steps share
    one bar and the best possible worst-gap is about 1/6. What is pinnable is
    that the run speaks early and often, which is what was actually wrong.
    """
    import tests.test_stage_strain as T
    from dfxm.stages import strain

    ccmth = T._synthetic_ccmth()
    root = tmp_path / "root"
    T._write_maps(str(root / "layer__1"), ccmth)
    params = {
        "mode": "batch",
        "root_folder": str(root),
        "folder_pattern": "layer__*",
        "ccmth_ref_deg": 7.144,
        "output_dir": str(tmp_path / "out"),
    }
    seen = trace(strain.run, params)
    fracs = [f for f, _ in seen]
    assert fracs[-1] == 1.0
    assert all(b >= a for a, b in zip(fracs, fracs[1:])), fracs
    assert len(seen) >= 9, f"one layer reported {len(seen)} times: {seen}"
    first_after_start = next(f for f in fracs if f > 0.0)
    assert first_after_start <= 0.25, f"first report is {first_after_start}: {seen}"


def test_mosaicity_progress_is_smooth_with_one_layer(tmp_path):
    """The reads and the writes both report, so one layer still clears the bar.

    Only the four dataset reads reported, over 0.8 of the layer's slot; the
    stacked-file writes that follow them were the remaining 0.2 in one jump. At
    three layers that is 0.067 and invisible, at one layer it is 0.19.
    """
    import tests.test_stage_mosaicity as T
    from dfxm.stages import mosaicity

    root = T._make_root(tmp_path, ("layer__1",))
    params = {"mode": "batch", "root_folder": str(root), "folder_pattern": "layer__*"}
    assert_progress_wellformed(trace(mosaicity.run, params), label="mosaicity one layer")


def test_rocking_reports_per_layer_while_rendering(tmp_path):
    """`_render` must forward its sub-range into `save_layer_pngs`.

    It took a `progress` argument, spent it on a single `report(0.0)` and passed
    nothing down, so the whole render half of the stage — minutes on a real
    78-layer dataset — reported nothing. The gap stayed under the ceiling
    because the *fixture* has three layers, which is why this asserts on the
    reports themselves rather than on smoothness.
    """
    import tests.test_stage_rocking as T
    from dfxm.stages import rocking

    raw = T._setup(tmp_path)
    raw = raw[0] if isinstance(raw, tuple) else raw
    params = {
        "raw_root": str(raw),
        "rocking_pattern": "rock__*",
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "pixel_size_x_um": 0.152,
        "pixel_size_y_um": 0.385,
        "output_dir": str(tmp_path / "rk"),
        "save_layers": True,
        "save_animation": False,
        "save_topview": False,
    }
    seen = trace(rocking.run, params)
    assert_progress_wellformed(seen, label="rocking")
    per_layer = [t for _, t in seen if "layer " in t and "/" in t]
    assert len(per_layer) >= 6, f"no per-layer render reports: {[t for _, t in seen]}"


def test_profiles_preview_reports_through_the_render(tmp_path):
    """A preview job must report through its render and close at its own `job_hi`.

    Deliberately not `assert_progress_wellformed`. The preview branch used to
    `continue` past both the render and the job's closing report, so a one-job
    preview traced `[0.02, 0.134, 0.97, 1.0]` — everything `render_single` does,
    which is the whole of a preview, passed in silence and the bar sat at 0.134.
    Those two omissions are what this pins.

    The gap ceiling is not asserted here because a one-job preview cannot meet
    it: the job ends in a single `savefig`, one indivisible matplotlib call that
    rasterises the figure and is most of the job's cost. Reporting more finely
    around it would mean inventing boundaries where no work ends, which is the
    milestone reporting this whole sweep replaced. `MAX_FRAC_GAP` stays where it
    is; this stage-and-mode simply has one atomic step, and saying so here is
    better than passing a looser `max_gap` and calling it covered.
    """
    import tests.test_stage_profiles as T
    from dfxm.stages import profiles

    h5 = tmp_path / "consolidated.h5"
    T._write_consolidated(str(h5))
    seen = trace(profiles.run, T._base_params(h5, tmp_path / "prev", mode="preview"))

    fracs = [f for f, _ in seen]
    assert fracs[-1] == 1.0
    assert all(b >= a for a, b in zip(fracs, fracs[1:])), fracs
    assert len(seen) >= 8, f"preview reported {len(seen)} times: {seen}"
    # render_single's own reports, the ones the `continue` used to skip.
    assert any("laid out" in t for _, t in seen), seen
    assert any(t.startswith("saved ") for _, t in seen), seen
    # The job closes at its slot's top rather than wherever it happened to stop.
    assert max(f for f, t in seen if "done" in t) == pytest.approx(0.97)


def test_concat_progress_is_smooth(bliss_factory, tmp_path):
    """`concat` alone needs a conftest fixture, so it takes one rather than
    being folded into a parametrised sweep with the other four."""
    from dfxm.stages import concat

    folder = bliss_factory(specs=(("1.1", 3), ("2.1", 2), ("3.1", 4)))
    params = {"mode": "single", "input_folder": folder}
    assert_progress_wellformed(trace(concat.run, params), label="concat")
