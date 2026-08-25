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

from tests.progress_trace import assert_progress_wellformed, trace


def test_visualize_progress_is_smooth(tmp_path):
    import tests.test_stage_visualize as T
    from dfxm.stages import visualize

    proc, raw = T._setup(tmp_path)
    params = T._stream_params(proc, raw, tmp_path / "viz")
    assert_progress_wellformed(trace(visualize.run, params), label="visualize")


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


def test_concat_progress_is_smooth(bliss_factory, tmp_path):
    """`concat` alone needs a conftest fixture, so it takes one rather than
    being folded into a parametrised sweep with the other four."""
    from dfxm.stages import concat

    folder = bliss_factory(specs=(("1.1", 3), ("2.1", 2), ("3.1", 4)))
    params = {"mode": "single", "input_folder": folder}
    assert_progress_wellformed(trace(concat.run, params), label="concat")
