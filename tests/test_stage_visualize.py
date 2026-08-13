"""Tests for dfxm.stages.visualize — produces aligned 2D products and records
datasets; alignment reuses the golden-tested common.alignment primitives.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest

from dfxm.stages import visualize as V

L, NY, NX = 4, 6, 8


def _write_mosa(path):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for grp in ("chi", "mu"):
            g = f.create_group(grp)
            g.create_dataset("Center of mass", data=rng.standard_normal((L, NY, NX)))
            g.create_dataset("FWHM", data=np.abs(rng.standard_normal((L, NY, NX))))


def _write_strain(path):
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        f.create_dataset("strain", data=rng.standard_normal((L, NY, NX)) * 1e-4)


def _write_raw(root, pattern_base, samy, samz):
    for i in range(L):
        folder = os.path.join(root, f"{pattern_base}__{i + 1}")
        os.makedirs(folder)
        name = os.path.basename(folder)
        with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
            f.create_dataset("1.1/instrument/positioners/samy", data=samy[i])
            f.create_dataset("1.1/instrument/positioners/samz", data=samz[i])


def _setup(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_mosa(str(proc / "stacked_volumes.h5"))
    _write_strain(str(proc / "stacked_strain_volumes.h5"))
    raw = tmp_path / "raw"
    raw.mkdir()
    samy = np.array([0.0, 0.001, 0.0025, 0.004])
    samz = np.array([0.0, 0.001, 0.0021, 0.0035])
    _write_raw(str(raw), "mosa", samy, samz)
    _write_raw(str(raw), "strain", samy, samz)
    return proc, raw


def test_run_produces_layers_and_animation(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "viz"
    res = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "output_dir": str(out),
            "save_topview": False,  # GL not guaranteed in CI
        }
    )
    names = {d.name for d in res.datasets}
    assert names == {
        "chi_Center_of_mass",
        "chi_FWHM",
        "mu_Center_of_mass",
        "mu_FWHM",
        "strain",
    }
    for d in res.datasets:
        assert d.layers_dir and os.path.isdir(d.layers_dir)
        pngs = [p for p in os.listdir(d.layers_dir) if p.endswith(".png")]
        assert len(pngs) == d.shape[0]  # one PNG per aligned Z layer
        assert d.animation and os.path.exists(d.animation)


def test_com_is_centered_strain_is_symmetric(tmp_path):
    proc, raw = _setup(tmp_path)
    res = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "output_dir": str(tmp_path / "viz"),
            "center_method": "midrange",
            "save_layers": False,
            "save_animation": False,
            "save_topview": False,
        }
    )
    by_name = {d.name: d for d in res.datasets}
    com = by_name["chi_Center_of_mass"]
    assert com.vmin == pytest.approx(-com.vmax)  # midrange -> symmetric
    strain = by_name["strain"]
    assert strain.vmin == pytest.approx(-strain.vmax)  # strain symmetric range


def test_alignment_shape_matches_primitives(tmp_path):
    proc, raw = _setup(tmp_path)
    res = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "output_dir": str(tmp_path / "viz"),
            "roi_y": "1,5",
            "save_layers": False,
            "save_animation": False,
            "save_topview": False,
        }
    )
    # ROI in Y -> 4 rows; X expanded by samy padding -> >= NX
    for d in res.datasets:
        assert d.shape[1] == 4
        assert d.shape[2] >= NX


def test_missing_inputs_recorded(tmp_path):
    res = V.run({"mosa_volume_file": str(tmp_path / "nope.h5"), "strain_volume_file": ""})
    assert any("not found" in s for s in res.skipped)


def test_parse_pair_and_display_info():
    assert V._parse_pair("") is None
    assert V._parse_pair("3, 7") == (3, 7)
    with pytest.raises(ValueError):
        V._parse_pair("1,2,3")
    # third element is the colormap GROUP (resolved via PlotStyle), not a cmap name
    assert V._display_info("strain", is_strain=True)[2] == "strain"
    assert V._display_info("chi_Center_of_mass")[2] == "mosa_com"
    assert V._display_info("mu_FWHM")[2] == "mosa_fwhm"
    assert V._display_info("something_else")[2] is None


def test_figures_resolve_cmap_from_style(tmp_path):
    """Visualize FigureSpecs resolve their colormap from the style at build time."""
    from dfxm.common.plotting import PlotStyle

    proc, raw = _setup(tmp_path)
    out = tmp_path / "viz"
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "output_dir": str(out),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
    }
    res = V.run(params)
    specs = V.figures(res, params)
    com_spec = next(s for s in specs if "Center_of_mass" in s.figure_id)
    fig = com_spec.build(PlotStyle(cmap_mosa_com="viridis"))
    assert fig.axes[0].images[0].cmap.name == "viridis"
    assert com_spec.build(None).axes[0].images[0].cmap.name == "fast"


def test_visualize_make_build_threads_group(monkeypatch):
    import numpy as np

    from dfxm.common import render as R
    from dfxm.common.plotting import PlotStyle
    from dfxm.stages import visualize as V

    captured = {}
    real = R.layer_figure

    def spy(*a, **k):
        captured["group"] = k.get("group")
        return real(*a, **k)

    monkeypatch.setattr(R, "layer_figure", spy)
    build = V._make_build(
        lambda: np.zeros((1, 4, 5)), 0, 0.0, 1.0, "strain", 5.0, 4.0, "title", "cbar"
    )
    build(PlotStyle())
    assert captured["group"] == "strain"


def test_process_dataset_rotation_video(tmp_path, monkeypatch):
    calls = {}

    def fake_rotation(scene, base_path, fmt, **kw):
        calls["base_path"] = base_path
        calls["fmt"] = fmt
        calls["cbar_label"] = kw.get("cbar_label")
        calls["spacing"] = scene.spacing
        return base_path + ".gif"

    monkeypatch.setattr(V.R3, "save_rotation_video", fake_rotation)
    p = {
        **V.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
        "save_rotation": True,
        "output_format": "gif",
    }
    data = np.zeros((2, 4, 5))
    prod = V._process_dataset(
        data, [0.0, 1.0], 1.0, "chi", 0.0, 1.0, "viridis", "chi", "deg", p, str(tmp_path)
    )
    assert prod.rotation_video == calls["base_path"] + ".gif"
    assert calls["base_path"].endswith(os.path.join("chi", "chi_rotation"))
    assert calls["fmt"] == "gif"
    # the stage hands render3d a Scene3D carrying the physical spacing + labels
    assert calls["cbar_label"] == "deg"
    assert calls["spacing"][2] == 1.0


def test_process_dataset_rotation_video_empty_volume_becomes_note(tmp_path, monkeypatch):
    monkeypatch.setattr(V.R3, "save_rotation_video", lambda *a, **kw: None)
    p = {
        **V.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
        "save_rotation": True,
    }
    prod = V._process_dataset(
        np.zeros((2, 4, 5)),
        [0.0, 1.0],
        1.0,
        "chi",
        0.0,
        1.0,
        "viridis",
        "chi",
        "deg",
        p,
        str(tmp_path),
    )
    assert prod.rotation_video is None
    assert any("no finite voxels" in n for n in prod.notes)


def test_process_dataset_rotation_video_failure_becomes_note(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("no GL")

    monkeypatch.setattr(V.R3, "save_rotation_video", boom)
    p = {
        **V.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
        "save_rotation": True,
    }
    prod = V._process_dataset(
        np.zeros((2, 4, 5)),
        [0.0, 1.0],
        1.0,
        "chi",
        0.0,
        1.0,
        "viridis",
        "chi",
        "deg",
        p,
        str(tmp_path),
    )
    assert prod.rotation_video is None
    assert any("rotation video skipped" in n for n in prod.notes)


def _oversize_params():
    return {
        **V.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": True,
        "save_rotation": False,
    }


def _run_process(p, tmp_path, nx=5):
    return V._process_dataset(
        np.zeros((2, 4, nx)),
        [0.0, 1.0],
        1.0,
        "chi",
        0.0,
        1.0,
        "viridis",
        "chi",
        "deg",
        p,
        str(tmp_path),
    )


def test_oversize_volume_becomes_a_note(tmp_path, monkeypatch):
    """Wider than the GL 3-D texture limit -> add_volume draws NOTHING and says
    nothing; the stage must not report success with a blank product."""
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: 4)
    prod = _run_process(_oversize_params(), tmp_path)
    assert any("texture limit" in n for n in prod.notes)


def test_no_oversize_note_for_a_small_volume_or_unknown_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: 4096)
    assert not _run_process(_oversize_params(), tmp_path).notes
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: None)  # no GL
    assert not _run_process(_oversize_params(), tmp_path).notes


def test_scene_carries_new_3d_params(tmp_path, monkeypatch):
    captured = {}

    def fake_top(scene, path, **kw):
        captured["scene"] = scene
        return path

    monkeypatch.setattr(V.R3, "save_top_view", fake_top)
    p = {
        **V.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": True,
        "save_rotation": False,
        "render_mode": "isosurface",
        "opacity_mapping": "sigmoid",
        "rotation_frames": 24,
    }
    V._process_dataset(
        np.zeros((2, 4, 5)),
        [0.0, 1.0],
        1.0,
        "chi",
        0.0,
        1.0,
        "viridis",
        "chi",
        "deg",
        p,
        str(tmp_path),
    )
    assert captured["scene"].mode == "isosurface"
    assert captured["scene"].opacity_mapping == "sigmoid"


def test_log_scale_guard_falls_back_with_note(tmp_path, monkeypatch):
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    p = {
        **V.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": True,
        "save_rotation": False,
        "log_scale": True,
    }
    # vmin/vmax straddle zero -> log mapping is invalid -> guard falls back + notes
    prod = V._process_dataset(
        np.zeros((2, 4, 5)),
        [0.0, 1.0],
        1.0,
        "chi",
        -1.0,
        1.0,
        "viridis",
        "chi",
        "deg",
        p,
        str(tmp_path),
    )
    assert any("log scale skipped" in n for n in prod.notes)


def test_rotation_frames_passed_through(tmp_path, monkeypatch):
    seen = {}

    def fake_rotation(scene, base, fmt, **kw):
        seen.update(kw)
        return base + ".gif"

    monkeypatch.setattr(V.R3, "save_rotation_video", fake_rotation)
    p = {
        **V.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
        "save_rotation": True,
        "rotation_frames": 24,
    }
    V._process_dataset(
        np.zeros((2, 4, 5)),
        [0.0, 1.0],
        1.0,
        "chi",
        0.0,
        1.0,
        "viridis",
        "chi",
        "deg",
        p,
        str(tmp_path),
    )
    assert seen["n_frames"] == 24
