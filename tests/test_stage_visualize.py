"""Tests for dfxm.stages.visualize — produces aligned 2D products and records
datasets; alignment reuses the golden-tested common.alignment primitives.
"""

from __future__ import annotations

import gc
import os
import weakref

import h5py
import numpy as np
import pytest

from dfxm.stages import visualize as V

L, NY, NX = 4, 6, 8


def _write_mosa(path, shape=(L, NY, NX)):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        for grp in ("chi", "mu"):
            g = f.create_group(grp)
            g.create_dataset("Center of mass", data=rng.standard_normal(shape))
            g.create_dataset("FWHM", data=np.abs(rng.standard_normal(shape)))


def _write_strain(path, shape=(L, NY, NX)):
    rng = np.random.default_rng(1)
    with h5py.File(path, "w") as f:
        f.create_dataset("strain", data=rng.standard_normal(shape) * 1e-4)


def _write_raw(root, pattern_base, samy, samz):
    for i in range(len(samy)):
        folder = os.path.join(root, f"{pattern_base}__{i + 1}")
        os.makedirs(folder)
        name = os.path.basename(folder)
        with h5py.File(os.path.join(folder, name + ".h5"), "w") as f:
            f.create_dataset("1.1/instrument/positioners/samy", data=samy[i])
            f.create_dataset("1.1/instrument/positioners/samz", data=samz[i])


def _setup(tmp_path, layers=L, ny=NY, nx=NX):
    proc = tmp_path / "proc"
    proc.mkdir()
    _write_mosa(str(proc / "stacked_volumes.h5"), (layers, ny, nx))
    _write_strain(str(proc / "stacked_strain_volumes.h5"), (layers, ny, nx))
    raw = tmp_path / "raw"
    raw.mkdir()
    # samy in sub-pixel steps (0.05 µm against a 0.152 µm pixel), so the layers
    # still overlap after the X-shift and the aligned volume is mostly FINITE
    # with a NaN rim. The 1-4 µm steps this fixture used to carry shifted an
    # 8 px-wide volume clean past itself: 6 finite voxels out of 840, which made
    # every colour-limit and centring assertion here a statement about NaN.
    # Irregular samz, so the uniform-Z interpolation really resamples.
    samy = np.arange(layers) * 5e-5
    samz = np.arange(layers) * 0.001 + np.linspace(0.0, 0.0004, layers)
    _write_raw(str(raw), "mosa", samy, samz)
    _write_raw(str(raw), "strain", samy, samz)
    _assert_mostly_finite(proc, raw)
    return proc, raw


# What the old fixture's motors destroyed, guarded at the source. A 99.3%-NaN
# aligned volume is not a failing fixture — every assertion in this module still
# passed on it, because none of them was capable of failing on NaN. So the guard
# has to be here, where the volume is defined, rather than inside whichever test
# happens to care.
_MIN_FINITE_FRACTION = 0.5


def _assert_mostly_finite(proc, raw):
    aligned, _z, _sz = V._align(
        V.load_mosa_field(str(proc / "stacked_volumes.h5"), "chi_FWHM"),
        *V._read_motors(
            str(raw),
            "mosa__*",
            "1.1/instrument/positioners/samy",
            "1.1/instrument/positioners/samz",
        ),
        scale_x=0.152,
        samy_direction=-1,
        roi_x=None,
        roi_y=None,
    )
    finite = float(np.isfinite(aligned).mean())
    assert finite > _MIN_FINITE_FRACTION, (
        f"the fixture's aligned volume is only {finite:.1%} finite — the samy steps have "
        "shifted the layers past each other again, and every colour-limit and centring "
        "assertion in this module becomes a statement about NaN"
    )


# -- lazy per-field loading ---------------------------------------------------
def test_mosa_field_names_matches_dict_keys(tmp_path):
    """The lazy API enumerates exactly what the old eager dict contained."""
    path = str(tmp_path / "stacked_volumes.h5")
    _write_mosa(path)

    names = V.mosa_field_names(path)
    assert names == sorted(names), "names must be deterministic across runs"
    assert names == ["chi_Center_of_mass", "chi_FWHM", "mu_Center_of_mass", "mu_FWHM"]

    # Every enumerated name reads back the same bytes an eager dict would hold.
    with h5py.File(path, "r") as f:
        eager = {
            f"{grp}_{ds.replace(' ', '_')}": f[grp][ds][:] for grp in ("chi", "mu") for ds in f[grp]
        }
    assert sorted(eager) == names
    for name in names:
        field = V.load_mosa_field(path, name)
        assert field is not None and field.ndim == 3
        np.testing.assert_array_equal(field, eager[name])


def test_load_mosa_field_unknown_name_returns_none(tmp_path):
    path = str(tmp_path / "stacked_volumes.h5")
    _write_mosa(path)
    assert V.load_mosa_field(path, "not_a_field") is None


def test_previous_aligned_volume_is_dead_before_the_next_align(tmp_path, monkeypatch):
    """No field's aligned volume survives into the next field's alignment.

    Only a 3-D product materialises a whole volume now (VTK uploads the grid in
    one piece), so `save_topview` is on and `render3d` is stubbed out — with it
    off there is no volume to outlive anything. A weakref on every materialised
    array, checked at each subsequent materialisation, is what distinguishes
    "released in time" from "released eventually".
    """
    proc, raw = _setup(tmp_path)
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: None)
    real_whole = V._LayerSource.whole
    refs: list[weakref.ref] = []
    alive_at_entry: list[list[bool]] = []

    def spy(self):
        gc.collect()
        alive_at_entry.append([r() is not None for r in refs])
        array = real_whole(self)
        refs.append(weakref.ref(array))
        return array

    monkeypatch.setattr(V._LayerSource, "whole", spy)
    V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "output_dir": str(tmp_path / "viz"),
            "save_layers": False,
            "save_animation": False,
            "save_topview": True,
        }
    )

    # 4 mosaicity fields + strain, in sorted order, one materialisation each.
    assert len(refs) == 5
    # The third entry is mu_Center_of_mass; index 1 in its list is chi_FWHM.
    assert alive_at_entry[2][1] is False, "chi_FWHM's aligned volume outlived its field"
    # And nothing else lingers either, including across into the strain section.
    assert not any(any(flags) for flags in alive_at_entry)


def test_run_never_materialises_a_volume_without_a_3d_product(tmp_path, monkeypatch):
    """With both 3-D products off, no whole aligned volume is ever built.

    The companion to the test above: that one pins *when* a materialised volume
    dies, this one pins that the streaming path does not create one at all. The
    layer PNGs and the animation read `_LayerSource[z]`, which holds one block.
    """
    proc, raw = _setup(tmp_path)
    calls = []
    monkeypatch.setattr(V._LayerSource, "whole", lambda self: calls.append(1))
    res = V.run({**_stream_params(proc, raw, tmp_path / "viz"), "_budget_bytes": 1 << 16})
    assert len(res.datasets) == 5, "the run must actually have rendered something"
    assert calls == []


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
    # ROI in Y -> 4 rows; X expanded by exactly the samy padding the motors imply.
    # `>= NX` alone was the bound that let the old fixture's 1-4 um samy steps
    # hide: it passed at 35 (layers shifted clean past each other) just as
    # happily as at the correct 9.
    samy, _samz = V._read_motors(
        str(raw), "mosa__*", "1.1/instrument/positioners/samy", "1.1/instrument/positioners/samz"
    )
    expected_nx = (
        NX + V.A.compute_pad_left(samy, 0.152, -1) + V.A.compute_pad_right(samy, 0.152, -1)
    )
    for d in res.datasets:
        assert d.shape[1] == 4
        assert d.shape[2] == expected_nx


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


def _capture_scene(monkeypatch, limit):
    """Stub the top-view render, keep the scene it was handed, set the GL limit."""
    captured = {}

    def fake_top(scene, path, **kw):
        captured["scene"] = scene
        return path

    monkeypatch.setattr(V.R3, "save_top_view", fake_top)
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: limit)
    return captured


def test_oversize_volume_becomes_a_note(tmp_path, monkeypatch):
    """Wider than the GL 3-D texture limit -> add_volume draws NOTHING and says
    nothing; opting out of the auto-fit must not report success with a blank
    product."""
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: 4)
    prod = _run_process({**_oversize_params(), "volume_downsample": 1}, tmp_path)
    assert any("texture limit" in n and "BLANK" in n for n in prod.notes)


def test_volume_downsample_auto_fits_the_texture_limit(tmp_path, monkeypatch):
    """The shipped default coarsens until it fits, so the product is a coarser
    picture instead of a silently blank one."""
    captured = _capture_scene(monkeypatch, 4)
    prod = _run_process(_oversize_params(), tmp_path)
    assert captured["scene"].downsample == 2  # (2, 4, 5) -> (2, 2, 2): 3 points <= 4
    assert any("coarsened 2x" in n for n in prod.notes)
    assert not any("BLANK" in n for n in prod.notes)


def test_volume_downsample_explicit_factor_is_honoured(tmp_path, monkeypatch):
    """An explicit factor is a user request, applied whether or not it is needed."""
    captured = _capture_scene(monkeypatch, 4096)  # nothing to fit
    prod = _run_process({**_oversize_params(), "volume_downsample": 2}, tmp_path)
    assert captured["scene"].downsample == 2
    assert not prod.notes


def test_no_oversize_note_for_a_small_volume_or_unknown_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: 4096)
    assert not _run_process(_oversize_params(), tmp_path).notes
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: None)  # no GL
    assert not _run_process(_oversize_params(), tmp_path).notes


def test_auto_fit_says_nothing_when_it_cannot_help(tmp_path, monkeypatch):
    """A volume too DEEP for the limit is not rescued by coarsening Y/X (Z is
    never block-averaged), so the honest blank-render warning survives."""
    captured = _capture_scene(monkeypatch, 2)  # even (2, 1, 1) needs 3 points
    prod = _run_process(_oversize_params(), tmp_path)
    assert captured["scene"].downsample == 1  # no resolution spent for nothing
    assert any("BLANK" in n for n in prod.notes)


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


# -- streamed alignment -------------------------------------------------------
def _stream_params(proc, raw, out):
    return {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "output_dir": str(out),
        "save_topview": False,  # GL not guaranteed in CI
    }


def test_align_streamed_matches_the_hand_rolled_chain(tmp_path):
    """`_align_streamed` drained equals `_align`, voxel for voxel."""
    proc, raw = _setup(tmp_path)
    # The motors `_setup` wrote, read back rather than restated, so this cannot
    # drift into comparing two all-NaN volumes if the fixture changes again.
    samy, samz = V._read_motors(
        str(raw), "mosa__*", "1.1/instrument/positioners/samy", "1.1/instrument/positioners/samz"
    )
    field = V.load_mosa_field(str(proc / "stacked_volumes.h5"), "chi_FWHM")
    ref, z_ref, sz_ref = V._align(
        field, samy, samz, scale_x=0.152, samy_direction=-1, roi_x=None, roi_y=None
    )
    with h5py.File(str(proc / "stacked_volumes.h5"), "r") as f:
        streamed, z_pos, scale_z = V._align_streamed(
            V._mosa_dataset(f, "chi_FWHM"),
            samy,
            samz,
            scale_x=0.152,
            samy_direction=-1,
            roi_x=None,
            roi_y=None,
            budget_bytes=1 << 16,
        )
        assert streamed.block_layers < streamed.shape[0], "budget did not block the stream"
        got = V._LayerSource(streamed.blocks, streamed.shape, streamed.dtype).whole()
    # (`_setup` guarantees `ref` is mostly finite, so this is not two NaN rims.)
    np.testing.assert_array_equal(got, ref)
    np.testing.assert_array_equal(z_pos, z_ref)
    assert scale_z == sz_ref


@pytest.mark.parametrize("method", ["midrange", "mean", "median"])
def test_streamed_clim_helpers_match_in_core(tmp_path, method):
    """Each streaming clim sibling reproduces its in-core original."""
    rng = np.random.default_rng(7)
    data = rng.standard_normal((5, 7, 9))
    data[0, 0, :3] = np.nan  # the alignment's NaN padding

    def blocks():
        return ((slice(z, z + 1), data[z : z + 1]) for z in range(data.shape[0]))

    ref_data, ref_lo, ref_hi = V._center_com_and_range(data, method, 99.5)
    got_blocks, lo, hi = V._center_com_and_range_streamed(blocks, method, 99.5)
    got = V._LayerSource(got_blocks, data.shape, data.dtype).whole()
    np.testing.assert_array_equal(got, ref_data)
    assert (lo, hi) == (ref_lo, ref_hi)

    assert V._colorbar_range_streamed(blocks) == V._colorbar_range(data)
    assert V._symmetric_range_streamed(blocks) == V._symmetric_range(data)


@pytest.mark.parametrize("method", ["midrange", "mean", "median"])
def test_visualize_streamed_matches_in_core(tmp_path, method):
    """Every rendered layer is identical whether the volume streamed or not.

    A real in-core-vs-streamed comparison, not budget-vs-budget: the machine's
    budget leaves the alignment one block, which sends `_source_and_clim` down
    the in-core rung with the original helpers, while the injected budget forces
    the streaming rung. `test_run_blocks_the_alignment_at_a_small_budget` is what
    keeps that second half from quietly becoming a second in-core run.

    Parametrised over the centring, because the three take different streaming
    reductions: `midrange` and `median` are percentiles (bit-equal to
    `np.percentile` by `stream_quantile`'s contract) while `mean` is a
    compensated sum against `np.nanmean`'s pairwise one, which is the case that
    could in principle move a pixel.
    """
    proc, raw = _setup(tmp_path)
    params = {**_stream_params(proc, raw, tmp_path / "ref"), "center_method": method}
    reference = V.run(params)
    streamed = V.run({**params, "output_dir": str(tmp_path / "str"), "_budget_bytes": 1 << 16})
    assert [d.name for d in streamed.datasets] == [d.name for d in reference.datasets]
    for a, b in zip(streamed.datasets, reference.datasets):
        assert (a.shape, a.vmin, a.vmax) == (b.shape, b.vmin, b.vmax)
    ref_pngs = sorted((tmp_path / "ref").rglob("*.png"))
    assert ref_pngs, "the reference run rendered nothing to compare"
    for ref_png in ref_pngs:
        rel = ref_png.relative_to(tmp_path / "ref")
        assert (tmp_path / "str" / rel).read_bytes() == ref_png.read_bytes(), rel


def test_visualize_never_hands_the_alignment_a_scratch_dir(tmp_path, monkeypatch):
    """No `scratch_dir=`, therefore no disk, therefore `scratch_bytes == 0`.

    This is the run-side half of
    `test_stage_estimates.py::test_visualize_never_prices_a_spill_it_cannot_perform`,
    and the reason that zero is the truth rather than a dropped term. Without a
    `scratch_dir` the multi-pass statistic re-reads instead of caching — slower,
    same result, no disk touched — so an estimator that priced a spill here
    would let `advice.plan_run` BLOCK a run on a full disk that the run would
    never have used.

    Checked with `center_method="median"`, which is the only setting that could
    cache at all, and at a budget small enough to force the blocked rung, where
    the caching would happen if it happened anywhere. Also asserts nothing is
    left on disk beside the products.
    """
    proc, raw = _setup(tmp_path)
    seen: list = []
    real = V.A.align_volume_streamed

    def spy(*a, **kw):
        seen.append(kw.get("scratch_dir", "NOT PASSED"))
        return real(*a, **kw)

    monkeypatch.setattr(V.A, "align_volume_streamed", spy)
    out = tmp_path / "median"
    V.run(
        {
            **_stream_params(proc, raw, out),
            "center_method": "median",
            "_budget_bytes": 1 << 16,
        }
    )
    assert seen, "no field went through the streaming alignment"
    assert set(seen) == {"NOT PASSED"}, seen
    assert not list(out.rglob("*scratch*")), "the run left a scratch directory behind"


def test_run_blocks_the_alignment_at_a_small_budget(tmp_path, monkeypatch):
    """The equivalence test is not vacuous: the small budget really blocks.

    Both halves are asserted, because both can rot: the injected budget must
    block the stream, and the machine's budget must *not* — if it ever did, the
    "reference" run would stop being the in-core path it is there to represent.
    """
    proc, raw = _setup(tmp_path)
    seen: dict[str, list] = {"small": [], "machine": []}
    real = V.A.align_volume_streamed
    key = "small"

    def spy(*a, **kw):
        streamed = real(*a, **kw)
        seen[key].append((streamed.block_layers, streamed.shape[0]))
        return streamed

    monkeypatch.setattr(V.A, "align_volume_streamed", spy)
    V.run({**_stream_params(proc, raw, tmp_path / "s"), "_budget_bytes": 1 << 16})
    key = "machine"
    V.run(_stream_params(proc, raw, tmp_path / "m"))
    assert seen["small"], "no field streamed"
    assert all(layers < nz for layers, nz in seen["small"]), seen["small"]
    assert all(layers >= nz for layers, nz in seen["machine"]), seen["machine"]


class _StrictVolume:
    """A `(Z, Y, X)` volume exposing ONLY the duck-type `_LayerSource` promises.

    `_LayerSource` stands in for an ndarray in `render.save_layer_pngs` and
    `render.save_layer_animation` on the contract that those two use `.shape`
    and `vol[z]` and nothing else. Nothing enforced that contract, so a future
    `volume.ravel()` or `volume[a:b]` in either renderer would break the
    streaming rung at run time rather than at review. This raises on any other
    attribute, and the test below drives both renderers through it.
    """

    _ALLOWED = {"shape", "__getitem__", "__class__"}

    def __init__(self, array):
        object.__setattr__(self, "_array", array)
        object.__setattr__(self, "shape", tuple(int(d) for d in array.shape))

    def __getitem__(self, z):
        if not isinstance(z, (int, np.integer)):
            raise AssertionError(
                f"the renderers may only index a volume by a single layer index, got {z!r}"
            )
        return object.__getattribute__(self, "_array")[int(z)]

    def __getattr__(self, name):
        raise AssertionError(
            f"render.py reached for volume.{name}, which is outside the duck-type "
            f"`_LayerSource` implements ({sorted(_StrictVolume._ALLOWED)}). Either add it "
            "to `_LayerSource` or stop using it in the renderers."
        )


def test_renderers_use_only_the_duck_type_layersource_implements(tmp_path):
    """`render.py` must not reach past `.shape` and `vol[z]`.

    The contract `_LayerSource` rests on, enforced instead of asserted in a
    comment. `_LayerSource` itself satisfies `_StrictVolume`'s interface by
    construction; what is at risk is the *renderers* growing a use it cannot
    serve, so they are what this drives.
    """
    from dfxm.common import render as Rnd

    vol = _StrictVolume(np.random.default_rng(0).standard_normal((3, 5, 7)))
    z_um = np.array([0.0, 1.0, 2.0])
    layers_dir = Rnd.save_layer_pngs(
        vol, z_um, str(tmp_path), "chi", -1.0, 1.0, "viridis", "t", "cb", 0.15, 0.38
    )
    assert len(os.listdir(layers_dir)) == 3
    assert Rnd.save_layer_animation(
        vol, z_um, str(tmp_path / "anim"), "chi", -1.0, 1.0, "viridis", "t", "cb", "gif", 0.15, 0.38
    )


def test_layer_source_rejects_a_negative_index(tmp_path):
    """A negative index is out of range for a forward-only stream, and says so.

    It used to surface as `TypeError: 'NoneType' object is not subscriptable`
    from the rewind's empty state, which names neither the cause nor the caller.
    """
    data = np.arange(24.0).reshape(2, 3, 4)

    def blocks():
        return iter([(slice(0, 2), data)])

    source = V._LayerSource(blocks, data.shape, data.dtype)
    np.testing.assert_array_equal(source[1], data[1])
    with pytest.raises(IndexError):
        source[-1]
    with pytest.raises(IndexError):
        source[2]


def _infinity_volume():
    """A volume carrying NaN padding, both infinities, AND a discriminating mean.

    The seed is load-bearing, not decorative. The rung equality below has two
    halves — the finite selection (`isfinite` vs `~isnan`) and the mean
    (`volumeio.stream_mean` vs `np.nanmean`) — and only the first is exercised by
    just putting infinities in. The compensated and the pairwise sum agree on
    many small samples: on the seed this fixture originally used they agreed
    exactly, so the whole test passed unchanged with `np.nanmean` restored and
    the mean half of the invariant was pinned by nothing.

    `_assert_mean_reductions_disagree` is what keeps that from happening again,
    and it is asserted rather than assumed for the twelfth time in this project:
    a fixture that stops discriminating is a test that stops testing.
    """
    rng = np.random.default_rng(0)
    data = rng.standard_normal((6, 8, 10))
    data[0, 0, :3] = np.nan
    data[1, 2, 4] = np.inf
    data[3, 5, 6] = -np.inf
    return data


def _assert_mean_reductions_disagree(data):
    """The two mean reductions must actually differ on *data*, or nothing is pinned."""
    from dfxm.common import volumeio

    finite = data[np.isfinite(data)]
    compensated = np.float64(volumeio.stream_mean([finite])).tobytes()
    pairwise = np.float64(np.nanmean(finite)).tobytes()
    assert compensated != pairwise, (
        "the fixture no longer separates `volumeio.stream_mean` from `np.nanmean` — "
        "they agree bit-for-bit on it, so the mean half of the rung equality below is "
        "asserted by nothing and reverting `_center_com_and_range` to `np.nanmean` "
        "would leave this test green. Change the seed or the shape until they differ."
    )


@pytest.mark.parametrize("method", ["midrange", "mean", "median"])
def test_both_rungs_agree_on_clims_with_infinities(method):
    """The rung boundary is an EQUALITY — and infinities are where it broke.

    `_source_and_clim` picks between the in-core helpers and the streaming ones
    by asking how much memory the machine has, so a value that differs between
    them is a value that depends on the machine. `_colorbar_range` used to
    select with `~np.isnan` (keeping `±inf`) while the streaming reductions
    filter on `np.isfinite`, and `np.nanmean` is not `volumeio.stream_mean`;
    both are fixed in the in-core helpers, and this is what holds them fixed.

    Note what the `mean` case needs that the others do not. The product-level
    check (rendered PNGs identical across the rungs) **cannot** see the mean
    divergence: a 1-ulp shift in the centring offset moves no percentile of
    `|value|`, so the colour limits come out bit-identical, and 8-bit PNG bytes
    could not move even if they did not. Only a value-level assertion on a
    fixture where the two reductions genuinely differ pins it — hence
    `_assert_mean_reductions_disagree`.

    Deliberately at the helper level rather than through `run()`: `run()` cannot
    be made to take both rungs on the same input without also changing the
    machine, and the equality is a property of the helpers.
    """
    data = _infinity_volume()
    # Both halves of the equality need a fixture that can see them fail.
    assert np.isinf(data).any() and np.isnan(data).any(), "fixture lost its non-finite values"
    _assert_mean_reductions_disagree(data)

    def blocks():
        return ((slice(z, z + 1), data[z : z + 1]) for z in range(data.shape[0]))

    # The convention every non-CoM mosaicity field takes — the one that diverged.
    assert V._colorbar_range_streamed(blocks) == V._colorbar_range(data)
    # And strain's.
    assert V._symmetric_range_streamed(blocks) == V._symmetric_range(data)
    # And the CoM centring, values as well as limits.
    ref_data, ref_lo, ref_hi = V._center_com_and_range(data, method, 99.5)
    got_blocks, lo, hi = V._center_com_and_range_streamed(blocks, method, 99.5)
    assert (lo, hi) == (ref_lo, ref_hi)
    np.testing.assert_array_equal(
        V._LayerSource(got_blocks, data.shape, data.dtype).whole(), ref_data
    )


def test_streamed_run_stays_within_its_working_set_budget(tmp_path):
    """The run's real allocation peak stays under the budget it was handed.

    This is what pins `REDUCTION_WORKING_SET_MULTIPLE`. That constant is a
    *divisor*, so getting it too small silently permits more than was counted —
    the under-predicting direction, and the one that ends in an OOM rather than
    in a slow run. Nothing but a measurement can catch it: every other test here
    passes at any value of it.

    `tracemalloc` rather than RSS on purpose. `budget_bytes` is priced in Python
    allocations (see `advice.working_set_budget_bytes`); comparing it against
    RSS would be the currency error this phase exists to avoid.

    The precondition is asserted, not assumed: at a budget this size the stream
    must actually block, or the in-core rung would run and the peak below would
    be measuring nothing.
    """
    import tracemalloc

    proc, raw = _setup(tmp_path, layers=32, ny=192, nx=192)
    seen = []
    real = V.A.align_volume_streamed

    def spy(*a, **kw):
        streamed = real(*a, **kw)
        seen.append((streamed.block_layers, streamed.shape[0]))
        return streamed

    for budget in (128 << 20, 256 << 20):
        params = {
            **_stream_params(proc, raw, tmp_path / f"viz{budget}"),
            "strain_volume_file": "",
            "save_layers": False,
            "save_animation": False,
            "_budget_bytes": budget,
        }
        seen.clear()
        V.A.align_volume_streamed = spy
        tracemalloc.start()
        try:
            V.run(params)
            _current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
            V.A.align_volume_streamed = real
        assert seen and all(layers < nz for layers, nz in seen), (
            f"budget {budget >> 20} MiB left the alignment in one block ({seen}) — the "
            "in-core rung ran and this measured nothing"
        )
        assert peak <= budget, (
            f"traced peak {peak / (1 << 20):.1f} MiB exceeded the "
            f"{budget >> 20} MiB budget it was handed"
        )


def test_visualize_peak_stays_under_budget(tmp_path):
    """The streaming rung really lowers peak RSS, not just the code path taken.

    `_budget_bytes` alone proves only that the blocks got smaller; a stage can
    block its read and then materialise a float64 copy anyway. Measured in the
    real child, four 64x256x256 float64 fields (33.5 MiB a volume) against a
    ~108 MiB child floor:

        previous commit (in-core)        340.4 MiB   (~7 volumes of data)
        this commit, machine budget      308.1 MiB   (the in-core rung, adopting
                                                      the single block instead of
                                                      copying it)
        this commit, 16 MiB budget       111.1 MiB   (~0.1 volumes)

    The 200 MiB limit sits between the two, so this fails against the previous
    commit — checked, not assumed — and has ~1.8x of margin on the passing side
    for a machine whose process image differs from this one's.

    The layer PNGs are off: 256 renders would dominate the runtime and none of
    them is where the memory goes (`_LayerSource` hands out one layer of one
    block). `_budget_bytes` is pinned rather than measured from the machine, so
    the figure does not depend on how much RAM the runner happens to have.
    """
    from tests.peak_rss import assert_peak_under

    proc, raw = _setup(tmp_path, layers=64, ny=256, nx=256)
    params = {
        **_stream_params(proc, raw, tmp_path / "viz"),
        "strain_volume_file": "",
        "save_layers": False,
        "save_animation": False,
        "_budget_bytes": 16 << 20,
    }
    result = assert_peak_under("dfxm.stages.visualize:run", params, 200 << 20, timeout=900)
    assert len(result.datasets) == 4, "the run must actually have visualized the fields"


def test_rss_floor_covers_the_measured_process_image(tmp_path):
    """`RSS_FLOOR_BYTES` must not sit below what this stage's child costs.

    The formula in `advice.working_set_budget_bytes` is pinned by
    `tests/test_common_advice.py`, but a *value* is not a formula: setting
    `RSS_FLOOR_BYTES = 1` passes every one of those and the budget it derives is
    then far too large. Only a live measurement catches that.

    **Every product on**, unlike the rest of this module, which turns the 3-D
    products off because CI has no GL. That is deliberate: the floor has to cover
    matplotlib *and* pyvista/VTK, because the 3-D import happens inside the first
    field's `_process_dataset` and every field after it streams with VTK already
    resident. The 3-D render itself is allowed to fail — `_probe_texture_limit`
    and `_process_dataset` both swallow a GL failure into a note — and the import
    is what is being measured either way.
    """
    from tests.peak_rss import assert_floor_covers

    proc, raw = _setup(tmp_path)
    params = {
        **_stream_params(proc, raw, tmp_path / "viz"),
        "strain_volume_file": "",
        "save_topview": True,
    }
    assert_floor_covers(
        V.RSS_FLOOR_BYTES,
        "dfxm.stages.visualize:run",
        params,
        data_bytes=4 * L * NY * NX * 8,
    )


def test_a_3d_product_that_overrides_the_budget_says_so(tmp_path, monkeypatch):
    """The one place the stage is NOT bounded by its budget must announce itself.

    A capped STO2 run peaks at 4.8 GiB against an 8 GB machine's 3.6 GiB
    headroom, entirely because a 3-D product materialises the whole aligned
    volume regardless of `_budget_bytes`. Silence there makes a setting look
    like a mystery. The note must name the size and the way out.
    """
    proc, raw = _setup(tmp_path)
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: None)

    result = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "output_dir": str(tmp_path / "viz"),
            "save_layers": False,
            "save_animation": False,
            "save_topview": True,
            # Force the streaming rung: without this the fixture fits in core and
            # nothing is being overridden, so there is nothing to warn about.
            "_budget_bytes": 4096,
        }
    )
    notes = [n for d in result.datasets for n in d.notes]
    overrides = [n for n in notes if "ignored the streaming budget" in n]
    assert overrides, f"no budget-override note among {notes}"
    assert "Save topview" in overrides[0]


def test_no_budget_override_note_when_the_volume_fits_in_core(tmp_path, monkeypatch):
    """Precondition guard for the test above.

    On the in-core rung the whole volume exists anyway, so nothing was
    overridden and the note would be a lie. Without this, the note could fire
    unconditionally and the test above would still pass.
    """
    proc, raw = _setup(tmp_path)
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    monkeypatch.setattr(V.R3, "volume_texture_limit", lambda *a, **kw: None)

    result = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "strain_pattern": "strain__*",
            "output_dir": str(tmp_path / "viz2"),
            "save_layers": False,
            "save_animation": False,
            "save_topview": True,
            "_budget_bytes": 8 << 30,
        }
    )
    notes = [n for d in result.datasets for n in d.notes]
    assert not [n for n in notes if "ignored the streaming budget" in n]


# -- already-aligned raw volumes ----------------------------------------------
# The rocking stage writes these files AFTER its own ROI crop, samy X-shift and
# uniform-Z interpolation, so visualize renders them as-is. Every assertion
# below is ultimately about that one word: as-is.
RAW_SCALES = (0.3, 0.4, 1.5)
RAW_FRAME_IDX = 7


def _write_aligned_raw(path, shape=(L, NY, NX), *, frame_idx=RAW_FRAME_IDX, scales=RAW_SCALES):
    """A rocking-stage aligned volume file: both datasets plus the alignment attrs.

    The scales deliberately differ from the stage defaults (0.152 / 0.385) so a
    render that silently used the form's pixel sizes instead of the file's own
    is visible rather than coincidentally right.
    """
    rng = np.random.default_rng(2)
    sx, sy, sz = scales
    with h5py.File(path, "w") as f:
        for name in ("sum_intensity", "specific_frame"):
            f.create_dataset(name, data=np.abs(rng.standard_normal(shape)).astype(np.float32))
        f.create_dataset("z_uniform_um", data=(np.arange(shape[0]) * sz).astype(np.float32))
        f.attrs["scale_x_um_per_px"] = sx
        f.attrs["scale_y_um_per_px"] = sy
        f.attrs["scale_z_um_per_px"] = sz
        f.attrs["specific_frame_idx"] = frame_idx


def _raw_params(tmp_path, out, **over):
    rock = tmp_path / "aligned_raw_rocking_volumes.h5"
    mosa = tmp_path / "aligned_raw_mosa_volumes.h5"
    _write_aligned_raw(str(rock))
    _write_aligned_raw(str(mosa))
    p = {
        "aligned_rocking_file": str(rock),
        "aligned_mosa_file": str(mosa),
        "output_dir": str(out),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
    }
    p.update(over)
    return p


ALL_RAW_KINDS = {"raw_sum", "raw_specific", "raw_mosa_sum", "raw_mosa_specific"}


def test_raw_volumes_render_without_being_realigned(tmp_path):
    """All four raw volumes render, at exactly their stored shape.

    The ROI is set and the raw volumes must ignore it: they were cropped once
    already, in the detector frame, and cropping them again would move them out
    of the frame the other volumes share.
    """
    out = tmp_path / "viz"
    res = V.run(_raw_params(tmp_path, out, roi_x="1,5", roi_y="1,4", save_layers=True))
    assert {d.name for d in res.datasets} == ALL_RAW_KINDS
    for d in res.datasets:
        assert d.shape == (L, NY, NX), f"{d.name} was re-cropped or re-aligned"
        assert d.layers_dir and len(os.listdir(d.layers_dir)) == L


def test_raw_toggles_select_which_volumes_render(tmp_path):
    res = V.run(
        _raw_params(
            tmp_path,
            tmp_path / "viz",
            include_raw_sum=False,
            include_mosa_specific=False,
        )
    )
    assert {d.name for d in res.datasets} == {"raw_specific", "raw_mosa_sum"}


def test_raw_volume_renders_at_the_files_own_scales(tmp_path, monkeypatch):
    """The pixel sizes come from the file's attrs, not from this stage's form."""
    seen = {}

    def _fake_pngs(data, z_pos, ds_dir, name, vmin, vmax, cmap, title, cbar, sx, sy, **kw):
        seen[name] = (sx, sy)
        return ds_dir

    monkeypatch.setattr(V.Rnd, "save_layer_pngs", _fake_pngs)
    V.run(
        _raw_params(
            tmp_path,
            tmp_path / "viz",
            save_layers=True,
            pixel_size_x_um=0.152,
            pixel_size_y_um=0.385,
        )
    )
    assert seen and set(seen) == ALL_RAW_KINDS
    for name, got in seen.items():
        assert got == pytest.approx(RAW_SCALES[:2]), name


def test_raw_shape_mismatch_becomes_a_note(tmp_path):
    """A raw volume cropped differently from the stacked ones will not overlay."""
    proc, raw = _setup(tmp_path)
    out = tmp_path / "viz"
    rock = tmp_path / "aligned_raw_rocking_volumes.h5"
    _write_aligned_raw(str(rock), (L, NY + 3, NX + 3))
    res = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "aligned_rocking_file": str(rock),
            "output_dir": str(out),
            "save_layers": False,
            "save_animation": False,
            "save_topview": False,
        }
    )
    by_name = {d.name: d for d in res.datasets}
    notes = " ".join(by_name["raw_sum"].notes)
    assert "overlay" in notes, notes
    assert not by_name["chi_FWHM"].notes  # the stacked volumes are the reference


def test_no_shape_note_when_the_raw_volume_matches(tmp_path):
    """Precondition guard: the note must not fire on a correctly-cropped volume."""
    proc, raw = _setup(tmp_path)
    rock = tmp_path / "aligned_raw_rocking_volumes.h5"
    # The stacked volumes gain samy padding along X, so match what run() produces.
    samy, _z = V._read_motors(
        str(raw), "mosa__*", "1.1/instrument/positioners/samy", "1.1/instrument/positioners/samz"
    )
    nx = NX + V.A.compute_pad_left(samy, 0.152, -1) + V.A.compute_pad_right(samy, 0.152, -1)
    _write_aligned_raw(str(rock), (L, NY, nx))
    res = V.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "aligned_rocking_file": str(rock),
            "output_dir": str(tmp_path / "viz"),
            "save_layers": False,
            "save_animation": False,
            "save_topview": False,
        }
    )
    raw_notes = [n for d in res.datasets if d.name.startswith("raw_") for n in d.notes]
    assert not [n for n in raw_notes if "overlay" in n], raw_notes


def test_raw_missing_file_is_recorded(tmp_path):
    res = V.run({"aligned_rocking_file": str(tmp_path / "nope.h5"), "output_dir": str(tmp_path)})
    assert any("not found" in s for s in res.skipped), res.skipped


def test_raw_display_info_carries_the_frame_index_and_the_raw_group(tmp_path):
    assert V._display_info("raw_sum") == (
        "Background-subtracted Sum Intensity",
        "Sum intensity (a.u.)",
        "raw",
    )
    title, cbar, group = V._display_info("raw_mosa_specific", frame_idx=RAW_FRAME_IDX)
    assert title == f"Mosa-integrated Frame {RAW_FRAME_IDX}"
    assert (cbar, group) == ("Intensity (a.u.)", "raw")


def test_available_fields_and_aligned_field_cover_the_raw_volumes(tmp_path):
    """The pop-out 3-D viewer offers the raw volumes, unaligned and at file scale."""
    p = _raw_params(tmp_path, tmp_path / "viz")
    assert set(V.available_fields(p)) == ALL_RAW_KINDS
    vol, spacing, _cmap, clim, meta = V.aligned_field(p, "raw_mosa_specific")
    with h5py.File(p["aligned_mosa_file"], "r") as f:
        stored = f["specific_frame"][:]
    assert np.allclose(vol, stored)  # loaded, not aligned
    assert spacing == pytest.approx(RAW_SCALES)
    assert meta["group"] == "raw"
    assert clim[0] <= clim[1]


def test_figures_cover_the_raw_volumes(tmp_path):
    p = _raw_params(tmp_path, tmp_path / "viz")
    res = V.run(p)
    specs = V.figures(res, p)
    ids = [s.figure_id for s in specs]
    assert sum(i.startswith("visualize_raw_mosa_sum_z") for i in ids) == L
    spec = next(s for s in specs if s.figure_id == "visualize_raw_mosa_sum_z0000")
    assert spec.build(None).axes  # renders without an alignment
    assert "Mosa-integrated Sum Intensity" in spec.title


def test_estimate_counts_the_aligned_raw_files(tmp_path):
    p = _raw_params(tmp_path, tmp_path / "viz")
    est = V.estimate({k: v for k, v in p.items() if k != "output_dir"})
    assert est.peak_bytes > 0
    assert est.chunkable


def test_raw_stream_reports_the_blocking_iter_blocks_uses(tmp_path):
    """`_fits_in_core` reads `block_layers`, so it must be the real blocking."""
    path = str(tmp_path / "aligned_raw_rocking_volumes.h5")
    _write_aligned_raw(path, (8, NY, NX))
    with h5py.File(path, "r") as f:
        dset = f["sum_intensity"]
        per_layer = V.volumeio.volume_bytes(dset) // 8
        for budget in (1, per_layer * 40, 1 << 30):
            streamed = V._raw_streamed(dset, z_um=np.arange(8.0), scale_z=1.5, budget_bytes=budget)
            first = next(iter(streamed.blocks()))[0]
            assert streamed.block_layers == first.stop - first.start
            assert sum(b.shape[0] for _sl, b in streamed.blocks()) == 8


def test_visualize_prefills_the_aligned_files_from_the_experiment(tmp_path):
    from dfxm.config.models import Experiment
    from gui.bindings import experiment_overrides

    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "aligned_raw_mosa_volumes.h5").write_bytes(b"")
    ov = experiment_overrides("visualize", Experiment(processed_root=str(proc)))
    assert ov["aligned_mosa_file"] == str(proc / "aligned_raw_mosa_volumes.h5")
    # Not written yet -> blank, NOT a path. Both fields are `must_exist`, and
    # `StageView._validate_inputs` blocks a run on any must_exist path that is
    # set but absent — so pre-filling one before rocking has run would make
    # visualize unrunnable for anyone who never runs rocking.
    assert ov["aligned_rocking_file"] == ""
    (proc / "aligned_raw_rocking_volumes.h5").write_bytes(b"")
    ov = experiment_overrides("visualize", Experiment(processed_root=str(proc)))
    assert ov["aligned_rocking_file"] == str(proc / "aligned_raw_rocking_volumes.h5")


def test_figures_route_a_raw_dataset_by_name_not_by_the_toggle(tmp_path):
    """A result outlives the form. Switching a toggle off must not misroute it.

    `figures` and `aligned_field` are asked about datasets a PREVIOUS run made;
    routing them through the toggle-respecting `_raw_configs` would send a raw
    dataset down the mosaicity branch and fail at build time.
    """
    p = _raw_params(tmp_path, tmp_path / "viz")
    res = V.run(p)
    off = {**p, "include_mosa_sum": False, "include_raw_sum": False}
    spec = next(s for s in V.figures(res, off) if s.figure_id == "visualize_raw_mosa_sum_z0000")
    assert spec.build(None).axes
    vol, _sp, _cm, _cl, meta = V.aligned_field(off, "raw_mosa_sum")
    assert vol.shape == (L, NY, NX)
    assert meta["group"] == "raw"


def test_figures_do_not_raise_when_a_raw_file_has_moved(tmp_path):
    """Listing specs is cheap and must stay so — the failure belongs in build()."""
    p = _raw_params(tmp_path, tmp_path / "viz")
    res = V.run(p)
    os.remove(p["aligned_mosa_file"])
    specs = V.figures(res, p)  # must not raise
    assert any(s.figure_id.startswith("visualize_raw_mosa_sum_z") for s in specs)


# -- per-volume selection + per-volume 3-D opacity ----------------------------
def _stacked_params(proc, raw, out, **over):
    p = {
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
    p.update(over)
    return p


def test_stacked_toggles_select_which_volumes_render(tmp_path):
    proc, raw = _setup(tmp_path)
    res = V.run(
        _stacked_params(
            proc,
            raw,
            tmp_path / "viz",
            include_mosa_com_chi=False,
            include_mosa_fwhm_mu=False,
            include_strain=False,
        )
    )
    assert {d.name for d in res.datasets} == {"chi_FWHM", "mu_Center_of_mass"}


def test_an_unknown_mosa_field_has_no_toggle_and_still_renders(tmp_path):
    """No existing run may silently lose a dataset: only the four named fields
    are gated, and anything else in the file keeps rendering."""
    proc, raw = _setup(tmp_path)
    with h5py.File(str(proc / "stacked_volumes.h5"), "a") as f:
        f["chi"].create_dataset(
            "Skew", data=np.abs(np.random.default_rng(9).standard_normal((L, NY, NX)))
        )
    res = V.run(
        _stacked_params(
            proc,
            raw,
            tmp_path / "viz",
            include_mosa_com_chi=False,
            include_mosa_fwhm_chi=False,
            include_mosa_com_mu=False,
            include_mosa_fwhm_mu=False,
            include_strain=False,
        )
    )
    assert {d.name for d in res.datasets} == {"chi_Skew"}


def test_available_fields_respects_the_stacked_toggles(tmp_path):
    """The pop-out viewer list must match what the run actually produced."""
    proc, raw = _setup(tmp_path)
    p = _stacked_params(
        proc, raw, tmp_path / "viz", include_mosa_fwhm_chi=False, include_strain=False
    )
    assert set(V.available_fields(p)) == {"chi_Center_of_mass", "mu_Center_of_mass", "mu_FWHM"}


def test_opacity_inherits_the_globals_when_nothing_is_overridden():
    p = {**V.STAGE.defaults(), "volume_opacity": 0.7, "opacity_mapping": "sigmoid"}
    for name in ("chi_FWHM", "strain", "raw_mosa_sum"):
        assert V._opacity_for(name, p) == (0.7, "sigmoid")


def test_per_dataset_opacity_and_mapping_override_the_globals():
    p = {
        **V.STAGE.defaults(),
        "volume_opacity": 0.7,
        "opacity_mapping": "sigmoid",
        "volume_opacity_raw_sum": "0.25",
        "opacity_mapping_raw_mosa_specific": "geom_r",
    }
    assert V._opacity_for("raw_sum", p) == (0.25, "sigmoid")  # opacity only
    assert V._opacity_for("raw_mosa_specific", p) == (0.7, "geom_r")  # mapping only
    assert V._opacity_for("mu_FWHM", p) == (0.7, "sigmoid")  # untouched


def test_every_volume_has_its_own_opacity_pair():
    """One pair per renderable dataset, and each keyed independently."""
    names = {p.name for p in V.STAGE.params}
    for key, _ds, _label in V._VOLUME_KEYS:
        assert f"volume_opacity_{key}" in names
        assert f"opacity_mapping_{key}" in names
    # and every key maps back to a dataset run() can actually produce
    assert set(V._KEY_BY_DATASET) == {ds for _k, ds, _l in V._VOLUME_KEYS}


def test_per_dataset_opacity_reaches_the_scene(tmp_path, monkeypatch):
    """The knob is worthless if it does not arrive at Scene3D."""
    captured = _capture_scene(monkeypatch, None)
    p = {
        **_oversize_params(),
        "volume_opacity": 0.9,
        "opacity_mapping": "linear",
        "volume_opacity_mosa_fwhm_chi": "0.3",
        "opacity_mapping_mosa_fwhm_chi": "geom",
    }
    V._process_dataset(
        np.zeros((2, 4, 5)),
        [0.0, 1.0],
        1.0,
        "chi_FWHM",
        0.0,
        1.0,
        "viridis",
        "chi",
        "deg",
        p,
        str(tmp_path),
    )
    assert captured["scene"].opacity == pytest.approx(0.3)
    assert captured["scene"].opacity_mapping == "geom"


def test_a_dataset_without_an_override_still_gets_the_globals_at_the_scene(tmp_path, monkeypatch):
    captured = _capture_scene(monkeypatch, None)
    p = {**_oversize_params(), "volume_opacity": 0.42, "opacity_mapping": "geom_r"}
    V._process_dataset(
        np.zeros((2, 4, 5)),
        [0.0, 1.0],
        1.0,
        "strain",
        0.0,
        1.0,
        "RdBu_r",
        "strain",
        "eps",
        p,
        str(tmp_path),
    )
    assert captured["scene"].opacity == pytest.approx(0.42)
    assert captured["scene"].opacity_mapping == "geom_r"


@pytest.mark.parametrize("bad", ["abc", "1.5", "-0.2", "0,5"])
def test_an_unusable_opacity_override_is_a_user_error(tmp_path, bad):
    """Fail before a voxel is read, with a hint — not 20 minutes into a run."""
    from dfxm.common.errors import StageUserError

    proc, raw = _setup(tmp_path)
    with pytest.raises(StageUserError) as exc:
        V.run(_stacked_params(proc, raw, tmp_path / "viz", volume_opacity_strain=bad))
    assert exc.value.hint
    assert "strain" in str(exc.value) or "strain" in exc.value.hint


def test_the_opacity_check_runs_before_any_volume_is_read(tmp_path, monkeypatch):
    """Precondition guard for the test above: it must not be a late failure."""
    from dfxm.common.errors import StageUserError

    proc, raw = _setup(tmp_path)
    monkeypatch.setattr(
        V, "_align_streamed", lambda *a, **kw: pytest.fail("a volume was read before validation")
    )
    with pytest.raises(StageUserError):
        V.run(_stacked_params(proc, raw, tmp_path / "viz", volume_opacity_mosa_com_chi="nope"))
