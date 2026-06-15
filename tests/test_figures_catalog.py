import subprocess
import sys

import h5py
import numpy as np
import pytest

from dfxm.common import figures, render
from dfxm.common.figures import volume_layer_specs
from dfxm.common.plotting import PlotStyle
from dfxm.stages import mosaicity as Mosaicity
from dfxm.stages import rocking as Rocking
from dfxm.stages import strain as Strain
from dfxm.stages import visualize as Visualize
from dfxm.stages.registry import STAGE_TARGETS


def _layer():
    return np.random.default_rng(0).normal(size=(20, 40))


def test_layer_figure_legacy_has_scale_bar_and_equal_aspect():
    fig, ax, im = render.layer_figure(_layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb")
    assert ax.get_aspect() == 1.0  # physical equal aspect
    assert len(ax.patches) >= 1  # legacy scale bar present


def test_layer_figure_style_off_drops_scale_bar():
    fig, ax, im = render.layer_figure(
        _layer(),
        -1,
        1,
        "viridis",
        40.0,
        20.0,
        "t",
        "cb",
        style=PlotStyle(scale_bar=False),
    )
    assert len(ax.patches) == 0


def test_every_stage_has_a_catalog_entry():
    assert set(figures._FIGURE_CATALOGS) == set(STAGE_TARGETS)


def test_figures_for_concat_is_empty():
    assert figures.figures_for("concat", object(), {}) == []


def test_figures_for_unknown_stage_returns_empty():
    assert figures.figures_for("__no_such_stage__", object(), {}) == []


def test_volume_layer_specs_one_per_layer(tmp_path):
    vol = np.random.rand(3, 20, 40)
    h5 = tmp_path / "v.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=vol)
    specs = volume_layer_specs(
        h5_path=str(h5),
        dataset="strain",
        id_prefix="strain",
        title="Strain",
        cbar_label="Strain (ε)",
        cmap="RdBu_r",
        sx=0.1,
        sy=0.3,
        vmin=-1.0,
        vmax=1.0,
    )
    assert len(specs) == 3
    fig = specs[0].build(None)
    assert fig.axes[0].get_aspect() == 1.0


def test_volume_layer_specs_zum_label_and_field_formats(tmp_path):
    vol = np.random.rand(3, 20, 40)
    h5 = tmp_path / "v.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=vol)
    specs = volume_layer_specs(
        h5_path=str(h5),
        dataset="strain",
        id_prefix="strain",
        title="Strain",
        cbar_label="Strain (ε)",
        cmap="RdBu_r",
        sx=0.1,
        sy=0.3,
        vmin=-1.0,
        vmax=1.0,
        z_um=[0.0, 1.0, 2.5],
    )
    assert specs[0].figure_id == "strain_z0000"
    assert specs[0].filename == "strain_layer_0000"
    # z_um appears only in the RENDERED axes title, not FigureSpec.title
    fig = specs[1].build(None)
    assert "Z = 1.00 µm" in fig.axes[0].get_title()


def test_volume_layer_specs_zum_length_mismatch_raises(tmp_path):
    vol = np.random.rand(3, 20, 40)
    h5 = tmp_path / "v.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=vol)
    with pytest.raises(ValueError):
        volume_layer_specs(
            h5_path=str(h5),
            dataset="strain",
            id_prefix="strain",
            title="Strain",
            cbar_label="cb",
            cmap="RdBu_r",
            sx=0.1,
            sy=0.3,
            vmin=-1.0,
            vmax=1.0,
            z_um=[0.0, 1.0],  # only 2 for 3 layers
        )


def test_importing_figures_does_not_eager_import_stage_modules():
    code = (
        "import sys, dfxm.common.figures as F; "
        "assert 'dfxm.stages.matched' not in sys.modules; "
        "assert len(F._FIGURE_CATALOGS) >= 9; "
        "print('ok')"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_strain_figures_no_stacked_path_returns_empty(tmp_path):
    res = Strain.StrainResult(
        stacked_path="", volume_shape=(0, 0, 0), output_dir=str(tmp_path), layers=[]
    )
    assert Strain.figures(res, {}) == []


def test_strain_catalog_map_per_layer(tmp_path):
    vol = np.random.rand(2, 15, 25) * 1e-3
    h5 = tmp_path / "stacked.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=vol)
    res = Strain.StrainResult(
        stacked_path=str(h5),
        volume_shape=vol.shape,
        output_dir=str(tmp_path),
        layers=[
            Strain.LayerResult("L0", (15, 25), -1e-3, 1e-3, 0, 0),
            Strain.LayerResult("L1", (15, 25), -1e-3, 1e-3, 0, 0),
        ],
    )
    all_specs = Strain.figures(res, {"pixel_size_x_um": 0.1, "pixel_size_y_um": 0.3})
    assert len(all_specs) == 2  # exactly 2 specs total, no unexpected non-map entries
    specs = [s for s in all_specs if s.kind == "map"]
    assert len(specs) == 2
    assert specs[0].build(None).axes[0].get_aspect() == 1.0


def test_mosaicity_figures_no_stacked_path_returns_empty():
    res = Mosaicity.MosaicityResult(stacked_path=None, datasets={}, layers=[], skipped=[])
    assert Mosaicity.figures(res, {}) == []


def test_mosaicity_catalog_map_per_layer(tmp_path):
    # Mirror the REAL h5 layout written by mosaicity.run():
    #   /chi/Center of mass  (Z, Y, X)
    # i.e. group "chi", dataset "Center of mass"
    vol = np.random.rand(3, 10, 20).astype(np.float32)
    h5 = tmp_path / "stacked_volumes.h5"
    with h5py.File(h5, "w") as f:
        grp = f.require_group("chi")
        grp.create_dataset("Center of mass", data=vol)

    # result.datasets key is "/{group}/{ds_name}" as written by mosaicity.run() line 279
    res = Mosaicity.MosaicityResult(
        stacked_path=str(h5),
        datasets={"/chi/Center of mass": vol.shape},
        layers=["layer0", "layer1", "layer2"],
        skipped=[],
    )
    specs = Mosaicity.figures(res, {"pixel_size_x_um": 0.1, "pixel_size_y_um": 0.3})
    # one map spec per Z layer
    assert len(specs) >= 3
    map_specs = [s for s in specs if s.kind == "map"]
    assert len(map_specs) == 3
    # distinct filenames (no collisions across layers)
    filenames = [s.filename for s in map_specs]
    assert len(set(filenames)) == 3
    # equal-aspect rendered axes
    assert map_specs[0].build(None).axes[0].get_aspect() == 1.0


def test_mosaicity_catalog_multiple_keys_no_collision(tmp_path):
    vol = np.random.rand(2, 8, 12).astype(np.float32)
    h5 = tmp_path / "stacked_volumes.h5"
    with h5py.File(h5, "w") as f:
        grp = f.require_group("chi")
        grp.create_dataset("Center of mass", data=vol)
        grp.create_dataset("FWHM", data=vol)
    res = Mosaicity.MosaicityResult(
        stacked_path=str(h5),
        datasets={"/chi/Center of mass": vol.shape, "/chi/FWHM": vol.shape},
        layers=["l0", "l1"],
        skipped=[],
    )
    specs = Mosaicity.figures(res, {"pixel_size_x_um": 0.1, "pixel_size_y_um": 0.3})
    # 2 keys x 2 layers = 4 map specs, all distinct ids/filenames
    assert len(specs) == 4
    assert all(s.kind == "map" for s in specs)
    assert len({s.figure_id for s in specs}) == 4
    assert len({s.filename for s in specs}) == 4


# ---------------------------------------------------------------------------
# visualize.figures() tests
# ---------------------------------------------------------------------------


def test_visualize_figures_no_datasets_returns_empty():
    res = Visualize.VisualizeResult(output_dir="", datasets=[], skipped=[])
    assert Visualize.figures(res, {}) == []


def test_visualize_figures_strain_map_per_layer(tmp_path, monkeypatch):
    """figures() with a strain DatasetProducts yields one map spec per Z layer."""
    rng = np.random.default_rng(1)
    aligned_vol = rng.random((3, 10, 20)).astype(np.float32)

    # Patch _align so we don't need real motor files
    def fake_align(volume, samy, samz, *, scale_x, samy_direction, roi_x, roi_y):
        return aligned_vol, np.arange(3) * 2.0, 2.0

    monkeypatch.setattr(Visualize, "_align", fake_align)

    # Write a minimal strain volume file
    strain_h5 = tmp_path / "stacked_strain.h5"
    with h5py.File(strain_h5, "w") as f:
        f.create_dataset("strain", data=aligned_vol)

    monkeypatch.setattr(Visualize, "load_strain_volume", lambda path: aligned_vol.copy())

    ds = Visualize.DatasetProducts(
        name="strain",
        shape=(3, 10, 20),
        vmin=-0.001,
        vmax=0.001,
    )
    res = Visualize.VisualizeResult(
        output_dir=str(tmp_path),
        datasets=[ds],
    )
    params = {
        "strain_volume_file": str(strain_h5),
        "mosa_volume_file": "",
        "raw_root": "",
        "mosa_pattern": "*",
        "strain_pattern": "*",
        "pixel_size_x_um": 0.1,
        "pixel_size_y_um": 0.3,
    }
    specs = Visualize.figures(res, params)

    # One map spec per Z layer
    assert len(specs) == 3
    assert all(s.kind == "map" for s in specs)
    # Distinct figure_id and filename per layer
    assert len({s.figure_id for s in specs}) == 3
    assert len({s.filename for s in specs}) == 3
    # build(None) returns a Figure with equal-aspect axes
    fig = specs[0].build(None)
    assert fig.axes[0].get_aspect() == 1.0


def test_visualize_figures_mosa_map_per_layer(tmp_path, monkeypatch):
    """figures() with a mosaicity DatasetProducts yields one map spec per Z layer."""
    rng = np.random.default_rng(2)
    aligned_vol = rng.random((2, 8, 16)).astype(np.float32)

    def fake_align(volume, samy, samz, *, scale_x, samy_direction, roi_x, roi_y):
        return aligned_vol, np.arange(2) * 2.0, 2.0

    monkeypatch.setattr(Visualize, "_align", fake_align)

    mosa_h5 = tmp_path / "stacked_mosa.h5"
    with h5py.File(mosa_h5, "w") as f:
        grp = f.require_group("chi")
        grp.create_dataset("Center of mass", data=aligned_vol)

    mosa_datasets = {"chi_Center_of_mass": aligned_vol.copy()}
    monkeypatch.setattr(Visualize, "load_mosa_datasets", lambda path: mosa_datasets)

    ds = Visualize.DatasetProducts(
        name="chi_Center_of_mass",
        shape=(2, 8, 16),
        vmin=-0.5,
        vmax=0.5,
    )
    res = Visualize.VisualizeResult(
        output_dir=str(tmp_path),
        datasets=[ds],
    )
    params = {
        "mosa_volume_file": str(mosa_h5),
        "strain_volume_file": "",
        "raw_root": "",
        "mosa_pattern": "*",
        "strain_pattern": "*",
        "pixel_size_x_um": 0.1,
        "pixel_size_y_um": 0.3,
    }
    specs = Visualize.figures(res, params)

    assert len(specs) == 2
    assert all(s.kind == "map" for s in specs)
    assert len({s.figure_id for s in specs}) == 2
    assert len({s.filename for s in specs}) == 2
    fig = specs[0].build(None)
    assert fig.axes[0].get_aspect() == 1.0


def test_visualize_figures_each_layer_renders_its_own_layer(tmp_path, monkeypatch):
    """Each layer spec's build() must render its own distinct Z slice (not all z=0)."""
    rng = np.random.default_rng(42)
    aligned_vol = rng.random((3, 10, 20)).astype(np.float32)

    def fake_align(volume, samy, samz, *, scale_x, samy_direction, roi_x, roi_y):
        return aligned_vol, np.arange(3) * 2.0, 2.0

    monkeypatch.setattr(Visualize, "_align", fake_align)

    strain_h5 = tmp_path / "stacked_strain.h5"
    with h5py.File(strain_h5, "w") as f:
        f.create_dataset("strain", data=aligned_vol)

    monkeypatch.setattr(Visualize, "load_strain_volume", lambda path: aligned_vol.copy())

    ds = Visualize.DatasetProducts(name="strain", shape=(3, 10, 20), vmin=-0.001, vmax=0.001)
    res = Visualize.VisualizeResult(output_dir=str(tmp_path), datasets=[ds])
    params = {
        "strain_volume_file": str(strain_h5),
        "mosa_volume_file": "",
        "raw_root": "",
        "mosa_pattern": "*",
        "strain_pattern": "*",
        "pixel_size_x_um": 0.1,
        "pixel_size_y_um": 0.3,
    }
    specs = Visualize.figures(res, params)
    assert len(specs) == 3

    figs = [s.build(None) for s in specs]
    titles = [f.axes[0].get_title() for f in figs]
    # The rendered axes title format is "<title> (layer <z>)"
    assert "(layer 0)" in titles[0]
    assert "(layer 1)" in titles[1]
    assert "(layer 2)" in titles[2]


def test_visualize_figures_multi_dataset_no_collision(tmp_path, monkeypatch):
    """Two DatasetProducts → distinct ids/filenames across all layers."""
    rng = np.random.default_rng(3)
    vol2 = rng.random((2, 6, 12)).astype(np.float32)

    def fake_align(volume, samy, samz, *, scale_x, samy_direction, roi_x, roi_y):
        return vol2, np.arange(2) * 2.0, 2.0

    monkeypatch.setattr(Visualize, "_align", fake_align)

    mosa_h5 = tmp_path / "stacked_mosa.h5"
    with h5py.File(mosa_h5, "w") as f:
        grp = f.require_group("chi")
        grp.create_dataset("Center of mass", data=vol2)
        grp.create_dataset("FWHM", data=vol2)

    strain_h5 = tmp_path / "stacked_strain.h5"
    with h5py.File(strain_h5, "w") as f:
        f.create_dataset("strain", data=vol2)

    mosa_datasets = {
        "chi_Center_of_mass": vol2.copy(),
        "chi_FWHM": vol2.copy(),
    }
    monkeypatch.setattr(Visualize, "load_mosa_datasets", lambda path: mosa_datasets)
    monkeypatch.setattr(Visualize, "load_strain_volume", lambda path: vol2.copy())

    res = Visualize.VisualizeResult(
        output_dir=str(tmp_path),
        datasets=[
            Visualize.DatasetProducts(
                name="chi_Center_of_mass", shape=(2, 6, 12), vmin=-0.5, vmax=0.5
            ),
            Visualize.DatasetProducts(name="chi_FWHM", shape=(2, 6, 12), vmin=0.0, vmax=1.0),
            Visualize.DatasetProducts(name="strain", shape=(2, 6, 12), vmin=-1e-3, vmax=1e-3),
        ],
    )
    params = {
        "mosa_volume_file": str(mosa_h5),
        "strain_volume_file": str(strain_h5),
        "raw_root": "",
        "mosa_pattern": "*",
        "strain_pattern": "*",
        "pixel_size_x_um": 0.1,
        "pixel_size_y_um": 0.3,
    }
    specs = Visualize.figures(res, params)

    # 3 datasets x 2 layers = 6 map specs
    assert len(specs) == 6
    assert all(s.kind == "map" for s in specs)
    assert len({s.figure_id for s in specs}) == 6
    assert len({s.filename for s in specs}) == 6


# ---------------------------------------------------------------------------
# rocking.figures() tests
# ---------------------------------------------------------------------------


def test_rocking_figures_no_aligned_path_returns_empty():
    """Guard: no aligned_path → empty list."""
    res = Rocking.RockingResult(aligned_path=None, datasets=[], output_dir="")
    assert Rocking.figures(res, {}) == []


def test_rocking_figures_empty_datasets_returns_empty(tmp_path):
    """Guard: aligned_path set but datasets=[] → empty list (no specs to emit)."""
    aligned_h5 = tmp_path / "aligned.h5"
    vol = np.zeros((2, 4, 4), dtype=np.float32)
    with h5py.File(aligned_h5, "w") as f:
        f.create_dataset("sum_intensity", data=vol)
        f.create_dataset("z_uniform_um", data=np.array([0.0, 5.0], dtype=np.float32))
    res = Rocking.RockingResult(aligned_path=str(aligned_h5), datasets=[], output_dir="")
    assert Rocking.figures(res, {}) == []


def test_rocking_catalog_map_per_layer(tmp_path):
    """figures() with a RockingResult yields one map spec per Z layer per dataset."""
    # Mirror the REAL h5 layout written by rocking.save_aligned_raw_volumes():
    #   /sum_intensity   (Z, Y, X)  float32
    #   /specific_frame  (Z, Y, X)  float32
    #   /z_uniform_um    (Z,)       float32
    n_z = 3
    rng = np.random.default_rng(7)
    sum_vol = rng.random((n_z, 10, 20)).astype(np.float32)
    spec_vol = rng.random((n_z, 10, 20)).astype(np.float32)
    z_um = np.array([0.0, 5.0, 10.0], dtype=np.float32)

    aligned_h5 = tmp_path / "aligned_raw_rocking_volumes.h5"
    with h5py.File(aligned_h5, "w") as f:
        f.create_dataset("sum_intensity", data=sum_vol)
        f.create_dataset("specific_frame", data=spec_vol)
        f.create_dataset("z_uniform_um", data=z_um)

    # Two RockingProducts (one per dataset, as produced by rocking._render)
    prod_sum = Rocking.RockingProducts(name="raw_sum_intensity", vmin=0.1, vmax=5.0)
    prod_spec = Rocking.RockingProducts(name="raw_specific_frame_000", vmin=0.2, vmax=4.0)

    res = Rocking.RockingResult(
        aligned_path=str(aligned_h5),
        datasets=[prod_sum, prod_spec],
        volume_shape=(n_z, 10, 20),
        output_dir=str(tmp_path),
    )
    params = {"pixel_size_x_um": 0.152, "pixel_size_y_um": 0.385}

    specs = Rocking.figures(res, params)

    # 2 products x n_z layers = 2*3 = 6 map specs
    assert len(specs) == 2 * n_z
    assert all(s.kind == "map" for s in specs)
    # All figure_ids and filenames distinct
    assert len({s.figure_id for s in specs}) == 2 * n_z
    assert len({s.filename for s in specs}) == 2 * n_z
    # build(None) returns a Figure with equal-aspect axes
    assert specs[0].build(None).axes[0].get_aspect() == 1.0


def test_rocking_catalog_zum_label_in_title(tmp_path):
    """Z coordinate from z_uniform_um appears in rendered axes title."""
    n_z = 2
    vol = np.random.default_rng(9).random((n_z, 8, 12)).astype(np.float32)
    z_um = np.array([0.0, 7.5], dtype=np.float32)

    aligned_h5 = tmp_path / "aligned.h5"
    with h5py.File(aligned_h5, "w") as f:
        f.create_dataset("sum_intensity", data=vol)
        f.create_dataset("specific_frame", data=vol)
        f.create_dataset("z_uniform_um", data=z_um)

    prod = Rocking.RockingProducts(name="raw_sum_intensity", vmin=0.0, vmax=1.0)
    res = Rocking.RockingResult(
        aligned_path=str(aligned_h5),
        datasets=[prod],
        volume_shape=(n_z, 8, 12),
        output_dir=str(tmp_path),
    )
    params = {"pixel_size_x_um": 0.152, "pixel_size_y_um": 0.385}
    specs = Rocking.figures(res, params)

    # Should produce n_z specs for the one product
    assert len(specs) == n_z
    # The second layer (z=1) title should contain the Z coordinate
    fig = specs[1].build(None)
    assert "Z = " in fig.axes[0].get_title()


def test_rocking_figures_via_figures_for(tmp_path):
    """Both import directions work: rocking.figures() and figures_for('rocking')."""
    n_z = 2
    vol = np.random.default_rng(11).random((n_z, 6, 10)).astype(np.float32)
    z_um = np.array([0.0, 4.0], dtype=np.float32)

    aligned_h5 = tmp_path / "aligned.h5"
    with h5py.File(aligned_h5, "w") as f:
        f.create_dataset("sum_intensity", data=vol)
        f.create_dataset("specific_frame", data=vol)
        f.create_dataset("z_uniform_um", data=z_um)

    prod = Rocking.RockingProducts(name="raw_sum_intensity", vmin=0.0, vmax=1.0)
    res = Rocking.RockingResult(
        aligned_path=str(aligned_h5),
        datasets=[prod],
        volume_shape=(n_z, 6, 10),
        output_dir=str(tmp_path),
    )
    params = {"pixel_size_x_um": 0.152, "pixel_size_y_um": 0.385}

    # Direct call
    direct = Rocking.figures(res, params)
    # Via figures_for
    via_catalog = figures.figures_for("rocking", res, params)

    assert len(direct) == n_z
    assert len(via_catalog) == n_z


def test_rocking_sum_cbar_label_normalize_sum_aware(tmp_path):
    """FIX 1: sum_intensity colorbar label includes 'normalized' when normalize_sum=True."""
    n_z = 1
    vol = np.ones((n_z, 4, 4), dtype=np.float32)
    z_um = np.array([0.0], dtype=np.float32)

    aligned_h5 = tmp_path / "aligned.h5"
    with h5py.File(aligned_h5, "w") as f:
        f.create_dataset("sum_intensity", data=vol)
        f.create_dataset("z_uniform_um", data=z_um)

    prod = Rocking.RockingProducts(name="raw_sum_intensity", vmin=0.0, vmax=1.0)
    res = Rocking.RockingResult(
        aligned_path=str(aligned_h5),
        datasets=[prod],
        volume_shape=(n_z, 4, 4),
        output_dir=str(tmp_path),
    )
    params = {"pixel_size_x_um": 0.152, "pixel_size_y_um": 0.385, "normalize_sum": True}

    specs = Rocking.figures(res, params)
    assert len(specs) == 1
    # The colorbar label is set via cb.set_label() which becomes axes[1].get_ylabel()
    fig = specs[0].build(None)
    cbar_label = fig.axes[1].get_ylabel()
    assert "normalized" in cbar_label, (
        f"Expected 'normalized' in colorbar label, got: {cbar_label!r}"
    )


def test_rocking_specific_frame_title_includes_idx(tmp_path):
    """FIX 2: specific_frame title includes the frame index from result.specific_frame_idx."""
    n_z = 1
    vol = np.ones((n_z, 4, 4), dtype=np.float32)
    z_um = np.array([0.0], dtype=np.float32)

    aligned_h5 = tmp_path / "aligned.h5"
    with h5py.File(aligned_h5, "w") as f:
        f.create_dataset("specific_frame", data=vol)
        f.create_dataset("z_uniform_um", data=z_um)

    frame_idx = 7
    prod = Rocking.RockingProducts(name="raw_specific_frame_007", vmin=0.0, vmax=1.0)
    res = Rocking.RockingResult(
        aligned_path=str(aligned_h5),
        datasets=[prod],
        volume_shape=(n_z, 4, 4),
        output_dir=str(tmp_path),
        specific_frame_idx=frame_idx,
    )
    params = {"pixel_size_x_um": 0.152, "pixel_size_y_um": 0.385}

    specs = Rocking.figures(res, params)
    assert len(specs) == 1
    # Title should contain the frame index (mirrors run()'s f"Background-subtracted Frame {spec_idx}")
    fig = specs[0].build(None)
    axes_title = fig.axes[0].get_title()
    assert str(frame_idx) in axes_title, (
        f"Expected frame index {frame_idx} in title, got: {axes_title!r}"
    )
    assert "Background-subtracted Frame" in axes_title
