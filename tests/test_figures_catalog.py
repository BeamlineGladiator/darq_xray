import subprocess
import sys

import h5py
import numpy as np
import pytest
from matplotlib.offsetbox import AnchoredOffsetbox

from dfxm.common import figures, render
from dfxm.common.figures import volume_layer_specs
from dfxm.common.plotting import PlotStyle
from dfxm.stages import matched as Matched
from dfxm.stages import mosaicity as Mosaicity
from dfxm.stages import profiles as Profiles
from dfxm.stages import rocking as Rocking
from dfxm.stages import slices as Slices
from dfxm.stages import strain as Strain
from dfxm.stages import visualize as Visualize
from dfxm.stages.registry import STAGE_TARGETS


def _layer():
    return np.random.default_rng(0).normal(size=(20, 40))


def test_layer_figure_legacy_has_scale_bar_and_equal_aspect():
    fig, ax, im = render.layer_figure(_layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb")
    assert ax.get_aspect() == 1.0  # physical equal aspect
    # legacy scale bar present (an AnchoredOffsetbox since the offsetbox rebuild)
    assert any(isinstance(a, AnchoredOffsetbox) for a in ax.artists)


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
    assert not any(isinstance(a, AnchoredOffsetbox) for a in ax.artists)


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
    # After Task 18: 3 specs per layer (map + histogram + detrend), 2 layers = 6 total
    map_specs = [s for s in all_specs if s.kind == "map"]
    assert len(map_specs) == 2
    assert map_specs[0].build(None).axes[0].get_aspect() == 1.0


def _make_strain_maps_h5(tmp_path, layer_name, ccmth_com_path):
    """Write a minimal maps.h5 with the ccmth COM dataset for the detrend-diag test."""
    rng = np.random.default_rng(17)
    ccmth = rng.normal(loc=7.144, scale=0.001, size=(15, 25)).astype(np.float32)
    # Nest the dataset under the full HDF5 path
    maps_h5 = tmp_path / layer_name / "maps.h5"
    maps_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(maps_h5, "w") as f:
        # ccmth_com_path may be a nested HDF5 path like "/entry/ccmth/…"
        # use require_group to create intermediate groups
        parts = [p for p in ccmth_com_path.split("/") if p]
        grp = f
        for part in parts[:-1]:
            grp = grp.require_group(part)
        grp.create_dataset(parts[-1], data=ccmth)
    return str(maps_h5)


def test_strain_catalog_plot_specs_per_layer(tmp_path):
    """figures() emits one histogram + one detrend-diag plot spec per layer."""
    n_layers = 2
    vol = np.random.default_rng(5).random((n_layers, 15, 25)) * 1e-3
    h5 = tmp_path / "stacked.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=vol)

    ccmth_com_path = "/entry/ccmth/Center of mass/Center of mass"
    layer_names = ["L0", "L1"]

    # Create a synthetic maps.h5 for each layer so the detrend build succeeds
    for name in layer_names:
        _make_strain_maps_h5(tmp_path, name, ccmth_com_path)

    res = Strain.StrainResult(
        stacked_path=str(h5),
        volume_shape=vol.shape,
        output_dir=str(tmp_path),
        layers=[
            Strain.LayerResult("L0", (15, 25), -1e-3, 1e-3, 0, 0),
            Strain.LayerResult("L1", (15, 25), -1e-3, 1e-3, 0, 0),
        ],
    )
    params = {
        "pixel_size_x_um": 0.1,
        "pixel_size_y_um": 0.3,
        "mode": "batch",
        "root_folder": str(tmp_path),
        "maps_filename": "maps.h5",
        "ccmth_com_path": ccmth_com_path,
        "roi": "",
    }
    all_specs = Strain.figures(res, params)

    plot_specs = [s for s in all_specs if s.kind == "plot"]
    # 2 plot specs per layer (histogram + detrend) × 2 layers = 4
    assert len(plot_specs) == 2 * n_layers

    # Split into histogram and detrend specs
    hist_specs = [s for s in plot_specs if "hist" in s.figure_id]
    detrend_specs = [s for s in plot_specs if "detrend" in s.figure_id]
    assert len(hist_specs) == n_layers
    assert len(detrend_specs) == n_layers

    # Total: 2 map + 4 plot = 6
    assert len(all_specs) == 3 * n_layers

    # All figure_ids and filenames are distinct
    assert len({s.figure_id for s in all_specs}) == 3 * n_layers
    assert len({s.filename for s in all_specs}) == 3 * n_layers

    # Histogram build(None) returns a real Figure
    hist_fig = hist_specs[0].build(None)
    assert hist_fig is not None
    assert len(hist_fig.axes) >= 1

    # Detrend build(None) returns a real Figure (3 titled image panels;
    # style=None adds 3 colorbar axes via fig.colorbar, total >= 3 axes)
    detrend_fig = detrend_specs[0].build(None)
    assert detrend_fig is not None
    titled = [ax for ax in detrend_fig.axes if ax.get_title()]
    assert len(titled) == 3

    # Per-layer distinctness: the two histogram specs must render DIFFERENT figures
    # (guards against late-binding regressions where both closures capture the same layer)
    hist_specs = [s for s in all_specs if "hist" in s.figure_id]
    t0 = hist_specs[0].build(None).axes[0].get_title()
    t1 = hist_specs[1].build(None).axes[0].get_title()
    assert t0 != t1, f"histogram titles should differ by layer name but both are {t0!r}"


def test_strain_detrend_spec_missing_maps_raises_file_not_found(tmp_path):
    """detrend build() raises FileNotFoundError when the source maps.h5 is absent."""
    vol = np.random.default_rng(6).random((1, 10, 10)) * 1e-3
    h5 = tmp_path / "stacked.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=vol)

    res = Strain.StrainResult(
        stacked_path=str(h5),
        volume_shape=vol.shape,
        output_dir=str(tmp_path),
        layers=[Strain.LayerResult("missing_layer", (10, 10), -1e-3, 1e-3, 0, 0)],
    )
    params = {
        "mode": "batch",
        "root_folder": str(tmp_path),  # missing_layer subfolder does not exist
        "maps_filename": "maps.h5",
        "ccmth_com_path": "/entry/ccmth/Center of mass/Center of mass",
        "roi": "",
    }
    specs = Strain.figures(res, params)
    detrend_specs = [s for s in specs if s.kind == "plot" and "detrend" in s.figure_id]
    assert len(detrend_specs) == 1
    with pytest.raises(FileNotFoundError, match="source maps.h5 not found"):
        detrend_specs[0].build(None)


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
    # one map spec + one histogram spec per Z layer (3 layers × 2 = 6 total)
    assert len(specs) == 6
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
    # 2 keys x 2 layers = 4 map specs + 4 histogram specs = 8 total, all distinct ids/filenames
    map_specs = [s for s in specs if s.kind == "map"]
    hist_specs = [s for s in specs if s.kind == "plot"]
    assert len(map_specs) == 4
    assert len(hist_specs) == 4
    assert len(specs) == 8
    assert len({s.figure_id for s in specs}) == 8
    assert len({s.filename for s in specs}) == 8


def test_mosaicity_catalog_histogram_specs_per_layer(tmp_path):
    """figures() emits one histogram plot spec per dataset key per Z layer."""
    n_z = 3
    vol = np.random.default_rng(42).random((n_z, 10, 20)).astype(np.float32)
    h5 = tmp_path / "stacked_volumes.h5"
    with h5py.File(h5, "w") as f:
        grp = f.require_group("chi")
        grp.create_dataset("Center of mass", data=vol)

    res = Mosaicity.MosaicityResult(
        stacked_path=str(h5),
        datasets={"/chi/Center of mass": vol.shape},
        layers=[f"layer{i}" for i in range(n_z)],
        skipped=[],
    )
    specs = Mosaicity.figures(res, {"pixel_size_x_um": 0.1, "pixel_size_y_um": 0.3})

    map_specs = [s for s in specs if s.kind == "map"]
    hist_specs = [s for s in specs if s.kind == "plot"]

    # n_z map specs + n_z histogram specs = 2*n_z total
    assert len(map_specs) == n_z
    assert len(hist_specs) == n_z
    assert len(specs) == 2 * n_z

    # All figure_ids and filenames are distinct (no collision between map and hist)
    assert len({s.figure_id for s in specs}) == 2 * n_z
    assert len({s.filename for s in specs}) == 2 * n_z

    # histogram ids contain "hist"; map ids do not
    assert all("hist" in s.figure_id for s in hist_specs)
    assert all("hist" not in s.figure_id for s in map_specs)

    # Histogram build(None) returns a real Figure with axes
    hist_fig = hist_specs[0].build(None)
    assert hist_fig is not None
    assert len(hist_fig.axes) >= 1

    # Per-layer distinctness: different layers must render different histogram titles
    t0 = hist_specs[0].build(None).axes[0].get_title()
    t1 = hist_specs[1].build(None).axes[0].get_title()
    assert t0 != t1, f"histogram titles should differ by layer but both are {t0!r}"

    # xlabel comes from _KEY_DISPLAY["/chi/Center of mass"] cbar_label = "Misorientation (°)"
    ax = hist_specs[0].build(None).axes[0]
    assert "°" in ax.get_xlabel() or "Misorientation" in ax.get_xlabel()


def test_plotting_build_histogram_importable():
    """build_histogram is importable from dfxm.common.plotting."""
    from dfxm.common.plotting import build_histogram

    data = np.random.default_rng(0).random((10, 10))
    fig = build_histogram(data, title="Test hist", xlabel="Values")
    assert fig is not None
    assert len(fig.axes) >= 1


def test_plotting_build_histogram_all_nan_returns_none():
    """build_histogram returns None when all values are NaN."""
    from dfxm.common.plotting import build_histogram

    data = np.full((5, 5), np.nan)
    assert build_histogram(data, title="t", xlabel="x") is None


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


# ---------------------------------------------------------------------------
# slices.figures() tests
# ---------------------------------------------------------------------------


def _make_oblique_slices_h5(tmp_path, n_planes=2, hu=8, wv=10):
    """Write a minimal oblique_slices.h5 mirroring write_volume_group's real layout.

    One volume group "strain" with all real attrs, one slice subgroup "z_sweep"
    with `slices` (n_planes, hu, wv), `u_um`, `v_um`, `offsets_um`.
    """
    h5 = tmp_path / "oblique_slices.h5"
    rng = np.random.default_rng(42)
    with h5py.File(h5, "w") as f:
        vg = f.create_group("strain")
        # attrs written by write_volume_group (lines 765-775 of slices.py)
        vg.attrs["kind"] = "strain"
        vg.attrs["dataset_path"] = "strain"
        vg.attrs["source_volume"] = "/fake/path.h5"
        vg.attrs["cbar_label"] = "Strain (ε)"
        vg.attrs["cmap"] = "RdBu_r"
        vg.attrs["vmin"] = float(-1e-3)
        vg.attrs["vmax"] = float(1e-3)
        vg.attrs["title"] = "Strain (cot method)"
        vg.attrs["scale_x_um_per_px"] = 0.152
        vg.attrs["scale_y_um_per_px"] = 0.385
        vg.attrs["scale_z_um_per_px"] = 1.0

        # one slice subgroup
        sg = vg.create_group("z_sweep")
        slices_data = rng.random((n_planes, hu, wv)).astype(np.float32)
        sg.create_dataset("slices", data=slices_data)
        sg.create_dataset("u_um", data=np.linspace(-40.0, 40.0, wv))
        sg.create_dataset("v_um", data=np.linspace(-30.0, 30.0, hu))
        sg.create_dataset("offsets_um", data=np.linspace(0.0, 10.0, n_planes))
        # subgroup attrs (from write_volume_group lines 784-788)
        sg.attrs["normal"] = np.array([0.0, 0.0, 1.0])
        sg.attrs["origin"] = np.array([0.0, 0.0, 0.0])
        sg.attrs["up"] = np.array([0.0, 1.0, 0.0])
        sg.attrs["u_hat"] = np.array([1.0, 0.0, 0.0])
        sg.attrs["v_hat"] = np.array([0.0, 1.0, 0.0])
        sg.attrs["n_hat"] = np.array([0.0, 0.0, 1.0])
        sg.attrs["half_u"] = 40.0
        sg.attrs["half_v"] = 30.0
        sg.attrs["du"] = 80.0 / wv
        sg.attrs["dv"] = 60.0 / hu
        sg.attrs["sweep_step_um"] = 5.0
        sg.attrs["n_planes"] = n_planes
    return str(h5)


def test_slices_catalog_two_planes(tmp_path):
    """figures() with 2 planes yields 2 map specs that build with equal aspect."""
    h5_path = _make_oblique_slices_h5(tmp_path, n_planes=2)
    res = Slices.SlicesResult(output_h5=h5_path)
    specs = Slices.figures(res, {})
    assert len(specs) == 2
    assert all(s.kind == "map" for s in specs)
    # distinct filenames
    filenames = [s.filename for s in specs]
    assert len(set(filenames)) == 2
    # equal aspect from build_slice_figure (aspect="equal")
    assert specs[0].build(None).axes[0].get_aspect() == 1.0


def test_slices_catalog_no_output_h5_returns_empty():
    """Guard: output_h5=None → empty list."""
    res = Slices.SlicesResult(output_h5=None)
    assert Slices.figures(res, {}) == []


def test_slices_catalog_distinct_ids_multi_plane(tmp_path):
    """All figure_ids and filenames are distinct across planes."""
    h5_path = _make_oblique_slices_h5(tmp_path, n_planes=3)
    res = Slices.SlicesResult(output_h5=h5_path)
    specs = Slices.figures(res, {})
    assert len(specs) == 3
    assert len({s.figure_id for s in specs}) == 3
    assert len({s.filename for s in specs}) == 3


def test_slices_catalog_via_figures_for(tmp_path):
    """Both import directions work: slices.figures() and figures_for('slices')."""
    h5_path = _make_oblique_slices_h5(tmp_path, n_planes=2)
    res = Slices.SlicesResult(output_h5=h5_path)
    direct = Slices.figures(res, {})
    via_catalog = figures.figures_for("slices", res, {})
    assert len(direct) == len(via_catalog) == 2


def _make_oblique_slices_h5_mosa_com(tmp_path, n_planes=2, hu=8, wv=10):
    """Write a minimal oblique_slices.h5 with kind='mosa_com' and asymmetric vmin/vmax.

    vmin=-0.5, vmax=1.0 → asymmetric straddle of zero → _make_norm returns TwoSlopeNorm.
    """
    h5 = tmp_path / "oblique_slices_mosa_com.h5"
    rng = np.random.default_rng(7)
    with h5py.File(h5, "w") as f:
        vg = f.create_group("mosa_com")
        vg.attrs["kind"] = "mosa_com"
        vg.attrs["dataset_path"] = "mosa_com/chi"
        vg.attrs["source_volume"] = "/fake/path.h5"
        vg.attrs["cbar_label"] = "Misorientation (°)"
        vg.attrs["cmap"] = "RdBu_r"
        vg.attrs["vmin"] = float(-0.5)
        vg.attrs["vmax"] = float(1.0)
        vg.attrs["title"] = "χ Misorientation"
        vg.attrs["scale_x_um_per_px"] = 0.152
        vg.attrs["scale_y_um_per_px"] = 0.385
        vg.attrs["scale_z_um_per_px"] = 1.0

        sg = vg.create_group("z_sweep")
        slices_data = rng.random((n_planes, hu, wv)).astype(np.float32)
        sg.create_dataset("slices", data=slices_data)
        sg.create_dataset("u_um", data=np.linspace(-40.0, 40.0, wv))
        sg.create_dataset("v_um", data=np.linspace(-30.0, 30.0, hu))
        sg.create_dataset("offsets_um", data=np.linspace(0.0, 10.0, n_planes))
        sg.attrs["normal"] = np.array([0.0, 0.0, 1.0])
        sg.attrs["origin"] = np.array([0.0, 0.0, 0.0])
        sg.attrs["up"] = np.array([0.0, 1.0, 0.0])
        sg.attrs["u_hat"] = np.array([1.0, 0.0, 0.0])
        sg.attrs["v_hat"] = np.array([0.0, 1.0, 0.0])
        sg.attrs["n_hat"] = np.array([0.0, 0.0, 1.0])
        sg.attrs["half_u"] = 40.0
        sg.attrs["half_v"] = 30.0
        sg.attrs["du"] = 80.0 / wv
        sg.attrs["dv"] = 60.0 / hu
        sg.attrs["sweep_step_um"] = 5.0
        sg.attrs["n_planes"] = n_planes
    return str(h5)


def test_slices_catalog_mosa_com_centered_norm(tmp_path):
    """mosa_com volume → TwoSlopeNorm (centered); strain volume → plain Normalize.

    Exercises the _CENTERED_KINDS branch end-to-end through figures() → build() →
    build_slice_figure() → _make_norm().
    """
    from matplotlib.colors import Normalize, TwoSlopeNorm

    # --- mosa_com path: asymmetric vmin/vmax straddle zero → TwoSlopeNorm ---
    h5_mosa = _make_oblique_slices_h5_mosa_com(tmp_path, n_planes=1)
    res_mosa = Slices.SlicesResult(output_h5=h5_mosa)
    specs_mosa = Slices.figures(res_mosa, {})
    assert len(specs_mosa) == 1
    fig_mosa = specs_mosa[0].build(None)
    norm_mosa = fig_mosa.axes[0].images[0].norm
    assert isinstance(norm_mosa, TwoSlopeNorm), (
        f"Expected TwoSlopeNorm for mosa_com, got {type(norm_mosa).__name__}"
    )

    # --- strain path: non-centered → plain Normalize, NOT TwoSlopeNorm ---
    strain_dir = tmp_path / "strain"
    strain_dir.mkdir()
    h5_strain = _make_oblique_slices_h5(strain_dir, n_planes=1)
    res_strain = Slices.SlicesResult(output_h5=h5_strain)
    specs_strain = Slices.figures(res_strain, {})
    assert len(specs_strain) == 1
    fig_strain = specs_strain[0].build(None)
    norm_strain = fig_strain.axes[0].images[0].norm
    assert not isinstance(norm_strain, TwoSlopeNorm), (
        f"Expected non-TwoSlopeNorm for strain, got {type(norm_strain).__name__}"
    )
    assert isinstance(norm_strain, Normalize)


# ---------------------------------------------------------------------------
# matched.figures() tests
# ---------------------------------------------------------------------------

_KNOWN_FRAME = np.random.default_rng(42).normal(size=(6, 8)).astype(np.float64)
# Expected output of _apply_shift_single(_KNOWN_FRAME, shift_px=0, pad_left=1, nx_new=10):
# 1 NaN column on the left, _KNOWN_FRAME in columns 1-8, 1 NaN column on the right.
_KNOWN_SHIFTED = np.concatenate(
    [np.full((6, 1), np.nan), _KNOWN_FRAME, np.full((6, 1), np.nan)], axis=1
)


def test_matched_figures_no_recorded_layers_returns_empty():
    """Guard: empty recorded list → empty figure catalog."""
    res = Matched.MatchedResult(recorded=[])
    assert Matched.figures(res, {}) == []


def test_matched_figures_one_layer_map_spec(monkeypatch, tmp_path):
    """figures() with one MatchedLayer yields one map spec that builds with equal aspect."""
    # Create a real (minimal) h5 file so the path check passes without FileNotFoundError
    h5 = tmp_path / "rock__1.h5"
    h5.write_bytes(b"")  # placeholder — load is monkeypatched

    layer = Matched.MatchedLayer(
        raw_h5=str(h5),
        pco_ff_path="1.1/measurement/pco_ff",
        frame_index=0,
        shift_px=0.0,
        pad_left=1,
        nx_new=10,
        ext_x=10 * 0.152,
        ext_y=6 * 0.385,
        vmin=0.0,
        vmax=50.0,
        title="Rocking Curve (frame 0, median-subtracted)\nZ = 0.00 µm (Layer 0/2)\nrock__1",
        colormap="gray",
        layer_index=0,
    )

    res = Matched.MatchedResult(recorded=[layer])

    # Monkeypatch load_pco_ff_frame to return the known array
    monkeypatch.setattr(Matched, "load_pco_ff_frame", lambda h5p, pco, fi: _KNOWN_FRAME.copy())

    specs = Matched.figures(res, {})
    assert len(specs) == 1
    assert specs[0].kind == "map"
    fig = specs[0].build(None)
    assert fig.axes[0].get_aspect() == 1.0
    # Pixel-fidelity: build() must reproduce the shifted frame exactly.
    arr = np.asarray(fig.axes[0].images[0].get_array())
    assert np.array_equal(arr, _KNOWN_SHIFTED, equal_nan=True)


def test_matched_figures_missing_raw_raises_file_not_found(monkeypatch, tmp_path):
    """build() must raise FileNotFoundError when the raw h5 file is gone."""
    missing_path = str(tmp_path / "gone.h5")
    layer = Matched.MatchedLayer(
        raw_h5=missing_path,
        pco_ff_path="1.1/measurement/pco_ff",
        frame_index=0,
        shift_px=0.0,
        pad_left=0,
        nx_new=8,
        ext_x=8 * 0.152,
        ext_y=6 * 0.385,
        vmin=0.0,
        vmax=50.0,
        title="t",
        colormap="gray",
        layer_index=0,
    )
    res = Matched.MatchedResult(recorded=[layer])
    specs = Matched.figures(res, {})
    assert len(specs) == 1
    with pytest.raises(FileNotFoundError):
        specs[0].build(None)


def test_matched_figures_distinct_ids_multi_layer(monkeypatch, tmp_path):
    """Multiple recorded layers → distinct figure_id and filename per layer."""
    h5 = tmp_path / "rock.h5"
    h5.write_bytes(b"")

    def _layer_obj(i):
        return Matched.MatchedLayer(
            raw_h5=str(h5),
            pco_ff_path="1.1/measurement/pco_ff",
            frame_index=0,
            shift_px=float(i),
            pad_left=0,
            nx_new=8,
            ext_x=8 * 0.152,
            ext_y=6 * 0.385,
            vmin=0.0,
            vmax=50.0,
            title=f"Layer {i}",
            colormap="gray",
            layer_index=i,
        )

    res = Matched.MatchedResult(recorded=[_layer_obj(i) for i in range(3)])
    monkeypatch.setattr(Matched, "load_pco_ff_frame", lambda h5p, pco, fi: _KNOWN_FRAME.copy())

    specs = Matched.figures(res, {})
    assert len(specs) == 3
    assert len({s.figure_id for s in specs}) == 3
    assert len({s.filename for s in specs}) == 3


def test_matched_figures_via_figures_for(monkeypatch, tmp_path):
    """Both import directions work: matched.figures() and figures_for('matched')."""
    h5 = tmp_path / "rock.h5"
    h5.write_bytes(b"")

    layer = Matched.MatchedLayer(
        raw_h5=str(h5),
        pco_ff_path="1.1/measurement/pco_ff",
        frame_index=0,
        shift_px=0.0,
        pad_left=0,
        nx_new=8,
        ext_x=8 * 0.152,
        ext_y=6 * 0.385,
        vmin=0.0,
        vmax=50.0,
        title="t",
        colormap="gray",
        layer_index=0,
    )
    res = Matched.MatchedResult(recorded=[layer])
    monkeypatch.setattr(Matched, "load_pco_ff_frame", lambda h5p, pco, fi: _KNOWN_FRAME.copy())

    direct = Matched.figures(res, {})
    via_catalog = figures.figures_for("matched", res, {})
    assert len(direct) == 1
    assert len(via_catalog) == 1


# ---------------------------------------------------------------------------
# profiles.figures() tests
# ---------------------------------------------------------------------------


def _write_profiles_h5(path, *, slice_name="oblique_full"):
    """Minimal oblique_slices.h5 mirroring what _collect() reads.

    Two volume groups sharing one slice subgroup (slice_name).
    Layout:
      /{vid}/             — attrs: kind, cbar_label, cmap, title, vmin, vmax,
                                   source_volume, dataset_path
      /{vid}/{slice_name}/slices      (3, 8, 10) float32
      /{vid}/{slice_name}/u_um        (10,)
      /{vid}/{slice_name}/v_um        (8,)
      /{vid}/{slice_name}/offsets_um  (3,)
    """
    rng = np.random.default_rng(77)
    u = np.linspace(-10.0, 10.0, 10)
    v = np.linspace(-8.0, 8.0, 8)
    offsets = np.array([-1.0, 0.0, 1.0])
    with h5py.File(path, "w") as f:
        for vid, cmap in (("raw_sum", "gray"), ("strain", "RdBu_r")):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cbar_label"] = "value"
            g.attrs["cmap"] = cmap
            g.attrs["title"] = vid
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            g.attrs["source_volume"] = ""
            g.attrs["dataset_path"] = vid
            sg = g.create_group(slice_name)
            slices_data = rng.random((3, 8, 10)).astype(np.float32)
            sg.create_dataset("slices", data=slices_data)
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offsets)
    return str(path)


def _profiles_job_spec(slice_name="oblique_full"):
    return {
        "name": slice_name,
        "offset_um": 0.0,
        "start_uv": [-5.0, -3.0],
        "end_uv": [5.0, 3.0],
        "n_samples": 20,
        "width_pixels": 1,
        "fig_name": f"profile_{slice_name}",
    }


def _profiles_result(h5_path, slice_name="oblique_full"):
    jr = Profiles.ProfileJobResult(
        name=slice_name,
        offset_used_um=0.0,
        figure=h5_path.replace(".h5", ".png"),
        fields=["raw_sum", "strain"],
    )
    return Profiles.ProfilesResult(output_dir="", mode="parameter", jobs=[jr])


def test_profiles_figures_no_h5_path_returns_empty(tmp_path):
    """Guard: consolidated_h5='' → empty list."""
    res = _profiles_result(str(tmp_path / "oblique_slices.h5"))
    assert Profiles.figures(res, {}) == []
    assert Profiles.figures(res, {"consolidated_h5": ""}) == []


def test_profiles_figures_no_jobs_with_fields_returns_empty(tmp_path):
    """Guard: no parameter-mode jobs (no fields) → empty list."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    # preview-mode job has fields=[]
    jr = Profiles.ProfileJobResult(name="oblique_full", offset_used_um=0.0, fields=[])
    res = Profiles.ProfilesResult(output_dir="", mode="preview", jobs=[jr])
    specs = Profiles.figures(res, {"consolidated_h5": str(h5)})
    assert specs == []


def test_profiles_catalog_one_spec_per_job(tmp_path):
    """figures() returns one kind='plot' FigureSpec per parameter-mode job."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    import json

    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {
        "consolidated_h5": str(h5),
        "jobs_json": json.dumps([job]),
    }
    specs = Profiles.figures(res, params)
    # 1 companion + 1 trace per field (raw_sum, strain), both toggles default on
    assert len(specs) == 3
    ids = {s.figure_id for s in specs}
    assert "profiles_oblique_full" in ids
    assert "profiles_oblique_full__trace__raw_sum" in ids
    assert "profiles_oblique_full__trace__strain" in ids
    assert all(s.kind == "plot" for s in specs)


def test_profiles_catalog_distinct_ids_multi_job(tmp_path):
    """Two jobs → two specs with distinct figure_id and filename."""
    import json

    h5 = tmp_path / "oblique_slices.h5"
    # Write two slice subgroups
    rng = np.random.default_rng(88)
    u = np.linspace(-10.0, 10.0, 10)
    v = np.linspace(-8.0, 8.0, 8)
    offsets = np.array([-1.0, 0.0, 1.0])
    with h5py.File(h5, "w") as f:
        for vid, cmap in (("raw_sum", "gray"), ("strain", "RdBu_r")):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cbar_label"] = "value"
            g.attrs["cmap"] = cmap
            g.attrs["title"] = vid
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            g.attrs["source_volume"] = ""
            g.attrs["dataset_path"] = vid
            for sname in ("slice_a", "slice_b"):
                sg = g.create_group(sname)
                sg.create_dataset("slices", data=rng.random((3, 8, 10)).astype(np.float32))
                sg.create_dataset("u_um", data=u)
                sg.create_dataset("v_um", data=v)
                sg.create_dataset("offsets_um", data=offsets)

    jr_a = Profiles.ProfileJobResult(name="slice_a", offset_used_um=0.0, fields=["raw_sum"])
    jr_b = Profiles.ProfileJobResult(name="slice_b", offset_used_um=0.0, fields=["raw_sum"])
    res = Profiles.ProfilesResult(output_dir="", mode="parameter", jobs=[jr_a, jr_b])
    job_a = {**_profiles_job_spec("slice_a"), "fig_name": "fig_a"}
    job_b = {**_profiles_job_spec("slice_b"), "fig_name": "fig_b"}
    params = {
        "consolidated_h5": str(h5),
        "jobs_json": json.dumps([job_a, job_b]),
    }
    specs = Profiles.figures(res, params)
    assert len(specs) == 4  # 2 jobs × (companion + 1 trace)
    assert len({s.figure_id for s in specs}) == 4
    assert len({s.filename for s in specs}) == 4


def test_profiles_catalog_build_returns_figure(tmp_path):
    """spec.build(None) returns a matplotlib Figure (the companion profile figure)."""
    import json

    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {
        "consolidated_h5": str(h5),
        "jobs_json": json.dumps([job]),
    }
    specs = Profiles.figures(res, params)
    assert len(specs) == 3  # companion (specs[0]) + one trace per field
    from matplotlib.figure import Figure

    fig = specs[0].build(None)
    assert isinstance(fig, Figure)
    # Image panel (ax_img) + N trace panels (one per field)
    assert len(fig.axes) >= 2


def test_profiles_catalog_traces_off_only_companion(tmp_path):
    """save_traces=False → only the companion spec per job."""
    import json

    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {"consolidated_h5": str(h5), "jobs_json": json.dumps([job]), "save_traces": False}
    specs = Profiles.figures(res, params)
    assert len(specs) == 1
    assert specs[0].figure_id == "profiles_oblique_full"


def test_profiles_catalog_companion_off_only_traces(tmp_path):
    """save_companion=False → only the per-field trace specs."""
    import json

    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {"consolidated_h5": str(h5), "jobs_json": json.dumps([job]), "save_companion": False}
    specs = Profiles.figures(res, params)
    assert len(specs) == 2
    assert all("__trace__" in s.figure_id for s in specs)


def test_profiles_catalog_trace_build_returns_single_axes_figure(tmp_path):
    """A trace spec's build(None) returns a single-axes Figure."""
    import json

    from matplotlib.figure import Figure

    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {"consolidated_h5": str(h5), "jobs_json": json.dumps([job]), "save_companion": False}
    specs = Profiles.figures(res, params)
    fig = specs[0].build(None)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1  # one trace panel, no colorbar


def test_profiles_catalog_build_missing_h5_raises(tmp_path):
    """build() raises FileNotFoundError when consolidated_h5 is absent."""
    import json

    missing = str(tmp_path / "gone.h5")
    job = _profiles_job_spec()
    jr = Profiles.ProfileJobResult(
        name="oblique_full", offset_used_um=0.0, fields=["raw_sum", "strain"]
    )
    res = Profiles.ProfilesResult(output_dir="", mode="parameter", jobs=[jr])
    params = {
        "consolidated_h5": missing,
        "jobs_json": json.dumps([job]),
    }
    specs = Profiles.figures(res, params)
    assert len(specs) == 3  # companion (specs[0]) + one trace per field
    with pytest.raises(FileNotFoundError):
        specs[0].build(None)


def test_profiles_figures_via_figures_for(tmp_path):
    """Both import directions work: profiles.figures() and figures_for('profiles')."""
    import json

    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    job = _profiles_job_spec()
    res = _profiles_result(str(h5))
    params = {
        "consolidated_h5": str(h5),
        "jobs_json": json.dumps([job]),
    }
    direct = Profiles.figures(res, params)
    via_catalog = figures.figures_for("profiles", res, params)
    assert len(direct) == 3  # companion + one trace per field
    assert len(via_catalog) == 3


def test_profiles_build_companion_figure_returns_figure():
    """build_companion_figure returns a Figure (unit test of the refactored builder)."""
    rng = np.random.default_rng(0)
    u = np.linspace(-10.0, 10.0, 20)
    v = np.linspace(-8.0, 8.0, 16)
    plane = rng.random((16, 20)).astype(np.float64)
    attrs = {
        "kind": "raw_sum",
        "cbar_label": "value",
        "cmap": "gray",
        "title": "Raw sum",
        "source_volume": "",
        "dataset_path": "",
        "vmin": 0.0,
        "vmax": 1.0,
    }
    geom = Profiles.line_geometry(u, v, (-5.0, -3.0), (5.0, 3.0), 20, 1, Profiles.grid_pitch(u, v))
    vm, vs, _ = Profiles.profile_plane(plane, geom)
    fields = [{"vid": "raw_sum", "attrs": attrs, "value_mean": vm, "value_std": vs}]
    ref = (plane, u, v, attrs, "raw_sum @ 0.000 µm")
    from matplotlib.figure import Figure

    fig = Profiles.build_companion_figure(ref, fields, geom, "cyan")
    assert isinstance(fig, Figure)
    # 1 image panel + 1 trace panel
    assert len(fig.axes) >= 2


def _companion_fixture():
    """Return (ref, fields, geom) for a minimal 1-field companion figure."""
    rng = np.random.default_rng(42)
    u = np.linspace(-10.0, 10.0, 20)
    v = np.linspace(-8.0, 8.0, 16)
    plane = rng.random((16, 20)).astype(np.float64)
    attrs = {
        "kind": "raw_sum",
        "cbar_label": "value",
        "cmap": "gray",
        "title": "Raw sum",
        "source_volume": "",
        "dataset_path": "",
        "vmin": 0.0,
        "vmax": 1.0,
    }
    geom = Profiles.line_geometry(u, v, (-5.0, -3.0), (5.0, 3.0), 20, 1, Profiles.grid_pitch(u, v))
    vm, vs, _ = Profiles.profile_plane(plane, geom)
    fields = [{"vid": "raw_sum", "attrs": attrs, "value_mean": vm, "value_std": vs}]
    ref = (plane, u, v, attrs, "raw_sum @ 0.000 µm")
    return ref, fields, geom


def test_profiles_colorbar_font_not_double_scaled():
    """FIX 1: styled colorbar fonts must be scaled once (10*fs), not twice (10*fs*fs).

    build_companion_figure(style=PlotStyle(font_scale=2.0)) must produce a
    colorbar whose label fontsize is 10*2.0 == 20, not 10*2.0*2.0 == 40.
    """
    from dfxm.common.plotting import PlotStyle

    fs = 2.0
    ref, fields, geom = _companion_fixture()
    fig = Profiles.build_companion_figure(ref, fields, geom, "cyan", style=PlotStyle(font_scale=fs))

    # matplotlib labels the colorbar axes '<colorbar>' internally.
    colorbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    content_axes = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]

    # There must be exactly one colorbar axes (style.colorbar=True by default).
    assert len(colorbar_axes) == 1, f"expected 1 colorbar axes, got {len(colorbar_axes)}"
    cb_ax = colorbar_axes[0]

    # Colorbar label fontsize must be 10*fs (set once by add_colorbar), NOT 10*fs*fs.
    expected_label_fs = 10 * fs
    actual_label_fs = cb_ax.yaxis.label.get_fontsize()
    assert actual_label_fs == pytest.approx(expected_label_fs, rel=0.05), (
        f"colorbar label fontsize {actual_label_fs:.1f} != expected {expected_label_fs:.1f} "
        f"(double-scaling would give {10 * fs * fs:.1f})"
    )

    # Colorbar tick fontsize must be 9*fs (set once by add_colorbar).
    expected_tick_fs = 9 * fs
    tick_labels = cb_ax.yaxis.get_ticklabels()
    if tick_labels:
        actual_tick_fs = tick_labels[0].get_fontsize()
        assert actual_tick_fs == pytest.approx(expected_tick_fs, rel=0.05), (
            f"colorbar tick fontsize {actual_tick_fs:.1f} != expected {expected_tick_fs:.1f}"
        )

    # Content axes (image + trace) must also be scaled exactly once.
    assert len(content_axes) >= 2, "expected at least image + 1 trace axes"


def test_profiles_build_missing_job_spec_raises_value_error(tmp_path):
    """FIX 2: build() must raise ValueError (not FileNotFoundError) when job spec is absent.

    The consolidated h5 exists; what's missing is the job spec entry in jobs_json
    (params mismatch, not a missing file).
    """
    h5 = tmp_path / "oblique_slices.h5"
    _write_profiles_h5(h5)
    jr = Profiles.ProfileJobResult(
        name="oblique_full", offset_used_um=0.0, fields=["raw_sum", "strain"]
    )
    res = Profiles.ProfilesResult(output_dir="", mode="parameter", jobs=[jr])
    # Omit jobs_json entirely so job_spec_by_name is empty → _job is None.
    params = {"consolidated_h5": str(h5)}
    specs = Profiles.figures(res, params)
    assert len(specs) == 3  # companion (specs[0]) + one trace per field
    with pytest.raises(ValueError, match="job spec for"):
        specs[0].build(None)


def test_profiles_colorbar_hidden_when_style_colorbar_false():
    """FIX 3: styled path with colorbar=False must produce no colorbar axes."""
    from dfxm.common.plotting import PlotStyle

    ref, fields, geom = _companion_fixture()
    fig = Profiles.build_companion_figure(
        ref, fields, geom, "cyan", style=PlotStyle(colorbar=False)
    )
    colorbar_axes = [ax for ax in fig.axes if ax.get_label() == "<colorbar>"]
    assert len(colorbar_axes) == 0, (
        f"expected no colorbar axes with colorbar=False, got {len(colorbar_axes)}"
    )


def test_layer_figure_threads_group_to_arbitrary_units():
    import numpy as np

    from dfxm.common.plotting import PlotStyle
    from dfxm.common.render import layer_figure

    style = PlotStyle(tickfmt_raw="arb")
    fig, ax, im = layer_figure(
        np.arange(100).reshape(10, 10).astype(float),
        0.0,
        99.0,
        "gray",
        50.0,
        30.0,
        "Raw",
        "Intensity",
        style=style,
        group="raw",
    )
    cbar_ax = fig.axes[1]
    assert list(cbar_ax.get_yticks()) == []  # raw+arb -> no numeric ticks
    assert cbar_ax.get_ylabel() == "Intensity (arb. units)"


def test_volume_layer_specs_cmap_group_resolves_from_style(tmp_path):
    import h5py
    import numpy as np

    from dfxm.common.figures import volume_layer_specs
    from dfxm.common.plotting import PlotStyle

    p = tmp_path / "v.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("vol", data=np.random.default_rng(0).random((2, 4, 5)))
    common = dict(
        h5_path=str(p), dataset="vol", title="T", cbar_label="c", sx=1.0, sy=1.0, vmin=0.0, vmax=1.0
    )
    specs = volume_layer_specs(id_prefix="t", cmap="magma", cmap_group="raw", **common)
    fig = specs[0].build(PlotStyle(cmap_raw="viridis"))
    assert fig.axes[0].images[0].cmap.name == "viridis"
    # no group -> fixed cmap wins regardless of style
    specs2 = volume_layer_specs(id_prefix="t2", cmap="bone", **common)
    fig2 = specs2[0].build(PlotStyle(cmap_raw="viridis"))
    assert fig2.axes[0].images[0].cmap.name == "bone"


def test_save_layer_pngs_accepts_style(tmp_path):
    import os

    import numpy as np

    from dfxm.common import render
    from dfxm.common.plotting import PlotStyle

    vol = np.zeros((1, 4, 5))
    d = render.save_layer_pngs(
        vol,
        [0.0],
        str(tmp_path),
        "x",
        0,
        1,
        "gray",
        "t",
        "c",
        1.0,
        1.0,
        style=PlotStyle(font_scale=3.0),
    )
    assert os.path.exists(os.path.join(d, "layer_0000.png"))


def test_mosaicity_map_specs_resolve_cmap_groups(tmp_path):
    from dfxm.common.plotting import PlotStyle

    vol = np.random.rand(1, 8, 12).astype(np.float32)
    h5 = tmp_path / "stacked_volumes.h5"
    with h5py.File(h5, "w") as f:
        grp = f.require_group("chi")
        grp.create_dataset("Center of mass", data=vol)
        grp.create_dataset("FWHM", data=vol)
    res = Mosaicity.MosaicityResult(
        stacked_path=str(h5),
        datasets={"/chi/Center of mass": vol.shape, "/chi/FWHM": vol.shape},
        layers=["l0"],
        skipped=[],
    )
    specs = Mosaicity.figures(res, {"pixel_size_x_um": 0.1, "pixel_size_y_um": 0.3})
    com = next(s for s in specs if s.kind == "map" and s.filename.startswith("chi_com"))
    fwhm = next(s for s in specs if s.kind == "map" and s.filename.startswith("chi_fwhm"))
    assert com.build(None).axes[0].images[0].cmap.name == "fast"
    assert fwhm.build(None).axes[0].images[0].cmap.name == "magma"
    assert com.build(PlotStyle(cmap_mosa_com="plasma")).axes[0].images[0].cmap.name == "plasma"


def test_save_layer_pngs_forwards_group(tmp_path, monkeypatch):
    import numpy as np

    from dfxm.common import render as R
    from dfxm.common.plotting import PlotStyle

    captured = {}
    real = R.layer_figure

    def spy(*a, **k):
        captured["group"] = k.get("group")
        return real(*a, **k)

    monkeypatch.setattr(R, "layer_figure", spy)
    vol = np.arange(2 * 4 * 5, dtype=float).reshape(2, 4, 5)
    R.save_layer_pngs(
        vol,
        [0.0, 1.0],
        str(tmp_path),
        "raw_sum_intensity",
        0.0,
        float(vol.max()),
        "gray",
        "Raw",
        "Intensity (a.u.)",
        1.0,
        1.0,
        style=PlotStyle(tickfmt_raw="arb"),
        group="raw",
    )
    assert captured["group"] == "raw"
