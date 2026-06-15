import subprocess
import sys

import h5py
import numpy as np
import pytest

from dfxm.common import figures, render
from dfxm.common.figures import volume_layer_specs
from dfxm.common.plotting import PlotStyle
from dfxm.stages import mosaicity as Mosaicity
from dfxm.stages import strain as Strain
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
