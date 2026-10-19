"""Tests for darq_xray.common.sort."""

from __future__ import annotations

import os

from darq_xray.common.sort import (
    APP_OUTPUT_DIRS,
    drop_output_dirs,
    find_matching_folders,
    natural_sort_key,
    resolve_layer_work,
)


def test_natural_sort_orders_numbers_numerically():
    items = ["x__10", "x__2", "x__1", "x__21", "x__3"]
    assert sorted(items, key=natural_sort_key) == ["x__1", "x__2", "x__3", "x__10", "x__21"]


def test_natural_sort_is_case_insensitive():
    assert sorted(["B", "a", "C"], key=natural_sort_key) == ["a", "B", "C"]


def test_find_matching_folders_returns_dirs_in_natural_order(tmp_path):
    for name in ["layer__10", "layer__2", "layer__1", "other"]:
        (tmp_path / name).mkdir()
    # a file that matches the glob must be ignored (dirs only)
    (tmp_path / "layer__99.txt").write_text("not a dir")

    found = find_matching_folders(str(tmp_path), "layer__*")
    assert [os.path.basename(f) for f in found] == ["layer__1", "layer__2", "layer__10"]


def test_find_matching_folders_empty_when_no_match(tmp_path):
    assert find_matching_folders(str(tmp_path), "nope__*") == []


def test_drop_output_dirs_keeps_layers_and_drops_the_apps_folders(tmp_path):
    def folder(name, *files):
        d = tmp_path / name
        d.mkdir()
        for f in files:
            (d / f).write_text("")
        return str(d)

    layer = folder("layer__1", "maps.h5")
    layer_with_png = folder("layer__2", "maps.h5", "layer__2_strain.png")
    no_maps = folder("layer__3")  # a broken layer: the stage reports it, so keep it
    app = [folder(name) for name in APP_OUTPUT_DIRS]  # by name, even while empty
    custom = folder("plots")  # the configured Output dir: by name, even while empty
    earlier = folder("old_plots", "layer__1_strain.png")  # recognised by its contents
    gone = str(tmp_path / "vanished")  # listdir fails: keep, let the stage report it

    kept = drop_output_dirs(
        [layer, layer_with_png, no_maps, *app, custom, earlier, gone],
        {"output_dir": custom},
        str(tmp_path),
    )
    assert kept == [layer, layer_with_png, no_maps, gone]


def test_drop_output_dirs_keeps_a_layer_the_output_dir_points_at(tmp_path):
    layer = tmp_path / "layer__1"
    layer.mkdir()
    (layer / "maps.h5").write_text("")
    assert drop_output_dirs([str(layer)], {"output_dir": str(layer)}, str(tmp_path)) == [str(layer)]


def test_resolve_layer_work_skips_the_apps_folders(tmp_path):
    """Filtered before the maps filename is joined on, so a maps filename with a
    folder in it still lines up with the run's own layer list."""
    for name in ("layer__1", "aligned_volume_visualizations"):
        (tmp_path / name / "sub").mkdir(parents=True)
    (tmp_path / "layer__1" / "sub" / "maps.h5").write_text("")
    params = {"mode": "batch", "root_folder": str(tmp_path), "folder_pattern": "*"}
    work = resolve_layer_work(
        {**params, "maps_filename": "sub/maps.h5"}, maps_filename="sub/maps.h5"
    )
    assert work == [str(tmp_path / "layer__1" / "sub" / "maps.h5")]
