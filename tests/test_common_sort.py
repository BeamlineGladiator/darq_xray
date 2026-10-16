"""Tests for darq_xray.common.sort."""

from __future__ import annotations

import os

from darq_xray.common.sort import drop_plot_dirs, find_matching_folders, natural_sort_key


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


def test_drop_plot_dirs_keeps_layers_and_drops_plot_folders(tmp_path):
    def folder(name, *files):
        d = tmp_path / name
        d.mkdir()
        for f in files:
            (d / f).write_text("")
        return str(d)

    layer = folder("layer__1", "maps.h5")
    layer_with_png = folder("layer__2", "maps.h5", "layer__2_strain.png")
    no_maps = folder("layer__3")  # a broken layer: the stage reports it, so keep it
    named = folder("plots")  # named by the caller: dropped even while empty
    earlier = folder("old_plots", "layer__1_strain.png")  # recognised by its contents
    gone = str(tmp_path / "vanished")  # listdir fails: keep, let the stage report it
    # the output folder pointed at a real layer: the maps file wins, the layer stays
    named_layer = folder("layer__4", "maps.h5")

    kept = drop_plot_dirs(
        [layer, layer_with_png, no_maps, named, earlier, gone, named_layer],
        plot_dirs=[named, None, "", named_layer],
        maps_filename="maps.h5",
        markers=("_strain.png",),
    )
    assert kept == [layer, layer_with_png, no_maps, gone, named_layer]
