"""Headless CLI — python3 -m dfxm.compose render."""

import os

import h5py
import numpy as np

from dfxm.compose.__main__ import _main
from dfxm.compose.recipe import (
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    PanelSource,
    Row,
    recipe_to_json,
)


def _write_obl(path):
    u = np.linspace(-10.0, 10.0, 41)
    v = np.linspace(-8.0, 8.0, 33)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(path, "w") as f:
        for vid, kind in (("raw_sum", "raw_sum"), ("strain", "strain")):
            g = f.create_group(vid)
            g.attrs.update(
                kind=kind, cbar_label="value", cmap="gray", title=vid, vmin=-10.0, vmax=10.0
            )
            sg = g.create_group("obl")
            sg.create_dataset("slices", data=(uu + vv)[None, ...].astype("f4"))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))
    return str(path)


JOB = {"name": "obl", "offset_um": 0.0, "start_uv": [-5.0, -3.0], "end_uv": [5.0, 3.0]}


def _two_panel_recipe(h5, **style):
    p1 = PanelDef(
        "a",
        PanelSource(h5, "slice_plane", {"volume_id": "raw_sum", "slice_name": "obl", "plane": 0}),
    )
    p2 = PanelDef(
        "b",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
    )
    return FigureRecipe(
        "demo",
        {"scale_um_per_cm": 10.0, "show_title": False, **style},
        ComposeStyle(),
        Row([PanelRef("a"), PanelRef("b")]),
        [p1, p2],
    )


def test_cli_renders_and_exits_zero(tmp_path, capsys):
    h5 = _write_obl(tmp_path / "obl.h5")
    rp = tmp_path / "r.json"
    rp.write_text(recipe_to_json(_two_panel_recipe(h5)))
    out = tmp_path / "out"
    rc = _main(["render", str(rp), "-o", str(out), "--formats", "png"])
    assert rc == 0
    assert os.path.exists(out / "demo.png")


def test_cli_all_placeholders_exits_one(tmp_path, capsys):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    for p in r.panels:
        p.source.h5_path = str(tmp_path / "gone.h5")
    rp = tmp_path / "r.json"
    rp.write_text(recipe_to_json(r))
    rc = _main(["render", str(rp), "-o", str(tmp_path / "out")])
    assert rc == 1
    assert "placeholder" in capsys.readouterr().out


def test_cli_bad_recipe_exits_two(tmp_path, capsys):
    rp = tmp_path / "bad.json"
    rp.write_text("{not json")
    rc = _main(["render", str(rp), "-o", str(tmp_path / "out")])
    assert rc == 2
    assert "hint" in capsys.readouterr().err.lower()
