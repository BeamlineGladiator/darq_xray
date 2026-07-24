"""Panel adapters: pure loaders + draw-into-axes dispatch — dfxm.compose.adapters."""

import h5py
import numpy as np
from matplotlib.figure import Figure

from dfxm.compose.adapters import draw_panel, load_panel
from dfxm.compose.recipe import PanelDef, PanelSource


def _write_mosa(path):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("/chi/Center of mass", data=rng.normal(size=(2, 6, 8)).astype("f4"))
    return str(path)


def _write_strain(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("strain", data=np.linspace(-2e-4, 2e-4, 2 * 6 * 8).reshape(2, 6, 8))
        f.attrs["scale_x_um"] = 0.2
        f.attrs["scale_y_um"] = 0.4
    return str(path)


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


def test_load_map_layer_mosaicity_extents_and_group(tmp_path):
    h5 = _write_mosa(tmp_path / "stack.h5")
    p = PanelDef(
        "m",
        PanelSource(
            h5,
            "map_layer",
            {"stage": "mosaicity", "dataset": "/chi/Center of mass", "z": 1, "sx": 0.5, "sy": 0.25},
        ),
    )
    d = load_panel(p)
    assert d.kind == "map_layer" and d.group == "mosa_com"
    assert d.ext_x_um == 8 * 0.5 and d.ext_y_um == 6 * 0.25
    assert d.vmin is not None and d.vmax is not None and d.vmin < d.vmax


def test_load_map_layer_strain_defaults_from_attrs_and_symmetric(tmp_path):
    h5 = _write_strain(tmp_path / "strain.h5")
    p = PanelDef("s", PanelSource(h5, "map_layer", {"stage": "strain", "z": 0}))
    d = load_panel(p)
    assert d.group == "strain"
    assert d.ext_x_um == 8 * 0.2 and d.ext_y_um == 6 * 0.4
    assert d.vmin == -d.vmax  # symmetric limits


def test_load_map_layer_roi_crops_extent(tmp_path):
    h5 = _write_mosa(tmp_path / "stack.h5")
    p = PanelDef(
        "m",
        PanelSource(
            h5,
            "map_layer",
            {"stage": "mosaicity", "dataset": "/chi/Center of mass", "z": 0, "sx": 1.0, "sy": 1.0},
        ),
        roi=(1, 4, 2, 6),
    )
    d = load_panel(p)
    assert d.ext_x_um == 4.0 and d.ext_y_um == 3.0


def test_load_slice_plane_and_profiles(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    sp = load_panel(
        PanelDef(
            "p",
            PanelSource(
                h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}
            ),
        )
    )
    assert sp.kind == "slice_plane" and sp.group == "strain"
    assert sp.ext_x_um == 20.0 and sp.ext_y_um == 16.0
    ref = load_panel(PanelDef("r", PanelSource(h5, "profiles_ref", {"job": JOB, "field": None})))
    assert ref.kind == "profiles_ref" and ref.ext_x_um == 20.0
    tr = load_panel(
        PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    )
    assert tr.kind == "profiles_trace"
    assert abs(tr.length_um - np.hypot(10.0, 6.0)) < 1e-9


def test_missing_file_and_missing_key_become_placeholders(tmp_path):
    d = load_panel(
        PanelDef("x", PanelSource(str(tmp_path / "gone.h5"), "map_layer", {"stage": "strain"}))
    )
    assert d.kind == "placeholder" and "gone.h5" in d.payload["reason"]
    h5 = _write_mosa(tmp_path / "stack.h5")
    d2 = load_panel(
        PanelDef(
            "y",
            PanelSource(
                h5,
                "map_layer",
                {"stage": "mosaicity", "dataset": "/nope", "z": 0, "sx": 1, "sy": 1},
            ),
        )
    )
    assert d2.kind == "placeholder"


def test_loader_cache_hit_skips_reread(tmp_path):
    h5 = _write_strain(tmp_path / "strain.h5")
    p = PanelDef("s", PanelSource(h5, "map_layer", {"stage": "strain", "z": 0}))
    cache = {}
    d1 = load_panel(p, cache=cache)
    import os

    os.remove(h5)
    d2 = load_panel(p, cache=cache)  # served from cache, file gone
    assert d2 is d1


def test_draw_panel_dispatch(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    fig = Figure(figsize=(6, 4))
    for sel_kind, sel in (
        ("slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
        ("profiles_ref", {"job": JOB, "field": None}),
        ("profiles_trace", {"job": JOB, "field": "strain"}),
    ):
        ax = fig.add_subplot(111)
        p = PanelDef("p", PanelSource(h5, sel_kind, sel))
        d = load_panel(p)
        draw_panel(ax, p, d, None, colorbar=False, scale_bar=False)
        if sel_kind == "profiles_trace":
            assert ax.lines  # the profile curve
        else:
            assert ax.images
        fig.clear()


def test_draw_placeholder_hatched(tmp_path):
    from dfxm.compose.adapters import draw_placeholder

    fig = Figure()
    ax = fig.add_subplot(111)
    draw_placeholder(ax, "missing file")
    assert any(p.get_hatch() for p in ax.patches)
    assert ax.get_xticks().size == 0
