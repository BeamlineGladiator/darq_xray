"""Spec acceptance figures — built headless from synthetic h5 fixtures."""

import h5py
import numpy as np

from dfxm.common.plotting import measured_box_in
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    PanelSource,
    Row,
)
from dfxm.compose.render import render_recipe


def _write_slices_two_planes(path):
    """oblique_slices.h5: mosa CoM + strain volumes, two single-plane slices."""
    u = np.linspace(-10.0, 10.0, 41)
    v = np.linspace(-8.0, 8.0, 33)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (
            ("mosa_com_chi", "mosa_com", "magma"),
            ("strain", "strain", "RdBu_r"),
        ):
            g = f.create_group(vid)
            g.attrs.update(
                kind=kind, cbar_label="value", cmap=cmap, title=vid, vmin=-10.0, vmax=10.0
            )
            for sname in ("slice_a", "slice_b"):
                sg = g.create_group(sname)
                sg.create_dataset("slices", data=(uu + vv)[None, ...].astype("f4"))
                sg.create_dataset("u_um", data=u)
                sg.create_dataset("v_um", data=v)
                sg.create_dataset("offsets_um", data=np.array([0.0]))
    return str(path)


def _write_profiles_three_fields(path):
    """oblique_slices.h5 with three fields on one slice (for figure 2)."""
    u = np.linspace(-20.0, 20.0, 81)
    v = np.linspace(-15.0, 15.0, 61)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (
            ("mosa_com_chi", "mosa_com", "magma"),
            ("strain", "strain", "RdBu_r"),
            ("raw_mosa_sum", "raw_mosa_sum", "gray"),
        ):
            g = f.create_group(vid)
            g.attrs.update(
                kind=kind, cbar_label="value", cmap=cmap, title=vid, vmin=-40.0, vmax=40.0
            )
            sg = g.create_group("obl")
            sg.create_dataset("slices", data=(uu + vv)[None, ...].astype("f4"))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))
    return str(path)


def test_acceptance_figure_1_two_by_two_grid(tmp_path):
    """2×2: cols = mosaicity CoM | strain, rows = two slices; labels A B / C D."""
    h5 = _write_slices_two_planes(tmp_path / "obl.h5")

    def sp(pid, vid, sname):
        return PanelDef(
            pid,
            PanelSource(h5, "slice_plane", {"volume_id": vid, "slice_name": sname, "plane": 0}),
        )

    scale = 10.0
    recipe = FigureRecipe(
        "fig1",
        {"scale_um_per_cm": scale, "show_title": False},
        ComposeStyle(label_template="A"),
        Col(
            [
                Row([PanelRef("p00"), PanelRef("p01")]),
                Row([PanelRef("p10"), PanelRef("p11")]),
            ]
        ),
        [
            sp("p00", "mosa_com_chi", "slice_a"),
            sp("p01", "strain", "slice_a"),
            sp("p10", "mosa_com_chi", "slice_b"),
            sp("p11", "strain", "slice_b"),
        ],
    )
    res = render_recipe(recipe)
    assert res.n_rendered == 4
    # one µm/cm on every map: every box exactly 20/scale × 16/scale cm
    for pid in ("p00", "p01", "p10", "p11"):
        w, h = measured_box_in(res.figure, res.axes_by_id[pid])
        assert abs(w - 20.0 / scale / 2.54) < 0.005 * w, pid
        assert abs(h - 16.0 / scale / 2.54) < 0.005 * h, pid
    # row-major labels A B / C D
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert {"A", "B", "C", "D"} <= set(texts)
    # grid alignment: row-mates share y, column-mates share x
    pos = {pid: res.axes_by_id[pid].get_position() for pid in res.axes_by_id}
    assert abs(pos["p00"].y0 - pos["p01"].y0) < 1e-6
    assert abs(pos["p10"].y0 - pos["p11"].y0) < 1e-6
    assert abs(pos["p00"].x0 - pos["p10"].x0) < 1e-6
    assert abs(pos["p01"].x0 - pos["p11"].x0) < 1e-6
    assert not any("scale is off" in n for n in res.notes)


def test_acceptance_figure_2_ragged_dual_scale(tmp_path):
    """Ragged 3 columns; maps at scale_um_per_cm, trace stacks at a DIFFERENT
    trace_scale_um_per_cm — both honoured exactly in one canvas."""
    h5 = _write_profiles_three_fields(tmp_path / "obl.h5")
    job_a = {"name": "obl", "offset_um": 0.0, "start_uv": [-8.0, -6.0], "end_uv": [8.0, 6.0]}
    job_b = {"name": "obl", "offset_um": 0.0, "start_uv": [-15.0, -9.0], "end_uv": [15.0, 9.0]}
    len_a = float(np.hypot(16.0, 12.0))  # 20 µm
    len_b = float(np.hypot(30.0, 18.0))  # ~34.99 µm (B longer than A)
    fields = ["mosa_com_chi", "strain", "raw_mosa_sum"]

    panels = [
        PanelDef(
            "A1",
            PanelSource(h5, "profiles_ref", {"job": job_a, "field": "mosa_com_chi"}),
            roi=(10, 50, 10, 60),
            label="A1",
        ),
        PanelDef(
            "B1",
            PanelSource(h5, "profiles_ref", {"job": job_b, "field": "mosa_com_chi"}),
            roi=(10, 50, 10, 60),
            label="B1",
        ),
    ]
    for tag, job in (("a", job_a), ("b", job_b)):
        for vid in fields:
            panels.append(
                PanelDef(
                    f"t_{tag}_{vid}",
                    PanelSource(h5, "profiles_trace", {"job": job, "field": vid}),
                )
            )

    map_scale, trace_scale, trace_h_cm = 5.0, 2.0, 2.0
    assert map_scale != trace_scale  # the spec's hard requirement
    recipe = FigureRecipe(
        "fig2",
        {
            "scale_um_per_cm": map_scale,
            "trace_scale_um_per_cm": trace_scale,
            "trace_height_cm": trace_h_cm,
            "show_title": False,
        },
        ComposeStyle(),
        Row(
            [
                Col([PanelRef("A1"), PanelRef("B1")]),
                Col([PanelRef(f"t_a_{v}") for v in fields], shared_x=True, group_label="A2"),
                Col([PanelRef(f"t_b_{v}") for v in fields], shared_x=True, group_label="B2"),
            ]
        ),
        panels,
    )
    res = render_recipe(recipe)
    assert res.n_rendered == 8
    fig = res.figure

    # maps honour the MAP scale exactly. u pitch 40/80=0.5, v pitch 30/60=0.5;
    # profiles_ref/slice_plane extent is the axis-endpoint difference (u[-1]-u[0]),
    # not count*pitch (that's the map_layer convention) — a 50-col crop (cols
    # 10:60 of 81) spans 49 pitches = 24.5 µm, a 40-row crop (rows 10:50 of 61)
    # spans 39 pitches = 19.5 µm.
    ext_x, ext_y = 24.5, 19.5
    for pid in ("A1", "B1"):
        w, h = measured_box_in(fig, res.axes_by_id[pid])
        assert abs(w - ext_x / map_scale / 2.54) < 0.005 * w, pid
        assert abs(h - ext_y / map_scale / 2.54) < 0.005 * h, pid

    # traces honour the TRACE scale exactly, height = trace_height_cm
    for tag, length in (("a", len_a), ("b", len_b)):
        for vid in fields:
            w, h = measured_box_in(fig, res.axes_by_id[f"t_{tag}_{vid}"])
            assert abs(w - length / trace_scale / 2.54) < 0.005 * w, (tag, vid)
            assert abs(h - trace_h_cm / 2.54) < 0.005 * h, (tag, vid)

    # both scales in ONE canvas, asserted against the same figure object
    wa, _ = measured_box_in(fig, res.axes_by_id["t_a_strain"])
    wm, _ = measured_box_in(fig, res.axes_by_id["A1"])
    assert abs(wa * 2.54 * trace_scale - len_a) < 0.005 * len_a
    assert abs(wm * 2.54 * map_scale - ext_x) < 0.005 * ext_x

    # group labels A2/B2 present; no titles anywhere
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "A2" in texts and "B2" in texts
    assert all(ax.get_title() == "" for ax in res.axes_by_id.values())

    # shared distance axis: bottom-only x labels within each stack
    for tag in ("a", "b"):
        for vid in fields[:-1]:
            assert res.axes_by_id[f"t_{tag}_{vid}"].get_xlabel() == ""
        assert res.axes_by_id[f"t_{tag}_{fields[-1]}"].get_xlabel() != ""

    # within each stack: identical box width/height and left-aligned x0
    for tag in ("a", "b"):
        xs = {round(res.axes_by_id[f"t_{tag}_{v}"].get_position().x0, 5) for v in fields}
        assert len(xs) == 1, tag

    # ragged padding: A stack narrower than B stack, yet column 3 starts at one x
    # for all B panels (column edges align — the A column envelope pads the gap)
    xa = max(res.axes_by_id[f"t_a_{v}"].get_position().x1 for v in fields)
    xb = {round(res.axes_by_id[f"t_b_{v}"].get_position().x0, 5) for v in fields}
    assert len(xb) == 1 and min(xb) > xa

    assert not any("scale is off" in n for n in res.notes)
