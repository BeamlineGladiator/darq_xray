"""Panel adapters: pure (source, roi) loaders + draw-into-axes dispatch (Qt-free).

Loaders are pure functions of (PanelSource, roi) so the GUI can cache results;
heavy deps (h5py, stage modules) are imported inside functions. A loader never
raises for missing DATA (file/dataset/field gone at render time) — it returns
a ``kind="placeholder"`` PanelData with the reason. A malformed SELECTOR
(a required key absent from ``selector``, checked before any h5 access)
raises :class:`~dfxm.common.errors.StageUserError` instead — that is a
recipe-authoring bug, not a data-availability issue.

Selector shapes (per ``PanelSource.kind``):

- ``map_layer``: ``{"stage": "strain"|"mosaicity"|"rocking", "dataset": str,
  "z": int, "sx": float?, "sy": float?}`` — for ``stage="strain"`` the dataset
  is fixed to ``"strain"`` and ``sx``/``sy`` default from the file attrs
  ``scale_x_um``/``scale_y_um``; for ``mosaicity``/``rocking`` ``sx``/``sy``
  are required inputs (the GUI/recipe author supplies them, matching the
  replot defaults 0.152/0.385 when omitted).
- ``slice_plane``: ``{"volume_id": str, "slice_name": str, "plane": int}``.
- ``profiles_ref``: ``{"job": dict, "field": str | None}`` (``None`` picks the
  job's reference field).
- ``profiles_trace``: ``{"job": dict, "field": str}``.
- ``image``: ``{}`` — no selector; ``h5_path`` is the PNG/JPEG/TIFF file itself.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from ..common.errors import StageUserError
from ..common.figures import crop_roi_2d, data_bbox_roi
from ..common.plotting import GROUP_BY_KIND, symmetric_limits
from .recipe import PANEL_KINDS, PanelDef


@dataclass
class PanelData:
    kind: str  # PANEL_KINDS value or "placeholder"
    ext_x_um: float | None = None  # sizing input for maps
    ext_y_um: float | None = None
    length_um: float | None = None  # sizing input for traces
    group: str | None = None  # quantity group (shared-colorbar compatibility)
    vmin: float | None = None  # default colour limits (pre-override)
    vmax: float | None = None
    payload: dict = field(default_factory=dict)  # kind-specific draw inputs


def _cache_key(panel: PanelDef) -> str:
    src = panel.source
    return json.dumps(
        [
            src.h5_path,
            src.kind,
            src.selector,
            list(panel.roi) if panel.roi else None,
            bool(panel.crop_to_data),
        ],
        sort_keys=True,
        default=str,
    )


def load_panel(panel: PanelDef, *, cache: dict | None = None) -> PanelData:
    """Read one panel's data from its source h5, applying its ROI crop.

    Never raises for a missing file/dataset/field — those become a
    ``kind="placeholder"`` :class:`PanelData` with ``payload["reason"]``
    describing why. Only a malformed selector (unknown kind, bad ``stage``)
    raises :class:`StageUserError` — those are recipe-authoring bugs, not
    data-availability issues.
    """
    if panel.source.kind not in PANEL_KINDS:
        raise StageUserError(
            f"panel {panel.id!r}: unknown kind {panel.source.kind!r}",
            hint=f"Valid kinds: {', '.join(PANEL_KINDS)}.",
        )
    key = _cache_key(panel)
    if cache is not None and key in cache:
        return cache[key]
    loader = _LOADERS[panel.source.kind]
    try:
        data = loader(
            panel.source.h5_path,
            panel.source.selector,
            panel.roi,
            crop_to_data=bool(panel.crop_to_data),
        )
    except StageUserError:
        raise
    except Exception as exc:  # noqa: BLE001 — the composition survives partial data
        data = PanelData(kind="placeholder", payload={"reason": f"{panel.source.h5_path}: {exc}"})
    if cache is not None:
        cache[key] = data
    return data


def _load_map_layer(h5_path, sel, roi, *, crop_to_data=False):
    import h5py

    stage = sel.get("stage")
    if stage not in ("strain", "mosaicity", "rocking"):
        raise StageUserError(
            f"map_layer selector needs stage strain/mosaicity/rocking (got {stage!r})",
            hint='Set selector["stage"], e.g. {"stage": "mosaicity", "dataset": ..., "z": 0}.',
        )
    if stage != "strain" and "dataset" not in sel:
        raise StageUserError(
            f"map_layer selector for stage={stage!r} needs a dataset",
            hint='Set selector["dataset"] to the HDF5 dataset path, '
            f'e.g. {{"stage": {stage!r}, "dataset": "/chi/Center of mass", "z": 0}}.',
        )
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"{h5_path} not found")
    with h5py.File(h5_path, "r") as f:
        if stage == "strain":
            dset = f["strain"]
            sx = float(sel.get("sx") or f.attrs.get("scale_x_um", 0.152))
            sy = float(sel.get("sy") or f.attrs.get("scale_y_um", 0.385))
            title, cbar, group, cmap = "Strain map (cot method)", "Strain (ε)", "strain", "RdBu_r"
        else:
            dset = f[sel["dataset"]]
            sx, sy = float(sel.get("sx", 0.152)), float(sel.get("sy", 0.385))
            if stage == "mosaicity":
                from ..stages.mosaicity import _KEY_DISPLAY, _streamed_clim

                group, cbar, title = _KEY_DISPLAY.get(sel["dataset"], (None, "(°)", sel["dataset"]))
                cmap = "magma"
            else:
                from ..stages.rocking import _DATASET_DISPLAY, _replot_default_clim

                title, cbar = _DATASET_DISPLAY.get(sel["dataset"], (sel["dataset"], "Intensity"))
                group, cmap = "raw", "gray"
        z = int(sel.get("z", 0))
        layer = dset[z][...]
        if stage == "mosaicity":
            vmin, vmax = _streamed_clim(dset)
        elif stage == "rocking":
            vmin, vmax = _replot_default_clim(dset, {}, None)
        else:
            vmin, vmax = symmetric_limits(layer)
    roi = _effective_roi(layer, roi, crop_to_data)
    layer = crop_roi_2d(layer, roi)
    if layer is None:
        raise ValueError(f"ROI {roi} crops to an empty layer")
    return PanelData(
        kind="map_layer",
        ext_x_um=layer.shape[1] * sx,
        ext_y_um=layer.shape[0] * sy,
        group=group,
        vmin=float(vmin),
        vmax=float(vmax),
        payload={"layer": layer, "sx": sx, "sy": sy, "title": title, "cbar": cbar, "cmap": cmap},
    )


def _load_slice_plane(h5_path, sel, roi, *, crop_to_data=False):
    import h5py

    from ..common.plotting import resolve_cmap

    missing = [k for k in ("volume_id", "slice_name") if k not in sel]
    if missing:
        raise StageUserError(
            f"slice_plane selector missing {', '.join(missing)}",
            hint='Set selector["volume_id"] and selector["slice_name"], '
            'e.g. {"volume_id": "strain", "slice_name": "obl", "plane": 0}.',
        )
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"{h5_path} not found")
    vid, sname = sel["volume_id"], sel["slice_name"]
    k = int(sel.get("plane", 0))
    with h5py.File(h5_path, "r") as f:
        vg = f[vid]
        kind = str(vg.attrs.get("kind", ""))
        prep = {
            "cmap_name": str(vg.attrs.get("cmap", "magma")),
            "title": str(vg.attrs.get("title", vid)),
            "cbar_label": str(vg.attrs.get("cbar_label", "")),
            "vmin": float(vg.attrs.get("vmin", 0.0)),
            "vmax": float(vg.attrs.get("vmax", 1.0)),
        }
        sg = vg[sname]
        s2d = sg["slices"][k]
        u = sg["u_um"][:]
        v = sg["v_um"][:]
        off = float(sg["offsets_um"][k])
    s2d, u, v = _crop_uv(s2d, u, v, _effective_roi(s2d, roi, crop_to_data))
    group = GROUP_BY_KIND.get(kind)
    prep["group"] = group
    prep["cmap_name"] = resolve_cmap(None, group, fallback=prep["cmap_name"])
    return PanelData(
        kind="slice_plane",
        ext_x_um=float(u[-1] - u[0]),
        ext_y_um=float(v[-1] - v[0]),
        group=group,
        vmin=prep["vmin"],
        vmax=prep["vmax"],
        payload={"prep": prep, "sname": sname, "plane2d": s2d, "u": u, "v": v, "offset_um": off},
    )


def _effective_roi(plane, roi, crop_to_data):
    """The ROI to crop *plane* with: ``roi`` as-is, or — when *crop_to_data* —
    the finite-data bounding box (``data_bbox_roi``, +3 % margin) searched
    inside ``roi``; falls back to ``roi`` when the plane has no finite data."""
    if not crop_to_data:
        return roi
    box = data_bbox_roi(plane, roi)
    return roi if box is None else box


def _crop_uv(plane, u, v, roi):
    """Crop *plane* (and its ``u``/``v`` axis arrays) to *roi*, clamping
    out-of-range indices to the plane's own bounds. ``roi is None`` is a
    no-op. Shared by ``_load_slice_plane`` and ``_load_profiles_ref`` — the
    two adapters that read a 2-D plane straight out of an h5 dataset."""
    if roi is None:
        return plane, u, v
    cropped = crop_roi_2d(plane, roi)
    if cropped is None:
        raise ValueError(f"ROI {roi} crops to an empty plane")
    r0, r1, c0, c1 = roi
    h, w = plane.shape[:2]
    r0 = max(0, min(int(r0), h))
    r1 = max(0, min(int(r1), h))
    c0 = max(0, min(int(c0), w))
    c1 = max(0, min(int(c1), w))
    return cropped, u[c0:c1], v[r0:r1]


def _load_profiles_ref(h5_path, sel, roi, *, crop_to_data=False):
    import h5py

    from ..stages import profiles

    if "job" not in sel:
        raise StageUserError(
            "profiles_ref selector needs a job",
            hint='Set selector["job"] to the profiles job dict, e.g. {"job": {...}, "field": None}.',
        )
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"{h5_path} not found")
    job = sel["job"]
    field_id = sel.get("field")
    p = profiles.STAGE.defaults()
    with h5py.File(h5_path, "r") as f:
        ref, fields, geom, off_used, _dropped = profiles._collect(
            f, job, p, ref_pref="", restrict=None
        )
        ref_plane, u_um, v_um, ref_attrs, ref_label = ref
        if field_id is not None:
            fld = next((fl for fl in fields if fl["vid"] == field_id), None)
            if fld is None:
                raise ValueError(f"field {field_id!r} not present in job {job.get('name')!r}")
            plane, attrs = fld["plane"], fld["attrs"]
        else:
            plane, attrs = ref_plane, ref_attrs
    plane, u, v = _crop_uv(plane, u_um, v_um, _effective_roi(plane, roi, crop_to_data))
    color = profiles.auto_line_color(attrs["cmap"], None)
    return PanelData(
        kind="profiles_ref",
        ext_x_um=float(u[-1] - u[0]),
        ext_y_um=float(v[-1] - v[0]),
        group=GROUP_BY_KIND.get(attrs.get("kind")),
        vmin=attrs["vmin"],
        vmax=attrs["vmax"],
        payload={"plane": plane, "u": u, "v": v, "attrs": attrs, "geom": geom, "line_color": color},
    )


def _load_profiles_trace(h5_path, sel, roi, *, crop_to_data=False):
    import h5py

    from ..stages import profiles

    missing = [k for k in ("job", "field") if k not in sel]
    if missing:
        raise StageUserError(
            f"profiles_trace selector missing {', '.join(missing)}",
            hint='Set selector["job"] and selector["field"], '
            'e.g. {"job": {...}, "field": "strain"}.',
        )
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"{h5_path} not found")
    job = sel["job"]
    field_id = sel["field"]
    p = profiles.STAGE.defaults()
    with h5py.File(h5_path, "r") as f:
        _ref, fields, geom, _off_used, _dropped = profiles._collect(
            f, job, p, ref_pref="", restrict=None
        )
    fld = next((fl for fl in fields if fl["vid"] == field_id), None)
    if fld is None:
        raise ValueError(f"field {field_id!r} not present in job {job.get('name')!r}")
    return PanelData(
        kind="profiles_trace",
        length_um=float(geom["L"]),
        group=GROUP_BY_KIND.get(fld["attrs"].get("kind")),
        payload={"fld": fld, "geom": geom},
    )


_TRACE_LW_PER_FONT = 1.8  # default trace linewidth (pt) per unit of font scale


def resolve_trace_opts(compose, style) -> dict:
    """Trace look for ``profiles_trace`` panels: ``{"linewidth", "color", "font_scale"}``.

    Fonts follow ``style.font_scale`` unless ``compose.trace_font_scale`` is set;
    the default linewidth is 1.8 pt x that (effective) font scale so one Font
    scale knob keeps maps and traces typographically consistent;
    ``compose.trace_linewidth`` (pt, absolute) and ``compose.trace_color``
    (``""`` = matplotlib ``C0``) override.
    """
    fs = compose.trace_font_scale
    if fs is None:
        fs = float(style.font_scale) if style is not None else 1.0
    fs = float(fs)
    lw = compose.trace_linewidth
    lw = _TRACE_LW_PER_FONT * fs if lw is None else float(lw)
    return {"linewidth": lw, "color": compose.trace_color or None, "font_scale": fs}


def draw_panel(
    ax,
    panel,
    data,
    style,
    *,
    cax=None,
    colorbar=None,
    scale_bar=None,
    fixed_scale_um_per_cm=None,
    show_xlabel=True,
    show_title=False,
    trace_opts=None,
):
    """Draw *data* into *ax* with the panel's overrides applied.

    Titles are OFF by default in composed figures (``panel.show_title`` or
    ``show_title=True`` re-enables). Returns the drawn ``AxesImage`` for
    maps/slices/refs, or ``None`` for traces and placeholders. *trace_opts*
    is :func:`resolve_trace_opts`'s dict for trace panels (``None`` = derive
    from a default :class:`ComposeStyle` and *style*).
    """
    if data.kind == "placeholder":
        draw_placeholder(ax, data.payload["reason"])
        return None
    if data.kind == "image":
        img = data.payload["image"]
        # the cell already has the image's aspect, so "auto" fills it edge to
        # edge; no title, colourbar or scale bar for an external image
        ax.imshow(img, interpolation="none", aspect="auto", cmap="gray" if img.ndim == 2 else None)
        ax.set_axis_off()
        return None
    from ..common.plotting import resolve_cmap

    titled = panel.show_title if panel.show_title is not None else show_title
    vmin, vmax = data.vmin, data.vmax
    if panel.clim is not None:
        lo, hi = panel.clim
        vmin = float(lo) if lo is not None else vmin
        vmax = float(hi) if hi is not None else vmax
    if data.kind == "map_layer":
        from ..common.render import draw_map_layer

        pay = data.payload
        cmap = panel.cmap or resolve_cmap(style, data.group, fallback=pay["cmap"])
        return draw_map_layer(
            ax,
            pay["layer"],
            vmin,
            vmax,
            cmap,
            data.ext_x_um,
            data.ext_y_um,
            pay["title"] if titled else "",
            pay["cbar"],
            style=style,
            group=data.group,
            cax=cax,
            colorbar=colorbar,
            scale_bar=scale_bar,
            fixed_scale_um_per_cm=fixed_scale_um_per_cm,
        )
    if data.kind == "slice_plane":
        from ..stages.slices import draw_slice_axes

        pay = data.payload
        prep = dict(pay["prep"], vmin=vmin, vmax=vmax)
        if panel.cmap:
            prep["cmap_name"] = panel.cmap
        else:
            prep["cmap_name"] = resolve_cmap(style, data.group, fallback=prep["cmap_name"])
        im = draw_slice_axes(
            ax,
            prep,
            {"name": pay["sname"]},
            pay["plane2d"],
            pay["u"],
            pay["v"],
            offset_um=pay["offset_um"],
            style=style,
            cax=cax,
            colorbar=colorbar,
            scale_bar=scale_bar,
            fixed_scale_um_per_cm=fixed_scale_um_per_cm,
        )
        if not titled:
            ax.set_title("")
        return im
    if data.kind == "profiles_ref":
        from ..common.plotting import add_colorbar
        from ..stages.profiles import draw_reference_axes

        pay = data.payload
        attrs = dict(pay["attrs"], vmin=vmin, vmax=vmax)
        if panel.cmap:
            attrs["cmap"] = panel.cmap
        im = draw_reference_axes(
            ax,
            pay["plane"],
            pay["u"],
            pay["v"],
            attrs,
            pay["line_color"],
            geom=pay["geom"],
            title=(attrs["title"] if titled else None),
            style=style,
            fixed_scale_um_per_cm=fixed_scale_um_per_cm,
            scale_bar=scale_bar,
        )
        want_cbar = (
            (style.colorbar if style is not None else True) if colorbar is None else colorbar
        )
        if want_cbar and style is not None:
            add_colorbar(
                ax.get_figure(), im, ax, attrs["cbar_label"], style, group=data.group, cax=cax
            )
        if style is not None:
            # same styling the standalone reference figure gets (font scale,
            # axes_mode) — the composer skipped both before 2026-08-18
            from ..common.plotting import apply_axes_mode, apply_text_scale

            apply_text_scale(ax, style)
            apply_axes_mode(ax, style)
        return im
    if data.kind == "profiles_trace":
        from ..stages.profiles import draw_trace_axes

        pay = data.payload
        if trace_opts is None:
            from .recipe import ComposeStyle

            trace_opts = resolve_trace_opts(ComposeStyle(), style)
        draw_trace_axes(
            ax,
            pay["fld"],
            pay["geom"],
            linewidth=trace_opts["linewidth"],
            color=trace_opts["color"],
            font_scale=trace_opts["font_scale"],
            style=style,
            show_xlabel=show_xlabel,
        )
        if not titled:
            ax.set_title("")
        if not show_xlabel:
            ax.tick_params(labelbottom=False)
        if not panel.y_tick_labels:
            # numbers only: tick marks, grid and the y-label stay
            ax.tick_params(labelleft=False)
            ax.yaxis.get_offset_text().set_visible(False)
            for t in ax.texts:  # apply_axis_tickfmt's static "scientific" x10^n
                if t.get_gid() == "tickfmt-exponent-y":
                    t.set_visible(False)
        return None
    raise StageUserError(f"unknown panel kind {data.kind!r}", hint="Recipe is corrupt.")


def panel_preview(panel: PanelDef):
    """Full-frame ``(array2d, sx_um, sy_um)`` of an image panel for the ROI
    picker — the panel's own ``roi``/``crop_to_data`` are deliberately ignored
    so the picker always shows the whole frame in the same pixel coordinates
    the ROI is expressed in. Raises ``ValueError`` for trace panels (no ROI)
    and for unavailable data (the placeholder reason)."""
    if panel.source.kind == "profiles_trace":
        raise ValueError("a trace panel has no ROI to pick")
    if panel.source.kind == "image":
        raise ValueError("an image panel's ROI is a pixel crop — type it in the ROI box")
    full = PanelDef(panel.id, panel.source)  # roi=None, crop_to_data=False
    data = load_panel(full)
    if data.kind == "placeholder":
        raise ValueError(str(data.payload.get("reason", "data unavailable")))
    pay = data.payload
    if data.kind == "map_layer":
        return pay["layer"], float(pay["sx"]), float(pay["sy"])

    def _step(axis):
        axis = [float(v) for v in axis]
        return abs(axis[1] - axis[0]) if len(axis) > 1 else 1.0

    if data.kind == "slice_plane":
        return pay["plane2d"], _step(pay["u"]), _step(pay["v"])
    if data.kind == "profiles_ref":
        return pay["plane"], _step(pay["u"]), _step(pay["v"])
    raise ValueError(f"no preview for panel kind {data.kind!r}")


def draw_placeholder(ax, reason: str) -> None:
    """Hatched grey cell for a panel whose data is unavailable — never a crash."""
    from matplotlib.patches import Rectangle

    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(
        Rectangle(
            (0, 0), 1, 1, transform=ax.transAxes, facecolor="0.92", edgecolor="0.6", hatch="///"
        )
    )
    ax.text(
        0.5,
        0.5,
        "unavailable",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        color="0.35",
    )


def _load_image(path, sel, roi, *, crop_to_data=False) -> PanelData:
    """An external raster (PNG/JPEG/TIFF) as a lettered panel.

    ``ext_x_um``/``ext_y_um`` carry PIXELS — used only for the aspect ratio by
    ``layout._image_cell``, never as a physical scale. ``roi`` is a plain
    ``(r0, r1, c0, c1)`` pixel crop; ``crop_to_data`` is ignored. A missing or
    undecodable file raises and becomes a placeholder in ``load_panel``.
    """
    import numpy as np
    from matplotlib.image import imread

    arr = np.asarray(imread(path))
    if arr.dtype.kind in "ui":
        arr = arr.astype("f4") / float(np.iinfo(arr.dtype).max)
    cropped = crop_roi_2d(arr, roi)
    if cropped is None or cropped.shape[0] == 0 or cropped.shape[1] == 0:
        raise ValueError(f"ROI {tuple(roi)} leaves no pixels")
    h, w = cropped.shape[:2]
    return PanelData(kind="image", ext_x_um=float(w), ext_y_um=float(h), payload={"image": cropped})


_LOADERS = {
    "map_layer": _load_map_layer,
    "slice_plane": _load_slice_plane,
    "profiles_ref": _load_profiles_ref,
    "profiles_trace": _load_profiles_trace,
    "image": _load_image,
}
