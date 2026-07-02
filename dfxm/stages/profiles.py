"""Line profiles stage — 1D profiles across oblique-slice planes.

Faithful port of ``line_profile_oblique_slices_v2.py`` (headless modes). It reads
the consolidated ``oblique_slices.h5`` written by the slices stage and profiles a
straight line (optionally a band of parallel lines) across one plane, for every
scalar field stored for that slice — so intensity, strain and misorientation are
profiled at the SAME in-plane positions.

Modes:
* ``parameter`` — the reproducible publication run from committed line coords
  (``jobs_json``): one companion figure + CSV(s) + per-field overview per job.
* ``preview`` — render each job's reference plane (with the line if given) to
  confirm the plane/line before committing.

The legacy TkAgg "interactive" click-picker is a GUI concern (the embedded
mpl canvas), not part of this headless stage; pick coordinates there, then run
``parameter`` here.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from scipy.ndimage import map_coordinates

from ..common import render as Rnd
from ..common.errors import StageUserError
from ..common.figures import FigureSpec, register
from ..common.plotting import PlotStyle, add_colorbar, apply_text_scale, style_from_params
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


_DEFAULT_JOBS = json.dumps(
    [
        {
            "name": "oblique_full",
            "offset_um": 0.0,
            "start_uv": [-50.0, -50.0],
            "end_uv": [50.0, 50.0],
            "n_samples": None,
            "width_pixels": 1,
            "fig_name": "profile_oblique_full_0um",
        }
    ],
    indent=2,
)

_LIGHT_MIDDLE = {
    "RdBu_r",
    "RdBu",
    "coolwarm",
    "bwr",
    "seismic",
    "PuOr",
    "PuOr_r",
    "BrBG",
    "BrBG_r",
    "PiYG",
    "Spectral",
    "fast",
}

STAGE = StageSpec(
    name="profiles",
    label="Line profiles",
    description=(
        "Draws 1-D line profiles across a slice plane — every field is sampled at the same "
        "in-plane positions, so intensity, strain and misorientation line up point by point. "
        "Writes a stacked figure plus CSVs. Use 'Pick line…' to choose the line by clicking "
        "on the plane."
    ),
    params=(
        Param(
            "consolidated_h5",
            ParamType.PATH,
            "Slices file",
            must_exist=True,
            help="The oblique_slices.h5 file written by the slices stage.",
        ),
        Param(
            "mode",
            ParamType.ENUM,
            "Mode",
            default="parameter",
            choices=("parameter", "preview"),
            help=(
                "'parameter' runs the jobs below and saves figures/CSVs (reproducible); "
                "'preview' just displays the plane so you can inspect it."
            ),
        ),
        Param(
            "reference_volume_id",
            ParamType.STR,
            "Reference field",
            default="",
            advanced=True,
            group="Selection",
            help=(
                "Which field is shown as the top image of the figure "
                "(blank = raw_sum if present, else the first field)."
            ),
        ),
        Param(
            "volume_ids",
            ParamType.STR,
            "Fields",
            default="",
            advanced=True,
            group="Selection",
            help="Comma-separated field ids to profile, in this order (blank = all fields).",
        ),
        Param(
            "jobs_json",
            ParamType.TEXT,
            "Jobs (JSON)",
            default=_DEFAULT_JOBS,
            help=(
                "JSON list of profile jobs: slice name, plane offset, line start/end in µm "
                "('start_uv'/'end_uv'), and band width in pixels. Easiest filled by 'Pick line…'."
            ),
        ),
        Param(
            "save_csv",
            ParamType.BOOL,
            "Save CSV",
            default=True,
            advanced=True,
            group="Output",
            help="Write one CSV per profiled field.",
        ),
        Param(
            "save_overview",
            ParamType.BOOL,
            "Save overviews",
            default=True,
            advanced=True,
            group="Output",
            help="Write per-field overview images with the profile line drawn on the plane.",
        ),
        Param(
            "line_color",
            ParamType.STR,
            "Line colour",
            default="",
            advanced=True,
            group="Appearance",
            help=(
                "Colour of the profile line drawn on the overview images "
                "(blank = automatic per colormap)."
            ),
        ),
        Param(
            "geom_tol_um",
            ParamType.FLOAT,
            "Geometry tol",
            unit="µm",
            default=1e-4,
            advanced=True,
            group="Matching",
            help=(
                "Maximum allowed geometry mismatch between fields sharing a plane, in µm — "
                "guards against profiling mis-registered slices."
            ),
        ),
        Param(
            "offset_tol_um",
            ParamType.FLOAT,
            "Offset tol",
            unit="µm",
            default=1e-3,
            advanced=True,
            group="Matching",
            help="Tolerance when matching the requested plane offset to the stored planes, in µm.",
        ),
        Param(
            "fig_dpi",
            ParamType.INT,
            "Figure DPI",
            default=200,
            advanced=True,
            group="Appearance",
            help="Resolution of the saved figures, in dots per inch.",
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help="Where the figures and CSVs are written (blank = next to the slices file).",
        ),
    ),
)


@dataclass
class ProfileJobResult:
    name: str
    offset_used_um: float
    figure: str | None = None
    csvs: list[str] = field(default_factory=list)
    overviews: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)


@dataclass
class ProfilesResult:
    output_dir: str = ""
    mode: str = "parameter"
    jobs: list[ProfileJobResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Profiling core (pure; faithful port — covered by the self-test)
# -----------------------------------------------------------------------------
def grid_pitch(u_um, v_um) -> float:
    du = abs(u_um[1] - u_um[0]) if len(u_um) > 1 else 1.0
    dv = abs(v_um[1] - v_um[0]) if len(v_um) > 1 else 1.0
    return float(min(du, dv))


def line_geometry(u_um, v_um, start_uv, end_uv, n_samples, width_pixels, pitch):
    """Sampling geometry for one line (and its band) in the (u, v) frame."""
    start = np.asarray(start_uv, dtype=np.float64)
    end = np.asarray(end_uv, dtype=np.float64)
    d = end - start
    length = float(np.hypot(d[0], d[1]))
    if length < 1e-9:
        raise ValueError(f"start_uv {tuple(start)} and end_uv {tuple(end)} coincide")
    dhat = d / length
    phat = np.array([-dhat[1], dhat[0]])
    if n_samples is None:
        n_samples = int(np.round(length / pitch)) + 1
    n_samples = max(2, int(n_samples))
    distance = np.linspace(0.0, length, n_samples)
    w = max(1, int(width_pixels))
    band_offsets = (np.arange(w) - (w - 1) / 2.0) * pitch
    du_step = (u_um[1] - u_um[0]) if len(u_um) > 1 else 1.0
    dv_step = (v_um[1] - v_um[0]) if len(v_um) > 1 else 1.0
    base = start[None, :] + distance[:, None] * dhat[None, :]
    pts = base[None, :, :] + band_offsets[:, None, None] * phat[None, None, :]
    iu = (pts[..., 0] - u_um[0]) / du_step
    iv = (pts[..., 1] - v_um[0]) / dv_step
    return {
        "distance": distance,
        "IV": iv,
        "IU": iu,
        "start": start,
        "end": end,
        "dhat": dhat,
        "phat": phat,
        "L": length,
        "band_offsets": band_offsets,
        "width": w,
    }


def sample_nan_aware(arr2d, iv, iu):
    """NaN-aware bilinear sampling: NaN wherever the stencil touched NaN/OOB."""
    finite = np.isfinite(arr2d)
    a0 = np.where(finite, arr2d, 0.0).astype(np.float64)
    wt = finite.astype(np.float64)
    num = map_coordinates(a0, np.stack([iv, iu]), order=1, mode="constant", cval=0.0)
    den = map_coordinates(wt, np.stack([iv, iu]), order=1, mode="constant", cval=0.0)
    out = np.full(num.shape, np.nan, dtype=np.float64)
    good = den > 1.0 - 1e-6
    out[good] = num[good] / den[good]
    return out


def profile_plane(plane2d, geom):
    """Profile one plane. Returns (value_mean, value_std_or_None, n_valid)."""
    import warnings

    w, ns = geom["IV"].shape
    sampled = sample_nan_aware(plane2d, geom["IV"].ravel(), geom["IU"].ravel()).reshape(w, ns)
    # all-NaN columns (a line fully outside the data) are expected; nanmean/nanstd
    # warn "Mean of empty slice" / "Degrees of freedom <= 0" there -> NaN, fine.
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        value_mean = np.nanmean(sampled, axis=0)
        value_std = None if w == 1 else np.nanstd(sampled, axis=0)
    n_valid = np.isfinite(sampled).sum(axis=0)
    return value_mean, value_std, n_valid


# -----------------------------------------------------------------------------
# HDF5 access
# -----------------------------------------------------------------------------
def _as_str(v, default=""):
    if v is None:
        return default
    return v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)


def list_volume_ids(f):
    return [k for k in f.keys() if isinstance(f[k], h5py.Group)]


def volume_ids_with_slice(f, slice_name):
    out = []
    for vid in list_volume_ids(f):
        g = f[vid]
        if slice_name in g and isinstance(g[slice_name], h5py.Group) and "slices" in g[slice_name]:
            out.append(vid)
    return sorted(out)


def read_volume_attrs(f, vid):
    a = dict(f[vid].attrs)
    return {
        "kind": _as_str(a.get("kind"), vid),
        "cbar_label": _as_str(a.get("cbar_label"), "value"),
        "cmap": _as_str(a.get("cmap"), "viridis"),
        "title": _as_str(a.get("title"), vid),
        "source_volume": _as_str(a.get("source_volume"), ""),
        "dataset_path": _as_str(a.get("dataset_path"), ""),
        "vmin": float(a["vmin"]) if "vmin" in a else None,
        "vmax": float(a["vmax"]) if "vmax" in a else None,
    }


def read_axes(sg):
    return (
        sg["u_um"][:].astype(np.float64),
        sg["v_um"][:].astype(np.float64),
        sg["offsets_um"][:].astype(np.float64),
    )


def resolve_plane_index(offsets_um, offset_um):
    idx = int(np.argmin(np.abs(offsets_um - float(offset_um))))
    return idx, float(offsets_um[idx])


def check_geometry(ref_u, ref_v, cand_u, cand_v, vid, tol):
    if ref_u.shape != cand_u.shape or ref_v.shape != cand_v.shape:
        raise ValueError(f"geometry shape mismatch for {vid!r}")
    if float(np.max(np.abs(ref_u - cand_u))) > tol or float(np.max(np.abs(ref_v - cand_v))) > tol:
        raise ValueError(f"geometry mismatch for {vid!r} exceeds tol {tol:.1e}")


# -----------------------------------------------------------------------------
# Style helpers
# -----------------------------------------------------------------------------
def auto_line_color(cmap_name, override):
    if override:
        return override
    return "black" if cmap_name in _LIGHT_MIDDLE else "cyan"


def _scale_bar(ax, color="black"):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = (x1 - x0), (y1 - y0)
    target = xr * 0.15
    if target >= 100:
        sl = round(target / 50) * 50
    elif target >= 10:
        sl = round(target / 10) * 10
    elif target >= 1:
        sl = round(target)
    else:
        sl = round(target, 1)
    sl = sl or target
    bx, by, bh = x1 - 0.05 * xr - sl, y0 + 0.05 * yr, 0.01 * yr
    ax.add_patch(Rectangle((bx, by), sl, bh, facecolor=color, edgecolor=color))
    ax.text(
        bx + sl / 2.0,
        by + bh * 3,
        f"{sl:.0f} µm",
        color=color,
        fontsize=10,
        ha="center",
        va="bottom",
        fontweight="bold",
    )


def _draw_reference_image(ax, plane2d, u_um, v_um, attrs, line_color, geom=None, title=None):
    extent = [float(u_um[0]), float(u_um[-1]), float(v_um[0]), float(v_um[-1])]
    vmin, vmax = attrs["vmin"], attrs["vmax"]
    norm = (
        mcolors.Normalize(vmin=vmin, vmax=vmax) if (vmin is not None and vmax is not None) else None
    )
    im = ax.imshow(
        plane2d,
        cmap=Rnd.cmap_nan_transparent(attrs["cmap"]),
        norm=norm,
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    ax.set_xlabel("u (µm)")
    ax.set_ylabel("v (µm)")
    if title:
        ax.set_title(title, fontsize=13)
    if geom is not None:
        s, e = geom["start"], geom["end"]
        ax.plot([s[0], e[0]], [s[1], e[1]], color=line_color, lw=2.0, zorder=5)
        ax.plot([s[0], e[0]], [s[1], e[1]], "o", color=line_color, ms=6, zorder=6)
        if geom["width"] > 1:
            half = float(geom["band_offsets"][-1])
            phat = geom["phat"]
            for sgn in (-1.0, 1.0):
                ss, ee = s + sgn * half * phat, e + sgn * half * phat
                ax.plot(
                    [ss[0], ee[0]],
                    [ss[1], ee[1]],
                    color=line_color,
                    lw=1.0,
                    ls="--",
                    alpha=0.5,
                    zorder=4,
                )
    _scale_bar(ax)
    return im


def build_companion_figure(
    ref, fields, geom, line_color, *, style: PlotStyle | None = None
) -> Figure:
    """Build and return a companion profile figure. Does NOT call savefig.

    When *style* is ``None`` the legacy appearance is reproduced exactly
    (image panel + N trace panels, same fonts and colorbar as before).
    When a :class:`~dfxm.common.plotting.PlotStyle` is supplied its font/
    colorbar settings are honoured (these are ``kind="plot"`` figures — no
    scale bar is drawn regardless of style).
    """
    ref_plane, u_um, v_um, ref_attrs, ref_label = ref
    n = len(fields)
    fig = Figure(figsize=(9.0, 4.8 + 1.85 * n), layout="constrained", facecolor="white")
    gs = fig.add_gridspec(nrows=n + 1, ncols=1, height_ratios=[3.0] + [1.0] * n)
    ax_img = fig.add_subplot(gs[0])
    im = _draw_reference_image(
        ax_img,
        ref_plane,
        u_um,
        v_um,
        ref_attrs,
        line_color,
        geom=geom,
        title=f"{ref_attrs['title']}\nreference: {ref_label}",
    )
    if style is None:
        # Legacy path: always draw a plain colorbar.
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04).set_label(ref_attrs["cbar_label"])
    elif style.colorbar:
        # Styled path: honour the colorbar flag — draw styled colorbar only when requested.
        add_colorbar(fig, im, ax_img, ref_attrs["cbar_label"], style)
    distance = geom["distance"]
    first_ax = None
    trace_axes = []
    for i, fld in enumerate(fields):
        ax = fig.add_subplot(gs[i + 1], sharex=first_ax)
        first_ax = first_ax or ax
        trace_axes.append(ax)
        vm = fld["value_mean"]
        ax.plot(distance, vm, "-", lw=1.8, color="C0", zorder=3)
        if fld["value_std"] is not None:
            vs = fld["value_std"]
            ax.fill_between(distance, vm - vs, vm + vs, color="C0", alpha=0.22, lw=0, zorder=2)
        ax.set_ylabel(fld["attrs"]["cbar_label"], fontsize=10)
        src = os.path.basename(fld["attrs"]["source_volume"]) or "(consolidated)"
        ax.set_title(f"{fld['attrs']['kind']}  |  {fld['vid']}  |  {src}", fontsize=10, loc="left")
        ax.grid(True, color="0.85", lw=0.6)
        ax.set_xlim(0.0, geom["L"])
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("distance along line (µm)", fontsize=12)
    if style is not None:
        # Scale only the content axes — colorbar axes are excluded because
        # add_colorbar() already scaled their fonts; hitting them again would
        # produce double-scaled fonts (~font_scale²).
        apply_text_scale(ax_img, style)
        for ax in trace_axes:
            apply_text_scale(ax, style)
    return fig


def save_companion_figure(ref, fields, geom, line_color, out_png, dpi, style=None):
    """Build a companion figure (legacy look when *style* is None) and save it."""
    build_companion_figure(ref, fields, geom, line_color, style=style).savefig(
        out_png, dpi=dpi, facecolor="white", edgecolor="none"
    )


def render_single(ref, geom, line_color, out_png, header, dpi, style=None):
    plane, u_um, v_um, attrs, label = ref
    fig = Figure(figsize=(11, 9), facecolor="white")
    ax = fig.add_subplot(111)
    im = _draw_reference_image(
        ax, plane, u_um, v_um, attrs, line_color, geom=geom, title=f"{header}\nreference: {label}"
    )
    if style is None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(attrs["cbar_label"])
    else:
        if style.colorbar:
            add_colorbar(fig, im, ax, attrs["cbar_label"], style)
        apply_text_scale(ax, style)
    fig.savefig(out_png, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")


# -----------------------------------------------------------------------------
# Drivers
# -----------------------------------------------------------------------------
def _pick_reference_id(present, ref_pref):
    if ref_pref and ref_pref in present:
        return ref_pref
    for vid in present:
        if "raw_sum" in vid:
            return vid
    return present[0]


def _ordered_field_ids(present, ref_id, restrict):
    if restrict:
        return [v for v in restrict if v in present]
    return [ref_id] + sorted(v for v in present if v != ref_id)


def _collect(f, job, p, ref_pref, restrict):
    name = job["name"]
    present = volume_ids_with_slice(f, name)
    if not present:
        raise KeyError(f"slice {name!r} not present in any field group")
    ref_id = _pick_reference_id(present, ref_pref)
    u_um, v_um, offsets = read_axes(f[f"{ref_id}/{name}"])
    idx, off_used = resolve_plane_index(offsets, job["offset_um"])
    ref_attrs = read_volume_attrs(f, ref_id)
    ref_plane = f[f"{ref_id}/{name}"]["slices"][idx].astype(np.float64)
    geom = line_geometry(
        u_um,
        v_um,
        job["start_uv"],
        job["end_uv"],
        job.get("n_samples"),
        job.get("width_pixels", 1),
        grid_pitch(u_um, v_um),
    )
    geom_tol, off_tol = float(p["geom_tol_um"]), float(p["offset_tol_um"])
    fields = []
    for vid in _ordered_field_ids(present, ref_id, restrict):
        sg = f[f"{vid}/{name}"]
        cu, cv, coff = read_axes(sg)
        check_geometry(u_um, v_um, cu, cv, vid, geom_tol)
        if abs(float(coff[idx]) - off_used) > off_tol:
            raise ValueError(f"offset mismatch for {vid!r}")
        plane = sg["slices"][idx].astype(np.float64)
        vm, vs, nv = profile_plane(plane, geom)
        fields.append(
            {
                "vid": vid,
                "attrs": read_volume_attrs(f, vid),
                "value_mean": vm,
                "value_std": vs,
                "n_valid": nv,
                "plane": plane,
            }
        )
    ref = (ref_plane, u_um, v_um, ref_attrs, f"{ref_id}  @ offset {off_used:+.3f} µm")
    return ref, fields, geom, off_used


def _write_csvs(out_dir, stem, distance, fields):
    paths = []
    for fld in fields:
        path = os.path.join(out_dir, f"{stem}__{fld['vid']}.csv")
        if fld["value_std"] is None:
            mat = np.column_stack([distance, fld["value_mean"]])
            header = "distance_um,value_mean"
        else:
            mat = np.column_stack([distance, fld["value_mean"], fld["value_std"]])
            header = "distance_um,value_mean,value_std"
        np.savetxt(path, mat, delimiter=",", header=header, comments="", fmt="%.8g")
        paths.append(path)
    return paths


def _save_overviews(out_dir, stem, ref, fields, geom, off_used, line_override, dpi, style=None):
    u_um, v_um = ref[1], ref[2]
    paths = []
    for fld in fields:
        ov_png = os.path.join(out_dir, f"{stem}__overview__{fld['vid']}.png")
        ov_ref = (
            fld["plane"],
            u_um,
            v_um,
            fld["attrs"],
            f"{fld['vid']}  @ offset {off_used:+.3f} µm",
        )
        color = auto_line_color(fld["attrs"]["cmap"], line_override)
        render_single(ov_ref, geom, color, ov_png, fld["attrs"]["title"], dpi, style=style)
        paths.append(ov_png)
    return paths


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> ProfilesResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    h5_path = p["consolidated_h5"]
    if not h5_path or not os.path.exists(h5_path):
        raise StageUserError(
            f"consolidated slice file not found: {h5_path!r}",
            hint=(
                "Run the slices stage first — it writes oblique_slices.h5, "
                "which this stage profiles."
            ),
        )
    style = style_from_params(p)
    mode = p["mode"].lower()
    if mode not in ("parameter", "preview"):
        raise ValueError(f"mode must be parameter/preview (got {p['mode']!r})")
    out_dir = p["output_dir"] or os.path.join(os.path.dirname(h5_path), "line_profiles")
    os.makedirs(out_dir, exist_ok=True)
    result = ProfilesResult(output_dir=out_dir, mode=mode)

    jobs = json.loads(p["jobs_json"])
    if not isinstance(jobs, list) or not jobs:
        raise StageUserError(
            "jobs_json must be a non-empty JSON list of jobs",
            hint="Define at least one job, or use 'Pick line…' to click a line on a slice plane.",
        )
    ref_pref = p["reference_volume_id"] or ""
    restrict = [v.strip() for v in p["volume_ids"].split(",") if v.strip()] or None
    line_override = p["line_color"] or None
    dpi = int(p["fig_dpi"])

    with h5py.File(h5_path, "r") as f:
        for ji, job in enumerate(jobs):
            progress((ji + 0.5) / len(jobs), f"{mode}: {job.get('name')}")
            name = job["name"]
            present = volume_ids_with_slice(f, name)
            if not present:
                result.skipped.append(f"slice {name!r} not present")
                continue
            if mode == "preview":
                ref_id = _pick_reference_id(present, ref_pref)
                u_um, v_um, offsets = read_axes(f[f"{ref_id}/{name}"])
                idx, off_used = resolve_plane_index(offsets, job["offset_um"])
                attrs = read_volume_attrs(f, ref_id)
                ref_plane = f[f"{ref_id}/{name}"]["slices"][idx].astype(np.float64)
                geom = None
                if job.get("start_uv") is not None and job.get("end_uv") is not None:
                    geom = line_geometry(
                        u_um,
                        v_um,
                        job["start_uv"],
                        job["end_uv"],
                        job.get("n_samples"),
                        job.get("width_pixels", 1),
                        grid_pitch(u_um, v_um),
                    )
                color = auto_line_color(attrs["cmap"], line_override)
                ref = (ref_plane, u_um, v_um, attrs, f"{ref_id}  @ offset {off_used:+.3f} µm")
                stem = (job.get("fig_name") or f"preview_{name}") + "__PREVIEW"
                out_png = os.path.join(out_dir, f"{stem}.png")
                header = f"PREVIEW :: slice {name!r} offset {off_used:+.3f} µm"
                render_single(ref, geom, color, out_png, header, dpi, style=style)
                result.jobs.append(
                    ProfileJobResult(name=name, offset_used_um=off_used, figure=out_png)
                )
                continue

            # parameter mode
            try:
                ref, fields, geom, off_used = _collect(f, job, p, ref_pref, restrict)
            except (KeyError, ValueError) as exc:
                result.skipped.append(f"{name}: {exc}")
                continue
            color = auto_line_color(ref[3]["cmap"], line_override)
            stem = job.get("fig_name") or f"profile_{name}_{off_used:+.2f}um".replace(
                "+", "p"
            ).replace("-", "m")
            out_png = os.path.join(out_dir, f"{stem}.png")
            save_companion_figure(ref, fields, geom, color, out_png, dpi, style=style)
            jr = ProfileJobResult(
                name=name,
                offset_used_um=off_used,
                figure=out_png,
                fields=[fl["vid"] for fl in fields],
            )
            if bool(p["save_csv"]):
                jr.csvs = _write_csvs(out_dir, stem, geom["distance"], fields)
            if bool(p["save_overview"]):
                jr.overviews = _save_overviews(
                    out_dir, stem, ref, fields, geom, off_used, line_override, dpi, style=style
                )
            result.jobs.append(jr)

    progress(1.0, f"{mode}: {len(result.jobs)} job(s) -> {out_dir}")
    return result


@register("profiles")
def figures(result: ProfilesResult, params: dict) -> list[FigureSpec]:
    """One ``kind="plot"`` FigureSpec per parameter-mode job in the result.

    Each spec's ``build(style)`` re-reads the profiling data from the
    consolidated ``oblique_slices.h5`` (``params["consolidated_h5"]``) and
    rebuilds that job's companion figure via :func:`build_companion_figure`.
    Returns ``[]`` when there is no consolidated h5 or no parameter-mode jobs.
    Raises :exc:`FileNotFoundError` at build time when the h5 is missing.
    """
    h5_path = params.get("consolidated_h5", "")
    if not h5_path:
        return []
    # Only parameter-mode jobs produce companion figures (preview jobs have no
    # fields list and their figure is a render_single overview, not a companion).
    jobs_with_fields = [jr for jr in result.jobs if jr.fields]
    if not jobs_with_fields:
        return []

    # Re-parse the original job specs so _collect knows start_uv / end_uv etc.
    try:
        job_specs: list[dict] = json.loads(params.get("jobs_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        job_specs = []
    # Build a name → job-spec lookup for the rebuild closure.
    job_spec_by_name = {j["name"]: j for j in job_specs if "name" in j}

    ref_pref = params.get("reference_volume_id") or ""
    restrict_raw = params.get("volume_ids", "") or ""
    restrict = [v.strip() for v in restrict_raw.split(",") if v.strip()] or None
    line_override = params.get("line_color") or None

    specs = []
    used_stems: dict[str, int] = {}
    for jr in jobs_with_fields:
        name = jr.name
        fig_name = job_spec_by_name.get(name, {}).get("fig_name") or f"profile_{name}"
        # fig_name is free-form user text: two jobs can share it. Disambiguate
        # the export stem so a shared fig_name does not silently overwrite a
        # previously written figure (both would otherwise report ok).
        if fig_name in used_stems:
            used_stems[fig_name] += 1
            fig_name = f"{fig_name}_{used_stems[fig_name]}"
        else:
            used_stems[fig_name] = 1
        job_spec = job_spec_by_name.get(name)

        # KEEP IN SYNC with the parameter-mode render in run() above:
        # _collect → auto_line_color → build_companion_figure (run() uses save_companion_figure).
        def _build(
            style,
            *,
            _h5=h5_path,
            _job=job_spec,
            _p=dict(params),
            _ref=ref_pref,
            _res=restrict,
            _lo=line_override,
            _name=name,
        ):
            if not _job:
                raise ValueError(
                    f"job spec for {_name!r} not found in jobs_json — "
                    "re-run profiles with the current jobs_json to rebuild this figure"
                )
            if not _h5 or not os.path.exists(_h5):
                raise FileNotFoundError(
                    f"consolidated h5 not found at {_h5!r} — re-run the slices stage"
                )
            import h5py as _h5py

            # _collect needs p["geom_tol_um"] and p["offset_tol_um"] — supply defaults
            _p.setdefault("geom_tol_um", STAGE.defaults().get("geom_tol_um", 1e-4))
            _p.setdefault("offset_tol_um", STAGE.defaults().get("offset_tol_um", 1e-3))
            with _h5py.File(_h5, "r") as f:
                ref, fields, geom, _ = _collect(f, _job, _p, _ref, _res)
            color = auto_line_color(ref[3]["cmap"], _lo)
            return build_companion_figure(ref, fields, geom, color, style=style)

        specs.append(
            FigureSpec(
                figure_id=f"profiles_{name}",
                title=f"Profile: {name}",
                kind="plot",
                filename=fig_name,
                build=_build,
            )
        )
    return specs


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Line profiles across oblique slices.")
    ap.add_argument("--consolidated-h5", default="")
    ap.add_argument("--mode", choices=("parameter", "preview"), default="parameter")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args(argv)
    res = run(
        dict(consolidated_h5=args.consolidated_h5, mode=args.mode, output_dir=args.output_dir),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\n{res.mode}: {len(res.jobs)} job(s) -> {res.output_dir}; skipped {len(res.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
