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
from ..common.figures import FigureSpec, register, resolve_clim
from ..common.plotting import (
    GROUP_BY_KIND,
    PlotStyle,
    add_colorbar,
    apply_axes_margins,
    apply_axes_mode,
    apply_text_scale,
    box_drift_note,
    draw_scale_bar,
    fit_axes_to_box,
    fixed_scale,
    fixed_scale_box,
    measure_axes_margins,
    place_axes_box,
    place_axes_stack,
    style_from_params,
    styled_figure,
    trace_fixed_box,
    trace_fixed_scale,
    trace_height_cm,
)
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
        "Writes one figure per field (plus an optional stacked companion) and CSVs. Use "
        "'Pick line…' to choose the line by clicking on the plane."
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
                "('start_uv'/'end_uv'), and band width in pixels. Optional per-job 'fields' "
                "(list of field ids to profile, in order) and 'reference' (top image) override "
                "the global Fields/Reference. Easiest filled by 'Pick line…'."
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
            "save_traces",
            ParamType.BOOL,
            "Save traces",
            default=True,
            advanced=True,
            group="Output",
            help=(
                "Write each profiled field as its own line-profile figure "
                "(separate from the stacked companion)."
            ),
        ),
        Param(
            "save_companion",
            ParamType.BOOL,
            "Save companion",
            default=True,
            advanced=True,
            group="Output",
            help=(
                "Also write the stacked companion figure (overview image + all traces in one). "
                "Turn off to export only the separate traces."
            ),
        ),
        Param(
            "trace_aspect",
            ParamType.STR,
            "Trace aspect",
            default="4:3",
            advanced=True,
            group="Appearance",
            help=(
                "Aspect ratio (width:height) of the plot box — the data area — of each "
                "separate trace figure, e.g. 4:3, 1:1, 16:9. The plotted rectangle keeps this "
                "ratio exactly, regardless of label/title margins."
            ),
        ),
        Param(
            "trace_width_in",
            ParamType.FLOAT,
            "Trace width",
            unit="in",
            default=6.0,
            advanced=True,
            group="Appearance",
            help="Width of each separate trace figure in inches; the height follows the aspect ratio.",
        ),
        Param(
            "trace_linewidth",
            ParamType.FLOAT,
            "Trace line width",
            unit="pt",
            default=2.0,
            advanced=True,
            group="Appearance",
            help="Line thickness of the plotted profile curve on the separate trace figures.",
        ),
        Param(
            "trace_color",
            ParamType.STR,
            "Trace colour",
            default="",
            advanced=True,
            group="Appearance",
            help=(
                "Colour of the profile curve and its std band on the separate trace figures "
                "(blank = default matplotlib blue)."
            ),
        ),
        Param(
            "trace_font_scale",
            ParamType.FLOAT,
            "Trace font scale",
            default=1.4,
            advanced=True,
            group="Appearance",
            help=(
                "Multiplies the label/tick/title font size of the separate trace figures "
                "(independent of the map figures' font scale)."
            ),
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
    traces: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    # position of the originating spec in jobs_json — two jobs may share a slice
    # name, so name alone cannot pair a result back to its spec
    job_index: int | None = None


@dataclass
class ProfilesResult:
    output_dir: str = ""
    mode: str = "parameter"
    jobs: list[ProfileJobResult] = field(default_factory=list)
    # one entry per job that produced NO output (reason included); per-field
    # drop notes for jobs that still ran go to `notes` instead
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


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


def resolve_job_slice_name(f, name, offset_um):
    """Effective slice-group name for a job: *name* when present, else the
    nearest single-plane group of a pinned slices run.

    A pinned run (slices ``use_pinned``) renames each plane to
    ``{name}_pin_{offset:+.2f}um``, so sweep-era jobs would otherwise miss
    every group. Among the pinned groups sharing the base name, the one whose
    stored offset is nearest ``offset_um`` wins — the same nearest-plane snap
    ``resolve_plane_index`` applies within a sweep. Returns
    ``(resolved_name, note)``; *note* is ``None`` unless a pin was substituted.
    """
    if volume_ids_with_slice(f, name):
        return name, None
    prefix = f"{name}_pin_"
    best = None
    for vid in list_volume_ids(f):
        g = f[vid]
        for key in g:
            sg = g[key]
            if not (
                key.startswith(prefix)
                and isinstance(sg, h5py.Group)
                and "slices" in sg
                and "offsets_um" in sg
            ):
                continue
            d = abs(float(sg["offsets_um"][0]) - float(offset_um))
            if best is None or d < best[0]:
                best = (d, key)
    if best is None:
        return name, None
    return best[1], f"job slice {name!r}: using pinned plane group {best[1]!r}"


@dataclass
class ReplotJobEntry:
    """One profile job as the replot dialog sees it: resolved slice + fields."""

    job_index: int
    name: str  # resolved (possibly pinned) slice-group name
    label: str  # display label: fig_name/name @ offset
    fields: list[str]  # volume ids carrying this slice, sorted
    note: str | None  # pin-substitution note, if any


def replot_catalog(h5_path: str, jobs: list[dict]) -> list[ReplotJobEntry]:
    """List each job's resolved slice group and the fields present for it.

    Jobs whose slice has no plain or pinned match are omitted (the dialog
    shows what will actually render; render_replot re-reports the skip).
    Raises StageUserError for an unreadable file.
    """
    try:
        fh = h5py.File(h5_path, "r")
    except OSError as exc:
        raise StageUserError(
            f"cannot read {h5_path!r}: {exc}",
            hint="Point at an oblique_slices.h5 written by the slices stage.",
        ) from exc
    entries: list[ReplotJobEntry] = []
    with fh as f:
        for ji, job in enumerate(jobs):
            if not isinstance(job, dict) or "name" not in job:
                continue
            name, note = resolve_job_slice_name(f, job["name"], job.get("offset_um", 0.0))
            present = volume_ids_with_slice(f, name)
            if not present:
                continue
            off = float(job.get("offset_um", 0.0))
            base = job.get("fig_name") or job["name"]
            entries.append(ReplotJobEntry(ji, name, f"{base}  @ {off:+.2f} µm", present, note))
    return entries


def render_replot(h5_path, jobs, style, clim, out_dir, *, dpi=None, params=None):
    """Re-render profile jobs cold with optional per-quantity colour limits.

    Appearance-only twin of a parameter-mode run: writes companion, overview
    and trace figures for *jobs* into *out_dir* — never CSVs. ``clim`` is a
    ``{key: (vmin, vmax)}`` mapping (field id first, colormap group fallback;
    ``None``/missing keeps stored limits). ``params`` (optional) is a dict of
    stage param overrides (e.g. the form's current values) layered on top of
    the stage defaults — this is how a replot honours the form's appearance
    knobs (trace styling, line colour, reference field, DPI, ...); ``save_csv``
    is always forced off regardless of what's in *params*. ``dpi``, if given,
    wins over ``params["fig_dpi"]``. Returns a ProfilesResult (jobs/skipped/notes).
    """
    if not h5_path or not os.path.exists(h5_path):
        raise StageUserError(
            f"consolidated slice file not found: {h5_path!r}",
            hint="Run the slices stage first, or Browse to an oblique_slices.h5.",
        )
    if not isinstance(jobs, list) or not jobs:
        raise StageUserError(
            "no jobs to replot",
            hint="Check at least one job in the tree (jobs come from the form's Jobs JSON).",
        )
    p = {**STAGE.defaults(), **(params or {}), "save_csv": False}
    if dpi is not None:
        p["fig_dpi"] = int(dpi)
    os.makedirs(out_dir, exist_ok=True)
    result = ProfilesResult(output_dir=out_dir, mode="parameter")
    used_stems: dict[str, int] = {}
    trace_deferred: list = []
    try:
        with h5py.File(h5_path, "r") as f:
            for ji, job in enumerate(jobs):
                if not isinstance(job, dict) or "name" not in job:
                    result.skipped.append(f"job {ji}: malformed job spec")
                    continue
                name, pin_note = resolve_job_slice_name(f, job["name"], job.get("offset_um", 0.0))
                if pin_note:
                    result.notes.append(pin_note)
                    job = {**job, "name": name}
                if not volume_ids_with_slice(f, name):
                    result.skipped.append(f"slice {name!r} not present")
                    continue
                _render_parameter_job(
                    f,
                    job,
                    ji,
                    (ji + 0.5) / len(jobs),
                    p,
                    result,
                    used_stems,
                    out_dir,
                    style,
                    _noop,
                    clim=clim,
                    trace_deferred=trace_deferred,
                )
    finally:
        # Runs even on a hard mid-loop failure (e.g. a corrupt h5 raising
        # partway through) so earlier jobs' already-built fixed-scale trace
        # figures still get placed and saved instead of silently discarded.
        _flush_deferred_traces(trace_deferred, int(p["fig_dpi"]), result.notes)
    return result


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
def parse_aspect(s: str) -> tuple[float, float]:
    """Parse a 'W:H' aspect string into positive (width, height) floats."""
    parts = str(s).split(":")
    hint = "Enter the ratio as two numbers separated by a colon, e.g. 4:3 or 1:1."
    if len(parts) != 2:
        raise StageUserError(f"aspect must be 'W:H' (got {s!r})", hint=hint)
    try:
        w, h = float(parts[0]), float(parts[1])
    except ValueError:
        raise StageUserError(
            f"aspect must be 'W:H' with numeric parts (got {s!r})", hint=hint
        ) from None
    if not (np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0):
        raise StageUserError(f"aspect parts must be positive and finite (got {s!r})", hint=hint)
    return w, h


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


def _draw_reference_image(
    ax,
    plane2d,
    u_um,
    v_um,
    attrs,
    line_color,
    geom=None,
    title=None,
    style=None,
    fixed_scale_um_per_cm=None,
    scale_bar=None,
):
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
    if style is None:
        _scale_bar(ax)  # legacy look, pinned
    elif style.scale_bar if scale_bar is None else scale_bar:
        draw_scale_bar(
            ax,
            style.scale_bar_length_um,
            style=style,
            fixed_scale_um_per_cm=fixed_scale_um_per_cm,
        )
    return im


def build_companion_figure(
    ref, fields, geom, line_color, *, style: PlotStyle | None = None, trace_opts=None, notes=None
) -> Figure:
    """Build and return a companion profile figure. Does NOT call savefig.

    When *style* has no effective fixed scale (map or trace) — including
    ``style=None`` — the legacy appearance is reproduced exactly (image panel
    + N trace panels, same fonts and colorbar as before): see
    :func:`_build_companion_legacy`. When a fixed scale IS in effect, the
    companion is instead built on the deterministic left-aligned stack layout
    — the map panel at the map scale, trace panels styled exactly like the
    standalone trace figures — see :func:`_build_companion_fixed`. That fixed
    path itself degrades back to :func:`_build_companion_legacy` (never
    raises) when the reference plane's own extent is degenerate (zero-width,
    single-point, or non-finite ``u_um``/``v_um`` — e.g. a pinned edge-of-ROI
    plane) so the trace scale alone cannot fit a physical map box.

    *trace_opts* (fixed-scale path only) is ``{"linewidth": float, "color":
    str | None, "font_scale": float}``; ``None`` keeps the trace panels'
    built-in default styling. *notes* (fixed-scale path only), when given, is
    a list that box-drift warnings are appended to (the caller's
    ``ProfilesResult.notes``).
    """
    if trace_fixed_box(style, float(geom["L"])) is None:
        return _build_companion_legacy(ref, fields, geom, line_color, style)
    return _build_companion_fixed(ref, fields, geom, line_color, style, trace_opts, notes)


def _build_companion_legacy(
    ref, fields, geom, line_color, style: PlotStyle | None = None
) -> Figure:
    """The pre-fixed-scale companion layout — pinned by tests, verbatim."""
    ref_plane, u_um, v_um, ref_attrs, ref_label = ref
    n = len(fields)
    fig = styled_figure((9.0, 4.8 + 1.85 * n), styled=True)
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
        style=style,
    )
    if style is None:
        # Legacy path: always draw a plain colorbar.
        fig.colorbar(im, ax=ax_img, fraction=0.046, pad=0.04).set_label(ref_attrs["cbar_label"])
    elif style.colorbar:
        # Styled path: honour the colorbar flag — draw styled colorbar only when requested.
        add_colorbar(
            fig,
            im,
            ax_img,
            ref_attrs["cbar_label"],
            style,
            group=GROUP_BY_KIND.get(ref_attrs.get("kind")),
        )
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


def _build_companion_fixed(ref, fields, geom, line_color, style, trace_opts, notes):
    """Fixed-scale companion: map panel at the MAP scale, trace panels styled
    exactly like the standalone trace figures, stacked left-aligned.

    Falls back to :func:`_build_companion_legacy` (never raises) when the
    reference plane's extent is degenerate and ``fixed_scale_box`` returns
    ``None`` for the map panel — a plausible pinned edge-of-ROI plane."""
    ref_plane, u_um, v_um, ref_attrs, ref_label = ref
    topts = {"linewidth": 1.8, "color": None, "font_scale": 1.0, **(trace_opts or {})}
    ext_u, ext_v = float(u_um[-1] - u_um[0]), float(v_um[-1] - v_um[0])
    map_scale = fixed_scale(style) or trace_fixed_scale(style)
    mbox = fixed_scale_box(style, ext_u, ext_v, scale=map_scale)
    if mbox is None:
        # Degenerate reference-plane extent (zero-width/single-point/non-finite
        # u_um or v_um — a plausible pinned edge-of-ROI plane): the map panel
        # cannot be fitted to a physical scale even though the trace scale is
        # set. Degrade to the legacy layout rather than indexing a None box —
        # guards in this module never raise (see fixed_scale/trace_fixed_box).
        if notes is not None:
            notes.append(
                "companion: reference plane extent is degenerate — rendered with the "
                "legacy layout (fixed scale not applied)"
            )
        return _build_companion_legacy(ref, fields, geom, line_color, style)
    tbox = trace_fixed_box(style, float(geom["L"]))
    fig = styled_figure((10.0, 10.0), styled=True)
    fig.set_layout_engine("none")
    n = len(fields)
    ax_img = fig.add_subplot(n + 1, 1, 1)
    im = _draw_reference_image(
        ax_img,
        ref_plane,
        u_um,
        v_um,
        ref_attrs,
        line_color,
        geom=geom,
        title=(f"{ref_attrs['title']}\nreference: {ref_label}" if style.show_title else None),
        style=style,
        fixed_scale_um_per_cm=mbox[2],
    )
    cax = None
    if style.colorbar:
        cax = fig.add_axes([0.9, 0.6, 0.03, 0.25])  # provisional; sync repositions it
        add_colorbar(
            fig,
            im,
            ax_img,
            ref_attrs["cbar_label"],
            style,
            group=GROUP_BY_KIND.get(ref_attrs.get("kind")),
            cax=cax,
        )
    apply_text_scale(ax_img, style)

    def _sync_cax(fig_, ax_, _w=mbox[0]):
        if cax is None:
            return
        pos = ax_.get_position()
        fw, _fh = fig_.get_size_inches()
        cax.set_position(
            [pos.x1 + 0.04 * _w / fw, pos.y0, style.colorbar_fraction * _w / fw, pos.height]
        )

    trace_axes = []
    for i, fld in enumerate(fields):
        ax = fig.add_subplot(n + 1, 1, i + 2)
        _draw_trace_axes(
            ax,
            fld,
            geom,
            linewidth=topts["linewidth"],
            color=topts["color"],
            font_scale=topts["font_scale"],
            style=style,
            show_xlabel=(i == n - 1),
        )
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        trace_axes.append(ax)
    panels = [(ax_img, mbox[0], mbox[1], (cax,) if cax is not None else (), _sync_cax)]
    panels += [(ax, tbox[0], tbox[1], (), None) for ax in trace_axes]
    place_axes_stack(fig, panels)
    if notes is not None:
        for label, ax, (w, h) in [("companion map", ax_img, (mbox[0], mbox[1]))] + [
            (f"companion trace {fld['vid']}", ax, (tbox[0], tbox[1]))
            for fld, ax in zip(fields, trace_axes)
        ]:
            note = box_drift_note(label, fig, ax, w, h)
            if note:
                notes.append(note)
    return fig


def save_companion_figure(
    ref, fields, geom, line_color, out_png, dpi, style=None, trace_opts=None, notes=None
):
    """Build a companion figure (legacy look when *style* is None) and save it.

    *trace_opts*/*notes* are forwarded to :func:`build_companion_figure`
    unchanged — see there (fixed-scale path only; ignored on the legacy path).
    """
    build_companion_figure(
        ref, fields, geom, line_color, style=style, trace_opts=trace_opts, notes=notes
    ).savefig(out_png, dpi=dpi, facecolor="white", edgecolor="none")


def _draw_trace_axes(ax, fld, geom, *, linewidth, color, font_scale, style, show_xlabel=True):
    """Draw one field's line profile into *ax* — the single source of the trace
    look, shared by the standalone trace figures and the companion panels."""
    fs = float(font_scale)
    curve_color = color or "C0"
    distance = geom["distance"]
    vm = fld["value_mean"]
    ax.plot(distance, vm, "-", lw=float(linewidth), color=curve_color, zorder=3)
    if fld["value_std"] is not None:
        vs = fld["value_std"]
        ax.fill_between(distance, vm - vs, vm + vs, color=curve_color, alpha=0.22, lw=0, zorder=2)
    ax.set_ylabel(fld["attrs"]["cbar_label"], fontsize=10 * fs)
    src = os.path.basename(fld["attrs"]["source_volume"]) or "(consolidated)"
    if style is None or style.show_title:
        title_fs = 10 * fs * (style.title_scale if style is not None else 1.0)
        ax.set_title(
            f"{fld['attrs']['kind']}  |  {fld['vid']}  |  {src}", fontsize=title_fs, loc="left"
        )
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_xlim(0.0, geom["L"])
    if show_xlabel:
        ax.set_xlabel("distance along line (µm)", fontsize=12 * fs)
    ax.tick_params(axis="both", labelsize=10 * fs)
    ax.yaxis.get_offset_text().set_fontsize(10 * fs)
    ax.xaxis.get_offset_text().set_fontsize(10 * fs)


# Public aliases for the compose adapters (the composer draws into its own axes;
# the underscore originals remain the in-module call sites).
draw_reference_axes = _draw_reference_image
draw_trace_axes = _draw_trace_axes


def build_trace_figure(
    fld,
    geom,
    *,
    aspect_wh,
    width_in,
    linewidth,
    color,
    font_scale,
    style: PlotStyle | None = None,
) -> Figure:
    """Build a standalone line-profile figure for a single field. Does NOT savefig.

    Two modes, chosen by the style's TRACE-effective fixed scale
    (``trace_scale_um_per_cm``, falling back to the map's ``scale_um_per_cm``):

    * **Fixed-scale mode** (scale set): the plot box is placed at an EXACT
      physical size — ``geom["L"] / scale`` cm wide by ``style.trace_height_cm``
      cm tall (default 3 cm) — via the deterministic ``place_axes_box`` engine
      (measure decorations once, size the figure to margins+box, no iteration,
      no ``set_box_aspect``). ``aspect_wh`` and ``width_in`` are both ignored in
      this mode (the box height comes from ``trace_height_cm``, not the
      aspect ratio); the distance axis prints at the same µm-per-cm as the map
      figures. Width clamps to 30 in like the map figures, raising the
      effective scale and logging a warning rather than producing an
      unreasonably wide canvas.
    * **Legacy mode** (no fixed scale, incl. ``style=None``): ``aspect_wh ==
      (w, h)`` pins the **plot box** (the data rectangle) to exactly ``w:h`` via
      ``ax.set_box_aspect(h / w)`` — so the plotted area keeps the requested
      ratio regardless of how much room the labels/title consume or how large
      ``font_scale`` is. The figure canvas is created at ``(width_in, width_in
      * h / w)`` so it roughly matches the box (minimal whitespace); on save
      the PNG is tight-cropped (its file dimensions hug box+labels).

    All trace text is multiplied by ``font_scale`` — this is the trace figures'
    own scale, independent of the map figures' ``style.font_scale``. The curve
    and its std band use ``color`` (blank/None -> ``"C0"``). No colorbar (it is a
    1-D plot); ``style`` selects the ``styled_figure`` layout engine so the
    background/layout matches the rest of the stage's output, and its
    ``show_title``/``title_scale`` flags apply to the trace title (``style=None``
    keeps the legacy always-on title).
    """
    w_ratio, h_ratio = aspect_wh
    box = trace_fixed_box(style, float(geom["L"]))
    if box is not None:
        # fixed-scale mode: exact physical box, deterministic placement — no
        # set_box_aspect, no constrained layout, no tight-crop reliance.
        fig = styled_figure((box[0] + 1.5, box[1] + 1.5), styled=True)
        ax = fig.add_subplot(111)
        _draw_trace_axes(
            ax, fld, geom, linewidth=linewidth, color=color, font_scale=font_scale, style=style
        )
        place_axes_box(fig, ax, box[0], box[1])
        return fig
    figsize = (float(width_in), float(width_in) * float(h_ratio) / float(w_ratio))
    fig = styled_figure(figsize, styled=style is not None)
    ax = fig.add_subplot(111)
    ax.set_box_aspect(float(h_ratio) / float(w_ratio))  # legacy: pin the box to w:h
    _draw_trace_axes(
        ax, fld, geom, linewidth=linewidth, color=color, font_scale=font_scale, style=style
    )
    return fig


def render_single(ref, geom, line_color, out_png, header, dpi, style=None, notes=None):
    plane, u_um, v_um, attrs, label = ref
    ext_u = float(u_um[-1] - u_um[0])
    ext_v = float(v_um[-1] - v_um[0])
    box = fixed_scale_box(style, ext_u, ext_v)
    figsize = (box[0] + 1.5, box[1] + 1.5) if box is not None else (11, 9)
    fig = styled_figure(figsize, styled=style is not None)
    ax = fig.add_subplot(111)
    im = _draw_reference_image(
        ax,
        plane,
        u_um,
        v_um,
        attrs,
        line_color,
        geom=geom,
        title=f"{header}\nreference: {label}",
        style=style,
        fixed_scale_um_per_cm=(box[2] if box is not None else None),
    )
    if style is None:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(attrs["cbar_label"])
    else:
        if style.colorbar:
            add_colorbar(
                fig, im, ax, attrs["cbar_label"], style, group=GROUP_BY_KIND.get(attrs.get("kind"))
            )
        apply_text_scale(ax, style)
        apply_axes_mode(ax, style)
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])
        if notes is not None:
            note = box_drift_note(os.path.basename(out_png), fig, ax, box[0], box[1])
            if note:
                notes.append(note)
    fig.savefig(out_png, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")


# -----------------------------------------------------------------------------
# Drivers
# -----------------------------------------------------------------------------
def _unique_name(used: dict[str, int], base: str) -> str:
    """Return *base* or ``base_N``, registering the result in *used* — so a later
    job whose own name equals an already-generated ``base_N`` still comes out
    unique instead of silently colliding with it."""
    if base not in used:
        used[base] = 1
        return base
    n = used[base]
    while True:
        n += 1
        candidate = f"{base}_{n}"
        if candidate not in used:
            used[base] = n
            used[candidate] = 1
            return candidate


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


def _clim_attrs(attrs, vid, clim):
    """Apply a per-quantity ``(vmin, vmax)`` override to a read_volume_attrs dict.

    Key resolution matches the slices replot: exact field id first (e.g.
    ``mosa_com_chi``), then the field kind's colormap group via GROUP_BY_KIND.
    A half-open pair keeps the stored value on the blank side. ``clim=None``
    (or no matching key) leaves *attrs* untouched.
    """
    pair = resolve_clim(clim, vid)
    if pair is None:
        pair = resolve_clim(clim, GROUP_BY_KIND.get(attrs.get("kind", ""), ""))
    if pair is None:
        return attrs
    lo, hi = pair
    if lo is not None:
        attrs["vmin"] = float(lo)
    if hi is not None:
        attrs["vmax"] = float(hi)
    return attrs


def _collect(f, job, p, ref_pref, restrict, clim=None):
    name = job["name"]
    present = volume_ids_with_slice(f, name)
    if not present:
        raise KeyError(f"slice {name!r} not present in any field group")
    # per-job overrides fall back to the global reference / restrict
    job_ref = job.get("reference") or ref_pref
    job_fields = job.get("fields") or restrict
    ref_id = _pick_reference_id(present, job_ref)
    u_um, v_um, offsets = read_axes(f[f"{ref_id}/{name}"])
    idx, off_used = resolve_plane_index(offsets, job["offset_um"])
    ref_attrs = _clim_attrs(read_volume_attrs(f, ref_id), ref_id, clim)
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
    dropped = []
    for vid in _ordered_field_ids(present, ref_id, job_fields):
        sg = f[f"{vid}/{name}"]
        cu, cv, coff = read_axes(sg)
        # A field on a different grid/sweep than the reference (e.g. a raw_mosa
        # volume sliced with its own pixel scale) drops with a note; the job
        # keeps going with the fields that do match.
        reason = None
        try:
            check_geometry(u_um, v_um, cu, cv, vid, geom_tol)
        except ValueError as exc:
            reason = str(exc)
        else:
            if idx >= len(coff):
                reason = f"plane sweep shorter than reference for {vid!r}"
            elif abs(float(coff[idx]) - off_used) > off_tol:
                reason = f"offset mismatch for {vid!r}"
        if reason is not None:
            dropped.append(f"field dropped — {reason}")
            continue
        plane = sg["slices"][idx].astype(np.float64)
        vm, vs, nv = profile_plane(plane, geom)
        fields.append(
            {
                "vid": vid,
                "attrs": _clim_attrs(read_volume_attrs(f, vid), vid, clim),
                "value_mean": vm,
                "value_std": vs,
                "n_valid": nv,
                "plane": plane,
            }
        )
    # No usable fields *because they were dropped* skips the job with reasons.
    # An empty list with nothing dropped (a job's "fields" naming only absent
    # ids) keeps the pre-drop behaviour: proceed and render reference-only.
    if not fields and dropped:
        raise ValueError("no usable fields: " + "; ".join(dropped))
    ref = (ref_plane, u_um, v_um, ref_attrs, f"{ref_id}  @ offset {off_used:+.3f} µm")
    return ref, fields, geom, off_used, dropped


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


def _save_overviews(
    out_dir, stem, ref, fields, geom, off_used, line_override, dpi, style=None, notes=None
):
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
        render_single(
            ov_ref, geom, color, ov_png, fld["attrs"]["title"], dpi, style=style, notes=notes
        )
        paths.append(ov_png)
    return paths


def _save_traces(
    out_dir,
    stem,
    fields,
    geom,
    *,
    aspect,
    width_in,
    linewidth,
    color,
    font_scale,
    dpi,
    style=None,
    deferred=None,
    notes=None,
):
    # Legacy mode (no fixed trace scale): the PNG is tight-cropped around
    # box+labels; the plot box keeps `aspect` exactly. Fixed-scale mode never
    # tight-crops — build_trace_figure already placed the axes at its own
    # exact box+margins, and (when *deferred* is given) the figure is handed
    # to _flush_deferred_traces for a second pass that re-places every figure
    # of the invocation at the shared max margins before saving the exact
    # canvas, so every trace PNG of one run/replot aligns in a grid.
    aspect_wh = parse_aspect(aspect)
    paths = []
    for fld in fields:
        tr_png = os.path.join(out_dir, f"{stem}__trace__{fld['vid']}.png")
        fig = build_trace_figure(
            fld,
            geom,
            aspect_wh=aspect_wh,
            width_in=width_in,
            linewidth=linewidth,
            color=color,
            font_scale=font_scale,
            style=style,
        )
        box = trace_fixed_box(style, float(geom["L"]))
        if box is not None:
            if notes is not None:
                if box[2] != trace_fixed_scale(style):
                    msg = (
                        f"{os.path.basename(tr_png)}: trace box clamped to 30 in — "
                        f"effective scale raised to {box[2]:.4g} µm/cm"
                    )
                    if msg not in notes:
                        notes.append(msg)
                if trace_height_cm(style) / 2.54 > box[1]:
                    msg = (
                        f"{os.path.basename(tr_png)}: trace box height clamped to 30 in "
                        f"(trace_height_cm={trace_height_cm(style):g} cm)"
                    )
                    if msg not in notes:
                        notes.append(msg)
            if deferred is not None:
                deferred.append((fig, box[0], box[1], tr_png))
            else:
                if notes is not None:
                    note = box_drift_note(
                        os.path.basename(tr_png), fig, fig.axes[0], box[0], box[1]
                    )
                    if note:
                        notes.append(note)
                fig.savefig(tr_png, dpi=dpi, facecolor="white", edgecolor="none")
        else:
            fig.savefig(tr_png, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
        paths.append(tr_png)
    return paths


def _flush_deferred_traces(deferred, dpi, notes):
    """Second pass for fixed-scale traces: re-place every figure of the
    invocation with the shared max margins (so all PNGs align in a grid),
    verify the box, save, then clear the figure (drops its artists/renderer
    buffers) so a big sweep does not accumulate every trace figure's memory
    until the whole invocation finishes. First pass (build) already placed
    each figure with its own margins, so single-figure consumers stay exact."""
    if not deferred:
        return
    shared = None
    for fig, w_in, h_in, _png in deferred:
        m = measure_axes_margins(fig, fig.axes[0])
        shared = m if shared is None else shared.max_with(m)
    for fig, w_in, h_in, png in deferred:
        apply_axes_margins(fig, fig.axes[0], w_in, h_in, shared)
        note = box_drift_note(os.path.basename(png), fig, fig.axes[0], w_in, h_in)
        if note:
            notes.append(note)
        fig.savefig(png, dpi=dpi, facecolor="white", edgecolor="none")
        fig.clear()


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def _render_parameter_job(
    f,
    job,
    ji,
    frac,
    p,
    result,
    used_stems,
    out_dir,
    style,
    progress,
    clim=None,
    trace_deferred=None,
):
    """Render one parameter-mode job into *out_dir* (companion/traces/CSVs/
    overviews per the p-flags), appending to *result*. Shared verbatim by
    run() and render_replot() so the two paths cannot drift; *frac* is the
    precomputed progress fraction for this job's drop notes and *clim* is the
    per-quantity override mapping threaded to _collect. *trace_deferred*, when
    given, collects fixed-scale trace figures for the caller to flush via
    :func:`_flush_deferred_traces` once every job in the invocation is built,
    so all trace PNGs of the run/replot share one set of margins."""
    ref_pref = p["reference_volume_id"] or ""
    restrict = [v.strip() for v in p["volume_ids"].split(",") if v.strip()] or None
    line_override = p["line_color"] or None
    dpi = int(p["fig_dpi"])
    save_traces = bool(p["save_traces"])
    save_companion = bool(p["save_companion"])
    trace_aspect = p["trace_aspect"]
    trace_width_in = float(p["trace_width_in"])
    trace_linewidth = float(p["trace_linewidth"])
    trace_color = p["trace_color"] or None
    trace_font_scale = float(p["trace_font_scale"])
    name = job["name"]
    try:
        ref, fields, geom, off_used, dropped = _collect(f, job, p, ref_pref, restrict, clim=clim)
    except (KeyError, ValueError) as exc:
        result.skipped.append(f"{name}: {exc}")
        return
    for reason in dropped:
        msg = f"{name}: {reason}"
        result.notes.append(msg)
        progress(frac, msg)
    color = auto_line_color(ref[3]["cmap"], line_override)
    stem = _unique_name(
        used_stems,
        job.get("fig_name")
        or f"profile_{name}_{off_used:+.2f}um".replace("+", "p").replace("-", "m"),
    )
    jr = ProfileJobResult(
        name=name,
        offset_used_um=off_used,
        fields=[fl["vid"] for fl in fields],
        job_index=ji,
    )
    if save_companion:
        out_png = os.path.join(out_dir, f"{stem}.png")
        save_companion_figure(
            ref,
            fields,
            geom,
            color,
            out_png,
            dpi,
            style=style,
            trace_opts={
                "linewidth": trace_linewidth,
                "color": trace_color,
                "font_scale": trace_font_scale,
            },
            notes=result.notes,
        )
        jr.figure = out_png
    if save_traces:
        jr.traces = _save_traces(
            out_dir,
            stem,
            fields,
            geom,
            aspect=trace_aspect,
            width_in=trace_width_in,
            linewidth=trace_linewidth,
            color=trace_color,
            font_scale=trace_font_scale,
            dpi=dpi,
            style=style,
            deferred=trace_deferred,
            notes=result.notes,
        )
    if bool(p["save_csv"]):
        jr.csvs = _write_csvs(out_dir, stem, geom["distance"], fields)
    if bool(p["save_overview"]):
        jr.overviews = _save_overviews(
            out_dir,
            stem,
            ref,
            fields,
            geom,
            off_used,
            line_override,
            dpi,
            style=style,
            notes=result.notes,
        )
    result.jobs.append(jr)


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
    line_override = p["line_color"] or None
    dpi = int(p["fig_dpi"])
    save_traces = bool(p["save_traces"])
    trace_aspect = p["trace_aspect"]
    trace_width_in = float(p["trace_width_in"])
    trace_linewidth = float(p["trace_linewidth"])
    trace_font_scale = float(p["trace_font_scale"])
    if save_traces:
        parse_aspect(trace_aspect)  # fail fast on a bad aspect before the h5 loop
        for _label, _val in (
            ("trace_width_in", trace_width_in),
            ("trace_linewidth", trace_linewidth),
            ("trace_font_scale", trace_font_scale),
        ):
            if not (np.isfinite(_val) and _val > 0):
                raise StageUserError(
                    f"{_label} must be a positive number (got {_val!r})",
                    hint="Enter a positive value, e.g. trace_width_in=6, "
                    "trace_linewidth=2, trace_font_scale=1.4.",
                )

    # Same-named jobs default to the same stem — suffix later ones so a second
    # job on one slice cannot silently overwrite the first's files.
    used_stems: dict[str, int] = {}
    trace_deferred: list = []

    try:
        with h5py.File(h5_path, "r") as f:
            for ji, job in enumerate(jobs):
                progress((ji + 0.5) / len(jobs), f"{mode}: {job.get('name')}")
                name, pin_note = resolve_job_slice_name(f, job["name"], job.get("offset_um", 0.0))
                if pin_note:
                    result.notes.append(pin_note)
                    job = {**job, "name": name}
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
                    stem = _unique_name(
                        used_stems, (job.get("fig_name") or f"preview_{name}") + "__PREVIEW"
                    )
                    out_png = os.path.join(out_dir, f"{stem}.png")
                    header = f"PREVIEW :: slice {name!r} offset {off_used:+.3f} µm"
                    render_single(
                        ref, geom, color, out_png, header, dpi, style=style, notes=result.notes
                    )
                    result.jobs.append(
                        ProfileJobResult(
                            name=name, offset_used_um=off_used, figure=out_png, job_index=ji
                        )
                    )
                    continue

                # parameter mode
                _render_parameter_job(
                    f,
                    job,
                    ji,
                    (ji + 0.5) / len(jobs),
                    p,
                    result,
                    used_stems,
                    out_dir,
                    style,
                    progress,
                    trace_deferred=trace_deferred,
                )
    finally:
        # Runs even on a hard mid-loop failure (e.g. a corrupt h5 raising
        # partway through) so earlier jobs' already-built fixed-scale trace
        # figures still get placed and saved instead of silently discarded.
        _flush_deferred_traces(trace_deferred, int(p["fig_dpi"]), result.notes)
    progress(1.0, f"{mode}: {len(result.jobs)} job(s) -> {out_dir}")
    return result


@register("profiles")
def figures(result: ProfilesResult, params: dict) -> list[FigureSpec]:
    """``kind="plot"`` FigureSpecs per parameter-mode job in the result.

    Per job (when the matching toggle is on): a companion FigureSpec
    (``profiles_<name>``, rebuilt via :func:`build_companion_figure`) followed by
    one trace FigureSpec per field (``profiles_<name>__trace__<vid>``, rebuilt via
    :func:`build_trace_figure`). Toggles (``save_companion``/``save_traces``, both
    default on) and the ``trace_*`` appearance params come from *params* with a
    ``STAGE.defaults()`` fallback. Each spec's ``build(style)`` re-reads the
    profiling data from the consolidated ``oblique_slices.h5``
    (``params["consolidated_h5"]``). Returns ``[]`` when there is no consolidated
    h5 or no parameter-mode jobs. Raises :exc:`FileNotFoundError` at build time
    when the h5 is missing.
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

    defaults = STAGE.defaults()
    save_companion = bool(params.get("save_companion", defaults["save_companion"]))
    save_traces = bool(params.get("save_traces", defaults["save_traces"]))
    trace_aspect = params.get("trace_aspect", defaults["trace_aspect"])
    trace_width_in = float(params.get("trace_width_in", defaults["trace_width_in"]))
    trace_linewidth = float(params.get("trace_linewidth", defaults["trace_linewidth"]))
    trace_color = params.get("trace_color", defaults["trace_color"]) or None
    trace_font_scale = float(params.get("trace_font_scale", defaults["trace_font_scale"]))

    specs = []
    used_stems: dict[str, int] = {}
    used_ids: dict[str, int] = {}
    for jr in jobs_with_fields:
        name = jr.name
        # Pair the result with its originating spec by position (job_index) —
        # two jobs may share one slice name, and a by-name lookup would rebuild
        # both from the last same-named spec. By-name stays as the fallback for
        # results predating job_index.
        job_spec = None
        if jr.job_index is not None and 0 <= jr.job_index < len(job_specs):
            candidate = job_specs[jr.job_index]
            if isinstance(candidate, dict) and candidate.get("name") == name:
                job_spec = candidate
        if job_spec is None:
            job_spec = job_spec_by_name.get(name)
        # fig_name is free-form user text: two jobs can share it. Disambiguate
        # the export stem so a shared fig_name does not silently overwrite a
        # previously written figure (both would otherwise report ok). The
        # figure_id must stay unique for same-named jobs too (export reporting).
        fig_name = _unique_name(used_stems, (job_spec or {}).get("fig_name") or f"profile_{name}")
        base_id = _unique_name(used_ids, f"profiles_{name}")

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
            _lw=trace_linewidth,
            _col=trace_color,
            _fs=trace_font_scale,
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
                ref, fields, geom, _, _dropped = _collect(f, _job, _p, _ref, _res)
            color = auto_line_color(ref[3]["cmap"], _lo)
            topts = {"linewidth": _lw, "color": _col, "font_scale": _fs}
            return build_companion_figure(ref, fields, geom, color, style=style, trace_opts=topts)

        if save_companion:
            specs.append(
                FigureSpec(
                    figure_id=base_id,
                    title=f"Profile: {name}",
                    kind="plot",
                    filename=fig_name,
                    build=_build,
                )
            )
        if save_traces:
            for vid in jr.fields:

                def _tbuild(
                    style,
                    *,
                    _h5=h5_path,
                    _job=job_spec,
                    _p=dict(params),
                    _ref=ref_pref,
                    _res=restrict,
                    _vid=vid,
                    _asp=trace_aspect,
                    _w=trace_width_in,
                    _lw=trace_linewidth,
                    _col=trace_color,
                    _fs=trace_font_scale,
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

                    _p.setdefault("geom_tol_um", STAGE.defaults().get("geom_tol_um", 1e-4))
                    _p.setdefault("offset_tol_um", STAGE.defaults().get("offset_tol_um", 1e-3))
                    with _h5py.File(_h5, "r") as f:
                        _, _fields, _geom, _, _dropped = _collect(f, _job, _p, _ref, _res)
                    _fld = next((fl for fl in _fields if fl["vid"] == _vid), None)
                    if _fld is None:
                        raise ValueError(f"field {_vid!r} not present for job {_name!r}")
                    return build_trace_figure(
                        _fld,
                        _geom,
                        aspect_wh=parse_aspect(_asp),
                        width_in=_w,
                        linewidth=_lw,
                        color=_col,
                        font_scale=_fs,
                        style=style,
                    )

                specs.append(
                    FigureSpec(
                        figure_id=f"{base_id}__trace__{vid}",
                        title=f"Profile trace: {name} · {vid}",
                        kind="plot",
                        filename=f"{fig_name}__trace__{vid}",
                        build=_tbuild,
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
