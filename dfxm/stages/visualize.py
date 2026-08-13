"""Visualize stage — aligned mosaicity/strain volumes to images + animation + 3D.

Ported from ``visualize_aligned_volumes_v6``. Per dataset (mosaicity chi/mu
Center-of-mass + FWHM, and strain) it:

1. aligns the stacked volume — ROI -> samy sub-pixel X-shift -> uniform-Z
   interpolation — reusing :mod:`dfxm.common.alignment` (golden-tested,
   voxel-identical to the PVTI exporter);
2. centres CoM volumes (midrange/mean/median) and picks colour limits (strain
   keeps its physical zero, symmetric limits);
3. writes per-layer PNGs, a layer-by-layer animation (MP4 with GIF fallback),
   a 3-D top-view render, and (optionally) a rotating 3-D orbit video (the 3-D
   renders are best-effort — skipped gracefully without a GL context).

Mosaicity uses ``magma``; strain uses ``RdBu_r``. Rendering uses the explicit
Figure/Agg API (no pyplot / matplotlib.use) so the module is import-safe in the
Qt GUI process.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np

from ..common import alignment as A
from ..common import render as Rnd
from ..common import render3d as R3
from ..common.figures import FigureSpec, register
from ..common.plotting import apply_round_clim, resolve_cmap, style_from_params
from ..common.raster import extract_motor_positions
from ..common.sort import find_matching_folders
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


STAGE = StageSpec(
    name="visualize",
    label="Visualize volumes",
    description=(
        "Aligns the stacked mosaicity/strain volumes into the shared sample frame "
        "(samy shift + uniform-Z interpolation) and renders per-layer PNGs, a layer "
        "animation, a 3-D top view, and an optional rotating 3-D video."
    ),
    params=(
        Param(
            "mosa_volume_file",
            ParamType.PATH,
            "Mosaicity volume",
            must_exist=True,
            help=(
                "The stacked mosaicity volume (stacked_volumes.h5) from the mosaicity stage. "
                "Leave blank to skip mosaicity rendering."
            ),
        ),
        Param(
            "strain_volume_file",
            ParamType.PATH,
            "Strain volume",
            must_exist=True,
            help=(
                "The stacked strain volume (stacked_strain_volumes.h5) from the strain stage. "
                "Leave blank to skip strain rendering."
            ),
        ),
        Param(
            "raw_root",
            ParamType.DIR,
            "Raw data root",
            must_exist=True,
            help=(
                "RAW_DATA root with the original scan folders — the samy/samz motor positions "
                "read from there drive the alignment."
            ),
        ),
        Param(
            "mosa_pattern",
            ParamType.STR,
            "Mosaicity raw pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob matching the raw mosaicity scan folders, used to read their "
                "samy/samz positions."
            ),
        ),
        Param(
            "strain_pattern",
            ParamType.STR,
            "Strain raw pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob matching the raw strain scan folders, used to read their samy/samz positions."
            ),
        ),
        Param(
            "samy_path",
            ParamType.STR,
            "samy path",
            default="1.1/instrument/positioners/samy",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path to the sample-Y motor position inside each scan file (under the first "
                "BLISS entry). Only change for a different beamline file layout."
            ),
        ),
        Param(
            "samz_path",
            ParamType.STR,
            "samz path",
            default="1.1/instrument/positioners/samz",
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path to the sample-Z motor position inside each scan file (under the first "
                "BLISS entry). Only change for a different beamline file layout."
            ),
        ),
        Param(
            "pixel_size_x_um",
            ParamType.FLOAT,
            "Pixel size X",
            unit="µm",
            default=0.152,
            calibration=True,
            advanced=True,
            group="Calibration",
            help=(
                "Physical size of one detector pixel along X, in µm, from the beamline optics "
                "calibration. This is what converts the sample-Y motor shift (mm) into detector "
                "pixels during alignment, so a wrong value misaligns layers along X as well as "
                "scaling every reported distance."
            ),
        ),
        Param(
            "pixel_size_y_um",
            ParamType.FLOAT,
            "Pixel size Y",
            unit="µm",
            default=0.385,
            calibration=True,
            advanced=True,
            group="Calibration",
            help=(
                "Physical size of one detector pixel along Y, in µm, from the beamline optics "
                "calibration. A wrong value skews the vertical physical scale of every rendered "
                "image and volume."
            ),
        ),
        Param(
            "samy_direction",
            ParamType.INT,
            "samy direction",
            default=-1,
            advanced=True,
            group="Alignment",
            help=(
                "Sign (+1 or −1) relating the samy motor direction to detector X. "
                "If features visibly march the wrong way between layers, flip the sign."
            ),
        ),
        Param(
            "roi_x",
            ParamType.STR,
            "Map ROI X",
            default="",
            roi_group="crop",
            roi_axis="x",
            roi_frame="map",
            help=(
                "Crop along map X as 'c0,c1' map pixels — columns of the darfix map, relative "
                "to the darfix window, NOT absolute detector pixels (blank = full width). "
                "Pre-filled from the experiment's analysis window. All volumes must share the "
                "same crop to stay co-registered."
            ),
        ),
        Param(
            "roi_y",
            ParamType.STR,
            "Map ROI Y",
            default="",
            roi_group="crop",
            roi_axis="y",
            roi_frame="map",
            help=(
                "Crop along map Y as 'r0,r1' map pixels — rows of the darfix map, relative to "
                "the darfix window, NOT absolute detector pixels (blank = full height). "
                "Pre-filled from the experiment's analysis window. All volumes must share the "
                "same crop to stay co-registered."
            ),
        ),
        Param(
            "center_method",
            ParamType.ENUM,
            "Centre method",
            default="midrange",
            choices=("midrange", "mean", "median"),
            advanced=True,
            group="Alignment",
            help=(
                "How the colour scale of the misorientation (CoM) maps is centred: "
                "midrange = midpoint of the robust limits, or mean/median of the data. "
                "Display only."
            ),
        ),
        Param(
            "range_pct",
            ParamType.FLOAT,
            "Range percentile",
            unit="%",
            default=99.5,
            advanced=True,
            group="Alignment",
            help=(
                "Robust percentile for colour limits, e.g. 99.5 ignores the most extreme "
                "0.5 % of pixels when setting the scale."
            ),
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help=(
                "Where the rendered PNGs, animation and top view are written "
                "(blank = next to the input volume)."
            ),
        ),
        Param(
            "output_format",
            ParamType.ENUM,
            "Animation format",
            default="mp4",
            choices=("mp4", "gif", "both"),
            advanced=True,
            group="Output",
            help=(
                "Animation container: mp4 needs ffmpeg on PATH; gif always works; both writes both."
            ),
        ),
        Param(
            "save_layers",
            ParamType.BOOL,
            "Save layer PNGs",
            default=True,
            advanced=True,
            group="Output",
            help="Write one PNG per layer of each volume.",
        ),
        Param(
            "save_animation",
            ParamType.BOOL,
            "Save animation",
            default=True,
            advanced=True,
            group="Output",
            help="Write the layer-by-layer animation.",
        ),
        Param(
            "save_topview",
            ParamType.BOOL,
            "Save 3D top-view",
            default=True,
            advanced=True,
            group="Output",
            help="Write the static 3-D top-view image.",
        ),
        Param(
            "save_rotation",
            ParamType.BOOL,
            "Save rotating 3-D video",
            default=False,
            advanced=True,
            group="Output",
            help=(
                "Write a movie of the 3-D volume render spinning once around "
                "(one 360° orbit; frame count from 'Rotation frames'). Uses the "
                "same opacity as the top view and the Animation format container. "
                "Slow — off by default."
            ),
        ),
        Param(
            "volume_opacity",
            ParamType.FLOAT,
            "3D opacity",
            default=0.85,
            advanced=True,
            group="Appearance",
            help=(
                "Opacity of the rendered 3-D top view and rotation video, 0–1. "
                "Scales the render's opacity in every render mode (volume mode: "
                "scales the transfer function)."
            ),
        ),
        Param(
            "render_mode",
            ParamType.ENUM,
            "3D render mode",
            default="volume",
            choices=("volume", "surface", "isosurface"),
            advanced=True,
            group="Appearance",
            help=(
                "How the 3-D top view and rotation video draw the volume: 'volume' is "
                "true volumetric rendering (shaded, transfer-function opacity), "
                "'surface' the legacy NaN-thresholded mesh, 'isosurface' stacked "
                "contour shells."
            ),
        ),
        Param(
            "opacity_mapping",
            ParamType.ENUM,
            "3D opacity mapping",
            default="linear",
            choices=("linear", "sigmoid", "geom", "geom_r"),
            advanced=True,
            group="Appearance",
            help=(
                "Opacity transfer function for volumetric 3-D rendering: linear, "
                "sigmoid (emphasises mid-range values), geom (high values), geom_r "
                "(low values). Ignored by the surface and isosurface modes."
            ),
        ),
        Param(
            "rotation_frames",
            ParamType.INT,
            "Rotation frames",
            default=180,
            advanced=True,
            group="Output",
            help="Frames in one 360-degree orbit of the rotation video (15 fps).",
        ),
        Param(
            "log_scale",
            ParamType.BOOL,
            "Log colour scale (3D)",
            default=False,
            advanced=True,
            group="Appearance",
            help=(
                "Logarithmic colour mapping for the 3-D top view and rotation video. "
                "Falls back to linear (with a note) when the colour range includes "
                "zero or negative values."
            ),
        ),
    ),
)


@dataclass
class DatasetProducts:
    name: str
    shape: tuple[int, int, int]
    vmin: float
    vmax: float
    layers_dir: str | None = None
    animation: str | None = None
    top_view: str | None = None
    rotation_video: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class VisualizeResult:
    output_dir: str = ""
    datasets: list[DatasetProducts] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Colour / range helpers (faithful port)
# -----------------------------------------------------------------------------
def _symmetric_range(data, pct=99):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return (-1.0, 1.0)
    am = float(np.percentile(np.abs(valid), pct))
    return (-am, am)


def _midrange_clim(data, pct=99.5):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return 0.0, (-1.0, 1.0)
    if pct >= 100.0:
        lo, hi = float(np.min(valid)), float(np.max(valid))
    else:
        lo, hi = (float(v) for v in np.percentile(valid, [100.0 - pct, pct]))
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo) or 1.0
    return center, (-half, half)


def _center_com_and_range(data, method, range_pct):
    method = method.lower()
    if method == "midrange":
        center, (vmin, vmax) = _midrange_clim(data, range_pct)
        return data - center, vmin, vmax
    valid = data[np.isfinite(data)]
    sub = (
        float(np.nanmean(valid))
        if method == "mean"
        else float(np.nanmedian(valid))
        if valid.size
        else 0.0
    )
    data = data - sub
    vmin, vmax = _symmetric_range(data)
    return data, vmin, vmax


def _colorbar_range(data):
    valid = data[~np.isnan(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(valid, 1)), float(np.percentile(valid, 99)))


def _display_info(dataset_name, is_strain=False):
    """(title, cbar_label, cmap_group) for a dataset; group None = not a std quantity."""
    if is_strain:
        return ("Strain (cot method)", "Strain (ε)", "strain")
    axis = (
        "χ"
        if dataset_name.startswith("chi_")
        else "μ"
        if dataset_name.startswith("mu_")
        else dataset_name
    )
    if "Center_of_mass" in dataset_name:
        return (f"{axis} Misorientation", "Misorientation (°)", "mosa_com")
    if "FWHM" in dataset_name:
        return (f"{axis} Peak Broadening", "Peak broadening (°)", "mosa_fwhm")
    return (dataset_name.replace("_", " "), "(°)", None)


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------
def load_mosa_datasets(path):
    out = {}
    with h5py.File(path, "r") as f:
        for group in ("chi", "mu"):
            if group in f:
                for ds in f[group].keys():
                    out[f"{group}_{ds.replace(' ', '_')}"] = f[group][ds][:]
    return out


def load_strain_volume(path):
    with h5py.File(path, "r") as f:
        return f["strain"][:] if "strain" in f else None


# -----------------------------------------------------------------------------
# Motor helper (shared by run(), aligned_field(), and figures())
# -----------------------------------------------------------------------------
def _read_motors(raw_root: str, pattern: str, samy_path: str, samz_path: str):
    """samy/samz positions from raw scan folders; empty arrays if unavailable."""
    if not raw_root or not pattern:
        return np.array([]), np.array([])
    folders = find_matching_folders(raw_root, pattern)
    if not folders:
        return np.array([]), np.array([])
    samy, samz, _ = extract_motor_positions(folders, samy_path, samz_path)
    return samy, samz


# -----------------------------------------------------------------------------
# Per-dataset + alignment
# -----------------------------------------------------------------------------
def _parse_pair(text):
    if text is None or str(text).strip() == "":
        return None
    parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'a,b', got {text!r}")
    return tuple(parts)


def _align(volume, samy, samz, *, scale_x, samy_direction, roi_x, roi_y):
    data = A.apply_roi_3d(volume, roi_x, roi_y)
    if len(samy) > 0:
        data = A.apply_samy_shifts_to_volume(data, samy, scale_x, samy_direction)
    if len(samz) > 0:
        data, z_pos, scale_z = A.interpolate_to_uniform_z(data, samz)
    else:
        scale_z = 2.0
        z_pos = np.arange(data.shape[0]) * scale_z
    return data, z_pos, scale_z


def _process_dataset(
    data, z_pos, scale_z, name, vmin, vmax, cmap, title, cbar, p, out_dir, style=None, group=None
):
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)
    sx, sy = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    prod = DatasetProducts(name=name, shape=tuple(data.shape), vmin=float(vmin), vmax=float(vmax))

    if p["save_layers"]:
        prod.layers_dir = Rnd.save_layer_pngs(
            data,
            z_pos,
            ds_dir,
            name,
            vmin,
            vmax,
            cmap,
            title,
            cbar,
            sx,
            sy,
            style=style,
            group=group,
        )
    if p["save_animation"]:
        prod.animation = Rnd.save_layer_animation(
            data,
            z_pos,
            os.path.join(ds_dir, f"{name}_layer_anim"),
            name,
            vmin,
            vmax,
            cmap,
            title,
            cbar,
            p["output_format"],
            sx,
            sy,
            style=style,
            group=group,
        )
    log_scale = bool(p["log_scale"])
    if log_scale and not R3.log_valid((vmin, vmax)):
        log_scale = False
        prod.notes.append("log scale skipped: colour range includes non-positive values")
    scene = R3.Scene3D(
        volume=data,
        spacing=(sx, sy, scale_z),
        mode=str(p["render_mode"]),
        cmap=cmap,
        clim=(float(vmin), float(vmax)),
        log_scale=log_scale,
        opacity=float(p["volume_opacity"]),
        opacity_mapping=str(p["opacity_mapping"]),
    )
    if p["save_topview"]:
        try:
            prod.top_view = R3.save_top_view(
                scene,
                os.path.join(ds_dir, f"{name}_top_view.png"),
                cbar_label=cbar,
                group=group,
                style=style,
            )
        except Exception as exc:  # noqa: BLE001 - no GL / pyvista issue -> note + continue
            prod.notes.append(f"3D top-view skipped: {exc}")
    if p["save_rotation"]:
        try:
            prod.rotation_video = R3.save_rotation_video(
                scene,
                os.path.join(ds_dir, f"{name}_rotation"),
                p["output_format"],
                cbar_label=cbar,
                group=group,
                style=style,
                n_frames=int(p["rotation_frames"]),
            )
            if prod.rotation_video is None:
                prod.notes.append("rotation video skipped: volume has no finite voxels")
        except Exception as exc:  # noqa: BLE001 - no GL / pyvista issue -> note + continue
            prod.notes.append(f"rotation video skipped: {exc}")
    return prod


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> VisualizeResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    style = style_from_params(p)
    scale_x = float(p["pixel_size_x_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["roi_x"]), _parse_pair(p["roi_y"])
    out_dir = p["output_dir"] or os.path.join(
        os.path.dirname(p["mosa_volume_file"] or p["strain_volume_file"] or "."),
        "aligned_volume_visualizations",
    )
    result = VisualizeResult(output_dir=out_dir)
    os.makedirs(out_dir, exist_ok=True)
    raw_root = (p["raw_root"] or "").rstrip("/")

    # --- mosaicity ---
    mosa_file = p["mosa_volume_file"]
    if mosa_file and os.path.exists(mosa_file):
        progress(0.05, "loading mosaicity volume")
        datasets = load_mosa_datasets(mosa_file)
        samy, samz = _read_motors(raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"])
        for i, (name, raw) in enumerate(datasets.items()):
            progress(0.1 + 0.4 * i / max(1, len(datasets)), f"mosaicity: {name}")
            title, cbar, group = _display_info(name)
            cmap = resolve_cmap(style, group)
            data, z_pos, scale_z = _align(
                raw, samy, samz, scale_x=scale_x, samy_direction=samy_dir, roi_x=roi_x, roi_y=roi_y
            )
            if "Center_of_mass" in name:
                data, vmin, vmax = _center_com_and_range(
                    data, p["center_method"], float(p["range_pct"])
                )
            else:
                vmin, vmax = _colorbar_range(data)
            vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
            if clim_note:
                progress(0.1 + 0.4 * i / max(1, len(datasets)), f"{name}: {clim_note}")
            prod = _process_dataset(
                data,
                z_pos,
                scale_z,
                name,
                vmin,
                vmax,
                cmap,
                title,
                cbar,
                p,
                out_dir,
                style=style,
                group=group,
            )
            if clim_note:
                prod.notes.append(clim_note)
            result.datasets.append(prod)
    elif mosa_file:
        result.skipped.append(f"mosaicity volume not found: {mosa_file}")

    # --- strain ---
    strain_file = p["strain_volume_file"]
    if strain_file and os.path.exists(strain_file):
        progress(0.6, "loading strain volume")
        vol = load_strain_volume(strain_file)
        if vol is not None:
            samy, samz = _read_motors(raw_root, p["strain_pattern"], p["samy_path"], p["samz_path"])
            title, cbar, group = _display_info("strain", is_strain=True)
            cmap = resolve_cmap(style, group)
            data, z_pos, scale_z = _align(
                vol, samy, samz, scale_x=scale_x, samy_direction=samy_dir, roi_x=roi_x, roi_y=roi_y
            )
            vmin, vmax = _symmetric_range(data)
            vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
            if clim_note:
                progress(0.6, f"strain: {clim_note}")
            prod = _process_dataset(
                data,
                z_pos,
                scale_z,
                "strain",
                vmin,
                vmax,
                cmap,
                title,
                cbar,
                p,
                out_dir,
                style=style,
                group=group,
            )
            if clim_note:
                prod.notes.append(clim_note)
            result.datasets.append(prod)
    elif strain_file:
        result.skipped.append(f"strain volume not found: {strain_file}")

    progress(1.0, f"visualized {len(result.datasets)} datasets -> {out_dir}")
    return result


# -----------------------------------------------------------------------------
# Single-field alignment (used by the GUI's lazy 3-D viewer)
# -----------------------------------------------------------------------------
def mosa_field_names(path: str) -> list[str]:
    """List the chi/mu field ids in a stacked mosaicity file (no data read)."""
    names = []
    with h5py.File(path, "r") as f:
        for grp in ("chi", "mu"):
            if grp in f:
                names.extend(f"{grp}_{ds.replace(' ', '_')}" for ds in f[grp].keys())
    return names


def available_fields(params: dict) -> list[str]:
    """Field ids that can be aligned for 3-D, given the configured volume files."""
    p = {**STAGE.defaults(), **params}
    out: list[str] = []
    if p["mosa_volume_file"] and os.path.exists(p["mosa_volume_file"]):
        out.extend(mosa_field_names(p["mosa_volume_file"]))
    if p["strain_volume_file"] and os.path.exists(p["strain_volume_file"]):
        out.append("strain")
    return out


def aligned_field(params: dict, name: str):
    """Align a single field for display. Returns (volume, spacing_xyz, cmap, clim, meta).

    Reuses the exact alignment + centering the stage applies, so the 3-D view
    matches the rendered PNGs. Heavy (loads + aligns one volume) — the GUI calls
    it only when the user asks to render. ``meta`` is
    ``{"cbar_label": str, "group": str | None}``.
    """
    p = {**STAGE.defaults(), **params}
    scale_x, scale_y = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["roi_x"]), _parse_pair(p["roi_y"])
    raw_root = (p["raw_root"] or "").rstrip("/")

    if name == "strain":
        vol = load_strain_volume(p["strain_volume_file"])
        if vol is None:
            raise KeyError("strain dataset not found")
        samy, samz = _read_motors(raw_root, p["strain_pattern"], p["samy_path"], p["samz_path"])
        data, _z, scale_z = _align(
            vol, samy, samz, scale_x=scale_x, samy_direction=samy_dir, roi_x=roi_x, roi_y=roi_y
        )
        vmin, vmax = _symmetric_range(data)
        cmap = resolve_cmap(None, "strain")
        meta = {"cbar_label": "Strain (ε)", "group": "strain"}
    else:
        datasets = load_mosa_datasets(p["mosa_volume_file"])
        if name not in datasets:
            raise KeyError(name)
        samy, samz = _read_motors(raw_root, p["mosa_pattern"], p["samy_path"], p["samz_path"])
        data, _z, scale_z = _align(
            datasets[name],
            samy,
            samz,
            scale_x=scale_x,
            samy_direction=samy_dir,
            roi_x=roi_x,
            roi_y=roi_y,
        )
        if "Center_of_mass" in name:
            data, vmin, vmax = _center_com_and_range(
                data, p["center_method"], float(p["range_pct"])
            )
        else:
            vmin, vmax = _colorbar_range(data)
        _t, label, group = _display_info(name)
        cmap = resolve_cmap(None, group)
        meta = {"cbar_label": label, "group": group}
    return data, (scale_x, scale_y, scale_z), cmap, (float(vmin), float(vmax)), meta


def _make_build(loader, z, vn, vx, cmap_group, ex, ey, t, cb):
    """Factory: returns a build(style) closure for one layer of an aligned volume.

    ``loader`` is a zero-arg callable that returns the full aligned 3-D volume
    (cached per dataset by the caller).  ``z`` is captured by value via the
    default-arg trick so late-binding is not an issue. The colormap is resolved
    from *style* at build time via the dataset's quantity group.
    """

    def build(
        style, _loader=loader, _z=z, _vn=vn, _vx=vx, _grp=cmap_group, _ex=ex, _ey=ey, _t=t, _cb=cb
    ):
        vol = _loader()
        layer = vol[_z]
        fig, _ax, _im = Rnd.layer_figure(
            layer,
            _vn,
            _vx,
            resolve_cmap(style, _grp),
            _ex,
            _ey,
            f"{_t} (layer {_z})",
            _cb,
            style=style,
            group=_grp,
        )
        return fig

    return build


@register("visualize")
def figures(result: "VisualizeResult", params: dict) -> list[FigureSpec]:
    """Return one ``map`` FigureSpec per Z layer per dataset in the VisualizeResult.

    Each ``build(style)`` closure reproduces the aligned layer exactly as the
    stage does: load the source volume → ``_align`` (ROI + samy shift + uniform-Z
    interp) → slice layer *z* → ``render.layer_figure``.  Motor data is read from
    ``params`` (``raw_root`` + pattern), so the alignment matches the original run.

    The full alignment is performed AT MOST ONCE per dataset (lazy cache): listing
    specs is cheap; the first ``build()`` call for a dataset loads and aligns its
    volume, and all subsequent layer builds for that same dataset reuse the result.

    Pixel scales fall back to the calibrated beamline defaults (0.152 / 0.385 µm)
    when not supplied in *params*.
    """
    if not result.datasets:
        return []

    p = {**STAGE.defaults(), **params}
    sx = float(p["pixel_size_x_um"])
    sy = float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x = _parse_pair(p["roi_x"])
    roi_y = _parse_pair(p["roi_y"])
    raw_root = (p["raw_root"] or "").rstrip("/")

    specs: list[FigureSpec] = []

    for ds in result.datasets:
        is_strain = ds.name == "strain"
        title, cbar_label, group = _display_info(ds.name, is_strain=is_strain)
        n_z = ds.shape[0]
        ext_x = ds.shape[2] * sx
        ext_y = ds.shape[1] * sy

        # Per-dataset lazy cache: shared across all layer builds for THIS dataset.
        # First build() call fills cache["vol"]; the rest reuse it.
        cache: dict = {}

        if is_strain:
            src_file = p["strain_volume_file"]
            pattern = p["strain_pattern"]

            def _aligned_vol(
                src=src_file,
                pat=pattern,
                _cache=cache,
                _sx=sx,
                _sd=samy_dir,
                _rx=roi_x,
                _ry=roi_y,
                _rr=raw_root,
                _sp=p["samy_path"],
                _szp=p["samz_path"],
            ):
                if "vol" not in _cache:
                    raw = load_strain_volume(src)
                    if raw is None:
                        raise ValueError(f"strain dataset not found in {src!r}")
                    samy, samz = _read_motors(_rr, pat, _sp, _szp)
                    _cache["vol"], _zp, _sz = _align(
                        raw, samy, samz, scale_x=_sx, samy_direction=_sd, roi_x=_rx, roi_y=_ry
                    )
                return _cache["vol"]

        else:
            src_file = p["mosa_volume_file"]
            ds_name = ds.name
            pattern = p["mosa_pattern"]

            def _aligned_vol(
                src=src_file,
                name=ds_name,
                pat=pattern,
                _cache=cache,
                _sx=sx,
                _sd=samy_dir,
                _rx=roi_x,
                _ry=roi_y,
                _rr=raw_root,
                _sp=p["samy_path"],
                _szp=p["samz_path"],
                _cm=p["center_method"],
                _rp=float(p["range_pct"]),
            ):
                if "vol" not in _cache:
                    all_datasets = load_mosa_datasets(src)
                    if name not in all_datasets:
                        raise KeyError(f"mosaicity dataset {name!r} not found in {src!r}")
                    raw = all_datasets[name]
                    samy, samz = _read_motors(_rr, pat, _sp, _szp)
                    vol, _zp, _sz = _align(
                        raw, samy, samz, scale_x=_sx, samy_direction=_sd, roi_x=_rx, roi_y=_ry
                    )
                    # CoM volumes are centred at run() time and ds.vmin/vmax were
                    # derived from the CENTRED data — reproduce that here so the
                    # export matches the saved PNG (and the 3-D viewer).
                    if "Center_of_mass" in name:
                        vol, _vn, _vx = _center_com_and_range(vol, _cm, _rp)
                    _cache["vol"] = vol
                return _cache["vol"]

        # Unique filename stem: sanitise the dataset name (spaces → _, slashes removed)
        stem = ds.name.replace("/", "_").replace(" ", "_")
        vmin_ds, vmax_ds = ds.vmin, ds.vmax

        for z in range(n_z):
            specs.append(
                FigureSpec(
                    figure_id=f"visualize_{stem}_z{z:04d}",
                    title=f"{title} — layer {z}",
                    kind="map",
                    filename=f"{stem}_layer_{z:04d}",
                    build=_make_build(
                        _aligned_vol, z, vmin_ds, vmax_ds, group, ext_x, ext_y, title, cbar_label
                    ),
                )
            )

    return specs


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Visualize aligned mosaicity/strain volumes.")
    ap.add_argument("--mosa-volume-file", default="")
    ap.add_argument("--strain-volume-file", default="")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--mosa-pattern", default="*")
    ap.add_argument("--strain-pattern", default="*")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--no-topview", action="store_true")
    args = ap.parse_args(argv)
    res = run(
        dict(
            mosa_volume_file=args.mosa_volume_file,
            strain_volume_file=args.strain_volume_file,
            raw_root=args.raw_root,
            mosa_pattern=args.mosa_pattern,
            strain_pattern=args.strain_pattern,
            output_dir=args.output_dir,
            save_topview=not args.no_topview,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\n{len(res.datasets)} datasets -> {res.output_dir}; skipped {len(res.skipped)}")
    return 0


def roi_previews(params: dict) -> list:
    """(label, thunk) ROI-picker previews from the stacked mosa/strain volume(s)."""
    from ..common.figures import stacked_volume_previews

    return stacked_volume_previews(params)


if __name__ == "__main__":
    raise SystemExit(_main())
