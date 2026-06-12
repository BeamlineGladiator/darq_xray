"""Visualize stage — aligned mosaicity/strain volumes to images + animation + 3D.

Ported from ``visualize_aligned_volumes_v6``. Per dataset (mosaicity chi/mu
Center-of-mass + FWHM, and strain) it:

1. aligns the stacked volume — ROI -> samy sub-pixel X-shift -> uniform-Z
   interpolation — reusing :mod:`dfxm.common.alignment` (golden-tested,
   voxel-identical to the PVTI exporter);
2. centres CoM volumes (midrange/mean/median) and picks colour limits (strain
   keeps its physical zero, symmetric limits);
3. writes per-layer PNGs, a layer-by-layer animation (MP4 with GIF fallback),
   and a 3-D top-view render (best-effort — skipped gracefully without a GL
   context).

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
        "animation, and a 3-D top view."
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
                "Physical size of one detector pixel along X, in µm — sets the lateral scale of "
                "the volumes. From the beamline optics calibration."
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
                "Physical size of one detector pixel along Y, in µm — sets the vertical scale of "
                "the volumes. From the beamline optics calibration."
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
            "ROI X",
            default="",
            help=(
                "Crop along detector X as 'x0,x1' in pixels (blank = full width). "
                "All volumes must share the same crop to stay co-registered."
            ),
        ),
        Param(
            "roi_y",
            ParamType.STR,
            "ROI Y",
            default="",
            help=(
                "Crop along detector Y as 'y0,y1' in pixels (blank = full height). "
                "All volumes must share the same crop to stay co-registered."
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
            "volume_opacity",
            ParamType.FLOAT,
            "3D opacity",
            default=0.85,
            advanced=True,
            group="Appearance",
            help="Opacity of the rendered 3-D top view, 0–1.",
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
    if is_strain:
        return ("Strain (cot method)", "Strain (ε)", "RdBu_r")
    axis = (
        "χ"
        if dataset_name.startswith("chi_")
        else "μ"
        if dataset_name.startswith("mu_")
        else dataset_name
    )
    if "Center_of_mass" in dataset_name:
        return (f"{axis} Misorientation", "Misorientation (°)", "magma")
    if "FWHM" in dataset_name:
        return (f"{axis} Peak Broadening", "Peak broadening (°)", "magma")
    return (dataset_name.replace("_", " "), "(°)", "magma")


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


def _process_dataset(data, z_pos, scale_z, name, vmin, vmax, cmap, title, cbar, p, out_dir):
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)
    sx, sy = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    prod = DatasetProducts(name=name, shape=tuple(data.shape), vmin=float(vmin), vmax=float(vmax))

    if p["save_layers"]:
        prod.layers_dir = Rnd.save_layer_pngs(
            data, z_pos, ds_dir, name, vmin, vmax, cmap, title, cbar, sx, sy
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
        )
    if p["save_topview"]:
        try:
            prod.top_view = Rnd.save_top_view(
                data,
                scale_z,
                sx,
                sy,
                vmin,
                vmax,
                cmap,
                float(p["volume_opacity"]),
                os.path.join(ds_dir, f"{name}_top_view.png"),
            )
        except Exception as exc:  # noqa: BLE001 - no GL / pyvista issue -> note + continue
            prod.notes.append(f"3D top-view skipped: {exc}")
    return prod


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> VisualizeResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
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

    def _motors(pattern):
        if not raw_root or not pattern:
            return np.array([]), np.array([])
        folders = find_matching_folders(raw_root, pattern)
        if not folders:
            return np.array([]), np.array([])
        samy, samz, _ = extract_motor_positions(folders, p["samy_path"], p["samz_path"])
        return samy, samz

    # --- mosaicity ---
    mosa_file = p["mosa_volume_file"]
    if mosa_file and os.path.exists(mosa_file):
        progress(0.05, "loading mosaicity volume")
        datasets = load_mosa_datasets(mosa_file)
        samy, samz = _motors(p["mosa_pattern"])
        for i, (name, raw) in enumerate(datasets.items()):
            progress(0.1 + 0.4 * i / max(1, len(datasets)), f"mosaicity: {name}")
            title, cbar, cmap = _display_info(name)
            data, z_pos, scale_z = _align(
                raw, samy, samz, scale_x=scale_x, samy_direction=samy_dir, roi_x=roi_x, roi_y=roi_y
            )
            if "Center_of_mass" in name:
                data, vmin, vmax = _center_com_and_range(
                    data, p["center_method"], float(p["range_pct"])
                )
            else:
                vmin, vmax = _colorbar_range(data)
            result.datasets.append(
                _process_dataset(
                    data, z_pos, scale_z, name, vmin, vmax, cmap, title, cbar, p, out_dir
                )
            )
    elif mosa_file:
        result.skipped.append(f"mosaicity volume not found: {mosa_file}")

    # --- strain ---
    strain_file = p["strain_volume_file"]
    if strain_file and os.path.exists(strain_file):
        progress(0.6, "loading strain volume")
        vol = load_strain_volume(strain_file)
        if vol is not None:
            samy, samz = _motors(p["strain_pattern"])
            title, cbar, cmap = _display_info("strain", is_strain=True)
            data, z_pos, scale_z = _align(
                vol, samy, samz, scale_x=scale_x, samy_direction=samy_dir, roi_x=roi_x, roi_y=roi_y
            )
            vmin, vmax = _symmetric_range(data)
            result.datasets.append(
                _process_dataset(
                    data, z_pos, scale_z, "strain", vmin, vmax, cmap, title, cbar, p, out_dir
                )
            )
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
    """Align a single field for display. Returns (volume, spacing_xyz, cmap, clim).

    Reuses the exact alignment + centering the stage applies, so the 3-D view
    matches the rendered PNGs. Heavy (loads + aligns one volume) — the GUI calls
    it only when the user asks to render.
    """
    p = {**STAGE.defaults(), **params}
    scale_x, scale_y = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["roi_x"]), _parse_pair(p["roi_y"])
    raw_root = (p["raw_root"] or "").rstrip("/")

    def motors(pattern):
        if not raw_root or not pattern:
            return np.array([]), np.array([])
        folders = find_matching_folders(raw_root, pattern)
        if not folders:
            return np.array([]), np.array([])
        samy, samz, _ = extract_motor_positions(folders, p["samy_path"], p["samz_path"])
        return samy, samz

    if name == "strain":
        vol = load_strain_volume(p["strain_volume_file"])
        if vol is None:
            raise KeyError("strain dataset not found")
        samy, samz = motors(p["strain_pattern"])
        data, _z, scale_z = _align(
            vol, samy, samz, scale_x=scale_x, samy_direction=samy_dir, roi_x=roi_x, roi_y=roi_y
        )
        vmin, vmax = _symmetric_range(data)
        cmap = "RdBu_r"
    else:
        datasets = load_mosa_datasets(p["mosa_volume_file"])
        if name not in datasets:
            raise KeyError(name)
        samy, samz = motors(p["mosa_pattern"])
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
        _, _, cmap = _display_info(name)
    return data, (scale_x, scale_y, scale_z), cmap, (float(vmin), float(vmax))


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


if __name__ == "__main__":
    raise SystemExit(_main())
