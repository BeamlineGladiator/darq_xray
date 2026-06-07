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
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from ..common import alignment as A
from ..common import plotting as P
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
        "Align stacked mosaicity/strain volumes (ROI -> samy shift -> uniform Z) "
        "and render per-layer PNGs, a layer animation, and a 3D top-view."
    ),
    params=(
        Param(
            "mosa_volume_file",
            ParamType.PATH,
            "Mosaicity volume",
            help="stacked_volumes.h5 (blank to skip)",
        ),
        Param(
            "strain_volume_file",
            ParamType.PATH,
            "Strain volume",
            help="stacked_strain_volumes.h5 (blank to skip)",
        ),
        Param("raw_root", ParamType.DIR, "Raw data root", help="for samy/samz motor positions"),
        Param("mosa_pattern", ParamType.STR, "Mosaicity raw pattern", default="*"),
        Param("strain_pattern", ParamType.STR, "Strain raw pattern", default="*"),
        Param("samy_path", ParamType.STR, "samy path", default="1.1/instrument/positioners/samy"),
        Param("samz_path", ParamType.STR, "samz path", default="1.1/instrument/positioners/samz"),
        Param(
            "pixel_size_x_um",
            ParamType.FLOAT,
            "Pixel size X",
            unit="µm",
            default=0.152,
            calibration=True,
        ),
        Param(
            "pixel_size_y_um",
            ParamType.FLOAT,
            "Pixel size Y",
            unit="µm",
            default=0.385,
            calibration=True,
        ),
        Param(
            "samy_direction",
            ParamType.INT,
            "samy direction",
            default=-1,
            help="+1 or -1 (motor sign convention)",
        ),
        Param("roi_x", ParamType.STR, "ROI X", default="", help="x0,x1 (blank = full)"),
        Param("roi_y", ParamType.STR, "ROI Y", default="", help="y0,y1 (blank = full)"),
        Param(
            "center_method",
            ParamType.ENUM,
            "Centre method",
            default="midrange",
            choices=("midrange", "mean", "median"),
        ),
        Param("range_pct", ParamType.FLOAT, "Range percentile", default=99.5),
        Param("output_dir", ParamType.DIR, "Output dir"),
        Param(
            "output_format",
            ParamType.ENUM,
            "Animation format",
            default="mp4",
            choices=("mp4", "gif", "both"),
        ),
        Param("save_layers", ParamType.BOOL, "Save layer PNGs", default=True),
        Param("save_animation", ParamType.BOOL, "Save animation", default=True),
        Param("save_topview", ParamType.BOOL, "Save 3D top-view", default=True),
        Param("volume_opacity", ParamType.FLOAT, "3D opacity", default=0.85),
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


def _cmap_nan_transparent(name):
    cmap = P.get_cmap(name).copy()
    cmap.set_bad(color="white", alpha=0.0)
    return cmap


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


def _add_scale_bar(ax, ext_x, ext_y, color="black"):
    target = ext_x * 0.15
    if target >= 100:
        sl = round(target / 50) * 50
    elif target >= 10:
        sl = round(target / 10) * 10
    elif target >= 1:
        sl = round(target)
    else:
        sl = round(target, 1)
    sl = sl or target
    bx, by, bh = ext_x * 0.95 - sl, ext_y * 0.05, ext_y * 0.01
    ax.add_patch(Rectangle((bx, by), sl, bh, facecolor=color, edgecolor=color))
    ax.text(
        bx + sl / 2,
        by + bh * 3,
        f"{sl:.0f} µm",
        color=color,
        fontsize=10,
        ha="center",
        va="bottom",
        fontweight="bold",
    )


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
# Rendering
# -----------------------------------------------------------------------------
def _layer_figure(layer, vmin, vmax, cmap, ext_x, ext_y, title, cbar_label):
    fig = Figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111)
    im = ax.imshow(
        layer,
        cmap=_cmap_nan_transparent(cmap),
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        extent=[0, ext_x, 0, ext_y],
        origin="lower",
        aspect="equal",
    )
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label)
    _add_scale_bar(ax, ext_x, ext_y)
    return fig, ax, im


def _save_layers(volume, z_um, out_dir, name, vmin, vmax, cmap, title, cbar, sx, sy):
    layers_dir = os.path.join(out_dir, f"{name}_layers")
    os.makedirs(layers_dir, exist_ok=True)
    ext_x, ext_y = volume.shape[2] * sx, volume.shape[1] * sy
    z_size = volume.shape[0]
    for z in range(z_size):
        full_title = f"{title}\nZ = {z_um[z]:.2f} µm (Layer {z}/{z_size - 1})"
        fig, _, _ = _layer_figure(volume[z], vmin, vmax, cmap, ext_x, ext_y, full_title, cbar)
        fig.savefig(
            os.path.join(layers_dir, f"layer_{z:04d}.png"),
            dpi=150,
            facecolor="white",
            bbox_inches="tight",
        )
    return layers_dir


def _save_animation(volume, z_um, base_path, name, vmin, vmax, cmap, title, cbar, fmt, sx, sy):
    ext_x, ext_y = volume.shape[2] * sx, volume.shape[1] * sy
    z_size = volume.shape[0]
    fig, ax, im = _layer_figure(volume[0], vmin, vmax, cmap, ext_x, ext_y, title, cbar)
    title_obj = ax.set_title(f"{title}\nZ = {z_um[0]:.2f} µm (Layer 0/{z_size - 1})")

    def update(frame):
        z = frame % z_size
        im.set_data(volume[z])
        title_obj.set_text(f"{title}\nZ = {z_um[z]:.2f} µm (Layer {z}/{z_size - 1})")
        return [im, title_obj]

    anim = FuncAnimation(fig, update, frames=z_size, blit=False)
    written = None
    want_mp4 = fmt in ("mp4", "both")
    want_gif = fmt in ("gif", "both")
    if want_mp4:
        try:
            anim.save(base_path + ".mp4", writer=FFMpegWriter(fps=15), dpi=120)
            written = base_path + ".mp4"
        except Exception:  # noqa: BLE001 - ffmpeg missing -> fall back to GIF
            want_gif = True
    if want_gif:
        anim.save(base_path + ".gif", writer=PillowWriter(fps=15), dpi=120)
        written = written or base_path + ".gif"
    return written


def _create_pyvista_grid(data, spacing):
    import pyvista as pv

    dt = np.transpose(data, (2, 1, 0))
    finite = dt[np.isfinite(dt)]
    sentinel = (
        (float(np.min(finite)) - 1000.0 * (float(np.ptp(finite)) + 1.0)) if finite.size else -1e30
    )
    dc = np.where(np.isfinite(dt), dt, sentinel)
    grid = pv.ImageData()
    grid.dimensions = np.array(dc.shape) + 1
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.cell_data["values"] = dc.flatten(order="F")
    thresh = sentinel * 0.5 if sentinel < 0 else sentinel + 1.0
    return grid.threshold(value=thresh, scalars="values")


def _save_top_view(volume, scale_z, sx, sy, vmin, vmax, cmap, opacity, path):
    import pyvista as pv

    pv.OFF_SCREEN = True
    grid = _create_pyvista_grid(volume, spacing=(sx, sy, scale_z))
    if grid.n_cells == 0:
        return None
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(
        grid,
        scalars="values",
        cmap=cmap,
        clim=[vmin, vmax],
        opacity=opacity,
        smooth_shading=True,
        show_edges=False,
    )
    pl.view_xy()
    pl.screenshot(path)
    pl.close()
    return path


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
        prod.layers_dir = _save_layers(
            data, z_pos, ds_dir, name, vmin, vmax, cmap, title, cbar, sx, sy
        )
    if p["save_animation"]:
        prod.animation = _save_animation(
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
            prod.top_view = _save_top_view(
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
