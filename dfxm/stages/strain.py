"""Strain stage — per-pixel axial strain (cot method, ccmth-only) + 3D stacking.

Port of the legacy ``calc_axial_strain_v7_batch`` calculator:

    ε = cot(ccmth_ref) · Δccmth

Pipeline per layer (order is a physics constraint — **detrend before ROI**):

1. load the ccmth Center-of-mass map from maps.h5;
2. detrend ccmth on the *full* map (separable 2-D arctan);
3. crop the ROI;
4. compute strain;
5. stack all layers into a 3-D volume.

Plotting uses the explicit Figure/Agg API (no pyplot) so this module is safe
to import in the Qt GUI process.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

import h5py
import numpy as np
from scipy.optimize import curve_fit

from ..common.plotting import new_figure, physical_extent, symmetric_limits
from ..common.sort import find_matching_folders
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


STAGE = StageSpec(
    name="strain",
    label="Axial strain",
    description=(
        "Per-pixel axial strain from ccmth COM (cot method, ccmth-only): detrend ccmth, "
        "crop ROI, compute strain, and stack layers into a 3D volume."
    ),
    params=(
        Param("mode", ParamType.ENUM, "Mode", default="batch", choices=("single", "batch")),
        Param(
            "input_folder",
            ParamType.DIR,
            "Input folder",
            help="folder holding maps.h5 (single mode)",
        ),
        Param(
            "root_folder", ParamType.DIR, "Root folder", help="parent of layer folders (batch mode)"
        ),
        Param("folder_pattern", ParamType.STR, "Folder pattern", default="*"),
        Param("maps_filename", ParamType.STR, "maps filename", default="maps.h5"),
        Param(
            "ccmth_com_path",
            ParamType.STR,
            "ccmth COM path",
            default="/entry/ccmth/Center of mass/Center of mass",
        ),
        Param(
            "ccmth_ref_deg",
            ParamType.FLOAT,
            "ccmth reference",
            unit="deg",
            default=7.144,
            calibration=True,
        ),
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
        Param("roi", ParamType.STR, "ROI", default="", help="r0,r1,c0,c1 (blank = full image)"),
        Param("vmin", ParamType.STR, "vmin", default="", help="colour-limit min (blank = auto)"),
        Param("vmax", ParamType.STR, "vmax", default="", help="colour-limit max (blank = auto)"),
        Param("output_dir", ParamType.DIR, "Output dir", help="where plots + volume are written"),
        Param(
            "stacked_filename",
            ParamType.STR,
            "Stacked filename",
            default="stacked_strain_volumes.h5",
        ),
        Param("save_plots", ParamType.BOOL, "Save plots", default=True),
    ),
)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
@dataclass
class LayerResult:
    name: str
    shape: tuple[int, int]
    vmin: float
    vmax: float
    mean: float
    std: float
    plots: list[str] = field(default_factory=list)


@dataclass
class StrainResult:
    stacked_path: str | None = None
    volume_shape: tuple[int, int, int] | None = None
    output_dir: str = ""
    layers: list[LayerResult] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)


# -----------------------------------------------------------------------------
# Numeric core (faithful port)
# -----------------------------------------------------------------------------
def cot(angle_rad):
    return np.cos(angle_rad) / np.sin(angle_rad)


def _arctan_model(x, a, b, c, d):
    return a * np.arctan(b * (x - c)) + d


def _fit_arctan_1d(coords: np.ndarray, profile: np.ndarray):
    mid = coords[len(coords) // 2]
    p0 = [(profile[-1] - profile[0]) / np.pi, 4.0 / len(coords), mid, np.nanmedian(profile)]
    bounds_lo = [-np.inf, 0.0, coords[0], -np.inf]
    bounds_hi = [np.inf, np.inf, coords[-1], np.inf]
    try:
        popt, _ = curve_fit(
            _arctan_model, coords, profile, p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=20000
        )
    except RuntimeError:
        popt = np.array(p0)
    return popt


def detrend_arctan_2d(map_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Separable 2-D arctan detrend; preserves the original absolute level."""
    ny, nx = map_2d.shape
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    original_median = np.nanmedian(map_2d)

    profile_x = np.nanmedian(map_2d, axis=0)
    finite_x = np.isfinite(profile_x)
    popt_x = _fit_arctan_1d(xs[finite_x], profile_x[finite_x])
    arctan_x_1d_noc = _arctan_model(xs, *popt_x) - popt_x[3]
    residual_1 = map_2d - arctan_x_1d_noc[np.newaxis, :]

    profile_y = np.nanmedian(residual_1, axis=1)
    finite_y = np.isfinite(profile_y)
    popt_y = _fit_arctan_1d(ys[finite_y], profile_y[finite_y])
    arctan_y_1d_noc = _arctan_model(ys, *popt_y) - popt_y[3]
    residual_2 = residual_1 - arctan_y_1d_noc[:, np.newaxis]

    residual_2 += original_median - np.nanmedian(residual_2)
    surface = arctan_x_1d_noc[np.newaxis, :] + arctan_y_1d_noc[:, np.newaxis]
    return residual_2, surface


def apply_roi(map_2d: np.ndarray, roi: list | None) -> np.ndarray:
    if roi is None:
        return map_2d
    r0, r1, c0, c1 = roi
    return map_2d[r0:r1, c0:c1]


def compute_strain(
    ccmth_map_deg: np.ndarray,
    ccmth_ref_deg: float,
) -> np.ndarray:
    """Per-pixel axial strain (cot method), ccmth-only.

    ``ε = cot(ccmth_ref) · (ccmth − ccmth_ref)`` with angles converted to radians.
    """
    ccmth_rad = np.deg2rad(ccmth_map_deg)
    ccmth_ref_rad = np.deg2rad(ccmth_ref_deg)
    return cot(ccmth_ref_rad) * (ccmth_rad - ccmth_ref_rad)


def load_map(filepath: str, dataset_path: str) -> np.ndarray:
    with h5py.File(filepath, "r") as f:
        if dataset_path not in f:
            raise KeyError(f"dataset {dataset_path!r} not found in {filepath!r}")
        return f[dataset_path][:]


# -----------------------------------------------------------------------------
# Plotting (Figure/Agg API — no pyplot)
# -----------------------------------------------------------------------------
def _save_strain_map(strain, px, py, roi, vlim, path):
    extent = physical_extent(strain.shape, px, py, roi)
    vmin, vmax = vlim if vlim != (None, None) else symmetric_limits(strain)
    fig = new_figure((7, 7 * (strain.shape[0] * py) / (strain.shape[1] * px) + 1.5))
    ax = fig.add_subplot(111)
    im = ax.imshow(
        strain,
        origin="lower",
        extent=extent,
        aspect="equal",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046).set_label("Strain (ε)")
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title("Strain map (cot method)")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")


def _save_histogram(data, path, title="Strain distribution", xlabel="Strain (ε)"):
    valid = data[np.isfinite(data)].ravel()
    if valid.size == 0:
        return
    fig = new_figure((8, 5))
    ax = fig.add_subplot(111)
    ax.hist(valid, bins=200, color="steelblue", alpha=0.85)
    ax.axvline(valid.mean(), color="red", ls="--", lw=1.5, label=f"mean = {valid.mean():.3e}")
    ax.axvline(
        np.median(valid), color="orange", ls="--", lw=1.5, label=f"median = {np.median(valid):.3e}"
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Pixel count")
    ax.set_title(title)
    ax.legend()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")


def _save_detrend_diag(original, detrended, surface, path):
    fig = new_figure((20, 6))
    axes = fig.subplots(1, 3)
    for ax, title, d in zip(
        axes,
        ["Original (after ROI)", "Arctan surface", "Detrended"],
        [original, surface, detrended],
    ):
        valid = d[np.isfinite(d)]
        if valid.size == 0:
            continue
        vlo, vhi = np.percentile(valid, [1, 99])
        im = ax.imshow(d, origin="lower", cmap="RdBu_r", vmin=vlo, vmax=vhi, aspect="auto")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=120, bbox_inches="tight", facecolor="white")


# -----------------------------------------------------------------------------
# Per-folder processing
# -----------------------------------------------------------------------------
def process_maps_file(
    maps_path: str,
    name: str,
    *,
    ccmth_com_path: str,
    ccmth_ref_deg: float,
    pixel_size_x_um: float,
    pixel_size_y_um: float,
    roi: list | None,
    vlim: tuple[float | None, float | None],
    out_dir: str | None,
    save_plots: bool,
) -> tuple[np.ndarray, LayerResult]:
    """Compute the 2-D strain map for one maps.h5 and (optionally) save plots."""
    ccmth_map = load_map(maps_path, ccmth_com_path)

    # detrend ccmth on the FULL map, THEN crop ROI (order matters)
    ccmth_original = ccmth_map.copy()
    ccmth_map, surface = detrend_arctan_2d(ccmth_map)
    ccmth_map = apply_roi(ccmth_map, roi)
    surface = apply_roi(surface, roi)
    ccmth_original = apply_roi(ccmth_original, roi)

    strain = compute_strain(ccmth_map, ccmth_ref_deg)

    plots: list[str] = []
    if save_plots and out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"{name}_strain.png")
        _save_strain_map(strain, pixel_size_x_um, pixel_size_y_um, roi, vlim, p)
        plots.append(p)
        ph = os.path.join(out_dir, f"{name}_hist.png")
        _save_histogram(strain, ph)
        plots.append(ph)
        pd = os.path.join(out_dir, f"{name}_detrend_diag.png")
        _save_detrend_diag(ccmth_original, ccmth_map, surface, pd)
        plots.append(pd)

    layer = LayerResult(
        name=name,
        shape=tuple(strain.shape),
        vmin=float(np.nanmin(strain)),
        vmax=float(np.nanmax(strain)),
        mean=float(np.nanmean(strain)),
        std=float(np.nanstd(strain)),
        plots=plots,
    )
    return strain, layer


def save_stacked_volume(path, slices, names, attrs, compression="gzip"):
    volume = np.stack(slices, axis=0)
    with h5py.File(path, "w") as f:
        kw = {}
        if compression:
            kw["compression"] = compression
            if compression == "gzip":
                kw["compression_opts"] = 4
        f.create_dataset("strain", data=volume, **kw)
        f.attrs["num_layers"] = len(names)
        f.attrs["source_folders"] = "\n".join(names)
        for k, v in attrs.items():
            f.attrs[k] = v
    return volume.shape


# -----------------------------------------------------------------------------
# Helpers for string params
# -----------------------------------------------------------------------------
def _parse_roi(text) -> list | None:
    if text is None or str(text).strip() == "":
        return None
    parts = [int(p) for p in str(text).replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError(f"ROI must be 'r0,r1,c0,c1', got {text!r}")
    return parts


def _parse_float(text) -> float | None:
    if text is None or str(text).strip() == "":
        return None
    return float(text)


# -----------------------------------------------------------------------------
# Stage entry point
# -----------------------------------------------------------------------------
def run(params: dict, progress: ProgressFn | None = None) -> StrainResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    roi = _parse_roi(p["roi"])
    vlim = (_parse_float(p["vmin"]), _parse_float(p["vmax"]))
    maps_filename = p["maps_filename"]

    # resolve the (folder_name, maps_path) work list
    if p["mode"] == "single":
        folder = p["input_folder"]
        if not folder:
            raise ValueError("single mode requires 'input_folder'")
        work = [(os.path.basename(folder.rstrip("/")), os.path.join(folder, maps_filename))]
        default_out_root = folder
    else:
        root = (p["root_folder"] or "").rstrip("/")
        if not root:
            raise ValueError("batch mode requires 'root_folder'")
        folders = find_matching_folders(root, p["folder_pattern"])
        if not folders:
            raise ValueError(f"no folders matching {p['folder_pattern']!r} in {root}")
        work = [(os.path.basename(f), os.path.join(f, maps_filename)) for f in folders]
        default_out_root = root

    out_dir = p["output_dir"] or os.path.join(default_out_root, "strain_maps")
    result = StrainResult(output_dir=out_dir)

    slices: list[np.ndarray] = []
    names: list[str] = []
    for i, (name, maps_path) in enumerate(work):
        progress(i / len(work), f"strain: {name}")
        if not os.path.exists(maps_path):
            result.skipped.append(name)
            continue
        try:
            strain, layer = process_maps_file(
                maps_path,
                name,
                ccmth_com_path=p["ccmth_com_path"],
                ccmth_ref_deg=float(p["ccmth_ref_deg"]),
                pixel_size_x_um=float(p["pixel_size_x_um"]),
                pixel_size_y_um=float(p["pixel_size_y_um"]),
                roi=roi,
                vlim=vlim,
                out_dir=out_dir,
                save_plots=bool(p["save_plots"]),
            )
        except (KeyError, OSError, ValueError) as exc:
            result.skipped.append(f"{name}: {exc}")
            continue
        slices.append(strain)
        names.append(name)
        result.layers.append(layer)

    if not slices:
        progress(1.0, "no strain layers produced")
        return result

    shapes = {s.shape for s in slices}
    if len(shapes) > 1:
        raise ValueError(f"strain maps have differing shapes {shapes}; fix ROI")

    stacked_path = os.path.join(default_out_root, p["stacked_filename"])
    attrs = dict(
        description="Stacked 3D strain volume (cot, ccmth-only)",
        ccmth_ref_deg=float(p["ccmth_ref_deg"]),
        scale_x_um=float(p["pixel_size_x_um"]),
        scale_y_um=float(p["pixel_size_y_um"]),
    )
    shape = save_stacked_volume(stacked_path, slices, names, attrs)
    result.stacked_path = stacked_path
    result.volume_shape = shape
    progress(1.0, f"stacked {len(slices)} layers -> {os.path.basename(stacked_path)}")
    return result


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Axial strain (cot method).")
    ap.add_argument("--mode", choices=("single", "batch"), default="batch")
    ap.add_argument("--input-folder", default="")
    ap.add_argument("--root-folder", default="")
    ap.add_argument("--folder-pattern", default="*")
    ap.add_argument("--ccmth-ref", type=float, default=7.144)
    ap.add_argument("--roi", default="")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)

    res = run(
        dict(
            mode=args.mode,
            input_folder=args.input_folder,
            root_folder=args.root_folder,
            folder_pattern=args.folder_pattern,
            ccmth_ref_deg=args.ccmth_ref,
            roi=args.roi,
            save_plots=not args.no_plots,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\n{res.n_layers} layers; stacked -> {res.stacked_path}; skipped {len(res.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
