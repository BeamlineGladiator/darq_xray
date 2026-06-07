"""Oblique slices stage — arbitrary planar cuts through aligned DFXM volumes.

Faithful port of ``extract_oblique_slices_v5.py``. It samples arbitrary planes
(defined by a normal + origin in physical µm, optionally swept along the normal)
from two volume categories and writes a consolidated ``oblique_slices.h5`` (the
file the line-profile stage consumes) plus a PNG per plane:

* ``source = "stacked"`` — mosaicity/strain stacks; this stage runs the SAME
  alignment as the visualize/PVTI stages (reusing :mod:`dfxm.common.alignment`)
  so the slices live in the origin-0 PVTI world frame.
* ``source = "aligned"`` — already-aligned raw rocking volumes; loaded directly
  with their stored spacing/attrs.

The set of volumes to slice is the standard pipeline output (toggled on/off);
the planes themselves are given as JSON (``slices_json``) — natural since
oblique normals/origins are typically copied from ParaView. ``extent="auto"``
fits a plane (and its sweep) to the common data bounding box across all volumes.
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

from ..common import alignment as A
from ..common import render as Rnd
from ..common.raster import extract_motor_positions
from ..common.sort import find_matching_folders
from ..config.models import Param, ParamType, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


_DEFAULT_SLICES = json.dumps(
    [
        {
            "name": "z_sweep",
            "normal": [0, 0, 1],
            "origin": [0, 0, 0],
            "extent": "auto",
            "sweep_step_um": 5.0,
        },
        {
            "name": "oblique_full",
            "normal": [0.647648, 0, 0.761939],
            "origin": [0, 0, 0],
            "extent": "auto",
            "sweep_step_um": 2.0,
        },
    ],
    indent=2,
)

# Standard sliceable volumes: (toggle param, source, file param, dataset, kind, cmap)
_STD_VOLUMES = (
    (
        "include_mosa_com_chi",
        "stacked",
        "mosa_volume_file",
        "chi/Center of mass",
        "mosa_com",
        "fast",
    ),
    ("include_mosa_fwhm_chi", "stacked", "mosa_volume_file", "chi/FWHM", "mosa_fwhm", "magma"),
    ("include_mosa_com_mu", "stacked", "mosa_volume_file", "mu/Center of mass", "mosa_com", "fast"),
    ("include_mosa_fwhm_mu", "stacked", "mosa_volume_file", "mu/FWHM", "mosa_fwhm", "magma"),
    ("include_strain", "stacked", "strain_volume_file", "strain", "strain", "RdBu_r"),
    ("include_raw_sum", "aligned", "aligned_rocking_file", "sum_intensity", "raw_sum", "gray"),
    (
        "include_raw_specific",
        "aligned",
        "aligned_rocking_file",
        "specific_frame",
        "raw_specific",
        "gray",
    ),
)


STAGE = StageSpec(
    name="slices",
    label="Oblique slices",
    description=(
        "Extract arbitrary planar slices (defined in physical µm, optionally "
        "swept) from aligned mosaicity/strain/rocking volumes -> oblique_slices.h5 + PNGs."
    ),
    params=(
        Param("mosa_volume_file", ParamType.PATH, "Mosaicity volume", help="stacked_volumes.h5"),
        Param(
            "strain_volume_file", ParamType.PATH, "Strain volume", help="stacked_strain_volumes.h5"
        ),
        Param(
            "aligned_rocking_file",
            ParamType.PATH,
            "Aligned rocking volume",
            help="aligned_raw_rocking_volumes.h5",
        ),
        Param("raw_root", ParamType.DIR, "Raw data root", help="for samy/samz of stacked volumes"),
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
        Param("samy_direction", ParamType.INT, "samy direction", default=-1),
        Param(
            "align_roi_x", ParamType.STR, "Align ROI X", default="", help="x0,x1 (match the export)"
        ),
        Param(
            "align_roi_y", ParamType.STR, "Align ROI Y", default="", help="y0,y1 (match the export)"
        ),
        Param("abs_fwhm", ParamType.BOOL, "abs() FWHM", default=True),
        Param(
            "center_method",
            ParamType.ENUM,
            "Centre method",
            default="midrange",
            choices=("midrange", "mean", "median"),
        ),
        Param("range_pct", ParamType.FLOAT, "Range percentile", default=99.5),
        Param("include_mosa_com_chi", ParamType.BOOL, "Slice χ misorientation", default=True),
        Param("include_mosa_fwhm_chi", ParamType.BOOL, "Slice χ FWHM", default=True),
        Param("include_mosa_com_mu", ParamType.BOOL, "Slice μ misorientation", default=True),
        Param("include_mosa_fwhm_mu", ParamType.BOOL, "Slice μ FWHM", default=True),
        Param("include_strain", ParamType.BOOL, "Slice strain", default=True),
        Param("include_raw_sum", ParamType.BOOL, "Slice raw sum", default=True),
        Param("include_raw_specific", ParamType.BOOL, "Slice raw specific", default=True),
        Param("slices_json", ParamType.STR, "Slices (JSON)", default=_DEFAULT_SLICES),
        Param("output_dir", ParamType.DIR, "Output dir"),
        Param("output_h5_name", ParamType.STR, "Output filename", default="oblique_slices.h5"),
        Param("save_png", ParamType.BOOL, "Save PNGs", default=True),
    ),
)


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------
@dataclass
class SlicesResult:
    output_dir: str = ""
    output_h5: str | None = None
    volume_ids: list[str] = field(default_factory=list)
    slice_names: list[str] = field(default_factory=list)
    n_planes_total: int = 0
    pngs: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Centering / colour-range helpers (faithful port)
# -----------------------------------------------------------------------------
def _center_offset(data, method):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return data, 0.0
    val = float(np.nanmean(valid)) if method == "mean" else float(np.nanmedian(valid))
    return data - val, val


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


def _percentile_range(data, lo=1, hi=99):
    valid = data[np.isfinite(data)]
    if valid.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(valid, lo)), float(np.percentile(valid, hi)))


def _parse_pair(text):
    if text is None or str(text).strip() == "":
        return None
    parts = [int(v) for v in str(text).replace(" ", "").split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'a,b', got {text!r}")
    return tuple(parts)


# -----------------------------------------------------------------------------
# Geometry + sampling (faithful port)
# -----------------------------------------------------------------------------
def build_basis(normal, up=None):
    """Orthonormal (u_hat, v_hat, n_hat) for the plane; vectors are (X, Y, Z)."""
    n = np.asarray(normal, dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        raise ValueError("normal vector has zero length")
    n_hat = n / nn
    if up is None:
        up_vec = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(n_hat, up_vec)) > 0.99:
            up_vec = np.array([0.0, 1.0, 0.0])
    else:
        up_vec = np.asarray(up, dtype=np.float64)
    v = up_vec - np.dot(up_vec, n_hat) * n_hat
    vn = np.linalg.norm(v)
    if vn < 1e-12:
        tmp = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(tmp, n_hat)) > 0.99:
            tmp = np.array([0.0, 1.0, 0.0])
        v = tmp - np.dot(tmp, n_hat) * n_hat
        vn = np.linalg.norm(v)
    v_hat = v / vn
    u_hat = np.cross(v_hat, n_hat)
    u_hat = u_hat / np.linalg.norm(u_hat)
    return u_hat, v_hat, n_hat


def slice_plane_offsets(sl):
    step = sl.get("sweep_step_um")
    if not step or step <= 0:
        return np.array([0.0])
    start = float(sl.get("sweep_start_um", 0.0))
    stop = float(sl.get("sweep_stop_um", start))
    if stop < start:
        start, stop = stop, start
    n = int(np.floor((stop - start) / step + 1e-9)) + 1
    return start + np.arange(n) * step


def sample_plane(prep, plane_origin, u_hat, v_hat, half_u, half_v, du, dv):
    """Sample one plane centred at plane_origin (X,Y,Z µm). Returns (slice, u_um, v_um)."""
    nu = max(1, int(np.round(2.0 * half_u / du)) + 1)
    nv = max(1, int(np.round(2.0 * half_v / dv)) + 1)
    u_um = np.linspace(-half_u, half_u, nu)
    v_um = np.linspace(-half_v, half_v, nv)
    uu, vv = np.meshgrid(u_um, v_um)
    pts = (
        np.asarray(plane_origin, np.float64)[None, None, :]
        + uu[..., None] * u_hat[None, None, :]
        + vv[..., None] * v_hat[None, None, :]
    )
    i = pts[..., 0] / prep["scale_x"] + prep["x_ref_shift_px"]
    j = pts[..., 1] / prep["scale_y"] + prep["y_ref_shift_px"]
    k = (pts[..., 2] - prep.get("z_ref_shift_um", 0.0)) / prep["scale_z"]
    coords = np.stack([k, j, i], axis=0)
    s = map_coordinates(prep["data"], coords, order=1, mode="constant", cval=np.nan)
    return s.astype(np.float32), u_um, v_um


def _world_box(shape, sx, sy, sz, xshift, yshift, zshift):
    nz, ny, nx = shape
    xs = [(0 - xshift) * sx, (nx - 1 - xshift) * sx]
    ys = [(0 - yshift) * sy, (ny - 1 - yshift) * sy]
    zs = [zshift + 0 * sz, zshift + (nz - 1) * sz]
    return (min(xs), max(xs), min(ys), max(ys), min(zs), max(zs))


def _union_box(boxes):
    a = np.array(boxes)
    return (
        a[:, 0].min(),
        a[:, 1].max(),
        a[:, 2].min(),
        a[:, 3].max(),
        a[:, 4].min(),
        a[:, 5].max(),
    )


def resolve_auto_extent(sl, box):
    if sl.get("extent") != "auto":
        return sl
    u_hat, v_hat, n_hat = build_basis(sl["normal"], sl.get("up"))
    origin = np.asarray(sl["origin"], dtype=np.float64)
    x0, x1, y0, y1, z0, z1 = box
    corners = (
        np.array(
            [[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float64
        )
        - origin[None, :]
    )
    u_p, v_p, n_p = corners @ u_hat, corners @ v_hat, corners @ n_hat
    margin = float(sl.get("auto_margin_um", 0.0))
    out = dict(sl)
    u_c, v_c = 0.5 * (u_p.min() + u_p.max()), 0.5 * (v_p.min() + v_p.max())
    out["origin"] = (origin + u_c * u_hat + v_c * v_hat).tolist()
    out["half_u"] = 0.5 * (u_p.max() - u_p.min()) + margin
    out["half_v"] = 0.5 * (v_p.max() - v_p.min()) + margin
    out["sweep_start_um"] = float(n_p.min() - margin)
    out["sweep_stop_um"] = float(n_p.max() + margin)
    if not out.get("sweep_step_um"):
        out["sweep_step_um"] = float(out.get("du", 0.152))
    return out


# -----------------------------------------------------------------------------
# Loaders + per-volume preparation
# -----------------------------------------------------------------------------
def _axis_suffix(dataset_path):
    p = dataset_path.lower()
    if "chi" in p:
        return "χ", "_chi"
    if "mu" in p:
        return "μ", "_mu"
    return "", ""


def _estimate_box(cfg, p, scale_x, scale_y, samy_dir):
    if cfg["source"] == "aligned":
        with h5py.File(cfg["h5_path"], "r") as f:
            shape = f[cfg["dataset_path"]].shape
            a = dict(f.attrs)
        sx = float(a.get("scale_x_um_per_px", scale_x))
        sy = float(a.get("scale_y_um_per_px", scale_y))
        sz = float(a.get("scale_z_um_per_px", 1.0))
        return _world_box(shape, sx, sy, sz, 0.0, 0.0, 0.0)
    with h5py.File(cfg["h5_path"], "r") as f:
        n_layers, ny0, nx0 = f[cfg["dataset_path"]].shape
    rx, ry = cfg.get("roi_x"), cfg.get("roi_y")
    nx_roi = (rx[1] - rx[0]) if rx else nx0
    ny_roi = (ry[1] - ry[0]) if ry else ny0
    samy, samz, _ = _motors(cfg, p)
    if len(samy) > 0:
        pad_l = A.compute_pad_left(samy, scale_x, samy_dir)
        pad_r = A.compute_pad_right(samy, scale_x, samy_dir)
    else:
        pad_l = pad_r = 0
    nx_pad = nx_roi + pad_l + pad_r
    if len(samz) > 0:
        z_um = (samz[: min(n_layers, len(samz))] - samz[0]) * 1000.0
        med = float(np.median(np.abs(np.diff(z_um)))) if len(z_um) > 1 else 1.0
        med = med if med >= 1e-6 else 1.0
        n_uniform = max(2, int(np.round((z_um.max() - z_um.min()) / med)) + 1)
        sz = (z_um.max() - z_um.min()) / (n_uniform - 1) if n_uniform > 1 else med
    else:
        n_uniform, sz = n_layers, 2.0
    return _world_box((n_uniform, ny_roi, nx_pad), scale_x, scale_y, sz, 0.0, 0.0, 0.0)


def _motors(cfg, p):
    raw_root = cfg.get("raw_root") or ""
    pattern = cfg.get("raw_pattern") or ""
    if not raw_root or not pattern:
        return np.array([]), np.array([]), []
    folders = find_matching_folders(raw_root, pattern)
    if not folders:
        return np.array([]), np.array([]), []
    return extract_motor_positions(folders, p["samy_path"], p["samz_path"])


def prepare_volume(cfg, p, scale_x, scale_y, samy_dir):
    """Load and (if stacked) align one volume, resolving render style per kind."""
    kind, source = cfg["kind"], cfg["source"]
    extra = {}
    if source == "stacked":
        with h5py.File(cfg["h5_path"], "r") as f:
            raw = f[cfg["dataset_path"]][:].astype(np.float64)
        if kind == "mosa_fwhm" and bool(p["abs_fwhm"]):
            raw = np.abs(raw)
        samy, samz, _ = _motors(cfg, p)
        data = A.apply_roi_3d(raw, cfg.get("roi_x"), cfg.get("roi_y"))
        if len(samy) > 0:
            data = A.apply_samy_shifts_to_volume(data, samy, scale_x, samy_dir)
        if len(samz) > 0:
            data, _z, scale_z = A.interpolate_to_uniform_z(data, samz)
        else:
            scale_z = 2.0
        sx, sy = scale_x, scale_y
        x_ref = y_ref = 0.0
        z_ref = float(cfg.get("z_ref_shift_um", 0.0))
    else:  # aligned
        with h5py.File(cfg["h5_path"], "r") as f:
            data = f[cfg["dataset_path"]][:].astype(np.float64)
            extra = dict(f.attrs)
        sx = float(extra.get("scale_x_um_per_px", scale_x))
        sy = float(extra.get("scale_y_um_per_px", scale_y))
        scale_z = float(extra.get("scale_z_um_per_px", 1.0))
        x_ref = float(cfg.get("x_ref_shift_px", 0))
        y_ref = float(cfg.get("y_ref_shift_px", 0))
        z_ref = float(cfg.get("z_ref_shift_um", 0.0))

    center_method = p["center_method"].lower()
    if kind in ("mosa_com", "strain"):
        if center_method == "midrange":
            center, (auto_vmin, auto_vmax) = _midrange_clim(data, float(p["range_pct"]))
            data = data - center
        else:
            data, _ = _center_offset(data, center_method)
            auto_vmin, auto_vmax = _symmetric_range(data)
    else:  # mosa_fwhm / raw_*
        auto_vmin, auto_vmax = _percentile_range(data, 1, 99)

    sym, suffix = _axis_suffix(cfg["dataset_path"])
    titles = {
        "mosa_com": (f"{sym} Misorientation", "Misorientation (°)", suffix),
        "mosa_fwhm": (f"{sym} Peak Broadening", "Peak broadening (°)", suffix),
        "strain": ("Strain (cot method)", "Strain (ε)", ""),
        "raw_sum": ("Background-subtracted Sum Intensity", "Sum intensity (a.u.)", ""),
        "raw_specific": (
            f"Background-subtracted Frame {int(extra.get('specific_frame_idx', -1))}",
            "Intensity (a.u.)",
            f"_frame{int(extra.get('specific_frame_idx', -1))}",
        ),
    }
    title, cbar_label, suffix = titles[kind]
    return {
        "data": np.ascontiguousarray(data, dtype=np.float64),
        "scale_x": float(sx),
        "scale_y": float(sy),
        "scale_z": float(scale_z),
        "x_ref_shift_px": float(x_ref),
        "y_ref_shift_px": float(y_ref),
        "z_ref_shift_um": float(z_ref),
        "vmin": float(auto_vmin),
        "vmax": float(auto_vmax),
        "cmap_name": cfg.get("cmap") or "magma",
        "center_zero": kind == "mosa_com",
        "title": title,
        "cbar_label": cbar_label,
        "kind": kind,
        "volume_id": f"{kind}{suffix}",
        "h5_path": cfg["h5_path"],
        "dataset_path": cfg["dataset_path"],
    }


# -----------------------------------------------------------------------------
# Rendering + HDF5
# -----------------------------------------------------------------------------
def _make_norm(prep):
    vmin, vmax = prep["vmin"], prep["vmax"]
    if prep.get("center_zero"):
        if vmin < 0.0 < vmax and not np.isclose(abs(vmin), abs(vmax)):
            return mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        v = max(abs(vmin), abs(vmax))
        return mcolors.Normalize(vmin=-v, vmax=v)
    return mcolors.Normalize(vmin=vmin, vmax=vmax)


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


def render_slice_png(prep, sl, slice2d, u_um, v_um, out_png, offset_um, dpi=150):
    extent = [float(u_um[0]), float(u_um[-1]), float(v_um[0]), float(v_um[-1])]
    fig = Figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111)
    im = ax.imshow(
        slice2d,
        cmap=Rnd.cmap_nan_transparent(prep["cmap_name"]),
        norm=_make_norm(prep),
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    ax.set_xlabel("u (µm)")
    ax.set_ylabel("v (µm)")
    sub = sl["name"] if offset_um is None else f"{sl['name']}  (offset {offset_um:+.2f} µm)"
    ax.set_title(f"{prep['title']}\nslice: {sub}")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(prep["cbar_label"])
    _scale_bar(ax)
    fig.savefig(out_png, dpi=dpi, facecolor="white", bbox_inches="tight")


def write_volume_group(fh, prep, slice_records):
    vg = fh.create_group(prep["volume_id"])
    vg.attrs["kind"] = prep["kind"]
    vg.attrs["dataset_path"] = prep["dataset_path"]
    vg.attrs["source_volume"] = prep["h5_path"]
    vg.attrs["cbar_label"] = prep["cbar_label"]
    vg.attrs["cmap"] = prep["cmap_name"]
    vg.attrs["vmin"] = float(prep["vmin"])
    vg.attrs["vmax"] = float(prep["vmax"])
    vg.attrs["title"] = prep["title"]
    vg.attrs["scale_x_um_per_px"] = prep["scale_x"]
    vg.attrs["scale_y_um_per_px"] = prep["scale_y"]
    vg.attrs["scale_z_um_per_px"] = prep["scale_z"]
    for rec in slice_records:
        sg = vg.create_group(rec["name"])
        sg.create_dataset(
            "slices", data=rec["stack"].astype(np.float32), compression="gzip", compression_opts=4
        )
        sg.create_dataset("u_um", data=rec["u_um"].astype(np.float64))
        sg.create_dataset("v_um", data=rec["v_um"].astype(np.float64))
        sg.create_dataset("offsets_um", data=rec["offsets"].astype(np.float64))
        for key in ("normal", "origin", "up", "u_hat", "v_hat", "n_hat"):
            sg.attrs[key] = np.asarray(rec[key], np.float64)
        for key in ("half_u", "half_v", "du", "dv", "sweep_step_um"):
            sg.attrs[key] = float(rec[key])
        sg.attrs["n_planes"] = int(rec["stack"].shape[0])


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def _standard_volumes(p, roi_x, roi_y):
    """Build the VOLUMES config list from the include_* toggles + file paths."""
    file_keys = {
        "mosa_volume_file": (p["mosa_volume_file"], p["raw_root"], p["mosa_pattern"]),
        "strain_volume_file": (p["strain_volume_file"], p["raw_root"], p["strain_pattern"]),
        "aligned_rocking_file": (p["aligned_rocking_file"], "", ""),
    }
    out = []
    for toggle, source, file_param, dataset, kind, cmap in _STD_VOLUMES:
        if not bool(p.get(toggle)):
            continue
        path, raw_root, pattern = file_keys[file_param]
        if not path or not os.path.exists(path):
            continue
        cfg = {
            "h5_path": path,
            "dataset_path": dataset,
            "kind": kind,
            "source": source,
            "cmap": cmap,
        }
        if source == "stacked":
            cfg.update(raw_root=raw_root, raw_pattern=pattern, roi_x=roi_x, roi_y=roi_y)
        out.append(cfg)
    return out


def run(params: dict, progress: ProgressFn | None = None) -> SlicesResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    if p["center_method"].lower() not in ("mean", "median", "midrange"):
        raise ValueError(f"center_method must be mean/median/midrange (got {p['center_method']!r})")
    scale_x, scale_y = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["align_roi_x"]), _parse_pair(p["align_roi_y"])

    out_dir = p["output_dir"] or os.path.join(
        os.path.dirname(p["mosa_volume_file"] or p["strain_volume_file"] or "."), "oblique_slices"
    )
    os.makedirs(out_dir, exist_ok=True)
    result = SlicesResult(output_dir=out_dir)

    volumes = _standard_volumes(p, roi_x, roi_y)
    if not volumes:
        result.skipped.append("no input volumes found / selected")
        progress(1.0, "no volumes to slice")
        return result
    slices = json.loads(p["slices_json"])
    if not isinstance(slices, list) or not slices:
        raise ValueError("slices_json must be a non-empty JSON list of slice specs")

    # Resolve extent='auto' planes against a common data box (shared grid).
    if any(sl.get("extent") == "auto" for sl in slices):
        progress(0.05, "estimating data bounding box")
        boxes = [_estimate_box(cfg, p, scale_x, scale_y, samy_dir) for cfg in volumes]
        box = _union_box(boxes)
        slices = [resolve_auto_extent(sl, box) for sl in slices]

    out_h5 = os.path.join(out_dir, p["output_h5_name"])
    save_png = bool(p["save_png"])
    fh = h5py.File(out_h5, "w")
    fh.attrs["created_by"] = "dfxm.stages.slices"
    fh.attrs["center_method"] = p["center_method"]
    fh.attrs["range_pct"] = float(p["range_pct"])
    fh.attrs["frame"] = "origin-0 PVTI frame: world(X,Y,Z) = (i*scale_x, j*scale_y, k*scale_z)"
    try:
        for vi, cfg in enumerate(volumes):
            progress(0.1 + 0.85 * vi / len(volumes), f"slicing {cfg['kind']} {cfg['dataset_path']}")
            try:
                prep = prepare_volume(cfg, p, scale_x, scale_y, samy_dir)
            except (KeyError, OSError, ValueError) as exc:
                result.skipped.append(f"{cfg['dataset_path']}: {exc}")
                continue
            records = []
            for sl in slices:
                du = float(sl.get("du", prep["scale_x"]))
                dv = float(sl.get("dv", prep["scale_x"]))
                u_hat, v_hat, n_hat = build_basis(sl["normal"], sl.get("up"))
                origin = np.asarray(sl["origin"], dtype=np.float64)
                offsets = slice_plane_offsets(sl)
                planes, u_um, v_um = [], None, None
                for pi, off in enumerate(offsets):
                    s2d, u_um, v_um = sample_plane(
                        prep,
                        origin + off * n_hat,
                        u_hat,
                        v_hat,
                        float(sl["half_u"]),
                        float(sl["half_v"]),
                        du,
                        dv,
                    )
                    planes.append(s2d)
                    if save_png:
                        if len(offsets) == 1:
                            png = os.path.join(out_dir, f"{sl['name']}__{prep['volume_id']}.png")
                            render_slice_png(prep, sl, s2d, u_um, v_um, png, None)
                        else:
                            png = os.path.join(
                                out_dir,
                                f"{sl['name']}__{prep['volume_id']}__p{pi:03d}_{off:+08.2f}um.png",
                            )
                            render_slice_png(prep, sl, s2d, u_um, v_um, png, off)
                        result.pngs.append(png)
                stack = np.stack(planes, axis=0)
                result.n_planes_total += stack.shape[0]
                up_used = sl.get("up", None)
                records.append(
                    {
                        "name": sl["name"],
                        "stack": stack,
                        "u_um": u_um,
                        "v_um": v_um,
                        "offsets": offsets,
                        "normal": sl["normal"],
                        "origin": sl["origin"],
                        "up": up_used if up_used is not None else v_hat,
                        "u_hat": u_hat,
                        "v_hat": v_hat,
                        "n_hat": n_hat,
                        "half_u": float(sl["half_u"]),
                        "half_v": float(sl["half_v"]),
                        "du": du,
                        "dv": dv,
                        "sweep_step_um": float(sl.get("sweep_step_um") or 0.0),
                    }
                )
                if sl["name"] not in result.slice_names:
                    result.slice_names.append(sl["name"])
            write_volume_group(fh, prep, records)
            result.volume_ids.append(prep["volume_id"])
    finally:
        fh.close()

    result.output_h5 = out_h5
    progress(1.0, f"sliced {len(result.volume_ids)} volumes -> {os.path.basename(out_h5)}")
    return result


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Extract oblique slices from aligned volumes.")
    ap.add_argument("--mosa-volume-file", default="")
    ap.add_argument("--strain-volume-file", default="")
    ap.add_argument("--aligned-rocking-file", default="")
    ap.add_argument("--raw-root", default="")
    ap.add_argument("--output-dir", default="")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args(argv)
    res = run(
        dict(
            mosa_volume_file=args.mosa_volume_file,
            strain_volume_file=args.strain_volume_file,
            aligned_rocking_file=args.aligned_rocking_file,
            raw_root=args.raw_root,
            output_dir=args.output_dir,
            save_png=not args.no_png,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\nsliced {len(res.volume_ids)} volumes -> {res.output_h5}; planes {res.n_planes_total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
