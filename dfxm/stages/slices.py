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

One deliberate deviation from the legacy script: the default plane ``up`` is
world Y — the detector-vertical axis (lab-frame X) — not Z, so slice plots are
oriented like the per-layer renders (detector-X-like horizontal, detector-Y
vertical) — see :func:`build_basis`.
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
from scipy.ndimage import map_coordinates

from ..common import alignment as A
from ..common import render as Rnd
from ..common.errors import StageUserError
from ..common.figures import FigureSpec, crop_roi_2d, register, resolve_clim
from ..common.plotting import (
    GROUP_BY_KIND,
    PlotStyle,
    add_colorbar,
    apply_axes_mode,
    apply_round_clim,
    apply_text_scale,
    draw_scale_bar,
    figure_size,
    fit_axes_to_box,
    fixed_scale_box,
    resolve_cmap,
    style_from_params,
    styled_figure,
)
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

# Standard sliceable volumes: (toggle param, source, file param, dataset, kind).
# The colormap is resolved per kind via GROUP_BY_KIND + the active PlotStyle.
_STD_VOLUMES = (
    ("include_mosa_com_chi", "stacked", "mosa_volume_file", "chi/Center of mass", "mosa_com"),
    ("include_mosa_fwhm_chi", "stacked", "mosa_volume_file", "chi/FWHM", "mosa_fwhm"),
    ("include_mosa_com_mu", "stacked", "mosa_volume_file", "mu/Center of mass", "mosa_com"),
    ("include_mosa_fwhm_mu", "stacked", "mosa_volume_file", "mu/FWHM", "mosa_fwhm"),
    ("include_strain", "stacked", "strain_volume_file", "strain", "strain"),
    ("include_raw_sum", "aligned", "aligned_rocking_file", "sum_intensity", "raw_sum"),
    ("include_raw_specific", "aligned", "aligned_rocking_file", "specific_frame", "raw_specific"),
    ("include_mosa_sum", "aligned", "aligned_mosa_file", "sum_intensity", "raw_mosa_sum"),
    (
        "include_mosa_specific",
        "aligned",
        "aligned_mosa_file",
        "specific_frame",
        "raw_mosa_specific",
    ),
)


STAGE = StageSpec(
    name="slices",
    label="Oblique slices",
    description=(
        "Cuts arbitrary planes — defined in physical µm, optionally swept along their normal — "
        "through all aligned volumes at once, so every quantity is sampled at identical "
        "positions. Writes oblique_slices.h5 (used by profiles) plus a PNG per plane."
    ),
    params=(
        Param(
            "mosa_volume_file",
            ParamType.PATH,
            "Mosaicity volume",
            must_exist=True,
            help=(
                "The stacked mosaicity volume (stacked_volumes.h5) from the mosaicity stage. "
                "Leave blank to skip mosaicity fields."
            ),
        ),
        Param(
            "strain_volume_file",
            ParamType.PATH,
            "Strain volume",
            must_exist=True,
            help=(
                "The stacked strain volume (stacked_strain_volumes.h5) from the strain stage. "
                "Leave blank to skip strain."
            ),
        ),
        Param(
            "aligned_rocking_file",
            ParamType.PATH,
            "Aligned rocking volume",
            must_exist=True,
            help=(
                "The aligned rocking volume (aligned_raw_rocking_volumes.h5) from the rocking "
                "stage. Leave blank to slice without raw intensity."
            ),
        ),
        Param(
            "aligned_mosa_file",
            ParamType.PATH,
            "Aligned mosa volume",
            must_exist=True,
            help=(
                "The aligned mosa-sum volume (aligned_raw_mosa_volumes.h5) from the rocking stage "
                "run with Source scan = mosaicity. Leave blank to skip the mosa raw fields."
            ),
        ),
        Param(
            "raw_root",
            ParamType.DIR,
            "Raw data root",
            must_exist=True,
            help=(
                "RAW_DATA root with the original scan folders — provides the samy/samz "
                "positions used to align the stacked volumes."
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
                "HDF5 path to the sample-Y motor position inside each scan file "
                "(under the first BLISS entry). Only change for a different beamline file layout."
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
                "HDF5 path to the sample-Z motor position inside each scan file "
                "(under the first BLISS entry). Only change for a different beamline file layout."
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
                "Physical size of one detector pixel along X, in µm — sets the physical scale "
                "the planes are defined in. From the beamline optics calibration."
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
                "Physical size of one detector pixel along Y, in µm — sets the physical scale "
                "the planes are defined in. From the beamline optics calibration."
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
            "align_roi_x",
            ParamType.STR,
            "Align ROI X",
            default="",
            advanced=True,
            group="Alignment",
            roi_group="crop",
            roi_axis="x",
            roi_frame="map",
            help=(
                "Map-frame crop 'c0,c1' (map pixels, relative to the darfix window) used "
                "during alignment — must match the crop used when the volumes were "
                "rendered/exported. Pre-filled from the experiment's analysis window."
            ),
        ),
        Param(
            "align_roi_y",
            ParamType.STR,
            "Align ROI Y",
            default="",
            advanced=True,
            group="Alignment",
            roi_group="crop",
            roi_axis="y",
            roi_frame="map",
            help=(
                "Map-frame crop 'r0,r1' (map pixels, relative to the darfix window) used "
                "during alignment — must match the crop used when the volumes were "
                "rendered/exported. Pre-filled from the experiment's analysis window."
            ),
        ),
        Param(
            "abs_fwhm",
            ParamType.BOOL,
            "abs() FWHM",
            default=True,
            advanced=True,
            group="Alignment",
            help="Use absolute FWHM values (darfix fits can produce negative widths).",
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
                "How the colour scale of the misorientation (CoM) fields is centred: "
                "midrange = midpoint of the robust limits, or mean/median. Display only."
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
                "Robust percentile for colour limits, e.g. 99.5 ignores the most "
                "extreme 0.5 % of pixels."
            ),
        ),
        Param(
            "include_mosa_com_chi",
            ParamType.BOOL,
            "Slice χ misorientation",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the χ misorientation (centre-of-mass) volume.",
        ),
        Param(
            "include_mosa_fwhm_chi",
            ParamType.BOOL,
            "Slice χ FWHM",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the χ FWHM (rocking-curve width) volume.",
        ),
        Param(
            "include_mosa_com_mu",
            ParamType.BOOL,
            "Slice μ misorientation",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the μ misorientation (centre-of-mass) volume.",
        ),
        Param(
            "include_mosa_fwhm_mu",
            ParamType.BOOL,
            "Slice μ FWHM",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the μ FWHM (curve width) volume.",
        ),
        Param(
            "include_strain",
            ParamType.BOOL,
            "Slice strain",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the axial strain volume.",
        ),
        Param(
            "include_raw_sum",
            ParamType.BOOL,
            "Slice raw sum",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the summed raw rocking intensity volume.",
        ),
        Param(
            "include_raw_specific",
            ParamType.BOOL,
            "Slice raw specific",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the specific-frame raw intensity volume.",
        ),
        Param(
            "include_mosa_sum",
            ParamType.BOOL,
            "Slice mosa sum",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the mosa-scan summed intensity volume (from the aligned mosa file).",
        ),
        Param(
            "include_mosa_specific",
            ParamType.BOOL,
            "Slice mosa specific",
            default=True,
            advanced=True,
            group="Quantities",
            help="Slice the mosa-scan specific-frame intensity volume.",
        ),
        Param(
            "slices_json",
            ParamType.TEXT,
            "Slices (JSON)",
            default=_DEFAULT_SLICES,
            help=(
                "JSON list of plane definitions. Each needs a name and a normal vector; "
                "'extent': 'auto' fits the plane to the data, and 'sweep_step_um' adds "
                "parallel planes along the normal. The default shows the format."
            ),
        ),
        Param(
            "use_pinned",
            ParamType.BOOL,
            "Run pinned planes only",
            default=False,
            help=(
                "Render only the planes in 'Pinned planes (JSON)' instead of the full sweep — "
                "fast re-computation of a few interesting planes. The sweep in 'Slices (JSON)' "
                "is kept untouched and ignored while this is on; while on, the default output "
                "filename becomes oblique_slices_pinned.h5 so the sweep file is never "
                "overwritten. Untick to run the full sweep again."
            ),
        ),
        Param(
            "pinned_slices_json",
            ParamType.TEXT,
            "Pinned planes (JSON)",
            default="",
            advanced=True,
            group="Pinned planes",
            help=(
                "JSON list of pinned single-plane specs, normally written by the Pin planes… "
                "dialog (exact stored sweep geometry, snapped to stored planes). Only used "
                "when 'Run pinned planes only' is ticked."
            ),
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help="Where oblique_slices.h5 and the per-plane PNGs are written.",
        ),
        Param(
            "output_h5_name",
            ParamType.STR,
            "Output filename",
            default="oblique_slices.h5",
            advanced=True,
            group="Output",
            help=(
                "Filename of the consolidated slices file. "
                "The profiles stage expects oblique_slices.h5. "
                "While 'Run pinned planes only' is on, this default is replaced by "
                "oblique_slices_pinned.h5 (an edited name is respected)."
            ),
        ),
        Param(
            "save_png",
            ParamType.BOOL,
            "Save PNGs",
            default=True,
            advanced=True,
            group="Output",
            help="Write a PNG per plane in addition to the HDF5.",
        ),
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
    notes: list[str] = field(default_factory=list)


@dataclass
class ReplotEntry:
    volume_id: str
    slice_name: str
    n_planes: int
    offsets_um: list[float] = field(default_factory=list)
    shape: tuple[int, int] | None = None  # stored plane (nv, nu) pixel shape (ROI-crop hint)
    group: str = ""  # kind-group (GROUP_BY_KIND[kind]) — keys per-kind clim overrides


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
    """Orthonormal (u_hat, v_hat, n_hat) for the plane; vectors are (X, Y, Z).

    u_hat is the plot's horizontal axis and v_hat its vertical axis. The default
    up is world Y — the detector-vertical axis (lab-frame X) — so slices read
    like the per-layer renders (detector-X-like horizontal, detector-Y
    vertical); pass an explicit ``up`` per slice to override.
    """
    n = np.asarray(normal, dtype=np.float64)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        raise ValueError("normal vector has zero length")
    n_hat = n / nn
    if up is None:
        up_vec = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(n_hat, up_vec)) > 0.99:
            up_vec = np.array([0.0, 0.0, 1.0])
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


def resolve_auto_extent(sl, box, default_du=0.152):
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
        out["sweep_step_um"] = float(out.get("du", default_du))
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


# Volume kinds whose colour norm is centered on zero (symmetric TwoSlopeNorm).
_CENTERED_KINDS: frozenset[str] = frozenset({"mosa_com"})


def prepare_volume(cfg, p, scale_x, scale_y, samy_dir, style=None):
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
        "raw_mosa_sum": ("Mosa-integrated Sum Intensity", "Sum intensity (a.u.)", ""),
        "raw_mosa_specific": (
            f"Mosa-integrated Frame {int(extra.get('specific_frame_idx', -1))}",
            "Intensity (a.u.)",
            f"_frame{int(extra.get('specific_frame_idx', -1))}",
        ),
    }
    title, cbar_label, suffix = titles[kind]
    vmin_f, vmax_f, clim_note = apply_round_clim(float(auto_vmin), float(auto_vmax), style)
    return {
        "data": np.ascontiguousarray(data, dtype=np.float64),
        "scale_x": float(sx),
        "scale_y": float(sy),
        "scale_z": float(scale_z),
        "x_ref_shift_px": float(x_ref),
        "y_ref_shift_px": float(y_ref),
        "z_ref_shift_um": float(z_ref),
        "vmin": vmin_f,
        "vmax": vmax_f,
        "vmin_raw": float(auto_vmin),
        "vmax_raw": float(auto_vmax),
        "clim_note": clim_note,
        "cmap_name": resolve_cmap(style, GROUP_BY_KIND.get(kind)),
        "group": GROUP_BY_KIND.get(kind),
        "center_zero": kind in _CENTERED_KINDS,
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


_LEGACY_STYLE = PlotStyle(scale_bar_color="black", colorbar_fraction=0.046)


def build_slice_figure(
    prep, sl, slice2d, u_um, v_um, *, offset_um, style: PlotStyle | None = None
) -> Figure:
    """Build and return a slice figure. Does NOT call savefig.

    When *style* is ``None`` the default un-styled appearance is used (black auto
    scale bar, 0.046 colourbar fraction, 12×10 in figsize) via the shared styled
    primitives — close to, but not byte-identical with, the pre-export legacy
    renderer. When a :class:`~dfxm.common.plotting.PlotStyle` is supplied its
    figsize/colourbar/fonts/scale-bar settings are honoured instead.
    """
    st = style if style is not None else _LEGACY_STYLE
    use_legacy = style is None

    if use_legacy:
        figsize = (12, 10)
        box = None
    else:
        ext_u = float(u_um[-1] - u_um[0])
        ext_v = float(v_um[-1] - v_um[0])
        box = fixed_scale_box(st, ext_u, ext_v)
        if box is not None:
            figsize = (box[0] + 1.5, box[1] + 1.5)
        else:
            figsize = figure_size(st, ext_u, ext_v) or (12, 10)

    fig = styled_figure(figsize, styled=not use_legacy)
    ax = fig.add_subplot(111)
    extent = [float(u_um[0]), float(u_um[-1]), float(v_um[0]), float(v_um[-1])]
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

    if st.colorbar:
        add_colorbar(fig, im, ax, prep["cbar_label"], st, group=prep.get("group"))
    if st.scale_bar:
        draw_scale_bar(
            ax,
            st.scale_bar_length_um,
            style=st,
            fixed_scale_um_per_cm=(box[2] if box is not None else None),
        )
    if not use_legacy:
        apply_text_scale(ax, st)
        apply_axes_mode(ax, st)
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])

    return fig


def save_slice_png(prep, sl, slice2d, u_um, v_um, out_png, *, offset_um, dpi=150, style=None):
    """Build a slice figure (legacy look when *style* is None) and save it."""
    build_slice_figure(prep, sl, slice2d, u_um, v_um, offset_um=offset_um, style=style).savefig(
        out_png, dpi=dpi, facecolor="white", bbox_inches="tight"
    )


def write_volume_group(fh, prep, slice_records):
    vg = fh.create_group(prep["volume_id"])
    vg.attrs["kind"] = prep["kind"]
    vg.attrs["dataset_path"] = prep["dataset_path"]
    vg.attrs["source_volume"] = prep["h5_path"]
    vg.attrs["cbar_label"] = prep["cbar_label"]
    vg.attrs["cmap"] = prep["cmap_name"]
    vg.attrs["vmin"] = float(prep["vmin"])
    vg.attrs["vmax"] = float(prep["vmax"])
    if prep.get("clim_note"):
        vg.attrs["vmin_raw"] = float(prep["vmin_raw"])
        vg.attrs["vmax_raw"] = float(prep["vmax_raw"])
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
# Pinned planes (shared by tools/pin_slice.py, the Pin planes… dialog and run())
# -----------------------------------------------------------------------------
def _find_slice_group(f, slice_name, volume=None):
    """Return (volume_id, slice_group) for the first volume holding *slice_name*.

    Slice geometry is identical across volumes, so any volume that carries the
    slice works; *volume* forces a specific one.
    """
    vids = [volume] if volume else list(f.keys())
    for vid in vids:
        if vid not in f:
            continue
        g = f[vid]
        if slice_name in g and "offsets_um" in g[slice_name]:
            return vid, g[slice_name]
    raise StageUserError(
        f"slice {slice_name!r} not found in any volume group of the file",
        hint=f"volumes present: {', '.join(f.keys()) or '(none)'} — pick a slice "
        "name from a swept oblique_slices.h5.",
    )


def build_pinned_spec(h5_path, slice_name, offsets, *, volume=None) -> list[dict]:
    """Pinned single-plane spec dicts for *slice_name*, snapped to stored planes.

    Geometry (normal/origin/up/half_u/half_v/du/dv) is read byte-exact off the
    stored slice-group attrs, so each pinned plane reproduces the sweep's plane
    exactly. Every requested offset snaps to the nearest stored plane; snaps
    landing on the same plane are collapsed. Raises StageUserError for an
    unreadable file or unknown slice name.
    """
    try:
        fh = h5py.File(h5_path, "r")
    except OSError as exc:
        raise StageUserError(
            f"cannot read {h5_path!r}: {exc}",
            hint="Point at an oblique_slices.h5 written by a slices sweep run.",
        ) from exc
    with fh as f:
        _vid, sg = _find_slice_group(f, slice_name, volume)
        stored = sg["offsets_um"][:].astype(np.float64)
        a = dict(sg.attrs)
    specs, seen = [], set()
    for off in offsets:
        idx = int(np.argmin(np.abs(stored - float(off))))
        if idx in seen:
            continue
        seen.add(idx)
        matched = float(stored[idx])
        specs.append(
            {
                "name": f"{slice_name}_pin_{matched:+.2f}um",
                "normal": np.asarray(a["normal"], np.float64).tolist(),
                "origin": np.asarray(a["origin"], np.float64).tolist(),
                "up": np.asarray(a["up"], np.float64).tolist(),
                "half_u": float(a["half_u"]),
                "half_v": float(a["half_v"]),
                "du": float(a["du"]),
                "dv": float(a["dv"]),
                "sweep_step_um": float(a.get("sweep_step_um") or 1.0) or 1.0,
                "sweep_start_um": matched,
                "sweep_stop_um": matched,
            }
        )
    return specs


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def _standard_volumes(p, roi_x, roi_y):
    """Build the VOLUMES config list from the include_* toggles + file paths."""
    file_keys = {
        "mosa_volume_file": (p["mosa_volume_file"], p["raw_root"], p["mosa_pattern"]),
        "strain_volume_file": (p["strain_volume_file"], p["raw_root"], p["strain_pattern"]),
        "aligned_rocking_file": (p["aligned_rocking_file"], "", ""),
        "aligned_mosa_file": (p["aligned_mosa_file"], "", ""),
    }
    out = []
    for toggle, source, file_param, dataset, kind in _STD_VOLUMES:
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
        }
        if source == "stacked":
            cfg.update(raw_root=raw_root, raw_pattern=pattern, roi_x=roi_x, roi_y=roi_y)
        out.append(cfg)
    return out


# Relative Y-height spread beyond which volumes are flagged as misregistered.
# Legitimate pixel-size differences between runs (calculator vs nominal values)
# are well under 1%; a wrong detector-row crop is tens of percent.
_Y_HEIGHT_RTOL = 0.05


def _y_height_notes(volumes, roi_y, scale_y):
    """Warn when the selected volumes disagree on physical Y height.

    Every volume anchors at Y=0 in the origin-0 world frame, so co-registration
    along Y relies on all volumes covering the same detector-row window. A
    height mismatch beyond pixel-size rounding means one volume was built with
    a different crop — classically a rocking-stage roi_y entered as darfix's
    origin+size instead of start,end detector rows. Reads only shapes/attrs.
    """
    entries = []
    for cfg in volumes:
        try:
            with h5py.File(cfg["h5_path"], "r") as f:
                ny = int(f[cfg["dataset_path"]].shape[1])
                attrs = dict(f.attrs) if cfg["source"] == "aligned" else {}
        except (KeyError, OSError):
            continue  # unreadable volumes get reported as skips by prepare_volume
        label = f"{cfg['kind']} ({os.path.basename(cfg['h5_path'])})"
        if cfg["source"] == "aligned":
            sy = float(attrs.get("scale_y_um_per_px", scale_y))
            ys, ye = attrs.get("roi_y_start"), attrs.get("roi_y_end")
            if ys is not None and ye is not None:
                label += f" [file roi_y {int(ys)},{int(ye)}]"
        else:
            sy = scale_y
            if roi_y:
                ny = roi_y[1] - roi_y[0]
                label += f" [align_roi_y {roi_y[0]},{roi_y[1]}]"
        entries.append((label, ny * sy))
    if len(entries) < 2:
        return []
    heights = [h for _, h in entries]
    lo, hi = min(heights), max(heights)
    if lo <= 0 or (hi - lo) / lo <= _Y_HEIGHT_RTOL:
        return []
    desc = "; ".join(f"{label}: {h:.1f} µm" for label, h in entries)
    return [
        f"volume Y heights differ — {desc}. All volumes anchor at Y=0 in the world "
        "frame, so different heights misregister features along v/Y. Rebuild the "
        "mismatched aligned raw volume with the same detector-row window the map "
        "volumes use (rocking-stage roi_y is 'start,end' in raw-detector pixels; "
        "note darfix displays its ROI as origin+size, not start,end)."
    ]


def run(params: dict, progress: ProgressFn | None = None) -> SlicesResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    style = style_from_params(p)
    if p["center_method"].lower() not in ("mean", "median", "midrange"):
        raise ValueError(f"center_method must be mean/median/midrange (got {p['center_method']!r})")
    scale_x, scale_y = float(p["pixel_size_x_um"]), float(p["pixel_size_y_um"])
    samy_dir = int(p["samy_direction"])
    roi_x, roi_y = _parse_pair(p["align_roi_x"]), _parse_pair(p["align_roi_y"])

    out_dir = p["output_dir"] or os.path.join(
        os.path.dirname(p["mosa_volume_file"] or p["strain_volume_file"] or "."), "oblique_slices"
    )
    result = SlicesResult(output_dir=out_dir)

    volumes = _standard_volumes(p, roi_x, roi_y)
    if not volumes:
        result.skipped.append("no input volumes found / selected")
        progress(1.0, "no volumes to slice")
        return result
    os.makedirs(out_dir, exist_ok=True)  # only once we know there is work to do
    for msg in _y_height_notes(volumes, roi_y, scale_y):
        result.notes.append(msg)
        progress(0.02, msg)
    if bool(p["use_pinned"]):
        raw_pinned = (p["pinned_slices_json"] or "").strip()
        try:
            slices = json.loads(raw_pinned) if raw_pinned else []
        except json.JSONDecodeError as exc:
            raise StageUserError(
                f"Pinned planes JSON is not valid JSON: {exc}",
                hint=(
                    "Open Pin planes… to regenerate the pinned list, or untick "
                    "'Run pinned planes only' to run the full sweep."
                ),
            ) from exc
        if not isinstance(slices, list) or not slices:
            raise StageUserError(
                "'Run pinned planes only' is on but the pinned planes list is empty",
                hint=(
                    "Open Pin planes… to pick planes, or untick "
                    "'Run pinned planes only' to run the full sweep."
                ),
            )
        msg = (
            f"PINNED RUN: rendering {len(slices)} pinned plane(s); "
            "the sweep in slices_json is ignored"
        )
        result.notes.append(msg)
        progress(0.03, msg)
    else:
        slices = json.loads(p["slices_json"])
        if not isinstance(slices, list) or not slices:
            raise StageUserError(
                "slices_json must be a non-empty JSON list of slice specs",
                hint=(
                    "Provide a JSON list of plane specs — the field's default "
                    "shows the format; 'extent': 'auto' fits the plane "
                    "automatically."
                ),
            )

    # Resolve extent='auto' planes against a common data box (shared grid).
    if any(sl.get("extent") == "auto" for sl in slices):
        progress(0.05, "estimating data bounding box")
        boxes = [_estimate_box(cfg, p, scale_x, scale_y, samy_dir) for cfg in volumes]
        box = _union_box(boxes)
        slices = [resolve_auto_extent(sl, box, default_du=scale_x) for sl in slices]

    # Validate each (resolved) slice up front so a bad spec fails clearly here
    # rather than as a ZeroDivisionError / KeyError deep inside sampling.
    for sl in slices:
        name = sl.get("name", "?")
        for key in ("du", "dv"):
            if key in sl and float(sl[key]) <= 0:
                raise ValueError(f"slice {name!r}: {key} must be > 0")
        if "half_u" not in sl or "half_v" not in sl:
            raise ValueError(f"slice {name!r}: needs half_u and half_v (or extent: 'auto')")
        if float(sl["half_u"]) <= 0 or float(sl["half_v"]) <= 0:
            raise ValueError(f"slice {name!r}: half_u and half_v must be > 0")

    h5_name = p["output_h5_name"]
    if bool(p["use_pinned"]) and h5_name == STAGE.defaults()["output_h5_name"]:
        h5_name = "oblique_slices_pinned.h5"
        result.notes.append(
            "pinned run: output filename switched to oblique_slices_pinned.h5 "
            "so the sweep file profiles reads is not overwritten"
        )
    out_h5 = os.path.join(out_dir, h5_name)
    save_png = bool(p["save_png"])
    grids_by_slice: dict[str, list[tuple[str, tuple[int, int]]]] = {}
    fh = h5py.File(out_h5, "w")
    fh.attrs["created_by"] = "dfxm.stages.slices"
    fh.attrs["center_method"] = p["center_method"]
    fh.attrs["range_pct"] = float(p["range_pct"])
    fh.attrs["frame"] = "origin-0 PVTI frame: world(X,Y,Z) = (i*scale_x, j*scale_y, k*scale_z)"
    try:
        for vi, cfg in enumerate(volumes):
            progress(0.1 + 0.85 * vi / len(volumes), f"slicing {cfg['kind']} {cfg['dataset_path']}")
            try:
                prep = prepare_volume(cfg, p, scale_x, scale_y, samy_dir, style=style)
            except (KeyError, OSError, ValueError) as exc:
                result.skipped.append(f"{cfg['dataset_path']}: {exc}")
                continue
            if prep["clim_note"]:
                msg = f"{prep['volume_id']}: {prep['clim_note']}"
                progress(0.1 + 0.85 * vi / len(volumes), msg)
                result.notes.append(msg)
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
                        slice_dir = os.path.join(out_dir, sl["name"])
                        os.makedirs(slice_dir, exist_ok=True)
                        if len(offsets) == 1:
                            png = os.path.join(slice_dir, f"{prep['volume_id']}.png")
                            save_slice_png(
                                prep, sl, s2d, u_um, v_um, png, offset_um=None, style=style
                            )
                        else:
                            png = os.path.join(
                                slice_dir,
                                f"{prep['volume_id']}__p{pi:03d}_{off:+08.2f}um.png",
                            )
                            save_slice_png(
                                prep, sl, s2d, u_um, v_um, png, offset_um=off, style=style
                            )
                        result.pngs.append(png)
                stack = np.stack(planes, axis=0)
                result.n_planes_total += stack.shape[0]
                grids_by_slice.setdefault(sl["name"], []).append(
                    (prep["volume_id"], (len(v_um), len(u_um)))
                )
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

    # Volumes with different pixel scales sample the same plane onto different
    # (nv, nu) grids when the slice spec has no explicit du/dv. Downstream
    # (profiles) can then only mix fields that share a grid — say so up front.
    for sname, entries in grids_by_slice.items():
        by_shape: dict[tuple[int, int], list[str]] = {}
        for vid, shape in entries:
            by_shape.setdefault(shape, []).append(vid)
        if len(by_shape) > 1:
            desc = "; ".join(
                f"{nv}×{nu} px: {', '.join(vids)}" for (nv, nu), vids in sorted(by_shape.items())
            )
            msg = (
                f"slice {sname!r}: plane grids differ across volumes — {desc}. "
                "Profiles jobs can only mix fields on one grid; set explicit "
                "du/dv in slices_json to sample all volumes onto the same grid."
            )
            result.notes.append(msg)
            progress(0.99, msg)

    result.output_h5 = out_h5
    progress(1.0, f"sliced {len(result.volume_ids)} volumes -> {os.path.basename(out_h5)}")
    return result


@register("slices")
def figures(result: SlicesResult, params: dict) -> list[FigureSpec]:
    """One map FigureSpec per plane per slice subgroup per volume in oblique_slices.h5."""
    import h5py

    if not result.output_h5:
        return []
    specs = []
    # attrs are read defensively inside _rebuild_plane_figure, so one group from an
    # older/partial run missing an attr is rendered with fallbacks, not skipped.
    with h5py.File(result.output_h5, "r") as f:
        for vid in f.keys():
            vg = f[vid]
            for sname in vg.keys():
                n_planes = vg[sname]["slices"].shape[0]
                for k in range(n_planes):

                    def build(style, vid=vid, sname=sname, k=k):
                        return _rebuild_plane_figure(result.output_h5, vid, sname, k, style)

                    specs.append(
                        FigureSpec(
                            figure_id=f"slice_{vid}_{sname}_{k:03d}",
                            title=f"{vid} / {sname} / plane {k}",
                            kind="map",
                            filename=f"{vid}_{sname}_{k:03d}",
                            build=build,
                        )
                    )
    return specs


def _rebuild_plane_figure(h5_path, vid, sname, k, style, *, clim=None, roi=None) -> Figure | None:
    """Rebuild one plane's slice figure from an oblique_slices.h5 group.

    Shared by :func:`figures` (catalog/export) and :func:`render_replot` so the
    prep-from-attrs reconstruction lives in exactly one place. ``clim`` is an
    optional ``(vmin, vmax)`` override; ``None`` entries keep the stored value.
    ``roi`` is an optional ``(r0, r1, c0, c1)`` pixel-index crop; returns
    ``None`` when the clamped crop is empty.
    """
    with h5py.File(h5_path, "r") as f:
        vg = f[vid]
        kind = str(vg.attrs.get("kind", ""))
        prep = {
            "cmap_name": str(vg.attrs.get("cmap", "magma")),
            "title": str(vg.attrs.get("title", vid)),
            "cbar_label": str(vg.attrs.get("cbar_label", "")),
            "vmin": float(vg.attrs.get("vmin", 0.0)),
            "vmax": float(vg.attrs.get("vmax", 1.0)),
            "center_zero": kind in _CENTERED_KINDS,
        }
        sg = vg[sname]
        s2d = sg["slices"][k]
        u = sg["u_um"][:]
        v = sg["v_um"][:]
        off = float(sg["offsets_um"][k])
    if roi is not None:
        cropped = crop_roi_2d(s2d, roi)
        if cropped is None:
            return None
        r0, r1, c0, c1 = roi
        h, w = s2d.shape[:2]
        r0 = max(0, min(int(r0), h))
        r1 = max(0, min(int(r1), h))
        c0 = max(0, min(int(c0), w))
        c1 = max(0, min(int(c1), w))
        s2d, u, v = cropped, u[c0:c1], v[r0:r1]
    if clim is not None:
        vmin_o, vmax_o = clim
        if vmin_o is not None:
            prep["vmin"] = float(vmin_o)
        if vmax_o is not None:
            prep["vmax"] = float(vmax_o)
    prep["cmap_name"] = resolve_cmap(style, GROUP_BY_KIND.get(kind), fallback=prep["cmap_name"])
    prep["group"] = GROUP_BY_KIND.get(kind)
    return build_slice_figure(prep, {"name": sname}, s2d, u, v, offset_um=off, style=style)


def replot_catalog(h5_path: str) -> list[ReplotEntry]:
    """List every (volume_id, slice_name, n_planes, offsets_um) in an oblique_slices.h5."""
    entries: list[ReplotEntry] = []
    with h5py.File(h5_path, "r") as f:
        for vid in f.keys():
            vg = f[vid]
            if not isinstance(vg, h5py.Group):
                continue
            kind = str(vg.attrs.get("kind", ""))
            group = GROUP_BY_KIND.get(kind, kind)
            for sname in vg.keys():
                sg = vg[sname]
                if not (isinstance(sg, h5py.Group) and "slices" in sg):
                    continue
                offsets = [float(o) for o in sg["offsets_um"][:]]
                entries.append(
                    ReplotEntry(
                        vid,
                        sname,
                        int(sg["slices"].shape[0]),
                        offsets,
                        shape=tuple(sg["slices"].shape[1:]),
                        group=group,
                    )
                )
    return entries


def plane_preview(h5_path: str, volume_id: str, slice_name: str) -> tuple[np.ndarray, float, float]:
    """Middle plane of a slice group + its (du, dv) µm/px pitch, for the ROI picker.

    Returns ``(array2d, sx, sy)`` where ``sx=du`` (cols/u/X) and ``sy=dv``
    (rows/v/Y) come from the stored ``u_um``/``v_um`` axes — the resampled slice
    pitch, NOT the detector pixel scale.
    """
    with h5py.File(h5_path, "r") as f:
        sg = f[f"{volume_id}/{slice_name}"]
        stack = sg["slices"]
        mid = stack.shape[0] // 2
        arr = stack[mid][...]
        u = sg["u_um"][:]
        v = sg["v_um"][:]
    du = float(abs(u[1] - u[0])) if len(u) > 1 else 1.0
    dv = float(abs(v[1] - v[0])) if len(v) > 1 else 1.0
    return np.asarray(arr, dtype=float), du, dv


def render_replot(
    h5_path: str,
    selections: list[tuple[str, str, list[int] | None]],
    style: PlotStyle | None,
    clim: tuple[float | None, float | None] | dict[str, tuple] | None,
    out_dir: str,
    roi: tuple[int, int, int, int] | None = None,
    *,
    dpi: int = 150,
) -> list[str]:
    """Rebuild + save the selected planes (appearance-only; no resampling).

    ``selections`` is a list of ``(volume_id, slice_name, plane_idxs)`` where
    ``plane_idxs`` is ``None`` for all planes. PNGs are written under
    ``{out_dir}/{slice_name}/`` mirroring the slices run layout; returns the
    written paths. ``clim`` overrides the stored colour limits: ``None`` keeps
    them, a single ``(vmin, vmax)`` applies to every plane, and a
    ``{key: (vmin, vmax)}`` mapping sets them per quantity — keyed by
    ``volume_id`` (e.g. ``mosa_com_chi`` vs ``mosa_com_mu``, each ``raw_*``),
    falling back to the colormap group (``mosa_com``/``mosa_fwhm``/``strain``/
    ``raw``) for keys not found by volume_id. ``roi`` is an optional
    ``(r0, r1, c0, c1)`` pixel-index crop applied to every rebuilt plane; planes
    whose clamped crop is empty are silently skipped.
    """
    catalog = {(e.volume_id, e.slice_name): e for e in replot_catalog(h5_path)}
    written: list[str] = []
    for vid, sname, plane_idxs in selections:
        entry = catalog.get((vid, sname))
        if entry is None:
            continue
        idxs = list(range(entry.n_planes)) if plane_idxs is None else list(plane_idxs)
        clim_k = resolve_clim(clim, entry.volume_id)
        if clim_k is None:
            clim_k = resolve_clim(clim, entry.group)
        slice_dir = os.path.join(out_dir, sname)
        os.makedirs(slice_dir, exist_ok=True)
        for k in idxs:
            if k < 0 or k >= entry.n_planes:
                continue
            fig = _rebuild_plane_figure(h5_path, vid, sname, k, style, clim=clim_k, roi=roi)
            if fig is None:
                continue
            if entry.n_planes == 1:
                png = os.path.join(slice_dir, f"{vid}.png")
            else:
                png = os.path.join(slice_dir, f"{vid}__p{k:03d}_{entry.offsets_um[k]:+08.2f}um.png")
            fig.savefig(png, dpi=dpi, facecolor="white", bbox_inches="tight")
            written.append(png)
    return written


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


def roi_previews(params: dict) -> list:
    """(label, thunk) ROI-picker previews from the stacked mosa/strain volume(s)."""
    from ..common.figures import stacked_volume_previews

    return stacked_volume_previews(params)


if __name__ == "__main__":
    raise SystemExit(_main())
