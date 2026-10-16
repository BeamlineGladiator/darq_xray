"""Mosaicity stage — stack per-layer darfix maps into 3-D volumes.

Faithful port of ``stack_h5_darfix_volumes.py``: for each ``maps.h5`` it reads
the chi/mu Center-of-mass and FWHM 2-D maps and stacks them (along a new layer
axis) into ``stacked_volumes.h5`` with the structure::

    /chi/Center of mass   (L, H, W)
    /chi/FWHM             (L, H, W)
    /mu/Center of mass    (L, H, W)
    /mu/FWHM              (L, H, W)

A folder is included if at least one of its four maps is present (matching the
legacy behaviour); each output volume stacks only the layers where its map
existed.

Optionally (``detrend_chi_com`` / ``detrend_mu_com``, both off by default) the
χ and/or μ centre-of-mass map of every layer is detrended before stacking with
the same separable 2-D arctan fit strain applies to ccmth
(:mod:`darq_xray.common.detrend`). The detrended map replaces the stored one,
marked by a ``detrend`` attribute on its dataset, and a 3-panel diagnostic PNG
is written per layer.
"""

from __future__ import annotations

import gc
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

import h5py
import numpy as np

from ..common import progress as _progress_mod
from ..common.errors import StageUserError
from ..common.figures import (
    FigureSpec,
    ReplotGroup,
    register,
    render_volume_layer,
    resolve_clim,
    volume_layer_specs,
)
from ..common.h5io import StackedVolumeFile
from ..common.plotting import build_histogram, style_from_params
from ..common.sort import find_matching_folders, resolve_layer_work
from ..config.models import CostEstimate, Param, ParamType, SeeAlso, StageSpec

ProgressFn = Callable[[float, str], None]


def _noop(_frac: float, _msg: str) -> None:
    pass


# (param name, source-path default, output group, output dataset name)
_DATASETS = (
    ("chi_com_path", "/entry/chi/Center of mass/Center of mass", "chi", "Center of mass"),
    ("chi_fwhm_path", "/entry/chi/FWHM/FWHM", "chi", "FWHM"),
    ("mu_com_path", "/entry/mu/Center of mass/Center of mass", "mu", "Center of mass"),
    ("mu_fwhm_path", "/entry/mu/FWHM/FWHM", "mu", "FWHM"),
)


class _Detrendable(NamedTuple):
    toggle: str  # the BOOL param that switches it on
    src_param: str  # the param holding its HDF5 path inside maps.h5
    out_key: str  # its dataset in stacked_volumes.h5
    label: str


# The detrendable maps. Only the centre-of-mass maps — a FWHM is a width, not a
# tilt, and a background gradient across the field of view is not what
# broadens it. Filename stems come from `_KEY_STEM`, like every other figure.
_DETREND = (
    _Detrendable("detrend_chi_com", "chi_com_path", "/chi/Center of mass", "χ COM"),
    _Detrendable("detrend_mu_com", "mu_com_path", "/mu/Center of mass", "μ COM"),
)

# Output key -> label, for the GUI's result summary.
DETREND_LABELS: dict[str, str] = {d.out_key: d.label for d in _DETREND}

# Default folder for the detrend diagnostic PNGs, inside the input/root folder.
DETREND_DIR = "mosaicity_detrend"


def _plot_dir(p: dict, out_root: str) -> str:
    """Where the detrend diagnostics go: ``output_dir``, else ``<out_root>/mosaicity_detrend``."""
    return p["output_dir"] or os.path.join(out_root, DETREND_DIR)


def _drop_plot_dirs(folders: list[str], p: dict, root: str) -> list[str]:
    """*folders* without the detrend diagnostics folders.

    Once a detrend run has created one inside the root, a broad folder pattern
    — the stage's own default ``*`` — matches it like a layer folder: every
    later run then reported it as a skipped layer, and the estimate sized
    ``work[0]``, which sorts first, and came back unknown. Dropped: the
    configured ``output_dir``, the default folder, and any folder that holds
    diagnostics but no maps file — that last one is how a folder an EARLIER
    run wrote to is recognised after ``Output dir`` has been changed.
    """
    plot_dirs = {os.path.realpath(os.path.join(root, DETREND_DIR))}
    if p.get("output_dir"):
        plot_dirs.add(os.path.realpath(p["output_dir"]))
    maps_filename = str(p.get("maps_filename") or "maps.h5")

    def is_plot_dir(folder: str) -> bool:
        if os.path.realpath(folder) in plot_dirs:
            return True
        if os.path.exists(os.path.join(folder, maps_filename)):
            return False
        try:
            return any(n.endswith("_detrend_diag.png") for n in os.listdir(folder))
        except OSError:
            return False

    return [f for f in folders if not is_plot_dir(f)]


STAGE = StageSpec(
    name="mosaicity",
    label="Mosaicity volume",
    description=(
        "Stacks the darfix χ (chi) and μ (mu) centre-of-mass and width (FWHM) maps of each layer"
        " into one 3-D mosaicity volume. Needs maps.h5 from darfix in each layer folder; writes"
        " stacked_volumes.h5."
    ),
    params=(
        Param(
            "mode",
            ParamType.ENUM,
            "Mode",
            default="batch",
            choices=("single", "batch"),
            help=(
                "single processes one layer folder ('Input folder'); batch processes every"
                " subfolder of 'Root folder' matching 'Folder pattern'."
            ),
        ),
        Param(
            "input_folder",
            ParamType.DIR,
            "Input folder",
            must_exist=True,
            help="Layer folder containing the darfix maps.h5 (single mode only).",
        ),
        Param(
            "root_folder",
            ParamType.DIR,
            "Root folder",
            must_exist=True,
            help="Parent of the mosaicity layer folders (batch mode only).",
        ),
        Param(
            "folder_pattern",
            ParamType.STR,
            "Folder pattern",
            default="*",
            advanced=True,
            group="Data layout",
            help=(
                "Glob selecting the mosaicity layer subfolders, usually the *_mosa__* naming"
                " pattern."
            ),
        ),
        Param(
            "maps_filename",
            ParamType.STR,
            "maps filename",
            default="maps.h5",
            advanced=True,
            group="Data layout",
            help="Filename of the darfix output inside each layer folder (normally maps.h5).",
        ),
        Param(
            "chi_com_path",
            ParamType.STR,
            "chi COM path",
            default=_DATASETS[0][1],
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path of the χ centre-of-mass map inside maps.h5 (darfix layout). χ CoM is"
                " the local lattice tilt about the rocking axis."
            ),
        ),
        Param(
            "chi_fwhm_path",
            ParamType.STR,
            "chi FWHM path",
            default=_DATASETS[1][1],
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path of the χ FWHM map — the local rocking-curve width, a measure of mosaic"
                " spread."
            ),
        ),
        Param(
            "mu_com_path",
            ParamType.STR,
            "mu COM path",
            default=_DATASETS[2][1],
            advanced=True,
            group="Data layout",
            help=(
                "HDF5 path of the μ centre-of-mass map — the local lattice tilt about the second"
                " tilt axis."
            ),
        ),
        Param(
            "mu_fwhm_path",
            ParamType.STR,
            "mu FWHM path",
            default=_DATASETS[3][1],
            advanced=True,
            group="Data layout",
            help="HDF5 path of the μ FWHM map — the local curve width about the second tilt axis.",
        ),
        Param(
            "output_dir",
            ParamType.DIR,
            "Output dir",
            help=(
                "Where the detrend diagnostic PNGs go (default: a mosaicity_detrend folder in the"
                " input/root folder). stacked_volumes.h5 is always written to the input/root"
                " folder, not here."
            ),
        ),
        Param(
            "stacked_filename",
            ParamType.STR,
            "Stacked filename",
            default="stacked_volumes.h5",
            advanced=True,
            group="Output",
            help=(
                "Filename of the stacked mosaicity volume. "
                "Downstream stages expect stacked_volumes.h5."
            ),
        ),
        Param(
            "compression",
            ParamType.ENUM,
            "Compression",
            default="gzip",
            choices=("gzip", "lzf", "none"),
            advanced=True,
            group="Output",
            help="HDF5 compression for the volume: gzip (small, slower), lzf (fast, larger), none.",
        ),
        Param(
            "detrend_chi_com",
            ParamType.BOOL,
            "Detrend χ COM",
            default=False,
            advanced=True,
            group="Detrend",
            help=(
                "Remove a slowly varying background (e.g. sample bending) from each layer's χ"
                " centre-of-mass map before stacking — the separable 2-D arctan fit strain applies"
                " to ccmth, on the full map, keeping its median level. The stored χ CoM is then"
                " the detrended map, so every later stage uses it; a diagnostic PNG per layer"
                " goes to 'Output dir'."
            ),
        ),
        Param(
            "detrend_mu_com",
            ParamType.BOOL,
            "Detrend μ COM",
            default=False,
            advanced=True,
            group="Detrend",
            help=(
                "Same as 'Detrend χ COM', for the μ centre-of-mass map. The FWHM maps are never"
                " detrended."
            ),
        ),
    ),
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” (left panel), not here.",
        ),
    ),
    estimate="darq_xray.stages.mosaicity:estimate",
)


@dataclass
class MosaicityResult:
    stacked_path: str | None = None
    datasets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    # The maps.h5 each entry of `layers` was read from, so figures() can rebuild
    # the detrend diagnostic without re-deriving the folder from its name.
    maps_paths: list[str] = field(default_factory=list)
    # Per output key (e.g. "/chi/Center of mass"), one `(z, layer_index)` per
    # detrended map: `z` is the slice in THAT dataset — which stacks only the
    # layers carrying its map, so it can trail the layer index — and
    # `layer_index` points into `layers`/`maps_paths`.
    detrend_layers: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    detrend_plots: list[str] = field(default_factory=list)
    detrend_dir: str = ""
    # Non-fatal remarks, e.g. a map too sparse to fit a trend to.
    notes: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def detrended(self) -> list[str]:
        """Output keys holding at least one detrended slice, χ before μ."""
        return [d.out_key for d in _DETREND if self.detrend_layers.get(d.out_key)]


# Per-key display metadata (cmap, colorbar label, title prefix).
# Keys match the "/{group}/{ds_name}" form stored in MosaicityResult.datasets.
# Conventions mirror visualize._display_info. The first element is the
# PlotStyle colormap GROUP (resolved at build time); unknown keys fall back
# to the fixed "magma". CoM = misorientation, FWHM = peak broadening.
_KEY_DISPLAY: dict[str, tuple[str | None, str, str]] = {
    "/chi/Center of mass": ("mosa_com", "Misorientation (°)", "χ Misorientation"),
    "/chi/FWHM": ("mosa_fwhm", "Peak broadening (°)", "χ Peak Broadening"),
    "/mu/Center of mass": ("mosa_com", "Misorientation (°)", "μ Misorientation"),
    "/mu/FWHM": ("mosa_fwhm", "Peak broadening (°)", "μ Peak Broadening"),
}

# Safe filename stem per key (no spaces/slashes).
_KEY_STEM: dict[str, str] = {
    "/chi/Center of mass": "chi_com",
    "/chi/FWHM": "chi_fwhm",
    "/mu/Center of mass": "mu_com",
    "/mu/FWHM": "mu_fwhm",
}


def _streamed_clim(dataset) -> tuple[float, float]:
    """Global (nanmin, nanmax) of a (Z,Y,X) HDF5 volume, one layer at a time.

    Memory-light: never materialises the whole volume (peak memory = one layer),
    so merely listing the figure catalog stays cheap even for multi-GB stacks.
    Falls back to (0, 1) for an all-NaN/empty volume.
    """
    vmin, vmax = np.inf, -np.inf
    for z in range(dataset.shape[0]):
        finite = dataset[z][np.isfinite(dataset[z])]
        if finite.size:
            vmin = min(vmin, float(finite.min()))
            vmax = max(vmax, float(finite.max()))
    if not np.isfinite(vmin):
        return 0.0, 1.0
    return vmin, vmax


def _detrend_map(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detrend one CoM map in float64; return ``(detrended in raw's dtype, surface)``.

    The stored map keeps the source dtype, so a float32 maps.h5 still yields a
    float32 volume and the estimator's itemsize stays the source's own.
    """
    from ..common.detrend import detrend_arctan_2d

    detrended, surface = detrend_arctan_2d(np.asarray(raw, dtype=np.float64))
    return detrended.astype(raw.dtype, copy=False), surface


def _detrend_diag(raw, detrended, surface, style):
    """The 3-panel diagnostic for one map, coloured with the mosaicity CoM group.

    The one place run() and figures() build it, so the PNG and the figure-list
    entry cannot drift apart.
    """
    from ..common.detrend import build_detrend_diag

    return build_detrend_diag(
        raw, detrended, surface, style=style, group="mosa_com", original_title="Original"
    )


def _build_detrend(maps_path: str, src_path: str, label: str, style):
    """Rebuild one layer's detrend diagnostic from its source maps.h5."""
    from ..common.detrend import can_detrend

    if not os.path.exists(maps_path):
        raise FileNotFoundError(
            f"source maps.h5 not found; the detrend diagnostic needs the original input"
            f" (looked for {maps_path!r})"
        )
    with h5py.File(maps_path, "r") as f:
        raw = _read_dataset(f, src_path)
    if raw is None:
        raise KeyError(f"{src_path!r} not in {maps_path!r}")
    if not can_detrend(raw):
        raise ValueError(f"{label} in {maps_path!r} {_unfittable(raw)}")
    detrended, surface = _detrend_map(raw)
    return _detrend_diag(raw, detrended, surface, style)


def _unfittable(raw: np.ndarray) -> str:
    """Why ``can_detrend`` refused *raw*, as the end of a sentence about the map."""
    if raw.ndim != 2:
        return f"is not a 2-D map (shape {tuple(raw.shape)})"
    return "has too few finite pixels to fit a trend"


def _detrend_specs(result: "MosaicityResult", params: dict) -> list[FigureSpec]:
    """One ``plot`` FigureSpec per detrended layer per detrended quantity."""
    p = {**STAGE.defaults(), **params}
    specs: list[FigureSpec] = []
    for d in _DETREND:
        stem = _KEY_STEM[d.out_key]
        for z, i in result.detrend_layers.get(d.out_key, []):
            name, maps_path = result.layers[i], result.maps_paths[i]

            def build(style, _mp=maps_path, _src=str(p[d.src_param]), _label=d.label):
                return _build_detrend(_mp, _src, _label, style)

            specs.append(
                FigureSpec(
                    figure_id=f"{stem}_detrend_z{z:04d}",
                    title=f"{d.label} detrend diagnostic — {name}",
                    kind="plot",
                    filename=f"{name}_{stem}_detrend_diag",
                    build=build,
                )
            )
    return specs


@register("mosaicity")
def figures(result: "MosaicityResult", params: dict) -> list[FigureSpec]:
    """Return one ``map`` + one ``plot`` (histogram) FigureSpec per Z layer per dataset key,
    plus one detrend diagnostic per detrended layer."""
    if not result.stacked_path or not result.datasets:
        return []

    # pixel scale — mosaicity has no pixel-size params of its own; use whatever
    # the caller passes (e.g. chained from strain), falling back to the same
    # calibrated beamline defaults that strain.figures() uses.
    px = float(params.get("pixel_size_x_um", 0.152))
    py = float(params.get("pixel_size_y_um", 0.385))

    specs: list[FigureSpec] = []
    for key in result.datasets:
        # The key is already the in-file HDF5 path (e.g. "/chi/Center of mass").
        group, cbar_label, title = _KEY_DISPLAY.get(
            key,
            (None, "(°)", key.lstrip("/").replace("/", " ")),
        )
        stem = _KEY_STEM.get(key, key.lstrip("/").replace("/", "_").replace(" ", "_"))

        # vmin/vmax/n_z without loading the whole volume into memory: read the
        # shape, then stream the clim one layer at a time (large stacks otherwise
        # spike RAM merely to LIST the catalog).
        with h5py.File(result.stacked_path, "r") as fh:
            dset = fh[key]
            n_z = dset.shape[0]
            vmin, vmax = _streamed_clim(dset)

        # map specs (one per layer)
        specs.extend(
            volume_layer_specs(
                h5_path=result.stacked_path,
                dataset=key,
                id_prefix=stem,
                title=title,
                cbar_label=cbar_label,
                cmap="magma",
                cmap_group=group,
                sx=px,
                sy=py,
                vmin=vmin,
                vmax=vmax,
            )
        )

        # histogram specs (one per layer, kind="plot")
        for z in range(n_z):

            def _build_hist(
                style,
                _path=result.stacked_path,
                _key=key,
                _z=z,
                _title=title,
                _xlabel=cbar_label,
            ):
                with h5py.File(_path, "r") as fh:
                    arr = fh[_key][_z]
                fig = build_histogram(
                    arr,
                    title=f"{_title} — layer {_z} distribution",
                    xlabel=_xlabel,
                    style=style,
                )
                if fig is None:
                    raise ValueError(f"layer {_z} of {_key!r} has no finite values to histogram")
                return fig

            specs.append(
                FigureSpec(
                    figure_id=f"{stem}_hist_z{z:04d}",
                    title=f"{title} — layer {z} distribution",
                    kind="plot",
                    filename=f"{stem}_hist_layer_{z:04d}",
                    build=_build_hist,
                )
            )

    specs.extend(_detrend_specs(result, params))
    return specs


def replot_catalog(h5_path: str) -> list[ReplotGroup]:
    """List every 3-D mosaicity dataset in a stacked h5 as a replot group."""
    groups: list[ReplotGroup] = []
    with h5py.File(h5_path, "r") as f:
        for key in _KEY_DISPLAY:
            obj = f.get(key)
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                continue
            _grp, _cbar, title = _KEY_DISPLAY[key]
            n_z = obj.shape[0]
            groups.append(
                ReplotGroup(
                    key=key,
                    label=title,
                    item_labels=[f"layer {z}" for z in range(n_z)],
                    shape=tuple(obj.shape[1:]),
                )
            )
    return groups


def render_replot(h5_path, selections, style, clim, out_dir, roi=None, params=None) -> list[str]:
    """Re-render selected mosaicity map layers cold from a stacked h5.

    ``selections`` is ``list[(dataset_key, item_idxs | None)]`` (``None`` = all
    layers). ``clim`` overrides vmin/vmax: ``None`` keeps the streamed default, a
    ``(vmin, vmax)`` tuple applies to every dataset, and a
    ``{dataset_key: (vmin, vmax)}`` mapping sets them per dataset — keyed by the
    in-file HDF5 path (``ReplotGroup.key``, e.g. ``"/chi/Center of mass"`` and
    ``"/chi/FWHM"``), not the colormap group name. ``roi`` crops each layer
    (pixel bounds). PNGs are written under ``{out_dir}/{stem}/``; returns paths.
    """
    params = params or {}
    px = float(params.get("pixel_size_x_um", 0.152))
    py = float(params.get("pixel_size_y_um", 0.385))
    written: list[str] = []
    with h5py.File(h5_path, "r") as f:
        for key, idxs in selections:
            obj = f.get(key)
            if not isinstance(obj, h5py.Dataset) or obj.ndim != 3:
                continue
            group, cbar_label, title = _KEY_DISPLAY.get(key, (None, "(°)", key))
            stem = _KEY_STEM.get(key, key.lstrip("/").replace("/", "_").replace(" ", "_"))
            clim_k = resolve_clim(clim, key)
            n_z = obj.shape[0]
            vmin, vmax = _streamed_clim(obj)
            layer_list = list(range(n_z)) if idxs is None else list(idxs)
            sub_dir = os.path.join(out_dir, stem)
            os.makedirs(sub_dir, exist_ok=True)
            for z in layer_list:
                if z < 0 or z >= n_z:
                    continue
                fig = render_volume_layer(
                    h5_path,
                    key,
                    z,
                    cmap="magma",
                    cmap_group=group,
                    title=title,
                    cbar_label=cbar_label,
                    sx=px,
                    sy=py,
                    vmin=vmin,
                    vmax=vmax,
                    style=style,
                    clim=clim_k,
                    roi=roi,
                )
                if fig is None:
                    continue
                png = os.path.join(sub_dir, f"{stem}_layer_{z:04d}.png")
                fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
                written.append(png)
    return written


def _read_dataset(h5f: h5py.File, path: str) -> np.ndarray | None:
    obj = h5f.get(path)
    return obj[:] if isinstance(obj, h5py.Dataset) else None


# --- Recalibrated 2026-08-26 against measured child peak RSS ------------------
#
# Measured on this box with `tests/peak_rss.py`: six layer shapes spanning 20x
# with all four datasets present, and 1/2/3/4 datasets at two of those shapes.
# The peak is flat in `n_layers` (176.1 MiB at one layer, 194.1 at four, eight
# and sixteen), and the per-dataset slope at the larger shape came out at
# **1.00, 1.01 and 1.00 float64 layers** — as exact as this campaign got. See
# `docs/Codebase.md` for the tables.

# The process image of a mosaicity child: interpreter, numpy, h5py, this module.
# It is the lightest in the pipeline because `run()` never imports matplotlib.
# Measured 74.1 MiB (`measure_process_floor`), charged with slack.
MOSAICITY_PROCESS_FLOOR_BYTES = 96 * 1024 * 1024

# Layers resident beyond the one-per-present-dataset the `data` dict holds:
# `StackedVolumeFile`'s gzip compression buffer and h5py's read buffer. Measured
# as the dataset-count-independent remainder, **1.77 layers** at both shapes.
MOSAICITY_WRITER_LAYERS = 3

# --- The optional χ/μ detrend, measured 2026-10-16 ----------------------------
#
# Charged once, whether one box or both is ticked: the two maps are detrended
# one after the other and `run()` collects each diagnostic figure before the
# next (without that, ticking both measured up to 38 MiB above ticking one).
# Measured with `tests/peak_rss.py` over four layer shapes spanning 16x
# (0.262 M to 4.194 M elements), float32 and float64, with and without the
# publication style: chi-only and chi+mu peaks agree within 2 MiB everywhere.

# What the first detrend adds regardless of size: scipy.optimize's import and
# the diagnostic's fixed 20x6 in, 120 dpi Agg canvas (the style cannot grow it).
# Measured as the detrend-on minus detrend-off intercept: 48.5 MiB (float32),
# 48.6 (float64), 51.0 (float64, styled).
MOSAICITY_DETREND_PROCESS_BYTES = 64 * 1024 * 1024

# Per element of one layer: the float64 working copy, the fit's residuals and
# surface, `np.nanmedian`'s copies, and the three `imshow` normalisations.
# Measured slope 37.6 B/elem (float32) and 43.4 (float64, plain and styled);
# 56 is seven float64 copies, the same figure strain charges for its chain.
MOSAICITY_DETREND_BYTES_PER_ELEM = 56


def _elems(shape: tuple[int, ...]) -> int:
    """Element count of *shape*; 0 for the empty tuple (no dataset seen yet)."""
    if not shape:
        return 0
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def estimate(params: dict) -> CostEstimate:
    """Peak memory for a mosaicity run, from HDF5 shapes only.

    ``run()`` reads one layer of each present dataset into the per-folder
    ``data`` dict, ``append``s each to a ``StackedVolumeFile`` and drops the
    dict, so the peak is **one layer per present dataset** plus the writer's
    buffers — it does not scale with ``n_layers``. The figure this estimator
    returned until 2026-08-26 (``(n_present + 1)`` whole *volumes*, a model of
    the deleted ``collected``-dict-then-``np.stack`` code) over-predicted the
    real STO2 run by **36x**, the worst error in the pipeline.

    The recalibrated model is
    :data:`MOSAICITY_PROCESS_FLOOR_BYTES` ``+ (present +``
    :data:`MOSAICITY_WRITER_LAYERS` ``) * layer_elems * itemsize``, in the
    source's own dtype because ``_read_dataset`` and ``out.append`` both keep
    it. ``present`` still matters and is still counted: switching a dataset off
    genuinely removes one layer from the peak.

    ``figures()``/``replot`` reach the same file through ``_streamed_clim`` and
    ``render_volume_layer``, which are layer-at-a-time by construction and are
    **not** part of ``run()``'s peak — they are deliberately not folded in.

    **Currency**: ``peak_bytes`` is whole-child peak RSS, process image
    included — see :func:`darq_xray.stages.strain.estimate` for why, and for the
    inconsistency with the three alignment-chain estimators that this records
    rather than hides.

    ``total_input`` and the reported ``shape`` still scale with ``n_layers``.
    Never raises — an unresolved work list or no matching datasets returns an
    unknown cost with the reason in ``note``.
    """
    p = {**STAGE.defaults(), **params}
    try:
        work = resolve_layer_work(p, maps_filename=str(p["maps_filename"] or "maps.h5"))
        if p["mode"] != "single":
            root = (p["root_folder"] or "").rstrip("/")
            kept = set(_drop_plot_dirs([os.path.dirname(w) for w in work], p, root))
            work = [w for w in work if os.path.dirname(w) in kept]
        if not work:
            return CostEstimate(0, 0, None, True, "no layer folders resolved yet")
        present = 0
        layer_shape: tuple[int, ...] = ()
        itemsize = 0
        with h5py.File(work[0], "r") as f:
            for key, _default, _group, _name in _DATASETS:
                ds_path = str(p.get(key) or "")
                if ds_path and ds_path in f:
                    ds = f[ds_path]
                    ds_shape = tuple(int(d) for d in ds.shape)
                    # The LARGEST of the present datasets on both counts, not
                    # whichever happens to come last in `_DATASETS`. A maps.h5
                    # with float64 chi and float32 mu was being priced at the
                    # mu itemsize — half the real cost, the direction that OOMs.
                    if _elems(ds_shape) > _elems(layer_shape):
                        layer_shape = ds_shape
                    itemsize = max(itemsize, int(ds.dtype.itemsize))
                    present += 1
        itemsize = itemsize or 8
        if not present:
            return CostEstimate(0, 0, None, True, "none of the mosaicity datasets found")
    except Exception as exc:  # noqa: BLE001 - an estimate is advisory, never fatal
        return CostEstimate(0, 0, None, True, f"cannot size input: {type(exc).__name__}")

    layer_elems = _elems(layer_shape)
    n_layers = len(work)
    per_volume = n_layers * layer_elems * itemsize
    input_bytes = present * per_volume
    peak_bytes = MOSAICITY_PROCESS_FLOOR_BYTES + (present + MOSAICITY_WRITER_LAYERS) * (
        layer_elems * itemsize
    )
    if any(p.get(d.toggle) for d in _DETREND):
        peak_bytes += MOSAICITY_DETREND_PROCESS_BYTES + MOSAICITY_DETREND_BYTES_PER_ELEM * (
            layer_elems
        )
    return CostEstimate(
        peak_bytes,
        input_bytes,
        (n_layers, *layer_shape),
        True,
        f"{present} datasets stacked together",
    )


def run(params: dict, progress: ProgressFn | None = None) -> MosaicityResult:
    progress = progress or _noop
    p = {**STAGE.defaults(), **params}
    maps_filename = p["maps_filename"]

    # source path per output key, plus the (group, name) routing
    config = {key: p[key] for key, _default, _g, _n in _DATASETS}
    routing = {key: (g, n) for key, _default, g, n in _DATASETS}

    if p["mode"] == "single":
        folder = p["input_folder"]
        if not folder:
            raise StageUserError(
                "single mode requires 'input_folder'",
                hint=(
                    "Pick the layer folder holding maps.h5 in 'Input folder', "
                    "or switch Mode to 'batch'."
                ),
            )
        work = [(os.path.basename(folder.rstrip("/")), os.path.join(folder, maps_filename))]
        default_out_root = folder
    else:
        root = (p["root_folder"] or "").rstrip("/")
        if not root:
            raise StageUserError(
                "batch mode requires 'root_folder'",
                hint=(
                    "Pick the parent of the mosaicity layer folders in "
                    "'Root folder', or switch Mode to 'single'."
                ),
            )
        folders = _drop_plot_dirs(find_matching_folders(root, p["folder_pattern"]), p, root)
        if not folders:
            raise StageUserError(
                f"no folders matching {p['folder_pattern']!r} in {root}",
                hint=(
                    "Check 'Folder pattern' — usually the *_mosa__* naming "
                    "pattern of the mosaicity layers."
                ),
            )
        work = [(os.path.basename(f), os.path.join(f, maps_filename)) for f in folders]
        default_out_root = root

    compression = None if p["compression"] == "none" else p["compression"]
    stacked_path = os.path.join(default_out_root, p["stacked_filename"])
    LAYERS_HI = 0.95
    result = MosaicityResult()

    to_detrend = [d for d in _DETREND if p[d.toggle]]
    if to_detrend:
        from ..common.detrend import DETREND_DIAG_DPI, can_detrend

        style = style_from_params(p)
        result.detrend_dir = _plot_dir(p, default_out_root)
    # With a detrend ticked, the reads give up half their share of the layer to
    # it: the arctan fits and the diagnostic render are the slow step then.
    read_hi = 0.3 if to_detrend else 0.6
    # Slices appended so far per output key — a dataset stacks only the layers
    # carrying its map, so its Z can trail the layer index.
    z_next: dict[str, int] = {}
    plot_error: OSError | None = None

    with StackedVolumeFile(stacked_path, compression=compression) as out:
        # Entry AND completion per layer, plus a report before the stacked
        # file's attrs/close: reporting only on entry left the last layer's read
        # and the whole write tail as one jump to 1.0.
        for i, (name, maps_path) in enumerate(work):
            lay_lo, lay_hi = _progress_mod.slice_for(i, len(work), 0.0, LAYERS_HI)
            # One reporter over this layer's own share, so the fractions below
            # read as "how far through this layer" instead of re-deriving the
            # map into the run-wide bar at every site.
            lp = _progress_mod.sub_progress(progress, lay_lo, lay_hi)
            progress(lay_lo, f"mosaicity: {name}")
            if not os.path.exists(maps_path):
                result.skipped.append(f"{name}: {maps_filename} not found")
                continue
            try:
                with h5py.File(maps_path, "r") as f:
                    # An explicit loop rather than a dict comprehension so each
                    # dataset read reports: four maps per layer is the only
                    # inner loop this stage has, and on a real layer each read
                    # is a whole 2-D map off disk.
                    data = {}
                    for di, (key, path) in enumerate(config.items()):
                        data[key] = _read_dataset(f, path)
                        lp(read_hi * (di + 1) / max(1, len(config)), f"mosaicity: {name} {key}")
            except OSError as exc:
                result.skipped.append(f"{name}: {exc}")
                continue
            if all(v is None for v in data.values()):
                result.skipped.append(f"{name}: no datasets")
                continue
            layer_idx = len(result.layers)
            for ki, d in enumerate(to_detrend):
                # Two reports per map — the fit and the diagnostic render are
                # each a slow step — and both fire whatever the map turns out
                # to be, so the bar's granularity does not depend on the data.
                k_lo, k_hi = _progress_mod.slice_for(ki, len(to_detrend), read_hi, 0.6)
                k_mid = 0.5 * (k_lo + k_hi)
                raw = data[d.src_param]
                fitted = None
                if raw is None:
                    lp(k_mid, f"mosaicity: {name} no {d.label} to detrend")
                elif not can_detrend(raw):
                    result.notes.append(f"{name}: {d.label} {_unfittable(raw)} — stored as read")
                    lp(k_mid, f"mosaicity: {name} could not detrend {d.label}")
                else:
                    # `can_detrend` is necessary, not sufficient (a column of
                    # infs passes it), and a detrend is an option on top of the
                    # stacking: a map the fit chokes on loses its detrend, never
                    # the whole volume.
                    try:
                        fitted = _detrend_map(raw)
                    except (ValueError, IndexError, TypeError, ArithmeticError) as exc:
                        result.notes.append(
                            f"{name}: {d.label} trend fit failed ({type(exc).__name__}: {exc})"
                            " — stored as read"
                        )
                    what = "detrended" if fitted is not None else "could not detrend"
                    lp(k_mid, f"mosaicity: {name} {what} {d.label}")
                if fitted is not None:
                    detrended, surface = fitted
                    data[d.src_param] = detrended
                    z = z_next.get(d.out_key, 0)
                    result.detrend_layers.setdefault(d.out_key, []).append((z, layer_idx))
                    if plot_error is None:
                        png = os.path.join(
                            result.detrend_dir,
                            f"{name}_{_KEY_STEM[d.out_key]}_detrend_diag.png",
                        )
                        try:
                            os.makedirs(result.detrend_dir, exist_ok=True)
                            _detrend_diag(raw, detrended, surface, style).savefig(
                                png, dpi=DETREND_DIAG_DPI, bbox_inches="tight", facecolor="white"
                            )
                            result.detrend_plots.append(png)
                        except OSError as exc:
                            # The diagnostics are a by-product: an unwritable
                            # folder costs them, not the volume. Said once and
                            # not retried — every later layer would fail the
                            # same way, after a second-long render each.
                            plot_error = exc
                            result.notes.append(f"detrend diagnostics not saved: {exc}")
                    del fitted, detrended, surface
                    # A Figure and its artists form reference cycles, so the
                    # diagnostic just saved — its canvas and the three maps it
                    # drew — outlives this scope until the cyclic collector
                    # happens to run. Collect now, or ticking both boxes peaks
                    # with the χ figure still alive while the μ one renders.
                    gc.collect()
                # Not `del`: the name must not keep this layer's raw map alive
                # into the next layer's reads, whichever branch ran.
                raw = None
                lp(k_hi, f"mosaicity: {name} {d.label} detrend done")
            # Appending a map to the stacked file is a write per key, and it is
            # the other half of a layer's cost. Reporting only the reads left
            # the whole write in one jump — 0.2 of a one-layer run's entire bar,
            # and `single` mode is one layer by definition.
            for wi, (key, arr) in enumerate(data.items()):
                if arr is not None:
                    group_name, ds_name = routing[key]
                    out.append(f"{group_name}/{ds_name}", arr)
                    out_key = f"/{group_name}/{ds_name}"
                    z_next[out_key] = z_next.get(out_key, 0) + 1
                    what = "wrote"
                else:
                    # `_read_dataset` returns None for a path this maps.h5 does
                    # not have. The report still fires — the bar's granularity
                    # must not depend on which optional datasets a layer
                    # happens to carry — but it must not claim a write that did
                    # not happen, which is the same defect this change fixes in
                    # `visualize`'s product boundaries.
                    what = "skipped"
                lp(0.6 + 0.4 * (wi + 1) / max(1, len(data)), f"mosaicity: {name} {what} {key}")
            del data
            result.layers.append(name)
            result.maps_paths.append(maps_path)
            progress(lay_hi, f"mosaicity: {name} done")

        if not result.layers:
            out.abort()
            progress(1.0, "no mosaicity layers produced")
            return result

        for ds_path in out.datasets():
            result.datasets[f"/{ds_path}"] = out.shape(ds_path)
        progress(LAYERS_HI, "writing stacked volume")
        # `detrended_z` names exactly the slices that were detrended: a map
        # the fit refused is stacked as read, so the method attribute alone
        # would claim the whole volume.
        for out_key in result.detrended:
            zs = [z for z, _layer in result.detrend_layers[out_key]]
            out.set_dataset_attrs(
                out_key, detrend="arctan_2d", detrended_z=np.asarray(zs, dtype=np.int64)
            )
        out.set_attrs(
            num_layers=len(result.layers),
            source_folders="\n".join(result.layers),
            description="Stacked 3D volumes from 2D darfix maps",
        )

    result.stacked_path = stacked_path
    progress(1.0, f"stacked {len(result.layers)} layers -> {os.path.basename(stacked_path)}")
    return result


def _main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Stack darfix mosaicity volumes.")
    ap.add_argument("--mode", choices=("single", "batch"), default="batch")
    ap.add_argument("--input-folder", default="")
    ap.add_argument("--root-folder", default="")
    ap.add_argument("--folder-pattern", default="*")
    ap.add_argument("--detrend-chi-com", action="store_true", help="detrend each χ CoM map")
    ap.add_argument("--detrend-mu-com", action="store_true", help="detrend each μ CoM map")
    ap.add_argument(
        "--output-dir",
        default="",
        help=f"detrend diagnostics folder (default: <root>/{DETREND_DIR})",
    )
    args = ap.parse_args(argv)
    res = run(
        dict(
            mode=args.mode,
            input_folder=args.input_folder,
            root_folder=args.root_folder,
            folder_pattern=args.folder_pattern,
            detrend_chi_com=args.detrend_chi_com,
            detrend_mu_com=args.detrend_mu_com,
            output_dir=args.output_dir,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\n{res.n_layers} layers; stacked -> {res.stacked_path}; datasets {list(res.datasets)}")
    if res.detrended:
        print(f"detrended {', '.join(res.detrended)}; diagnostics -> {res.detrend_dir}")
    for note in res.notes:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
