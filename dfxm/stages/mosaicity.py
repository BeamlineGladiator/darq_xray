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
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field

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
from ..common.plotting import build_histogram
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
            help="Where stacked_volumes.h5 is written (blank = the input/root folder).",
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
    ),
    see_also=(
        SeeAlso(
            "",
            "Colormaps are set per quantity group in “Publication style…” (left panel), not here.",
        ),
    ),
    estimate="dfxm.stages.mosaicity:estimate",
)


@dataclass
class MosaicityResult:
    stacked_path: str | None = None
    datasets: dict[str, tuple[int, ...]] = field(default_factory=dict)
    layers: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def n_layers(self) -> int:
        return len(self.layers)


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


@register("mosaicity")
def figures(result: "MosaicityResult", params: dict) -> list[FigureSpec]:
    """Return one ``map`` + one ``plot`` (histogram) FigureSpec per Z layer per dataset key."""
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


def estimate(params: dict) -> CostEstimate:
    """Peak memory for a mosaicity run, from HDF5 shapes only.

    The arithmetic below, ``(n_present + 1)`` whole volumes, models the *old*
    ``run()``: it held every layer of all present datasets (chi/mu x com/fwhm)
    in a ``collected`` dict at once and then ``np.stack``ed one more contiguous
    volume per dataset on top, which made this one of the heaviest stages. The
    ``collected`` dict and the ``np.stack`` are gone. ``run()`` now ``append``s
    each layer to a ``StackedVolumeFile`` as it is read and drops it, so the
    resident set no longer scales with ``n_layers``: the real peak is one
    layer of each present dataset (the per-folder ``data`` dict) plus the
    writer's per-dataset compression buffers.

    **Recalibration warning — the fix is not simply dropping ``n_layers``.**
    The current figure over-predicts, which is the safe direction, and is
    deliberately left unchanged here. What a per-layer model must still count:

    * ``StackedVolumeFile`` chunks every volume one layer per chunk and (at the
      default ``compression="gzip"``) holds a compression buffer per dataset, so
      the writer's own footprint scales with ``n_present``, not with 1.
    * ``figures()``/``replot`` paths reach the same file through
      ``_streamed_clim`` and ``render_volume_layer``, which are layer-at-a-time
      by construction and are *not* part of ``run()``'s peak — do not fold them
      in.

    A model in bare layer units would sit close enough to the noise floor that
    under-prediction becomes easy, and under-prediction is the dangerous
    direction (it greenlights a run that then OOMs). ``total_input`` and the
    reported ``shape`` still legitimately scale with ``n_layers``; only
    ``peak_bytes`` is the stale part.
    """
    p = {**STAGE.defaults(), **params}
    try:
        work = resolve_layer_work(p, maps_filename=str(p["maps_filename"] or "maps.h5"))
        if not work:
            return CostEstimate(0, 0, None, True, "no layer folders resolved yet")
        present = 0
        layer_shape: tuple[int, ...] = ()
        itemsize = 8
        with h5py.File(work[0], "r") as f:
            for key, _default, _group, _name in _DATASETS:
                ds_path = str(p.get(key) or "")
                if ds_path and ds_path in f:
                    ds = f[ds_path]
                    layer_shape = tuple(int(d) for d in ds.shape)
                    itemsize = int(ds.dtype.itemsize)
                    present += 1
        if not present:
            return CostEstimate(0, 0, None, True, "none of the mosaicity datasets found")
    except Exception as exc:  # noqa: BLE001 - an estimate is advisory, never fatal
        return CostEstimate(0, 0, None, True, f"cannot size input: {type(exc).__name__}")

    layer_elems = 1
    for dim in layer_shape:
        layer_elems *= dim
    n_layers = len(work)
    per_volume = n_layers * layer_elems * itemsize
    input_bytes = present * per_volume
    peak_bytes = (present + 1) * per_volume
    return CostEstimate(
        peak_bytes,
        input_bytes,
        (n_layers, *layer_shape),
        True,
        f"{present} datasets stacked together",
        confidence="conservative",
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
        folders = find_matching_folders(root, p["folder_pattern"])
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
                        lp(0.6 * (di + 1) / max(1, len(config)), f"mosaicity: {name} {key}")
            except OSError as exc:
                result.skipped.append(f"{name}: {exc}")
                continue
            if all(v is None for v in data.values()):
                result.skipped.append(f"{name}: no datasets")
                continue
            # Appending a map to the stacked file is a write per key, and it is
            # the other half of a layer's cost. Reporting only the reads left
            # the whole write in one jump — 0.2 of a one-layer run's entire bar,
            # and `single` mode is one layer by definition.
            for wi, (key, arr) in enumerate(data.items()):
                if arr is not None:
                    group_name, ds_name = routing[key]
                    out.append(f"{group_name}/{ds_name}", arr)
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
            progress(lay_hi, f"mosaicity: {name} done")

        if not result.layers:
            out.abort()
            progress(1.0, "no mosaicity layers produced")
            return result

        for ds_path in out.datasets():
            result.datasets[f"/{ds_path}"] = out.shape(ds_path)
        progress(LAYERS_HI, "writing stacked volume")
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
    args = ap.parse_args(argv)
    res = run(
        dict(
            mode=args.mode,
            input_folder=args.input_folder,
            root_folder=args.root_folder,
            folder_pattern=args.folder_pattern,
        ),
        progress=lambda f, m: print(f"  [{f * 100:5.1f}%] {m}"),
    )
    print(f"\n{res.n_layers} layers; stacked -> {res.stacked_path}; datasets {list(res.datasets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
