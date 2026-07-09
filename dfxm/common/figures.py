"""Per-stage figure catalog: enumerate + rebuild saved figures at any PlotStyle.

Qt-free. The GUI calls :func:`figures_for` to list a result's figures and
``spec.build(style)`` to get a Matplotlib Figure rebuilt from the persisted data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from matplotlib.figure import Figure

from .plotting import PlotStyle


@dataclass
class ReplotGroup:
    """One selectable group in a replot catalog: a dataset/product with N layers."""

    key: str  # in-file dataset key (mosaicity/rocking) or logical group id
    label: str  # tree display label
    item_labels: list[str] = field(default_factory=list)  # per-layer labels
    shape: tuple[int, int] | None = None  # stored layer (Y, X) pixel shape (ROI-crop hint)


def resolve_clim(clim, key):
    """Pick the ``(vmin, vmax)`` override for one replot group *key*.

    ``clim`` may be ``None`` (keep stored/default limits), a single
    ``(vmin, vmax)`` tuple (legacy — applies to every group), or a
    ``{group_key: (vmin, vmax)}`` mapping (per-kind limits). A key missing from
    the mapping resolves to ``None`` so that group keeps its stored limits.
    """
    if clim is None:
        return None
    if isinstance(clim, dict):
        return clim.get(key)
    return clim


@dataclass
class FigureSpec:
    figure_id: str
    title: str
    kind: str  # "map" (gets a scale bar) | "plot"
    filename: str  # suggested export stem (no extension)
    build: Callable[[PlotStyle | None], Figure]


# stage name -> fn(result, params) -> list[FigureSpec]
_FIGURE_CATALOGS: dict[str, Callable[[object, dict], list[FigureSpec]]] = {}


def register(stage_name: str) -> Callable[[Callable], Callable]:
    def deco(fn):
        _FIGURE_CATALOGS[stage_name] = fn
        return fn

    return deco


def _empty(result: object, params: dict) -> list[FigureSpec]:
    return []


# concat and paraview never produce saved figures — register immediately.
for _name in ("concat", "paraview"):
    _FIGURE_CATALOGS[_name] = _empty

# Pre-populate all 7 map-stage keys with the empty placeholder so that
# ``set(_FIGURE_CATALOGS) == set(STAGE_TARGETS)`` holds immediately after
# import (no figures_for call needed). Real @register(name) catalogs added by
# later tasks will overwrite these via direct dict assignment.
# Keep this list explicit — do NOT derive from STAGE_TARGETS so the drift
# test stays meaningful.
for _name in ("strain", "mosaicity", "visualize", "rocking", "slices", "matched", "profiles"):
    _FIGURE_CATALOGS.setdefault(_name, _empty)

# Laziness guard: stage modules are NOT imported at module level.  They are
# imported on the first figures_for() call so that ``import dfxm.common.figures``
# remains cheap and headless-safe (no h5py/scipy pulled in).
_stage_catalogs_loaded: bool = False


def _load_stage_catalogs() -> None:
    """Import all stage modules so their @register decorators fire.

    Idempotent: subsequent calls are no-ops.
    """
    global _stage_catalogs_loaded
    if _stage_catalogs_loaded:
        return
    # imported here to avoid importing heavy deps at module load time
    from dfxm.stages import (  # noqa: F401
        matched,
        mosaicity,
        profiles,
        rocking,
        slices,
        strain,
        visualize,
    )

    _stage_catalogs_loaded = True


def _load_layer(h5_path: str, dataset: str, z: int):
    import h5py

    with h5py.File(h5_path, "r") as f:
        return f[dataset][z]


def load_middle_layer(h5_path: str, dataset: str) -> np.ndarray:
    """Return the middle-Z 2-D layer of a (Z,Y,X) HDF5 dataset (ROI-picker preview)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        dset = f[dataset]
        z = dset.shape[0] // 2
        return dset[z][...]


def stacked_volume_previews(params: dict) -> list:
    """(label, thunk) ROI-picker previews from a stacked mosa/strain volume file.

    Middle-Z layers of the chi/mu CoM (mosa) and strain datasets named in the
    form. Qt-free; returns [] when no readable volume file is set. Shared by the
    co-registration stages' ``roi_previews`` (visualize/paraview/slices).
    """
    import os

    p = dict(params)
    sx = float(p.get("pixel_size_x_um", 0.152))
    sy = float(p.get("pixel_size_y_um", 0.385))
    out = []
    mosa = p.get("mosa_volume_file", "") or ""
    strain = p.get("strain_volume_file", "") or ""
    if mosa and os.path.exists(mosa):
        import h5py

        try:
            with h5py.File(mosa, "r") as f:
                present = [ds for ds in ("chi/Center of mass", "mu/Center of mass") if ds in f]
        except Exception:  # noqa: BLE001
            present = []
        for ds in present:
            out.append(
                (
                    f"{ds} · {os.path.basename(mosa)}",
                    (lambda _m=mosa, _d=ds, _sx=sx, _sy=sy: (load_middle_layer(_m, _d), _sx, _sy)),
                )
            )
    if strain and os.path.exists(strain):
        out.append(
            (
                f"strain · {os.path.basename(strain)}",
                (lambda _s=strain, _sx=sx, _sy=sy: (load_middle_layer(_s, "strain"), _sx, _sy)),
            )
        )
    return out


def crop_roi_2d(layer: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray | None:
    """Crop a 2-D array to ``(r0, r1, c0, c1)`` pixel bounds, clamped to shape.

    ``roi=None`` returns *layer* unchanged. Returns ``None`` when the (clamped)
    crop is empty. Replot ROI is a reframe of stored data, never a recompute.
    """
    if roi is None:
        return layer
    r0, r1, c0, c1 = roi
    h, w = layer.shape[:2]
    r0 = max(0, min(int(r0), h))
    r1 = max(0, min(int(r1), h))
    c0 = max(0, min(int(c0), w))
    c1 = max(0, min(int(c1), w))
    if r1 <= r0 or c1 <= c0:
        return None
    return layer[r0:r1, c0:c1]


def _apply_clim(vmin: float, vmax: float, clim: tuple[float | None, float | None] | None):
    if clim is None:
        return vmin, vmax
    lo, hi = clim
    return (lo if lo is not None else vmin, hi if hi is not None else vmax)


def render_volume_layer(
    h5_path: str,
    dataset: str,
    z: int,
    *,
    cmap: str,
    cmap_group: str | None,
    title: str,
    cbar_label: str,
    sx: float,
    sy: float,
    vmin: float,
    vmax: float,
    style,
    clim: tuple[float | None, float | None] | None = None,
    roi: tuple[int, int, int, int] | None = None,
    z_um: list[float] | None = None,
) -> Figure | None:
    """Read one (Z,Y,X) layer, optionally crop + clim-override, return a map Figure.

    Returns ``None`` when the ROI crop is empty. Shared by ``volume_layer_specs``
    (export) and the mosaicity/rocking ``render_replot`` (cold replot).
    """
    from . import render
    from .plotting import resolve_cmap

    layer = _load_layer(h5_path, dataset, z)
    layer = crop_roi_2d(layer, roi)
    if layer is None:
        return None
    ext_x = layer.shape[1] * sx
    ext_y = layer.shape[0] * sy
    v0, v1 = _apply_clim(vmin, vmax, clim)
    zlabel = f"\nZ = {z_um[z]:.2f} µm" if z_um is not None else ""
    fig, _, _ = render.layer_figure(
        layer,
        v0,
        v1,
        resolve_cmap(style, cmap_group, fallback=cmap),
        ext_x,
        ext_y,
        f"{title}{zlabel} (layer {z})",
        cbar_label,
        style=style,
        group=cmap_group,
    )
    return fig


def volume_layer_specs(
    *,
    h5_path: str,
    dataset: str,
    id_prefix: str,
    title: str,
    cbar_label: str,
    cmap: str,
    cmap_group: str | None = None,
    sx: float,
    sy: float,
    vmin: float,
    vmax: float,
    z_um: list[float] | None = None,
) -> list[FigureSpec]:
    """One ``map`` FigureSpec per Z layer of a (Z,Y,X) HDF5 volume.

    Opens the file once eagerly (for the shape); each ``build(style)`` re-opens
    it to read exactly one layer (memory-light for large volumes). ``z_um``, if
    given, must have length == n_z and is used only in the rendered axes title,
    not in ``FigureSpec.title``.
    """
    import h5py

    with h5py.File(h5_path, "r") as f:
        n_z = f[dataset].shape[0]

    if z_um is not None and len(z_um) != n_z:
        raise ValueError(f"z_um has {len(z_um)} entries but dataset '{dataset}' has {n_z} Z layers")

    def make(z):
        def build(style):
            fig = render_volume_layer(
                h5_path,
                dataset,
                z,
                cmap=cmap,
                cmap_group=cmap_group,
                title=title,
                cbar_label=cbar_label,
                sx=sx,
                sy=sy,
                vmin=vmin,
                vmax=vmax,
                style=style,
                z_um=z_um,
            )
            return fig

        return build

    return [
        FigureSpec(
            figure_id=f"{id_prefix}_z{z:04d}",
            title=f"{title} — layer {z}",
            kind="map",
            filename=f"{id_prefix}_layer_{z:04d}",
            build=make(z),
        )
        for z in range(n_z)
    ]


def figures_for(stage_name: str, result, params: dict) -> list[FigureSpec]:
    _load_stage_catalogs()
    fn = _FIGURE_CATALOGS.get(stage_name)
    return fn(result, params) if fn else []
