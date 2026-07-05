"""Per-stage figure catalog: enumerate + rebuild saved figures at any PlotStyle.

Qt-free. The GUI calls :func:`figures_for` to list a result's figures and
``spec.build(style)`` to get a Matplotlib Figure rebuilt from the persisted data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from matplotlib.figure import Figure

from .plotting import PlotStyle


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

    from . import render

    with h5py.File(h5_path, "r") as f:
        n_z = f[dataset].shape[0]
        ext_x = f[dataset].shape[2] * sx
        ext_y = f[dataset].shape[1] * sy

    if z_um is not None and len(z_um) != n_z:
        raise ValueError(f"z_um has {len(z_um)} entries but dataset '{dataset}' has {n_z} Z layers")

    def make(z):
        def build(style):
            from .plotting import resolve_cmap

            layer = _load_layer(h5_path, dataset, z)
            zlabel = f"\nZ = {z_um[z]:.2f} µm" if z_um is not None else ""
            fig, _, _ = render.layer_figure(
                layer,
                vmin,
                vmax,
                resolve_cmap(style, cmap_group, fallback=cmap),
                ext_x,
                ext_y,
                f"{title}{zlabel} (layer {z})",
                cbar_label,
                style=style,
                group=cmap_group,
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
