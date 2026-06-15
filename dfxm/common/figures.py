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


def figures_for(stage_name: str, result, params: dict) -> list[FigureSpec]:
    _load_stage_catalogs()
    fn = _FIGURE_CATALOGS.get(stage_name)
    return fn(result, params) if fn else []
