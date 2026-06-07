"""Lazy registry of stage entry points.

Maps a stage name to a ``"module:function"`` dotted target string. The targets
are resolved (imported) only on demand, so importing this registry never drags
in matplotlib / pyvista / h5py — important for keeping the worker child and the
GUI startup light.

New stages add a line here as they land in later phases.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

# name -> "module:function" (a stage's run(params, progress=None) entry point)
STAGE_TARGETS: dict[str, str] = {
    "concat": "dfxm.stages.concat:run",
    "strain": "dfxm.stages.strain:run",
    "mosaicity": "dfxm.stages.mosaicity:run",
    "visualize": "dfxm.stages.visualize:run",
    "rocking": "dfxm.stages.rocking:run",
    "paraview": "dfxm.stages.paraview:run",
}


def resolve(target: str | Callable) -> Callable:
    """Resolve a ``"module:function"`` string (or a callable) to a callable."""
    if callable(target):
        return target
    module_name, _, func_name = target.partition(":")
    if not func_name:
        raise ValueError(f"target must be 'module:function', got {target!r}")
    return getattr(importlib.import_module(module_name), func_name)


def resolve_stage(name: str) -> Callable:
    """Resolve a registered stage *name* to its run callable."""
    if name not in STAGE_TARGETS:
        raise KeyError(f"unknown stage {name!r}; known: {sorted(STAGE_TARGETS)}")
    return resolve(STAGE_TARGETS[name])
