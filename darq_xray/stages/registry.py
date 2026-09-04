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
    "concat": "darq_xray.stages.concat:run",
    "strain": "darq_xray.stages.strain:run",
    "mosaicity": "darq_xray.stages.mosaicity:run",
    "visualize": "darq_xray.stages.visualize:run",
    "rocking": "darq_xray.stages.rocking:run",
    "paraview": "darq_xray.stages.paraview:run",
    "slices": "darq_xray.stages.slices:run",
    "profiles": "darq_xray.stages.profiles:run",
    "matched": "darq_xray.stages.matched:run",
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
