"""Compose a machine profile, a cost estimate and a run plan into one Advisory.

Qt-free and side-effect-free apart from the probes it delegates to. This is the
single place that decides *what the user is told* about a run's cost; the four
GUI surfaces render an :class:`Advisory` and compute no policy of their own.

Nothing here influences what a stage does. Since phase 5 each volume stage
derives its own streaming budget from
:func:`~dfxm.common.advice.working_set_budget_bytes` with its own **measured**
``RSS_FLOOR_BYTES``, which a caller cannot guess; the advisory path and the
execution path are parallel, not sequential.
"""

from __future__ import annotations

import os

from ..config.models import StageSpec


def disk_probe_dir(spec: StageSpec, params: dict) -> str:
    """Which directory's filesystem to measure for free space.

    ``output_dir`` when the user set one; otherwise the directory of the first
    filled-in ``must_exist`` input; otherwise the working directory.

    The fallback is not cosmetic. ``output_dir`` is optional on every estimating
    stage — each ``run()`` computes its own default internally — so reading it
    alone would measure the filesystem the *app* was started from while the data
    sits on an external drive, and the scratch-disk check that decides whether a
    run is blocked would be answered about the wrong disk.
    """
    out = str(params.get("output_dir") or "").strip()
    if out:
        return out
    for p in spec.params:
        if not p.must_exist:
            continue
        value = str(params.get(p.name) or "").strip()
        if not value:
            continue
        if os.path.isdir(value):
            return value
        return os.path.dirname(value) or os.getcwd()
    return os.getcwd()
