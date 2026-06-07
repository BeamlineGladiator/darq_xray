"""GUI glue for the lazy interactive viewers.

* :func:`volume_sources` returns, per stage, a mapping ``name -> callable`` where
  each callable loads (and, for the visualize stage, aligns) ONE volume ready for
  3-D rendering. The callables are invoked only when the user clicks "Render 3-D",
  so nothing heavy (volume load / alignment / pyvista) happens otherwise.
* :func:`inject_line_into_jobs` writes a picked line back into a profiles
  ``jobs_json`` string (pure; unit-tested).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable

import h5py
import numpy as np

# A source returns (volume (Z,Y,X), spacing_xyz µm, cmap name, clim or None).
VolumeSource = Callable[[], tuple]


def _rocking_source(aligned_path: str, dataset: str) -> VolumeSource:
    def _load():
        with h5py.File(aligned_path, "r") as f:
            vol = f[dataset][:].astype(float)
            sx = float(f.attrs.get("scale_x_um_per_px", 1.0))
            sy = float(f.attrs.get("scale_y_um_per_px", 1.0))
            sz = float(f.attrs.get("scale_z_um_per_px", 1.0))
        valid = vol[np.isfinite(vol)]
        clim = (
            (float(np.percentile(valid, 1)), float(np.percentile(valid, 99)))
            if valid.size
            else None
        )
        return vol, (sx, sy, sz), "magma", clim

    return _load


def volume_sources(stage_name: str, result, params: dict) -> dict[str, VolumeSource]:
    """Lazy 3-D volume sources for a finished stage run (empty for most stages)."""
    sources: dict[str, VolumeSource] = {}
    if stage_name == "rocking":
        path = getattr(result, "aligned_path", None)
        if path and os.path.exists(path):
            for ds in ("sum_intensity", "specific_frame"):
                sources[ds] = _rocking_source(path, ds)
    elif stage_name == "visualize":
        from dfxm.stages import visualize

        for name in visualize.available_fields(params):
            sources[name] = lambda n=name: visualize.aligned_field(params, n)
    return sources


def inject_line_into_jobs(
    jobs_json: str, slice_name: str, start_uv, end_uv, offset_um: float
) -> str:
    """Return a new jobs_json with the picked line written into the matching job.

    Updates the first job whose ``name`` equals *slice_name* (else the first job);
    if there are no jobs, creates one. ``start_uv``/``end_uv`` are 2-tuples (µm).
    """
    try:
        jobs = json.loads(jobs_json) if jobs_json.strip() else []
    except json.JSONDecodeError:
        jobs = []
    if not isinstance(jobs, list):
        jobs = []
    target = None
    for job in jobs:
        if isinstance(job, dict) and job.get("name") == slice_name:
            target = job
            break
    if target is None:
        target = jobs[0] if jobs and isinstance(jobs[0], dict) else {}
        if target not in jobs:
            jobs.append(target)
        target["name"] = slice_name
    target["offset_um"] = round(float(offset_um), 4)
    target["start_uv"] = [round(float(start_uv[0]), 4), round(float(start_uv[1]), 4)]
    target["end_uv"] = [round(float(end_uv[0]), 4), round(float(end_uv[1]), 4)]
    return json.dumps(jobs, indent=2)
