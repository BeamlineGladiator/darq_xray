"""GUI glue for the lazy interactive viewers.

* :func:`volume_sources` returns, per stage, a mapping ``name -> VolumeSourceSpec``
  where each spec's ``load`` callable loads (and, for the visualize stage,
  aligns) ONE volume ready for 3-D rendering, plus a JSON-able ``loader`` recipe
  describing how to reload it without pickling arrays. The ``load`` callables
  are invoked only when the user clicks "Render 3-D", so nothing heavy (volume
  load / alignment / pyvista) happens otherwise.
* :func:`inject_line_into_jobs` writes a picked line back into a profiles
  ``jobs_json`` string (pure; unit-tested).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import h5py
import numpy as np


@dataclass
class LoadedVolume:
    """One loaded, ready-to-render 3-D volume plus its display metadata."""

    volume: "np.ndarray"
    spacing: tuple
    cmap: str
    clim: tuple | None
    cbar_label: str
    group: str | None


@dataclass
class VolumeSourceSpec:
    """One openable 3-D volume: a lazy loader + a JSON-able reload recipe.

    ``loader`` lets the viewer's child-process video job reload the same
    volume without pickling arrays: ``{"kind": "visualize_field", "stage_params",
    "field"}`` or ``{"kind": "h5_dataset", "path", "dataset"}``.
    """

    name: str
    load: Callable[[], LoadedVolume]
    loader: dict


def _rocking_source(aligned_path: str, dataset: str) -> Callable[[], LoadedVolume]:
    def _load() -> LoadedVolume:
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
        from dfxm.common.plotting import resolve_cmap

        return LoadedVolume(
            volume=vol,
            spacing=(sx, sy, sz),
            cmap=resolve_cmap(None, "raw"),
            clim=clim,
            cbar_label="Intensity",
            group="raw",
        )

    return _load


def _visualize_load(params: dict, name: str) -> LoadedVolume:
    from dfxm.stages import visualize

    vol, spacing, cmap, clim, meta = visualize.aligned_field(params, name)
    return LoadedVolume(
        volume=vol,
        spacing=spacing,
        cmap=cmap,
        clim=clim,
        cbar_label=meta["cbar_label"],
        group=meta["group"],
    )


def volume_sources(stage_name: str, result, params: dict) -> dict[str, VolumeSourceSpec]:
    """Lazy 3-D volume sources for a finished stage run (empty for most stages)."""
    sources: dict[str, VolumeSourceSpec] = {}
    if stage_name == "rocking":
        path = getattr(result, "aligned_path", None)
        if path and os.path.exists(path):
            for ds in ("sum_intensity", "specific_frame"):
                sources[ds] = VolumeSourceSpec(
                    name=ds,
                    load=_rocking_source(path, ds),
                    loader={"kind": "h5_dataset", "path": path, "dataset": ds},
                )
    elif stage_name == "visualize":
        from dfxm.stages import visualize

        for name in visualize.available_fields(params):
            sources[name] = VolumeSourceSpec(
                name=name,
                load=lambda n=name: _visualize_load(params, n),
                loader={
                    "kind": "visualize_field",
                    "stage_params": dict(params),
                    "field": name,
                },
            )
    return sources


def _parse_jobs(jobs_json: str) -> list:
    """Jobs list from a jobs_json string; anything unparseable -> []."""
    try:
        jobs = json.loads(jobs_json) if jobs_json.strip() else []
    except json.JSONDecodeError:
        jobs = []
    return jobs if isinstance(jobs, list) else []


def _set_line(job: dict, start_uv, end_uv, offset_um, fields, reference) -> None:
    """Write the picked-line keys into *job* (shared by inject/append)."""
    job["offset_um"] = round(float(offset_um), 4)
    job["start_uv"] = [round(float(start_uv[0]), 4), round(float(start_uv[1]), 4)]
    job["end_uv"] = [round(float(end_uv[0]), 4), round(float(end_uv[1]), 4)]
    if fields is not None:
        job["fields"] = list(fields)
    else:
        job.pop("fields", None)
    if reference:
        job["reference"] = str(reference)


def inject_line_into_jobs(
    jobs_json: str,
    slice_name: str,
    start_uv,
    end_uv,
    offset_um: float,
    fields=None,
    reference=None,
) -> str:
    """Return a new jobs_json with the picked line written into the matching job.

    Updates the first job whose ``name`` equals *slice_name* (else the first job);
    if there are no jobs, creates one. ``start_uv``/``end_uv`` are 2-tuples (µm).
    A truthy *reference* sets the job's ``"reference"`` (the background group the
    line was drawn against); ``None`` leaves any existing value untouched.
    """
    jobs = _parse_jobs(jobs_json)
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
    _set_line(target, start_uv, end_uv, offset_um, fields, reference)
    return json.dumps(jobs, indent=2)


def append_line_job(
    jobs_json: str,
    slice_name: str,
    start_uv,
    end_uv,
    offset_um: float,
    fields=None,
    reference=None,
) -> str:
    """Append ONE complete job to *jobs_json* — never edits existing jobs.

    Unlike :func:`inject_line_into_jobs` (which updates the first job matching
    the slice name), this always appends, so several marks on one slice each
    become their own job (the profiles stage de-duplicates output stems for
    same-named jobs).
    """
    jobs = _parse_jobs(jobs_json)
    job: dict = {"name": slice_name}
    _set_line(job, start_uv, end_uv, offset_um, fields, reference)
    jobs.append(job)
    return json.dumps(jobs, indent=2)
