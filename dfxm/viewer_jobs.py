"""Qt-free jobs the GUI runs in a child process (via dfxm.runner.StageRunner).

Sits ABOVE dfxm.common and dfxm.stages in the layering (it may import both);
gui/ builds the JSON-able params, the runner resolves
"dfxm.viewer_jobs:rotation_video_job" in the child.
"""

from __future__ import annotations

import h5py

from .common import render3d as R3
from .common import volumeio
from .common.plotting import style_from_json


def _load_volume(loader: dict):
    """``(volume, spacing, notes)`` from a JSON-able loader spec.

    Like the interactive viewer this feeds, the ``h5_dataset`` path is a *render*
    load: the whole array is uploaded to VTK, so streaming cannot help it and an
    oversized volume is decimated on read instead of dying. Both loaders share
    :mod:`dfxm.common.volumeio`'s display policy, so the exported video is
    coarsened by the same *rule* as the view the user was looking at — not
    necessarily by the same *factor*, since ``display_headroom_bytes()`` re-reads
    available RAM here, in a separate process that may meet different memory
    pressure than the window did. ``notes`` carries the user-facing sentence
    naming the factor actually used (empty when nothing was decimated).
    """
    if loader["kind"] == "h5_dataset":
        notes: tuple[str, ...] = ()
        with h5py.File(loader["path"], "r") as f:
            dset = f[loader["dataset"]]
            step = volumeio.display_decimation(dset, volumeio.display_headroom_bytes())
            if step > 1:
                notes = (volumeio.decimation_note(step, dset.shape),)
            vol = dset[::step, ::step, ::step].astype(float)
            # Scaled by the stride, or the video would show the volume at
            # 1/step of its true physical size (wrong scale bar, wrong bounds).
            spacing = tuple(
                float(f.attrs.get(f"scale_{ax}_um_per_px", 1.0)) * step for ax in ("x", "y", "z")
            )
        return vol, spacing, notes
    if loader["kind"] == "visualize_field":
        from .stages.visualize import aligned_field

        vol, spacing, _cmap, _clim, _meta = aligned_field(loader["stage_params"], loader["field"])
        return vol, spacing, ()
    raise ValueError(f"unknown loader kind {loader.get('kind')!r}")


def rotation_video_job(params: dict, progress=None) -> dict:
    """Render a rotation video from a viewer window's settings.

    Returns ``{"video": path, "notes": [...]}`` — ``notes`` carries anything the
    user needs told about the export (currently: that the volume was decimated
    to fit this machine). It is also emitted through *progress* as soon as it is
    known, so the message is visible while the render runs, not only after it.
    """
    vol, spacing, notes = _load_volume(params["loader"])
    if notes and progress is not None:
        progress(0.0, notes[0])
    s = dict(params["scene"])
    scene = R3.Scene3D(
        volume=vol,
        spacing=spacing,
        mode=s["mode"],
        cmap=s["cmap"],
        clim=tuple(s["clim"]) if s.get("clim") else None,
        log_scale=bool(s.get("log_scale", False)),
        opacity=float(s.get("opacity", 0.85)),
        opacity_mapping=s.get("opacity_mapping", "linear"),
        threshold=tuple(s["threshold"]) if s.get("threshold") else None,
        clip=(tuple(s["clip"][0]), tuple(s["clip"][1])) if s.get("clip") else None,
        downsample=int(s.get("downsample", 1)),
        background=s.get("background", "white"),
    )
    base_camera = (
        tuple(tuple(float(x) for x in v) for v in params["base_camera"])
        if params.get("base_camera")
        else None
    )
    video = R3.save_rotation_video(
        scene,
        params["base_path"],
        params["fmt"],
        cbar_label=params["cbar_label"],
        group=params.get("group"),
        style=style_from_json(params.get("style_json") or ""),
        n_frames=int(params.get("n_frames", 180)),
        fps=int(params.get("fps", 15)),
        elevation=float(params.get("elevation", 20.0)),
        zoom=float(params.get("zoom", 1.2)),
        base_camera=base_camera,
        progress=progress,
    )
    return {"video": video, "notes": list(notes)}
