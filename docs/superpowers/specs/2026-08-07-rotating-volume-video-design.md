# Rotating 3-D volume video (visualize stage) — design

**Date:** 2026-08-07
**Status:** approved (Albert, via AskUserQuestion)
**Execution mode:** inline, single end review

## Goal

Restore the rotating 3-D volume movie that the legacy Scripts2
`build_aligned_raw_rocking_volumes_v3.py` produced and that the Phase 2 port
deliberately dropped (`dfxm/stages/rocking.py:19` — "pure eye-candy"). The
modern version renders the **same pyvista volume** the top-view PNG and the 3D
tab show (NaN padding hidden, publication colormap, opacity knob) with the
camera orbiting 360° around it, written as MP4 (GIF fallback).

**Scope:** visualize stage only (Albert's choice). The renderer lands in the
shared `dfxm/common/render.py`, so wiring rocking later is a small follow-up.

## Approach

Chosen: **orbit with pyvista, write with the existing matplotlib writers.**
No new dependencies; inherits the `mp4`/`gif`/`both` + ffmpeg→GIF fallback
semantics the layer animation already has. (Rejected: pyvista `open_movie` —
new `imageio` deps; ffmpeg-CLI frame assembly — temp-file churn, no GIF
fallback.)

## Components

### 1. `dfxm/common/render.py`

New public function beside `save_top_view`:

```python
def save_rotation_video(volume, scale_z, sx, sy, vmin, vmax, cmap, opacity,
                        base_path, fmt, *, n_frames=120, fps=15):
```

- Lazy `import pyvista` (module stays import-light; headless-safe).
- Reuses `_pyvista_grid` (NaN voxels thresholded out); returns `None` when the
  grid is empty (same contract as `save_top_view`).
- Off-screen `Plotter`, same `add_mesh` styling as the top view; camera starts
  from the isometric view, then steps azimuth by `360 / n_frames` per frame
  (120 frames @ 15 fps ≈ 8 s loop).
- Frames are grabbed one at a time (`screenshot(return_img=True)`) inside the
  animation update — no full frame stack in memory.
- Frame→movie assembly is factored into a private helper
  `_write_image_video(get_frame, n_frames, base_path, fmt, fps)` — where
  `get_frame(i)` returns the RGB ndarray for frame *i* — that drives a
  borderless matplotlib figure (`imshow`, axes off, figsize matched
  to the frame's pixel size) through `FuncAnimation` and saves via
  `FFMpegWriter` → `PillowWriter` exactly like `save_layer_animation`
  (mp4 attempt, GIF fallback, returns the written path). This helper takes
  plain numpy frames, so it is unit-testable without GL.
- Returns the written path (`base_path + ".mp4"`/".gif") or `None`.

### 2. `dfxm/stages/visualize.py`

- New param (Output group, advanced):
  - `save_rotation` (BOOL, label "Save rotating 3-D video", default **False**
    — it is the expensive render; help text written for a first-time beamline
    user, mentioning it reuses "Animation format" and "3D opacity").
- Reuses existing `output_format` (mp4/gif/both) and `volume_opacity`.
- `DatasetProducts` gains `rotation_video: str | None = None`.
- After the top-view block, when `p["save_rotation"]`: call
  `Rnd.save_rotation_video(...)` writing `<name>_rotation.{mp4,gif}` in the
  dataset dir; wrap in the same try/except pattern as the top view — any
  pyvista/GL failure appends `notes.append(f"rotation video skipped: {exc}")`,
  never raises.
- Progress message before rendering (it is slow).

### 3. GUI (`gui/stage_view.py`)

- Form: automatic (schema-driven) — no GUI code for the param.
- `_summarize_visualize` artifact list gains `"spin"` when
  `d.rotation_video` is set (alongside `layers`/`anim`/`3d`).
- Image pickers untouched (a video is not a preview image).

### 4. Docs (same change, both files)

- `docs/Usage.md`: visualize section — output list + param table row; note
  that colormaps/opacity follow the same knobs as the top view.
- `docs/Codebase.md`: `dfxm/common/render.py` entry (+ new function) and the
  visualize stage entry (`save_rotation`, `rotation_video`).

### 5. Tests

- Existing suites keep `save_topview` off and get `save_rotation` default-off
  (GL not guaranteed in CI — same policy as the top view).
- New unit test for `_write_image_video` with small synthetic numpy frames
  (GIF path; no GL, no ffmpeg needed).
- Summarizer test: `spin` appears in the visualize summary when
  `rotation_video` is set.
- `tests/test_param_metadata.py` covers the new param automatically.
- The real GL orbit render remains a real-data eyeball item (as the top view
  is today).

## Error handling

Identical philosophy to `save_top_view`: degrade, never fail the run. Empty
grid → `None`; pyvista/GL exception → per-dataset note; missing ffmpeg → GIF.

## Out of scope

- Rocking-stage wiring (follow-up; the shared helper makes it ~5 lines).
- Camera-path/elevation/fps GUI knobs (constants; YAGNI until asked).
- Interactive 3D-tab recording.
