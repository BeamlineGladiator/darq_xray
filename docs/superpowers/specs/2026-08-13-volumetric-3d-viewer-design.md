# Volumetric 3-D rendering + pop-out 3-D viewer — design

**Date:** 2026-08-13
**Status:** approved (brainstorm with Albert)

## Problem

1. **The rotation video does not rotate.** `save_rotation_video`
   (`dfxm/common/render.py`) was merged 2026-08-07 with mocked tests only; on a
   real GL run the camera never visibly moves. Albert's original standalone
   script (predecessor of the visualize stage) produced correct orbits with a
   different camera recipe and a different rendering technique.
2. **The embedded 3-D tab is minimal.** One `add_mesh` render, no controls —
   no opacity, colour-range, log-scale, colormap, threshold, clipping or camera
   control, no way to export a figure with a colorbar, and the VTK context and
   volume stay in memory as long as the tab lives.

## Decisions (from brainstorm)

- **Render technique:** true volumetric rendering (`add_volume`, shaded) is the
  default everywhere; mode stays selectable (volumetric / surface / isosurface).
- **Video design:** publication-ready **white** style matching the pipeline's
  2-D figures — colorbar with quantity label + units, µm scale bar, per-quantity
  tick formats. No side-by-side layer panel (the old script's layout is dead).
- **Viewer:** pop-out window per volume (multiple allowed), launched from a
  slimmed 3-D tab; closing a window frees its VTK context and volume memory.
- **Controls v1:** render mode, colormap, clim + auto-reset, log scale, opacity
  value + transfer-function mapping, value threshold, clipping plane, downsample,
  background, camera presets + editable azimuth/elevation/zoom, µm bounds axes.
- **Exports v1:** styled figure (PNG), raw screenshot, rotation video from the
  current settings. **Not** in v1: PDF/SVG 3-D figures, viewer-settings
  persistence.

## Part A — shared 3-D render core (`dfxm/common/render3d.py`, Qt-free)

New module; `_pyvista_grid`, `_volume_plotter`, `save_top_view` and
`save_rotation_video` move here out of `render.py` (which stays 2-D).
`visualize.py` is the only caller and is updated in the same change. All pyvista
imports stay lazy (inside functions), per the repo rule.

### `Scene3D` dataclass

One description of "what to render", shared verbatim by stage exports and the
GUI viewer:

- `volume` (Z, Y, X float array), `spacing` (sx, sy, sz µm)
- `mode`: `"volume" | "surface" | "isosurface"`, `n_isosurfaces` (default 10)
- `cmap`, `clim` (vmin, vmax), `log_scale: bool`
- `opacity: float` (0–1) and `opacity_mapping`:
  `"linear" | "sigmoid" | "geom" | "geom_r"`
- `threshold: (tmin, tmax) | None` — hide voxels outside the value window
  (independent of clim, which only recolours)
- `clip: (origin, normal) | None` — single clipping plane
- `downsample: int` (block-average factor, 1 = off)
- `background` colour

### `populate(plotter, scene)`

Builds actors into **any** pyvista plotter — the GUI hands its `QtInteractor`,
export paths hand an off-screen `Plotter`. Per mode:

- **volume:** `add_volume(grid, cmap, clim, opacity=<mapping>, shade=True,
  ambient=0.3, diffuse=0.6, specular=0.2)` — the old script's look. NaN → 0
  before upload (transparent under the transfer function).
- **surface:** current behaviour — NaN-sentinel `threshold()` mesh via
  `add_mesh(smooth_shading=True)`.
- **isosurface:** N contour levels between clim, opacity ramped by level
  (ported from the old script).

Value-threshold and clip are applied as grid filters before the actor is added.
`log_scale` maps the scalars/colour transfer logarithmically and is only valid
for all-positive clim (callers guard).

### `CameraSpec` + the proven camera recipe

`CameraSpec(azimuth, elevation, zoom)` (plus named presets top/front/side/iso).
Applied **exactly** as the working script did: reset `camera_position = "xy"`,
then `camera.azimuth = value`, `camera.elevation = value`, `camera.zoom(zoom)`,
and `plotter.show(auto_close=False)` before any screenshot. Every off-screen
frame sets an *absolute* pose from a clean reset — no incremental `Azimuth()`
calls (the current broken approach), and idempotent in the frame index, which
`_write_image_video`'s replay (`fmt="both"`, MP4→GIF fallback) requires.

### Styled figure compositor

`styled_scene_figure(img, *, quantity/group, clim, log_scale, cmap,
extent_um, style)` → matplotlib `Figure` (explicit-Figure API, Agg-safe):
white background, the rendered image, a colorbar built from a
`ScalarMappable` (`LogNorm` when log), quantity label + units and the
pipeline's per-quantity tick formats, and a µm scale bar.

**Exact scale bar:** export renders use **parallel projection**; px-per-µm then
follows directly from `camera.parallel_scale` and the image height, so the
scale bar is exact rather than the old script's estimate. The interactive
viewer keeps its default projection; only exports force parallel.

### Rewritten products

- `save_top_view` — off-screen `populate` + top `CameraSpec` + compositor.
- `save_rotation_video` — **one** off-screen plotter built once; per frame the
  absolute camera recipe at `azimuth = i * 360/n`, elevation 20°, zoom 1.2
  (script defaults); frames feed the compositor figure built **once**, with
  only the imshow data swapped per frame, so every frame carries the colorbar
  at video-friendly cost. Defaults: 180 frames, 15 fps, MP4 with GIF fallback
  (existing `_write_image_video` container semantics).

### Visualize stage parameters (new, `StageSpec`-driven)

- `render_mode` enum (`volume` default / `surface` / `isosurface`)
- `opacity_mapping` enum (`linear` default)
- `rotation_frames` int (default 180)
- `log_scale` bool (default off; falls back to linear with a result **note** —
  not an exception — when the data window includes non-positive values)

Existing `volume_opacity` and `save_topview`/`save_rotation` params keep their
meaning. Docs contract: `Usage.md` + `Codebase.md` updated in the same change.

## Part B — pop-out 3-D viewer (`gui/`)

### Launcher

`Volume3DPanel` (the 3-D tab) slims to: volume dropdown + **"Open 3D viewer…"**
button + status label. The embedded `PvCanvas` render path is removed from the
tab. Each click opens an independent `Viewer3DWindow` titled
`<stage> — <volume>`; multiple windows may coexist (side-by-side comparison).

### `gui/widgets/viewer3d_window.py`

A top-level window: `QtInteractor` centre (lazy GL, same graceful degradation
as `PvCanvas`), control panel at the side driving a single `Scene3D`:

- render mode combo (+ isosurface count), colormap combo (pipeline cmaps),
- clim min/max spinboxes + "auto (1–99%)" reset button,
- log-scale checkbox — disabled with a hint when the clim window includes
  values ≤ 0,
- opacity slider + mapping combo,
- threshold enable + min/max,
- clip plane enable + axis presets + flip (interactive plane widget in view),
- downsample spinbox,
- background picker (theme / white / black),
- camera preset buttons (Top / Front / Side / Iso) + editable
  azimuth / elevation / zoom fields (reproducible POVs),
- µm-labelled bounds-axes toggle (`show_bounds`).

Cheap changes (opacity value, cmap, clim, background, camera) mutate the actor
or plotter in place; structural changes (mode, mapping, threshold, clip,
downsample) rebuild via `populate`. Closing the window closes the plotter,
destroys the interactor and drops the volume reference — memory returns.

### Exports (toolbar)

- **Save figure…** — off-screen re-render of the current `Scene3D` +
  `CameraSpec` at a user-chosen resolution (not window-bound), through the
  Part A compositor → styled white PNG with colorbar + scale bar.
- **Save screenshot…** — raw grab of the on-screen render.
- **Save rotation video…** — orbit around the current settings; runs in a
  **child process** (the `dfxm.runner` pattern) so the GUI stays responsive
  and off-screen GL never contends with the window's context; progress +
  cancel like a stage run.

### Sources metadata

`volume_sources()` (`gui/viewers.py`) grows per-volume metadata: quantity
group / units label (for the colorbar) and initial clim — so the window can
label its colorbar and validate log without guessing.

## Error handling

- No GL / pyvistaqt import failure → the window shows the same placeholder
  label pattern as `PvCanvas`; launcher stays usable.
- Empty grid after threshold/clip → in-window status note (skip-style, no
  exception), matching the stage's empty-grid note behaviour.
- ffmpeg missing → existing GIF fallback, unchanged.

## Testing

- **Qt-free core:** unit tests for `Scene3D` validation, downsample, threshold
  /clip grid maths, compositor figure (Agg — colorbar label, log norm, scale-bar
  length), video container plumbing with a fake frame source.
- **Real-GL regression (the one that was missing):** render frames at azimuth
  0°/90°/180° and assert the images differ pairwise; `pytest.mark.skipif` when
  no usable GL context. Run locally before merge.
- **GUI smoke:** open viewer window with mocked/absent GL, toggle controls,
  close (memory path); launcher tab wiring.
- **Real-data eyeball:** headless CLI visualize run on STO2 (top view + video)
  + on-screen viewer session, before merge is called done.

## Out of scope (v1)

- PDF/SVG containers for 3-D figures.
- Per-quantity persistence of viewer settings (QSettings) — future follow-up.
- Rocking-stage wiring of the rotation video (tracked separately in memory).
