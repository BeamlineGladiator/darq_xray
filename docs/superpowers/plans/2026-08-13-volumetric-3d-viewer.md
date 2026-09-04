# Volumetric 3-D Render Core + Pop-out Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the non-rotating orbit video by rebuilding the 3-D pipeline on true volumetric rendering with publication-styled output, and replace the bare 3-D tab with pop-out viewer windows carrying ParaView-style controls and figure/video export.

**Architecture:** A new Qt-free `dfxm/common/render3d.py` owns everything 3-D: a `Scene3D` dataclass (one description of "what to render", shared verbatim by stage exports and the GUI), `populate(plotter, scene)` that builds actors into any pyvista plotter, pure-numpy camera-orbit math (absolute per-frame poses — the fix for the rotation bug), and a matplotlib compositor that wraps rendered images in the pipeline's white publication style (colorbar + exact µm scale bar via parallel projection). The GUI gains `Viewer3DWindow` (pop-out, one per volume, memory freed on close) driving a `Scene3D` against a lazy `PvCanvas`; its video export runs `dfxm/viewer_jobs.py:rotation_video_job` in a child process via the existing `StageRunner`.

**Tech Stack:** Python 3, numpy, pyvista (lazy), pyvistaqt (GUI only, lazy), matplotlib explicit-Figure/Agg API, PySide6 (gui/ only), h5py, pytest.

**Spec:** `docs/superpowers/specs/2026-08-13-volumetric-3d-viewer-design.md`

## Global Constraints

- `dfxm/` stays Qt-free: never import PySide6/pyvistaqt there; `pyvista` imports only **inside** functions.
- Plotting uses the explicit `matplotlib.figure.Figure` API — never `pyplot`, never `matplotlib.use(...)`.
- Docs contract: any task changing `dfxm/stages/` or `gui/` behaviour updates `docs/Usage.md` and/or `docs/Codebase.md` **in the same commit**.
- `ruff format` runs automatically on Write/Edit (hook); run `ruff check .` before each commit.
- This repo has **no git remote** — commit locally, never push.
- Read any pre-existing file once before its first Edit; never reconstruct `old_string` from memory (`hint=` strings in `dfxm/stages/*.py` contain em-dashes at varying indents).
- The GUI smoke test is `tests/gui_smoke.py` (not pytest); run as `python3 tests/gui_smoke.py`. Full suite: `python3 -m pytest -q`.
- Existing behaviour to preserve: visualize's 3-D products are best-effort (`notes` on failure, never exceptions); `_save_animation`'s MP4→GIF fallback semantics; the `(Z, Y, X)` volume convention and `(sx, sy, sz)` µm spacing.
- If `ParamType.INT` does not exist in `dfxm/config/models.py`, use `ParamType.FLOAT` with an `int(...)` cast at the use site (check before writing Task 5).

## File Structure

- **Create** `dfxm/common/render3d.py` — Scene3D/CameraSpec, volume prep (downsample/threshold/clip masks), grid builders, `populate`, camera math, `render_scene_image`, `scene_figure` compositor, `save_top_view`, `save_rotation_video`, `_video_from_frames`.
- **Create** `dfxm/viewer_jobs.py` — JSON-able child-process job: `rotation_video_job(params, progress)` (may import `dfxm.stages.visualize`; `common/` may not).
- **Create** `gui/widgets/viewer3d_window.py` — the pop-out viewer window (controls + exports).
- **Modify** `dfxm/common/render.py` — remove `_pyvista_grid`, `_volume_plotter`, `save_top_view`, `save_rotation_video`, `_write_image_video` (module becomes 2-D only).
- **Modify** `dfxm/stages/visualize.py` — new params, Scene3D-based call sites, log-scale guard note, `aligned_field` returns metadata.
- **Modify** `gui/viewers.py` — `VolumeSourceSpec`/`LoadedVolume` dataclasses, loader specs for child-process reload.
- **Modify** `gui/widgets/volume3d.py` — becomes the launcher (dropdown + "Open 3D viewer…").
- **Modify** `gui/widgets/pv_canvas.py` — drop `show_volume` (superseded by `render3d.populate`).
- **Tests:** create `tests/test_render3d.py`, `tests/test_render3d_gl.py`, `tests/test_viewer_jobs.py`, `tests/test_gui_viewer3d.py`; rewrite `tests/test_render_rotation.py`; extend `tests/test_gui_viewers.py`, `tests/gui_smoke.py`.

---

### Task 1: Scene3D, CameraSpec and pure volume/camera math

**Files:**
- Create: `dfxm/common/render3d.py`
- Test: `tests/test_render3d.py`

**Interfaces:**
- Consumes: nothing new (numpy only at module level).
- Produces (later tasks rely on these exact names):
  - `Scene3D(volume, spacing, mode="volume", n_isosurfaces=10, cmap="magma", clim=None, log_scale=False, opacity=0.85, opacity_mapping="linear", threshold=None, clip=None, downsample=1, background="white")` with methods `resolved_clim() -> tuple[float, float]` and `prepared() -> tuple[np.ndarray, tuple[float, float, float]]`.
  - `CameraSpec(preset="front", azimuth=0.0, elevation=0.0, zoom=1.0)`; `PRESETS = ("front", "top", "side", "iso")`.
  - `downsample_volume(vol, factor) -> np.ndarray`, `threshold_mask(vol, window) -> np.ndarray`, `clip_mask(vol, spacing, origin, normal) -> np.ndarray`, `auto_clim(vol, lo=1.0, hi=99.0) -> tuple[float, float]`, `log_valid(clim) -> bool`, `orbit_positions(base_camera, elevation_deg, n_frames) -> list[tuple]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Qt-free 3-D scene core — pure numpy parts (no pyvista, no GL)."""

from __future__ import annotations

import numpy as np
import pytest

from dfxm.common import render3d as R3


def _vol():
    # (Z=2, Y=4, X=6) ramp with one NaN
    v = np.arange(48, dtype=float).reshape(2, 4, 6)
    v[0, 0, 0] = np.nan
    return v


def test_downsample_volume_block_means_yx_only():
    v = _vol()
    d = R3.downsample_volume(v, 2)
    assert d.shape == (2, 2, 3)  # Z untouched, Y/X halved
    # block (z=1, rows 0-1, cols 0-1) mean
    assert d[1, 0, 0] == pytest.approx(np.nanmean(v[1, 0:2, 0:2]))
    assert np.array_equal(R3.downsample_volume(v, 1), v, equal_nan=True)


def test_threshold_mask_nans_outside_window():
    v = _vol()
    t = R3.threshold_mask(v, (10.0, 20.0))
    assert np.isnan(t[0, 0, 1])  # value 1 < 10 -> NaN
    assert t[0, 2, 3] == 15.0  # inside window kept
    assert np.array_equal(R3.threshold_mask(v, None), v, equal_nan=True)


def test_clip_mask_halves_volume_on_plane():
    v = np.ones((2, 4, 6))
    # plane through x=3 µm (spacing sx=1), normal +x: keep x >= 3 µm side
    c = R3.clip_mask(v, (1.0, 1.0, 1.0), (3.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert np.isnan(c[:, :, :3]).all() and (c[:, :, 3:] == 1.0).all()


def test_auto_clim_and_resolved_clim():
    v = _vol()
    lo, hi = R3.auto_clim(v)
    assert lo < hi
    s = R3.Scene3D(volume=v, spacing=(1, 1, 1))
    assert s.resolved_clim() == pytest.approx((lo, hi))
    s2 = R3.Scene3D(volume=v, spacing=(1, 1, 1), clim=(0.0, 5.0))
    assert s2.resolved_clim() == (0.0, 5.0)


def test_log_valid():
    assert R3.log_valid((0.1, 2.0))
    assert not R3.log_valid((0.0, 2.0))
    assert not R3.log_valid((-1.0, 2.0))
    assert not R3.log_valid(None)


def test_scene_prepared_applies_downsample_threshold_clip():
    v = _vol()
    s = R3.Scene3D(volume=v, spacing=(1.0, 2.0, 3.0), downsample=2, threshold=(10.0, 40.0))
    out, spacing = s.prepared()
    assert out.shape == (2, 2, 3)
    assert spacing == (2.0, 4.0, 3.0)  # sx, sy scaled; sz untouched
    assert np.isnan(out[0, 0, 0])  # block mean 3.75 < 10 -> thresholded


def test_orbit_positions_absolute_and_equidistant():
    base = ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    poses = R3.orbit_positions(base, 0.0, 4)
    assert len(poses) == 4
    eyes = [np.array(p[0]) for p in poses]
    # frame 0 with no elevation reproduces the base eye
    assert eyes[0] == pytest.approx(np.array(base[0]))
    # all eyes stay on the orbit sphere around the focal point
    for e in eyes:
        assert np.linalg.norm(e) == pytest.approx(10.0)
    # 90° steps about +y: eye moves into the x-z plane
    assert abs(eyes[1][0]) == pytest.approx(10.0, abs=1e-6)
    # focal + up unchanged
    for p in poses:
        assert p[1] == (0.0, 0.0, 0.0) and p[2] == (0.0, 1.0, 0.0)


def test_orbit_positions_elevation_tilts_eye():
    base = ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    poses = R3.orbit_positions(base, 20.0, 2)
    # elevation lifts the eye along +y (view-up side), distance preserved
    assert poses[0][0][1] > 0.0
    assert np.linalg.norm(np.array(poses[0][0])) == pytest.approx(10.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_render3d.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfxm.common.render3d'`.

- [ ] **Step 3: Write the implementation**

Create `dfxm/common/render3d.py`:

```python
"""Shared 3-D volume scene — one description of "what to render" for everything.

:class:`Scene3D` + :func:`populate` are the single 3-D setup used by the
visualize stage's top view and rotation video AND the GUI's pop-out viewer, so
an exported figure is guaranteed to look like the interactive view. ``pyvista``
is imported lazily inside functions (a missing GL stack only disables 3-D);
this module stays Qt-free and figure code uses the explicit Figure/Agg API.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PRESETS = ("front", "top", "side", "iso")
OPACITY_MAPPINGS = ("linear", "sigmoid", "geom", "geom_r")
RENDER_MODES = ("volume", "surface", "isosurface")


def downsample_volume(vol: np.ndarray, factor: int) -> np.ndarray:
    """Block-average (nanmean) over factor×factor Y/X blocks; Z untouched."""
    if factor <= 1:
        return vol
    z, y, x = vol.shape
    yc, xc = (y // factor) * factor, (x // factor) * factor
    v = vol[:, :yc, :xc].reshape(z, yc // factor, factor, xc // factor, factor)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN blocks -> NaN
        return np.nanmean(v, axis=(2, 4))


def threshold_mask(vol: np.ndarray, window) -> np.ndarray:
    """NaN out voxels outside the (tmin, tmax) value window; None = no-op."""
    if window is None:
        return vol
    tmin, tmax = float(window[0]), float(window[1])
    out = vol.copy()
    out[(out < tmin) | (out > tmax)] = np.nan
    return out


def clip_mask(vol: np.ndarray, spacing, origin, normal) -> np.ndarray:
    """NaN out voxels on the negative side of the plane (world µm, cell centres)."""
    sx, sy, sz = (float(s) for s in spacing)
    z, y, x = vol.shape
    xs = (np.arange(x) + 0.5) * sx
    ys = (np.arange(y) + 0.5) * sy
    zs = (np.arange(z) + 0.5) * sz
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")
    n = np.asarray(normal, dtype=float)
    o = np.asarray(origin, dtype=float)
    side = (xx - o[0]) * n[0] + (yy - o[1]) * n[1] + (zz - o[2]) * n[2]
    out = vol.copy()
    out[side < 0.0] = np.nan
    return out


def auto_clim(vol: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    valid = vol[np.isfinite(vol)]
    if valid.size == 0:
        return (0.0, 1.0)
    return (float(np.percentile(valid, lo)), float(np.percentile(valid, hi)))


def log_valid(clim) -> bool:
    """Log colour mapping is only meaningful for an all-positive colour range."""
    return clim is not None and float(clim[0]) > 0.0 and float(clim[1]) > 0.0


@dataclass
class Scene3D:
    """Everything needed to render one volume in 3-D (Qt-free, JSON-friendly)."""

    volume: np.ndarray  # (Z, Y, X) float
    spacing: tuple  # (sx, sy, sz) µm/px
    mode: str = "volume"  # RENDER_MODES
    n_isosurfaces: int = 10
    cmap: str = "magma"
    clim: tuple | None = None  # None -> auto_clim
    log_scale: bool = False
    opacity: float = 0.85  # surface/isosurface modes
    opacity_mapping: str = "linear"  # volume mode transfer function
    threshold: tuple | None = None  # (tmin, tmax) value window
    clip: tuple | None = None  # ((ox, oy, oz), (nx, ny, nz)) µm
    downsample: int = 1
    background: str = "white"

    def resolved_clim(self):
        return self.clim if self.clim is not None else auto_clim(self.volume)

    def prepared(self):
        """Volume after downsample -> threshold -> clip, plus adjusted spacing."""
        vol = downsample_volume(self.volume, int(self.downsample))
        sx, sy, sz = (float(s) for s in self.spacing)
        if int(self.downsample) > 1:
            sx, sy = sx * int(self.downsample), sy * int(self.downsample)
        vol = threshold_mask(vol, self.threshold)
        if self.clip is not None:
            vol = clip_mask(vol, (sx, sy, sz), self.clip[0], self.clip[1])
        return vol, (sx, sy, sz)


@dataclass
class CameraSpec:
    """A reproducible camera pose: preset base + azimuth/elevation/zoom."""

    preset: str = "front"  # PRESETS
    azimuth: float = 0.0
    elevation: float = 0.0
    zoom: float = 1.0


def _rotate(vec: np.ndarray, axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rodrigues rotation of *vec* about unit *axis* by *angle_deg*."""
    a = np.deg2rad(angle_deg)
    axis = axis / np.linalg.norm(axis)
    return (
        vec * np.cos(a)
        + np.cross(axis, vec) * np.sin(a)
        + axis * np.dot(axis, vec) * (1.0 - np.cos(a))
    )


def orbit_positions(base_camera, elevation_deg: float, n_frames: int):
    """Absolute (eye, focal, up) poses for a 360° orbit — pure numpy.

    Frame *i* = base eye rotated about the view-up axis through the focal
    point by ``i*360/n`` (azimuth), then lifted by *elevation_deg* about the
    horizontal axis. Absolute poses (never incremental camera mutation) are
    what make video frame generation idempotent — the fix for the
    "video doesn't rotate" bug and a requirement of the MP4→GIF replay.
    """
    eye, focal, up = (np.asarray(v, dtype=float) for v in base_camera)
    out = []
    for i in range(n_frames):
        angle = 360.0 * i / n_frames
        e = _rotate(eye - focal, up, angle) + focal
        if elevation_deg:
            horiz = np.cross(up, focal - e)
            if np.linalg.norm(horiz) > 1e-12:
                e = _rotate(e - focal, horiz, float(elevation_deg)) + focal
        out.append((tuple(e), tuple(focal), tuple(np.asarray(base_camera[2], dtype=float))))
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_render3d.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check dfxm/common/render3d.py tests/test_render3d.py
git add dfxm/common/render3d.py tests/test_render3d.py
git commit -m "feat(render3d): Scene3D/CameraSpec core with pure volume prep + orbit math"
```

---

### Task 2: Grid builders, populate(), camera application, off-screen image render

**Files:**
- Modify: `dfxm/common/render3d.py`
- Test: `tests/test_render3d.py` (append; pyvista-needing tests in their own section)

**Interfaces:**
- Consumes: Task 1's `Scene3D`, `CameraSpec`, `orbit_positions`.
- Produces:
  - `populate(plotter, scene, *, scalar_bar_title=None) -> bool` — builds actors into any pyvista plotter; `False` (and nothing added) if no finite voxels survive `prepared()`. Returns `True` otherwise. Stores nothing on the plotter beyond actors.
  - `apply_camera(plotter, cam: CameraSpec) -> None`.
  - `render_scene_image(scene, camera, *, window_size=(1920, 1080)) -> tuple[np.ndarray, float] | None` — off-screen render with **parallel projection**; returns `(rgb_image, px_per_um)`; `None` when the scene is empty. `camera` is a `CameraSpec` **or** an explicit `(eye, focal, up)` camera-position triple (the viewer passes its live pose).
  - Internal: `_grid_for_scene(scene) -> ("volume"|"mesh", grid_or_mesh) | None`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_render3d.py`)

```python
# --- pyvista-dependent (no GL needed: grid building only) -----------------


def test_grid_for_scene_surface_thresholds_nans():
    pytest.importorskip("pyvista")
    v = _vol()
    s = R3.Scene3D(volume=v, spacing=(1, 1, 1), mode="surface")
    kind, mesh = R3._grid_for_scene(s)
    assert kind == "mesh"
    assert mesh.n_cells == 47  # 48 voxels, 1 NaN thresholded out


def test_grid_for_scene_volume_keeps_grid_shape():
    pytest.importorskip("pyvista")
    v = _vol()
    s = R3.Scene3D(volume=v, spacing=(1.0, 2.0, 3.0), mode="volume")
    kind, grid = R3._grid_for_scene(s)
    assert kind == "volume"
    assert tuple(grid.dimensions) == (7, 5, 3)  # cells+1 in (X, Y, Z)
    assert grid.spacing == (1.0, 2.0, 3.0)
    # NaN voxel uploaded as 0 (transparent under the transfer function)
    assert float(grid.cell_data["values"].min()) == 0.0


def test_grid_for_scene_empty_returns_none():
    pytest.importorskip("pyvista")
    s = R3.Scene3D(volume=np.full((2, 3, 4), np.nan), spacing=(1, 1, 1))
    assert R3._grid_for_scene(s) is None


def test_grid_for_scene_log_uploads_log10_values():
    pytest.importorskip("pyvista")
    v = np.full((1, 2, 2), 100.0)
    s = R3.Scene3D(volume=v, spacing=(1, 1, 1), mode="volume", clim=(1.0, 100.0), log_scale=True)
    kind, grid = R3._grid_for_scene(s)
    assert float(grid.cell_data["values"].max()) == pytest.approx(2.0)  # log10(100)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_render3d.py -q`
Expected: new tests FAIL with `AttributeError: ... has no attribute '_grid_for_scene'`; Task 1 tests still pass.

- [ ] **Step 3: Write the implementation** (append to `render3d.py`)

```python
def _grid_for_scene(scene: Scene3D):
    """(kind, grid) for the scene — ("volume", ImageData) or ("mesh", mesh).

    Lazy pyvista import. Returns None when no finite voxel survives
    :meth:`Scene3D.prepared`. Volume mode uploads NaN as 0 so padding is
    transparent under the opacity transfer function (the old standalone
    script's approach); surface/isosurface modes threshold NaN out via the
    sentinel trick (previously ``render._pyvista_grid``).
    """
    import pyvista as pv

    vol, spacing = scene.prepared()
    dt = np.transpose(vol, (2, 1, 0))  # (Z,Y,X) -> (X,Y,Z)
    finite = dt[np.isfinite(dt)]
    if finite.size == 0:
        return None
    if scene.log_scale:
        dt = np.where(dt > 0, dt, np.nan)
        dt = np.log10(dt)
        finite = dt[np.isfinite(dt)]
        if finite.size == 0:
            return None
    grid = pv.ImageData()
    grid.dimensions = np.array(dt.shape) + 1
    grid.spacing = tuple(float(s) for s in spacing)
    grid.origin = (0.0, 0.0, 0.0)
    if scene.mode == "volume":
        grid.cell_data["values"] = np.nan_to_num(dt, nan=0.0).flatten(order="F")
        return ("volume", grid)
    sentinel = float(np.min(finite)) - 1000.0 * (float(np.ptp(finite)) + 1.0)
    dc = np.where(np.isfinite(dt), dt, sentinel)
    grid.cell_data["values"] = dc.flatten(order="F")
    thresh = sentinel * 0.5 if sentinel < 0 else sentinel + 1.0
    return ("mesh", grid.threshold(value=thresh, scalars="values"))


def _display_clim(scene: Scene3D):
    """clim in uploaded-scalar space (log10 when log_scale) + original clim."""
    vmin, vmax = scene.resolved_clim()
    if scene.log_scale:
        return (float(np.log10(vmin)), float(np.log10(vmax))), (vmin, vmax)
    return (float(vmin), float(vmax)), (vmin, vmax)


def populate(plotter, scene: Scene3D, *, scalar_bar_title=None) -> bool:
    """Build the scene's actors into *plotter* (works for QtInteractor too).

    Returns False (adding nothing) when the scene has no finite voxels.
    ``scalar_bar_title=None`` suppresses the pyvista scalar bar (exports add a
    matplotlib colorbar instead); a string shows the interactive scalar bar.
    """
    built = _grid_for_scene(scene)
    if built is None:
        return False
    kind, grid = built
    clim, _ = _display_clim(scene)
    sb = (
        {"title": scalar_bar_title + (" (log10)" if scene.log_scale else "")}
        if scalar_bar_title
        else None
    )
    common = dict(scalars="values", cmap=scene.cmap, clim=list(clim))
    if kind == "volume":
        plotter.add_volume(
            grid,
            opacity=scene.opacity_mapping,
            shade=True,
            ambient=0.3,
            diffuse=0.6,
            specular=0.2,
            show_scalar_bar=sb is not None,
            scalar_bar_args=sb,
            **common,
        )
    elif scene.mode == "isosurface":
        lo, hi = clim
        levels = np.linspace(lo, hi, scene.n_isosurfaces + 2)[1:-1]
        for i, level in enumerate(levels):
            contour = grid.contour([float(level)], scalars="values")
            if contour.n_points:
                plotter.add_mesh(
                    contour,
                    opacity=scene.opacity * (i + 1) / len(levels),
                    smooth_shading=True,
                    show_scalar_bar=sb is not None,
                    scalar_bar_args=sb,
                    **common,
                )
    else:
        plotter.add_mesh(
            grid,
            opacity=scene.opacity,
            smooth_shading=True,
            show_edges=False,
            show_scalar_bar=sb is not None,
            scalar_bar_args=sb,
            **common,
        )
    plotter.set_background(scene.background)
    return True


def _top_camera(bounds):
    """The old script's top view: eye above in +Y, Z up (bounds = pyvista tuple)."""
    cx = 0.5 * (bounds[0] + bounds[1])
    cy = 0.5 * (bounds[2] + bounds[3])
    cz = 0.5 * (bounds[4] + bounds[5])
    dist = 1.5 * max(bounds[1] - bounds[0], bounds[5] - bounds[4])
    return ((cx, cy + dist, cz), (cx, cy, cz), (0.0, 0.0, 1.0))


def apply_camera(plotter, cam: CameraSpec) -> None:
    """Apply a CameraSpec with the proven recipe (preset reset, then offsets)."""
    if cam.preset == "top":
        plotter.camera_position = _top_camera(plotter.bounds)
    elif cam.preset == "side":
        plotter.camera_position = "yz"
    elif cam.preset == "iso":
        plotter.view_isometric()
    else:  # "front"
        plotter.camera_position = "xy"
    if cam.azimuth:
        plotter.camera.azimuth = float(cam.azimuth)
    if cam.elevation:
        plotter.camera.elevation = float(cam.elevation)
    if cam.zoom and cam.zoom != 1.0:
        plotter.camera.zoom(float(cam.zoom))


def render_scene_image(scene: Scene3D, camera, *, window_size=(1920, 1080)):
    """One off-screen render -> (rgb array, px_per_um). None if scene empty.

    Uses parallel projection so px-per-µm follows exactly from the camera's
    parallel scale (the compositor's scale bar is exact, not estimated).
    *camera* is a CameraSpec or an explicit (eye, focal, up) triple.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    try:
        if not populate(pl, scene):
            return None
        if isinstance(camera, CameraSpec):
            apply_camera(pl, camera)
        else:
            pl.camera_position = camera
        pl.enable_parallel_projection()
        pl.show(auto_close=False)
        img = pl.screenshot(return_img=True)
        px_per_um = window_size[1] / (2.0 * float(pl.camera.parallel_scale))
        return np.asarray(img), px_per_um
    finally:
        pl.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_render3d.py -q`
Expected: all PASS (grid tests need pyvista importable but no GL).

- [ ] **Step 5: Lint and commit**

```bash
ruff check dfxm/common/render3d.py tests/test_render3d.py
git add dfxm/common/render3d.py tests/test_render3d.py
git commit -m "feat(render3d): grid builders, populate(), camera recipe, parallel-projection image render"
```

---

### Task 3: Styled figure compositor

**Files:**
- Modify: `dfxm/common/render3d.py`
- Test: `tests/test_render3d.py` (append)

**Interfaces:**
- Consumes: `dfxm.common.plotting` — `PlotStyle`, `add_colorbar(fig, im, ax, label, style, *, group=None)`, `draw_scale_bar(ax, length_um=None, *, style, fixed_scale_um_per_cm=None)`, `apply_text_scale`, `get_cmap`, `styled_figure`.
- Produces: `scene_figure(img, *, px_per_um, cbar_label, group=None, clim, log_scale=False, cmap="magma", title=None, style=None) -> (fig, ax, im)` — white publication figure: rendered image with µm data coordinates, matplotlib colorbar (LogNorm when `log_scale`), µm scale bar. `im` is the `AxesImage` (later tasks swap its data per video frame).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_render3d.py`)

```python
# --- compositor (Agg, no pyvista) -----------------------------------------


def _fake_render():
    return np.full((120, 200, 3), 255, dtype=np.uint8)


def test_scene_figure_extent_is_micron_true():
    fig, ax, im = R3.scene_figure(
        _fake_render(), px_per_um=2.0, cbar_label="Misorientation (°)", clim=(0.0, 1.0)
    )
    # 200 px wide at 2 px/µm -> 100 µm x-extent (exact scale-bar basis)
    assert ax.get_xlim() == (0.0, 100.0)
    assert ax.get_ylim() == (0.0, 60.0)
    assert len(fig.axes) == 2  # image + colorbar


def test_scene_figure_log_uses_lognorm():
    from matplotlib.colors import LogNorm

    fig, ax, im = R3.scene_figure(
        _fake_render(), px_per_um=2.0, cbar_label="I", clim=(1.0, 100.0), log_scale=True
    )
    # the colorbar was built from a ScalarMappable with LogNorm
    assert isinstance(fig._scene_mappable.norm, LogNorm)


def test_scene_figure_saves_png(tmp_path):
    fig, ax, im = R3.scene_figure(
        _fake_render(), px_per_um=2.0, cbar_label="ε", group="strain", clim=(-1e-3, 1e-3)
    )
    out = tmp_path / "f.png"
    fig.savefig(out, dpi=100)
    assert out.stat().st_size > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_render3d.py -q`
Expected: new tests FAIL with `AttributeError: ... 'scene_figure'`.

- [ ] **Step 3: Write the implementation** (append to `render3d.py`; add the imports to the module header)

```python
def scene_figure(
    img,
    *,
    px_per_um: float,
    cbar_label: str,
    group: str | None = None,
    clim,
    log_scale: bool = False,
    cmap: str = "magma",
    title: str | None = None,
    style=None,
):
    """Publication-styled figure around a rendered 3-D image (white background).

    The image is drawn in true µm data coordinates (from *px_per_um*, exact
    under parallel projection), so :func:`~dfxm.common.plotting.draw_scale_bar`
    needs no estimation. The colorbar comes from a ScalarMappable with the
    ORIGINAL (non-log) limits — LogNorm when *log_scale* — so log videos and
    figures label real values. Returns (fig, ax, im); *im* is the AxesImage
    whose data the rotation video swaps per frame.
    """
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    from .plotting import PlotStyle, add_colorbar, apply_text_scale, draw_scale_bar, get_cmap

    st = style if style is not None else PlotStyle(scale_bar_color="black", colorbar_fraction=0.046)
    h, w = np.asarray(img).shape[:2]
    ext_x, ext_y = w / float(px_per_um), h / float(px_per_um)
    fig = Figure(figsize=(12, 12 * h / w + 1.0), facecolor="white")
    ax = fig.add_subplot(111)
    im = ax.imshow(np.asarray(img), extent=[0, ext_x, 0, ext_y], origin="lower", aspect="equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    vmin, vmax = float(clim[0]), float(clim[1])
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log_scale else mcolors.Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap=get_cmap(cmap))
    sm.set_array([])
    fig._scene_mappable = sm  # test/debug hook: the mappable behind the colorbar
    add_colorbar(fig, sm, ax, cbar_label, st, group=group)
    draw_scale_bar(ax, st.scale_bar_length_um, style=st)
    apply_text_scale(ax, st)
    return fig, ax, im
```

Add `from matplotlib.figure import Figure` to the module-level imports (matplotlib at module level is fine — the repo rule is only about pyvista/Qt). Note: for log + `group` whose tick format is "scientific", `add_colorbar` calls `colorbar_tick_values` on `im.norm.vmin/vmax` of the mappable — works because the mappable carries the true limits. If the strain group's scientific offset path errors under LogNorm during Step 4, pass `group=None` for `log_scale=True` figures and note it in the docstring (log strain is not a meaningful combination; the visualize guard from Task 5 already blocks non-positive ranges).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_render3d.py -q`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check dfxm/common/render3d.py tests/test_render3d.py
git add dfxm/common/render3d.py tests/test_render3d.py
git commit -m "feat(render3d): publication-styled scene figure compositor (colorbar, LogNorm, exact scale bar)"
```

---

### Task 4: save_top_view + save_rotation_video rewrite; render.py slimmed; GL regression test

**Files:**
- Modify: `dfxm/common/render3d.py` (new save functions + `_video_from_frames`)
- Modify: `dfxm/common/render.py` (delete `_pyvista_grid`, `_volume_plotter`, `save_top_view`, `save_rotation_video`, `_write_image_video`; update module docstring to "2-D only"; keep `_save_animation`)
- Modify: `dfxm/stages/visualize.py` (call sites only — imports + `_process_dataset`; new params come in Task 5)
- Rewrite: `tests/test_render_rotation.py`
- Create: `tests/test_render3d_gl.py`
- Modify: `docs/Codebase.md` (render.py section: 3-D moved to render3d.py; add render3d.py section)

**Interfaces:**
- Consumes: Tasks 1–3 (`Scene3D`, `orbit_positions`, `populate`, `apply_camera`, `render_scene_image`, `scene_figure`), `render._save_animation(anim, base_path, fmt, fps, dpi)`.
- Produces:
  - `save_top_view(scene, path, *, cbar_label, group=None, style=None, window_size=(1920, 1080)) -> str | None`
  - `save_rotation_video(scene, base_path, fmt, *, cbar_label, group=None, style=None, n_frames=180, fps=15, elevation=20.0, zoom=1.2, base_camera=None, window_size=(1280, 960), progress=None) -> str | None` — `base_camera` (explicit `(eye, focal, up)`) overrides the front-preset default orbit centre; `progress` is a `(frac, msg)` callable or None.
  - `_video_from_frames(get_frame, n_frames, base_path, fmt, *, fps, cbar_label, group, clim, log_scale, cmap, px_per_um, style) -> str` — GL-free assembly seam (testable with fake frames).
  - `_orbit_frames(scene, *, elevation, zoom, base_camera, window_size) -> tuple[callable, float] | None` — `(get_frame(i)->rgb, px_per_um)`; None if empty. GL-gated tests call this directly.

- [ ] **Step 1: Write the failing tests**

Rewrite `tests/test_render_rotation.py` (same filename, new content — the old `_write_image_video` tests become `_video_from_frames` tests, container semantics preserved):

```python
"""Rotation-video assembly — GL-free frame pipeline and the empty-volume guard."""

from __future__ import annotations

import os

import numpy as np
import pytest

from dfxm.common import render3d as R3


def _gradient_frame(i):
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :, 0] = (i * 40) % 256
    frame[:, i % 48, 1] = 255
    return frame


def _kw():
    return dict(
        fps=5, cbar_label="I", group=None, clim=(0.0, 1.0),
        log_scale=False, cmap="magma", px_per_um=2.0, style=None,
    )


def test_video_from_frames_writes_gif(tmp_path):
    base = os.path.join(tmp_path, "spin")
    written = R3._video_from_frames(_gradient_frame, 4, base, "gif", **_kw())
    assert written == base + ".gif"
    assert os.path.getsize(written) > 0


def test_video_from_frames_both_prefers_mp4_or_falls_back(tmp_path):
    base = os.path.join(tmp_path, "spin")
    written = R3._video_from_frames(_gradient_frame, 3, base, "both", **_kw())
    assert written in (base + ".mp4", base + ".gif")


def test_video_from_frames_failed_mp4_is_removed(tmp_path, monkeypatch):
    from dfxm.common import render

    class BoomWriter:
        def __init__(self, *a, **kw):
            raise RuntimeError("ffmpeg died")

    monkeypatch.setattr(render, "FFMpegWriter", BoomWriter)
    base = os.path.join(tmp_path, "spin")
    with open(base + ".mp4", "wb") as fh:
        fh.write(b"partial")
    written = R3._video_from_frames(_gradient_frame, 3, base, "mp4", **_kw())
    assert written == base + ".gif"
    assert not os.path.exists(base + ".mp4")


def test_save_rotation_video_empty_volume_returns_none(tmp_path):
    pytest.importorskip("pyvista")
    scene = R3.Scene3D(volume=np.full((2, 3, 4), np.nan), spacing=(0.15, 0.38, 1.0))
    out = R3.save_rotation_video(scene, os.path.join(tmp_path, "r"), "gif", cbar_label="x")
    assert out is None


def test_save_top_view_empty_volume_returns_none(tmp_path):
    pytest.importorskip("pyvista")
    scene = R3.Scene3D(volume=np.full((2, 3, 4), np.nan), spacing=(0.15, 0.38, 1.0))
    assert R3.save_top_view(scene, os.path.join(tmp_path, "t.png"), cbar_label="x") is None
```

Create `tests/test_render3d_gl.py` (the regression the mocked tests could never catch):

```python
"""Real-GL 3-D rendering checks — skipped wholesale without an off-screen GL context."""

from __future__ import annotations

import os

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from dfxm.common import render3d as R3  # noqa: E402


def _gl_available() -> bool:
    try:
        pl = pv.Plotter(off_screen=True, window_size=[64, 48])
        pl.add_mesh(pv.Cube())
        img = pl.screenshot(return_img=True)
        pl.close()
        return img is not None and np.asarray(img).size > 0
    except Exception:  # noqa: BLE001 - any GL/driver failure -> skip the module
        return False


pytestmark = pytest.mark.skipif(not _gl_available(), reason="no usable off-screen GL context")


def _scene(mode="volume"):
    # asymmetric ramp so every orbit angle looks different
    z, y, x = np.meshgrid(np.arange(6), np.arange(20), np.arange(30), indexing="ij")
    vol = (x * 2.0 + y * 0.5 + z).astype(float)
    return R3.Scene3D(volume=vol, spacing=(0.15, 0.38, 2.0), mode=mode, clim=(0.0, 70.0))


def test_orbit_frames_actually_rotate():
    got = R3._orbit_frames(_scene(), elevation=20.0, zoom=1.2, base_camera=None,
                           window_size=(160, 120))
    assert got is not None
    get_frame, px_per_um = got
    assert px_per_um > 0
    f0, f90, f180 = (np.asarray(get_frame(i), dtype=float) for i in (0, 45, 90))
    # THE regression check: distinct azimuths must produce distinct images
    assert np.abs(f0 - f90).mean() > 1.0
    assert np.abs(f0 - f180).mean() > 1.0
    assert np.abs(f90 - f180).mean() > 1.0
    # idempotent: re-rendering frame 0 reproduces it exactly (MP4->GIF replay)
    assert np.array_equal(np.asarray(get_frame(0)), np.asarray(get_frame(0)))


def test_save_rotation_video_end_to_end(tmp_path):
    out = R3.save_rotation_video(
        _scene(), os.path.join(tmp_path, "orbit"), "gif",
        cbar_label="I", n_frames=6, fps=5, window_size=(160, 120),
    )
    assert out is not None and os.path.getsize(out) > 0


def test_save_top_view_end_to_end_all_modes(tmp_path):
    for mode in ("volume", "surface", "isosurface"):
        out = R3.save_top_view(
            _scene(mode), os.path.join(tmp_path, f"tv_{mode}.png"),
            cbar_label="I", window_size=(160, 120),
        )
        assert out is not None and os.path.getsize(out) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_render_rotation.py tests/test_render3d_gl.py -q`
Expected: FAIL with `AttributeError: ... '_video_from_frames'` / `'_orbit_frames'` / `'save_rotation_video'`.

- [ ] **Step 3: Write the implementation** (append to `render3d.py`)

```python
def _video_from_frames(
    get_frame, n_frames, base_path, fmt, *, fps, cbar_label, group, clim,
    log_scale, cmap, px_per_um, style,
):
    """Assemble RGB frames into a styled MP4/GIF (colorbar in every frame).

    Builds the :func:`scene_figure` ONCE from frame 0 and swaps only the image
    per frame. ``get_frame`` must be idempotent in ``i`` — the animation is
    replayed for ``fmt="both"`` and the MP4→GIF fallback (see
    ``render._save_animation``).
    """
    from matplotlib.animation import FuncAnimation

    from .render import _save_animation

    fig, _ax, im = scene_figure(
        np.asarray(get_frame(0)), px_per_um=px_per_um, cbar_label=cbar_label,
        group=group, clim=clim, log_scale=log_scale, cmap=cmap, style=style,
    )

    def update(frame):
        if frame:
            im.set_data(np.asarray(get_frame(frame)))
        return [im]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    return _save_animation(anim, base_path, fmt, fps=fps, dpi=100)


def _orbit_frames(scene, *, elevation, zoom, base_camera, window_size):
    """(get_frame, px_per_um) closure rendering absolute-pose orbit frames.

    One off-screen plotter, built once; each frame assigns an absolute
    camera_position from :func:`orbit_positions` — never incremental vtk
    Azimuth() calls (the bug this file replaces). None if the scene is empty.
    The caller must invoke get_frame at least once and is responsible for the
    plotter's lifetime via the returned closure's ``.close()`` attribute.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    if not populate(pl, scene):
        pl.close()
        return None
    if base_camera is None:
        apply_camera(pl, CameraSpec(preset="front", zoom=zoom))
        base = pl.camera_position
        poses = orbit_positions(tuple(tuple(v) for v in base), elevation, 3600)
    else:
        poses = orbit_positions(tuple(tuple(v) for v in base_camera), 0.0, 3600)
    pl.enable_parallel_projection()
    pl.show(auto_close=False)
    px_per_um = window_size[1] / (2.0 * float(pl.camera.parallel_scale))

    def get_frame(i, _n=[None]):
        # poses is a dense 3600-step table; frame i of n maps to step i*3600//n
        pl.camera_position = poses[(i * 3600 // get_frame.n_frames) % 3600]
        return pl.screenshot(return_img=True)

    get_frame.n_frames = 360  # caller overwrites with the real count
    get_frame.close = pl.close
    return get_frame, px_per_um


def save_top_view(scene, path, *, cbar_label, group=None, style=None, window_size=(1920, 1080)):
    """Styled top-view figure (colorbar + scale bar); returns path or None if empty."""
    got = render_scene_image(scene, CameraSpec(preset="top"), window_size=window_size)
    if got is None:
        return None
    img, px_per_um = got
    fig, _ax, _im = scene_figure(
        img, px_per_um=px_per_um, cbar_label=cbar_label, group=group,
        clim=scene.resolved_clim(), log_scale=scene.log_scale, cmap=scene.cmap, style=style,
    )
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    return path


def save_rotation_video(
    scene, base_path, fmt, *, cbar_label, group=None, style=None, n_frames=180,
    fps=15, elevation=20.0, zoom=1.2, base_camera=None, window_size=(1280, 960),
    progress=None,
):
    """360° orbit movie, publication-styled; returns path or None if empty.

    ``base_camera`` (an explicit (eye, focal, up) triple, e.g. the GUI
    viewer's live pose) orbits around that pose instead of the front preset.
    """
    got = _orbit_frames(
        scene, elevation=elevation, zoom=zoom, base_camera=base_camera, window_size=window_size
    )
    if got is None:
        return None
    get_frame, px_per_um = got
    get_frame.n_frames = int(n_frames)
    if progress is not None:
        inner = get_frame

        def get_frame(i, _inner=inner):  # noqa: F811 - deliberate wrap
            progress(min(0.99, i / max(1, n_frames)), f"rendering orbit frame {i}/{n_frames}")
            return _inner(i)

        get_frame.n_frames = int(n_frames)
        get_frame.close = inner.close
    try:
        return _video_from_frames(
            get_frame, int(n_frames), base_path, fmt, fps=fps, cbar_label=cbar_label,
            group=group, clim=scene.resolved_clim(), log_scale=scene.log_scale,
            cmap=scene.cmap, px_per_um=px_per_um, style=style,
        )
    finally:
        get_frame.close()
```

Implementation note for the executor: the dense-table indirection in `_orbit_frames` exists so a single closure serves any `n_frames` without regenerating poses; if you find it awkward, generating `orbit_positions(base, elev, n_frames)` lazily inside `get_frame` per call is equally correct — keep poses **absolute** either way. Simplify to direct per-frame `orbit_positions(...)[i]` computation if the table feels clever-but-fragile; the tests only require absolute-pose idempotency.

Then in `dfxm/common/render.py` (Read it first): delete `_pyvista_grid`, `_volume_plotter`, `save_top_view`, `save_rotation_video`, `_write_image_video`; trim the module docstring's 3-D claims to say 3-D lives in `render3d`. `_save_animation` stays (used by `save_layer_animation` and `render3d`).

Then in `dfxm/stages/visualize.py` (call sites only): in `_process_dataset`, replace the two `Rnd.save_top_view(...)`/`Rnd.save_rotation_video(...)` calls with:

```python
    from ..common import render3d as R3

    scene = R3.Scene3D(
        volume=data,
        spacing=(sx, sy, scale_z),
        cmap=cmap,
        clim=(float(vmin), float(vmax)),
        opacity=float(p["volume_opacity"]),
        mode="volume",
    )
    if p["save_topview"]:
        try:
            prod.top_view = R3.save_top_view(
                scene, os.path.join(ds_dir, f"{name}_top_view.png"),
                cbar_label=cbar, group=group, style=style,
            )
        except Exception as exc:  # noqa: BLE001 - no GL / pyvista issue -> note + continue
            prod.notes.append(f"3D top-view skipped: {exc}")
    if p["save_rotation"]:
        try:
            prod.rotation_video = R3.save_rotation_video(
                scene, os.path.join(ds_dir, f"{name}_rotation"), p["output_format"],
                cbar_label=cbar, group=group, style=style,
            )
            if prod.rotation_video is None:
                prod.notes.append("rotation video skipped: volume has no finite voxels")
        except Exception as exc:  # noqa: BLE001 - no GL / pyvista issue -> note + continue
            prod.notes.append(f"rotation video skipped: {exc}")
```

Put the `render3d` import at module top (`from ..common import render3d as R3` — module-level import is fine, pyvista inside it is lazy). `_process_dataset` needs `group` — it already receives `group=None` keyword; pass the same `group` through (check the existing signature when editing; `style`/`group` are already parameters). Update `tests/test_stage_visualize.py` monkeypatches from `V.Rnd.save_rotation_video`/`save_top_view` to `V.R3.save_rotation_video`/`V.R3.save_top_view` — Read that file first; the fakes must accept the new keyword signatures (`lambda *a, **kw: ...` fakes already do).

`docs/Codebase.md`: in the `dfxm/common/render.py` section, remove the 3-D function entries and add a `dfxm/common/render3d.py` section listing `Scene3D`, `CameraSpec`, `populate`, `apply_camera`, `render_scene_image`, `scene_figure`, `save_top_view`, `save_rotation_video`, `orbit_positions` with one-line descriptions (Read the doc's render.py section first and match its style).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_render_rotation.py tests/test_render3d.py tests/test_render3d_gl.py tests/test_stage_visualize.py -q`
Expected: all PASS. **On this machine the GL tests must actually run (not skip)** — if they skip, stop and investigate the GL context before proceeding (this is the bug's regression net).

- [ ] **Step 5: Full suite, lint, commit**

```bash
python3 -m pytest -q && ruff check .
git add -A
git commit -m "feat(render3d): volumetric top view + rotation video on absolute-pose orbit math; render.py is 2-D only"
```

---

### Task 5: Visualize stage — new 3-D params, log-scale guard, docs

**Files:**
- Modify: `dfxm/stages/visualize.py` (STAGE params + `_process_dataset` threading)
- Modify: `docs/Usage.md`, `docs/Codebase.md`
- Test: `tests/test_stage_visualize.py` (append)

**Interfaces:**
- Consumes: Task 4's Scene3D call sites; `render3d.log_valid`.
- Produces: stage params `render_mode` (enum volume/surface/isosurface, default volume), `opacity_mapping` (enum linear/sigmoid/geom/geom_r, default linear), `rotation_frames` (int, default 180), `log_scale` (bool, default False). Note text (exact string later tasks/tests match): `"log scale skipped: colour range includes non-positive values"`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_stage_visualize.py` — Read the file first; follow its existing fixture/params pattern for a minimal mosa run with monkeypatched `V.R3`)

```python
def test_scene_carries_new_3d_params(tmp_path, monkeypatch, mosa_file_fixture_used_by_existing_tests):
    captured = {}

    def fake_top(scene, path, **kw):
        captured["scene"] = scene
        return path

    monkeypatch.setattr(V.R3, "save_top_view", fake_top)
    # build params exactly like the existing save_topview test, plus:
    params.update(render_mode="isosurface", opacity_mapping="sigmoid",
                  rotation_frames=24, save_topview=True, save_rotation=False)
    V.run(params)
    assert captured["scene"].mode == "isosurface"
    assert captured["scene"].opacity_mapping == "sigmoid"


def test_log_scale_guard_falls_back_with_note(tmp_path, monkeypatch, ...):
    monkeypatch.setattr(V.R3, "save_top_view", lambda scene, path, **kw: path)
    params.update(log_scale=True, save_topview=True)  # CoM data centres around 0 -> invalid
    result = V.run(params)
    prod = result.datasets[0]
    assert any("log scale skipped" in n for n in prod.notes)


def test_rotation_frames_passed_through(tmp_path, monkeypatch, ...):
    seen = {}
    monkeypatch.setattr(
        V.R3, "save_rotation_video",
        lambda scene, base, fmt, **kw: seen.update(kw) or (base + ".gif"),
    )
    params.update(save_rotation=True, rotation_frames=24)
    V.run(params)
    assert seen["n_frames"] == 24
```

(The `...` fixture placeholders above mean: reuse whatever minimal-run fixture the existing `test_stage_visualize.py` rotation tests use — copy their params dict verbatim. The executor must adapt names after Reading the file; the assertions are the contract.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_visualize.py -q`
Expected: new tests FAIL (`KeyError: 'render_mode'` or scene lacking the values).

- [ ] **Step 3: Implement**

In `STAGE.params`, after the existing `volume_opacity` param, add (exact blocks — check `ParamType.INT` exists first, see Global Constraints):

```python
        Param(
            "render_mode",
            ParamType.ENUM,
            "3D render mode",
            default="volume",
            choices=("volume", "surface", "isosurface"),
            advanced=True,
            group="Appearance",
            help=(
                "How the 3-D top view and rotation video draw the volume: 'volume' is "
                "true volumetric rendering (shaded, transfer-function opacity), "
                "'surface' the legacy NaN-thresholded mesh, 'isosurface' stacked "
                "contour shells."
            ),
        ),
        Param(
            "opacity_mapping",
            ParamType.ENUM,
            "3D opacity mapping",
            default="linear",
            choices=("linear", "sigmoid", "geom", "geom_r"),
            advanced=True,
            group="Appearance",
            help=(
                "Opacity transfer function for volumetric 3-D rendering: linear, "
                "sigmoid (emphasises mid-range values), geom (high values), geom_r "
                "(low values). Ignored by the surface and isosurface modes."
            ),
        ),
        Param(
            "rotation_frames",
            ParamType.INT,
            "Rotation frames",
            default=180,
            advanced=True,
            group="Output",
            help="Frames in one 360-degree orbit of the rotation video (15 fps).",
        ),
        Param(
            "log_scale",
            ParamType.BOOL,
            "Log colour scale (3D)",
            default=False,
            advanced=True,
            group="Appearance",
            help=(
                "Logarithmic colour mapping for the 3-D top view and rotation video. "
                "Falls back to linear (with a note) when the colour range includes "
                "zero or negative values."
            ),
        ),
```

In `_process_dataset`, build the scene with the new params and the guard:

```python
    log_scale = bool(p["log_scale"])
    if log_scale and not R3.log_valid((vmin, vmax)):
        log_scale = False
        prod.notes.append("log scale skipped: colour range includes non-positive values")
    scene = R3.Scene3D(
        volume=data,
        spacing=(sx, sy, scale_z),
        mode=str(p["render_mode"]),
        cmap=cmap,
        clim=(float(vmin), float(vmax)),
        log_scale=log_scale,
        opacity=float(p["volume_opacity"]),
        opacity_mapping=str(p["opacity_mapping"]),
    )
```

and pass `n_frames=int(p["rotation_frames"])` to `save_rotation_video`. Update the `save_rotation` param's help text (Read the exact bytes first — em-dash hazard): replace the stale "(120 frames, ~8 s)" wording with "(one 360° orbit; frame count from 'Rotation frames')". Also update `volume_opacity` help to mention it applies to surface/isosurface modes.

Docs (same commit):
- `docs/Usage.md` visualize section: document the four new fields (what each does, when log falls back, that volume mode is the new default look).
- `docs/Codebase.md` visualize section: note `_process_dataset` builds a `render3d.Scene3D` and the new params.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_visualize.py tests/test_param_metadata.py -q`
Expected: PASS (param-metadata enforcement tests pick up the new params — if they fail on missing help/group, fix the param blocks, not the tests).

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && git add -A
git commit -m "feat(visualize): render_mode/opacity_mapping/rotation_frames/log_scale params + log guard (docs synced)"
```

---

### Task 6: Volume source metadata + JSON-able loader specs

**Files:**
- Modify: `dfxm/stages/visualize.py` (`aligned_field` returns 5-tuple)
- Modify: `gui/viewers.py` (`LoadedVolume`, `VolumeSourceSpec`, `volume_sources` returns specs)
- Modify: `gui/widgets/volume3d.py` (unpack the new shape — minimal edit; full launcher rewrite is Task 8)
- Modify: `docs/Codebase.md`
- Test: `tests/test_gui_viewers.py` (append; Read first)

**Interfaces:**
- Consumes: `visualize.aligned_field`, `visualize._display_info`.
- Produces:
  - `visualize.aligned_field(params, name) -> (volume, spacing, cmap, clim, meta)` where `meta = {"cbar_label": str, "group": str | None}` (strain: `("Strain (ε)", "strain")`; CoM/FWHM from `_display_info`).
  - `gui.viewers.LoadedVolume(volume, spacing, cmap, clim, cbar_label, group)` dataclass.
  - `gui.viewers.VolumeSourceSpec(name, load, loader)` dataclass — `load: Callable[[], LoadedVolume]`; `loader` a JSON-able dict: visualize → `{"kind": "visualize_field", "stage_params": {…}, "field": name}`; rocking → `{"kind": "h5_dataset", "path": aligned_path, "dataset": ds}`.
  - `volume_sources(stage_name, result, params) -> dict[str, VolumeSourceSpec]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_gui_viewers.py` after Reading it — reuse its existing synthetic-h5 helpers if present)

```python
def test_rocking_source_spec_carries_meta_and_loader(tmp_path):
    path = _make_aligned_h5(tmp_path)  # reuse/adapt the file builder already in this test module
    sources = viewers.volume_sources("rocking", _result_with(aligned_path=path), {})
    spec = sources["sum_intensity"]
    assert spec.loader == {"kind": "h5_dataset", "path": path, "dataset": "sum_intensity"}
    loaded = spec.load()
    assert loaded.group == "raw"
    assert loaded.cbar_label == "Intensity"
    assert loaded.volume.ndim == 3


def test_visualize_source_spec_loader_is_jsonable(monkeypatch):
    import json
    monkeypatch.setattr(
        "dfxm.stages.visualize.available_fields", lambda p: ["chi_Center_of_mass"]
    )
    params = {"mosa_volume_file": "/x/maps.h5"}
    sources = viewers.volume_sources("visualize", object(), params)
    spec = sources["chi_Center_of_mass"]
    assert spec.loader["kind"] == "visualize_field"
    assert spec.loader["field"] == "chi_Center_of_mass"
    json.dumps(spec.loader)  # must not raise


def test_aligned_field_returns_meta(monkeypatch, visualize_minimal_fixture):
    vol, spacing, cmap, clim, meta = V.aligned_field(params, "chi_Center_of_mass")
    assert meta == {"cbar_label": "Misorientation (°)", "group": "mosa_com"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_viewers.py -q`
Expected: FAIL (`AttributeError: VolumeSourceSpec` / tuple unpack count).

- [ ] **Step 3: Implement**

`visualize.aligned_field`: change the two returns to append `meta`:
strain branch → `meta = {"cbar_label": "Strain (ε)", "group": "strain"}`; mosa branch → `_t, label, group = _display_info(name)` then `meta = {"cbar_label": label, "group": group}`; final line `return data, (scale_x, scale_y, scale_z), cmap, (float(vmin), float(vmax)), meta`.

`gui/viewers.py`:

```python
@dataclass
class LoadedVolume:
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
    volume without pickling arrays: {"kind": "visualize_field", "stage_params",
    "field"} or {"kind": "h5_dataset", "path", "dataset"}.
    """

    name: str
    load: Callable[[], LoadedVolume]
    loader: dict
```

Rework `_rocking_source(aligned_path, dataset)` to return a `LoadedVolume` (`cbar_label="Intensity"`, `group="raw"`) and `volume_sources` to build `VolumeSourceSpec`s:
rocking → `loader={"kind": "h5_dataset", "path": path, "dataset": ds}`;
visualize → `load=lambda n=name: _visualize_load(params, n)` where `_visualize_load` calls `visualize.aligned_field` and wraps the 5-tuple into `LoadedVolume`, and `loader={"kind": "visualize_field", "stage_params": dict(params), "field": name}`.

`gui/widgets/volume3d.py` `_on_render` minimal fix (full rewrite comes in Task 8):

```python
            spec = self._sources[name]
            lv = spec.load()
            ok = self._canvas.show_volume(lv.volume, lv.spacing, cmap=lv.cmap, clim=lv.clim)
```

Update the module docstring in `viewers.py` (the "(volume, spacing, cmap, clim)" comment) and `docs/Codebase.md` entries for `viewers.py` + `aligned_field`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_viewers.py tests/test_stage_visualize.py -q`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && git add -A
git commit -m "feat(viewers): VolumeSourceSpec with colorbar metadata + JSON-able reload loaders (docs synced)"
```

---

### Task 7: viewer_jobs — child-process rotation-video job

**Files:**
- Create: `dfxm/viewer_jobs.py`
- Test: `tests/test_viewer_jobs.py`
- Modify: `docs/Codebase.md`

**Interfaces:**
- Consumes: `render3d.Scene3D`, `render3d.save_rotation_video`, `visualize.aligned_field`, `plotting.style_from_json`, h5py.
- Produces: `rotation_video_job(params: dict, progress=None) -> dict` — runnable via `StageRunner("dfxm.viewer_jobs:rotation_video_job", params)`. Params (all JSON-able):

```python
{
  "loader": {"kind": "visualize_field", "stage_params": {...}, "field": "..."}
          | {"kind": "h5_dataset", "path": "...", "dataset": "..."},
  "scene": {"mode": "volume", "cmap": "magma", "clim": [0, 1] | None,
             "log_scale": False, "opacity": 0.85, "opacity_mapping": "linear",
             "threshold": [a, b] | None, "clip": [[ox,oy,oz],[nx,ny,nz]] | None,
             "downsample": 1, "background": "white"},
  "base_camera": [[ex,ey,ez],[fx,fy,fz],[ux,uy,uz]] | None,
  "elevation": 20.0, "zoom": 1.2, "n_frames": 180, "fps": 15,
  "base_path": "...", "fmt": "mp4",
  "cbar_label": "...", "group": "mosa_com" | None, "style_json": "" ,
}
```

Returns `{"video": path | None}`. Also `_load_volume(loader) -> (volume, spacing)` (h5_dataset kind reads the dataset + `scale_[xyz]_um_per_px` attrs like `gui.viewers._rocking_source` does).

- [ ] **Step 1: Write the failing tests**

```python
"""Child-process viewer jobs — loader resolution and scene threading (no GL)."""

from __future__ import annotations

import h5py
import numpy as np

from dfxm import viewer_jobs as VJ


def _h5(tmp_path):
    path = str(tmp_path / "aligned.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("sum_intensity", data=np.ones((2, 3, 4)))
        f.attrs["scale_x_um_per_px"] = 0.15
        f.attrs["scale_y_um_per_px"] = 0.38
        f.attrs["scale_z_um_per_px"] = 2.0
    return path


def test_load_volume_h5_dataset(tmp_path):
    vol, spacing = VJ._load_volume({"kind": "h5_dataset", "path": _h5(tmp_path), "dataset": "sum_intensity"})
    assert vol.shape == (2, 3, 4)
    assert spacing == (0.15, 0.38, 2.0)


def test_rotation_video_job_threads_scene_and_camera(tmp_path, monkeypatch):
    seen = {}

    def fake_save(scene, base_path, fmt, **kw):
        seen.update(scene=scene, base_path=base_path, fmt=fmt, **kw)
        return base_path + ".mp4"

    monkeypatch.setattr(VJ.R3, "save_rotation_video", fake_save)
    out = VJ.rotation_video_job(
        {
            "loader": {"kind": "h5_dataset", "path": _h5(tmp_path), "dataset": "sum_intensity"},
            "scene": {"mode": "surface", "cmap": "viridis", "clim": [0.0, 2.0],
                       "log_scale": False, "opacity": 0.5, "opacity_mapping": "linear",
                       "threshold": None, "clip": None, "downsample": 2, "background": "white"},
            "base_camera": [[0, 0, 10], [0, 0, 0], [0, 1, 0]],
            "elevation": 0.0, "zoom": 1.0, "n_frames": 12, "fps": 5,
            "base_path": str(tmp_path / "orbit"), "fmt": "gif",
            "cbar_label": "Intensity", "group": "raw", "style_json": "",
        }
    )
    assert out == {"video": str(tmp_path / "orbit") + ".mp4"}
    assert seen["scene"].mode == "surface" and seen["scene"].downsample == 2
    assert seen["base_camera"] == ((0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0))
    assert seen["n_frames"] == 12 and seen["group"] == "raw"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_viewer_jobs.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** `dfxm/viewer_jobs.py`

```python
"""Qt-free jobs the GUI runs in a child process (via dfxm.runner.StageRunner).

Sits ABOVE dfxm.common and dfxm.stages in the layering (it may import both);
gui/ builds the JSON-able params, the runner resolves
"dfxm.viewer_jobs:rotation_video_job" in the child.
"""

from __future__ import annotations

import h5py
import numpy as np

from .common import render3d as R3
from .common.plotting import style_from_json


def _load_volume(loader: dict):
    """(volume, spacing) from a JSON-able loader spec."""
    if loader["kind"] == "h5_dataset":
        with h5py.File(loader["path"], "r") as f:
            vol = f[loader["dataset"]][:].astype(float)
            spacing = tuple(
                float(f.attrs.get(f"scale_{ax}_um_per_px", 1.0)) for ax in ("x", "y", "z")
            )
        return vol, spacing
    if loader["kind"] == "visualize_field":
        from .stages.visualize import aligned_field

        vol, spacing, _cmap, _clim, _meta = aligned_field(loader["stage_params"], loader["field"])
        return vol, spacing
    raise ValueError(f"unknown loader kind {loader.get('kind')!r}")


def rotation_video_job(params: dict, progress=None) -> dict:
    """Render a rotation video from a viewer window's settings; returns {"video": path}."""
    vol, spacing = _load_volume(params["loader"])
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
    return {"video": video}
```

(`np` import only if used — drop it if ruff flags it.) Add a `docs/Codebase.md` entry for the module.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_viewer_jobs.py -q` → PASS.

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && git add -A
git commit -m "feat(viewer-jobs): child-process rotation_video_job with JSON-able loader/scene specs (docs synced)"
```

---

### Task 8: Viewer3DWindow skeleton + launcher rewrite

**Files:**
- Create: `gui/widgets/viewer3d_window.py`
- Modify: `gui/widgets/volume3d.py` (launcher rewrite), `gui/widgets/pv_canvas.py` (delete `show_volume`)
- Modify: `docs/Usage.md`, `docs/Codebase.md`
- Test: `tests/test_gui_viewer3d.py` (create)

**Interfaces:**
- Consumes: `VolumeSourceSpec`/`LoadedVolume` (Task 6), `render3d.Scene3D`/`populate`/`apply_camera` (Tasks 1–2), `PvCanvas.ensure()/available/plotter`.
- Produces:
  - `Viewer3DWindow(spec: VolumeSourceSpec, stage_name: str, style_json: str = "")` — a top-level `QWidget` (`Qt.Window` flag, `WA_DeleteOnClose`); attributes used by tests/Task 9-10: `.scene: Scene3D | None`, `.loaded: LoadedVolume | None`, `._canvas: PvCanvas`, `.rebuild()` (clears + `populate`s + renders; no-op safely without GL), `.load_and_render()` (calls `spec.load()`, builds the initial Scene3D with `mode="volume"`, auto clim from `LoadedVolume.clim`, then `rebuild()` + front camera).
  - `Volume3DPanel` keeps its class name and `set_sources(dict[str, VolumeSourceSpec])` API (stage_view wiring untouched) but now shows dropdown + "Open 3D viewer…" button; `_windows: list[Viewer3DWindow]` keeps refs; closed windows are pruned via `destroyed` signal.

- [ ] **Step 1: Write the failing tests** (`tests/test_gui_viewer3d.py`, same offscreen header as `test_gui_clim_section.py`)

```python
"""Viewer3DWindow + launcher — construction, scene wiring, close/free (offscreen)."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.viewers import LoadedVolume, VolumeSourceSpec  # noqa: E402
from gui.widgets.viewer3d_window import Viewer3DWindow  # noqa: E402
from gui.widgets.volume3d import Volume3DPanel  # noqa: E402

_app = QApplication.instance() or QApplication([])


def _spec(name="vol"):
    lv = LoadedVolume(
        volume=np.ones((2, 3, 4)), spacing=(0.15, 0.38, 2.0), cmap="magma",
        clim=(0.0, 1.0), cbar_label="Intensity", group="raw",
    )
    return VolumeSourceSpec(name=name, load=lambda: lv,
                            loader={"kind": "h5_dataset", "path": "/x", "dataset": name})


def test_window_builds_scene_from_source():
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    assert w.loaded is not None
    assert w.scene.mode == "volume"
    assert w.scene.clim == (0.0, 1.0)
    assert w.windowTitle() == "visualize — vol (3D)"
    w.close()


def test_window_survives_without_gl():
    # offscreen: PvCanvas.ensure() may fail -> placeholder; nothing raises
    w = Viewer3DWindow(_spec(), "rocking")
    w.load_and_render()
    w.rebuild()  # second rebuild also safe
    w.close()


def test_launcher_opens_windows_and_prunes_closed():
    panel = Volume3DPanel()
    panel.set_sources({"a": _spec("a"), "b": _spec("b")})
    panel._combo.setCurrentText("a")
    panel._open_btn.click()
    assert len(panel._windows) == 1
    panel._windows[0].close()
    _app.processEvents()
    assert len(panel._windows) == 0


def test_launcher_disabled_without_sources():
    panel = Volume3DPanel()
    panel.set_sources({})
    assert not panel._open_btn.isEnabled()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_viewer3d.py -q`
Expected: FAIL — `viewer3d_window` module missing.

- [ ] **Step 3: Implement**

`gui/widgets/viewer3d_window.py` (skeleton this task; controls Task 9, exports Task 10):

```python
"""Pop-out 3-D viewer window — one volume, full controls, freed on close.

Each window owns its own lazy PvCanvas (GL context) and one
:class:`~dfxm.common.render3d.Scene3D`; closing the window closes the plotter
and drops the volume reference, returning the memory. All rendering setup goes
through ``render3d.populate`` so the view matches the stage's exports exactly.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QVBoxLayout, QWidget

from dfxm.common import render3d as R3

from ..viewers import LoadedVolume, VolumeSourceSpec
from .pv_canvas import PvCanvas


class Viewer3DWindow(QWidget):
    """Interactive 3-D view of ONE volume with ParaView-style controls."""

    def __init__(self, spec: VolumeSourceSpec, stage_name: str, style_json: str = "") -> None:
        super().__init__(None, Qt.WindowType.Window)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setWindowTitle(f"{stage_name} — {spec.name} (3D)")
        self.resize(1100, 750)
        self._spec = spec
        self._stage_name = stage_name
        self._style_json = style_json
        self.loaded: LoadedVolume | None = None
        self.scene: R3.Scene3D | None = None
        self._canvas = PvCanvas()
        self._status = QLabel("")
        self._status.setWordWrap(True)

        self._controls = QWidget()  # Task 9 fills this
        controls_scroll = QScrollArea()
        controls_scroll.setWidget(self._controls)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFixedWidth(280)

        centre = QVBoxLayout()
        centre.addWidget(self._canvas, 1)
        centre.addWidget(self._status)
        lay = QHBoxLayout(self)
        row = QHBoxLayout()
        row.addLayout(centre, 1)
        row.addWidget(controls_scroll)
        lay.addLayout(row)

    # -- lifecycle --------------------------------------------------------
    def load_and_render(self) -> None:
        """Load the volume (heavy) and do the first render."""
        self.loaded = self._spec.load()
        self.scene = R3.Scene3D(
            volume=self.loaded.volume,
            spacing=self.loaded.spacing,
            cmap=self.loaded.cmap,
            clim=self.loaded.clim,
        )
        self.rebuild()
        if self._canvas.available:
            R3.apply_camera(self._canvas.plotter, R3.CameraSpec(preset="front"))

    def rebuild(self) -> None:
        """Clear and re-populate the plotter from the current scene."""
        if self.scene is None or not self._canvas.ensure():
            if not self._canvas.available:
                self._status.setText("3-D unavailable (no OpenGL context) — controls disabled")
            return
        pl = self._canvas.plotter
        pl.clear()
        ok = R3.populate(pl, self.scene, scalar_bar_title=self.loaded.cbar_label)
        self._status.setText(
            f"{self._spec.name}: shape {tuple(self.scene.volume.shape)}"
            if ok
            else "nothing to show (no finite voxels after threshold/clip)"
        )
        pl.render()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._canvas.plotter is not None:
            self._canvas.plotter.close()
        self.loaded = None
        self.scene = None
        super().closeEvent(event)
```

`gui/widgets/volume3d.py` rewrite (keep class name + `set_sources` API; docstring updated):

```python
class Volume3DPanel(QWidget):
    """Launcher: pick a volume, open it in an independent 3-D viewer window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._sources: dict[str, VolumeSourceSpec] = {}
        self._windows: list[Viewer3DWindow] = []
        self._stage_name = ""
        self._combo = QComboBox()
        self._open_btn = QPushButton("Open 3D viewer…")
        self._open_btn.clicked.connect(self._on_open)
        self._status = QLabel("(run the stage, then open a volume in the 3-D viewer)")
        ...  # same layout pattern as before

    def set_sources(self, sources: dict[str, VolumeSourceSpec], stage_name: str = "") -> None:
        ...

    def _on_open(self) -> None:
        spec = self._sources.get(self._combo.currentText())
        if spec is None:
            return
        w = Viewer3DWindow(spec, self._stage_name or "stage")
        self._windows.append(w)
        w.destroyed.connect(lambda *_a, _w=w: self._windows.remove(_w) if _w in self._windows else None)
        w.show()
        w.load_and_render()
```

`stage_view.py:823` passes no stage name today — extend the call to `set_sources(volume_sources(...), self._stage_name)` (Read the site first; keep the lazy comment). Delete `PvCanvas.show_volume` (nothing calls it after this task) and its numpy import if now unused; keep `ensure/available/plotter/apply_theme/clear`.

Docs (same commit): `Usage.md` — replace the 3-D tab description with the launcher + window workflow; `Codebase.md` — `viewer3d_window.py` entry, `volume3d.py` + `pv_canvas.py` updates.

**Note:** remove the "3-D unavailable" placeholder copy duplication — the window shows status text; PvCanvas keeps its own label for the embedded fallback.

- [ ] **Step 4: Run tests + smoke to verify**

Run: `python3 -m pytest tests/test_gui_viewer3d.py tests/test_gui_viewers.py -q && python3 tests/gui_smoke.py`
Expected: pytest PASS; smoke still all-green (its 3-D-tab checks may need updating if they reference "Render 3-D" — Read `gui_smoke.py`'s relevant check and update it to the launcher in the same commit).

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && git add -A
git commit -m "feat(gui): pop-out Viewer3DWindow + launcher 3-D tab; memory freed on close (docs synced)"
```

---

### Task 9: Viewer controls — appearance set (mode, cmap, clim, log, opacity, background)

**Files:**
- Modify: `gui/widgets/viewer3d_window.py`
- Modify: `docs/Usage.md`
- Test: `tests/test_gui_viewer3d.py` (append)

**Interfaces:**
- Consumes: Task 8's window; `render3d.RENDER_MODES`, `OPACITY_MAPPINGS`, `log_valid`, `auto_clim`; `dfxm.common.plotting.CMAP_GROUPS` is NOT needed — the cmap combo lists matplotlib names + the pipeline's registered ParaView-Fast cmap (Read `dfxm/common/cmaps.py:register` for its registered name and use `["magma", "viridis", "plasma", "inferno", "RdBu_r", "gray", <fast-name>]`).
- Produces: controls that mutate `self.scene` and call `self.rebuild()`; `_sync_log_enabled()` disabling the log checkbox (with tooltip `"log needs an all-positive colour range"`) when `not log_valid(scene.clim or auto)`; `_auto_clim()` reset button handler. Widget attribute names (tests + Task 10 rely on them): `_mode_combo`, `_cmap_combo`, `_clim_min`, `_clim_max` (QDoubleSpinBox, generous ranges, 6 decimals), `_clim_auto_btn`, `_log_check`, `_opacity_slider` (0–100), `_mapping_combo`, `_bg_combo` ("theme", "white", "black").

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_controls_mutate_scene_and_trigger_rebuild(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    calls = []
    monkeypatch.setattr(w, "rebuild", lambda: calls.append(1))
    w._mode_combo.setCurrentText("isosurface")
    assert w.scene.mode == "isosurface"
    w._clim_min.setValue(0.2); w._clim_max.setValue(0.8)
    assert w.scene.clim == (0.2, 0.8)
    w._opacity_slider.setValue(40)
    assert w.scene.opacity == pytest.approx(0.4)
    w._mapping_combo.setCurrentText("sigmoid")
    assert w.scene.opacity_mapping == "sigmoid"
    assert len(calls) >= 4
    w.close()


def test_log_checkbox_guard():
    w = Viewer3DWindow(_spec(), "visualize")  # clim (0.0, 1.0) -> vmin not > 0
    w.load_and_render()
    assert not w._log_check.isEnabled()
    w._clim_min.setValue(0.1)
    assert w._log_check.isEnabled()
    w._log_check.setChecked(True)
    assert w.scene.log_scale is True
    w.close()


def test_auto_clim_button_resets_from_volume():
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    w._clim_min.setValue(0.4)
    w._clim_auto_btn.click()
    lo, hi = w.scene.clim
    assert lo == pytest.approx(1.0) and hi == pytest.approx(1.0)  # all-ones volume
    w.close()
```

- [ ] **Step 2: Run to verify failure** — `python3 -m pytest tests/test_gui_viewer3d.py -q` → FAIL (`AttributeError: _mode_combo`).

- [ ] **Step 3: Implement** — build the controls into `self._controls` (a `QFormLayout`), each handler pattern:

```python
    def _on_mode(self, text: str) -> None:
        self.scene.mode = text
        self.rebuild()
```

`_clim_min`/`_clim_max` handlers write `self.scene.clim = (self._clim_min.value(), self._clim_max.value())`, call `self._sync_log_enabled()`, then `self.rebuild()`. `_clim_auto_btn` → `self.scene.clim = R3.auto_clim(self.scene.volume)`, write both spinboxes (blocking their signals with `QSignalBlocker`), sync log, rebuild. `_sync_log_enabled()`:

```python
    def _sync_log_enabled(self) -> None:
        ok = R3.log_valid(self.scene.resolved_clim())
        self._log_check.setEnabled(ok)
        self._log_check.setToolTip("" if ok else "log needs an all-positive colour range")
        if not ok and self._log_check.isChecked():
            self._log_check.setChecked(False)  # emits -> scene.log_scale=False + rebuild
```

Background: "theme" → `ThemeController.instance().palette.pv_background` (see `pv_canvas.apply_theme`), else the literal colour; sets `scene.background` and rebuilds. Initialise all widget values from the scene inside `load_and_render()` with signals blocked. Wrap every combo/spinbox in the existing `wheel_guard` helper if the repo applies it to forms (Read `gui/widgets/wheel_guard.py` usage in `param_form.py` and follow the same pattern). `Usage.md`: document each control briefly.

- [ ] **Step 4: Run to verify pass** — `python3 -m pytest tests/test_gui_viewer3d.py -q && python3 tests/gui_smoke.py` → PASS/green.

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && git add -A
git commit -m "feat(gui): viewer appearance controls — mode/cmap/clim/log/opacity/background (docs synced)"
```

---

### Task 10: Viewer controls — threshold, clip, downsample, camera, bounds axes

**Files:**
- Modify: `gui/widgets/viewer3d_window.py`
- Modify: `docs/Usage.md`
- Test: `tests/test_gui_viewer3d.py` (append)

**Interfaces:**
- Consumes: Task 9's window; `render3d.CameraSpec`, `apply_camera`.
- Produces: widget attributes `_thresh_check`, `_thresh_min`, `_thresh_max`, `_clip_check`, `_clip_axis_combo` ("X", "Y", "Z"), `_clip_flip_btn`, `_downsample_spin` (1–16), `_bounds_check`, camera preset buttons `_cam_front`, `_cam_top`, `_cam_side`, `_cam_iso`, and `_az_spin`, `_el_spin`, `_zoom_spin` + `_cam_apply_btn`. Methods: `_current_clip() -> tuple | None` (axis-aligned plane through the volume centre, flipped sign when toggled), `_apply_camera_fields()`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_threshold_and_downsample_controls(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    monkeypatch.setattr(w, "rebuild", lambda: None)
    w._thresh_min.setValue(0.2); w._thresh_max.setValue(0.9)
    w._thresh_check.setChecked(True)
    assert w.scene.threshold == (0.2, 0.9)
    w._thresh_check.setChecked(False)
    assert w.scene.threshold is None
    w._downsample_spin.setValue(4)
    assert w.scene.downsample == 4
    w.close()


def test_clip_plane_axis_and_flip(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    monkeypatch.setattr(w, "rebuild", lambda: None)
    w._clip_axis_combo.setCurrentText("Y")
    w._clip_check.setChecked(True)
    origin, normal = w.scene.clip
    # centre of (2,3,4) at spacing (0.15, 0.38, 2.0): y = 3*0.38/2
    assert origin[1] == pytest.approx(0.57)
    assert normal == (0.0, 1.0, 0.0)
    w._clip_flip_btn.click()
    assert w.scene.clip[1] == (0.0, -1.0, 0.0)
    w._clip_check.setChecked(False)
    assert w.scene.clip is None
    w.close()


def test_camera_fields_build_cameraspec(monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    seen = {}
    monkeypatch.setattr(
        "gui.widgets.viewer3d_window.R3.apply_camera", lambda pl, cam: seen.update(cam=cam)
    )
    w._az_spin.setValue(30.0); w._el_spin.setValue(15.0); w._zoom_spin.setValue(1.5)
    w._cam_apply_btn.click()
    if w._canvas.available:  # offscreen may lack GL; the guard itself is under test
        assert seen["cam"].azimuth == 30.0 and seen["cam"].elevation == 15.0
    w.close()
```

- [ ] **Step 2: Run to verify failure** — FAIL (`AttributeError: _thresh_check`).

- [ ] **Step 3: Implement.** Threshold: checkbox gates `scene.threshold = (min, max)`/`None` + rebuild; spin edits while checked update and rebuild. Clip (`_current_clip`):

```python
    def _current_clip(self):
        if not self._clip_check.isChecked():
            return None
        vol, (sx, sy, sz) = self.scene.volume, self.scene.spacing
        z, y, x = vol.shape
        centre = (x * sx / 2.0, y * sy / 2.0, z * sz / 2.0)
        axis = {"X": 0, "Y": 1, "Z": 2}[self._clip_axis_combo.currentText()]
        normal = [0.0, 0.0, 0.0]
        normal[axis] = -1.0 if self._clip_flipped else 1.0
        return (centre, tuple(normal))
```

(voxel-mask clip via `Scene3D.clip` — the spec's v1; a live vtk plane widget is a follow-up). Camera preset buttons call `R3.apply_camera(self._canvas.plotter, R3.CameraSpec(preset=...))` + `render()` guarded by `self._canvas.available`; `_cam_apply_btn` builds `CameraSpec(preset="front", azimuth=_az, elevation=_el, zoom=_zoom)`. `_bounds_check` toggles `plotter.show_bounds(xtitle="X (µm)", ytitle="Y (µm)", ztitle="Z (µm)")` / `plotter.remove_bounds_axes()`. Every handler that touches structure (threshold/clip/downsample) rebuilds; camera/bounds only render. `Usage.md` updated.

- [ ] **Step 4: Run to verify pass** — `python3 -m pytest tests/test_gui_viewer3d.py -q && python3 tests/gui_smoke.py`.

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && git add -A
git commit -m "feat(gui): viewer threshold/clip/downsample/camera/bounds controls (docs synced)"
```

---

### Task 11: Viewer exports — styled figure, screenshot, rotation video (child process)

**Files:**
- Modify: `gui/widgets/viewer3d_window.py`
- Modify: `docs/Usage.md`, `docs/Codebase.md`
- Test: `tests/test_gui_viewer3d.py` (append)

**Interfaces:**
- Consumes: `render3d.render_scene_image` + `scene_figure` (figure), `plotter.screenshot` (raw), `StageRunner("dfxm.viewer_jobs:rotation_video_job", job)` + Task 7's params shape (video), `plotting.style_from_json`.
- Produces: toolbar buttons `_fig_btn` ("Save figure…"), `_shot_btn` ("Save screenshot…"), `_video_btn` ("Save rotation video…"); `_video_job_params(base_path, fmt, n_frames, fps) -> dict` (pure-ish, tested); all three disabled when `not self._canvas.available`.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_video_job_params_round_trip_jsonable():
    import json

    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    w.scene.mode = "surface"; w.scene.downsample = 2
    job = w._video_job_params("/tmp/out/orbit", "mp4", 90, 15)
    json.dumps(job)  # JSON-able end to end
    assert job["loader"] == w._spec.loader
    assert job["scene"]["mode"] == "surface" and job["scene"]["downsample"] == 2
    assert job["cbar_label"] == "Intensity" and job["group"] == "raw"
    assert job["n_frames"] == 90 and job["base_path"] == "/tmp/out/orbit"
    # no GL offscreen -> orbit around the default pose
    assert job["base_camera"] is None or len(job["base_camera"]) == 3
    w.close()


def test_save_figure_writes_png(tmp_path, monkeypatch):
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    fake_img = np.full((60, 80, 3), 200, dtype=np.uint8)
    monkeypatch.setattr(
        "gui.widgets.viewer3d_window.R3.render_scene_image",
        lambda scene, cam, window_size: (fake_img, 2.0),
    )
    out = tmp_path / "fig.png"
    w._save_figure_to(str(out), window_size=(80, 60))
    assert out.stat().st_size > 0
    w.close()


def test_export_buttons_disabled_without_gl():
    w = Viewer3DWindow(_spec(), "visualize")
    w.load_and_render()
    if not w._canvas.available:
        assert not w._fig_btn.isEnabled()
        assert not w._video_btn.isEnabled()
    w.close()
```

- [ ] **Step 2: Run to verify failure** — FAIL (`AttributeError: _video_job_params`).

- [ ] **Step 3: Implement.**

```python
    def _video_job_params(self, base_path: str, fmt: str, n_frames: int, fps: int) -> dict:
        s = self.scene
        base_cam = None
        if self._canvas.available:
            base_cam = [list(v) for v in self._canvas.plotter.camera_position]
        return {
            "loader": self._spec.loader,
            "scene": {
                "mode": s.mode, "cmap": s.cmap,
                "clim": list(s.resolved_clim()), "log_scale": s.log_scale,
                "opacity": s.opacity, "opacity_mapping": s.opacity_mapping,
                "threshold": list(s.threshold) if s.threshold else None,
                "clip": [list(s.clip[0]), list(s.clip[1])] if s.clip else None,
                "downsample": s.downsample, "background": s.background,
            },
            "base_camera": base_cam,
            "elevation": 0.0 if base_cam else 20.0, "zoom": 1.2,
            "n_frames": int(n_frames), "fps": int(fps),
            "base_path": base_path, "fmt": fmt,
            "cbar_label": self.loaded.cbar_label, "group": self.loaded.group,
            "style_json": self._style_json,
        }

    def _save_figure_to(self, path: str, *, window_size=(1920, 1080)) -> None:
        cam = (
            tuple(tuple(v) for v in self._canvas.plotter.camera_position)
            if self._canvas.available
            else R3.CameraSpec(preset="front")
        )
        got = R3.render_scene_image(self.scene, cam, window_size=window_size)
        if got is None:
            self._status.setText("nothing to export (empty scene)")
            return
        img, px_per_um = got
        from dfxm.common.plotting import style_from_json

        fig, _ax, _im = R3.scene_figure(
            img, px_per_um=px_per_um, cbar_label=self.loaded.cbar_label,
            group=self.loaded.group, clim=self.scene.resolved_clim(),
            log_scale=self.scene.log_scale, cmap=self.scene.cmap,
            style=style_from_json(self._style_json),
        )
        fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
```

Button handlers: figure → `QFileDialog.getSaveFileName` (PNG) + a small width/height dialog (two spinboxes, defaults 1920×1080) then `_save_figure_to`; screenshot → file dialog + `self._canvas.plotter.screenshot(path)`; video → file dialog (base path + mp4/gif combo + frames spinbox default 180) then:

```python
        runner = StageRunner("dfxm.viewer_jobs:rotation_video_job", job)
        runner.start()
```

polled by a `QTimer` (200 ms) draining `runner.poll()` into a `QProgressDialog` (`Progress.frac` → percent; cancel → `runner.cancel()`; `Done` → status text with the written path; `Failed` → status text with the error). Keep the runner+timer as window attributes so they outlive the handler; disable `_video_btn` while running. Look at how `stage_view.py` drives `StageRunner.poll()` (Read its run/poll region) and mirror the message handling. Where the video needs the viewer's style: `stage_view` passes the current style JSON when constructing windows — thread `style_json` from `Volume3DPanel.set_sources` if a style is available (optional; empty string = default style is acceptable for v1 if stage_view has no style handy — check `stage_view` for the style params it holds and wire if trivial).

Docs: `Usage.md` exports section (figure/screenshot/video, what each produces); `Codebase.md` window entry updated.

- [ ] **Step 4: Run to verify pass** — `python3 -m pytest tests/test_gui_viewer3d.py tests/test_viewer_jobs.py -q && python3 tests/gui_smoke.py`.

- [ ] **Step 5: Lint and commit**

```bash
ruff check . && git add -A
git commit -m "feat(gui): viewer exports — styled figure, screenshot, child-process rotation video (docs synced)"
```

---

### Task 12: Smoke coverage, full verification, real-data canary

**Files:**
- Modify: `tests/gui_smoke.py` (new numbered check)
- Test: whole suite + smoke + headless real-data run

**Interfaces:** consumes everything above; produces no new API.

- [ ] **Step 1: Add smoke check** (Read `gui_smoke.py`'s final checks and follow the pattern; next free number):

```python
    # [40] 3-D viewer: launcher on the visualize view opens a Viewer3DWindow;
    # controls mutate the scene; close prunes the window list (offscreen: the
    # GL canvas degrades to its placeholder and export buttons disable).
    from gui.viewers import LoadedVolume, VolumeSourceSpec
    import numpy as np

    lv = LoadedVolume(np.ones((2, 3, 4)), (0.15, 0.38, 2.0), "magma", (0.5, 1.0), "I", "raw")
    spec = VolumeSourceSpec("smoke_vol", lambda: lv, {"kind": "h5_dataset", "path": "/x", "dataset": "d"})
    panel = view._vol3d  # the visualize stage view's launcher tab
    panel.set_sources({"smoke_vol": spec}, "visualize")
    panel._open_btn.click()
    assert len(panel._windows) == 1
    win = panel._windows[0]
    win._mode_combo.setCurrentText("surface")
    assert win.scene.mode == "surface"
    win.close()
    app.processEvents()
    assert len(panel._windows) == 0
    print("[40] 3-D viewer: launcher opens window, controls live, close frees")
```

- [ ] **Step 2: Full verification (verify-suite skill applies)**

```bash
python3 -m pytest -q          # expect: prior count + new tests, 13 skips, 0 warnings-regression
python3 tests/gui_smoke.py    # expect: [1]-[40] all green
ruff check . && ruff format --check .
```

- [ ] **Step 3: Real-GL + real-data canary (this machine has GL + the STO2 data)**

```bash
# GL tests must RUN here, not skip:
python3 -m pytest tests/test_render3d_gl.py -v
# Headless stage run on real data (adjust paths to the STO2 experiment; small ROI to keep it quick):
python3 -m dfxm.stages.visualize \
  --mosa_volume_file /path/to/data/ESRF/ma6778/id03/20251029/PROCESSED_DATA/STO2_overnight/stacked_volumes.h5 \
  --raw_root /path/to/data/ESRF/ma6778/id03/20251029/RAW_DATA \
  --save_layers false --save_animation false --save_topview true --save_rotation true \
  --rotation_frames 36 --output_format mp4 \
  --output_dir /tmp/claude-1000/-home-albert-Desktop-dfxm-pipeline/09b341dc-cf83-47b4-b973-e975d32c1493/scratchpad/vis3d_canary
```

(Check the stage CLI's exact flag syntax with `python3 -m dfxm.stages.visualize -h` first; use the STO2 preset's mosa_pattern/pixel sizes — see the experiment preset. If the raw data drive is not mounted, note it and hand the command to Albert instead.) Open 3–4 spread frames of the MP4 (ffmpeg frame extraction) and confirm the volume orbits — visibly different viewpoints — and every frame carries the colorbar + scale bar on white.

- [ ] **Step 4: Send Albert the canary video + a top view for eyeball; note remaining manual checks** (on-screen viewer session, real pop-out interaction, clip/threshold on real data).

- [ ] **Step 5: Commit any smoke/doc touch-ups**

```bash
git add -A && git commit -m "test(smoke): 3-D viewer window check [40]; canary artifacts noted"
```

---

## Self-Review Notes

- **Spec coverage:** rendering technique switch (T2/T4/T5), camera-recipe fix + regression test (T1/T4), styled white video/top-view with colorbar + exact scale bar (T3/T4), new stage params + log guard (T5), source metadata (T6), child-process video job (T7), pop-out window + launcher + memory-on-close (T8), full v1 control set (T9/T10), three exports (T11), smoke + real-data verification (T12). Out-of-scope items (PDF/SVG, settings persistence, rocking video wiring, live vtk clip-plane widget) are deliberately absent; the clip control ships as axis presets + flip over `Scene3D.clip` voxel masking, which the spec's export path requires anyway — the draggable widget is noted as a follow-up in Usage.md.
- **Type consistency check done:** `Scene3D.prepared()` returns `(vol, spacing)`; `render_scene_image` returns `(img, px_per_um) | None`; `scene_figure` returns `(fig, ax, im)`; `aligned_field` returns a 5-tuple; `VolumeSourceSpec.load` returns `LoadedVolume`; `rotation_video_job` returns `{"video": path}` — used consistently across T4–T11.
- **Known judgment calls for the executor:** `_orbit_frames`' dense pose table may be simplified to per-frame `orbit_positions` computation (noted inline); `ParamType.INT` existence must be checked (noted); smoke-test check numbers and `test_stage_visualize.py` fixture names must be adapted after Reading those files (assertions are the contract, not the scaffolding names).
