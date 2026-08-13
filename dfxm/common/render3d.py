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
from matplotlib.figure import Figure

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
    opacity: float = 0.85  # scalar transparency, honoured by EVERY mode
    opacity_mapping: str = "linear"  # volume mode transfer function, scaled by opacity
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


def _contour_meshes(grid, clim, n_isosurfaces):
    """[(level, contour mesh)] at evenly spaced interior levels of *clim*.

    ``_grid_for_scene`` uploads the scalars as CELL data (the sentinel
    threshold needs them there), but VTK's contour filter only accepts POINT
    data — so the scalars are interpolated to points once, before the level
    loop. Meshes are returned for every level, empty ones included, so callers
    can ramp opacity over the full level list.
    """
    lo, hi = float(clim[0]), float(clim[1])
    levels = np.linspace(lo, hi, int(n_isosurfaces) + 2)[1:-1]
    if levels.size == 0:
        return []
    point_grid = grid.cell_data_to_point_data()
    return [(float(lv), point_grid.contour([float(lv)], scalars="values")) for lv in levels]


def _volume_opacity(scene: Scene3D):
    """The named opacity mapping as a 256-step curve scaled by ``scene.opacity``.

    Volume mode cannot just hand ``opacity_mapping`` to ``add_volume``: the name
    alone always maps to the full 0–255 alpha ramp, so the scalar opacity (the
    stages' ``volume_opacity``) would be a silent no-op and only surface /
    isosurface modes would honour it. Building the curve explicitly and scaling
    it makes one opacity knob mean the same thing in every render mode.
    """
    import pyvista as pv

    curve = np.asarray(pv.opacity_transfer_function(scene.opacity_mapping, 256), dtype=float)
    return np.clip(curve * float(scene.opacity), 0.0, 255.0)


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
            opacity=_volume_opacity(scene),
            shade=True,
            ambient=0.3,
            diffuse=0.6,
            specular=0.2,
            show_scalar_bar=sb is not None,
            scalar_bar_args=sb,
            **common,
        )
    elif scene.mode == "isosurface":
        contours = _contour_meshes(grid, clim, scene.n_isosurfaces)
        for i, (_level, contour) in enumerate(contours):
            if contour.n_points:
                plotter.add_mesh(
                    contour,
                    opacity=scene.opacity * (i + 1) / len(contours),
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

    ``origin="upper"``: a pyvista screenshot's row 0 is the TOP of the render,
    so the image must be drawn top-row-first (``origin="lower"`` published every
    3-D figure and video frame upside-down). The µm extent is unchanged either
    way, so the scale bar stays exact. Colorbar and scale bar honour the style's
    ``colorbar``/``scale_bar`` flags, like ``render.draw_map_layer``.
    """
    import matplotlib.colors as mcolors
    from matplotlib.cm import ScalarMappable

    from .plotting import PlotStyle, add_colorbar, apply_text_scale, draw_scale_bar, get_cmap

    st = style if style is not None else PlotStyle(scale_bar_color="black", colorbar_fraction=0.046)
    h, w = np.asarray(img).shape[:2]
    ext_x, ext_y = w / float(px_per_um), h / float(px_per_um)
    fig = Figure(figsize=(12, 12 * h / w + 1.0), facecolor="white")
    ax = fig.add_subplot(111)
    im = ax.imshow(np.asarray(img), extent=[0, ext_x, 0, ext_y], origin="upper", aspect="equal")
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    vmin, vmax = float(clim[0]), float(clim[1])
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax) if log_scale else mcolors.Normalize(vmin, vmax)
    sm = ScalarMappable(norm=norm, cmap=get_cmap(cmap))
    sm.set_array([])
    fig._scene_mappable = sm  # test/debug hook: the mappable behind the colorbar
    if st.colorbar:
        add_colorbar(fig, sm, ax, cbar_label, st, group=group)
    if st.scale_bar:
        draw_scale_bar(ax, st.scale_bar_length_um, style=st)
    apply_text_scale(ax, st)
    return fig, ax, im


def _video_from_frames(
    get_frame,
    n_frames,
    base_path,
    fmt,
    *,
    fps,
    cbar_label,
    group,
    clim,
    log_scale,
    cmap,
    px_per_um,
    style,
):
    """Assemble RGB frames into a styled MP4/GIF (colorbar in every frame).

    Builds the :func:`scene_figure` ONCE from frame 0 and swaps only the image
    per frame — so the colorbar, scale bar and figure geometry are identical in
    every frame (they must be: the frames all share one (H, W)). ``get_frame``
    must be idempotent in ``i`` — the animation is replayed for ``fmt="both"``
    and for the MP4→GIF fallback (see ``render._save_animation``).
    """
    from matplotlib.animation import FuncAnimation

    from .render import _save_animation

    fig, _ax, im = scene_figure(
        np.asarray(get_frame(0)),
        px_per_um=px_per_um,
        cbar_label=cbar_label,
        group=group,
        clim=clim,
        log_scale=log_scale,
        cmap=cmap,
        style=style,
    )

    def update(frame):
        if frame:
            im.set_data(np.asarray(get_frame(frame)))
        return [im]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    return _save_animation(anim, base_path, fmt, fps=fps, dpi=100)


def _orbit_frames(scene: Scene3D, *, elevation, zoom, base_camera, window_size):
    """``(get_frame, px_per_um)`` closure rendering absolute-pose orbit frames.

    ONE off-screen plotter, built once and reused for every frame (so every
    frame shares the same ``window_size`` and therefore the same image shape);
    each frame assigns an ABSOLUTE ``camera_position`` from
    :func:`orbit_positions` — never incremental vtk ``Azimuth()`` calls, which
    made the old video re-render the same pose over and over ("the rotation
    video doesn't rotate"). Returns ``None`` (plotter closed) when the scene has
    no finite voxels.

    ``base_camera`` — an explicit ``(eye, focal, up)`` triple — orbits around
    that pose instead of the ``"front"`` preset. The orbit table is regenerated
    whenever the caller changes ``get_frame.n_frames`` (default 360), so one
    closure serves any frame count while staying idempotent in ``i``. The caller
    owns the plotter's lifetime through ``get_frame.close()``.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    # Not try/FINALLY: on success the plotter must stay OPEN for the returned
    # closure — only the failure paths (and the empty scene) close it here.
    try:
        if not populate(pl, scene):
            pl.close()
            return None
        if base_camera is None:
            apply_camera(pl, CameraSpec(preset="front", zoom=zoom))
            base = tuple(tuple(float(c) for c in v) for v in pl.camera_position)
            elev = float(elevation)
        else:
            base = tuple(tuple(float(c) for c in v) for v in base_camera)
            elev = 0.0
        pl.enable_parallel_projection()
        pl.show(auto_close=False)
        px_per_um = window_size[1] / (2.0 * float(pl.camera.parallel_scale))
    except BaseException:
        pl.close()
        raise
    orbit: dict = {"n": None, "poses": None}

    def get_frame(i):
        n = max(1, int(get_frame.n_frames))
        if orbit["n"] != n:
            orbit["n"], orbit["poses"] = n, orbit_positions(base, elev, n)
        pl.camera_position = orbit["poses"][int(i) % n]
        # Explicit render: after the first one, pyvista's screenshot() only grabs
        # the window buffer, so without this every frame is a copy of frame 0 —
        # the second half of the "video doesn't rotate" bug.
        pl.render()
        return pl.screenshot(return_img=True)

    get_frame.n_frames = 360  # the caller overwrites this with the real count
    get_frame.close = pl.close
    return get_frame, px_per_um


def save_top_view(
    scene: Scene3D, path, *, cbar_label, group=None, style=None, window_size=(1920, 1080)
):
    """Styled top-view figure (colorbar + scale bar); returns path or None if empty."""
    got = render_scene_image(scene, CameraSpec(preset="top"), window_size=window_size)
    if got is None:
        return None
    img, px_per_um = got
    fig, _ax, _im = scene_figure(
        img,
        px_per_um=px_per_um,
        cbar_label=cbar_label,
        group=group,
        clim=scene.resolved_clim(),
        log_scale=scene.log_scale,
        cmap=scene.cmap,
        style=style,
    )
    fig.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    return path


def save_rotation_video(
    scene: Scene3D,
    base_path,
    fmt,
    *,
    cbar_label,
    group=None,
    style=None,
    n_frames=180,
    fps=15,
    elevation=20.0,
    zoom=1.2,
    base_camera=None,
    window_size=(1280, 960),
    progress=None,
):
    """360° orbit movie, publication-styled; returns path or None if empty.

    ``base_camera`` (an explicit (eye, focal, up) triple, e.g. the GUI viewer's
    live pose) orbits around that pose instead of the front preset. ``progress``
    is an optional ``(frac, msg)`` callable reporting per-frame progress.
    """
    got = _orbit_frames(
        scene, elevation=elevation, zoom=zoom, base_camera=base_camera, window_size=window_size
    )
    if got is None:
        return None
    frames, px_per_um = got
    n = int(n_frames)
    frames.n_frames = n
    get_frame = frames
    if progress is not None:

        def get_frame(i, _inner=frames):  # noqa: F811 - deliberate progress wrapper
            progress(min(0.99, i / max(1, n)), f"rendering orbit frame {i}/{n}")
            return _inner(i)

    try:
        return _video_from_frames(
            get_frame,
            n,
            base_path,
            fmt,
            fps=fps,
            cbar_label=cbar_label,
            group=group,
            clim=scene.resolved_clim(),
            log_scale=scene.log_scale,
            cmap=scene.cmap,
            px_per_um=px_per_um,
            style=style,
        )
    finally:
        frames.close()
