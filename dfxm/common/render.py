"""Shared volume rendering — per-layer PNGs, layer animation, 3D top-view.

Generic over the scalar field: the caller passes a ``(Z, Y, X)`` volume, the Z
coordinates (µm), colour limits, a colormap name and labels. The visualize and
rocking stages both render through here so there is exactly one renderer.

Uses the explicit :class:`~matplotlib.figure.Figure`/Agg API (never ``pyplot``
or ``matplotlib.use``) so it is import-safe inside the Qt GUI process. ``pyvista``
is imported lazily, so a missing GL/driver stack only disables the 3D top-view.
"""

from __future__ import annotations

import os

import matplotlib.colors as mcolors
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from .plotting import get_cmap


def cmap_nan_transparent(name: str):
    """Colormap copy that renders NaN (padded) voxels as transparent white."""
    cmap = get_cmap(name).copy()
    cmap.set_bad(color="white", alpha=0.0)
    return cmap


def add_scale_bar(ax, ext_x: float, ext_y: float, color: str = "black") -> None:
    """Draw a rounded µm scale bar (~15% of the X extent) in the lower-right."""
    target = ext_x * 0.15
    if target >= 100:
        sl = round(target / 50) * 50
    elif target >= 10:
        sl = round(target / 10) * 10
    elif target >= 1:
        sl = round(target)
    else:
        sl = round(target, 1)
    sl = sl or target
    bx, by, bh = ext_x * 0.95 - sl, ext_y * 0.05, ext_y * 0.01
    ax.add_patch(Rectangle((bx, by), sl, bh, facecolor=color, edgecolor=color))
    ax.text(
        bx + sl / 2,
        by + bh * 3,
        f"{sl:.0f} µm",
        color=color,
        fontsize=10,
        ha="center",
        va="bottom",
        fontweight="bold",
    )


def layer_figure(layer, vmin, vmax, cmap, ext_x, ext_y, title, cbar_label):
    """Build a single equal-aspect layer figure (µm axes, scale bar, colorbar)."""
    fig = Figure(figsize=(12, 10), facecolor="white")
    ax = fig.add_subplot(111)
    im = ax.imshow(
        layer,
        cmap=cmap_nan_transparent(cmap),
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
        extent=[0, ext_x, 0, ext_y],
        origin="lower",
        aspect="equal",
    )
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(cbar_label)
    add_scale_bar(ax, ext_x, ext_y)
    return fig, ax, im


def save_layer_pngs(volume, z_um, out_dir, name, vmin, vmax, cmap, title, cbar, sx, sy):
    """Write one PNG per Z layer into ``<out_dir>/<name>_layers/``; return the dir."""
    layers_dir = os.path.join(out_dir, f"{name}_layers")
    os.makedirs(layers_dir, exist_ok=True)
    ext_x, ext_y = volume.shape[2] * sx, volume.shape[1] * sy
    z_size = volume.shape[0]
    for z in range(z_size):
        full_title = f"{title}\nZ = {z_um[z]:.2f} µm (Layer {z}/{z_size - 1})"
        fig, _, _ = layer_figure(volume[z], vmin, vmax, cmap, ext_x, ext_y, full_title, cbar)
        fig.savefig(
            os.path.join(layers_dir, f"layer_{z:04d}.png"),
            dpi=150,
            facecolor="white",
            bbox_inches="tight",
        )
    return layers_dir


def save_layer_animation(volume, z_um, base_path, name, vmin, vmax, cmap, title, cbar, fmt, sx, sy):
    """Layer-by-layer flip-through movie. MP4 (ffmpeg) with GIF fallback."""
    ext_x, ext_y = volume.shape[2] * sx, volume.shape[1] * sy
    z_size = volume.shape[0]
    fig, ax, im = layer_figure(volume[0], vmin, vmax, cmap, ext_x, ext_y, title, cbar)
    title_obj = ax.set_title(f"{title}\nZ = {z_um[0]:.2f} µm (Layer 0/{z_size - 1})")

    def update(frame):
        z = frame % z_size
        im.set_data(volume[z])
        title_obj.set_text(f"{title}\nZ = {z_um[z]:.2f} µm (Layer {z}/{z_size - 1})")
        return [im, title_obj]

    anim = FuncAnimation(fig, update, frames=z_size, blit=False)
    written = None
    want_mp4 = fmt in ("mp4", "both")
    want_gif = fmt in ("gif", "both")
    if want_mp4:
        try:
            anim.save(base_path + ".mp4", writer=FFMpegWriter(fps=15), dpi=120)
            written = base_path + ".mp4"
        except Exception:  # noqa: BLE001 - ffmpeg missing -> fall back to GIF
            want_gif = True
    if want_gif:
        anim.save(base_path + ".gif", writer=PillowWriter(fps=15), dpi=120)
        written = written or base_path + ".gif"
    return written


def _pyvista_grid(data, spacing):
    """ImageData grid with NaN voxels thresholded out (lazy pyvista import)."""
    import pyvista as pv

    dt = np.transpose(data, (2, 1, 0))
    finite = dt[np.isfinite(dt)]
    sentinel = (
        (float(np.min(finite)) - 1000.0 * (float(np.ptp(finite)) + 1.0)) if finite.size else -1e30
    )
    dc = np.where(np.isfinite(dt), dt, sentinel)
    grid = pv.ImageData()
    grid.dimensions = np.array(dc.shape) + 1
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.cell_data["values"] = dc.flatten(order="F")
    thresh = sentinel * 0.5 if sentinel < 0 else sentinel + 1.0
    return grid.threshold(value=thresh, scalars="values")


def save_top_view(volume, scale_z, sx, sy, vmin, vmax, cmap, opacity, path):
    """Single top-view (XY) 3D render via pyvista; returns path or None if empty."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    grid = _pyvista_grid(volume, spacing=(sx, sy, scale_z))
    if grid.n_cells == 0:
        return None
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(
        grid,
        scalars="values",
        cmap=cmap,
        clim=[vmin, vmax],
        opacity=opacity,
        smooth_shading=True,
        show_edges=False,
    )
    pl.view_xy()
    pl.screenshot(path)
    pl.close()
    return path
