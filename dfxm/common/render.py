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

from .plotting import (
    PlotStyle,
    add_colorbar,
    apply_text_scale,
    draw_scale_bar,
    figure_size,
    fit_axes_to_box,
    fixed_scale_box,
    get_cmap,
    styled_figure,
)


def cmap_nan_transparent(name: str):
    """Colormap copy that renders NaN (padded) voxels as transparent white."""
    cmap = get_cmap(name).copy()
    cmap.set_bad(color="white", alpha=0.0)
    return cmap


def layer_figure(
    layer, vmin, vmax, cmap, ext_x, ext_y, title, cbar_label, *, style=None, group=None
):
    """Single equal-aspect layer figure (µm axes).

    ``style=None`` renders with the default un-styled :class:`PlotStyle` (black
    scale bar, 0.046 colourbar fraction). Note this still routes through the
    shared styled primitives, so it is close to — but not byte-identical with —
    the pre-export legacy renderer (scale-bar length/thickness and colourbar tick
    font differ slightly); see ``draw_scale_bar``/``add_colorbar``.
    """
    st = style if style is not None else PlotStyle(scale_bar_color="black", colorbar_fraction=0.046)
    box = fixed_scale_box(st, ext_x, ext_y) if style is not None else None
    if box is not None:
        figsize = (box[0] + 1.5, box[1] + 1.5)  # headroom; fit_axes_to_box converges regardless
    else:
        figsize = (figure_size(st, ext_x, ext_y) or (12, 10)) if style is not None else (12, 10)
    fig = styled_figure(figsize, styled=style is not None)
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
    if st.colorbar:
        add_colorbar(fig, im, ax, cbar_label, st, group=group)
    if st.scale_bar:
        draw_scale_bar(
            ax,
            st.scale_bar_length_um,
            style=st,
            fixed_scale_um_per_cm=(box[2] if box is not None else None),
        )
    apply_text_scale(ax, st)
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])
    return fig, ax, im


def save_layer_pngs(
    volume, z_um, out_dir, name, vmin, vmax, cmap, title, cbar, sx, sy, *, style=None, group=None
):
    """Write one PNG per Z layer into ``<out_dir>/<name>_layers/``; return the dir."""
    layers_dir = os.path.join(out_dir, f"{name}_layers")
    os.makedirs(layers_dir, exist_ok=True)
    ext_x, ext_y = volume.shape[2] * sx, volume.shape[1] * sy
    z_size = volume.shape[0]
    for z in range(z_size):
        full_title = f"{title}\nZ = {z_um[z]:.2f} µm (Layer {z}/{z_size - 1})"
        fig, _, _ = layer_figure(
            volume[z], vmin, vmax, cmap, ext_x, ext_y, full_title, cbar, style=style, group=group
        )
        fig.savefig(
            os.path.join(layers_dir, f"layer_{z:04d}.png"),
            dpi=150,
            facecolor="white",
            bbox_inches="tight",
        )
    return layers_dir


def save_layer_animation(
    volume,
    z_um,
    base_path,
    name,
    vmin,
    vmax,
    cmap,
    title,
    cbar,
    fmt,
    sx,
    sy,
    *,
    style=None,
    group=None,
):
    """Layer-by-layer flip-through movie. MP4 (ffmpeg) with GIF fallback."""
    ext_x, ext_y = volume.shape[2] * sx, volume.shape[1] * sy
    z_size = volume.shape[0]
    fig, ax, im = layer_figure(
        volume[0], vmin, vmax, cmap, ext_x, ext_y, title, cbar, style=style, group=group
    )
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
