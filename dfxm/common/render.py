"""Shared volume rendering, 2-D only — per-layer PNGs and the layer animation.

Generic over the scalar field: the caller passes a ``(Z, Y, X)`` volume, the Z
coordinates (µm), colour limits, a colormap name and labels. The visualize and
rocking stages both render through here so there is exactly one renderer.

Uses the explicit :class:`~matplotlib.figure.Figure`/Agg API (never ``pyplot``
or ``matplotlib.use``) so it is import-safe inside the Qt GUI process. Nothing
here touches ``pyvista``: all 3-D rendering (top view, rotation video, the GUI
viewer) lives in :mod:`dfxm.common.render3d`, which reuses this module's
:func:`_save_animation` for its MP4/GIF container policy.
"""

from __future__ import annotations

import os

import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from . import progress as _progress
from .plotting import (
    PlotStyle,
    add_colorbar,
    apply_axes_mode,
    apply_text_scale,
    draw_scale_bar,
    figure_size,
    fit_axes_to_box,
    fixed_scale_box,
    get_cmap,
    styled_figure,
)


def cmap_nan_transparent(name: str):
    """Colormap copy that renders NaN (padded) voxels as transparent white.

    `with_extremes(bad=...)` rather than `copy()` + `set_bad(...)`: matplotlib
    3.11 raises a `PendingDeprecationWarning` from `set_bad`, which turns any
    caller running warnings-as-errors into a failure (it broke
    `test_compose_render.py::test_degenerate_roi_extent_renders_placeholder_not_singular_imshow`,
    whose whole point is that rendering that panel is warning-free). The
    alpha goes in the RGBA tuple because `with_extremes` takes no separate
    `alpha`; the two spellings were verified to produce identical `get_bad()`
    and identical mapping of a NaN-bearing row under matplotlib 3.6.3 and
    3.11.1. `with_extremes` already returns a copy, so the explicit one is gone
    — the shared registry colormap is still never mutated.
    """
    return get_cmap(name).with_extremes(bad=(1.0, 1.0, 1.0, 0.0))


def draw_map_layer(
    ax,
    layer,
    vmin,
    vmax,
    cmap,
    ext_x,
    ext_y,
    title,
    cbar_label,
    *,
    style=None,
    group=None,
    cax=None,
    colorbar=None,
    scale_bar=None,
    fixed_scale_um_per_cm=None,
):
    """Draw one equal-aspect map layer into *ax* (µm axes); returns the image.

    Extracted verbatim from :func:`layer_figure` so the single-figure path and
    the compose adapters share one look. ``colorbar``/``scale_bar`` default to
    the style's flags; an explicit bool overrides (the composer switches them
    off when a shared bar covers the panel). ``cax`` routes the colourbar into
    an already-placed axes (steal-free, see ``add_colorbar``).
    """
    st = style if style is not None else PlotStyle(scale_bar_color="black", colorbar_fraction=0.046)
    fig = ax.get_figure()
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
    if st.colorbar if colorbar is None else colorbar:
        add_colorbar(fig, im, ax, cbar_label, st, group=group, cax=cax)
    if st.scale_bar if scale_bar is None else scale_bar:
        draw_scale_bar(
            ax, st.scale_bar_length_um, style=st, fixed_scale_um_per_cm=fixed_scale_um_per_cm
        )
    apply_text_scale(ax, st)
    apply_axes_mode(ax, st)
    return im


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
    im = draw_map_layer(
        ax,
        layer,
        vmin,
        vmax,
        cmap,
        ext_x,
        ext_y,
        title,
        cbar_label,
        style=style,
        group=group,
        fixed_scale_um_per_cm=(box[2] if box is not None else None),
    )
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])
    return fig, ax, im


def save_layer_pngs(
    volume,
    z_um,
    out_dir,
    name,
    vmin,
    vmax,
    cmap,
    title,
    cbar,
    sx,
    sy,
    *,
    style=None,
    group=None,
    progress=None,
):
    """Write one PNG per Z layer into ``<out_dir>/<name>_layers/``; return the dir.

    *progress* takes a **local** 0..1 fraction — wrap it with
    `dfxm.common.progress.sub_progress` to place it in a caller's range. This is
    the longest inner loop either of its callers has (a real `visualize` run
    writes ~78 files per dataset), so it is where a run otherwise goes quiet for
    minutes at a time.
    """
    layers_dir = os.path.join(out_dir, f"{name}_layers")
    os.makedirs(layers_dir, exist_ok=True)
    report = progress or _progress.noop
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
        # After the write, not before: the fraction reports work *done*, so a
        # cancelled run's last report is never one layer ahead of the disk.
        report((z + 1) / max(1, z_size), f"{name}: layer {z + 1}/{z_size}")
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
    progress=None,
):
    """Layer-by-layer flip-through movie. MP4 (ffmpeg) with GIF fallback.

    *progress* takes a **local** 0..1 fraction — wrap it with
    `dfxm.common.progress.sub_progress` to place it in a caller's range. Like
    :func:`save_layer_pngs`, this renders every layer, so it is one of the two
    long inner loops in a `visualize` or `rocking` dataset; without a reporter
    its caller can only mark the boundaries around it, and a run with the layer
    PNGs switched off has nothing else to speak between them.

    The denominator counts the render passes, not the layers: ``fmt="both"``
    drives the animation twice, once per container.
    """
    report = progress or _progress.noop
    # `FuncAnimation` calls `update` once for the first frame while setting up,
    # before the save loop calls it for that frame again. Counting calls put the
    # fraction a whole frame ahead of the frame its message names; counting
    # *distinct* frames keeps the two saying the same thing. A second container
    # replays the same frame numbers, which is a change of `z` and so still
    # advances.
    ext_x, ext_y = volume.shape[2] * sx, volume.shape[1] * sy
    z_size = volume.shape[0]
    fig, ax, im = layer_figure(
        volume[0], vmin, vmax, cmap, ext_x, ext_y, title, cbar, style=style, group=group
    )
    title_obj = ax.set_title(f"{title}\nZ = {z_um[0]:.2f} µm (Layer 0/{z_size - 1})")
    total_frames = max(1, z_size * (2 if fmt == "both" else 1))
    rendered = [0]
    last_z = [-1]

    def update(frame):
        z = frame % z_size
        im.set_data(volume[z])
        title_obj.set_text(f"{title}\nZ = {z_um[z]:.2f} µm (Layer {z}/{z_size - 1})")
        # After the frame is composed, and counted rather than derived from
        # `frame`: a GIF fallback after a failed MP4 replays the same frame
        # numbers, and only a running count keeps the fraction monotonic.
        if z != last_z[0]:
            rendered[0] += 1
            last_z[0] = z
        report(rendered[0] / total_frames, f"{name}: animation frame {z + 1}/{z_size}")
        return [im, title_obj]

    anim = FuncAnimation(fig, update, frames=z_size, blit=False)
    return _save_animation(anim, base_path, fmt, fps=15, dpi=120)


def _save_animation(anim, base_path, fmt, fps, dpi):
    """Save a FuncAnimation as MP4 (ffmpeg) with GIF fallback; returns the path.

    ``fmt`` is ``mp4``/``gif``/``both``. A failed MP4 write (ffmpeg missing or
    dying mid-save) falls back to GIF and removes the partial ``.mp4``.
    """
    written = None
    want_mp4 = fmt in ("mp4", "both")
    want_gif = fmt in ("gif", "both")
    if want_mp4:
        try:
            anim.save(base_path + ".mp4", writer=FFMpegWriter(fps=fps), dpi=dpi)
            written = base_path + ".mp4"
        except Exception:  # noqa: BLE001 - ffmpeg missing -> fall back to GIF
            if os.path.exists(base_path + ".mp4"):
                os.remove(base_path + ".mp4")
            want_gif = True
    if want_gif:
        anim.save(base_path + ".gif", writer=PillowWriter(fps=fps), dpi=dpi)
        written = written or base_path + ".gif"
    return written
