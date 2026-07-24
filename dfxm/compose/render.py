"""Render a FigureRecipe into one Figure at exact physical size; export it.

No tight-crop anywhere: the solver owns all margins, so the saved canvas IS
the figure geometry. No matplotlib auto-layout."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace

from matplotlib.figure import Figure

from ..common.errors import StageUserError
from ..common.plotting import (
    PlotStyle,
    add_colorbar,
    box_drift_note,
    draw_scale_bar,
    style_from_params,
)
from .adapters import draw_panel, load_panel
from .layout import SizedCell, measure_cells, place_tree, size_cells
from .recipe import Col, PanelRef, Row, Spacer, TextCell, iter_leaves, validate_recipe

_IN_PER_CM = 1.0 / 2.54

_MIXED_GROUP_HINT = (
    "Group only panels of one quantity (e.g. all strain) under a shared bar, "
    "or give each its own bar."
)
_NO_SCALE_BAR_PANEL_HINT = (
    "Set scale_bar_panel to one of the panel ids in this recipe, or switch scale_bar_mode."
)
_GUTTER_MISMATCH_HINT = (
    "A shared scale bar needs every map at one µm/cm — remove per-panel scale "
    "overrides or use per-panel bars."
)


@dataclass
class ComposeResult:
    figure: Figure
    notes: list[str] = field(default_factory=list)
    n_panels: int = 0
    n_rendered: int = 0
    axes_by_id: dict = field(default_factory=dict)


def _format_label(template: str, index: int) -> str:
    for j, ch in enumerate(template):
        if ch in ("A", "a"):
            return template[:j] + chr(ord(ch) + index) + template[j + 1 :]
    return template


def _assign_labels(layout, panels_by_id, compose):
    """Depth-first auto-increment; a group_label node consumes ONE slot for the
    whole group; manual labels replace the slot's text; '' suppresses."""
    out: dict[object, str] = {}
    seq = 0

    def leaf_panels(node):
        if isinstance(node, (Row, Col)):
            for c in node.children:
                yield from leaf_panels(c)
        elif isinstance(node, PanelRef):
            yield node

    def walk(node):
        nonlocal seq
        if isinstance(node, (Row, Col)):
            if node.group_label is not None:
                text = _format_label(compose.label_template, seq)
                if node.group_label not in ("", "auto"):
                    text = node.group_label
                seq += 1
                first = next(leaf_panels(node), None)
                if first is not None:
                    out[id(first)] = text
                return
            for c in node.children:
                walk(c)
        elif isinstance(node, PanelRef):
            p = panels_by_id[node.panel_id]
            if p.label == "":
                return
            text = p.label if p.label is not None else _format_label(compose.label_template, seq)
            seq += 1
            out[id(node)] = text

    walk(layout)
    return out


def _draw_label(ax, text, style, compose):
    size = 12.0 * style.font_scale * compose.label_font_scale
    ax.annotate(
        text,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(0.0, 4.0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=size,
        fontweight="bold",
        annotation_clip=False,
    )


def _panel_leaves(node):
    """PanelRef leaves under *node* (any depth), depth-first order."""
    return [leaf for leaf in iter_leaves(node) if isinstance(leaf, PanelRef)]


def _cbar_label(data) -> str:
    if data.kind == "map_layer":
        return data.payload["cbar"]
    if data.kind == "slice_plane":
        return data.payload["prep"]["cbar_label"]
    if data.kind == "profiles_ref":
        return data.payload["attrs"]["cbar_label"]
    return ""


def _find_shared_bar_nodes(node):
    if isinstance(node, (Row, Col)):
        if node.shared_colorbar:
            yield node
        for c in node.children:
            yield from _find_shared_bar_nodes(c)


def _find_shared_x_nodes(node):
    if isinstance(node, (Row, Col)):
        if isinstance(node, Col) and node.shared_x:
            yield node
        for c in node.children:
            yield from _find_shared_x_nodes(c)


def _resolve_show_xlabel(recipe, data_by_id):
    """PanelRef ids whose trace should NOT draw an x-label/tick labels: every
    trace leaf under a ``shared_x`` Col except the last one."""
    hide: set[str] = set()
    for node in _find_shared_x_nodes(recipe.layout):
        trace_leaves = [
            m for m in _panel_leaves(node) if data_by_id[m.panel_id].kind == "profiles_trace"
        ]
        for m in trace_leaves[:-1]:
            hide.add(m.panel_id)
    return hide


def _make_cax_sync(cax, style, cell):
    """Re-glue a per-panel colourbar axes beside *cell*'s main axes after every
    placement pass (cell.w_in is read live so a later pinned-width rescale
    still lands the bar in the right place)."""

    def _sync(fig_, ax_):
        pos = ax_.get_position()
        fw, _fh = fig_.get_size_inches()
        w_in = cell.w_in
        cax.set_position(
            [
                pos.x1 + 0.04 * w_in / fw,
                pos.y0,
                style.colorbar_fraction * w_in / fw,
                pos.height,
            ]
        )

    return _sync


def _apply_shared_colorbars(recipe, style, panels_by_id, data_by_id, cells, fig):
    """Unify clim across each shared-colorbar group and build its bar cell.

    Mutates *panels_by_id* (replacing member entries with a copy carrying the
    unified ``clim``) and *cells* (adding one synthetic entry per bar leaf).
    Returns ``(no_colorbar_pids, bar_specs)`` where each spec is
    ``(node, group, member_pids, bar_leaf, bar_ax)``.
    """
    no_colorbar_pids: set[str] = set()
    bar_specs = []
    gutter_in = recipe.compose.gutter_cm * _IN_PER_CM
    for node in _find_shared_bar_nodes(recipe.layout):
        members = _panel_leaves(node)
        pids = [m.panel_id for m in members]
        member_data = [data_by_id[pid] for pid in pids]
        groups = {d.group for d in member_data if d.kind != "placeholder"}
        if len(groups) > 1:
            raise StageUserError(
                f"shared colorbar mixes quantity groups {sorted(g for g in groups if g)}",
                hint=_MIXED_GROUP_HINT,
            )
        grp = next(iter(groups), None)

        vmins, vmaxs = [], []
        for pid, d in zip(pids, member_data):
            if d.kind == "placeholder":
                continue
            p = panels_by_id[pid]
            lo, hi = p.clim if p.clim is not None else (None, None)
            vmins.append(lo if lo is not None else d.vmin)
            vmaxs.append(hi if hi is not None else d.vmax)
        unified = node.shared_clim or ((min(vmins), max(vmaxs)) if vmins else None)
        if unified is not None:
            for pid, d in zip(pids, member_data):
                if d.kind != "placeholder":
                    panels_by_id[pid] = dc_replace(panels_by_id[pid], clim=unified)
                    no_colorbar_pids.add(pid)

        member_cells = [cells[id(m)] for m in members]
        # Basis for the bar's cross-dimension: the first non-placeholder member
        # when one exists (a placeholder's box is the fixed PLACEHOLDER_CM
        # fallback, not representative of the group's real content size).
        first = next(
            (c for c, d in zip(member_cells, member_data) if d.kind != "placeholder"),
            member_cells[0],
        )
        # This box only reserves the solver's PROVISIONAL space for the bar
        # (content-box sums, no member decoration margins) — it is what lets
        # size_cells/place_tree give the bar a slot in the tree at all. The
        # bar's real cross-dimension is corrected after place_tree to the
        # group's ACTUAL placed envelope (member margins included), see
        # `_stretch_shared_bar` — a raw content-box sum under-counts the real
        # span by each member's own top/bottom (or left/right) decoration
        # margins, which is exactly what left the bar short of the group in
        # the pre-fix geometry.
        if isinstance(node, Col):
            content_h = sum(c.h_in for c in member_cells) + gutter_in * max(
                0, len(member_cells) - 1
            )
            bar_w_in = style.colorbar_fraction * first.w_in + 0.1
            bar_h_in = content_h
        else:
            content_w = sum(c.w_in for c in member_cells) + gutter_in * max(
                0, len(member_cells) - 1
            )
            bar_w_in = content_w
            bar_h_in = style.colorbar_fraction * first.h_in + 0.1

        bar_leaf = Spacer(bar_w_in / _IN_PER_CM, bar_h_in / _IN_PER_CM)
        bar_ax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
        cells[id(bar_leaf)] = SizedCell(bar_leaf, None, "spacer", bar_w_in, bar_h_in, ax=bar_ax)
        bar_specs.append((node, grp, pids, bar_leaf, bar_ax))
    return no_colorbar_pids, bar_specs


def _stretch_shared_bar(node, pids, bar_ax, axes_by_id, data_by_id):
    """Stretch/reposition *bar_ax* to the group's REAL placed span.

    `_apply_shared_colorbars` sizes the bar leaf from a content-box sum before
    placement (a provisional reservation — see the comment there); the real
    placed envelope of a Row/Col of panels also includes each member's own
    measured decoration margins (tick labels, title, etc.), which
    `place_tree`'s margin-sharing can make asymmetric besides. Calling this
    once placement is final (member axes have their true
    ``get_position()``) makes the bar span exactly the group's real top/bottom
    (a Col group) or left/right (a Row group) instead of drifting short.
    """
    member_axes = [axes_by_id[pid] for pid in pids if data_by_id[pid].kind != "placeholder"]
    if not member_axes:
        bar_ax.set_axis_off()
        return
    bpos = bar_ax.get_position()
    if isinstance(node, Col):
        top = max(ax.get_position().y1 for ax in member_axes)
        bottom = min(ax.get_position().y0 for ax in member_axes)
        bar_ax.set_position([bpos.x0, bottom, bpos.width, top - bottom])
    else:
        left = min(ax.get_position().x0 for ax in member_axes)
        right = max(ax.get_position().x1 for ax in member_axes)
        bar_ax.set_position([left, bpos.y0, right - left, bpos.height])


def _wrap_bar_node(node, bar_leaf):
    """A Col group's bar sits beside it (Row); a Row group's bar sits below (Col)."""
    if isinstance(node, Col):
        return Row([node, bar_leaf])
    return Col([node, bar_leaf])


def _build_working_layout(node, bar_map):
    """Rebuild *node* (never mutating the recipe's own tree), wrapping any
    node found in *bar_map* (``id(node) -> bar_leaf``) with its bar cell."""
    if isinstance(node, (Row, Col)):
        new_children = [_build_working_layout(c, bar_map) for c in node.children]
        changed = any(nc is not oc for nc, oc in zip(new_children, node.children))
        base = dc_replace(node, children=new_children) if changed else node
        bar_leaf = bar_map.get(id(node))
        if bar_leaf is not None:
            return _wrap_bar_node(base, bar_leaf)
        return base
    return node


def _resolve_scale_bar_kwargs(recipe, panels_by_id, data_by_id, cell_by_pid):
    """Per-panel ``scale_bar`` kwargs from ``compose.scale_bar_mode``, plus an
    optional gutter leaf (``"gutter"`` mode) and its shared µm/cm scale."""
    mode = recipe.compose.scale_bar_mode
    scale_bar_by_pid: dict[str, bool] = {}
    # `data_by_id`/`panels_by_id` are keyed by EVERY recipe.panels entry, but
    # `cell_by_pid` only has layout leaves — a PanelDef the layout no longer
    # references (e.g. left behind by a GUI delete that orphaned it, see
    # FigureBuilderWindow.delete_selected) must not reach a `cell_by_pid[pid]`
    # lookup below, so filter it out here.
    map_pids = [
        pid
        for pid, d in data_by_id.items()
        if d.kind in ("map_layer", "slice_plane", "profiles_ref") and pid in cell_by_pid
    ]
    gutter_leaf = None
    gutter_scale = None

    if mode == "one-panel":
        target = recipe.compose.scale_bar_panel
        if target not in panels_by_id:
            raise StageUserError(
                f"compose.scale_bar_panel {target!r} is not a known panel id",
                hint=_NO_SCALE_BAR_PANEL_HINT,
            )
        for pid in map_pids:
            scale_bar_by_pid[pid] = pid == target
    elif mode == "gutter":
        for pid in map_pids:
            scale_bar_by_pid[pid] = False
        effs = {
            round(data_by_id[pid].ext_x_um / (cell_by_pid[pid].w_in * 2.54), 6)
            for pid in map_pids
            if cell_by_pid[pid].w_in
        }
        if len(effs) > 1:
            raise StageUserError(
                "shared scale bar needs every map at one µm/cm", hint=_GUTTER_MISMATCH_HINT
            )
        if effs:
            gutter_scale = next(iter(effs))
            # Deliberate deviation from the brief's literal sizing ("xlim
            # spanning gutter_in * 2.54 * shared_scale" reads as: reuse
            # compose.gutter_cm — the between-cell spacing gutter, 0.5 cm by
            # default — directly as the cell's own width). That is too narrow
            # to hold the drawn bar + its "N µm" label at any reasonable font
            # scale; the cell would clip its own content. Use a practical
            # minimum box instead (still derived from gutter_cm so a larger
            # gutter setting still grows it) — the µm-per-cm SPAN drawn inside
            # (`span = gcell.w_in * 2.54 * gutter_scale`, right where the brief
            # calls for it) still comes from this cell's real final width.
            gutter_w_cm = max(recipe.compose.gutter_cm * 4.0, 2.0)
            gutter_h_cm = max(recipe.compose.gutter_cm * 1.2, 0.6)
            gutter_leaf = Spacer(gutter_w_cm, gutter_h_cm)
    # "per-panel": scale_bar_by_pid stays empty -> draw_panel follows the style flag.

    return scale_bar_by_pid, gutter_leaf, gutter_scale


def render_recipe(
    recipe, style_overrides: dict | None = None, *, loader_cache: dict | None = None
) -> ComposeResult:
    validate_recipe(recipe)
    style = (
        style_from_params({"plot_style": {**recipe.style, **(style_overrides or {})}})
        or PlotStyle()
    )

    panels_by_id = dict(recipe.panel_by_id())
    data_by_id = {pid: load_panel(p, cache=loader_cache) for pid, p in panels_by_id.items()}

    notes: list[str] = []
    cells = size_cells(recipe, style, data_by_id, notes)

    fig = Figure(facecolor="white")

    leaves = list(iter_leaves(recipe.layout))
    axes_by_id: dict[str, object] = {}
    cell_by_pid: dict[str, SizedCell] = {}
    for leaf in leaves:
        cell = cells[id(leaf)]
        if isinstance(leaf, PanelRef):
            ax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
            cell.ax = ax
            axes_by_id[leaf.panel_id] = ax
            cell_by_pid[leaf.panel_id] = cell
        elif isinstance(leaf, TextCell):
            ax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
            ax.set_axis_off()
            cell.ax = ax

    n_panels = sum(1 for leaf in leaves if isinstance(leaf, PanelRef))
    n_rendered = sum(
        1
        for leaf in leaves
        if isinstance(leaf, PanelRef) and data_by_id[leaf.panel_id].kind != "placeholder"
    )

    no_colorbar_pids, bar_specs = _apply_shared_colorbars(
        recipe, style, panels_by_id, data_by_id, cells, fig
    )
    bar_map = {id(node): bar_leaf for node, _grp, _pids, bar_leaf, _ax in bar_specs}

    scale_bar_by_pid, gutter_leaf, gutter_scale = _resolve_scale_bar_kwargs(
        recipe, panels_by_id, data_by_id, cell_by_pid
    )
    if gutter_leaf is not None:
        gax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
        gax.set_axis_off()
        cells[id(gutter_leaf)] = SizedCell(
            gutter_leaf,
            None,
            "spacer",
            gutter_leaf.w_cm * _IN_PER_CM,
            gutter_leaf.h_cm * _IN_PER_CM,
            ax=gax,
        )

    working_layout = _build_working_layout(recipe.layout, bar_map)
    if gutter_leaf is not None:
        working_layout = Col([working_layout, gutter_leaf])

    hide_xlabel_pids = _resolve_show_xlabel(recipe, data_by_id)

    # Per-panel scale bars are deliberately NOT drawn here — see the deferred
    # loop after placement below (a pinned_width_cm rescale changes every
    # panel's true effective µm/cm, so baking the bar's thickness from the
    # pre-placement cell size would leave it wrong at final size). We only
    # record here WHETHER a bar is wanted, matching draw_map_layer's own
    # "explicit bool overrides, None follows style.scale_bar" convention.
    scale_bar_wanted: dict[str, bool] = {}
    im_by_pid: dict[str, object] = {}
    for leaf in leaves:
        if not isinstance(leaf, PanelRef):
            continue
        pid = leaf.panel_id
        panel = panels_by_id[pid]
        data = data_by_id[pid]
        cell = cell_by_pid[pid]
        ax = cell.ax

        if cell.kind == "map":
            scale_bar_pref = scale_bar_by_pid.get(pid)
            scale_bar_wanted[pid] = style.scale_bar if scale_bar_pref is None else scale_bar_pref
            cax = None
            if pid in no_colorbar_pids:
                colorbar_kw = False
            else:
                colorbar_kw = None
                if style.colorbar:
                    cax = fig.add_axes([0.0, 0.0, 0.01, 0.01])
                    cell.extras = (cax,)
                    cell.sync = _make_cax_sync(cax, style, cell)
            im = draw_panel(
                ax,
                panel,
                data,
                style,
                cax=cax,
                colorbar=colorbar_kw,
                scale_bar=False,
                fixed_scale_um_per_cm=None,
                show_title=False,
            )
            im_by_pid[pid] = im
        elif cell.kind == "trace":
            show_xlabel = pid not in hide_xlabel_pids
            draw_panel(ax, panel, data, style, show_xlabel=show_xlabel, show_title=False)
        else:  # placeholder
            draw_panel(ax, panel, data, style, show_title=False)

    labels = _assign_labels(recipe.layout, panels_by_id, recipe.compose)
    for node_id, text in labels.items():
        cell = cells.get(node_id)
        if cell is not None and cell.ax is not None:
            _draw_label(cell.ax, text, style, recipe.compose)

    measure_cells(fig, list(cells.values()), pad_in=0.02)
    gutter_in = recipe.compose.gutter_cm * _IN_PER_CM
    pad_in = recipe.compose.padding_cm * _IN_PER_CM
    fig_w, fig_h = place_tree(fig, working_layout, cells, gutter_in=gutter_in, pad_in=pad_in)

    if recipe.compose.pinned_width_cm:
        target_w_in = recipe.compose.pinned_width_cm * _IN_PER_CM
        factor = target_w_in / fig_w
        for pid, cell in cell_by_pid.items():
            if cell.kind in ("map", "trace") and cell.w_in:
                d = data_by_id[pid]
                ext = d.ext_x_um if cell.kind == "map" else d.length_um
                if ext:
                    old_eff = ext / (cell.w_in * 2.54)
                    new_eff = old_eff / factor
                    notes.append(
                        f"panel {pid}: pinned total width — implied scale {new_eff:.4g} µm/cm"
                    )
        for cell in cells.values():
            cell.w_in *= factor
            cell.h_in *= factor
        measure_cells(fig, list(cells.values()), pad_in=0.02)
        fig_w, fig_h = place_tree(fig, working_layout, cells, gutter_in=gutter_in, pad_in=pad_in)
        if abs(fig_w - target_w_in) > 0.02 * target_w_in:
            notes.append(
                f"pinned width {recipe.compose.pinned_width_cm:.4g} cm missed — "
                f"rendered {fig_w * 2.54:.4g} cm"
            )

    # Per-panel scale bars, drawn NOW (final geometry) — see the note at the
    # draw loop above for why this can't happen before placement.
    for pid, want in scale_bar_wanted.items():
        if not want:
            continue
        cell = cell_by_pid[pid]
        if not cell.w_in:
            continue
        d = data_by_id[pid]
        final_eff = d.ext_x_um / (cell.w_in * 2.54)
        draw_scale_bar(
            axes_by_id[pid], style.scale_bar_length_um, style=style, fixed_scale_um_per_cm=final_eff
        )

    for node, grp, pids, _bar_leaf, bar_ax in bar_specs:
        # Stretch the bar to the group's REAL placed span before drawing its
        # content — the provisional box from `_apply_shared_colorbars` only
        # reserved solver space (see the comment there).
        _stretch_shared_bar(node, pids, bar_ax, axes_by_id, data_by_id)
        rep_pid = next((pid for pid in pids if im_by_pid.get(pid) is not None), None)
        if rep_pid is not None:
            add_colorbar(
                fig,
                im_by_pid[rep_pid],
                axes_by_id[rep_pid],
                _cbar_label(data_by_id[rep_pid]),
                style,
                group=grp,
                cax=bar_ax,
            )

    if gutter_leaf is not None and gutter_scale is not None:
        # Recompute the shared scale FRESH from final cell sizes (a
        # pinned_width_cm rescale after `_resolve_scale_bar_kwargs` ran would
        # otherwise leave `gutter_scale` stale, same issue as the per-panel
        # bars above); all map panels were rescaled by the same factor, so the
        # common value is preserved, just recomputed at the final size.
        map_pids_final = [
            pid
            for pid, d in data_by_id.items()
            if d.kind in ("map_layer", "slice_plane", "profiles_ref") and pid in cell_by_pid
        ]
        effs_final = {
            round(data_by_id[pid].ext_x_um / (cell_by_pid[pid].w_in * 2.54), 6)
            for pid in map_pids_final
            if cell_by_pid[pid].w_in
        }
        gutter_scale = (
            next(iter(effs_final), gutter_scale) if len(effs_final) == 1 else gutter_scale
        )
        gcell = cells[id(gutter_leaf)]
        gax = gcell.ax
        span = gcell.w_in * 2.54 * gutter_scale
        gax.set_xlim(0, span if span > 0 else 1.0)
        gax.set_ylim(0, 1)
        # Force a sensible, non-clipping placement in the narrow dedicated
        # gutter cell — the brief calls for this explicitly ("loc forced
        # sensible via the style"); an arbitrary corner loc from the user's
        # style (meant for full-size map panels) can clip in this small box.
        gutter_style = dc_replace(style, scale_bar_loc="center", scale_bar_inset_pt=0.0)
        draw_scale_bar(gax, None, style=gutter_style, fixed_scale_um_per_cm=gutter_scale)

    for leaf in leaves:
        if isinstance(leaf, TextCell):
            ax = cells[id(leaf)].ax
            ax.text(
                0.5,
                0.5,
                leaf.text,
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=12.0 * style.font_scale,
                wrap=True,
            )

    for pid, ax in axes_by_id.items():
        cell = cell_by_pid[pid]
        note = box_drift_note(f"panel {pid}", fig, ax, cell.w_in, cell.h_in)
        if note:
            notes.append(note)

    return ComposeResult(
        figure=fig, notes=notes, n_panels=n_panels, n_rendered=n_rendered, axes_by_id=axes_by_id
    )


def export_recipe(
    recipe,
    out_dir,
    *,
    formats=None,
    dpi=None,
    style_overrides: dict | None = None,
    loader_cache: dict | None = None,
):
    res = render_recipe(recipe, style_overrides, loader_cache=loader_cache)
    style = style_from_params({"plot_style": dict(recipe.style)}) or PlotStyle()
    fmts = tuple(formats) if formats else tuple(style.formats)
    the_dpi = int(dpi) if dpi else int(style.dpi)
    os.makedirs(out_dir, exist_ok=True)
    stem = re.sub(r"[^\w.-]+", "_", recipe.name or "figure").strip("_") or "figure"
    paths = []
    for fmt in fmts:
        path = os.path.join(out_dir, f"{stem}.{fmt}")
        res.figure.savefig(path, dpi=the_dpi, facecolor="white")  # NO bbox_inches
        paths.append(path)
    return paths, res
