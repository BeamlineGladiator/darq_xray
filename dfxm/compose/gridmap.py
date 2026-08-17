"""Grid <-> layout-tree mapping for the drag arranger (Qt-free).

GridModel = list of columns (left->right), each a list of panel ids
(top->bottom). Only "plain grid" layouts map cleanly: a root Row of PanelRefs
and/or single-level Cols of PanelRefs. A Col's group/shared flags are allowed
only where the Col survives the rebuild (>= 2 members) — the caller
(ArrangeDialog) re-applies them to rebuilt Cols by member-id set. Everything
else — spacers, text cells, nested containers, ghost refs, flags the rebuilt
grid cannot represent — makes ``layout_to_grid`` return ``None``; callers then
offer the flatten-with-warning path seeded from :func:`flatten_panel_ids`.

Round-trip law (tested): ``layout_to_grid(grid_to_layout(g), panels) == g``
for any normalized grid (no empty columns, every id known)."""

from __future__ import annotations

from .recipe import Col, PanelRef, Row, iter_leaves

GridModel = list  # list[list[str]] — columns of panel ids


def _row_has_flags(row: Row) -> bool:
    return (
        row.group_label is not None
        or row.shared_colorbar
        or row.shared_clim is not None
        or row.pinned_height_cm is not None
    )


def _col_has_flags(col: Col) -> bool:
    return (
        col.group_label is not None
        or col.shared_x
        or col.shared_colorbar
        or col.shared_clim is not None
        or col.pinned_width_cm is not None
    )


def _col_ids(col: Col, panels_by_id) -> list[str] | None:
    ids: list[str] = []
    for child in col.children:
        if not isinstance(child, PanelRef) or child.panel_id not in panels_by_id:
            return None
        ids.append(child.panel_id)
    if _col_has_flags(col) and len(ids) < 2:
        return None  # the rebuild would silently drop this Col (and its flags)
    return ids


def layout_to_grid(layout, panels_by_id) -> GridModel | None:
    """Map *layout* to columns of panel ids, or ``None`` when it isn't a plain grid."""
    if isinstance(layout, PanelRef):
        return [[layout.panel_id]] if layout.panel_id in panels_by_id else None
    if isinstance(layout, Col):
        ids = _col_ids(layout, panels_by_id)
        return None if ids is None else ([ids] if ids else [])
    if not isinstance(layout, Row) or _row_has_flags(layout):
        return None
    grid: GridModel = []
    for child in layout.children:
        if isinstance(child, PanelRef):
            if child.panel_id not in panels_by_id:
                return None
            grid.append([child.panel_id])
        elif isinstance(child, Col):
            ids = _col_ids(child, panels_by_id)
            if ids is None:
                return None
            if ids:
                grid.append(ids)
        else:
            return None
    return grid


def flatten_panel_ids(layout) -> list[str]:
    """Every PanelRef id under *layout*, depth-first — the flatten-path seed."""
    return [leaf.panel_id for leaf in iter_leaves(layout) if isinstance(leaf, PanelRef)]


def grid_to_layout(grid) -> Row:
    """Build ``Row([...])`` from *grid*: a ``Col([PanelRef...])`` per multi-tile
    column, a bare ``PanelRef`` per single-tile column; empty columns dropped;
    an empty grid yields ``Row([])``."""
    children = []
    for column in grid:
        if not column:
            continue
        if len(column) == 1:
            children.append(PanelRef(column[0]))
        else:
            children.append(Col([PanelRef(pid) for pid in column]))
    return Row(children)


def panel_group_hint(panel) -> str | None:
    """Cheap, file-free quantity-group guess for schematic tile chips ONLY —
    rendering always uses the loaded ``PanelData.group``, never this."""
    src = panel.source
    if src.kind == "profiles_trace":
        return "trace"
    if src.kind == "map_layer":
        stage = src.selector.get("stage")
        if stage == "strain":
            return "strain"
        if stage == "rocking":
            return "raw"
        if stage == "mosaicity":
            return "mosa_fwhm" if "FWHM" in str(src.selector.get("dataset", "")) else "mosa_com"
        return None
    key = "volume_id" if src.kind == "slice_plane" else "field"
    v = str(src.selector.get(key) or "").lower()
    if "raw" in v:
        return "raw"
    if "fwhm" in v:
        return "mosa_fwhm"
    if "strain" in v:
        return "strain"
    if "mosa" in v or "com" in v or "chi" in v or "mu" in v:
        return "mosa_com"
    return None
