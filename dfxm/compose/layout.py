"""Deterministic box-tree layout for composed figures (Qt-free).

Sizing (this half): every layout leaf resolves to an exact (w_in, h_in)
content box from physical scales BEFORE any drawing. Placement (second half)
measures decorations at final box size and places every axes absolutely —
no matplotlib auto-layout anywhere (generalizes place_axes_stack)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from dataclasses import replace as dc_replace

from ..common.errors import StageUserError
from ..common.plotting import (
    fixed_scale,
    fixed_scale_box,
    trace_fixed_box,
    trace_fixed_scale,
    trace_height_cm,
)
from .recipe import Col, PanelRef, Row, Spacer, TextCell

_IN_PER_CM = 1.0 / 2.54
PLACEHOLDER_CM = (4.0, 3.0)
_NO_SCALE_HINT = (
    "Set Scale (µm/cm) in the style, a per-panel scale override, or pin the "
    "row height / column width so the composer can size this panel."
)
_BAD_SCALE_HINT = "Fix the per-panel scale override — it must be a positive, finite number."


def _finite_positive(v) -> bool:
    """True for a real (non-``None``, non-NaN/inf) positive number."""
    return v is not None and math.isfinite(v) and v > 0


def _validate_scale(value, panel_id: str, what: str) -> float:
    """Validate a per-panel scale override: float-castable, finite, and > 0.

    Raises :class:`StageUserError` (never a bare ``ValueError``) for a
    non-numeric value (e.g. a hand-edited recipe JSON — ``recipe.py`` reads
    ``scale_um_per_cm`` uncast) or a numeric-but-non-finite/non-positive one
    (negative, zero, NaN, inf) — this is a recipe-authoring bug, not a
    data-availability issue, so it must never silently produce a
    negative/garbage box or a silent fallback to a different scale.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise StageUserError(
            f"panel {panel_id}: {what} override {value!r} is not a number",
            hint=_BAD_SCALE_HINT,
        ) from None
    if not (math.isfinite(v) and v > 0):
        raise StageUserError(
            f"panel {panel_id}: {what} override {v!r} must be a positive number",
            hint=_BAD_SCALE_HINT,
        )
    return v


@dataclass
class SizedCell:
    leaf: object  # the layout leaf (PanelRef/Spacer/TextCell)
    panel: object | None  # PanelDef, or None for Spacer/TextCell
    kind: str  # "map"|"trace"|"spacer"|"text"|"placeholder"
    w_in: float
    h_in: float
    # filled by the placement pass (Task 6):
    ax: object | None = None
    extras: tuple = ()
    sync: object | None = None
    margins: object | None = None
    label: str | None = None


def size_cells(recipe, style, data_by_id, notes):
    """Walk *recipe*'s layout tree and resolve every leaf to an exact box.

    Returns ``{id(leaf): SizedCell}``. Appends implied-scale / clamp /
    placeholder notes to *notes* (mutated in place). Raises
    :class:`StageUserError` when a physical (map/trace) panel has no scale to
    size from and no pinned row-height / column-width covers it.
    """
    panels = recipe.panel_by_id()
    cells: dict[int, SizedCell] = {}

    def leaf_cell(leaf, pinned_h_in, pinned_w_in):
        if isinstance(leaf, Spacer):
            return SizedCell(leaf, None, "spacer", leaf.w_cm * _IN_PER_CM, leaf.h_cm * _IN_PER_CM)
        if isinstance(leaf, TextCell):
            return SizedCell(leaf, None, "text", leaf.w_cm * _IN_PER_CM, leaf.h_cm * _IN_PER_CM)
        panel = panels[leaf.panel_id]
        data = data_by_id[panel.id]
        if data.kind == "placeholder":
            notes.append(f"panel {panel.id}: {data.payload['reason']} — rendered as placeholder")
            return SizedCell(
                leaf,
                panel,
                "placeholder",
                PLACEHOLDER_CM[0] * _IN_PER_CM,
                PLACEHOLDER_CM[1] * _IN_PER_CM,
            )
        if data.kind == "profiles_trace":
            return _trace_cell(leaf, panel, data, pinned_h_in, pinned_w_in)
        return _map_cell(leaf, panel, data, pinned_h_in, pinned_w_in)

    def _map_cell(leaf, panel, data, pinned_h_in, pinned_w_in):
        ext_x, ext_y = data.ext_x_um, data.ext_y_um
        if not (_finite_positive(ext_x) and _finite_positive(ext_y)):
            notes.append(f"panel {panel.id}: degenerate extent — rendered as placeholder")
            return SizedCell(
                leaf,
                panel,
                "placeholder",
                PLACEHOLDER_CM[0] * _IN_PER_CM,
                PLACEHOLDER_CM[1] * _IN_PER_CM,
            )
        # Pins are checked BEFORE the scale is resolved: a pinned row-height or
        # column-width sizes purely from (ext_x, ext_y) and needs no scale at
        # all, so an unused/irrelevant bad override must not raise here.
        if pinned_h_in is not None:
            h = pinned_h_in
            w = h * ext_x / ext_y
            implied = ext_y / (h / _IN_PER_CM)
            notes.append(f"panel {panel.id}: pinned row height — implied scale {implied:.4g} µm/cm")
            return SizedCell(leaf, panel, "map", w, h)
        if pinned_w_in is not None:
            w = pinned_w_in
            h = w * ext_y / ext_x
            implied = ext_x / (w / _IN_PER_CM)
            notes.append(
                f"panel {panel.id}: pinned column width — implied scale {implied:.4g} µm/cm"
            )
            return SizedCell(leaf, panel, "map", w, h)
        override = panel.scale_um_per_cm
        eff = _validate_scale(override, panel.id, "scale") if override else fixed_scale(style)
        if eff is None:
            raise StageUserError(
                f"panel {panel.id} has no physical scale to size from", hint=_NO_SCALE_HINT
            )
        # eff and (ext_x, ext_y) are now both known finite and positive, so
        # fixed_scale_box cannot return None here.
        box = fixed_scale_box(style, ext_x, ext_y, scale=eff)
        if box[2] != eff:
            notes.append(
                f"panel {panel.id}: box clamped to 30 in — effective scale {box[2]:.4g} µm/cm"
            )
        return SizedCell(leaf, panel, "map", box[0], box[1])

    def _trace_cell(leaf, panel, data, pinned_h_in, pinned_w_in):
        length = data.length_um or 0.0
        # A pinned column width sizes purely from (length, trace_height_cm)
        # and needs no scale at all — checked first so an unused/irrelevant
        # bad override cannot raise here (mirrors the map-cell ordering).
        if pinned_w_in is not None:
            w = pinned_w_in
            h = trace_height_cm(style) * _IN_PER_CM
            implied = length / (w / _IN_PER_CM) if w > 0 else 0.0
            notes.append(
                f"panel {panel.id}: pinned column width — implied trace scale {implied:.4g} µm/cm"
            )
            return SizedCell(leaf, panel, "trace", w, h)
        st = style
        if panel.scale_um_per_cm:
            st = dc_replace(
                style,
                trace_scale_um_per_cm=_validate_scale(
                    panel.scale_um_per_cm, panel.id, "trace scale"
                ),
            )
        box = trace_fixed_box(st, float(length))
        if box is None:
            raise StageUserError(
                f"trace panel {panel.id} has no trace scale to size from", hint=_NO_SCALE_HINT
            )
        if box[2] != trace_fixed_scale(st):
            notes.append(
                f"panel {panel.id}: trace box clamped to 30 in — effective scale {box[2]:.4g} µm/cm"
            )
        # Unlike a map, a trace's height is purely cosmetic (trace_height_cm),
        # not derived from any physical extent — so a pinned ROW height can
        # only override the height field; the width still needs a real scale
        # (already resolved above) and is never rescaled by the height pin.
        if pinned_h_in is not None:
            notes.append(
                f"panel {panel.id}: pinned row height — implied trace scale {box[2]:.4g} µm/cm"
            )
            return SizedCell(leaf, panel, "trace", box[0], pinned_h_in)
        return SizedCell(leaf, panel, "trace", box[0], box[1])

    def walk(node, pinned_h_in, pinned_w_in):
        if isinstance(node, Row):
            ph = node.pinned_height_cm * _IN_PER_CM if node.pinned_height_cm else pinned_h_in
            for child in node.children:
                walk(child, ph, pinned_w_in)
        elif isinstance(node, Col):
            pw = node.pinned_width_cm * _IN_PER_CM if node.pinned_width_cm else pinned_w_in
            for child in node.children:
                walk(child, pinned_h_in, pw)
        else:
            cells[id(node)] = leaf_cell(node, pinned_h_in, pinned_w_in)

    walk(recipe.layout, None, None)
    return cells


def measure_cells(fig, cells, pad_in: float = 0.02) -> None:
    """Fill each cell's margins, measured at FINAL box size (mandatory: tick
    density depends on size — same rule as place_axes_box)."""
    from ..common.plotting import AxesMargins, measure_axes_margins

    fig.set_layout_engine("none")
    live = [c for c in cells if c.ax is not None]
    prov_w = max((c.w_in for c in live), default=1.0) + 4.0
    prov_h = max((c.h_in for c in live), default=1.0) + 4.0
    fig.set_size_inches(prov_w, prov_h, forward=False)
    for c in cells:
        if c.ax is None:
            c.margins = AxesMargins(0.0, 0.0, 0.0, 0.0)
            continue
        c.ax.set_position([2.0 / prov_w, 2.0 / prov_h, c.w_in / prov_w, c.h_in / prov_h])
        if c.sync is not None:
            c.sync(fig, c.ax)
    for c in live:
        c.margins = measure_axes_margins(fig, c.ax, extras=c.extras, pad_in=pad_in)


def _cell_env(c):
    m = c.margins
    return (m.left + c.w_in + m.right, m.bottom + c.h_in + m.top)


def place_tree(fig, layout, cells, *, gutter_in, pad_in):
    """Share max margins within direct-panel Rows/Cols, compute envelope sizes
    (composite children align by envelope with automatic trailing padding),
    size *fig* to the root envelope + padding, and absolute-place every axes.

    Returns ``(fig_w_in, fig_h_in)``.
    """
    from ..common.plotting import AxesMargins

    env: dict[int, tuple[float, float]] = {}

    def _share_row(node):
        leaf_cells = [
            cells[id(ch)]
            for ch in node.children
            if isinstance(ch, PanelRef) and cells[id(ch)].ax is not None
        ]
        if len(leaf_cells) > 1:
            top = max(c.margins.top for c in leaf_cells)
            bottom = max(c.margins.bottom for c in leaf_cells)
            for c in leaf_cells:
                c.margins = AxesMargins(c.margins.left, c.margins.right, top, bottom)

    def _share_col(node):
        leaf_cells = [
            cells[id(ch)]
            for ch in node.children
            if isinstance(ch, PanelRef) and cells[id(ch)].ax is not None
        ]
        if len(leaf_cells) > 1:
            left = max(c.margins.left for c in leaf_cells)
            right = max(c.margins.right for c in leaf_cells)
            for c in leaf_cells:
                c.margins = AxesMargins(left, right, c.margins.top, c.margins.bottom)

    def _envelope(node):
        if isinstance(node, Row):
            _share_row(node)
            child_envs = [_envelope(c) for c in node.children]
            w = sum(e[0] for e in child_envs) + gutter_in * max(0, len(child_envs) - 1)
            h = max(e[1] for e in child_envs)
            env[id(node)] = (w, h)
            return (w, h)
        if isinstance(node, Col):
            _share_col(node)
            child_envs = [_envelope(c) for c in node.children]
            w = max(e[0] for e in child_envs)
            h = sum(e[1] for e in child_envs) + gutter_in * max(0, len(child_envs) - 1)
            env[id(node)] = (w, h)
            return (w, h)
        e = _cell_env(cells[id(node)])
        env[id(node)] = e
        return e

    root_w, root_h = _envelope(layout)
    fig_w, fig_h = root_w + 2 * pad_in, root_h + 2 * pad_in
    fig.set_size_inches(fig_w, fig_h, forward=False)

    def _place(node, x, y):
        # (x, y) = this node's envelope top-left, inches from figure top-left
        if isinstance(node, Row):
            cx = x
            for child in node.children:
                _place(child, cx, y)  # top-aligned; trailing pad below shorter kids
                cx += env[id(child)][0] + gutter_in
            return
        if isinstance(node, Col):
            cy = y
            for child in node.children:
                _place(child, x, cy)  # left-aligned; trailing pad right of narrower kids
                cy += env[id(child)][1] + gutter_in
            return
        c = cells[id(node)]
        if c.ax is None:
            return
        m = c.margins
        x0 = (x + m.left) / fig_w
        y0 = (fig_h - y - m.top - c.h_in) / fig_h
        c.ax.set_position([x0, y0, c.w_in / fig_w, c.h_in / fig_h])
        if c.sync is not None:
            c.sync(fig, c.ax)

    _place(layout, pad_in, pad_in)
    return (fig_w, fig_h)
