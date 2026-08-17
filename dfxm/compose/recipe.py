"""Figure-recipe data model + JSON (de)serialization + validation (Qt-free)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

from ..common.errors import StageUserError

RECIPE_VERSION = 1
PANEL_KINDS = ("map_layer", "slice_plane", "profiles_ref", "profiles_trace")
SCALE_BAR_MODES = ("per-panel", "one-panel", "gutter")
COLORBAR_MODES = ("per-panel", "united")
COLORBAR_POSITIONS = ("right", "bottom")


@dataclass
class ComposeStyle:
    label_template: str = "A"
    label_font_scale: float = 1.0
    gutter_cm: float = 0.5
    padding_cm: float = 0.3
    scale_bar_mode: str = "per-panel"
    scale_bar_panel: str | None = None
    pinned_width_cm: float | None = None
    colorbar_mode: str = "per-panel"  # one of COLORBAR_MODES
    colorbar_pos: str = "right"  # one of COLORBAR_POSITIONS (united mode only)


@dataclass
class PanelSource:
    h5_path: str
    kind: str  # one of PANEL_KINDS
    selector: dict  # kind-specific selection key (see Task 4)


@dataclass
class PanelDef:
    id: str
    source: PanelSource
    roi: tuple[int, int, int, int] | None = None  # (r0, r1, c0, c1) px, replot convention
    clim: tuple[float | None, float | None] | None = None
    cmap: str | None = None
    label: str | None = None  # None = auto sequence; "" = no label; text = manual
    show_title: bool | None = None  # None = composed default (off)
    scale_um_per_cm: float | None = None
    colorbar: bool | None = None  # None = follow style; False when a shared bar covers it
    title: str | None = None  # human-readable data name (display only); None = show the id


@dataclass
class PanelRef:
    panel_id: str


@dataclass
class Spacer:
    w_cm: float
    h_cm: float


@dataclass
class TextCell:
    text: str
    w_cm: float = 2.0
    h_cm: float = 1.0


@dataclass
class Row:
    children: list
    pinned_height_cm: float | None = None
    # None = not a group; "auto" = auto-lettered slot; text = manual label.
    # "" is normalized to None on load — a blank GROUP label means "not a
    # group", unlike PanelDef.label where "" means "suppress the label".
    group_label: str | None = None
    shared_colorbar: bool = False
    shared_clim: tuple[float, float] | None = None


@dataclass
class Col:
    children: list
    pinned_width_cm: float | None = None
    # None = not a group; "auto" = auto-lettered slot; text = manual label.
    # "" is normalized to None on load — a blank GROUP label means "not a
    # group", unlike PanelDef.label where "" means "suppress the label".
    group_label: str | None = None
    shared_x: bool = False
    shared_colorbar: bool = False
    shared_clim: tuple[float, float] | None = None


@dataclass
class FigureRecipe:
    name: str
    style: dict  # PlotStyle field overrides, JSON-safe
    compose: ComposeStyle
    layout: Row | Col | PanelRef | Spacer | TextCell
    panels: list[PanelDef]
    version: int = RECIPE_VERSION

    def panel_by_id(self) -> dict[str, PanelDef]:
        return {p.id: p for p in self.panels}


def iter_leaves(node):
    """Yield layout leaves (PanelRef/Spacer/TextCell) in depth-first order."""
    if isinstance(node, (Row, Col)):
        for child in node.children:
            yield from iter_leaves(child)
    else:
        yield node


def _node_to_dict(node, rel):
    if isinstance(node, Row):
        d = {"type": "row", "children": [_node_to_dict(c, rel) for c in node.children]}
        for k in ("pinned_height_cm", "group_label", "shared_clim"):
            if getattr(node, k) is not None:
                d[k] = getattr(node, k)
        if node.shared_colorbar:
            d["shared_colorbar"] = True
        return d
    if isinstance(node, Col):
        d = {"type": "col", "children": [_node_to_dict(c, rel) for c in node.children]}
        for k in ("pinned_width_cm", "group_label", "shared_clim"):
            if getattr(node, k) is not None:
                d[k] = getattr(node, k)
        if node.shared_colorbar:
            d["shared_colorbar"] = True
        if node.shared_x:
            d["shared_x"] = True
        return d
    if isinstance(node, PanelRef):
        return {"type": "panel", "panel_id": node.panel_id}
    if isinstance(node, Spacer):
        return {"type": "spacer", "w_cm": node.w_cm, "h_cm": node.h_cm}
    if isinstance(node, TextCell):
        return {"type": "text", "text": node.text, "w_cm": node.w_cm, "h_cm": node.h_cm}
    raise StageUserError(
        f"unknown layout node {type(node).__name__!r}",
        hint="Layout nodes must be Row/Col/PanelRef/Spacer/TextCell.",
    )


def _node_from_dict(d):
    t = d.get("type")
    if t == "row":
        return Row(
            [_node_from_dict(c) for c in d.get("children", [])],
            pinned_height_cm=d.get("pinned_height_cm"),
            group_label=d.get("group_label") or None,
            shared_colorbar=bool(d.get("shared_colorbar", False)),
            shared_clim=tuple(d["shared_clim"]) if d.get("shared_clim") else None,
        )
    if t == "col":
        return Col(
            [_node_from_dict(c) for c in d.get("children", [])],
            pinned_width_cm=d.get("pinned_width_cm"),
            group_label=d.get("group_label") or None,
            shared_x=bool(d.get("shared_x", False)),
            shared_colorbar=bool(d.get("shared_colorbar", False)),
            shared_clim=tuple(d["shared_clim"]) if d.get("shared_clim") else None,
        )
    if t == "panel":
        return PanelRef(d["panel_id"])
    if t == "spacer":
        return Spacer(float(d["w_cm"]), float(d["h_cm"]))
    if t == "text":
        return TextCell(d["text"], float(d.get("w_cm", 2.0)), float(d.get("h_cm", 1.0)))
    raise StageUserError(
        f"unknown layout node type {t!r}",
        hint="Valid node types: row, col, panel, spacer, text.",
    )


def _rel_path(path, base_dir):
    """Store ``path`` relative to ``base_dir`` when that succeeds (same drive)."""
    if base_dir is None:
        return path
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:
        return path


def _resolve_path(path, base_dir):
    """Resolve a stored (possibly relative) path against ``base_dir``."""
    if base_dir is not None and not os.path.isabs(path):
        return os.path.join(base_dir, path)
    return path


def _panel_source_to_dict(src, rel):
    return {"h5_path": rel(src.h5_path), "kind": src.kind, "selector": src.selector}


def _panel_source_from_dict(d, base_dir):
    return PanelSource(
        h5_path=_resolve_path(d["h5_path"], base_dir),
        kind=d["kind"],
        selector=d.get("selector", {}),
    )


def _panel_def_to_dict(p, rel):
    return {
        "id": p.id,
        "source": _panel_source_to_dict(p.source, rel),
        "roi": list(p.roi) if p.roi is not None else None,
        "clim": list(p.clim) if p.clim is not None else None,
        "cmap": p.cmap,
        "label": p.label,
        "show_title": p.show_title,
        "scale_um_per_cm": p.scale_um_per_cm,
        "colorbar": p.colorbar,
        "title": p.title,
    }


def _panel_def_from_dict(d, base_dir):
    roi = d.get("roi")
    clim = d.get("clim")
    return PanelDef(
        id=d["id"],
        source=_panel_source_from_dict(d["source"], base_dir),
        roi=tuple(roi) if roi is not None else None,
        clim=tuple(clim) if clim is not None else None,
        cmap=d.get("cmap"),
        label=d.get("label"),
        show_title=d.get("show_title"),
        scale_um_per_cm=d.get("scale_um_per_cm"),
        colorbar=d.get("colorbar"),
        title=d.get("title"),
    )


def recipe_to_json(recipe: FigureRecipe, *, base_dir: str | None = None) -> str:
    """Serialize ``recipe`` to a JSON string.

    When ``base_dir`` is given, each panel's ``h5_path`` is stored relative to
    it (falling back to the absolute path if ``os.path.relpath`` can't
    compute one, e.g. a different drive on Windows).
    """

    def rel(path):
        return _rel_path(path, base_dir)

    d = {
        "name": recipe.name,
        "style": recipe.style,
        "compose": asdict(recipe.compose),
        "layout": _node_to_dict(recipe.layout, rel),
        "panels": [_panel_def_to_dict(p, rel) for p in recipe.panels],
        "version": recipe.version,
    }
    return json.dumps(d, indent=2)


def recipe_from_json(text: str, *, base_dir: str | None = None) -> FigureRecipe:
    """Parse a recipe previously written by :func:`recipe_to_json`.

    Raises :class:`StageUserError` (with a hint) for invalid JSON, JSON that
    is not a figure recipe at all, an unsupported/missing ``version``, or a
    structurally malformed v1 recipe (e.g. a hand-edited file missing a
    required key or carrying an unknown one) — the latter wraps the
    underlying ``TypeError``/``KeyError`` instead of letting it escape.
    """
    try:
        d = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StageUserError(
            "recipe is not valid JSON — the file may be corrupted or truncated",
            hint="Re-export the recipe from the figure builder, or restore it from a backup.",
        ) from exc

    if not isinstance(d, dict) or not any(k in d for k in ("version", "layout", "panels")):
        raise StageUserError(
            "this JSON file is not a figure recipe",
            hint="Pick a recipe .json saved by the figure builder "
            "(it has version/layout/panels keys).",
        )

    version = d.get("version")
    if version != RECIPE_VERSION:
        raise StageUserError(
            f"unsupported recipe version {version!r}",
            hint="This app writes version 1 recipes — re-save the recipe with this version of the app.",
        )

    if "layout" not in d or "panels" not in d:
        raise StageUserError(
            "recipe is missing 'layout' or 'panels'",
            hint="This does not look like a figure-recipe file.",
        )

    try:
        compose = ComposeStyle(**d.get("compose", {}))
        panels = [_panel_def_from_dict(pd, base_dir) for pd in d["panels"]]
        layout = _node_from_dict(d["layout"])
    except StageUserError:
        raise
    except (TypeError, KeyError) as exc:
        raise StageUserError(
            f"recipe is malformed ({type(exc).__name__}: {exc})",
            hint="A hand-edited or corrupted field is the likely cause — "
            "re-save the recipe from the figure builder.",
        ) from exc

    return FigureRecipe(
        name=d.get("name", ""),
        style=d.get("style", {}),
        compose=compose,
        layout=layout,
        panels=panels,
        version=version,
    )


def validate_recipe(recipe: FigureRecipe) -> None:
    """Sanity-check a recipe, raising :class:`StageUserError` on the first problem found."""
    seen: set[str] = set()
    for p in recipe.panels:
        if p.id in seen:
            raise StageUserError(
                f"duplicate panel id {p.id!r}",
                hint="Every panel in a recipe needs a unique id.",
            )
        seen.add(p.id)

    by_id = recipe.panel_by_id()
    ref_counts: dict[str, int] = {}
    for leaf in iter_leaves(recipe.layout):
        if isinstance(leaf, PanelRef):
            if leaf.panel_id not in by_id:
                raise StageUserError(
                    f"layout refers to unknown panel id {leaf.panel_id!r} (ghost reference)",
                    hint="Every panel referenced by the layout must exist in the recipe's panel list.",
                )
            ref_counts[leaf.panel_id] = ref_counts.get(leaf.panel_id, 0) + 1
    dupes = sorted(pid for pid, n in ref_counts.items() if n > 1)
    if dupes:
        raise StageUserError(
            f"layout references panel(s) {', '.join(repr(p) for p in dupes)} more than once",
            hint="Each panel can appear in the layout once — add a second PanelDef "
            "(new id, same source) to show the same data twice.",
        )

    for p in recipe.panels:
        if p.source.kind not in PANEL_KINDS:
            raise StageUserError(
                f"panel {p.id!r} has unknown source kind {p.source.kind!r}",
                hint=f"source.kind must be one of {PANEL_KINDS}.",
            )

    if recipe.compose.scale_bar_mode not in SCALE_BAR_MODES:
        raise StageUserError(
            f"invalid compose.scale_bar_mode {recipe.compose.scale_bar_mode!r}",
            hint=f"scale_bar_mode must be one of {SCALE_BAR_MODES}.",
        )

    if recipe.compose.colorbar_mode not in COLORBAR_MODES:
        raise StageUserError(
            f"invalid compose.colorbar_mode {recipe.compose.colorbar_mode!r}",
            hint=f"colorbar_mode must be one of {COLORBAR_MODES}.",
        )
    if recipe.compose.colorbar_pos not in COLORBAR_POSITIONS:
        raise StageUserError(
            f"invalid compose.colorbar_pos {recipe.compose.colorbar_pos!r}",
            hint=f"colorbar_pos must be one of {COLORBAR_POSITIONS}.",
        )

    label_template = recipe.compose.label_template
    if not any(c in ("A", "a") for c in label_template):
        raise StageUserError(
            f"compose.label_template {label_template!r} has no placeholder letter",
            hint="label_template must contain at least one 'A' or 'a' marking where the label goes.",
        )

    if recipe.compose.gutter_cm <= 0:
        raise StageUserError(
            f"compose.gutter_cm must be positive, got {recipe.compose.gutter_cm!r}",
            hint="Set gutter_cm to a positive number of centimetres.",
        )
    if recipe.compose.padding_cm <= 0:
        raise StageUserError(
            f"compose.padding_cm must be positive, got {recipe.compose.padding_cm!r}",
            hint="Set padding_cm to a positive number of centimetres.",
        )
