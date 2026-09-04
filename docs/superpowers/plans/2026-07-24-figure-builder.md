# Publication figure builder (grid-based composer) — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL — use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task by task, with a reviewer gate after every task.

**Spec:** `docs/superpowers/specs/2026-07-24-figure-builder-design.md` (approved 2026-07-24)
**Repo:** `darq_xray`

## Goal

A composer that renders publication figures — panels from multiple stages and files placed in one matplotlib Figure at exact physical scale (µm/cm), with global style applied once, template subplot labels, shared colorbars/scale bars, margin compensation, JSON recipes, and PNG/PDF/SVG export at exact physical size. Phase A: the Qt-free `dfxm/compose/` core proven headless by the spec's two acceptance figures. Phase B: a dedicated non-modal GUI window (`gui/figure_builder.py`) on top of the proven core.

## Architecture

A new Qt-free package `dfxm/compose/` (recipe schema → panel adapters → deterministic box-tree solver → single-Figure render/export) generalizes the existing `place_axes_box`/`place_axes_stack` engine: intrinsic cm boxes from physical scales, one measure pass at final box size, shared row/col margins, absolute `ax.set_position` placement. Panel content comes from `draw_<kind>(ax, …)` functions extracted from the existing single-figure builders (which re-call them, pinned by regression tests). The GUI window is a thin shell: outline-tree recipe editing, cached debounced preview, style/compose knobs, notes bar.

## Tech stack

Python 3.10, numpy, h5py, matplotlib (explicit `Figure` API + Agg), PySide6 (Phase B only), pytest. No new dependencies.

## Global constraints

- **Qt-free `dfxm/compose`** — never import PySide6/pyvista anywhere under `dfxm/`.
- **Figure API only** — no `pyplot`, no `matplotlib.use(...)`; Agg via `FigureCanvasAgg` when a renderer is needed (use the existing `_ensure_agg` pattern).
- **No tight-crop on composed export** — `fig.savefig(..., facecolor="white")` without `bbox_inches`; the solver owns all margins.
- **No matplotlib auto-layout in the solver** — `fig.set_layout_engine("none")`; never constrained/tight layout on the composed figure.
- **Measure-at-final-size rule** — decorations are measured with each axes provisionally placed at its final box size (tick density depends on size), exactly like `place_axes_box`.
- **Split scales** — maps size from `scale_um_per_cm`, traces from `trace_scale_um_per_cm` (fallback map scale) × `trace_height_cm`, same rules as `trace_fixed_box` including the 30-in clamps; the composer never guesses a scale (`StageUserError` when nothing pins a size).
- **Error contract** — refused configs (mixed-quantity shared bar, no scale anywhere, bad schema/version) → `StageUserError(message, hint)`; missing h5/dataset at render time → hatched placeholder panel + note (never a crash); CLI exits non-zero only when *no* panel rendered; degenerate extents degrade per panel with a note (companion precedent).
- **Drift guard** — `measured_box_in` + `box_drift_note` (0.5 % tol) on every placed panel; notes surface in CLI output and the GUI notes bar.
- **Ruff** — line length 100, double quotes, py310, rules E/F/I; `ruff format` runs on Write/Edit via hook.
- **Docs same-change contract** — any task changing stage/GUI behaviour or public structure updates `docs/Usage.md` and/or `docs/Codebase.md` in the same task.
- **TDD** — every task: failing test first (extraction tasks: characterization test that passes *before* the refactor and must still pass after), then minimal implementation, then commit.
- **This repo has no git remote** — commit locally only; branch for the feature (e.g. `git checkout -b figure-builder` before Task 1).

---

## Phase A — core (`dfxm/compose/`)

### Task 1 — Recipe schema + JSON round-trip (`dfxm/compose/recipe.py`)

**Files**
- Create: `dfxm/compose/__init__.py`
- Create: `dfxm/compose/recipe.py`
- Create: `tests/test_compose_recipe.py`
- Modify: `docs/Codebase.md` (add `dfxm/compose/recipe.py` under "Layer 1 — `dfxm/` core library"; Read the file section first)

**Interfaces**

*Consumes:* `dfxm.common.errors.StageUserError(message, hint="")`.

*Produces (exact public surface other tasks import):*
```python
RECIPE_VERSION: int = 1
PANEL_KINDS: tuple[str, ...] = ("map_layer", "slice_plane", "profiles_ref", "profiles_trace")
SCALE_BAR_MODES: tuple[str, ...] = ("per-panel", "one-panel", "gutter")

@dataclass
class ComposeStyle:
    label_template: str = "A"
    label_font_scale: float = 1.0
    gutter_cm: float = 0.5
    padding_cm: float = 0.3
    scale_bar_mode: str = "per-panel"
    scale_bar_panel: str | None = None
    pinned_width_cm: float | None = None

@dataclass
class PanelSource:
    h5_path: str
    kind: str          # one of PANEL_KINDS
    selector: dict     # kind-specific selection key (see Task 4)

@dataclass
class PanelDef:
    id: str
    source: PanelSource
    roi: tuple[int, int, int, int] | None = None       # (r0, r1, c0, c1) px, replot convention
    clim: tuple[float | None, float | None] | None = None
    cmap: str | None = None
    label: str | None = None          # None = auto sequence; "" = no label; text = manual
    show_title: bool | None = None    # None = composed default (off)
    scale_um_per_cm: float | None = None
    colorbar: bool | None = None      # None = follow style; False when a shared bar covers it

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
    group_label: str | None = None    # None = not a group; "auto" = auto slot; text = manual
    shared_colorbar: bool = False
    shared_clim: tuple[float, float] | None = None

@dataclass
class Col:
    children: list
    pinned_width_cm: float | None = None
    group_label: str | None = None
    shared_x: bool = False
    shared_colorbar: bool = False
    shared_clim: tuple[float, float] | None = None

@dataclass
class FigureRecipe:
    name: str
    style: dict                        # PlotStyle field overrides, JSON-safe
    compose: ComposeStyle
    layout: Row | Col | PanelRef | Spacer | TextCell
    panels: list[PanelDef]
    version: int = RECIPE_VERSION

    def panel_by_id(self) -> dict[str, PanelDef]: ...

def recipe_to_json(recipe: FigureRecipe, *, base_dir: str | None = None) -> str
def recipe_from_json(text: str, *, base_dir: str | None = None) -> FigureRecipe
def validate_recipe(recipe: FigureRecipe) -> None   # raises StageUserError
def iter_leaves(node) -> "Iterator[PanelRef | Spacer | TextCell]"
```
`base_dir`: on save, `PanelSource.h5_path` is stored relative to it when `os.path.relpath` succeeds on the same drive; on load, relative paths are resolved against it. Layout leaves serialize as `{"type": "panel", "panel_id": ...}` / `{"type": "spacer", ...}` / `{"type": "text", ...}`; nodes as `{"type": "row"|"col", "children": [...], ...}`.

**Steps**

- [ ] Write the failing tests — `tests/test_compose_recipe.py`:
```python
"""Recipe schema + JSON round-trip — dfxm.compose.recipe."""

import pytest

from dfxm.common.errors import StageUserError
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    Row,
    Spacer,
    TextCell,
    iter_leaves,
    recipe_from_json,
    recipe_to_json,
    validate_recipe,
)


def _mini_recipe():
    p1 = PanelDef("m0", __src("/data/stack.h5", "map_layer", {"stage": "mosaicity"}))
    p2 = PanelDef("t0", __src("/data/obl.h5", "profiles_trace", {"field": "strain"}))
    layout = Row([PanelRef("m0"), Col([PanelRef("t0"), Spacer(1.0, 1.0)], shared_x=True)])
    return FigureRecipe(
        name="demo",
        style={"scale_um_per_cm": 10.0},
        compose=ComposeStyle(label_template="(A)"),
        layout=layout,
        panels=[p1, p2],
    )


def __src(path, kind, sel):
    from dfxm.compose.recipe import PanelSource

    return PanelSource(path, kind, sel)


def test_round_trip_preserves_everything():
    r = _mini_recipe()
    r2 = recipe_from_json(recipe_to_json(r))
    assert r2.name == "demo" and r2.version == r.version
    assert r2.style == {"scale_um_per_cm": 10.0}
    assert r2.compose.label_template == "(A)"
    assert [p.id for p in r2.panels] == ["m0", "t0"]
    assert isinstance(r2.layout, Row)
    col = r2.layout.children[1]
    assert isinstance(col, Col) and col.shared_x is True
    assert isinstance(col.children[1], Spacer)
    leaves = list(iter_leaves(r2.layout))
    assert [type(x).__name__ for x in leaves] == ["PanelRef", "PanelRef", "Spacer"]


def test_relative_h5_paths_round_trip(tmp_path):
    r = _mini_recipe()
    r.panels[0].source.h5_path = str(tmp_path / "sub" / "stack.h5")
    text = recipe_to_json(r, base_dir=str(tmp_path))
    assert str(tmp_path) not in text  # stored relative
    r2 = recipe_from_json(text, base_dir=str(tmp_path))
    assert r2.panels[0].source.h5_path == str(tmp_path / "sub" / "stack.h5")


def test_unknown_version_raises_stageusererror():
    text = recipe_to_json(_mini_recipe()).replace('"version": 1', '"version": 99')
    with pytest.raises(StageUserError) as e:
        recipe_from_json(text)
    assert "version" in str(e.value) and e.value.hint


@pytest.mark.parametrize(
    "mutate,frag",
    [
        (lambda r: r.panels.append(PanelDef("m0", __src("x", "map_layer", {}))), "duplicate"),
        (lambda r: r.layout.children.append(PanelRef("ghost")), "ghost"),
        (lambda r: setattr(r.panels[0].source, "kind", "hologram"), "hologram"),
        (lambda r: setattr(r.compose, "scale_bar_mode", "everywhere"), "scale_bar_mode"),
        (lambda r: setattr(r.compose, "label_template", "xx"), "label_template"),
    ],
)
def test_validate_refuses_bad_recipes(mutate, frag):
    r = _mini_recipe()
    mutate(r)
    with pytest.raises(StageUserError) as e:
        validate_recipe(r)
    assert frag in str(e.value)


def test_bad_json_raises_stageusererror():
    with pytest.raises(StageUserError):
        recipe_from_json("{not json")
    with pytest.raises(StageUserError):
        recipe_from_json('{"no": "layout"}')
```
- [ ] Run to see it fail: `python3 -m pytest -q tests/test_compose_recipe.py` — expect `ModuleNotFoundError: No module named 'dfxm.compose'`.
- [ ] Implement. `dfxm/compose/__init__.py`:
```python
"""Publication figure composer (Qt-free): recipes, layout solver, adapters, render."""
```
`dfxm/compose/recipe.py` (core of the (de)serializer; the dataclasses are exactly the Interfaces block above):
```python
"""Figure-recipe data model + JSON (de)serialization + validation (Qt-free)."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields

from ..common.errors import StageUserError

RECIPE_VERSION = 1
PANEL_KINDS = ("map_layer", "slice_plane", "profiles_ref", "profiles_trace")
SCALE_BAR_MODES = ("per-panel", "one-panel", "gutter")

# ... dataclasses exactly as in the Interfaces block ...


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
            group_label=d.get("group_label"),
            shared_colorbar=bool(d.get("shared_colorbar", False)),
            shared_clim=tuple(d["shared_clim"]) if d.get("shared_clim") else None,
        )
    if t == "col":
        return Col(
            [_node_from_dict(c) for c in d.get("children", [])],
            pinned_width_cm=d.get("pinned_width_cm"),
            group_label=d.get("group_label"),
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
```
plus `recipe_to_json` (asdict panels with tuple→list, layout via `_node_to_dict`, relpath the `h5_path`s when `base_dir` given and `os.path.relpath` doesn't raise/escape to another drive), `recipe_from_json` (json.loads wrapped → `StageUserError("recipe is not valid JSON…", hint=…)`; version check `!= RECIPE_VERSION` → `StageUserError(f"unsupported recipe version {v}", hint="This app writes version 1 recipes — re-save the recipe with this version of the app.")`; missing `layout`/`panels` → `StageUserError`; tuples restored for `roi`/`clim`; paths resolved via `os.path.join(base_dir, p)` when not absolute), and `validate_recipe` (duplicate panel ids, every layout `PanelRef.panel_id` present in `panels` — message contains the offending id, `source.kind in PANEL_KINDS`, `compose.scale_bar_mode in SCALE_BAR_MODES`, `label_template` contains at least one `A`/`a` — message names `label_template`, positive gutter/padding). Every raise carries a `hint`.
- [ ] Run to pass: `python3 -m pytest -q tests/test_compose_recipe.py` then `ruff check dfxm/compose tests/test_compose_recipe.py`.
- [ ] Update `docs/Codebase.md`: under Layer 1 add a `dfxm/compose/` subsection stub listing `recipe.py` (dataclasses + JSON + validation) — later tasks extend it.
- [ ] Commit: `git add -A && git commit -m "feat(compose): recipe schema, JSON round-trip and validation"`

---

### Task 2 — Extract `draw_map_layer` from `render.layer_figure` (regression-pinned)

**Files**
- Modify: `dfxm/common/render.py` (function `layer_figure`, lines 41–84)
- Create: `tests/test_draw_map_layer.py`
- Modify: `docs/Codebase.md` (the `dfxm/common/render.py` entry gains `draw_map_layer`)

**Interfaces**

*Consumes (verified against `dfxm/common/render.py` + `dfxm/common/plotting.py`):* `PlotStyle`, `add_colorbar(fig, im, ax, label, style, *, group=None, cax=None)`, `draw_scale_bar(ax, length_um=None, *, style, fixed_scale_um_per_cm=None)`, `apply_text_scale(ax, style)`, `apply_axes_mode(ax, style)`, `cmap_nan_transparent(name)`, `fixed_scale_box`, `figure_size`, `styled_figure`, `fit_axes_to_box`.

*Produces:*
```python
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
    colorbar=None,     # None = follow style flag; explicit bool overrides
    scale_bar=None,    # None = follow style flag; explicit bool overrides
    fixed_scale_um_per_cm=None,
):
    """Draw one equal-aspect map layer into *ax* (µm axes); returns the image."""
```
`layer_figure(...)` keeps its exact current signature and behaviour, re-calling `draw_map_layer`.

**Steps**

- [ ] Write the characterization test FIRST (it must pass against the *unmodified* code — this pins the behaviour the refactor must preserve) — `tests/test_draw_map_layer.py`:
```python
"""Pin layer_figure's single-figure output across the draw_map_layer extraction."""

import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox

from dfxm.common.plotting import PlotStyle
from dfxm.common.render import layer_figure

LAYER = np.linspace(0.0, 1.0, 24 * 30).reshape(24, 30)


def _bar_boxes(ax):
    return [a for a in ax.artists if isinstance(a, AnchoredOffsetbox)]


def test_layer_figure_unstyled_pinned():
    fig, ax, im = layer_figure(LAYER, 0.0, 1.0, "magma", 3.0, 2.4, "T", "C")
    assert tuple(fig.get_size_inches()) == (12.0, 10.0)
    assert fig.get_layout_engine() is None  # legacy: plain figure
    assert im.get_extent() == (0.0, 3.0, 0.0, 2.4)
    assert (im.norm.vmin, im.norm.vmax) == (0.0, 1.0)
    assert ax.get_xlabel() == "X (µm)" and ax.get_ylabel() == "Y (µm)"
    assert ax.get_title() == "T"
    assert len(fig.axes) == 2  # main + stolen colorbar
    assert fig.axes[1].get_ylabel() == "C"
    assert len(_bar_boxes(ax)) == 1  # scale bar drawn


def test_layer_figure_styled_flags_and_fixed_scale():
    style = PlotStyle(scale_um_per_cm=10.0, colorbar=False, scale_bar=False, show_title=False)
    fig, ax, im = layer_figure(LAYER, 0.0, 1.0, "magma", 30.0, 24.0, "T", "C", style=style)
    assert len(fig.axes) == 1  # colorbar honoured off
    assert _bar_boxes(ax) == []  # scale bar honoured off
    assert ax.get_title() == ""  # show_title off via apply_text_scale
    from dfxm.common.plotting import measured_box_in

    w, h = measured_box_in(fig, ax)
    assert abs(w - 30.0 / 10.0 / 2.54) < 0.02 and abs(h - 24.0 / 10.0 / 2.54) < 0.02


def test_layer_figure_styled_group_cmap_resolution():
    style = PlotStyle(cmap_strain="coolwarm")
    fig, ax, im = layer_figure(
        LAYER, -1.0, 1.0, "magma", 3.0, 2.4, "T", "C", style=style, group="strain"
    )
    assert im.get_cmap().name == "coolwarm"
```
Note the third test pins that cmap-group resolution happens in the *caller* (`render_volume_layer` passes the resolved name) — verify by reading `layer_figure`: it receives `cmap` already resolved, so `im.get_cmap().name` equals the passed name only when the caller resolved it. `layer_figure` itself does NOT resolve groups — so this third test must pass the resolved cmap. Correct it to call with `cmap="coolwarm"` and assert `"coolwarm"`; keep the test minimal and true to current behaviour (write the assertion, run, and fix the test until it passes on unmodified code).
- [ ] Run to see it PASS on unmodified code: `python3 -m pytest -q tests/test_draw_map_layer.py` (fix any assertion that mismatches current behaviour — the point is an accurate pin, not red).
- [ ] Refactor `dfxm/common/render.py` — add `draw_map_layer` and rewrite `layer_figure` to call it:
```python
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
    """Single equal-aspect layer figure (µm axes).  (docstring unchanged)"""
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
```
(Keep the original module docstring/imports; `mcolors` is already imported. One behaviour subtlety to preserve: in the original, `add_colorbar`/`draw_scale_bar`/`apply_*` receive `st` — identical here.)
- [ ] Run the pins + neighbours: `python3 -m pytest -q tests/test_draw_map_layer.py tests/test_figures_replot.py tests/test_stage_visualize.py tests/test_stage_rocking.py tests/test_stage_mosaicity.py tests/test_export_fidelity.py tests/test_figure_layout.py` — all green.
- [ ] Update `docs/Codebase.md` render.py entry (add `draw_map_layer`, note the extraction).
- [ ] Commit: `git add -A && git commit -m "refactor(render): extract draw_map_layer from layer_figure (pinned by regression tests)"`

---

### Task 3 — Extract `draw_slice_axes` from `slices.build_slice_figure`; public profiles draw aliases

**Files**
- Modify: `dfxm/stages/slices.py` (`build_slice_figure`, lines 821–878)
- Modify: `dfxm/stages/profiles.py` (add two aliases right after `_draw_trace_axes`, ~line 942)
- Modify: `tests/test_stage_slices.py` (append characterization tests)
- Modify: `docs/Codebase.md` (slices/profiles function lists)

**Interfaces**

*Consumes (verified):* `slices._make_norm(prep)`, `slices._LEGACY_STYLE`, `Rnd.cmap_nan_transparent`, `add_colorbar`, `draw_scale_bar`, `apply_text_scale`, `apply_axes_mode`; `profiles._draw_reference_image(ax, plane2d, u_um, v_um, attrs, line_color, geom=None, title=None, style=None, fixed_scale_um_per_cm=None)`; `profiles._draw_trace_axes(ax, fld, geom, *, linewidth, color, font_scale, style, show_xlabel=True)`.

*Produces:*
```python
# dfxm/stages/slices.py
def draw_slice_axes(
    ax, prep, sl, slice2d, u_um, v_um, *, offset_um, style=None,
    cax=None, colorbar=None, scale_bar=None, fixed_scale_um_per_cm=None,
):
    """Draw one oblique-slice plane into *ax*; returns the image."""

# dfxm/stages/profiles.py — public aliases for the compose adapters
draw_reference_axes = _draw_reference_image
draw_trace_axes = _draw_trace_axes
```
`build_slice_figure` keeps its exact signature/behaviour, re-calling `draw_slice_axes`.

**Steps**

- [ ] Append characterization tests to `tests/test_stage_slices.py` (they must pass pre-refactor):
```python
# -- draw_slice_axes extraction pins ------------------------------------------
def _prep(cmap="magma", center=False):
    return {
        "cmap_name": cmap,
        "title": "χ CoM",
        "cbar_label": "deg",
        "vmin": -1.0,
        "vmax": 3.0,
        "center_zero": center,
        "group": "mosa_com",
    }


def test_build_slice_figure_unstyled_pinned_shape_and_decor():
    import numpy as np

    from dfxm.stages.slices import build_slice_figure

    u = np.linspace(-5.0, 5.0, 21)
    v = np.linspace(-4.0, 4.0, 17)
    fig = build_slice_figure(
        _prep(), {"name": "obl"}, np.zeros((17, 21)), u, v, offset_um=1.0
    )
    assert tuple(fig.get_size_inches()) == (12.0, 10.0)
    ax = fig.axes[0]
    im = ax.images[0]
    assert im.get_extent() == (-5.0, 5.0, -4.0, 4.0)
    assert ax.get_xlabel() == "u (µm)" and ax.get_ylabel() == "v (µm)"
    assert "offset +1.00" in ax.get_title()
    assert len(fig.axes) == 2  # stolen colorbar present


def test_build_slice_figure_centered_norm_pinned():
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    from dfxm.stages.slices import build_slice_figure

    u = np.linspace(0.0, 2.0, 5)
    v = np.linspace(0.0, 2.0, 5)
    fig = build_slice_figure(
        _prep(center=True), {"name": "obl"}, np.zeros((5, 5)), u, v, offset_um=None
    )
    assert isinstance(fig.axes[0].images[0].norm, TwoSlopeNorm)
```
- [ ] Run to see them PASS on unmodified code: `python3 -m pytest -q tests/test_stage_slices.py -k "draw_slice or build_slice_figure_unstyled or centered_norm"`.
- [ ] Refactor `slices.py` — insert above `build_slice_figure`:
```python
def draw_slice_axes(
    ax,
    prep,
    sl,
    slice2d,
    u_um,
    v_um,
    *,
    offset_um,
    style: PlotStyle | None = None,
    cax=None,
    colorbar=None,
    scale_bar=None,
    fixed_scale_um_per_cm=None,
):
    """Draw one oblique-slice plane into *ax*; returns the image.

    Extracted verbatim from :func:`build_slice_figure` so the single-figure path
    and the compose adapters share one look. ``colorbar``/``scale_bar`` default
    to the style flags; explicit bools override. ``cax`` routes the colourbar
    into an already-placed axes (steal-free).
    """
    st = style if style is not None else _LEGACY_STYLE
    use_legacy = style is None
    fig = ax.get_figure()
    extent = [float(u_um[0]), float(u_um[-1]), float(v_um[0]), float(v_um[-1])]
    im = ax.imshow(
        slice2d,
        cmap=Rnd.cmap_nan_transparent(prep["cmap_name"]),
        norm=_make_norm(prep),
        extent=extent,
        origin="lower",
        aspect="equal",
    )
    ax.set_xlabel("u (µm)")
    ax.set_ylabel("v (µm)")
    sub = sl["name"] if offset_um is None else f"{sl['name']}  (offset {offset_um:+.2f} µm)"
    ax.set_title(f"{prep['title']}\nslice: {sub}")
    if st.colorbar if colorbar is None else colorbar:
        add_colorbar(fig, im, ax, prep["cbar_label"], st, group=prep.get("group"), cax=cax)
    if st.scale_bar if scale_bar is None else scale_bar:
        draw_scale_bar(
            ax, st.scale_bar_length_um, style=st, fixed_scale_um_per_cm=fixed_scale_um_per_cm
        )
    if not use_legacy:
        apply_text_scale(ax, st)
        apply_axes_mode(ax, st)
    return im
```
and reduce `build_slice_figure`'s body after `ax = fig.add_subplot(111)` to a single `draw_slice_axes(ax, prep, sl, slice2d, u_um, v_um, offset_um=offset_um, style=style, fixed_scale_um_per_cm=(box[2] if box is not None else None))` followed by the unchanged `if box is not None: fit_axes_to_box(...)` and `return fig` (figsize/box logic stays in `build_slice_figure`).
- [ ] Add the two aliases in `profiles.py` immediately after `_draw_trace_axes` (Read the exact region first per repo rules):
```python
# Public aliases for the compose adapters (the composer draws into its own axes;
# the underscore originals remain the in-module call sites).
draw_reference_axes = _draw_reference_image
draw_trace_axes = _draw_trace_axes
```
- [ ] Run: `python3 -m pytest -q tests/test_stage_slices.py tests/test_stage_profiles.py` — green.
- [ ] Update `docs/Codebase.md` (slices gains `draw_slice_axes`; profiles gains the aliases).
- [ ] Commit: `git add -A && git commit -m "refactor(slices,profiles): extract draw_slice_axes; public draw aliases for compose"`

---

### Task 4 — Panel adapters + pure loaders (`dfxm/compose/adapters.py`)

**Files**
- Create: `dfxm/compose/adapters.py`
- Create: `tests/test_compose_adapters.py`
- Modify: `docs/Codebase.md` (`dfxm/compose` section grows `adapters.py`)

**Interfaces**

*Consumes (verified):* `dfxm.common.render.draw_map_layer` (Task 2), `dfxm.stages.slices.draw_slice_axes` (Task 3), `dfxm.stages.profiles.draw_reference_axes` / `draw_trace_axes` / `_collect(f, job, p, ref_pref, restrict, clim=None)` / `auto_line_color(cmap_name, override)` / `STAGE.defaults()`, `dfxm.stages.mosaicity._KEY_DISPLAY` / `_streamed_clim`, `dfxm.stages.rocking._DATASET_DISPLAY` / `_replot_default_clim(dataset, params, style)`, `dfxm.common.figures.crop_roi_2d`, `dfxm.common.plotting.GROUP_BY_KIND`, `symmetric_limits`, `resolve_cmap`, `StageUserError`.

*Produces:*
```python
@dataclass
class PanelData:
    kind: str                       # PANEL_KINDS value or "placeholder"
    ext_x_um: float | None = None   # sizing input for maps
    ext_y_um: float | None = None
    length_um: float | None = None  # sizing input for traces
    group: str | None = None        # quantity group (shared-colorbar compatibility)
    vmin: float | None = None       # default colour limits (pre-override)
    vmax: float | None = None
    payload: dict = field(default_factory=dict)  # kind-specific draw inputs

def load_panel(panel: PanelDef, *, cache: dict | None = None) -> PanelData
    # never raises for missing file/dataset — returns kind="placeholder" with
    # payload={"reason": str}; raises StageUserError only for malformed selectors

def draw_panel(
    ax, panel: PanelDef, data: PanelData, style, *,
    cax=None, colorbar=None, scale_bar=None, fixed_scale_um_per_cm=None,
    show_xlabel=True, show_title=False,
):
    # dispatch by data.kind; returns the AxesImage for maps/slices/refs, None for traces
    # kind == "placeholder": hatched grey cell, no ticks

def draw_placeholder(ax, reason: str) -> None
```
Selector shapes (documented in the module docstring, enforced by `load_panel`):
- `map_layer`: `{"stage": "strain"|"mosaicity"|"rocking", "dataset": str, "z": int, "sx": float?, "sy": float?}` (strain: dataset fixed `"strain"`, sx/sy default from file attrs `scale_x_um`/`scale_y_um`; mosaicity/rocking: sx/sy required — the GUI/recipe author supplies them, defaults 0.152/0.385 like the replots).
- `slice_plane`: `{"volume_id": str, "slice_name": str, "plane": int}`.
- `profiles_ref`: `{"job": dict, "field": str | None}` (`None` → job reference field).
- `profiles_trace`: `{"job": dict, "field": str}`.

**Steps**

- [ ] Write failing tests — `tests/test_compose_adapters.py`. Reuse the synthetic-h5 idioms already in the suite (`tests/test_figures_replot.py::_write_vol` for stacked volumes; `tests/test_stage_profiles.py::_write_consolidated` shape for oblique_slices):
```python
"""Panel adapters: pure loaders + draw-into-axes dispatch — dfxm.compose.adapters."""

import h5py
import numpy as np
from matplotlib.figure import Figure

from dfxm.compose.adapters import PanelData, draw_panel, load_panel
from dfxm.compose.recipe import PanelDef, PanelSource


def _write_mosa(path):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        f.create_dataset("/chi/Center of mass", data=rng.normal(size=(2, 6, 8)).astype("f4"))
    return str(path)


def _write_strain(path):
    with h5py.File(path, "w") as f:
        f.create_dataset("strain", data=np.linspace(-2e-4, 2e-4, 2 * 6 * 8).reshape(2, 6, 8))
        f.attrs["scale_x_um"] = 0.2
        f.attrs["scale_y_um"] = 0.4
    return str(path)


def _write_obl(path):
    u = np.linspace(-10.0, 10.0, 41)
    v = np.linspace(-8.0, 8.0, 33)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(path, "w") as f:
        for vid, kind in (("raw_sum", "raw_sum"), ("strain", "strain")):
            g = f.create_group(vid)
            g.attrs.update(
                kind=kind, cbar_label="value", cmap="gray", title=vid, vmin=-10.0, vmax=10.0
            )
            sg = g.create_group("obl")
            sg.create_dataset("slices", data=(uu + vv)[None, ...].astype("f4"))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))
    return str(path)


JOB = {"name": "obl", "offset_um": 0.0, "start_uv": [-5.0, -3.0], "end_uv": [5.0, 3.0]}


def test_load_map_layer_mosaicity_extents_and_group(tmp_path):
    h5 = _write_mosa(tmp_path / "stack.h5")
    p = PanelDef(
        "m",
        PanelSource(
            h5,
            "map_layer",
            {"stage": "mosaicity", "dataset": "/chi/Center of mass", "z": 1, "sx": 0.5, "sy": 0.25},
        ),
    )
    d = load_panel(p)
    assert d.kind == "map_layer" and d.group == "mosa_com"
    assert d.ext_x_um == 8 * 0.5 and d.ext_y_um == 6 * 0.25
    assert d.vmin is not None and d.vmax is not None and d.vmin < d.vmax


def test_load_map_layer_strain_defaults_from_attrs_and_symmetric(tmp_path):
    h5 = _write_strain(tmp_path / "strain.h5")
    p = PanelDef("s", PanelSource(h5, "map_layer", {"stage": "strain", "z": 0}))
    d = load_panel(p)
    assert d.group == "strain"
    assert d.ext_x_um == 8 * 0.2 and d.ext_y_um == 6 * 0.4
    assert d.vmin == -d.vmax  # symmetric limits


def test_load_map_layer_roi_crops_extent(tmp_path):
    h5 = _write_mosa(tmp_path / "stack.h5")
    p = PanelDef(
        "m",
        PanelSource(
            h5,
            "map_layer",
            {"stage": "mosaicity", "dataset": "/chi/Center of mass", "z": 0, "sx": 1.0, "sy": 1.0},
        ),
        roi=(1, 4, 2, 6),
    )
    d = load_panel(p)
    assert d.ext_x_um == 4.0 and d.ext_y_um == 3.0


def test_load_slice_plane_and_profiles(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    sp = load_panel(
        PanelDef(
            "p",
            PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
        )
    )
    assert sp.kind == "slice_plane" and sp.group == "strain"
    assert sp.ext_x_um == 20.0 and sp.ext_y_um == 16.0
    ref = load_panel(PanelDef("r", PanelSource(h5, "profiles_ref", {"job": JOB, "field": None})))
    assert ref.kind == "profiles_ref" and ref.ext_x_um == 20.0
    tr = load_panel(
        PanelDef("t", PanelSource(h5, "profiles_trace", {"job": JOB, "field": "strain"}))
    )
    assert tr.kind == "profiles_trace"
    assert abs(tr.length_um - np.hypot(10.0, 6.0)) < 1e-9


def test_missing_file_and_missing_key_become_placeholders(tmp_path):
    d = load_panel(
        PanelDef("x", PanelSource(str(tmp_path / "gone.h5"), "map_layer", {"stage": "strain"}))
    )
    assert d.kind == "placeholder" and "gone.h5" in d.payload["reason"]
    h5 = _write_mosa(tmp_path / "stack.h5")
    d2 = load_panel(
        PanelDef(
            "y",
            PanelSource(
                h5, "map_layer", {"stage": "mosaicity", "dataset": "/nope", "z": 0, "sx": 1, "sy": 1}
            ),
        )
    )
    assert d2.kind == "placeholder"


def test_loader_cache_hit_skips_reread(tmp_path):
    h5 = _write_strain(tmp_path / "strain.h5")
    p = PanelDef("s", PanelSource(h5, "map_layer", {"stage": "strain", "z": 0}))
    cache = {}
    d1 = load_panel(p, cache=cache)
    import os

    os.remove(h5)
    d2 = load_panel(p, cache=cache)  # served from cache, file gone
    assert d2 is d1


def test_draw_panel_dispatch(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    fig = Figure(figsize=(6, 4))
    for sel_kind, sel in (
        ("slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
        ("profiles_ref", {"job": JOB, "field": None}),
        ("profiles_trace", {"job": JOB, "field": "strain"}),
    ):
        ax = fig.add_subplot(111)
        p = PanelDef("p", PanelSource(h5, sel_kind, sel))
        d = load_panel(p)
        draw_panel(ax, p, d, None, colorbar=False, scale_bar=False)
        if sel_kind == "profiles_trace":
            assert ax.lines  # the profile curve
        else:
            assert ax.images
        fig.clear()


def test_draw_placeholder_hatched(tmp_path):
    from dfxm.compose.adapters import draw_placeholder

    fig = Figure()
    ax = fig.add_subplot(111)
    draw_placeholder(ax, "missing file")
    assert any(p.get_hatch() for p in ax.patches)
    assert ax.get_xticks().size == 0
```
- [ ] Run to fail: `python3 -m pytest -q tests/test_compose_adapters.py` — `ImportError: cannot import name 'load_panel'` (module absent).
- [ ] Implement `dfxm/compose/adapters.py`:
```python
"""Panel adapters: pure (source, roi) loaders + draw-into-axes dispatch (Qt-free).

Loaders are pure functions of (PanelSource, roi) so the GUI can cache results;
heavy deps (h5py, stage modules) are imported inside functions. A loader never
raises for missing files/keys — it returns a ``kind="placeholder"`` PanelData
with the reason; only a malformed selector raises StageUserError.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from ..common.errors import StageUserError
from ..common.figures import crop_roi_2d
from ..common.plotting import GROUP_BY_KIND, symmetric_limits
from .recipe import PANEL_KINDS, PanelDef


@dataclass
class PanelData:
    kind: str
    ext_x_um: float | None = None
    ext_y_um: float | None = None
    length_um: float | None = None
    group: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    payload: dict = field(default_factory=dict)


def _cache_key(panel: PanelDef) -> str:
    src = panel.source
    return json.dumps(
        [src.h5_path, src.kind, src.selector, list(panel.roi) if panel.roi else None],
        sort_keys=True,
        default=str,
    )


def load_panel(panel: PanelDef, *, cache: dict | None = None) -> PanelData:
    if panel.source.kind not in PANEL_KINDS:
        raise StageUserError(
            f"panel {panel.id!r}: unknown kind {panel.source.kind!r}",
            hint=f"Valid kinds: {', '.join(PANEL_KINDS)}.",
        )
    key = _cache_key(panel)
    if cache is not None and key in cache:
        return cache[key]
    loader = _LOADERS[panel.source.kind]
    try:
        data = loader(panel.source.h5_path, panel.source.selector, panel.roi)
    except StageUserError:
        raise
    except Exception as exc:  # noqa: BLE001 — the composition survives partial data
        data = PanelData(kind="placeholder", payload={"reason": f"{panel.source.h5_path}: {exc}"})
    if cache is not None:
        cache[key] = data
    return data
```
Loaders (each with lazy `import h5py`):
```python
def _load_map_layer(h5_path, sel, roi):
    import h5py

    stage = sel.get("stage")
    if stage not in ("strain", "mosaicity", "rocking"):
        raise StageUserError(
            f"map_layer selector needs stage strain/mosaicity/rocking (got {stage!r})",
            hint='Set selector["stage"], e.g. {"stage": "mosaicity", "dataset": ..., "z": 0}.',
        )
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"{h5_path} not found")
    with h5py.File(h5_path, "r") as f:
        if stage == "strain":
            dset = f["strain"]
            sx = float(sel.get("sx") or f.attrs.get("scale_x_um", 0.152))
            sy = float(sel.get("sy") or f.attrs.get("scale_y_um", 0.385))
            title, cbar, group, cmap = "Strain map (cot method)", "Strain (ε)", "strain", "RdBu_r"
        else:
            dset = f[sel["dataset"]]
            sx, sy = float(sel.get("sx", 0.152)), float(sel.get("sy", 0.385))
            if stage == "mosaicity":
                from ..stages.mosaicity import _KEY_DISPLAY, _streamed_clim

                group, cbar, title = _KEY_DISPLAY.get(sel["dataset"], (None, "(°)", sel["dataset"]))
                cmap = "magma"
            else:
                from ..stages.rocking import _DATASET_DISPLAY, _replot_default_clim

                title, cbar = _DATASET_DISPLAY.get(sel["dataset"], (sel["dataset"], "Intensity"))
                group, cmap = "raw", "gray"
        z = int(sel.get("z", 0))
        layer = dset[z][...]
        if stage == "mosaicity":
            vmin, vmax = _streamed_clim(dset)
        elif stage == "rocking":
            vmin, vmax = _replot_default_clim(dset, {}, None)
        else:
            vmin, vmax = symmetric_limits(layer)
    layer = crop_roi_2d(layer, panel_roi := roi)
    if layer is None:
        raise ValueError(f"ROI {panel_roi} crops to an empty layer")
    return PanelData(
        kind="map_layer",
        ext_x_um=layer.shape[1] * sx,
        ext_y_um=layer.shape[0] * sy,
        group=group,
        vmin=float(vmin),
        vmax=float(vmax),
        payload={"layer": layer, "sx": sx, "sy": sy, "title": title, "cbar": cbar, "cmap": cmap},
    )
```
`_load_slice_plane` mirrors `slices._rebuild_plane_figure`'s reading block (attrs → prep dict with `cmap_name/title/cbar_label/vmin/vmax/center_zero/group`, plane + `u_um`/`v_um`, ROI cropping `u[c0:c1]`, `v[r0:r1]` with the same clamping code, `offset_um`), returning `PanelData(kind="slice_plane", ext_x_um=float(u[-1]-u[0]), ext_y_um=float(v[-1]-v[0]), group=GROUP_BY_KIND.get(kind), vmin=prep["vmin"], vmax=prep["vmax"], payload={"prep": prep, "sname": sname, "plane2d": s2d, "u": u, "v": v, "offset_um": off})`. `_load_profiles_ref` / `_load_profiles_trace` open the h5 and call `profiles._collect(f, job, p, ref_pref="", restrict=None)` with `p = profiles.STAGE.defaults()`; ref: pick `ref = (plane, u, v, attrs, label)` (or the named field's plane like `_save_overviews` builds `ov_ref`), apply ROI cropping to plane/u/v as in slices, line colour via `profiles.auto_line_color(attrs["cmap"], None)`, `PanelData(kind="profiles_ref", ext_x_um=float(u[-1]-u[0]), ext_y_um=float(v[-1]-v[0]), group=GROUP_BY_KIND.get(attrs.get("kind")), vmin=attrs["vmin"], vmax=attrs["vmax"], payload={"plane": plane, "u": u, "v": v, "attrs": attrs, "geom": geom, "line_color": color})`; trace: find `fld` by `vid` in `fields` (missing → `ValueError` → placeholder), `PanelData(kind="profiles_trace", length_um=float(geom["L"]), group=GROUP_BY_KIND.get(fld["attrs"].get("kind")), payload={"fld": fld, "geom": geom})`.
`draw_panel` dispatch:
```python
def draw_panel(
    ax,
    panel,
    data,
    style,
    *,
    cax=None,
    colorbar=None,
    scale_bar=None,
    fixed_scale_um_per_cm=None,
    show_xlabel=True,
    show_title=False,
):
    """Draw *data* into *ax* with the panel's overrides applied. Titles are OFF
    by default in composed figures (panel.show_title=True re-enables)."""
    if data.kind == "placeholder":
        draw_placeholder(ax, data.payload["reason"])
        return None
    from ..common.plotting import resolve_cmap

    titled = panel.show_title if panel.show_title is not None else show_title
    vmin, vmax = data.vmin, data.vmax
    if panel.clim is not None:
        lo, hi = panel.clim
        vmin = float(lo) if lo is not None else vmin
        vmax = float(hi) if hi is not None else vmax
    if data.kind == "map_layer":
        from ..common.render import draw_map_layer

        pay = data.payload
        cmap = panel.cmap or resolve_cmap(style, data.group, fallback=pay["cmap"])
        return draw_map_layer(
            ax, pay["layer"], vmin, vmax, cmap, data.ext_x_um, data.ext_y_um,
            pay["title"] if titled else "", pay["cbar"], style=style, group=data.group,
            cax=cax, colorbar=colorbar, scale_bar=scale_bar,
            fixed_scale_um_per_cm=fixed_scale_um_per_cm,
        )
    if data.kind == "slice_plane":
        from ..stages.slices import draw_slice_axes

        pay = data.payload
        prep = dict(pay["prep"], vmin=vmin, vmax=vmax)
        if panel.cmap:
            prep["cmap_name"] = panel.cmap
        else:
            prep["cmap_name"] = resolve_cmap(style, data.group, fallback=prep["cmap_name"])
        im = draw_slice_axes(
            ax, prep, {"name": pay["sname"]}, pay["plane2d"], pay["u"], pay["v"],
            offset_um=pay["offset_um"], style=style, cax=cax, colorbar=colorbar,
            scale_bar=scale_bar, fixed_scale_um_per_cm=fixed_scale_um_per_cm,
        )
        if not titled:
            ax.set_title("")
        return im
    if data.kind == "profiles_ref":
        from ..common.plotting import add_colorbar
        from ..stages.profiles import draw_reference_axes

        pay = data.payload
        attrs = dict(pay["attrs"], vmin=vmin, vmax=vmax)
        if panel.cmap:
            attrs["cmap"] = panel.cmap
        im = draw_reference_axes(
            ax, pay["plane"], pay["u"], pay["v"], attrs, pay["line_color"], geom=pay["geom"],
            title=(attrs["title"] if titled else None), style=style,
            fixed_scale_um_per_cm=fixed_scale_um_per_cm,
        )
        want_cbar = (style.colorbar if style is not None else True) if colorbar is None else colorbar
        if want_cbar and style is not None:
            add_colorbar(fig := ax.get_figure(), im, ax, attrs["cbar_label"], style,
                         group=data.group, cax=cax)
        return im
    if data.kind == "profiles_trace":
        from ..stages.profiles import draw_trace_axes

        pay = data.payload
        draw_trace_axes(
            ax, pay["fld"], pay["geom"], linewidth=1.8, color=None, font_scale=1.0,
            style=style, show_xlabel=show_xlabel,
        )
        if not titled:
            ax.set_title("")
        if not show_xlabel:
            ax.tick_params(labelbottom=False)
        return None
    raise StageUserError(f"unknown panel kind {data.kind!r}", hint="Recipe is corrupt.")
```
`draw_placeholder`:
```python
def draw_placeholder(ax, reason: str) -> None:
    """Hatched grey cell for a panel whose data is unavailable — never a crash."""
    from matplotlib.patches import Rectangle

    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(
        Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="0.92",
                  edgecolor="0.6", hatch="///")
    )
    ax.text(0.5, 0.5, "unavailable", transform=ax.transAxes, ha="center", va="center",
            fontsize=8, color="0.35")


_LOADERS = {
    "map_layer": _load_map_layer,
    "slice_plane": _load_slice_plane,
    "profiles_ref": _load_profiles_ref,
    "profiles_trace": _load_profiles_trace,
}
```
(`draw_reference_axes` and `draw_trace_axes` are the `_draw_reference_image` / `_draw_trace_axes` aliases from Task 3 — trace styling knobs `linewidth/color/font_scale` stay at their trace defaults in v1; the style's `font_scale` does not multiply trace fonts, matching the standalone trace figures' independent scale.)
- [ ] Run to pass: `python3 -m pytest -q tests/test_compose_adapters.py` + `ruff check dfxm/compose`.
- [ ] Update `docs/Codebase.md` compose section (`adapters.py`: PanelData, load_panel, draw_panel, selector shapes).
- [ ] Commit: `git add -A && git commit -m "feat(compose): panel adapters — pure loaders + draw dispatch with placeholder fallback"`

---

### Task 5 — Layout solver, sizing pass (`dfxm/compose/layout.py`, part 1)

**Files**
- Create: `dfxm/compose/layout.py`
- Create: `tests/test_compose_layout.py`
- Modify: `docs/Codebase.md`

**Interfaces**

*Consumes (verified):* `fixed_scale(style)`, `fixed_scale_box(style, ext_x_um, ext_y_um, scale=None)`, `trace_fixed_box(style, length_um)`, `trace_fixed_scale(style)`, `trace_height_cm(style)`, `dataclasses.replace`, recipe types (Task 1), `PanelData` (Task 4), `StageUserError`.

*Produces:*
```python
_IN_PER_CM = 1.0 / 2.54
PLACEHOLDER_CM = (4.0, 3.0)

@dataclass
class SizedCell:
    leaf: object                 # the layout leaf (PanelRef/Spacer/TextCell)
    panel: PanelDef | None       # None for Spacer/TextCell
    kind: str                    # "map"|"trace"|"spacer"|"text"|"placeholder"
    w_in: float
    h_in: float
    # filled by the placement pass (Task 6):
    ax: object | None = None
    extras: tuple = ()
    sync: object | None = None
    margins: object | None = None
    label: str | None = None

def size_cells(
    recipe: FigureRecipe, style, data_by_id: dict[str, PanelData], notes: list[str]
) -> dict[int, SizedCell]
    # keyed by id(leaf); appends implied-scale / clamp / placeholder notes;
    # raises StageUserError when a physical panel has no scale and no pinned dim
```

**Steps**

- [ ] Write failing tests (append to a new `tests/test_compose_layout.py`):
```python
"""Layout solver, sizing pass — dfxm.compose.layout."""

import pytest

from dfxm.common.errors import StageUserError
from dfxm.common.plotting import PlotStyle
from dfxm.compose.adapters import PanelData
from dfxm.compose.layout import size_cells
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    PanelSource,
    Row,
    Spacer,
    TextCell,
)


def _panel(pid, kind="map_layer"):
    return PanelDef(pid, PanelSource("/x.h5", kind, {}))


def _recipe(layout, panels, style=None):
    return FigureRecipe("t", style or {}, ComposeStyle(), layout, panels)


def _map_data(x=20.0, y=10.0):
    return PanelData(kind="map_layer", ext_x_um=x, ext_y_um=y, group="mosa_com")


def _trace_data(length=30.0):
    return PanelData(kind="profiles_trace", length_um=length)


def test_map_and_trace_intrinsic_boxes_exact():
    style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=5.0, trace_height_cm=2.0)
    pa, pt = _panel("a"), _panel("t", "profiles_trace")
    layout = Row([PanelRef("a"), PanelRef("t")])
    cells = size_cells(
        _recipe(layout, [pa, pt]), style, {"a": _map_data(), "t": _trace_data()}, notes := []
    )
    ca, ct = cells[id(layout.children[0])], cells[id(layout.children[1])]
    assert abs(ca.w_in - 20.0 / 10.0 / 2.54) < 1e-9
    assert abs(ca.h_in - 10.0 / 10.0 / 2.54) < 1e-9
    assert abs(ct.w_in - 30.0 / 5.0 / 2.54) < 1e-9
    assert abs(ct.h_in - 2.0 / 2.54) < 1e-9
    assert notes == []


def test_per_panel_scale_override_wins():
    style = PlotStyle(scale_um_per_cm=10.0)
    p = _panel("a")
    p.scale_um_per_cm = 4.0
    layout = PanelRef("a")
    cells = size_cells(_recipe(layout, [p]), style, {"a": _map_data()}, [])
    assert abs(cells[id(layout)].w_in - 20.0 / 4.0 / 2.54) < 1e-9


def test_no_scale_anywhere_refused_with_hint():
    layout = PanelRef("a")
    with pytest.raises(StageUserError) as e:
        size_cells(_recipe(layout, [_panel("a")]), PlotStyle(), {"a": _map_data()}, [])
    assert "scale" in str(e.value).lower() and e.value.hint


def test_pinned_row_height_rescales_and_notes_implied_scale():
    style = PlotStyle(scale_um_per_cm=10.0)
    layout = Row([PanelRef("a")], pinned_height_cm=2.0)  # intrinsic h would be 1 cm
    cells = size_cells(_recipe(layout, [_panel("a")]), style, {"a": _map_data()}, notes := [])
    c = cells[id(layout.children[0])]
    assert abs(c.h_in - 2.0 / 2.54) < 1e-9
    assert abs(c.w_in - 2.0 * (20.0 / 10.0) / 2.54) < 1e-9  # aspect preserved
    assert any("implied" in n and "5" in n for n in notes)  # 10 µm/cm -> implied 5 µm/cm


def test_pinned_col_width_covers_missing_scale():
    layout = Col([PanelRef("a")], pinned_width_cm=4.0)
    cells = size_cells(_recipe(layout, [_panel("a")]), PlotStyle(), {"a": _map_data()}, notes := [])
    c = cells[id(layout.children[0])]
    assert abs(c.w_in - 4.0 / 2.54) < 1e-9
    assert abs(c.h_in - 4.0 * (10.0 / 20.0) / 2.54) < 1e-9
    assert any("implied" in n for n in notes)


def test_spacer_text_placeholder_fixed_boxes():
    style = PlotStyle(scale_um_per_cm=10.0)
    layout = Row([Spacer(1.0, 2.0), TextCell("hdr", 3.0, 1.0), PanelRef("a")])
    data = {"a": PanelData(kind="placeholder", payload={"reason": "gone"})}
    cells = size_cells(_recipe(layout, [_panel("a")]), style, data, notes := [])
    sp = cells[id(layout.children[0])]
    tx = cells[id(layout.children[1])]
    ph = cells[id(layout.children[2])]
    assert (sp.w_in, sp.h_in) == (1.0 / 2.54, 2.0 / 2.54)
    assert (tx.w_in, tx.h_in) == (3.0 / 2.54, 1.0 / 2.54)
    assert (ph.w_in, ph.h_in) == (4.0 / 2.54, 3.0 / 2.54)
    assert any("placeholder" in n for n in notes)


def test_trace_clamp_note_surfaces():
    style = PlotStyle(trace_scale_um_per_cm=0.1)  # 500 µm line -> >30 in, clamps
    layout = PanelRef("t")
    cells = size_cells(
        _recipe(layout, [_panel("t", "profiles_trace")]), style, {"t": _trace_data(500.0)},
        notes := [],
    )
    assert cells[id(layout)].w_in == 30.0
    assert any("clamp" in n.lower() for n in notes)
```
- [ ] Run to fail: `python3 -m pytest -q tests/test_compose_layout.py` — `ModuleNotFoundError`/`ImportError`.
- [ ] Implement in `dfxm/compose/layout.py`:
```python
"""Deterministic box-tree layout for composed figures (Qt-free).

Sizing (this half): every layout leaf resolves to an exact (w_in, h_in)
content box from physical scales BEFORE any drawing. Placement (second half)
measures decorations at final box size and places every axes absolutely —
no matplotlib auto-layout anywhere (generalizes place_axes_stack)."""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace

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


def size_cells(recipe, style, data_by_id, notes):
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
                leaf, panel, "placeholder",
                PLACEHOLDER_CM[0] * _IN_PER_CM, PLACEHOLDER_CM[1] * _IN_PER_CM,
            )
        if data.kind == "profiles_trace":
            return _trace_cell(leaf, panel, data, pinned_w_in)
        return _map_cell(leaf, panel, data, pinned_h_in, pinned_w_in)

    def _map_cell(leaf, panel, data, pinned_h_in, pinned_w_in):
        ext_x, ext_y = data.ext_x_um, data.ext_y_um
        if not (ext_x and ext_y and ext_x > 0 and ext_y > 0):
            notes.append(f"panel {panel.id}: degenerate extent — rendered as placeholder")
            return SizedCell(
                leaf, panel, "placeholder",
                PLACEHOLDER_CM[0] * _IN_PER_CM, PLACEHOLDER_CM[1] * _IN_PER_CM,
            )
        eff = panel.scale_um_per_cm or fixed_scale(style)
        if pinned_h_in is not None:
            h = pinned_h_in
            w = h * ext_x / ext_y
            implied = ext_y / (h / _IN_PER_CM)
            notes.append(
                f"panel {panel.id}: pinned row height — implied scale {implied:.4g} µm/cm"
            )
            return SizedCell(leaf, panel, "map", w, h)
        if pinned_w_in is not None:
            w = pinned_w_in
            h = w * ext_y / ext_x
            implied = ext_x / (w / _IN_PER_CM)
            notes.append(
                f"panel {panel.id}: pinned column width — implied scale {implied:.4g} µm/cm"
            )
            return SizedCell(leaf, panel, "map", w, h)
        if eff is None:
            raise StageUserError(
                f"panel {panel.id} has no physical scale to size from", hint=_NO_SCALE_HINT
            )
        box = fixed_scale_box(style, ext_x, ext_y, scale=float(eff))
        if box is None:
            raise StageUserError(
                f"panel {panel.id}: scale {eff!r} is not a positive number", hint=_NO_SCALE_HINT
            )
        if box[2] != float(eff):
            notes.append(
                f"panel {panel.id}: box clamped to 30 in — effective scale {box[2]:.4g} µm/cm"
            )
        return SizedCell(leaf, panel, "map", box[0], box[1])

    def _trace_cell(leaf, panel, data, pinned_w_in):
        length = data.length_um or 0.0
        if pinned_w_in is not None:
            w = pinned_w_in
            h = trace_height_cm(style) * _IN_PER_CM
            implied = length / (w / _IN_PER_CM) if w > 0 else 0.0
            notes.append(
                f"panel {panel.id}: pinned column width — implied trace scale "
                f"{implied:.4g} µm/cm"
            )
            return SizedCell(leaf, panel, "trace", w, h)
        st = style
        if panel.scale_um_per_cm:
            st = dc_replace(style, trace_scale_um_per_cm=float(panel.scale_um_per_cm))
        box = trace_fixed_box(st, float(length))
        if box is None:
            raise StageUserError(
                f"trace panel {panel.id} has no trace scale to size from", hint=_NO_SCALE_HINT
            )
        if box[2] != trace_fixed_scale(st):
            notes.append(
                f"panel {panel.id}: trace box clamped to 30 in — effective scale "
                f"{box[2]:.4g} µm/cm"
            )
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
```
(with the `SizedCell` dataclass from the Interfaces block at module top).
- [ ] Run to pass: `python3 -m pytest -q tests/test_compose_layout.py` + `ruff check dfxm/compose`.
- [ ] Update `docs/Codebase.md` compose section (`layout.py` sizing半 entry: `size_cells`, `SizedCell`, sizing rules incl. pinned-dim implied-scale notes).
- [ ] Commit: `git add -A && git commit -m "feat(compose): layout sizing pass — intrinsic boxes, pinned dims, implied-scale notes"`

---

### Task 6 — Layout solver, measure/align/place engine (`dfxm/compose/layout.py`, part 2)

**Files**
- Modify: `dfxm/compose/layout.py`
- Modify: `tests/test_compose_layout.py` (append)
- Modify: `docs/Codebase.md`

**Interfaces**

*Consumes (verified):* `AxesMargins(left, right, top, bottom)` + `.max_with`, `measure_axes_margins(fig, ax, extras=(), pad_in=0.02)`, `measured_box_in(fig, ax)`, recipe node types, `SizedCell` (Task 5).

*Produces:*
```python
def measure_cells(fig, cells: list[SizedCell], pad_in: float = 0.02) -> None
    # provisional placement of every cell's ax at FINAL box size in *fig*,
    # one draw, fills cell.margins (extras counted; sync re-glued first)

def place_tree(
    fig, layout, cells: dict[int, SizedCell], *, gutter_in: float, pad_in: float
) -> tuple[float, float]
    # shares margins (Row: top/bottom among direct leaf panels; Col: left/right),
    # computes envelopes with trailing padding, sizes *fig*, absolute-places
    # every axes (+ sync); returns (fig_w_in, fig_h_in)
```
Alignment rules implemented here (from the spec): direct `PanelRef` children of a `Row` share max top/bottom margins; direct `PanelRef` children of a `Col` share max left/right margins (y-labels align); composite children align by envelope (top-aligned in a Row, left-aligned in a Col) and shorter/narrower siblings get trailing padding automatically because envelopes are rectangles; `shared_x` Cols must have had interior x tick-labels/xlabel suppressed *before* `measure_cells` (the render step does this — Task 7).

**Steps**

- [ ] Append failing tests to `tests/test_compose_layout.py`:
```python
# -- measure/align/place ------------------------------------------------------
import numpy as np
from matplotlib.figure import Figure

from dfxm.common.plotting import measured_box_in
from dfxm.compose.layout import SizedCell, measure_cells, place_tree


def _plot_cell(fig, leaf, w_in, h_in, ylabel="y"):
    ax = fig.add_subplot(111)
    ax.plot(np.linspace(0, 10, 50), np.sin(np.linspace(0, 10, 50)))
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel(ylabel)
    return SizedCell(leaf, None, "trace", w_in, h_in, ax=ax)


def test_row_shared_top_bottom_margins_and_exact_boxes():
    fig = Figure(facecolor="white")
    a, b = PanelRef("a"), PanelRef("b")
    layout = Row([a, b])
    ca = _plot_cell(fig, a, 2.0, 1.5)
    cb = _plot_cell(fig, b, 1.2, 1.5, ylabel="a much longer label (units)")
    cells = {id(a): ca, id(b): cb}
    measure_cells(fig, [ca, cb])
    place_tree(fig, layout, cells, gutter_in=0.2, pad_in=0.1)
    assert ca.margins.top == cb.margins.top and ca.margins.bottom == cb.margins.bottom
    for c, w in ((ca, 2.0), (cb, 1.2)):
        bw, bh = measured_box_in(fig, c.ax)
        assert abs(bw - w) < 0.01 and abs(bh - 1.5) < 0.01
    # boxes top-align: same y1 in figure inches
    figh = fig.get_size_inches()[1]
    y1a = ca.ax.get_position().y1 * figh
    y1b = cb.ax.get_position().y1 * figh
    assert abs(y1a - y1b) < 0.01


def test_col_shared_left_margin_left_aligns_boxes():
    fig = Figure(facecolor="white")
    a, b = PanelRef("a"), PanelRef("b")
    layout = Col([a, b])
    ca = _plot_cell(fig, a, 2.5, 1.0, ylabel="s")
    cb = _plot_cell(fig, b, 1.4, 1.0, ylabel="a very long y label (deg)")
    cells = {id(a): ca, id(b): cb}
    measure_cells(fig, [ca, cb])
    place_tree(fig, layout, cells, gutter_in=0.15, pad_in=0.1)
    assert abs(ca.ax.get_position().x0 - cb.ax.get_position().x0) < 1e-6
    # no vertical overlap, a above b
    assert ca.ax.get_position().y0 > cb.ax.get_position().y1 - 1e-6


def test_ragged_row_of_cols_trailing_padding_aligns_envelopes():
    fig = Figure(facecolor="white")
    a1, a2, b1 = PanelRef("a1"), PanelRef("a2"), PanelRef("b1")
    col_a = Col([a1, a2])   # two panels -> taller envelope
    col_b = Col([b1])       # one panel -> padded at the bottom
    layout = Row([col_a, col_b])
    cells = {
        id(a1): _plot_cell(fig, a1, 1.5, 1.0),
        id(a2): _plot_cell(fig, a2, 1.5, 1.0),
        id(b1): _plot_cell(fig, b1, 1.5, 1.0),
    }
    measure_cells(fig, list(cells.values()))
    fw, fh = place_tree(fig, layout, cells, gutter_in=0.2, pad_in=0.1)
    # col_b's single panel top-aligns with col_a's first panel
    assert abs(cells[id(b1)].ax.get_position().y1 - cells[id(a1)].ax.get_position().y1) < 1e-3
    # figure is exactly the envelope + padding, no auto layout
    assert fig.get_layout_engine() is None
    assert tuple(np.round(fig.get_size_inches(), 3)) == (round(fw, 3), round(fh, 3))


def test_spacer_and_text_cells_occupy_their_boxes():
    fig = Figure(facecolor="white")
    sp = Spacer(2.54, 2.54)  # 1 in
    a = PanelRef("a")
    layout = Row([sp, a])
    csp = SizedCell(sp, None, "spacer", 1.0, 1.0)
    ca = _plot_cell(fig, a, 1.5, 1.0)
    cells = {id(sp): csp, id(a): ca}
    measure_cells(fig, [csp, ca])
    place_tree(fig, layout, cells, gutter_in=0.0, pad_in=0.0)
    figw = fig.get_size_inches()[0]
    # panel starts 1 in (spacer) + its own left margin from the left edge
    assert abs(ca.ax.get_position().x0 * figw - (1.0 + ca.margins.left)) < 0.02
```
- [ ] Run to fail: `python3 -m pytest -q tests/test_compose_layout.py -k "measure or shared or ragged or spacer"` — `ImportError: cannot import name 'measure_cells'`.
- [ ] Implement (append to `layout.py`):
```python
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
```
- [ ] Run to pass: `python3 -m pytest -q tests/test_compose_layout.py` (all sizing + placement tests) + `python3 -m pytest -q tests/test_axes_placement.py` (engine neighbours untouched) + `ruff check dfxm/compose`.
- [ ] Update `docs/Codebase.md` (`layout.py` placement half: `measure_cells`, `place_tree`, alignment rules).
- [ ] Commit: `git add -A && git commit -m "feat(compose): measure/align/place engine — shared margins, ragged envelopes, absolute placement"`

---

### Task 7 — `render_recipe` + export + error/notes contract (`dfxm/compose/render.py`)

**Files**
- Create: `dfxm/compose/render.py`
- Create: `tests/test_compose_render.py`
- Modify: `docs/Codebase.md` (compose section completed; data-flow table row)

**Interfaces**

*Consumes:* everything from Tasks 1–6; `style_from_params`, `PlotStyle`, `add_colorbar`, `draw_scale_bar`, `box_drift_note`, `measured_box_in`, `auto_scale_bar_length_um`.

*Produces:*
```python
@dataclass
class ComposeResult:
    figure: Figure
    notes: list[str]
    n_panels: int          # PanelRef leaves in the layout
    n_rendered: int        # panels drawn with real data (placeholders excluded)
    axes_by_id: dict[str, object]   # panel id -> main axes (tests + GUI click-pick)

def render_recipe(
    recipe: FigureRecipe, style_overrides: dict | None = None, *, loader_cache: dict | None = None
) -> ComposeResult
    # (the spec's "(Figure, notes)" return is realized as ComposeResult.figure /
    #  .notes — the CLI exit-code contract additionally needs the panel counts)

def export_recipe(
    recipe, out_dir, *, formats=None, dpi=None, style_overrides=None, loader_cache=None
) -> tuple[list[str], ComposeResult]
    # saves <out_dir>/<safe(recipe.name)>.<fmt> per format; NO bbox_inches
```
Render pipeline (order matters): `validate_recipe` → style = `style_from_params({"plot_style": {**recipe.style, **(style_overrides or {})}}) or PlotStyle()` → load all panels (placeholders substituted) → `size_cells` → create one `Figure(facecolor="white")`, add an axes per non-spacer cell → shared-colorbar/scale-bar solver-cell transforms → draw panel contents (`colorbar=False` for members of a shared-bar group; `scale_bar` per compose mode; shared-x Cols: `show_xlabel` only on the last trace leaf) → assign + draw labels → `measure_cells` → `place_tree` → pinned-total-width correction (one rescale + re-measure + re-place when `compose.pinned_width_cm` set, implied scales noted) → text-cell contents → drift guard per panel.

**Steps**

- [ ] Write failing tests — `tests/test_compose_render.py` (reuse `_write_obl`, `_write_mosa`, `JOB` helpers — import them from `tests/test_compose_adapters.py` or duplicate; duplication is fine and keeps files standalone):
```python
"""render_recipe / export_recipe — dfxm.compose.render."""

import os

import h5py
import numpy as np
import pytest

from dfxm.common.errors import StageUserError
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    PanelSource,
    Row,
)
from dfxm.compose.render import export_recipe, render_recipe

# ... duplicate _write_obl / JOB from test_compose_adapters.py verbatim ...


def _two_panel_recipe(h5, **style):
    p1 = PanelDef(
        "a", PanelSource(h5, "slice_plane", {"volume_id": "raw_sum", "slice_name": "obl", "plane": 0})
    )
    p2 = PanelDef(
        "b", PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0})
    )
    return FigureRecipe(
        "demo",
        {"scale_um_per_cm": 10.0, "show_title": False, **style},
        ComposeStyle(),
        Row([PanelRef("a"), PanelRef("b")]),
        [p1, p2],
    )


def test_render_two_maps_exact_boxes_and_labels(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    res = render_recipe(_two_panel_recipe(h5))
    assert res.n_panels == 2 and res.n_rendered == 2
    from dfxm.common.plotting import measured_box_in

    for pid in ("a", "b"):
        w, h = measured_box_in(res.figure, res.axes_by_id[pid])
        assert abs(w - 20.0 / 10.0 / 2.54) < 0.005 * w
        assert abs(h - 16.0 / 10.0 / 2.54) < 0.005 * h
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert "A" in texts and "B" in texts  # auto label sequence
    assert not any("drift" in n or "scale is off" in n for n in res.notes)


def test_label_template_and_manual_override(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.compose.label_template = "(a)"
    r.panels[1].label = "X9"
    res = render_recipe(r)
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert "(a)" in texts and "X9" in texts and "(b)" not in texts


def test_missing_file_renders_placeholder_and_notes(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    r.panels[1].source.h5_path = str(tmp_path / "gone.h5")
    res = render_recipe(r)
    assert res.n_rendered == 1
    assert any("placeholder" in n for n in res.notes)


def test_shared_colorbar_unified_clim_and_single_bar(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    p1 = PanelDef(
        "a", PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0})
    )
    p2 = PanelDef(
        "b",
        PanelSource(h5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}),
        clim=(-5.0, 5.0),
    )
    r = FigureRecipe(
        "shared",
        {"scale_um_per_cm": 10.0, "show_title": False},
        ComposeStyle(),
        Col([PanelRef("a"), PanelRef("b")], shared_colorbar=True),
        [p1, p2],
    )
    res = render_recipe(r)
    ima = res.axes_by_id["a"].images[0]
    imb = res.axes_by_id["b"].images[0]
    assert (ima.norm.vmin, ima.norm.vmax) == (imb.norm.vmin, imb.norm.vmax)  # unified
    # exactly one colorbar axes beyond the two panel axes + label texts
    cbar_axes = [ax for ax in res.figure.axes if ax not in res.axes_by_id.values()]
    assert len(cbar_axes) == 1


def test_shared_colorbar_mixed_groups_refused(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)  # raw + strain groups
    r.layout = Col([PanelRef("a"), PanelRef("b")], shared_colorbar=True)
    with pytest.raises(StageUserError) as e:
        render_recipe(r)
    assert "quantity" in str(e.value) and e.value.hint


def test_shared_x_stack_bottom_labels_only(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    pt = [
        PanelDef(f"t{i}", PanelSource(h5, "profiles_trace", {"job": JOB, "field": vid}))
        for i, vid in enumerate(["raw_sum", "strain"])
    ]
    r = FigureRecipe(
        "stack",
        {"trace_scale_um_per_cm": 5.0, "trace_height_cm": 2.0, "show_title": False},
        ComposeStyle(),
        Col([PanelRef("t0"), PanelRef("t1")], shared_x=True),
        pt,
    )
    res = render_recipe(r)
    top, bot = res.axes_by_id["t0"], res.axes_by_id["t1"]
    assert top.get_xlabel() == "" and bot.get_xlabel() != ""
    assert not any(t.get_visible() for t in top.get_xticklabels())


def test_export_no_tightcrop_all_formats(tmp_path):
    h5 = _write_obl(tmp_path / "obl.h5")
    out = tmp_path / "out"
    out.mkdir()
    paths, res = export_recipe(
        _two_panel_recipe(h5), str(out), formats=("png", "pdf", "svg"), dpi=120
    )
    assert sorted(os.path.splitext(p)[1] for p in paths) == [".pdf", ".png", ".svg"]
    from PIL import Image  # pillow ships with matplotlib's test deps; if absent, use
    # matplotlib.image.imread instead:
    import matplotlib.image as mpimg

    img = mpimg.imread([p for p in paths if p.endswith(".png")][0])
    fw, fh = res.figure.get_size_inches()
    assert img.shape[1] == round(fw * 120) and img.shape[0] == round(fh * 120)  # exact canvas
```
(Drop the PIL import — use only `matplotlib.image.imread` as shown; the assertion is the exact-canvas pin that proves no tight-crop.)
- [ ] Run to fail: `python3 -m pytest -q tests/test_compose_render.py` — `ModuleNotFoundError: dfxm.compose.render`.
- [ ] Implement `dfxm/compose/render.py`. Key blocks (full pipeline per the Interfaces order):
```python
"""Render a FigureRecipe into one Figure at exact physical size; export it.

No tight-crop anywhere: the solver owns all margins, so the saved canvas IS
the figure geometry. No matplotlib auto-layout."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

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
from .recipe import Col, PanelRef, Row, Spacer, TextCell, validate_recipe

_IN_PER_CM = 1.0 / 2.54


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
```
Label drawing (before `measure_cells`, so labels count toward margins):
```python
def _draw_label(ax, text, style, compose):
    size = 12.0 * style.font_scale * compose.label_font_scale
    ax.annotate(
        text, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0.0, 4.0),
        textcoords="offset points", ha="left", va="bottom",
        fontsize=size, fontweight="bold", annotation_clip=False,
    )
```
Shared bars — collect groups by walking the layout for `shared_colorbar` nodes: members = all `PanelRef` leaves below; compatibility check `len({data.group for members}) == 1` else `StageUserError("shared colorbar mixes quantity groups …", hint="Group only panels of one quantity (e.g. all strain) under a shared bar, or give each its own bar.")`; unified clim = `node.shared_clim` or `(min(vmins), max(vmaxs))` over member panel data with per-panel `clim` overrides applied first; members drawn with `colorbar=False` and their `PanelDef.clim` forced to the unified pair; the bar is a synthetic solver cell — wrap the group node internally: `Row([node, _bar_leaf])` (for a `Col` group) / `Col`-analog, where `_bar_leaf` is a `Spacer`-typed marker whose `SizedCell` gets `w_in = style.colorbar_fraction * first_member.w_in + 0.1`, `h_in = ` group content height, `ax = fig.add_axes(...)` and after placement `add_colorbar(fig, member_im, member_ax, cbar_label, style, group=grp, cax=bar_ax)`. Shared scale bars per `compose.scale_bar_mode`: `"per-panel"` → `scale_bar=None` (style flag) on every map draw; `"one-panel"` → `scale_bar=True` only for `compose.scale_bar_panel` (missing/unknown id → `StageUserError` with hint), `False` elsewhere; `"gutter"` → all panels `scale_bar=False`, plus a gutter cell (bare axes, `ax.set_axis_off()`, xlim spanning `gutter_in * 2.54 * shared_scale` µm, `draw_scale_bar(ax, None, style=style, fixed_scale_um_per_cm=shared_scale)` with `loc` forced sensible via the style) — refuse when panels' effective µm/cm differ (`StageUserError(..., hint="A shared scale bar needs every map at one µm/cm — remove per-panel scale overrides or use per-panel bars.")`). `fixed_scale_um_per_cm` passed to each map/ref draw = that panel's effective µm/cm (`data.ext_x_um / (cell.w_in * 2.54)`), so bar thickness pins true points. Shared-x Cols: before measuring, every trace leaf except the last gets `show_xlabel=False` at draw time (the adapter already suppresses labels+xlabel). Drift guard at the end:
```python
    for pid, ax in result.axes_by_id.items():
        cell = cell_by_pid[pid]
        note = box_drift_note(f"panel {pid}", fig, ax, cell.w_in, cell.h_in)
        if note:
            result.notes.append(note)
```
Pinned total width: after the first `place_tree`, when `compose.pinned_width_cm` set — `factor = (compose.pinned_width_cm * _IN_PER_CM) / fig_w`; multiply every cell's `w_in`/`h_in` by `factor`, append per-panel implied-scale notes (`old_eff / factor`), re-run `measure_cells` + `place_tree` once, and append a drift note if the final width misses the pin by > 2 %.
`export_recipe`:
```python
def export_recipe(recipe, out_dir, *, formats=None, dpi=None,
                  style_overrides=None, loader_cache=None):
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
```
- [ ] Run to pass: `python3 -m pytest -q tests/test_compose_render.py tests/test_compose_layout.py tests/test_compose_adapters.py` + `ruff check dfxm/compose`.
- [ ] Update `docs/Codebase.md`: complete the `dfxm/compose` section (all five modules once Task 8 lands the CLI; here: `render.py` — ComposeResult, render pipeline order, shared-bar rules, drift guard) and add a "figure recipes" row to the Data & artifact flow table (`recipe.json` + stage h5s → `dfxm.compose.render` → `<name>.png/pdf/svg`).
- [ ] Commit: `git add -A && git commit -m "feat(compose): render_recipe + export — labels, shared bars, placeholders, drift guard, no tight-crop"`

---

### Task 8 — Headless CLI (`dfxm/compose/__main__.py`) + Usage docs

**Files**
- Create: `dfxm/compose/__main__.py`
- Create: `tests/test_compose_cli.py`
- Modify: `docs/Usage.md` ("Running without the GUI (CLI)" section + a new "Figure builder" chapter stub with the recipe-file + CLI workflow)
- Modify: `docs/Codebase.md` (`__main__.py` entry)

**Interfaces**

*Consumes:* `recipe_from_json`, `export_recipe`, `StageUserError`.

*Produces:*
```python
def _main(argv: list[str] | None = None) -> int
# usage: python3 -m dfxm.compose render recipe.json -o OUTDIR [--formats png,pdf,svg] [--dpi N]
# exit 0: at least one panel rendered (placeholder notes still printed)
# exit 1: rendered figure but NO panel had data (all placeholders)
# exit 2: StageUserError (message + hint printed to stderr)
```

**Steps**

- [ ] Write failing tests — `tests/test_compose_cli.py`:
```python
"""Headless CLI — python3 -m dfxm.compose render."""

import json
import os

from dfxm.compose.__main__ import _main
from dfxm.compose.recipe import recipe_to_json

# ... duplicate _write_obl / JOB / _two_panel_recipe helpers from test_compose_render.py ...


def test_cli_renders_and_exits_zero(tmp_path, capsys):
    h5 = _write_obl(tmp_path / "obl.h5")
    rp = tmp_path / "r.json"
    rp.write_text(recipe_to_json(_two_panel_recipe(h5)))
    out = tmp_path / "out"
    rc = _main(["render", str(rp), "-o", str(out), "--formats", "png"])
    assert rc == 0
    assert os.path.exists(out / "demo.png")


def test_cli_all_placeholders_exits_one(tmp_path, capsys):
    h5 = _write_obl(tmp_path / "obl.h5")
    r = _two_panel_recipe(h5)
    for p in r.panels:
        p.source.h5_path = str(tmp_path / "gone.h5")
    rp = tmp_path / "r.json"
    rp.write_text(recipe_to_json(r))
    rc = _main(["render", str(rp), "-o", str(tmp_path / "out")])
    assert rc == 1
    assert "placeholder" in capsys.readouterr().out


def test_cli_bad_recipe_exits_two(tmp_path, capsys):
    rp = tmp_path / "bad.json"
    rp.write_text("{not json")
    rc = _main(["render", str(rp), "-o", str(tmp_path / "out")])
    assert rc == 2
    assert "hint" in capsys.readouterr().err.lower()
```
- [ ] Run to fail: `python3 -m pytest -q tests/test_compose_cli.py` — module absent.
- [ ] Implement `dfxm/compose/__main__.py`:
```python
"""Headless recipe renderer: python3 -m dfxm.compose render recipe.json -o outdir."""

from __future__ import annotations

import argparse
import os
import sys


def _main(argv: list[str] | None = None) -> int:
    from ..common.errors import StageUserError
    from .recipe import recipe_from_json
    from .render import export_recipe

    ap = argparse.ArgumentParser(prog="python3 -m dfxm.compose")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="render a figure recipe to PNG/PDF/SVG")
    r.add_argument("recipe", help="path to a recipe .json")
    r.add_argument("-o", "--out", required=True, help="output directory")
    r.add_argument("--formats", default="", help="comma list, e.g. png,pdf,svg (default: style)")
    r.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        with open(args.recipe, encoding="utf-8") as fh:
            recipe = recipe_from_json(fh.read(), base_dir=os.path.dirname(os.path.abspath(args.recipe)))
        fmts = tuple(f for f in args.formats.split(",") if f) or None
        paths, res = export_recipe(recipe, args.out, formats=fmts, dpi=args.dpi)
    except StageUserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.hint:
            print(f"hint: {exc.hint}", file=sys.stderr)
        return 2
    for note in res.notes:
        print(f"note: {note}")
    for path in paths:
        print(f"wrote {path}")
    if res.n_rendered == 0:
        print("error: no panel rendered (all placeholders)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```
- [ ] Run to pass: `python3 -m pytest -q tests/test_compose_cli.py` + `ruff check dfxm/compose`.
- [ ] Docs (same change): `docs/Usage.md` — add `python3 -m dfxm.compose render …` to the CLI section and start the "Figure builder" chapter (concepts: recipe file, panels/rows/cols, physical scales, placeholder behaviour, CLI re-render; the GUI workflow lands in Task 13). `docs/Codebase.md` — `__main__.py` entry + exit-code contract.
- [ ] Commit: `git add -A && git commit -m "feat(compose): headless CLI renderer with exit-code contract + docs"`

---

### Task 9 — Acceptance figures 1 & 2 (spec §Acceptance)

**Files**
- Create: `tests/test_compose_acceptance.py`

**Interfaces**

*Consumes:* everything from Tasks 1–7 (no new production code expected; any bug found here is fixed in the module that owns it, with the fix noted in the commit).

**Steps**

- [ ] Write the two acceptance tests (they may fail initially — this is the proving task):
```python
"""Spec acceptance figures — built headless from synthetic h5 fixtures."""

import h5py
import numpy as np

from dfxm.common.plotting import measured_box_in
from dfxm.compose.recipe import (
    Col,
    ComposeStyle,
    FigureRecipe,
    PanelDef,
    PanelRef,
    PanelSource,
    Row,
)
from dfxm.compose.render import render_recipe


def _write_slices_two_planes(path):
    """oblique_slices.h5: mosa CoM + strain volumes, two single-plane slices."""
    u = np.linspace(-10.0, 10.0, 41)
    v = np.linspace(-8.0, 8.0, 33)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (("mosa_com_chi", "mosa_com", "magma"), ("strain", "strain", "RdBu_r")):
            g = f.create_group(vid)
            g.attrs.update(
                kind=kind, cbar_label="value", cmap=cmap, title=vid, vmin=-10.0, vmax=10.0
            )
            for sname in ("slice_a", "slice_b"):
                sg = g.create_group(sname)
                sg.create_dataset("slices", data=(uu + vv)[None, ...].astype("f4"))
                sg.create_dataset("u_um", data=u)
                sg.create_dataset("v_um", data=v)
                sg.create_dataset("offsets_um", data=np.array([0.0]))
    return str(path)


def _write_profiles_three_fields(path):
    """oblique_slices.h5 with three fields on one slice (for figure 2)."""
    u = np.linspace(-20.0, 20.0, 81)
    v = np.linspace(-15.0, 15.0, 61)
    uu, vv = np.meshgrid(u, v)
    with h5py.File(path, "w") as f:
        for vid, kind, cmap in (
            ("mosa_com_chi", "mosa_com", "magma"),
            ("strain", "strain", "RdBu_r"),
            ("raw_mosa_sum", "raw_mosa_sum", "gray"),
        ):
            g = f.create_group(vid)
            g.attrs.update(
                kind=kind, cbar_label="value", cmap=cmap, title=vid, vmin=-40.0, vmax=40.0
            )
            sg = g.create_group("obl")
            sg.create_dataset("slices", data=(uu + vv)[None, ...].astype("f4"))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0]))
    return str(path)


def test_acceptance_figure_1_two_by_two_grid(tmp_path):
    """2×2: cols = mosaicity CoM | strain, rows = two slices; labels A B / C D."""
    h5 = _write_slices_two_planes(tmp_path / "obl.h5")

    def sp(pid, vid, sname):
        return PanelDef(
            pid,
            PanelSource(h5, "slice_plane", {"volume_id": vid, "slice_name": sname, "plane": 0}),
        )

    scale = 10.0
    recipe = FigureRecipe(
        "fig1",
        {"scale_um_per_cm": scale, "show_title": False},
        ComposeStyle(label_template="A"),
        Col(
            [
                Row([PanelRef("p00"), PanelRef("p01")]),
                Row([PanelRef("p10"), PanelRef("p11")]),
            ]
        ),
        [
            sp("p00", "mosa_com_chi", "slice_a"),
            sp("p01", "strain", "slice_a"),
            sp("p10", "mosa_com_chi", "slice_b"),
            sp("p11", "strain", "slice_b"),
        ],
    )
    res = render_recipe(recipe)
    assert res.n_rendered == 4
    # one µm/cm on every map: every box exactly 20/scale × 16/scale cm
    for pid in ("p00", "p01", "p10", "p11"):
        w, h = measured_box_in(res.figure, res.axes_by_id[pid])
        assert abs(w - 20.0 / scale / 2.54) < 0.005 * w, pid
        assert abs(h - 16.0 / scale / 2.54) < 0.005 * h, pid
    # row-major labels A B / C D
    texts = [t.get_text() for ax in res.figure.axes for t in ax.texts]
    assert {"A", "B", "C", "D"} <= set(texts)
    # grid alignment: row-mates share y, column-mates share x
    pos = {pid: res.axes_by_id[pid].get_position() for pid in res.axes_by_id}
    assert abs(pos["p00"].y0 - pos["p01"].y0) < 1e-6
    assert abs(pos["p10"].y0 - pos["p11"].y0) < 1e-6
    assert abs(pos["p00"].x0 - pos["p10"].x0) < 1e-6
    assert abs(pos["p01"].x0 - pos["p11"].x0) < 1e-6
    assert not any("scale is off" in n for n in res.notes)


def test_acceptance_figure_2_ragged_dual_scale(tmp_path):
    """Ragged 3 columns; maps at scale_um_per_cm, trace stacks at a DIFFERENT
    trace_scale_um_per_cm — both honoured exactly in one canvas."""
    h5 = _write_profiles_three_fields(tmp_path / "obl.h5")
    job_a = {"name": "obl", "offset_um": 0.0, "start_uv": [-8.0, -6.0], "end_uv": [8.0, 6.0]}
    job_b = {"name": "obl", "offset_um": 0.0, "start_uv": [-15.0, -9.0], "end_uv": [15.0, 9.0]}
    len_a = float(np.hypot(16.0, 12.0))  # 20 µm
    len_b = float(np.hypot(30.0, 18.0))  # ~34.99 µm (B longer than A)
    fields = ["mosa_com_chi", "strain", "raw_mosa_sum"]

    panels = [
        PanelDef(
            "A1",
            PanelSource(h5, "profiles_ref", {"job": job_a, "field": "mosa_com_chi"}),
            roi=(10, 50, 10, 60),
            label="A1",
        ),
        PanelDef(
            "B1",
            PanelSource(h5, "profiles_ref", {"job": job_b, "field": "mosa_com_chi"}),
            roi=(10, 50, 10, 60),
            label="B1",
        ),
    ]
    for tag, job in (("a", job_a), ("b", job_b)):
        for vid in fields:
            panels.append(
                PanelDef(
                    f"t_{tag}_{vid}",
                    PanelSource(h5, "profiles_trace", {"job": job, "field": vid}),
                )
            )

    map_scale, trace_scale, trace_h_cm = 5.0, 2.0, 2.0
    assert map_scale != trace_scale  # the spec's hard requirement
    recipe = FigureRecipe(
        "fig2",
        {
            "scale_um_per_cm": map_scale,
            "trace_scale_um_per_cm": trace_scale,
            "trace_height_cm": trace_h_cm,
            "show_title": False,
        },
        ComposeStyle(),
        Row(
            [
                Col([PanelRef("A1"), PanelRef("B1")]),
                Col(
                    [PanelRef(f"t_a_{v}") for v in fields], shared_x=True, group_label="A2"
                ),
                Col(
                    [PanelRef(f"t_b_{v}") for v in fields], shared_x=True, group_label="B2"
                ),
            ]
        ),
        panels,
    )
    res = render_recipe(recipe)
    assert res.n_rendered == 8
    fig = res.figure

    # maps honour the MAP scale exactly (ROI crop: 50 cols × 0.5 µm/px = 25 µm,
    # 40 rows × 0.5 = 20 µm — u pitch 40/80=0.5, v pitch 30/60=0.5)
    for pid in ("A1", "B1"):
        w, h = measured_box_in(fig, res.axes_by_id[pid])
        assert abs(w - 25.0 / map_scale / 2.54) < 0.005 * w, pid
        assert abs(h - 20.0 / map_scale / 2.54) < 0.005 * h, pid

    # traces honour the TRACE scale exactly, height = trace_height_cm
    for tag, length in (("a", len_a), ("b", len_b)):
        for vid in fields:
            w, h = measured_box_in(fig, res.axes_by_id[f"t_{tag}_{vid}"])
            assert abs(w - length / trace_scale / 2.54) < 0.005 * w, (tag, vid)
            assert abs(h - trace_h_cm / 2.54) < 0.005 * h, (tag, vid)

    # both scales in ONE canvas, asserted against the same figure object
    wa, _ = measured_box_in(fig, res.axes_by_id["t_a_strain"])
    wm, _ = measured_box_in(fig, res.axes_by_id["A1"])
    assert abs(wa * 2.54 * trace_scale - len_a) < 0.005 * len_a
    assert abs(wm * 2.54 * map_scale - 25.0) < 0.005 * 25.0

    # group labels A2/B2 present; no titles anywhere
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert "A2" in texts and "B2" in texts
    assert all(ax.get_title() == "" for ax in res.axes_by_id.values())

    # shared distance axis: bottom-only x labels within each stack
    for tag in ("a", "b"):
        for vid in fields[:-1]:
            assert res.axes_by_id[f"t_{tag}_{vid}"].get_xlabel() == ""
        assert res.axes_by_id[f"t_{tag}_{fields[-1]}"].get_xlabel() != ""

    # within each stack: identical box width/height and left-aligned x0
    for tag in ("a", "b"):
        xs = {round(res.axes_by_id[f"t_{tag}_{v}"].get_position().x0, 5) for v in fields}
        assert len(xs) == 1, tag

    # ragged padding: A stack narrower than B stack, yet column 3 starts at one x
    # for all B panels (column edges align — the A column envelope pads the gap)
    xa = max(res.axes_by_id[f"t_a_{v}"].get_position().x1 for v in fields)
    xb = {round(res.axes_by_id[f"t_b_{v}"].get_position().x0, 5) for v in fields}
    assert len(xb) == 1 and min(xb) > xa

    assert not any("scale is off" in n for n in res.notes)
```
- [ ] Run: `python3 -m pytest -q tests/test_compose_acceptance.py`. Debug any failure with `superpowers:systematic-debugging`; fixes land in the owning module (`layout.py`/`render.py`/`adapters.py`) with their own minimal regression test when the bug was not covered.
- [ ] Full Phase A gate: `python3 -m pytest -q` (whole suite) + `ruff check . && ruff format .`.
- [ ] Commit: `git add -A && git commit -m "test(compose): acceptance figures 1+2 — exact dual-scale grid proven headless"`

---

## Phase B — GUI (`gui/figure_builder.py`)

### Task 10 — Panel picker + builder window skeleton + recipe file I/O

**Files**
- Create: `gui/widgets/panel_picker.py`
- Create: `gui/figure_builder.py`
- Create: `tests/test_gui_figure_builder.py`
- Modify: `docs/Codebase.md` (Layer 2 gains both modules)

**Interfaces**

*Consumes (verified):* stage catalogs — `strain.replot_catalog(h5) -> list[ReplotGroup]`, `mosaicity.replot_catalog(h5)`, `rocking.replot_catalog(h5)`, `slices.replot_catalog(h5) -> list[ReplotEntry]`, `profiles.replot_catalog(h5, jobs) -> list[ReplotJobEntry]`; `gui.bindings.experiment_overrides(stage_name, exp)` (h5 pre-fill paths); recipe API (Task 1). Qt test convention: module-level `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` + `_app = QApplication.instance() or QApplication([])` (pattern from `tests/test_gui_replot_dialog.py`).

*Produces:*
```python
# gui/widgets/panel_picker.py
class AddPanelDialog(QDialog):
    def __init__(self, defaults: dict[str, dict], parent=None) -> None
        # defaults: stage -> {"h5": str, "sx": float, "sy": float, "jobs": list[dict]}
    selected_panels: list[PanelDef]   # filled on accept()
    def _build_panels(self) -> list[PanelDef]   # testable without exec()

# gui/figure_builder.py
class FigureBuilderWindow(QMainWindow):
    def __init__(self, defaults_provider, style: PlotStyle, parent=None) -> None
        # defaults_provider: Callable[[], dict[str, dict]] — the main window closes
        # over its experiment + stage forms to build the AddPanelDialog defaults
    def recipe(self) -> FigureRecipe
    def load_recipe_file(self, path: str) -> None
    def save_recipe_file(self, path: str) -> None
    def is_dirty(self) -> bool
    # outline ops (all testable): add_row(), add_col(), add_spacer(), add_text(),
    # add_panels(list[PanelDef]), move_selected(delta), delete_selected(),
    # toggle_group_selected(), set_selected_label(text)
```
Outline: a `QTreeWidget` mirroring the layout tree; each item stores a reference to its recipe node in `Qt.ItemDataRole.UserRole`; structural edits mutate the `FigureRecipe` and rebuild the tree (structured editing only — no canvas dragging). Window title: `"Figure builder — {name}{' *' if dirty else ''}"`.

**Steps**

- [ ] Write failing tests — `tests/test_gui_figure_builder.py`:
```python
"""Figure-builder window + panel picker (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.plotting import PlotStyle  # noqa: E402
from dfxm.compose.recipe import PanelDef, PanelSource, Row  # noqa: E402
from gui.figure_builder import FigureBuilderWindow  # noqa: E402


def _win():
    return FigureBuilderWindow(lambda: {}, PlotStyle(scale_um_per_cm=10.0))


def _panel(pid):
    return PanelDef(pid, PanelSource("/x.h5", "map_layer", {"stage": "strain", "z": 0}))


def test_add_panels_and_outline_roundtrip():
    w = _win()
    w.add_panels([_panel("a"), _panel("b")])
    r = w.recipe()
    assert [p.id for p in r.panels] == ["a", "b"]
    assert w.is_dirty()


def test_move_and_delete_edit_the_recipe():
    w = _win()
    w.add_panels([_panel("a"), _panel("b")])
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(1))  # select "b"
    w.move_selected(-1)
    assert [getattr(x, "panel_id", None) for x in _row(w).children] == ["b", "a"]
    w._tree.setCurrentItem(w._tree.topLevelItem(0).child(0))
    w.delete_selected()
    assert [getattr(x, "panel_id", None) for x in _row(w).children] == ["a"]


def _row(w):
    layout = w.recipe().layout
    assert isinstance(layout, Row)
    return layout


def test_save_open_round_trip_and_dirty_state(tmp_path):
    w = _win()
    w.add_panels([_panel("a")])
    path = str(tmp_path / "r.json")
    w.save_recipe_file(path)
    assert not w.is_dirty()
    w2 = _win()
    w2.load_recipe_file(path)
    assert [p.id for p in w2.recipe().panels] == ["a"]
    assert not w2.is_dirty()
    assert "r" in w2.windowTitle() or w2.recipe().name


def test_panel_picker_builds_slice_panel_defs(tmp_path):
    import h5py
    import numpy as np

    from gui.widgets.panel_picker import AddPanelDialog

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((2, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0, 1, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 1, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))
    dlg = AddPanelDialog({"slices": {"h5": str(h5), "sx": 0.5, "sy": 0.5, "jobs": []}})
    dlg._stage.setCurrentText("slices")
    dlg._reload()
    dlg._check_all()  # helper that checks every leaf, mirrors replot select_all
    panels = dlg._build_panels()
    assert len(panels) == 2
    assert panels[0].source.kind == "slice_plane"
    assert panels[0].source.selector["volume_id"] == "strain"
```
- [ ] Run to fail: `python3 -m pytest -q tests/test_gui_figure_builder.py` — `ModuleNotFoundError: gui.figure_builder`.
- [ ] Implement `gui/widgets/panel_picker.py`: stage combo (`strain/mosaicity/rocking/slices/profiles`), h5 field + Browse (pre-filled from `defaults[stage]["h5"]`), Load button, a check-tree built per stage from the core catalog functions (strain/mosaicity/rocking: group → layer items; slices: `(vid, sname)` → plane items; profiles: job → `reference` + per-field `trace` items), `_check_all()` helper, `_build_panels()` translating checked leaves into `PanelDef`s with unique auto-ids (`f"{stage}_{n}"` via a counter) and selectors per Task 4's shapes (map layers carry `sx`/`sy` from `defaults`; profiles items carry the job dict from `defaults["profiles"]["jobs"]`). Catalog errors → status label, never a crash (pattern from `ReplotDialog._reload`).
Implement `gui/figure_builder.py`: `QMainWindow` with a left `QSplitter` pane holding an "Add panels…" button (opens `AddPanelDialog(self._defaults_provider())`; on accept, `add_panels(dlg.selected_panels)`), the outline `QTreeWidget` (`self._tree`), and outline-edit buttons (Row/Col/Spacer/Text/↑/↓/Delete/Group/Label…). Internal state: `self._recipe = FigureRecipe("untitled", {}, ComposeStyle(), Row([]), [])`; every mutator sets `self._dirty = True` and calls `self._rebuild_tree()` + `self._update_title()`. `add_panels` appends `PanelDef`s to `panels` and `PanelRef`s to the current container (selected Row/Col, else root). `save_recipe_file` uses `recipe_to_json(self._recipe, base_dir=os.path.dirname(path))`, clears dirty; `load_recipe_file` uses `recipe_from_json(..., base_dir=...)`, sets `self._recipe.name` from the file stem when the recipe has no name. `closeEvent` warns on dirty (QMessageBox Save/Discard/Cancel). The center/right panes are placeholders (`QLabel("preview")`, filled by Tasks 11–12).
- [ ] Run to pass: `python3 -m pytest -q tests/test_gui_figure_builder.py` + `ruff check gui`.
- [ ] Update `docs/Codebase.md` Layer 2 (both modules, public methods).
- [ ] Commit: `git add -A && git commit -m "feat(gui): figure-builder window skeleton — outline editing, panel picker, recipe I/O"`

---

### Task 11 — Cached preview with debounce + Refresh + notes bar + click-to-select

**Files**
- Modify: `gui/figure_builder.py`
- Modify: `tests/test_gui_figure_builder.py` (append)
- Modify: `docs/Usage.md` + `docs/Codebase.md`

**Interfaces**

*Consumes:* `render_recipe(recipe, style_overrides=None, *, loader_cache=None) -> ComposeResult` (Task 7); `FigureCanvasQTAgg` (pattern from `gui/widgets/mpl_canvas.py`); `QTimer`.

*Produces (methods on `FigureBuilderWindow`):*
```python
def schedule_preview(self) -> None      # 300 ms single-shot debounce → render_now()
def render_now(self) -> ComposeResult | None   # renders from self._cache; errors → notes bar
def refresh_data(self) -> None          # clears self._cache, then render_now()
# self._notes_label: QLabel under the preview (implied-scale / drift / placeholder notes,
#   StageUserError message+hint on refused configs)
# clicking a panel in the preview selects its outline node (result.axes_by_id reverse map)
```
The preview hosts the *rendered* Figure directly: on each render the old `FigureCanvasQTAgg` is replaced by a new one wrapping `result.figure` (the composed figure is placed absolutely — it must never be re-laid-out by a display canvas; a fresh canvas per render behind the debounce is cheap and keeps WYSIWYG exact). Every recipe/style mutator calls `schedule_preview()`.

**Steps**

- [ ] Append failing tests to `tests/test_gui_figure_builder.py`:
```python
# -- preview + cache ----------------------------------------------------------
def _obl_recipe_panels(tmp_path):
    import h5py
    import numpy as np

    h5 = tmp_path / "obl.h5"
    with h5py.File(h5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=np.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=np.linspace(0.0, 2.0, 5))
        sg.create_dataset("v_um", data=np.linspace(0.0, 1.5, 4))
        sg.create_dataset("offsets_um", data=np.array([0.0]))
    from dfxm.compose.recipe import PanelDef, PanelSource

    return [
        PanelDef(
            "a",
            PanelSource(
                str(h5), "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}
            ),
        )
    ]


def test_render_now_populates_preview_and_notes(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    res = w.render_now()
    assert res is not None and res.n_rendered == 1
    assert w._canvas is not None  # a live FigureCanvasQTAgg wrapping res.figure
    assert w._canvas.figure is res.figure


def test_render_error_lands_in_notes_bar_not_crash(tmp_path):
    w = FigureBuilderWindow(lambda: {}, PlotStyle())  # NO scale anywhere
    w.add_panels(_obl_recipe_panels(tmp_path))
    res = w.render_now()
    assert res is None
    assert "scale" in w._notes_label.text().lower()
    assert "hint" in w._notes_label.text().lower() or "Set Scale" in w._notes_label.text()


def test_cache_survives_file_deletion_until_refresh(tmp_path):
    w = _win()
    panels = _obl_recipe_panels(tmp_path)
    w.add_panels(panels)
    assert w.render_now() is not None
    os.remove(panels[0].source.h5_path)
    res2 = w.render_now()  # served from cache
    assert res2 is not None and res2.n_rendered == 1
    w.refresh_data()  # cache cleared -> placeholder now
    res3 = w.render_now()
    assert res3 is not None and res3.n_rendered == 0
    assert "placeholder" in w._notes_label.text()


def test_click_preview_selects_outline_node(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    res = w.render_now()
    ax = res.axes_by_id["a"]
    w._on_preview_pick(ax)  # the slot the mpl button_press handler calls
    item = w._tree.currentItem()
    assert item is not None and "a" in item.text(0)
```
- [ ] Run to fail: `python3 -m pytest -q tests/test_gui_figure_builder.py -k "preview or cache or pick"` — `AttributeError: 'FigureBuilderWindow' object has no attribute 'render_now'`.
- [ ] Implement in `gui/figure_builder.py`:
```python
# in __init__:
self._cache: dict = {}
self._canvas = None
self._result = None
self._preview_host = QWidget()
self._preview_layout = QVBoxLayout(self._preview_host)
self._preview_layout.setContentsMargins(0, 0, 0, 0)
self._notes_label = QLabel("")
self._notes_label.setWordWrap(True)
self._refresh_btn = QPushButton("Refresh data")
self._refresh_btn.clicked.connect(self.refresh_data)
self._debounce = QTimer(self)
self._debounce.setSingleShot(True)
self._debounce.setInterval(300)
self._debounce.timeout.connect(self.render_now)


def schedule_preview(self) -> None:
    self._debounce.start()


def refresh_data(self) -> None:
    self._cache.clear()
    self.render_now()


def render_now(self):
    from dfxm.common.errors import StageUserError
    from dfxm.compose.render import render_recipe

    if not self._recipe.panels:
        self._notes_label.setText("add panels to preview")
        return None
    try:
        result = render_recipe(self._recipe, loader_cache=self._cache)
    except StageUserError as exc:
        hint = f"  Hint: {exc.hint}" if exc.hint else ""
        self._notes_label.setText(f"cannot render: {exc}{hint}")
        return None
    except Exception as exc:  # noqa: BLE001 — preview must never crash the window
        self._notes_label.setText(f"render failed: {exc}")
        return None
    self._show_figure(result.figure)
    self._result = result
    self._notes_label.setText("; ".join(result.notes) if result.notes else "")
    return result


def _show_figure(self, figure) -> None:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

    if self._canvas is not None:
        self._preview_layout.removeWidget(self._canvas)
        self._canvas.deleteLater()
    self._canvas = FigureCanvasQTAgg(figure)
    self._canvas.mpl_connect("button_press_event", self._on_preview_click)
    self._preview_layout.addWidget(self._canvas, 1)
    self._canvas.draw_idle()


def _on_preview_click(self, event) -> None:
    if event.inaxes is not None:
        self._on_preview_pick(event.inaxes)


def _on_preview_pick(self, ax) -> None:
    if self._result is None:
        return
    for pid, panel_ax in self._result.axes_by_id.items():
        if panel_ax is ax:
            self._select_outline_panel(pid)
            return
```
`_select_outline_panel(pid)` walks the tree items comparing the stored node's `panel_id`; `_show_figure` replaces the canvas widget wholesale (no reuse of a themed `MplCanvas` — exports and preview must show the white publication figure exactly, matching the "exports stay white" convention). Wire `schedule_preview()` into every mutator from Task 10 (`add_panels`, `move_selected`, `delete_selected`, `toggle_group_selected`, `set_selected_label`, load).
- [ ] Run to pass: `python3 -m pytest -q tests/test_gui_figure_builder.py` + `ruff check gui`.
- [ ] Docs (same change): `docs/Usage.md` Figure-builder chapter gains the preview/Refresh/notes-bar paragraphs; `docs/Codebase.md` figure_builder entry gains the preview/cache methods.
- [ ] Commit: `git add -A && git commit -m "feat(gui): figure-builder preview — cached debounced render, Refresh, notes bar, click-to-select"`

---

### Task 12 — Right pane: style controls, compose knobs, per-node overrides, export

**Files**
- Modify: `gui/figure_builder.py`
- Modify: `tests/test_gui_figure_builder.py` (append)
- Modify: `docs/Usage.md` + `docs/Codebase.md`

**Interfaces**

*Consumes (verified):* `gui.widgets.export_dialog.StyleControls(style: PlotStyle, parent=None)` — mutates the bound `PlotStyle` in place, emits `changed`, `sync_from_style()` re-pushes values; `dataclasses.asdict`; `export_recipe` (Task 7); recipe override fields (Task 1).

*Produces (methods/attrs on `FigureBuilderWindow`):*
```python
# self._style: PlotStyle — the working style object bound to StyleControls;
#   on every `changed` emission it is serialized into recipe.style:
def _sync_style_to_recipe(self) -> None   # recipe.style = asdict(self._style); dirty; schedule
# compose knobs (QLineEdit/QComboBox/QDoubleSpinBox bound to recipe.compose):
#   label template, label font scale, gutter cm, padding cm, scale-bar mode
#   (+ panel id combo for one-panel mode), pinned total width
# per-node override editor for the selected outline node:
def _apply_panel_overrides(self, panel: PanelDef, values: dict) -> None
#   values keys: roi ("r0,r1,c0,c1" text -> tuple|None), clim ("lo,hi" halves ok),
#   cmap (combo, "" = follow style), label, show_title (tri-state), scale_um_per_cm,
#   colorbar (tri-state)
def export_now(self) -> None   # dir picker + export_recipe; result paths/notes in notes bar
```

**Steps**

- [ ] Append failing tests:
```python
# -- style pane + overrides + export ------------------------------------------
def test_style_controls_edit_lands_in_recipe_style(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w._style.font_scale = 2.0
    w._sync_style_to_recipe()
    assert w.recipe().style["font_scale"] == 2.0
    assert w.is_dirty()


def test_compose_knob_edits_land_in_recipe(tmp_path):
    w = _win()
    w._compose_template.setText("(A)")
    w._on_compose_edited()
    assert w.recipe().compose.label_template == "(A)"


def test_panel_override_editor_applies(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    panel = w.recipe().panels[0]
    w._apply_panel_overrides(
        panel,
        {"roi": "0,3,1,4", "clim": "-2,", "cmap": "viridis", "label": "Z1",
         "show_title": None, "scale_um_per_cm": 4.0, "colorbar": False},
    )
    assert panel.roi == (0, 3, 1, 4)
    assert panel.clim == (-2.0, None)
    assert panel.cmap == "viridis" and panel.label == "Z1"
    assert panel.scale_um_per_cm == 4.0 and panel.colorbar is False


def test_export_now_writes_files(tmp_path, monkeypatch):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    out = tmp_path / "out"
    monkeypatch.setattr(
        "gui.figure_builder.QFileDialog.getExistingDirectory", lambda *a, **k: str(out)
    )
    w.export_now()
    assert os.path.exists(out / "untitled.png")
    assert "wrote" in w._notes_label.text()
```
- [ ] Run to fail: `python3 -m pytest -q tests/test_gui_figure_builder.py -k "style_controls or compose_knob or override or export_now"` — missing attributes.
- [ ] Implement. Right pane = `QScrollArea` over a `QVBoxLayout` with three sections. (1) *Style*: `StyleControls(self._style)` where `self._style` is built in `__init__` from the constructor's `PlotStyle` via `dataclasses.replace` (an independent copy — builder edits must not mutate the app-wide session style); connect `controls.changed → self._sync_style_to_recipe`:
```python
def _sync_style_to_recipe(self) -> None:
    from dataclasses import asdict

    self._recipe.style = asdict(self._style)
    self._dirty = True
    self._update_title()
    self.schedule_preview()
```
On `load_recipe_file`, rebuild `self._style` from the loaded dict (`style_from_params({"plot_style": self._recipe.style}) or PlotStyle()`) and call `self._controls.set_style(self._style)`. (2) *Compose*: widgets bound to `recipe.compose` (`self._compose_template = QLineEdit`, `QDoubleSpinBox` for label font scale/gutter/padding/pinned width — pinned width 0 = off → `None`, `QComboBox` for scale-bar mode over `SCALE_BAR_MODES`, `QComboBox` of current panel ids for the designated panel), one `_on_compose_edited()` slot reading all widgets into `recipe.compose`, marking dirty, scheduling. (3) *Selected node*: when the outline selection is a panel item, show the override editor (ROI four-int text à la `ReplotDialog._roi`, clim "lo,hi" with blank halves, cmap combo over `("",) + CMAP_CHOICES`, label edit, show-title/colorbar tri-state combos Follow/On/Off, scale spinbox 0=off); `_apply_panel_overrides` parses (malformed ROI/clim → notes-bar message, no mutation) and schedules a preview. Export button in the toolbar area:
```python
def export_now(self) -> None:
    from dfxm.common.errors import StageUserError
    from dfxm.compose.render import export_recipe

    out = QFileDialog.getExistingDirectory(self, "Export directory")
    if not out:
        return
    try:
        paths, res = export_recipe(self._recipe, out, loader_cache=self._cache)
    except StageUserError as exc:
        hint = f"  Hint: {exc.hint}" if exc.hint else ""
        self._notes_label.setText(f"export failed: {exc}{hint}")
        return
    notes = f"; {'; '.join(res.notes)}" if res.notes else ""
    self._notes_label.setText(f"wrote {len(paths)} file(s) → {out}{notes}")
```
(`QFileDialog` imported at module top of `figure_builder.py` so the monkeypatch target `gui.figure_builder.QFileDialog` resolves.)
- [ ] Run to pass: `python3 -m pytest -q tests/test_gui_figure_builder.py` + `ruff check gui`.
- [ ] Docs (same change): Usage chapter — style/compose/override workflow + export; Codebase — the new methods.
- [ ] Commit: `git add -A && git commit -m "feat(gui): figure-builder right pane — style controls, compose knobs, per-node overrides, export"`

---

### Task 13 — Main-window launch, gui_smoke step, docs chapters, final verify

**Files**
- Modify: `gui/main_window.py` (button next to "Publication style…", ~line 115)
- Modify: `tests/gui_smoke.py` (append step `[37]`)
- Modify: `docs/Usage.md` (complete the "Figure builder" chapter; add the builder to the Contents list and the "Publication export" cross-reference)
- Modify: `docs/Codebase.md` (final sweep: compose modules + gui modules + data-flow complete and consistent)

**Interfaces**

*Consumes (verified):* `MainWindow._pub_style_btn` wiring pattern (`gui/main_window.py:114-116`, left column `left_layout` at lines 127-132), `MainWindow._views[name]._form` (stage forms — see `gui_smoke.py:102`), `experiment_overrides` (bindings), `FigureBuilderWindow(defaults_provider, style, parent=None)`.

*Produces:*
```python
# gui/main_window.py
self._figure_builder_btn = QPushButton("Figure builder…")   # left column, below pub-style
def _on_figure_builder(self) -> None    # lazy import; non-modal .show(); one instance reused
def _builder_defaults(self) -> dict[str, dict]
    # per stage: h5 (from the stage form's current input/output field, falling back to
    # experiment_overrides chaining), sx/sy (experiment pixel sizes), jobs (profiles form's
    # jobs_json parsed, [] on parse failure)
```

**Steps**

- [ ] Failing check first — append a pytest-side wiring test to `tests/test_gui_figure_builder.py`:
```python
def test_main_window_launches_builder_non_modal():
    from gui.main_window import MainWindow

    win = MainWindow()
    win._on_figure_builder()
    assert win._figure_builder is not None
    assert win._figure_builder.isVisible()
    first = win._figure_builder
    win._on_figure_builder()  # reuse, not a second window
    assert win._figure_builder is first
```
Run: `python3 -m pytest -q tests/test_gui_figure_builder.py -k main_window` — `AttributeError: _on_figure_builder`.
- [ ] Implement in `main_window.py` (Read the target region first; the button rows sit at lines 114-132):
```python
# after the pub-style button block:
self._figure_builder = None
self._figure_builder_btn = QPushButton("Figure builder…")
self._figure_builder_btn.clicked.connect(self._on_figure_builder)
# in left_layout, after self._pub_style_btn:
left_layout.addWidget(self._figure_builder_btn)
```
```python
def _on_figure_builder(self) -> None:
    """Open (or raise) the non-modal figure-builder window."""
    from .figure_builder import FigureBuilderWindow

    if self._figure_builder is None:
        self._figure_builder = FigureBuilderWindow(
            self._builder_defaults, replace(self._plot_style), parent=self
        )
    self._figure_builder.show()
    self._figure_builder.raise_()
    self._figure_builder.activateWindow()


def _builder_defaults(self) -> dict[str, dict]:
    import json

    from .bindings import experiment_overrides

    exp = self._experiment_panel.current_experiment()
    sx, sy = exp.pixel_size_x_um, exp.pixel_size_y_um
    out: dict[str, dict] = {}
    field_for = {
        "strain": "root_folder",
        "mosaicity": "root_folder",
        "rocking": "raw_root",
        "slices": "mosa_volume_file",
        "profiles": "consolidated_h5",
    }
    for stage, fld in field_for.items():
        values = self._views[stage]._form.values()
        chained = experiment_overrides(stage, exp)
        h5 = values.get(fld) or chained.get(fld) or ""
        jobs: list = []
        if stage == "profiles":
            try:
                jobs = json.loads(values.get("jobs_json") or "[]")
            except (TypeError, ValueError):
                jobs = []
        out[stage] = {"h5": h5, "sx": sx, "sy": sy, "jobs": jobs if isinstance(jobs, list) else []}
    return out
```
(Verify `ParamForm.values()` is the getter name by grepping `gui/widgets/param_form.py` before writing; if it is `get_values()`, use that — the smoke test at `gui_smoke.py:102` shows `set_values`, so mirror the actual getter.)
- [ ] Append gui_smoke step `[37]` after `[36]` (Read the tail region first; follow the existing style):
```python
    # [37] figure builder: open from the main window, build a 1-panel recipe from a
    # synthetic slices h5, preview renders, export writes a PNG without tight-crop.
    import h5py as _h5b
    import numpy as _npb

    from dfxm.compose.recipe import PanelDef as _PD, PanelSource as _PS

    _bdir = tempfile.mkdtemp()
    _bh5 = os.path.join(_bdir, "obl.h5")
    with _h5b.File(_bh5, "w") as f:
        g = f.create_group("strain")
        g.attrs.update(kind="strain", cbar_label="v", cmap="RdBu_r", title="s", vmin=-1, vmax=1)
        sg = g.create_group("obl")
        sg.create_dataset("slices", data=_npb.zeros((1, 4, 5), "f4"))
        sg.create_dataset("u_um", data=_npb.linspace(0.0, 2.0, 5))
        sg.create_dataset("v_um", data=_npb.linspace(0.0, 1.5, 4))
        sg.create_dataset("offsets_um", data=_npb.array([0.0]))
    win._on_figure_builder()
    fb = win._figure_builder
    assert fb.isVisible()
    fb._style.scale_um_per_cm = 10.0
    fb._sync_style_to_recipe()
    fb.add_panels(
        [_PD("s0", _PS(_bh5, "slice_plane", {"volume_id": "strain", "slice_name": "obl", "plane": 0}))]
    )
    res = fb.render_now()
    assert res is not None and res.n_rendered == 1, fb._notes_label.text()
    _bout = os.path.join(_bdir, "export")
    from PySide6.QtWidgets import QFileDialog as _QFD

    import gui.figure_builder as _fbmod

    _orig_dir = _fbmod.QFileDialog.getExistingDirectory
    _fbmod.QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: _bout)
    try:
        fb.export_now()
    finally:
        _fbmod.QFileDialog.getExistingDirectory = _orig_dir
    assert os.path.exists(os.path.join(_bout, "untitled.png"))
    _rp = os.path.join(_bdir, "r.json")
    fb.save_recipe_file(_rp)
    fb.load_recipe_file(_rp)
    assert not fb.is_dirty()
    print("[37] figure builder: open, preview, export, recipe save/load round-trip")
```
- [ ] Docs finalization (same change): `docs/Usage.md` — complete the "Figure builder" chapter (workflow: sources → layout → labels → shared bars → export; recipe files; CLI; troubleshooting: placeholder panels, implied-scale notes, drift notes; add to Contents). `docs/Codebase.md` — verify the `dfxm/compose` section covers all five modules, `gui/figure_builder.py` + `gui/widgets/panel_picker.py` entries complete, data-flow table row present.
- [ ] Final verify (invoke the repo `verify-suite` skill when executing): `python3 -m pytest -q` (expect the pre-branch count + all new tests, 13 skips unchanged), `ruff check . && ruff format .`, `python3 tests/gui_smoke.py` (expect `[1]`…`[37]` all printed), and one manual-style CLI run: `python3 -m dfxm.compose render <tmp recipe> -o <tmp>` exits 0.
- [ ] Commit: `git add -A && git commit -m "feat(gui): figure-builder launch button, smoke step 37, docs chapters"`

---

## Architectural trade-offs (decided; recorded for reviewers)

- **Composer core + box-tree + one Figure** (per spec) over GridSpec — GridSpec hands sizes to matplotlib auto-layout and cannot guarantee µm/cm (the exact failure the trace project removed); over PNG pasting — raster-only, no SVG.
- **`ComposeResult` instead of a bare `(Figure, notes)` tuple** — the spec's CLI contract (exit non-zero only when *no* panel rendered) needs panel counts, and the GUI click-pick needs `axes_by_id`; the spec's intent (figure + notes out) is preserved as fields.
- **Margin sharing at direct-leaf level, envelope alignment for nested nodes** — implements the spec's row/col compensation with a strictly recursive, testable rule; both acceptance figures exercise it (fig 1 leaf sharing, fig 2 ragged envelopes).
- **Group labels attach to the group's first leaf panel's axes** — the label is then measured with that panel's margins, so it can never collide or be cropped; a floating figure-level label would need its own margin bookkeeping.
- **Preview swaps in a fresh `FigureCanvasQTAgg` per render** rather than reusing the themed `MplCanvas` — the composed figure's absolute placement must not be touched by a display layout engine, and exports/preview must both show the white publication canvas.
- **Loaders pure of `(source, roi)` with an explicit cache dict** — style/layout edits redraw from cache; Refresh is a cache clear; the cache key is the JSON-serialized source+roi.

## Risks & mitigations

- **Measure-pass fidelity** (tick density, offset text, scale-bar overflow at extreme `font_scale`) — mitigated by measuring at final box size with `measure_axes_margins` (the proven engine) and by the per-panel drift guard; any miss surfaces as a note, never silently.
- **Shared-colorbar cell height vs group height** — the bar cell is sized from the group's content boxes before measuring; a mismatch shows as visible misalignment in acceptance-test position asserts (Task 7 test pins one-bar-count and unified clim; eyeball follows post-merge).
- **`rocking._replot_default_clim` reads the whole volume** — acceptable for composer loads (same cost as a replot); the cache prevents repeats.
- **Stage-module imports inside `dfxm/compose/adapters.py` loaders** are lazy (function-level), keeping `import dfxm.compose` light and the GUI startup unaffected.
- **`em`-dash/indentation edit hazards in `profiles.py`/`slices.py` hint strings** — every Modify step above requires Reading the exact target region before the first Edit (repo rule restated in the tasks).
- **Known post-plan follow-ups (out of scope, per spec):** visualize/matched/histogram adapters, rich text-cell styling, journal width presets, on-screen eyeball with real STO2 data + PDF/SVG inspection (record in the project memory note at finish).
