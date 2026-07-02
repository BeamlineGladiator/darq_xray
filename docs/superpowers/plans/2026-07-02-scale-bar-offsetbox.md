# Scale-Bar Offsetbox Rebuild Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild `draw_scale_bar()` on matplotlib's offsetbox machinery so the background box always hugs the rendered label + bar at any font scale, with label and bar mutually centred.

**Architecture:** The hand-rolled `Rectangle` + `FancyBboxPatch` data-coordinate geometry in `dfxm/common/plotting.py` is replaced by an `AnchoredOffsetbox` whose child is a `VPacker(align="center")` stacking the label `TextArea` above the bar (a `Rectangle` in an `AuxTransformBox(ax.transData)`, keeping the bar true to data µm). Offsetbox layout happens at draw time, so the frame is exact even under constrained layout. Signature and the three call sites are unchanged. Spec: `docs/superpowers/specs/2026-07-02-scale-bar-offsetbox-design.md`.

**Tech Stack:** matplotlib (`offsetbox`, `patches`, `font_manager`), pytest with `FigureCanvasAgg`.

## Global Constraints

- `dfxm/` stays Qt-free; never import `pyplot` or call `matplotlib.use(...)` — use the explicit `Figure` API (this plan only touches figure code already following that rule).
- Docs (`docs/Usage.md` + `docs/Codebase.md`) must be updated **in the same commit** as the code change.
- Ruff: line length 100, double quotes (auto-formatted on Write/Edit by hook).
- No new `PlotStyle` fields, no GUI changes — all existing style knobs keep working.
- The only pinned legacy property is the *absent layout engine* on `style=None` figures; scale-bar pixel geometry is NOT byte-pinned (see `dfxm/common/render.py:41-45`) and may change on both render paths.
- `tests/gui_smoke.py` is not a pytest file — run it as `python3 tests/gui_smoke.py`.

---

### Task 1: Rebuild `draw_scale_bar` on offsetbox (code + all tests + docs, one commit)

**Files:**
- Modify: `dfxm/common/plotting.py:353-430` (`draw_scale_bar` body only)
- Test: `tests/test_plot_style.py` (rewrite lines 69-121, the `_box_pad` helper ~194-213; add 2 new tests + 3 helpers)
- Test: `tests/test_stage_strain.py:175-192`
- Test: `tests/test_stage_slices.py:233` and `:254`
- Test: `tests/test_figures_catalog.py:25-43`
- Modify: `docs/Usage.md:429-430` (Scale bar table rows)
- Modify: `docs/Codebase.md:225` (`draw_scale_bar` row)

**Interfaces:**
- Consumes: existing `PlotStyle` fields (`scale_bar_*`), `auto_scale_bar_length_um(ext_x)`.
- Produces: `draw_scale_bar(ax, length_um=None, *, style)` — same signature; the scale bar is now a single `matplotlib.offsetbox.AnchoredOffsetbox` in `ax.artists` (nothing added to `ax.patches`/`ax.texts`). Callers (`dfxm/stages/strain.py:408`, `dfxm/stages/slices.py:777`, `dfxm/common/render.py:65`) need no change.

- [ ] **Step 1: Create the feature branch**

```bash
git checkout -b scale-bar-offsetbox
```

- [ ] **Step 2: Rewrite the scale-bar tests in `tests/test_plot_style.py` (failing first)**

Read `tests/test_plot_style.py` in full first (it is ~220 lines; `_ax` is its fixture helper). Then:

**(a)** Add these helpers near the top of the file (after the existing `_ax` helper):

```python
def _drawn_renderer(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def _scale_bar_artist(ax):
    from matplotlib.offsetbox import AnchoredOffsetbox

    return next(a for a in ax.artists if isinstance(a, AnchoredOffsetbox))


def _offsetbox_children(artist):
    """Flatten every artist nested under an offsetbox tree (Texts, Rectangles, ...)."""
    out, stack = [], [artist]
    while stack:
        a = stack.pop()
        out.append(a)
        if hasattr(a, "get_children"):
            stack.extend(a.get_children())
    return out
```

**(b)** Replace `test_draw_scale_bar_adds_patch_and_text` (line 69) with:

```python
def test_draw_scale_bar_adds_anchored_box_with_label_and_bar():
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    fig, ax = _ax()
    draw_scale_bar(ax, length_um=10.0, style=PlotStyle(scale_bar_color="white"))
    abox = _scale_bar_artist(ax)  # raises StopIteration if missing
    kids = _offsetbox_children(abox)
    assert any(isinstance(t, Text) and "µm" in t.get_text() for t in kids)
    bar = next(p for p in kids if isinstance(p, Rectangle))
    ec = bar.get_edgecolor()  # (R, G, B, A) — no doubled point-based edge
    assert ec[3] == 0 or bar.get_linewidth() == 0
    assert len(ax.patches) == 0  # nothing leaks into ax.patches any more
```

**(c)** Replace `test_draw_scale_bar_box_adds_a_second_patch` (line 77) with:

```python
def test_draw_scale_bar_box_toggles_frame():
    fig, ax = _ax()
    draw_scale_bar(ax, length_um=10.0, style=PlotStyle(scale_bar_box=True))
    assert _scale_bar_artist(ax).patch.get_visible()

    fig2, ax2 = _ax()
    draw_scale_bar(ax2, length_um=10.0, style=PlotStyle(scale_bar_box=False))
    assert not _scale_bar_artist(ax2).patch.get_visible()
```

**(d)** Replace `test_draw_scale_bar_box_geometry_is_sane` (lines 84-121) with the regression test for the reported bug (box must hug the rendered text at large font scale, and stay a snug corner element):

```python
def test_scale_bar_box_hugs_label_at_large_font_scale():
    from matplotlib.text import Text

    fig, ax = _ax(ext_x=50.0, ext_y=30.0)
    style = PlotStyle(
        scale_bar_box=True,
        font_scale=2.2,
        scale_bar_label_scale=1.1,
        scale_bar_thickness_pt=4.0,
    )
    draw_scale_bar(ax, length_um=10.0, style=style)
    renderer = _drawn_renderer(fig)
    abox = _scale_bar_artist(ax)
    frame_bb = abox.patch.get_window_extent(renderer)
    label = next(
        t
        for t in _offsetbox_children(abox)
        if isinstance(t, Text) and "µm" in t.get_text()
    )
    text_bb = label.get_window_extent(renderer)
    # The reported bug: at large font scale the label spilled out of the box.
    assert frame_bb.contains(text_bb.x0, text_bb.y0), "label bottom-left outside box"
    assert frame_bb.contains(text_bb.x1, text_bb.y1), "label top-right outside box"
    # Snug corner element, not half the figure.
    ax_bb = ax.get_window_extent(renderer)
    assert frame_bb.height < 0.35 * ax_bb.height
```

**(e)** Add a new centring test right after it (short bar + big font → label wider than bar; the two must share a centre):

```python
def test_scale_bar_label_and_bar_are_centred():
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text

    fig, ax = _ax()
    draw_scale_bar(ax, length_um=2.0, style=PlotStyle(scale_bar_box=True, font_scale=2.2))
    renderer = _drawn_renderer(fig)
    kids = _offsetbox_children(_scale_bar_artist(ax))
    label = next(t for t in kids if isinstance(t, Text) and "µm" in t.get_text())
    bar = next(p for p in kids if isinstance(p, Rectangle))
    text_bb = label.get_window_extent(renderer)
    bar_bb = bar.get_window_extent(renderer)
    text_cx = (text_bb.x0 + text_bb.x1) / 2
    bar_cx = (bar_bb.x0 + bar_bb.x1) / 2
    assert abs(text_cx - bar_cx) < 2.0  # px
```

**(f)** Replace the `_box_pad` helper and its `test_box_margin_control_affects_box_padding` (lines ~194-213 — Read the exact current bytes first) with a drawn-extent comparison:

```python
def _box_frame_width(margin_pt):
    fig, ax = _ax()
    draw_scale_bar(
        ax,
        5.0,
        style=PlotStyle(scale_bar_box=True, scale_bar_box_margin_pt=margin_pt),
    )
    renderer = _drawn_renderer(fig)
    return _scale_bar_artist(ax).patch.get_window_extent(renderer).width


def test_box_margin_control_affects_box_padding():
    assert _box_frame_width(10.0) > _box_frame_width(2.0)
```

- [ ] **Step 3: Run the rewritten tests to verify they fail**

Run: `python3 -m pytest tests/test_plot_style.py -q`
Expected: the five rewritten/new tests FAIL with `StopIteration` from `_scale_bar_artist` (no `AnchoredOffsetbox` exists yet); the untouched tests still pass.

- [ ] **Step 4: Rewrite the `draw_scale_bar` body**

Replace the entire body of `draw_scale_bar` in `dfxm/common/plotting.py` (lines 353-430 — Read the current bytes first; keep the signature line) with:

```python
def draw_scale_bar(ax, length_um: float | None = None, *, style: "PlotStyle") -> None:
    """Draw a µm scale bar (label centred over the bar, optional background box).

    *ax* must use data coordinates in microns. ``length_um=None`` auto-sizes.

    Built on matplotlib's offsetbox machinery: the ``AnchoredOffsetbox`` frame
    is laid out at draw time around the *rendered* label + bar, so the
    background box hugs its content at any ``font_scale`` (exact even under
    constrained layout, whose axes positions are only final at first draw),
    and the ``VPacker`` centres label and bar on each other by construction.
    """
    from matplotlib.font_manager import FontProperties
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, TextArea, VPacker
    from matplotlib.patches import Rectangle

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = (x1 - x0), (y1 - y0)
    sl = length_um if length_um is not None else auto_scale_bar_length_um(abs(xr))
    # Bar height in data coords: 0.004·thickness_pt·|yr| (≈0.012·|yr| at the
    # default thickness_pt=3.0) — unchanged from the previous hand-rolled geometry.
    bh = abs(yr) * 0.004 * style.scale_bar_thickness_pt
    label_size = 10.0 * style.font_scale * style.scale_bar_label_scale

    bar = AuxTransformBox(ax.transData)  # width stays true to data µm
    bar.add_artist(
        Rectangle((0, 0), sl, bh, facecolor=style.scale_bar_color, edgecolor="none", linewidth=0)
    )
    label = TextArea(
        f"{sl:g} µm",
        textprops={"color": style.scale_bar_color, "fontsize": label_size, "fontweight": "bold"},
    )
    box = AnchoredOffsetbox(
        loc=style.scale_bar_loc,
        child=VPacker(children=[label, bar], align="center", pad=0, sep=0.25 * label_size),
        # pad/borderpad are in font-size units of *prop*; pinning prop to the
        # label size makes box_margin_pt mean real points.
        prop=FontProperties(size=label_size),
        pad=style.scale_bar_box_margin_pt / label_size,
        borderpad=0.5,
        frameon=style.scale_bar_box,
    )
    if style.scale_bar_box:
        box.patch.set(
            facecolor=style.scale_bar_box_color,
            edgecolor="none",
            alpha=style.scale_bar_box_alpha,
        )
        # Rounded corners without extra growth: all padding comes from the
        # offsetbox pad above; rounding_size is in mutation-scale (font) units.
        box.patch.set_boxstyle("round", pad=0, rounding_size=0.4)
    box.set_zorder(5)
    # In-axes decoration: keep the constrained-layout solver from budgeting
    # figure margin for it (the old code clipped the label for the same reason).
    box.set_in_layout(False)
    ax.add_artist(box)
```

Note: `FancyBboxPatch` may become an unused import at the top of the old body's `from matplotlib.patches import ...` line — the ruff hook will flag it; the imports above are function-local, so just delete the old local import line with the body.

- [ ] **Step 5: Run the plot-style tests to verify they pass**

Run: `python3 -m pytest tests/test_plot_style.py -q`
Expected: ALL PASS. If `_offsetbox_children` finds no `Text` (a matplotlib version where `TextArea.get_children()` differs), extend the helper to also follow `get_child()` — do not reach into private attributes.

- [ ] **Step 6: Update the dependent stage/catalog tests**

Read each target region first (exact bytes). All three files follow the same pattern — the scale bar moved from `ax.patches` to an `AnchoredOffsetbox` in `ax.artists`.

**`tests/test_stage_strain.py:175-192`** — replace both tests' assertions:

```python
def test_build_strain_map_legacy_has_no_scale_bar():
    from matplotlib.offsetbox import AnchoredOffsetbox

    fig = S.build_strain_map(np.random.rand(20, 30) * 1e-3, 0.1, 0.3, None, (None, None))
    ax = fig.axes[0]
    assert not any(isinstance(a, AnchoredOffsetbox) for a in ax.artists)


def test_build_strain_map_style_adds_scale_bar():
    from matplotlib.offsetbox import AnchoredOffsetbox

    from dfxm.common.plotting import PlotStyle

    fig = S.build_strain_map(
        np.random.rand(20, 30) * 1e-3,
        0.1,
        0.3,
        None,
        (None, None),
        style=PlotStyle(scale_bar=True),
    )
    assert any(isinstance(a, AnchoredOffsetbox) for a in fig.axes[0].artists)
```

**`tests/test_stage_slices.py`** — line 233 (`scale_bar=False` case): replace
`assert len(fig.axes[0].patches) == 0` with:

```python
    from matplotlib.offsetbox import AnchoredOffsetbox

    assert not any(isinstance(a, AnchoredOffsetbox) for a in fig.axes[0].artists)
```

Line 254 (legacy draws the black scale bar): replace
`assert len(fig.axes[0].patches) >= 1` with:

```python
    from matplotlib.offsetbox import AnchoredOffsetbox

    assert any(isinstance(a, AnchoredOffsetbox) for a in fig.axes[0].artists)
```

(Put the import at the top of each test function, matching the local-import style already used in these files.)

**`tests/test_figures_catalog.py:25-43`** — same substitution: line 28
`assert len(ax.patches) >= 1` → `assert any(isinstance(a, AnchoredOffsetbox) for a in ax.artists)`; line 43 `assert len(ax.patches) == 0` → `assert not any(isinstance(a, AnchoredOffsetbox) for a in ax.artists)`. This file imports at module level (line 10 area), so add `from matplotlib.offsetbox import AnchoredOffsetbox` to the top-level imports instead.

- [ ] **Step 7: Update both docs (same commit — contract)**

**`docs/Usage.md`** — replace the two table rows at lines 429-430:

```markdown
| Background box | Optionally draw a semi-transparent box behind the bar + label. The box sizes itself to the rendered label and bar at any font scale, with the label centred over the bar |
| Box colour / alpha / margin | Control the background box appearance; margin is the padding inside the box, in points |
```

**`docs/Codebase.md`** — replace the `draw_scale_bar` table row (line 225; Read the exact bytes first):

```markdown
| `draw_scale_bar(ax, length_um, *, style)` | Draw a µm scale bar on `ax` whose data coordinates are in µm. Built as an `AnchoredOffsetbox` in `ax.artists` (`VPacker`: bold label `TextArea` over a bar `Rectangle` in an `AuxTransformBox(transData)`): the optional background box is laid out at draw time around the *rendered* label + bar, so it hugs its content at any `font_scale` (exact under constrained layout), and label/bar are mutually centred. `length_um=None` calls `auto_scale_bar_length_um`. Box padding (`scale_bar_box_margin_pt`) is in real points; the anchor inset is fixed in font units, no longer proportional to the data extent. `set_in_layout(False)` keeps constrained layout from budgeting figure margin for it (the old code clipped the label for the same reason). |
```

- [ ] **Step 8: Full verification**

Run:
```bash
python3 -m pytest -q
ruff check . && ruff format --check .
python3 tests/gui_smoke.py
```
Expected: suite green (baseline 301 passed / 13 skipped; 4 tests replaced 1:1 plus 1 new centring test → 302/13), ruff clean, smoke `[1]`-`[25]` green.

- [ ] **Step 9: Commit (code + tests + docs together)**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py tests/test_stage_strain.py \
  tests/test_stage_slices.py tests/test_figures_catalog.py docs/Usage.md docs/Codebase.md
git commit -m "fix(plotting): scale-bar box hugs label via offsetbox; label/bar centred"
```
