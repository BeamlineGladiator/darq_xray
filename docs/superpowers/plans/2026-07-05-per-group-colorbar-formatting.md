# Per-group Colourbar Number Formatting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give each quantity group (mosa_com, mosa_fwhm, strain, raw) its own colourbar tick format, with independently sizeable, top-or-bottom scientific-notation offset text, and an "arbitrary units" mode that drops ticks for raw images.

**Architecture:** Replace the single `PlotStyle.colorbar_tick_format` with per-group fields (`tickfmt_<group>`, `offset_scale_<group>`, `offset_pos_<group>`) plus `*_for(group)` helpers mirroring the existing `cmap_for`. `add_colorbar` gains a keyword `group` and renders scientific notation itself (custom exponent label) rather than relying on matplotlib's un-styleable built-in offset text. The group is threaded to every `add_colorbar` call site (already known wherever `resolve_cmap` is called). The tuned per-group defaults live in `PUBLICATION_STYLE` and an old-dict migration, so the byte-identical `style=None` legacy render path is untouched.

**Tech Stack:** Python, matplotlib (`Figure`/`FuncFormatter`/offsetbox), PySide6 (export dialog), pytest.

## Global Constraints

- **Keep `dfxm/` Qt-free** — no PySide6/pyvista imports in `dfxm/common/` or `dfxm/stages/`.
- **Build figures with the explicit `Figure` API** — never `pyplot` / `matplotlib.use(...)`.
- **`style=None` output must stay byte-identical** with the pre-export legacy renderers. Bare `PlotStyle()` therefore keeps `tickfmt_*="auto"`, `offset_scale_*=1.0`, `offset_pos_*="top"`. Tuned defaults live ONLY in `PUBLICATION_STYLE` and the `_style_from_dict` migration (serialized/GUI path, never the `style=None` code path).
- **Docs in the same change** — `docs/Usage.md` (user-facing) and `docs/Codebase.md` (code reference) are updated in the task that changes the behaviour, not as a follow-up.
- **ruff** — line length 100, double quotes, target py310; `ruff format` runs on save via hook. Run `ruff check .` before each commit.
- **Read before first Edit** — any file not created this session must be Read once before editing. Never reconstruct `old_string` from memory (hint strings have em-dashes; indentation varies).
- **No git remote** — no pull/push/PR. `python3 -m pytest -q` is the suite; `python3 tests/gui_smoke.py` is the GUI smoke (not a pytest file).
- **Four groups** (verbatim): `CMAP_GROUPS = ("mosa_com", "mosa_fwhm", "strain", "raw")`.

---

### Task 1: Per-group `PlotStyle` fields + lookup helpers + `GROUP_BY_KIND`

Add the schema surface without removing anything, so the suite stays green. `PUBLICATION_STYLE`, the migration, and the field removal come later (Tasks 5–6).

**Files:**
- Modify: `dfxm/common/plotting.py` (PlotStyle dataclass ~L60-88; add `GROUP_BY_KIND` near `CMAP_GROUPS` ~L26)
- Test: `tests/test_plot_style.py`

**Interfaces:**
- Produces:
  - `PlotStyle.tickfmt_mosa_com/_mosa_fwhm/_strain/_raw: str` (default `"auto"`)
  - `PlotStyle.offset_scale_mosa_com/_mosa_fwhm/_strain/_raw: float` (default `1.0`)
  - `PlotStyle.offset_pos_mosa_com/_mosa_fwhm/_strain/_raw: str` (default `"top"`)
  - `PlotStyle.tickfmt_for(group: str | None) -> str` (None → `"auto"`)
  - `PlotStyle.offset_scale_for(group: str | None) -> float` (None → `1.0`)
  - `PlotStyle.offset_pos_for(group: str | None) -> str` (None → `"top"`)
  - `GROUP_BY_KIND: dict[str, str]` mapping volume kind → group.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_plot_style.py`:

```python
def test_per_group_tickfmt_defaults_and_lookup():
    from dfxm.common.plotting import GROUP_BY_KIND

    s = PlotStyle()
    # bare defaults preserve the legacy look
    assert s.tickfmt_for("strain") == "auto"
    assert s.tickfmt_for("raw") == "auto"
    assert s.offset_scale_for("mosa_com") == 1.0
    assert s.offset_pos_for("mosa_fwhm") == "top"
    # group=None is the neutral fallback (callers that don't know their group)
    assert s.tickfmt_for(None) == "auto"
    assert s.offset_scale_for(None) == 1.0
    assert s.offset_pos_for(None) == "top"
    # unknown non-None group raises, like cmap_for
    import pytest

    with pytest.raises(KeyError):
        s.tickfmt_for("bogus")
    # explicit per-group values round-trip through the lookups
    s2 = PlotStyle(tickfmt_strain="scientific", offset_scale_strain=1.5, offset_pos_strain="bottom")
    assert s2.tickfmt_for("strain") == "scientific"
    assert s2.offset_scale_for("strain") == 1.5
    assert s2.offset_pos_for("strain") == "bottom"
    # GROUP_BY_KIND collapses raw_sum / raw_specific onto the raw group
    assert GROUP_BY_KIND["raw_sum"] == "raw"
    assert GROUP_BY_KIND["raw_specific"] == "raw"
    assert GROUP_BY_KIND["strain"] == "strain"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_plot_style.py::test_per_group_tickfmt_defaults_and_lookup -q`
Expected: FAIL (`TypeError: unexpected keyword argument 'tickfmt_strain'` / `ImportError: GROUP_BY_KIND`).

- [ ] **Step 3: Add the `GROUP_BY_KIND` constant**

In `dfxm/common/plotting.py`, immediately after the `CMAP_CHOICES` tuple (~L40), add:

```python
# Volume "kind" (as stored in HDF5 attrs by the map stages) -> quantity group.
# Shared by slices and profiles so the kind->group mapping lives in one place.
GROUP_BY_KIND: dict[str, str] = {
    "mosa_com": "mosa_com",
    "mosa_fwhm": "mosa_fwhm",
    "strain": "strain",
    "raw_sum": "raw",
    "raw_specific": "raw",
}
```

- [ ] **Step 4: Add the per-group fields**

In the `PlotStyle` dataclass, REPLACE the single colourbar tick-format line:

```python
    colorbar_tick_format: str = "auto"  # "auto" | "scientific" | a digit count like "2"
```

with the per-group fields (keep `colorbar_tick_format` for now — it is removed in Task 6):

```python
    colorbar_tick_format: str = "auto"  # DEPRECATED (removed once GUI migrates); see tickfmt_*
    # per-group colourbar number format: "auto" | "scientific" | "arb" | a digit count like "2"
    tickfmt_mosa_com: str = "auto"
    tickfmt_mosa_fwhm: str = "auto"
    tickfmt_strain: str = "auto"
    tickfmt_raw: str = "auto"
    # per-group scientific-notation ×10ⁿ offset text: size multiplier (×font_scale) + placement
    offset_scale_mosa_com: float = 1.0
    offset_scale_mosa_fwhm: float = 1.0
    offset_scale_strain: float = 1.0
    offset_scale_raw: float = 1.0
    offset_pos_mosa_com: str = "top"  # "top" | "bottom"
    offset_pos_mosa_fwhm: str = "top"
    offset_pos_strain: str = "top"
    offset_pos_raw: str = "top"
```

- [ ] **Step 5: Add the lookup helpers**

Directly after the existing `cmap_for` method in `PlotStyle`, add:

```python
    def tickfmt_for(self, group: str | None) -> str:
        """Tick format for a quantity group; ``group=None`` -> the neutral ``"auto"``."""
        if group is None:
            return "auto"
        if group not in CMAP_GROUPS:
            raise KeyError(f"unknown colormap group {group!r}")
        return getattr(self, f"tickfmt_{group}")

    def offset_scale_for(self, group: str | None) -> float:
        """Scientific-offset size multiplier for a group; ``group=None`` -> ``1.0``."""
        if group is None:
            return 1.0
        if group not in CMAP_GROUPS:
            raise KeyError(f"unknown colormap group {group!r}")
        return getattr(self, f"offset_scale_{group}")

    def offset_pos_for(self, group: str | None) -> str:
        """Scientific-offset placement for a group; ``group=None`` -> ``"top"``."""
        if group is None:
            return "top"
        if group not in CMAP_GROUPS:
            raise KeyError(f"unknown colormap group {group!r}")
        return getattr(self, f"offset_pos_{group}")
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_plot_style.py::test_per_group_tickfmt_defaults_and_lookup -q`
Expected: PASS.

- [ ] **Step 7: Run the full suite + lint**

Run: `python3 -m pytest -q && ruff check dfxm/common/plotting.py tests/test_plot_style.py`
Expected: PASS (no existing test broke — nothing was removed).

- [ ] **Step 8: Commit**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py
git commit -m "feat(plotting): per-group colourbar tickfmt/offset fields + GROUP_BY_KIND"
```

---

### Task 2: `add_colorbar` per-group rendering — arbitrary units + custom scientific offset

Rewrite `add_colorbar` to branch on `style.tickfmt_for(group)`. `colorbar_tick_format` still exists but is no longer read here.

**Files:**
- Modify: `dfxm/common/plotting.py` (`_tick_formatter` ~L454-466, `add_colorbar` ~L517-528; add `_apply_scientific` helper)
- Modify: `docs/Codebase.md` (`add_colorbar` row ~L228)
- Test: `tests/test_plot_style.py`

**Interfaces:**
- Consumes: `PlotStyle.tickfmt_for`, `offset_scale_for`, `offset_pos_for` (Task 1).
- Produces:
  - `add_colorbar(fig, im, ax, label, style, *, group=None)` — `group` keyword-only, default `None`.
  - `_apply_scientific(cb, im, style, group)` — internal.
  - `_tick_formatter(fmt)` now returns `None` for `"auto"`, `"scientific"`, and `"arb"` (scientific/arb handled in `add_colorbar`); a `FuncFormatter` for non-negative digit strings.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_plot_style.py` (keep the existing imports; `Text` is already imported for the scale-bar tests):

```python
def test_tick_formatter_scientific_and_arb_are_deferred():
    # scientific + arb are handled inside add_colorbar, not by _tick_formatter
    assert _tick_formatter("scientific") is None
    assert _tick_formatter("arb") is None
    # digit + auto behaviour unchanged
    assert _tick_formatter("2") is not None
    assert _tick_formatter("auto") is None


def test_add_colorbar_arbitrary_units_drops_ticks_and_marks_label():
    fig, ax = _ax()
    im = ax.imshow(np.arange(100).reshape(10, 10), extent=[0, 50, 0, 30], origin="lower")
    style = PlotStyle(tickfmt_raw="arb")
    cb = add_colorbar(fig, im, ax, "Intensity", style, group="raw")
    assert list(cb.get_ticks()) == []  # no numeric ticks
    assert cb.ax.get_ylabel() == "Intensity (arb. units)"


def test_add_colorbar_arbitrary_units_does_not_double_up_existing_au():
    fig, ax = _ax()
    im = ax.imshow(np.arange(100).reshape(10, 10), extent=[0, 50, 0, 30], origin="lower")
    style = PlotStyle(tickfmt_raw="arb")
    cb = add_colorbar(fig, im, ax, "Sum intensity (a.u.)", style, group="raw")
    assert cb.ax.get_ylabel() == "Sum intensity (a.u.)"  # already mentions a.u. -> no suffix


def test_add_colorbar_scientific_hides_builtin_offset_and_draws_custom():
    from matplotlib.text import Text

    fig, ax = _ax()
    im = ax.imshow(
        np.linspace(-2e-3, 2e-3, 100).reshape(10, 10), extent=[0, 50, 0, 30], origin="lower"
    )
    style = PlotStyle(
        tickfmt_strain="scientific",
        offset_pos_strain="bottom",
        offset_scale_strain=2.0,
        font_scale=1.0,
    )
    cb = add_colorbar(fig, im, ax, "Strain (ε)", style, group="strain")
    # matplotlib's built-in offset text is hidden
    assert cb.ax.yaxis.get_offset_text().get_visible() is False
    # exactly one custom exponent label exists, sized by font_scale*offset_scale
    exps = [
        t for t in cb.ax.texts if isinstance(t, Text) and "10" in t.get_text() and "times" in t.get_text()
    ]
    assert len(exps) == 1
    assert abs(exps[0].get_fontsize() - 9 * 1.0 * 2.0) < 1e-6
    # placed below the axes (va="top", y < 0)
    assert exps[0].get_position()[1] < 0.0
```

Note: the mathtext string is `r"$\times\mathdefault{10^{-3}}$"`, so it contains both `"10"` and `"times"`.

Also UPDATE the two existing tests that used the removed-in-Task-6 global field. Replace in `test_add_colorbar_sets_label_and_tick_count`:

```python
    cb = add_colorbar(
        fig, im, ax, "Strain (ε)", PlotStyle(colorbar_ticks=5, colorbar_tick_format="scientific")
    )
```

with:

```python
    cb = add_colorbar(
        fig,
        im,
        ax,
        "Strain (ε)",
        PlotStyle(colorbar_ticks=5, tickfmt_strain="scientific"),
        group="strain",
    )
```

And in `test_apply_text_scale_increases_title_pad_on_constrained_figure`, replace:

```python
    style = PlotStyle(font_scale=2.2, colorbar_ticks=5, colorbar_tick_format="scientific")
```
with
```python
    style = PlotStyle(font_scale=2.2, colorbar_ticks=5, tickfmt_strain="scientific")
```
and both `add_colorbar(...)` calls in that test gain `group="strain"`:
```python
    add_colorbar(fig_ref, im_ref, ax_ref, "label", style, group="strain")
```
```python
    add_colorbar(fig_treated, im_treated, ax_treated, "label", style, group="strain")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plot_style.py -q -k "scientific or arbitrary or tick_count or title_pad or deferred"`
Expected: FAIL (new tests: `add_colorbar() got unexpected keyword 'group'`; scientific still uses ScalarFormatter offset).

- [ ] **Step 3: Rewrite `_tick_formatter`**

Replace the existing `_tick_formatter` body:

```python
def _tick_formatter(fmt: str):
    """Formatter for plain/digit formats. ``"auto"``, ``"scientific"`` and ``"arb"``
    return ``None`` here — scientific/arb are handled directly in ``add_colorbar``
    because they need the colour limits / axis, and ``auto`` means matplotlib default.
    """
    if fmt in ("auto", "scientific", "arb"):
        return None
    try:
        d = int(fmt)
        if d >= 0:
            return FuncFormatter(lambda v, _pos: f"{v:.{d}f}")
    except ValueError:
        pass
    return None  # matplotlib default
```

- [ ] **Step 4: Add the `_apply_scientific` helper**

Add above `add_colorbar` (uses the module-level `math` and `FuncFormatter` imports already present):

```python
def _apply_scientific(cb, im, style, group) -> None:
    """Render scientific notation on *cb* with a custom, styleable exponent label.

    Computes one common order of magnitude from the colour limits, formats the
    ticks as mantissas, hides matplotlib's built-in (un-styleable, top-only)
    offset text, and draws our own ``×10ⁿ`` label at the group's chosen
    top/bottom position and size. Deterministic and redraw-safe (a static Text
    artist), unlike the built-in offset whose position matplotlib re-derives on
    every draw.
    """
    vmin, vmax = im.norm.vmin, im.norm.vmax
    maxabs = max(abs(vmin), abs(vmax)) if (vmin is not None and vmax is not None) else 0.0
    oom = int(math.floor(math.log10(maxabs))) if maxabs > 0 and math.isfinite(maxabs) else 0

    if oom == 0:
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    else:
        scale = 10.0**oom
        cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos, s=scale: f"{v / s:.2f}"))

    # Silence matplotlib's built-in offset text; we draw our own below.
    cb.ax.yaxis.get_offset_text().set_visible(False)
    if oom == 0:
        return  # mantissas are the values themselves — no exponent label needed

    size = max(9 * style.font_scale * style.offset_scale_for(group), 0.1)
    exp = r"$\times\mathdefault{10^{%d}}$" % oom
    if style.offset_pos_for(group) == "bottom":
        cb.ax.text(0.5, -0.02, exp, transform=cb.ax.transAxes, ha="center", va="top", fontsize=size)
    else:  # top
        cb.ax.text(
            0.5, 1.02, exp, transform=cb.ax.transAxes, ha="center", va="bottom", fontsize=size
        )
```

- [ ] **Step 5: Rewrite `add_colorbar`**

Replace the existing `add_colorbar` body:

```python
def add_colorbar(fig, im, ax, label: str, style: "PlotStyle", *, group: str | None = None):
    """Add a colourbar honouring thickness, label, tick count and per-group number format.

    *group* (one of :data:`CMAP_GROUPS`, or ``None`` for the neutral default)
    selects the tick format via ``style.tickfmt_for(group)``:
    ``"auto"``/digit as before; ``"scientific"`` renders a custom, styleable
    ``×10ⁿ`` exponent (see :func:`_apply_scientific`); ``"arb"`` drops all
    numeric ticks and marks the label "arbitrary units".
    """
    cb = fig.colorbar(im, ax=ax, fraction=style.colorbar_fraction, pad=0.04)
    text = style.colorbar_label if style.colorbar_label is not None else label
    fmt = style.tickfmt_for(group)

    if fmt == "arb":
        cb.set_ticks([])  # no numeric scale for arbitrary units
        if style.colorbar_label is None and not ("a.u." in text.lower() or "arb" in text.lower()):
            text = f"{text} (arb. units)"
    else:
        if style.colorbar_ticks and style.colorbar_ticks >= 2:
            cb.set_ticks(colorbar_tick_values(im.norm.vmin, im.norm.vmax, style.colorbar_ticks))
        if fmt == "scientific":
            _apply_scientific(cb, im, style, group)
        else:
            f = _tick_formatter(fmt)
            if f is not None:
                cb.ax.yaxis.set_major_formatter(f)

    cb.set_label(text, fontsize=10 * style.font_scale)
    cb.ax.tick_params(labelsize=9 * style.font_scale)
    return cb
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_plot_style.py -q`
Expected: PASS.

- [ ] **Step 7: Update `docs/Codebase.md`**

Replace the `add_colorbar` row (~L228):

```markdown
| `add_colorbar(fig, im, ax, label, style, *, group=None)` | Add a colourbar honouring `style.colorbar_fraction`, label, tick count, and the per-group number format `style.tickfmt_for(group)`. `group` is one of `CMAP_GROUPS` (or `None` = neutral). Formats: `"auto"`/digit as before; `"scientific"` draws a custom, styleable `×10ⁿ` exponent label at the group's `offset_pos_*` (top/bottom) and `offset_scale_*` size, hiding matplotlib's built-in top offset; `"arb"` drops all ticks and appends " (arb. units)" to the label (unless it already says a.u./arb, or a manual `colorbar_label` override is set). |
```

- [ ] **Step 8: Run the full suite + lint, then commit**

```bash
python3 -m pytest -q && ruff check dfxm/common/plotting.py tests/test_plot_style.py
git add dfxm/common/plotting.py tests/test_plot_style.py docs/Codebase.md
git commit -m "feat(plotting): add_colorbar per-group format — arb units + custom scientific offset"
```

---

### Task 3: Thread the group through the volume render path

`render.layer_figure` → `add_colorbar`, fed by `figures.volume_layer_specs`. Covers mosaicity, rocking, and visualize (all render via `layer_figure`).

**Files:**
- Modify: `dfxm/common/render.py` (`layer_figure` ~L38-67)
- Modify: `dfxm/common/figures.py` (`volume_layer_specs.make.build` ~L124-141)
- Modify: `docs/Codebase.md` (`layer_figure` bullet ~L237)
- Test: `tests/test_figures_catalog.py`

**Interfaces:**
- Consumes: `add_colorbar(..., group=…)` (Task 2); `volume_layer_specs(cmap_group=…)` (existing).
- Produces: `layer_figure(layer, vmin, vmax, cmap, ext_x, ext_y, title, cbar_label, *, style=None, group=None)`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_figures_catalog.py` (a focused unit test on `layer_figure`):

```python
def test_layer_figure_threads_group_to_arbitrary_units():
    import numpy as np

    from dfxm.common.plotting import PlotStyle
    from dfxm.common.render import layer_figure

    style = PlotStyle(tickfmt_raw="arb")
    fig, ax, im = layer_figure(
        np.arange(100).reshape(10, 10).astype(float),
        0.0,
        99.0,
        "gray",
        50.0,
        30.0,
        "Raw",
        "Intensity",
        style=style,
        group="raw",
    )
    cbar_ax = fig.axes[1]
    assert list(cbar_ax.get_yticks()) == []  # raw+arb -> no numeric ticks
    assert cbar_ax.get_ylabel() == "Intensity (arb. units)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_figures_catalog.py::test_layer_figure_threads_group_to_arbitrary_units -q`
Expected: FAIL (`layer_figure() got an unexpected keyword argument 'group'`).

- [ ] **Step 3: Add `group` to `layer_figure`**

In `dfxm/common/render.py`, change the signature:

```python
def layer_figure(layer, vmin, vmax, cmap, ext_x, ext_y, title, cbar_label, *, style=None, group=None):
```

and the `add_colorbar` call inside it:

```python
    if st.colorbar:
        add_colorbar(fig, im, ax, cbar_label, st, group=group)
```

- [ ] **Step 4: Forward `cmap_group` as `group` in `volume_layer_specs`**

In `dfxm/common/figures.py`, inside `volume_layer_specs`'s nested `build`, pass the group to `layer_figure`:

```python
            fig, _, _ = render.layer_figure(
                layer,
                vmin,
                vmax,
                resolve_cmap(style, cmap_group, fallback=cmap),
                ext_x,
                ext_y,
                f"{title}{zlabel} (layer {z})",
                cbar_label,
                style=style,
                group=cmap_group,
            )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_figures_catalog.py::test_layer_figure_threads_group_to_arbitrary_units -q`
Expected: PASS.

- [ ] **Step 6: Update `docs/Codebase.md`**

Update the `layer_figure` bullet (~L237) signature to include `group=None`:

```markdown
- `layer_figure(layer, vmin, vmax, cmap, ext_x, ext_y, title, cbar_label, *, style=None, group=None)` — one equal-aspect layer figure. `style=None` reproduces the legacy look (12×10 in, plain colourbar, no scale bar). When a `PlotStyle` is passed, figsize/colourbar/scale-bar/text-scaling are honoured; `group` (a `CMAP_GROUPS` name) selects the per-group colourbar tick format. Returns `(fig, ax, im)`.
```

- [ ] **Step 7: Run the full suite + lint, then commit**

```bash
python3 -m pytest -q && ruff check dfxm/common/render.py dfxm/common/figures.py tests/test_figures_catalog.py
git add dfxm/common/render.py dfxm/common/figures.py docs/Codebase.md tests/test_figures_catalog.py
git commit -m "feat(render): thread quantity group into layer_figure colourbars"
```

---

### Task 4: Thread the group in stages that call `add_colorbar` directly

`strain.py` (always `"strain"`), `slices.py` (via `GROUP_BY_KIND[kind]`, stored in `prep`), `profiles.py` (via the field/ref `kind`). Reuse the shared `GROUP_BY_KIND` from Task 1.

**Files:**
- Modify: `dfxm/stages/strain.py` (add_colorbar calls L405, L458)
- Modify: `dfxm/stages/slices.py` (import `GROUP_BY_KIND`; `_GROUP_BY_KIND` → shared; `prep["group"]` in `prepare_volume` ~L697-718 and in the catalog `build` ~L1021-1025; `build_slice_figure` add_colorbar L775)
- Modify: `dfxm/stages/profiles.py` (add_colorbar calls L470, L520)
- Modify: `docs/Codebase.md` (`build_slice_figure` bullet ~L330)
- Test: `tests/test_stage_slices.py`

**Interfaces:**
- Consumes: `add_colorbar(..., group=…)` (Task 2), `GROUP_BY_KIND` (Task 1).
- Produces: `prep["group"]` key on slices prep dicts (a `CMAP_GROUPS` name or `None`).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_stage_slices.py`:

```python
def test_build_slice_figure_raw_arbitrary_units_drops_ticks():
    import numpy as np

    from dfxm.common.plotting import PlotStyle
    from dfxm.stages.slices import build_slice_figure

    u = np.linspace(0.0, 40.0, 40)
    v = np.linspace(0.0, 30.0, 30)
    data = np.arange(30 * 40, dtype=float).reshape(30, 40)
    prep = {
        "cmap_name": "gray",
        "vmin": 0.0,
        "vmax": float(data.max()),
        "center_zero": False,
        "title": "Sum intensity",
        "cbar_label": "Sum intensity (a.u.)",
        "group": "raw",
    }
    style = PlotStyle(tickfmt_raw="arb")
    fig = build_slice_figure(prep, {"name": "s"}, data, u, v, offset_um=None, style=style)
    cbar_ax = fig.axes[1]
    assert list(cbar_ax.get_yticks()) == []
    assert cbar_ax.get_ylabel() == "Sum intensity (a.u.)"  # already a.u. -> unchanged
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_slices.py::test_build_slice_figure_raw_arbitrary_units_drops_ticks -q`
Expected: FAIL (ticks still present — `build_slice_figure` does not pass `group` yet).

- [ ] **Step 3: `strain.py` — pass `group="strain"`**

At L405: `add_colorbar(fig, im, ax, "Strain (ε)", style, group="strain")`
At L458: `add_colorbar(fig, im, ax, title, style, group="strain")`

- [ ] **Step 4: `slices.py` — share `GROUP_BY_KIND`, store the group in prep, pass it through**

Add to the imports from `..common.plotting` (the block that already imports `resolve_cmap`):

```python
    GROUP_BY_KIND,
```

Delete the local definition (L631-638):

```python
# PlotStyle colormap group per volume kind.
_GROUP_BY_KIND: dict[str, str] = {
    "mosa_com": "mosa_com",
    "mosa_fwhm": "mosa_fwhm",
    "strain": "strain",
    "raw_sum": "raw",
    "raw_specific": "raw",
}
```

Replace the two `_GROUP_BY_KIND` references with `GROUP_BY_KIND`:
- L710: `"cmap_name": resolve_cmap(style, GROUP_BY_KIND.get(kind)),`
- add a sibling key right after it: `"group": GROUP_BY_KIND.get(kind),`
- L1024: `style, GROUP_BY_KIND.get(kind), fallback=prep["cmap_name"]`

In the catalog `build` closure (~L1021), add the group to the rebuilt prep right after the `cmap_name` line:

```python
                        prep["cmap_name"] = resolve_cmap(
                            style, GROUP_BY_KIND.get(kind), fallback=prep["cmap_name"]
                        )
                        prep["group"] = GROUP_BY_KIND.get(kind)
```

In `build_slice_figure` (L775):

```python
    if st.colorbar:
        add_colorbar(fig, im, ax, prep["cbar_label"], st, group=prep.get("group"))
```

- [ ] **Step 5: `profiles.py` — pass the ref/field group**

`profiles` stores `kind` on each field's attrs. Import the shared map — add to the `from ..common.plotting import (...)` block:

```python
    GROUP_BY_KIND,
```

At L470 (companion reference image):

```python
        add_colorbar(fig, im, ax_img, ref_attrs["cbar_label"], style, group=GROUP_BY_KIND.get(ref_attrs.get("kind")))
```

At L520 (single reference image):

```python
            add_colorbar(fig, im, ax, attrs["cbar_label"], style, group=GROUP_BY_KIND.get(attrs.get("kind")))
```

(`GROUP_BY_KIND.get(...)` returns `None` for an unknown/missing kind → neutral `"auto"`, i.e. today's behaviour — safe.)

- [ ] **Step 6: Run the test + full suite**

Run: `python3 -m pytest tests/test_stage_slices.py::test_build_slice_figure_raw_arbitrary_units_drops_ticks -q && python3 -m pytest -q`
Expected: PASS.

- [ ] **Step 7: Update `docs/Codebase.md`**

Update the `build_slice_figure` bullet (~L330) to note the prep carries `group`:

```markdown
- `build_slice_figure(prep, sl, slice2d, u_um, v_um, *, offset_um, style=None)` — build and return a slice `Figure` (equal-aspect, µm axes). When `style` is `None` the legacy appearance is reproduced; otherwise figsize/colourbar/scale-bar/text-scaling are honoured. `prep["group"]` (a `CMAP_GROUPS` name, via `GROUP_BY_KIND[kind]`) selects the per-group colourbar tick format. Does NOT call `savefig`.
```

- [ ] **Step 8: Lint + commit**

```bash
ruff check dfxm/stages/strain.py dfxm/stages/slices.py dfxm/stages/profiles.py tests/test_stage_slices.py
git add dfxm/stages/strain.py dfxm/stages/slices.py dfxm/stages/profiles.py docs/Codebase.md tests/test_stage_slices.py
git commit -m "feat(stages): pass quantity group to add_colorbar in strain/slices/profiles"
```

---

### Task 5: GUI — per-group colourbar controls + scroll area

Replace the single "Tick format" row with a compact per-group subsection and stop referencing `colorbar_tick_format`. Wrap `StyleControls` in a `QScrollArea` in the export dialog.

**Files:**
- Modify: `gui/widgets/export_dialog.py` (`_TICK_FMTS`/labels L37-45; `StyleControls.sync_from_style` L116-117; `_all_widgets` L164-167; `_build_controls` colourbar section L393-404; `ExportDialog.__init__` `right` layout L579-583; add `QScrollArea` import)
- Modify: `docs/Usage.md` (Tick-format row L452 + the ×10ⁿ tip L455-456)
- Test: `tests/gui_smoke.py`

**Interfaces:**
- Consumes: per-group `PlotStyle` fields + `CMAP_GROUPS` (Task 1).
- Produces: per-group widget dicts `StyleControls._w_tickfmt/_w_offscale/_w_offpos` keyed by group.

- [ ] **Step 1: Extend the module constants**

Replace L37-45:

```python
_TICK_FMTS = ["auto", "scientific", "arb", "0", "1", "2", "3"]
_TICK_FMT_LABELS = {
    "auto": "auto (matplotlib default)",
    "scientific": "scientific (×10ⁿ offset)",
    "arb": "arbitrary units (no ticks)",
    "0": "0 decimals (plain numbers)",
    "1": "1 decimal (plain numbers)",
    "2": "2 decimals (plain numbers)",
    "3": "3 decimals (plain numbers)",
}
_OFFSET_POS = ["top", "bottom"]
# (group field-suffix, friendly label) — drives the per-group colourbar rows.
_CBAR_GROUPS = (
    ("mosa_com", "Mosa misorientation"),
    ("mosa_fwhm", "Mosa FWHM"),
    ("strain", "Strain"),
    ("raw", "Raw intensity"),
)
```

- [ ] **Step 2: Replace the single Tick-format widget with per-group rows**

In `_build_controls`, DELETE the old block (L393-404):

```python
        self._w_cbar_fmt = QComboBox()
        for fmt in _TICK_FMTS:
            self._w_cbar_fmt.addItem(_TICK_FMT_LABELS[fmt], fmt)
        cur_fmt = s.colorbar_tick_format if s.colorbar_tick_format in _TICK_FMTS else "auto"
        self._w_cbar_fmt.setCurrentIndex(_TICK_FMTS.index(cur_fmt))
        self._w_cbar_fmt.currentIndexChanged.connect(
            lambda _i: (
                setattr(self._style, "colorbar_tick_format", self._w_cbar_fmt.currentData()),
                self._emit(),
            )
        )
        form.addRow("Tick format", self._w_cbar_fmt)
```

and REPLACE it with per-group controls:

```python
        form.addRow(QLabel("<b>Colourbar — per group</b>"))
        self._w_tickfmt: dict[str, QComboBox] = {}
        self._w_offscale: dict[str, QDoubleSpinBox] = {}
        self._w_offpos: dict[str, QComboBox] = {}
        for grp, label in _CBAR_GROUPS:
            fmt_combo = QComboBox()
            for fmt in _TICK_FMTS:
                fmt_combo.addItem(_TICK_FMT_LABELS[fmt], fmt)
            cur = getattr(s, f"tickfmt_{grp}")
            fmt_combo.setCurrentIndex(_TICK_FMTS.index(cur if cur in _TICK_FMTS else "auto"))
            fmt_combo.currentIndexChanged.connect(
                lambda _i, g=grp, c=fmt_combo: (
                    setattr(self._style, f"tickfmt_{g}", c.currentData()),
                    self._emit(),
                )
            )

            off_scale = QDoubleSpinBox()
            off_scale.setRange(0.2, 5.0)
            off_scale.setDecimals(2)
            off_scale.setSingleStep(0.1)
            off_scale.setValue(getattr(s, f"offset_scale_{grp}"))
            off_scale.setToolTip("Size of the scientific ×10ⁿ exponent (only when format = scientific).")
            off_scale.valueChanged.connect(
                lambda v, g=grp: (setattr(self._style, f"offset_scale_{g}", v), self._emit())
            )

            off_pos = QComboBox()
            off_pos.addItems(_OFFSET_POS)
            off_pos.setCurrentText(getattr(s, f"offset_pos_{grp}"))
            off_pos.setToolTip("Where the scientific ×10ⁿ exponent sits (only when format = scientific).")
            off_pos.currentTextChanged.connect(
                lambda v, g=grp: (setattr(self._style, f"offset_pos_{g}", v), self._emit())
            )

            row = QHBoxLayout()
            row.addWidget(fmt_combo, 2)
            row.addWidget(off_scale, 1)
            row.addWidget(off_pos, 1)
            form.addRow(label, row)
            self._w_tickfmt[grp] = fmt_combo
            self._w_offscale[grp] = off_scale
            self._w_offpos[grp] = off_pos
```

- [ ] **Step 3: Update `sync_from_style`**

Replace the two old tick-format lines (L116-117):

```python
        cur_fmt = s.colorbar_tick_format if s.colorbar_tick_format in _TICK_FMTS else "auto"
        self._w_cbar_fmt.setCurrentIndex(_TICK_FMTS.index(cur_fmt))
```

with the per-group sync:

```python
        for grp, _label in _CBAR_GROUPS:
            cur = getattr(s, f"tickfmt_{grp}")
            self._w_tickfmt[grp].setCurrentIndex(_TICK_FMTS.index(cur if cur in _TICK_FMTS else "auto"))
            self._w_offscale[grp].setValue(getattr(s, f"offset_scale_{grp}"))
            self._w_offpos[grp].setCurrentText(getattr(s, f"offset_pos_{grp}"))
```

- [ ] **Step 4: Update `_all_widgets`**

Remove `self._w_cbar_fmt,` from the returned list (L166) and append the per-group widgets to the flat list. Change the `return [ ... ]` to build and extend a list:

```python
    def _all_widgets(self) -> list[QWidget]:
        """Return a flat list of all leaf widgets (for blockSignals)."""
        widgets = [
            self._w_cmap_mosa_com,
            self._w_cmap_mosa_fwhm,
            self._w_cmap_strain,
            self._w_cmap_raw,
            self._w_scale_bar,
            self._w_bar_auto,
            self._w_bar_len,
            self._w_bar_thick,
            self._w_bar_label_scale,
            self._w_bar_color,
            self._w_bar_loc,
            self._w_bar_box,
            self._w_box_color,
            self._w_box_alpha,
            self._w_box_margin,
            self._w_font_scale,
            self._w_title_scale,
            self._w_show_title,
            self._w_center_labels,
            self._w_colorbar,
            self._w_cbar_label,
            self._w_cbar_frac,
            self._w_cbar_ticks,
            self._w_round_clim,
            self._w_fig_width,
            self._w_fmt_png,
            self._w_fmt_pdf,
            self._w_fmt_svg,
            self._w_dpi,
        ]
        for grp, _label in _CBAR_GROUPS:
            widgets += [self._w_tickfmt[grp], self._w_offscale[grp], self._w_offpos[grp]]
        return widgets
```

- [ ] **Step 5: Wrap `StyleControls` in a `QScrollArea` (export dialog)**

Add `QScrollArea` to the `PySide6.QtWidgets` import block. Then in `ExportDialog.__init__`, replace `right.addWidget(self._controls)` (L581) with:

```python
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setWidget(self._controls)

        right = QVBoxLayout()
        right.addWidget(self._selector)
        right.addWidget(controls_scroll, 1)
        right.addLayout(btns)
```

(Remove the now-duplicated `right = QVBoxLayout()` / `right.addWidget(self._selector)` / `right.addWidget(self._controls)` / `right.addStretch(1)` / `right.addLayout(btns)` block those five lines replace.)

- [ ] **Step 6: Run the GUI smoke test**

Run: `python3 tests/gui_smoke.py`
Expected: all steps pass (the export dialog opens; per-group controls render; no `colorbar_tick_format` AttributeError).

If `gui_smoke.py` asserts a specific `StyleControls` widget count or references `_w_cbar_fmt`, update that assertion to match the new per-group widgets.

- [ ] **Step 7: Update `docs/Usage.md`**

Replace the Tick-format table row (L452) with a per-group description:

```markdown
| Colourbar — per group | One row per quantity group (Mosa misorientation, Mosa FWHM, Strain, Raw intensity), each with **Tick format** + **offset size** + **offset position**. Tick format: **auto** (matplotlib default), **scientific (×10ⁿ offset)** (forces offset notation), **arbitrary units (no ticks)** (drops numeric ticks and marks the label "arb. units" — the default for Raw), or a digit-count (**0/1/2/3 decimals**). Offset size scales the scientific `×10ⁿ` exponent; offset position places it at the **top** or **bottom** of the bar (both only take effect when that group's format is scientific). |
```

Update the ×10ⁿ tip (L455-456) to mention the size/position + arbitrary-units controls:

```markdown
> [!tip] Why does my colourbar say ×10⁻²?
> The **scientific** tick-format forces offset notation: the ticks show plain multipliers (e.g. −8, 0, 8) and a `×10⁻³` exponent is drawn separately. That exponent is now a styleable label — set its **offset size** and **offset position** (top/bottom) per group. Prefer full numbers? Switch that group's Tick format to a digit-count (e.g. **3 decimals**) or **auto**. For images with no absolute scale (raw intensity), choose **arbitrary units (no ticks)**.
```

- [ ] **Step 8: Lint + commit**

```bash
python3 tests/gui_smoke.py && ruff check gui/widgets/export_dialog.py
git add gui/widgets/export_dialog.py docs/Usage.md tests/gui_smoke.py
git commit -m "feat(gui): per-group colourbar controls (tick format, offset size/position) + scroll area"
```

---

### Task 6: Retire `colorbar_tick_format` — tuned `PUBLICATION_STYLE` + old-dict migration

Nothing reads `colorbar_tick_format` after Task 5. Remove it, tune `PUBLICATION_STYLE`, and migrate old serialized styles to the tuned per-group profile.

**Files:**
- Modify: `dfxm/common/plotting.py` (PlotStyle field removal; `PUBLICATION_STYLE` ~L90-103; `_style_from_dict` ~L117-122)
- Modify: `docs/Codebase.md` (`PUBLICATION_STYLE` row ~L222)
- Modify: `tests/test_plot_style.py` (migration test), `tests/test_figure_layout.py` (`_BIG` + fixture)
- Test: `tests/test_plot_style.py`, `tests/test_figure_layout.py`

**Interfaces:**
- Consumes: per-group fields + helpers (Task 1), `add_colorbar(group=…)` (Task 2), slices `prep["group"]` (Task 4).
- Produces: `PUBLICATION_STYLE` with tuned per-group defaults; `_style_from_dict` migration for pre-feature dicts.

- [ ] **Step 1: Write the failing migration test**

Add to `tests/test_plot_style.py`:

```python
def test_style_from_dict_migrates_old_snapshot_to_tuned_defaults():
    from dfxm.common.plotting import _style_from_dict

    # An old GUI snapshot: no per-group tickfmt_* keys at all.
    old = {"font_scale": 2.2, "colorbar_ticks": 5}
    s = _style_from_dict(old)
    assert s.tickfmt_for("strain") == "scientific"
    assert s.tickfmt_for("raw") == "arb"
    assert s.tickfmt_for("mosa_com") == "auto"
    assert s.offset_pos_for("strain") == "bottom"

    # A current snapshot carrying the keys is left exactly as-is (no migration).
    new = {"tickfmt_strain": "auto", "tickfmt_raw": "auto", "font_scale": 1.0}
    s2 = _style_from_dict(new)
    assert s2.tickfmt_for("strain") == "auto"
    assert s2.tickfmt_for("raw") == "auto"


def test_publication_style_is_tuned_per_group():
    from dfxm.common.plotting import PUBLICATION_STYLE

    assert PUBLICATION_STYLE.tickfmt_for("strain") == "scientific"
    assert PUBLICATION_STYLE.tickfmt_for("raw") == "arb"
    assert PUBLICATION_STYLE.tickfmt_for("mosa_com") == "auto"
    assert PUBLICATION_STYLE.offset_pos_for("strain") == "bottom"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plot_style.py -q -k "migrat or tuned_per_group"`
Expected: FAIL (no migration; `PUBLICATION_STYLE` still uses `colorbar_tick_format`).

- [ ] **Step 3: Remove the deprecated field**

In `PlotStyle`, delete the line:

```python
    colorbar_tick_format: str = "auto"  # DEPRECATED (removed once GUI migrates); see tickfmt_*
```

- [ ] **Step 4: Tune `PUBLICATION_STYLE`**

Replace the `colorbar_tick_format="scientific",` line in `PUBLICATION_STYLE` with the tuned per-group block:

```python
    tickfmt_mosa_com="auto",
    tickfmt_mosa_fwhm="auto",
    tickfmt_strain="scientific",
    tickfmt_raw="arb",
    offset_pos_mosa_com="bottom",
    offset_pos_mosa_fwhm="bottom",
    offset_pos_strain="bottom",
    offset_pos_raw="bottom",
```

- [ ] **Step 5: Add the migration to `_style_from_dict`**

Replace `_style_from_dict`:

```python
def _style_from_dict(data: dict) -> PlotStyle:
    names = {f.name for f in fields(PlotStyle)}
    kwargs = {k: v for k, v in dict(data).items() if k in names}
    if isinstance(kwargs.get("formats"), list):
        kwargs["formats"] = tuple(kwargs["formats"])
    # Migration: a snapshot predating per-group tick formats has none of the
    # tickfmt_* keys. Give it the tuned profile (same as PUBLICATION_STYLE) so
    # old persisted/injected styles gain the sensible defaults. Reached only from
    # the serialized/GUI path — never the bare-PlotStyle style=None code path.
    _tickfmt_keys = ("tickfmt_mosa_com", "tickfmt_mosa_fwhm", "tickfmt_strain", "tickfmt_raw")
    if not any(k in data for k in _tickfmt_keys):
        kwargs.setdefault("tickfmt_strain", "scientific")
        kwargs.setdefault("tickfmt_raw", "arb")
        for grp in CMAP_GROUPS:
            kwargs.setdefault(f"offset_pos_{grp}", "bottom")
    return PlotStyle(**kwargs)
```

- [ ] **Step 6: Update `tests/test_figure_layout.py`**

Replace the `colorbar_tick_format="scientific",` line in `_BIG` (L14) with `tickfmt_mosa_com="scientific",` (the fixture is a χ-Misorientation / mosa_com map), and add a `"group": "mosa_com"` key to the fixture prep so `build_slice_figure` actually renders the scientific offset the overlap test measures. In `_slice_fixture`, the prep dict (L25-32) gains:

```python
        "cbar_label": "Misorientation (°)",
        "group": "mosa_com",
    }
```

- [ ] **Step 7: Run the migration + layout tests, then the full suite**

Run: `python3 -m pytest tests/test_plot_style.py tests/test_figure_layout.py -q && python3 -m pytest -q`
Expected: PASS. In particular `test_slice_figure_texts_do_not_overlap_at_publication_scale` must still pass — the custom scientific exponent (in layout) plus constrained layout keep the title and colourbar apart. If it fails, verify the custom exponent Text is left in layout (do NOT call `set_in_layout(False)` on it) so constrained layout reserves room for it.

- [ ] **Step 8: Update `docs/Codebase.md`**

Update the `PUBLICATION_STYLE` row (~L222):

```markdown
| `PUBLICATION_STYLE` | A ready-made `PlotStyle` tuned for publication: white scale bar with a box, font_scale=2.2, colourbar_ticks=5, single-column width, PNG+PDF+SVG at 300 dpi, and per-group tick formats (strain → scientific with the ×10ⁿ exponent at the **bottom**, mosaicity → auto, raw → arbitrary units). |
```

Also add a one-line note to the `PlotStyle` description (find where the dataclass is documented in Codebase.md) that per-group colourbar formatting is controlled by `tickfmt_<group>` / `offset_scale_<group>` / `offset_pos_<group>` with `tickfmt_for`/`offset_scale_for`/`offset_pos_for` lookups. If there is no such prose entry, add the fields to the nearest `PlotStyle` reference.

- [ ] **Step 9: Verify no stray references remain**

Run: `grep -rn "colorbar_tick_format" dfxm/ gui/ tests/ docs/`
Expected: no matches (empty output).

- [ ] **Step 10: Lint + commit**

```bash
python3 -m pytest -q && python3 tests/gui_smoke.py && ruff check dfxm/common/plotting.py tests/
git add dfxm/common/plotting.py tests/test_plot_style.py tests/test_figure_layout.py docs/Codebase.md
git commit -m "feat(plotting): tuned per-group PUBLICATION_STYLE + old-style migration; drop colorbar_tick_format"
```

---

## Self-Review

**Spec coverage:**
- Per-group tick format (schema + helpers) → Task 1. ✓
- `add_colorbar` arb + custom scientific offset (size + top/bottom) → Task 2. ✓
- Group threading: volume path → Task 3; strain/slices/profiles → Task 4. ✓
- GUI per-group controls + `_TICK_FMTS` "arb" + scroll area → Task 5. ✓ (pub-style dialog already scrolls — verified in `main_window._on_pub_style`.)
- Legacy `style=None` byte-identical (bare defaults auto/1.0/top) → Global Constraints + Task 1 defaults; never touched. ✓
- Tuned defaults in `PUBLICATION_STYLE` + old-dict migration → Task 6. ✓
- Field removal sequenced AFTER the GUI stops reading it → Task 6 after Task 5. ✓
- Docs in the same change → each task ships its Codebase.md/Usage.md edits. ✓
- Title-pad hack left as-is (top placement still uses it) → not removed; noted in Task 6 Step 7. ✓
- Tests updated for removed field (test_plot_style L202/L510, test_figure_layout L14) → Tasks 2 & 6. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. The one "if gui_smoke asserts a widget count, update it" (Task 5 Step 6) and "if no prose PlotStyle entry, add fields to nearest reference" (Task 6 Step 8) are conditional-on-inspection instructions with a concrete action, not deferred work.

**Type consistency:** `tickfmt_for`/`offset_scale_for`/`offset_pos_for` names match across Tasks 1→2→GUI. `group` keyword name consistent across `add_colorbar`, `layer_figure`, `volume_layer_specs`. `GROUP_BY_KIND` (public) replaces `_GROUP_BY_KIND` in every slices reference (Task 4) and is imported in slices + profiles. `prep["group"]` produced in Task 4, consumed by `build_slice_figure` in the same task. `_CBAR_GROUPS` / `_OFFSET_POS` used only within `export_dialog.py`.

**Ordering invariant:** field removal (Task 6) strictly follows the GUI migration (Task 5); between Tasks 1–5 `colorbar_tick_format` still exists (harmless, unread after Task 2), so every intermediate commit keeps the suite green.
