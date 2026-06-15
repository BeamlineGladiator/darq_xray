# Publication-Quality Plot Export — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the user re-render any figure the pipeline saves at a publication style (bigger text, sized colourbar, µm scale bar with optional background box) with a live preview, exporting to PNG/PDF/SVG.

**Architecture:** Approach A — a Qt-free `PlotStyle` + styled primitives in `dfxm/common/plotting.py`, plus a per-stage figure *catalog* (`dfxm/common/figures.py`) whose `FigureSpec.build(style)` rebuilds a figure from the persisted output data. The GUI is a thin shell: an export dialog with a live preview and controls bound to a `PlotStyle`. Legacy look is preserved by `style=None` on every builder, so normal stage runs are byte-for-byte unchanged.

**Tech Stack:** Python, NumPy, h5py, Matplotlib (explicit `Figure`/Agg API — never `pyplot`/`matplotlib.use`), PySide6 (GUI only), pytest. Spec: `docs/superpowers/specs/2026-06-15-plot-export-design.md`.

---

## Conventions for every task

- Run tests with `python3 -m pytest <path> -v`; lint with `ruff check . && ruff format --check .`.
- `dfxm/` MUST stay Qt-free (no PySide6/pyvista import at module load). The GUI imports the core, never the reverse.
- The offscreen GUI smoke is `tests/gui_smoke.py` — run as a script (`python3 tests/gui_smoke.py`), NEVER renamed into pytest collection.
- Commit messages end with: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- `ruff format` runs automatically on save; if a long line trips the 100-char house style, wrap strings with parenthesised implicit concatenation (trailing-space continuation), as the stage `help=`/`hint=` texts do.

## File Structure

- **`dfxm/common/plotting.py`** (modify) — add `PlotStyle`, `PUBLICATION_STYLE`, and the styled primitives `draw_scale_bar`, `apply_text_scale`, `add_colorbar`. Becomes the single home for figure styling.
- **`dfxm/common/render.py`** (modify) — `layer_figure` gains a `style` argument and routes through the primitives; the local `add_scale_bar` is removed (folded into `draw_scale_bar`).
- **`dfxm/common/figures.py`** (create) — `FigureSpec`, the `_FIGURE_CATALOGS` dispatch, `figures_for(...)`, and the shared `map_layer_spec(...)` / volume-loader helpers.
- **`dfxm/stages/*.py`** (modify) — each stage's figure functions gain a `style` argument (legacy when `None`); each stage gains a module-level `figures(result, params) -> list[FigureSpec]`.
- **`gui/widgets/export_dialog.py`** (create) — `ExportDialog` (live preview + controls) and the global style editor.
- **`gui/stage_view.py`** (modify) — Output-tab "Export…" / "Export all…" entry points.
- **`tests/test_plot_style.py`**, **`tests/test_figures_catalog.py`** (create); **`tests/gui_smoke.py`** (extend).

## Implementation note — legacy default

Current figures differ in whether they draw a scale bar (volume layers and slices do, in black; the strain map does not). So there is no single style value that reproduces *every* current figure. The mechanism: **every builder takes `style: PlotStyle | None = None`; `None` means "render exactly as today".** A non-`None` style is honored fully (this is what export uses). The regression tests call builders with `style=None`. `PUBLICATION_STYLE` is the explicit publication preset used as the GUI's starting global style.

---

# PHASE 1 — core styling layer (no behaviour change)

### Task 1: `PlotStyle` dataclass + `PUBLICATION_STYLE`

**Files:**
- Modify: `dfxm/common/plotting.py`
- Test: `tests/test_plot_style.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_plot_style.py
from dataclasses import replace
from dfxm.common.plotting import PlotStyle, PUBLICATION_STYLE


def test_default_style_is_conservative():
    s = PlotStyle()
    assert s.font_scale == 1.0
    assert s.formats == ("png",)
    assert s.colorbar_ticks == 0  # 0 => matplotlib default ticks


def test_publication_style_is_bigger():
    assert PUBLICATION_STYLE.font_scale >= 2.0
    assert PUBLICATION_STYLE.scale_bar is True
    assert PUBLICATION_STYLE.scale_bar_box is True
    assert set(PUBLICATION_STYLE.formats) == {"png", "pdf", "svg"}


def test_replace_makes_independent_copy():
    s = replace(PUBLICATION_STYLE, font_scale=1.5)
    assert s.font_scale == 1.5
    assert PUBLICATION_STYLE.font_scale != 1.5
```

- [ ] **Step 2: Run it — expect ImportError (`PlotStyle` not defined)**

Run: `python3 -m pytest tests/test_plot_style.py -v`

- [ ] **Step 3: Add `PlotStyle` + `PUBLICATION_STYLE` to `dfxm/common/plotting.py`**

Add near the top (after the imports):

```python
from dataclasses import dataclass


@dataclass
class PlotStyle:
    """How to render a figure for export. ``None`` (not this) means 'as today'."""

    # scale bar (map figures only)
    scale_bar: bool = True
    scale_bar_length_um: float | None = None  # None -> auto (~15% of X extent)
    scale_bar_thickness_pt: float = 3.0
    scale_bar_label_scale: float = 1.0  # multiplies font_scale for the bar label
    scale_bar_loc: str = "lower right"  # "lower right" | "lower left" | "upper right" | "upper left"
    scale_bar_color: str = "black"
    scale_bar_box: bool = False
    scale_bar_box_color: str = "black"
    scale_bar_box_alpha: float = 0.45
    scale_bar_box_margin_pt: float = 4.0
    # text
    font_scale: float = 1.0  # multiplies axis labels, ticks, title
    show_title: bool = True
    center_axis_labels: bool = True
    # colourbar
    colorbar: bool = True
    colorbar_label: str | None = None  # None -> the figure's own label
    colorbar_fraction: float = 0.046  # matplotlib colorbar `fraction` (thickness)
    colorbar_ticks: int = 0  # 0 -> matplotlib default; >=2 -> N evenly spaced incl min/mid/max
    colorbar_tick_format: str = "auto"  # "auto" | "scientific" | a digit count like "2"
    # figure
    figure_width: str | float = "auto"  # "single" | "double" | "auto" | width in inches
    # output
    formats: tuple[str, ...] = ("png",)
    dpi: int = 300


PUBLICATION_STYLE = PlotStyle(
    scale_bar=True,
    scale_bar_thickness_pt=4.0,
    scale_bar_label_scale=1.1,
    scale_bar_color="white",
    scale_bar_box=True,
    font_scale=2.2,
    colorbar_fraction=0.07,
    colorbar_ticks=5,
    colorbar_tick_format="scientific",
    figure_width="single",
    formats=("png", "pdf", "svg"),
    dpi=300,
)


def figure_size(style: PlotStyle, ext_x: float, ext_y: float) -> tuple[float, float] | None:
    """Figure (w, h) in inches from the width preset, preserving physical aspect.

    Returns ``None`` for ``figure_width="auto"`` so the builder keeps its own
    figsize (the legacy path). Height follows the physical aspect plus ~1in of
    headroom for the title/colourbar.
    """
    presets = {"single": 3.5, "double": 7.0}
    w = presets.get(style.figure_width) if isinstance(style.figure_width, str) else style.figure_width
    if w in (None, "auto"):
        return None
    aspect = (ext_y / ext_x) if ext_x else 1.0
    return (float(w), float(w) * aspect + 1.0)
```

Add a test to `tests/test_plot_style.py`:

```python
from dfxm.common.plotting import figure_size


def test_figure_size_auto_returns_none():
    assert figure_size(PlotStyle(figure_width="auto"), 50.0, 30.0) is None


def test_figure_size_single_preserves_aspect():
    w, h = figure_size(PlotStyle(figure_width="single"), 50.0, 25.0)
    assert w == 3.5 and abs(h - (3.5 * 0.5 + 1.0)) < 1e-9
```

- [ ] **Step 4: Run the test — expect PASS**

Run: `python3 -m pytest tests/test_plot_style.py -v`

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py
git commit -m "plotting: add PlotStyle + PUBLICATION_STYLE"
```

---

### Task 2: `draw_scale_bar` — one scale bar to replace the three

**Files:**
- Modify: `dfxm/common/plotting.py`
- Test: `tests/test_plot_style.py`

Context: three near-identical scale bars exist today — `plotting.add_scale_bar` (line-based, white), `render.add_scale_bar` (Rectangle, auto-rounded length, black), `slices._scale_bar` (Rectangle, auto-rounded, black). This unifies them; the auto length uses the rounded ~15% rule the renderers use.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_plot_style.py
import numpy as np
from matplotlib.figure import Figure
from dfxm.common.plotting import PlotStyle, draw_scale_bar, auto_scale_bar_length_um


def _ax(ext_x=50.0, ext_y=30.0):
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.zeros((10, 10)), extent=[0, ext_x, 0, ext_y], origin="lower")
    return fig, ax


def test_auto_length_rounds_to_nice_value():
    assert auto_scale_bar_length_um(50.0) == 10  # ~15% of 50 = 7.5 -> rounds to 10


def test_draw_scale_bar_adds_patch_and_text():
    fig, ax = _ax()
    n_before = len(ax.patches)
    draw_scale_bar(ax, length_um=10.0, style=PlotStyle(scale_bar_color="white"))
    assert len(ax.patches) == n_before + 1  # the bar
    assert any("µm" in t.get_text() for t in ax.texts)


def test_draw_scale_bar_box_adds_a_second_patch():
    fig, ax = _ax()
    draw_scale_bar(ax, length_um=10.0, style=PlotStyle(scale_bar_box=True))
    # one patch for the bar, one for the background box
    assert len(ax.patches) == 2
```

- [ ] **Step 2: Run it — expect ImportError**

Run: `python3 -m pytest tests/test_plot_style.py -k scale_bar -v`

- [ ] **Step 3: Implement `auto_scale_bar_length_um` + `draw_scale_bar`**

Replace the existing `add_scale_bar` in `dfxm/common/plotting.py` with:

```python
def auto_scale_bar_length_um(ext_x: float) -> float:
    """A 'nice' bar length ~15% of the X extent (rounded to 1/10/50 steps)."""
    target = ext_x * 0.15
    if target >= 100:
        sl = round(target / 50) * 50
    elif target >= 10:
        sl = round(target / 10) * 10
    elif target >= 1:
        sl = round(target)
    else:
        sl = round(target, 1)
    return sl or target


def draw_scale_bar(ax, length_um: float | None = None, *, style: "PlotStyle") -> None:
    """Draw a µm scale bar (and optional background box) per *style*.

    *ax* must use data coordinates in microns. ``length_um=None`` auto-sizes.
    """
    from matplotlib.patches import FancyBboxPatch, Rectangle

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xr, yr = (x1 - x0), (y1 - y0)
    sl = length_um if length_um is not None else auto_scale_bar_length_um(abs(xr))
    bh = abs(yr) * 0.012
    pad_x, pad_y = 0.05 * abs(xr), 0.05 * abs(yr)
    bx = (x1 - pad_x - sl) if "right" in style.scale_bar_loc else (x0 + pad_x)
    by = (y1 - pad_y - bh) if "upper" in style.scale_bar_loc else (y0 + pad_y)
    label = f"{sl:g} µm"
    label_size = 10.0 * style.font_scale * style.scale_bar_label_scale

    if style.scale_bar_box:
        m = style.scale_bar_box_margin_pt
        box = FancyBboxPatch(
            (bx, by),
            sl,
            bh + label_size * 0.02 * abs(yr),
            boxstyle=f"round,pad={m * 0.01 * abs(yr)}",
            transform=ax.transData,
            facecolor=style.scale_bar_box_color,
            edgecolor="none",
            alpha=style.scale_bar_box_alpha,
            zorder=4,
        )
        ax.add_patch(box)

    ax.add_patch(
        Rectangle(
            (bx, by),
            sl,
            bh,
            facecolor=style.scale_bar_color,
            edgecolor=style.scale_bar_color,
            linewidth=style.scale_bar_thickness_pt,
            zorder=5,
        )
    )
    ax.text(
        bx + sl / 2.0,
        by + bh * 2.5,
        label,
        color=style.scale_bar_color,
        fontsize=label_size,
        fontweight="bold",
        ha="center",
        va="bottom",
        zorder=6,
    )
```

(Note: the box geometry above is approximate; the executor should eyeball one rendered PNG and adjust the box padding so it visibly encloses bar + label. The test only asserts patch/text counts.)

- [ ] **Step 4: Run — expect PASS**

Run: `python3 -m pytest tests/test_plot_style.py -k scale_bar -v`

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py
git commit -m "plotting: unified draw_scale_bar with optional background box"
```

---

### Task 3: `apply_text_scale`

**Files:**
- Modify: `dfxm/common/plotting.py`
- Test: `tests/test_plot_style.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_plot_style.py
from dfxm.common.plotting import apply_text_scale


def test_apply_text_scale_grows_label_fonts():
    fig, ax = _ax()
    ax.set_xlabel("X (µm)")
    base = ax.xaxis.label.get_fontsize()
    apply_text_scale(ax, PlotStyle(font_scale=2.0))
    assert ax.xaxis.label.get_fontsize() == base * 2.0


def test_apply_text_scale_hides_title_when_asked():
    fig, ax = _ax()
    ax.set_title("keep me?")
    apply_text_scale(ax, PlotStyle(show_title=False))
    assert ax.get_title() == ""
```

- [ ] **Step 2: Run — expect ImportError**

Run: `python3 -m pytest tests/test_plot_style.py -k text_scale -v`

- [ ] **Step 3: Implement `apply_text_scale`**

```python
def apply_text_scale(ax, style: "PlotStyle") -> None:
    """Scale axis-label/tick/title fonts by ``style.font_scale``; apply title/centre options."""
    fs = style.font_scale
    for label in (ax.xaxis.label, ax.yaxis.label):
        label.set_fontsize(label.get_fontsize() * fs)
        if style.center_axis_labels:
            label.set_ha("center")
    ax.tick_params(labelsize=ax.xaxis.get_ticklabels()[0].get_fontsize() * fs
                   if ax.xaxis.get_ticklabels() else 10 * fs)
    title = ax.title
    if not style.show_title:
        ax.set_title("")
    else:
        title.set_fontsize(title.get_fontsize() * fs)
```

(If `tick_params(labelsize=...)` interacts awkwardly with already-drawn ticks, set it before drawing in the builder; the test above only checks the label font and the title.)

- [ ] **Step 4: Run — expect PASS**

Run: `python3 -m pytest tests/test_plot_style.py -k text_scale -v`

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py
git commit -m "plotting: apply_text_scale (font scaling + title/centre options)"
```

---

### Task 4: `add_colorbar` with configurable label, ticks, format, thickness

**Files:**
- Modify: `dfxm/common/plotting.py`
- Test: `tests/test_plot_style.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_plot_style.py
from dfxm.common.plotting import add_colorbar, colorbar_tick_values


def test_colorbar_tick_values_includes_extremes_and_mid():
    vals = colorbar_tick_values(-2e-3, 2e-3, n=5)
    assert len(vals) == 5
    assert vals[0] == -2e-3 and vals[-1] == 2e-3
    assert abs(vals[2]) < 1e-12  # midpoint of a symmetric range is 0


def test_add_colorbar_sets_label_and_tick_count():
    fig, ax = _ax()
    im = ax.imshow(np.linspace(-2e-3, 2e-3, 100).reshape(10, 10),
                   extent=[0, 50, 0, 30], origin="lower")
    cb = add_colorbar(fig, im, ax, "Strain (ε)",
                      PlotStyle(colorbar_ticks=5, colorbar_tick_format="scientific"))
    assert cb.ax.get_ylabel() == "Strain (ε)"
    assert len(cb.get_ticks()) == 5
```

- [ ] **Step 2: Run — expect ImportError**

Run: `python3 -m pytest tests/test_plot_style.py -k colorbar -v`

- [ ] **Step 3: Implement `colorbar_tick_values` + `add_colorbar`**

```python
from matplotlib.ticker import FuncFormatter, ScalarFormatter


def colorbar_tick_values(vmin: float, vmax: float, n: int):
    """``n`` evenly-spaced tick values from vmin..vmax (always includes both ends)."""
    import numpy as np

    return list(np.linspace(vmin, vmax, max(2, n)))


def _tick_formatter(fmt: str):
    if fmt == "scientific":
        f = ScalarFormatter(useMathText=True)
        f.set_powerlimits((0, 0))
        return f
    if fmt != "auto":
        try:
            d = int(fmt)
            return FuncFormatter(lambda v, _pos: f"{v:.{d}f}")
        except ValueError:
            pass
    return None  # matplotlib default


def add_colorbar(fig, im, ax, label, style: "PlotStyle"):
    """Add a colourbar honouring thickness, label, tick count and number format."""
    cb = fig.colorbar(im, ax=ax, fraction=style.colorbar_fraction, pad=0.04)
    text = style.colorbar_label if style.colorbar_label is not None else label
    cb.set_label(text, fontsize=10 * style.font_scale)
    if style.colorbar_ticks and style.colorbar_ticks >= 2:
        cb.set_ticks(colorbar_tick_values(im.norm.vmin, im.norm.vmax, style.colorbar_ticks))
    fmt = _tick_formatter(style.colorbar_tick_format)
    if fmt is not None:
        cb.ax.yaxis.set_major_formatter(fmt)
    cb.ax.tick_params(labelsize=9 * style.font_scale)
    return cb
```

- [ ] **Step 4: Run — expect PASS**

Run: `python3 -m pytest tests/test_plot_style.py -k colorbar -v`

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py
git commit -m "plotting: add_colorbar with configurable label/ticks/format/thickness"
```

---

### Task 5: `render.layer_figure` accepts a style (legacy when None)

**Files:**
- Modify: `dfxm/common/render.py`
- Test: `tests/test_figures_catalog.py`

Context: `layer_figure` is the shared per-layer map renderer used by visualize, rocking, and matched. Today it always draws a black auto scale bar + a 0.046 colourbar. Add `style`; `None` reproduces today; a style routes through the Task 2–4 primitives. Remove the local `add_scale_bar`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_figures_catalog.py
import numpy as np
from dfxm.common import render
from dfxm.common.plotting import PlotStyle


def _layer():
    return np.random.default_rng(0).normal(size=(20, 40))


def test_layer_figure_legacy_has_scale_bar_and_equal_aspect():
    fig, ax, im = render.layer_figure(_layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb")
    assert ax.get_aspect() == 1.0  # physical equal aspect
    assert len(ax.patches) >= 1    # legacy scale bar present


def test_layer_figure_style_off_drops_scale_bar():
    fig, ax, im = render.layer_figure(
        _layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb",
        style=PlotStyle(scale_bar=False),
    )
    assert len(ax.patches) == 0
```

- [ ] **Step 2: Run — expect failure (unexpected `style` kwarg)**

Run: `python3 -m pytest tests/test_figures_catalog.py -k layer_figure -v`

- [ ] **Step 3: Refactor `layer_figure`; delete `render.add_scale_bar`**

In `dfxm/common/render.py`, delete the `add_scale_bar` function and rewrite `layer_figure`:

```python
from .plotting import add_colorbar, apply_text_scale, draw_scale_bar, get_cmap  # noqa: F401


def layer_figure(layer, vmin, vmax, cmap, ext_x, ext_y, title, cbar_label, *, style=None):
    """Single equal-aspect layer figure (µm axes). ``style=None`` == legacy look."""
    from .plotting import PlotStyle

    st = style if style is not None else PlotStyle(scale_bar_color="black", colorbar_fraction=0.046)
    from .plotting import figure_size

    figsize = (figure_size(st, ext_x, ext_y) or (12, 10)) if style is not None else (12, 10)
    fig = Figure(figsize=figsize, facecolor="white")
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
        add_colorbar(fig, im, ax, cbar_label, st)
    if st.scale_bar:
        draw_scale_bar(ax, st.scale_bar_length_um, style=st)
    apply_text_scale(ax, st)
    return fig, ax, im
```

Note: legacy callers pass no `style`, so they get the black-bar / 0.046-cbar look. `apply_text_scale` at `font_scale=1.0` is a no-op on sizes. The legacy default also keeps `show_title=True`.

- [ ] **Step 4: Run the suite (catch any caller breakage) — expect PASS**

Run: `python3 -m pytest tests/test_figures_catalog.py -k layer_figure -v && python3 -m pytest -q`

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/render.py tests/test_figures_catalog.py
git commit -m "render: layer_figure routes through styled primitives (legacy when style=None)"
```

---

### Task 6: strain builders take a style + return Figures

**Files:**
- Modify: `dfxm/stages/strain.py` (`_save_strain_map`, `_save_histogram`, `_save_detrend_diag`, and their callers in `process_maps_file`)
- Test: `tests/test_stage_strain.py` (add cases)

- [ ] **Step 1: Read the current functions** (`strain.py` ~331–386 and the caller ~417–428) so the refactor preserves the legacy `savefig` dpi values (200 map, 150 hist, 120 diag).

- [ ] **Step 2: Write the failing test**

```python
# add to tests/test_stage_strain.py
import numpy as np
from dfxm.stages import strain
from dfxm.common.plotting import PlotStyle


def test_build_strain_map_legacy_has_no_scale_bar():
    fig = strain.build_strain_map(np.random.rand(20, 30) * 1e-3, 0.1, 0.3, None, (None, None))
    ax = fig.axes[0]
    assert len(ax.patches) == 0  # today's strain map has no scale bar


def test_build_strain_map_style_adds_scale_bar():
    fig = strain.build_strain_map(
        np.random.rand(20, 30) * 1e-3, 0.1, 0.3, None, (None, None),
        style=PlotStyle(scale_bar=True),
    )
    assert len(fig.axes[0].patches) >= 1
```

- [ ] **Step 3: Refactor.** Rename `_save_strain_map` → `build_strain_map(strain, px, py, roi, vlim, *, style=None) -> Figure` that builds and **returns** the figure (no `savefig`); when `style is None` it draws no scale bar (legacy) and uses the current colourbar; when a style is given, route the colourbar through `add_colorbar`, fonts through `apply_text_scale`, and draw a scale bar via `draw_scale_bar` when `style.scale_bar`. Do the same for `_save_histogram` → `build_strain_histogram(data, *, title=..., xlabel=..., style=None) -> Figure | None` and `_save_detrend_diag` → `build_detrend_diag(original, detrended, surface, *, style=None) -> Figure`. In `process_maps_file`, replace the three `_save_*` calls with `fig = build_*(...); fig.savefig(path, dpi=<legacy dpi>, bbox_inches="tight", facecolor="white")` so saved output is unchanged. When a non-`None` style is passed to `build_strain_map`, derive the figsize from `plotting.figure_size(style, nx*px, ny*py)` (fall back to the current computed size when it returns `None`).

- [ ] **Step 4: Run strain tests + full suite — expect PASS (no behaviour change)**

Run: `python3 -m pytest tests/test_stage_strain.py -v && python3 -m pytest -q`

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/strain.py tests/test_stage_strain.py
git commit -m "strain: figure builders return Figures + accept a style (legacy preserved)"
```

---

### Task 7: slices builder takes a style + returns a Figure

**Files:**
- Modify: `dfxm/stages/slices.py` (`render_slice_png`, delete `_scale_bar`, update the caller)
- Test: `tests/test_stage_slices.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_stage_slices.py
import numpy as np
from dfxm.stages import slices as S
from dfxm.common.plotting import PlotStyle


def _prep():
    return {"cmap_name": "viridis", "title": "t", "cbar_label": "cb",
            "vmin": -1.0, "vmax": 1.0, "center_zero": False}


def test_build_slice_figure_returns_figure_with_equal_aspect():
    sl = {"name": "p0"}
    s2d = np.random.rand(10, 12)
    fig = S.build_slice_figure(_prep(), sl, s2d, np.linspace(0, 12, 12),
                               np.linspace(0, 10, 10), offset_um=None, style=PlotStyle(scale_bar=False))
    assert fig.axes[0].get_aspect() == 1.0
    assert len(fig.axes[0].patches) == 0
```

(`_make_norm(prep)` reads `prep` keys; mirror the keys the current `render_slice_png` uses — the executor confirms them when reading the function.)

- [ ] **Step 2: Run — expect AttributeError (`build_slice_figure` missing)**

Run: `python3 -m pytest tests/test_stage_slices.py -k build_slice_figure -v`

- [ ] **Step 3: Refactor.** Delete `slices._scale_bar`. Rename `render_slice_png(prep, sl, slice2d, u_um, v_um, out_png, offset_um, dpi=150)` → `build_slice_figure(prep, sl, slice2d, u_um, v_um, *, offset_um, style=None) -> Figure` that returns the figure; route colourbar/fonts/scale-bar through the primitives (legacy when `style is None`: black auto bar, 0.046 cbar). Add a thin `save_slice_png(...)` wrapper used by `run()` that calls `build_slice_figure(..., style=None).savefig(out_png, dpi=dpi, facecolor="white", bbox_inches="tight")`. Update the `run()` call site. When a non-`None` style is passed, derive the figsize from `plotting.figure_size(style, u_um[-1]-u_um[0], v_um[-1]-v_um[0])` (fall back to `(12, 10)` when it returns `None`).

- [ ] **Step 4: Run slices tests + full suite — expect PASS**

Run: `python3 -m pytest tests/test_stage_slices.py -v && python3 -m pytest -q`

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/slices.py tests/test_stage_slices.py
git commit -m "slices: build_slice_figure returns a Figure + accepts a style (legacy preserved)"
```

---

# PHASE 2 — figure catalog + map figures + export dialog

### Task 8: `FigureSpec` + catalog dispatch + empty catalogs

**Files:**
- Create: `dfxm/common/figures.py`
- Test: `tests/test_figures_catalog.py`

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_figures_catalog.py
from dfxm.common import figures
from dfxm.stages.registry import STAGE_TARGETS


def test_every_stage_has_a_catalog_entry():
    assert set(figures._FIGURE_CATALOGS) == set(STAGE_TARGETS)


def test_figures_for_concat_is_empty():
    assert figures.figures_for("concat", object(), {}) == []
```

- [ ] **Step 2: Run — expect ImportError**

Run: `python3 -m pytest tests/test_figures_catalog.py -k catalog -v`

- [ ] **Step 3: Create `dfxm/common/figures.py`**

```python
"""Per-stage figure catalog: enumerate + rebuild saved figures at any PlotStyle.

Qt-free. The GUI calls :func:`figures_for` to list a result's figures and
``spec.build(style)`` to get a Matplotlib Figure rebuilt from the persisted data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from matplotlib.figure import Figure

from .plotting import PlotStyle


@dataclass
class FigureSpec:
    figure_id: str
    title: str
    kind: str  # "map" (gets a scale bar) | "plot"
    filename: str  # suggested export stem (no extension)
    build: Callable[[PlotStyle | None], Figure]


# stage name -> fn(result, params) -> list[FigureSpec]
_FIGURE_CATALOGS: dict[str, Callable[[object, dict], list[FigureSpec]]] = {}


def register(stage_name: str):
    def deco(fn):
        _FIGURE_CATALOGS[stage_name] = fn
        return fn
    return deco


def figures_for(stage_name: str, result, params: dict) -> list[FigureSpec]:
    fn = _FIGURE_CATALOGS.get(stage_name)
    return fn(result, params) if fn else []


def _empty(result, params):  # concat, paraview: no saved figures
    return []


for _name in ("concat", "paraview"):
    _FIGURE_CATALOGS[_name] = _empty
```

Then make each stage register its catalog at import time. In `dfxm/common/figures.py`, after the empties, import the per-stage registrars (added in later tasks):

```python
def _load_stage_catalogs() -> None:
    # imported lazily to avoid import cycles; each module calls register(...)
    from dfxm.stages import (  # noqa: F401
        matched, mosaicity, profiles, rocking, slices, strain, visualize,
    )


_load_stage_catalogs()
```

Until later tasks add `figures()` to those modules, define them to register an empty list there (added per task). For Task 8, give every not-yet-implemented stage a temporary empty registration so the sync test passes:

```python
for _name in ("strain", "mosaicity", "visualize", "rocking", "slices", "matched", "profiles"):
    _FIGURE_CATALOGS.setdefault(_name, _empty)
```

(The `setdefault` lets later tasks override with real catalogs.)

- [ ] **Step 4: Run — expect PASS**

Run: `python3 -m pytest tests/test_figures_catalog.py -k catalog -v`

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/figures.py tests/test_figures_catalog.py
git commit -m "figures: FigureSpec + catalog dispatch (empty entries for every stage)"
```

---

### Task 9: shared volume helpers for map catalogs

**Files:**
- Modify: `dfxm/common/figures.py`
- Test: `tests/test_figures_catalog.py`

Context: strain/mosaicity/visualize/rocking all render per-Z-layer 2D arrays of a `(Z,Y,X)` volume via `render.layer_figure`. A shared helper builds the `FigureSpec`s.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_figures_catalog.py
import h5py
from dfxm.common.figures import volume_layer_specs


def test_volume_layer_specs_one_per_layer(tmp_path):
    vol = np.random.rand(3, 20, 40)
    h5 = tmp_path / "v.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=vol)
    specs = volume_layer_specs(
        h5_path=str(h5), dataset="strain", id_prefix="strain",
        title="Strain", cbar_label="Strain (ε)", cmap="RdBu_r",
        sx=0.1, sy=0.3, vmin=-1.0, vmax=1.0,
    )
    assert len(specs) == 3
    fig = specs[0].build(None)
    assert fig.axes[0].get_aspect() == 1.0
```

- [ ] **Step 2: Run — expect ImportError**

Run: `python3 -m pytest tests/test_figures_catalog.py -k volume_layer -v`

- [ ] **Step 3: Implement `volume_layer_specs` in `figures.py`**

```python
def _load_layer(h5_path: str, dataset: str, z: int):
    import h5py

    with h5py.File(h5_path, "r") as f:
        return f[dataset][z]


def volume_layer_specs(
    *, h5_path, dataset, id_prefix, title, cbar_label, cmap, sx, sy, vmin, vmax, z_um=None
):
    """One ``map`` FigureSpec per Z layer of a (Z,Y,X) HDF5 volume."""
    import h5py

    from . import render

    with h5py.File(h5_path, "r") as f:
        n_z = f[dataset].shape[0]
        ext_x = f[dataset].shape[2] * sx
        ext_y = f[dataset].shape[1] * sy

    def make(z):
        def build(style):
            layer = _load_layer(h5_path, dataset, z)
            zlabel = f"\nZ = {z_um[z]:.2f} µm" if z_um is not None else ""
            fig, _, _ = render.layer_figure(
                layer, vmin, vmax, cmap, ext_x, ext_y,
                f"{title}{zlabel} (layer {z})", cbar_label, style=style,
            )
            return fig
        return build

    return [
        FigureSpec(
            figure_id=f"{id_prefix}_z{z:04d}",
            title=f"{title} — layer {z}",
            kind="map",
            filename=f"{id_prefix}_layer_{z:04d}",
            build=make(z),
        )
        for z in range(n_z)
    ]
```

- [ ] **Step 4: Run — expect PASS**

Run: `python3 -m pytest tests/test_figures_catalog.py -k volume_layer -v`

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/figures.py tests/test_figures_catalog.py
git commit -m "figures: shared volume_layer_specs helper"
```

---

### Task 10: `strain.figures()` — map (strain map per layer)

**Files:**
- Modify: `dfxm/stages/strain.py`
- Test: `tests/test_figures_catalog.py`

- [ ] **Step 1: Write the failing test** (build a synthetic `StrainResult` with a stacked h5)

```python
# add to tests/test_figures_catalog.py
from dfxm.stages import strain as Strain


def test_strain_catalog_map_per_layer(tmp_path):
    vol = np.random.rand(2, 15, 25) * 1e-3
    h5 = tmp_path / "stacked.h5"
    with h5py.File(h5, "w") as f:
        f.create_dataset("strain", data=vol)
    res = Strain.StrainResult(
        stacked_path=str(h5), volume_shape=vol.shape, output_dir=str(tmp_path),
        layers=[Strain.LayerResult("L0", (15, 25), -1e-3, 1e-3, 0, 0),
                Strain.LayerResult("L1", (15, 25), -1e-3, 1e-3, 0, 0)],
    )
    specs = [s for s in Strain.figures(res, {"pixel_size_x_um": 0.1, "pixel_size_y_um": 0.3})
             if s.kind == "map"]
    assert len(specs) == 2
    assert specs[0].build(None).axes[0].get_aspect() == 1.0
```

- [ ] **Step 2: Run — expect AttributeError (`strain.figures` missing)**

Run: `python3 -m pytest tests/test_figures_catalog.py -k strain_catalog -v`

- [ ] **Step 3: Add `figures()` to `strain.py`** and register it

The strain map uses `build_strain_map` (Task 6) over `strain[i]` from `stacked_path`, not the generic `layer_figure` (it uses RdBu_r + `physical_extent`). One `map` spec per layer:

```python
from dfxm.common.figures import FigureSpec, register


@register("strain")
def figures(result, params):
    if not result.stacked_path:
        return []
    px = float(params.get("pixel_size_x_um", 1.0))
    py = float(params.get("pixel_size_y_um", 1.0))
    specs = []
    for i, layer in enumerate(result.layers):
        def build(style, i=i, lr=layer):
            arr = load_map(result.stacked_path, "strain")[i]
            return build_strain_map(arr, px, py, None, (lr.vmin, lr.vmax), style=style)
        specs.append(FigureSpec(
            figure_id=f"strain_map_{i:04d}", title=f"Strain map — {layer.name}",
            kind="map", filename=f"{layer.name}_strain", build=build,
        ))
    return specs
```

Replace the temporary `setdefault("strain", _empty)` reliance — the `@register` decorator overrides it at import. (The `_load_stage_catalogs` import in figures.py already imports `strain`.)

- [ ] **Step 4: Run — expect PASS; full suite green**

Run: `python3 -m pytest tests/test_figures_catalog.py -k strain_catalog -v && python3 -m pytest -q`

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/strain.py tests/test_figures_catalog.py
git commit -m "strain: figures() catalog — strain map per layer"
```

---

### Task 11: `mosaicity.figures()` — map

**Files:**
- Modify: `dfxm/stages/mosaicity.py`
- Test: `tests/test_figures_catalog.py`

Context: `MosaicityResult.stacked_path` holds an h5 with per-dataset-key groups (see `mosaicity.run` ~266–278 and `result.datasets` = `{key: shape}`). Each key is a `(Z,Y,X)` volume; render each layer as a map.

- [ ] **Step 1: Read** `mosaicity.run`'s h5 write (~262–278) to confirm the group/dataset path for each key and the per-key cmap/label/vmin/vmax (mirror what its own plotting used). Note the dataset path inside the h5 for a key.

- [ ] **Step 2: Write the failing test** (synthetic stacked h5 with one group/dataset; assert `figures()` yields map specs that build with equal aspect). Model it on Task 10's test using the real group/dataset path you found.

- [ ] **Step 3: Add `figures()`** registered with `@register("mosaicity")`, using `volume_layer_specs(...)` (Task 9) per dataset key — `h5_path=result.stacked_path`, `dataset=<the key's path in the h5>`, `sx/sy` from params (`pixel_size_x_um`/`pixel_size_y_um`), `vmin/vmax` per key (use the key's data range if the result doesn't carry them: compute with `np.nanmin/nanmax` on first read, or reuse the stage's colour-range helper). cmap/label per key as the stage uses today.

- [ ] **Step 4: Run** `pytest tests/test_figures_catalog.py -k mosaicity -v && python3 -m pytest -q` — expect PASS.

- [ ] **Step 5: Commit** `git commit -m "mosaicity: figures() catalog — maps per dataset/layer"`.

---

### Task 12: `visualize.figures()` — map

**Files:**
- Modify: `dfxm/stages/visualize.py`
- Test: `tests/test_figures_catalog.py`

Context: `VisualizeResult.datasets: list[DatasetProducts]` (name, shape, vmin, vmax). The volume comes from the **input** volume files (`strain_volume_file` / `mosa_volume_file` in params); visualize loads + aligns them. Reuse the stage's own volume-loading path so the layer arrays match what was rendered.

- [ ] **Step 1: Read** `visualize.run` to find the function that loads/aligns a volume from a file and the per-dataset cmap/label, plus the µm pixel scales used for `ext_x/ext_y`.

- [ ] **Step 2: Write the failing test** — synthetic: monkeypatch/feed a small volume file the loader accepts, build a `VisualizeResult` with one `DatasetProducts`, assert `figures()` yields per-layer map specs that build. (If the loader is heavy, test at the `volume_layer_specs` level with a written h5 and assert `visualize.figures()` returns the right count from `result.datasets[].shape[0]`.)

- [ ] **Step 3: Add `figures()`** registered `@register("visualize")`: for each `DatasetProducts`, produce per-layer map specs whose `build` loads that dataset's aligned volume via the stage's loader and renders layer `z` through `render.layer_figure(..., style=style)` with the dataset's cmap/label and vmin/vmax. Prefer reusing `volume_layer_specs` if you first materialise the aligned volume to a temp/known array; otherwise write an inline `build` closure mirroring it.

- [ ] **Step 4: Run** `-k visualize` + full suite — expect PASS.

- [ ] **Step 5: Commit** `git commit -m "visualize: figures() catalog — map per dataset/layer"`.

---

### Task 13: `rocking.figures()` — map

**Files:**
- Modify: `dfxm/stages/rocking.py`
- Test: `tests/test_figures_catalog.py`

Context: `RockingResult.aligned_path` is an h5 holding the aligned volume; `rocking.run` ~473–489 writes it (confirm the main dataset name and the `z_uniform_um` dataset). vmin/vmax live on `RockingProducts`. Pixel scales from params (`pixel_size`).

- [ ] **Step 1: Read** `rocking.run` ~470–490 to confirm the aligned-volume dataset name and `z_uniform_um`.

- [ ] **Step 2: Write the failing test** — synthetic aligned h5 (volume + z), a `RockingResult`, assert per-layer map specs build with equal aspect (model on Task 10).

- [ ] **Step 3: Add `figures()`** `@register("rocking")` using `volume_layer_specs(h5_path=result.aligned_path, dataset=<aligned name>, z_um=<read z_uniform_um>, sx/sy from params, vmin/vmax from the product, cmap/label as the stage uses)`.

- [ ] **Step 4: Run** `-k rocking` + full suite — expect PASS.

- [ ] **Step 5: Commit** `git commit -m "rocking: figures() catalog — aligned volume layers"`.

---

### Task 14: `slices.figures()` — map (cleanest; from oblique_slices.h5)

**Files:**
- Modify: `dfxm/stages/slices.py`
- Test: `tests/test_figures_catalog.py`

Context: `SlicesResult.output_h5` is `oblique_slices.h5`; its structure (`write_volume_group`, slices.py ~756–781): one group per `volume_id` with attrs (`cmap`, `cbar_label`, `vmin`, `vmax`, `title`, `dataset_path`, `kind`), each slice subgroup has datasets `slices` (n_planes, Hu, Wv), `u_um`, `v_um`, `offsets_um`. Rebuild each plane via `build_slice_figure` (Task 7).

- [ ] **Step 1: Write the failing test** — write a tiny `oblique_slices.h5` mirroring `write_volume_group`'s layout (one volume group, one slice subgroup with 2 planes), build a `SlicesResult(output_h5=...)`, assert `figures()` returns 2 map specs that build with equal aspect.

- [ ] **Step 2: Run — expect AttributeError**

Run: `python3 -m pytest tests/test_figures_catalog.py -k slices_catalog -v`

- [ ] **Step 3: Add `figures()`** `@register("slices")`:

```python
from dfxm.common.figures import FigureSpec, register


@register("slices")
def figures(result, params):
    import h5py

    if not result.output_h5:
        return []
    specs = []
    with h5py.File(result.output_h5, "r") as f:
        for vid in f.keys():
            vg = f[vid]
            prep = {
                "cmap_name": vg.attrs["cmap"], "title": vg.attrs["title"],
                "cbar_label": vg.attrs["cbar_label"],
                "vmin": float(vg.attrs["vmin"]), "vmax": float(vg.attrs["vmax"]),
            }
            for sname in vg.keys():
                n_planes = vg[sname]["slices"].shape[0]
                for k in range(n_planes):
                    def build(style, vid=vid, sname=sname, k=k, prep=dict(prep)):
                        with h5py.File(result.output_h5, "r") as g:
                            sg = g[vid][sname]
                            s2d = sg["slices"][k]
                            u = sg["u_um"][:]
                            v = sg["v_um"][:]
                            off = float(sg["offsets_um"][k])
                        return build_slice_figure(prep, {"name": sname}, s2d, u, v,
                                                  offset_um=off, style=style)
                    specs.append(FigureSpec(
                        figure_id=f"slice_{vid}_{sname}_{k:03d}",
                        title=f"{vid} / {sname} / plane {k}",
                        kind="map", filename=f"{vid}_{sname}_{k:03d}", build=build,
                    ))
    return specs
```

(Confirm `build_slice_figure`/`_make_norm` use exactly these `prep` keys; add any the norm needs, e.g. a center flag, from `vg.attrs`.)

- [ ] **Step 4: Run** `-k slices_catalog` + full suite — expect PASS.

- [ ] **Step 5: Commit** `git commit -m "slices: figures() catalog — one map per plane from oblique_slices.h5"`.

---

### Task 15: `matched.figures()` — map (recompute from raw)

**Files:**
- Modify: `dfxm/stages/matched.py`
- Test: `tests/test_figures_catalog.py`

Context: matched saves only PNGs; the data (`shifted` frame) is computed in `run()` from raw (`load_pco_ff_frame` + `_apply_shift_single`) and rendered via `render.layer_figure`. Rebuild requires the raw frames + match/shift info.

- [ ] **Step 1: Read** `matched.run` to identify the smallest reusable unit that, given the result/params, recomputes a single layer's `shifted` frame + its `ext_x/ext_y/vmin/vmax/title`. If `run()` doesn't retain matches/shifts on the result, **extend `MatchedResult`** to record, per saved layer: the rocking file used, `frame_index`, the shift, the canvas size, and `vmin/vmax` (a `list[MatchedLayer]`). Persist enough to recompute without re-deriving matches.

- [ ] **Step 2: Write the failing test** — synthetic: monkeypatch `matched.load_pco_ff_frame` to return a known array; build a `MatchedResult` carrying one layer's recompute info; assert `matched.figures()` yields one map spec that builds with equal aspect.

- [ ] **Step 3: Add `figures()`** `@register("matched")`: for each recorded layer, a `map` spec whose `build` re-reads the frame, re-applies the shift, and calls `render.layer_figure(shifted, vmin, vmax, colormap, ext_x, ext_y, title, "Intensity − background (a.u.)", style=style)`. If raw is unavailable at build time, raise a clear `FileNotFoundError` (the GUI surfaces it).

- [ ] **Step 4: Run** `-k matched` + full suite — expect PASS.

- [ ] **Step 5: Commit** `git commit -m "matched: figures() catalog — recompute frame from raw"`.

---

### Task 16: `ExportDialog` — live preview + controls + savefig

**Files:**
- Create: `gui/widgets/export_dialog.py`
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add a failing smoke assertion** (after the last existing step in `tests/gui_smoke.py`, before `GUI SMOKE PASSED`):

```python
    # Export dialog: build a figure spec, render a preview, export 3 formats.
    import tempfile, os as _os
    from dfxm.common import figures as _figs
    from dfxm.common.plotting import PUBLICATION_STYLE
    from gui.widgets.export_dialog import ExportDialog

    sres = win._views["strain"]  # any view; we drive the dialog directly with a spec
    # Build a synthetic one-figure catalog so the smoke doesn't need a real run:
    import numpy as _np
    from matplotlib.figure import Figure as _Fig

    def _mk(style):
        f = _Fig(); ax = f.add_subplot(111)
        ax.imshow(_np.zeros((8, 8)), extent=[0, 8, 0, 8], origin="lower")
        return f

    spec = _figs.FigureSpec("t", "Test", "map", "test_fig", _mk)
    dlg = ExportDialog([spec], 0, PUBLICATION_STYLE)
    dlg.show(); app.processEvents()
    assert dlg._canvas.figure is not None
    out = tempfile.mkdtemp()
    dlg._style.formats = ("png", "pdf", "svg")
    paths = dlg.export_to(out)
    app.processEvents()
    assert all(_os.path.exists(p) and _os.path.getsize(p) > 0 for p in paths)
    print("[N] export dialog renders preview + writes png/pdf/svg")
```

(Renumber `[N]` to the next free step number.)

- [ ] **Step 2: Run — expect ImportError (`export_dialog` missing)**

Run: `python3 tests/gui_smoke.py`

- [ ] **Step 3: Create `gui/widgets/export_dialog.py`**

```python
"""Per-figure publication export dialog: live preview + style controls."""

from __future__ import annotations

import os
from dataclasses import replace

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFormLayout, QHBoxLayout, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

from dfxm.common.figures import FigureSpec
from dfxm.common.plotting import PlotStyle
from .mpl_canvas import MplCanvas


class ExportDialog(QDialog):
    def __init__(self, specs: list[FigureSpec], index: int, global_style: PlotStyle,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Export figure")
        self.resize(900, 620)
        self._specs = specs
        self._index = index
        self._style = replace(global_style)  # working copy
        self._global = global_style

        self._canvas = MplCanvas()
        self._selector = QComboBox()
        self._selector.addItems([s.title for s in specs])
        self._selector.setCurrentIndex(index)
        self._selector.currentIndexChanged.connect(self._on_select)

        self._controls = self._build_controls()  # binds widgets -> self._style
        self._debounce = QTimer(self); self._debounce.setSingleShot(True)
        self._debounce.setInterval(150); self._debounce.timeout.connect(self._render)

        export_btn = QPushButton("Export"); export_btn.clicked.connect(self._on_export)
        reset_btn = QPushButton("Reset to global style"); reset_btn.clicked.connect(self._on_reset)
        btns = QHBoxLayout(); btns.addStretch(1); btns.addWidget(reset_btn); btns.addWidget(export_btn)

        right = QVBoxLayout(); right.addWidget(self._selector); right.addLayout(self._controls)
        right.addStretch(1); right.addLayout(btns)
        root = QHBoxLayout(self); root.addWidget(self._canvas, 2)
        rw = QWidget(); rw.setLayout(right); root.addWidget(rw, 1)
        self._render()

    def _spec(self) -> FigureSpec:
        return self._specs[self._index]

    def _on_select(self, i): self._index = i; self._render()
    def _on_reset(self): self._style = replace(self._global); self._render()
    def _schedule(self): self._debounce.start()

    def _render(self) -> None:
        spec = self._spec()
        style = self._style if spec.kind == "map" else replace(self._style, scale_bar=False)
        fig = spec.build(style)
        self._canvas.figure.clear()
        # swap the canvas figure to the freshly built one:
        self._canvas.figure = fig
        self._canvas.canvas.figure = fig
        fig.set_canvas(self._canvas.canvas)
        self._canvas.canvas.draw_idle()

    def export_to(self, out_dir: str) -> list[str]:
        os.makedirs(out_dir, exist_ok=True)
        spec = self._spec()
        style = self._style if spec.kind == "map" else replace(self._style, scale_bar=False)
        fig = spec.build(style)
        written = []
        for fmt in self._style.formats:
            path = os.path.join(out_dir, f"{spec.filename}.{fmt}")
            fig.savefig(path, dpi=self._style.dpi, bbox_inches="tight",
                        facecolor="white")
            written.append(path)
        return written

    def _on_export(self) -> None:
        from PySide6.QtWidgets import QFileDialog
        d = QFileDialog.getExistingDirectory(self, "Export to folder")
        if d:
            self.export_to(d)

    def _build_controls(self) -> QFormLayout:
        form = QFormLayout()
        # font scale
        fs = QDoubleSpinBox(); fs.setRange(1.0, 3.0); fs.setSingleStep(0.1)
        fs.setValue(self._style.font_scale)
        fs.valueChanged.connect(lambda v: (setattr(self._style, "font_scale", v), self._schedule()))
        form.addRow("Text size", fs)
        # colourbar ticks
        ticks = QSpinBox(); ticks.setRange(2, 11); ticks.setValue(self._style.colorbar_ticks or 5)
        ticks.valueChanged.connect(
            lambda v: (setattr(self._style, "colorbar_ticks", v), self._schedule()))
        form.addRow("Colourbar ticks", ticks)
        # ... the executor adds the remaining controls (scale bar show/length/thickness/label
        # size/pos/colour + background box show/colour/opacity/margin; colourbar label/thickness/
        # format; figure width; output formats/dpi) following the same bind-and-_schedule pattern.
        return form
```

(The control panel is mechanical: every widget mutates a `self._style` field and calls `self._schedule()`. The smoke only needs the dialog to build/preview/export; the full control set is added here following the shown pattern and the locked spec list.)

- [ ] **Step 4: Run the smoke — expect it to print the new step + `GUI SMOKE PASSED`; run full suite**

Run: `python3 tests/gui_smoke.py && python3 -m pytest -q && ruff check .`

- [ ] **Step 5: Commit**

```bash
git add gui/widgets/export_dialog.py tests/gui_smoke.py
git commit -m "gui: ExportDialog with live preview + PNG/PDF/SVG export"
```

---

### Task 17: wire "Export…" into the Output tab

**Files:**
- Modify: `gui/stage_view.py`
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Read** `gui/stage_view.py` to find the Output tab construction and how the result/params are held on the view (e.g. `self._last_params`, the result from `_finish_ok`). Store the last result on the view if not already (`self._last_result = result` in `_finish_ok`).

- [ ] **Step 2: Add a failing smoke assertion** — after a strain run in the smoke, assert `sview._export_btn` exists and that clicking it (when there are figures) constructs an `ExportDialog`. Simpler: assert `sview._figures()` returns a list and an `Export…` button widget exists.

- [ ] **Step 3: Implement.** Add an "Export…" `QPushButton` to the Output tab; on click, call `figures_for(self._stage_name, self._last_result, self._last_params)`; if empty, show an info message ("This stage produced no exportable figures"); else open `ExportDialog(specs, 0, <global style>)`. Disable the button until a successful run has populated `self._last_result`.

- [ ] **Step 4: Run** the smoke + full suite + ruff — expect PASS.

- [ ] **Step 5: Commit** `git commit -m "gui: Export… button on the Output tab"`.

---

# PHASE 3 — plot figures + global style + export all

### Task 18: strain plot figures (histogram + detrend diagnostic)

**Files:**
- Modify: `dfxm/stages/strain.py`
- Test: `tests/test_figures_catalog.py`

- [ ] **Step 1: Write the failing test** — extend the strain catalog test to assert `kind=="plot"` specs exist: one histogram per layer (builds from the stacked volume layer) and one detrend diagnostic per layer.

- [ ] **Step 2: Run — expect failure (only map specs today)**

- [ ] **Step 3: Extend `strain.figures()`.** Append per layer:
  - **histogram:** `FigureSpec(kind="plot", build=lambda style,i=i: build_strain_histogram(load_map(result.stacked_path,"strain")[i], style=style))`.
  - **detrend diagnostic:** recompute from the source `maps.h5`. Derive the layer's maps path from params (`root_folder`/`input_folder` + the layer `name` folder + `maps_filename`) exactly as `run()` does; in `build`, `ccmth = load_map(maps_path, params["ccmth_com_path"]); detrended, surface = detrend_arctan_2d(ccmth); ... apply_roi(...); return build_detrend_diag(orig, detrended, surface, style=style)`. If the maps path is missing, raise `FileNotFoundError("source maps.h5 not found; detrend diagnostic needs the original input")`.

- [ ] **Step 4: Run** `-k strain_catalog` + full suite — expect PASS.

- [ ] **Step 5: Commit** `git commit -m "strain: catalog adds histogram + detrend-diagnostic plot figures"`.

---

### Task 19: mosaicity plot figures (histogram)

**Files:**
- Modify: `dfxm/stages/mosaicity.py`
- Test: `tests/test_figures_catalog.py`

- [ ] **Step 1: Write the failing test** — assert mosaicity catalog includes `kind=="plot"` histogram specs per dataset/layer.
- [ ] **Step 2: Run — expect failure.**
- [ ] **Step 3: Extend `mosaicity.figures()`** to append a histogram `FigureSpec` per dataset/layer using `strain.build_strain_histogram` (it is generic over the data array — import it, or move it to `dfxm/common/plotting.py` as `build_histogram` if cleaner; if you move it, update strain to import it and keep its tests green).
- [ ] **Step 4: Run** `-k mosaicity` + full suite — expect PASS.
- [ ] **Step 5: Commit** `git commit -m "mosaicity: catalog adds histogram plot figures"`.

---

### Task 20: `profiles.figures()` — plot

**Files:**
- Modify: `dfxm/stages/profiles.py`
- Test: `tests/test_figures_catalog.py`

Context: profiles produces line-profile figures (image panel + trace panels) from `oblique_slices.h5` + the job specs (see `profiles.run` and its figure builders ~430–478). These are `kind="plot"` (no µm scale bar).

- [ ] **Step 1: Read** the profiles figure builder(s) and how a job maps to data (which datasets/coords it reads from `oblique_slices.h5`).
- [ ] **Step 2: Refactor** the profile figure builder to return a `Figure` and accept `style=None` (route fonts + colourbar through the primitives; legacy when `None`), with a thin save wrapper used by `run()` (same pattern as Tasks 6–7).
- [ ] **Step 3: Write the failing test** + add `@register("profiles") def figures(result, params)` returning one `kind="plot"` spec per job that rebuilds from the consolidated h5 + the job spec.
- [ ] **Step 4: Run** `-k profiles` + full suite — expect PASS.
- [ ] **Step 5: Commit** `git commit -m "profiles: catalog + style-aware line-profile builder"`.

---

### Task 21: global publication-style editor + session global style

**Files:**
- Modify: `gui/main_window.py`, `gui/stage_view.py`, `gui/widgets/export_dialog.py`
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add a failing smoke assertion** — assert `win.global_plot_style()` returns a `PlotStyle` and that a `StyleEditor` (or the reused control panel) can mutate it.
- [ ] **Step 2: Run — expect failure.**
- [ ] **Step 3: Implement.** Hold one session `PlotStyle` on `MainWindow` (seeded from `PUBLICATION_STYLE`), exposed via `global_plot_style()`. Add a "Publication style…" button (Output tab or window) that opens an editor reusing the export dialog's control panel (factor the controls into a `StyleControls(QWidget)` so both the dialog and the global editor share it — extract during this task). `ExportDialog` is constructed with `win.global_plot_style()` as its starting global.
- [ ] **Step 4: Run** smoke + suite + ruff — expect PASS.
- [ ] **Step 5: Commit** `git commit -m "gui: global publication style + shared StyleControls editor"`.

---

### Task 22: "Export all…" (batch a stage's catalog)

**Files:**
- Modify: `gui/stage_view.py`
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add a failing smoke assertion** — after a strain run, call `sview.export_all(tmpdir)` and assert files are written for every map spec and that a returned summary lists per-figure success/failure.
- [ ] **Step 2: Run — expect AttributeError.**
- [ ] **Step 3: Implement `export_all(out_dir)`** on the view: loop `figures_for(...)`, `spec.build(global_style)` (scale bar forced off for `plot` kinds), `savefig` each format; collect `(figure_id, ok, error)` and return a summary; skip-and-report on per-figure failure (never abort the batch). Wire an "Export all…" button that picks a folder, runs it, and shows the summary in the banner/results.
- [ ] **Step 4: Run** smoke + suite + ruff — expect PASS.
- [ ] **Step 5: Commit** `git commit -m "gui: Export all… batches a stage's figure catalog"`.

---

# PHASE 4 — documentation

### Task 23: docs sweep (Usage.md + Codebase.md)

**Files:**
- Modify: `docs/Usage.md`, `docs/Codebase.md`

- [ ] **Step 1: `docs/Usage.md`** — add a "Publication export" subsection under the stage-panel docs: how to open **Export…** / **Export all…**, what each control does (scale bar incl. background box, text size, colourbar label/ticks/format, figure width, formats), that exported files land in an `exports/` subfolder, and that physical aspect is always preserved.

- [ ] **Step 2: `docs/Codebase.md`** — document `PlotStyle` + `PUBLICATION_STYLE` and the primitives (`draw_scale_bar`, `apply_text_scale`, `add_colorbar`) under `dfxm/common/plotting.py`; the new `dfxm/common/figures.py` (`FigureSpec`, `figures_for`, `volume_layer_specs`); note `render.layer_figure`/strain/slices/profiles builders now take a `style`; add `gui/widgets/export_dialog.py` (`ExportDialog`, `StyleControls`) and the Output-tab Export entry points.

- [ ] **Step 3: Verify everything**

Run: `python3 -m pytest -q && ruff check . && ruff format --check . && python3 tests/gui_smoke.py`
Expected: all green; smoke through the new export step + `GUI SMOKE PASSED`.

- [ ] **Step 4: Commit**

```bash
git add docs/Usage.md docs/Codebase.md
git commit -m "docs: publication plot export (PlotStyle, figure catalog, export dialog)"
```

---

## Final verification (after Task 23)

- `python3 -m pytest -q` — all green (new `test_plot_style.py`, `test_figures_catalog.py`, extended stage tests).
- `ruff check . && ruff format --check .` — clean.
- `python3 tests/gui_smoke.py` — through the export step + `GUI SMOKE PASSED`.
- `dfxm/` import-clean with PySide6/pyvista/vtk poisoned (Qt-free invariant): the catalog + builders must import and run headless.
- Manual (with a display): run a stage, **Export…**, dial up the text/scale bar/box, confirm the live preview updates and PNG/PDF/SVG are written; **Export all…** on a volume stage.
```
