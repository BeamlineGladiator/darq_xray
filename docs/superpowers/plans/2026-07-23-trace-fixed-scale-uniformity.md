# Trace Fixed-Scale Fidelity + Uniform Trace Figures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the profiles trace/companion figures hold their µm-per-cm scale exactly (today they drift silently — real files measured 5.7 µm/cm where 10 was set) and make every trace figure of a run physically uniform (same box height in cm, same fonts, shared margins) so they compose into a final figure with zero rescaling.

**Architecture:** A new deterministic placement engine in `dfxm/common/plotting.py` (measure decorations once → set figure size and axes position exactly; no iteration, no `set_box_aspect`) replaces `fit_axes_to_box` for the trace/companion path only. Traces get a fixed box height (`PlotStyle.trace_height_cm`, width = L/scale); the run/replot drivers defer trace saving so all traces of an invocation share max-margins; the companion is rebuilt on the same engine (map panel at map scale, trace panels styled identically to standalone traces). Every fixed-scale figure self-checks its box after render and reports drift into `ProfilesResult.notes`.

**Tech Stack:** Python 3.10, matplotlib (explicit `Figure` API — never pyplot), h5py fixtures, pytest, PySide6 only under `gui/`.

**Spec:** `docs/superpowers/specs/2026-07-23-trace-fixed-scale-uniformity-design.md`

## Global Constraints

- `dfxm/` stays Qt-free; figures via `matplotlib.figure.Figure` only, never pyplot.
- Legacy paths byte-stable: `style=None` and styled-without-fixed-scale trace/companion rendering must not change (existing pinned tests must keep passing untouched unless a task explicitly updates them).
- Drift guard tolerance: **0.5 %** relative, per side. Default trace box height: **3.0 cm**. Max side clamp stays **30 in** (`_MAX_FIXED_SIDE_IN`).
- Guards report via `ProfilesResult.notes` + `logging` WARNING — never exceptions (`StageUserError` only for user-fixable input).
- Docs contract: any task that changes stage/GUI behaviour updates `docs/Usage.md` and/or `docs/Codebase.md` **in the same commit**.
- Ruff: line length 100, double quotes; `ruff format` runs via hook on Write/Edit.
- Map builders (`render.layer_figure`, strain/slices, `render_single`) keep `fit_axes_to_box` — out of scope except the WARNING upgrade and the overview drift note.
- Work on a feature branch `trace-fixed-scale` off master.

---

### Task 1: Deterministic placement primitives in `dfxm/common/plotting.py`

**Files:**
- Modify: `dfxm/common/plotting.py` (after `fit_axes_to_box`, ~line 362)
- Test: `tests/test_axes_placement.py` (new)

**Interfaces:**
- Produces (used by Tasks 2–5):
  - `@dataclass(frozen=True) AxesMargins(left, right, top, bottom)` with `max_with(other) -> AxesMargins`
  - `measure_axes_margins(fig, ax, extras=(), pad_in=0.02) -> AxesMargins`
  - `apply_axes_margins(fig, ax, w_in, h_in, margins) -> None`
  - `place_axes_box(fig, ax, w_in, h_in, margins=None, pad_in=0.02) -> AxesMargins`
  - `trace_height_cm(style) -> float` (defensive read, default 3.0)
  - `trace_fixed_box(style, length_um) -> (w_in, h_in, eff_scale) | None`
  - `measured_box_in(fig, ax) -> (w_in, h_in)`
  - `box_drift_note(label, fig, ax, w_in, h_in, rel_tol=0.005) -> str | None`
- `fit_axes_to_box` non-convergence log: `_log.info` → `_log.warning` (message text unchanged).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_axes_placement.py`:

```python
"""Deterministic axes placement (trace fixed-scale engine) — dfxm.common.plotting."""

import numpy as np
from matplotlib.figure import Figure

from dfxm.common.plotting import (
    AxesMargins,
    PlotStyle,
    box_drift_note,
    measure_axes_margins,
    measured_box_in,
    place_axes_box,
    trace_fixed_box,
    trace_height_cm,
)


def _plot_fig(ylabel="value", title="a title", font=10.0):
    fig = Figure(figsize=(6, 4), facecolor="white")
    ax = fig.add_subplot(111)
    x = np.linspace(0, 29.668647, 200)
    ax.plot(x, np.sin(x / 5.0) * 1e-4)
    ax.set_xlim(0, x[-1])
    ax.set_xlabel("distance along line (µm)", fontsize=font * 1.2)
    ax.set_ylabel(ylabel, fontsize=font)
    ax.set_title(title, loc="left", fontsize=font)
    ax.tick_params(labelsize=font)
    return fig, ax


def test_place_axes_box_exact_small_box_large_fonts():
    # the exact configuration that defeated fit_axes_to_box (L=29.67, 10 um/cm)
    for font in (10.0, 14.0, 20.0):
        fig, ax = _plot_fig(ylabel="COM mu (deg)", font=font)
        place_axes_box(fig, ax, 29.668647 / 10.0 / 2.54, 3.0 / 2.54)
        w, h = measured_box_in(fig, ax)
        assert abs(w - 29.668647 / 10.0 / 2.54) < 0.005 * w, (font, w)
        assert abs(h - 3.0 / 2.54) < 0.005 * h, (font, h)


def test_place_axes_box_with_shared_margins_keeps_box_and_margins():
    fig1, ax1 = _plot_fig(ylabel="short")
    fig2, ax2 = _plot_fig(ylabel="a much longer y-axis label (units)")
    m1 = place_axes_box(fig1, ax1, 2.0, 1.2)
    m2 = place_axes_box(fig2, ax2, 2.0, 1.2)
    shared = m1.max_with(m2)
    place_axes_box(fig1, ax1, 2.0, 1.2, margins=shared)
    place_axes_box(fig2, ax2, 2.0, 1.2, margins=shared)
    # identical canvas sizes and identical box positions
    assert np.allclose(fig1.get_size_inches(), fig2.get_size_inches())
    assert np.allclose(list(ax1.get_position().bounds), list(ax2.get_position().bounds))
    for fig, ax in ((fig1, ax1), (fig2, ax2)):
        w, h = measured_box_in(fig, ax)
        assert abs(w - 2.0) < 0.01 and abs(h - 1.2) < 0.01


def test_measure_axes_margins_covers_decorations():
    fig, ax = _plot_fig()
    place_axes_box(fig, ax, 2.5, 1.5)
    m = measure_axes_margins(fig, ax)
    assert m.left > 0.2 and m.bottom > 0.2 and m.top > 0.05  # labels/ticks/title exist
    fw, fh = fig.get_size_inches()
    assert fw >= m.left + 2.5 and fh >= m.bottom + 1.5  # canvas holds box+margins


def test_axes_margins_max_with():
    a = AxesMargins(1.0, 0.1, 0.2, 0.5)
    b = AxesMargins(0.5, 0.4, 0.1, 0.9)
    m = a.max_with(b)
    assert (m.left, m.right, m.top, m.bottom) == (1.0, 0.4, 0.2, 0.9)


def test_trace_height_cm_defensive():
    assert trace_height_cm(PlotStyle()) == 3.0  # default when unset
    assert trace_height_cm(PlotStyle(trace_height_cm=4.5)) == 4.5
    assert trace_height_cm(PlotStyle(trace_height_cm=-1)) == 3.0
    assert trace_height_cm(PlotStyle(trace_height_cm="junk")) == 3.0
    assert trace_height_cm(None) == 3.0


def test_trace_fixed_box_geometry_and_clamp():
    st = PlotStyle(trace_scale_um_per_cm=10.0, trace_height_cm=3.0)
    box = trace_fixed_box(st, 44.941256)
    assert box is not None
    w, h, s = box
    assert abs(w - 44.941256 / 10.0 / 2.54) < 1e-9
    assert abs(h - 3.0 / 2.54) < 1e-9
    assert s == 10.0
    assert trace_fixed_box(PlotStyle(), 40.0) is None  # knob off
    assert trace_fixed_box(st, 0.0) is None  # degenerate line
    w, h, s = trace_fixed_box(PlotStyle(trace_scale_um_per_cm=0.1, trace_height_cm=3.0), 500.0)
    assert w == 30.0 and s > 0.1  # width clamped, effective scale raised


def test_box_drift_note_fires_only_on_miss():
    fig, ax = _plot_fig()
    place_axes_box(fig, ax, 2.0, 1.2)
    assert box_drift_note("t", fig, ax, 2.0, 1.2) is None
    note = box_drift_note("t", fig, ax, 3.0, 1.2)  # deliberately wrong target
    assert note is not None and "t" in note and "cm" in note
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_axes_placement.py -q`
Expected: FAIL — `ImportError: cannot import name 'AxesMargins'`

- [ ] **Step 3: Implement the primitives**

In `dfxm/common/plotting.py`, add `from dataclasses import dataclass, field` if not present (check the existing imports — `dataclass` is already imported for `PlotStyle`).

First, add the new `PlotStyle` field (after `trace_scale_um_per_cm`, line ~119 — the GUI knob for it comes in Task 5, but the core reads it from Task 1 on):

```python
    # fixed box HEIGHT for the profiles trace figures in fixed-scale mode, in cm
    # of page. None/blank -> 3.0. Ignored when no trace/map scale is set.
    trace_height_cm: float | None = None
```

Then insert after `fit_axes_to_box` (and change that function's final `_log.info(` to `_log.warning(` — same message):

```python
@dataclass(frozen=True)
class AxesMargins:
    """Decoration extents around an axes box, in inches."""

    left: float
    right: float
    top: float
    bottom: float

    def max_with(self, other: "AxesMargins") -> "AxesMargins":
        return AxesMargins(
            max(self.left, other.left),
            max(self.right, other.right),
            max(self.top, other.top),
            max(self.bottom, other.bottom),
        )


def _ensure_agg(fig):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if fig.canvas is None or not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)


def measure_axes_margins(fig, ax, extras=(), pad_in: float = 0.02) -> AxesMargins:
    """Measure *ax*'s decoration margins (labels/ticks/title/offset text) in inches.

    Draws once; *extras* are additional axes (e.g. a manually placed colorbar)
    whose extents count toward this axes' decoration envelope. ``pad_in`` is a
    small breathing margin added on every side.
    """
    _ensure_agg(fig)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = ax.get_tightbbox(r)
    for ex in extras:
        if ex is not None:
            tb = tb.union([tb, ex.get_tightbbox(r)])
    bb = ax.get_window_extent(r)
    d = fig.dpi
    return AxesMargins(
        left=max(0.0, (bb.x0 - tb.x0) / d) + pad_in,
        right=max(0.0, (tb.x1 - bb.x1) / d) + pad_in,
        top=max(0.0, (tb.y1 - bb.y1) / d) + pad_in,
        bottom=max(0.0, (bb.y0 - tb.y0) / d) + pad_in,
    )


def apply_axes_margins(fig, ax, w_in: float, h_in: float, m: AxesMargins) -> None:
    """Size *fig* to exactly margins+box and pin *ax* there. No iteration."""
    fig.set_layout_engine("none")
    fw, fh = m.left + w_in + m.right, m.bottom + h_in + m.top
    fig.set_size_inches(fw, fh, forward=False)
    ax.set_position([m.left / fw, m.bottom / fh, w_in / fw, h_in / fh])


def place_axes_box(fig, ax, w_in, h_in, margins: AxesMargins | None = None, pad_in=0.02):
    """Deterministically give *ax* an exactly (w_in, h_in)-inch box.

    With ``margins=None``: place provisionally at the final box size (so tick
    density is measured at the real geometry), measure the decorations, then
    apply. With explicit *margins* (e.g. the max over a figure set): apply
    directly. Returns the margins used. Exact by construction — replaces
    ``fit_axes_to_box`` + ``set_box_aspect`` for the trace path, whose coupling
    made the iterative fit stall and silently keep a wrong physical scale.
    """
    fig.set_layout_engine("none")
    if margins is None:
        apply_axes_margins(fig, ax, w_in, h_in, AxesMargins(1.2, 0.6, 0.8, 0.9))
        margins = measure_axes_margins(fig, ax, pad_in=pad_in)
    apply_axes_margins(fig, ax, w_in, h_in, margins)
    return margins


_TRACE_HEIGHT_CM_DEFAULT = 3.0


def trace_height_cm(style: "PlotStyle | None") -> float:
    """Defensive read of ``style.trace_height_cm``: positive finite float, else 3.0."""
    v = getattr(style, "trace_height_cm", None)
    if v is None or v == "":
        return _TRACE_HEIGHT_CM_DEFAULT
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _TRACE_HEIGHT_CM_DEFAULT
    return v if (v > 0 and math.isfinite(v)) else _TRACE_HEIGHT_CM_DEFAULT


def trace_fixed_box(style: "PlotStyle | None", length_um: float):
    """Target trace box (w_in, h_in, effective_um_per_cm), or None when off.

    Width = line length / trace-effective scale; height = the fixed
    ``trace_height_cm`` (trace_aspect does NOT apply in fixed-scale mode).
    Width clamps to 30 in, raising the effective scale like the map clamp.
    """
    s = trace_fixed_scale(style)
    if s is None:
        return None
    if not math.isfinite(length_um) or length_um <= 0:
        return None
    w = length_um / s / 2.54
    h = min(trace_height_cm(style) / 2.54, _MAX_FIXED_SIDE_IN)
    if w > _MAX_FIXED_SIDE_IN:
        s = s * (w / _MAX_FIXED_SIDE_IN)
        w = _MAX_FIXED_SIDE_IN
        _log.warning(
            "trace fixed-scale box clamped to %.0f in wide; effective scale raised to %.4g um/cm",
            _MAX_FIXED_SIDE_IN,
            s,
        )
    return (w, h, s)


def measured_box_in(fig, ax) -> tuple[float, float]:
    """The axes box as actually rendered, in inches (draws once)."""
    _ensure_agg(fig)
    fig.canvas.draw()
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    return (bb.width / fig.dpi, bb.height / fig.dpi)


def box_drift_note(label: str, fig, ax, w_in, h_in, rel_tol: float = 0.005) -> str | None:
    """None when the rendered box is within *rel_tol* of target; else a user note.

    The no-silent-drift guard: callers append the note to the stage result
    notes (GUI Results tab) and we log a WARNING. Never raises.
    """
    w, h = measured_box_in(fig, ax)
    if abs(w - w_in) <= rel_tol * w_in and abs(h - h_in) <= rel_tol * h_in:
        return None
    msg = (
        f"{label}: plot box rendered {w * 2.54:.2f}x{h * 2.54:.2f} cm, "
        f"expected {w_in * 2.54:.2f}x{h_in * 2.54:.2f} cm — physical scale is off"
    )
    _log.warning(msg)
    return msg
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_axes_placement.py tests/test_figure_layout.py -q`
Expected: all PASS (the `fit_axes_to_box` tests in test_figure_layout.py must survive the INFO→WARNING change; `test_fit_axes_to_box_nonconvergence_is_nonfatal` asserts non-fatality, not the log level — if it captures the log, adjust its `caplog` level to WARNING).

- [ ] **Step 5: Commit**

```bash
git checkout -b trace-fixed-scale
git add dfxm/common/plotting.py tests/test_axes_placement.py tests/test_figure_layout.py
git commit -m "feat(plotting): deterministic axes placement primitives + trace box helpers"
```

---

### Task 2: `build_trace_figure` fixed-scale mode on the new engine

**Files:**
- Modify: `dfxm/stages/profiles.py:781-866` (`build_trace_figure`)
- Modify: `tests/test_stage_profiles.py:358-450` (fixed-scale trace tests)
- Modify: `docs/Usage.md` (profiles → trace figures paragraph), `docs/Codebase.md` (profiles entry)

**Interfaces:**
- Consumes: `trace_fixed_box`, `place_axes_box`, `measured_box_in` (Task 1).
- Produces: `build_trace_figure(...)` — same signature, but in fixed-scale mode the returned figure has an exactly `(L/scale) × trace_height_cm` box, no `set_box_aspect`, no constrained layout. New extracted helper `_draw_trace_axes(ax, fld, geom, *, linewidth, color, font_scale, style, show_xlabel=True)` (module-level, reused by Task 4's companion).

- [ ] **Step 1: Update/write the failing tests**

In `tests/test_stage_profiles.py` replace `test_build_trace_figure_fixed_scale_pins_box_width` (line ~368) and the clamp test (~408), keep `test_build_trace_figure_fixed_scale_ignores_width_in` as-is, and add the real-data regression. `_trace_box_inches` (line 358) already measures the drawn box — reuse it:

```python
def test_build_trace_figure_fixed_scale_box_is_length_by_height():
    # fixed-scale mode: box width = L/scale, box height = trace_height_cm —
    # trace_aspect no longer shapes the box (it only governs the legacy mode).
    fld, geom = _fake_field(std=True)  # L = 10 um
    st = PlotStyle(trace_scale_um_per_cm=2.0, trace_height_cm=3.0)
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(2.0, 1.0), width_in=6.0, linewidth=2.0, color="",
        font_scale=1.0, style=st,
    )
    w_in, h_in = _trace_box_inches(fig)
    assert abs(w_in - 10.0 / 2.0 / 2.54) < 0.005 * w_in
    assert abs(h_in - 3.0 / 2.54) < 0.005 * h_in
    assert fig.axes[0].get_box_aspect() is None  # no aspect pin in fixed mode


def test_build_trace_figure_fixed_scale_exact_for_short_line_real_repro():
    # regression: L=29.668647 at 10 um/cm rendered at ~5.7 um/cm on real data
    # (set_box_aspect defeated fit_axes_to_box, which silently kept the miss).
    n = 200
    dist = np.linspace(0.0, 29.668647, n)
    fld = {
        "vid": "mosa_com_mu",
        "attrs": {
            "cbar_label": "COM mu (deg)",
            "kind": "mosa_com",
            "source_volume": "aligned_raw_mosa_volumes.h5",
            "title": "t",
            "cmap": "viridis",
        },
        "value_mean": np.sin(dist / 5.0) * 1e-4,
        "value_std": None,
    }
    geom = {"distance": dist, "L": 29.668647}
    for fs in (1.0, 1.4, 2.0):
        st = PlotStyle(trace_scale_um_per_cm=10.0, trace_height_cm=3.0)
        fig = PR.build_trace_figure(
            fld, geom, aspect_wh=(4.0, 3.0), width_in=2.0, linewidth=2.0, color="",
            font_scale=fs, style=st,
        )
        w_in, _ = _trace_box_inches(fig)
        implied = 29.668647 / (w_in * 2.54)
        assert abs(implied - 10.0) < 0.05, (fs, implied)


def test_build_trace_figure_fixed_scale_clamps_width_only():
    fld, geom = _fake_field(std=True)
    geom = {**geom, "L": 10.0}
    st = PlotStyle(trace_scale_um_per_cm=0.01, trace_height_cm=3.0)  # 10um/0.01 -> 393 in
    fig = PR.build_trace_figure(
        fld, geom, aspect_wh=(4.0, 3.0), width_in=6.0, linewidth=2.0, color="",
        font_scale=1.0, style=st,
    )
    w_in, h_in = _trace_box_inches(fig)
    assert w_in <= 30.0 + 0.2  # clamped width
    assert abs(h_in - 3.0 / 2.54) < 0.02  # height keeps trace_height_cm
```

Keep `test_build_trace_figure_pins_plot_box_aspect` (legacy mode, no fixed scale) untouched — it must still pass.

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python3 -m pytest tests/test_stage_profiles.py -q -k "fixed_scale or box_aspect"`
Expected: the three new/replaced tests FAIL (box height still comes from aspect); `pins_plot_box_aspect` and `ignores_width_in` PASS.

- [ ] **Step 3: Rework `build_trace_figure`**

Extract the axes-content block (plot/band/ylabel/title/grid/xlim/xlabel/ticks/offset-text, currently lines 838–863) into a module-level helper so Task 4 can reuse it verbatim:

```python
def _draw_trace_axes(ax, fld, geom, *, linewidth, color, font_scale, style, show_xlabel=True):
    """Draw one field's line profile into *ax* — the single source of the trace
    look, shared by the standalone trace figures and the companion panels."""
    fs = float(font_scale)
    curve_color = color or "C0"
    distance = geom["distance"]
    vm = fld["value_mean"]
    ax.plot(distance, vm, "-", lw=float(linewidth), color=curve_color, zorder=3)
    if fld["value_std"] is not None:
        vs = fld["value_std"]
        ax.fill_between(distance, vm - vs, vm + vs, color=curve_color, alpha=0.22, lw=0, zorder=2)
    ax.set_ylabel(fld["attrs"]["cbar_label"], fontsize=10 * fs)
    src = os.path.basename(fld["attrs"]["source_volume"]) or "(consolidated)"
    if style is None or style.show_title:
        title_fs = 10 * fs * (style.title_scale if style is not None else 1.0)
        ax.set_title(
            f"{fld['attrs']['kind']}  |  {fld['vid']}  |  {src}", fontsize=title_fs, loc="left"
        )
    ax.grid(True, color="0.85", lw=0.6)
    ax.set_xlim(0.0, geom["L"])
    if show_xlabel:
        ax.set_xlabel("distance along line (µm)", fontsize=12 * fs)
    ax.tick_params(axis="both", labelsize=10 * fs)
    ax.yaxis.get_offset_text().set_fontsize(10 * fs)
    ax.xaxis.get_offset_text().set_fontsize(10 * fs)
```

Then `build_trace_figure` becomes (docstring updated to describe the two modes; fixed mode = exact `(L/scale) × trace_height_cm` box, aspect/width_in ignored, deterministic placement, no tight-crop reliance):

```python
def build_trace_figure(fld, geom, *, aspect_wh, width_in, linewidth, color, font_scale,
                       style: PlotStyle | None = None) -> Figure:
    w_ratio, h_ratio = aspect_wh
    box = trace_fixed_box(style, float(geom["L"]))
    if box is not None:
        # fixed-scale mode: exact physical box, deterministic placement.
        fig = styled_figure((box[0] + 1.5, box[1] + 1.5), styled=True)
        ax = fig.add_subplot(111)
        _draw_trace_axes(ax, fld, geom, linewidth=linewidth, color=color,
                         font_scale=font_scale, style=style)
        place_axes_box(fig, ax, box[0], box[1])
        return fig
    figsize = (float(width_in), float(width_in) * float(h_ratio) / float(w_ratio))
    fig = styled_figure(figsize, styled=style is not None)
    ax = fig.add_subplot(111)
    ax.set_box_aspect(float(h_ratio) / float(w_ratio))  # legacy: pin the box to w:h
    _draw_trace_axes(ax, fld, geom, linewidth=linewidth, color=color,
                     font_scale=font_scale, style=style)
    return fig
```

Update the imports from `dfxm.common.plotting` at the top of `profiles.py`: add `place_axes_box`, `trace_fixed_box`, keep `fixed_scale_box`/`fit_axes_to_box` (still used by `render_single`). Remove the now-unused `trace_fixed_scale` import only if nothing else in the module uses it (Task 3/4 will use `trace_fixed_box` instead — check with grep before removing).

- [ ] **Step 4: Run the full profiles test file**

Run: `python3 -m pytest tests/test_stage_profiles.py -q`
Expected: PASS. If `test_build_trace_figure_aspect_linewidth_color` fails on figure size, that test exercises the legacy path (no style) and must still pass unchanged — the legacy branch must keep `styled_figure(figsize, styled=style is not None)` semantics exactly.

- [ ] **Step 5: Docs + commit**

`docs/Usage.md` (profiles stage, trace-figures paragraph): trace boxes under a fixed scale are now `line length / scale` wide × "Trace height (cm)" tall (aspect/width apply only when no scale is set); PNGs are no longer tight-cropped in that mode. `docs/Codebase.md` (profiles): mention `_draw_trace_axes` + the two `build_trace_figure` modes.

```bash
git add dfxm/stages/profiles.py tests/test_stage_profiles.py docs/Usage.md docs/Codebase.md
git commit -m "feat(profiles): exact fixed-scale trace boxes via deterministic placement"
```

---

### Task 3: Uniform margins across a run/replot + drift notes for traces

**Files:**
- Modify: `dfxm/stages/profiles.py` — `_save_traces` (~1057), `_render_parameter_job` (~1096), `run()` (~1216 with-block), `render_replot` (~527 with-block)
- Test: `tests/test_stage_profiles.py`
- Modify: `docs/Usage.md` (same section as Task 2 — add the uniform-margin semantics)

**Interfaces:**
- Consumes: `trace_fixed_box`, `measure_axes_margins`, `apply_axes_margins`, `box_drift_note` (Task 1).
- Produces: `_save_traces(..., deferred=None, notes=None)`; `_flush_deferred_traces(deferred, dpi, notes) -> None`; `_render_parameter_job(..., trace_deferred=None)`. Deferred entry shape: `(fig, w_in, h_in, png_path)`.

- [ ] **Step 1: Write the failing tests**

The existing fixture `_write_consolidated` writes slices the jobs draw lines on; add a second job with a different line length so the two jobs' traces must come out with identical canvas heights and identical margins:

```python
def _png_size(path):
    import matplotlib.image as mpimg

    img = mpimg.imread(path)
    return img.shape[1], img.shape[0]  # (w_px, h_px)


def test_run_fixed_scale_traces_share_height_and_margins(tmp_path):
    h5 = tmp_path / "c.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    jobs = (
        '[{"name":"oblique_full","offset_um":0.0,"start_uv":[-5,-3],"end_uv":[5,3],'
        '"n_samples":40,"width_pixels":1,"fig_name":"jobA"},'
        '{"name":"oblique_full","offset_um":0.0,"start_uv":[-2,-1],"end_uv":[2,1],'
        '"n_samples":40,"width_pixels":1,"fig_name":"jobB"}]'
    )
    res = PR.run(
        _base_params(
            h5, out, jobs_json=jobs,
            plot_style={"trace_scale_um_per_cm": 2.0, "trace_height_cm": 3.0},
        )
    )
    assert len(res.jobs) == 2
    sizes = [_png_size(t) for jr in res.jobs for t in jr.traces]
    heights = {h for _, h in sizes}
    assert len(heights) == 1, sizes  # every trace PNG of the run: same pixel height
    # widths track line length: jobA line is ~2.5x jobB's
    wA = _png_size(res.jobs[0].traces[0])[0]
    wB = _png_size(res.jobs[1].traces[0])[0]
    assert wA > wB
    # and no drift notes were emitted
    assert not any("physical scale is off" in n for n in res.notes)
```

Also add the clamp-note test (spec section D — the 30-in clamp must surface visibly):

```python
def test_run_fixed_scale_clamp_appends_note(tmp_path):
    h5 = tmp_path / "c.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "prof"
    res = PR.run(
        _base_params(
            h5, out,
            plot_style={"trace_scale_um_per_cm": 0.001, "trace_height_cm": 3.0},
        )
    )
    assert any("clamped to 30 in" in n for n in res.notes), res.notes
```

Check how `_base_params` threads `plot_style` (grep `style_from_params` in `profiles.py` for the param name — pass the dict the same way existing styled run() tests do; if they build it via `plot_style` param containing a dict of PlotStyle fields, mirror that).

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_stage_profiles.py -q -k share_height`
Expected: FAIL — heights differ (each figure still placed with its own margins) or, before implementation, tight-crop produces per-figure heights.

- [ ] **Step 3: Implement deferral + flush**

`_save_traces`: add keyword params `deferred=None, notes=None`. In the loop, after `build_trace_figure`:

```python
        box = trace_fixed_box(style, float(geom["L"]))
        if box is not None:
            if notes is not None and box[2] != trace_fixed_scale(style):
                notes.append(
                    f"{os.path.basename(tr_png)}: trace box clamped to 30 in — "
                    f"effective scale raised to {box[2]:.4g} µm/cm"
                )
            if deferred is not None:
                deferred.append((fig, box[0], box[1], tr_png))
            else:
                if notes is not None:
                    note = box_drift_note(os.path.basename(tr_png), fig, fig.axes[0], box[0], box[1])
                    if note:
                        notes.append(note)
                fig.savefig(tr_png, dpi=dpi, facecolor="white", edgecolor="none")
        else:
            fig.savefig(tr_png, dpi=dpi, facecolor="white", edgecolor="none", bbox_inches="tight")
        paths.append(tr_png)
```

(The docstring comment at the top of `_save_traces` about "always tight-cropped" must be updated: tight-crop is legacy-mode only now.)

Module-level flush helper (place near `_save_traces`):

```python
def _flush_deferred_traces(deferred, dpi, notes):
    """Second pass for fixed-scale traces: re-place every figure of the
    invocation with the shared max margins (so all PNGs align in a grid),
    verify the box, save, close. First pass (build) already placed each
    figure with its own margins, so single-figure consumers stay exact."""
    if not deferred:
        return
    shared = None
    for fig, w_in, h_in, _png in deferred:
        m = measure_axes_margins(fig, fig.axes[0])
        shared = m if shared is None else shared.max_with(m)
    for fig, w_in, h_in, png in deferred:
        apply_axes_margins(fig, fig.axes[0], w_in, h_in, shared)
        note = box_drift_note(os.path.basename(png), fig, fig.axes[0], w_in, h_in)
        if note:
            notes.append(note)
        fig.savefig(png, dpi=dpi, facecolor="white", edgecolor="none")
```

`_render_parameter_job`: add param `trace_deferred=None`; pass `deferred=trace_deferred, notes=result.notes` into `_save_traces`.

`run()` parameter-mode loop (~1259) and `render_replot` loop (~539): create `trace_deferred: list = []` before the `with h5py.File(...)` block, pass it into `_render_parameter_job`, and after the loop (still inside or right after the with-block — the figures are in memory, h5 no longer needed) call `_flush_deferred_traces(trace_deferred, int(p["fig_dpi"]), result.notes)`.

Imports: add `measure_axes_margins`, `apply_axes_margins`, `box_drift_note` to the plotting imports in `profiles.py`.

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_stage_profiles.py tests/test_figures_replot.py -q`
Expected: PASS (replot shares `_render_parameter_job`, so its tests exercise the deferral too).

- [ ] **Step 5: Docs + commit**

`docs/Usage.md`: add to the trace-figures paragraph — all trace PNGs of one run/replot share the same margins (sized to the largest labels in the set), so they align in any grid; margins can shift slightly if you re-render a subset.

```bash
git add dfxm/stages/profiles.py tests/test_stage_profiles.py docs/Usage.md
git commit -m "feat(profiles): per-invocation uniform trace margins + drift/clamp notes"
```

---

### Task 4: Companion figure on the deterministic engine

**Files:**
- Modify: `dfxm/common/plotting.py` — `add_colorbar` (~784) gains `cax=None`; new `place_axes_stack`
- Modify: `dfxm/stages/profiles.py` — `build_companion_figure` (~702), `save_companion_figure` (~774), `_render_parameter_job` (~1137), replot `_build` closure (~1339)
- Test: `tests/test_stage_profiles.py`, `tests/test_axes_placement.py`
- Modify: `docs/Usage.md` (companion paragraph), `docs/Codebase.md`

**Interfaces:**
- Consumes: Task 1 primitives, `_draw_trace_axes` (Task 2), `trace_fixed_box`.
- Produces:
  - `add_colorbar(fig, im, ax, label, style, *, group=None, cax=None)` — when `cax` is given, `fig.colorbar(im, cax=cax, ...)` (no parent-axes steal); map builders unaffected (they don't pass it).
  - `place_axes_stack(fig, panels, pad_in=0.02, gap_in=0.15) -> None` in plotting.py; `panels` = list of `(ax, w_in, h_in, extras, sync)` where `extras` is a tuple of attached axes measured with the panel and `sync` is `None` or `callable(fig, ax)` invoked after each placement pass (used to keep a manual colorbar glued to its panel).
  - `build_companion_figure(ref, fields, geom, line_color, *, style=None, trace_opts=None, notes=None)`; `trace_opts` = `{"linewidth": float, "color": str|None, "font_scale": float}` — `None` keeps legacy panel styling.
  - `save_companion_figure(ref, fields, geom, line_color, out_png, dpi, style=None, trace_opts=None, notes=None)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_axes_placement.py`:

```python
def test_place_axes_stack_left_aligned_exact_boxes():
    fig = Figure(figsize=(8, 10), facecolor="white")
    axs = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    labels = ["short", "a very very long y label (units)", "mid label"]
    for ax, lab in zip(axs, labels):
        ax.plot([0, 1], [0, 1])
        ax.set_ylabel(lab)
    from dfxm.common.plotting import place_axes_stack

    boxes = [(2.5, 1.6), (1.4, 1.0), (2.0, 1.0)]
    place_axes_stack(fig, [(ax, w, h, (), None) for ax, (w, h) in zip(axs, boxes)])
    x0 = {round(ax.get_position().x0, 4) for ax in axs}
    assert len(x0) == 1  # shared left edge
    for ax, (w, h) in zip(axs, boxes):
        bw, bh = measured_box_in(fig, ax)
        assert abs(bw - w) < 0.01 and abs(bh - h) < 0.01
    # panels must not overlap: y-intervals strictly descending
    ys = [ax.get_position() for ax in axs]
    assert ys[0].y0 > ys[1].y1 - 1e-6 and ys[1].y0 > ys[2].y1 - 1e-6
```

In `tests/test_stage_profiles.py` (reuse `_companion_inputs()` at ~677):

```python
def test_companion_fixed_scale_panel_boxes_and_trace_style():
    ref, fields, geom = _companion_inputs()
    st = PlotStyle(scale_um_per_cm=20.0, trace_scale_um_per_cm=2.0, trace_height_cm=3.0)
    topts = {"linewidth": 2.5, "color": "red", "font_scale": 1.4}
    fig = PR.build_companion_figure(ref, fields, geom, "white", style=st, trace_opts=topts)
    from dfxm.common.plotting import measured_box_in

    # trace axes carry the plotted lines; the manual colorbar axes has none
    ax_map, ax_traces = fig.axes[0], [a for a in fig.axes[1:] if a.lines]
    u, v = ref[1], ref[2]
    ext_u, ext_v = float(u[-1] - u[0]), float(v[-1] - v[0])
    mw, mh = measured_box_in(fig, ax_map)
    assert abs(mw - ext_u / 20.0 / 2.54) < 0.01 * max(1.0, mw)
    assert abs(mh - ext_v / 20.0 / 2.54) < 0.01 * max(1.0, mh)
    for ax in ax_traces:
        tw, th = measured_box_in(fig, ax)
        assert abs(tw - geom["L"] / 2.0 / 2.54) < 0.01 * tw
        assert abs(th - 3.0 / 2.54) < 0.01 * th
        assert abs(ax.lines[0].get_linewidth() - 2.5) < 1e-9  # trace_opts, not 1.8
        assert ax.yaxis.label.get_fontsize() == 10 * 1.4  # trace font scale, not map


def test_companion_fixed_scale_show_title_false_no_panel_titles():
    ref, fields, geom = _companion_inputs()
    st = PlotStyle(
        scale_um_per_cm=20.0, trace_scale_um_per_cm=2.0, trace_height_cm=3.0, show_title=False
    )
    fig = PR.build_companion_figure(ref, fields, geom, "white", style=st)
    for ax in fig.axes:
        assert ax.get_title() == "" and ax.get_title(loc="left") == ""


def test_companion_without_fixed_scale_keeps_legacy_layout():
    ref, fields, geom = _companion_inputs()
    fig_none = PR.build_companion_figure(ref, fields, geom, "white", style=None)
    w, h = fig_none.get_size_inches()
    assert abs(w - 9.0) < 1e-6  # legacy canvas untouched
    fig_styled = PR.build_companion_figure(ref, fields, geom, "white", style=PlotStyle())
    w2, _ = fig_styled.get_size_inches()
    assert abs(w2 - 9.0) < 1e-6  # styled-but-no-scale also legacy
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_axes_placement.py tests/test_stage_profiles.py -q -k "stack or companion"`
Expected: new tests FAIL (`place_axes_stack` missing; companion ignores trace_opts / boxes not physical); `test_companion_map_styled_scale_bar_honours_style` may now route into the new layout if it sets a fixed scale — read it and keep its assertion valid in whichever path it exercises.

- [ ] **Step 3: Implement**

`plotting.py` — `add_colorbar`: change signature to `(fig, im, ax, label, style, *, group=None, cax=None)` and the first line to:

```python
    if cax is not None:
        cb = fig.colorbar(im, cax=cax)
    else:
        cb = fig.colorbar(im, ax=ax, fraction=style.colorbar_fraction, pad=0.04)
```

`plotting.py` — `place_axes_stack` (after `place_axes_box`):

```python
def place_axes_stack(fig, panels, pad_in: float = 0.02, gap_in: float = 0.15) -> None:
    """Stack *panels* top→bottom, each with an EXACT (w_in, h_in) box, sharing
    one left margin (the max over panels) so their boxes left-align.

    panels: list of (ax, w_in, h_in, extras, sync). *extras* are attached axes
    (a manual colorbar) counted in the panel's decoration envelope; *sync* is
    an optional callable(fig, ax) re-gluing attachments after placement.
    Two passes: provisional placement at final box sizes → measure → final.
    """
    fig.set_layout_engine("none")
    n = len(panels)
    prov_w = max(w for _, w, _, _, _ in panels) + 2.5
    prov_h = sum(h for _, _, h, _, _ in panels) + 1.5 * (n + 1)
    fig.set_size_inches(prov_w, prov_h, forward=False)
    y = prov_h - 1.5
    for ax, w, h, _extras, sync in panels:
        y -= h
        ax.set_position([1.5 / prov_w, y / prov_h, w / prov_w, h / prov_h])
        y -= 1.5
        if sync is not None:
            sync(fig, ax)
    margins = [
        measure_axes_margins(fig, ax, extras=extras, pad_in=pad_in)
        for ax, _w, _h, extras, _s in panels
    ]
    left = max(m.left for m in margins)
    fig_w = left + max(w + m.right for (_a, w, _h, _e, _s), m in zip(panels, margins))
    fig_h = sum(m.top + h + m.bottom for (_a, _w, h, _e, _s), m in zip(panels, margins))
    fig_h += gap_in * (n - 1)
    fig.set_size_inches(fig_w, fig_h, forward=False)
    y = fig_h
    for (ax, w, h, _extras, sync), m in zip(panels, margins):
        y -= m.top + h
        ax.set_position([left / fig_w, y / fig_h, w / fig_w, h / fig_h])
        y -= m.bottom + gap_in
        if sync is not None:
            sync(fig, ax)
```

`profiles.py` — split `build_companion_figure`: rename the current body to `_build_companion_legacy(ref, fields, geom, line_color, style)` **verbatim** (it is pinned by tests), and make the public function dispatch:

```python
def build_companion_figure(
    ref, fields, geom, line_color, *, style=None, trace_opts=None, notes=None
) -> Figure:
    if trace_fixed_box(style, float(geom["L"])) is None:
        return _build_companion_legacy(ref, fields, geom, line_color, style)
    return _build_companion_fixed(ref, fields, geom, line_color, style, trace_opts, notes)
```

`_build_companion_fixed`:

```python
def _build_companion_fixed(ref, fields, geom, line_color, style, trace_opts, notes):
    """Fixed-scale companion: map panel at the MAP scale, trace panels styled
    exactly like the standalone trace figures, stacked left-aligned."""
    ref_plane, u_um, v_um, ref_attrs, ref_label = ref
    topts = {"linewidth": 1.8, "color": None, "font_scale": 1.0, **(trace_opts or {})}
    ext_u, ext_v = float(u_um[-1] - u_um[0]), float(v_um[-1] - v_um[0])
    map_scale = fixed_scale(style) or trace_fixed_scale(style)
    mbox = fixed_scale_box(style, ext_u, ext_v, scale=map_scale)
    tbox = trace_fixed_box(style, float(geom["L"]))
    fig = styled_figure((10.0, 10.0), styled=True)
    fig.set_layout_engine("none")
    n = len(fields)
    ax_img = fig.add_subplot(n + 1, 1, 1)
    im = _draw_reference_image(
        ax_img, ref_plane, u_um, v_um, ref_attrs, line_color, geom=geom,
        title=(f"{ref_attrs['title']}\nreference: {ref_label}" if style.show_title else None),
        style=style, fixed_scale_um_per_cm=mbox[2],
    )
    cax = None
    if style.colorbar:
        cax = fig.add_axes([0.9, 0.6, 0.03, 0.25])  # provisional; sync repositions it
        add_colorbar(fig, im, ax_img, ref_attrs["cbar_label"], style,
                     group=GROUP_BY_KIND.get(ref_attrs.get("kind")), cax=cax)
    apply_text_scale(ax_img, style)

    def _sync_cax(fig_, ax_, _w=mbox[0]):
        if cax is None:
            return
        pos = ax_.get_position()
        fw, _fh = fig_.get_size_inches()
        cax.set_position(
            [pos.x1 + 0.04 * _w / fw, pos.y0, style.colorbar_fraction * _w / fw, pos.height]
        )

    trace_axes = []
    for i, fld in enumerate(fields):
        ax = fig.add_subplot(n + 1, 1, i + 2)
        _draw_trace_axes(
            ax, fld, geom, linewidth=topts["linewidth"], color=topts["color"],
            font_scale=topts["font_scale"], style=style, show_xlabel=(i == n - 1),
        )
        if i < n - 1:
            ax.tick_params(labelbottom=False)
        trace_axes.append(ax)
    panels = [(ax_img, mbox[0], mbox[1], (cax,) if cax is not None else (), _sync_cax)]
    panels += [(ax, tbox[0], tbox[1], (), None) for ax in trace_axes]
    place_axes_stack(fig, panels)
    if notes is not None:
        for label, ax, (w, h) in [("companion map", ax_img, (mbox[0], mbox[1]))] + [
            (f"companion trace {fld['vid']}", ax, (tbox[0], tbox[1]))
            for fld, ax in zip(fields, trace_axes)
        ]:
            note = box_drift_note(label, fig, ax, w, h)
            if note:
                notes.append(note)
    return fig
```

`save_companion_figure`: add `trace_opts=None, notes=None` pass-throughs; **no `bbox_inches`** change needed (it already saves full-canvas).

`_render_parameter_job` (~1137): pass
`trace_opts={"linewidth": trace_linewidth, "color": trace_color, "font_scale": trace_font_scale}, notes=result.notes` into `save_companion_figure`.

Replot `_build` closure (~1339): pass the same dict built from the already-extracted `trace_linewidth`/`trace_color`/`trace_font_scale` locals (close over them like `_asp`/`_w` in `_tbuild`).

Imports in `profiles.py`: add `place_axes_stack`, `fixed_scale` (check whether already imported).

- [ ] **Step 4: Run the tests**

Run: `python3 -m pytest tests/test_stage_profiles.py tests/test_axes_placement.py tests/test_figures_replot.py -q`
Expected: PASS, including the legacy pinned companion tests.

- [ ] **Step 5: Docs + commit**

`docs/Usage.md` (companion paragraph): under a fixed scale the companion becomes a left-aligned stack — map panel at the map scale, trace panels identical to the standalone trace figures (trace fonts/line width/colour), panel titles follow "Show title". `docs/Codebase.md`: `place_axes_stack`, `add_colorbar(cax=)`, `_build_companion_fixed`/`_build_companion_legacy`.

```bash
git add dfxm/common/plotting.py dfxm/stages/profiles.py tests/ docs/Usage.md docs/Codebase.md
git commit -m "feat(profiles): fixed-scale companion on the deterministic stack layout"
```

---

### Task 5: Overview drift guard + the `trace_height_cm` GUI knob

**Files:**
- Modify: `dfxm/common/plotting.py:116-119` (PlotStyle fields)
- Modify: `dfxm/stages/profiles.py` — `render_single` (~869), `_save_overviews` (~1039), preview call site in `run()` (~1251), `_render_parameter_job` overview call (~1157)
- Modify: `gui/widgets/export_dialog.py` — `StyleControls` (`sync_from_style` ~101, `_all_widgets` ~183, `_build_controls` trace-scale row ~566-578, handler ~632)
- Test: `tests/test_plot_style.py`, `tests/test_stage_profiles.py`, `tests/gui_smoke.py` (StyleControls block, step [16])
- Modify: `docs/Usage.md` (publication-style knob table), `docs/Codebase.md`

**Interfaces:**
- Consumes: `box_drift_note` (Task 1).
- Produces: `PlotStyle.trace_height_cm: float | None = None` (None/blank → 3.0 via `trace_height_cm()` reader); `render_single(..., notes=None)`; `_save_overviews(..., notes=None)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_plot_style.py` (find the existing round-trip test for `style_to_json`/`style_from_json` and extend it, plus):

```python
def test_trace_height_cm_roundtrips_and_defaults():
    from dfxm.common.plotting import PlotStyle, style_from_json, style_to_json, trace_height_cm

    st = PlotStyle(trace_height_cm=4.2)
    assert style_from_json(style_to_json(st)).trace_height_cm == 4.2
    # old persisted styles (no field) load with the default
    st_old = style_from_json(style_to_json(PlotStyle()))
    assert trace_height_cm(st_old) == 3.0
```

`tests/test_stage_profiles.py`:

```python
def test_render_single_appends_drift_note_on_forced_miss(tmp_path, monkeypatch):
    # force fit_axes_to_box to do nothing so the guard must catch the miss
    import dfxm.stages.profiles as prof

    monkeypatch.setattr(prof, "fit_axes_to_box", lambda *a, **k: False)
    ref, fields, geom = _companion_inputs()
    notes = []
    prof.render_single(
        ref, geom, "white", str(tmp_path / "ov.png"), "hdr", 100,
        style=PlotStyle(scale_um_per_cm=20.0), notes=notes,
    )
    assert notes and "physical scale is off" in notes[0]
```

`tests/gui_smoke.py`: in the StyleControls step [16] block (~line 439), after the existing assertions add:

```python
    # trace height knob writes through to the style (blank -> None -> 3.0 default)
    ctrls = _StyleControls(session_style)
    ctrls._w_trace_height_cm.setText("4.5")
    assert session_style.trace_height_cm == 4.5, session_style.trace_height_cm
    ctrls._w_trace_height_cm.setText("")
    assert session_style.trace_height_cm is None
```

(Match the surrounding smoke style — it uses plain asserts with `[16]`-style print markers; keep its numbering scheme intact.)

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_plot_style.py tests/test_stage_profiles.py -q -k "trace_height or drift_note"`
Expected: FAIL — no `trace_height_cm` field / no `notes` param.

- [ ] **Step 3: Implement**

(`PlotStyle.trace_height_cm` was already added in Task 1 — this task only wires the GUI and the guards.)

`render_single`: add `notes=None` keyword; after the existing `fit_axes_to_box(fig, ax, box[0], box[1])` call:

```python
    if box is not None and notes is not None:
        note = box_drift_note(os.path.basename(out_png), fig, ax, box[0], box[1])
        if note:
            notes.append(note)
```

`_save_overviews`: add `notes=None`, pass through to `render_single`. `_render_parameter_job`: pass `notes=result.notes` at the overview call (~1157). Preview call in `run()` (~1251): pass `notes=result.notes`.

`StyleControls`: mirror the trace-scale row exactly —
- `_build_controls` (after the trace-scale `form.addRow`, ~578): `self._w_trace_height_cm = QLineEdit()`, placeholder `"(blank = 3)"`, tooltip "Fixed height of every trace plot box in cm of page. All traces of a run share it, so they align side-by-side. Blank = 3 cm.", `textChanged` → `self._on_trace_height_cm`, `form.addRow("Trace height (cm)", self._w_trace_height_cm)`.
- handler (copy `_on_trace_scale_umcm` ~632, adjusted):

```python
    def _on_trace_height_cm(self, text: str) -> None:
        t = text.strip()
        val = None
        if t:
            try:
                val = float(t)
            except ValueError:
                return  # ignore partial input
            if not (val > 0):
                return
        self._style.trace_height_cm = val
        self._emit()
```

(Match the real `_on_trace_scale_umcm` body when copying — read it first; it may guard/emit differently.)
- `sync_from_style` (~166): `_thv = getattr(s, "trace_height_cm", None)` → `self._w_trace_height_cm.setText(f"{_thv:g}" if _thv is not None else "")`.
- `_all_widgets` (~214): append `self._w_trace_height_cm`.

- [ ] **Step 4: Run tests + smoke**

Run: `python3 -m pytest tests/test_plot_style.py tests/test_stage_profiles.py -q && python3 tests/gui_smoke.py`
Expected: pytest PASS; smoke prints all steps OK (offscreen — run with `QT_QPA_PLATFORM=offscreen` if the environment needs it; check how the repo invokes it in `verify-suite`).

- [ ] **Step 5: Docs + commit**

`docs/Usage.md`: add "Trace height (cm)" to the publication-style knob list (default 3, what it does, that it pairs with Trace scale). `docs/Codebase.md`: PlotStyle field + `render_single(notes=)`.

```bash
git add dfxm/common/plotting.py dfxm/stages/profiles.py gui/widgets/export_dialog.py tests/ docs/
git commit -m "feat(style): trace_height_cm knob + overview drift guard notes"
```

---

### Task 6: Full verification + docs coherence pass

**Files:**
- Modify (if drift found): `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:** none — verification only.

- [ ] **Step 1: Docs coherence read-through**

Read the profiles sections of `docs/Usage.md` and `docs/Codebase.md` end-to-end once; confirm every behaviour change from Tasks 2–5 is described (fixed box height, uniform margins, no tight-crop in fixed mode, companion stack, drift notes, the new knob) and nothing stale remains (e.g. "always tight-cropped", `trace_aspect` claims in fixed mode). Fix in place.

- [ ] **Step 2: Run the full verification suite**

Use the repo's `verify-suite` skill (it runs pytest + ruff check + ruff format check + gui_smoke). Expected: suite green (625+new passed / 13 skipped), ruff clean, smoke all steps.

- [ ] **Step 3: Real-data canary (this machine has the failing files)**

Run a headless replot against the real STO2 data that exposed the bug:

```bash
python3 - <<'EOF'
import json
from dfxm.common.plotting import PlotStyle
from dfxm.stages import profiles as PR

jobs = [
    {"name": "oblique_full", "offset_um": -3.72, "start_uv": [-14.8, -1.9], "end_uv": [11.9, 11.0], "n_samples": 197, "width_pixels": 1, "fig_name": "canary_full3"},
]
# NOTE: exact job specs live in the GUI form's jobs_json — if available, paste the
# real jobs_json here instead of the synthetic line above.
style = PlotStyle(scale_um_per_cm=10.0, trace_scale_um_per_cm=10.0, trace_height_cm=3.0)
res = PR.render_replot(
    "/mnt/data/ESRF/STO2_overnight_plots/oblique_slices.h5",
    jobs, style, {}, "/tmp/trace_canary",
)
print("notes:", res.notes)
print("skipped:", res.skipped)
for jr in res.jobs:
    print(jr.name, jr.traces)
EOF
```

Then measure the produced trace PNGs (spine-to-spine) the same way the diagnosis did and confirm implied scale = 10.00 ± 0.05 µm/cm and all heights identical. Report the numbers to the user — this is the acceptance evidence.

- [ ] **Step 4: Commit any doc fixes**

```bash
git add docs/
git commit -m "docs: coherence pass for trace fixed-scale project"  # only if changed
```

Then hand over to the finishing flow (`finish-and-record` skill) — merge decision belongs to Albert.
