# Fixed-scale figures + slice selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (Part I) A publication-style knob `scale_um_per_cm` that renders every styled **map** figure at an identical physical µm-per-cm scale (draw–measure–resize fitting, point-exact scale bar), opt-in and byte-identical to today when blank. (Part II) Fast pinned-plane re-runs on the slices stage (`use_pinned` + `pinned_slices_json` + Pin planes… dialog) and a planes-first selection widget (planes listed once, filter box, check-all-visible) replacing the per-volume trees in both replot dialogs.

**Architecture:** All figure logic stays in the Qt-free `dfxm/` core (`dfxm/common/plotting.py` grows `fixed_scale`/`fixed_scale_box`/`fit_axes_to_box`; the three styled builders in `dfxm/common/render.py`, `dfxm/stages/slices.py`, `dfxm/stages/strain.py` plus the profiles overview call them). GUI work is confined to `gui/`: one new field in `StyleControls`, a new pure-Python selection model (`gui/widgets/plane_selection_model.py`, PySide6-free, unit-testable), a new `PlaneSelectionPanel` widget, a new `PinPlanesDialog`, and swaps inside the two existing replot dialogs. The pin-snap core moves from `tools/pin_slice.py` into `dfxm/stages/slices.py`.

**Tech Stack:** Python 3.10, numpy, h5py, matplotlib (explicit `Figure`/Agg API — never pyplot), PySide6 (GUI only), pytest. Lint: ruff (line length 100, double quotes).

## Global Constraints

- **`dfxm/` stays Qt-free.** Never import PySide6/pyvista in `dfxm/`. Figures via `matplotlib.figure.Figure` only; never `pyplot` or `matplotlib.use(...)`.
- **Docs contract:** every task that changes stage params/behaviour, a public function, or a viewer updates `docs/Usage.md` (user behaviour) AND `docs/Codebase.md` (code structure) **in the same commit**. Locate sections by grepping headings; do not restructure the docs.
- **Params:** declared via the `Param`/`StageSpec` schema (`dfxm/config/models.py`). Every param needs `help` written for a first-time beamline user; `advanced=True` params need `group=`; `tests/test_param_metadata.py` enforces this. User-fixable input errors raise `StageUserError(message, hint=...)` from `dfxm.common.errors`.
- **Style parsing is defensive:** bad persisted values (strings, ≤0, junk) degrade to "off", never raise.
- **Opt-in fidelity:** with `scale_um_per_cm` unset, and with `use_pinned` off, behaviour must be byte-identical to master — the existing suite is the regression net; never change a legacy default.
- **Form persistence:** `gui/form_state.py` `FormStateStore` persists stage form fields per experiment automatically. New slices params just work — do NOT add custom persistence.
- Verification commands: `python3 -m pytest -q` (full suite), targeted `python3 -m pytest -q tests/<file>.py`, `ruff check . && ruff format .`, `python3 tests/gui_smoke.py` (numbered steps — extend the numbering when adding GUI features; it is NOT a pytest file).
- **Git:** work on branch `feature/fixed-scale-slice-selection` (create from master if absent: `git checkout -b feature/fixed-scale-slice-selection`). This repo has **no remote** — never pull/push/PR. Commit after every green task; end every commit message with the standard repo trailer lines (Co-Authored-By + Claude-Session) your harness instructions specify.
- **Read before edit:** always Read a file (or the exact target region) before editing it; `hint=` strings in `dfxm/stages/*.py` contain em-dashes at varying indents — never reconstruct `old_string` from memory.
- Line length 100; double quotes; `ruff format` runs via hook on Write/Edit.

---

# Phase A — Fixed-scale (µm-per-cm) map figures

## Task A1: `PlotStyle.scale_um_per_cm` knob + defensive accessor + target-box helper

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/common/plotting.py` (dataclass at ~line 56; add module functions after `figure_size` ~line 241)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_plot_style.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md` (plotting.py section)

**Interfaces:**
- Produces: `PlotStyle.scale_um_per_cm: float | None = None` (new dataclass field, in the `# figure` block next to `figure_width`).
- Produces: `fixed_scale(style) -> float | None` — defensive read: `None` for `style=None`, missing attr, blank/non-numeric/≤0/non-finite values.
- Produces: `fixed_scale_box(style, ext_x_um, ext_y_um) -> tuple[float, float, float] | None` — `(w_in, h_in, effective_um_per_cm)` of the target axes box, or `None` when the knob is off or extents are degenerate; sides clamped to 30 in (aspect preserved, effective scale raised, warning logged).
- Consumes: nothing new; `style_to_json`/`style_from_json`/`_style_from_dict` pick the field up automatically (dataclass `fields()`-driven).

**Steps:**

- [ ] Read `/home/albert/Desktop/dfxm_pipeline/dfxm/common/plotting.py` lines 55–160 and 225–245, and `/home/albert/Desktop/dfxm_pipeline/tests/test_plot_style.py` in full.
- [ ] Append to `tests/test_plot_style.py` (add `import pytest` and extend the existing `from dfxm.common.plotting import ...` import with `fixed_scale, fixed_scale_box, style_from_json, style_to_json` as needed):

```python
def test_fixed_scale_defensive_parse():
    assert fixed_scale(None) is None
    assert fixed_scale(PlotStyle()) is None
    assert fixed_scale(PlotStyle(scale_um_per_cm=50.0)) == 50.0
    assert fixed_scale(PlotStyle(scale_um_per_cm="50")) == 50.0  # stale persisted string
    assert fixed_scale(PlotStyle(scale_um_per_cm="junk")) is None
    assert fixed_scale(PlotStyle(scale_um_per_cm=-3)) is None
    assert fixed_scale(PlotStyle(scale_um_per_cm=0)) is None


def test_fixed_scale_box_geometry_clamp_and_degenerate():
    box = fixed_scale_box(PlotStyle(scale_um_per_cm=50.0), 200.0, 100.0)
    assert box is not None
    w, h, eff = box
    assert w == pytest.approx(200.0 / 50.0 / 2.54)
    assert h == pytest.approx(100.0 / 50.0 / 2.54)
    assert eff == 50.0
    # typo scale (0.1 µm/cm on 200 µm ≈ 787 in): clamped to 30 in, aspect kept, scale raised
    w, h, eff = fixed_scale_box(PlotStyle(scale_um_per_cm=0.1), 200.0, 100.0)
    assert max(w, h) == pytest.approx(30.0)
    assert h / w == pytest.approx(0.5)
    assert eff > 0.1
    # degenerate extents / knob off -> None
    assert fixed_scale_box(PlotStyle(scale_um_per_cm=50.0), 0.0, 100.0) is None
    assert fixed_scale_box(PlotStyle(), 200.0, 100.0) is None
    assert fixed_scale_box(None, 200.0, 100.0) is None


def test_scale_um_per_cm_json_roundtrip_and_old_snapshots():
    s2 = style_from_json(style_to_json(PlotStyle(scale_um_per_cm=75.0)))
    assert s2 is not None and s2.scale_um_per_cm == 75.0
    old = style_from_json('{"font_scale": 2.0}')  # snapshot predating the knob
    assert old is not None and old.scale_um_per_cm is None
```

- [ ] Run `python3 -m pytest -q tests/test_plot_style.py` — expect ImportError (`fixed_scale` not defined) / `TypeError: unexpected keyword argument 'scale_um_per_cm'`.
- [ ] Implement in `dfxm/common/plotting.py`. Add to the dataclass, directly under `figure_width`:

```python
    # fixed physical scale for MAP figures: µm of data per cm of page. None/blank = off.
    # When set (>0), figure_width is ignored for maps (trace figures keep it).
    scale_um_per_cm: float | None = None
```

Add near the top (after the existing imports): `import logging` and `_log = logging.getLogger(__name__)`. Add after `figure_size`:

```python
_MAX_FIXED_SIDE_IN = 30.0  # sanity cap: a typo scale must not request a 47k-pixel render


def fixed_scale(style: "PlotStyle | None") -> float | None:
    """Defensively read ``style.scale_um_per_cm``: a positive finite float, else None.

    Stale persisted styles may carry strings or nonsense — degrade to None
    (today's behaviour), matching the other style-field guards. Never raises.
    """
    if style is None:
        return None
    v = getattr(style, "scale_um_per_cm", None)
    if v is None or v == "":
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if (v > 0 and math.isfinite(v)) else None


def fixed_scale_box(
    style: "PlotStyle | None", ext_x_um: float, ext_y_um: float
) -> tuple[float, float, float] | None:
    """Target axes-box (w_in, h_in, effective_um_per_cm) for fixed-scale mode.

    Returns None when the knob is off or the extents are degenerate (skip
    fitting). Sides are clamped to 30 in preserving aspect — the scale is
    effectively raised and a warning logged, never an exception.
    """
    s = fixed_scale(style)
    if s is None:
        return None
    if not (math.isfinite(ext_x_um) and math.isfinite(ext_y_um)) or ext_x_um <= 0 or ext_y_um <= 0:
        return None
    w, h = ext_x_um / s / 2.54, ext_y_um / s / 2.54
    m = max(w, h)
    if m > _MAX_FIXED_SIDE_IN:
        f = _MAX_FIXED_SIDE_IN / m
        w, h, s = w * f, h * f, s / f
        _log.warning(
            "fixed-scale box clamped to %.0f in per side; effective scale raised to %.4g um/cm",
            _MAX_FIXED_SIDE_IN,
            s,
        )
    return (w, h, s)
```

- [ ] Run `python3 -m pytest -q tests/test_plot_style.py` — expect all pass.
- [ ] Update `docs/Codebase.md`: in the `dfxm/common/plotting.py` section, document the new `scale_um_per_cm` field and the `fixed_scale`/`fixed_scale_box` functions (grep for `figure_size` in the doc to find the right spot).
- [ ] `ruff check . && ruff format .`, then `git add -A && git commit -m "feat(plotting): scale_um_per_cm knob + fixed_scale_box helper (fixed-scale groundwork)"`.

## Task A2: `fit_axes_to_box` draw–measure–resize helper

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/common/plotting.py` (add after `fixed_scale_box`)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_figure_layout.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Produces: `fit_axes_to_box(fig, ax, w_in, h_in, tol_in=0.02, max_iter=3) -> bool` — attaches a `FigureCanvasAgg` if the canvas cannot render, then iteratively corrects the figure size ADDITIVELY until the axes box is within `tol_in` of `(w_in, h_in)`; non-convergence keeps the last size, logs, returns `False` (never raises).
- Produces: `finalize_fixed_scale(fig, ax, style, ext_x_um, ext_y_um) -> None` — convenience: `box = fixed_scale_box(style, ...)`; if set, fit to it; else no-op.
- Consumes: `fixed_scale_box` (Task A1); `matplotlib.backends.backend_agg.FigureCanvasAgg`.

**Steps:**

- [ ] Read `/home/albert/Desktop/dfxm_pipeline/tests/test_figure_layout.py` in full (match its import style). Append:

```python
def _box_inches(fig, ax):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    return bb.width / fig.dpi, bb.height / fig.dpi


def test_fit_axes_to_box_reaches_target_under_two_decoration_loads():
    import numpy as np

    from dfxm.common.plotting import fit_axes_to_box, styled_figure

    for title, with_cbar in (
        ("A long two-line title\nwith even more text on the second line", True),
        ("t", False),
    ):
        fig = styled_figure((6.0, 5.0), styled=True)
        ax = fig.add_subplot(111)
        im = ax.imshow(
            np.random.default_rng(0).random((10, 20)),
            extent=[0, 200, 0, 100],
            origin="lower",
            aspect="equal",
        )
        ax.set_title(title)
        if with_cbar:
            fig.colorbar(im, ax=ax, fraction=0.07)
        assert fit_axes_to_box(fig, ax, 3.0, 1.5) is True
        w, h = _box_inches(fig, ax)
        assert abs(w - 3.0) <= 0.05 and abs(h - 1.5) <= 0.05


def test_fit_axes_to_box_nonconvergence_is_nonfatal():
    import numpy as np

    from dfxm.common.plotting import fit_axes_to_box, styled_figure

    fig = styled_figure((2.0, 2.0), styled=True)
    ax = fig.add_subplot(111)
    ax.imshow([[0.0, 1.0]], extent=[0, 2, 0, 1], origin="lower", aspect="equal")
    ok = fit_axes_to_box(fig, ax, 5.0, 2.5, tol_in=1e-9, max_iter=1)
    assert ok is False  # kept the last size, did not raise
    assert np.all(np.isfinite(fig.get_size_inches()))


def test_finalize_fixed_scale_noop_when_knob_off():
    from dfxm.common.plotting import PlotStyle, finalize_fixed_scale, styled_figure

    fig = styled_figure((6.0, 5.0), styled=True)
    ax = fig.add_subplot(111)
    finalize_fixed_scale(fig, ax, PlotStyle(), 200.0, 100.0)
    finalize_fixed_scale(fig, ax, None, 200.0, 100.0)
    assert tuple(fig.get_size_inches()) == (6.0, 5.0)
```

- [ ] Run `python3 -m pytest -q tests/test_figure_layout.py` — expect ImportError on `fit_axes_to_box`.
- [ ] Implement in `dfxm/common/plotting.py`:

```python
def fit_axes_to_box(fig, ax, w_in: float, h_in: float, tol_in: float = 0.02, max_iter: int = 3):
    """Resize *fig* until *ax*'s box is (w_in, h_in) inches, within *tol_in*.

    Draws, measures ``ax.get_window_extent()``, and corrects the figure size
    ADDITIVELY by the miss (decorations are constant in inches, so the first
    correction is nearly exact; the loop is insurance). The target box must
    have the data aspect so aspect="equal" does not fight the fit. Returns
    True on convergence; non-convergence keeps the last size, logs, and
    returns False — never fatal.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if fig.canvas is None or not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    for _ in range(max(1, int(max_iter))):
        fig.canvas.draw()
        bb = ax.get_window_extent(fig.canvas.get_renderer())
        cur_w, cur_h = bb.width / fig.dpi, bb.height / fig.dpi
        dw, dh = w_in - cur_w, h_in - cur_h
        if abs(dw) <= tol_in and abs(dh) <= tol_in:
            return True
        fw, fh = fig.get_size_inches()
        fig.set_size_inches(max(fw + dw, 0.5), max(fh + dh, 0.5), forward=False)
    _log.info("fit_axes_to_box: miss > %.3f in after %d iterations (kept last size)", tol_in, max_iter)
    return False


def finalize_fixed_scale(fig, ax, style: "PlotStyle | None", ext_x_um: float, ext_y_um: float) -> None:
    """Fit *ax* to the fixed-scale target box when the knob is on; else no-op."""
    box = fixed_scale_box(style, ext_x_um, ext_y_um)
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])
```

- [ ] Run `python3 -m pytest -q tests/test_figure_layout.py` — expect pass. Then `python3 -m pytest -q` for the None-regression net.
- [ ] Update `docs/Codebase.md` plotting.py section with both functions.
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(plotting): fit_axes_to_box draw-measure-resize helper + finalize_fixed_scale"`.

## Task A3: scale bar fixed mode (point-exact bar height)

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/common/plotting.py` (`draw_scale_bar`, ~line 419; the `bh = ...` line is at ~442)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_plot_style.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Produces: `draw_scale_bar(ax, length_um=None, *, style, fixed_scale_um_per_cm=None)` — new keyword-only arg. When set (>0): bar height in data units = `style.scale_bar_thickness_pt * (2.54 / 72.0) * fixed_scale_um_per_cm` (true points at the known scale). When `None`: today's `abs(yr) * 0.004 * style.scale_bar_thickness_pt` byte-for-byte. The bar **never** infers the scale from the style — callers pass it explicitly (only builders that actually fit do).
- Consumes: nothing new. All existing call sites keep working unchanged (kwarg defaults to `None`).

**Steps:**

- [ ] Read `dfxm/common/plotting.py` lines 419–513. Append to `tests/test_plot_style.py`:

```python
def _bar_rect(ax):
    """The scale-bar Rectangle inside the AnchoredOffsetbox assembly."""
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox

    box = next(a for a in ax.artists if isinstance(a, AnchoredOffsetbox))
    stack = [box.get_child()]
    while stack:
        a = stack.pop()
        if isinstance(a, AuxTransformBox):
            return a.get_children()[0]
        if hasattr(a, "get_children"):
            stack.extend(a.get_children())
    raise AssertionError("no bar rectangle found")


def _bar_axes(xr=200.0, yr=100.0):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, xr)
    ax.set_ylim(0, yr)
    return ax


def test_draw_scale_bar_fixed_mode_height_is_point_exact():
    style = PlotStyle(scale_bar_thickness_pt=4.0)
    ax = _bar_axes()
    draw_scale_bar(ax, 50.0, style=style, fixed_scale_um_per_cm=100.0)
    assert _bar_rect(ax).get_height() == pytest.approx(4.0 * (2.54 / 72.0) * 100.0)
    assert _bar_rect(ax).get_width() == pytest.approx(50.0)


def test_draw_scale_bar_default_mode_geometry_unchanged():
    style = PlotStyle(scale_bar_thickness_pt=3.0)
    ax = _bar_axes(yr=100.0)
    draw_scale_bar(ax, 50.0, style=style)  # no kwarg -> today's geometry
    assert _bar_rect(ax).get_height() == pytest.approx(100.0 * 0.004 * 3.0)
```

(Extend the file's `from dfxm.common.plotting import ...` line to include `draw_scale_bar` if not already imported.)
- [ ] Run `python3 -m pytest -q tests/test_plot_style.py` — first test fails: `TypeError: draw_scale_bar() got an unexpected keyword argument`.
- [ ] Implement: change the signature to `def draw_scale_bar(ax, length_um: float | None = None, *, style: "PlotStyle", fixed_scale_um_per_cm: float | None = None) -> None:` and replace the single `bh = abs(yr) * 0.004 * style.scale_bar_thickness_pt` line with:

```python
    if fixed_scale_um_per_cm:
        # Fixed-scale mode: bar height = thickness in TRUE points at the known scale
        # (1 pt = 2.54/72 cm of page = that many cm x um-per-cm of data).
        bh = style.scale_bar_thickness_pt * (2.54 / 72.0) * float(fixed_scale_um_per_cm)
    else:
        # Bar height in data coords: 0.004*thickness_pt*|yr| (unchanged legacy geometry).
        bh = abs(yr) * 0.004 * style.scale_bar_thickness_pt
```

Also extend the docstring: fixed mode is opt-in per call; un-fitted callers must not pass it.
- [ ] Run `python3 -m pytest -q tests/test_plot_style.py tests/test_stage_slices.py tests/test_export_fidelity.py` — expect pass (default mode untouched). Then full `python3 -m pytest -q`.
- [ ] Update `docs/Codebase.md` `draw_scale_bar` entry with the new kwarg semantics.
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(plotting): draw_scale_bar fixed_scale_um_per_cm mode (point-exact bar height)"`.

## Task A4: wire the shared layer builder + strain diagnostic

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/common/render.py` (`layer_figure`, ~line 38)
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/stages/strain.py` (`build_strain_map`, ~lines 353–412)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_figure_layout.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Consumes: `fixed_scale_box`, `fit_axes_to_box` from `dfxm.common.plotting`; `draw_scale_bar(..., fixed_scale_um_per_cm=...)` (Task A3).
- Produces: `layer_figure(...)` and `build_strain_map(...)` (signatures unchanged) render fixed-scale when `style.scale_um_per_cm` is set: initial figsize = target box + 1.5 in headroom (figure_width ignored for maps), bar gets the effective scale, fit runs after full assembly. This covers visualize/paraview/rocking/mosaicity runs, exports, and their replots via `figures.render_volume_layer` — no changes needed there.

**Steps:**

- [ ] Read `dfxm/common/render.py` lines 1–92 and `dfxm/stages/strain.py` lines 340–415. Append to `tests/test_figure_layout.py` (reuses the `_box_inches` helper added in Task A2):

```python
def test_layer_figure_fixed_scale_equal_boxes_across_decoration_loads():
    import numpy as np

    from dfxm.common import render
    from dfxm.common.plotting import PlotStyle

    layer = np.random.default_rng(1).random((10, 20))
    style = PlotStyle(scale_um_per_cm=50.0, figure_width="single", tickfmt_raw="scientific")
    boxes = []
    for vmax, title in ((1.0e-4, "short"), (123456.0, "a much longer two-line\ntitle text here")):
        fig, ax, _ = render.layer_figure(
            layer * vmax, 0.0, vmax, "gray", 200.0, 100.0, title, "I (a.u.)",
            style=style, group="raw",
        )
        boxes.append(_box_inches(fig, ax))
    target_w, target_h = 200.0 / 50.0 / 2.54, 100.0 / 50.0 / 2.54
    for w, h in boxes:
        assert abs(w - target_w) <= 0.05 and abs(h - target_h) <= 0.05


def test_build_strain_map_fixed_scale_box():
    import numpy as np

    from dfxm.common.plotting import PlotStyle
    from dfxm.stages.strain import build_strain_map

    strain = np.random.default_rng(2).standard_normal((50, 100)) * 1e-4
    style = PlotStyle(scale_um_per_cm=10.0)
    fig = build_strain_map(strain, 1.0, 1.0, None, (None, None), style=style)
    w, h = _box_inches(fig, fig.axes[0])
    assert abs(w - 100.0 / 10.0 / 2.54) <= 0.05
    assert abs(h - 50.0 / 10.0 / 2.54) <= 0.05
```

- [ ] Run `python3 -m pytest -q tests/test_figure_layout.py` — the two new tests fail (boxes differ / wrong size).
- [ ] Implement `render.py`: extend the `from .plotting import (...)` block with `fit_axes_to_box, fixed_scale_box`. In `layer_figure`, replace the figsize line and the scale-bar call, and add the fit before `return`:

```python
    st = style if style is not None else PlotStyle(scale_bar_color="black", colorbar_fraction=0.046)
    box = fixed_scale_box(st, ext_x, ext_y) if style is not None else None
    if box is not None:
        figsize = (box[0] + 1.5, box[1] + 1.5)  # headroom; fit_axes_to_box converges regardless
    else:
        figsize = (figure_size(st, ext_x, ext_y) or (12, 10)) if style is not None else (12, 10)
```

…(unchanged body)…

```python
    if st.scale_bar:
        draw_scale_bar(
            ax,
            st.scale_bar_length_um,
            style=st,
            fixed_scale_um_per_cm=(box[2] if box is not None else None),
        )
    apply_text_scale(ax, st)
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])
    return fig, ax, im
```

- [ ] Implement `strain.py` `build_strain_map` the same way: import `fit_axes_to_box, fixed_scale_box` (extend the existing `from ..common.plotting import (...)` block); compute `box = fixed_scale_box(style, nx * px, ny * py)` right before the figsize expression; `figsize = (box[0] + 1.5, box[1] + 1.5)` when `box is not None`, else the existing expression; pass `fixed_scale_um_per_cm=(box[2] if box is not None else None)` to the `draw_scale_bar` call in the styled branch; add `if box is not None: fit_axes_to_box(fig, ax, box[0], box[1])` immediately before `return fig`.
- [ ] Run `python3 -m pytest -q tests/test_figure_layout.py tests/test_stage_strain.py tests/test_export_fidelity.py tests/test_figures_replot.py` — expect pass; then full `python3 -m pytest -q`.
- [ ] Update `docs/Codebase.md`: `render.py layer_figure` and `strain.py build_strain_map` notes (fixed-scale fitting, bar in point-exact mode).
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(render,strain): fixed-scale fitting for shared layer figures + strain diagnostic"`.

## Task A5: wire `build_slice_figure` (slices run, replot, export)

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/stages/slices.py` (`build_slice_figure`, ~lines 786–830)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_stage_slices.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`, `/home/albert/Desktop/dfxm_pipeline/docs/Usage.md`

**Interfaces:**
- Consumes: `fixed_scale_box`, `fit_axes_to_box` (extend the existing `from ..common.plotting import (...)` block in slices.py), `draw_scale_bar(..., fixed_scale_um_per_cm=...)`.
- Produces: `build_slice_figure(prep, sl, slice2d, u_um, v_um, *, offset_um, style=None) -> Figure` (signature unchanged) — fixed-scale when the style knob is set; legacy (`style=None`) path untouched. `_rebuild_plane_figure`/`render_replot`/export inherit automatically.

**Steps:**

- [ ] Read `dfxm/stages/slices.py` lines 783–840 and `tests/test_stage_slices.py` lines 210–260 (the `_prep()` helper). Append to `tests/test_stage_slices.py`:

```python
def _box_inches(fig, ax):
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    if not hasattr(fig.canvas, "get_renderer"):
        FigureCanvasAgg(fig)
    fig.canvas.draw()
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    return bb.width / fig.dpi, bb.height / fig.dpi


def test_build_slice_figure_fixed_scale_equal_boxes_across_colorbar_text():
    u = np.linspace(0.0, 200.0, 21)
    v = np.linspace(0.0, 100.0, 11)
    s2d = np.random.default_rng(3).random((11, 21))
    style = PlotStyle(scale_um_per_cm=50.0, figure_width="single", tickfmt_strain="scientific")
    boxes = []
    for vmin, vmax, group in ((-1.0e-4, 1.0e-4, "strain"), (-1.0, 1.0, None)):
        prep = dict(_prep(), vmin=vmin, vmax=vmax, group=group)
        fig = SL.build_slice_figure(prep, {"name": "p"}, s2d, u, v, offset_um=None, style=style)
        boxes.append(_box_inches(fig, fig.axes[0]))
    tw, th = 200.0 / 50.0 / 2.54, 100.0 / 50.0 / 2.54
    for w, h in boxes:
        assert abs(w - tw) <= 0.05 and abs(h - th) <= 0.05
```

- [ ] Run `python3 -m pytest -q tests/test_stage_slices.py -k fixed_scale` — expect failure (boxes at figure_width sizing).
- [ ] Implement in `build_slice_figure` (mirror Task A4's pattern):

```python
    if use_legacy:
        figsize = (12, 10)
        box = None
    else:
        ext_u = float(u_um[-1] - u_um[0])
        ext_v = float(v_um[-1] - v_um[0])
        box = fixed_scale_box(st, ext_u, ext_v)
        if box is not None:
            figsize = (box[0] + 1.5, box[1] + 1.5)
        else:
            figsize = figure_size(st, ext_u, ext_v) or (12, 10)
```

…and at the tail of the function:

```python
    if st.scale_bar:
        draw_scale_bar(
            ax,
            st.scale_bar_length_um,
            style=st,
            fixed_scale_um_per_cm=(box[2] if box is not None else None),
        )
    if not use_legacy:
        apply_text_scale(ax, st)
    if box is not None:
        fit_axes_to_box(fig, ax, box[0], box[1])
    return fig
```

(Keep the `if st.colorbar:` line above the bar exactly as-is.)
- [ ] Run `python3 -m pytest -q tests/test_stage_slices.py tests/test_gui_slice_replot.py tests/test_export_fidelity.py`, then full `python3 -m pytest -q` — expect green.
- [ ] Docs: `docs/Codebase.md` — `build_slice_figure` note (fixed-scale fitting; replot/export inherit). `docs/Usage.md` — in the slices stage section, one sentence: slice PNGs honour the publication-style "Scale (µm/cm)" once set (full knob doc lands with the GUI task).
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(slices): fixed-scale fitting for slice figures (run/replot/export inherit)"`.

## Task A6: wire the profiles standalone overview; pin the companion

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/stages/profiles.py` (`_draw_reference_image` ~line 523, `render_single` ~line 716; `build_companion_figure` stays untouched)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_stage_profiles.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`, `/home/albert/Desktop/dfxm_pipeline/docs/Usage.md`

**Interfaces:**
- Produces: `_draw_reference_image(ax, plane2d, u_um, v_um, attrs, line_color, geom=None, title=None, style=None, fixed_scale_um_per_cm=None)` — new trailing kwarg forwarded to `draw_scale_bar` in the styled branch only.
- Produces: `render_single(ref, geom, line_color, out_png, header, dpi, style=None)` (signature unchanged) — fitted in fixed mode. `build_companion_figure` does NOT pass the kwarg → companion map panel keeps today's geometry byte-identically even when the knob is set.
- Consumes: `fixed_scale_box`, `fit_axes_to_box` (extend profiles.py's `from ..common.plotting import (...)` block).

**Steps:**

- [ ] Read `dfxm/stages/profiles.py` lines 510–645 and 716–745; read `tests/test_stage_profiles.py` far enough to find its figure-level tests and fixtures (grep `build_companion_figure` in it). Append to `tests/test_stage_profiles.py` (adapt the `ref`/`geom` construction to the file's existing helpers if one already builds them — otherwise use this standalone version):

```python
def _bar_rect(ax):
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox

    box = next(a for a in ax.artists if isinstance(a, AnchoredOffsetbox))
    stack = [box.get_child()]
    while stack:
        a = stack.pop()
        if isinstance(a, AuxTransformBox):
            return a.get_children()[0]
        if hasattr(a, "get_children"):
            stack.extend(a.get_children())
    raise AssertionError("no bar rectangle found")


def _mini_ref_geom():
    import numpy as np

    u = np.linspace(0.0, 200.0, 21)
    v = np.linspace(0.0, 100.0, 11)
    plane = np.random.default_rng(4).random((11, 21))
    attrs = {
        "vmin": 0.0, "vmax": 1.0, "cmap": "gray", "title": "T",
        "cbar_label": "I (a.u.)", "kind": "raw_sum", "source_volume": "x.h5",
    }
    geom = {
        "start": np.array([10.0, 10.0]), "end": np.array([150.0, 80.0]),
        "width": 1, "band_offsets": np.array([0.0]), "phat": np.array([0.0, 1.0]),
        "distance": np.linspace(0.0, 100.0, 50), "L": 100.0,
    }
    return (plane, u, v, attrs, "ref"), geom


def test_render_single_overview_fits_fixed_scale(tmp_path):
    from dfxm.common.plotting import PlotStyle
    from dfxm.stages.profiles import render_single

    ref, geom = _mini_ref_geom()
    style = PlotStyle(scale_um_per_cm=50.0)
    out = str(tmp_path / "ov.png")
    render_single(ref, geom, "red", out, "hdr", 100, style=style)
    assert os.path.exists(out)


def test_companion_map_panel_bar_geometry_unchanged_by_scale_knob():
    from dfxm.common.plotting import PlotStyle
    from dfxm.stages.profiles import build_companion_figure

    ref, geom = _mini_ref_geom()
    fld = {
        "vid": "raw_sum", "attrs": ref[3], "plane": ref[0],
        "value_mean": geom["distance"] * 0.0, "value_std": None,
    }
    style = PlotStyle(scale_um_per_cm=50.0, scale_bar_thickness_pt=3.0)
    fig = build_companion_figure(ref, [fld], geom, "red", style=style)
    ax_img = fig.axes[0]
    yr = ax_img.get_ylim()[1] - ax_img.get_ylim()[0]
    # companion is NOT fitted: bar height stays the data-fraction geometry
    assert _bar_rect(ax_img).get_height() == pytest.approx(abs(yr) * 0.004 * 3.0)
```

- [ ] Run `python3 -m pytest -q tests/test_stage_profiles.py -k "render_single_overview or companion_map_panel"` — companion test may already pass (that is the pinned behaviour); the overview test must pass only after wiring — first confirm current failure/pass state, then wire.
- [ ] Implement: add `fixed_scale_box, fit_axes_to_box` to profiles.py's plotting import block. In `_draw_reference_image`, add the kwarg and change only the styled bar call:

```python
def _draw_reference_image(
    ax, plane2d, u_um, v_um, attrs, line_color, geom=None, title=None, style=None,
    fixed_scale_um_per_cm=None,
):
```

```python
    if style is None:
        _scale_bar(ax)  # legacy look, pinned
    elif style.scale_bar:
        draw_scale_bar(
            ax, style.scale_bar_length_um, style=style,
            fixed_scale_um_per_cm=fixed_scale_um_per_cm,
        )
```

In `render_single`, before the figure is created:

```python
    plane, u_um, v_um, attrs, label = ref
    ext_u = float(u_um[-1] - u_um[0])
    ext_v = float(v_um[-1] - v_um[0])
    box = fixed_scale_box(style, ext_u, ext_v)
    figsize = (box[0] + 1.5, box[1] + 1.5) if box is not None else (11, 9)
    fig = styled_figure(figsize, styled=style is not None)
```

Pass `fixed_scale_um_per_cm=(box[2] if box is not None else None)` in its `_draw_reference_image(...)` call, and add `if box is not None: fit_axes_to_box(fig, ax, box[0], box[1])` immediately before `fig.savefig(...)`. Do **not** touch `build_companion_figure` (it calls `_draw_reference_image` without the kwarg → default `None`).
- [ ] Run `python3 -m pytest -q tests/test_stage_profiles.py`, then full `python3 -m pytest -q` — green.
- [ ] Docs: `docs/Codebase.md` — `_draw_reference_image`/`render_single` notes (overview fitted, companion pinned). `docs/Usage.md` — profiles section: overview maps honour the fixed scale; the companion figure does not (by design).
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(profiles): fixed-scale overview maps; companion map panel pinned"`.

## Task A7: GUI "Scale (µm/cm)" field + smoke step + user docs — **Phase A gate**

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/gui/widgets/export_dialog.py` (`StyleControls`: `_build_controls` Figure section ~line 470, `sync_from_style` ~line 134, `_all_widgets` ~line 155)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/gui_smoke.py` (new step `[29]`, inserted before `print("\nGUI SMOKE PASSED")`)
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Usage.md`, `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Produces: `StyleControls._w_scale_umcm: QLineEdit` — blank = off; defensive parse writes `self._style.scale_um_per_cm` (float > 0 or `None`) and emits `changed`. Persistence is automatic: the global style round-trips through `style_to_json`/`style_from_json` in `gui/main_window.py` (`_save_plot_style`, QSettings key `plot_style`) — no new persistence code.
- Consumes: `fixed_scale` from `dfxm.common.plotting` (for `sync_from_style`).

**Steps:**

- [ ] Read `gui/widgets/export_dialog.py` lines 80–200 and 470–515. In `_build_controls`, after the "Figure width" row, add:

```python
        self._w_scale_umcm = QLineEdit()
        self._w_scale_umcm.setPlaceholderText("(blank = off)")
        _sv = fixed_scale(s)
        if _sv is not None:
            self._w_scale_umcm.setText(f"{_sv:g}")
        self._w_scale_umcm.setToolTip(
            "Fixed physical scale for map figures: µm of data per cm of page. When set, every "
            "map's data box is fitted so the printed scale (and the scale bar) is identical "
            "across figures; Figure width is ignored for maps (trace figures keep it). "
            "Blank turns it off. For identical bars across different crops also set an "
            "explicit Bar length."
        )
        self._w_scale_umcm.textChanged.connect(self._on_scale_umcm)
        form.addRow("Scale (µm/cm)", self._w_scale_umcm)
```

Add the slot after `_on_bar_auto_toggled`:

```python
    def _on_scale_umcm(self, text: str) -> None:
        t = text.strip()
        val: float | None = None
        if t:
            try:
                val = float(t)
            except ValueError:
                val = None
            else:
                if val <= 0:
                    val = None
        self._style.scale_um_per_cm = val
        self._emit()
```

Add `from dfxm.common.plotting import CMAP_CHOICES, PlotStyle, fixed_scale` to the imports; add `self._w_scale_umcm` to `_all_widgets`; add to `sync_from_style` (next to the fig-width sync):

```python
        _sv = fixed_scale(s)
        self._w_scale_umcm.setText(f"{_sv:g}" if _sv is not None else "")
```

- [ ] Add smoke step `[29]` to `tests/gui_smoke.py`, immediately after step `[28]`'s `print` and before `print("\nGUI SMOKE PASSED")`:

```python
    # [29] StyleControls: Scale (µm/cm) field parses defensively and mutates the style.
    from dfxm.common.plotting import PlotStyle as _PS29
    from gui.widgets.export_dialog import StyleControls as _SC29

    _st29 = _PS29()
    _sc29 = _SC29(_st29)
    _sc29._w_scale_umcm.setText("50")
    assert _st29.scale_um_per_cm == 50.0
    _sc29._w_scale_umcm.setText("junk")
    assert _st29.scale_um_per_cm is None
    _sc29._w_scale_umcm.setText("-2")
    assert _st29.scale_um_per_cm is None
    _sc29._w_scale_umcm.setText("")
    assert _st29.scale_um_per_cm is None
    print("[29] StyleControls Scale (µm/cm) field mutates the style defensively")
```

- [ ] Run `python3 tests/gui_smoke.py` — expect all 29 steps to pass (before implementing, step [29] fails with `AttributeError: _w_scale_umcm` — run once before wiring to confirm the failure, then after).
- [ ] Docs (same commit): `docs/Usage.md` publication-style section (grep for "Publication style"): document the knob — what µm/cm means, that a value fixes the printed data scale across all map figures (slices, per-layer maps, strain diagnostic, profiles overviews; NOT the companion or trace figures), that Figure width is then ignored for maps, the 30-in clamp, and the **identical-bar recipe**: set an explicit Bar length (e.g. 50 µm) because auto length still picks ~15% of each crop's extent. `docs/Codebase.md`: `StyleControls` widget list.
- [ ] Full gate: `python3 -m pytest -q` && `ruff check . && ruff format .` && `python3 tests/gui_smoke.py` — all green.
- [ ] `git add -A && git commit -m "feat(gui): Scale (µm/cm) publication-style field + smoke [29] + docs (Phase A complete)"`.

---

# Phase B — Pin planes + planes-first replot selection

## Task B1: `build_pinned_spec` shared core; `tools/pin_slice.py` delegates

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/stages/slices.py` (add `_find_slice_group` + `build_pinned_spec` after `write_volume_group`, ~line 869)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tools/pin_slice.py` (replace `_find_slice_group`/`pin_spec` bodies with delegation)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_stage_slices.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Produces (Qt-free, in `dfxm/stages/slices.py`):
  - `_find_slice_group(f, slice_name, volume=None) -> tuple[str, h5py.Group]` — raises `StageUserError` when not found.
  - `build_pinned_spec(h5_path: str, slice_name: str, offsets: list[float], *, volume: str | None = None) -> list[dict]` — one single-plane spec per requested offset, snapped to the nearest stored plane, duplicates collapsed; geometry (`normal`/`origin`/`up`/`half_u`/`half_v`/`du`/`dv`) read byte-exact from the stored slice-group attrs; each spec has `sweep_start_um == sweep_stop_um == matched offset`; raises `StageUserError` (with hint) on unreadable file / unknown slice.
- Consumes: `StageUserError` (already imported in slices.py), `h5py`, `numpy`.
- `tools/pin_slice.py` keeps its CLI (`h5_path slice_name --offset [--name] [--volume]`) and stdout JSON list; its core becomes a call to `dfxm.stages.slices.build_pinned_spec`.

**Steps:**

- [ ] Read `tools/pin_slice.py` in full and `dfxm/stages/slices.py` lines 840–900. Append to `tests/test_stage_slices.py` (add `from dfxm.common.errors import StageUserError` and `import subprocess, sys` to the imports):

```python
# -- pinned planes ------------------------------------------------------------
def test_build_pinned_spec_snaps_and_reproduces_sweep_plane(tmp_path):
    proc, raw = _setup(tmp_path)
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "slices_json": (
            '[{"name":"zsweep","normal":[0,0,1],"origin":[0,0,0],'
            '"extent":"auto","sweep_step_um":1.0}]'
        ),
        "output_dir": str(tmp_path / "sl"),
        "save_png": False,
    }
    res = SL.run(dict(params))
    with h5py.File(res.output_h5, "r") as f:
        sweep_offsets = f["strain/zsweep/offsets_um"][:]
        sweep_stack = f["strain/zsweep/slices"][:]
        attrs = dict(f["strain/zsweep"].attrs)
    target = float(sweep_offsets[1]) + 0.2  # off-grid: must snap to plane 1
    specs = SL.build_pinned_spec(res.output_h5, "zsweep", [target, target])
    assert len(specs) == 1  # duplicate snap collapsed
    spec = specs[0]
    assert spec["sweep_start_um"] == spec["sweep_stop_um"]
    assert spec["sweep_start_um"] == pytest.approx(float(sweep_offsets[1]))
    for key in ("normal", "origin", "up"):
        np.testing.assert_allclose(spec[key], np.asarray(attrs[key], float))
    for key in ("half_u", "half_v", "du", "dv"):
        assert spec[key] == pytest.approx(float(attrs[key]))
    # golden: a run on the pinned spec reproduces the sweep's plane 1 exactly
    res2 = SL.run({**params, "slices_json": json.dumps(specs), "output_dir": str(tmp_path / "pin")})
    with h5py.File(res2.output_h5, "r") as f:
        pinned = f[f"strain/{spec['name']}/slices"][:]
    np.testing.assert_array_equal(pinned[0], sweep_stack[1])


def test_build_pinned_spec_unknown_slice_raises_user_error(tmp_path):
    p = str(tmp_path / "s.h5")
    with h5py.File(p, "w") as f:
        f.create_group("strain")
    with pytest.raises(StageUserError):
        SL.build_pinned_spec(p, "nope", [0.0])


def test_pin_slice_cli_delegates_to_core(tmp_path):
    p = str(tmp_path / "s.h5")
    with h5py.File(p, "w") as f:
        sg = f.create_group("strain").create_group("oblique")
        sg.create_dataset("offsets_um", data=np.array([0.0, 1.0, 2.0]))
        sg.attrs["normal"] = [0.0, 0.0, 1.0]
        sg.attrs["origin"] = [0.0, 0.0, 0.0]
        sg.attrs["up"] = [0.0, 1.0, 0.0]
        for k, v in (("half_u", 4.0), ("half_v", 3.0), ("du", 0.2), ("dv", 0.2),
                     ("sweep_step_um", 1.0)):
            sg.attrs[k] = v
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run(
        [sys.executable, os.path.join(root, "tools", "pin_slice.py"), p, "oblique",
         "--offset", "1.4"],
        capture_output=True, text=True, cwd=root,
    )
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert len(out) == 1 and out[0]["sweep_start_um"] == pytest.approx(1.0)
```

- [ ] Run `python3 -m pytest -q tests/test_stage_slices.py -k pinned` — expect `AttributeError: build_pinned_spec`.
- [ ] Implement in `dfxm/stages/slices.py` (after `write_volume_group`):

```python
# -----------------------------------------------------------------------------
# Pinned planes (shared by tools/pin_slice.py, the Pin planes… dialog and run())
# -----------------------------------------------------------------------------
def _find_slice_group(f, slice_name, volume=None):
    """Return (volume_id, slice_group) for the first volume holding *slice_name*.

    Slice geometry is identical across volumes, so any volume that carries the
    slice works; *volume* forces a specific one.
    """
    vids = [volume] if volume else list(f.keys())
    for vid in vids:
        if vid not in f:
            continue
        g = f[vid]
        if slice_name in g and "offsets_um" in g[slice_name]:
            return vid, g[slice_name]
    raise StageUserError(
        f"slice {slice_name!r} not found in any volume group of the file",
        hint=f"volumes present: {', '.join(f.keys()) or '(none)'} — pick a slice "
        "name from a swept oblique_slices.h5.",
    )


def build_pinned_spec(h5_path, slice_name, offsets, *, volume=None) -> list[dict]:
    """Pinned single-plane spec dicts for *slice_name*, snapped to stored planes.

    Geometry (normal/origin/up/half_u/half_v/du/dv) is read byte-exact off the
    stored slice-group attrs, so each pinned plane reproduces the sweep's plane
    exactly. Every requested offset snaps to the nearest stored plane; snaps
    landing on the same plane are collapsed. Raises StageUserError for an
    unreadable file or unknown slice name.
    """
    try:
        fh = h5py.File(h5_path, "r")
    except OSError as exc:
        raise StageUserError(
            f"cannot read {h5_path!r}: {exc}",
            hint="Point at an oblique_slices.h5 written by a slices sweep run.",
        ) from exc
    with fh as f:
        _vid, sg = _find_slice_group(f, slice_name, volume)
        stored = sg["offsets_um"][:].astype(np.float64)
        a = dict(sg.attrs)
    specs, seen = [], set()
    for off in offsets:
        idx = int(np.argmin(np.abs(stored - float(off))))
        if idx in seen:
            continue
        seen.add(idx)
        matched = float(stored[idx])
        specs.append(
            {
                "name": f"{slice_name}_pin_{matched:+.2f}um",
                "normal": np.asarray(a["normal"], np.float64).tolist(),
                "origin": np.asarray(a["origin"], np.float64).tolist(),
                "up": np.asarray(a["up"], np.float64).tolist(),
                "half_u": float(a["half_u"]),
                "half_v": float(a["half_v"]),
                "du": float(a["du"]),
                "dv": float(a["dv"]),
                "sweep_step_um": float(a.get("sweep_step_um") or 1.0) or 1.0,
                "sweep_start_um": matched,
                "sweep_stop_um": matched,
            }
        )
    return specs
```

- [ ] Rewrite `tools/pin_slice.py`'s core: delete its `_find_slice_group` and the body of `pin_spec`; keep the module docstring and CLI. New content below the docstring:

```python
from __future__ import annotations

import argparse
import json
import sys

from dfxm.common.errors import StageUserError
from dfxm.stages.slices import build_pinned_spec


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("h5_path", help="path to a swept oblique_slices.h5")
    ap.add_argument("slice_name", help="slice name (the sweep's 'name', e.g. oblique_full)")
    ap.add_argument("--offset", type=float, required=True, help="desired offset along normal, µm")
    ap.add_argument("--name", default=None, help="name for the pinned plane (default: auto)")
    ap.add_argument("--volume", default=None, help="read geometry from this volume group only")
    args = ap.parse_args(argv)
    try:
        spec = build_pinned_spec(args.h5_path, args.slice_name, [args.offset],
                                 volume=args.volume)[0]
    except StageUserError as exc:
        raise SystemExit(f"{exc} ({exc.hint})") from exc
    if args.name:
        spec["name"] = args.name
    print(
        f"requested {args.offset:+.2f} µm -> nearest stored plane "
        f"{spec['sweep_start_um']:+.2f} µm",
        file=sys.stderr,
    )
    print(json.dumps([spec], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
```

(First check `dfxm/common/errors.py` for the exact `StageUserError.hint` attribute name and adjust the f-string if it differs.)
- [ ] Run `python3 -m pytest -q tests/test_stage_slices.py`, then full `python3 -m pytest -q` — green.
- [ ] `docs/Codebase.md`: add `build_pinned_spec`/`_find_slice_group` under `dfxm/stages/slices.py`; update the `tools/pin_slice.py` entry (now a thin CLI over the core).
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(slices): build_pinned_spec shared core; tools/pin_slice.py delegates"`.

## Task B2: `use_pinned` + `pinned_slices_json` params, run() routing, clobber guard

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/dfxm/stages/slices.py` (STAGE params after `slices_json` ~line 405; `run()` routing at the `slices = json.loads(p["slices_json"])` block ~line 976 and `out_h5 = ...` ~line 1006; `output_h5_name` help text)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_stage_slices.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Usage.md`, `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Produces: two new `Param`s on the slices `STAGE`:
  - `Param("use_pinned", ParamType.BOOL, "Run pinned planes only", default=False, help=...)` — placed directly after `slices_json` (non-advanced, so the toggle is visible next to the sweep field).
  - `Param("pinned_slices_json", ParamType.TEXT, "Pinned planes (JSON)", default="", advanced=True, group="Pinned planes", help=...)`.
- Produces: `run()` behaviour — toggle on: parse `pinned_slices_json` (StageUserError with hint when empty/invalid), loud `PINNED RUN` note in `result.notes` + progress log, and while `output_h5_name` still equals the stage default (`"oblique_slices.h5"`) the output filename becomes `"oblique_slices_pinned.h5"` (clobber guard; a user-edited name is respected). Toggle off: byte-identical to today.
- Consumes: `STAGE.defaults()`, `StageUserError`, existing `json` import. Form persistence of both fields is automatic (`FormStateStore`); no GUI code in this task.

**Steps:**

- [ ] Read `dfxm/stages/slices.py` lines 395–435 and 952–1010. Append to `tests/test_stage_slices.py`:

```python
def _pinned_params(proc, raw, out):
    return {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "slices_json": "[]",  # would raise on the sweep path — proves routing skips it
        "use_pinned": True,
        "pinned_slices_json": (
            '[{"name":"pin","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
            '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
        ),
        "output_dir": str(out),
        "save_png": False,
    }


def test_run_use_pinned_routes_and_renames_output(tmp_path):
    proc, raw = _setup(tmp_path)
    res = SL.run(_pinned_params(proc, raw, tmp_path / "sl"))
    assert res.output_h5 and res.output_h5.endswith("oblique_slices_pinned.h5")
    assert any("PINNED RUN" in n for n in res.notes)
    assert res.slice_names == ["pin"]


def test_run_use_pinned_respects_user_edited_name(tmp_path):
    proc, raw = _setup(tmp_path)
    p = _pinned_params(proc, raw, tmp_path / "sl")
    p["output_h5_name"] = "custom.h5"
    res = SL.run(p)
    assert res.output_h5.endswith("custom.h5")


def test_run_use_pinned_empty_or_invalid_raises_user_error(tmp_path):
    proc, raw = _setup(tmp_path)
    for bad in ("", "   ", "{not json", "[]"):
        p = _pinned_params(proc, raw, tmp_path / "sl")
        p["pinned_slices_json"] = bad
        with pytest.raises(StageUserError, match="[Pp]inned"):
            SL.run(p)
```

- [ ] Run `python3 -m pytest -q tests/test_stage_slices.py -k use_pinned` — expect failures (unknown param ignored → `"[]"` sweep raises the wrong error, no rename).
- [ ] Implement the two params (insert directly after the `slices_json` Param, before `output_dir`):

```python
        Param(
            "use_pinned",
            ParamType.BOOL,
            "Run pinned planes only",
            default=False,
            help=(
                "Render only the planes in 'Pinned planes (JSON)' instead of the full sweep — "
                "fast re-computation of a few interesting planes. The sweep in 'Slices (JSON)' "
                "is kept untouched and ignored while this is on; while on, the default output "
                "filename becomes oblique_slices_pinned.h5 so the sweep file is never "
                "overwritten. Untick to run the full sweep again."
            ),
        ),
        Param(
            "pinned_slices_json",
            ParamType.TEXT,
            "Pinned planes (JSON)",
            default="",
            advanced=True,
            group="Pinned planes",
            help=(
                "JSON list of pinned single-plane specs, normally written by the Pin planes… "
                "dialog (exact stored sweep geometry, snapped to stored planes). Only used "
                "when 'Run pinned planes only' is ticked."
            ),
        ),
```

- [ ] Implement the routing in `run()`. Replace the block

```python
    slices = json.loads(p["slices_json"])
    if not isinstance(slices, list) or not slices:
        raise StageUserError(...)
```

with:

```python
    if bool(p["use_pinned"]):
        raw_pinned = (p["pinned_slices_json"] or "").strip()
        try:
            slices = json.loads(raw_pinned) if raw_pinned else []
        except json.JSONDecodeError as exc:
            raise StageUserError(
                f"Pinned planes JSON is not valid JSON: {exc}",
                hint=(
                    "Open Pin planes… to regenerate the pinned list, or untick "
                    "'Run pinned planes only' to run the full sweep."
                ),
            ) from exc
        if not isinstance(slices, list) or not slices:
            raise StageUserError(
                "'Run pinned planes only' is on but the pinned planes list is empty",
                hint=(
                    "Open Pin planes… to pick planes, or untick "
                    "'Run pinned planes only' to run the full sweep."
                ),
            )
        msg = (
            f"PINNED RUN: rendering {len(slices)} pinned plane(s); "
            "the sweep in slices_json is ignored"
        )
        result.notes.append(msg)
        progress(0.03, msg)
    else:
        slices = json.loads(p["slices_json"])
        if not isinstance(slices, list) or not slices:
            raise StageUserError(
                "slices_json must be a non-empty JSON list of slice specs",
                hint=(
                    "Provide a JSON list of plane specs — the field's default "
                    "shows the format; 'extent': 'auto' fits the plane "
                    "automatically."
                ),
            )
```

(Read the existing block first — the `hint=` string contains em-dashes; keep it byte-identical inside the `else`.) Then replace `out_h5 = os.path.join(out_dir, p["output_h5_name"])` with:

```python
    h5_name = p["output_h5_name"]
    if bool(p["use_pinned"]) and h5_name == STAGE.defaults()["output_h5_name"]:
        h5_name = "oblique_slices_pinned.h5"
        result.notes.append(
            "pinned run: output filename switched to oblique_slices_pinned.h5 "
            "so the sweep file profiles reads is not overwritten"
        )
    out_h5 = os.path.join(out_dir, h5_name)
```

Also append one sentence to the `output_h5_name` Param help: `"While 'Run pinned planes only' is on, this default is replaced by oblique_slices_pinned.h5 (an edited name is respected)."`
- [ ] Run `python3 -m pytest -q tests/test_stage_slices.py tests/test_param_metadata.py`, then full `python3 -m pytest -q` — green.
- [ ] Docs (same commit): `docs/Usage.md` slices stage section — new "Pinned planes (fast re-runs)" subsection: workflow (sweep once → Pin planes… → tick toggle → Run → `oblique_slices_pinned.h5`), and update the existing "Pinning one plane from a sweep" text to mention the toggle supersedes hand-copying JSON. `docs/Codebase.md`: new params + routing + clobber guard under `slices.py`; add `oblique_slices_pinned.h5` to the data-flow table.
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(slices): use_pinned/pinned_slices_json params, run routing + clobber guard"`.

## Task B3: planes-first selection model (pure logic + unit tests)

**Files:**
- Create: `/home/albert/Desktop/dfxm_pipeline/gui/widgets/plane_selection_model.py` (PySide6-free)
- Create: `/home/albert/Desktop/dfxm_pipeline/tests/test_plane_selection_model.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Produces (all in `gui/widgets/plane_selection_model.py`; consumed by Tasks B4–B6):
  - `@dataclass(frozen=True) PlaneRow: key: object; section: str; number: int; offset: float | None; label: str`
  - `build_slice_rows(entries) -> list[PlaneRow]` — entries are `dfxm.stages.slices.ReplotEntry`-shaped (`volume_id`, `slice_name`, `n_planes`, `offsets_um`); one row per `(slice_name, plane_idx)` union across volumes, `key=(slice_name, plane_idx)`, `section=slice_name`, label `f"p{k:03d}  {off:+.2f} µm"`.
  - `build_layer_rows(groups) -> list[PlaneRow]` — groups are `dfxm.common.figures.ReplotGroup`-shaped (`key`, `item_labels`); one row per layer index union across groups, `key=z`, `section=""`, offset parsed from a `Z=<float>` fragment in the label when present.
  - `parse_tokens(text) -> list[tuple[str, float]]` — comma-split; bare unsigned integers → `("number", v)`; signed/decimal floats → `("offset", v)`; unparseable → `("invalid", 0.0)` (matches nothing).
  - `filter_rows(rows, text) -> list[PlaneRow]` — blank text → all rows; else the union of token matches. Number tokens match `row.number` in every section; offset tokens match, per section, the nearest row by offset when within half the section's median sweep step (single-plane sections: nearest matches unconditionally). Narrows only — never selects.
  - `slice_selections(entries, checked_plane_keys, checked_vids) -> tuple[list[tuple[str, str, list[int]]], list[str]]` — cartesian product of checked planes × checked volume ids in `slices.render_replot` selection format, plus human-readable skip reasons for missing combos.
  - `layer_selections(groups, checked_layers, checked_keys) -> tuple[list[tuple[object, list[int]]], list[str]]` — same for the generic `render_replot(h5, [(key, idxs)], ...)` format.

**Steps:**

- [ ] Create `tests/test_plane_selection_model.py`:

```python
"""Tests for the Qt-free planes-first selection model."""

from __future__ import annotations

import pytest

from gui.widgets.plane_selection_model import (
    build_layer_rows,
    build_slice_rows,
    filter_rows,
    layer_selections,
    parse_tokens,
    slice_selections,
)


class _E:  # duck-typed dfxm.stages.slices.ReplotEntry
    def __init__(self, vid, sname, offsets):
        self.volume_id = vid
        self.slice_name = sname
        self.offsets_um = list(offsets)
        self.n_planes = len(self.offsets_um)


class _G:  # duck-typed dfxm.common.figures.ReplotGroup
    def __init__(self, key, labels):
        self.key = key
        self.item_labels = list(labels)


def _entries():
    return [
        _E("strain", "oblique", [-2.0, 0.0, 2.0]),
        _E("mosa_com_chi", "oblique", [-2.0, 0.0, 2.0]),
        _E("strain", "zsweep", [0.0, 5.0]),
    ]


def test_build_slice_rows_unions_across_volumes():
    rows = build_slice_rows(_entries())
    assert len(rows) == 5  # 3 oblique planes (once, not per volume) + 2 zsweep
    keys = {r.key for r in rows}
    assert ("oblique", 1) in keys and ("zsweep", 1) in keys
    r = next(r for r in rows if r.key == ("oblique", 2))
    assert r.section == "oblique" and r.number == 2 and r.offset == 2.0
    assert r.label == "p002  +2.00 µm"


def test_build_layer_rows_parses_z_and_unions():
    rows = build_layer_rows(
        [_G("sum_intensity", ["layer 0  (Z=0.00 µm)", "layer 1  (Z=2.00 µm)"]),
         _G("specific_frame", ["layer 0  (Z=0.00 µm)"])]
    )
    assert [r.number for r in rows] == [0, 1]
    assert rows[1].offset == 2.0 and rows[0].section == ""


def test_parse_tokens_classification():
    assert parse_tokens("118, 7") == [("number", 118.0), ("number", 7.0)]
    assert parse_tokens("-3.7, +4, 2.0") == [("offset", -3.7), ("offset", 4.0), ("offset", 2.0)]
    assert parse_tokens("") == []
    assert parse_tokens("zz")[0][0] == "invalid"


def test_filter_rows_number_offset_and_no_match():
    rows = build_slice_rows(_entries())
    assert filter_rows(rows, "") == rows  # blank = everything
    vis = filter_rows(rows, "1")  # plane number 1 in BOTH sections
    assert {r.key for r in vis} == {("oblique", 1), ("zsweep", 1)}
    vis = filter_rows(rows, "1.8")  # nearest oblique plane 2.0 (step 2 -> tol 1);
    assert ("oblique", 2) in {r.key for r in vis}  # zsweep nearest 0.0 is off by 1.8 > tol 2.5? no:
    # zsweep offsets [0,5]: step 5, tol 2.5 -> nearest 0.0 within tol, also matches
    assert ("zsweep", 0) in {r.key for r in vis}
    assert filter_rows(rows, "99") == []  # no plane 99 anywhere
    assert filter_rows(rows, "zz") == []  # invalid token matches nothing


def test_slice_selections_product_and_skips():
    entries = _entries()
    sels, skipped = slice_selections(
        entries,
        [("oblique", 1), ("zsweep", 1), ("zsweep", 7)],
        ["strain", "mosa_com_chi"],
    )
    assert ("strain", "oblique", [1]) in sels
    assert ("strain", "zsweep", [1]) in sels
    assert ("mosa_com_chi", "oblique", [1]) in sels
    # mosa has no zsweep group at all; zsweep plane 7 out of range
    assert any("mosa_com_chi/zsweep" in s for s in skipped)
    assert any("no plane" in s for s in skipped)


def test_layer_selections_product_and_skips():
    groups = [_G("sum_intensity", ["l0", "l1", "l2"]), _G("specific_frame", ["l0"])]
    sels, skipped = layer_selections(groups, [0, 2], ["sum_intensity", "specific_frame"])
    assert ("sum_intensity", [0, 2]) in sels
    assert ("specific_frame", [0]) in sels
    assert any("specific_frame" in s for s in skipped)
    assert layer_selections(groups, [], ["sum_intensity"])[0] == []
```

- [ ] Run `python3 -m pytest -q tests/test_plane_selection_model.py` — ModuleNotFoundError.
- [ ] Create `gui/widgets/plane_selection_model.py`:

```python
"""Qt-free selection model behind the planes-first replot/pin dialogs.

Planes/layers are listed ONCE (union across volumes/quantity groups); the
filter box only narrows visibility; render selections are the cartesian
product of checked planes × checked quantities, with missing combinations
reported as skip reasons (never errors). No PySide6 imports — unit-testable
without a QApplication.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PlaneRow:
    """One selectable plane/layer row in the left panel."""

    key: object  # (slice_name, plane_idx) for slices; layer int for generic
    section: str  # section header ("" = flat list)
    number: int  # plane/layer number (integer-token filtering)
    offset: float | None  # µm offset / Z (decimal-token filtering); None = unknown
    label: str  # display text, e.g. "p118  -3.72 µm"


def build_slice_rows(entries) -> list[PlaneRow]:
    """Rows from slices ReplotEntry list — one per (slice_name, plane_idx)."""
    seen: dict[tuple[str, int], PlaneRow] = {}
    for e in entries:
        for k, off in enumerate(e.offsets_um):
            key = (e.slice_name, k)
            if key not in seen:
                seen[key] = PlaneRow(
                    key=key,
                    section=e.slice_name,
                    number=k,
                    offset=float(off),
                    label=f"p{k:03d}  {off:+.2f} µm",
                )
    return list(seen.values())


_Z_IN_LABEL = re.compile(r"Z=([-+]?\d+(?:\.\d+)?)")


def build_layer_rows(groups) -> list[PlaneRow]:
    """Rows from generic ReplotGroup list — one per layer index, union across groups."""
    rows: dict[int, PlaneRow] = {}
    for g in groups:
        for z, lab in enumerate(g.item_labels):
            if z in rows:
                continue
            m = _Z_IN_LABEL.search(lab)
            rows[z] = PlaneRow(
                key=z,
                section="",
                number=z,
                offset=float(m.group(1)) if m else None,
                label=lab,
            )
    return [rows[z] for z in sorted(rows)]


def parse_tokens(text: str) -> list[tuple[str, float]]:
    """Classify comma tokens: bare unsigned int -> number; signed/decimal -> offset.

    Unparseable tokens become ("invalid", 0.0) — they match nothing (so a
    nonsense filter shows an empty list, not the full one).
    """
    out: list[tuple[str, float]] = []
    for tok in (t.strip() for t in (text or "").split(",")):
        if not tok:
            continue
        if re.fullmatch(r"\d+", tok):
            out.append(("number", float(tok)))
        else:
            try:
                out.append(("offset", float(tok)))
            except ValueError:
                out.append(("invalid", 0.0))
    return out


def _half_step(offsets: list[float]) -> float:
    if len(offsets) < 2:
        return float("inf")  # single plane: nearest matches unconditionally
    s = sorted(offsets)
    diffs = sorted(b - a for a, b in zip(s, s[1:]) if b > a)
    return diffs[len(diffs) // 2] / 2.0 if diffs else float("inf")


def filter_rows(rows: list[PlaneRow], text: str) -> list[PlaneRow]:
    """Rows visible under *text*. Blank -> all. Narrows only — never selects."""
    toks = parse_tokens(text)
    if not toks:
        return list(rows)
    by_section: dict[str, list[PlaneRow]] = {}
    for r in rows:
        by_section.setdefault(r.section, []).append(r)
    matched: set = set()
    for kind, v in toks:
        if kind == "number":
            matched.update(r.key for r in rows if r.number == int(v))
        elif kind == "offset":
            for sec_rows in by_section.values():
                with_off = [r for r in sec_rows if r.offset is not None]
                if not with_off:
                    continue
                tol = _half_step([r.offset for r in with_off])
                best = min(with_off, key=lambda r: abs(r.offset - v))
                if abs(best.offset - v) <= tol:
                    matched.add(best.key)
    return [r for r in rows if r.key in matched]


def slice_selections(entries, checked_plane_keys, checked_vids):
    """Checked planes × checked volume ids -> (selections, skip reasons).

    Selections match ``dfxm.stages.slices.render_replot``:
    ``[(volume_id, slice_name, [plane_idx, ...]), ...]``.
    """
    sels, skipped = [], []
    by_vid_slice = {(e.volume_id, e.slice_name): e for e in entries}
    wanted: dict[str, list[int]] = {}
    for sname, idx in checked_plane_keys:
        wanted.setdefault(sname, []).append(int(idx))
    for vid in checked_vids:
        for sname, idxs in wanted.items():
            e = by_vid_slice.get((vid, sname))
            if e is None:
                skipped.append(f"{vid}/{sname}: volume has no such slice group")
                continue
            ok = sorted(i for i in idxs if 0 <= i < e.n_planes)
            missing = sorted(set(idxs) - set(ok))
            if missing:
                skipped.append(f"{vid}/{sname}: no plane(s) {missing}")
            if ok:
                sels.append((vid, sname, ok))
    return sels, skipped


def layer_selections(groups, checked_layers, checked_keys):
    """Checked layers × checked quantity groups -> (selections, skip reasons).

    Selections match the generic ``render_replot``: ``[(group_key, [z, ...]), ...]``.
    """
    sels, skipped = [], []
    n_by_key = {g.key: len(g.item_labels) for g in groups}
    layers = sorted(int(z) for z in checked_layers)
    for key in checked_keys:
        n = n_by_key.get(key, 0)
        ok = [z for z in layers if 0 <= z < n]
        missing = [z for z in layers if z >= n]
        if missing:
            skipped.append(f"{key}: no layer(s) {missing}")
        if ok:
            sels.append((key, ok))
    return sels, skipped
```

- [ ] Run `python3 -m pytest -q tests/test_plane_selection_model.py` — green; then full `python3 -m pytest -q`.
- [ ] `docs/Codebase.md`: add the module under the `gui/widgets` listing.
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(gui): planes-first selection model (pure logic + tests)"`.

## Task B4: `PlaneSelectionPanel` widget + swap the slices replot dialog onto it

**Files:**
- Create: `/home/albert/Desktop/dfxm_pipeline/gui/widgets/plane_selection.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/gui/widgets/slice_replot.py` (replace the volume→slice→plane `QTreeWidget` with the panel)
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_gui_slice_replot.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/gui_smoke.py` (update `[26]`, add `[30]`)
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Usage.md`, `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Produces: `class PlaneSelectionPanel(QWidget)`:
  - `__init__(self, show_quantities: bool = True, parent=None)`; `selectionChanged = Signal()`
  - `set_rows(rows: list[PlaneRow])` — rebuild left tree (section header items non-checkable when sections exist; plane rows checkable), **all checked**; re-applies the current filter.
  - `set_quantities(quantities: list[tuple[object, str]])` — right flat checkbox list, all checked (hidden entirely when `show_quantities=False`).
  - `checked_plane_keys() -> list`, `checked_quantity_keys() -> list`, `has_selection() -> bool`, `set_all_checked(checked: bool)`, `check_all_visible()`.
  - Filter `QLineEdit` narrows visibility via `filter_rows`; a "no match" hint label appears when a non-blank filter hides everything; buttons: Check all / Uncheck all / Check all visible.
- Modified `SliceReplotDialog`: same constructor/public API (`select_all()`, `render_selection(out_dir)`, `written`); internally `self._panel: PlaneSelectionPanel`, `self._catalog: list[ReplotEntry]`; `_selections()` uses `slice_selections(...)`; skip reasons appended to the status label; Render button disabled while `has_selection()` is false. Clim (`ClimGroupSection` keyed by volume_id), ROI picker, timestamped `replots/<stamp>/` output default, and all-checked-on-load are unchanged.

**Steps:**

- [ ] Read `gui/widgets/slice_replot.py` in full (already quoted above), `tests/test_gui_slice_replot.py` in full, and `tests/gui_smoke.py` lines 801–838.
- [ ] Update `tests/test_gui_slice_replot.py`: adapt any `_tree`-poking assertions to the panel API and add:

```python
def test_panel_default_all_checked_renders_everything(tmp_path, qapp):
    # build a 2-plane, 2-volume oblique_slices.h5 exactly like the existing fixture
    # in this file, then:
    dlg = SliceReplotDialog(str(h5_path), style=None, out_default=str(tmp_path / "out"))
    assert dlg._panel.has_selection()
    written = dlg.render_selection(str(tmp_path / "out"))
    assert len(written) == 4  # 2 volumes x 2 planes


def test_panel_filter_and_check_all_visible_subsets(tmp_path, qapp):
    dlg = SliceReplotDialog(str(h5_path), style=None, out_default=str(tmp_path / "out"))
    dlg._panel.set_all_checked(False)
    dlg._panel._filter.setText("1")  # narrows to plane 1 only
    dlg._panel.check_all_visible()
    written = dlg.render_selection(str(tmp_path / "out"))
    assert len(written) == 2  # plane 1 in both volumes
```

(Match the file's actual fixture names/`qapp` conventions — read first; if it uses a shared `_make_h5` helper, reuse it.)
- [ ] Run `python3 -m pytest -q tests/test_gui_slice_replot.py` — expect failures (`_panel` missing).
- [ ] Create `gui/widgets/plane_selection.py`:

```python
"""Planes-first selection panel: planes listed once, filter narrows, quantities right.

Shared by the slices + generic replot dialogs and the Pin planes… dialog. All
selection logic lives in the Qt-free ``plane_selection_model``; this widget is
the checkbox/visibility shell around it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .plane_selection_model import PlaneRow, filter_rows


class PlaneSelectionPanel(QWidget):
    """Left: plane/layer rows (listed once). Right: quantity checkboxes."""

    selectionChanged = Signal()

    def __init__(self, show_quantities: bool = True, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[PlaneRow] = []
        self._items: dict[object, QTreeWidgetItem] = {}
        self._show_quantities = show_quantities

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("filter: plane numbers (118) or offsets (-3.7), commas")
        self._filter.textChanged.connect(self._apply_filter)
        self._no_match = QLabel("no match")
        self._no_match.setVisible(False)

        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Plane"])
        self._tree.itemChanged.connect(lambda *_: self.selectionChanged.emit())

        check_all = QPushButton("Check all")
        check_all.clicked.connect(lambda: self.set_all_checked(True))
        uncheck_all = QPushButton("Uncheck all")
        uncheck_all.clicked.connect(lambda: self.set_all_checked(False))
        check_visible = QPushButton("Check all visible")
        check_visible.clicked.connect(self.check_all_visible)
        btns = QHBoxLayout()
        for b in (check_all, uncheck_all, check_visible):
            btns.addWidget(b)
        btns.addStretch(1)

        left = QVBoxLayout()
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Filter:"))
        frow.addWidget(self._filter, 1)
        frow.addWidget(self._no_match)
        left.addLayout(frow)
        left.addWidget(self._tree, 1)
        left.addLayout(btns)

        root = QHBoxLayout(self)
        lw = QWidget()
        lw.setLayout(left)
        root.addWidget(lw, 2)

        self._qty = QListWidget()
        self._qty.itemChanged.connect(lambda *_: self.selectionChanged.emit())
        if show_quantities:
            right = QVBoxLayout()
            right.addWidget(QLabel("Quantities:"))
            right.addWidget(self._qty, 1)
            rw = QWidget()
            rw.setLayout(right)
            root.addWidget(rw, 1)

    # -- population -------------------------------------------------------
    def set_rows(self, rows: list[PlaneRow]) -> None:
        """Rebuild the plane list; everything checked (a plain Render remakes all)."""
        self._rows = list(rows)
        self._tree.blockSignals(True)
        self._tree.clear()
        self._items.clear()
        sections: dict[str, QTreeWidgetItem] = {}
        for r in self._rows:
            if r.section:
                parent = sections.get(r.section)
                if parent is None:
                    parent = QTreeWidgetItem(self._tree, [r.section])
                    parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
                    sections[r.section] = parent
                item = QTreeWidgetItem(parent, [r.label])
            else:
                item = QTreeWidgetItem(self._tree, [r.label])
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(0, Qt.CheckState.Checked)
            item.setData(0, Qt.ItemDataRole.UserRole, r.key)
            self._items[r.key] = item
        self._tree.expandAll()
        self._tree.blockSignals(False)
        self._apply_filter(self._filter.text())
        self.selectionChanged.emit()

    def set_quantities(self, quantities: list[tuple[object, str]]) -> None:
        self._qty.blockSignals(True)
        self._qty.clear()
        for key, label in quantities:
            it = QListWidgetItem(label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked)
            it.setData(Qt.ItemDataRole.UserRole, key)
            self._qty.addItem(it)
        self._qty.blockSignals(False)
        self.selectionChanged.emit()

    # -- filtering (visibility only; never selects) -----------------------
    def _apply_filter(self, text: str) -> None:
        visible = {r.key for r in filter_rows(self._rows, text)}
        for key, item in self._items.items():
            item.setHidden(key not in visible)
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            if top.childCount():  # section header: hide when all children hidden
                top.setHidden(all(top.child(j).isHidden() for j in range(top.childCount())))
        self._no_match.setVisible(bool(text.strip()) and not visible)

    # -- bulk actions ------------------------------------------------------
    def set_all_checked(self, checked: bool) -> None:
        state = Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        for item in self._items.values():
            item.setCheckState(0, state)

    def check_all_visible(self) -> None:
        for item in self._items.values():
            if not item.isHidden():
                item.setCheckState(0, Qt.CheckState.Checked)

    # -- selection ---------------------------------------------------------
    def checked_plane_keys(self) -> list:
        return [k for k, it in self._items.items() if it.checkState(0) == Qt.CheckState.Checked]

    def checked_quantity_keys(self) -> list:
        return [
            self._qty.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._qty.count())
            if self._qty.item(i).checkState() == Qt.CheckState.Checked
        ]

    def has_selection(self) -> bool:
        if not self.checked_plane_keys():
            return False
        return (not self._show_quantities) or bool(self.checked_quantity_keys())
```

- [ ] Swap `SliceReplotDialog` internals (`gui/widgets/slice_replot.py`): import `PlaneSelectionPanel` and `build_slice_rows, slice_selections` from the new modules; delete the `QTreeWidget` construction + tree toolbar; construct `self._panel = PlaneSelectionPanel(show_quantities=True)` and add it where the tree/toolbar were; keep a `self._render_btn` reference and wire `self._panel.selectionChanged.connect(lambda: self._render_btn.setEnabled(self._panel.has_selection()))`. Rewrite:

```python
    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        if not self._out_pinned:
            self._out_edit.setText(self._default_out_for(self._h5_path))
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._catalog = []
            self._panel.set_rows([])
            self._panel.set_quantities([])
            self._clim.set_groups([])
            self._status.setText("no such file")
            return
        try:
            self._catalog = _sl.replot_catalog(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._catalog = []
            self._panel.set_rows([])
            self._panel.set_quantities([])
            self._clim.set_groups([])
            self._status.setText(f"cannot read: {exc}")
            return
        self._clim.set_groups(self._clim_groups(self._catalog))
        self._panel.set_rows(build_slice_rows(self._catalog))
        vids = list(dict.fromkeys(e.volume_id for e in self._catalog))
        self._panel.set_quantities([(vid, _volume_label(vid)) for vid in vids])
        self._status.setText(f"{len(self._catalog)} slice group(s)")

    def select_all(self) -> None:  # kept for smoke/back-compat
        self._panel.set_all_checked(True)

    def _selections(self):
        sels, self._skipped = slice_selections(
            self._catalog,
            self._panel.checked_plane_keys(),
            self._panel.checked_quantity_keys(),
        )
        return sels
```

In `_on_render`, after a successful render, append skip info: `if self._skipped: self._status.setText(self._status.text() + f"; skipped {len(self._skipped)} combo(s)")`. Initialise `self._catalog: list = []` and `self._skipped: list[str] = []` in `__init__` before `_reload()`. Delete the now-unused `_deselect_all` and `QTreeWidget/QTreeWidgetItem` imports.
- [ ] Run `python3 -m pytest -q tests/test_gui_slice_replot.py tests/test_gui_clim_section.py` — green. Update smoke `[26]`: replace `assert _dlg26._tree.topLevelItemCount() == 1` with `assert len(_dlg26._panel._rows) == 1`, keep `select_all()` + `render_selection`. Add step `[30]` after `[29]`:

```python
    # [30] Planes-first slices replot: filter narrows visibility; check-all-visible selects.
    from gui.widgets.slice_replot import SliceReplotDialog as _SRD30

    _dlg30 = _SRD30(_h5_path26, style=None, out_default=_out26)  # reuse [26]'s file
    _dlg30._panel.set_all_checked(False)
    assert not _dlg30._panel.has_selection()
    _dlg30._panel._filter.setText("0")
    _dlg30._panel.check_all_visible()
    assert _dlg30._panel.has_selection()
    _dlg30._panel._filter.setText("999")
    assert _dlg30._panel._no_match.isVisible()
    print("[30] planes-first slices replot: filter + check-all-visible + no-match hint")
```

- [ ] Run `python3 tests/gui_smoke.py` (30 steps) and full `python3 -m pytest -q` — green.
- [ ] Docs: `docs/Usage.md` slices Replot… section — planes listed once, filter semantics (int = plane number, decimal/signed = nearest offset within half a sweep step, narrows only), check-all-visible, quantities panel, skip behaviour. `docs/Codebase.md`: `plane_selection.py` + `slice_replot.py` changes.
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(gui): PlaneSelectionPanel; slices replot dialog goes planes-first"`.

## Task B5: swap the generic replot dialog (strain/mosaicity/rocking)

**Files:**
- Modify: `/home/albert/Desktop/dfxm_pipeline/gui/widgets/replot_dialog.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_gui_replot_dialog.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/gui_smoke.py` (add `[31]`)
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Usage.md`, `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Modified `ReplotDialog(h5_default, catalog_fn, render_fn, style=None, out_default="", preview_fn=None, parent=None)` — constructor and `render_selection(out_dir)`/`select_all()`/`written` unchanged (so `gui/stage_view.py` needs **no** changes). Internals: `self._panel = PlaneSelectionPanel(show_quantities=True)`; left rows via `build_layer_rows(catalog)` (layers listed once, `Z=… µm` parsed when present); right quantities = `[(grp.key, grp.label + shape hint)]`; `_selections()` via `layer_selections(catalog, ...)` (returns `[(key, idxs)]`, accepted by every `render_replot`); skip reasons in status; Render disabled on empty selection. Clim (`ClimGroupSection` keyed per group key), ROI picker (`preview_fn`), timestamped output default, all-checked-on-load unchanged.
- Consumes: `PlaneSelectionPanel` (Task B4), `build_layer_rows`/`layer_selections` (Task B3).

**Steps:**

- [ ] Read `tests/test_gui_replot_dialog.py` in full; adapt its tree-based assertions to the panel API and add a default-all-checked render test plus a filter+check-all-visible subset test, mirroring Task B4's two tests but against `ReplotDialog` with a fake `catalog_fn`/`render_fn`:

```python
def test_generic_dialog_planes_first_product(tmp_path, qapp):
    class _G:
        def __init__(self, key, labels):
            self.key, self.label, self.item_labels, self.shape = key, key, labels, None

    calls = []

    def catalog_fn(_path):
        return [_G("sum_intensity", ["layer 0", "layer 1"]), _G("specific_frame", ["layer 0"])]

    def render_fn(h5, selections, st, clim, roi, out):
        calls.append(selections)
        return ["x.png"]

    h5 = tmp_path / "a.h5"
    h5.write_bytes(b"")
    dlg = ReplotDialog(str(h5), catalog_fn, render_fn, out_default=str(tmp_path))
    dlg.render_selection(str(tmp_path))
    sels = dict(calls[-1])
    assert sels["sum_intensity"] == [0, 1]
    assert sels["specific_frame"] == [0]  # layer 1 skipped for this product, no error
```

(`catalog_fn` is called on a path that exists — the empty file is fine because the fake never opens it; keep the existing tests' `qapp` fixture conventions.)
- [ ] Run `python3 -m pytest -q tests/test_gui_replot_dialog.py` — expect failures.
- [ ] Implement the swap in `replot_dialog.py`, exactly mirroring Task B4's `SliceReplotDialog` changes: keep the file row / clim / ROI / out-dir / status / buttons; replace the tree + toolbar with `self._panel`; `_reload` stores `self._catalog = self._catalog_fn(...)`, calls `self._panel.set_rows(build_layer_rows(self._catalog))` and `self._panel.set_quantities([(g.key, g.label if g.shape is None else f"{g.label}   ·   {g.shape[0]}×{g.shape[1]} px (Y×X)") for g in self._catalog])`, and re-populates `self._clim` as before; `select_all()` delegates to `self._panel.set_all_checked(True)`; `_selections()` returns `layer_selections(self._catalog, self._panel.checked_plane_keys(), self._panel.checked_quantity_keys())[0]` with the skip list stored on `self._skipped` and surfaced in `_on_render`'s status; Render button disabled when `not self._panel.has_selection()`. `_on_pick_roi` keeps using `self._catalog_fn` unchanged.
- [ ] Run `python3 -m pytest -q tests/test_gui_replot_dialog.py tests/test_figures_replot.py`, then full `python3 -m pytest -q` — green.
- [ ] Add smoke `[31]` (after `[30]`): construct a `ReplotDialog` with the same fake catalog/render pattern as the test above, assert `select_all()` + `render_selection` invokes the render_fn with both groups, and that the filter no-match hint appears for `"999"`. Print `"[31] generic replot dialog planes-first: product selection + filter"`. Run `python3 tests/gui_smoke.py`.
- [ ] Docs: `docs/Usage.md` — the strain/mosaicity/rocking Replot… section gets the same planes-first description (layers listed once). `docs/Codebase.md`: `replot_dialog.py` entry.
- [ ] `ruff check . && ruff format .`, `git add -A && git commit -m "feat(gui): generic replot dialog goes planes-first (layers listed once)"`.

## Task B6: Pin planes… dialog + slices form wiring — **Phase B gate**

**Files:**
- Create: `/home/albert/Desktop/dfxm_pipeline/gui/widgets/pin_planes.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/gui/stage_view.py` (button block ~line 140; new `_on_pin_planes` slot next to `_replot_slices` ~line 496)
- Create: `/home/albert/Desktop/dfxm_pipeline/tests/test_gui_pin_planes.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/test_stage_view_buttons.py`
- Modify: `/home/albert/Desktop/dfxm_pipeline/tests/gui_smoke.py` (add `[32]`)
- Modify: `/home/albert/Desktop/dfxm_pipeline/docs/Usage.md`, `/home/albert/Desktop/dfxm_pipeline/docs/Codebase.md`

**Interfaces:**
- Produces: `class PinPlanesDialog(QDialog)` — `__init__(self, h5_default="", parent=None)`; attribute `result_json: str | None` (set on accept). Layout: file row (edit + Browse… + Load), `PlaneSelectionPanel(show_quantities=False)` (planes start **unchecked** — pinning means explicit picks), status label, OK/Cancel. Load: `dfxm.stages.slices.replot_catalog(h5)` → `build_slice_rows`; unreadable/empty file → inline status error, nothing written. OK: checked keys grouped by slice_name → `dfxm.stages.slices.build_pinned_spec(h5, sname, offsets)` per group → `self.result_json = json.dumps(specs, indent=2)`; empty selection or `StageUserError` → inline status, dialog stays open, nothing written.
- Produces: on the slices `StageView`, a "Pin planes…" button (`self._pin_btn`) whose handler opens the dialog pre-filled with the chained/last output h5 (same path computation as `_replot_slices`), and on accept writes the form: `self._form.set_values({"pinned_slices_json": dlg.result_json, "use_pinned": True})` (this flows through the normal dirty-gated form persistence — no custom persistence).
- Consumes: `PlaneSelectionPanel` (B4), `build_slice_rows` (B3), `build_pinned_spec` (B1), `StageUserError`.

**Steps:**

- [ ] Read `gui/stage_view.py` lines 130–160 and 439–521, and `tests/test_stage_view_buttons.py` in full. Create `tests/test_gui_pin_planes.py`:

```python
"""Tests for the Pin planes… dialog and its slices-form wiring."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest  # noqa: F401

from gui.widgets.pin_planes import PinPlanesDialog


def _sweep_h5(tmp_path):
    p = str(tmp_path / "oblique_slices.h5")
    with h5py.File(p, "w") as f:
        sg = f.create_group("strain").create_group("oblique")
        sg.create_dataset("slices", data=np.zeros((3, 4, 5), dtype=np.float32))
        sg.create_dataset("u_um", data=np.linspace(0, 4, 5))
        sg.create_dataset("v_um", data=np.linspace(0, 3, 4))
        sg.create_dataset("offsets_um", data=np.array([-2.0, 0.0, 2.0]))
        sg.attrs["normal"] = [0.0, 0.0, 1.0]
        sg.attrs["origin"] = [0.0, 0.0, 0.0]
        sg.attrs["up"] = [0.0, 1.0, 0.0]
        for k, v in (("half_u", 2.0), ("half_v", 1.5), ("du", 1.0), ("dv", 1.0),
                     ("sweep_step_um", 2.0)):
            sg.attrs[k] = v
        f["strain"].attrs["kind"] = "strain"
    return p


def test_pin_dialog_writes_pinned_specs_for_checked_planes(tmp_path, qapp):
    dlg = PinPlanesDialog(_sweep_h5(tmp_path))
    assert not dlg._panel.checked_plane_keys()  # pinning starts unchecked
    dlg._panel._items[("oblique", 2)].setCheckState(0, __import__("PySide6.QtCore",
        fromlist=["Qt"]).Qt.CheckState.Checked)
    dlg._on_ok()
    specs = json.loads(dlg.result_json)
    assert len(specs) == 1
    assert specs[0]["sweep_start_um"] == specs[0]["sweep_stop_um"] == 2.0
    assert specs[0]["half_u"] == 2.0


def test_pin_dialog_empty_selection_or_bad_file_writes_nothing(tmp_path, qapp):
    dlg = PinPlanesDialog(_sweep_h5(tmp_path))
    dlg._on_ok()  # nothing checked
    assert dlg.result_json is None
    bad = PinPlanesDialog(str(tmp_path / "missing.h5"))
    assert bad._panel._rows == [] and bad.result_json is None
```

(Use the same `qapp` fixture the other `test_gui_*` files use — read `tests/conftest.py` to confirm its name; replace the awkward inline Qt import with a plain `from PySide6.QtCore import Qt` at the top.)
- [ ] Run `python3 -m pytest -q tests/test_gui_pin_planes.py` — ModuleNotFoundError.
- [ ] Create `gui/widgets/pin_planes.py`:

```python
"""Pin planes… dialog: pick sweep planes to pin, emit pinned_slices_json.

Loads a swept oblique_slices.h5, lists slice group -> planes with the shared
planes-first panel (same number/offset filter as the replot dialogs), and on
OK builds exact-geometry pinned specs via the Qt-free
``dfxm.stages.slices.build_pinned_spec``. The caller (slices StageView) writes
the JSON into the form and ticks 'Run pinned planes only'.
"""

from __future__ import annotations

import json
import os

from PySide6.QtCore import Qt  # noqa: F401 (checkstate use in tests/tools)
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)

from dfxm.common.errors import StageUserError
from dfxm.stages import slices as _sl

from .plane_selection import PlaneSelectionPanel
from .plane_selection_model import build_slice_rows


class PinPlanesDialog(QDialog):
    """Pick planes from a swept oblique_slices.h5; OK yields result_json."""

    def __init__(self, h5_default: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pin planes")
        self.result_json: str | None = None
        self._h5_path = h5_default or ""

        self._file_edit = QLineEdit(self._h5_path)
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._on_browse)
        load = QPushButton("Load")
        load.clicked.connect(self._reload)
        frow = QHBoxLayout()
        frow.addWidget(QLabel("Slices file:"))
        frow.addWidget(self._file_edit, 1)
        frow.addWidget(browse)
        frow.addWidget(load)

        self._panel = PlaneSelectionPanel(show_quantities=False)
        self._status = QLabel("")
        ok = QPushButton("OK")
        ok.setProperty("role", "primary")
        ok.clicked.connect(self._on_ok)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        brow = QHBoxLayout()
        brow.addWidget(self._status, 1)
        brow.addWidget(ok)
        brow.addWidget(cancel)

        layout = QVBoxLayout(self)
        layout.addLayout(frow)
        layout.addWidget(self._panel, 1)
        layout.addLayout(brow)
        self._reload()

    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._panel.set_rows([])
            self._status.setText("no such file")
            return
        try:
            catalog = _sl.replot_catalog(self._h5_path)
        except Exception as exc:  # noqa: BLE001 — show inline, never crash
            self._panel.set_rows([])
            self._status.setText(f"cannot read: {exc}")
            return
        self._panel.set_rows(build_slice_rows(catalog))
        self._panel.set_all_checked(False)  # pinning = explicit picks, not all
        self._status.setText(f"{len(self._panel._rows)} plane(s) — check the ones to pin")

    def _on_ok(self) -> None:
        keys = self._panel.checked_plane_keys()
        if not keys:
            self._status.setText("no planes checked")
            return
        row_by_key = {r.key: r for r in self._panel._rows}
        by_slice: dict[str, list[float]] = {}
        for sname, _idx in keys:
            by_slice.setdefault(sname, []).append(row_by_key[(sname, _idx)].offset)
        specs: list[dict] = []
        try:
            for sname, offs in by_slice.items():
                specs.extend(_sl.build_pinned_spec(self._h5_path, sname, offs))
        except StageUserError as exc:
            self._status.setText(str(exc))
            return
        self.result_json = json.dumps(specs, indent=2)
        self.accept()

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open oblique_slices.h5", "", "HDF5 (*.h5)")
        if path:
            self._file_edit.setText(path)
            self._reload()
```

- [ ] Wire `gui/stage_view.py`: in `__init__` next to the replot button block add

```python
        # slices: pin sweep planes into pinned_slices_json (built lazily on click)
        self._pin_btn: QPushButton | None = None
        if stage_name == "slices":
            self._pin_btn = QPushButton("Pin planes…")
            self._pin_btn.clicked.connect(self._on_pin_planes)
            btn_row.addWidget(self._pin_btn)
```

and add the slot after `_replot_slices`:

```python
    def _on_pin_planes(self) -> None:
        """Open Pin planes… and write pinned_slices_json + use_pinned into the form."""
        vals = self._form.values()
        out_dir = vals.get("output_dir", "") or os.path.join(
            os.path.dirname(
                vals.get("mosa_volume_file", "") or vals.get("strain_volume_file", "") or "."
            ),
            "oblique_slices",
        )
        h5 = os.path.join(out_dir, vals.get("output_h5_name", "") or "oblique_slices.h5")

        from .widgets.pin_planes import PinPlanesDialog  # imported on demand

        dlg = PinPlanesDialog(h5, parent=self)
        if dlg.exec() and dlg.result_json:
            self._form.set_values({"pinned_slices_json": dlg.result_json, "use_pinned": True})
            self._log.append(
                "Pinned planes written; 'Run pinned planes only' ticked. Run to render "
                "(output goes to oblique_slices_pinned.h5 unless you set a name)."
            )
            self._tabs.setCurrentWidget(self._log)
```

- [ ] Extend `tests/test_stage_view_buttons.py`: assert the slices view has `_pin_btn` and (e.g.) strain/profiles have `_pin_btn is None` — follow the file's existing assertion pattern for `_replot_btn`/`_pick_btn`.
- [ ] Run `python3 -m pytest -q tests/test_gui_pin_planes.py tests/test_stage_view_buttons.py` — green.
- [ ] Add smoke `[32]` (after `[31]`): reuse the `[26]` slices file (`_h5_path26`), construct `PinPlanesDialog(_h5_path26)`, check its single plane via `_dlg32._panel._items[("oblique", 0)].setCheckState(0, Qt.CheckState.Checked)` (import Qt locally), call `_on_ok()`, assert `json.loads(_dlg32.result_json)[0]["sweep_start_um"] == 0.0`; also assert `win._views["slices"]._pin_btn is not None`. Print `"[32] Pin planes… dialog emits pinned specs; button wired on slices view"`.
- [ ] Docs (same commit): `docs/Usage.md` slices section — full Pin planes… workflow (open, filter, check, OK ticks the toggle; untick to return to the sweep); `docs/Codebase.md` — `pin_planes.py` module + the `stage_view.py` button.
- [ ] Phase B gate: `python3 -m pytest -q` && `ruff check . && ruff format .` && `python3 tests/gui_smoke.py` (32 steps) — all green.
- [ ] `git add -A && git commit -m "feat(gui): Pin planes… dialog + slices form wiring + smoke [32] (Phase B complete)"`.

---

## Final verification (whole branch)

- [ ] `python3 -m pytest -q` — full suite green (expected: prior count + ~25 new tests, 13 skips unchanged).
- [ ] `ruff check . && ruff format .` — clean.
- [ ] `python3 tests/gui_smoke.py` — steps [1]–[32] pass.
- [ ] Sanity greps: `grep -rn "PySide6\|pyvista" dfxm/` → no new hits; `grep -n "pyplot" dfxm/ gui/ -r` → none.
- [ ] Request the standard whole-branch review (fable, xhigh) before merging; known intentional behaviours to tell the reviewer: (1) `scale_um_per_cm` blank keeps byte-identical output — the fixed path only activates via the knob; (2) the profiles companion figure is deliberately NOT fitted and its bar keeps data-fraction geometry; (3) pinned runs rename the default output file on purpose; (4) `render_replot` receives explicit index lists where it previously often got `None` (semantically identical: all indices).

**Risks / watch items for reviewers:** `fit_axes_to_box` interacting with `TwoSlopeNorm` colorbars and the `_apply_scientific` static text (redraw-safe by design — verify no offset-text drift after resize); `filter_rows` half-step tolerance on irregular sweeps (median-diff based); the generic dialog's layer-union when products have unequal layer counts (skip path, never KeyError); form-persistence of a large `pinned_slices_json` TEXT blob (JSON in QSettings — same mechanism as `slices_json`, no new code).agentId: acfa54a73d6f6c058 (use SendMessage with to: 'acfa54a73d6f6c058', summary: '<5-10 word recap>' to continue this agent)
<usage>subagent_tokens: 221619
tool_uses: 31
duration_ms: 877014</usage>