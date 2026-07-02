# Styled-Figure Layout Overhaul Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Styled figure exports never overlap (constrained layout), the title gets its own independent size control, and colour limits can be auto-rounded to nice values with the rounding reported explicitly.

**Architecture:** All changes centre on `dfxm/common/plotting.py` (three new/changed primitives: `styled_figure`, `title_scale` in `apply_text_scale`, `round_limits_outward`/`apply_round_clim`), which every stage's figure builder already routes through. Stages then wire `round_clim` where they compute auto colour limits. The GUI only touches `gui/widgets/export_dialog.py` (`StyleControls`). The legacy `style=None` render path stays byte-identical.

**Tech Stack:** Python 3.10, matplotlib (explicit `Figure` API — never pyplot), h5py, PySide6 (GUI only), pytest.

**Spec:** `docs/superpowers/specs/2026-07-02-figure-layout-title-scale-round-clim-design.md`

## Global Constraints

- `dfxm/` stays Qt-free: never import PySide6/pyvista at module level there.
- Never use `pyplot` or `matplotlib.use(...)` — build figures with `matplotlib.figure.Figure`.
- The legacy path (`style is None`) must keep today's behaviour: plain `Figure`, no constrained layout, no rounding.
- Ruff: line length 100, double quotes. `ruff format` runs automatically on Write/Edit via hook; run `ruff check .` before each commit.
- Run tests with `python3 -m pytest -q` (baseline: 281 passed / 13 skipped).
- Docs contract: `docs/Usage.md` + `docs/Codebase.md` must be updated in the same change set (Task 7).
- Commit messages end with:
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_013ZTFrGWMWLKrpsM3G9fUVg`

---

### Task 1: `PlotStyle.title_scale` — independent title sizing

**Files:**
- Modify: `dfxm/common/plotting.py` (PlotStyle dataclass ~line 60; `apply_text_scale` ~line 231)
- Test: `tests/test_plot_style.py`

**Interfaces:**
- Produces: `PlotStyle.title_scale: float = 1.0` (new dataclass field); `apply_text_scale(ax, style)` now sizes the title as `base_fontsize * style.title_scale` (the title no longer scales with `font_scale`). JSON/params round-trip picks the field up automatically (`_style_from_dict` filters by `fields(PlotStyle)`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plot_style.py`:

```python
def test_title_scale_is_independent_of_font_scale():
    fig, ax = _ax()
    ax.set_xlabel("X (µm)")
    ax.set_title("χ Misorientation")
    label_base = ax.xaxis.label.get_fontsize()
    title_base = ax.title.get_fontsize()
    apply_text_scale(ax, PlotStyle(font_scale=3.0, title_scale=0.5))
    assert ax.xaxis.label.get_fontsize() == label_base * 3.0
    assert ax.title.get_fontsize() == title_base * 0.5  # font_scale must NOT touch the title


def test_title_scale_default_leaves_title_at_base_size():
    fig, ax = _ax()
    ax.set_title("t")
    title_base = ax.title.get_fontsize()
    apply_text_scale(ax, PlotStyle(font_scale=2.2))  # title_scale defaults to 1.0
    assert ax.title.get_fontsize() == title_base


def test_title_scale_survives_json_roundtrip():
    from dfxm.common.plotting import style_from_json, style_to_json

    s = PlotStyle(title_scale=0.4)
    assert style_from_json(style_to_json(s)).title_scale == 0.4
    # Old persisted blobs (no title_scale key) default to 1.0
    assert style_from_json(style_to_json(PlotStyle())).title_scale == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plot_style.py -q -k title_scale`
Expected: FAIL — `TypeError: PlotStyle.__init__() got an unexpected keyword argument 'title_scale'`

- [ ] **Step 3: Implement**

In `dfxm/common/plotting.py`, the `# text` block of `PlotStyle` (around line 60):

```python
    # text
    font_scale: float = 1.0  # multiplies axis labels + ticks (NOT the title)
    title_scale: float = 1.0  # multiplies the title alone (independent of font_scale)
    show_title: bool = True
    center_axis_labels: bool = True
```

In `apply_text_scale` (around line 254), change the title branch:

```python
    title = ax.title
    if not style.show_title:
        ax.set_title("")
    else:
        title.set_fontsize(title.get_fontsize() * style.title_scale)
```

Also update the docstring of `apply_text_scale`:

```python
    """Scale axis-label/tick fonts by ``style.font_scale`` and the title by the
    independent ``style.title_scale``; apply title/centre options."""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_plot_style.py -q`
Expected: PASS (all tests in the file — the existing `noop_at_font_scale_1` test still holds because `title_scale` defaults to 1.0)

- [ ] **Step 5: Run the full suite (behaviour change guard)**

Run: `python3 -m pytest -q`
Expected: 284+ passed / 13 skipped. If any test fails because it asserted the title grows with `font_scale`, update that test to the new contract (title follows `title_scale` only) — this behaviour change is the point of the task.

- [ ] **Step 6: Commit**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py
git commit -m "feat(style): title_scale sizes the title independently of font_scale"
```

---

### Task 2: `round_limits_outward` + `apply_round_clim` helpers

**Files:**
- Modify: `dfxm/common/plotting.py` (add both functions after `symmetric_limits`, ~line 190; add `round_clim` field to `PlotStyle`; add `import math` is already present)
- Test: `tests/test_plot_style.py`

**Interfaces:**
- Produces:
  - `PlotStyle.round_clim: bool = False` (new field, in the `# colourbar` block).
  - `round_limits_outward(vmin: float, vmax: float) -> tuple[float, float]` — pure rounding rule.
  - `apply_round_clim(vmin: float, vmax: float, style: PlotStyle | None) -> tuple[float, float, str | None]` — returns possibly-rounded limits plus a human-readable note (`None` when disabled or unchanged). Stages (Tasks 4–5) call this and surface the note.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plot_style.py`:

```python
def test_round_limits_outward_symmetric_stays_symmetric():
    from dfxm.common.plotting import round_limits_outward

    lo, hi = round_limits_outward(-0.0778, 0.0778)
    assert (lo, hi) == (-0.08, 0.08)


def test_round_limits_outward_examples():
    from dfxm.common.plotting import round_limits_outward

    assert round_limits_outward(0.0, 0.11)[1] == 0.15
    assert round_limits_outward(0.0, 0.0432)[1] == 0.045
    assert abs(round_limits_outward(0.0, 1.7e-4)[1] - 2e-4) < 1e-12
    # asymmetric: vmin floors, vmax ceils
    lo, hi = round_limits_outward(-5.3, -1.2)
    assert (lo, hi) == (-5.5, -1.0)


def test_round_limits_outward_already_round_is_unchanged():
    from dfxm.common.plotting import round_limits_outward

    assert round_limits_outward(-0.08, 0.08) == (-0.08, 0.08)  # no float-epsilon inflation
    assert round_limits_outward(0.0, 0.1) == (0.0, 0.1)


def test_round_limits_outward_degenerate_and_zero():
    import math

    from dfxm.common.plotting import round_limits_outward

    assert round_limits_outward(0.5, 0.5) == (0.5, 0.5)  # degenerate: unchanged
    assert round_limits_outward(0.0, 0.0778) == (0.0, 0.08)  # zero endpoint stays 0
    lo, hi = round_limits_outward(float("nan"), 1.0)  # non-finite: passthrough
    assert math.isnan(lo) and hi == 1.0


def test_apply_round_clim_notes_and_gating():
    from dfxm.common.plotting import apply_round_clim

    # disabled (default style) and style=None: passthrough, no note
    assert apply_round_clim(-0.0778, 0.0778, PlotStyle()) == (-0.0778, 0.0778, None)
    assert apply_round_clim(-0.0778, 0.0778, None) == (-0.0778, 0.0778, None)
    # enabled: rounded + symmetric note
    lo, hi, note = apply_round_clim(-0.0778, 0.0778, PlotStyle(round_clim=True))
    assert (lo, hi) == (-0.08, 0.08)
    assert note == "colour limits rounded ±0.0778 → ±0.08 (round_clim)"
    # enabled but already round: no note
    assert apply_round_clim(-0.08, 0.08, PlotStyle(round_clim=True))[2] is None
    # asymmetric note shows both pairs
    _, _, note = apply_round_clim(0.0, 0.11, PlotStyle(round_clim=True))
    assert note == "colour limits rounded (0, 0.11) → (0, 0.15) (round_clim)"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plot_style.py -q -k "round_limits or round_clim"`
Expected: FAIL — `ImportError: cannot import name 'round_limits_outward'`

- [ ] **Step 3: Implement**

In `PlotStyle`, `# colourbar` block (after `colorbar_tick_format`):

```python
    colorbar_tick_format: str = "auto"  # "auto" | "scientific" | a digit count like "2"
    round_clim: bool = False  # round auto colour limits outward to nice values
```

After `symmetric_limits` in `dfxm/common/plotting.py`:

```python
def round_limits_outward(vmin: float, vmax: float) -> tuple[float, float]:
    """Round colour limits OUTWARD (vmin down, vmax up) to 'nice' values.

    Each non-zero endpoint moves to the next multiple of half its
    leading-digit unit (step = 0.5 * 10**floor(log10(|v|))): ±0.0778 → ±0.08,
    0.11 → 0.15, 0.0432 → 0.045, 1.7e-4 → 2e-4. Results have at most two
    significant digits (last digit 0 or 5), so evenly spaced colourbar ticks
    land on round numbers. Symmetric input stays exactly symmetric; zero
    endpoints, non-finite values and degenerate ranges (vmin >= vmax) are
    returned unchanged.
    """

    def _out(v: float, up: bool) -> float:
        if v == 0.0 or not math.isfinite(v):
            return v
        step = 0.5 * 10.0 ** math.floor(math.log10(abs(v)))
        n = v / step
        # epsilon guard so already-round values do not inflate by a whole step
        n = math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)
        return n * step

    if not (math.isfinite(vmin) and math.isfinite(vmax)) or vmin >= vmax:
        return (vmin, vmax)
    return (_out(vmin, up=False), _out(vmax, up=True))


def apply_round_clim(
    vmin: float, vmax: float, style: "PlotStyle | None"
) -> tuple[float, float, str | None]:
    """Round (vmin, vmax) outward when ``style.round_clim`` is set.

    Returns ``(vmin, vmax, note)``. The note is a user-facing description of
    what changed (``None`` when rounding is off, style is None, or the limits
    were already round) — stages surface it in the run log / results.
    """
    if style is None or not style.round_clim:
        return vmin, vmax, None
    rlo, rhi = round_limits_outward(vmin, vmax)
    if rlo == vmin and rhi == vmax:
        return vmin, vmax, None
    if math.isclose(-vmin, vmax, rel_tol=1e-9) and math.isclose(-rlo, rhi, rel_tol=1e-9):
        note = f"colour limits rounded ±{vmax:.4g} → ±{rhi:.4g} (round_clim)"
    else:
        note = f"colour limits rounded ({vmin:.4g}, {vmax:.4g}) → ({rlo:.4g}, {rhi:.4g}) (round_clim)"
    return rlo, rhi, note
```

Note: `-0.0778 / 0.005 = -15.56 → floor → -16 → -0.08` and `0.0778 / 0.005 → ceil → 16 → 0.08`, so symmetry is preserved by construction (floor(-x) == -ceil(x)).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_plot_style.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py
git commit -m "feat(style): round_clim — outward rounding of auto colour limits to nice values"
```

---

### Task 3: Constrained layout for all styled figures + no-overlap regression test

**Files:**
- Modify: `dfxm/common/plotting.py` (add `styled_figure` next to `new_figure` ~line 214; use it in `build_histogram` ~line 380)
- Modify: `dfxm/stages/slices.py:751` (`build_slice_figure`)
- Modify: `dfxm/common/render.py:49` (`layer_figure`)
- Modify: `dfxm/stages/profiles.py:446` (`build_companion_figure`) and `dfxm/stages/profiles.py:505` (`render_single`)
- Modify: `dfxm/stages/strain.py:380` (`build_strain_map`) and `dfxm/stages/strain.py:437` (`build_detrend_diag`)
- Test: create `tests/test_figure_layout.py`

**Interfaces:**
- Produces: `styled_figure(figsize: tuple[float, float], *, styled: bool) -> Figure` in `dfxm.common.plotting` — white-face Figure; `layout="constrained"` when `styled=True`, plain margins when `False` (legacy). All figure builders call it with `styled=(style is not None)`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_figure_layout.py`:

```python
"""Regression tests for styled-figure layout: no text may overlap at large font scales."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfxm.common.plotting import PlotStyle, styled_figure
from dfxm.stages.slices import build_slice_figure

# The exact style family that produced the overlapping export in the bug report
_BIG = PlotStyle(
    font_scale=2.2,
    figure_width="single",
    colorbar_ticks=5,
    colorbar_tick_format="scientific",
    scale_bar=True,
    scale_bar_box=True,
    scale_bar_color="white",
)


def _slice_fixture():
    u = np.linspace(-200.0, 200.0, 80)
    v = np.linspace(-120.0, 120.0, 50)
    data = np.outer(np.linspace(-0.0778, 0.0778, 50), np.ones(80))
    prep = {
        "cmap_name": "RdBu_r",
        "vmin": -0.0778,
        "vmax": 0.0778,
        "center_zero": True,
        "title": "χ Misorientation",
        "cbar_label": "Misorientation (°)",
    }
    return prep, {"name": "oblique_full"}, data, u, v


def _drawn(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    return canvas.get_renderer()


def test_styled_figure_layout_flag():
    assert styled_figure((4, 3), styled=True).get_layout_engine() is not None
    assert styled_figure((4, 3), styled=False).get_layout_engine() is None


def test_slice_figure_legacy_path_has_no_layout_engine():
    prep, sl, data, u, v = _slice_fixture()
    fig = build_slice_figure(prep, sl, data, u, v, offset_um=None, style=None)
    assert fig.get_layout_engine() is None


def test_slice_figure_texts_do_not_overlap_at_publication_scale():
    prep, sl, data, u, v = _slice_fixture()
    fig = build_slice_figure(prep, sl, data, u, v, offset_um=194.0, style=_BIG)
    renderer = _drawn(fig)
    ax, cax = fig.axes[0], fig.axes[1]

    title_bb = ax.title.get_window_extent(renderer)
    cbar_bb = cax.get_tightbbox(renderer)  # includes ticks, label AND the ×10ⁿ offset text
    image_bb = ax.bbox

    # The three failures visible in the bug report:
    assert not title_bb.overlaps(cbar_bb), "title collides with colorbar"
    assert not title_bb.overlaps(image_bb), "title collides with the map"
    cb_label_bb = cax.yaxis.label.get_window_extent(renderer)
    for tick in cax.yaxis.get_ticklabels():
        assert not cb_label_bb.overlaps(
            tick.get_window_extent(renderer)
        ), "colorbar label collides with its tick labels"


def test_slice_figure_keeps_exact_single_column_width():
    prep, sl, data, u, v = _slice_fixture()
    fig = build_slice_figure(prep, sl, data, u, v, offset_um=194.0, style=_BIG)
    assert fig.get_size_inches()[0] == 3.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_figure_layout.py -q`
Expected: FAIL — `ImportError: cannot import name 'styled_figure'` (and, once that exists, the overlap assertions fail against the current fixed-margin layout)

- [ ] **Step 3: Implement `styled_figure`**

In `dfxm/common/plotting.py`, directly after `new_figure`:

```python
def styled_figure(figsize: tuple[float, float], *, styled: bool) -> Figure:
    """A white-background Figure for the shared figure builders.

    ``styled=True`` (a PlotStyle is in play) uses matplotlib's constrained
    layout, which measures every text element at its final font size and
    reserves space so title, axis labels, colorbar and offset text can never
    overlap — the figure keeps its exact width and the axes shrink instead.
    ``styled=False`` is the legacy path: plain fixed margins, byte-identical
    with the pre-export renderers.
    """
    if styled:
        return Figure(figsize=figsize, facecolor="white", layout="constrained")
    return Figure(figsize=figsize, facecolor="white")
```

- [ ] **Step 4: Switch the construction sites**

`dfxm/stages/slices.py` (~751), inside `build_slice_figure` (`use_legacy = style is None` already exists just above):

```python
    fig = styled_figure(figsize, styled=not use_legacy)
```

and extend the existing `from ..common.plotting import (...)` block with `styled_figure`.

`dfxm/common/render.py` (~49), inside `layer_figure`:

```python
    fig = styled_figure(figsize, styled=style is not None)
```

adding `styled_figure` to the existing `from .plotting import (...)` block.

`dfxm/stages/profiles.py` (~446), `build_companion_figure` — replace the direct constrained call for consistency:

```python
    fig = styled_figure((9.0, 4.8 + 1.85 * n), styled=True)
```

Note: the companion figure is constrained on BOTH paths today (`layout="constrained"` is unconditional at line 446) — keep that by passing `styled=True` unconditionally; do not gate it on `style`.

`dfxm/stages/profiles.py` (~505), `render_single`:

```python
    fig = styled_figure((11, 9), styled=style is not None)
```

and add `styled_figure` to the profiles import from `..common.plotting`.

`dfxm/stages/strain.py` (~380), `build_strain_map`:

```python
    fig = styled_figure(figsize, styled=style is not None)
```

`dfxm/stages/strain.py` (~437), `build_detrend_diag`:

```python
    fig = styled_figure((20, 6), styled=style is not None)
```

and swap `new_figure` for `styled_figure` in the strain import block (keep `new_figure` imported only if still used elsewhere in the module — check with grep; if unused, remove it from the import).

`dfxm/common/plotting.py` `build_histogram` (~380):

```python
    fig = styled_figure(figsize, styled=style is not None)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_figure_layout.py -q`
Expected: PASS (4 tests)

- [ ] **Step 6: Full suite (fidelity guard)**

Run: `python3 -m pytest -q`
Expected: all pass. If a test in `tests/test_export_fidelity.py` or a stage test pins exact styled-figure geometry (margins/positions), inspect it: legacy-path assertions must still pass untouched; styled-path geometry assertions may legitimately change with constrained layout — update only those, stating why in the commit.

- [ ] **Step 7: Commit**

```bash
git add dfxm/common/plotting.py dfxm/common/render.py dfxm/stages/slices.py \
        dfxm/stages/profiles.py dfxm/stages/strain.py tests/test_figure_layout.py
git commit -m "fix(figures): constrained layout for all styled figures — no text overlap at any font scale"
```

---

### Task 4: Wire `round_clim` into slices (log + summary + HDF5) and strain

**Files:**
- Modify: `dfxm/stages/slices.py` (`SlicesResult` ~398, `prepare_volume` ~669–711, `write_volume_group` ~784, `run` ~900)
- Modify: `dfxm/stages/strain.py:370` (`build_strain_map`)
- Modify: `gui/stage_view.py` (`_summarize_slices` ~599)
- Test: `tests/test_stage_slices.py`, `tests/test_stage_summaries.py`

**Interfaces:**
- Consumes: `apply_round_clim(vmin, vmax, style) -> (vmin, vmax, note | None)` from Task 2.
- Produces: `prep` dict gains `"vmin_raw"`, `"vmax_raw"` (floats) and `"clim_note"` (str | None); `SlicesResult` gains `notes: list[str]`; slice HDF5 volume groups gain `vmin_raw`/`vmax_raw` attrs when rounding changed the limits.

- [ ] **Step 1: Write the failing tests**

In `tests/test_stage_slices.py` (reusing the existing `_setup(tmp_path)` fixture at line 92 — its volumes are `standard_normal` random data, so auto limits are essentially never already round and rounding WILL fire):

```python
def test_run_round_clim_rounds_notes_and_h5_attrs(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    params = {
        "mosa_volume_file": str(proc / "stacked_volumes.h5"),
        "strain_volume_file": str(proc / "stacked_strain_volumes.h5"),
        "raw_root": str(raw),
        "mosa_pattern": "mosa__*",
        "strain_pattern": "strain__*",
        "slices_json": slices_json,
        "output_dir": str(out),
        "plot_style": {"round_clim": True},
    }
    res = S.run(params)
    assert res.notes and all("rounded" in n for n in res.notes)
    with h5py.File(res.output_h5, "r") as f:
        for note in res.notes:
            vid = note.split(":")[0]
            vg = f[vid]
            assert "vmin_raw" in vg.attrs and "vmax_raw" in vg.attrs
            # final limits enclose the raw ones (outward rounding never clips)
            assert vg.attrs["vmin"] <= vg.attrs["vmin_raw"]
            assert vg.attrs["vmax"] >= vg.attrs["vmax_raw"]


def test_run_without_round_clim_has_no_notes_or_raw_attrs(tmp_path):
    proc, raw = _setup(tmp_path)
    out = tmp_path / "sl"
    slices_json = (
        '[{"name":"mid","normal":[0,0,1],"origin":[0.5,0.5,1.5],'
        '"half_u":0.4,"half_v":0.4,"du":0.2,"dv":0.2,"sweep_step_um":null}]'
    )
    res = S.run(
        {
            "mosa_volume_file": str(proc / "stacked_volumes.h5"),
            "raw_root": str(raw),
            "mosa_pattern": "mosa__*",
            "slices_json": slices_json,
            "output_dir": str(out),
        }
    )
    assert res.notes == []
    with h5py.File(res.output_h5, "r") as f:
        for vid in res.volume_ids:
            assert "vmin_raw" not in f[vid].attrs
```

In `tests/test_stage_summaries.py` (this file constructs the REAL result dataclasses and calls the shared `_summarize(name, result)` helper — follow that):

```python
def test_summarize_slices_shows_clim_notes():
    result = SlicesResult(
        output_h5="slices.h5",
        volume_ids=["mosa_com_chi"],
        slice_names=["oblique_full"],
        n_planes_total=3,
        notes=["mosa_com_chi: colour limits rounded ±0.0778 → ±0.08 (round_clim)"],
    )
    text = _summarize("slices", result)
    assert "rounded ±0.0778 → ±0.08" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_slices.py tests/test_stage_summaries.py -q`
Expected: FAIL — `KeyError: 'vmin_raw'` / missing `notes` attribute handling

- [ ] **Step 3: Implement — slices**

`SlicesResult` (~398) gains:

```python
    notes: list[str] = field(default_factory=list)
```

`prepare_volume` return block (~693): compute rounding just before building the dict —

```python
    vmin_f, vmax_f, clim_note = apply_round_clim(float(auto_vmin), float(auto_vmax), style)
    return {
        "data": np.ascontiguousarray(data, dtype=np.float64),
        ...existing keys...
        "vmin": vmin_f,
        "vmax": vmax_f,
        "vmin_raw": float(auto_vmin),
        "vmax_raw": float(auto_vmax),
        "clim_note": clim_note,
        ...rest unchanged...
    }
```

(add `apply_round_clim` to the slices import from `..common.plotting`).

`run` (~903), right after `prep = prepare_volume(...)` succeeds:

```python
            if prep["clim_note"]:
                msg = f"{prep['volume_id']}: {prep['clim_note']}"
                progress(0.1 + 0.85 * vi / len(volumes), msg)
                result.notes.append(msg)
```

`write_volume_group` (~784), next to the existing `vg.attrs["vmin"]`/`vmax` writes:

```python
    if prep.get("clim_note"):
        vg.attrs["vmin_raw"] = float(prep["vmin_raw"])
        vg.attrs["vmax_raw"] = float(prep["vmax_raw"])
```

`gui/stage_view.py` `_summarize_slices` — after the `volume_ids` lines:

```python
    lines += [f"  {n}" for n in getattr(result, "notes", [])]
```

- [ ] **Step 4: Implement — strain map (auto path only)**

`dfxm/stages/strain.py:370` in `build_strain_map`, replace:

```python
    vmin, vmax = vlim if vlim != (None, None) else symmetric_limits(strain)
```

with:

```python
    if vlim != (None, None):
        vmin, vmax = vlim  # user-specified limits are never rounded
    else:
        vmin, vmax = symmetric_limits(strain)
        vmin, vmax, _ = apply_round_clim(vmin, vmax, style)
```

(add `apply_round_clim` to the strain import from `..common.plotting`). The rounding is visible on the colourbar itself; the per-layer strain diagnostics have no run-log line (documented in Task 7).

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_slices.py tests/test_stage_summaries.py tests/test_stage_strain.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add dfxm/stages/slices.py dfxm/stages/strain.py gui/stage_view.py \
        tests/test_stage_slices.py tests/test_stage_summaries.py
git commit -m "feat(slices,strain): round_clim wiring — rounded limits logged, summarized and stored raw in HDF5"
```

---

### Task 5: Wire `round_clim` into visualize, rocking and matched

**Files:**
- Modify: `dfxm/stages/visualize.py` (`run` ~496–501 and ~533)
- Modify: `dfxm/stages/rocking.py` (`_render` ~543–550)
- Modify: `dfxm/stages/matched.py` (result dataclass + ~384–403)
- Modify: `gui/stage_view.py` (`_summarize_matched` ~626)
- Test: `tests/test_stage_summaries.py`

**Interfaces:**
- Consumes: `apply_round_clim` (Task 2); `DatasetProducts.notes` / `RockingProducts.notes` (existing fields, already printed by `_dataset_lines`).
- Produces: `MatchedResult` gains `vmin_raw: float | None = None`, `vmax_raw: float | None = None`; matched summary shows `(rounded from (a, b))`.

- [ ] **Step 1: Write the failing test (matched summary)**

In `tests/test_stage_summaries.py` (real dataclass + shared `_summarize` helper, matching the file's existing matched tests at ~line 120):

```python
def test_summarize_matched_shows_rounded_clim():
    result = MatchedResult(
        layers_dir="/out/rocking_layers",
        n_strain=5,
        n_matched=5,
        n_saved=5,
        frame_index=42,
        max_match_dist_um=0.5,
        vmin=0.0,
        vmax=1500.0,
        vmin_raw=3.2,
        vmax_raw=1487.0,
    )
    text = _summarize("matched", result)
    assert "clim=(0, 1500)" in text
    assert "rounded from (3.2, 1487)" in text


def test_summarize_matched_without_rounding_has_no_rounded_text():
    result = MatchedResult(
        layers_dir="/out/rocking_layers", n_strain=3, n_matched=2, n_saved=2, vmin=0.0, vmax=1.0
    )
    assert "rounded from" not in _summarize("matched", result)
```

(If `MatchedResult` has no `frame_index`/`max_match_dist_um` defaults matching these names, mirror the constructor arguments the existing tests at lines 120–150 use.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_summaries.py -q -k matched`
Expected: FAIL — summary has no "rounded from" text

- [ ] **Step 3: Implement — visualize**

Both clim sites in `run` get the same three lines. Mosaicity loop (~496–501) becomes:

```python
            if "Center_of_mass" in name:
                data, vmin, vmax = _center_com_and_range(
                    data, p["center_method"], float(p["range_pct"])
                )
            else:
                vmin, vmax = _colorbar_range(data)
            vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
            if clim_note:
                progress(0.1 + 0.4 * i / max(1, len(datasets)), f"{name}: {clim_note}")
            prod = _process_dataset(
                data, z_pos, scale_z, name, vmin, vmax, cmap, title, cbar, p, out_dir, style=style
            )
            if clim_note:
                prod.notes.append(clim_note)
            result.datasets.append(prod)
```

Strain block (~533) analogously:

```python
            vmin, vmax = _symmetric_range(data)
            vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
            if clim_note:
                progress(0.6, f"strain: {clim_note}")
            prod = _process_dataset(
                data, z_pos, scale_z, "strain", vmin, vmax, cmap, title, cbar, p, out_dir, style=style
            )
            if clim_note:
                prod.notes.append(clim_note)
            result.datasets.append(prod)
```

(add `apply_round_clim` to the visualize import from `..common.plotting`). The stored `DatasetProducts.vmin/vmax` are then the rounded values, so the figure-export rebuild path (~754, which reuses `ds.vmin/ds.vmax`) stays consistent automatically. Do NOT touch `aligned_field` (~602–626) — it feeds the interactive 3-D viewer, not exports.

- [ ] **Step 4: Implement — rocking**

`_render` (~547):

```python
    vmin, vmax = _colorbar_range(vol, float(p["cbar_pct_lo"]), float(p["cbar_pct_hi"]))
    vmin, vmax, clim_note = apply_round_clim(vmin, vmax, style)
    ...
    prod = RockingProducts(name=name, vmin=vmin, vmax=vmax)
    if clim_note:
        prod.notes.append(clim_note)
```

(add `apply_round_clim` to the rocking import from `..common.plotting`; `notes` are already shown by `_dataset_lines` in the Results tab).

- [ ] **Step 5: Implement — matched**

In the `MatchedResult` dataclass (grep `class MatchedResult` in `dfxm/stages/matched.py`), after `vmin`/`vmax` add:

```python
    vmin_raw: float | None = None
    vmax_raw: float | None = None
```

At ~384–403: round ONLY when both limits were auto-computed (a half-manual pair must not be half-rounded):

```python
    vmin_user, vmax_user = _parse_float(p["vmin"]), _parse_float(p["vmax"])
    vmin, vmax = vmin_user, vmax_user
    if vmin is None or vmax is None:
        ...existing pooling code unchanged...
    result.vmin, result.vmax = float(vmin), float(vmax)
    if vmin_user is None and vmax_user is None:
        rlo, rhi, clim_note = apply_round_clim(result.vmin, result.vmax, style)
        if clim_note:
            result.vmin_raw, result.vmax_raw = result.vmin, result.vmax
            result.vmin, result.vmax = rlo, rhi
            vmin, vmax = rlo, rhi  # the loop below renders with vmin/vmax
            progress(0.2, clim_note)
```

(add `apply_round_clim` to the matched import from `..common.plotting`).

`gui/stage_view.py` `_summarize_matched` — extend the clim line:

```python
        clim = f"clim=({result.vmin:.4g}, {result.vmax:.4g})"
        if getattr(result, "vmin_raw", None) is not None:
            clim += f" (rounded from ({result.vmin_raw:.4g}, {result.vmax_raw:.4g}))"
        lines = [
            f"output: {result.layers_dir}",
            f"matched {result.n_matched}/{result.n_strain}, saved {result.n_saved} "
            f"(frame {result.frame_index})",
            f"max match dist: {result.max_match_dist_um:.3f} µm   {clim}",
        ]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_summaries.py tests/test_stage_visualize.py tests/test_stage_rocking.py tests/test_stage_matched.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add dfxm/stages/visualize.py dfxm/stages/rocking.py dfxm/stages/matched.py \
        gui/stage_view.py tests/test_stage_summaries.py
git commit -m "feat(visualize,rocking,matched): round_clim wiring with notes in log and results"
```

---

### Task 6: GUI — Title scale + Round colour limits controls, self-explanatory tick formats

**Files:**
- Modify: `gui/widgets/export_dialog.py` (`StyleControls`: `_build_controls` ~167, `sync_from_style` ~63, `_all_widgets` ~131; `_TICK_FMTS` ~37)
- Test: `tests/gui_smoke.py` (extend check [16])

**Interfaces:**
- Consumes: `PlotStyle.title_scale`, `PlotStyle.round_clim` (Tasks 1–2).
- Produces: two new widgets `self._w_title_scale`, `self._w_round_clim`; the tick-format combo stores the format VALUE in `Qt userData` (`currentData()`), display text becomes descriptive.

- [ ] **Step 1: Add the controls**

In `_build_controls`, Text section, directly after the "Show title" row (~319):

```python
        self._w_title_scale = QDoubleSpinBox()
        self._w_title_scale.setRange(0.1, 5.0)
        self._w_title_scale.setDecimals(2)
        self._w_title_scale.setSingleStep(0.1)
        self._w_title_scale.setValue(s.title_scale)
        self._w_title_scale.setToolTip(
            "Size of the title alone, independent of Font scale — set small if the "
            "title is only there to identify the plot."
        )
        self._w_title_scale.valueChanged.connect(
            lambda v: (setattr(self._style, "title_scale", v), self._emit())
        )
        form.addRow("Title scale", self._w_title_scale)
```

Colourbar section, after the "Tick format" row (~376):

```python
        self._w_round_clim = QCheckBox()
        self._w_round_clim.setChecked(s.round_clim)
        self._w_round_clim.setToolTip(
            "Round the automatic colour limits outward to nice values (e.g. ±0.0778 → "
            "±0.08) so evenly spaced ticks are round numbers. The run log and Results "
            "tab state exactly what was rounded."
        )
        self._w_round_clim.toggled.connect(
            lambda v: (setattr(self._style, "round_clim", v), self._emit())
        )
        form.addRow("Round colour limits", self._w_round_clim)
```

- [ ] **Step 2: Descriptive tick-format labels**

Replace the module constant (~37):

```python
_TICK_FMTS = ["auto", "scientific", "0", "1", "2", "3"]
_TICK_FMT_LABELS = {
    "auto": "auto (matplotlib default)",
    "scientific": "scientific (×10ⁿ offset)",
    "0": "0 decimals (plain numbers)",
    "1": "1 decimal (plain numbers)",
    "2": "2 decimals (plain numbers)",
    "3": "3 decimals (plain numbers)",
}
```

In `_build_controls` (~368) replace the combo population and wiring:

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

In `sync_from_style` (~107) replace the `setCurrentText` call:

```python
        cur_fmt = s.colorbar_tick_format if s.colorbar_tick_format in _TICK_FMTS else "auto"
        self._w_cbar_fmt.setCurrentIndex(_TICK_FMTS.index(cur_fmt))
```

and add the two new widgets to `sync_from_style`:

```python
        self._w_font_scale.setValue(s.font_scale)
        self._w_title_scale.setValue(s.title_scale)
        self._w_show_title.setChecked(s.show_title)
        ...
        self._w_round_clim.setChecked(s.round_clim)
```

and to `_all_widgets` (insert `self._w_title_scale` after `self._w_font_scale`, `self._w_round_clim` after `self._w_cbar_fmt`).

- [ ] **Step 3: Extend gui_smoke check [16]**

In `tests/gui_smoke.py` around line 430, after the existing `StyleControls` mutation assertions, add:

```python
    sc._w_title_scale.setValue(0.4)
    assert session_style.title_scale == 0.4, "Title scale widget did not mutate the style"
    sc._w_round_clim.setChecked(True)
    assert session_style.round_clim is True, "Round colour limits widget did not mutate the style"
    sc._w_cbar_fmt.setCurrentIndex(2)  # "0 decimals (plain numbers)"
    assert session_style.colorbar_tick_format == "0", "Tick-format combo must store the format value"
```

- [ ] **Step 4: Run the smoke + suite**

Run: `python3 tests/gui_smoke.py` — expected: all numbered checks pass, including the extended [16].
Run: `python3 -m pytest -q` — expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add gui/widgets/export_dialog.py tests/gui_smoke.py
git commit -m "feat(gui): Title scale + Round colour limits controls; descriptive tick-format labels"
```

---

### Task 7: Documentation (Usage.md + Codebase.md)

**Files:**
- Modify: `docs/Usage.md` (export-dialog / publication-style section)
- Modify: `docs/Codebase.md` (`dfxm/common/plotting`, `dfxm/common/render`, touched stage sections)

**Interfaces:** none (documentation contract from CLAUDE.md — this completes Tasks 1–6; the code changes are incomplete without it).

- [ ] **Step 1: Update `docs/Usage.md`**

In the section that documents the export dialog / plot style controls (grep for "Font scale" in the file), add rows/paragraphs covering, in the same voice as the surrounding text:

- **Title scale** — sizes the title independently of Font scale; set it small (e.g. 0.3) when the title is only needed to identify the plot while composing figures elsewhere.
- **Round colour limits** — rounds automatic colour limits outward to nice values (±0.0778 → ±0.08) so evenly spaced colourbar ticks are round; the run log and Results tab state exactly what was rounded (and the slices HDF5 keeps `vmin_raw`/`vmax_raw`); manual limits (e.g. matched's vmin/vmax, strain's vmin/vmax) are never rounded.
- **Tick format** — a short "Why does my colourbar say ×10⁻²?" note: `scientific` forces offset notation; the digit-count formats print plain decimals; `auto` is matplotlib's default.
- A note under the styled-export description: styled figures now use matplotlib constrained layout, so titles/labels/colourbars cannot overlap at any font scale; the figure keeps its exact single/double-column width and the map shrinks to make room instead.

- [ ] **Step 2: Update `docs/Codebase.md`**

In the `dfxm/common/plotting` section: document `title_scale` and `round_clim` fields, `styled_figure`, `round_limits_outward`, `apply_round_clim`, and the changed `apply_text_scale` contract (title scales by `title_scale` only). In the stage sections for slices/visualize/rocking/matched/strain: one line each on where `round_clim` applies and how it is reported (`prep["clim_note"]` + `SlicesResult.notes` + HDF5 `vmin_raw`/`vmax_raw`; `DatasetProducts.notes`/`RockingProducts.notes`; `MatchedResult.vmin_raw`/`vmax_raw`; strain rounds in-figure on the auto path with no log line).

- [ ] **Step 3: Commit**

```bash
git add docs/Usage.md docs/Codebase.md
git commit -m "docs: title scale, round colour limits, tick formats, constrained layout"
```

---

### Task 8: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `python3 -m pytest -q`
Expected: ≥ 290 passed / 13 skipped, 0 failed.

- [ ] **Step 2: Lint + format**

Run: `ruff check . && ruff format --check .`
Expected: no findings.

- [ ] **Step 3: GUI smoke**

Run: `python3 tests/gui_smoke.py`
Expected: all numbered checks green.

- [ ] **Step 4: Visual proof (the actual bug)**

Render a slice figure with the bug-report style and confirm by eye that nothing overlaps and the title is small:

```bash
python3 - <<'EOF'
import numpy as np
from dfxm.common.plotting import PlotStyle
from dfxm.stages.slices import build_slice_figure

style = PlotStyle(font_scale=2.2, title_scale=0.5, figure_width="single",
                  colorbar_ticks=5, colorbar_tick_format="2", round_clim=True,
                  scale_bar=True, scale_bar_box=True, scale_bar_color="white")
u = np.linspace(-200, 200, 80); v = np.linspace(-120, 120, 50)
data = np.outer(np.linspace(-0.0778, 0.0778, 50), np.ones(80))
prep = {"cmap_name": "RdBu_r", "vmin": -0.08, "vmax": 0.08, "center_zero": True,
        "title": "χ Misorientation", "cbar_label": "Misorientation (°)"}
fig = build_slice_figure(prep, {"name": "oblique_full"}, data, u, v,
                         offset_um=194.0, style=style)
fig.savefig("/tmp/claude-1000/-home-albert-Desktop-dfxm-pipeline/bb967ef3-0c6d-4cf7-bbdf-f8b383545228/scratchpad/layout_check.png",
            dpi=150, facecolor="white", bbox_inches="tight")
print("wrote layout_check.png")
EOF
```

Send `layout_check.png` to the user for confirmation.

- [ ] **Step 5: Report**

Summarize suite counts, list the commits, and hand back for the user's manual test on real data (real slice PNGs, animations, PDF/SVG exports).
