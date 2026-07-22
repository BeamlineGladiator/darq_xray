# Axis-Free Plot Style Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an `axes_mode` publication-style knob (`full` / `no_frame` / `none`) so styled map figures can be exported without the plot frame, or without any axis decoration at all.

**Architecture:** One new `PlotStyle` field + a small `apply_axes_mode(ax, style)` helper in `dfxm/common/plotting.py`, called at exactly four styled-map-builder sites (the shared `render.layer_figure` covers mosaicity/rocking/visualize/matched/replots/animations at one stroke; plus the strain, slices and profiles map builders). One `QComboBox` in `StyleControls`. Serialization/persistence/stage-injection come free from the existing dataclass machinery.

**Tech Stack:** Python 3.10, matplotlib (`Figure` API only — never pyplot), PySide6 (GUI layer only), pytest.

**Spec:** `docs/superpowers/specs/2026-07-22-axis-free-style-design.md` (approved).

## Global Constraints

- `dfxm/` stays Qt-free: no PySide6 imports anywhere under `dfxm/`.
- Never use `pyplot` or `matplotlib.use(...)` — explicit `Figure` API only.
- `"full"` (and any stale/unknown persisted value) must be a byte-exact no-op: default output must not change. `PUBLICATION_STYLE` keeps `axes_mode="full"`.
- The mode applies to **map axes only**. Never call `apply_axes_mode` on: profiles trace figures, `build_companion_figure` (its map panel included — it shares the distance frame with the traces below it), `build_histogram`/`build_strain_histogram`, `build_detrend_diag`, or any legacy `style=None`-branch code path.
- Docs contract: `docs/Codebase.md` updates land in the same commit as the code they describe; `docs/Usage.md` lands with the GUI task (Task 3) where the feature becomes user-visible.
- Ruff: line length 100, double quotes (`ruff format` runs on Write/Edit via hook).
- Read every file before its first Edit (session rule); never reconstruct `old_string` from this plan — the plan's "existing code" excerpts are for orientation, always re-read the real bytes.
- Run `python3 -m pytest -q` (full suite) before each task's commit; the GUI smoke test is `python3 tests/gui_smoke.py` (not a pytest file).

---

### Task 1: Core — `axes_mode` field + `apply_axes_mode` helper

**Files:**
- Modify: `dfxm/common/plotting.py` (field in `PlotStyle` ~line 82; helper after `apply_text_scale`, ~line 538)
- Test: `tests/test_plot_style.py`
- Modify: `docs/Codebase.md` (`plotting.py` section, ~line 246–278)

**Interfaces:**
- Consumes: existing `PlotStyle`, `style_to_json`, `style_from_json`.
- Produces: `PlotStyle.axes_mode: str = "full"` (values `"full" | "no_frame" | "none"`) and `apply_axes_mode(ax, style) -> None`, importable from `dfxm.common.plotting`. Tasks 2 and 3 rely on exactly these names.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_plot_style.py` (add `apply_axes_mode` to the existing `from dfxm.common.plotting import (...)` block, keeping it alphabetical; add `import json` to the top-level imports — the file does not have it yet):

```python
def _bare_ax():
    fig = Figure()
    return fig.add_subplot(111)


def test_axes_mode_default_is_full():
    assert PlotStyle().axes_mode == "full"
    assert PUBLICATION_STYLE.axes_mode == "full"


def test_apply_axes_mode_no_frame_hides_spines_keeps_ticks():
    ax = _bare_ax()
    apply_axes_mode(ax, PlotStyle(axes_mode="no_frame"))
    assert all(not sp.get_visible() for sp in ax.spines.values())
    assert ax.axison  # ticks and labels survive


def test_apply_axes_mode_none_removes_axes():
    ax = _bare_ax()
    apply_axes_mode(ax, PlotStyle(axes_mode="none"))
    assert not ax.axison


def test_apply_axes_mode_full_and_stale_values_are_noops():
    for mode in ("full", "boxless", "", 0, None):
        ax = _bare_ax()
        apply_axes_mode(ax, replace(PlotStyle(), axes_mode=mode))
        assert ax.axison
        assert all(sp.get_visible() for sp in ax.spines.values())


def test_axes_mode_json_roundtrip_and_legacy_snapshot_default():
    assert style_from_json(style_to_json(PlotStyle(axes_mode="none"))).axes_mode == "none"
    # a persisted snapshot from before this feature has no axes_mode key
    assert style_from_json(json.dumps({"font_scale": 2.0})).axes_mode == "full"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/test_plot_style.py -q -k axes_mode`
Expected: FAIL / ERROR — `ImportError: cannot import name 'apply_axes_mode'`.

- [ ] **Step 3: Implement field + helper**

In `dfxm/common/plotting.py`, add the field to `PlotStyle` directly after `center_axis_labels: bool = True` (~line 82):

```python
    # axes decoration on MAP figures only: "full" (today's look) | "no_frame"
    # (spines hidden; ticks and labels stay) | "none" (spines, ticks and
    # labels all removed — scale bar + colorbar carry the context)
    axes_mode: str = "full"
```

Add the helper directly after `apply_text_scale` (after its closing line, ~line 538):

```python
def apply_axes_mode(ax, style: "PlotStyle") -> None:
    """Hide map-axes decoration per ``style.axes_mode``.

    ``"no_frame"`` hides the four spines (ticks and labels stay); ``"none"``
    removes spines, ticks and labels entirely; ``"full"`` — or any
    stale/unknown persisted value — is a no-op (defensive, like
    :func:`fixed_scale`; never raises). Map axes only: callers must not apply
    this to trace/companion/histogram/diagnostic axes.
    """
    mode = getattr(style, "axes_mode", "full")
    if mode == "no_frame":
        for spine in ax.spines.values():
            spine.set_visible(False)
    elif mode == "none":
        ax.set_axis_off()
```

`PUBLICATION_STYLE` is not touched (it inherits the `"full"` default). `_style_from_dict`/`style_to_json` need no change — dataclass fields serialize automatically and missing keys default.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_plot_style.py -q`
Expected: all pass (new tests + no regressions in the file).

- [ ] **Step 5: Update `docs/Codebase.md`**

Read the `plotting.py` section (~line 246–278). In the function table, add a row directly below the `apply_text_scale` row:

```markdown
| `apply_axes_mode(ax, style)` | Hide map-axes decoration per `style.axes_mode`: `"no_frame"` hides the four spines (ticks/labels stay), `"none"` calls `ax.set_axis_off()`; `"full"` or any stale/unknown persisted value is a no-op (defensive, like `fixed_scale`). Map axes only — never applied to trace, companion, histogram or diagnostic axes. |
```

Where the section describes the `PlotStyle` dataclass, mention the new field: `axes_mode` (`"full"`/`"no_frame"`/`"none"`, default `"full"`) — axes decoration on map figures.

- [ ] **Step 6: Full suite + commit**

Run: `python3 -m pytest -q` — expected: same pass/skip counts as master plus the new tests, 0 failures. Then:

```bash
git add dfxm/common/plotting.py tests/test_plot_style.py docs/Codebase.md
git commit -m "feat(plotting): PlotStyle.axes_mode + apply_axes_mode helper"
```

---

### Task 2: Wire the four map-builder call sites

**Files:**
- Modify: `dfxm/common/render.py` (`layer_figure`, ~line 79)
- Modify: `dfxm/stages/strain.py` (`build_strain_map` styled branch, ~line 428)
- Modify: `dfxm/stages/slices.py` (`build_slice_figure` styled branch, ~line 872)
- Modify: `dfxm/stages/profiles.py` (`render_single` styled branch, ~line 895)
- Test: `tests/test_figures_catalog.py`, `tests/test_stage_strain.py`, `tests/test_figure_layout.py`
- Modify: `docs/Codebase.md` (render/strain/slices/profiles entries)

**Interfaces:**
- Consumes: `apply_axes_mode(ax, style)` from Task 1 (import from `..common.plotting` in stages, `.plotting` in render).
- Produces: no new names — behaviour only. Every styled map figure honours `style.axes_mode`; excluded builders (`build_companion_figure`, `build_histogram`, `build_strain_histogram`, `build_detrend_diag`, trace figures) are untouched.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_figures_catalog.py`, next to the existing `layer_figure` tests (~line 26–40; `render`, `_layer()` and `PlotStyle` are already available in that file — add any missing import to the existing import block):

```python
def test_layer_figure_axes_mode_no_frame_hides_spines():
    fig, ax, _ = render.layer_figure(
        _layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb", style=PlotStyle(axes_mode="no_frame")
    )
    assert all(not sp.get_visible() for sp in ax.spines.values())
    assert ax.axison


def test_layer_figure_axes_mode_none_removes_axes_but_not_colorbar():
    fig, ax, _ = render.layer_figure(
        _layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb", style=PlotStyle(axes_mode="none")
    )
    assert not ax.axison
    cax = [a for a in fig.axes if a is not ax][0]
    assert cax.axison  # colorbar untouched


def test_layer_figure_legacy_and_full_keep_axes():
    for style in (None, PlotStyle()):
        fig, ax, _ = render.layer_figure(
            _layer(), -1, 1, "viridis", 40.0, 20.0, "t", "cb", style=style
        )
        assert ax.axison
        assert all(sp.get_visible() for sp in ax.spines.values())


def test_companion_figure_keeps_axes_under_axes_mode_none():
    """The companion (map panel + traces) is excluded from axes_mode by design."""
    ref, fields, geom = _companion_fixture()
    fig = Profiles.build_companion_figure(
        ref, fields, geom, "cyan", style=PlotStyle(axes_mode="none")
    )
    content_axes = [a for a in fig.axes]
    assert all(a.axison for a in content_axes)
```

(`_companion_fixture()` exists at ~line 1411 of that file and returns `(ref, fields, geom)`; `Profiles` is the already-imported stage module alias used by the neighbouring companion tests — reuse whatever alias that file uses.)

Append to `tests/test_stage_strain.py` (match its existing imports; it already tests `build_strain_map`):

```python
def test_build_strain_map_axes_mode_none_map_only():
    strain = np.linspace(-1e-4, 1e-4, 400).reshape(20, 20)
    fig = build_strain_map(strain, 0.5, 0.5, None, (None, None), style=PlotStyle(axes_mode="none"))
    assert not fig.axes[0].axison  # the map
    assert fig.axes[1].axison  # its colorbar

    diag = build_detrend_diag(strain, strain, strain, style=PlotStyle(axes_mode="none"))
    assert all(a.axison for a in diag.axes)  # diagnostic excluded by design
```

Append to `tests/test_plot_style.py`:

```python
def test_histogram_keeps_axes_under_axes_mode_none():
    fig = build_histogram(
        np.linspace(-1.0, 1.0, 100),
        title="t",
        xlabel="x",
        style=PlotStyle(axes_mode="none"),
    )
    assert fig.axes[0].axison
```

Append to `tests/test_figure_layout.py` (its `_slice_fixture()` and `_box_inches()` helpers already exist; model is `test_layer_figure_fixed_scale_equal_boxes_across_decoration_loads`, ~line 163):

```python
def test_slice_figure_axes_mode_none_removes_axes():
    prep, sl, data, u, v = _slice_fixture()
    fig = build_slice_figure(
        prep, sl, data, u, v, offset_um=None, style=PlotStyle(axes_mode="none")
    )
    assert not fig.axes[0].axison


def test_layer_figure_fixed_scale_box_unchanged_by_axes_mode():
    import numpy as np

    from dfxm.common import render

    layer = np.random.default_rng(3).random((10, 20))
    boxes = []
    for mode in ("full", "no_frame", "none"):
        style = PlotStyle(scale_um_per_cm=50.0, axes_mode=mode)
        fig, ax, _ = render.layer_figure(
            layer, 0.0, 1.0, "gray", 200.0, 100.0, "t", "I (a.u.)", style=style, group="raw"
        )
        boxes.append(_box_inches(fig, ax))
    target_w, target_h = 200.0 / 50.0 / 2.54, 100.0 / 50.0 / 2.54
    for w, h in boxes:
        assert abs(w - target_w) <= 0.05 and abs(h - target_h) <= 0.05
```

- [ ] **Step 2: Run the new tests to verify the inclusion tests fail**

Run: `python3 -m pytest tests/test_figures_catalog.py tests/test_stage_strain.py tests/test_figure_layout.py tests/test_plot_style.py -q -k "axes_mode or keeps_axes"`
Expected: the `layer_figure`/`strain_map`/`slice_figure` axes-mode tests FAIL (axes still on); the exclusion tests (companion/histogram/diag/legacy) already PASS — they pin behaviour that must not change.

- [ ] **Step 3: Add the four calls**

Each call goes immediately after the existing `apply_text_scale` call, styled branch only. Read each site first; the surrounding code shown here is orientation, not `old_string` material.

`dfxm/common/render.py` — add `apply_axes_mode` to the existing `from .plotting import (...)` block (alphabetical, before `apply_text_scale`). In `layer_figure` (~line 79):

```python
    apply_text_scale(ax, st)
    apply_axes_mode(ax, st)
```

(The legacy `style=None` path builds a default `st` with `axes_mode="full"`, so this is a no-op there — no gating needed.)

`dfxm/stages/strain.py` — add `apply_axes_mode` to the `from ..common.plotting import (...)` block. In `build_strain_map`'s `else` (styled) branch (~line 428):

```python
        add_colorbar(fig, im, ax, "Strain (ε)", style, group="strain")
        apply_text_scale(ax, style)
        apply_axes_mode(ax, style)
```

Do NOT touch `build_detrend_diag` (~line 488) or `build_strain_histogram`.

`dfxm/stages/slices.py` — add `apply_axes_mode` to the `from ..common.plotting import (...)` block. In `build_slice_figure` (~line 872):

```python
    if not use_legacy:
        apply_text_scale(ax, st)
        apply_axes_mode(ax, st)
```

`dfxm/stages/profiles.py` — add `apply_axes_mode` to the `from ..common.plotting import (...)` block. In `render_single`'s styled `else` branch (~line 895):

```python
        apply_text_scale(ax, style)
        apply_axes_mode(ax, style)
```

Do NOT touch `build_companion_figure` (~line 701 — neither its `ax_img` nor its trace axes) or the trace-figure code.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python3 -m pytest tests/test_figures_catalog.py tests/test_stage_strain.py tests/test_figure_layout.py tests/test_plot_style.py -q`
Expected: all pass.

- [ ] **Step 5: Update `docs/Codebase.md`**

Read the relevant entries and add, in the same style as the surrounding text:
- `render.py` → `layer_figure`: note it applies `apply_axes_mode` after `apply_text_scale`, so `axes_mode` reaches every styled volume-layer figure (mosaicity, rocking, visualize, matched, replot renders, animation frames); the legacy default-style path resolves to `"full"` (no-op).
- `strain.py` → `build_strain_map`: honours `style.axes_mode` (the detrend diagnostic and histogram deliberately do not).
- `slices.py` → `build_slice_figure`: honours `style.axes_mode` on the styled path.
- `profiles.py` → `render_single`: honours `style.axes_mode`; `build_companion_figure` and trace figures deliberately do not.

- [ ] **Step 6: Full suite + commit**

Run: `python3 -m pytest -q` — expected: 0 failures. Then:

```bash
git add dfxm/common/render.py dfxm/stages/strain.py dfxm/stages/slices.py dfxm/stages/profiles.py tests/test_figures_catalog.py tests/test_stage_strain.py tests/test_figure_layout.py tests/test_plot_style.py docs/Codebase.md
git commit -m "feat(figures): honour axes_mode in the four styled map builders"
```

---

### Task 3: GUI dropdown + smoke step + user docs

**Files:**
- Modify: `gui/widgets/export_dialog.py` (`StyleControls`: `_build_controls` Text section ~line 407, `sync_from_style` ~line 140, `_all_widgets` ~line 200; module constants near the top)
- Test: `tests/gui_smoke.py` (new step `[36]`)
- Modify: `docs/Usage.md` (style-controls list, ~line 1099 section), `docs/Codebase.md` (`export_dialog.py` entry, ~line 516)

**Interfaces:**
- Consumes: `PlotStyle.axes_mode` from Task 1; nothing from Task 2.
- Produces: `StyleControls._w_axes_mode` (`QComboBox` with display text Full / No frame / None and `itemData` values `"full"` / `"no_frame"` / `"none"`); module constant `_AXES_MODES = ("full", "no_frame", "none")`.

- [ ] **Step 1: Add the control**

Read `gui/widgets/export_dialog.py` first. Near the other module constants (`_CMAPS`/`_COLORS`/`_WIDTHS` etc.), add:

```python
_AXES_MODES = ("full", "no_frame", "none")
```

In `_build_controls`, after the "Centre axis labels" row (~line 407, still in the **Text** section):

```python
        self._w_axes_mode = QComboBox()
        for label, value in (("Full", "full"), ("No frame", "no_frame"), ("None", "none")):
            self._w_axes_mode.addItem(label, value)
        self._w_axes_mode.setCurrentIndex(
            self._w_axes_mode.findData(s.axes_mode if s.axes_mode in _AXES_MODES else "full")
        )
        self._w_axes_mode.setToolTip(
            "Axis decoration on map figures: 'No frame' hides the box around the plot "
            "(ticks and numbers stay); 'None' removes ticks, numbers and axis labels too — "
            "the scale bar and colourbar then carry the physical context. Trace, companion "
            "and diagnostic figures always keep their axes."
        )
        self._w_axes_mode.currentIndexChanged.connect(
            lambda i: (setattr(self._style, "axes_mode", self._w_axes_mode.itemData(i)), self._emit())
        )
        form.addRow("Axes", self._w_axes_mode)
```

In `sync_from_style`, after `self._w_center_labels.setChecked(s.center_axis_labels)` (~line 140):

```python
        self._w_axes_mode.setCurrentIndex(
            self._w_axes_mode.findData(s.axes_mode if s.axes_mode in _AXES_MODES else "full")
        )
```

In `_all_widgets`, add `self._w_axes_mode,` after `self._w_center_labels,` (~line 200).

- [ ] **Step 2: Add smoke step [36]**

Read the end of `tests/gui_smoke.py` (step `[35]` ends ~line 1071) and append a step following the file's local conventions (`_StyleControls` is already imported at step `[16]`):

```python
    # [36] Axes mode: StyleControls dropdown mutates axes_mode; sync restores; JSON round-trips
    from dfxm.common.plotting import PlotStyle as _PS36
    from dfxm.common.plotting import style_from_json as _sfj36
    from dfxm.common.plotting import style_to_json as _stj36

    s36 = _PS36()
    sc36 = _StyleControls(s36)
    idx36 = sc36._w_axes_mode.findData("none")
    assert idx36 >= 0, "Axes combo missing the 'none' entry"
    sc36._w_axes_mode.setCurrentIndex(idx36)
    assert s36.axes_mode == "none", "Axes combo did not mutate style.axes_mode"
    assert _sfj36(_stj36(s36)).axes_mode == "none", "axes_mode lost in JSON round-trip"
    s36.axes_mode = "no_frame"
    sc36.sync_from_style()
    assert sc36._w_axes_mode.currentData() == "no_frame", "sync_from_style did not restore combo"
    print("[36] axes-mode: dropdown mutates style + sync restores + JSON round-trip")
```

- [ ] **Step 3: Run the smoke test**

Run: `python3 tests/gui_smoke.py`
Expected: all steps print through `[36]`, exit 0.

- [ ] **Step 4: Update the docs**

`docs/Usage.md` — Read the shared style-controls section ("Both the global "Publication style…" editor and the per-figure **Export…** dialog offer the same controls:", ~line 1099) and add, in the Text group after Centre axis labels, matching the section's format:

> **Axes** — axis decoration on map figures: **Full** (default, today's look), **No frame** (hides the box/spines around the plot; ticks and numbers stay), or **None** (removes ticks, numbers and axis labels entirely — an axis-free map where the scale bar and colourbar carry the physical context, so consider keeping **Show scale bar** on). Applies to every styled map figure and all Replot… dialogs; profiles trace figures, the companion figure and diagnostic images always keep their axes. Headless CLI runs (no injected style) are unaffected.

`docs/Codebase.md` — in the `export_dialog.py` entry (~line 516), extend the `StyleControls` description: the **Text** group gains `_w_axes_mode` (combo, display Full / No frame / None ↔ `PlotStyle.axes_mode` values `full`/`no_frame`/`none` via `itemData`; `_AXES_MODES` guards stale persisted values back to `full` in sync).

- [ ] **Step 5: Full suite + commit**

Run: `python3 -m pytest -q` and `python3 tests/gui_smoke.py` — expected: 0 failures, smoke through `[36]`. Then:

```bash
git add gui/widgets/export_dialog.py tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): Axes mode dropdown in publication style controls + docs + smoke [36]"
```

---

## Verification after all tasks

- `python3 -m pytest -q` — full suite green (baseline on master: 612 passed / 13 skipped before this work).
- `ruff check . && ruff format --check .` — clean.
- `python3 tests/gui_smoke.py` — steps `[1]`–`[36]` pass.
- Manual (deferred to Albert's next real-data session): Publication style… → Axes = None → run/replot a map stage and eyeball the axis-free export; PDF/SVG included.
