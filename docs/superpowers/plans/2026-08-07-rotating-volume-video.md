# Rotating 3-D Volume Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional 360° orbiting movie of the pyvista volume render to the visualize stage (`<name>_rotation.mp4`/`.gif`), restoring the legacy "spin" video the Phase 2 port dropped.

**Architecture:** A GL-free frame→movie assembler (`_write_image_video`) plus an orbiting pyvista frame source (`save_rotation_video`) land in the shared renderer `dfxm/common/render.py`; the visualize stage gains one default-off bool param `save_rotation` and a `rotation_video` product field, wired with the same degrade-never-fail pattern as the top view. Spec: `docs/superpowers/specs/2026-08-07-rotating-volume-video-design.md`.

**Tech Stack:** numpy, matplotlib (Figure/Agg + FFMpegWriter→PillowWriter fallback), pyvista (lazy import), existing StageSpec schema, pytest.

## Global Constraints

- `dfxm/` stays Qt-free; pyvista imported ONLY inside the rendering function (lazy).
- Figures via explicit `matplotlib.figure.Figure` — never `pyplot`, never `matplotlib.use(...)`.
- `save_rotation` defaults to **False** (expensive render; CI has no guaranteed GL).
- MP4→GIF semantics identical to `save_layer_animation`: `fmt` in `("mp4", "gif", "both")`, ffmpeg failure falls back to GIF.
- Any pyvista/GL failure becomes a `DatasetProducts.notes` entry, never an exception.
- Docs (`docs/Usage.md` + `docs/Codebase.md`) updated in the SAME task as the code they describe.
- Every new Param has `help` written for a first-time beamline user; advanced params have `group` (`tests/test_param_metadata.py` enforces).
- Work on branch `rotating-volume-video`; ruff (line length 100, double quotes) auto-formats on Write/Edit.

---

### Task 1: Frame→movie assembler `_write_image_video` (render.py)

**Files:**
- Modify: `dfxm/common/render.py` (imports ~line 12-18; new function after `save_layer_animation`, ~line 196)
- Test: `tests/test_render_rotation.py` (create)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_write_image_video(get_frame, n_frames, base_path, fmt, fps=15) -> str | None` — `get_frame(i)` returns an `(H, W, 3)` uint8 RGB ndarray for frame *i*; returns the written path (`base_path + ".mp4"` or `".gif"`).

- [ ] **Step 1: Write the failing test**

```python
"""Rotation-video rendering — the GL-free assembly path and the empty-grid guard."""

from __future__ import annotations

import os

import numpy as np

from dfxm.common import render


def _gradient_frame(i):
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    frame[:, :, 0] = (i * 40) % 256
    frame[:, i % 48, 1] = 255
    return frame


def test_write_image_video_writes_gif(tmp_path):
    base = os.path.join(tmp_path, "spin")
    written = render._write_image_video(_gradient_frame, 4, base, "gif", fps=5)
    assert written == base + ".gif"
    assert os.path.getsize(written) > 0


def test_write_image_video_both_prefers_mp4_or_falls_back(tmp_path):
    base = os.path.join(tmp_path, "spin")
    written = render._write_image_video(_gradient_frame, 3, base, "both", fps=5)
    # mp4 when ffmpeg is on PATH, else the GIF fallback; either way a file exists
    assert written in (base + ".mp4", base + ".gif")
    assert os.path.getsize(base + ".gif") > 0 or written == base + ".mp4"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_render_rotation.py -v`
Expected: FAIL — `AttributeError: module 'dfxm.common.render' has no attribute '_write_image_video'`

- [ ] **Step 3: Implement**

In `dfxm/common/render.py`, add to the imports (keep ruff import order):

```python
from matplotlib.figure import Figure
```

After `save_layer_animation` add:

```python
def _write_image_video(get_frame, n_frames, base_path, fmt, fps=15):
    """Assemble RGB frames (``get_frame(i) -> (H, W, 3) uint8``) into MP4/GIF.

    Same container semantics as :func:`save_layer_animation`: try MP4 for
    ``mp4``/``both`` (ffmpeg), fall back to GIF. Returns the written path.
    """
    first = np.asarray(get_frame(0))
    h, w = first.shape[:2]
    dpi = 100.0
    fig = Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    im = ax.imshow(first)

    def update(frame):
        if frame:
            im.set_data(np.asarray(get_frame(frame)))
        return [im]

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    written = None
    want_mp4 = fmt in ("mp4", "both")
    want_gif = fmt in ("gif", "both")
    if want_mp4:
        try:
            anim.save(base_path + ".mp4", writer=FFMpegWriter(fps=fps), dpi=dpi)
            written = base_path + ".mp4"
        except Exception:  # noqa: BLE001 - ffmpeg missing -> fall back to GIF
            want_gif = True
    if want_gif:
        anim.save(base_path + ".gif", writer=PillowWriter(fps=fps), dpi=dpi)
        written = written or base_path + ".gif"
    return written
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_render_rotation.py -v`
Expected: 2 PASS

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/render.py tests/test_render_rotation.py
git commit -m "feat(render): GL-free frame-to-movie assembler _write_image_video"
```

---

### Task 2: Orbiting frame source `save_rotation_video` (render.py)

**Files:**
- Modify: `dfxm/common/render.py` (after `save_top_view`, end of file; module docstring line 1)
- Test: `tests/test_render_rotation.py` (extend)

**Interfaces:**
- Consumes: `_pyvista_grid(data, spacing)` and `_write_image_video` (Task 1).
- Produces: `save_rotation_video(volume, scale_z, sx, sy, vmin, vmax, cmap, opacity, base_path, fmt, *, n_frames=120, fps=15) -> str | None` — `None` when the NaN-thresholded grid is empty, else the written movie path.

- [ ] **Step 1: Write the failing test** (append to `tests/test_render_rotation.py`)

```python
def test_save_rotation_video_empty_volume_returns_none(tmp_path):
    import pytest

    pytest.importorskip("pyvista")
    volume = np.full((2, 3, 4), np.nan)
    out = render.save_rotation_video(
        volume, 1.0, 0.15, 0.38, 0.0, 1.0, "viridis", 0.85, os.path.join(tmp_path, "r"), "gif"
    )
    assert out is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_render_rotation.py::test_save_rotation_video_empty_volume_returns_none -v`
Expected: FAIL — `AttributeError: ... no attribute 'save_rotation_video'`

- [ ] **Step 3: Implement**

Append after `save_top_view`:

```python
def save_rotation_video(
    volume, scale_z, sx, sy, vmin, vmax, cmap, opacity, base_path, fmt, *, n_frames=120, fps=15
):
    """360° orbit movie of the 3D volume render; returns path or None if empty."""
    import pyvista as pv

    pv.OFF_SCREEN = True
    grid = _pyvista_grid(volume, spacing=(sx, sy, scale_z))
    if grid.n_cells == 0:
        return None
    pl = pv.Plotter(off_screen=True)
    pl.add_mesh(
        grid,
        scalars="values",
        cmap=cmap,
        clim=[vmin, vmax],
        opacity=opacity,
        smooth_shading=True,
        show_edges=False,
    )
    pl.view_isometric()
    step = 360.0 / n_frames

    def get_frame(i):
        if i:
            pl.camera.Azimuth(step)
        return pl.screenshot(return_img=True)

    try:
        return _write_image_video(get_frame, n_frames, base_path, fmt, fps=fps)
    finally:
        pl.close()
```

Update the module docstring (line 1) to `"""Shared volume rendering — per-layer PNGs, layer animation, 3D top-view + orbit video.` and line 9's lazy-import note to `...only disables the 3D top-view and rotation video.`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_render_rotation.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add dfxm/common/render.py tests/test_render_rotation.py
git commit -m "feat(render): save_rotation_video — 360-degree orbit movie via pyvista"
```

---

### Task 3: Visualize stage wiring (`save_rotation` param + product field)

**Files:**
- Modify: `dfxm/stages/visualize.py` (module docstring ~line 11; STAGE description ~line 50; params after `save_topview` ~line 277; `DatasetProducts` ~line 297; `_process_dataset` after the topview block ~line 488)
- Test: `tests/test_stage_visualize.py` (extend)

**Interfaces:**
- Consumes: `Rnd.save_rotation_video` (Task 2 signature).
- Produces: param `save_rotation` (BOOL, default False), `DatasetProducts.rotation_video: str | None` — Task 4's summarizer reads `rotation_video`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_stage_visualize.py`; it already imports `visualize` — reuse its existing import name/fixtures style)

```python
def test_process_dataset_rotation_video(tmp_path, monkeypatch):
    calls = {}

    def fake_rotation(data, scale_z, sx, sy, vmin, vmax, cmap, opacity, base_path, fmt, **kw):
        calls["base_path"] = base_path
        calls["fmt"] = fmt
        return base_path + ".gif"

    monkeypatch.setattr(visualize.Rnd, "save_rotation_video", fake_rotation)
    p = {
        **visualize.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
        "save_rotation": True,
        "output_format": "gif",
    }
    data = np.zeros((2, 4, 5))
    prod = visualize._process_dataset(
        data, [0.0, 1.0], 1.0, "chi", 0.0, 1.0, "viridis", "chi", "deg", p, str(tmp_path)
    )
    assert prod.rotation_video == calls["base_path"] + ".gif"
    assert calls["base_path"].endswith(os.path.join("chi", "chi_rotation"))
    assert calls["fmt"] == "gif"


def test_process_dataset_rotation_video_failure_becomes_note(tmp_path, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("no GL")

    monkeypatch.setattr(visualize.Rnd, "save_rotation_video", boom)
    p = {
        **visualize.STAGE.defaults(),
        "save_layers": False,
        "save_animation": False,
        "save_topview": False,
        "save_rotation": True,
    }
    prod = visualize._process_dataset(
        np.zeros((2, 4, 5)), [0.0, 1.0], 1.0, "chi", 0.0, 1.0, "viridis", "chi", "deg", p, str(tmp_path)
    )
    assert prod.rotation_video is None
    assert any("rotation video skipped" in n for n in prod.notes)
```

(If `tests/test_stage_visualize.py` imports the module differently — e.g. `from dfxm.stages import visualize` vs bare names — match its existing import; add `import os`/`import numpy as np` only if absent.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_stage_visualize.py -k rotation -v`
Expected: FAIL — `KeyError: 'save_rotation'` (or AttributeError on `save_rotation_video`)

- [ ] **Step 3: Implement**

In `dfxm/stages/visualize.py`:

(a) After the `save_topview` Param (~line 277) add:

```python
        Param(
            "save_rotation",
            ParamType.BOOL,
            "Save rotating 3-D video",
            default=False,
            advanced=True,
            group="Output",
            help=(
                "Write a movie of the 3-D volume render spinning once around "
                "(120 frames, ~8 s). Uses the same opacity as the top view and "
                "the Animation format container. Slow — off by default."
            ),
        ),
```

(b) In `DatasetProducts`, after `top_view: str | None = None`:

```python
    rotation_video: str | None = None
```

(c) In `_process_dataset`, after the `save_topview` try/except block:

```python
    if p["save_rotation"]:
        try:
            prod.rotation_video = Rnd.save_rotation_video(
                data,
                scale_z,
                sx,
                sy,
                vmin,
                vmax,
                cmap,
                float(p["volume_opacity"]),
                os.path.join(ds_dir, f"{name}_rotation"),
                p["output_format"],
            )
        except Exception as exc:  # noqa: BLE001 - no GL / pyvista issue -> note + continue
            prod.notes.append(f"rotation video skipped: {exc}")
```

No new progress call is needed: `run()` already emits `progress(..., f"mosaicity: {name}")` / `"strain: ..."` immediately before each `_process_dataset` call, which satisfies the spec's "progress before the slow render" requirement.

(d) Module docstring item 3 (~line 11): append `, and (optionally) a rotating 3-D orbit video` to the outputs sentence. STAGE `description` (~line 50): extend `"animation, and a 3-D top view."` → `"animation, a 3-D top view, and an optional rotating 3-D video."`

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_stage_visualize.py tests/test_param_metadata.py -v`
Expected: all PASS (param-metadata picks up the new param automatically)

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/visualize.py tests/test_stage_visualize.py
git commit -m "feat(visualize): save_rotation param + rotation_video product (default off)"
```

---

### Task 4: GUI summary line + docs

**Files:**
- Modify: `gui/stage_view.py` (`_dataset_lines`, ~line 995-1005)
- Modify: `docs/Usage.md` (visualize section ~line 606-634: outputs bullet + param table)
- Modify: `docs/Codebase.md` (dfxm/common/render.py entry + dfxm/stages/visualize entry — Read the current sections first, match their format)
- Test: `tests/test_stage_summaries.py` (extend)

**Interfaces:**
- Consumes: `DatasetProducts.rotation_video` (Task 3). Note `_dataset_lines` is ALSO called for rocking's `RockingProducts`, which has NO `rotation_video` field — use `getattr`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_stage_summaries.py`, which already imports `DatasetProducts`, `VisualizeResult`, `_summarize`)

```python
def test_summarize_visualize_lists_rotation_video():
    result = VisualizeResult(
        output_dir="/out",
        datasets=[
            DatasetProducts(
                name="chi", shape=(1, 2, 3), vmin=0.0, vmax=1.0, rotation_video="spin.gif"
            )
        ],
    )
    assert "spin" in _summarize("visualize", result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_stage_summaries.py -k rotation -v`
Expected: FAIL — `"spin" not in ...`

- [ ] **Step 3: Implement**

In `gui/stage_view.py` `_dataset_lines`, extend the `made` tuple list (getattr because rocking's products lack the field):

```python
        made = [
            n
            for n, v in (
                ("layers", d.layers_dir),
                ("anim", d.animation),
                ("3d", d.top_view),
                ("spin", getattr(d, "rotation_video", None)),
            )
            if v
        ]
```

Docs (same commit):
- `docs/Usage.md` visualize section: in the **Output** bullet add `an optional rotating 3-D orbit video (`<name>_rotation.mp4`/`.gif`),` and add a param-table row: `| `save_rotation` | write a 360° orbiting movie of the 3-D volume render (same look/opacity as the top view, container from `output_format`); slow, off by default |`
- `docs/Codebase.md`: add `_write_image_video` + `save_rotation_video` to the `dfxm/common/render.py` function list and `save_rotation`/`rotation_video` to the visualize stage entry, matching the surrounding format (Read those sections before editing).

- [ ] **Step 4: Run tests + smoke to verify**

Run: `python3 -m pytest tests/test_stage_summaries.py -v && python3 tests/gui_smoke.py`
Expected: all PASS; smoke stays green

- [ ] **Step 5: Commit**

```bash
git add gui/stage_view.py tests/test_stage_summaries.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui,docs): spin artifact in visualize summary + rotating-video docs"
```

---

### Task 5: Full verification + end review

- [ ] **Step 1:** Run the full gate (verify-suite skill): `ruff check . && ruff format --check . && python3 -m pytest -q && python3 tests/gui_smoke.py`
- [ ] **Step 2:** Single end code review of the branch diff (code-review, high effort scoped to the changed files); fix findings, re-run the gate.
- [ ] **Step 3:** Merge via finish-and-record flow (no remote — local merge to master), update memory notes.
