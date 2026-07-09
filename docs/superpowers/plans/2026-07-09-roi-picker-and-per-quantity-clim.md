# App-wide Interactive ROI Picker + Per-quantity Colour Limits — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every ROI selection in the GUI an interactive, WYSIWYG "drag a rectangle" picker, and let the slices replot dialog set colour limits per quantity (χ vs μ, and each `raw_*`) instead of per colormap group.

**Architecture:** A single dumb Qt+matplotlib widget (`ROIPickerDialog`) renders a preview array at true physical aspect and returns four half-open pixel indices; every call site supplies previews via lazy `() -> (array2d, sx, sy)` thunks and does its own write-back. The slices replot clim key moves from colormap group to `volume_id` with a group-key fallback. Run-time ROI params become schema-driven picker targets via two new `Param` fields.

**Tech Stack:** Python 3.10, PySide6, matplotlib (explicit `Figure` API + `FigureCanvasQTAgg` + `RectangleSelector`), h5py, numpy, pytest.

## Global Constraints

- Keep `dfxm/` **Qt-free** — no PySide6/pyvista imports there. The picker widget lives under `gui/`.
- **Lazy heavy deps:** import `roi_picker` and matplotlib's Qt backend only inside the button handlers that open it (mirror `line_picker` at `gui/stage_view.py:386`).
- Build figures with the explicit `matplotlib.figure.Figure` API — **never** `pyplot`, never `matplotlib.use(...)`.
- Every new `Param` needs `help`; advanced params need `group`; input paths set `must_exist=True`. `tests/test_param_metadata.py` enforces this.
- User-facing input errors raise `StageUserError(message, hint=...)` from `dfxm.common.errors` (not applicable to the picker, which only reframes data).
- **Docs same-change contract:** any change to a stage's params/behaviour/IO or a viewer updates BOTH `docs/Usage.md` (user) and `docs/Codebase.md` (code) in the SAME task.
- **No git remote** — commit locally; never push/PR.
- Run `ruff check . && ruff format .` before every commit (format hook runs on Write/Edit already).
- Suite: `python3 -m pytest -q`. GUI smoke (not a pytest file): `python3 tests/gui_smoke.py`.
- ROI ints are **half-open** `(r0, r1, c0, c1)` == `data[r0:r1, c0:c1]`; `r`=rows/Y, `c`=cols/X. This matches `crop_roi_2d` (`dfxm/common/figures.py:116`) and `apply_roi_3d` (`dfxm/common/alignment.py:24`).
- Map exports render with `origin="lower"` (`dfxm/common/render.py:58`) — the picker MUST match or picked r-ranges are vertically mirrored.

---

## Task 1: Slices replot clim keyed per `volume_id` (core)

**Files:**
- Modify: `dfxm/stages/slices.py` (`render_replot`, ~line 1175; docstring ~1160-1166)
- Test: `tests/test_stage_slices.py`

**Interfaces:**
- Consumes: `resolve_clim(clim, key)` from `dfxm.common.figures` (unchanged); `ReplotEntry.volume_id`, `ReplotEntry.group` (`slices.py:448`).
- Produces: `render_replot` now resolves clim by `entry.volume_id`, falling back to `entry.group`, then to a bare tuple. No signature change.

- [ ] **Step 1: Write the failing test** — add to `tests/test_stage_slices.py` a builder for a mosa file with χ/μ sharing one group, and a test that they get different clims:

```python
def _write_mosa_consolidated(path):
    """Two mosa-COM volumes (chi, mu) sharing group 'mosa_com', one slice, 2 planes."""
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offsets = np.array([0.0, 1.0])
    with h5py.File(path, "w") as f:
        for vid in ("mosa_com_chi", "mosa_com_mu"):
            g = f.create_group(vid)
            g.attrs["kind"] = "mosa_com"
            g.attrs["cmap"] = "magma"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "deg"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("plane_a")
            sg.create_dataset("slices", data=np.zeros((2, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offsets)


def test_render_replot_clim_keyed_by_volume_id(tmp_path, monkeypatch):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mosa_consolidated(str(h5))
    seen: dict[str, tuple] = {}

    def fake_rebuild(h5_path, vid, sname, k, style, *, clim=None, roi=None):
        seen[vid] = clim
        return None

    monkeypatch.setattr(SL, "_rebuild_plane_figure", fake_rebuild)
    SL.render_replot(
        str(h5),
        [("mosa_com_chi", "plane_a", [0]), ("mosa_com_mu", "plane_a", [0])],
        style=None,
        clim={"mosa_com_chi": (-2.0, 2.0), "mosa_com_mu": (-9.0, 9.0)},
        out_dir=str(tmp_path / "r"),
    )
    assert seen["mosa_com_chi"] == (-2.0, 2.0)
    assert seen["mosa_com_mu"] == (-9.0, 9.0)  # chi and mu NO LONGER share a limit


def test_render_replot_clim_group_key_still_works(tmp_path, monkeypatch):
    """Back-compat: a group-keyed dict still applies via the fallback."""
    h5 = tmp_path / "oblique_slices.h5"
    _write_mosa_consolidated(str(h5))
    seen: dict[str, tuple] = {}
    monkeypatch.setattr(
        SL, "_rebuild_plane_figure",
        lambda h5_path, vid, sname, k, style, *, clim=None, roi=None: seen.__setitem__(vid, clim),
    )
    SL.render_replot(
        str(h5),
        [("mosa_com_chi", "plane_a", [0]), ("mosa_com_mu", "plane_a", [0])],
        style=None,
        clim={"mosa_com": (-3.0, 3.0)},  # group key
        out_dir=str(tmp_path / "r"),
    )
    assert seen["mosa_com_chi"] == (-3.0, 3.0)
    assert seen["mosa_com_mu"] == (-3.0, 3.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_stage_slices.py::test_render_replot_clim_keyed_by_volume_id -q`
Expected: FAIL — both χ and μ currently receive `None` (neither vid is in the dict; only group `mosa_com` would match, and the code keys by group so the vid-keyed dict misses entirely).

- [ ] **Step 3: Implement the two-key fallback** — in `dfxm/stages/slices.py::render_replot`, change the clim resolution (currently `clim_k = resolve_clim(clim, entry.group)`):

```python
        clim_k = resolve_clim(clim, entry.volume_id)
        if clim_k is None:
            clim_k = resolve_clim(clim, entry.group)
```

Update the `render_replot` docstring block (slices.py ~1160-1166): replace the "`{kind_group: (vmin, vmax)}` mapping (`mosa_com`/`mosa_fwhm`/`strain`/`raw`)" sentence with:

```
    ``clim`` overrides the stored colour limits: ``None`` keeps them, a single
    ``(vmin, vmax)`` applies to every plane, and a ``{key: (vmin, vmax)}`` mapping
    sets them per quantity — keyed by ``volume_id`` (e.g. ``mosa_com_chi`` vs
    ``mosa_com_mu``, each ``raw_*``), falling back to the colormap group
    (``mosa_com``/``mosa_fwhm``/``strain``/``raw``) for keys not found by volume_id.
```

- [ ] **Step 4: Run to verify it passes** (and the legacy per-group test still passes)

Run: `python3 -m pytest tests/test_stage_slices.py -q -k "render_replot"`
Expected: PASS — including the pre-existing `test_render_replot_per_group_clim_maps_by_kind` (its `{"raw":..,"strain":..}` still resolves via the group fallback).

- [ ] **Step 5: Update `docs/Codebase.md`** — in the slices `render_replot` entry, note clim is keyed by `volume_id` with a colormap-group fallback.

- [ ] **Step 6: Commit**

```bash
ruff check . && ruff format .
git add dfxm/stages/slices.py tests/test_stage_slices.py docs/Codebase.md
git commit -m "feat(slices): replot clim keyed per volume_id (chi/mu split) with group fallback"
```

---

## Task 2: Per-quantity clim rows in the slices replot dialog (GUI)

**Files:**
- Modify: `gui/widgets/slice_replot.py` (`_SLICE_GROUP_ORDER`/`_SLICE_GROUP_LABELS` at 32-40; `_clim_groups` at 222-233)
- Test: `tests/test_gui_slice_replot.py`

**Interfaces:**
- Consumes: `ReplotEntry.volume_id` (per-entry); `ClimGroupSection.set_groups([(key, label)])` (unchanged).
- Produces: the dialog's `ClimGroupSection` now has one row per distinct `volume_id`; `clim_by_group()` returns a `{volume_id: (vmin, vmax)}` mapping that Task 1's core consumes.

- [ ] **Step 1: Write the failing test** — append to `tests/test_gui_slice_replot.py`:

```python
def test_clim_rows_are_per_volume_id(tmp_path):
    from gui.widgets.slice_replot import SliceReplotDialog

    _ = QApplication.instance() or QApplication([])
    h5 = tmp_path / "oblique_slices.h5"
    # two mosa-COM volumes sharing group 'mosa_com'
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    with h5py.File(h5, "w") as f:
        for vid in ("mosa_com_chi", "mosa_com_mu"):
            g = f.create_group(vid)
            g.attrs["kind"] = "mosa_com"
            g.attrs["cmap"] = "magma"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "deg"
            g.attrs["vmin"], g.attrs["vmax"] = -1.0, 1.0
            sg = g.create_group("plane_a")
            sg.create_dataset("slices", data=np.zeros((2, v.size, u.size), dtype=np.float32))
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=np.array([0.0, 1.0]))
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path / "o"))
    keys = set(dlg._clim._edits.keys())
    assert keys == {"mosa_com_chi", "mosa_com_mu"}  # one row per quantity, not one 'mosa_com'
    dlg.deleteLater()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_gui_slice_replot.py::test_clim_rows_are_per_volume_id -q`
Expected: FAIL — keys are `{"mosa_com"}` (group), not the two vids.

- [ ] **Step 3: Implement per-vid rows** — in `gui/widgets/slice_replot.py`, replace the group tables and `_clim_groups`:

Replace `_SLICE_GROUP_ORDER`/`_SLICE_GROUP_LABELS` (32-40) with a per-vid label helper:

```python
# Friendly labels for the per-quantity colour-limit rows, keyed by volume_id.
# volume_id is f"{kind}{suffix}" where suffix is ""/"_chi"/"_mu" (slices.py:_axis_suffix).
_KIND_LABELS = {
    "mosa_com": "Mosaicity COM",
    "mosa_fwhm": "Mosaicity FWHM",
    "strain": "Strain",
    "raw_sum": "Raw sum intensity",
    "raw_specific": "Raw frame",
    "raw_mosa_sum": "Raw mosa-sum intensity",
    "raw_mosa_specific": "Raw mosa frame",
}


def _volume_label(volume_id: str) -> str:
    """Human label for a clim row, e.g. 'mosa_com_chi' -> 'Mosaicity COM (χ)'."""
    for comp, sym in (("_chi", "χ"), ("_mu", "μ")):
        if volume_id.endswith(comp):
            base = volume_id[: -len(comp)]
            return f"{_KIND_LABELS.get(base, base)} ({sym})"
    return _KIND_LABELS.get(volume_id, volume_id)
```

Replace `_clim_groups` (222-233) with a per-vid version (stable order: first-seen):

```python
    @staticmethod
    def _clim_groups(catalog):
        """One (volume_id, label) row per distinct quantity, in first-seen order."""
        vids = list(dict.fromkeys(e.volume_id for e in catalog))
        return [(vid, _volume_label(vid)) for vid in vids]
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_gui_slice_replot.py -q`
Expected: PASS (all, including the pre-existing dialog tests).

- [ ] **Step 5: Update `docs/Usage.md`** — in the slices Replot section, note the colour-limit rows are now per quantity (χ and μ separate; each raw variant separate).

- [ ] **Step 6: Commit**

```bash
ruff check . && ruff format .
git add gui/widgets/slice_replot.py tests/test_gui_slice_replot.py docs/Usage.md
git commit -m "feat(slices-replot): per-quantity colour-limit rows (chi/mu + raw split)"
```

---

## Task 3: `rect_to_indices` pure helper

**Files:**
- Create: `gui/widgets/roi_picker.py`
- Test: `tests/test_roi_picker.py`

**Interfaces:**
- Produces: `rect_to_indices(xmin, xmax, ymin, ymax, w, h) -> tuple[int, int, int, int]` returning half-open `(r0, r1, c0, c1)` clamped to `[0,h]`/`[0,w]`.

- [ ] **Step 1: Write the failing test** — create `tests/test_roi_picker.py`:

```python
"""Pure-function tests for the ROI picker's rectangle→indices mapping (no QApplication)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")  # module imports PySide6 at import time, but needs no app

from gui.widgets.roi_picker import rect_to_indices  # noqa: E402


def test_basic_floor_ceil_halfopen():
    # a rect from x[12.2, 88.9], y[40.0, 160.7] on a 200x100 (h x w) grid
    assert rect_to_indices(12.2, 88.9, 40.0, 160.7, w=100, h=200) == (40, 161, 12, 89)


def test_swapped_min_max_normalised():
    assert rect_to_indices(88.9, 12.2, 160.7, 40.0, w=100, h=200) == (40, 161, 12, 89)


def test_clamped_to_bounds():
    assert rect_to_indices(-5.0, 500.0, -5.0, 500.0, w=100, h=200) == (0, 200, 0, 100)


def test_degenerate_zero_area():
    assert rect_to_indices(50.0, 50.0, 30.0, 30.0, w=100, h=200) == (30, 30, 50, 50)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_roi_picker.py -q`
Expected: FAIL — `gui/widgets/roi_picker.py` does not exist.

- [ ] **Step 3: Create the module with the pure helper** — `gui/widgets/roi_picker.py`:

```python
"""Interactive ROI picker: drag a rectangle on a preview plane, get pixel bounds.

Source-agnostic and dumb — it imports nothing from ``dfxm`` beyond numpy-level
types. Each call site supplies previews as ``(label, thunk)`` pairs where
``thunk() -> (array2d, sx, sy)`` (lazy). The plane is drawn exactly like the
map exports (``origin="lower"``, physical aspect via ``set_aspect(sy/sx)``), so
the crop you draw is the crop you get. On accept, :attr:`result` is the
half-open ``(r0, r1, c0, c1)`` pixel-index tuple; otherwise ``None``.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np


def rect_to_indices(xmin, xmax, ymin, ymax, w, h) -> tuple[int, int, int, int]:
    """Map a selector rectangle (data coords on pixel-edge extents) to half-open
    ``(r0, r1, c0, c1)`` pixel indices, clamped to ``[0, w]`` / ``[0, h]``.

    ``x`` is columns (X), ``y`` is rows (Y). floor(min)/ceil(max) on pixel-edge
    extents gives inclusive-of-touched-pixels behaviour with no ±0.5 fencepost.
    """
    c0 = max(0, min(int(math.floor(min(xmin, xmax))), w))
    c1 = max(0, min(int(math.ceil(max(xmin, xmax))), w))
    r0 = max(0, min(int(math.floor(min(ymin, ymax))), h))
    r1 = max(0, min(int(math.ceil(max(ymin, ymax))), h))
    return r0, r1, c0, c1
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_roi_picker.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
ruff check . && ruff format .
git add gui/widgets/roi_picker.py tests/test_roi_picker.py
git commit -m "feat(roi-picker): rect_to_indices pixel-edge/half-open mapping"
```

---

## Task 4: `ROIPickerDialog` widget

**Files:**
- Modify: `gui/widgets/roi_picker.py` (append the dialog class)
- Test: `tests/test_gui_roi_picker.py`

**Interfaces:**
- Consumes: `rect_to_indices` (Task 3).
- Produces: `ROIPickerDialog(previews, initial=None, parent=None)` where `previews: list[tuple[str, Callable[[], tuple[np.ndarray, float, float]]]]`; attribute `result: tuple[int,int,int,int] | None`; internal `_on_rect_change(...)` callable directly in tests. Method `_current_shape() -> tuple[int,int]`.

- [ ] **Step 1: Write the failing test** — create `tests/test_gui_roi_picker.py`:

```python
"""Offscreen construction + selection test for the ROI picker dialog."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


def _previews():
    arr = np.arange(200 * 100, dtype=float).reshape(200, 100)  # (H=200, W=100)
    return [("layer 0", lambda a=arr: (a, 0.152, 0.385))]  # sx=0.152 (X), sy=0.385 (Y)


def test_dialog_selection_returns_indices_and_readout():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())
    # simulate a drag by calling the selector callback with data coords
    dlg._on_rect_change(12.0, 89.0, 40.0, 161.0)
    assert dlg._use.isEnabled()
    assert "µm" in dlg._readout.text() and "px" in dlg._readout.text()
    dlg._accept()
    assert dlg.result == (40, 161, 12, 89)
    dlg.deleteLater()


def test_dialog_no_selection_result_none():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())
    assert dlg.result is None
    assert not dlg._use.isEnabled()  # disabled until a non-degenerate rect exists
    dlg.deleteLater()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_gui_roi_picker.py -q`
Expected: FAIL — `ROIPickerDialog` not defined.

- [ ] **Step 3: Append the dialog class** to `gui/widgets/roi_picker.py`:

```python
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.widgets import RectangleSelector  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ROIPickerDialog(QDialog):
    """Drag a rectangle on a preview plane; read back half-open pixel bounds."""

    def __init__(self, previews, initial=None, parent=None) -> None:
        super().__init__(parent)
        self._previews = list(previews)
        self._initial = initial
        self.result: tuple[int, int, int, int] | None = None
        self._arr: np.ndarray | None = None
        self._sx = self._sy = 1.0
        self._rect: tuple[float, float, float, float] | None = None  # xmin,xmax,ymin,ymax

        self.setWindowTitle("Pick ROI")
        self._combo = QComboBox()
        for label, _thunk in self._previews:
            self._combo.addItem(label)
        self._combo.currentIndexChanged.connect(self._load_current)

        self._fig = Figure(figsize=(5, 6), layout="tight")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._ax = self._fig.add_subplot(111)
        self._selector = None
        self._readout = QLabel("drag a rectangle")

        self._reset = QPushButton("Reset")
        self._use = QPushButton("Use")
        self._cancel = QPushButton("Cancel")
        self._use.setEnabled(False)
        self._reset.clicked.connect(self._on_reset)
        self._use.clicked.connect(self._accept)
        self._cancel.clicked.connect(self.reject)

        top = QHBoxLayout()
        top.addWidget(QLabel("Preview:"))
        top.addWidget(self._combo, 1)
        btns = QHBoxLayout()
        btns.addWidget(self._readout, 1)
        btns.addWidget(self._reset)
        btns.addWidget(self._use)
        btns.addWidget(self._cancel)
        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._canvas, 1)
        lay.addLayout(btns)

        if self._previews:
            self._load_current()

    def _current_shape(self) -> tuple[int, int] | None:
        return None if self._arr is None else (self._arr.shape[0], self._arr.shape[1])

    def _load_current(self) -> None:
        prev_shape = self._current_shape()
        idx = max(0, self._combo.currentIndex())
        try:
            arr, sx, sy = self._previews[idx][1]()
        except Exception as exc:  # noqa: BLE001 — bad path/dataset: show, don't crash
            self._arr = None
            self._readout.setText(f"cannot preview: {exc}")
            return
        self._arr, self._sx, self._sy = np.asarray(arr, dtype=float), float(sx), float(sy)
        h, w = self._arr.shape[:2]
        self._ax.clear()
        self._ax.imshow(
            self._arr, origin="lower", extent=[0, w, 0, h], cmap="magma", aspect="auto"
        )
        self._ax.set_aspect(self._sy / self._sx)  # physical aspect (Y stretched by sy/sx)
        self._ax.set_xlabel("cols / X (px)")
        self._ax.set_ylabel("rows / Y (px)")
        self._selector = RectangleSelector(
            self._ax,
            onselect=lambda e_press, e_release: self._on_rect_change(
                e_press.xdata, e_release.xdata, e_press.ydata, e_release.ydata
            ),
            useblit=False,
            interactive=True,
            button=[1],
        )
        # keep the rect only if the new preview has the same shape
        if prev_shape is not None and prev_shape != (h, w):
            self._rect = None
            self._use.setEnabled(False)
            self._readout.setText(f"shape {h}×{w} px — previous selection cleared")
        elif self._rect is None and self._initial is not None:
            r0, r1, c0, c1 = self._initial
            self._on_rect_change(c0, c1, r0, r1)
        elif self._rect is not None:
            xmin, xmax, ymin, ymax = self._rect
            self._on_rect_change(xmin, xmax, ymin, ymax)
        self._canvas.draw_idle()

    def _on_rect_change(self, xmin, xmax, ymin, ymax) -> None:
        if self._arr is None or None in (xmin, xmax, ymin, ymax):
            return
        self._rect = (xmin, xmax, ymin, ymax)
        h, w = self._arr.shape[:2]
        r0, r1, c0, c1 = rect_to_indices(xmin, xmax, ymin, ymax, w, h)
        dr, dc = r1 - r0, c1 - c0
        self._use.setEnabled(dr >= 1 and dc >= 1)
        self._readout.setText(
            f"{dr}×{dc} px  =  {dr * self._sy:.1f} × {dc * self._sx:.1f} µm (Y×X)"
        )

    def _on_reset(self) -> None:
        self._rect = None
        self._use.setEnabled(False)
        self._readout.setText("drag a rectangle")
        if self._selector is not None:
            self._selector.set_visible(False)
        self._canvas.draw_idle()

    def _accept(self) -> None:
        if self._arr is None or self._rect is None:
            return
        h, w = self._arr.shape[:2]
        self.result = rect_to_indices(*self._rect, w=w, h=h)
        self.accept()
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_gui_roi_picker.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Update `docs/Codebase.md`** — add `gui/widgets/roi_picker.py` (`rect_to_indices`, `ROIPickerDialog`) to the widgets section.

- [ ] **Step 6: Commit**

```bash
ruff check . && ruff format .
git add gui/widgets/roi_picker.py tests/test_gui_roi_picker.py docs/Codebase.md
git commit -m "feat(roi-picker): ROIPickerDialog (physical-aspect WYSIWYG rectangle select)"
```

---

## Task 5: Preview helpers (Qt-free core)

**Files:**
- Modify: `dfxm/common/figures.py` (add `load_middle_layer`)
- Modify: `dfxm/stages/slices.py` (add `plane_preview`)
- Test: `tests/test_figures_replot.py`, `tests/test_stage_slices.py`

**Interfaces:**
- Produces:
  - `figures.load_middle_layer(h5_path, dataset) -> np.ndarray` — the middle-Z 2-D layer of a (Z,Y,X) dataset.
  - `slices.plane_preview(h5_path, volume_id, slice_name) -> tuple[np.ndarray, float, float]` — middle plane + `(du, dv)` pitch (du=X/cols, dv=Y/rows) from `u_um`/`v_um`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_figures_replot.py`:

```python
def test_load_middle_layer(tmp_path):
    import h5py
    import numpy as np
    from dfxm.common.figures import load_middle_layer

    p = tmp_path / "vol.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("/chi/Center of mass", data=np.arange(5 * 4 * 3).reshape(5, 4, 3))
    layer = load_middle_layer(str(p), "/chi/Center of mass")
    assert layer.shape == (4, 3)
    assert np.array_equal(layer, np.arange(5 * 4 * 3).reshape(5, 4, 3)[2])  # middle z=2
```

Append to `tests/test_stage_slices.py`:

```python
def test_plane_preview_returns_middle_plane_and_du_dv(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_mini_consolidated(str(h5))  # u: 9 pts over [-4,4] -> du=1.0; v: 7 pts over [-3,3] -> dv=1.0
    arr, sx, sy = SL.plane_preview(str(h5), "strain", "plane_a")
    assert arr.shape == (7, 9)  # (nv, nu)
    assert sx == pytest.approx(1.0)  # du (cols/X)
    assert sy == pytest.approx(1.0)  # dv (rows/Y)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_figures_replot.py::test_load_middle_layer tests/test_stage_slices.py::test_plane_preview_returns_middle_plane_and_du_dv -q`
Expected: FAIL — `load_middle_layer` / `plane_preview` not defined.

- [ ] **Step 3: Implement `load_middle_layer`** in `dfxm/common/figures.py` (below `_load_layer`, ~line 114):

```python
def load_middle_layer(h5_path: str, dataset: str) -> np.ndarray:
    """Return the middle-Z 2-D layer of a (Z,Y,X) HDF5 dataset (ROI-picker preview)."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        dset = f[dataset]
        z = dset.shape[0] // 2
        return dset[z][...]
```

- [ ] **Step 4: Implement `plane_preview`** in `dfxm/stages/slices.py` (near `replot_catalog`, ~line 1118):

```python
def plane_preview(h5_path: str, volume_id: str, slice_name: str) -> tuple[np.ndarray, float, float]:
    """Middle plane of a slice group + its (du, dv) µm/px pitch, for the ROI picker.

    Returns ``(array2d, sx, sy)`` where ``sx=du`` (cols/u/X) and ``sy=dv``
    (rows/v/Y) come from the stored ``u_um``/``v_um`` axes — the resampled slice
    pitch, NOT the detector pixel scale.
    """
    with h5py.File(h5_path, "r") as f:
        sg = f[f"{volume_id}/{slice_name}"]
        stack = sg["slices"]
        mid = stack.shape[0] // 2
        arr = stack[mid][...]
        u = sg["u_um"][:]
        v = sg["v_um"][:]
    du = float(abs(u[1] - u[0])) if len(u) > 1 else 1.0
    dv = float(abs(v[1] - v[0])) if len(v) > 1 else 1.0
    return np.asarray(arr, dtype=float), du, dv
```

- [ ] **Step 5: Run to verify they pass**

Run: `python3 -m pytest tests/test_figures_replot.py::test_load_middle_layer tests/test_stage_slices.py::test_plane_preview_returns_middle_plane_and_du_dv -q`
Expected: PASS.

- [ ] **Step 6: Update `docs/Codebase.md`** — note `load_middle_layer` (figures) and `plane_preview` (slices) as ROI-picker preview helpers.

- [ ] **Step 7: Commit**

```bash
ruff check . && ruff format .
git add dfxm/common/figures.py dfxm/stages/slices.py tests/test_figures_replot.py tests/test_stage_slices.py docs/Codebase.md
git commit -m "feat(core): ROI-picker preview helpers (load_middle_layer, slices.plane_preview)"
```

---

## Task 6: Wire "Pick ROI…" into the generic replot dialog (P1)

**Files:**
- Modify: `gui/widgets/replot_dialog.py` (constructor gains `preview_fn`; add button + handler)
- Modify: `gui/stage_view.py` (`_on_replot`, ~441-452 — build + pass `preview_fn`)
- Test: `tests/test_gui_replot_dialog.py`

**Interfaces:**
- Consumes: `ROIPickerDialog` (Task 4); `figures.load_middle_layer` (Task 5); `ReplotGroup.key` (catalog).
- Produces: `ReplotDialog(h5_default, catalog_fn, render_fn, style=None, out_default="", preview_fn=None, parent=None)`. `preview_fn(h5_path, key) -> tuple[np.ndarray, float, float]`. New method `_on_pick_roi()` and button `_pick_roi_btn`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_gui_replot_dialog.py` (follow its existing offscreen/app header):

```python
def test_pick_roi_fills_boxes(tmp_path, monkeypatch):
    import numpy as np
    from gui.widgets.replot_dialog import ReplotDialog
    from dfxm.common.figures import ReplotGroup

    _ = QApplication.instance() or QApplication([])

    def catalog_fn(_h5):
        return [ReplotGroup(key="/chi/Center of mass", label="χ", item_labels=["layer 0"], shape=(200, 100))]

    def preview_fn(_h5, _key):
        return np.zeros((200, 100)), 0.152, 0.385

    dlg = ReplotDialog("nofile.h5", catalog_fn, lambda *a, **k: [], preview_fn=preview_fn)

    # stub the modal picker: pretend the user dragged (r0,r1,c0,c1)
    import gui.widgets.replot_dialog as RD

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (40, 160, 12, 88)
        def exec(self):
            return 1

    monkeypatch.setattr(RD, "ROIPickerDialog", _FakePicker, raising=False)
    dlg._on_pick_roi()
    assert (dlg._r0.text(), dlg._r1.text(), dlg._c0.text(), dlg._c1.text()) == ("40", "160", "12", "88")
    dlg.deleteLater()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_gui_replot_dialog.py::test_pick_roi_fills_boxes -q`
Expected: FAIL — `preview_fn` kw / `_on_pick_roi` not present.

- [ ] **Step 3: Implement in `gui/widgets/replot_dialog.py`.**

Add `preview_fn=None` to `__init__` signature and store it:

```python
    def __init__(
        self, h5_default, catalog_fn, render_fn, style=None, out_default="", preview_fn=None, parent=None
    ) -> None:
        ...
        self._preview_fn = preview_fn
```

Add a "Pick ROI…" button into `roi_row` (after the four boxes, ~line 78):

```python
        self._pick_roi_btn = QPushButton("Pick ROI…")
        self._pick_roi_btn.clicked.connect(self._on_pick_roi)
        roi_row.addWidget(self._pick_roi_btn)
```

Add the handler (previews = one per catalog group, middle layer via `preview_fn`):

```python
    def _on_pick_roi(self) -> None:
        if not self._preview_fn or not self._h5_path or not os.path.exists(self._h5_path):
            self._status.setText("load a volume file first to pick an ROI")
            return
        try:
            catalog = self._catalog_fn(self._h5_path)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"cannot read: {exc}")
            return
        previews = [
            (grp.label, (lambda k=grp.key: self._preview_fn(self._h5_path, k))) for grp in catalog
        ]
        if not previews:
            self._status.setText("nothing to preview")
            return
        from .roi_picker import ROIPickerDialog  # imported on demand

        dlg = ROIPickerDialog(previews, initial=self._roi(), parent=self)
        if dlg.exec() and dlg.result:
            r0, r1, c0, c1 = dlg.result
            for edit, val in ((self._r0, r0), (self._r1, r1), (self._c0, c0), (self._c1, c1)):
                edit.setText(str(val))
```

- [ ] **Step 4: Pass `preview_fn` from `gui/stage_view.py::_on_replot`** — build a preview closure using the stage's pixel scales (mosaicity/rocking/strain all expose `pixel_size_x_um`/`pixel_size_y_um` on their forms) and pass it into `ReplotDialog` (after `render_fn`, ~441-452):

```python
        from dfxm.common.figures import load_middle_layer

        def preview_fn(h5, key, _p=dict(vals)):
            sx = float(_p.get("pixel_size_x_um", 0.152))
            sy = float(_p.get("pixel_size_y_um", 0.385))
            return load_middle_layer(h5, key), sx, sy

        dlg = ReplotDialog(
            h5_default,
            module.replot_catalog,
            render_fn,
            style=style,
            out_default="",
            preview_fn=preview_fn,
            parent=self,
        )
```

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_gui_replot_dialog.py -q`
Expected: PASS.

- [ ] **Step 6: Update `docs/Usage.md`** — in the replot (strain/mosaicity/rocking) section, document the "Pick ROI…" button and that the preview is oriented exactly like the exported maps.

- [ ] **Step 7: Commit**

```bash
ruff check . && ruff format .
git add gui/widgets/replot_dialog.py gui/stage_view.py tests/test_gui_replot_dialog.py docs/Usage.md
git commit -m "feat(replot): Pick ROI… button in the generic replot dialog"
```

---

## Task 7: Wire "Pick ROI…" into the slices replot dialog (P2)

**Files:**
- Modify: `gui/widgets/slice_replot.py` (add button + handler using `slices.plane_preview`)
- Test: `tests/test_gui_slice_replot.py`

**Interfaces:**
- Consumes: `ROIPickerDialog` (Task 4); `dfxm.stages.slices.plane_preview` (Task 5); the dialog's `catalog` (`ReplotEntry.volume_id`, `.slice_name`).
- Produces: `SliceReplotDialog._on_pick_roi()`; button `_pick_roi_btn`; the four ROI boxes get filled from the picker result.

- [ ] **Step 1: Write the failing test** — append to `tests/test_gui_slice_replot.py`:

```python
def test_slice_pick_roi_fills_boxes(tmp_path, monkeypatch):
    from gui.widgets.slice_replot import SliceReplotDialog

    _ = QApplication.instance() or QApplication([])
    h5 = tmp_path / "oblique_slices.h5"
    _mini(str(h5))  # helper already in this file: raw_sum + strain, plane_a
    dlg = SliceReplotDialog(str(h5), style=None, out_default=str(tmp_path / "o"))

    import gui.widgets.slice_replot as SR

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (2, 6, 1, 8)
        def exec(self):
            return 1

    monkeypatch.setattr(SR, "ROIPickerDialog", _FakePicker, raising=False)
    dlg._on_pick_roi()
    assert (dlg._r0.text(), dlg._r1.text(), dlg._c0.text(), dlg._c1.text()) == ("2", "6", "1", "8")
    dlg.deleteLater()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_gui_slice_replot.py::test_slice_pick_roi_fills_boxes -q`
Expected: FAIL — `_on_pick_roi` not present.

- [ ] **Step 3: Implement in `gui/widgets/slice_replot.py`.**

Add a "Pick ROI…" button into `roi_row` (after the four boxes, ~line 96):

```python
        self._pick_roi_btn = QPushButton("Pick ROI…")
        self._pick_roi_btn.clicked.connect(self._on_pick_roi)
        roi_row.addWidget(self._pick_roi_btn)
```

Add the handler (previews = one per (volume_id, slice_name) in the loaded catalog):

```python
    def _on_pick_roi(self) -> None:
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._status.setText("load a slices file first to pick an ROI")
            return
        try:
            catalog = _sl.replot_catalog(self._h5_path)
        except Exception as exc:  # noqa: BLE001
            self._status.setText(f"cannot read: {exc}")
            return
        previews = [
            (
                f"{e.volume_id} · {e.slice_name}",
                (lambda v=e.volume_id, s=e.slice_name: _sl.plane_preview(self._h5_path, v, s)),
            )
            for e in catalog
        ]
        if not previews:
            self._status.setText("nothing to preview")
            return
        from .roi_picker import ROIPickerDialog  # imported on demand

        dlg = ROIPickerDialog(previews, initial=self._roi(), parent=self)
        if dlg.exec() and dlg.result:
            r0, r1, c0, c1 = dlg.result
            for edit, val in ((self._r0, r0), (self._r1, r1), (self._c0, c0), (self._c1, c1)):
                edit.setText(str(val))
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_gui_slice_replot.py -q`
Expected: PASS.

- [ ] **Step 5: Update `docs/Usage.md`** — note the "Pick ROI…" button in the slices Replot section (preview per volume/slice; different slice groups can differ in shape so the selection clears when you switch to a differently-shaped preview).

- [ ] **Step 6: Commit**

```bash
ruff check . && ruff format .
git add gui/widgets/slice_replot.py tests/test_gui_slice_replot.py docs/Usage.md
git commit -m "feat(slices-replot): Pick ROI… button (per-plane preview at slice pitch)"
```

---

## Task 8: `Param.roi_group` / `roi_axis` schema fields

**Files:**
- Modify: `dfxm/config/models.py` (`Param`, 40-66)
- Test: `tests/test_param_metadata.py`

**Interfaces:**
- Produces: `Param(roi_group: str = "", roi_axis: str = "")`; `__post_init__` validates `roi_axis in {"", "x", "y", "both"}` and that a non-empty `roi_axis` requires a non-empty `roi_group`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_param_metadata.py`:

```python
def test_roi_fields_default_empty():
    from dfxm.config.models import Param, ParamType

    p = Param("x", ParamType.STR, "X")
    assert p.roi_group == ""
    assert p.roi_axis == ""


def test_roi_axis_requires_group_and_valid_value():
    from dfxm.config.models import Param, ParamType

    Param("roi_x", ParamType.STR, "ROI x", roi_group="align", roi_axis="x")  # ok
    with pytest.raises(ValueError):
        Param("roi_x", ParamType.STR, "ROI x", roi_axis="x")  # axis without group
    with pytest.raises(ValueError):
        Param("roi_x", ParamType.STR, "ROI x", roi_group="align", roi_axis="diagonal")  # bad value
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_param_metadata.py -q -k roi`
Expected: FAIL — `Param` has no `roi_group`/`roi_axis`.

- [ ] **Step 3: Implement in `dfxm/config/models.py`.** Add fields after `must_exist` (line 62):

```python
    must_exist: bool = False  # input path/dir: GUI checks existence before a run
    roi_group: str = ""  # params sharing a roi_group are one ROI-picker target
    roi_axis: str = ""  # "" | "x" | "y" | "both" ("both" = one 4-int "r0,r1,c0,c1" field)
```

Extend `__post_init__` (after the enum check, line 66):

```python
        if self.roi_axis and not self.roi_group:
            raise ValueError(f"roi param {self.name!r}: roi_axis set but roi_group is empty")
        if self.roi_axis not in ("", "x", "y", "both"):
            raise ValueError(f"roi param {self.name!r}: bad roi_axis {self.roi_axis!r}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `python3 -m pytest tests/test_param_metadata.py -q`
Expected: PASS (all).

- [ ] **Step 5: Update `docs/Codebase.md`** — document the `Param.roi_group`/`roi_axis` fields in the config schema section.

- [ ] **Step 6: Commit**

```bash
ruff check . && ruff format .
git add dfxm/config/models.py tests/test_param_metadata.py docs/Codebase.md
git commit -m "feat(config): Param.roi_group/roi_axis for schema-driven ROI pickers"
```

---

## Task 9: strain — tag `roi` + `roi_previews`

**Files:**
- Modify: `dfxm/stages/strain.py` (tag the `roi` Param ~168-178; add `roi_previews`)
- Test: `tests/test_strain.py` (or the existing strain test module — use `tests/test_stage_strain.py` if present; otherwise create `tests/test_strain_roi_previews.py`)

**Interfaces:**
- Consumes: `strain._derive_maps_path(layer_name, params)` (485), `strain.load_map(maps_path, ccmth_com_path)` (476 area).
- Produces: strain's `roi` Param carries `roi_group="crop"`, `roi_axis="both"`; `strain.roi_previews(params) -> list[tuple[str, Callable[[], tuple[np.ndarray, float, float]]]]` (empty list when the maps file can't be resolved).

- [ ] **Step 1: Write the failing test** — create `tests/test_strain_roi_previews.py`:

```python
"""strain.roi_previews resolves a CoM map layer + pixel scales for the ROI picker."""

from __future__ import annotations

import h5py
import numpy as np

from dfxm.stages import strain as ST


def test_roi_param_is_tagged():
    spec = ST.STAGE
    roi = next(p for p in spec.params if p.name == "roi")
    assert roi.roi_group and roi.roi_axis == "both"


def test_roi_previews_reads_com_map(tmp_path):
    maps = tmp_path / "maps.h5"
    com = np.arange(20 * 12, dtype=float).reshape(20, 12)
    with h5py.File(maps, "w") as f:
        f.create_dataset("/entry/ccmth/Center of mass/Center of mass", data=com)
    params = {
        "mode": "single",
        "folder": str(tmp_path),
        "maps_filename": "maps.h5",
        "ccmth_com_path": "/entry/ccmth/Center of mass/Center of mass",
        "pixel_size_x_um": 0.152,
        "pixel_size_y_um": 0.385,
    }
    previews = ST.roi_previews(params)
    assert previews, "expected at least one preview"
    arr, sx, sy = previews[0][1]()
    assert arr.shape == (20, 12)
    assert (sx, sy) == (0.152, 0.385)


def test_roi_previews_missing_file_returns_empty():
    assert ST.roi_previews({"mode": "single", "folder": "/no/such", "maps_filename": "maps.h5"}) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_strain_roi_previews.py -q`
Expected: FAIL — `roi` untagged and `roi_previews` undefined.

- [ ] **Step 3: Tag the `roi` Param** in `dfxm/stages/strain.py` (find the `Param(name="roi", ...)` block ~168-178) — add the two kwargs:

```python
            roi_group="crop",
            roi_axis="both",
```

- [ ] **Step 4: Add `roi_previews`** to `dfxm/stages/strain.py` (module level, near `_derive_maps_path`):

```python
def roi_previews(params: dict):
    """(label, thunk) previews for the ROI picker: the CoM map + pixel scales.

    Best-effort: returns [] when the maps file can't be resolved from the form.
    The picker draws on the same 2-D map the run-time ``roi`` crops.
    """
    import os

    p = dict(params)
    try:
        layer = ""
        if p.get("mode") == "single":
            maps_path = os.path.join(p.get("folder", ""), p.get("maps_filename", "maps.h5"))
        else:
            maps_path = _derive_maps_path(layer, p)
    except Exception:  # noqa: BLE001
        return []
    if not maps_path or not os.path.exists(maps_path):
        return []
    ccmth_com_path = p.get("ccmth_com_path", "/entry/ccmth/Center of mass/Center of mass")
    sx = float(p.get("pixel_size_x_um", 0.152))
    sy = float(p.get("pixel_size_y_um", 0.385))

    def _thunk(_mp=maps_path, _ds=ccmth_com_path, _sx=sx, _sy=sy):
        import numpy as np

        return np.asarray(load_map(_mp, _ds), dtype=float), _sx, _sy

    return [(f"CoM map · {os.path.basename(maps_path)}", _thunk)]
```

> **Note for the implementer:** verify `load_map` and `_derive_maps_path` names/signatures by reading `dfxm/stages/strain.py:469-498` before wiring; adjust the single-vs-multi branch to whatever param names strain actually uses for the folder (`folder`/`input_folder`). Keep the "return [] on any failure" contract.

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_strain_roi_previews.py -q`
Expected: PASS.

- [ ] **Step 6: Update docs** — `docs/Usage.md` (strain: the `roi` field now has a "Pick ROI…" picker) + `docs/Codebase.md` (strain `roi_previews`).

- [ ] **Step 7: Commit**

```bash
ruff check . && ruff format .
git add dfxm/stages/strain.py tests/test_strain_roi_previews.py docs/Usage.md docs/Codebase.md
git commit -m "feat(strain): tag roi param + roi_previews for the ROI picker"
```

---

## Task 10: visualize / paraview / slices — tag `roi_x`/`roi_y` + `roi_previews`

**Files:**
- Modify: `dfxm/stages/visualize.py` (tag `roi_x`/`roi_y` ~170-189; add `roi_previews`)
- Modify: `dfxm/stages/paraview.py` (tag `roi_x`/`roi_y` ~164-183; add `roi_previews`)
- Modify: `dfxm/stages/slices.py` (tag `align_roi_x`/`align_roi_y` ~252-275; add `roi_previews`)
- Test: `tests/test_volume_roi_previews.py`

**Interfaces:**
- Consumes: `figures.load_middle_layer` (Task 5).
- Produces: each stage's x/y ROI params carry `roi_group="crop"` with `roi_axis="x"`/`"y"`; each stage exposes `roi_previews(params) -> list[(label, thunk)]` reading the middle Z layer of one dataset in the stacked mosa/strain volume file.

- [ ] **Step 1: Write the failing test** — create `tests/test_volume_roi_previews.py`:

```python
"""visualize/paraview/slices: roi params tagged + roi_previews read a stacked-volume layer."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from dfxm.stages import paraview as PV
from dfxm.stages import slices as SL
from dfxm.stages import visualize as VZ


def _mosa_volume(path):
    with h5py.File(path, "w") as f:
        for grp in ("chi", "mu"):
            g = f.create_group(grp)
            g.create_dataset("Center of mass", data=np.arange(4 * 6 * 5).reshape(4, 6, 5).astype(float))
            g.create_dataset("FWHM", data=np.abs(np.arange(4 * 6 * 5).reshape(4, 6, 5)).astype(float))


@pytest.mark.parametrize(
    "mod,xname,yname",
    [(VZ, "roi_x", "roi_y"), (PV, "roi_x", "roi_y"), (SL, "align_roi_x", "align_roi_y")],
)
def test_roi_params_tagged(mod, xname, yname):
    by_name = {p.name: p for p in mod.STAGE.params}
    assert by_name[xname].roi_group and by_name[xname].roi_axis == "x"
    assert by_name[yname].roi_group and by_name[yname].roi_axis == "y"
    assert by_name[xname].roi_group == by_name[yname].roi_group  # same picker target


@pytest.mark.parametrize("mod", [VZ, PV, SL])
def test_roi_previews_reads_volume_layer(mod, tmp_path):
    vol = tmp_path / "mosa.h5"
    _mosa_volume(str(vol))
    params = {"mosa_volume_file": str(vol), "strain_volume_file": "",
              "pixel_size_x_um": 0.152, "pixel_size_y_um": 0.385}
    previews = mod.roi_previews(params)
    assert previews
    arr, sx, sy = previews[0][1]()
    assert arr.shape == (6, 5)  # middle Z layer of (4,6,5)
    assert (sx, sy) == (0.152, 0.385)


@pytest.mark.parametrize("mod", [VZ, PV, SL])
def test_roi_previews_missing_file_empty(mod):
    assert mod.roi_previews({"mosa_volume_file": "", "strain_volume_file": ""}) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_volume_roi_previews.py -q`
Expected: FAIL — params untagged, `roi_previews` undefined.

- [ ] **Step 3: Tag the params.** In each stage add to the x-param `Param(...)` block `roi_group="crop", roi_axis="x"` and to the y-param `roi_group="crop", roi_axis="y"`:
  - `dfxm/stages/visualize.py`: `roi_x` (~170), `roi_y` (~181).
  - `dfxm/stages/paraview.py`: `roi_x` (~164), `roi_y` (~174).
  - `dfxm/stages/slices.py`: `align_roi_x` (~253), `align_roi_y` (~265).

- [ ] **Step 4: Add a shared `roi_previews` to each of the three stages.** The body is identical (reads the mosa or strain stacked volume). Add this function to each of `visualize.py`, `paraview.py`, `slices.py`:

```python
def roi_previews(params: dict):
    """(label, thunk) previews for the ROI picker: middle Z layer of a stacked volume."""
    import os

    from ..common.figures import load_middle_layer

    p = dict(params)
    sx = float(p.get("pixel_size_x_um", 0.152))
    sy = float(p.get("pixel_size_y_um", 0.385))
    out = []
    mosa = p.get("mosa_volume_file", "")
    strain = p.get("strain_volume_file", "")
    if mosa and os.path.exists(mosa):
        for ds in ("chi/Center of mass", "mu/Center of mass"):
            try:
                with __import__("h5py").File(mosa, "r") as f:
                    if ds not in f:
                        continue
                out.append(
                    (
                        f"{ds} · {os.path.basename(mosa)}",
                        (lambda _m=mosa, _d=ds: (load_middle_layer(_m, _d), sx, sy)),
                    )
                )
            except Exception:  # noqa: BLE001
                continue
    if strain and os.path.exists(strain):
        out.append(
            (
                f"strain · {os.path.basename(strain)}",
                (lambda _s=strain: (load_middle_layer(_s, "strain"), sx, sy)),
            )
        )
    return out
```

> **Note for the implementer:** the three copies are byte-identical — that is acceptable here (each stage owns its own preview provider, consistent with the per-site adapter design; there is no shared stage base module). If you prefer, hoist it to a private helper in `dfxm/common/figures.py` (e.g. `stacked_volume_previews(params)`) and have each stage's `roi_previews` delegate — but keep the stage-level name `roi_previews` so `StageView` (Task 11) resolves it uniformly. Verify the strain-volume dataset key (`"strain"`) against `load_strain_volume` in each module before finalizing.

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_volume_roi_previews.py -q`
Expected: PASS.

- [ ] **Step 6: Update docs** — `docs/Usage.md` (visualize/paraview/slices: `roi_x`/`roi_y` fields gain a "Pick ROI…" picker) + `docs/Codebase.md` (the three `roi_previews`).

- [ ] **Step 7: Commit**

```bash
ruff check . && ruff format .
git add dfxm/stages/visualize.py dfxm/stages/paraview.py dfxm/stages/slices.py tests/test_volume_roi_previews.py docs/Usage.md docs/Codebase.md
git commit -m "feat(volume stages): tag roi_x/roi_y + roi_previews for the ROI picker"
```

---

## Task 11: StageView — schema-driven "Pick ROI…" button + write-back

**Files:**
- Modify: `gui/stage_view.py` (build ROI-group buttons in `__init__` btn_row area ~114-129; add `_on_pick_roi_group`)
- Test: `tests/gui_smoke.py` (add a numbered step) + `tests/test_gui_stage_view_roi.py`

**Interfaces:**
- Consumes: `self._spec.params` (each `Param.roi_group`/`roi_axis`); the stage module's `roi_previews(vals)` (Tasks 9-10); `ROIPickerDialog`; `self._form.values()` / `self._form.set_values(...)`.
- Produces: for each distinct non-empty `roi_group` in the spec, a "Pick ROI…" button that opens the picker with the stage's previews and writes the encoded strings back into the group's params.

- [ ] **Step 1: Write the failing test** — create `tests/test_gui_stage_view_roi.py`:

```python
"""StageView shows a Pick ROI… button for roi-grouped specs and writes back per axis."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


def test_pick_roi_writes_pair_encoding(monkeypatch, tmp_path):
    from dfxm.config import Experiment
    from gui.main_window import MainWindow

    _ = QApplication.instance() or QApplication([])
    win = MainWindow(Experiment(name="t"))
    view = win._views["visualize"]  # roi_x/roi_y, roi_axis x/y
    assert view._roi_buttons, "expected a Pick ROI… button for the crop group"

    import gui.stage_view as SV

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (2, 6, 1, 8)  # r0,r1,c0,c1
        def exec(self):
            return 1

    monkeypatch.setattr(SV, "ROIPickerDialog", _FakePicker, raising=False)
    monkeypatch.setattr(SV, "_roi_previews_for", lambda name, vals: [("x", lambda: (None, 1, 1))])
    view._on_pick_roi_group("crop")
    vals = view._form.values()
    assert vals["roi_x"] == "1,8"   # c0,c1
    assert vals["roi_y"] == "2,6"   # r0,r1
    win.close()


def test_pick_roi_writes_both_encoding(monkeypatch):
    from dfxm.config import Experiment
    from gui.main_window import MainWindow

    _ = QApplication.instance() or QApplication([])
    win = MainWindow(Experiment(name="t"))
    view = win._views["strain"]  # single 'roi', roi_axis both

    import gui.stage_view as SV

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (2, 6, 1, 8)
        def exec(self):
            return 1

    monkeypatch.setattr(SV, "ROIPickerDialog", _FakePicker, raising=False)
    monkeypatch.setattr(SV, "_roi_previews_for", lambda name, vals: [("x", lambda: (None, 1, 1))])
    view._on_pick_roi_group("crop")
    assert view._form.values()["roi"] == "2,6,1,8"  # r0,r1,c0,c1
    win.close()
```

- [ ] **Step 2: Run to verify it fails**

Run: `python3 -m pytest tests/test_gui_stage_view_roi.py -q`
Expected: FAIL — `_roi_buttons` / `_on_pick_roi_group` / `_roi_previews_for` not present.

- [ ] **Step 3: Implement in `gui/stage_view.py`.**

Add a module-level resolver (near the top-level imports) so the picker stays schema-driven and testable via monkeypatch:

```python
def _roi_previews_for(stage_name: str, vals: dict):
    """Resolve a stage's roi_previews(params) provider, or [] if it has none."""
    mods = {}
    if stage_name == "strain":
        from dfxm.stages import strain as m
        mods["strain"] = m
    elif stage_name == "visualize":
        from dfxm.stages import visualize as m
        mods["visualize"] = m
    elif stage_name == "paraview":
        from dfxm.stages import paraview as m
        mods["paraview"] = m
    elif stage_name == "slices":
        from dfxm.stages import slices as m
        mods["slices"] = m
    mod = mods.get(stage_name)
    if mod is None or not hasattr(mod, "roi_previews"):
        return []
    return mod.roi_previews(vals)
```

In `StageView.__init__`, after building `btn_row` (~129, before `btn_row.addStretch(1)`), add one "Pick ROI…" button per distinct roi_group:

```python
        self._roi_buttons: dict[str, QPushButton] = {}
        seen_groups: list[str] = []
        for p in spec.params:
            if p.roi_group and p.roi_group not in seen_groups:
                seen_groups.append(p.roi_group)
        for grp in seen_groups:
            btn = QPushButton("Pick ROI…")
            btn.clicked.connect(lambda _checked=False, g=grp: self._on_pick_roi_group(g))
            btn_row.addWidget(btn)
            self._roi_buttons[grp] = btn
```

Add the handler:

```python
    def _on_pick_roi_group(self, roi_group: str) -> None:
        vals = self._form.values()
        previews = _roi_previews_for(self._stage_name, vals)
        if not previews:
            self._log.append("Pick ROI: no preview available (set the volume/map file first).")
            self._tabs.setCurrentWidget(self._log)
            return
        members = [p for p in self._spec.params if p.roi_group == roi_group]

        def _cur_initial():
            # pre-fill the picker from existing box values, if fully set
            axis = {p.roi_axis: str(vals.get(p.name, "") or "") for p in members}
            try:
                if "both" in axis:
                    r0, r1, c0, c1 = (int(t) for t in axis["both"].split(","))
                    return (r0, r1, c0, c1)
                c0, c1 = (int(t) for t in axis.get("x", "").split(","))
                r0, r1 = (int(t) for t in axis.get("y", "").split(","))
                return (r0, r1, c0, c1)
            except Exception:  # noqa: BLE001
                return None

        from .widgets.roi_picker import ROIPickerDialog  # imported on demand

        dlg = ROIPickerDialog(previews, initial=_cur_initial(), parent=self)
        if not (dlg.exec() and dlg.result):
            return
        r0, r1, c0, c1 = dlg.result
        updates: dict[str, str] = {}
        for p in members:
            if p.roi_axis == "both":
                updates[p.name] = f"{r0},{r1},{c0},{c1}"
            elif p.roi_axis == "x":
                updates[p.name] = f"{c0},{c1}"
            elif p.roi_axis == "y":
                updates[p.name] = f"{r0},{r1}"
        self._form.set_values(updates)
        self._log.append(f"Picked ROI → {updates}")
        self._tabs.setCurrentWidget(self._log)
```

Add `from .widgets.roi_picker import ROIPickerDialog` reference for the monkeypatch seam — the tests patch `gui.stage_view.ROIPickerDialog`, so import the name at module scope too:

```python
# near other lazy-usage comments; the actual open is lazy inside the handler, but expose the
# name at module scope so tests can monkeypatch it:
from .widgets.roi_picker import ROIPickerDialog  # noqa: F401,E402  (also imported lazily above)
```

> **Note for the implementer:** to keep GUI startup light per the lazy-deps rule, prefer NOT importing `roi_picker` at module top. Instead, in the handler use `import gui.stage_view as _self_mod` + `getattr(_self_mod, "ROIPickerDialog", None)`, falling back to the lazy `from .widgets.roi_picker import ROIPickerDialog`. Adjust the test's monkeypatch target to match whichever seam you choose (module attribute vs the widgets module). The behavioural assertions (encoded strings) are what matter.

- [ ] **Step 4: Add a gui_smoke step** — in `tests/gui_smoke.py`, after the forms step (~[8]), add a check that a roi-grouped stage view exposes the button:

```python
    vview = win._views["visualize"]
    assert getattr(vview, "_roi_buttons", None), "visualize StageView missing Pick ROI… button"
    print("[9] visualize StageView exposes a Pick ROI… button")
```

(Renumber subsequent smoke steps if needed.)

- [ ] **Step 5: Run to verify it passes**

Run: `python3 -m pytest tests/test_gui_stage_view_roi.py -q && python3 tests/gui_smoke.py`
Expected: PASS + smoke prints through the new step.

- [ ] **Step 6: Update `docs/Usage.md`** — a short "Picking an ROI" subsection: run-time crop params on strain/visualize/slices/paraview show a "Pick ROI…" button; per-stage (does not propagate to sibling stages); rocking is deliberately manual (raw-frame).

- [ ] **Step 7: Commit**

```bash
ruff check . && ruff format .
git add gui/stage_view.py tests/test_gui_stage_view_roi.py tests/gui_smoke.py docs/Usage.md
git commit -m "feat(stage-view): schema-driven Pick ROI… button + per-axis write-back"
```

---

## Task 12: Documentation sweep + pipeline consistency

**Files:**
- Modify: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:** none (docs only).

- [ ] **Step 1: Re-read the two docs** for the ROI-picker + clim changes and ensure coverage is complete and consistent:
  - `docs/Usage.md`: the ROI picker appears in (a) the strain/mosaicity/rocking replot section, (b) the slices replot section, (c) a "Picking an ROI" note for run-time crop params. The per-quantity colour limits (χ/μ + raw split) are described in the slices replot section. Include the one-line "the preview is oriented exactly like the exported maps (origin lower)" note.
  - `docs/Codebase.md`: `gui/widgets/roi_picker.py` (`rect_to_indices`, `ROIPickerDialog`), `figures.load_middle_layer`, `slices.plane_preview`, each stage's `roi_previews`, `Param.roi_group`/`roi_axis`, and the slices `render_replot` volume_id clim key are all present.

- [ ] **Step 2: Verify no stale references** — grep the docs for "per kind"/"per group" clim wording in the slices replot area and correct to "per quantity".

Run: `grep -n "per group\|per kind\|per-kind" docs/Usage.md docs/Codebase.md`
Expected: any hit in the slices-replot context is updated to "per quantity".

- [ ] **Step 3: Commit** (only if this task made doc edits not already committed with their code task)

```bash
git add docs/Usage.md docs/Codebase.md
git commit -m "docs: ROI picker + per-quantity clim (Usage + Codebase sync)"
```

---

## Task 13: Full verification + review

**Files:** none (verification).

- [ ] **Step 1: Full suite**

Run: `python3 -m pytest -q`
Expected: all pass (prior baseline 409 passed / 13 skipped, now higher). No new warnings.

- [ ] **Step 2: GUI smoke**

Run: `python3 tests/gui_smoke.py`
Expected: exits 0, all numbered steps print.

- [ ] **Step 3: Lint + format**

Run: `ruff check . && ruff format --check .`
Expected: clean.

- [ ] **Step 4: Manual eyeball checklist** (record results in the handoff note, do not block the branch on hardware-only checks): open the GUI, load a real `oblique_slices.h5`, open the slices Replot dialog → confirm one clim row per χ/μ and each raw; click "Pick ROI…" → the plane shows stretched (physical) aspect matching an export, drag a box, confirm the r0/r1/c0/c1 boxes fill and a render honours the crop; repeat on the generic replot dialog and on the visualize/strain/slices/paraview run-time crop buttons.

- [ ] **Step 5: Request whole-branch review** (fable final review per repo convention) before merge.

---

## Self-Review (author checklist — completed at write time)

**Spec coverage:**
- Section A (per-quantity clim) → Tasks 1 (core) + 2 (GUI). ✓ (incl. raw_* split, group fallback)
- Section B (ROIPickerDialog) → Tasks 3 (rect_to_indices) + 4 (dialog). ✓ (origin lower, set_aspect(sy/sx), readout, initial, shape-change clear)
- Section C (preview helpers) → Task 5 (`load_middle_layer`, `plane_preview`) + Tasks 9/10 (`roi_previews`). ✓
- Section D wiring: replot dialogs P1/P2 → Tasks 6/7; schema fields → Task 8; run-time tagging + providers → Tasks 9/10; StageView button + per-axis write-back, per-stage only → Task 11; rocking excluded → not tagged (Tasks 9/10 omit rocking). ✓
- Testing → each task’s tests + Task 13. Docs same-change → in-task doc steps + Task 12 sweep. ✓

**Placeholder scan:** two "Note for the implementer" blocks (Tasks 9, 10, 11) flag real verification-before-wiring points with concrete file:line anchors — not placeholders for missing content; the code to write is fully given. No TBD/TODO.

**Type consistency:** thunk contract `() -> (array2d, sx, sy)` uniform across `plane_preview`, `roi_previews`, and the replot `preview_fn` closures; `rect_to_indices(...) -> (r0,r1,c0,c1)` and `ROIPickerDialog.result` agree; `preview_fn(h5, key)` (Task 6) vs `roi_previews(params)` (Tasks 9/10) are distinct-by-design (catalog-key vs form-driven) and each call site uses the matching one; `roi_axis` values `{"","x","y","both"}` consistent across Tasks 8/9/10/11.
