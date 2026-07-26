# Slice Marks + UX Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the four approved features from
`docs/superpowers/specs/2026-07-26-slice-marks-and-ux-features-design.md`:
scroll-wheel guard on param forms, line-picker background switch, persistent
oblique-slice marks with a visual marking dialog, marks→profiles-jobs bridge,
plus a tooltip precision sweep.

**Architecture:** Marks live as `/marks/<slice_name>` offset datasets inside
`oblique_slices.h5`, read/written by Qt-free helpers in `dfxm/stages/slices.py`;
every root-group enumerator learns to skip that group. A new shared
`PlaneBrowser` widget (extracted from `LinePickerDialog`) renders planes for
both the line picker (which gains a background dropdown) and the new
`MarkPlanesDialog`. The profiles stage gains a "Jobs from marks…" flow that
opens one line picker per checked mark and appends complete jobs.

**Tech Stack:** Python 3.10, PySide6, h5py, numpy, matplotlib (Figure API only),
pytest with `QT_QPA_PLATFORM=offscreen`.

## Global Constraints

- `dfxm/` stays Qt-free: never import PySide6/pyvista there.
- Matplotlib via explicit `matplotlib.figure.Figure` API only — never `pyplot`,
  never `matplotlib.use(...)`.
- Docs contract: any task changing stage behaviour/params or GUI behaviour
  updates `docs/Usage.md` (user-visible) and `docs/Codebase.md` (structure)
  **in the same commit**.
- Ruff: line length 100, double quotes, target py310, rules E/F/I. `ruff format`
  runs automatically on Write/Edit via hook.
- User-facing input errors raise `StageUserError(message, hint=...)` from
  `dfxm.common.errors`.
- Read every edit-target region before the first Edit (repo rule; `hint=`
  strings contain em-dashes at varying indents — never reconstruct
  `old_string` from memory).
- This repo has no git remote — no pull/push/PR steps.
- Run the full check before claiming a task done:
  `python3 -m pytest -q && ruff check .` (suite baseline at plan time:
  785 passed / 13 skipped / 0 warnings).

**Branch:** create `slice-marks-ux` off master before Task 1:
`git checkout -b slice-marks-ux`

---

### Task 1: Scroll-wheel guard on param forms

**Files:**
- Create: `gui/widgets/wheel_guard.py`
- Modify: `gui/widgets/param_form.py` (methods `_enum_editor`, `_int_editor`,
  `_float_editor`; lines ≈250–283)
- Test: `tests/test_gui_wheel_guard.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: nothing new.
- Produces: `install_wheel_guard(widget: QWidget) -> None` in
  `gui.widgets.wheel_guard` (idempotent per widget; safe to call on any
  widget). Later tasks do not depend on this.

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_wheel_guard.py`:

```python
"""Offscreen tests: wheel over an unfocused spin/combo field must not edit it."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import QPoint, QPointF, Qt  # noqa: E402
from PySide6.QtGui import QWheelEvent  # noqa: E402
from PySide6.QtWidgets import QApplication, QComboBox, QDoubleSpinBox, QSpinBox  # noqa: E402

from dfxm.config.models import Param, ParamType  # noqa: E402
from gui.widgets.param_form import ParamForm  # noqa: E402


def _wheel(widget, delta=120):
    ev = QWheelEvent(
        QPointF(5, 5),
        widget.mapToGlobal(QPointF(5, 5)),
        QPoint(0, 0),
        QPoint(0, delta),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, ev)


def _form():
    params = [
        Param("count", ParamType.INT, "Count", default=3),
        Param("ratio", ParamType.FLOAT, "Ratio", default=1.5),
        Param("mode", ParamType.ENUM, "Mode", default="a", choices=("a", "b", "c")),
    ]
    return ParamForm(params)


def test_unfocused_fields_ignore_wheel():
    _app = QApplication.instance() or QApplication([])
    form = _form()
    spin = form._editors["count"]
    dspin = form._editors["ratio"]
    combo = form._editors["mode"]
    assert isinstance(spin, QSpinBox)
    assert isinstance(dspin, QDoubleSpinBox)
    assert isinstance(combo, QComboBox)
    for w in (spin, dspin, combo):
        assert w.focusPolicy() == Qt.FocusPolicy.StrongFocus
        assert not w.hasFocus()
    _wheel(spin)
    _wheel(dspin)
    _wheel(combo)
    assert form.values() == {"count": 3, "ratio": 1.5, "mode": "a"}


def test_focused_spinbox_still_wheels():
    _app = QApplication.instance() or QApplication([])
    form = _form()
    form.show()
    QApplication.processEvents()
    spin = form._editors["count"]
    spin.setFocus()
    QApplication.processEvents()
    if not spin.hasFocus():
        pytest.skip("offscreen platform denied focus")
    _wheel(spin)
    assert form.values()["count"] == 4
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_gui_wheel_guard.py -v`
Expected: FAIL — `ModuleNotFoundError`/assert on focus policy (default is
`WheelFocus` for spin boxes, and values change on wheel).

- [ ] **Step 3: Create `gui/widgets/wheel_guard.py`**

```python
"""Stop unfocused arrow-fields from eating scroll-wheel events.

Spin boxes and combo boxes change value on wheel by default, so scrolling a
form inside a QScrollArea silently edits fields under the cursor — and form
persistence then saves the stray edit. The guard sets StrongFocus (wheel can
no longer *give* focus) and swallows wheel events while the widget is
unfocused; because the event is left unaccepted, Qt propagates it up to the
scroll area, which scrolls the page as expected.
"""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt


class _WheelGuard(QObject):
    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt API
        if event.type() == QEvent.Type.Wheel and not obj.hasFocus():
            event.ignore()
            return True
        return False


_guard: _WheelGuard | None = None


def install_wheel_guard(widget) -> None:
    """Make *widget* ignore wheel events unless it has keyboard focus."""
    global _guard
    if _guard is None:
        _guard = _WheelGuard()
    widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    widget.installEventFilter(_guard)
```

- [ ] **Step 4: Wire it into `ParamForm`**

In `gui/widgets/param_form.py`: add the import after the existing
`from .help_panel import param_help_html`:

```python
from .wheel_guard import install_wheel_guard
```

Then add one call in each of the three wheel-sensitive editor builders
(Read the methods first; insert directly after the widget is constructed):

- `_enum_editor`: after `box = QComboBox()` → `install_wheel_guard(box)`
- `_int_editor`: after `sb = QSpinBox()` → `install_wheel_guard(sb)`
- `_float_editor`: after `sb = QDoubleSpinBox()` → `install_wheel_guard(sb)`

- [ ] **Step 5: Run the test to verify it passes**

Run: `python3 -m pytest tests/test_gui_wheel_guard.py -v`
Expected: 2 passed (or 1 passed + 1 skipped if offscreen denies focus).

- [ ] **Step 6: Docs (same commit)**

- `docs/Usage.md`: find the section describing stage forms (grep for
  `Advanced (` or "stage form"); add one sentence: scrolling a form never
  changes spin/dropdown values any more — a field only reacts to the wheel
  after you click into it.
- `docs/Codebase.md`: add `wheel_guard.py` to the `gui/widgets` listing with a
  one-line description, and mention in the `param_form` entry that all
  spin/combo editors are wheel-guarded.

- [ ] **Step 7: Full check + commit**

Run: `python3 -m pytest -q && ruff check .`
Expected: baseline + 2 new tests pass, ruff clean.

```bash
git add gui/widgets/wheel_guard.py gui/widgets/param_form.py \
  tests/test_gui_wheel_guard.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): wheel over unfocused spin/combo fields scrolls the page, never edits"
```

---

### Task 2: Marks core (read/write) + reader hardening

**Files:**
- Modify: `dfxm/stages/slices.py` (new constant + two functions after
  `build_pinned_spec` ≈line 1023; harden `_find_slice_group` ≈957,
  `figures` ≈1316, `replot_catalog` ≈1388)
- Modify: `dfxm/stages/profiles.py` (`list_volume_ids` ≈420)
- Test: `tests/test_slices_marks.py`
- Docs: `docs/Codebase.md`

**Interfaces:**
- Consumes: existing `_find_slice_group(f, slice_name, volume=None)`,
  `StageUserError`.
- Produces (used by Tasks 4–6):
  - `dfxm.stages.slices.MARKS_GROUP: str = "marks"`
  - `dfxm.stages.slices.read_marks(h5_path_or_file) -> dict[str, list[float]]`
    (accepts a path **or** an open `h5py.File`; missing/malformed `/marks` →
    `{}` / skipped entries; offsets sorted)
  - `dfxm.stages.slices.write_marks(h5_path, slice_name, offsets_um) -> list[float]`
    (snaps to nearest stored planes, dedupes, sorts, replaces the dataset;
    empty list deletes it; returns the snapped list; `StageUserError` on
    unwritable file or unknown slice)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slices_marks.py`:

```python
"""Marks storage in oblique_slices.h5: round-trip, snapping, reader hardening."""

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from dfxm.common.errors import StageUserError
from dfxm.stages import profiles as pr
from dfxm.stages import slices as sl


def _mini(path, offsets=(-2.0, 0.0, 2.0)):
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offs = np.asarray(offsets, np.float64)
    with h5py.File(path, "w") as f:
        for vid in ("raw_sum", "strain"):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cmap"] = "gray"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "v"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("oblique_full")
            sg.create_dataset(
                "slices", data=np.zeros((offs.size, v.size, u.size), dtype=np.float32)
            )
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offs)
            for key, val in (
                ("normal", (0.0, 0.0, 1.0)),
                ("origin", (0.0, 0.0, 0.0)),
                ("up", (0.0, 1.0, 0.0)),
                ("u_hat", (1.0, 0.0, 0.0)),
                ("v_hat", (0.0, 1.0, 0.0)),
                ("n_hat", (0.0, 0.0, 1.0)),
            ):
                sg.attrs[key] = np.asarray(val, np.float64)
            for key, val in (
                ("half_u", 4.0),
                ("half_v", 3.0),
                ("du", 1.0),
                ("dv", 1.0),
                ("sweep_step_um", 2.0),
            ):
                sg.attrs[key] = float(val)
            sg.attrs["n_planes"] = int(offs.size)
    return str(path)


def test_write_and_read_marks_roundtrip(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    snapped = sl.write_marks(h5, "oblique_full", [0.3, -1.7, 0.4])
    assert snapped == [-2.0, 0.0]  # snapped to stored planes, deduped, sorted
    assert sl.read_marks(h5) == {"oblique_full": [-2.0, 0.0]}
    with h5py.File(h5, "r") as f:  # open-file variant
        assert sl.read_marks(f) == {"oblique_full": [-2.0, 0.0]}


def test_write_marks_replaces_and_deletes(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    sl.write_marks(h5, "oblique_full", [0.0])
    sl.write_marks(h5, "oblique_full", [2.0])  # replace, not append
    assert sl.read_marks(h5) == {"oblique_full": [2.0]}
    sl.write_marks(h5, "oblique_full", [])  # empty -> dataset and group gone
    assert sl.read_marks(h5) == {}
    with h5py.File(h5, "r") as f:
        assert sl.MARKS_GROUP not in f


def test_write_marks_unknown_slice_raises(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    with pytest.raises(StageUserError):
        sl.write_marks(h5, "nope", [0.0])


def test_read_marks_absent_and_malformed(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    assert sl.read_marks(h5) == {}  # no /marks group at all
    with h5py.File(h5, "a") as f:
        mg = f.require_group(sl.MARKS_GROUP)
        mg.create_group("weird_subgroup")  # non-dataset child: skipped
        mg.create_dataset("strs", data=np.bytes_([b"x"]))  # non-numeric: skipped
        mg.create_dataset("oblique_full", data=np.asarray([2.0], np.float64))
    assert sl.read_marks(h5) == {"oblique_full": [2.0]}


def test_readers_skip_marks_group(tmp_path):
    h5 = _mini(tmp_path / "s.h5")
    sl.write_marks(h5, "oblique_full", [0.0])
    # replot_catalog: volume groups only
    cat = sl.replot_catalog(h5)
    assert sorted({e.volume_id for e in cat}) == ["raw_sum", "strain"]
    # build_pinned_spec: still resolves geometry (would crash on /marks before)
    specs = sl.build_pinned_spec(h5, "oblique_full", [0.0])
    assert len(specs) == 1
    # figures catalog: one spec per (vid, slice, plane) — none for /marks
    result = SimpleNamespace(output_h5=h5)
    specs = sl.figures(result, {})
    assert len(specs) == 6  # 2 vids x 1 slice x 3 planes
    # profiles enumerators
    with h5py.File(h5, "r") as f:
        assert pr.list_volume_ids(f) == ["raw_sum", "strain"]
        assert pr.volume_ids_with_slice(f, "oblique_full") == ["raw_sum", "strain"]
```

Note: if `sl.figures` turns out to need more of `SlicesResult` than
`output_h5`, Read `figures()` and extend the `SimpleNamespace` with the
missing attributes (keep the test's intent: a marked file yields exactly the
6 volume specs).

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_slices_marks.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'write_marks'`.

- [ ] **Step 3: Implement in `dfxm/stages/slices.py`**

Read the region around lines 950–1030 first. Add near the top of the file
(after the imports, before the first constant/dataclass):

```python
# Root-level group holding starred plane offsets (/marks/<slice_name>); every
# enumerator of oblique_slices.h5 root groups must skip it.
MARKS_GROUP = "marks"
```

Harden `_find_slice_group` (≈line 957) — replace its body's first lines:

```python
    vids = [volume] if volume else [k for k in f.keys() if k != MARKS_GROUP]
    for vid in vids:
        if vid not in f or not isinstance(f[vid], h5py.Group):
            continue
        g = f[vid]
        if (
            slice_name in g
            and isinstance(g[slice_name], h5py.Group)
            and "offsets_um" in g[slice_name]
        ):
            return vid, g[slice_name]
```

(keep the existing `raise StageUserError(...)` tail unchanged).

Add a new section after `build_pinned_spec` (≈line 1023):

```python
# -----------------------------------------------------------------------------
# Marks (starred planes; shared by the GUI dialogs and the profiles bridge)
# -----------------------------------------------------------------------------
def read_marks(h5_path_or_file) -> dict[str, list[float]]:
    """All marked offsets, ``{slice_name: [offset_um, ...]}`` (sorted).

    Accepts a path or an open ``h5py.File``. A missing ``/marks`` group means
    no marks; malformed children (non-datasets, non-numeric) are skipped, so a
    hand-edited file degrades to fewer marks, never an error.
    """

    def _read(f):
        out: dict[str, list[float]] = {}
        mg = f.get(MARKS_GROUP)
        if not isinstance(mg, h5py.Group):
            return out
        for sname, ds in mg.items():
            if not isinstance(ds, h5py.Dataset):
                continue
            try:
                offs = np.asarray(ds[()], np.float64).ravel()
            except (TypeError, ValueError):
                continue
            out[str(sname)] = sorted(float(o) for o in offs)
        return out

    if isinstance(h5_path_or_file, h5py.File):
        return _read(h5_path_or_file)
    with h5py.File(h5_path_or_file, "r") as f:
        return _read(f)


def write_marks(h5_path, slice_name, offsets_um) -> list[float]:
    """Replace *slice_name*'s marks with *offsets_um* (snapped to stored planes).

    Offsets snap to the nearest stored plane (the ``resolve_plane_index``
    rule), collapse duplicates, and are stored sorted; an empty list deletes
    the dataset (and the ``/marks`` group once it is empty). Returns the
    snapped offsets actually stored.
    """
    try:
        fh = h5py.File(h5_path, "r+")
    except OSError as exc:
        raise StageUserError(
            f"cannot open {h5_path!r} for writing marks: {exc}",
            hint="Close any dialog/viewer holding the file open, then retry; "
            "the file must be an oblique_slices.h5 from a slices run.",
        ) from exc
    with fh as f:
        _vid, sg = _find_slice_group(f, slice_name)
        stored = sg["offsets_um"][:].astype(np.float64)
        snapped = sorted(
            {float(stored[int(np.argmin(np.abs(stored - float(o))))]) for o in offsets_um}
        )
        mg = f.require_group(MARKS_GROUP)
        if slice_name in mg:
            del mg[slice_name]
        if snapped:
            mg.create_dataset(slice_name, data=np.asarray(snapped, np.float64))
        elif len(mg.keys()) == 0:
            del f[MARKS_GROUP]
    return snapped
```

Harden `figures()` (Read the loop at ≈1316 first) — the inner loop becomes:

```python
        for vid in f.keys():
            vg = f[vid]
            if vid == MARKS_GROUP or not isinstance(vg, h5py.Group):
                continue
            for sname in vg.keys():
                sub = vg[sname]
                if not (isinstance(sub, h5py.Group) and "slices" in sub):
                    continue
                n_planes = sub["slices"].shape[0]
```

(the existing `for k in range(n_planes):` body, the `build` closure and the
`FigureSpec(...)` append stay exactly as they are).

Harden `replot_catalog` (≈1388): after `vg = f[vid]`, extend the existing
guard to also skip marks:

```python
            if vid == MARKS_GROUP or not isinstance(vg, h5py.Group):
                continue
```

- [ ] **Step 4: Harden `profiles.list_volume_ids`**

In `dfxm/stages/profiles.py`, add to the relative-import block at the top
(no circularity — `slices.py` does not import `profiles`):

```python
from .slices import MARKS_GROUP
```

and change `list_volume_ids` (≈line 420) to:

```python
def list_volume_ids(f):
    return [k for k in f.keys() if k != MARKS_GROUP and isinstance(f[k], h5py.Group)]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_slices_marks.py -v`
Expected: 5 passed.

- [ ] **Step 6: Docs (same commit)**

`docs/Codebase.md`: in the `dfxm/stages` → slices section, document
`MARKS_GROUP`, `read_marks`, `write_marks` and note that every root-group
enumerator of `oblique_slices.h5` (slices `figures`/`replot_catalog`/
`_find_slice_group`, profiles `list_volume_ids`) skips `/marks`. Mention the
file-format addition in the data-flow table row for `oblique_slices.h5`.

- [ ] **Step 7: Full check + commit**

Run: `python3 -m pytest -q && ruff check .`
Expected: green.

```bash
git add dfxm/stages/slices.py dfxm/stages/profiles.py \
  tests/test_slices_marks.py docs/Codebase.md
git commit -m "feat(slices): persistent plane marks in oblique_slices.h5 (/marks) + reader hardening"
```

---

### Task 3: Shared PlaneBrowser + line-picker background switch

**Files:**
- Create: `gui/widgets/plane_browser.py`
- Rewrite: `gui/widgets/line_picker.py`
- Modify: `gui/viewers.py` (`inject_line_into_jobs` ≈60–91),
  `gui/stage_view.py` (`_on_pick_line` ≈413–453)
- Test: `tests/test_gui_plane_browser.py` (new);
  `tests/test_gui_line_picker_fields.py` must pass **unchanged**
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `profiles.volume_ids_with_slice / _pick_reference_id / read_axes /
  read_volume_attrs / resolve_plane_index / list_volume_ids`,
  `render.cmap_nan_transparent`.
- Produces (used by Tasks 4 and 6):
  - `PlaneBrowser(h5_path, parent=None)` QWidget with: attributes
    `present: list[str]`, `slice_name: str | None`, `group_id: str | None`,
    `plane_index: int`, `offsets/u/v: np.ndarray`, `attrs: dict`,
    `ax`, `canvas`, `post_draw: Callable | None`, property `file` (open
    `h5py.File`); methods `slice_names() -> list[str]`,
    `open_slice(slice_name, *, ref_pref="", init_offset=0.0)`,
    `set_group(vid)`, `set_plane(idx)`, `step(d)`,
    `current_offset() -> float`, `redraw()`, `close_file()`, `reopen()`;
    signal `viewChanged` (no args, emitted after every redraw).
  - `LinePickerDialog(h5_path, slice_name, init_offset=0.0, ref_pref="",
    parent=None)` — public surface kept: `_pts`, `_field_boxes`, `_use`,
    `_refresh_use_button()`, `selected_fields()`, `field_restriction()`,
    `done()`; **`result` is now a 5-tuple**
    `(start_uv, end_uv, offset_um, fields, reference)`.
  - `inject_line_into_jobs(..., fields=None, reference=None)` — writes
    `target["reference"]` when *reference* is truthy, leaves it untouched
    otherwise.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_plane_browser.py`:

```python
"""Offscreen tests: shared PlaneBrowser + line-picker background switch."""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


def _mini(path, offsets=(-2.0, 0.0, 2.0)):
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offs = np.asarray(offsets, np.float64)
    with h5py.File(path, "w") as f:
        for vid in ("raw_sum", "strain"):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cmap"] = "gray"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "v"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("oblique_full")
            sg.create_dataset(
                "slices", data=np.zeros((offs.size, v.size, u.size), dtype=np.float32)
            )
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offs)
    return str(path)


def test_browser_open_step_and_group_switch(tmp_path):
    from gui.widgets.plane_browser import PlaneBrowser

    _app = QApplication.instance() or QApplication([])
    b = PlaneBrowser(_mini(tmp_path / "s.h5"))
    assert b.slice_names() == ["oblique_full"]
    b.open_slice("oblique_full", init_offset=1.7)
    assert b.present == ["raw_sum", "strain"]
    assert b.group_id == "raw_sum"  # raw_sum preferred as reference
    assert b.plane_index == 2  # 1.7 snaps to +2.0
    b.step(-1)
    assert b.current_offset() == 0.0
    b.set_group("strain")
    assert b.attrs["title"] == "strain"
    assert b.plane_index == 1  # plane cursor survives the switch
    b.close_file()


def test_picker_background_dropdown_and_result_reference(tmp_path):
    from gui.widgets.line_picker import LinePickerDialog

    _app = QApplication.instance() or QApplication([])
    dlg = LinePickerDialog(_mini(tmp_path / "s.h5"), "oblique_full")
    assert dlg._bg.currentText() == "raw_sum"
    dlg._pts = [(0.0, 0.0), (1.0, 0.5)]
    dlg._bg.setCurrentText("strain")  # switch background
    assert dlg._pts == [(0.0, 0.0), (1.0, 0.5)]  # picked points survive
    assert dlg._browser.attrs["title"] == "strain"
    dlg._refresh_use_button()
    dlg._accept()
    start, end, off, fields, reference = dlg.result
    assert (start, end) == ((0.0, 0.0), (1.0, 0.5))
    assert fields is None  # all fields checked -> no restriction
    assert reference == "strain"  # the group the line was drawn against
    dlg.done(0)


def test_inject_line_reference_kwarg():
    from gui.viewers import inject_line_into_jobs

    base = json.dumps([{"name": "oblique_full", "offset_um": 0.0, "reference": "old"}])
    out = inject_line_into_jobs(
        base, "oblique_full", (0.0, 0.0), (1.0, 0.0), 0.0, reference="strain"
    )
    assert json.loads(out)[0]["reference"] == "strain"
    out = inject_line_into_jobs(base, "oblique_full", (0.0, 0.0), (1.0, 0.0), 0.0)
    assert json.loads(out)[0]["reference"] == "old"  # None leaves it untouched
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_plane_browser.py -v`
Expected: FAIL — no module `gui.widgets.plane_browser`.

- [ ] **Step 3: Create `gui/widgets/plane_browser.py`**

```python
"""Shared read-only plane browser over one oblique_slices.h5.

Owns the open h5 handle, the (slice, group, plane) cursor, and one matplotlib
canvas that redraws the current plane with the group's stored cmap/clim.
LinePickerDialog and MarkPlanesDialog compose it and add their own controls;
owner overlays go through ``post_draw`` and owners resync labels on
``viewChanged``. Built lazily on dialog open — never at stage-view
construction.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QVBoxLayout, QWidget

from dfxm.common import render as _rnd
from dfxm.stages import profiles as _pr


class PlaneBrowser(QWidget):
    """One plane of one field group of one slice, with stepping and switching."""

    viewChanged = Signal()  # emitted after every redraw

    def __init__(self, h5_path, parent=None) -> None:
        super().__init__(parent)
        import h5py

        self._path = str(h5_path)
        self._f = h5py.File(self._path, "r")
        self.post_draw = None  # callable(ax) for owner overlays, or None
        self.slice_name: str | None = None
        self.group_id: str | None = None
        self.plane_index = 0
        self.present: list[str] = []
        self.u = self.v = self.offsets = None
        self.attrs: dict | None = None
        self._sg = None

        self._fig = Figure(figsize=(7, 6), layout="tight")
        self.canvas = FigureCanvasQTAgg(self._fig)
        self.ax = self._fig.add_subplot(111)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)

    # -- file ---------------------------------------------------------------
    @property
    def file(self):
        """The open h5py.File (read-only)."""
        return self._f

    def close_file(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001 - already closed / never opened
            pass

    def reopen(self) -> None:
        """Re-open after an external write; re-binds the current view."""
        import h5py

        self._f = h5py.File(self._path, "r")
        if self.slice_name and self.group_id:
            self._sg = self._f[f"{self.group_id}/{self.slice_name}"]

    # -- navigation -----------------------------------------------------------
    def slice_names(self) -> list[str]:
        """All slice-group names in the file (union across field groups)."""
        import h5py

        names: set[str] = set()
        for vid in _pr.list_volume_ids(self._f):
            g = self._f[vid]
            for k in g:
                if isinstance(g[k], h5py.Group) and "slices" in g[k]:
                    names.add(str(k))
        return sorted(names)

    def open_slice(self, slice_name, *, ref_pref="", init_offset=0.0) -> None:
        present = _pr.volume_ids_with_slice(self._f, slice_name)
        if not present:
            raise KeyError(f"slice {slice_name!r} not present in {self._path}")
        self.slice_name = slice_name
        self.present = present
        self._bind_group(_pr._pick_reference_id(present, ref_pref))
        self.plane_index, _ = _pr.resolve_plane_index(self.offsets, init_offset)
        self.redraw()

    def _bind_group(self, vid) -> None:
        self.group_id = vid
        self._sg = self._f[f"{vid}/{self.slice_name}"]
        self.u, self.v, self.offsets = _pr.read_axes(self._sg)
        self.attrs = _pr.read_volume_attrs(self._f, vid)
        self.plane_index = max(0, min(self.plane_index, len(self.offsets) - 1))

    def set_group(self, vid) -> None:
        self._bind_group(vid)
        self.redraw()

    def set_plane(self, idx) -> None:
        self.plane_index = max(0, min(int(idx), len(self.offsets) - 1))
        self.redraw()

    def step(self, d) -> None:
        self.set_plane(self.plane_index + d)

    def current_offset(self) -> float:
        return float(self.offsets[self.plane_index])

    # -- drawing --------------------------------------------------------------
    def redraw(self) -> None:
        self.ax.clear()
        extent = [float(self.u[0]), float(self.u[-1]), float(self.v[0]), float(self.v[-1])]
        vmin, vmax = self.attrs["vmin"], self.attrs["vmax"]
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax) if vmin is not None else None
        self.ax.imshow(
            self._sg["slices"][self.plane_index].astype(float),
            cmap=_rnd.cmap_nan_transparent(self.attrs["cmap"]),
            norm=norm,
            extent=extent,
            origin="lower",
            aspect="equal",
        )
        self.ax.set_xlabel("u (µm)")
        self.ax.set_ylabel("v (µm)")
        if self.post_draw is not None:
            self.post_draw(self.ax)
        self.canvas.draw_idle()
        self.viewChanged.emit()
```

- [ ] **Step 4: Rewrite `gui/widgets/line_picker.py`**

Read the current file once more, then replace it wholesale with:

```python
"""Interactive line picker for the profiles stage (built lazily on demand).

Scroll the planes of one slice on the shared PlaneBrowser canvas, switch the
background field group, click two endpoints, and read back
``(start_uv, end_uv, offset_um, fields, reference)`` in the slice's (u, v)
frame. Opened only when the user clicks "Pick line…" (or once per mark in the
"Jobs from marks…" flow), so the consolidated file is read and the canvas
built on demand, never at stage-view construction.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from .plane_browser import PlaneBrowser


class LinePickerDialog(QDialog):
    """Modal picker over one slice of an oblique_slices.h5 file.

    On accept, :attr:`result` is ``(start_uv, end_uv, offset_um, fields,
    reference)``; otherwise None. *reference* is the background group the line
    was drawn against — it becomes the job's per-job ``"reference"``.
    """

    def __init__(self, h5_path, slice_name, init_offset=0.0, ref_pref="", parent=None) -> None:
        super().__init__(parent)
        self.result = None
        self._pts: list[tuple[float, float]] = []

        self._browser = PlaneBrowser(h5_path)
        try:
            self._browser.open_slice(slice_name, ref_pref=ref_pref, init_offset=init_offset)
        except Exception:
            self._browser.close_file()
            raise
        self._browser.post_draw = self._post_draw
        self._browser.viewChanged.connect(self._update_info)

        self.setWindowTitle(f"Pick line — {slice_name} ({self._browser.group_id})")
        self._info = QLabel()

        self._bg = QComboBox()
        self._bg.addItems(self._browser.present)
        self._bg.setCurrentText(self._browser.group_id)
        self._bg.currentTextChanged.connect(self._on_bg_changed)
        bg_row = QHBoxLayout()
        bg_row.addWidget(QLabel("Background:"))
        bg_row.addWidget(self._bg)
        bg_row.addStretch(1)

        self._prev = QPushButton("◀ plane")
        self._next = QPushButton("plane ▶")
        self._use = QPushButton("Use line")
        self._cancel = QPushButton("Cancel")
        self._use.setEnabled(False)
        self._prev.clicked.connect(lambda: self._browser.step(-1))
        self._next.clicked.connect(lambda: self._browser.step(+1))
        self._use.clicked.connect(self._accept)
        self._cancel.clicked.connect(self.reject)

        self._field_boxes: dict[str, QCheckBox] = {}
        fields_row = QHBoxLayout()
        fields_row.addWidget(QLabel("Fields:"))
        for vid in self._browser.present:
            box = QCheckBox(vid)
            box.setChecked(True)
            self._field_boxes[vid] = box
            fields_row.addWidget(box)
        fields_row.addStretch(1)

        nav = QHBoxLayout()
        nav.addWidget(self._prev)
        nav.addWidget(self._next)
        nav.addStretch(1)
        nav.addWidget(self._use)
        nav.addWidget(self._cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(self._browser, 1)
        lay.addWidget(self._info)
        lay.addLayout(bg_row)
        lay.addLayout(fields_row)
        lay.addLayout(nav)

        for box in self._field_boxes.values():
            box.toggled.connect(self._refresh_use_button)

        self._browser.canvas.mpl_connect("button_press_event", self._on_click)
        self._update_info()

    # -- plane display ----------------------------------------------------
    def _post_draw(self, ax) -> None:
        if len(self._pts) == 2:
            (u0, v0), (u1, v1) = self._pts
            ax.plot([u0, u1], [v0, v1], "-o", color="cyan", lw=2, ms=6, zorder=6)

    def _on_bg_changed(self, vid: str) -> None:
        self._browser.set_group(vid)
        self.setWindowTitle(f"Pick line — {self._browser.slice_name} ({vid})")

    def _update_info(self) -> None:
        off = self._browser.current_offset()
        n = len(self._browser.offsets)
        pts = " -> ".join(f"({u:.2f}, {v:.2f})" for u, v in self._pts) or "click two points"
        self._info.setText(
            f"plane {self._browser.plane_index + 1}/{n}  offset {off:+.3f} µm   |   line: {pts}"
        )

    # -- interaction ------------------------------------------------------
    def _refresh_use_button(self) -> None:
        """Enable the Use button only when 2 points are picked AND ≥1 field is checked."""
        self._use.setEnabled(len(self._pts) == 2 and bool(self.selected_fields()))

    def _on_click(self, event) -> None:
        if event.inaxes is not self._browser.ax or event.xdata is None or event.ydata is None:
            return
        if len(self._pts) >= 2:
            self._pts = []
        self._pts.append((float(event.xdata), float(event.ydata)))
        self._refresh_use_button()
        self._browser.redraw()

    def _accept(self) -> None:
        if len(self._pts) != 2:
            return
        self.result = (
            self._pts[0],
            self._pts[1],
            self._browser.current_offset(),
            self.field_restriction(),
            self._browser.group_id,
        )
        self.accept()

    def selected_fields(self) -> list[str]:
        """Ticked field ids, in catalog order (all present when none unticked)."""
        return [vid for vid in self._browser.present if self._field_boxes[vid].isChecked()]

    def field_restriction(self) -> list[str] | None:
        """The per-job ``fields`` restriction, or None when the user left ALL
        present fields checked (= no restriction; job auto-adapts to new fields)."""
        sel = self.selected_fields()
        return None if set(sel) == set(self._browser.present) else sel

    # -- cleanup ----------------------------------------------------------
    def done(self, code) -> None:  # noqa: D401 - Qt override
        self._browser.close_file()
        super().done(code)
```

- [ ] **Step 5: Thread `reference` through `gui/viewers.py` and `gui/stage_view.py`**

`inject_line_into_jobs` (Read it first): change the signature to

```python
def inject_line_into_jobs(
    jobs_json: str,
    slice_name: str,
    start_uv,
    end_uv,
    offset_um: float,
    fields=None,
    reference=None,
) -> str:
```

and add, right after the existing `fields` handling:

```python
    if reference:
        target["reference"] = str(reference)
```

(update the docstring: mention that a truthy *reference* sets the job's
``"reference"``; ``None`` leaves any existing value untouched).

`gui/stage_view.py` `_on_pick_line` (Read ≈413–453 first): change the unpack
and call to:

```python
        if dlg.exec() and dlg.result:
            start, end, off, fields, reference = dlg.result
            new_jobs = inject_line_into_jobs(
                vals.get("jobs_json", "") or "[]",
                slice_name,
                start,
                end,
                off,
                fields=fields,
                reference=reference,
            )
```

- [ ] **Step 6: Run tests — new AND untouched old ones**

Run: `python3 -m pytest tests/test_gui_plane_browser.py tests/test_gui_line_picker_fields.py -v`
Expected: all pass; `test_gui_line_picker_fields.py` was NOT edited.

- [ ] **Step 7: Docs (same commit)**

- `docs/Usage.md`: in the profiles Pick-line description (grep "Pick line"),
  add: the Background dropdown switches which field group is displayed while
  you draw; the group you accept with becomes the job's `reference`.
- `docs/Codebase.md`: add `plane_browser.py` to `gui/widgets`; update the
  `line_picker` entry (5-tuple result, composes PlaneBrowser) and
  `viewers.inject_line_into_jobs` (reference kwarg).

- [ ] **Step 8: Full check + commit**

Run: `python3 -m pytest -q && ruff check .`
Expected: green.

```bash
git add gui/widgets/plane_browser.py gui/widgets/line_picker.py gui/viewers.py \
  gui/stage_view.py tests/test_gui_plane_browser.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): shared PlaneBrowser; line picker gains background switch, reference on accept"
```

---

### Task 4: MarkPlanesDialog + "Mark planes…" button

**Files:**
- Create: `gui/widgets/mark_planes.py`
- Modify: `gui/stage_view.py` (button row ≈146–151; `_on_pin_planes` ≈582–602
  — factor the h5-path resolution into a helper both handlers share)
- Test: `tests/test_gui_mark_planes.py`
- Modify: `tests/gui_smoke.py` (new numbered check after the current last one)
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `PlaneBrowser` (Task 3: `slice_names/open_slice/set_group/step/
  current_offset/viewChanged/file/close_file/reopen`),
  `slices.read_marks/write_marks/replot_catalog` (Task 2).
- Produces: `MarkPlanesDialog(h5_path, parent=None)` with attribute
  `saved: bool` (True after ≥1 successful Save); `StageView._slices_output_h5()
  -> str` (used by pin + mark handlers).

- [ ] **Step 1: Write the failing test**

Create `tests/test_gui_mark_planes.py` (reuse the `_mini` builder — copy it
verbatim from `tests/test_gui_plane_browser.py`):

```python
"""Offscreen tests: MarkPlanesDialog toggles and persists marks."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.stages import slices as sl  # noqa: E402


def _mini(path, offsets=(-2.0, 0.0, 2.0)):
    u = np.linspace(-4.0, 4.0, 9)
    v = np.linspace(-3.0, 3.0, 7)
    offs = np.asarray(offsets, np.float64)
    with h5py.File(path, "w") as f:
        for vid in ("raw_sum", "strain"):
            g = f.create_group(vid)
            g.attrs["kind"] = vid
            g.attrs["cmap"] = "gray"
            g.attrs["title"] = vid
            g.attrs["cbar_label"] = "v"
            g.attrs["vmin"] = -1.0
            g.attrs["vmax"] = 1.0
            sg = g.create_group("oblique_full")
            sg.create_dataset(
                "slices", data=np.zeros((offs.size, v.size, u.size), dtype=np.float32)
            )
            sg.create_dataset("u_um", data=u)
            sg.create_dataset("v_um", data=v)
            sg.create_dataset("offsets_um", data=offs)
    return str(path)


def test_mark_toggle_and_save(tmp_path):
    from gui.widgets.mark_planes import MarkPlanesDialog

    _app = QApplication.instance() or QApplication([])
    h5 = _mini(tmp_path / "s.h5")
    dlg = MarkPlanesDialog(h5)
    assert dlg._slice_box.currentText() == "oblique_full"
    dlg._browser.set_plane(2)  # offset +2.0
    dlg._mark_btn.setChecked(True)  # mark it
    dlg._browser.set_plane(0)
    assert not dlg._mark_btn.isChecked()  # button tracks the current plane
    dlg._mark_btn.setChecked(True)  # mark -2.0 too
    assert dlg._dirty()
    dlg._on_save()
    assert dlg.saved
    assert not dlg._dirty()
    assert sl.read_marks(h5) == {"oblique_full": [-2.0, 2.0]}
    # unmark one and save again -> replaced
    dlg._browser.set_plane(0)
    dlg._mark_btn.setChecked(False)
    dlg._on_save()
    dlg.done(0)
    assert sl.read_marks(h5) == {"oblique_full": [2.0]}


def test_dialog_loads_existing_marks(tmp_path):
    from gui.widgets.mark_planes import MarkPlanesDialog

    _app = QApplication.instance() or QApplication([])
    h5 = _mini(tmp_path / "s.h5")
    sl.write_marks(h5, "oblique_full", [0.0])
    dlg = MarkPlanesDialog(h5)
    dlg._browser.set_plane(1)  # offset 0.0
    assert dlg._mark_btn.isChecked()
    assert not dlg._dirty()
    dlg.done(0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/test_gui_mark_planes.py -v`
Expected: FAIL — no module `gui.widgets.mark_planes`.

- [ ] **Step 3: Create `gui/widgets/mark_planes.py`**

```python
"""Mark planes… dialog: browse slice planes visually and star the interesting ones.

Marks are offsets stored under ``/marks/<slice_name>`` inside the
oblique_slices.h5 itself (``dfxm.stages.slices.read_marks/write_marks``), so
every plane list in the app can show them. Save briefly closes the browser's
read handle, writes, and reopens (HDF5 file locking forbids r+ while a read
handle is open in the same process).
"""

from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from dfxm.common.errors import StageUserError
from dfxm.stages import slices as _sl

from .plane_browser import PlaneBrowser


class MarkPlanesDialog(QDialog):
    """Star planes of an oblique_slices.h5; Save persists to ``/marks``."""

    def __init__(self, h5_path, parent=None) -> None:
        super().__init__(parent)
        self._path = str(h5_path)
        self.saved = False
        self._browser = PlaneBrowser(h5_path)
        names = self._browser.slice_names()
        if not names:
            self._browser.close_file()
            raise KeyError(f"no slice groups in {h5_path}")

        # marks state: slice_name -> set of plane indices (offsets resolve on save)
        self._offsets: dict[str, list[float]] = {}
        for e in _sl.replot_catalog(self._path):
            self._offsets.setdefault(e.slice_name, list(e.offsets_um))
        self._marks: dict[str, set[int]] = {}
        for sname, offs in _sl.read_marks(self._browser.file).items():
            stored = self._offsets.get(sname)
            if not stored:
                continue
            self._marks[sname] = {
                min(range(len(stored)), key=lambda i: abs(stored[i] - o)) for o in offs
            }
        self._baseline = {k: set(v) for k, v in self._marks.items()}

        self.setWindowTitle(f"Mark planes — {os.path.basename(self._path)}")
        self._info = QLabel()
        self._slice_box = QComboBox()
        self._slice_box.addItems(names)
        self._group_box = QComboBox()
        self._mark_btn = QPushButton("★ Mark")
        self._mark_btn.setCheckable(True)
        self._prev = QPushButton("◀ plane")
        self._next = QPushButton("plane ▶")
        save = QPushButton("Save")
        save.setProperty("role", "primary")
        close = QPushButton("Close")

        top = QHBoxLayout()
        top.addWidget(QLabel("Slice:"))
        top.addWidget(self._slice_box)
        top.addWidget(QLabel("Background:"))
        top.addWidget(self._group_box)
        top.addStretch(1)

        nav = QHBoxLayout()
        nav.addWidget(self._prev)
        nav.addWidget(self._next)
        nav.addWidget(self._mark_btn)
        nav.addStretch(1)
        nav.addWidget(save)
        nav.addWidget(close)

        lay = QVBoxLayout(self)
        lay.addLayout(top)
        lay.addWidget(self._browser, 1)
        lay.addWidget(self._info)
        lay.addLayout(nav)

        self._prev.clicked.connect(lambda: self._browser.step(-1))
        self._next.clicked.connect(lambda: self._browser.step(+1))
        self._mark_btn.toggled.connect(self._on_mark_toggled)
        self._slice_box.currentTextChanged.connect(self._on_slice_changed)
        self._group_box.currentTextChanged.connect(self._on_group_changed)
        self._browser.viewChanged.connect(self._sync_controls)
        save.clicked.connect(self._on_save)
        close.clicked.connect(self.reject)

        self._on_slice_changed(names[0])

    # -- state --------------------------------------------------------------
    def _dirty(self) -> bool:
        a = {k: v for k, v in self._marks.items() if v}
        b = {k: v for k, v in self._baseline.items() if v}
        return a != b

    def _current_marked(self) -> bool:
        s = self._browser.slice_name
        return s is not None and self._browser.plane_index in self._marks.get(s, ())

    # -- slots ----------------------------------------------------------------
    def _on_slice_changed(self, sname: str) -> None:
        self._browser.open_slice(sname)
        self._group_box.blockSignals(True)
        self._group_box.clear()
        self._group_box.addItems(self._browser.present)
        self._group_box.setCurrentText(self._browser.group_id)
        self._group_box.blockSignals(False)
        self._sync_controls()

    def _on_group_changed(self, vid: str) -> None:
        if vid:
            self._browser.set_group(vid)

    def _on_mark_toggled(self, checked: bool) -> None:
        s = self._browser.slice_name
        if s is None:
            return
        idxs = self._marks.setdefault(s, set())
        if checked:
            idxs.add(self._browser.plane_index)
        else:
            idxs.discard(self._browser.plane_index)
        self._sync_controls()

    def _sync_controls(self) -> None:
        s = self._browser.slice_name
        n_marked = len(self._marks.get(s, ()))
        off = self._browser.current_offset()
        n = len(self._browser.offsets)
        star = "  ★" if self._current_marked() else ""
        self._info.setText(
            f"plane {self._browser.plane_index + 1}/{n}  offset {off:+.3f} µm{star}"
            f"   |   ★ {n_marked} marked in this slice"
            + ("   |   unsaved changes" if self._dirty() else "")
        )
        self._mark_btn.blockSignals(True)
        self._mark_btn.setChecked(self._current_marked())
        self._mark_btn.blockSignals(False)

    def _on_save(self) -> None:
        self._browser.close_file()
        try:
            for sname in sorted(set(self._baseline) | set(self._marks)):
                offs = [self._offsets[sname][i] for i in sorted(self._marks.get(sname, set()))]
                _sl.write_marks(self._path, sname, offs)
        except StageUserError as exc:
            QMessageBox.warning(
                self, "Save marks", f"{exc}\n\n{exc.hint}" if exc.hint else str(exc)
            )
            return
        finally:
            self._browser.reopen()
        self._baseline = {k: set(v) for k, v in self._marks.items()}
        self.saved = True
        self._sync_controls()

    # -- close ------------------------------------------------------------
    def reject(self) -> None:  # noqa: D401 - Qt override
        if self._dirty():
            ans = QMessageBox.question(
                self, "Discard mark changes?", "You have unsaved mark changes. Discard them?"
            )
            if ans != QMessageBox.StandardButton.Yes:
                return
        super().reject()

    def done(self, code) -> None:  # noqa: D401 - Qt override
        self._browser.close_file()
        super().done(code)
```

- [ ] **Step 4: Wire the button in `gui/stage_view.py`**

Read the button row (≈131–163) and `_on_pin_planes` (≈582–602) first.

In the button row, after the Pin planes… block:

```python
        # slices: star interesting planes into /marks (built lazily on click)
        self._mark_btn: QPushButton | None = None
        if stage_name == "slices":
            self._mark_btn = QPushButton("Mark planes…")
            self._mark_btn.clicked.connect(self._on_mark_planes)
            btn_row.addWidget(self._mark_btn)
```

Factor the h5-path resolution out of `_on_pin_planes` into a shared helper and
add the new handler (place both next to `_on_pin_planes`):

```python
    def _slices_output_h5(self) -> str:
        """The slices run's consolidated h5, resolved like the run would."""
        vals = self._form.values()
        out_dir = vals.get("output_dir", "") or os.path.join(
            os.path.dirname(
                vals.get("mosa_volume_file", "") or vals.get("strain_volume_file", "") or "."
            ),
            "oblique_slices",
        )
        return os.path.join(out_dir, vals.get("output_h5_name", "") or "oblique_slices.h5")

    def _on_mark_planes(self) -> None:
        """Open Mark planes… on the slices output file (marks persist in the h5)."""
        h5 = self._slices_output_h5()
        if not os.path.exists(h5):
            self._log.append(f"Mark planes: no slices file at {h5} — run slices first.")
            self._tabs.setCurrentWidget(self._log)
            return
        from .widgets.mark_planes import MarkPlanesDialog  # imported on demand

        try:
            dlg = MarkPlanesDialog(h5, parent=self)
        except Exception as exc:  # noqa: BLE001 - unreadable / empty file
            self._log.append(f"Mark planes failed: {exc}")
            self._tabs.setCurrentWidget(self._log)
            return
        dlg.exec()
        if dlg.saved:
            self._log.append(
                "Marks saved into the slices file — ★ in plane lists; turn them into "
                "profile jobs with 'Jobs from marks…' on the profiles stage."
            )
            self._tabs.setCurrentWidget(self._log)
```

Then edit `_on_pin_planes` to use the helper — its first lines become:

```python
    def _on_pin_planes(self) -> None:
        """Open Pin planes… and write pinned_slices_json + use_pinned into the form."""
        h5 = self._slices_output_h5()
```

(delete the old inline `vals`/`out_dir`/`h5` computation; keep the rest).

- [ ] **Step 5: Run the tests**

Run: `python3 -m pytest tests/test_gui_mark_planes.py -v`
Expected: 2 passed.

- [ ] **Step 6: gui_smoke entry**

Read the end of `tests/gui_smoke.py` to find the last numbered check (`[NN]`,
currently ≈[37]) and the `[32]` Pin planes… block as the pattern. Append a new
check `[NN+1]` that: builds a synthetic slices h5 (reuse the file-building
style of check [26]/[32]), constructs `MarkPlanesDialog`, marks plane 0 via
`dlg._browser.set_plane(0); dlg._mark_btn.setChecked(True); dlg._on_save()`,
asserts `dfxm.stages.slices.read_marks` returns the mark, and asserts the
slices StageView has a `_mark_btn`. Print
`"[NN+1] Mark planes… dialog saves /marks; button wired on slices view"`.

Run: `python3 tests/gui_smoke.py`
Expected: all checks incl. the new one pass.

- [ ] **Step 7: Docs (same commit)**

- `docs/Usage.md`: slices stage section — new subsection "Marking interesting
  planes": browse planes visually (slice + background dropdowns, ◀ ▶), star
  with ★ Mark, Save writes into the slices file; marks show as ★ in Pin
  planes…/Replot… lists and feed "Jobs from marks…" on profiles; re-running
  slices rewrites the file and clears marks. Cross-reference from the
  "Pinning one plane from a sweep" note.
- `docs/Codebase.md`: add `mark_planes.py`; note `StageView._slices_output_h5`
  and the third slices button.

- [ ] **Step 8: Full check + commit**

Run: `python3 -m pytest -q && ruff check .`
Expected: green.

```bash
git add gui/widgets/mark_planes.py gui/stage_view.py tests/test_gui_mark_planes.py \
  tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): Mark planes… visual dialog stars planes into /marks on the slices stage"
```

---

### Task 5: ★ in plane lists (model, panel, pin + slice-replot dialogs, line picker)

**Files:**
- Modify: `gui/widgets/plane_selection_model.py` (`PlaneRow` ≈16–24,
  `build_slice_rows` ≈27–41, `filter_rows` ≈93–114)
- Modify: `gui/widgets/plane_selection.py` (filter row ≈38–66,
  `set_rows` ≈84–115, `_apply_filter` ≈130–138)
- Modify: `gui/widgets/pin_planes.py` (`_reload` ≈70–84)
- Modify: `gui/widgets/slice_replot.py` (its `_reload`; grep
  `build_slice_rows` for the call site)
- Modify: `gui/widgets/line_picker.py` (★ in the info line)
- Test: `tests/test_plane_selection_marks.py`; extend
  `tests/test_gui_plane_browser.py`
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `slices.read_marks` (Task 2), `PlaneBrowser.file/offsets` (Task 3),
  `profiles.resolve_plane_index`.
- Produces: `PlaneRow.marked: bool = False`;
  `build_slice_rows(entries, marks=None)`;
  `filter_rows(rows, text, *, marked_only=False)`. All existing callers keep
  working without changes (new args optional).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_plane_selection_marks.py`:

```python
"""Qt-free tests: marked rows in the plane-selection model."""

from __future__ import annotations

from dfxm.stages.slices import ReplotEntry
from gui.widgets.plane_selection_model import PlaneRow, build_slice_rows, filter_rows


def _entries():
    return [
        ReplotEntry("raw_sum", "oblique_full", 3, [-2.0, 0.0, 2.0], shape=(7, 9), group="raw"),
        ReplotEntry("strain", "oblique_full", 3, [-2.0, 0.0, 2.0], shape=(7, 9), group="strain"),
    ]


def test_build_slice_rows_marks():
    rows = build_slice_rows(_entries(), marks={"oblique_full": [0.1]})  # snaps to 0.0
    by_key = {r.key: r for r in rows}
    assert by_key[("oblique_full", 1)].marked
    assert by_key[("oblique_full", 1)].label.startswith("★ ")
    assert not by_key[("oblique_full", 0)].marked
    assert not by_key[("oblique_full", 0)].label.startswith("★")


def test_build_slice_rows_no_marks_unchanged():
    rows = build_slice_rows(_entries())
    assert all(not r.marked for r in rows)
    assert rows[0].label == "p000  -2.00 µm"


def test_filter_rows_marked_only():
    rows = [
        PlaneRow(key=1, section="", number=1, offset=0.0, label="a", marked=True),
        PlaneRow(key=2, section="", number=2, offset=1.0, label="b"),
    ]
    assert [r.key for r in filter_rows(rows, "", marked_only=True)] == [1]
    assert [r.key for r in filter_rows(rows, "2", marked_only=True)] == []
    assert [r.key for r in filter_rows(rows, "")] == [1, 2]
```

Check `ReplotEntry`'s exact constructor first (grep
`class ReplotEntry` in `dfxm/stages/slices.py`) and adapt the two `_entries()`
lines to its real field order/names if they differ.

Also append to `tests/test_gui_plane_browser.py`:

```python
def test_picker_info_shows_star_for_marked_plane(tmp_path):
    from dfxm.stages import slices as sl
    from gui.widgets.line_picker import LinePickerDialog

    _app = QApplication.instance() or QApplication([])
    h5 = _mini(tmp_path / "m.h5")
    sl.write_marks(h5, "oblique_full", [0.0])
    dlg = LinePickerDialog(h5, "oblique_full", init_offset=0.0)
    assert "★" in dlg._info.text()
    dlg._browser.step(+1)
    assert "★" not in dlg._info.text()
    dlg.done(0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_plane_selection_marks.py tests/test_gui_plane_browser.py -v`
Expected: FAIL — `PlaneRow` has no field `marked` /
`build_slice_rows() got an unexpected keyword argument 'marks'` / no ★.

- [ ] **Step 3: Model changes (`plane_selection_model.py`)**

Add to `PlaneRow` (after `label`):

```python
    marked: bool = False  # starred in /marks — cosmetic; filtering-only
```

Replace `build_slice_rows` with:

```python
def build_slice_rows(entries, marks=None) -> list[PlaneRow]:
    """Rows from slices ReplotEntry list — one per (slice_name, plane_idx).

    *marks* is the ``read_marks`` mapping (slice_name -> [offset_um, ...]);
    each marked offset stars its nearest stored plane's row.
    """
    marked_idx: dict[str, set[int]] = {}
    if marks:
        stored_by_slice: dict[str, list[float]] = {}
        for e in entries:
            stored_by_slice.setdefault(e.slice_name, list(e.offsets_um))
        for sname, offs in marks.items():
            stored = stored_by_slice.get(sname)
            if not stored:
                continue
            marked_idx[sname] = {
                min(range(len(stored)), key=lambda i: abs(stored[i] - o)) for o in offs
            }
    seen: dict[tuple[str, int], PlaneRow] = {}
    for e in entries:
        for k, off in enumerate(e.offsets_um):
            key = (e.slice_name, k)
            if key not in seen:
                is_marked = k in marked_idx.get(e.slice_name, ())
                seen[key] = PlaneRow(
                    key=key,
                    section=e.slice_name,
                    number=k,
                    offset=float(off),
                    label=("★ " if is_marked else "") + f"p{k:03d}  {off:+.2f} µm",
                    marked=is_marked,
                )
    return list(seen.values())
```

In `filter_rows`, change the signature to
`def filter_rows(rows: list[PlaneRow], text: str, *, marked_only: bool = False) -> list[PlaneRow]:`
and insert as the first lines of the body:

```python
    if marked_only:
        rows = [r for r in rows if r.marked]
```

- [ ] **Step 4: Panel checkbox (`plane_selection.py`)**

Add `QCheckBox` to the PySide6 import list. In `__init__`, right after the
`self._no_match` lines:

```python
        self._marked_only = QCheckBox("★ only")
        self._marked_only.setVisible(False)
        self._marked_only.toggled.connect(lambda *_: self._apply_filter(self._filter.text()))
```

and add it to the filter row (after `frow.addWidget(self._no_match)`):

```python
        frow.addWidget(self._marked_only)
```

In `set_rows`, after `self._rows = list(rows)`:

```python
        has_marks = any(r.marked for r in self._rows)
        if not has_marks:
            self._marked_only.setChecked(False)
        self._marked_only.setVisible(has_marks)
```

In `_apply_filter`, change the first line to:

```python
        visible = {
            r.key
            for r in filter_rows(self._rows, text, marked_only=self._marked_only.isChecked())
        }
```

and change the no-match condition's last line to also count the checkbox as
an active narrowing (so "no match" shows when ★-only hides everything):

```python
        self._no_match.setVisible(
            (bool(text.strip()) or self._marked_only.isChecked()) and not visible
        )
```

- [ ] **Step 5: Thread marks into the two dialogs**

`gui/widgets/pin_planes.py` `_reload` — replace the
`self._panel.set_rows(build_slice_rows(catalog))` line with:

```python
        try:
            marks = _sl.read_marks(self._h5_path)
        except Exception:  # noqa: BLE001 — marks are cosmetic here
            marks = {}
        self._panel.set_rows(build_slice_rows(catalog, marks=marks))
```

`gui/widgets/slice_replot.py` — grep for its `build_slice_rows(` call inside
`_reload` and make the identical change (it already imports
`dfxm.stages.slices as _sl`).

- [ ] **Step 6: ★ in the line-picker info line**

In `gui/widgets/line_picker.py`: add imports

```python
from dfxm.stages import profiles as _pr
from dfxm.stages import slices as _sl
```

In `__init__`, right after the `open_slice` try/except block:

```python
        moffs = _sl.read_marks(self._browser.file).get(slice_name, [])
        self._marked_idx = {
            _pr.resolve_plane_index(self._browser.offsets, o)[0] for o in moffs
        }
```

In `_update_info`, mark the offset segment:

```python
        star = "  ★" if self._browser.plane_index in self._marked_idx else ""
```

and include it in the f-string right after the offset:
`f"plane {self._browser.plane_index + 1}/{n}  offset {off:+.3f} µm{star}   |   line: {pts}"`.

- [ ] **Step 7: Run tests**

Run: `python3 -m pytest tests/test_plane_selection_marks.py tests/test_gui_plane_browser.py tests/test_gui_pin_planes.py tests/test_gui_slice_replot.py -v`
Expected: all pass (existing pin/slice-replot tests unaffected — new args are
optional).

- [ ] **Step 8: Docs (same commit)**

- `docs/Usage.md`: in the Pin planes… / Replot slices descriptions, one
  sentence: marked planes show a ★ and a "★ only" toggle appears when the
  file has marks; the Pick-line info line also flags marked planes.
- `docs/Codebase.md`: `PlaneRow.marked`, `build_slice_rows(marks=)`,
  `filter_rows(marked_only=)`, panel "★ only" checkbox.

- [ ] **Step 9: Full check + commit**

Run: `python3 -m pytest -q && ruff check .`
Expected: green.

```bash
git add gui/widgets/plane_selection_model.py gui/widgets/plane_selection.py \
  gui/widgets/pin_planes.py gui/widgets/slice_replot.py gui/widgets/line_picker.py \
  tests/test_plane_selection_marks.py tests/test_gui_plane_browser.py \
  docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): marked planes show ★ + '★ only' filter in plane lists and the line picker"
```

---

### Task 6: Jobs from marks (profiles stage)

**Files:**
- Create: `gui/widgets/jobs_from_marks.py`
- Modify: `gui/viewers.py` (add `append_line_job` after
  `inject_line_into_jobs`), `gui/stage_view.py` (button row ≈134–139; new
  handler next to `_on_pick_line`; extend the `from .viewers import` line ≈39)
- Test: `tests/test_gui_jobs_from_marks.py`
- Modify: `tests/gui_smoke.py` (one more numbered check)
- Docs: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `slices.read_marks` (Task 2), `LinePickerDialog` 5-tuple result
  (Task 3).
- Produces: `append_line_job(jobs_json, slice_name, start_uv, end_uv,
  offset_um, fields=None, reference=None) -> str` in `gui.viewers`;
  `JobsFromMarksDialog(marks: dict[str, list[float]], parent=None)` with
  attribute `selected: list[tuple[str, float]]` after accept.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_jobs_from_marks.py`:

```python
"""append_line_job (pure) + JobsFromMarksDialog checklist (offscreen)."""

from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.viewers import append_line_job  # noqa: E402


def test_append_line_job_appends_never_updates():
    base = json.dumps([{"name": "oblique_full", "offset_um": 0.0, "start_uv": [9, 9]}])
    out = append_line_job(
        base, "oblique_full", (0.0, 0.0), (1.0, 0.5), 2.0, fields=["strain"], reference="strain"
    )
    jobs = json.loads(out)
    assert len(jobs) == 2  # appended, existing job untouched
    assert jobs[0]["start_uv"] == [9, 9]
    assert jobs[1] == {
        "name": "oblique_full",
        "offset_um": 2.0,
        "start_uv": [0.0, 0.0],
        "end_uv": [1.0, 0.5],
        "fields": ["strain"],
        "reference": "strain",
    }


def test_append_line_job_minimal_and_bad_json():
    out = append_line_job("not json", "s", (0.0, 0.0), (1.0, 0.0), 0.0)
    jobs = json.loads(out)
    assert len(jobs) == 1
    assert "fields" not in jobs[0] and "reference" not in jobs[0]


def test_jobs_from_marks_dialog_selection():
    from gui.widgets.jobs_from_marks import JobsFromMarksDialog

    _app = QApplication.instance() or QApplication([])
    dlg = JobsFromMarksDialog({"b_slice": [1.0], "a_slice": [-2.0, 0.0]})
    assert dlg._list.count() == 3  # sorted by slice then offset, all checked
    dlg._list.item(1).setCheckState(Qt.CheckState.Unchecked)  # drop a_slice @ 0.0
    dlg._on_ok()
    assert dlg.selected == [("a_slice", -2.0), ("b_slice", 1.0)]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_jobs_from_marks.py -v`
Expected: FAIL — `append_line_job` not found.

- [ ] **Step 3: Add `append_line_job` to `gui/viewers.py`**

After `inject_line_into_jobs`:

```python
def append_line_job(
    jobs_json: str,
    slice_name: str,
    start_uv,
    end_uv,
    offset_um: float,
    fields=None,
    reference=None,
) -> str:
    """Append ONE complete job to *jobs_json* — never edits existing jobs.

    Unlike :func:`inject_line_into_jobs` (which updates the first job matching
    the slice name), this always appends, so several marks on one slice each
    become their own job (the profiles stage de-duplicates output stems for
    same-named jobs).
    """
    try:
        jobs = json.loads(jobs_json) if jobs_json.strip() else []
    except json.JSONDecodeError:
        jobs = []
    if not isinstance(jobs, list):
        jobs = []
    job = {
        "name": slice_name,
        "offset_um": round(float(offset_um), 4),
        "start_uv": [round(float(start_uv[0]), 4), round(float(start_uv[1]), 4)],
        "end_uv": [round(float(end_uv[0]), 4), round(float(end_uv[1]), 4)],
    }
    if fields is not None:
        job["fields"] = list(fields)
    if reference:
        job["reference"] = str(reference)
    jobs.append(job)
    return json.dumps(jobs, indent=2)
```

- [ ] **Step 4: Create `gui/widgets/jobs_from_marks.py`**

```python
"""Jobs from marks… checklist: pick which marked planes become profile jobs.

A thin selection dialog over ``dfxm.stages.slices.read_marks`` output; the
caller (profiles StageView) opens one LinePickerDialog per checked row and
appends complete jobs via ``gui.viewers.append_line_job``. Holds no h5 handle.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)


class JobsFromMarksDialog(QDialog):
    """Checklist of marked planes; OK -> ``selected = [(slice_name, offset_um), ...]``."""

    def __init__(self, marks: dict[str, list[float]], parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Jobs from marks")
        self.selected: list[tuple[str, float]] = []

        self._list = QListWidget()
        for sname in sorted(marks):
            for off in marks[sname]:
                it = QListWidgetItem(f"{sname}  @ {off:+.2f} µm")
                it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                it.setCheckState(Qt.CheckState.Checked)
                it.setData(Qt.ItemDataRole.UserRole, (sname, float(off)))
                self._list.addItem(it)

        ok = QPushButton("OK")
        ok.setProperty("role", "primary")
        ok.clicked.connect(self._on_ok)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        brow = QHBoxLayout()
        brow.addStretch(1)
        brow.addWidget(ok)
        brow.addWidget(cancel)

        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("A line picker opens for each checked plane:"))
        lay.addWidget(self._list, 1)
        lay.addLayout(brow)

    def _on_ok(self) -> None:
        self.selected = [
            self._list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self._list.count())
            if self._list.item(i).checkState() == Qt.CheckState.Checked
        ]
        self.accept()
```

- [ ] **Step 5: Wire the button + guided loop in `gui/stage_view.py`**

Extend the viewers import (≈line 39) to:

```python
from .viewers import append_line_job, inject_line_into_jobs, volume_sources
```

In the button row, inside the existing `if stage_name == "profiles":` block,
after the Pick line… lines:

```python
            self._jobs_marks_btn = QPushButton("Jobs from marks…")
            self._jobs_marks_btn.clicked.connect(self._on_jobs_from_marks)
            btn_row.addWidget(self._jobs_marks_btn)
```

Add the handler next to `_on_pick_line`:

```python
    def _on_jobs_from_marks(self) -> None:
        """One line picker per checked mark; each accepted line appends a job."""
        vals = self._form.values()
        h5 = vals.get("consolidated_h5", "")
        if not h5 or not os.path.exists(h5):
            self._log.append("Jobs from marks: set a valid 'consolidated_h5' (run slices first).")
            self._tabs.setCurrentWidget(self._log)
            return
        from dfxm.stages import slices as _sl  # local import: lazy, Qt-free

        try:
            marks = _sl.read_marks(h5)
        except Exception as exc:  # noqa: BLE001 - unreadable file
            self._log.append(f"Jobs from marks: cannot read marks: {exc}")
            self._tabs.setCurrentWidget(self._log)
            return
        if not any(marks.values()):
            self._log.append(
                "Jobs from marks: no marked planes in this file — star planes with "
                "'Mark planes…' on the slices stage first."
            )
            self._tabs.setCurrentWidget(self._log)
            return
        from .widgets.jobs_from_marks import JobsFromMarksDialog  # imported on demand
        from .widgets.line_picker import LinePickerDialog

        sel_dlg = JobsFromMarksDialog(marks, parent=self)
        if not sel_dlg.exec() or not sel_dlg.selected:
            return
        jobs_json = vals.get("jobs_json", "") or "[]"
        added = skipped = 0
        n = len(sel_dlg.selected)
        for k, (sname, off) in enumerate(sel_dlg.selected, start=1):
            try:
                dlg = LinePickerDialog(
                    h5,
                    sname,
                    init_offset=off,
                    ref_pref=vals.get("reference_volume_id", ""),
                    parent=self,
                )
            except Exception as exc:  # noqa: BLE001 - missing slice / unreadable file
                self._log.append(f"Jobs from marks: {sname} @ {off:+.2f} µm failed: {exc}")
                skipped += 1
                continue
            dlg.setWindowTitle(f"Pick line ({k}/{n}) — {sname} @ {off:+.2f} µm")
            if dlg.exec() and dlg.result:
                start, end, o, fields, reference = dlg.result
                jobs_json = append_line_job(
                    jobs_json, sname, start, end, o, fields=fields, reference=reference
                )
                added += 1
            else:
                skipped += 1
        if added:
            self._form.set_values({"jobs_json": jobs_json})
        self._log.append(
            f"Jobs from marks: added {added} job(s), skipped {skipped} — Run to profile."
        )
        self._tabs.setCurrentWidget(self._log)
```

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/test_gui_jobs_from_marks.py -v`
Expected: 3 passed.

- [ ] **Step 7: gui_smoke entry**

Append the next numbered check after Task 4's: assert the profiles StageView
has `_jobs_marks_btn`; construct `JobsFromMarksDialog({"s": [0.0, 2.0]})`,
uncheck one row, call `_on_ok()`, assert `selected == [("s", 0.0)]` (or the
kept row). Print
`"[NN] Jobs from marks: button wired on profiles view; checklist selection"`.

Run: `python3 tests/gui_smoke.py`
Expected: all checks pass.

- [ ] **Step 8: Docs (same commit)**

- `docs/Usage.md`: profiles stage — new "Jobs from marks…" paragraph
  (checklist of marks → one Pick-line dialog per checked plane, pre-navigated;
  each accepted line appends a complete job; cancelling skips that mark).
- `docs/Codebase.md`: `jobs_from_marks.py`, `viewers.append_line_job`, the new
  profiles button/handler.

- [ ] **Step 9: Full check + commit**

Run: `python3 -m pytest -q && ruff check .`
Expected: green.

```bash
git add gui/widgets/jobs_from_marks.py gui/viewers.py gui/stage_view.py \
  tests/test_gui_jobs_from_marks.py tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): Jobs from marks… — guided line picking turns marked planes into profile jobs"
```

---

### Task 7: Tooltip precision sweep

**Files:**
- Modify: `dfxm/stages/{concat,strain,mosaicity,rocking,visualize,paraview,slices,profiles,matched}.py`
  (help strings only), `dfxm/config/models.py` + `dfxm/config/detect.py`
  (experiment-schema help strings)
- Modify: `docs/Usage.md` where changed wording describes behaviour
- Test: existing `tests/test_param_metadata.py` must stay green

**Interfaces:** none — strings only, zero behaviour change. Verify with
`git diff --stat` that only `.py` help strings and `docs/` change.

- [ ] **Step 1: Inventory**

Run: `grep -rn "help=" dfxm/stages/*.py dfxm/config/*.py | wc -l`
(~209 at plan time). Read each stage's `STAGE: StageSpec` block in full before
editing it (never reconstruct `old_string` — em-dashes at 12 or 16 spaces).

- [ ] **Step 2: Apply the precision checklist to every help string**

For each param, tighten the help so it states — in this priority order:

1. **Coordinate frame first** for anything positional: ROI bounds, shifts,
   offsets, origins. Every ROI-ish param's FIRST sentence names its frame:
   "absolute detector pixels" (rocking `roi_x`/`roi_y` — already precise from
   the ROI-unification work; verify, don't churn) vs "darfix-map pixels
   (the maps.h5 frame after darfix's crop)" (slices `align_roi_x/y`, strain
   `roi`, the experiment `analysis_roi_x/y`) vs "origin+size as darfix shows
   it" (`darfix_roi`). Cross-check frames against `dfxm/common/roi.py` and the
   Usage.md ROI section — do not restate from memory. Where a ROI param
   already carries `roi_frame="detector"|"map"` metadata
   (`dfxm/config/models.py` `Param.roi_frame`), the help's frame wording MUST
   agree with it.
2. **Units** on every numeric param (µm, px, degrees, keV) — in the label
   `unit=` if missing, and consistently in the help.
3. **Blank/default semantics**: what happens when the field is left blank or
   at its default ("blank = full frame", "blank = follow the map scale", …).
4. **Consequences over mechanisms** for calibration params: what goes wrong
   when it's set incorrectly.
5. Written for a first-time beamline user; no bare jargon without a gloss.

Worked example — rocking `subtract_background` (rocking.py ≈lines 269–281;
verify the applies-to-both-sources claim against `_sum_scan` usage at ≈446 and
≈763 before writing it):

Before:
```
"Subtract a per-pixel median background (across the scan's frames) before "
"summing. On for the standard rocking sum; turn off for a plain intensity sum "
"(e.g. a mosa-scan topograph that keeps the background)."
```

After:
```
"Before summing, compute each pixel's median across the scan's frames and "
"subtract it, so only above-background diffraction signal accumulates. "
"Applies to whichever scan type the run reads (rocking or mosaicity source). "
"Keep on for the standard rocking sum; turn off for a plain intensity sum, "
"e.g. a mosa-scan topograph where the background level itself is meaningful."
```

- [ ] **Step 3: Keep Usage.md in sync**

Wherever a help rewrite changes documented wording (grep Usage.md for the
param's label), update the matching sentence in `docs/Usage.md`.

- [ ] **Step 4: Verify**

Run: `python3 -m pytest tests/test_param_metadata.py -q && python3 -m pytest -q && ruff check .`
Expected: green. Then `git diff` — review that the diff is strings-only
(no logic lines touched).

- [ ] **Step 5: Commit**

```bash
git add dfxm/stages/ dfxm/config/ docs/Usage.md
git commit -m "docs(stages): tooltip precision sweep — coordinate frames first, units, blank semantics"
```

---

### Task 8: Final verification

- [ ] **Step 1: Whole-suite verify** — use the repo's `verify-suite` skill
  (full pytest + ruff check + ruff format check + `python3 tests/gui_smoke.py`).
  Expected: suite ≥ baseline 785 passed / 13 skipped / 0 warnings, smoke all
  checks (now +2), ruff clean.
- [ ] **Step 2: Spec cross-check** — re-read
  `docs/superpowers/specs/2026-07-26-slice-marks-and-ux-features-design.md`;
  confirm every section (incl. error handling + out-of-scope) is implemented
  or explicitly deferred; fix gaps before review.
- [ ] **Step 3: Whole-branch review** — per repo convention (final
  whole-branch review on fable; match effort to change size).
- [ ] **Step 4: Merge** — via the `finish-and-record` skill (no remote:
  merge `slice-marks-ux` into master locally, `--no-ff`, delete branch,
  update memory notes: retire `scroll-wheel-spinbox-wish`,
  `line-picker-reference-switch-wish`, `oblique-slice-selection-wish`,
  `tooltip-precision-pass`; new project note with rollback point).
