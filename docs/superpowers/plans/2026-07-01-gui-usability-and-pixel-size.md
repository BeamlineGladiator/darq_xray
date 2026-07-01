# GUI usability polish + effective-pixel-size calculator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compute-effective-pixel-size button to the Experiment editor, make the help panel revert to the stage description (with richer hover tooltips), and make the middle column wider by default while remembering window geometry and a shared middle/right splitter width across stages.

**Architecture:** Three independent features. Feature 2 is a Qt-free core function (`dfxm/common/pixel_size.py`) plus a button in `gui/experiment_panel.py`. Feature 3 edits `gui/widgets/param_form.py`, `gui/widgets/help_panel.py`, and `gui/stage_view.py`. Feature 1 adds `gui/window_state.py` and wires it into `gui/main_window.py` and `gui/stage_view.py`. Core stays Qt-free; all GUI persistence uses the existing `QSettings` (org `dfxm`, app `pipeline`).

**Tech Stack:** Python 3.10, PySide6 (Qt), h5py, numpy. Tests: `pytest` for Qt-free core; the manually-run `tests/gui_smoke.py` script (offscreen Qt, numbered checkpoints) for GUI behaviour.

## Global Constraints

- **Keep `dfxm/` Qt-free.** Never import PySide6/pyvista/vtk under `dfxm/`.
- **Input problems raise `StageUserError(message, hint=...)`** from `dfxm.common.errors` — never a bare exception for a user-fixable input.
- **Lint/format:** `ruff check . && ruff format .` must pass. Ruff config: line length 100, double quotes, target py310, rules E/F/I. (`ruff format` also runs automatically on Write/Edit via the repo hook.)
- **Docs are part of the change:** update `docs/Usage.md` (user-visible behaviour) and `docs/Codebase.md` (code structure) in the SAME task that changes behaviour — not as a follow-up.
- **Test commands:**
  - Core: `python3 -m pytest -q`
  - GUI smoke (manual, offscreen): `python3 tests/gui_smoke.py` — must print `GUI SMOKE PASSED`.
- **Commit frequently** — one commit per task (after its tests pass).

---

## File map

- **Create** `dfxm/common/pixel_size.py` — Qt-free pixel-size computation (Task 1).
- **Create** `tests/test_common_pixel_size.py` — core tests (Task 1).
- **Create** `gui/window_state.py` — geometry + shared-splitter persistence (Task 4).
- **Modify** `gui/experiment_panel.py` — "Compute pixel size from scan…" button (Task 2).
- **Modify** `gui/widgets/help_panel.py` — factor a shared `param_help_html` (Task 3).
- **Modify** `gui/widgets/param_form.py` — `focusCleared` signal + enriched tooltips (Task 3).
- **Modify** `gui/stage_view.py` — `showEvent` help reset, `inner_splitter`, shared default width, `focusCleared` wiring (Tasks 3 & 4).
- **Modify** `gui/main_window.py` — register stage splitters, restore/save window state (Tasks 4 & 5).
- **Modify** `tests/gui_smoke.py` — new checkpoints + QSettings isolation (Tasks 2, 3, 4, 5).
- **Modify** `docs/Usage.md`, `docs/Codebase.md` — in each behaviour-changing task.

---

## Task 1: Core `compute_pixel_size` (Qt-free)

**Files:**
- Create: `dfxm/common/pixel_size.py`
- Test: `tests/test_common_pixel_size.py`

**Interfaces:**
- Consumes: `dfxm.common.h5io.get_filtered_entries`, `dfxm.common.h5io.read_positioners`, `dfxm.common.errors.StageUserError`.
- Produces:
  - `PixelSizeResult` (frozen dataclass) with fields: `pixel_size_x_um: float`, `pixel_size_y_um: float`, `magnification: float`, `two_theta_deg: float`, `objective: str` (`"2x"`|`"10x"`), `condenser_in: bool`, `mainx: float`, `obx: float`, `ffsel: float`, `ffz: float`, `lenssel: float`.
  - `compute_pixel_size(scan_h5: str, positioners_path: str = "instrument/positioners", entry_suffix: str = ".1") -> PixelSizeResult`.
  - Module constants: `DEFAULT_POSITIONERS_PATH`, motor names `MAINX/OBX/FFSEL/FFZ/LENSSEL`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_common_pixel_size.py`:

```python
"""Tests for dfxm.common.pixel_size."""

from __future__ import annotations

import math

import h5py
import numpy as np
import pytest

from dfxm.common.errors import StageUserError
from dfxm.common.pixel_size import compute_pixel_size


def _write_scan(path, *, mainx, obx, ffsel, ffz, lenssel, entry="1.1", motors=None):
    """Write a minimal BLISS-style scan with only a positioners group."""
    with h5py.File(path, "w") as f:
        pos = f.create_group(f"{entry}/instrument/positioners")
        values = {
            "mainx": mainx,
            "obx": obx,
            "ffsel": ffsel,
            "ffz": ffz,
            "lenssel": lenssel,
        }
        if motors is not None:
            values = {k: v for k, v in values.items() if k in motors}
        for name, val in values.items():
            pos.create_dataset(name, data=val)
    return str(path)


def test_2x_condenser_in(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5", mainx=5000.0, obx=273.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0
    )
    res = compute_pixel_size(p)
    m = 5000.0 / 273.0 - 1.0
    two_theta = math.atan2(2100.0, 5000.0)
    assert res.objective == "2x"
    assert res.condenser_in is True
    assert res.magnification == pytest.approx(m)
    assert res.pixel_size_x_um == pytest.approx(3.25 / m)
    assert res.pixel_size_y_um == pytest.approx((3.25 / m) / math.sin(two_theta))
    assert res.two_theta_deg == pytest.approx(math.degrees(two_theta))


def test_10x_condenser_out(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5", mainx=5000.0, obx=273.0, ffsel=0.0, ffz=2100.0, lenssel=3.0
    )
    res = compute_pixel_size(p)
    m = 5000.0 / 273.0 - 1.0
    assert res.objective == "10x"
    assert res.condenser_in is False
    assert res.pixel_size_x_um == pytest.approx(0.65 / m)
    # condenser out -> Y equals X (no sin(2theta) division)
    assert res.pixel_size_y_um == pytest.approx(0.65 / m)


def test_unrecognized_ffsel_raises(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5", mainx=5000.0, obx=273.0, ffsel=-30.0, ffz=2100.0, lenssel=0.0
    )
    with pytest.raises(StageUserError):
        compute_pixel_size(p)


def test_missing_motor_raises(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5",
        mainx=5000.0,
        obx=273.0,
        ffsel=-60.0,
        ffz=2100.0,
        lenssel=0.0,
        motors={"mainx", "obx", "ffsel", "ffz"},  # no lenssel
    )
    with pytest.raises(StageUserError):
        compute_pixel_size(p)


def test_no_matching_entry_raises(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5",
        mainx=5000.0,
        obx=273.0,
        ffsel=-60.0,
        ffz=2100.0,
        lenssel=0.0,
        entry="1.2",  # does not end in the default ".1"
    )
    with pytest.raises(StageUserError):
        compute_pixel_size(p)


def test_zero_obx_raises(tmp_path):
    p = _write_scan(
        tmp_path / "s.h5", mainx=5000.0, obx=0.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0
    )
    with pytest.raises(StageUserError):
        compute_pixel_size(p)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_common_pixel_size.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfxm.common.pixel_size'`.

- [ ] **Step 3: Write the implementation**

Create `dfxm/common/pixel_size.py`:

```python
"""Effective detector pixel size from a raw (pre-darfix) scan's motors.

Qt-free. Reads the far-field geometry motors from the first BLISS entry of a
raw scan and turns them into the physical detector pixel size (micrometres per
pixel) that the strain/mosaicity maps are scaled by.

Geometry (lens-maker):
    M   = mainx / obx - 1                 (CRL magnification)
    E_x = base / M                        (horizontal pixel size, um)
    2th = atan2(ffz, mainx)               (detector angle)
    E_y = E_x / sin(2th)  if condenser in, else E_x

``base`` is the far-field camera pixel (6.5 um) divided by the objective
magnification, selected from the ``ffsel`` motor:
    ffsel = -60 -> 2x  (base 3.25)
    ffsel =   0 -> 10x (base 0.65)
The condenser is detected from ``lenssel`` (0 -> in).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import h5py
import numpy as np

from .errors import StageUserError
from .h5io import get_filtered_entries, read_positioners

DEFAULT_POSITIONERS_PATH = "instrument/positioners"

# Positioner motor names (ID03 far-field geometry).
MAINX = "mainx"
OBX = "obx"
FFSEL = "ffsel"
FFZ = "ffz"
LENSSEL = "lenssel"

# ffsel value -> (objective label, base detector pixel in um).
_OBJECTIVES: dict[float, tuple[str, float]] = {
    -60.0: ("2x", 3.25),
    0.0: ("10x", 0.65),
}
_FFSEL_TOL = 1.0  # ffsel motor tolerance when matching an objective
_LENSSEL_TOL = 0.5  # |lenssel| below this = condenser in


@dataclass(frozen=True)
class PixelSizeResult:
    """Effective pixel sizes plus the derived geometry that produced them."""

    pixel_size_x_um: float
    pixel_size_y_um: float
    magnification: float
    two_theta_deg: float
    objective: str
    condenser_in: bool
    mainx: float
    obx: float
    ffsel: float
    ffz: float
    lenssel: float


def _scalar(pos: dict, key: str, where: str) -> float:
    if key not in pos:
        raise StageUserError(
            f"motor {key!r} not found in {where}",
            hint=f"This scan's positioners have no {key!r}; set the pixel size manually.",
        )
    return float(np.asarray(pos[key]).reshape(-1)[0])


def _match_objective(ffsel: float) -> tuple[str, float]:
    for value, (label, base) in _OBJECTIVES.items():
        if abs(ffsel - value) <= _FFSEL_TOL:
            return label, base
    raise StageUserError(
        f"unrecognized ffsel={ffsel:g}; cannot pick the far-field objective",
        hint="Expected ffsel -60 (2x) or 0 (10x). Set the pixel size manually.",
    )


def compute_pixel_size(
    scan_h5: str,
    positioners_path: str = DEFAULT_POSITIONERS_PATH,
    entry_suffix: str = ".1",
) -> PixelSizeResult:
    """Effective pixel size for the first ``entry_suffix`` entry of *scan_h5*."""
    with h5py.File(scan_h5, "r") as f:
        entries = get_filtered_entries(f, entry_suffix)
        if not entries:
            raise StageUserError(
                f"no scan entry ending in {entry_suffix!r} in {scan_h5}",
                hint="Point at a raw BLISS scan .h5 (entries like 1.1, 2.1, ...).",
            )
        group = f"{entries[0]}/{positioners_path}"
        try:
            pos = read_positioners(f, group)
        except KeyError as exc:
            raise StageUserError(
                str(exc),
                hint="Check the experiment's 'Positioners path' matches this file.",
            ) from exc

    mainx = _scalar(pos, MAINX, group)
    obx = _scalar(pos, OBX, group)
    ffsel = _scalar(pos, FFSEL, group)
    ffz = _scalar(pos, FFZ, group)
    lenssel = _scalar(pos, LENSSEL, group)

    if obx == 0.0:
        raise StageUserError(
            "obx = 0, cannot compute the magnification (mainx/obx - 1)",
            hint="Check obx; set the pixel size manually.",
        )
    magnification = mainx / obx - 1.0
    if magnification <= 0.0:
        raise StageUserError(
            f"non-physical magnification M={magnification:g} (mainx/obx - 1)",
            hint="Check mainx and obx (expected mainx > obx). Set the pixel size manually.",
        )

    objective, base = _match_objective(ffsel)
    px_x = base / magnification

    two_theta = math.atan2(ffz, mainx)
    condenser_in = abs(lenssel) <= _LENSSEL_TOL
    if condenser_in:
        sin2t = math.sin(two_theta)
        if abs(sin2t) < 1e-9:
            raise StageUserError(
                "2theta ~ 0, cannot divide the Y pixel size by sin(2theta)",
                hint="Check ffz/mainx; set the pixel size manually.",
            )
        px_y = px_x / sin2t
    else:
        px_y = px_x

    return PixelSizeResult(
        pixel_size_x_um=px_x,
        pixel_size_y_um=px_y,
        magnification=magnification,
        two_theta_deg=math.degrees(two_theta),
        objective=objective,
        condenser_in=condenser_in,
        mainx=mainx,
        obx=obx,
        ffsel=ffsel,
        ffz=ffz,
        lenssel=lenssel,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_pixel_size.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Update `docs/Codebase.md`**

Under `### dfxm/common — shared primitives`, after the `#### h5io.py` block, add:

```markdown
#### `pixel_size.py` (new)
`compute_pixel_size(scan_h5, positioners_path="instrument/positioners",
entry_suffix=".1") -> PixelSizeResult`. Reads the far-field geometry motors
(`mainx`, `obx`, `ffsel`, `ffz`, `lenssel`) from the first matching entry of a
raw (pre-darfix) scan and derives the effective detector pixel size:
`M = mainx/obx − 1`, `E_x = base/M` (base 3.25 for 2× at `ffsel=−60`, 0.65 for
10× at `ffsel=0`), `2θ = arctan(ffz/mainx)`, and `E_y = E_x/sin(2θ)` when the
condenser is in (`lenssel=0`) else `E_y = E_x`. Raises `StageUserError` for a
missing entry/motor, an unrecognized `ffsel`, or a non-physical magnification.
`PixelSizeResult` carries both pixel sizes plus `magnification`,
`two_theta_deg`, `objective`, `condenser_in`, and the raw motor values.
```

- [ ] **Step 6: Run the full core suite + lint, then commit**

Run: `python3 -m pytest -q && ruff check . && ruff format .`
Expected: all pass; no lint errors.

```bash
git add dfxm/common/pixel_size.py tests/test_common_pixel_size.py docs/Codebase.md
git commit -m "feat(common): compute_pixel_size from a raw scan's far-field motors

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: "Compute pixel size from scan…" button in the Experiment editor

**Files:**
- Modify: `gui/experiment_panel.py`
- Modify: `tests/gui_smoke.py`
- Modify: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: `dfxm.common.pixel_size.compute_pixel_size`, `dfxm.common.errors.StageUserError`, `ParamForm.values()`, `ParamForm.set_values()`.
- Produces: `ExperimentDialog._apply_pixel_size(path: str) -> PixelSizeResult` (no modal dialogs — computes and writes the two pixel-size fields, raising `StageUserError` on failure) and `ExperimentDialog._on_compute_pixel_size()` (file picker + result/warning message boxes).

- [ ] **Step 1: Add the failing GUI-smoke checkpoint**

In `tests/gui_smoke.py`, immediately after checkpoint `[10]` (the compact experiment header block that ends with the `print("[10] ...")` line), add:

```python
    # [10b] Compute pixel size from a raw scan fills the two calibration fields.
    import h5py as _h5py

    _scan_dir = tempfile.mkdtemp()
    _scan = os.path.join(_scan_dir, "mosa_scan.h5")
    with _h5py.File(_scan, "w") as _f:
        _pos = _f.create_group("1.1/instrument/positioners")
        _pos.create_dataset("mainx", data=5000.0)
        _pos.create_dataset("obx", data=273.0)
        _pos.create_dataset("ffsel", data=-60.0)
        _pos.create_dataset("ffz", data=2100.0)
        _pos.create_dataset("lenssel", data=0.0)
    pdlg = panel._make_dialog()
    pdlg.show()
    app.processEvents()
    pres = pdlg._apply_pixel_size(_scan)  # no modal dialogs on this path
    app.processEvents()
    assert pres.objective == "2x" and pres.condenser_in is True
    vals = pdlg._form.values()
    assert vals["pixel_size_x_um"] == pytest.approx(pres.pixel_size_x_um)
    assert vals["pixel_size_y_um"] == pytest.approx(pres.pixel_size_y_um)
    pdlg.reject()
    app.processEvents()
    print("[10b] compute-pixel-size button fills X/Y from a scan's motors")
```

Add `import pytest` to the imports near the top of `tests/gui_smoke.py` (after the existing `import numpy as np` line):

```python
import pytest  # noqa: E402
```

- [ ] **Step 2: Run the smoke to verify it fails**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL at `[10b]` — `AttributeError: 'ExperimentDialog' object has no attribute '_apply_pixel_size'`.

- [ ] **Step 3: Implement the button + handlers**

In `gui/experiment_panel.py`, add `QMessageBox` to the `PySide6.QtWidgets` import block:

```python
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)
```

In `ExperimentDialog.__init__`, after the existing `save_btn` is added to `buttons` and before `buttons.accepted.connect(self.accept)`, add a compute button:

```python
        compute_btn = QPushButton("Compute pixel size from scan…")
        compute_btn.setToolTip(
            "Read a raw (pre-darfix) scan .h5 and fill Pixel size X/Y from its motors"
        )
        compute_btn.clicked.connect(self._on_compute_pixel_size)
        buttons.addButton(compute_btn, QDialogButtonBox.ButtonRole.ActionRole)
```

Add these two methods to `ExperimentDialog` (e.g. after `_on_save_as`):

```python
    def _apply_pixel_size(self, path: str):
        """Compute pixel sizes from *path* and write them into the form.

        No dialogs — raises StageUserError on a user-fixable problem so the
        caller can surface message/hint. Returns the PixelSizeResult.
        """
        from dfxm.common.pixel_size import compute_pixel_size

        vals = self._form.values()
        res = compute_pixel_size(
            path,
            positioners_path=vals.get("positioners_path") or "instrument/positioners",
            entry_suffix=vals.get("entry_suffix") or ".1",
        )
        self._form.set_values(
            {
                "pixel_size_x_um": res.pixel_size_x_um,
                "pixel_size_y_um": res.pixel_size_y_um,
            }
        )
        return res

    def _on_compute_pixel_size(self) -> None:
        from dfxm.common.errors import StageUserError

        vals = self._form.values()
        path, _ = QFileDialog.getOpenFileName(
            self, "Pick a raw scan .h5", vals.get("raw_root") or "", "HDF5 (*.h5 *.hdf5)"
        )
        if not path:
            return
        try:
            res = self._apply_pixel_size(path)
        except StageUserError as exc:
            QMessageBox.warning(self, "Compute pixel size", f"{exc}\n\n{exc.hint}")
            return
        except Exception as exc:  # noqa: BLE001 — unreadable/foreign file
            QMessageBox.warning(self, "Compute pixel size", f"Could not read scan:\n{exc}")
            return
        QMessageBox.information(
            self,
            "Compute pixel size",
            f"Objective {res.objective} (ffsel={res.ffsel:g})\n"
            f"M = {res.magnification:.3f}\n"
            f"2θ = {res.two_theta_deg:.3f}°\n"
            f"condenser {'in' if res.condenser_in else 'out'}\n\n"
            f"Pixel size X = {res.pixel_size_x_um:.4f} µm\n"
            f"Pixel size Y = {res.pixel_size_y_um:.4f} µm",
        )
```

- [ ] **Step 4: Run the smoke to verify it passes**

Run: `python3 tests/gui_smoke.py`
Expected: PASS — prints `[10b] compute-pixel-size button fills X/Y from a scan's motors` and ends with `GUI SMOKE PASSED`.

- [ ] **Step 5: Update docs**

`docs/Usage.md` — in `### Experiment presets`, add a paragraph:

```markdown
The experiment editor has a **Compute pixel size from scan…** button: pick a raw
(pre-darfix) scan `.h5` and it reads the far-field motors (`mainx`, `obx`,
`ffsel`, `ffz`, `lenssel`) and fills **Pixel size X** and **Pixel size Y** for
you, reporting the magnification, 2θ, the detected objective (2× / 10×) and
whether the condenser was in. Any unrecognized `ffsel` leaves the fields
untouched and explains what to set manually.
```

`docs/Codebase.md` — in `## Layer 2 — gui/` under the `experiment_panel.py` description (or the `ExperimentDialog` mention), add:

```markdown
`ExperimentDialog` also exposes **Compute pixel size from scan…**:
`_on_compute_pixel_size()` picks a scan and calls `_apply_pixel_size(path)`,
which runs `dfxm.common.pixel_size.compute_pixel_size` and writes
`pixel_size_x_um` / `pixel_size_y_um` back into the form (`_apply_pixel_size`
raises `StageUserError`; `_on_compute_pixel_size` shows the result/warning).
```

- [ ] **Step 6: Commit**

Run: `ruff check . && ruff format .`
Expected: pass.

```bash
git add gui/experiment_panel.py tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): Compute-pixel-size-from-scan button in the experiment editor

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: Hybrid help panel + richer hover tooltips

**Files:**
- Modify: `gui/widgets/help_panel.py`
- Modify: `gui/widgets/param_form.py`
- Modify: `gui/stage_view.py`
- Modify: `gui/experiment_panel.py`
- Modify: `tests/gui_smoke.py`
- Modify: `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Produces:
  - `help_panel.param_help_html(p: Param, error_color: str | None = None) -> str` — the rich-text used by both the panel and the tooltips (calibration line is coloured only when `error_color` is given).
  - `ParamForm.focusCleared` — a `Signal()` emitted when focus leaves all of the form's fields.
  - `StageView.showEvent` resets its help panel to the stage description.
- Consumes: existing `ParamForm.focusedParamChanged` (unchanged), `HelpPanel.show_idle`, `HelpPanel.show_param`.

- [ ] **Step 1: Add failing GUI-smoke assertions**

In `tests/gui_smoke.py`, replace the checkpoint `[9]` block (the four lines from `sview = win._views["strain"]` through `print("[9] help panel idles on description and follows focus")`) with:

```python
    # Help panel: idles on the stage description, follows focus, reverts on
    # focus-clear, resets on stage switch; tooltips carry the same rich help.
    sview = win._views["strain"]
    assert "strain" in sview._help._label.text().lower()
    sview._form.focus_param("ccmth_ref_deg")
    app.processEvents()
    help_text = sview._help._label.text()
    assert "Bragg" in help_text and "calibration" in help_text.lower()
    # Focus leaving the fields reverts the panel to the stage description.
    sview._form.focusCleared.emit()
    app.processEvents()
    assert sview._help._current is None
    assert "strain" in sview._help._label.text().lower()
    # Switching away and back resets the panel to the stage description.
    sview._help.show_param(sview._spec.params[0])  # force it onto a field
    assert sview._help._current is not None
    win._stack.setCurrentWidget(win._overview)
    app.processEvents()
    win._stack.setCurrentWidget(sview)
    app.processEvents()
    assert sview._help._current is None  # showEvent reset it to idle
    # Enriched hover tooltip on a calibration field.
    tip = sview._form._editors["ccmth_ref_deg"].toolTip()
    assert "Bragg" in tip and "calibration" in tip.lower()
    print("[9] help panel idles/follows/reverts/resets + enriched tooltips")
```

- [ ] **Step 2: Run the smoke to verify it fails**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL at `[9]` — `AttributeError: 'ParamForm' object has no attribute 'focusCleared'`.

- [ ] **Step 3: Factor `param_help_html` in `help_panel.py`**

In `gui/widgets/help_panel.py`, add a module-level function (after the imports, before `class HelpPanel`):

```python
def param_help_html(p: Param, error_color: str | None = None) -> str:
    """Rich-text help for *p*: label (+unit), calibration note, help text.

    The calibration note is coloured with *error_color* when given (the help
    panel), otherwise rendered plain (tooltips, which do not restyle on theme
    change).
    """
    head = f"<b>{html.escape(p.label)}</b>"
    if p.unit:
        head += f" ({html.escape(p.unit)})"
    parts = [head]
    if p.calibration:
        warn = (
            "⚠ calibration — physically meaningful; confirm against the beamline "
            "calibration for your experiment."
        )
        if error_color:
            parts.append(f'<span style="color:{error_color};">{warn}</span>')
        else:
            parts.append(warn)
    if p.help:
        parts.append(html.escape(p.help))
    return "<br>".join(parts)
```

Replace `HelpPanel._cal_warning` and `HelpPanel._render` so the panel reuses the helper:

```python
    def set_idle(self, title: str, description: str) -> None:
        """Set (and show) the text used when no field is focused."""
        self._idle_html = f"<b>{html.escape(title)}</b> — {html.escape(description)}"
        self._current = None
        self._render()

    def show_idle(self) -> None:
        self._current = None
        self._render()

    def show_param(self, p: Param) -> None:
        self._current = p
        self._render()

    def _render(self) -> None:
        if self._current is None:
            self._label.setText(self._idle_html)
            return
        self._label.setText(param_help_html(self._current, self._error_color))
```

Delete the now-unused `_cal_warning` method.

- [ ] **Step 4: Add `focusCleared` + enriched tooltips in `param_form.py`**

In `gui/widgets/param_form.py`:

Add `QApplication` to the `PySide6.QtWidgets` import block, and import the helper + theme:

```python
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from dfxm.config.models import Param, ParamType

from ..theme import ThemeController
from .help_panel import param_help_html
```

Add the new signal next to the existing ones:

```python
    changed = Signal()
    focusedParamChanged = Signal(object)  # the focused Param
    focusCleared = Signal()  # focus left every field in this form
```

At the end of `__init__` (after the essentials/advanced layout is built), connect to the app-wide focus signal:

```python
        app = QApplication.instance()
        if app is not None:
            app.focusChanged.connect(self._on_focus_changed)
```

Add the handler (near `eventFilter`):

```python
    def _on_focus_changed(self, old: QObject, new: QObject) -> None:  # Qt slot
        # Only react when focus leaves one of *our* fields for something that
        # is not one of our fields -> revert the help panel to the stage text.
        if old in self._param_for_widget and new not in self._param_for_widget:
            self.focusCleared.emit()
```

(Keep the existing `eventFilter` — it still emits `focusedParamChanged` on `FocusIn`.)

Set the enriched tooltip once, in `_make_editor`, and drop the per-builder `setToolTip(p.help)` calls. Replace `_make_editor` with:

```python
    def _make_editor(self, p: Param, initial: dict[str, Any]) -> QWidget:
        editor = self._build_editor(p, initial.get(p.name, p.default))
        self._editors[p.name] = editor
        tip = param_help_html(p, ThemeController.instance().palette.error)
        for w in (editor, *editor.findChildren(QWidget)):
            w.installEventFilter(self)
            self._param_for_widget[w] = p
            w.setToolTip(tip)
        return editor
```

Now remove every `if p.help: <widget>.setToolTip(p.help)` line inside `_enum_editor`, `_bool_editor`, `_int_editor`, `_float_editor`, `_str_editor`, `_text_editor`, and `_path_editor` (the `_make_editor` loop sets a richer tooltip on all of them). In `_label_for`, replace:

```python
        if p.help:
            lbl.setToolTip(p.help)
```

with:

```python
        lbl.setToolTip(param_help_html(p, ThemeController.instance().palette.error))
```

- [ ] **Step 5: Reset help on stage show + wire `focusCleared` in `stage_view.py`**

In `gui/stage_view.py`, where the form/help are wired (after `self._form.focusedParamChanged.connect(self._help.show_param)`), add:

```python
        self._form.focusCleared.connect(self._help.show_idle)
```

Add a `showEvent` override to `StageView` (e.g. just after `__init__`):

```python
    def showEvent(self, event) -> None:  # Qt hook
        super().showEvent(event)
        self._help.show_idle()  # every stage opens on its description
```

- [ ] **Step 6: Wire `focusCleared` in the experiment dialog**

In `gui/experiment_panel.py`, `ExperimentDialog.__init__`, after
`self._form.focusedParamChanged.connect(help_panel.show_param)`, add:

```python
        self._form.focusCleared.connect(help_panel.show_idle)
```

- [ ] **Step 7: Run the smoke to verify it passes**

Run: `python3 tests/gui_smoke.py`
Expected: PASS — `[9] help panel idles/follows/reverts/resets + enriched tooltips` and `GUI SMOKE PASSED`.

- [ ] **Step 8: Update docs**

`docs/Usage.md` — in `### The stage panel`, add:

```markdown
A help box under the form shows the current stage's description by default. Click
a field and it shows that field's help; click away (or open another stage) and it
returns to the stage description. The same per-field help is also available as a
hover tooltip on each field and its label.
```

`docs/Codebase.md` — in `### gui/widgets/`, update the `help_panel.py` / `param_form.py` entries to note: `param_help_html(p, error_color=None)` renders the shared rich help (panel + tooltips); `ParamForm` emits `focusCleared` (via `QApplication.focusChanged`) when focus leaves its fields and sets the enriched tooltip on every editor/label; `StageView.showEvent` resets the help panel to the stage description.

- [ ] **Step 9: Full suite + lint, commit**

Run: `python3 -m pytest -q && ruff check . && ruff format . && python3 tests/gui_smoke.py`
Expected: pytest passes; no lint errors; smoke prints `GUI SMOKE PASSED`.

```bash
git add gui/widgets/help_panel.py gui/widgets/param_form.py gui/stage_view.py gui/experiment_panel.py tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): help panel reverts to stage description + richer hover tooltips

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Shared middle/right splitter width across stages + `WindowState`

**Files:**
- Create: `gui/window_state.py`
- Modify: `gui/stage_view.py`
- Modify: `gui/main_window.py`
- Modify: `tests/gui_smoke.py`
- Modify: `docs/Codebase.md`

**Interfaces:**
- Produces:
  - `gui/window_state.py`: `DEFAULT_STAGE_SIZES: list[int]` (first-run middle|right default) and `WindowState` with:
    - `__init__(self, settings: QSettings | None = None)`
    - `register_stage_splitter(self, splitter: QSplitter) -> None`
    - `restore(self, window: QMainWindow, main_splitter: QSplitter) -> None`
    - `save(self, window: QMainWindow, main_splitter: QSplitter) -> None`
  - `StageView.inner_splitter: QSplitter` (the middle|right splitter).
- Consumes: nothing new from earlier tasks.

- [ ] **Step 1: Add the failing GUI-smoke checkpoint**

In `tests/gui_smoke.py`, just before the final `print("\nGUI SMOKE PASSED")`, add:

```python
    # [21] Stage splitters share one middle|right width via WindowState.
    from PySide6.QtCore import QSettings as _QSettings
    from PySide6.QtWidgets import QSplitter as _QSplitter
    from PySide6.QtWidgets import QWidget as _QWidget

    from gui.window_state import WindowState as _WindowState

    _ws = _WindowState(_QSettings())
    _a = _QSplitter()
    _a.addWidget(_QWidget())
    _a.addWidget(_QWidget())
    _a.resize(1000, 200)
    _a.show()
    _b = _QSplitter()
    _b.addWidget(_QWidget())
    _b.addWidget(_QWidget())
    _b.resize(1000, 200)
    _b.show()
    app.processEvents()
    _ws.register_stage_splitter(_a)
    _ws.register_stage_splitter(_b)
    _a.setSizes([700, 300])
    _a.splitterMoved.emit(700, 1)  # simulate a user drag
    app.processEvents()
    assert _b.sizes() == _a.sizes(), (_a.sizes(), _b.sizes())
    # Real stage views expose an inner splitter and share the same sizes.
    assert win._views["strain"].inner_splitter is not None
    assert (
        win._views["strain"].inner_splitter.sizes()
        == win._views["mosaicity"].inner_splitter.sizes()
    )
    _a.deleteLater()
    _b.deleteLater()
    app.processEvents()
    print("[21] shared stage-splitter width via WindowState")
```

- [ ] **Step 2: Run the smoke to verify it fails**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL at `[21]` — `ModuleNotFoundError: No module named 'gui.window_state'`.

- [ ] **Step 3: Create `gui/window_state.py`**

```python
"""Persist and share window/splitter geometry (Qt-side; core stays Qt-free).

Uses the app-wide QSettings (org ``dfxm``, app ``pipeline``):
  - ``geometry``      : QMainWindow.saveGeometry() (size, position, maximized)
  - ``mainSplitter``  : the top-level (left rail | stack) splitter state
  - ``stageSplitter`` : the shared middle|right split applied to every stage

Every stage's inner splitter is registered here and kept in lock-step: dragging
any one broadcasts its sizes to the others and writes them back, so all stages
show the same middle|right width across a session and across restarts.
"""

from __future__ import annotations

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QMainWindow, QSplitter

_KEY_GEOMETRY = "geometry"
_KEY_MAIN_SPLIT = "mainSplitter"
_KEY_STAGE_SPLIT = "stageSplitter"

#: First-run middle|right split — favours the middle (parameter) column.
DEFAULT_STAGE_SIZES: list[int] = [560, 460]


class WindowState:
    """Save/restore window geometry and keep the stage splitters in sync."""

    def __init__(self, settings: QSettings | None = None) -> None:
        self._settings = settings or QSettings()
        self._stage_splitters: list[QSplitter] = []
        self._applying = False

    # -- shared stage splitter -------------------------------------------
    def register_stage_splitter(self, splitter: QSplitter) -> None:
        """Track *splitter*, apply the shared sizes, and mirror future drags."""
        self._stage_splitters.append(splitter)
        splitter.setSizes(self._saved_stage_sizes())
        splitter.splitterMoved.connect(lambda _pos, _i, s=splitter: self._on_stage_moved(s))

    def _on_stage_moved(self, source: QSplitter) -> None:
        if self._applying:
            return
        sizes = source.sizes()
        if not sizes or sum(sizes) == 0:
            return
        self._settings.setValue(_KEY_STAGE_SPLIT, sizes)
        self._applying = True
        try:
            for s in self._stage_splitters:
                if s is not source:
                    s.setSizes(sizes)
        finally:
            self._applying = False

    def _saved_stage_sizes(self) -> list[int]:
        raw = self._settings.value(_KEY_STAGE_SPLIT)
        if raw is None:
            return list(DEFAULT_STAGE_SIZES)
        if isinstance(raw, str):
            raw = raw.split(",")
        try:
            sizes = [int(float(x)) for x in raw]
        except (TypeError, ValueError):
            return list(DEFAULT_STAGE_SIZES)
        return sizes if len(sizes) >= 2 else list(DEFAULT_STAGE_SIZES)

    # -- geometry + main splitter ----------------------------------------
    def restore(self, window: QMainWindow, main_splitter: QSplitter) -> None:
        """Restore window geometry + the top-level splitter, if saved."""
        geo = self._settings.value(_KEY_GEOMETRY)
        if geo is not None:
            try:
                window.restoreGeometry(geo)
            except (TypeError, ValueError):  # corrupt/foreign state
                pass
        state = self._settings.value(_KEY_MAIN_SPLIT)
        if state is not None:
            try:
                main_splitter.restoreState(state)
            except (TypeError, ValueError):
                pass

    def save(self, window: QMainWindow, main_splitter: QSplitter) -> None:
        """Persist window geometry + the top-level splitter state."""
        self._settings.setValue(_KEY_GEOMETRY, window.saveGeometry())
        self._settings.setValue(_KEY_MAIN_SPLIT, main_splitter.saveState())
```

- [ ] **Step 4: Expose `inner_splitter` + shared default in `stage_view.py`**

In `gui/stage_view.py`, add the import near the other `.` imports:

```python
from .window_state import DEFAULT_STAGE_SIZES
```

In `StageView.__init__`, replace the splitter block:

```python
        splitter = QSplitter()
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([360, 600])
```

with:

```python
        splitter = QSplitter()
        splitter.addWidget(left_scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes(list(DEFAULT_STAGE_SIZES))
        self.inner_splitter = splitter
```

(The middle pane now gets a non-zero stretch factor so it grows with the window instead of staying pinned narrow.)

- [ ] **Step 5: Register the stage splitters in `main_window.py`**

In `gui/main_window.py`, add the import:

```python
from .window_state import WindowState
```

In `MainWindow.__init__`, right after `super().__init__()` / `self.resize(1100, 720)`, create the state object:

```python
        self._window_state = WindowState()
```

At the end of `__init__` (after `self.setCentralWidget(splitter)`), keep a reference to the top-level splitter and register every stage's inner splitter:

```python
        self._main_splitter = splitter
        for name in STAGE_ORDER:
            self._window_state.register_stage_splitter(self._views[name].inner_splitter)
```

- [ ] **Step 6: Run the smoke to verify it passes**

Run: `python3 tests/gui_smoke.py`
Expected: PASS — `[21] shared stage-splitter width via WindowState` and `GUI SMOKE PASSED`.

- [ ] **Step 7: Update docs**

`docs/Codebase.md` — in `## Layer 2 — gui/`, add a `window_state.py` entry:

```markdown
#### `window_state.py` (new)
`WindowState` persists window geometry (size/position/maximized) and the
top-level splitter via `QSettings`, and keeps every stage's middle|right
splitter in lock-step: `register_stage_splitter` applies the shared width and
mirrors future drags to all stages (`DEFAULT_STAGE_SIZES` is the first-run
middle-favoured default). `MainWindow` owns one instance; each `StageView`
exposes its `inner_splitter`.
```

- [ ] **Step 8: Full suite + lint, commit**

Run: `python3 -m pytest -q && ruff check . && ruff format . && python3 tests/gui_smoke.py`
Expected: pytest passes; no lint errors; smoke `GUI SMOKE PASSED`.

```bash
git add gui/window_state.py gui/stage_view.py gui/main_window.py tests/gui_smoke.py docs/Codebase.md
git commit -m "feat(gui): shared, wider middle|right splitter width across stages

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Remember window geometry + maximized state + top-level splitter

**Files:**
- Modify: `gui/main_window.py`
- Modify: `tests/gui_smoke.py`
- Modify: `docs/Usage.md`

**Interfaces:**
- Consumes: `WindowState.restore` / `WindowState.save` (Task 4).
- Produces: `MainWindow.closeEvent` persists window state; `MainWindow.__init__` restores it.

- [ ] **Step 1: Isolate QSettings in the smoke, and add the failing checkpoint**

In `tests/gui_smoke.py`, near the top where the Qt platform is forced (just after `os.environ.setdefault("QT_API", "pyside6")`), add QSettings isolation so the smoke never clobbers the real user config:

```python
import tempfile as _tempfile_isolate  # noqa: E402

from PySide6.QtCore import QSettings as _QSettingsIsolate  # noqa: E402
from PySide6.QtWidgets import QApplication as _QAppIsolate  # noqa: E402

_QAppIsolate.setOrganizationName("dfxm-smoke")
_QAppIsolate.setApplicationName("pipeline-smoke")
_QSettingsIsolate.setDefaultFormat(_QSettingsIsolate.Format.IniFormat)
_QSettingsIsolate.setPath(
    _QSettingsIsolate.Format.IniFormat,
    _QSettingsIsolate.Scope.UserScope,
    _tempfile_isolate.mkdtemp(),
)
```

Then, just before the final `print("\nGUI SMOKE PASSED")` (after checkpoint `[21]`), add:

```python
    # [22] WindowState saves geometry and restores without raising.
    from PySide6.QtCore import QSettings as _QSettings22
    from PySide6.QtWidgets import QMainWindow as _QMainWindow22
    from PySide6.QtWidgets import QSplitter as _QSplitter22
    from PySide6.QtWidgets import QWidget as _QWidget22

    from gui.window_state import WindowState as _WindowState22

    _iso = _QSettings22()
    _ws22 = _WindowState22(_iso)
    _w1 = _QMainWindow22()
    _w1.resize(900, 640)
    _w1.show()
    app.processEvents()
    _ms1 = _QSplitter22()
    _ms1.addWidget(_QWidget22())
    _ms1.addWidget(_QWidget22())
    _ws22.save(_w1, _ms1)
    assert _iso.value("geometry") is not None
    _w2 = _QMainWindow22()
    _ms2 = _QSplitter22()
    _ms2.addWidget(_QWidget22())
    _ms2.addWidget(_QWidget22())
    _ws22.restore(_w2, _ms2)  # must not raise
    app.processEvents()
    # MainWindow persists on close without raising.
    win.close()
    app.processEvents()
    _w1.deleteLater()
    _w2.deleteLater()
    app.processEvents()
    print("[22] window geometry + splitter state persist/restore")
```

- [ ] **Step 2: Run the smoke to verify it fails**

Run: `python3 tests/gui_smoke.py`
Expected: FAIL at `[22]` — the geometry key is absent / `win.close()` does not persist, i.e. `assert _iso.value("geometry") is not None` fails only if save is wrong; the real failing point is the missing restore/close wiring proven next. (If `[22]` passes purely on the standalone `WindowState`, the wiring is still added in Step 3 for the real window.)

Note: the standalone `WindowState` from Task 4 already works, so the meaningful new wiring is `MainWindow.__init__` restore + `MainWindow.closeEvent`. Confirm `win.close()` triggers no exception and, after Step 3, that the real window persisted geometry: extend the assertion in Step 3.

- [ ] **Step 3: Wire restore on startup + save on close in `main_window.py`**

In `gui/main_window.py`, at the very end of `__init__` (after the stage-splitter registration loop from Task 4), restore saved state:

```python
        self._window_state.restore(self, self._main_splitter)
```

Add a `closeEvent` override to `MainWindow` (e.g. after `_show_stage`):

```python
    def closeEvent(self, event) -> None:  # Qt hook
        self._window_state.save(self, self._main_splitter)
        super().closeEvent(event)
```

Now strengthen the `[22]` checkpoint to prove the *real* window persisted: replace the `win.close()` / `print` tail of `[22]` with:

```python
    # MainWindow persists on close using the app-wide (isolated) settings.
    win.close()
    app.processEvents()
    assert _QSettings22().value("geometry") is not None
    _w1.deleteLater()
    _w2.deleteLater()
    app.processEvents()
    print("[22] window geometry + splitter state persist/restore")
```

- [ ] **Step 4: Run the smoke to verify it passes**

Run: `python3 tests/gui_smoke.py`
Expected: PASS — `[22] window geometry + splitter state persist/restore` and `GUI SMOKE PASSED`.

- [ ] **Step 5: Update docs**

`docs/Usage.md` — in `### The main window`, add:

```markdown
The window remembers its size, position and maximized state between runs, along
with the left-rail width and the shared middle/right column width — so the layout
you set stays put next time you open the app. Drag the divider between the
parameter form and the Log/Results/Output panel to rebalance them; the new width
applies to every stage and is remembered.
```

- [ ] **Step 6: Full suite + lint, commit**

Run: `python3 -m pytest -q && ruff check . && ruff format . && python3 tests/gui_smoke.py`
Expected: pytest passes; no lint errors; smoke `GUI SMOKE PASSED`.

```bash
git add gui/main_window.py tests/gui_smoke.py docs/Usage.md
git commit -m "feat(gui): remember window geometry, maximized state, and splitter widths

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Feature 1 (wider middle + shared width + geometry/maximized/splitter persistence) → Tasks 4 (shared width + default) & 5 (geometry/maximized/top-splitter). ✔
- Feature 2 (compute-from-scan pixel size: core formulas, 2×/10× via ffsel, condenser via lenssel, error on bad ffsel/missing motor; button in Experiment editor) → Tasks 1 (core) & 2 (button). ✔
- Feature 3 (help reverts on click-away + resets on stage switch + richer hover tooltips) → Task 3. ✔
- Docs contract (Usage.md + Codebase.md) → covered in Tasks 1, 2, 3, 4, 5. ✔
- Non-goals honoured: no new Experiment motor-name fields; 2×/10× only; unified (not per-stage) width. ✔

**Placeholder scan:** No TBD/TODO; every code step shows complete code and exact commands.

**Type consistency:** `compute_pixel_size(...) -> PixelSizeResult` and its field names are used identically in Tasks 1 and 2 (`res.objective`, `res.condenser_in`, `res.pixel_size_x_um/y_um`, `res.magnification`, `res.two_theta_deg`, `res.ffsel`). `WindowState` method names (`register_stage_splitter`, `save`, `restore`) and `DEFAULT_STAGE_SIZES` / `StageView.inner_splitter` match across Tasks 4 and 5. `ParamForm.focusCleared` and `param_help_html` are defined and consumed consistently in Task 3.

**Note on Task 5 Step 2:** the standalone `WindowState` already works from Task 4, so the genuinely new coverage is the *real* `MainWindow` restore/close wiring — the strengthened assertion (`_QSettings22().value("geometry") is not None` after `win.close()`) is what fails before Step 3 and passes after.
