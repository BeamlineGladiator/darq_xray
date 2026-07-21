# ROI Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-07-21-roi-unification-design.md` (approved)

**Goal:** One canonical ROI entry on the experiment (darfix window as origin+size + analysis window in map-frame) from which every stage's ROI field pre-fills in its own native frame — making the 2026-07-18 misregistration class of error structurally impossible.

**Architecture:** Plain string fields on `Experiment` + a pure Qt-free conversion module `dfxm/common/roi.py`; pre-fill rides the existing `experiment_overrides` mechanism in `gui/bindings.py`; the experiment editor gains a live derived-frames read-out, save-time validation, and a picker; ROI stage fields get a deviation marker and frame-honest help. A test-enforced `Param.roi_frame` declaration documents each field's frame.

**Tech Stack:** Python 3.10, dataclasses, PySide6 (GUI layer only), pytest, existing `ROIPickerDialog` + `stacked_volume_previews`.

## Global Constraints

- `dfxm/` stays Qt-free — `dfxm/common/roi.py` must not import PySide6/pyvista, and takes/returns plain strings/tuples (no Qt, no I/O).
- Docs contract: every task that changes stage params, GUI behaviour, or public functions updates `docs/Usage.md` and/or `docs/Codebase.md` **in the same commit**.
- `ruff format` runs automatically on Write/Edit (hook); line length 100, double quotes.
- Read every file before its first Edit this session; `hint=`/help strings contain em-dashes at varying indents — never reconstruct `old_string` from memory.
- Figures: explicit `matplotlib.figure.Figure` API only; never pyplot (not needed in this plan).
- Suite baseline at branch point (`cf6eec7`): ~526 passed / 13 skipped; `python3 -m pytest -q`, `ruff check .`, and `python3 tests/gui_smoke.py` must be green at every commit.
- GUI tests set `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")` before importing PySide6 (see `tests/test_gui_stage_view_roi.py` for the pattern).
- Worked STO2 numbers used as golden values throughout: darfix origin (105, 230) size (1832, 1266); analysis map-frame x `0,1832`, y `400,1100`; derived detector x `105,1937`, y `630,1330`.

## File Structure

- **Create** `dfxm/common/roi.py` — `DarfixWindow`, `parse_pair`, `parse_darfix_roi`, `map_to_detector`, `detector_to_map`, `analysis_detector_window`, `format_pair`, `validate_rois`. The only place `detector = darfix_origin + map` exists.
- **Create** `tests/test_common_roi.py` — unit tests incl. the STO2 golden case.
- **Modify** `dfxm/config/models.py` — 3 new `Experiment` fields + 3 `EXPERIMENT_SCHEMA` params (order must match the dataclass) + `Param.roi_frame`.
- **Modify** `experiments/STO2_overnight.yaml` — real ROI values.
- **Modify** `dfxm/stages/{rocking,visualize,paraview,slices,strain}.py` — `roi_frame` + frame-honest labels/help.
- **Modify** `gui/bindings.py` — `_roi_overrides` merged into `experiment_overrides`.
- **Create** `tests/test_bindings_roi.py` — per-stage pre-fill tests.
- **Modify** `gui/experiment_panel.py` — derived read-out, validation on OK/Save-as, Pick analysis ROI… button.
- **Create** `tests/test_gui_experiment_roi.py` — dialog tests.
- **Modify** `gui/widgets/param_form.py` — `set_field_marker` API; **Modify** `gui/stage_view.py` — `_update_roi_markers` wiring.
- **Modify** `tests/test_gui_stage_view_roi.py`, `tests/test_config.py`, `tests/test_param_metadata.py`, `tests/gui_smoke.py`, `docs/Usage.md`, `docs/Codebase.md`.

Branch: create `roi-unification` from `master` before Task 1 (`git checkout -b roi-unification`). No remote — no push anywhere.

---

### Task 1: `dfxm/common/roi.py` — frames and conversions

**Files:**
- Create: `dfxm/common/roi.py`
- Test: `tests/test_common_roi.py`
- Modify: `docs/Codebase.md` (add module row under `### dfxm/common — shared primitives`)

**Interfaces:**
- Consumes: nothing (pure module).
- Produces (later tasks call these exact signatures):
  - `parse_pair(text: str | None) -> tuple[int, int] | None` (raises `ValueError` on malformed)
  - `parse_darfix_roi(text: str | None) -> DarfixWindow | None` (raises `ValueError`)
  - `DarfixWindow(origin_x, origin_y, width, height)` with properties `.x0 .x1 .y0 .y1`
  - `map_to_detector(pair, origin) -> tuple[int, int]`, `detector_to_map(pair, origin) -> tuple[int, int]`
  - `analysis_detector_window(darfix_roi: str, analysis_roi_x: str, analysis_roi_y: str) -> tuple[tuple[int,int]|None, tuple[int,int]|None]`
  - `format_pair(pair: tuple[int, int]) -> str`
  - `validate_rois(darfix_roi: str, analysis_roi_x: str, analysis_roi_y: str) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_common_roi.py`:

```python
"""dfxm.common.roi — darfix-window / map-frame conversions and validation."""

from __future__ import annotations

import pytest

from dfxm.common import roi as R

STO2_DARFIX = "105,230,1832,1266"


def test_parse_pair():
    assert R.parse_pair("400,1100") == (400, 1100)
    assert R.parse_pair(" 400 , 1100 ") == (400, 1100)
    assert R.parse_pair("") is None
    assert R.parse_pair(None) is None
    with pytest.raises(ValueError):
        R.parse_pair("400")
    with pytest.raises(ValueError):
        R.parse_pair("a,b")


def test_parse_darfix_roi():
    win = R.parse_darfix_roi(STO2_DARFIX)
    assert (win.origin_x, win.origin_y, win.width, win.height) == (105, 230, 1832, 1266)
    assert (win.x0, win.x1, win.y0, win.y1) == (105, 1937, 230, 1496)
    assert R.parse_darfix_roi("") is None
    with pytest.raises(ValueError):
        R.parse_darfix_roi("105,230,1832")


def test_map_detector_round_trip():
    assert R.map_to_detector((400, 1100), 230) == (630, 1330)
    assert R.detector_to_map((630, 1330), 230) == (400, 1100)
    pair = (12, 345)
    assert R.detector_to_map(R.map_to_detector(pair, 105), 105) == pair


def test_analysis_detector_window_sto2_golden():
    """The 2026-07-18 incident's hand-conversion, as a derivation."""
    det_x, det_y = R.analysis_detector_window(STO2_DARFIX, "0,1832", "400,1100")
    assert det_x == (105, 1937)
    assert det_y == (630, 1330)


def test_analysis_detector_window_blank_axis_falls_back_to_full_window():
    det_x, det_y = R.analysis_detector_window(STO2_DARFIX, "", "400,1100")
    assert det_x == (105, 1937)  # full darfix width
    assert det_y == (630, 1330)


def test_analysis_detector_window_no_darfix_derives_nothing():
    assert R.analysis_detector_window("", "0,1832", "400,1100") == (None, None)


def test_format_pair():
    assert R.format_pair((630, 1330)) == "630,1330"


def test_validate_rois_ok_and_blank():
    assert R.validate_rois(STO2_DARFIX, "0,1832", "400,1100") == []
    assert R.validate_rois("", "", "") == []
    assert R.validate_rois("", "0,100", "") == []  # analysis without darfix is allowed


def test_validate_rois_problems():
    assert R.validate_rois("105,230", "", "")  # malformed darfix
    assert R.validate_rois("105,230,0,1266", "", "")  # non-positive size
    assert R.validate_rois(STO2_DARFIX, "banana", "")  # malformed pair
    assert R.validate_rois(STO2_DARFIX, "", "1100,400")  # end <= start
    assert R.validate_rois(STO2_DARFIX, "", "-5,100")  # negative start
    msgs = R.validate_rois(STO2_DARFIX, "", "400,1300")  # 1300 > height 1266
    assert msgs and "1266" in msgs[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_common_roi.py -q`
Expected: collection error — `ModuleNotFoundError: No module named 'dfxm.common.roi'`.

- [ ] **Step 3: Write the implementation**

Create `dfxm/common/roi.py`:

```python
"""Darfix-window / map-frame ROI conversions and validation.

Two regions of interest describe every DFXM dataset:

* The **darfix window** — the detector crop darfix used when fitting the maps,
  displayed by darfix as *origin + size* ``x,y,w,h``. A fact about how the maps
  were made: map pixel (0, 0) sits at detector pixel ``(x, y)``.
* The **analysis window** — the sub-region chosen for study, expressed in
  *map-frame* start,end pairs (columns ``c0,c1``, rows ``r0,r1``).

Stages consume these in different frames: rocking crops raw detector frames
(absolute pixels), the map stages crop darfix maps (map pixels). The
converters here are the single place the frames meet:
``detector = darfix_origin + map``. Pure functions — no Qt, no I/O.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DarfixWindow:
    """The darfix detector crop, as origin + size (what darfix displays)."""

    origin_x: int
    origin_y: int
    width: int
    height: int

    @property
    def x0(self) -> int:
        return self.origin_x

    @property
    def x1(self) -> int:
        return self.origin_x + self.width

    @property
    def y0(self) -> int:
        return self.origin_y

    @property
    def y1(self) -> int:
        return self.origin_y + self.height


def parse_pair(text: str | None) -> tuple[int, int] | None:
    """'start,end' -> (start, end); blank/None -> None; malformed -> ValueError."""
    if text is None or not str(text).strip():
        return None
    parts = [s.strip() for s in str(text).split(",")]
    if len(parts) != 2:
        raise ValueError(f"expected 'start,end' (two integers), got {text!r}")
    try:
        return int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"expected 'start,end' (two integers), got {text!r}") from exc


def parse_darfix_roi(text: str | None) -> DarfixWindow | None:
    """'x,y,w,h' (origin+size, darfix's display) -> DarfixWindow; blank -> None."""
    if text is None or not str(text).strip():
        return None
    parts = [s.strip() for s in str(text).split(",")]
    if len(parts) != 4:
        raise ValueError(f"expected 'x,y,w,h' (four integers, origin+size), got {text!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"expected 'x,y,w,h' (four integers, origin+size), got {text!r}") from exc
    return DarfixWindow(x, y, w, h)


def map_to_detector(pair: tuple[int, int], origin: int) -> tuple[int, int]:
    """Map-frame start,end -> absolute detector pixels along one axis."""
    return pair[0] + origin, pair[1] + origin


def detector_to_map(pair: tuple[int, int], origin: int) -> tuple[int, int]:
    """Absolute detector start,end -> map-frame pixels along one axis."""
    return pair[0] - origin, pair[1] - origin


def format_pair(pair: tuple[int, int]) -> str:
    return f"{pair[0]},{pair[1]}"


def analysis_detector_window(
    darfix_roi: str, analysis_roi_x: str, analysis_roi_y: str
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    """The analysis window in absolute detector pixels (what rocking crops).

    A blank analysis axis falls back to the full darfix window; without a
    darfix window nothing is derivable -> (None, None). Malformed input raises
    ValueError (use :func:`validate_rois` for user-facing messages).
    """
    win = parse_darfix_roi(darfix_roi)
    if win is None:
        return None, None
    ax = parse_pair(analysis_roi_x)
    ay = parse_pair(analysis_roi_y)
    det_x = map_to_detector(ax, win.origin_x) if ax else (win.x0, win.x1)
    det_y = map_to_detector(ay, win.origin_y) if ay else (win.y0, win.y1)
    return det_x, det_y


def validate_rois(darfix_roi: str, analysis_roi_x: str, analysis_roi_y: str) -> list[str]:
    """Human-readable problems with the experiment ROI fields ([] = all fine)."""
    problems: list[str] = []
    win = None
    try:
        win = parse_darfix_roi(darfix_roi)
    except ValueError as exc:
        problems.append(f"Darfix ROI: {exc}")
    else:
        if win is not None and (win.width <= 0 or win.height <= 0):
            problems.append("Darfix ROI: width and height must be positive (it is origin+size)")
            win = None
    for label, text, size in (
        ("Analysis window X", analysis_roi_x, win.width if win else None),
        ("Analysis window Y", analysis_roi_y, win.height if win else None),
    ):
        try:
            pair = parse_pair(text)
        except ValueError as exc:
            problems.append(f"{label}: {exc}")
            continue
        if pair is None:
            continue
        start, end = pair
        if start < 0 or end <= start:
            problems.append(f"{label}: need 0 <= start < end, got {start},{end}")
        elif size is not None and end > size:
            problems.append(
                f"{label}: end {end} exceeds the darfix window size {size} "
                "(analysis windows are map-frame, relative to the darfix window)"
            )
    return problems
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_common_roi.py -q`
Expected: all pass.

- [ ] **Step 5: Docs**

In `docs/Codebase.md`, section `### dfxm/common — shared primitives`: add a `roi.py` entry alongside the existing module descriptions (Read the section first to match its format), describing the two frames, the `detector = darfix_origin + map` rule, and the public functions above.

- [ ] **Step 6: Commit**

```bash
git add dfxm/common/roi.py tests/test_common_roi.py docs/Codebase.md
git commit -m "feat(common): roi.py — darfix-window/map-frame conversions + validation"
```

---

### Task 2: Experiment ROI fields + schema + STO2 preset

**Files:**
- Modify: `dfxm/config/models.py` (Experiment dataclass ~line 146; EXPERIMENT_SCHEMA after the `pixel_size_y_um` Param ~line 279)
- Modify: `experiments/STO2_overnight.yaml`
- Test: `tests/test_config.py`
- Modify: `docs/Usage.md` (Experiment presets section), `docs/Codebase.md` (`### dfxm/config`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Experiment.darfix_roi`, `Experiment.analysis_roi_x`, `Experiment.analysis_roi_y` (all `str`, default `""`) — read by Tasks 4–6.

**Ordering constraint:** `test_experiment_schema_matches_dataclass_fields` compares the dataclass field list to the schema name list — insert the three fields **after `pixel_size_y_um` and before `maps_filename` in BOTH places**, same order.

- [ ] **Step 1: Write the failing test additions**

Read `tests/test_config.py` in full, then extend `test_sto2_preset_ships_expected_values` with (match its existing style for loading the preset):

```python
    # regions of interest (confirmed on real data 2026-07-18/19)
    assert exp.darfix_roi == "105,230,1832,1266"
    assert exp.analysis_roi_x == "0,1832"
    assert exp.analysis_roi_y == "400,1100"
```

And add a new test:

```python
def test_experiment_roi_fields_default_blank():
    exp = Experiment()
    assert exp.darfix_roi == ""
    assert exp.analysis_roi_x == ""
    assert exp.analysis_roi_y == ""
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_config.py -q`
Expected: FAIL — `Experiment` has no attribute `darfix_roi` (and the schema-sync test fails once fields exist but schema entries don't — both directions get exercised during this task).

- [ ] **Step 3: Implement**

In `dfxm/config/models.py`, after `pixel_size_y_um: float = 1.0` add:

```python
    # --- regions of interest (frames + conversions: dfxm/common/roi.py) ------
    darfix_roi: str = ""  # darfix detector crop as darfix displays it: "x,y,w,h" origin+size
    analysis_roi_x: str = ""  # analysis window columns, map-frame "c0,c1" (blank = full)
    analysis_roi_y: str = ""  # analysis window rows, map-frame "r0,r1" (blank = full)
```

In `EXPERIMENT_SCHEMA`, after the `pixel_size_y_um` Param add:

```python
    Param(
        "darfix_roi",
        ParamType.STR,
        "Darfix ROI (origin+size)",
        help=(
            "The detector crop used in darfix, exactly as darfix's ROI widget shows it: "
            "'x,y,w,h' — origin then size (e.g. 105,230,1832,1266). Copy the four numbers "
            "verbatim, no conversion. Map pixel (0,0) sits at detector pixel (x,y); stages "
            "derive their detector-frame crops from this. Leave blank if darfix ran uncropped."
        ),
    ),
    Param(
        "analysis_roi_x",
        ParamType.STR,
        "Analysis window X (map px)",
        help=(
            "Columns of the darfix map to study, as 'c0,c1' start,end map pixels — relative "
            "to the darfix window, NOT absolute detector pixels. Pre-fills every stage's "
            "map-frame ROI X and (with the darfix ROI) rocking's detector crop. "
            "Blank = full width."
        ),
    ),
    Param(
        "analysis_roi_y",
        ParamType.STR,
        "Analysis window Y (map px)",
        help=(
            "Rows of the darfix map to study, as 'r0,r1' start,end map pixels — relative "
            "to the darfix window, NOT absolute detector pixels. Pre-fills every stage's "
            "map-frame ROI Y and (with the darfix ROI) rocking's detector crop. "
            "Blank = full height."
        ),
    ),
```

In `experiments/STO2_overnight.yaml`, after the `pixel_size_y_um` line add:

```yaml
darfix_roi: 105,230,1832,1266
analysis_roi_x: 0,1832
analysis_roi_y: 400,1100
```

(YAML reads these as plain strings; the round-trip test confirms.)

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_config.py tests/test_param_metadata.py -q`
Expected: all pass (schema-sync, help-present, STO2 pins, round-trip).

- [ ] **Step 5: Docs**

- `docs/Usage.md` → `### Experiment presets`: document the three new fields — darfix ROI is copied verbatim as origin+size; analysis windows are map-frame start,end; both pre-fill the stages (full explanation lands in Task 8's concept subsection).
- `docs/Codebase.md` → `### dfxm/config — typed config & presets`: add the three fields to the Experiment description.

- [ ] **Step 6: Commit**

```bash
git add dfxm/config/models.py experiments/STO2_overnight.yaml tests/test_config.py docs/Usage.md docs/Codebase.md
git commit -m "feat(config): experiment-level darfix ROI (origin+size) + map-frame analysis window"
```

---

### Task 3: `Param.roi_frame` + frame-honest labels/help sweep

**Files:**
- Modify: `dfxm/config/models.py` (Param dataclass ~lines 52–72)
- Modify: `dfxm/stages/rocking.py` (~lines 205–228), `dfxm/stages/visualize.py` (~lines 170–193), `dfxm/stages/paraview.py` (~lines 163–186), `dfxm/stages/slices.py` (~lines 254–281), `dfxm/stages/strain.py` (~lines 170–182)
- Test: `tests/test_param_metadata.py`
- Modify: `docs/Usage.md` (stage-reference ROI paragraphs), `docs/Codebase.md` (Param description in `### dfxm/config`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `Param.roi_frame: str` (`""|"detector"|"map"`) — Task 7 uses `p.roi_group or p.roi_frame` to find markable ROI fields, so **rocking's `roi_x`/`roi_y` MUST get `roi_frame="detector"`** here.

- [ ] **Step 1: Write the failing tests**

Read `tests/test_param_metadata.py` in full first. Extend `test_roi_fields_default_empty` with `assert p.roi_frame == ""`, and add:

```python
def test_roi_frame_validated():
    Param("roi_x", ParamType.STR, "ROI x", roi_frame="detector")  # ok, no group needed
    with pytest.raises(ValueError):
        Param("roi_x", ParamType.STR, "ROI x", roi_frame="galactic")


def test_roi_params_declare_frame():
    """Every ROI param states its coordinate frame — and says so in its help."""
    from gui.bindings import STAGE_SPECS

    for stage_name, spec in STAGE_SPECS.items():
        for p in spec.params:
            if not (p.roi_group or p.roi_frame):
                continue
            assert p.roi_frame in ("detector", "map"), f"{stage_name}.{p.name}: no roi_frame"
            assert p.roi_frame in (p.help or "").lower(), (
                f"{stage_name}.{p.name}: help must state its '{p.roi_frame}' frame"
            )


def test_rocking_roi_params_are_detector_frame():
    from dfxm.stages import rocking

    assert rocking.STAGE.get("roi_x").roi_frame == "detector"
    assert rocking.STAGE.get("roi_y").roi_frame == "detector"
```

(If the module has no `pytest`/`Param` imports at top level yet, follow its existing import style.)

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_param_metadata.py -q`
Expected: FAIL — `Param` has no `roi_frame`, then missing declarations per stage.

- [ ] **Step 3: Implement `Param.roi_frame`**

In `dfxm/config/models.py` add after `roi_axis`:

```python
    roi_frame: str = ""  # "" | "detector" | "map" — the coordinate frame of a ROI param
```

and in `__post_init__`:

```python
        if self.roi_frame not in ("", "detector", "map"):
            raise ValueError(f"roi param {self.name!r}: bad roi_frame {self.roi_frame!r}")
```

- [ ] **Step 4: Sweep the five stages** (Read each Param block first; exact current texts are in the files at the line ranges above)

- `dfxm/stages/rocking.py` `roi_x`/`roi_y`: add `roi_frame="detector"`, and append to each help string: `" Pre-filled from the experiment's darfix + analysis ROIs — normally leave as-is."` (keep the existing origin+size warning text).
- `dfxm/stages/visualize.py` and `dfxm/stages/paraview.py` `roi_x`/`roi_y`: add `roi_frame="map"`, change labels to `"Map ROI X"` / `"Map ROI Y"`, and replace the help (both stages currently claim "detector X/Y", which is wrong) with:
  - roi_x: `"Crop along map X as 'c0,c1' map pixels — columns of the darfix map, relative to the darfix window, NOT absolute detector pixels (blank = full width). Pre-filled from the experiment's analysis window. All volumes must share the same crop to stay co-registered."`
  - roi_y: same with rows/'r0,r1'/full height.
- `dfxm/stages/slices.py` `align_roi_x`/`align_roi_y`: add `roi_frame="map"`, help → `"Map-frame crop 'c0,c1' (map pixels, relative to the darfix window) used during alignment — must match the crop used when the volumes were rendered/exported. Pre-filled from the experiment's analysis window."` (rows variant for y).
- `dfxm/stages/strain.py` `roi`: add `roi_frame="map"`, help → `"Region of interest as 'r0,r1,c0,c1' in map pixels (rows then columns, relative to the darfix window; blank = full image). Pre-filled from the experiment's analysis window. Cropped after detrending, so the trend fit always uses the full map — that order is a physics constraint."`

- [ ] **Step 5: Run the suite**

Run: `python3 -m pytest tests/test_param_metadata.py tests/test_config.py -q` then `python3 -m pytest -q`
Expected: all pass (labels/help are metadata-only; no behaviour change).

- [ ] **Step 6: Docs**

- `docs/Usage.md` stage reference: update the ROI wording in sections 2 (strain), 4 (rocking), 5 (visualize), 6 (paraview), 7 (slices) to the new labels and frame statements (grep each section for "ROI" first).
- `docs/Codebase.md` `### dfxm/config`: add `roi_frame` to the Param field description.

- [ ] **Step 7: Commit**

```bash
git add dfxm/config/models.py dfxm/stages/rocking.py dfxm/stages/visualize.py dfxm/stages/paraview.py dfxm/stages/slices.py dfxm/stages/strain.py tests/test_param_metadata.py docs/Usage.md docs/Codebase.md
git commit -m "feat(schema): Param.roi_frame + frame-honest ROI labels/help across five stages"
```

---

### Task 4: bindings pre-fill — every stage in its native frame

**Files:**
- Modify: `gui/bindings.py`
- Test: create `tests/test_bindings_roi.py`
- Modify: `docs/Codebase.md` (`## Layer 2` bindings paragraph), `docs/Usage.md` (Shared project state & auto-chaining section: ROI pre-fill sentence)

**Interfaces:**
- Consumes: `Experiment.darfix_roi/analysis_roi_x/analysis_roi_y` (Task 2); `analysis_detector_window`, `format_pair`, `parse_pair` from `dfxm.common.roi` (Task 1).
- Produces: `experiment_overrides(stage_name, exp)` now may include `roi_x`/`roi_y` (rocking: detector frame; visualize/paraview: map frame), `align_roi_x`/`align_roi_y` (slices), `roi` (strain). Keys are **omitted** (not blank) when underivable — Tasks 5–7 rely on this.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bindings_roi.py`:

```python
"""experiment_overrides derives each stage's ROI fields in its native frame."""

from __future__ import annotations

from dfxm.config.models import Experiment
from gui.bindings import experiment_overrides

STO2_ROIS = dict(
    darfix_roi="105,230,1832,1266", analysis_roi_x="0,1832", analysis_roi_y="400,1100"
)

ROI_KEYS = ("roi_x", "roi_y", "align_roi_x", "align_roi_y", "roi")


def test_rocking_gets_absolute_detector_window():
    ov = experiment_overrides("rocking", Experiment(**STO2_ROIS))
    assert ov["roi_x"] == "105,1937"
    assert ov["roi_y"] == "630,1330"  # the incident's hand-conversion, automated


def test_map_stages_get_map_frame_values():
    exp = Experiment(**STO2_ROIS)
    for stage in ("visualize", "paraview"):
        ov = experiment_overrides(stage, exp)
        assert ov["roi_x"] == "0,1832" and ov["roi_y"] == "400,1100"
    ov = experiment_overrides("slices", exp)
    assert ov["align_roi_x"] == "0,1832" and ov["align_roi_y"] == "400,1100"
    assert experiment_overrides("strain", exp)["roi"] == "400,1100,0,1832"


def test_blank_rois_prefill_nothing():
    exp = Experiment()
    for stage in ("rocking", "visualize", "paraview", "slices", "strain"):
        ov = experiment_overrides(stage, exp)
        assert not any(k in ov for k in ROI_KEYS), (stage, ov)


def test_partial_analysis_falls_back_to_full_window():
    exp = Experiment(darfix_roi="105,230,1832,1266", analysis_roi_y="400,1100")
    ov = experiment_overrides("rocking", exp)
    assert ov["roi_x"] == "105,1937"  # blank X -> full darfix width
    assert ov["roi_y"] == "630,1330"
    assert "align_roi_x" not in experiment_overrides("slices", exp)
    assert "roi" not in experiment_overrides("strain", exp)  # strain needs both axes


def test_analysis_without_darfix_fills_map_stages_only():
    exp = Experiment(analysis_roi_y="400,1100")
    assert "roi_y" not in experiment_overrides("rocking", exp)  # underivable
    assert experiment_overrides("slices", exp)["align_roi_y"] == "400,1100"


def test_malformed_preset_prefills_nothing():
    exp = Experiment(darfix_roi="banana", analysis_roi_y="400,1100")
    ov = experiment_overrides("rocking", exp)
    assert not any(k in ov for k in ROI_KEYS)


def test_existing_overrides_untouched():
    ov = experiment_overrides("rocking", Experiment(**STO2_ROIS, raw_root="/r"))
    assert ov["raw_root"] == "/r"  # ROI merge does not clobber the base dict
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_bindings_roi.py -q`
Expected: FAIL — no ROI keys in the overrides.

- [ ] **Step 3: Implement**

In `gui/bindings.py`: add import `from dfxm.common.roi import analysis_detector_window, format_pair, parse_pair`. Rename the existing `def experiment_overrides(` to `def _base_overrides(` (docstring stays), then add:

```python
def _roi_overrides(stage_name: str, exp: Experiment) -> dict:
    """ROI pre-fill for *stage_name*, each stage in its native frame.

    Only derivable values are returned (keys omitted otherwise), so a preset
    without ROIs — or a malformed one, which the experiment editor flags —
    leaves every stage form exactly as before.
    """
    ax = (exp.analysis_roi_x or "").strip()
    ay = (exp.analysis_roi_y or "").strip()
    out: dict = {}
    if stage_name == "rocking":
        try:
            det_x, det_y = analysis_detector_window(exp.darfix_roi, ax, ay)
        except ValueError:
            return {}
        if det_x:
            out["roi_x"] = format_pair(det_x)
        if det_y:
            out["roi_y"] = format_pair(det_y)
    elif stage_name in ("visualize", "paraview"):
        if ax:
            out["roi_x"] = ax
        if ay:
            out["roi_y"] = ay
    elif stage_name == "slices":
        if ax:
            out["align_roi_x"] = ax
        if ay:
            out["align_roi_y"] = ay
    elif stage_name == "strain" and ax and ay:
        try:
            (c0, c1), (r0, r1) = parse_pair(ax), parse_pair(ay)
        except ValueError:
            return {}
        out["roi"] = f"{r0},{r1},{c0},{c1}"
    return out


def experiment_overrides(stage_name: str, exp: Experiment) -> dict:
    """Experiment-derived defaults that pre-fill *stage_name*'s form."""
    out = _base_overrides(stage_name, exp)
    out.update(_roi_overrides(stage_name, exp))
    return out
```

Note for the map stages: a malformed `analysis_roi_*` string passes through verbatim (the stage's own `_parse_pair`/`StageUserError` handles it at run time, and the editor blocks saving such presets) — only rocking/strain, which must *convert*, guard with try/except.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_bindings_roi.py -q` then `python3 -m pytest -q`
Expected: all pass. `tests/test_gui_stage_view_persistence.py` exercises `experiment_overrides` — it must stay green (the default `Experiment()` derives nothing, so existing expectations hold).

- [ ] **Step 5: Docs**

- `docs/Usage.md` → `### Shared project state & auto-chaining`: add that the experiment's ROI fields pre-fill every stage's crop in that stage's own frame (rocking in detector pixels).
- `docs/Codebase.md` → gui/bindings paragraph: mention `_roi_overrides` and the native-frame derivation.

- [ ] **Step 6: Commit**

```bash
git add gui/bindings.py tests/test_bindings_roi.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): experiment ROIs pre-fill every stage in its native frame"
```

---

### Task 5: Experiment editor — derived read-out + validation on OK/Save

**Files:**
- Modify: `gui/experiment_panel.py` (ExperimentDialog `__init__` ~lines 39–77, `_on_save_as` ~line 82)
- Test: create `tests/test_gui_experiment_roi.py`
- Modify: `docs/Usage.md` (Experiment presets section), `docs/Codebase.md` (gui section: ExperimentDialog)

**Interfaces:**
- Consumes: `dfxm.common.roi` (Task 1); the three schema fields render automatically via `ParamForm(EXPERIMENT_SCHEMA, …)` (Task 2 — no form work needed).
- Produces: `ExperimentDialog._roi_note` (QLabel), `ExperimentDialog._roi_problems() -> list[str]`, `ExperimentDialog._on_accept()` — reused by Task 6's test and Task 8's smoke step.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_experiment_roi.py`:

```python
"""ExperimentDialog: ROI derived read-out + validation on accept/save."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402

STO2 = dict(darfix_roi="105,230,1832,1266", analysis_roi_x="0,1832", analysis_roi_y="400,1100")


def _dlg(**fields):
    from dfxm.config.models import Experiment
    from gui.experiment_panel import ExperimentDialog

    _ = QApplication.instance() or QApplication([])
    return ExperimentDialog(Experiment(**fields))


def test_derived_readout_shows_both_frames():
    text = _dlg(**STO2)._roi_note.text()
    assert "x 105→1937" in text and "y 230→1496" in text  # darfix window, detector px
    assert "y 630→1330" in text  # analysis window translated to detector rows


def test_readout_blank_without_darfix():
    assert _dlg()._roi_note.text() == ""


def test_readout_tracks_edits():
    dlg = _dlg(darfix_roi="105,230,1832,1266")
    dlg._form.set_values({"analysis_roi_y": "400,1100"})
    assert "y 630→1330" in dlg._roi_note.text()


def test_readout_survives_malformed_input():
    dlg = _dlg(darfix_roi="105,230,1832,1266")
    dlg._form.set_values({"darfix_roi": "banana"})  # must not raise mid-typing
    assert "expected" in dlg._roi_note.text()


def test_accept_blocked_on_invalid(monkeypatch):
    dlg = _dlg(darfix_roi="105,230,1832,1266", analysis_roi_y="1100,400")
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dlg._on_accept()
    assert warned
    assert dlg.result() != QDialog.DialogCode.Accepted


def test_accept_passes_when_valid():
    dlg = _dlg(**STO2)
    dlg._on_accept()
    assert dlg.result() == QDialog.DialogCode.Accepted
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_gui_experiment_roi.py -q`
Expected: FAIL — no `_roi_note` / `_on_accept`.

- [ ] **Step 3: Implement** (Read `gui/experiment_panel.py` first)

In `ExperimentDialog.__init__`, change `buttons.accepted.connect(self.accept)` to `buttons.accepted.connect(self._on_accept)`, and before the layout block add:

```python
        self._roi_note = QLabel("")
        self._roi_note.setProperty("role", "muted")
        self._roi_note.setWordWrap(True)
        self._form.changed.connect(self._update_roi_note)
        self._update_roi_note()
```

In the layout, insert the note between the scroll area and the help panel:

```python
        layout.addWidget(scroll, 1)
        layout.addWidget(self._roi_note)
        layout.addWidget(help_panel)
        layout.addWidget(buttons)
```

Add the methods:

```python
    def _update_roi_note(self) -> None:
        """Live ROI read-out: darfix window + analysis window in detector px."""
        from dfxm.common import roi as R

        vals = self._form.values()
        try:
            win = R.parse_darfix_roi(vals.get("darfix_roi", ""))
            det_x, det_y = R.analysis_detector_window(
                vals.get("darfix_roi", ""),
                vals.get("analysis_roi_x", ""),
                vals.get("analysis_roi_y", ""),
            )
        except ValueError as exc:
            self._roi_note.setText(f"ROI: {exc}")
            return
        if win is None or det_x is None or det_y is None:
            self._roi_note.setText("")
            return
        self._roi_note.setText(
            f"Detector window: x {win.x0}→{win.x1}, y {win.y0}→{win.y1} · analysis in "
            f"detector px: x {det_x[0]}→{det_x[1]}, y {det_y[0]}→{det_y[1]}"
        )

    def _roi_problems(self) -> list[str]:
        from dfxm.common.roi import validate_rois

        vals = self._form.values()
        return validate_rois(
            vals.get("darfix_roi", ""),
            vals.get("analysis_roi_x", ""),
            vals.get("analysis_roi_y", ""),
        )

    def _warn_roi_problems(self) -> bool:
        """True (and a dialog shown) when the ROI fields are unsaveable."""
        problems = self._roi_problems()
        if not problems:
            return False
        QMessageBox.warning(
            self,
            "Regions of interest",
            "\n".join(problems)
            + "\n\nDarfix ROI is 'x,y,w,h' exactly as darfix shows it (origin+size); "
            "analysis windows are map-frame 'start,end' relative to that window.",
        )
        return True

    def _on_accept(self) -> None:
        if self._warn_roi_problems():
            return
        self.accept()
```

And at the top of `_on_save_as`, first line: `if self._warn_roi_problems(): return`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_gui_experiment_roi.py -q` then the GUI test batch `python3 -m pytest tests/test_gui_stage_view_persistence.py tests/test_gui_stage_view_roi.py -q`
Expected: all pass.

- [ ] **Step 5: Docs**

- `docs/Usage.md` → `### Experiment presets`: describe the read-out line ("the numbers you'd previously derive by hand are displayed, never typed") and that OK/Save-as validate the ROI fields.
- `docs/Codebase.md` → gui layer, ExperimentDialog entry: mention `_roi_note`/`_on_accept` validation.

- [ ] **Step 6: Commit**

```bash
git add gui/experiment_panel.py tests/test_gui_experiment_roi.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): experiment editor ROI read-out (both frames live) + save-time validation"
```

---

### Task 6: Experiment editor — Pick analysis ROI… button

**Files:**
- Modify: `gui/experiment_panel.py`
- Test: extend `tests/test_gui_experiment_roi.py`
- Modify: `docs/Usage.md` (Experiment presets), `docs/Codebase.md` (ExperimentDialog entry)

**Interfaces:**
- Consumes: `stacked_volume_previews(params: dict) -> list[(label, thunk)]` from `dfxm.common.figures` (exists); `ROIPickerDialog(previews, initial=None, parent=None)` with `.result = (r0, r1, c0, c1)` from `gui.widgets.roi_picker` (exists); `parse_pair` (Task 1); `_update_roi_note` fires automatically via `ParamForm.changed` (Task 5).
- Produces: `ExperimentDialog._on_pick_analysis_roi()`.

**Spec deviation (deliberate):** previews come from the *stacked volumes* (`stacked_volumes.h5` / `stacked_strain_volumes.h5` beside `processed_root`) rather than a per-layer `maps.h5` — same map frame, reuses `stacked_volume_previews` unchanged, no new HDF5 layout knowledge. Record this in the spec's Section 2 with one sentence when committing this task.

- [ ] **Step 1: Write the failing test** (append to `tests/test_gui_experiment_roi.py`)

```python
def test_pick_analysis_roi_writes_map_pairs(monkeypatch):
    import dfxm.common.figures as F
    import gui.widgets.roi_picker as RP

    dlg = _dlg(darfix_roi="105,230,1832,1266")
    monkeypatch.setattr(
        F, "stacked_volume_previews", lambda params: [("mosa", lambda: (None, 1.0, 1.0))]
    )

    class _FakePicker:
        def __init__(self, *a, **k):
            self.result = (400, 1100, 0, 1832)  # r0, r1, c0, c1

        def exec(self):
            return 1

    monkeypatch.setattr(RP, "ROIPickerDialog", _FakePicker)
    dlg._on_pick_analysis_roi()
    vals = dlg._form.values()
    assert vals["analysis_roi_x"] == "0,1832"
    assert vals["analysis_roi_y"] == "400,1100"
    assert "y 630→1330" in dlg._roi_note.text()  # read-out updated by the write-back
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_gui_experiment_roi.py -q`
Expected: FAIL — no `_on_pick_analysis_roi`.

- [ ] **Step 3: Implement**

In `ExperimentDialog.__init__`, next to the compute-pixel-size button:

```python
        roi_btn = QPushButton("Pick analysis ROI…")
        roi_btn.setToolTip(
            "Draw the analysis window on a stacked-volume layer — fills Analysis window X/Y"
        )
        roi_btn.clicked.connect(self._on_pick_analysis_roi)
        buttons.addButton(roi_btn, QDialogButtonBox.ButtonRole.ActionRole)
```

Add the methods (lazy imports — nothing matplotlib/h5py is built until the click, per house rule):

```python
    def _roi_preview_params(self, vals: dict) -> dict:
        proc = (vals.get("processed_root") or "").rstrip("/")
        return {
            "mosa_volume_file": os.path.join(proc, "stacked_volumes.h5") if proc else "",
            "strain_volume_file": os.path.join(proc, "stacked_strain_volumes.h5") if proc else "",
            "pixel_size_x_um": vals.get("pixel_size_x_um") or 1.0,
            "pixel_size_y_um": vals.get("pixel_size_y_um") or 1.0,
        }

    def _on_pick_analysis_roi(self) -> None:
        from dfxm.common.figures import stacked_volume_previews
        from dfxm.common.roi import parse_pair

        from .widgets.roi_picker import ROIPickerDialog

        vals = self._form.values()
        previews = stacked_volume_previews(self._roi_preview_params(vals))
        if not previews:
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Pick a stacked volume .h5 (map preview)",
                vals.get("processed_root") or "",
                "HDF5 (*.h5 *.hdf5)",
            )
            if not path:
                return
            params = self._roi_preview_params(vals)
            params["mosa_volume_file"] = path
            params["strain_volume_file"] = path
            previews = stacked_volume_previews(params)
        if not previews:
            QMessageBox.warning(
                self,
                "Pick analysis ROI",
                "No map preview available — run the mosaicity or strain stage first "
                "(needs stacked_volumes.h5 or stacked_strain_volumes.h5).",
            )
            return
        initial = None
        try:
            ax = parse_pair(vals.get("analysis_roi_x", ""))
            ay = parse_pair(vals.get("analysis_roi_y", ""))
            if ax and ay:
                initial = (ay[0], ay[1], ax[0], ax[1])  # picker wants r0, r1, c0, c1
        except ValueError:
            pass
        dlg = ROIPickerDialog(previews, initial=initial, parent=self)
        if dlg.exec() and dlg.result:
            r0, r1, c0, c1 = dlg.result
            self._form.set_values(
                {"analysis_roi_x": f"{c0},{c1}", "analysis_roi_y": f"{r0},{r1}"}
            )
```

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_gui_experiment_roi.py -q`
Expected: all pass.

- [ ] **Step 5: Docs + spec note**

- `docs/Usage.md` → Experiment presets: one paragraph on the Pick button (draws on a stacked-volume mid-layer; result is map-frame; needs mosaicity/strain output to exist — otherwise Browse to any stacked h5).
- `docs/Codebase.md`: extend the ExperimentDialog entry.
- `docs/superpowers/specs/2026-07-21-roi-unification-design.md` Section 2: one sentence recording the stacked-volumes preview source.

- [ ] **Step 6: Commit**

```bash
git add gui/experiment_panel.py tests/test_gui_experiment_roi.py docs/Usage.md docs/Codebase.md docs/superpowers/specs/2026-07-21-roi-unification-design.md
git commit -m "feat(gui): Pick analysis ROI… in the experiment editor (map-frame picker)"
```

---

### Task 7: Deviation markers on ROI stage fields

**Files:**
- Modify: `gui/widgets/param_form.py` (`__init__` ~line 68, `_label_for` ~lines 202–212)
- Modify: `gui/stage_view.py` (`__init__` after the form is built ~line 130; `set_experiment` ~lines 260–284)
- Test: extend `tests/test_gui_stage_view_roi.py`
- Modify: `docs/Usage.md` (The stage panel section), `docs/Codebase.md` (param_form + stage_view entries)

**Interfaces:**
- Consumes: `Param.roi_frame` (Task 3 — how rocking's fields, which have no `roi_group`, are found); `experiment_overrides` omitting underivable keys (Task 4).
- Produces: `ParamForm.set_field_marker(name: str, marked: bool, tooltip: str = "") -> None`; `StageView._update_roi_markers()`; `ParamForm._labels: dict[str, QLabel]` (used by tests).

- [ ] **Step 1: Write the failing tests** (append to `tests/test_gui_stage_view_roi.py`; its offscreen/QApplication preamble already exists)

```python
def _exp_sto2_rois():
    from dfxm.config.models import Experiment

    return Experiment(
        darfix_roi="105,230,1832,1266", analysis_roi_x="0,1832", analysis_roi_y="400,1100"
    )


def test_roi_deviation_marker_toggles():
    from dfxm.stages import visualize
    from gui.stage_view import StageView

    _ = QApplication.instance() or QApplication([])
    view = StageView("visualize", visualize.STAGE, _exp_sto2_rois())
    lbl = view._form._labels["roi_x"]
    assert "⚠" not in lbl.text()  # pre-filled value matches the experiment
    view._form.set_values({"roi_x": "1,2"})
    assert "⚠" in lbl.text()
    assert "0,1832" in lbl.toolTip()  # tooltip names the experiment value
    view._form.set_values({"roi_x": "0,1832"})
    assert "⚠" not in lbl.text()


def test_rocking_marker_catches_the_incident_entry():
    from dfxm.stages import rocking
    from gui.stage_view import StageView

    _ = QApplication.instance() or QApplication([])
    view = StageView("rocking", rocking.STAGE, _exp_sto2_rois())
    assert view._form.values()["roi_y"] == "630,1330"  # pre-filled, detector frame
    view._form.set_values({"roi_y": "230,1266"})  # darfix origin+size — the classic mistake
    assert "⚠" in view._form._labels["roi_y"].text()


def test_no_marker_without_experiment_rois():
    from dfxm.config.models import Experiment
    from dfxm.stages import visualize
    from gui.stage_view import StageView

    _ = QApplication.instance() or QApplication([])
    view = StageView("visualize", visualize.STAGE, Experiment())
    view._form.set_values({"roi_x": "1,2"})
    assert "⚠" not in view._form._labels["roi_x"].text()  # nothing to deviate from
```

- [ ] **Step 2: Run to verify failure**

Run: `python3 -m pytest tests/test_gui_stage_view_roi.py -q`
Expected: FAIL — `ParamForm` has no `_labels`.

- [ ] **Step 3: Implement `ParamForm.set_field_marker`** (Read `gui/widgets/param_form.py` first)

In `__init__`, next to the other dict initialisers (before the row-building loop):

```python
        self._labels: dict[str, QLabel] = {}
        self._base_label: dict[str, str] = {}
```

In `_label_for`, before `return lbl`:

```python
        self._labels[p.name] = lbl
        self._base_label[p.name] = text
```

New public method after `reset_values`:

```python
    def set_field_marker(self, name: str, marked: bool, tooltip: str = "") -> None:
        """Toggle a '⚠' suffix on *name*'s row label (deviates-from-experiment)."""
        lbl = self._labels.get(name)
        p = self._param_by_name.get(name)
        if lbl is None or p is None:
            return
        base = self._base_label[name]
        lbl.setText(f"{base}  ⚠" if marked else base)
        lbl.setToolTip(tooltip if (marked and tooltip) else param_help_html(p))
```

- [ ] **Step 4: Implement StageView wiring** (Read the target regions of `gui/stage_view.py` first)

In `__init__`, right after the ROI Pick-button block (~line 162, after `self._roi_buttons[_grp] = _btn` loop ends):

```python
        # ROI fields: mark any value that deviates from the experiment-derived one
        self._roi_param_names = tuple(p.name for p in spec.params if p.roi_group or p.roi_frame)
        if self._roi_param_names:
            self._form.changed.connect(self._update_roi_markers)
            self._update_roi_markers()
```

At the end of `set_experiment` — in **both** branches (after the early-return `set_values` call, and after `self._dirty = False`):

```python
        if self._roi_param_names:
            self._update_roi_markers()
```

New method next to `_on_pick_roi_group`:

```python
    def _update_roi_markers(self) -> None:
        """Flag ROI fields whose value differs from the experiment-derived one."""
        expected = experiment_overrides(self._stage_name, self._experiment)
        vals = self._form.values()
        for name in self._roi_param_names:
            want = str(expected.get(name, "") or "")
            have = str(vals.get(name, "") or "")
            deviates = bool(want) and have != want
            self._form.set_field_marker(
                name, deviates, f"differs from experiment: {want}" if deviates else ""
            )
```

(`experiment_overrides` is already imported at the top of `stage_view.py`. It is string-ops-only, so calling it on every form edit is cheap.)

- [ ] **Step 5: Run tests**

Run: `python3 -m pytest tests/test_gui_stage_view_roi.py tests/test_gui_stage_view_persistence.py -q` then `python3 -m pytest -q`
Expected: all pass — including persistence tests (markers touch only labels, never values, so no dirty-flag interaction; `set_field_marker` is called from the `changed` signal which persistence already tolerates).

- [ ] **Step 6: Docs**

- `docs/Usage.md` → `### The stage panel`: one paragraph — ROI fields pre-fill from the experiment; a ⚠ on the label means the value differs from the experiment-derived one (tooltip shows it); that's fine for deliberate overrides, suspicious otherwise.
- `docs/Codebase.md` → `gui/widgets/` param_form entry (+ stage_view entry): `set_field_marker` / `_update_roi_markers`.

- [ ] **Step 7: Commit**

```bash
git add gui/widgets/param_form.py gui/stage_view.py tests/test_gui_stage_view_roi.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): ⚠ deviation markers on ROI fields that differ from the experiment"
```

---

### Task 8: "ROI frames" user docs + smoke step + final verification

**Files:**
- Modify: `docs/Usage.md` (new subsection under Core concepts + Contents entry)
- Modify: `tests/gui_smoke.py` (new step `[34]` before the final `print("\nGUI SMOKE PASSED")`)

**Interfaces:**
- Consumes: `ExperimentDialog._roi_note` / `._roi_problems()` (Task 5).
- Produces: nothing downstream — this closes the branch.

- [ ] **Step 1: Write the smoke step** (Read the end of `tests/gui_smoke.py` first; follow the `[33]` step's local-alias style)

```python
    # [34] Experiment editor ROI: derived read-out translates map -> detector px
    # and validation catches an inverted analysis pair.
    from dfxm.config.models import Experiment as _Exp34

    from gui.experiment_panel import ExperimentDialog as _ED34

    _dlg34 = _ED34(
        _Exp34(darfix_roi="105,230,1832,1266", analysis_roi_x="0,1832", analysis_roi_y="400,1100")
    )
    assert "y 630→1330" in _dlg34._roi_note.text()
    _dlg34._form.set_values({"analysis_roi_y": "0,700"})
    assert "y 230→930" in _dlg34._roi_note.text()  # read-out is live
    assert not _dlg34._roi_problems()
    _dlg34._form.set_values({"analysis_roi_y": "1100,400"})
    assert _dlg34._roi_problems()  # end <= start is unsaveable
    print("[34] experiment editor ROI: derived read-out live + validation blocks bad pairs")
```

- [ ] **Step 2: Run the smoke test**

Run: `python3 tests/gui_smoke.py`
Expected: steps [1]–[34] print, ends `GUI SMOKE PASSED`.

- [ ] **Step 3: Write the Usage concept section**

In `docs/Usage.md`, after `### Experiment presets` (Read the surrounding sections first; match tone), add `### Regions of interest — two windows, two frames` and register it in `## Contents`. Content requirements:

- The **darfix window** (origin+size, e.g. `105,230,1832,1266`) is a *fact*: the detector crop darfix used. Copy it verbatim from darfix's ROI widget.
- The **analysis window** (map-frame start,end, e.g. y `400,1100`) is a *choice*: the sub-region you study. Map pixel 0 sits at the darfix origin, so detector row = darfix origin_y + map row.
- Worked STO2 example table: darfix y 230→1496 (1266 rows); analysis y 400,1100 (map) = 630,1330 (detector); which stage field receives which number (rocking `roi_x/roi_y` detector; visualize/paraview Map ROI, slices Align ROI, strain `roi` map-frame).
- Enter both once in the experiment editor; every stage pre-fills; ⚠ marks deviations; the slices Y-height note remains the last-line guard.
- Explicitly name the classic mistake (typing darfix's origin+size as start,end) and state it is now displayed, never typed.

- [ ] **Step 4: Full verification**

Run (all three must be green):
- `python3 -m pytest -q` — expected: baseline + ~25 new tests, 0 failures, 13 skips.
- `ruff check .` — expected: clean.
- `python3 tests/gui_smoke.py` — expected: `GUI SMOKE PASSED` with [1]–[34].

- [ ] **Step 5: Commit**

```bash
git add docs/Usage.md tests/gui_smoke.py
git commit -m "docs(usage): ROI frames concept section + smoke [34] (experiment editor ROI)"
```

---

## Out of scope (explicit, from the approved spec)

- paraview `mosa_darfix_origin_xy` / `strain_darfix_origin_xy`: completely untouched (no pre-fill, no default change).
- Replot dialogs' pixel-index crops: untouched.
- Picking the darfix window itself from a raw detector frame: not built (typed only).

## Merge

After Task 8 is green and reviewed: merge `roi-unification` into `master` with `--no-ff` (no remote — nothing to push), then run the suite once more on master. Update the `roi-unification-wish` memory to DONE with the merge commit and rollback point.
