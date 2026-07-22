# Experiment Initializer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An "Initialize from data…" button in the Edit-experiment dialog that detects
folder patterns, entry suffix, pixel sizes, ccmth reference and darfix-ROI size from
the data on disk, presents them in a review table, applies checked rows into the form,
and offers to save the preset YAML.

**Architecture:** Qt-free detection core (`dfxm/config/detect.py`: per-detector
functions + `detect_experiment` orchestrator + CLI) feeding a thin GUI review dialog
(`gui/widgets/detect_review.py`) wired into `ExperimentDialog`. Spec:
`docs/superpowers/specs/2026-07-22-experiment-initializer-design.md`.

**Tech Stack:** Python 3.10, h5py, numpy, PySide6 (GUI layer only), pytest
(offscreen Qt tests).

## Global Constraints

- `dfxm/` stays Qt-free — no PySide6 imports in `dfxm/config/detect.py`.
- Ruff: line length 100, double quotes, target py310, rules E/F/I (auto-format hook
  runs on Write/Edit).
- Detectors never raise for "not there yet" — a missing input becomes a
  `Detection` with `error` set (same skip-with-reason style as the stages).
  `StageUserError` from `compute_pixel_size` is caught, never propagated.
- `compute_pixel_size` and `dfxm/common/roi.py` are consumed as-is — no changes.
- Docs are part of each change: Task 3 updates `docs/Codebase.md` (new module);
  Task 5 updates `docs/Usage.md` + `docs/Codebase.md` (GUI behaviour) in the SAME
  commit as the code.
- Qt test files start with `os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")`
  before PySide6 imports, and `pytest.importorskip("PySide6")`.
- Existing APIs used (verified): `Param(name, type, label, ..., help=...)` with
  `.label`; `ParamForm.values()/set_values(dict)`; `Experiment.to_dict()/from_dict`;
  `parse_darfix_roi(text) -> DarfixWindow(origin_x, origin_y, width, height) | None`
  (raises `ValueError` on malformed); `find_matching_folders(root, pattern)` →
  natural-sorted full dir paths; `get_filtered_entries(h5f, suffix)`;
  `read_positioners(h5f, group_path)`; `StageUserError(message, hint=...)` with
  `.hint`.
- The GUI smoke test is `tests/gui_smoke.py` (not pytest; run
  `python3 tests/gui_smoke.py`).
- Full suite gate per task: `python3 -m pytest -q` and `ruff check .`.

---

### Task 1: Detection scaffolding — patterns, scan file, entry suffix

**Files:**
- Create: `dfxm/config/detect.py`
- Test: `tests/test_config_detect.py`

**Interfaces:**
- Consumes: nothing new (stdlib + h5py).
- Produces (used by Tasks 2–5):
  - `@dataclass(frozen=True) Detection(field: str, value: Any | None = None, note: str = "", error: str | None = None)` — `value is None and error is None` = info-only row.
  - `folder_families(raw_root: str) -> dict[str, int]` (stem → folder count)
  - `detect_patterns(raw_root: str) -> list[Detection]` (fields `folder_pattern`, `mosa_pattern`, `rocking_pattern`)
  - `select_scan_file(folder: str) -> str | None`
  - `detect_entry_suffix(scan_h5: str) -> Detection` (field `entry_suffix`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config_detect.py`:

```python
"""Tests for dfxm.config.detect (data-driven experiment initialization)."""

from __future__ import annotations

import h5py

from dfxm.config.detect import (
    Detection,
    detect_entry_suffix,
    detect_patterns,
    folder_families,
    select_scan_file,
)


def _mkdirs(root, *names):
    for n in names:
        (root / n).mkdir(parents=True)


def _by_field(rows: list[Detection]) -> dict[str, Detection]:
    return {d.field: d for d in rows}


# -- folder families / patterns -----------------------------------------------


def test_folder_families_groups_and_counts(tmp_path):
    _mkdirs(tmp_path, "s_strain__0", "s_strain__1", "s_mosa__0", "loose_folder")
    (tmp_path / "s_strain__2").write_text("")  # a FILE — must be ignored
    fams = folder_families(str(tmp_path))
    assert fams == {"s_strain": 2, "s_mosa": 1}


def test_folder_families_missing_root_is_empty():
    assert folder_families("/nonexistent/nowhere") == {}


def test_detect_patterns_classifies_families(tmp_path):
    _mkdirs(
        tmp_path,
        "s_energy_strain__0",
        "s_energy_strain__1",
        "s_energy_strain__2",
        "s_mosa__0",
        "s_mosa__1",
        "s_rocking__0",
    )
    rows = _by_field(detect_patterns(str(tmp_path)))
    assert rows["folder_pattern"].value == "s_energy_strain__*"
    assert rows["mosa_pattern"].value == "s_mosa__*"
    assert rows["rocking_pattern"].value == "s_rocking__*"
    assert "3 folders" in rows["folder_pattern"].note


def test_detect_patterns_missing_family_skips_with_reason(tmp_path):
    _mkdirs(tmp_path, "s_strain__0")
    rows = _by_field(detect_patterns(str(tmp_path)))
    assert rows["folder_pattern"].value == "s_strain__*"
    assert rows["mosa_pattern"].value is None and "mosa" in rows["mosa_pattern"].error
    assert rows["rocking_pattern"].value is None


def test_detect_patterns_no_families_at_all(tmp_path):
    _mkdirs(tmp_path, "no_numeric_suffix")
    rows = detect_patterns(str(tmp_path))
    assert len(rows) == 1
    assert rows[0].field == "folder_pattern" and rows[0].error


def test_detect_patterns_largest_family_wins(tmp_path):
    _mkdirs(tmp_path, "a__0", "b__0", "b__1")
    rows = _by_field(detect_patterns(str(tmp_path)))
    assert rows["folder_pattern"].value == "b__*"


# -- scan file selection ------------------------------------------------------


def test_select_scan_file_prefers_folder_name(tmp_path):
    d = tmp_path / "layer__0"
    d.mkdir()
    (d / "aaa_first_alphabetically.h5").write_text("")
    (d / "layer__0.h5").write_text("")
    assert select_scan_file(str(d)).endswith("layer__0.h5")


def test_select_scan_file_excludes_concat(tmp_path):
    d = tmp_path / "layer__0"
    d.mkdir()
    (d / "layer__0_concat.h5").write_text("")
    (d / "other_scan.h5").write_text("")
    assert select_scan_file(str(d)).endswith("other_scan.h5")


def test_select_scan_file_none_when_only_concat(tmp_path):
    d = tmp_path / "layer__0"
    d.mkdir()
    (d / "layer__0_concat.h5").write_text("")
    assert select_scan_file(str(d)) is None


# -- entry suffix -------------------------------------------------------------


def _write_entries(path, *entries):
    with h5py.File(path, "w") as f:
        for e in entries:
            f.create_group(e)
    return str(path)


def test_detect_entry_suffix_majority(tmp_path):
    p = _write_entries(tmp_path / "s.h5", "1.1", "2.1", "3.1", "2.2")
    d = detect_entry_suffix(p)
    assert d.value == ".1"
    assert "mixed" in d.note  # the minority .2 is called out


def test_detect_entry_suffix_clean(tmp_path):
    p = _write_entries(tmp_path / "s.h5", "1.1", "2.1")
    d = detect_entry_suffix(p)
    assert d.value == ".1" and "mixed" not in d.note


def test_detect_entry_suffix_no_entries(tmp_path):
    p = _write_entries(tmp_path / "s.h5", "not_an_entry")
    d = detect_entry_suffix(p)
    assert d.value is None and d.error


def test_detect_entry_suffix_unreadable_file(tmp_path):
    p = tmp_path / "junk.h5"
    p.write_text("this is not hdf5")
    d = detect_entry_suffix(str(p))
    assert d.value is None and d.error
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_config_detect.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'dfxm.config.detect'`

- [ ] **Step 3: Write the implementation**

Create `dfxm/config/detect.py`:

```python
"""Detect experiment settings from the data on disk.

Qt-free. Each detector inspects the raw/processed trees and returns
:class:`Detection` rows — never raising for "not there yet": a missing
input becomes a row with ``error`` set (the same skip-with-reason style the
stages use), so one unreadable file cannot block the other detections.

The orchestrator :func:`detect_experiment` runs every detector against an
:class:`~dfxm.config.models.Experiment` (typically the live form values) and
is re-runnable: a first pass before darfix fills the raw-data facts
(patterns, entry suffix, pixel sizes, ccmth positioner fallback); run again
after darfix and the maps.h5-derived rows (ccmth COM median, darfix-ROI
size) appear. darfix records no ROI metadata, so only the crop *size* is
recoverable — the origin must still be typed from the darfix widget.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import h5py

_FAMILY_RE = re.compile(r"^(?P<stem>.+)__(?P<num>\d+)$")
_ENTRY_RE = re.compile(r"^\d+(\.\d+)$")


@dataclass(frozen=True)
class Detection:
    """One detected experiment value (or the reason there isn't one).

    ``value is None and error is None`` marks an info-only row (e.g.
    "darfix ROI size consistent") — nothing to apply, nothing wrong.
    """

    field: str
    value: Any | None = None
    note: str = ""
    error: str | None = None


# -- folder patterns ----------------------------------------------------------


def folder_families(raw_root: str) -> dict[str, int]:
    """Group ``<stem>__<N>`` subfolders of *raw_root*: stem -> folder count."""
    try:
        names = sorted(os.listdir(raw_root))
    except OSError:
        return {}
    fams: Counter[str] = Counter()
    for n in names:
        m = _FAMILY_RE.match(n)
        if m and os.path.isdir(os.path.join(raw_root, n)):
            fams[m.group("stem")] += 1
    return dict(fams)


def detect_patterns(raw_root: str) -> list[Detection]:
    """Suggest folder/mosa/rocking glob patterns from the folder families."""
    fams = folder_families(raw_root)
    if not fams:
        return [
            Detection(
                "folder_pattern",
                error=f"no '<name>__<N>' folder families under {raw_root or '(unset)'}",
            )
        ]
    out: list[Detection] = []
    remaining = dict(fams)

    def take(field: str, key: str) -> None:
        match = {s: c for s, c in remaining.items() if key in s.lower()}
        if not match:
            out.append(Detection(field, error=f"no folder family containing {key!r}"))
            return
        stem = max(match, key=match.__getitem__)  # largest family wins
        del remaining[stem]
        out.append(Detection(field, f"{stem}__*", f"{fams[stem]} folders"))

    take("mosa_pattern", "mosa")
    take("rocking_pattern", "rocking")
    if remaining:
        stem = max(remaining, key=remaining.__getitem__)
        out.insert(0, Detection("folder_pattern", f"{stem}__*", f"{fams[stem]} folders"))
    else:
        out.insert(
            0, Detection("folder_pattern", error="no folder family besides mosa/rocking")
        )
    return out


# -- scan file + entry suffix -------------------------------------------------


def select_scan_file(folder: str) -> str | None:
    """The raw scan .h5 inside a layer *folder* (concat output excluded).

    Prefers ``<folder name>.h5`` exactly (the BLISS convention); otherwise the
    first ``*.h5`` that is not a ``*_concat.h5``.
    """
    preferred = os.path.join(folder, os.path.basename(os.path.normpath(folder)) + ".h5")
    if os.path.isfile(preferred):
        return preferred
    try:
        names = sorted(os.listdir(folder))
    except OSError:
        return None
    for n in names:
        if n.endswith(".h5") and not n.endswith("_concat.h5"):
            return os.path.join(folder, n)
    return None


def detect_entry_suffix(scan_h5: str) -> Detection:
    """Majority BLISS entry suffix (``1.1`` -> ``.1``) among *scan_h5* entries."""
    try:
        with h5py.File(scan_h5, "r") as f:
            suffixes = Counter(m.group(1) for k in f.keys() if (m := _ENTRY_RE.match(k)))
    except OSError as exc:
        return Detection("entry_suffix", error=f"could not read {scan_h5}: {exc}")
    if not suffixes:
        return Detection(
            "entry_suffix", error=f"no '<n>.<m>' entries in {os.path.basename(scan_h5)}"
        )
    suffix, count = suffixes.most_common(1)[0]
    note = f"{count} entries in {os.path.basename(scan_h5)}"
    if len(suffixes) > 1:
        others = ", ".join(s for s in sorted(suffixes) if s != suffix)
        note += f" (mixed with {others} — majority wins)"
    return Detection("entry_suffix", suffix, note)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config_detect.py -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check dfxm/config/detect.py tests/test_config_detect.py
git add dfxm/config/detect.py tests/test_config_detect.py
git commit -m "feat(detect): Detection dataclass + pattern/scan-file/entry-suffix detectors"
```

---

### Task 2: Calibration detectors — pixel sizes, ccmth, darfix-ROI size

**Files:**
- Modify: `dfxm/config/detect.py` (append)
- Test: `tests/test_config_detect.py` (append)

**Interfaces:**
- Consumes: `Detection` (Task 1); `compute_pixel_size` (existing, untouched);
  `parse_darfix_roi`, `find_matching_folders`, `get_filtered_entries`,
  `read_positioners` (existing).
- Produces (used by Task 3):
  - `detect_pixel_sizes(scan_h5: str, positioners_path: str, entry_suffix: str) -> list[Detection]` (two rows: `pixel_size_x_um`, `pixel_size_y_um`; on failure both carry the same `error`)
  - `find_strain_maps(processed_root: str, pattern: str, maps_filename: str, ccmth_com_path: str) -> tuple[str, str] | None` (→ `(maps_path, folder_name)`)
  - `detect_ccmth_from_maps(maps_path: str, folder_name: str, ccmth_com_path: str) -> Detection`
  - `detect_ccmth_from_positioners(scan_h5: str, positioners_path: str, entry_suffix: str) -> Detection`
  - `detect_darfix_roi(maps_path: str, folder_name: str, ccmth_com_path: str, current_roi: str) -> Detection` — blank current → value `"?,?,{w},{h}"`; matching size → info row (`value None, error None`); mismatch → corrected `"x,y,w,h"` keeping the current origin.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_detect.py`:

```python
# -- calibration detectors ----------------------------------------------------

import numpy as np  # noqa: E402  (test-section import, keeps diffs local)

from dfxm.config.detect import (  # noqa: E402
    detect_ccmth_from_maps,
    detect_ccmth_from_positioners,
    detect_darfix_roi,
    detect_pixel_sizes,
    find_strain_maps,
)

CCMTH_COM = "/entry/ccmth/Center of mass/Center of mass"


def _write_scan(path, *, entry="1.1", ccmth=7.1, **extra):
    """Minimal BLISS scan: the five pixel-size motors + ccmth."""
    motors = dict(mainx=-5000.0, obx=273.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0)
    motors.update(extra)
    if ccmth is not None:
        motors["ccmth"] = ccmth
    with h5py.File(path, "w") as f:
        pos = f.create_group(f"{entry}/instrument/positioners")
        for name, val in motors.items():
            pos.create_dataset(name, data=val)
    return str(path)


def _write_maps(path, *, with_ccmth=True, shape=(6, 8), fill=7.5):
    with h5py.File(path, "w") as f:
        if with_ccmth:
            data = np.full(shape, fill)
            data[0, 0] = np.nan  # nanmedian must survive NaNs
            f.create_dataset(CCMTH_COM, data=data)
        else:
            f.create_dataset("/entry/chi/Center of mass/Center of mass", data=np.zeros(shape))
    return str(path)


def test_detect_pixel_sizes_success(tmp_path):
    p = _write_scan(tmp_path / "s.h5")
    rows = detect_pixel_sizes(p, "instrument/positioners", ".1")
    by = _by_field(rows)
    m = 5000.0 / 273.0 - 1.0
    assert by["pixel_size_x_um"].value == round(3.25 / m, 6)
    assert by["pixel_size_y_um"].value > by["pixel_size_x_um"].value  # sin(2θ) division
    assert "2x" in by["pixel_size_x_um"].note and "M=" in by["pixel_size_x_um"].note


def test_detect_pixel_sizes_user_error_becomes_rows(tmp_path):
    p = _write_scan(tmp_path / "s.h5", ffsel=-30.0)  # unrecognized objective
    rows = detect_pixel_sizes(p, "instrument/positioners", ".1")
    assert len(rows) == 2
    assert all(d.value is None and d.error for d in rows)
    assert "ffsel" in rows[0].error


def test_detect_pixel_sizes_unreadable_file(tmp_path):
    p = tmp_path / "junk.h5"
    p.write_text("nope")
    rows = detect_pixel_sizes(str(p), "instrument/positioners", ".1")
    assert all(d.error for d in rows)


def test_find_strain_maps_skips_mosa_style(tmp_path):
    proc = tmp_path / "proc"
    for name, ccm in (("s__0", False), ("s__1", True)):
        d = proc / name
        d.mkdir(parents=True)
        _write_maps(d / "maps.h5", with_ccmth=ccm)
    found = find_strain_maps(str(proc), "s__*", "maps.h5", CCMTH_COM)
    assert found is not None
    maps_path, folder = found
    assert folder == "s__1" and maps_path.endswith("s__1/maps.h5")


def test_find_strain_maps_none_when_absent(tmp_path):
    assert find_strain_maps(str(tmp_path), "s__*", "maps.h5", CCMTH_COM) is None
    assert find_strain_maps("", "s__*", "maps.h5", CCMTH_COM) is None


def test_detect_ccmth_from_maps_nanmedian(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", fill=7.1442)
    d = detect_ccmth_from_maps(p, "s__0", CCMTH_COM)
    assert d.field == "ccmth_ref_deg" and d.value == 7.1442
    assert "median" in d.note and "s__0" in d.note


def test_detect_ccmth_from_positioners(tmp_path):
    p = _write_scan(tmp_path / "s.h5", ccmth=7.144236)
    d = detect_ccmth_from_positioners(p, "instrument/positioners", ".1")
    assert d.value == 7.1442
    assert "confirm" in d.note  # flags itself as a snapshot needing confirmation


def test_detect_ccmth_from_positioners_missing_motor(tmp_path):
    p = _write_scan(tmp_path / "s.h5", ccmth=None)
    d = detect_ccmth_from_positioners(p, "instrument/positioners", ".1")
    assert d.value is None and "ccmth" in d.error


def test_detect_darfix_roi_blank_current_gives_partial(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", shape=(1266, 1832))
    d = detect_darfix_roi(p, "s__0", CCMTH_COM, "")
    assert d.value == "?,?,1832,1266"
    assert "origin" in d.note


def test_detect_darfix_roi_consistent_is_info_row(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", shape=(1266, 1832))
    d = detect_darfix_roi(p, "s__0", CCMTH_COM, "105,230,1832,1266")
    assert d.value is None and d.error is None
    assert "matches" in d.note


def test_detect_darfix_roi_mismatch_keeps_origin(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", shape=(1266, 1832))
    d = detect_darfix_roi(p, "s__0", CCMTH_COM, "105,230,999,999")
    assert d.value == "105,230,1832,1266"
    assert "not 999×999" in d.note


def test_detect_darfix_roi_malformed_current_treated_as_blank(tmp_path):
    p = _write_maps(tmp_path / "maps.h5", shape=(6, 8))
    d = detect_darfix_roi(p, "s__0", CCMTH_COM, "banana")
    assert d.value == "?,?,8,6"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_config_detect.py -q`
Expected: ImportError — `detect_pixel_sizes` etc. not defined (Task 1 tests still pass).

- [ ] **Step 3: Write the implementation**

Append to `dfxm/config/detect.py`:

```python
# -- calibration --------------------------------------------------------------


def detect_pixel_sizes(scan_h5: str, positioners_path: str, entry_suffix: str) -> list[Detection]:
    """Pixel sizes X/Y via :func:`~dfxm.common.pixel_size.compute_pixel_size`.

    Two rows sharing one geometry note; on failure both rows carry the same
    error so the review table stays honest about what is missing.
    """
    from dfxm.common.errors import StageUserError
    from dfxm.common.pixel_size import compute_pixel_size

    fields = ("pixel_size_x_um", "pixel_size_y_um")
    try:
        res = compute_pixel_size(
            scan_h5, positioners_path=positioners_path, entry_suffix=entry_suffix
        )
    except StageUserError as exc:
        err = f"{exc} — {exc.hint}" if exc.hint else str(exc)
        return [Detection(f, error=err) for f in fields]
    except Exception as exc:  # noqa: BLE001 — unreadable/foreign file must not block others
        return [
            Detection(f, error=f"could not read {os.path.basename(scan_h5)}: {exc}")
            for f in fields
        ]
    note = (
        f"M={res.magnification:.3f}, {res.objective} objective, "
        f"2θ={res.two_theta_deg:.3f}°, condenser {'in' if res.condenser_in else 'out'} — "
        f"{os.path.basename(scan_h5)}"
    )
    return [
        Detection("pixel_size_x_um", round(res.pixel_size_x_um, 6), note),
        Detection("pixel_size_y_um", round(res.pixel_size_y_um, 6), note),
    ]


def find_strain_maps(
    processed_root: str, pattern: str, maps_filename: str, ccmth_com_path: str
) -> tuple[str, str] | None:
    """First layer folder under *processed_root* whose maps file has the ccmth COM.

    Returns ``(maps_path, folder_name)`` or None. Mosa-family maps carry chi/mu
    only, so every candidate is probed for *ccmth_com_path* before acceptance.
    """
    from dfxm.common.sort import find_matching_folders

    if not processed_root or not os.path.isdir(processed_root):
        return None
    for folder in find_matching_folders(processed_root, pattern):
        maps_path = os.path.join(folder, maps_filename)
        if not os.path.isfile(maps_path):
            continue
        try:
            with h5py.File(maps_path, "r") as f:
                if ccmth_com_path in f:
                    return maps_path, os.path.basename(folder)
        except OSError:
            continue
    return None


def detect_ccmth_from_maps(maps_path: str, folder_name: str, ccmth_com_path: str) -> Detection:
    """ccmth reference suggestion: nanmedian of the darfix ccmth COM map."""
    import numpy as np

    try:
        with h5py.File(maps_path, "r") as f:
            com = f[ccmth_com_path][()]
        value = float(np.nanmedian(com))
    except (OSError, KeyError) as exc:
        return Detection(
            "ccmth_ref_deg", error=f"could not read ccmth COM from {folder_name}: {exc}"
        )
    return Detection("ccmth_ref_deg", round(value, 4), f"median of ccmth COM map, {folder_name}")


def detect_ccmth_from_positioners(
    scan_h5: str, positioners_path: str, entry_suffix: str
) -> Detection:
    """Fallback ccmth reference from the raw scan's motor snapshot (pre-darfix)."""
    import numpy as np

    from dfxm.common.h5io import get_filtered_entries, read_positioners

    try:
        with h5py.File(scan_h5, "r") as f:
            entries = get_filtered_entries(f, entry_suffix)
            if not entries:
                return Detection(
                    "ccmth_ref_deg",
                    error=f"no {entry_suffix!r} entries in {os.path.basename(scan_h5)}",
                )
            pos = read_positioners(f, f"{entries[0]}/{positioners_path}")
    except (OSError, KeyError) as exc:
        return Detection("ccmth_ref_deg", error=f"could not read positioners: {exc}")
    if "ccmth" not in pos:
        return Detection(
            "ccmth_ref_deg", error=f"no 'ccmth' motor in {os.path.basename(scan_h5)}"
        )
    value = float(np.asarray(pos["ccmth"]).reshape(-1)[0])
    return Detection(
        "ccmth_ref_deg",
        round(value, 4),
        "single motor snapshot — confirm against the beamline alignment",
    )


def detect_darfix_roi(
    maps_path: str, folder_name: str, ccmth_com_path: str, current_roi: str
) -> Detection:
    """Darfix-ROI row from the map shape: size only — darfix records no origin.

    Blank/malformed current -> an applicable-but-partial ``?,?,w,h`` value (the
    GUI keeps it uncheckable until the origin is typed). Filled current ->
    validation: matching size is an info row; a mismatch offers the corrected
    size with the existing origin kept.
    """
    from dfxm.common.roi import parse_darfix_roi

    try:
        with h5py.File(maps_path, "r") as f:
            h, w = f[ccmth_com_path].shape[:2]
    except (OSError, KeyError) as exc:
        return Detection(
            "darfix_roi", error=f"could not read map shape from {folder_name}: {exc}"
        )
    try:
        win = parse_darfix_roi(current_roi)
    except ValueError:
        win = None
    if win is None:
        return Detection(
            "darfix_roi",
            f"?,?,{w},{h}",
            f"map size {w}×{h} from {folder_name} — replace ?,? with the darfix origin",
        )
    if (win.width, win.height) == (w, h):
        return Detection("darfix_roi", None, f"✓ size matches maps.h5 ({w}×{h})")
    return Detection(
        "darfix_roi",
        f"{win.origin_x},{win.origin_y},{w},{h}",
        f"size in maps.h5 is {w}×{h}, not {win.width}×{win.height} — origin kept",
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config_detect.py -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check dfxm/config/detect.py tests/test_config_detect.py
git add dfxm/config/detect.py tests/test_config_detect.py
git commit -m "feat(detect): pixel-size/ccmth/darfix-ROI-size detectors"
```

---

### Task 3: Orchestrator + CLI + Codebase.md entry

**Files:**
- Modify: `dfxm/config/detect.py` (append)
- Modify: `docs/Codebase.md` (the `### dfxm/config — typed config & presets` section)
- Test: `tests/test_config_detect.py` (append)

**Interfaces:**
- Consumes: every detector from Tasks 1–2; `Experiment` (existing).
- Produces (used by Tasks 4–5):
  - `detect_experiment(current: Experiment) -> list[Detection]`
  - `main(argv: list[str] | None = None) -> int` (CLI: `python3 -m dfxm.config.detect <raw_root> [--processed-root DIR] [--maps-filename NAME]`)

**Orchestrator contract** (encode exactly):
1. `raw_root` unset or not a directory → single row `Detection("raw_root", error="set Raw data root to an existing folder first")`.
2. Pattern rows always emitted. The *working pattern* for scan/maps selection is `current.folder_pattern` when it is neither `""` nor `"*"`, else the detected `folder_pattern` value.
3. Scan file = `select_scan_file(first natural-sorted folder matching the working pattern)`. No scan → skip rows for `entry_suffix` + both pixel-size fields; the working suffix falls back to `current.entry_suffix or ".1"`.
4. With a scan: entry-suffix row, then pixel sizes using the *detected* suffix (fallback as above).
5. Maps: `find_strain_maps(current.processed_root, working pattern, current.maps_filename, current.ccmth_com_path)`. Found → ccmth-from-maps + darfix-ROI rows. Not found → ccmth-from-positioners (only if a scan exists) + a `darfix_roi` skip row saying to re-run after darfix.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_config_detect.py`:

```python
# -- orchestrator + CLI -------------------------------------------------------

from dfxm.config.detect import detect_experiment, main  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402


def _make_tree(tmp_path, *, with_maps=True):
    """Full synthetic experiment: raw families + scan, optional processed maps."""
    raw = tmp_path / "RAW"
    _mkdirs(raw, "s_strain__0", "s_strain__1", "s_mosa__0", "s_rocking__0")
    _write_scan(raw / "s_strain__0" / "s_strain__0.h5", ccmth=7.10)
    proc = tmp_path / "PROC"
    if with_maps:
        d = proc / "s_strain__0"
        d.mkdir(parents=True)
        _write_maps(d / "maps.h5", shape=(1266, 1832), fill=7.1442)
    else:
        proc.mkdir()
    return str(raw), str(proc)


def test_detect_experiment_full_pass(tmp_path):
    raw, proc = _make_tree(tmp_path)
    rows = _by_field(detect_experiment(Experiment(raw_root=raw, processed_root=proc)))
    assert rows["folder_pattern"].value == "s_strain__*"
    assert rows["entry_suffix"].value == ".1"
    assert rows["pixel_size_x_um"].value and rows["pixel_size_x_um"].error is None
    assert rows["ccmth_ref_deg"].value == 7.1442  # maps median wins over positioner 7.10
    assert "median" in rows["ccmth_ref_deg"].note
    assert rows["darfix_roi"].value == "?,?,1832,1266"


def test_detect_experiment_pre_darfix_falls_back(tmp_path):
    raw, proc = _make_tree(tmp_path, with_maps=False)
    rows = _by_field(detect_experiment(Experiment(raw_root=raw, processed_root=proc)))
    assert rows["ccmth_ref_deg"].value == 7.1  # positioner snapshot fallback
    assert "confirm" in rows["ccmth_ref_deg"].note
    assert rows["darfix_roi"].error and "re-run" in rows["darfix_roi"].error


def test_detect_experiment_explicit_pattern_wins(tmp_path):
    raw, proc = _make_tree(tmp_path)
    # user set a pattern that matches nothing -> scan-dependent rows skip
    exp = Experiment(raw_root=raw, processed_root=proc, folder_pattern="zzz__*")
    rows = _by_field(detect_experiment(exp))
    assert rows["folder_pattern"].value == "s_strain__*"  # suggestion still shown
    assert rows["pixel_size_x_um"].error  # but zzz__* found no scan


def test_detect_experiment_no_raw_root():
    rows = detect_experiment(Experiment(raw_root=""))
    assert len(rows) == 1 and rows[0].field == "raw_root" and rows[0].error


def test_detect_experiment_survives_bad_scan_file(tmp_path):
    raw = tmp_path / "RAW"
    _mkdirs(raw, "s_strain__0")
    (raw / "s_strain__0" / "s_strain__0.h5").write_text("not hdf5")
    rows = _by_field(detect_experiment(Experiment(raw_root=str(raw))))
    assert rows["folder_pattern"].value == "s_strain__*"  # patterns still detected
    assert rows["entry_suffix"].error and rows["pixel_size_x_um"].error


def test_cli_main_prints_table(tmp_path, capsys):
    raw, proc = _make_tree(tmp_path)
    assert main([raw, "--processed-root", proc]) == 0
    out = capsys.readouterr().out
    assert "folder_pattern" in out and "s_strain__*" in out
    assert "ccmth_ref_deg" in out and "7.1442" in out
    assert "SKIP" not in out.split("darfix_roi")[0]  # detected rows are not skips
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_config_detect.py -q`
Expected: ImportError — `detect_experiment`/`main` not defined.

- [ ] **Step 3: Write the implementation**

Append to `dfxm/config/detect.py`:

```python
# -- orchestrator -------------------------------------------------------------


def detect_experiment(current) -> list[Detection]:
    """Run every detector against *current* (an :class:`Experiment`).

    Re-runnable: rows for data that does not exist yet come back as
    skip-with-reason, so a pre-darfix pass already shows what a later pass
    will add. Never overwrites anything — callers decide what to apply.
    """
    from dfxm.common.sort import find_matching_folders

    raw_root = (current.raw_root or "").rstrip("/")
    if not raw_root or not os.path.isdir(raw_root):
        return [Detection("raw_root", error="set Raw data root to an existing folder first")]

    out = detect_patterns(raw_root)
    detected = {d.field: d.value for d in out if d.value}
    pattern = (
        current.folder_pattern
        if current.folder_pattern not in ("", "*")
        else detected.get("folder_pattern", "")
    )

    scan = None
    if pattern:
        folders = find_matching_folders(raw_root, pattern)
        scan = select_scan_file(folders[0]) if folders else None
    suffix = current.entry_suffix or ".1"
    if scan is None:
        out.append(
            Detection("entry_suffix", error=f"no layer folder matching {pattern!r} has a scan .h5")
        )
        out.extend(
            Detection(f, error="pixel sizes need a raw scan — none found")
            for f in ("pixel_size_x_um", "pixel_size_y_um")
        )
    else:
        suffix_row = detect_entry_suffix(scan)
        out.append(suffix_row)
        suffix = suffix_row.value or suffix
        out.extend(detect_pixel_sizes(scan, current.positioners_path, suffix))

    found = (
        find_strain_maps(
            current.processed_root, pattern, current.maps_filename, current.ccmth_com_path
        )
        if pattern
        else None
    )
    if found:
        maps_path, folder_name = found
        out.append(detect_ccmth_from_maps(maps_path, folder_name, current.ccmth_com_path))
        out.append(
            detect_darfix_roi(maps_path, folder_name, current.ccmth_com_path, current.darfix_roi)
        )
    else:
        if scan is not None:
            out.append(detect_ccmth_from_positioners(scan, current.positioners_path, suffix))
        out.append(
            Detection(
                "darfix_roi",
                error="no darfix maps.h5 under the processed root yet — re-run after darfix",
            )
        )
    return out


# -- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Print the detection table for a raw (and optionally processed) tree."""
    import argparse

    from dfxm.config.models import Experiment

    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("raw_root", help="RAW_DATA root (the folder holding the layer subfolders)")
    ap.add_argument(
        "--processed-root", default="", help="PROCESSED_DATA root (enables the maps.h5 rows)"
    )
    ap.add_argument("--maps-filename", default="maps.h5")
    args = ap.parse_args(argv)
    exp = Experiment(
        raw_root=args.raw_root,
        processed_root=args.processed_root,
        maps_filename=args.maps_filename,
    )
    for d in detect_experiment(exp):
        if d.error:
            print(f"{d.field:18} SKIP  {d.error}")
        elif d.value is None:
            print(f"{d.field:18} INFO  {d.note}")
        else:
            print(f"{d.field:18} {d.value!s:26} {d.note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_config_detect.py -q`
Expected: all PASS.
Also run the CLI once by hand against nothing: `python3 -m dfxm.config.detect /nonexistent`
Expected: one `raw_root  SKIP  set Raw data root...` line, exit 0.

- [ ] **Step 5: Document the module in Codebase.md**

In `docs/Codebase.md`, inside `### dfxm/config — typed config & presets` (find it
with `grep -n "dfxm/config" docs/Codebase.md`), after the existing
`models.py`/`presets.py` descriptions, add:

```markdown
**`detect.py` — data-driven experiment initialization.** Qt-free detectors that
suggest experiment values from the trees on disk; each returns `Detection`
rows (`field`, `value`, `note`, `error` — `error` set = skip-with-reason;
`value` and `error` both `None` = info-only row). `folder_families` /
`detect_patterns` classify `<stem>__<N>` raw subfolders into
folder/mosa/rocking globs; `select_scan_file` picks the raw scan .h5 (concat
output excluded); `detect_entry_suffix` reads the majority BLISS suffix;
`detect_pixel_sizes` wraps `common.pixel_size.compute_pixel_size`;
`find_strain_maps` + `detect_ccmth_from_maps` take the nanmedian of the ccmth
COM map (mosa maps are skipped — chi/mu only), with
`detect_ccmth_from_positioners` as the pre-darfix fallback;
`detect_darfix_roi` recovers the crop *size* from the map shape (darfix
records no origin — blank current → partial `?,?,w,h`, filled current →
consistency check). `detect_experiment(current)` orchestrates all of the
above (re-runnable, never overwrites); `main` is the
`python3 -m dfxm.config.detect` CLI. GUI consumer:
`gui/widgets/detect_review.py` + the Edit dialog's "Initialize from data…"
button.
```

Also add `detect.py` to the repository-layout tree in the same file if
`dfxm/config` files are listed there (`grep -n "presets.py" docs/Codebase.md`).

- [ ] **Step 6: Full suite, lint, commit**

Run: `python3 -m pytest -q` — expected: all pass (560+new, 13 skipped).
Run: `ruff check .`

```bash
git add dfxm/config/detect.py tests/test_config_detect.py docs/Codebase.md
git commit -m "feat(detect): detect_experiment orchestrator + CLI + Codebase docs"
```

---

### Task 4: Review dialog widget

**Files:**
- Create: `gui/widgets/detect_review.py`
- Test: `tests/test_gui_detect_review.py`

**Interfaces:**
- Consumes: `Detection` (Task 1); `EXPERIMENT_SCHEMA` (existing);
  `parse_darfix_roi` (existing).
- Produces (used by Task 5):
  - `DetectReviewDialog(detections: list[Detection], current: dict[str, Any], defaults: dict[str, Any], parent=None)` — a `QDialog`; OK button reads "Apply checked".
  - `.applied_values() -> dict[str, Any]` — checked rows only; for string-valued
    detections the (possibly edited) cell text, otherwise `Detection.value`.

**Behaviour contract:**
- Columns: Field | Current | Detected | Note | Apply. Field shows the schema
  label (`EXPERIMENT_SCHEMA` lookup, fallback to the raw name).
- Pre-check rules: checked when current is `""`/`None` **or** equals
  `defaults[field]`; otherwise unchecked with "differs from current" appended
  to the note.
- Error rows and info rows (value `None`): Detected shows "—", note shows the
  error/note, whole row disabled (greyed, uncheckable).
- darfix-ROI partial row (`value` starts with `"?,?"`): Detected cell is
  editable; checkbox disabled+unchecked until the text parses via
  `parse_darfix_roi`; once valid it auto-checks.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_detect_review.py`:

```python
"""DetectReviewDialog: pre-check rules, row states, ROI gating, applied_values."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.config.detect import Detection  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402

DEFAULTS = Experiment().to_dict()


def _dlg(detections, current=None):
    from gui.widgets.detect_review import DetectReviewDialog

    _ = QApplication.instance() or QApplication([])
    cur = dict(DEFAULTS)
    cur.update(current or {})
    return DetectReviewDialog(detections, current=cur, defaults=DEFAULTS)


def _check_item(dlg, row):
    return dlg._table.item(row, 4)


def test_blank_current_prechecked_and_applied():
    dlg = _dlg([Detection("folder_pattern", "s__*", "3 folders")])
    assert _check_item(dlg, 0).checkState() == Qt.CheckState.Checked
    assert dlg.applied_values() == {"folder_pattern": "s__*"}


def test_default_valued_current_prechecked():
    # pixel_size_x_um default is 1.0 -> still counts as "not user-set"
    dlg = _dlg([Detection("pixel_size_x_um", 0.151733, "M=20.4")])
    assert _check_item(dlg, 0).checkState() == Qt.CheckState.Checked
    assert dlg.applied_values() == {"pixel_size_x_um": 0.151733}


def test_user_set_current_unchecked_and_marked():
    dlg = _dlg(
        [Detection("ccmth_ref_deg", 7.1442, "median")], current={"ccmth_ref_deg": 7.144}
    )
    item = _check_item(dlg, 0)
    assert item.checkState() == Qt.CheckState.Unchecked
    assert "differs" in dlg._table.item(0, 3).text()
    assert dlg.applied_values() == {}  # nothing checked -> nothing applied


def test_error_row_disabled():
    dlg = _dlg([Detection("mosa_pattern", error="no folder family containing 'mosa'")])
    item = _check_item(dlg, 0)
    assert item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "mosa" in dlg._table.item(0, 3).text()
    assert dlg.applied_values() == {}


def test_info_row_disabled():
    dlg = _dlg([Detection("darfix_roi", None, "✓ size matches maps.h5 (1832×1266)")])
    item = _check_item(dlg, 0)
    assert item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert dlg.applied_values() == {}


def test_partial_roi_gated_until_origin_typed():
    dlg = _dlg([Detection("darfix_roi", "?,?,1832,1266", "map size — replace ?,?")])
    check = _check_item(dlg, 0)
    assert not (check.flags() & Qt.ItemFlag.ItemIsEnabled)  # gated
    dlg._table.item(0, 2).setText("105,230,1832,1266")  # user types the origin
    check = _check_item(dlg, 0)
    assert check.flags() & Qt.ItemFlag.ItemIsEnabled
    assert check.checkState() == Qt.CheckState.Checked  # auto-checks once valid
    assert dlg.applied_values() == {"darfix_roi": "105,230,1832,1266"}


def test_partial_roi_invalid_edit_regates():
    dlg = _dlg([Detection("darfix_roi", "?,?,1832,1266", "map size")])
    dlg._table.item(0, 2).setText("105,230,1832,1266")
    dlg._table.item(0, 2).setText("banana")  # edited back to nonsense
    check = _check_item(dlg, 0)
    assert not (check.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert dlg.applied_values() == {}


def test_unchecking_excludes_from_applied():
    dlg = _dlg(
        [
            Detection("folder_pattern", "s__*", ""),
            Detection("entry_suffix", ".1", ""),
        ]
    )
    _check_item(dlg, 0).setCheckState(Qt.CheckState.Unchecked)
    assert dlg.applied_values() == {"entry_suffix": ".1"}


def test_field_column_shows_schema_label():
    dlg = _dlg([Detection("pixel_size_x_um", 0.15, "")])
    assert dlg._table.item(0, 0).text() == "Pixel size X"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_detect_review.py -q`
Expected: FAIL — `ModuleNotFoundError: gui.widgets.detect_review`

- [ ] **Step 3: Write the implementation**

Create `gui/widgets/detect_review.py`:

```python
"""Review table for data-detected experiment values.

One row per :class:`~dfxm.config.detect.Detection` — current vs detected,
with a per-row Apply checkbox. Pre-check rules: checked when the current
value is blank or still the schema default; unchecked (and marked "differs
from current") when applying would overwrite something the user set.
Skipped and info-only detections render as greyed, uncheckable rows, so a
pre-darfix pass already shows what a later re-run will add.

The darfix-ROI row is special: detection recovers only the crop size, so
its Detected cell is editable (``?,?,w,h``) and its checkbox stays disabled
until the text parses as a full origin+size ROI — then it auto-checks.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from dfxm.config.detect import Detection
from dfxm.config.models import EXPERIMENT_SCHEMA

_LABELS = {p.name: p.label for p in EXPERIMENT_SCHEMA}
_COLS = ("Field", "Current", "Detected", "Note", "Apply")
_CHECKABLE = Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled


def _fmt(value: Any) -> str:
    if value in ("", None):
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _is_unset(field: str, current: Any, defaults: dict[str, Any]) -> bool:
    """True when *current* is blank or still the schema default (safe to fill)."""
    return current in ("", None) or current == defaults.get(field)


class DetectReviewDialog(QDialog):
    """Apply-what-you-check review of :func:`~dfxm.config.detect.detect_experiment`."""

    def __init__(
        self,
        detections: list[Detection],
        current: dict[str, Any],
        defaults: dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Initialize from data — review")
        self.resize(780, 420)
        self._detections = list(detections)

        self._table = QTableWidget(len(self._detections), len(_COLS))
        self._table.setHorizontalHeaderLabels(_COLS)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        for row, d in enumerate(self._detections):
            cur = current.get(d.field, "")
            self._set_text(row, 0, _LABELS.get(d.field, d.field))
            self._set_text(row, 1, _fmt(cur))
            if d.error is not None or d.value is None:
                # skip-with-reason or info-only: greyed, nothing to apply
                self._set_text(row, 2, "—")
                self._set_text(row, 3, d.error if d.error is not None else d.note)
                for col in range(len(_COLS) - 1):
                    self._table.item(row, col).setFlags(Qt.ItemFlag.ItemIsSelectable)
                check = QTableWidgetItem()
                check.setFlags(Qt.ItemFlag.ItemIsSelectable)
                self._table.setItem(row, 4, check)
                continue
            partial = isinstance(d.value, str) and d.value.startswith("?,?")
            cell = QTableWidgetItem(_fmt(d.value))
            if not partial:
                cell.setFlags(cell.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, 2, cell)
            check = QTableWidgetItem()
            note = d.note
            if partial:
                check.setFlags(Qt.ItemFlag.ItemIsUserCheckable)  # gated: greyed until valid
                check.setCheckState(Qt.CheckState.Unchecked)
            elif _is_unset(d.field, cur, defaults):
                check.setFlags(_CHECKABLE)
                check.setCheckState(Qt.CheckState.Checked)
            else:
                check.setFlags(_CHECKABLE)
                check.setCheckState(Qt.CheckState.Unchecked)
                note = f"{note} · differs from current" if note else "differs from current"
            self._set_text(row, 3, note)
            self._table.setItem(row, 4, check)
        self._table.itemChanged.connect(self._on_item_changed)

        hint = QLabel(
            "Checked rows are written into the experiment form. Greyed rows show "
            "what a re-run will add once the data exists."
        )
        hint.setProperty("role", "muted")
        hint.setWordWrap(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply checked")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addWidget(hint)
        layout.addWidget(buttons)

    # -- helpers ----------------------------------------------------------
    def _set_text(self, row: int, col: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._table.setItem(row, col, item)

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Gate the partial darfix-ROI row on its edited text parsing cleanly."""
        if item.column() != 2:
            return
        row = item.row()
        d = self._detections[row]
        if not (isinstance(d.value, str) and d.value.startswith("?,?")):
            return
        from dfxm.common.roi import parse_darfix_roi

        try:
            valid = parse_darfix_roi(item.text()) is not None
        except ValueError:
            valid = False
        check = self._table.item(row, 4)
        if check is None:
            return
        if valid:
            check.setFlags(_CHECKABLE)
            check.setCheckState(Qt.CheckState.Checked)
        else:
            check.setCheckState(Qt.CheckState.Unchecked)
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable)

    # -- result -----------------------------------------------------------
    def applied_values(self) -> dict[str, Any]:
        """Checked rows as {field: value} — edited cell text for string values."""
        out: dict[str, Any] = {}
        for row, d in enumerate(self._detections):
            check = self._table.item(row, 4)
            if check is None or check.checkState() != Qt.CheckState.Checked:
                continue
            if not (check.flags() & Qt.ItemFlag.ItemIsEnabled):
                continue
            cell = self._table.item(row, 2)
            out[d.field] = cell.text() if isinstance(d.value, str) else d.value
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_detect_review.py -q`
Expected: all PASS

- [ ] **Step 5: Lint and commit**

```bash
ruff check gui/widgets/detect_review.py tests/test_gui_detect_review.py
git add gui/widgets/detect_review.py tests/test_gui_detect_review.py
git commit -m "feat(gui): DetectReviewDialog — review table for detected experiment values"
```

---

### Task 5: Wire into ExperimentDialog + save prompt + smoke + docs

**Files:**
- Modify: `gui/experiment_panel.py` (button row `__init__` block at ~lines 58–78; `_on_accept` at ~line 142; new methods after `_on_compute_pixel_size`)
- Modify: `tests/gui_smoke.py` (append step `[35]` before the final `print("\nGUI SMOKE PASSED")`)
- Modify: `docs/Usage.md` (`### Experiment presets` section) and `docs/Codebase.md` (`## Layer 2` `gui/experiment_panel` prose + `### gui/widgets/` list)
- Test: `tests/test_gui_experiment_init.py` (create)

**Interfaces:**
- Consumes: `detect_experiment` (Task 3), `DetectReviewDialog.applied_values()`
  (Task 4), `ParamForm.set_values`, `Experiment.from_dict/to_dict` (existing).
- Produces: `ExperimentDialog._on_initialize_from_data()` (button slot),
  `ExperimentDialog._detect(vals) -> list[Detection]` (dialog-free seam for
  tests), `self._applied_detections: bool` (drives the OK-time save prompt).

**Read `gui/experiment_panel.py` in full before editing** (CLAUDE.md: read
before first Edit; `hint=`/em-dash hazards).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gui_experiment_init.py`:

```python
"""ExperimentDialog: Initialize from data… wiring + OK-time save prompt."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import h5py
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox  # noqa: E402


def _dlg(**fields):
    from dfxm.config.models import Experiment
    from gui.experiment_panel import ExperimentDialog

    _ = QApplication.instance() or QApplication([])
    return ExperimentDialog(Experiment(**fields))


def _make_raw(tmp_path):
    raw = tmp_path / "RAW"
    d = raw / "s_strain__0"
    d.mkdir(parents=True)
    with h5py.File(d / "s_strain__0.h5", "w") as f:
        pos = f.create_group("1.1/instrument/positioners")
        for k, v in dict(
            mainx=-5000.0, obx=273.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0, ccmth=7.1
        ).items():
            pos.create_dataset(k, data=v)
    return str(raw)


def test_initialize_button_exists():
    from PySide6.QtWidgets import QPushButton

    dlg = _dlg()
    labels = [b.text() for b in dlg.findChildren(QPushButton)]
    assert any("Initialize from data" in t for t in labels)


def test_blank_raw_root_warns_and_aborts(monkeypatch):
    dlg = _dlg()
    warned = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warned.append(a))
    dlg._on_initialize_from_data()
    assert warned and "Raw data root" in warned[0][2]


def test_detect_seam_runs_on_form_values(tmp_path):
    raw = _make_raw(tmp_path)
    dlg = _dlg()
    dlg._form.set_values({"raw_root": raw})  # typed into the form, never saved
    rows = {d.field: d for d in dlg._detect(dlg._form.values())}
    assert rows["folder_pattern"].value == "s_strain__*"
    assert rows["pixel_size_x_um"].value


def test_apply_marks_and_ok_prompts_save(tmp_path, monkeypatch):
    raw = _make_raw(tmp_path)
    dlg = _dlg(raw_root=raw)
    # simulate the review dialog having applied values
    dlg._form.set_values({"folder_pattern": "s_strain__*"})
    dlg._applied_detections = True
    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: (asked.append(a), QMessageBox.StandardButton.No)[1],
    )
    dlg._on_accept()
    assert asked  # prompted
    assert dlg.result() == QDialog.DialogCode.Accepted  # "No" still accepts
    assert dlg._applied_detections is False  # asks once


def test_ok_without_apply_never_prompts(monkeypatch):
    dlg = _dlg()
    asked = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *a, **k: (asked.append(a), QMessageBox.StandardButton.No)[1],
    )
    dlg._on_accept()
    assert not asked
    assert dlg.result() == QDialog.DialogCode.Accepted


def test_save_choice_routes_to_save_as(monkeypatch):
    dlg = _dlg()
    dlg._applied_detections = True
    monkeypatch.setattr(
        QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Save
    )
    saved = []
    monkeypatch.setattr(dlg, "_on_save_as", lambda: saved.append(True))
    dlg._on_accept()
    assert saved
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_gui_experiment_init.py -q`
Expected: FAIL — no button, no `_on_initialize_from_data`, no `_detect`.

- [ ] **Step 3: Implement the wiring**

In `gui/experiment_panel.py`:

(a) In `ExperimentDialog.__init__`, right after `buttons.addButton(save_btn, ...)`
and before the `compute_btn` block, add the button and the flag:

```python
        init_btn = QPushButton("Initialize from data…")
        init_btn.setToolTip(
            "Scan the data roots and suggest patterns, calibration and ROI values"
        )
        init_btn.clicked.connect(self._on_initialize_from_data)
        buttons.addButton(init_btn, QDialogButtonBox.ButtonRole.ActionRole)
```

and at the end of `__init__` (after `self._update_roi_note()`):

```python
        self._applied_detections = False
```

(b) Replace `_on_accept` with:

```python
    def _on_accept(self) -> None:
        if self._warn_roi_problems():
            return
        if self._applied_detections:
            self._applied_detections = False  # ask once per apply
            ret = QMessageBox.question(
                self,
                "Initialize from data",
                "Save the preset to a YAML now? The applied values otherwise "
                "live only in this session.",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.No,
            )
            if ret == QMessageBox.StandardButton.Save:
                self._on_save_as()
        self.accept()
```

(c) After `_on_compute_pixel_size`, add:

```python
    def _detect(self, vals: dict) -> list:
        """Run detection on the current form values (no dialogs — test seam)."""
        from dfxm.config.detect import detect_experiment

        return detect_experiment(Experiment.from_dict(vals))

    def _on_initialize_from_data(self) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QApplication

        vals = self._form.values()
        raw = (vals.get("raw_root") or "").strip()
        if not raw or not os.path.isdir(raw):
            QMessageBox.warning(
                self,
                "Initialize from data",
                "Set 'Raw data root' to an existing folder first — detection "
                "starts from the raw scan tree.",
            )
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            detections = self._detect(vals)
        finally:
            QApplication.restoreOverrideCursor()

        from .widgets.detect_review import DetectReviewDialog

        dlg = DetectReviewDialog(
            detections, current=vals, defaults=Experiment().to_dict(), parent=self
        )
        try:
            if dlg.exec():
                applied = dlg.applied_values()
                if applied:
                    self._form.set_values(applied)
                    self._applied_detections = True
        finally:
            dlg.deleteLater()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_gui_experiment_init.py tests/test_gui_experiment_roi.py -q`
Expected: all PASS (including the pre-existing ROI dialog tests — `_on_accept`
still routes through `_warn_roi_problems` first).

- [ ] **Step 5: Add smoke step [35]**

Read the end of `tests/gui_smoke.py` (step [34] block, ~lines 1028–1045).
Insert before `print("\nGUI SMOKE PASSED")`:

```python
    # [35] Initialize from data: detect on a synthetic raw tree -> review dialog
    # pre-checks blank fields -> apply lands in the experiment form.
    import h5py as _h535

    _raw35 = os.path.join(tempfile.mkdtemp(prefix="smoke_detect_"), "RAW")
    for _fam35, _n35 in (("s1_strain", 2), ("s1_mosa", 1), ("s1_rocking", 1)):
        for _i35 in range(_n35):
            os.makedirs(os.path.join(_raw35, f"{_fam35}__{_i35}"))
    with _h535.File(os.path.join(_raw35, "s1_strain__0", "s1_strain__0.h5"), "w") as _f35:
        _pos35 = _f35.create_group("1.1/instrument/positioners")
        for _k35, _v35 in dict(
            mainx=-5000.0, obx=273.0, ffsel=-60.0, ffz=2100.0, lenssel=0.0, ccmth=7.1
        ).items():
            _pos35.create_dataset(_k35, data=_v35)
    from gui.widgets.detect_review import DetectReviewDialog as _DRD35

    _dlg35 = _ED34(_Exp34(raw_root=_raw35))
    _rows35 = _dlg35._detect(_dlg35._form.values())
    assert {d.field: d.value for d in _rows35 if d.value}["folder_pattern"] == "s1_strain__*"
    _rev35 = _DRD35(_rows35, current=_dlg35._form.values(), defaults=_Exp34().to_dict())
    _applied35 = _rev35.applied_values()  # pre-checked = blank/default fields
    assert _applied35["folder_pattern"] == "s1_strain__*"
    assert _applied35["mosa_pattern"] == "s1_mosa__*"
    assert 0 < _applied35["pixel_size_x_um"] < _applied35["pixel_size_y_um"]
    assert "darfix_roi" not in _applied35  # skip row pre-darfix — never auto-applied
    _dlg35._form.set_values(_applied35)
    assert _dlg35.experiment().folder_pattern == "s1_strain__*"
    print("[35] initialize-from-data: detectors → review pre-checks → applied into the form")
```

If `tempfile` is not already imported at the top of `gui_smoke.py`, check with
`grep -n "^import tempfile\|^import " tests/gui_smoke.py` and add it to the
existing import block if missing. `_ED34`/`_Exp34` are in scope from step [34];
keep this step directly after it.

Run: `python3 tests/gui_smoke.py`
Expected: ends with `[35] initialize-from-data: ...` then `GUI SMOKE PASSED`.

- [ ] **Step 6: Update both docs (same commit)**

`docs/Usage.md` — at the end of the `### Experiment presets` section (before
`### Regions of interest — two windows, two frames`), add:

```markdown
#### Initializing an experiment from data

Instead of hand-filling the editor, click **Initialize from data…** (in
*Edit…*). It scans the roots currently typed in the form and suggests values,
shown in a review table — current value vs detected, with a per-row *Apply*
checkbox (pre-checked only where it would not overwrite something you set):

- **Folder patterns** and **entry suffix**, from the `<name>__<N>` layer
  folders under the raw root.
- **Pixel sizes X/Y**, computed from the far-field motors of the first raw
  scan (same physics as *Compute pixel size from scan…*).
- **ccmth reference**, preferably the median of a strain-layer ccmth
  centre-of-mass map under the processed root; before darfix has run it
  falls back to the raw `ccmth` motor snapshot — confirm either against the
  beamline alignment before trusting strain maps.
- **Darfix ROI size**: darfix does not record its crop anywhere, so only the
  window *size* can be read back (from the map shape). The row shows
  `?,?,w,h` — type the origin from darfix's ROI widget to enable applying
  it. If your Darfix ROI is already filled, the row instead verifies the
  size against `maps.h5`.

The flow is **re-runnable**: run it on day one for the raw-data facts (the
maps rows appear greyed with the reason), then again after darfix to add the
maps-derived rows. On *OK*, if you applied anything, the dialog offers to
save the preset YAML — otherwise the values live only in this session.

The same detection runs headless:
`python3 -m dfxm.config.detect RAW_ROOT --processed-root PROC_ROOT`.
```

`docs/Codebase.md` — two touches:
1. In the Layer 2 `gui/experiment_panel` description, mention the new button:
   "**Initialize from data…** runs `dfxm.config.detect.detect_experiment` on
   the live form values and opens `DetectReviewDialog`; applying sets
   `_applied_detections`, which makes OK offer a save-to-YAML prompt."
2. In `### gui/widgets/`, add:
   "`detect_review.py` — `DetectReviewDialog`: review table for detected
   experiment values (pre-check rules, greyed skip/info rows, editable
   origin-gated darfix-ROI row); `applied_values()` returns the checked
   rows."

- [ ] **Step 7: Full verification and commit**

Run: `python3 -m pytest -q` — expected all pass.
Run: `ruff check .` — expected clean.
Run: `python3 tests/gui_smoke.py` — expected `GUI SMOKE PASSED` with 35 steps.

```bash
git add gui/experiment_panel.py tests/test_gui_experiment_init.py tests/gui_smoke.py \
        docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): Initialize from data… — detection review wired into the experiment editor + docs + smoke [35]"
```

---

## Verification (whole feature)

1. `python3 -m pytest -q` — full suite green.
2. `ruff check . && ruff format --check .` — clean.
3. `python3 tests/gui_smoke.py` — 35/35.
4. Real-data CLI check (SSD mounted):
   `python3 -m dfxm.config.detect /media/albert/DIC_SSD_3/ESRF/ma6778/id03/20251029/RAW_DATA/STO2_overnight --processed-root /media/albert/DIC_SSD_3/ESRF/ma6778/id03/20251029/PROCESSED_DATA/STO2_overnight`
   Expected: `folder_pattern STO2_overnight_layer_2x_energy_strain__*`,
   pixel sizes ≈ 0.151733 / 0.387584, `ccmth_ref_deg` ≈ 7.1442 (median note),
   `darfix_roi` `?,?,1832,1266` row (preset ROI not passed on the CLI).
5. On-screen GUI eyeball (user): STO2 preset → Edit… → Initialize from data… →
   review table shows ✓-consistent darfix-ROI row and "differs" markers on the
   calibrated fields.
