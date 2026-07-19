# Profiles Replot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Replot… button on the profiles stage that re-renders profile figures
(overviews + companion + traces, never CSVs) cold from `oblique_slices.h5` with
per-quantity colour-limit overrides, mirroring the slices replot workflow.

**Architecture:** Qt-free core first — thread a `{key: (vmin, vmax)}` clim
mapping into `_collect`'s attrs (field-id key, colormap-group fallback), extract
the parameter-mode job body of `run()` into `_render_parameter_job`, and add
`render_replot()` on top of both. Then a thin GUI dialog
(`gui/widgets/profiles_replot.py`, modeled on `slice_replot.py`, reusing
`ClimGroupSection`) plus the Replot… button branch in `stage_view.py`.

**Tech Stack:** Python 3.10, h5py, matplotlib (Figure API only), PySide6 (GUI
layer only), pytest + `tests/gui_smoke.py`.

**Spec:** `docs/superpowers/specs/2026-07-20-profiles-replot-design.md` (approved).

## Global Constraints

- `dfxm/` stays Qt-free; never import PySide6/pyvista there.
- Figures via `matplotlib.figure.Figure` only — never pyplot.
- Docs contract: every behaviour change updates `docs/Usage.md` and
  `docs/Codebase.md` **in the same task/commit** as the code.
- `hint=` strings in `dfxm/stages/*.py` contain em-dashes at 12 or 16 spaces —
  Read the exact bytes before any Edit touching them; never reconstruct
  `old_string` from memory.
- Ruff: line length 100, double quotes (auto-format hook runs on Write/Edit).
- Existing profiles run-path behaviour must be byte-identical for `clim=None`
  (existing tests in `tests/test_stage_profiles.py` are the guard).
- Work on branch `profiles-replot`; no remote exists (never pull/push).
- The GUI smoke test is `tests/gui_smoke.py` (not a pytest file); run it as
  `python3 tests/gui_smoke.py`.

## File Structure

- `dfxm/stages/profiles.py` — modify: `_clim_attrs` (new), `_collect` (clim
  param), `_render_parameter_job` (extracted from `run()`), `render_replot` +
  `replot_catalog` (new public API).
- `gui/widgets/clim_section.py` — modify: gains shared `KIND_LABELS` +
  `volume_label` (moved from `slice_replot.py`).
- `gui/widgets/slice_replot.py` — modify: import the moved labels (alias kept).
- `gui/widgets/profiles_replot.py` — create: `ProfilesReplotDialog`.
- `gui/stage_view.py` — modify: Replot… button for `profiles`, `_replot_profiles`.
- `tests/test_stage_profiles.py` — modify: core tests.
- `tests/gui_smoke.py` — modify: step [33].
- `docs/Usage.md`, `docs/Codebase.md` — modify (in the same tasks as the code).

---

### Task 1: Core clim threading (`_clim_attrs` + `_collect(clim=…)`)

**Files:**
- Modify: `dfxm/stages/profiles.py` (imports; `_collect` at ~line 834)
- Test: `tests/test_stage_profiles.py`

**Interfaces:**
- Consumes: `dfxm.common.figures.resolve_clim(clim, key)` (exists),
  `dfxm.common.plotting.GROUP_BY_KIND` (exists: kind → colormap group).
- Produces: `_clim_attrs(attrs: dict, vid: str, clim) -> dict` (mutates+returns
  attrs) and `_collect(f, job, p, ref_pref, restrict, clim=None)` — Task 2
  passes `clim` through; all other callers stay unchanged.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_stage_profiles.py`)

```python
def test_clim_attrs_field_id_beats_group_fallback():
    attrs = {"kind": "strain", "vmin": -10.0, "vmax": 10.0}
    out = PR._clim_attrs(dict(attrs), "strain", {"strain": (-1.0, 1.0)})
    assert (out["vmin"], out["vmax"]) == (-1.0, 1.0)
    # group fallback: vid not in mapping, kind's colormap group is
    attrs2 = {"kind": "raw_sum", "vmin": 0.0, "vmax": 100.0}
    out2 = PR._clim_attrs(dict(attrs2), "raw_sum", {"raw": (5.0, None)})
    assert (out2["vmin"], out2["vmax"]) == (5.0, 100.0)  # half-open keeps stored vmax
    # vid key wins over group key when both present
    out3 = PR._clim_attrs(dict(attrs2), "raw_sum", {"raw": (5.0, 50.0), "raw_sum": (7.0, 70.0)})
    assert (out3["vmin"], out3["vmax"]) == (7.0, 70.0)
    # no matching key / clim None -> untouched
    out4 = PR._clim_attrs(dict(attrs), "strain", {"mosa_com": (0.0, 1.0)})
    assert (out4["vmin"], out4["vmax"]) == (-10.0, 10.0)
    assert PR._clim_attrs(dict(attrs), "strain", None)["vmin"] == -10.0


def test_collect_applies_clim_to_ref_and_fields(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    job = {"name": "oblique_full", "offset_um": 0.0, "start_uv": [-5, -3], "end_uv": [5, 3]}
    p = PR.STAGE.defaults()
    with h5py.File(str(h5), "r") as f:
        ref, fields, _geom, _off, _dropped = PR._collect(
            f, job, p, "", None, clim={"strain": (-2.0, 2.0)}
        )
    by_vid = {fl["vid"]: fl["attrs"] for fl in fields}
    assert (by_vid["strain"]["vmin"], by_vid["strain"]["vmax"]) == (-2.0, 2.0)
    assert (by_vid["raw_sum"]["vmin"], by_vid["raw_sum"]["vmax"]) == (-10.0, 10.0)  # stored
    assert (ref[3]["vmin"], ref[3]["vmax"]) == (-10.0, 10.0)  # ref is raw_sum -> stored
```

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_stage_profiles.py -q -k clim`
Expected: FAIL — `AttributeError: module ... has no attribute '_clim_attrs'`.

- [ ] **Step 3: Implement.** In `dfxm/stages/profiles.py`: add `GROUP_BY_KIND`
to the existing `from ..common.plotting import (...)` block and `resolve_clim`
to the existing `from ..common.figures import (...)` block (Read the import
region first — grep `from ..common` for exact lines). Insert above `_collect`:

```python
def _clim_attrs(attrs, vid, clim):
    """Apply a per-quantity ``(vmin, vmax)`` override to a read_volume_attrs dict.

    Key resolution matches the slices replot: exact field id first (e.g.
    ``mosa_com_chi``), then the field kind's colormap group via GROUP_BY_KIND.
    A half-open pair keeps the stored value on the blank side. ``clim=None``
    (or no matching key) leaves *attrs* untouched.
    """
    pair = resolve_clim(clim, vid)
    if pair is None:
        pair = resolve_clim(clim, GROUP_BY_KIND.get(attrs.get("kind", ""), ""))
    if pair is None:
        return attrs
    lo, hi = pair
    if lo is not None:
        attrs["vmin"] = float(lo)
    if hi is not None:
        attrs["vmax"] = float(hi)
    return attrs
```

Change `_collect`'s signature to `def _collect(f, job, p, ref_pref, restrict, clim=None):`
and wrap both `read_volume_attrs` calls:
`ref_attrs = _clim_attrs(read_volume_attrs(f, ref_id), ref_id, clim)` and, in
the field loop, `"attrs": _clim_attrs(read_volume_attrs(f, vid), vid, clim),`.

- [ ] **Step 4: Run tests**

Run: `python3 -m pytest tests/test_stage_profiles.py -q`
Expected: all pass (new clim tests + every existing test — `clim` defaults to
`None`, so the run path is unchanged).

- [ ] **Step 5: Docs + commit.** `docs/Codebase.md` profiles entry (the
"HDF5 access" bullet area): add one sentence — `_clim_attrs(attrs, vid, clim)`
applies a per-quantity limit override (field-id key, GROUP_BY_KIND fallback,
half-open pairs keep the stored side); `_collect` threads `clim` through. Then:

```bash
git add dfxm/stages/profiles.py tests/test_stage_profiles.py docs/Codebase.md
git commit -m "feat(profiles): thread per-quantity clim overrides into _collect"
```

---

### Task 2: `_render_parameter_job` extraction + `render_replot` + `replot_catalog`

**Files:**
- Modify: `dfxm/stages/profiles.py` (`run()` parameter-mode body at ~1080–1127)
- Test: `tests/test_stage_profiles.py`

**Interfaces:**
- Consumes: Task 1's `_collect(..., clim=None)`; existing
  `resolve_job_slice_name`, `_unique_name`, `save_companion_figure`,
  `_save_traces`, `_save_overviews`, `_write_csvs`, `ProfilesResult`,
  `ProfileJobResult`, `StageUserError`.
- Produces (Task 3 relies on these exact signatures):
  - `render_replot(h5_path, jobs, style, clim, out_dir, *, dpi=None) -> ProfilesResult`
  - `replot_catalog(h5_path, jobs) -> list[ReplotJobEntry]` with
    `ReplotJobEntry(job_index: int, name: str, label: str, fields: list[str], note: str | None)`

- [ ] **Step 1: Write the failing tests** (append; `_write_pinned` exists from
the pinned-name fix)

```python
def test_render_replot_writes_figures_no_csvs(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    out = tmp_path / "replots"
    jobs = [
        {
            "name": "oblique_full",
            "offset_um": 0.0,
            "start_uv": [-5, -3],
            "end_uv": [5, 3],
            "n_samples": 40,
            "width_pixels": 1,
            "fig_name": "rp0",
        }
    ]
    res = PR.render_replot(str(h5), jobs, None, {"strain": (-2.0, 2.0)}, str(out))
    assert len(res.jobs) == 1
    jr = res.jobs[0]
    assert jr.figure and os.path.exists(jr.figure)  # companion
    assert len(jr.overviews) == 2 and all(os.path.exists(p) for p in jr.overviews)
    assert len(jr.traces) == 2 and all(os.path.exists(p) for p in jr.traces)
    assert jr.csvs == []  # replots never write CSVs
    assert not any(fn.endswith(".csv") for fn in os.listdir(out))


def test_render_replot_resolves_pinned_names(tmp_path):
    h5 = tmp_path / "oblique_slices_pinned.h5"
    _write_pinned(str(h5))
    jobs = [{"name": "oblique_full", "offset_um": 0.8, "start_uv": [-5, -3], "end_uv": [5, 3]}]
    res = PR.render_replot(str(h5), jobs, None, None, str(tmp_path / "rp"))
    assert len(res.jobs) == 1 and res.jobs[0].name == "oblique_full_pin_+1.00um"
    assert any("pinned" in n for n in res.notes)


def test_render_replot_bad_inputs_raise_stageusererror(tmp_path):
    with pytest.raises(PR.StageUserError):
        PR.render_replot(str(tmp_path / "missing.h5"), [{"name": "x"}], None, None, str(tmp_path))
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    with pytest.raises(PR.StageUserError):
        PR.render_replot(str(h5), [], None, None, str(tmp_path))


def test_replot_catalog_lists_jobs_and_fields(tmp_path):
    h5 = tmp_path / "oblique_slices.h5"
    _write_consolidated(str(h5))
    jobs = [
        {"name": "oblique_full", "offset_um": 0.0, "fig_name": "rp0"},
        {"name": "no_such_slice", "offset_um": 0.0},
    ]
    cat = PR.replot_catalog(str(h5), jobs)
    assert len(cat) == 1  # jobs whose slice is absent (plain or pinned) are omitted
    e = cat[0]
    assert e.job_index == 0 and e.name == "oblique_full"
    assert e.fields == ["raw_sum", "strain"]
    assert "rp0" in e.label and e.note is None
```

Check the test file imports `pytest` and that `PR.StageUserError` is reachable
(`profiles.py` imports it from `dfxm.common.errors`); if the module doesn't
re-export it, import it in the test from `dfxm.common.errors` instead.

- [ ] **Step 2: Run to verify they fail**

Run: `python3 -m pytest tests/test_stage_profiles.py -q -k "render_replot or replot_catalog"`
Expected: FAIL — `AttributeError ... 'render_replot'`.

- [ ] **Step 3: Extract the helper.** Read `run()`'s parameter-mode branch
first (`dfxm/stages/profiles.py` ~lines 1080–1127; line numbers have shifted —
grep `# parameter mode`). Move that body into a module-level function directly
above `run()`; it re-reads every knob from `p` so it is self-contained:

```python
def _render_parameter_job(f, job, ji, p, result, used_stems, out_dir, style, progress, clim=None):
    """Render one parameter-mode job into *out_dir* (companion/traces/CSVs/
    overviews per the p-flags), appending to *result*. Shared verbatim by
    run() and render_replot() so the two paths cannot drift; *clim* is the
    per-quantity override mapping threaded to _collect."""
    ref_pref = p["reference_volume_id"] or ""
    restrict = [v.strip() for v in p["volume_ids"].split(",") if v.strip()] or None
    line_override = p["line_color"] or None
    dpi = int(p["fig_dpi"])
    name = job["name"]
    try:
        ref, fields, geom, off_used, dropped = _collect(f, job, p, ref_pref, restrict, clim=clim)
    except (KeyError, ValueError) as exc:
        result.skipped.append(f"{name}: {exc}")
        return
    for reason in dropped:
        msg = f"{name}: {reason}"
        result.notes.append(msg)
        progress((ji + 0.5) / max(1, len(json.loads(p["jobs_json"]) or [])), msg)
    ...
```

**Correction to the progress line above:** do NOT re-parse jobs_json — give the
helper the original fraction instead. Final signature:
`_render_parameter_job(f, job, frac, p, result, used_stems, out_dir, style, progress, clim=None)`
where `frac` is the precomputed progress fraction (`(ji + 0.5) / len(jobs)`),
and the drop-note line becomes `progress(frac, msg)`. The rest of the moved
body is byte-identical to today's: `auto_line_color` → `_unique_name` stem →
`ProfileJobResult(name=..., offset_used_um=..., fields=[...], job_index=ji)` —
**keep `job_index`: pass `ji` as a separate argument** (final form:
`_render_parameter_job(f, job, ji, frac, p, result, used_stems, out_dir, style, progress, clim=None)`)
— then the four `if save_companion / save_traces / p["save_csv"] / p["save_overview"]`
blocks exactly as they are (the trace kwargs read from `p` inside the helper:
`aspect=p["trace_aspect"]` etc., matching the locals run() currently builds),
ending with `result.jobs.append(jr)`. In `run()`, replace the moved branch with:

```python
            # parameter mode
            _render_parameter_job(
                f, job, ji, (ji + 0.5) / len(jobs), p, result, used_stems, out_dir, style, progress
            )
```

`run()` keeps its fail-fast trace validation and its local `save_traces` etc.
for that validation; the helper re-derives its own from `p` (same values).

- [ ] **Step 4: Run the full profiles file — extraction must be behaviour-neutral**

Run: `python3 -m pytest tests/test_stage_profiles.py -q`
Expected: every pre-existing test passes; the 4 new ones still fail.

- [ ] **Step 5: Add the public API** (below `resolve_job_slice_name`):

```python
@dataclass
class ReplotJobEntry:
    """One profile job as the replot dialog sees it: resolved slice + fields."""

    job_index: int
    name: str  # resolved (possibly pinned) slice-group name
    label: str  # display label: fig_name/name @ offset
    fields: list[str]  # volume ids carrying this slice, sorted
    note: str | None  # pin-substitution note, if any


def replot_catalog(h5_path: str, jobs: list[dict]) -> list[ReplotJobEntry]:
    """List each job's resolved slice group and the fields present for it.

    Jobs whose slice has no plain or pinned match are omitted (the dialog
    shows what will actually render; render_replot re-reports the skip).
    Raises StageUserError for an unreadable file.
    """
    try:
        fh = h5py.File(h5_path, "r")
    except OSError as exc:
        raise StageUserError(
            f"cannot read {h5_path!r}: {exc}",
            hint="Point at an oblique_slices.h5 written by the slices stage.",
        ) from exc
    entries: list[ReplotJobEntry] = []
    with fh as f:
        for ji, job in enumerate(jobs):
            if not isinstance(job, dict) or "name" not in job:
                continue
            name, note = resolve_job_slice_name(f, job["name"], job.get("offset_um", 0.0))
            present = volume_ids_with_slice(f, name)
            if not present:
                continue
            off = float(job.get("offset_um", 0.0))
            base = job.get("fig_name") or job["name"]
            entries.append(
                ReplotJobEntry(ji, name, f"{base}  @ {off:+.2f} µm", present, note)
            )
    return entries


def render_replot(h5_path, jobs, style, clim, out_dir, *, dpi=None):
    """Re-render profile jobs cold with optional per-quantity colour limits.

    Appearance-only twin of a parameter-mode run: writes companion, overview
    and trace figures for *jobs* into *out_dir* — never CSVs. ``clim`` is a
    ``{key: (vmin, vmax)}`` mapping (field id first, colormap group fallback;
    ``None``/missing keeps stored limits). ``dpi=None`` uses the stage's
    ``fig_dpi`` default. Returns a ProfilesResult (jobs/skipped/notes).
    """
    if not h5_path or not os.path.exists(h5_path):
        raise StageUserError(
            f"consolidated slice file not found: {h5_path!r}",
            hint="Run the slices stage first, or Browse to an oblique_slices.h5.",
        )
    if not isinstance(jobs, list) or not jobs:
        raise StageUserError(
            "no jobs to replot",
            hint="Check at least one job in the tree (jobs come from the form's Jobs JSON).",
        )
    p = {**STAGE.defaults(), "save_csv": False}
    if dpi is not None:
        p["fig_dpi"] = int(dpi)
    os.makedirs(out_dir, exist_ok=True)
    result = ProfilesResult(output_dir=out_dir, mode="parameter")
    used_stems: dict[str, int] = {}
    with h5py.File(h5_path, "r") as f:
        for ji, job in enumerate(jobs):
            name, pin_note = resolve_job_slice_name(f, job["name"], job.get("offset_um", 0.0))
            if pin_note:
                result.notes.append(pin_note)
                job = {**job, "name": name}
            if not volume_ids_with_slice(f, name):
                result.skipped.append(f"slice {name!r} not present")
                continue
            _render_parameter_job(
                f, job, ji, (ji + 0.5) / len(jobs), p, result, used_stems, out_dir, style, _noop
            )
    return result
```

(`dataclass`/`field` are already imported at the top of profiles.py — verify
with a grep before assuming.)

- [ ] **Step 6: Run tests**

Run: `python3 -m pytest tests/test_stage_profiles.py -q`
Expected: all pass, including the 4 new ones.

- [ ] **Step 7: Docs + commit.** `docs/Codebase.md` (profiles entry): document
`_render_parameter_job` (shared by run/replot), `render_replot`,
`replot_catalog`/`ReplotJobEntry` with the signatures above. `docs/Usage.md`:
no user-visible change yet (button lands in Task 3).

```bash
git add dfxm/stages/profiles.py tests/test_stage_profiles.py docs/Codebase.md
git commit -m "feat(profiles): Qt-free render_replot + replot_catalog (cold replot core)"
```

---

### Task 3: `ProfilesReplotDialog` + shared quantity labels

**Files:**
- Modify: `gui/widgets/clim_section.py` (move labels in), `gui/widgets/slice_replot.py` (import them)
- Create: `gui/widgets/profiles_replot.py`
- Test: `tests/gui_smoke.py` (step [33])

**Interfaces:**
- Consumes: Task 2's `replot_catalog(h5_path, jobs)` / `render_replot(h5_path,
  jobs, style, clim, out_dir, dpi=...)`; `ClimGroupSection` (unchanged API:
  `set_groups([(key, label)])`, `clim_by_group()`, `validate()`).
- Produces: `ProfilesReplotDialog(h5_path, jobs, style=None, out_default="",
  parent=None)` with `.written: list[str]`, `.render_selection(out_dir)`,
  `.select_all()`; `clim_section.volume_label(vid)` (shared label helper).

- [ ] **Step 1: Move the label helpers.** Read `gui/widgets/slice_replot.py`
lines 31–50 (`_KIND_LABELS` + `_volume_label`) and move them into
`gui/widgets/clim_section.py` as module-level `KIND_LABELS` and
`volume_label(volume_id: str) -> str` (same bodies, public names, docstring
noting they label clim rows for slices-file field ids). In `slice_replot.py`
replace the definitions with:

```python
from .clim_section import ClimGroupSection, volume_label as _volume_label
```

(and delete the now-unused `_KIND_LABELS`). Run:
`python3 -m pytest tests/ -q -k "replot or clim"` then
`python3 tests/gui_smoke.py` — all green (pure move).

- [ ] **Step 2: Write the dialog.** Create `gui/widgets/profiles_replot.py`:

```python
"""Profiles replot dialog (built lazily on demand).

Re-renders profile jobs cold from an oblique_slices.h5 with per-quantity
colour-limit overrides — overviews, companion and traces only, never CSVs.
Jobs come from the profiles form's Jobs (JSON); all figure work happens in the
Qt-free core (dfxm.stages.profiles.render_replot); this dialog is a thin shell.
"""

from __future__ import annotations

import os
import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from dfxm.stages import profiles as _pr

from .clim_section import ClimGroupSection, volume_label


class ProfilesReplotDialog(QDialog):
    """Pick jobs/fields from the form's jobs list and re-render profile figures."""

    def __init__(self, h5_path, jobs, style=None, out_default="", parent=None) -> None:
        super().__init__(parent)
        self._h5_path = h5_path
        self._jobs = [j for j in (jobs or []) if isinstance(j, dict) and "name" in j]
        self._style = style
        self.written: list[str] = []
        self._ts = time.strftime("%Y%m%d-%H%M%S")
        self._out_pinned = bool(out_default)

        self.setWindowTitle(f"Replot profiles — {os.path.basename(h5_path or '(no file)')}")

        self._file_edit = QLineEdit(h5_path)
        file_browse = QPushButton("Browse…")
        file_browse.clicked.connect(self._on_browse_h5)
        file_reload = QPushButton("Load")
        file_reload.clicked.connect(self._reload)
        file_row = QHBoxLayout()
        file_row.addWidget(QLabel("Slices file:"))
        file_row.addWidget(self._file_edit, 1)
        file_row.addWidget(file_browse)
        file_row.addWidget(file_reload)

        # job → fields checkbox tree
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Job / field"])
        self._tree.itemChanged.connect(self._on_item_changed)

        self._clim = ClimGroupSection()
        clim_header = QLabel("Colour limits (per quantity; blank = stored):")

        self._out_edit = QLineEdit(out_default or self._default_out_for(h5_path))
        self._out_edit.textEdited.connect(lambda _t: setattr(self, "_out_pinned", True))
        out_browse = QPushButton("Browse…")
        out_browse.clicked.connect(self._on_browse_out)
        self._dpi = QSpinBox()
        self._dpi.setRange(50, 1200)
        self._dpi.setValue(int(_pr.STAGE.defaults()["fig_dpi"]))
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output dir:"))
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(out_browse)
        out_row.addWidget(QLabel("DPI:"))
        out_row.addWidget(self._dpi)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._render_btn = QPushButton("Render")
        self._render_btn.setProperty("role", "primary")
        self._render_btn.clicked.connect(self._on_render)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._status, 1)
        btn_row.addWidget(self._render_btn)
        btn_row.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addLayout(file_row)
        layout.addWidget(self._tree, 1)
        layout.addWidget(clim_header)
        layout.addWidget(self._clim)
        layout.addLayout(out_row)
        layout.addLayout(btn_row)

        self._catalog: list = []
        self._reload()

    def _default_out_for(self, h5_path: str) -> str:
        if not h5_path:
            return ""
        return os.path.join(os.path.dirname(os.path.abspath(h5_path)), "replots", self._ts)

    # -- population -----------------------------------------------------------
    def _reload(self) -> None:
        self._h5_path = self._file_edit.text().strip()
        if not self._out_pinned:
            self._out_edit.setText(self._default_out_for(self._h5_path))
        self._tree.clear()
        self._catalog = []
        if not self._h5_path or not os.path.exists(self._h5_path):
            self._clim.set_groups([])
            self._status.setText("no such file")
            self._update_render_enabled()
            return
        if not self._jobs:
            self._clim.set_groups([])
            self._status.setText("no jobs — fill Jobs (JSON) on the form (e.g. via Pick line…)")
            self._update_render_enabled()
            return
        try:
            self._catalog = _pr.replot_catalog(self._h5_path, self._jobs)
        except Exception as exc:  # noqa: BLE001 — GUI reload: show status, never crash
            self._clim.set_groups([])
            self._status.setText(f"cannot read: {exc}")
            self._update_render_enabled()
            return
        vids = list(dict.fromkeys(v for e in self._catalog for v in e.fields))
        self._clim.set_groups([(vid, volume_label(vid)) for vid in vids])
        self._tree.blockSignals(True)
        for e in self._catalog:
            top = QTreeWidgetItem([e.label + ("   · pinned" if e.note else "")])
            top.setData(0, Qt.ItemDataRole.UserRole, e.job_index)
            top.setFlags(top.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsAutoTristate)
            job_fields = self._jobs[e.job_index].get("fields") or None
            for vid in e.fields:
                child = QTreeWidgetItem([vid])
                child.setFlags(child.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                checked = job_fields is None or vid in job_fields
                child.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                top.addChild(child)
            self._tree.addTopLevelItem(top)
            top.setExpanded(True)
        self._tree.blockSignals(False)
        n_missing = len(self._jobs) - len(self._catalog)
        msg = f"{len(self._catalog)} job(s)"
        if n_missing:
            msg += f"; {n_missing} job(s) reference a slice not in this file"
        self._status.setText(msg)
        self._update_render_enabled()

    def select_all(self) -> None:
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            for j in range(top.childCount()):
                top.child(j).setCheckState(0, Qt.CheckState.Checked)

    def _on_item_changed(self, _item, _col) -> None:
        self._update_render_enabled()

    def _update_render_enabled(self) -> None:
        self._render_btn.setEnabled(bool(self._checked_jobs()))

    # -- selection → core -----------------------------------------------------
    def _checked_jobs(self) -> list[dict]:
        """Jobs to render: checked fields become the job's 'fields' override."""
        out = []
        for i in range(self._tree.topLevelItemCount()):
            top = self._tree.topLevelItem(i)
            vids = [
                top.child(j).text(0)
                for j in range(top.childCount())
                if top.child(j).checkState(0) == Qt.CheckState.Checked
            ]
            if not vids:
                continue
            ji = top.data(0, Qt.ItemDataRole.UserRole)
            out.append({**self._jobs[ji], "fields": vids})
        return out

    def render_selection(self, out_dir):
        """Render the checked jobs into *out_dir*; returns written paths."""
        res = _pr.render_replot(
            self._h5_path,
            self._checked_jobs(),
            self._style,
            self._clim.clim_by_group(),
            out_dir,
            dpi=int(self._dpi.value()),
        )
        self.written = [
            p
            for jr in res.jobs
            for p in ([jr.figure] if jr.figure else []) + list(jr.overviews) + list(jr.traces)
        ]
        self._last_result = res
        return self.written

    # -- slots ----------------------------------------------------------------
    def _on_render(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._status.setText("set an output dir")
            return
        err = self._clim.validate()
        if err:
            self._status.setText(err)
            return
        try:
            written = self.render_selection(out_dir)
        except Exception as exc:  # noqa: BLE001 — surface render errors in the status bar
            self._status.setText(f"render failed: {exc}")
            return
        res = self._last_result
        msg = f"wrote {len(written)} PNG(s) → {out_dir}"
        if res.skipped:
            msg += f"; skipped: {'; '.join(res.skipped)}"
        if res.notes:
            msg += f"; notes: {'; '.join(res.notes)}"
        self._status.setText(msg)

    def _on_browse_h5(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open oblique_slices.h5", "", "HDF5 (*.h5)")
        if path:
            self._file_edit.setText(path)
            self._reload()

    def _on_browse_out(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output directory")
        if path:
            self._out_pinned = True
            self._out_edit.setText(path)
```

Note `ProfileJobResult.overviews`/`traces` default types — grep the dataclass
first; if they default to `None` rather than `[]`, guard with `or []` in
`render_selection`.

- [ ] **Step 3: Quick import check**

Run: `QT_QPA_PLATFORM=offscreen python3 -c "import gui.widgets.profiles_replot as m; print(m.ProfilesReplotDialog)"`
Expected: prints the class, no import error.

- [ ] **Step 4: Commit**

```bash
git add gui/widgets/profiles_replot.py gui/widgets/clim_section.py gui/widgets/slice_replot.py
git commit -m "feat(gui): ProfilesReplotDialog (job/field tree + per-quantity clim)"
```

(Docs for the dialog land with the button wiring in Task 4 — one user-visible
feature, one docs update.)

---

### Task 4: Replot… button on the profiles view + smoke step [33] + docs

**Files:**
- Modify: `gui/stage_view.py` (button condition ~line 142; `_on_replot`
  dispatch ~line 446; new `_replot_profiles` next to `_replot_slices` ~line 502)
- Modify: `tests/gui_smoke.py` (new step [33]), `docs/Usage.md`, `docs/Codebase.md`

**Interfaces:**
- Consumes: Task 3's `ProfilesReplotDialog(h5_path, jobs, style=None,
  out_default="", parent=None)` / `.written` / `.select_all()` /
  `.render_selection(out_dir)`.
- Produces: profiles stage view gains `_replot_btn` (same attribute the other
  stages use — the smoke step relies on it).

- [ ] **Step 1: Wire the button.** In `gui/stage_view.py` (Read each region
first): change the condition at ~line 142 to
`if stage_name in ("slices", "strain", "mosaicity", "rocking", "profiles"):`
and update its comment. In `_on_replot` (~line 446), after the slices branch add:

```python
        if self._stage_name == "profiles":
            self._replot_profiles(vals, style)
            return
```

Add next to `_replot_slices`:

```python
    def _replot_profiles(self, vals: dict, style) -> None:
        """Open the profiles replot dialog (jobs from the form's Jobs JSON)."""
        import json as _json

        h5 = vals.get("consolidated_h5", "") or ""
        try:
            jobs = _json.loads(vals.get("jobs_json", "") or "[]")
        except (ValueError, TypeError):
            jobs = []
        if not isinstance(jobs, list):
            jobs = []

        from .widgets.profiles_replot import ProfilesReplotDialog  # imported on demand

        # out_default="" lets the dialog default the output beside the loaded h5.
        dlg = ProfilesReplotDialog(h5, jobs, style=style, out_default="", parent=self)
        dlg.exec()
        if dlg.written:
            self._log.append(
                f"Replotted {len(dlg.written)} PNG(s) → {os.path.dirname(dlg.written[0])}"
            )
            self._tabs.setCurrentWidget(self._log)
```

- [ ] **Step 2: Smoke step [33].** Read the end of `tests/gui_smoke.py` (step
[32] and the step-count assertion/print pattern — follow it exactly). Append a
step that: builds a tiny consolidated h5 in the smoke tmp dir (reuse the
existing smoke slices output if a prior step left one — check what step [26]
uses — else write one inline with h5py mirroring `_write_consolidated` from
`tests/test_stage_profiles.py`: two field groups, one `oblique_full` slice
group each), then:

```python
    # [33] ProfilesReplotDialog: opens from the profiles view, renders checked jobs.
    from gui.widgets.profiles_replot import ProfilesReplotDialog as _PRD

    profiles_view = ...  # fetch the profiles StageView the same way earlier steps fetch views
    assert profiles_view._replot_btn is not None, "profiles view missing _replot_btn"
    _jobs33 = [
        {"name": "oblique_full", "offset_um": 0.0, "start_uv": [-5, -3], "end_uv": [5, 3],
         "n_samples": 20, "width_pixels": 1, "fig_name": "smoke33"}
    ]
    _dlg33 = _PRD(_h5_33, _jobs33, style=None, out_default="")
    assert _dlg33._tree.topLevelItemCount() == 1
    assert _dlg33._render_btn.isEnabled()  # opens with fields checked
    _out33 = os.path.join(_tmp33, "replots")
    _written33 = _dlg33.render_selection(_out33)
    assert _written33 and all(os.path.exists(p) for p in _written33)
    assert not any(p.endswith(".csv") for p in os.listdir(_out33))
    print("[33] ProfilesReplotDialog: Replot… button wired; tree + render writes PNGs, no CSVs")
```

(`_h5_33`/`_tmp33`/view fetching are placeholders only in THIS plan snippet
because the smoke file's local variable names must be read from the file —
match the surrounding steps' actual patterns when writing the real step. The
assertions shown are the required content.) Update the smoke test's final
expected-step count (grep for `[32]`/total assertions).

- [ ] **Step 3: Run the smoke test**

Run: `python3 tests/gui_smoke.py`
Expected: `GUI SMOKE PASSED` with step [33] printed.

- [ ] **Step 4: Docs.** `docs/Usage.md`: add subsection "Replotting line
profiles" under the profiles stage (mirror the slices "Replotting slices
without re-running" section): button always enabled; dialog reads the form's
Slices file + Jobs (JSON); job→field checkbox tree (a job's own `fields` list
seeds the checks, else all); per-quantity colour limits (blank = stored, same
semantics as the slices replot); traces re-render but are unaffected by clim;
CSVs are never rewritten; output defaults to a timestamped `replots/<stamp>/`
beside the h5; pinned files work (substitution shown in the status line).
`docs/Codebase.md`: add `gui/widgets/profiles_replot.py` to the widgets list;
note `clim_section.volume_label` as the shared label helper; note the
stage_view `_replot_profiles` branch.

- [ ] **Step 5: Full verification + commit**

Run: `python3 -m pytest -q && ruff check . && ruff format --check . && python3 tests/gui_smoke.py`
Expected: suite green, ruff clean, smoke passes with [33].

```bash
git add gui/stage_view.py tests/gui_smoke.py docs/Usage.md docs/Codebase.md
git commit -m "feat(gui): Replot… button on profiles view + smoke [33] + docs"
```

---

### Task 5: Whole-branch review + merge

- [ ] **Step 1:** Run the verify-suite skill (all four checks, canonical line).
- [ ] **Step 2:** Whole-branch review (fable, xhigh) of `master..profiles-replot`
  against the spec — clim resolution parity with slices, run/replot drift, GUI
  thread-safety of the in-process render, docs coverage.
- [ ] **Step 3:** Fix findings (implementer/fix agents per CLAUDE.md tiering),
  re-verify, then merge with `git merge --no-ff profiles-replot` on master and
  delete the branch (no remote — no push).
