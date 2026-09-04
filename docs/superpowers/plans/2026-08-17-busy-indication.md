# App-wide busy indication — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Spec:** `docs/superpowers/specs/2026-08-17-busy-indication-design.md` (approved 2026-08-17).
**Repo:** `darq_xray`. Runs AFTER the trace-autoscale/collisions plan (`2026-08-17-trace-autoscale-collisions-design.md`) is merged — rebase/verify the baseline first.

## Goal

The user can always tell when the app is working:

1. **Threaded figure-builder renders/exports** (`gui/figure_builder.py`) — non-blocking `render_now`/`export_now` on a `QThread`, latest-wins serialization, animated `BusyOverlay` spinner over the preview.
2. **Threaded replot batch renders** — the replot dialogs run their per-item render loops in a shared `BatchWorker`, with a determinate overlay (`{i}/{N}` + ETA) and cancel-after-current-item.
3. **Wait-cursor sweep** — every listed short synchronous load wrapped in a `busy_cursor(...)` context manager.
4. **Stage-run ETA** — `StageView` progress text gains `"{text} — ~{eta} left"` from an `EtaEstimator`.

Shared vocabulary: new `gui/widgets/busy.py` (Qt) + new `dfxm/common/eta.py` (Qt-free).

## Architecture

- **`dfxm/common/eta.py` (new, Qt-free):** `format_eta(elapsed_s, frac) -> str` and `class EtaEstimator` (monotonic-frac clamp, EMA smoothing over `(t, frac)` samples, quiet below 5 % / 2 s). Imported by `gui/widgets/busy.py` and `gui/stage_view.py`. Zero Qt, zero matplotlib.
- **`gui/widgets/busy.py` (new):** `BusyOverlay(QWidget)` (translucent input-swallowing child overlay; indeterminate rotating-arc spinner painted in `paintEvent`, KIT-green accent from `gui.theme.ThemeController.instance().palette`; determinate mode = `QProgressBar` + `"{done}/{total} — {eta}"` sub-label; optional Cancel button), `busy_cursor(text="", widget=None)` context manager, `keep_alive(worker)` (module-level registry pinning running `QThread`s until `finished`), `BatchWorker(QThread)` (per-item loop, `itemDone`/`batchFinished` signals, cooperative stop), `DialogBatchRunner(QObject)` (glues a `BatchWorker` + determinate overlay + ETA + button gating onto a dialog).
- **`gui/figure_builder.py`:** private `_ComposeWorker(QThread)` renders/exports a JSON-snapshotted recipe (via `recipe_to_json`/`recipe_from_json`, `base_dir=None` — internal recipes hold absolute h5 paths, confirmed in `dfxm/compose/recipe.py::_rel_path/_resolve_path`) against the window's loader cache. `render_recipe`/`export_recipe` are canvas-less and Qt-free (verified: `dfxm/compose/render.py` imports only matplotlib `Figure` + dfxm; no `rcParams` mutation anywhere under `dfxm/`), so worker-thread rendering is safe. Generation counter = latest-wins; at most one worker; one pending request slot. `FigureCanvasQTAgg` attachment stays on the GUI thread in `_show_figure` (unchanged).
- **Replot dialogs:** `ReplotDialog`/`SliceReplotDialog` split their batch into per-selection items (verified output-identical: `strain.render_replot`, `mosaicity.render_replot`, `rocking.render_replot`, `slices.render_replot` each loop selections with **no cross-selection state**). `ProfilesReplotDialog` runs the whole checked-jobs batch as **one** item — `profiles.render_replot` shares `used_stems` (same-name stem dedup) and `trace_deferred` (per-run shared trace margins) across jobs, so splitting would change filenames and trace geometry. This is a deliberate, documented deviation from the spec's uniform per-item wording; the spec's own "no risky changes" principle wins. The synchronous `render_selection(out_dir)` methods stay **unchanged** (test/smoke/back-compat seam); only the `_on_render` button path is threaded.
- **Tests:** new `tests/qt_helpers.py` (importable by pytest files AND `tests/gui_smoke.py`, which puts the repo root on `sys.path`) provides `wait_builder_idle` / `render_and_wait` / `export_and_wait` / `wait_batch_idle`.

## Tech stack

PySide6 (QThread/Signal/QPainter), matplotlib explicit-`Figure` API only, h5py (worker-side reads only, one open at a time), pytest offscreen Qt (`QT_QPA_PLATFORM=offscreen`).

## Global constraints

- **Qt-free `dfxm/`** — `EtaEstimator`/`format_eta` go in `dfxm/common/eta.py` with zero Qt imports. Never import PySide6/pyvista under `dfxm/`.
- **QThread rules** — workers never touch Qt widgets; results are delivered ONLY via signals connected to **bound methods of a QObject** (a plain lambda receiver executes in the emitting thread — forbidden for anything touching widgets). A running `QThread` must never be garbage-collected: pin via `keep_alive`. Never `thread.wait()` on the GUI thread in production code (tests may).
- **Plotting** — explicit `matplotlib.figure.Figure`; never `pyplot` or `matplotlib.use(...)`.
- **Docs contract** — every behaviour change updates `docs/Usage.md` AND `docs/Codebase.md` **in the same task**.
- **Suite** — `DISPLAY= python3 -m pytest -q` for full-suite runs (this shell's DISPLAY has a broken GL stack that kills the vtk tests). Smoke = `python3 tests/gui_smoke.py`; the 3-D step `[41]` stays last. Baseline: record fresh counts after the trace-autoscale plan merges (slightly above 949 passed / 13 skipped); every task ends green at baseline + its own new tests.
- **Lint** — `ruff check . && ruff format .` (format also runs via the Write/Edit hook).
- **No git remote** — no pull/push/PR.
- **Anchors** — the trace-autoscale plan edits `gui/figure_builder.py::_build_compose_form` and `dfxm/compose/render.py`'s `render_recipe` tail. This plan does not touch `dfxm/compose/` at all, and in `figure_builder.py` anchors only on function defs (`def render_now`, `def export_now`, `def refresh_data`, `def closeEvent`, `def _build_center_pane`, `def _build_right_pane`, `def load_recipe_file`), never on neighbouring lines. **Read every target region before editing** (em-dash/indent hazards in `hint=` strings).
- **Error-message parity** — the async paths must reproduce today's exact user-facing strings (`"cannot render: {exc}  Hint: {hint}"`, `"render failed: {exc}"`, `"export failed: …"`, `"wrote N PNG(s) → {out}"`); existing tests assert substrings of them.

---

## Task 1 — Qt-free ETA core: `dfxm/common/eta.py`

**Files**
- Create `dfxm/common/eta.py`
- Create `tests/test_common_eta.py`
- Edit `docs/Codebase.md` (new `#### eta.py` subsection under `### dfxm/common — shared primitives`, inserted after the `#### roi.py (new)` block — find it with grep, don't assume line numbers)
- Edit `docs/Usage.md` (one sentence in `### The stage panel` under Core concepts: progress readouts may show a `~… left` estimate once a run/batch is >5 % done and >2 s old)

**Interfaces (Produces)**
```python
def format_eta(elapsed_s: float, frac: float) -> str
    # "" when frac < 0.05, elapsed_s < 2.0, or frac >= 1.0;
    # else "~{s} s left" (< 90 s remaining) or "~{m} min left"

class EtaEstimator:
    def __init__(self, clock=time.monotonic) -> None
    def reset(self) -> None            # new t0; forgets samples
    def update(self, frac: float) -> None   # clamps frac to [0,1]; ignores regressions; EMA-smooths
    def eta_text(self) -> str          # "" when not estimable, else formatted remaining
```

**Steps**
- [ ] Write `tests/test_common_eta.py` (no Qt anywhere):
```python
"""Qt-free ETA helpers (dfxm/common/eta.py)."""

from dfxm.common.eta import EtaEstimator, format_eta


def test_format_eta_quiet_below_thresholds():
    assert format_eta(1.0, 0.5) == ""  # < 2 s elapsed
    assert format_eta(10.0, 0.01) == ""  # < 5 % done
    assert format_eta(10.0, 1.0) == ""  # finished — nothing left to estimate


def test_format_eta_seconds_and_minutes():
    assert format_eta(10.0, 0.5) == "~10 s left"
    assert format_eta(60.0, 0.25) == "~3 min left"  # 180 s remaining -> minutes


def test_estimator_estimates_and_smooths():
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    assert est.eta_text() == ""
    t[0] = 10.0
    est.update(0.5)
    assert est.eta_text() == "~10 s left"
    t[0] = 12.0
    est.update(0.6)  # raw remaining = 8 s; EMA(0.7*10 + 0.3*8) = 9.4 -> ~9 s
    assert est.eta_text() == "~9 s left"


def test_estimator_ignores_regressing_frac():
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.5)
    before = est.eta_text()
    t[0] = 11.0
    est.update(0.4)  # monotonic clamp: a regression never poisons the estimate
    assert est.eta_text() == before


def test_estimator_reset_forgets_everything():
    t = [0.0]
    est = EtaEstimator(clock=lambda: t[0])
    t[0] = 10.0
    est.update(0.5)
    assert est.eta_text() != ""
    est.reset()
    assert est.eta_text() == ""
```
- [ ] Run it: `DISPLAY= python3 -m pytest -q tests/test_common_eta.py` — must FAIL (module missing).
- [ ] Write `dfxm/common/eta.py`:
```python
"""Qt-free ETA estimation for progress readouts.

Shared by the GUI's busy overlay/progress text (gui/widgets/busy.py,
gui/stage_view.py) but importable and testable without Qt.
"""

from __future__ import annotations

import time

_MIN_FRAC = 0.05  # below this, any extrapolation is noise
_MIN_ELAPSED_S = 2.0
_EMA_KEEP = 0.7  # weight of the previous smoothed estimate


def _format_remaining(remaining_s: float) -> str:
    if remaining_s >= 90.0:
        return f"~{max(1, int(round(remaining_s / 60.0)))} min left"
    return f"~{max(1, int(round(remaining_s)))} s left"


def format_eta(elapsed_s: float, frac: float) -> str:
    """Human remaining-time estimate; "" when too early/noisy to say."""
    if frac < _MIN_FRAC or elapsed_s < _MIN_ELAPSED_S or frac >= 1.0:
        return ""
    return _format_remaining(elapsed_s * (1.0 - frac) / frac)


class EtaEstimator:
    """Smoothed remaining-time estimate from monotonic (t, frac) samples.

    ``update(frac)`` records a progress sample against the injected *clock*
    (``time.monotonic`` by default; tests inject a fake). Fractions are
    clamped to [0, 1] and a regressing fraction is ignored, so a jittery
    reporter can never produce a negative or exploding estimate. Estimates
    are EMA-smoothed; ``eta_text()`` stays "" until >= 5 % done and >= 2 s
    elapsed (mirroring :func:`format_eta`).
    """

    def __init__(self, clock=time.monotonic) -> None:
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        self._t0 = self._clock()
        self._frac = 0.0
        self._smoothed: float | None = None

    def update(self, frac: float) -> None:
        frac = min(1.0, max(0.0, float(frac)))
        if frac < self._frac:
            return
        self._frac = frac
        elapsed = self._clock() - self._t0
        if frac < _MIN_FRAC or elapsed < _MIN_ELAPSED_S or frac >= 1.0:
            return
        raw = elapsed * (1.0 - frac) / frac
        self._smoothed = (
            raw if self._smoothed is None else _EMA_KEEP * self._smoothed + (1.0 - _EMA_KEEP) * raw
        )

    def eta_text(self) -> str:
        if self._smoothed is None or self._frac >= 1.0:
            return ""
        return _format_remaining(self._smoothed)
```
- [ ] `DISPLAY= python3 -m pytest -q tests/test_common_eta.py` — green. Verify zero Qt: `grep -i pyside dfxm/common/eta.py` → nothing.
- [ ] Docs: `docs/Codebase.md` — new `#### eta.py` subsection (module purpose, both public names, the 5 %/2 s thresholds, who consumes it); `docs/Usage.md` — the one-sentence stage-panel note.
- [ ] `ruff check . && ruff format .`; full `DISPLAY= python3 -m pytest -q` green; commit.

---

## Task 2 — `gui/widgets/busy.py`: BusyOverlay + busy_cursor + keep_alive

**Files**
- Create `gui/widgets/busy.py`
- Create `tests/test_gui_busy.py`
- Edit `docs/Codebase.md` (`### gui/widgets/` section: add `busy.py` entry)
- Edit `docs/Usage.md` (short "Busy indication" paragraph under `## Core concepts` → after `### The stage panel`: what the spinner overlay and wait cursor mean)

**Interfaces (Produces)**
```python
class BusyOverlay(QWidget):
    cancelRequested: Signal  # emitted by the optional Cancel button
    def __init__(self, host: QWidget, cancellable: bool = False) -> None
    @property
    def active(self) -> bool                     # isVisible()
    def start(self, text: str) -> None           # indeterminate spinner + text; covers host; swallows input
    def set_text(self, text: str) -> None
    def set_progress(self, done: int, total: int, eta_text: str = "") -> None  # determinate mode
    def stop(self) -> None                       # hide + stop the spin timer

def busy_cursor(text: str = "", widget=None)     # context manager; WaitCursor + optional widget.setText(text)
def keep_alive(worker) -> None                   # pin a QThread until its finished signal fires
```

**Steps**
- [ ] Write `tests/test_gui_busy.py`:
```python
"""BusyOverlay / busy_cursor / keep_alive (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

_app = QApplication.instance() or QApplication([])

from gui.widgets.busy import BusyOverlay, busy_cursor  # noqa: E402


def _host():
    host = QWidget()
    host.resize(300, 200)
    host.show()
    return host


def test_overlay_start_stop_visibility_and_text():
    host = _host()
    ov = BusyOverlay(host)
    assert not ov.active
    ov.start("Rendering…")
    assert ov.active and ov._label.text() == "Rendering…"
    assert ov.geometry() == host.rect()
    ov.stop()
    assert not ov.active and not ov._timer.isActive()


def test_overlay_determinate_progress_line():
    ov = BusyOverlay(_host())
    ov.start("Rendering…")
    ov.set_progress(2, 5, "~10 s left")
    assert ov._bar.value() == 2 and ov._bar.maximum() == 5
    assert ov._sub.text() == "2/5 — ~10 s left"
    ov.set_progress(3, 5)
    assert ov._sub.text() == "3/5"
    ov.stop()


def test_overlay_cancel_button_only_when_cancellable():
    ov = BusyOverlay(_host())
    ov.start("x")
    assert not ov._cancel_btn.isVisible()
    ov.stop()
    ov2 = BusyOverlay(_host(), cancellable=True)
    hits = []
    ov2.cancelRequested.connect(lambda: hits.append(1))
    ov2.start("x")
    assert ov2._cancel_btn.isVisible()
    ov2._cancel_btn.click()
    assert hits == [1]
    ov2.stop()


def test_overlay_tracks_host_resize():
    host = _host()
    ov = BusyOverlay(host)
    ov.start("x")
    host.resize(500, 400)
    _app.processEvents()
    assert ov.geometry() == host.rect()
    ov.stop()


def test_busy_cursor_sets_and_restores():
    assert _app.overrideCursor() is None
    with busy_cursor():
        assert _app.overrideCursor() is not None
    assert _app.overrideCursor() is None


def test_busy_cursor_restores_on_exception_and_writes_text():
    from PySide6.QtWidgets import QLabel

    label = QLabel("")
    with pytest.raises(RuntimeError):
        with busy_cursor("loading…", widget=label):
            assert label.text() == "loading…"
            raise RuntimeError("boom")
    assert _app.overrideCursor() is None
```
- [ ] Run — FAILS (module missing).
- [ ] Write `gui/widgets/busy.py`:
```python
"""Shared busy-indication vocabulary: overlay spinner, wait cursor, thread pins.

Everything user-visible about "the app is working" lives here (spec
2026-08-17-busy-indication-design.md): :class:`BusyOverlay` (animated
indeterminate spinner or determinate progress over a host widget),
:func:`busy_cursor` (honest wait-cursor for short synchronous blocks) and
:func:`keep_alive` (pins running QThreads so they are never garbage-collected
mid-flight). Batch machinery (:class:`BatchWorker`/:class:`DialogBatchRunner`)
is added by the replot-threading tasks.
"""

from __future__ import annotations

from contextlib import contextmanager

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..theme import ThemeController

# Running QThreads pinned here until their finished signal fires — a running
# QThread that gets garbage-collected aborts the process.
_LIVE_WORKERS: set = set()


def keep_alive(worker) -> None:
    """Pin *worker* (a QThread) until it finishes."""
    _LIVE_WORKERS.add(worker)
    worker.finished.connect(lambda w=worker: _LIVE_WORKERS.discard(w))


@contextmanager
def busy_cursor(text: str = "", widget=None):
    """Wait-cursor (and optional status text) around a short synchronous block.

    Forces one ``processEvents()`` so the cursor/text actually appear BEFORE
    the block runs; always restores the cursor, including on raise. The status
    text is deliberately left for the call site's completion message to
    overwrite.
    """
    app = QApplication.instance()
    if app is not None:
        app.setOverrideCursor(Qt.CursorShape.WaitCursor)
    if widget is not None and text:
        widget.setText(text)
    if app is not None:
        app.processEvents()
    try:
        yield
    finally:
        if app is not None:
            app.restoreOverrideCursor()


class BusyOverlay(QWidget):
    """Translucent, input-swallowing overlay over a host widget.

    Indeterminate mode (``start``): a rotating KIT-green arc painted in
    :meth:`paintEvent`, driven by a 50 ms QTimer, plus a one-line text label.
    Determinate mode (``set_progress``): a progress bar plus a
    ``"{done}/{total} — {eta}"`` sub-label. ``stop()`` in EVERY finish path —
    success, error, cancel, close — is the call sites' contract.
    """

    cancelRequested = Signal()

    def __init__(self, host: QWidget, cancellable: bool = False) -> None:
        super().__init__(host)
        self._host = host
        self._cancellable = cancellable
        self._angle = 0
        self._determinate = False
        self._timer = QTimer(self)
        self._timer.setInterval(50)
        self._timer.timeout.connect(self._spin)

        self._label = QLabel("", self)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub = QLabel("", self)
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bar = QProgressBar(self)
        self._bar.setFixedWidth(220)
        self._bar.setTextVisible(False)
        self._bar.hide()
        self._cancel_btn = QPushButton("Cancel", self)
        self._cancel_btn.clicked.connect(self.cancelRequested.emit)
        self._cancel_btn.hide()

        lay = QVBoxLayout(self)
        lay.addStretch(2)
        lay.addSpacing(56)  # room for the painted arc above the label
        lay.addWidget(self._label)
        lay.addWidget(self._bar, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addWidget(self._sub)
        lay.addWidget(self._cancel_btn, 0, Qt.AlignmentFlag.AlignHCenter)
        lay.addStretch(3)

        host.installEventFilter(self)
        self.hide()

    @property
    def active(self) -> bool:
        return self.isVisible()

    def start(self, text: str) -> None:
        self._determinate = False
        self._label.setText(text)
        self._sub.setText("")
        self._bar.hide()
        self._cancel_btn.setVisible(self._cancellable)
        self.setGeometry(self._host.rect())
        self.raise_()
        self.show()
        self._timer.start()

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def set_progress(self, done: int, total: int, eta_text: str = "") -> None:
        self._determinate = True
        self._bar.setRange(0, max(1, int(total)))
        self._bar.setValue(int(done))
        self._bar.show()
        self._sub.setText(f"{done}/{total} — {eta_text}" if eta_text else f"{done}/{total}")
        self.update()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    # -- internals --------------------------------------------------------
    def _spin(self) -> None:
        self._angle = (self._angle - 12) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 — Qt override
        p = ThemeController.instance().palette
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor(p.surface)
        bg.setAlpha(190)
        painter.fillRect(self.rect(), bg)
        if not self._determinate:
            pen = QPen(QColor(p.accent), 4)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            r = 18
            cx, cy = self.width() // 2, self.height() // 2 - 44
            painter.drawArc(cx - r, cy - r, 2 * r, 2 * r, self._angle * 16, 100 * 16)
        painter.end()

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 — Qt override
        if obj is self._host and event.type() == QEvent.Type.Resize and self.isVisible():
            self.setGeometry(self._host.rect())
        return False

    # swallow interaction with the host while active
    def mousePressEvent(self, event) -> None:  # noqa: N802 — Qt override
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 — Qt override
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 — Qt override
        event.accept()

    def wheelEvent(self, event) -> None:  # noqa: N802 — Qt override
        event.accept()
```
- [ ] `DISPLAY= python3 -m pytest -q tests/test_gui_busy.py` — green. (Offscreen note: if `isVisible()` on the cancel button is unreliable offscreen before the overlay is shown, assert `ov._cancel_btn.isVisibleTo(ov)` instead — decide from the actual failure, don't guess.)
- [ ] Docs: `Codebase.md` `busy.py` entry (all four public names + the "stop() in every finish path" contract); `Usage.md` busy-indication paragraph.
- [ ] `ruff check . && ruff format .`; full suite green; commit.

---

## Task 3 — Figure-builder async render (worker + latest-wins + overlay) and test migration

The riskiest task. Only the **render** path moves here; `export_now` stays synchronous until Task 4, so the suite is green at this boundary.

**Files**
- Edit `gui/figure_builder.py` (anchors: `def __init__`, `def _build_center_pane`, `def _build_right_pane`, `def refresh_data`, `def render_now`; do NOT touch `_build_compose_form` internals or anything under `dfxm/compose/`)
- Create `tests/qt_helpers.py`
- Edit `tests/test_gui_figure_builder.py`
- Edit `tests/gui_smoke.py` (steps `[37]` and `[40]` render lines only)
- Edit `docs/Usage.md` (`## Figure builder` section: preview renders in the background under a spinner; Refresh/Export disabled while rendering; rapid edits coalesce — latest wins)
- Edit `docs/Codebase.md` (figure_builder entry: `_ComposeWorker`, generation counter, pending slot, overlay)

**Interfaces**

Consumes: `BusyOverlay`, `keep_alive` (Task 2); `recipe_to_json`/`recipe_from_json` (existing); `render_recipe(recipe, style_overrides=None, *, loader_cache=None) -> ComposeResult` (existing).

Produces (in `gui/figure_builder.py`):
```python
class _ComposeWorker(QThread):
    resultReady = Signal(int, str, object, object)  # (generation, kind, payload | None, exception | None)
    def __init__(self, generation: int, kind: str, recipe_json: str, cache: dict, out_dir: str | None) -> None
    # kind: "render" -> payload ComposeResult; "export" -> payload (paths, ComposeResult)

# FigureBuilderWindow additions:
#   self._worker: _ComposeWorker | None ; self._generation: int ; self._pending: tuple[str, str | None] | None
#   self._overlay: BusyOverlay ; self._last_outcome ; self._export_btn (promoted from local)
#   render_now(self) -> None   # CHANGED: no return value; async
```

Produces (in `tests/qt_helpers.py`):
```python
def wait_builder_idle(w, timeout_s: float = 30.0) -> None
def render_and_wait(w, timeout_s: float = 30.0)   # -> ComposeResult | None (old render_now contract)
def export_and_wait(w, timeout_s: float = 30.0)   # -> ComposeResult | None (Task 4 uses it)
```

**Steps**
- [ ] Read `gui/figure_builder.py` in full first (it has changed since this plan: the trace-autoscale plan added a checkbox in `_build_compose_form`). Grep `render_now(` across `tests/` to enumerate ALL call sites (baseline list, by test name: `test_delete_row_purges_nested_panel_defs_and_gutter_renders`, `test_render_now_populates_preview_and_notes`, `test_render_error_lands_in_notes_bar_not_crash`, `test_cache_survives_file_deletion_until_refresh`, `test_click_preview_selects_outline_node`, `test_render_now_clears_canvas_when_no_panels_left`, `test_figure2_authored_through_window_methods` in `tests/test_gui_figure_builder.py`, plus smoke `[37]`/`[40]` — migrate every hit the grep finds, including any the autoscale plan added).
- [ ] Create `tests/qt_helpers.py`:
```python
"""Shared Qt test helpers, importable by pytest files AND tests/gui_smoke.py.

Not a test module (no ``test_`` prefix). gui_smoke.py puts the repo root on
sys.path, so both worlds import this as ``tests.qt_helpers``.
"""

from __future__ import annotations

import time

from PySide6.QtWidgets import QApplication


def wait_builder_idle(w, timeout_s: float = 30.0) -> None:
    """Drive the event loop until the figure-builder window has no live or
    pending compose worker."""
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_s
    while w._worker is not None or w._pending is not None:
        assert time.monotonic() < deadline, "compose worker did not finish in time"
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()  # flush any just-queued result delivery


def render_and_wait(w, timeout_s: float = 30.0):
    """Async twin of the old synchronous ``render_now()`` contract: request a
    render, wait for it, return the ComposeResult of THIS request (None on
    error or nothing-to-render — the notes bar carries the explanation)."""
    w.render_now()
    wait_builder_idle(w, timeout_s)
    return w._last_outcome


def export_and_wait(w, timeout_s: float = 30.0):
    """Same, for ``export_now()`` (async from the busy-indication Task 4 on)."""
    w.export_now()
    wait_builder_idle(w, timeout_s)
    return w._last_outcome
```
- [ ] Write the NEW tests first (append to `tests/test_gui_figure_builder.py`; they fail until the worker exists):
```python
# -- busy indication: async render worker (latest-wins) ----------------------
from tests.qt_helpers import render_and_wait, wait_builder_idle  # noqa: E402


def test_async_render_shows_overlay_then_clears(tmp_path):
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w.render_now()
    # synchronous guarantee: the overlay is up and buttons gated BEFORE return
    assert w._overlay.active and not w._refresh_btn.isEnabled()
    assert not w._export_btn.isEnabled()
    wait_builder_idle(w)
    assert not w._overlay.active
    assert w._refresh_btn.isEnabled() and w._export_btn.isEnabled()
    assert w._result is not None and w._canvas is not None


def test_latest_wins_two_rapid_renders_one_canvas(tmp_path, monkeypatch):
    import threading

    import dfxm.compose.render as _render

    real = _render.render_recipe
    release = threading.Event()
    calls: list[str] = []

    def gated(recipe, *a, **k):
        calls.append(recipe.name)
        if len(calls) == 1:
            release.wait(30)  # hold render #1 until #2 has been requested
        return real(recipe, *a, **k)

    monkeypatch.setattr(_render, "render_recipe", gated)
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    shows: list = []
    orig_show = w._show_figure
    monkeypatch.setattr(w, "_show_figure", lambda fig: (shows.append(fig), orig_show(fig))[1])
    w.render_now()  # worker 1 (gen 1) — parked in gated()
    w.recipe().name = "second"
    w.render_now()  # gen 2 — queued behind worker 1
    release.set()
    wait_builder_idle(w)
    assert calls == ["untitled", "second"]  # serialized, both ran
    assert len(shows) == 1  # worker 1's stale result was DROPPED, never attached
    assert w._last_outcome is not None and w._canvas.figure is w._result.figure
```
- [ ] Implement in `gui/figure_builder.py`:
  - Imports: add `from PySide6.QtCore import QThread, Signal` to the existing QtCore import line; add `from .widgets.busy import BusyOverlay, keep_alive` beside the other `.widgets` imports.
  - Module-level worker class (place directly above `class FigureBuilderWindow`):
```python
class _ComposeWorker(QThread):
    """One render/export of a JSON-snapshotted recipe, off the GUI thread.

    Holds NO Qt widgets — only the JSON snapshot, the shared loader cache
    (serialized: at most one worker at a time, the GUI thread rebinds rather
    than mutates it) and an output dir for exports. Results/exceptions are
    delivered via ``resultReady`` connected to a BOUND METHOD of the window,
    so delivery is queued onto the GUI thread.
    """

    resultReady = Signal(int, str, object, object)  # (generation, kind, payload, exception)

    def __init__(
        self, generation: int, kind: str, recipe_json: str, cache: dict, out_dir: str | None
    ) -> None:
        super().__init__()
        self._generation = generation
        self._kind = kind
        self._recipe_json = recipe_json
        self._cache = cache
        self._out_dir = out_dir

    def run(self) -> None:  # worker thread — no Qt widgets in here
        from dfxm.compose.render import export_recipe, render_recipe

        try:
            recipe = recipe_from_json(self._recipe_json)
            if self._kind == "export":
                payload = export_recipe(recipe, self._out_dir, loader_cache=self._cache)
            else:
                payload = render_recipe(recipe, loader_cache=self._cache)
        except Exception as exc:  # noqa: BLE001 — delivered to the GUI as data
            self.resultReady.emit(self._generation, self._kind, None, exc)
            return
        self.resultReady.emit(self._generation, self._kind, payload, None)
```
  - `__init__` (after the `self._result = None` group): `self._worker: _ComposeWorker | None = None`, `self._generation = 0`, `self._pending: tuple[str, str | None] | None = None`, `self._last_outcome = None`; after `self._preview_host`/layout exist: `self._overlay = BusyOverlay(self._preview_host)`.
  - `_build_right_pane`: promote the local `export_btn` to `self._export_btn` (same two lines, renamed).
  - Replace `render_now` and add the worker plumbing:
```python
    def render_now(self) -> None:
        """Request an async preview render of the current recipe (latest-wins).

        Non-blocking: snapshots the recipe through JSON, runs
        ``render_recipe`` on a worker thread, and attaches the canvas in the
        result slot. At most one worker runs; a request made while one is
        running is queued (one slot, newest wins) and any superseded worker's
        result is dropped via the generation counter. The old synchronous
        return-the-result contract lives on for tests as
        ``tests.qt_helpers.render_and_wait``.
        """
        if not self._recipe.panels:
            self._clear_canvas()
            self._notes_label.setText("add panels to preview")
            self._last_outcome = None
            return
        self._request_work("render", None)

    def _request_work(self, kind: str, out_dir: str | None) -> None:
        self._generation += 1
        if self._worker is not None:
            self._pending = (kind, out_dir)
            return
        self._start_worker(kind, out_dir)

    def _start_worker(self, kind: str, out_dir: str | None) -> None:
        self._overlay.start("Exporting…" if kind == "export" else "Rendering…")
        self._refresh_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        worker = _ComposeWorker(
            self._generation, kind, recipe_to_json(self._recipe), self._cache, out_dir
        )
        worker.resultReady.connect(self._on_worker_result)  # bound method -> queued to GUI thread
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        keep_alive(worker)
        worker.start()

    def _on_worker_result(self, gen: int, kind: str, payload, error) -> None:
        if gen != self._generation:
            return  # superseded (latest-wins) or window closed — drop silently
        from dfxm.common.errors import StageUserError

        if error is not None:
            verb = "export failed" if kind == "export" else (
                "cannot render" if isinstance(error, StageUserError) else "render failed"
            )
            hint = ""
            if isinstance(error, StageUserError) and error.hint:
                hint = f"  Hint: {error.hint}"
            self._notes_label.setText(f"{verb}: {error}{hint}")
            self._last_outcome = None
            return
        if kind == "export":
            paths, res = payload
            out = os.path.dirname(paths[0]) if paths else ""
            notes = f"; {'; '.join(res.notes)}" if res.notes else ""
            self._notes_label.setText(f"wrote {len(paths)} file(s) → {out}{notes}")
            self._last_outcome = res
            return
        self._show_figure(payload.figure)
        self._result = payload
        self._notes_label.setText("; ".join(payload.notes) if payload.notes else "")
        self._last_outcome = payload

    def _on_worker_finished(self) -> None:
        self._worker = None
        if self._pending is not None:
            kind, out_dir = self._pending
            self._pending = None
            self._start_worker(kind, out_dir)
            return
        self._overlay.stop()
        self._refresh_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
```
    Note the error-path detail: today's synchronous code shows `"cannot render: …  Hint: …"` for `StageUserError` and `"render failed: …"` for everything else — the `verb` selection above reproduces that exactly; on error `self._result`/canvas are left untouched (parity with today: the stale preview stays visible).
  - `refresh_data`: change `self._cache.clear()` to `self._cache = {}` with the comment `# rebind, never clear() — an in-flight worker may still be reading the old dict`.
- [ ] Migrate the existing tests in `tests/test_gui_figure_builder.py` — mechanical, per call site: `res = w.render_now()` → `res = render_and_wait(w)`; bare `w.render_now()` used for its side effect → `render_and_wait(w)`. In `test_cache_survives_file_deletion_until_refresh`, `w.refresh_data()` stays followed by `res3 = render_and_wait(w)` (the helper's request supersedes the refresh-triggered one against the same fresh cache — equivalent outcome). Update the autouse fixture to also drain workers:
```python
@pytest.fixture(autouse=True)
def _no_leaked_debounce_timers():
    yield
    while _live_windows:
        w = _live_windows.pop()
        w._debounce.stop()  # never let a pending render fire into a later test
        w._pending = None
        if w._worker is not None:
            w._worker.wait(30000)  # tests may join; production code must not
        _app.processEvents()  # deliver the worker's queued result/finished
        w.deleteLater()
    _app.processEvents()
```
- [ ] Migrate smoke: in step `[37]` replace `res = fb.render_now()` with:
```python
    from tests.qt_helpers import wait_builder_idle

    fb.render_now()
    assert fb._overlay.active, "busy overlay should cover the preview during a render"
    wait_builder_idle(fb)
    assert not fb._overlay.active
    res = fb._last_outcome
```
  and in step `[40]` replace `_res40 = fb.render_now()` with `fb.render_now()` + `wait_builder_idle(fb)` + `_res40 = fb._last_outcome` (import already done in [37]). Leave both steps' `export_now` lines untouched (still synchronous until Task 4).
- [ ] `DISPLAY= python3 -m pytest -q tests/test_gui_figure_builder.py` green; `python3 tests/gui_smoke.py` green through `[41]`.
- [ ] Docs (`Usage.md` figure-builder + `Codebase.md` figure_builder entry, incl. the changed `render_now` return contract).
- [ ] `ruff check . && ruff format .`; full suite green; commit.

---

## Task 4 — Figure-builder async export + closeEvent worker handling

**Files**
- Edit `gui/figure_builder.py` (anchors: `def export_now`, `def closeEvent`)
- Edit `tests/test_gui_figure_builder.py`
- Edit `tests/gui_smoke.py` (step `[37]` export lines)
- Edit `docs/Usage.md` + `docs/Codebase.md` (export threading + close semantics)

**Interfaces** — Consumes Task 3's `_request_work`/`_on_worker_result` (the `"export"` branch already exists there). `export_now(self) -> None` becomes async; `closeEvent` additionally discards in-flight work.

**Steps**
- [ ] Migrate the two export tests first: in `test_export_now_writes_files` and `test_export_now_unexpected_error_reports_to_notes_bar_not_crash`, replace `w.export_now()` with `export_and_wait(w)` (add `export_and_wait` to the existing `tests.qt_helpers` import). They still pass against the synchronous implementation (the helper is a superset) — run to confirm, then keep.
- [ ] Add the close-with-live-worker test:
```python
def test_close_with_live_worker_drops_result_never_attaches(tmp_path, monkeypatch):
    import threading

    import dfxm.compose.render as _render

    real = _render.render_recipe
    release = threading.Event()

    def gated(recipe, *a, **k):
        release.wait(30)
        return real(recipe, *a, **k)

    monkeypatch.setattr(_render, "render_recipe", gated)
    w = _win()
    w.add_panels(_obl_recipe_panels(tmp_path))
    w.render_now()
    assert w._worker is not None
    w._dirty = False
    assert w.close()  # returns immediately — never joins the thread on the GUI thread
    assert w._pending is None and not w._debounce.isActive()
    release.set()
    w._worker.wait(30000)  # test-only join so monkeypatch outlives the worker
    _app.processEvents()
    assert w._canvas is None  # generation was bumped on close: result dropped
```
- [ ] Rewrite `export_now`:
```python
    def export_now(self) -> None:
        """Ask for an export directory, then run ``export_recipe`` on the
        shared compose worker (spinner text "Exporting…"). Export re-renders
        rather than reusing the preview result — simplest correct thing; the
        result slot reports "wrote N file(s) → dir" or the failure text."""
        out = QFileDialog.getExistingDirectory(self, "Export directory")
        if not out:
            return
        self._request_work("export", out)
```
  (The `StageUserError` export message: today's synchronous export shows `"export failed: {exc}  Hint: {hint}"` — Task 3's `_on_worker_result` `verb` logic already picks `"export failed"` for `kind == "export"` regardless of exception type and appends the hint; verify against the old code text and keep parity.)
- [ ] Extend `closeEvent`: in BOTH accept paths (the early not-dirty return and the tail), alongside the existing `self._debounce.stop()`, add:
```python
        self._pending = None
        self._generation += 1  # any in-flight worker result is now stale — dropped on arrival
```
  with a comment that the worker holds no Qt objects, keeps running to completion under `keep_alive`, and its queued delivery hits the generation guard (or auto-disconnects if the window is deleted). Never `wait()` here.
- [ ] Migrate smoke `[37]` export: replace `fb.export_now()` inside the try/finally with `fb.export_now()` followed (still inside the `try`) by `wait_builder_idle(fb)`; the existing `assert os.path.exists(...)`/notes assertions stay.
- [ ] `DISPLAY= python3 -m pytest -q tests/test_gui_figure_builder.py` + smoke green; docs; ruff; full suite; commit.

---

## Task 5 — `BatchWorker` + `DialogBatchRunner` in busy.py

**Files**
- Edit `gui/widgets/busy.py` (append; anchor: end of file)
- Edit `tests/test_gui_busy.py` (append)
- Edit `tests/qt_helpers.py` (append `wait_batch_idle`)
- Edit `docs/Codebase.md` (busy.py entry grows)

**Interfaces (Produces)**
```python
class BatchWorker(QThread):
    itemDone = Signal(int, int)          # (done, total) — emitted after each finished item
    batchFinished = Signal(list, str)    # (written_paths, error_text; "" on success/cancel)
    def __init__(self, items: list, fn) -> None   # fn(item) -> list[str]; runs on the worker thread
    def request_stop(self) -> None       # cooperative: stops AFTER the current item
    # attrs: written: list[str]; cancelled: bool

class DialogBatchRunner(QObject):
    def __init__(self, dialog: QWidget, buttons: tuple) -> None
    @property
    def running(self) -> bool
    def start(self, items: list, fn, on_finished, text: str = "Rendering…") -> None
        # on_finished(written: list[str], error: str, cancelled: bool) -> None, called on the GUI thread
    def request_cancel(self) -> None
```

**Steps**
- [ ] Append tests to `tests/test_gui_busy.py`:
```python
def _run_batch(worker):
    worker.start()
    assert worker.wait(20000)
    _app.processEvents()


def test_batch_worker_runs_items_and_reports():
    from gui.widgets.busy import BatchWorker

    ticks, finishes = [], []
    w = BatchWorker([1, 2, 3], lambda i: [f"p{i}"])
    w.itemDone.connect(lambda d, t: ticks.append((d, t)))
    w.batchFinished.connect(lambda paths, err: finishes.append((paths, err)))
    _run_batch(w)
    assert ticks == [(1, 3), (2, 3), (3, 3)]
    assert finishes == [(["p1", "p2", "p3"], "")]
    assert not w.cancelled


def test_batch_worker_cancel_stops_after_current_item():
    from gui.widgets.busy import BatchWorker

    seen: list[int] = []
    holder: list = []

    def fn(i):
        seen.append(i)
        holder[0].request_stop()
        return [f"p{i}"]

    w = BatchWorker([1, 2, 3], fn)
    holder.append(w)
    results: list = []
    w.batchFinished.connect(lambda paths, err: results.append((paths, err)))
    _run_batch(w)
    assert seen == [1] and w.cancelled
    assert results == [(["p1"], "")]


def test_batch_worker_error_carries_partial_written_and_hint():
    from dfxm.common.errors import StageUserError
    from gui.widgets.busy import BatchWorker

    def fn(i):
        if i == 2:
            raise StageUserError("bad item", hint="fix it")
        return [f"p{i}"]

    results: list = []
    w = BatchWorker([1, 2, 3], fn)
    w.batchFinished.connect(lambda paths, err: results.append((paths, err)))
    _run_batch(w)
    assert results == [(["p1"], "bad item — fix it")]


def test_dialog_batch_runner_overlay_buttons_and_finish():
    from PySide6.QtWidgets import QPushButton

    from gui.widgets.busy import DialogBatchRunner

    host = _host()
    btn = QPushButton("Render", host)
    runner = DialogBatchRunner(host, (btn,))
    done: list = []
    runner.start([1, 2], lambda i: [f"p{i}"], lambda w, e, c: done.append((w, e, c)))
    assert runner.running and runner._overlay.active and not btn.isEnabled()
    import time as _t

    deadline = _t.monotonic() + 20
    while runner.running:
        assert _t.monotonic() < deadline
        _app.processEvents()
        _t.sleep(0.01)
    _app.processEvents()
    assert done == [(["p1", "p2"], "", False)]
    assert not runner._overlay.active and btn.isEnabled()
```
- [ ] Run — fails. Append to `gui/widgets/busy.py` (new imports: `QObject`, `QThread` from QtCore; `EtaEstimator` from `dfxm.common.eta` — module top):
```python
class BatchWorker(QThread):
    """Per-item batch on a worker thread: ``fn(item) -> list[str]`` per item.

    Emits ``itemDone(done, total)`` after each item and ``batchFinished
    (written, error_text)`` once — error_text "" on success AND on cancel
    (``cancelled`` distinguishes). ``request_stop()`` is cooperative: the
    current item always completes. Exceptions are formatted with a
    StageUserError hint when present and carried as data, with the partial
    ``written`` list preserved (those files are really on disk).
    """

    itemDone = Signal(int, int)
    batchFinished = Signal(list, str)

    def __init__(self, items: list, fn) -> None:
        super().__init__()
        self._items = list(items)
        self._fn = fn
        self._stop = False
        self.written: list[str] = []
        self.cancelled = False

    def request_stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # worker thread — no Qt widgets in here
        total = len(self._items)
        err = ""
        try:
            for i, item in enumerate(self._items):
                if self._stop:
                    self.cancelled = True
                    break
                self.written.extend(self._fn(item))
                self.itemDone.emit(i + 1, total)
        except Exception as exc:  # noqa: BLE001 — delivered to the dialog as data
            hint = getattr(exc, "hint", "")
            err = f"{exc} — {hint}" if hint else str(exc)
        self.batchFinished.emit(list(self.written), err)


class DialogBatchRunner(QObject):
    """Owns one batch run for a dialog: cancellable determinate BusyOverlay
    over the dialog, ETA from EtaEstimator, button gating, and GUI-thread
    delivery (it is a QObject parented to the dialog, so the worker's signals
    queue onto the GUI thread). Single-item batches keep the indeterminate
    spinner — a 0/1 progress bar with an ETA is noise."""

    def __init__(self, dialog: QWidget, buttons: tuple) -> None:
        super().__init__(dialog)
        self._buttons = tuple(buttons)
        self._overlay = BusyOverlay(dialog, cancellable=True)
        self._overlay.cancelRequested.connect(self.request_cancel)
        self._eta = EtaEstimator()
        self._worker: BatchWorker | None = None
        self._on_finished_cb = None

    @property
    def running(self) -> bool:
        return self._worker is not None

    def start(self, items: list, fn, on_finished, text: str = "Rendering…") -> None:
        if self._worker is not None:
            return
        self._on_finished_cb = on_finished
        for b in self._buttons:
            b.setEnabled(False)
        self._eta.reset()
        self._overlay.start(text)
        if len(items) > 1:
            self._overlay.set_progress(0, len(items), "")
        worker = BatchWorker(items, fn)
        worker.itemDone.connect(self._on_item_done)  # bound methods -> queued to GUI thread
        worker.batchFinished.connect(self._on_batch_finished)
        self._worker = worker
        keep_alive(worker)
        worker.start()

    def request_cancel(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()
            self._overlay.set_text("Cancelling — finishing current item…")

    def _on_item_done(self, done: int, total: int) -> None:
        self._eta.update(done / total)
        if total > 1:
            self._overlay.set_progress(done, total, self._eta.eta_text())

    def _on_batch_finished(self, written: list, error: str) -> None:
        worker, self._worker = self._worker, None
        cancelled = bool(worker.cancelled) if worker is not None else False
        self._overlay.stop()  # stop() in EVERY finish path: success, error, cancel
        for b in self._buttons:
            b.setEnabled(True)
        if self._on_finished_cb is not None:
            self._on_finished_cb(written, error, cancelled)
```
- [ ] Append to `tests/qt_helpers.py`:
```python
def wait_batch_idle(dialog, timeout_s: float = 60.0) -> None:
    """Drive the event loop until *dialog*._batch (a DialogBatchRunner) is idle."""
    app = QApplication.instance()
    deadline = time.monotonic() + timeout_s
    while dialog._batch.running:
        assert time.monotonic() < deadline, "replot batch did not finish in time"
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
```
- [ ] `DISPLAY= python3 -m pytest -q tests/test_gui_busy.py` green; docs (`Codebase.md`); ruff; full suite; commit.

---

## Task 6 — Threaded batch in the generic `ReplotDialog` (+ smoke)

**Files**
- Edit `gui/widgets/replot_dialog.py` (anchors: `def __init__` tail, `def _on_render`; `render_selection` stays byte-identical)
- Edit `tests/test_gui_replot_dialog.py` (append)
- Edit `tests/gui_smoke.py` (extend step `[31]`)
- Edit `docs/Usage.md` (the three "Replotting … without re-running" sections for strain/mosaicity/rocking: progress bar `{i}/{N}` + ETA + Cancel-after-current-item) and `docs/Codebase.md` (replot_dialog entry)

**Interfaces** — Consumes `DialogBatchRunner` (Task 5). Per-item fn: one selection tuple `(key, idxs)` → `self._render_fn(h5, [sel], style, clim, roi, out_dir) -> list[str]` (verified output-identical to the whole-batch call for strain/mosaicity/rocking). Dialog gains `self._batch: DialogBatchRunner` and `reject()` gating.

**Steps**
- [ ] Append test (reuse the file's existing dialog-construction helpers — read the file first):
```python
def test_on_render_runs_batch_with_overlay_and_status(tmp_path):
    from tests.qt_helpers import wait_batch_idle

    dlg = ...  # build exactly as test_replot_dialog_collects_selection_clim_roi does
    dlg.select_all()
    dlg._out_edit.setText(str(tmp_path / "out"))
    dlg._on_render()
    assert dlg._batch.running and dlg._batch._overlay.active
    assert not dlg._render_btn.isEnabled()
    wait_batch_idle(dlg)
    assert not dlg._batch._overlay.active and dlg._render_btn.isEnabled()
    assert dlg.written and all(os.path.exists(p) for p in dlg.written)
    assert "wrote" in dlg._status.text()


def test_reject_while_running_cancels_instead_of_closing(tmp_path):
    ...  # start _on_render as above, then immediately:
    dlg.reject()
    assert dlg.isVisible() is not False or dlg._batch.running  # dialog did not close mid-run
    wait_batch_idle(dlg)
    dlg.reject()  # now it closes normally
```
  (The `...` lines are construction boilerplate the implementer copies verbatim from the first existing test in the same file — it builds a small h5 + catalog/render fns; do not invent a new fixture. For the reject test, drive with a real multi-selection so the batch is running when `reject()` fires; if it completes too fast offscreen, wrap the render fn with a `threading.Event`-gated monkeypatch as in Task 3's latest-wins test.)
- [ ] Implement in `replot_dialog.py`:
  - Imports: `from .busy import DialogBatchRunner`.
  - `__init__` tail (after `self._skipped`): `self._batch = DialogBatchRunner(self, (self._render_btn,))` and keep a `self._close_btn` reference by promoting the local `close_btn` variable → pass `(self._render_btn, self._close_btn)` as the gated buttons.
  - Replace `_on_render`:
```python
    def _on_render(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._status.setText("set an output dir")
            return
        sels = self._selections()
        if not sels:
            self._status.setText("nothing selected")
            return
        err = self._clim.validate()
        if err:
            self._status.setText(err)
            return
        # Snapshot EVERYTHING on the GUI thread; the per-item fn is Qt-free.
        h5, style, render_fn = self._h5_path, self._style, self._render_fn
        clim, roi = self._clim.clim_by_group(), self._roi()
        self._last_out_dir = out_dir

        def _one(sel):
            return render_fn(h5, [sel], style, clim, roi, out_dir)

        self._batch.start(sels, _one, self._on_batch_done)

    def _on_batch_done(self, written: list, error: str, cancelled: bool) -> None:
        self.written = written  # partial results are real files — always record them
        if error:
            self._status.setText(f"render failed: {error}")
            return
        msg = f"wrote {len(written)} PNG(s) → {self._last_out_dir}"
        if cancelled:
            msg = "cancelled — " + msg
        if self._skipped:
            msg += f"; skipped {len(self._skipped)} combo(s)"
        self._status.setText(msg)

    def reject(self) -> None:  # noqa: D401 — Qt override
        """Close gates on a running batch: first Esc/Close requests cancel."""
        if self._batch.running:
            self._batch.request_cancel()
            return
        super().reject()
```
- [ ] Extend smoke step `[31]` (after its existing `render_selection` call, before the step's `print`):
```python
    from tests.qt_helpers import wait_batch_idle

    _dlg31._on_render()
    assert _dlg31._batch.running and _dlg31._batch._overlay.active, "replot overlay missing"
    wait_batch_idle(_dlg31)
    assert not _dlg31._batch._overlay.active
    assert "wrote" in _dlg31._status.text()
```
  (verify step `[31]` sets `_out_edit`; it renders into the same out dir again — harmless overwrite).
- [ ] `DISPLAY= python3 -m pytest -q tests/test_gui_replot_dialog.py` + smoke green; docs; ruff; full suite; commit.

---

## Task 7 — Threaded batch in `SliceReplotDialog` and `ProfilesReplotDialog`

**Files**
- Edit `gui/widgets/slice_replot.py` (anchors: `def __init__` tail, `def _on_render`)
- Edit `gui/widgets/profiles_replot.py` (same anchors)
- Edit `tests/test_gui_slice_replot.py` (append one batch test, same shape as Task 6's)
- Create-or-extend a profiles-dialog test: `tests/test_gui_profiles_replot.py` does not exist — the profiles dialog is currently covered by smoke `[33]` only; add the batch test to a new small `tests/test_gui_profiles_replot.py` reusing smoke `[33]`'s h5/jobs construction pattern (read `tests/gui_smoke.py` around step `[33]` for it)
- Edit `docs/Usage.md` (`#### Replotting slices without re-running` + `#### Replotting line profiles`: progress/ETA/cancel for slices; single-spinner note for profiles) and `docs/Codebase.md` (both widget entries)

**Interfaces** — Consumes `DialogBatchRunner`. Slice per-item fn: `(vid, sname, plane_idxs)` → `_sl.render_replot(h5, [sel], style, clims, out_dir, roi=roi)`. Profiles: ONE item = the whole checked-jobs list → `_pr.render_replot(h5, jobs, style, clims, out_dir, dpi=dpi, params=params)` (single item because `used_stems` dedup + per-run shared trace margins are per-call state — splitting changes filenames and trace geometry).

**Steps**
- [ ] Tests first (slice test mirrors Task 6's overlay/status test using this file's existing `_make`-style helper; profiles test builds the dialog cold from a synthetic oblique_slices.h5 + one job, calls `_on_render`, waits via `wait_batch_idle`, asserts written files + "wrote" status + overlay cleared). Run — fail.
- [ ] `slice_replot.py`: identical pattern to Task 6 — `self._batch = DialogBatchRunner(self, (self._render_btn, self._close_btn))` (promote `close_btn`), `_on_render` snapshots `h5/style/clims/roi/out_dir` and starts `self._batch.start(sels, _one, self._on_batch_done)` with
```python
        def _one(sel):
            return _sl.render_replot(h5, [sel], style, clims, out_dir, roi=roi)
```
  `_on_batch_done` + `reject()` gating copied from Task 6 (same strings). `render_selection` untouched.
- [ ] `profiles_replot.py`: `self._batch = DialogBatchRunner(self, (self._render_btn, self._close_btn))`; `_on_render` becomes:
```python
    def _on_render(self) -> None:
        out_dir = self._out_edit.text().strip()
        if not out_dir:
            self._status.setText("set an output dir")
            return
        err = self._clim.validate()
        if err:
            self._status.setText(err)
            return
        jobs = self._checked_jobs()
        h5, style, params = self._h5_path, self._style, self._params
        clims, dpi = self._clim.clim_by_group(), int(self._dpi.value())
        self._last_out_dir = out_dir
        result_box: list = []

        def _whole_batch(_jobs):
            # ONE item: profiles' stem dedup + shared trace margins are
            # per-call state — never split this batch (plan Task 7 note).
            res = _pr.render_replot(h5, _jobs, style, clims, out_dir, dpi=dpi, params=params)
            result_box.append(res)  # plain attr/list append: GIL-safe, read only after finish
            return [
                p
                for jr in res.jobs
                for p in ([jr.figure] if jr.figure else []) + list(jr.overviews) + list(jr.traces)
            ]

        self._result_box = result_box
        self._batch.start(
            [jobs], _whole_batch, self._on_batch_done, text=f"Rendering {len(jobs)} job(s)…"
        )

    def _on_batch_done(self, written: list, error: str, cancelled: bool) -> None:
        self.written = written
        if error:
            self._status.setText(f"render failed: {error}")
            return
        res = self._result_box[0] if self._result_box else None
        self._last_result = res
        msg = f"wrote {len(written)} PNG(s) → {self._last_out_dir}"
        if res is not None and res.skipped:
            msg += f"; skipped: {'; '.join(res.skipped)}"
        if res is not None and res.notes:
            msg += f"; notes: {'; '.join(res.notes)}"
        self._status.setText(msg)
```
  plus the same `reject()` gating. (Note: `BatchWorker` already formats `StageUserError` hints into the error string — parity with `_fmt_error`.) `render_selection` untouched.
- [ ] Run new tests + `tests/test_gui_slice_replot.py` + smoke (steps `[26]`/`[30]`/`[33]` exercise the untouched sync path) — green; docs; ruff; full suite; commit.

---

## Task 8 — Wait-cursor sweep

**Files** (every site listed exactly; wrap the named statements only — read each region first)
1. `gui/stage_view.py::_on_pick_line` — wrap the `dlg = LinePickerDialog(...)` try-body statement: `with busy_cursor("Loading slice planes…"):` (the construction opens the h5 and builds the canvas).
2. `gui/stage_view.py::_on_jobs_from_marks` — wrap `marks = _sl.read_marks(h5)` AND, inside the per-mark loop, the `dlg = LinePickerDialog(...)` statement.
3. `gui/stage_view.py::_on_mark_planes` — wrap `dlg = MarkPlanesDialog(h5, parent=self)` (construction reads catalog + marks + first slice).
4. `gui/widgets/mark_planes.py::_on_slice_changed` — wrap `self._browser.open_slice(sname)`.
5. `gui/widgets/mark_planes.py::_on_save` — wrap ONLY the `for sname in sorted(...)` write loop (inside its existing `try`), i.e. `with busy_cursor():` around the loop — message boxes must show with a normal cursor.
6. `gui/widgets/roi_picker.py::_load_current` — wrap `arr, sx, sy = self._previews[idx][1]()` with `busy_cursor("loading preview…", widget=self._readout)` (covers Pick ROI everywhere: stage forms, both replot dialogs, experiment panel).
7. `gui/experiment_panel.py::_on_initialize_from_data` — replace the manual `QApplication.setOverrideCursor/restoreOverrideCursor` block (and its local `Qt`/`QApplication` imports) with `with busy_cursor(): detections = self._detect(vals)`.
8. `gui/figure_builder.py::load_recipe_file` — wrap the file read + `recipe_from_json` lines (only those two) with `busy_cursor("Opening recipe…", widget=self._notes_label)`.
9. `gui/widgets/viewer3d_window.py::load_and_render` — wrap ONLY `self.loaded = self._spec.load()` with `busy_cursor("Loading volume…", widget=self._status)`; the GL scene build/`rebuild()` below stays untouched (spec: placeholder/label behaviour preserved).

Plus docs: `docs/Usage.md` (extend the Task 2 "Busy indication" paragraph with "loads and saves show a wait cursor") and `docs/Codebase.md` (one line per touched module is NOT needed — instead note in the busy.py entry that these nine sites consume `busy_cursor`).

**Interfaces** — Consumes `busy_cursor` only. Imports: `from .widgets.busy import busy_cursor` (stage_view, figure_builder, experiment_panel uses `from .widgets.busy import busy_cursor`), `from .busy import busy_cursor` (mark_planes, roi_picker, viewer3d_window).

**Steps**
- [ ] Batch-read all nine regions first (one Read per file covering the sites).
- [ ] Apply the wraps site by site; no logic changes, no reordering of existing statements; existing `try/except` shapes preserved (busy_cursor sits INSIDE the `try` where one exists, so error paths still report as today).
- [ ] Verify by grep: `grep -n "busy_cursor" gui/stage_view.py gui/widgets/mark_planes.py gui/widgets/roi_picker.py gui/experiment_panel.py gui/figure_builder.py gui/widgets/viewer3d_window.py` → 10 hits (site 2 has two).
- [ ] `DISPLAY= python3 -m pytest -q` — the existing dialog/picker tests exercise every wrapped path (e.g. `tests/test_gui_mark_planes.py`, `tests/test_gui_roi_picker.py`, `tests/test_gui_experiment_init.py`, `tests/test_gui_viewer3d.py`); the cursor CM itself is covered by Task 2. Then smoke `[1]`-`[41]` green (steps `[26]`/`[38]`/`[41]` walk sites 3-5 and 9).
- [ ] Docs; ruff; commit.

---

## Task 9 — Stage-run ETA in `StageView` + final verification

**Files**
- Edit `gui/stage_view.py` (anchors: the top import block, `def __init__` progress-widget region, `def _on_run`, `def _handle`)
- Create `tests/test_gui_stage_eta.py`
- Edit `docs/Usage.md` (`### The stage panel`: the progress text shows `— ~… left` once estimable; resets each run) and `docs/Codebase.md` (stage_view entry: `EtaEstimator` fed from `_on_progress`/`_handle`)

**Interfaces** — Consumes `EtaEstimator` (Task 1) and the existing `Progress(frac: float, text: str)` dataclass from `dfxm/runner.py`. Produces: `StageView._eta: EtaEstimator`; progress text format `f"{msg.text} — {eta}"` where `eta = self._eta.eta_text()` (already includes `~… left`), plain `msg.text` while not estimable. Progress bar itself unchanged.

**Steps**
- [ ] Write `tests/test_gui_stage_eta.py`:
```python
"""StageView progress text gains an ETA (offscreen Qt, synthetic Progress msgs)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.eta import EtaEstimator  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402
from dfxm.runner import Progress  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import StageView  # noqa: E402


def _view_with_fake_clock():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    t = [0.0]
    view._eta = EtaEstimator(clock=lambda: t[0])
    return view, t


def test_progress_text_plain_before_estimable():
    view, _t = _view_with_fake_clock()
    view._handle(Progress(0.02, "warming up"))
    assert view._progress_text.text() == "warming up"


def test_progress_text_gains_eta_once_estimable():
    view, t = _view_with_fake_clock()
    view._handle(Progress(0.02, "warming up"))
    t[0] = 10.0
    view._handle(Progress(0.5, "stacking layers"))
    assert view._progress_text.text() == "stacking layers — ~10 s left"
    assert view._progress.value() == 50  # bar itself unchanged


def test_run_start_resets_estimator():
    view, t = _view_with_fake_clock()
    t[0] = 10.0
    view._handle(Progress(0.5, "x"))
    assert "left" in view._progress_text.text()
    view._eta.reset()  # what _on_run does — the wiring itself is asserted below
    view._handle(Progress(0.5, "x"))  # elapsed 0 since reset -> not estimable yet
    assert view._progress_text.text() == "x"
```
- [ ] Run — fails (no `_eta`, no suffix). Implement in `gui/stage_view.py`:
  - Import: `from dfxm.common.eta import EtaEstimator` in the dfxm import block.
  - `__init__` (next to the progress widgets): `self._eta = EtaEstimator()`.
  - `_on_run` (immediately after `self._progress_text.setText("")`): `self._eta.reset()`.
  - `_handle`, `Progress` branch — replace the text lines:
```python
        if isinstance(msg, Progress):
            self._log.set_progress(msg.frac, msg.text)
            self._progress.setValue(max(0, min(100, int(round(msg.frac * 100)))))
            self._eta.update(msg.frac)
            if msg.text:
                eta = self._eta.eta_text()
                self._progress_text.setText(f"{msg.text} — {eta}" if eta else msg.text)
                self._log.append(f"  [{msg.frac * 100:5.1f}%] {msg.text}")
```
  - Verify by reading the diff that `_on_run`'s `self._eta.reset()` line is present (the third test only proves reset-behaviour, not the wiring — the wiring is one greppable line: `grep -n "_eta.reset" gui/stage_view.py`).
- [ ] `DISPLAY= python3 -m pytest -q tests/test_gui_stage_eta.py` green.
- [ ] Docs; ruff.
- [ ] **Final whole-plan verification:** `DISPLAY= python3 -m pytest -q` (expect baseline + all new tests from Tasks 1-9, 13 skips, no new warnings), `ruff check . && ruff format .` clean, `python3 tests/gui_smoke.py` prints `[1]`-`[41]` with `[41]` last and `GUI SMOKE PASSED`. Walk the spec-coverage table below and confirm every row lands on real code. Commit.

---

## Spec coverage

| Spec section | Where in this plan |
|---|---|
| §1 `BusyOverlay` (spinner, determinate mode, resize-follow, input swallow) | Task 2 |
| §1 `busy_cursor` (override cursor, text, pre-block repaint, finally-restore) | Task 2 |
| §1 `format_eta` / `EtaEstimator` in Qt-free `dfxm/common/eta.py` | Task 1 |
| §2 threaded `render_now` (JSON recipe snapshot, cache handed to worker, canvas attached on GUI thread) | Task 3 |
| §2 latest-wins serialization (generation counter, one pending slot, stale result dropped) | Task 3 |
| §2 overlay over preview host, Refresh/Export gated, error → notes bar with today's text | Tasks 3-4 |
| §2 `export_now` through the same worker ("Exporting…", re-renders) | Task 4 |
| §2 debounce kept; `closeEvent` stops timer + discards in-flight work, never joins | Tasks 3 (kept) + 4 (close) |
| §3 shared `BatchWorker` + per-item loop moved off the GUI thread | Task 5 (mechanism) |
| §3 determinate overlay `{i}/{N}` + ETA, Render/Close gated, cancel-after-item | Tasks 5-7 (profiles = single-item batch, justified: `used_stems` + shared trace margins are per-call state) |
| §4 wait-cursor sweep (9 exact sites incl. 3-D viewer h5-read-only) | Task 8 |
| §5 stage-run ETA in `StageView._handle`/`_on_progress` path, reset on run start, bar unchanged | Task 9 |
| Error handling: worker exceptions as signals, today's message text, `stop()` in every finish path, cursor always restored | Tasks 2-7 (asserted in tests) |
| Testing: Qt-free eta; overlay; fb worker success/error/latest-wins; batch itemDone/cancel; StageView ETA; busy_cursor exception | Tasks 1, 2, 3, 5, 9 |
| Testing: `render_and_wait` migration of every synchronous `render_now()` test | Tasks 3-4 (+ `tests/qt_helpers.py`) |
| Testing: smoke — fb step waits on async render, overlay appears/clears; one replot batch with overlay; 3-D `[41]` stays last | Tasks 3, 4, 6, 9 |
| Docs in the same change (`Usage.md` + `Codebase.md`) | every task |
