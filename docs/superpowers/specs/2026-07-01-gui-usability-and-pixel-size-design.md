# Design — GUI usability polish + effective-pixel-size calculator

Date: 2026-07-01
Status: approved (brainstorming), pre-plan

Three independent changes, bundled because they are small and touch adjacent GUI
code. Each is its own phase in the implementation plan; they share no state.

---

## Feature 1 — Wider middle column + remembered window/splitter state

### Problem
- The middle column (parameter form + help) is much narrower than the
  Log/Results/Output panel. Users want the middle wider at the expense of the
  right panel.
- The middle/right width **resets when switching stages**, because each
  `StageView` owns its own inner `QSplitter` (`gui/stage_view.py`, sizes
  `[360, 600]`). There is no shared state.
- Nothing persists between runs: window size, position, maximized ("optimized")
  state, or any splitter widths.

### Decision
- **One shared inner-split width across all stages** (user preference: "make the
  width the same for all of them"). Dragging the splitter on any stage updates
  every stage and persists.
- Persist, via the existing `QSettings` (org `dfxm`, app `pipeline`, already used
  for `theme`):
  - **Window geometry + maximized state** — `QMainWindow.saveGeometry()` /
    `restoreGeometry()` (these encode size, position, and maximized).
  - **Top-level splitter** (left rail | stack) — `saveState()`/`restoreState()`.
  - **Shared inner (middle | right) splitter** width — one state value applied to
    every `StageView` splitter.

### Implementation sketch
- New module `gui/window_state.py` with a `WindowState` helper (Qt-side; the
  `dfxm/` core stays untouched). Responsibilities:
  - `register_stage_splitter(splitter)` — store the splitter, apply the saved
    (or default) shared sizes, and connect its `splitterMoved` to a handler that
    captures the new sizes, applies them to every other registered splitter, and
    writes them to `QSettings`.
  - `save(window, main_splitter)` / `restore(window, main_splitter)` for geometry
    + top-level splitter state.
  - Centralises the `QSettings` keys: `geometry`, `mainSplitter`, `stageSplitter`.
- `StageView` exposes its inner splitter (e.g. attribute `inner_splitter`) so the
  window can register it. `StageView` keeps a sensible default in case it is used
  standalone.
- `MainWindow`:
  - builds views, registers each view's inner splitter with `WindowState`,
  - restores geometry + top-level splitter state at the end of `__init__`,
  - overrides `closeEvent` to persist geometry + top-level splitter state
    (the shared inner width is already persisted live on drag).
- **New first-run default** for the inner split: ≈ **55/45 favouring the middle**
  (replaces today's right-heavy `[360, 600]`). Persistence takes over after the
  first drag, so the exact number only matters on a fresh profile.
- Guard against a `QSettings` value from an incompatible build (wrong widget
  count / corrupt state): restore calls are defensive — on failure, fall back to
  the coded defaults rather than raising.

### Tests
- A GUI-smoke test (headless `QApplication`): construct `MainWindow`, drag one
  stage's inner splitter (set sizes), assert another stage's inner splitter now
  reports the same sizes. Round-trip `WindowState` save/restore through a
  temporary `QSettings` scope.

---

## Feature 2 — "Compute from scan…" effective-pixel-size button

### Physics (from the beamline note, with the user's corrections)
Read motor positions from a raw (pre-darfix) scan's positioners group:
`mainx`, `obx`, `ffsel`, `ffz`, `lenssel`.

- CRL magnification: `M = mainx / obx − 1`
- Objective (far-field) from `ffsel`:
  - `ffsel = −60` → **2×**, base `3.25`
  - `ffsel = 0`  → **10×**, base `0.65`
  - anything else → error (set pixel size manually)
- Horizontal pixel size: `E_x = base / M`  (µm)
- Detector angle: `2θ = arctan(ffz / mainx)`
- Condenser auto-detected via `lenssel`:
  - `lenssel = 0` → condenser **in** → `E_y = E_x / sin(2θ)`
  - otherwise    → condenser **out** → `E_y = E_x`

(Corrections vs the original note: base is **3.25 / 0.65**, not 3.75 / 0.75.)

### Decision
- Live in the **Edit-experiment dialog** as a `Compute from scan…` button next to
  Pixel size X / Y. It reads a chosen raw scan `.h5`, fills both pixel-size
  fields, and reports the derived `M`, `2θ`, objective, and condenser state.

### Implementation sketch
- **Core (Qt-free), new `dfxm/common/pixel_size.py`:**
  - `@dataclass PixelSizeResult`: `pixel_size_x_um`, `pixel_size_y_um`,
    `magnification`, `two_theta_deg`, `objective` (`"2x"|"10x"`),
    `condenser_in: bool`, and the raw motor values used.
  - `compute_pixel_size(scan_h5, positioners_path, entry_suffix=".1")
    -> PixelSizeResult`. Opens the file, picks the first entry ending in
    `entry_suffix` (via `h5io.get_filtered_entries`), reads
    `f"{entry}/{positioners_path}"` with `h5io.read_positioners`, applies the
    formulas above.
  - Motor key names are documented module constants
    (`MAINX="mainx"`, `OBX="obx"`, `FFSEL="ffsel"`, `FFZ="ffz"`,
    `LENSSEL="lenssel"`). `ffsel`/`lenssel` matched with a small tolerance.
  - Raises `StageUserError(message, hint=...)` (from `dfxm.common.errors`) for:
    no matching entry, missing positioners group, missing motor, unrecognised
    `ffsel`, or a non-physical `M ≤ 0`.
- **GUI (`gui/experiment_panel.py`):** add the button to `ExperimentDialog`.
  On click: `QFileDialog.getOpenFileName` for `.h5`; call the core function with
  the dialog's current `positioners_path` and `entry_suffix` field values; on
  success `self._form.set_values({"pixel_size_x_um": ..., "pixel_size_y_um": ...})`
  and show an info line/box (`M`, `2θ°`, objective, condenser in/out); on
  `StageUserError` show a warning with `message` + `hint`.

### Tests
- `tests/` with a synthetic HDF5 fixture writing a positioners group:
  - 2× (`ffsel=−60`) and 10× (`ffsel=0`) each give the expected `M`, `E_x`;
  - condenser in (`lenssel=0`) applies `/sin(2θ)`, out (`lenssel≠0`) does not;
  - unrecognised `ffsel`, missing motor, and missing entry each raise
    `StageUserError`.

---

## Feature 3 — Hybrid help panel + richer hover tooltips

### Problem
- The bottom `HelpPanel` follows focus (`ParamForm.focusedParamChanged →
  HelpPanel.show_param`) but has **no revert path**: once a field is clicked it
  sticks on that field's help forever, even after clicking elsewhere, and the
  stage description never comes back.
- Per-field help exists only in the bottom panel and (already) as a bare-`help`
  tooltip.

### Decision (hybrid)
- Bottom panel shows the **stage description by default** and **resets to it every
  time a stage is opened**.
- While a field is focused, the panel shows that field's help (as today); when
  focus leaves all fields (to Run/Cancel/tabs/log, or a click on empty space),
  it **reverts to the stage description**.
- Per-field help is also available as a **richer hover tooltip** matching the
  panel (label + unit + ⚠ calibration + help), not just bare `help`.

### Implementation sketch
- `StageView.showEvent` → `self._help.show_idle()` so opening any stage always
  starts on the stage description.
- `ParamForm` focus tracking: connect to `QApplication.instance().focusChanged`
  (receiver is the `ParamForm`, so the connection is torn down with it). When the
  new focus widget is in `_param_for_widget` → emit `focusedParamChanged(param)`
  (existing signal); otherwise emit a new `focusCleared` signal. `HelpPanel`
  connects `focusCleared → show_idle`. (Replaces the current FocusIn-only event
  filter for this purpose.)
- Make blank areas of the param column focusable (`Qt.FocusPolicy.ClickFocus` on
  the left content container) so clicking empty space moves focus off the field
  and triggers the revert.
- Enrich tooltips: build one shared helper that renders a `Param` to the same
  rich text the `HelpPanel` uses, and set it as each editor/label's tooltip in
  `ParamForm`.

### Tests
- GUI-smoke: focusing a field shows its help; moving focus to the Run button (or
  emitting `focusCleared`) reverts to the stage description; showing a stage view
  resets the panel to idle.

---

## Documentation (same change, per CLAUDE.md contract)
- `docs/Usage.md`: window size/position/maximized + column widths are now
  remembered; the shared middle/right width; the `Compute from scan…` pixel-size
  button (what it reads and reports); the help-panel + hover-tooltip behaviour.
- `docs/Codebase.md`: new `gui/window_state.py` and `dfxm/common/pixel_size.py`;
  updated `HelpPanel`, `ParamForm` (new `focusCleared` signal / focus tracking),
  `ExperimentDialog` (compute button), `StageView` (`inner_splitter`,
  `showEvent`), `MainWindow` (persistence wiring).

## Non-goals
- No new configurable Experiment fields for the motor names (kept as documented
  constants).
- No 5× objective support (2×/10× only; other `ffsel` → manual).
- No per-stage independent widths (deliberately unified).
