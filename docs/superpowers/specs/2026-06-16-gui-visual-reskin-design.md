# GUI visual reskin — Direction "Lab" + KIT green (light/dark)

**Date:** 2026-06-16
**Status:** approved design, ready for implementation plan
**Scope:** pure visual reskin of the PySide6 GUI. No layout, workflow, stage-order,
or behavior changes. `dfxm/` core is untouched.

## Goal

Give the DFXM pipeline GUI a deliberate, non-default look: soft/rounded "Lab"
direction, KIT corporate green as the single accent, in both a light and a dark
theme with a user toggle. The app currently runs on the platform-default Qt look
with ~8 ad-hoc inline `setStyleSheet` calls and no central theme; this introduces
one styling lever where none exists.

## Decisions (locked)

- **Direction:** "Lab" — cool light surfaces, 9px rounded corners, roomy spacing,
  monospace numeric fields.
- **Accent:** KIT-Grün. The brand's CMYK 100/0/60/0 maps to the official on-screen
  value **`#009682`** (RGB 0/150/130), *not* the naïve CMYK→RGB `#00FF66`. Light
  mode uses exact `#009682`. Dark mode uses a slightly brightened `#12a890` for
  legibility against dark panels (documented to the user).
- **Modes:** light + dark, user-toggleable; choice persisted across launches.
- **Toggle placement:** compact checkable ☀/☾ button in the left column, beside
  the existing "Publication style…" button. No new menu bar / toolbar.
- **Deploy target:** single Linux machine, but we still base on the **Fusion**
  style because it honors `QPalette`/QSS theming predictably.
- **Implementation approach:** global QSS stylesheet + `QPalette` on a Fusion base
  (vs. palette-only — too plain; vs. custom `QStyle` — overkill).

## Palette tokens

A `Palette` dataclass with two instances. Values:

| token | LIGHT | DARK | used for |
|---|---|---|---|
| `surface` | `#eef1f0` | `#15191b` | window/background |
| `panel` | `#ffffff` | `#1d2326` | content panels, inputs bg(light) |
| `ink` | `#1f2a27` | `#e7ecea` | primary text |
| `ink_muted` | `#62706b` | `#93a09b` | secondary text, muted rail rows |
| `border` | `#dde3e1` | `#2c3439` | borders, separators |
| `accent` | `#009682` | `#12a890` | buttons, focus ring, active-tab underline |
| `accent_strong` | `#00786a` | `#46c9b6` | accent text/links on the surface (contrast) |
| `accent_soft` | `#d8efe9` | `#163a34` | active-row / hover fills, focus-ring glow |
| `accent_on` | `#ffffff` | `#ffffff` | text on the accent button |
| `error` | `#b00020` | `#ef5a6a` | error labels, fail banner |
| `warning` | `#b06a00` | `#e0a23a` | warning banner |
| `success` | `#00786a` | `#46c9b6` | success banner / ✓ |
| `rail_bg` | `#f5f7f6` | `#181d20` | left navigation rail |
| `mpl_facecolor` | `#ffffff` | `#1d2326` | embedded matplotlib **display** canvas |
| `pv_background` | `#eef1f0` | `#15191b` | embedded pyvista 3-D background |

Exact hex may be tuned during implementation; the mockup approved these.

## Architecture

### `gui/theme.py` (new, Qt-only — must not be imported by `dfxm/`)

- `@dataclass(frozen=True) Palette` with the tokens above; module constants
  `LIGHT` and `DARK`.
- `build_qss(palette: Palette) -> str` — the global stylesheet. Covers: base
  widget colors, 9px radii, primary `QPushButton` (accent bg, white text) and a
  ghost/secondary variant — distinguished by a dynamic property
  `QPushButton[role="primary"]` (the stage "Run" button is tagged `role="primary"`;
  other buttons keep the default styling), `QLineEdit`/`QAbstractSpinBox`/`QComboBox` with accent
  focus ring, `QTabBar::tab:selected` accent underline, the `QListWidget` rail
  selection (accent_soft bg, accent_strong text), `QGroupBox` headers, scrollbars.
  Semantic colored widgets are addressed via **dynamic properties** so they
  restyle automatically when the stylesheet is rebuilt — e.g.
  `QLabel[role="error"] { color: <error>; }`, banner `QFrame[role="error|warning|success|info"]`.
- `apply_theme(app: QApplication, mode: str) -> Palette` — `app.setStyle("Fusion")`
  (once), build & set a `QPalette` from the tokens, `app.setStyleSheet(build_qss(p))`,
  set the application font. Returns the active palette.
- `class ThemeController(QObject)` — process singleton; holds `current: Palette`
  and a `mode` ("light"/"dark"); `themeChanged = Signal(object)`. `set_mode(mode)`
  calls `apply_theme`, stores it via `QSettings`, and emits `themeChanged`.

### Two refresh mechanisms (deliberate split)

- **Standard widgets** restyle for free: rebuilding the app stylesheet + palette
  on toggle re-evaluates every QSS rule, including the `[role=...]` dynamic
  properties. No per-widget wiring needed for colored labels/banners.
- **Embedded canvases** can't be reached by QSS, so they subscribe to
  `ThemeController.themeChanged`:
  - `MplCanvas.apply_theme(palette)`: set `figure.patch`/axes facecolor, tick,
    label, and spine colors — **display only**.
  - `PvCanvas.apply_theme(palette)`: `plotter.set_background(palette.pv_background)`
    guarded by `self._plotter is not None` (no-op when 3-D is unavailable).

### `gui/app.py`

Set `QApplication.setApplicationName("pipeline")` /
`setOrganizationName("dfxm")` (so `QSettings` has a stable home), read the saved
mode (default **light**) via `QSettings("dfxm","pipeline")`, call
`ThemeController.instance().set_mode(mode)` (which applies the theme) **before**
`window.show()`.

### `gui/main_window.py`

Add a compact checkable ☀/☾ `QPushButton` to the left column next to
"Publication style…"; toggling calls `ThemeController.set_mode(...)`. Replace the
hardcoded `QColor("#888888")` muted-concat brush with `ink_muted` from the active
palette (and refresh it on `themeChanged`, since `QListWidgetItem` foreground is
not QSS-reachable).

## Style consolidation (concrete edits)

Fold these into theme tokens / dynamic properties (one source of truth, dark-aware):

| file:line | current | becomes |
|---|---|---|
| `gui/main_window.py:82` | `QColor("#888888")` | `ink_muted` token (refresh on themeChanged) |
| `gui/overview_page.py:59,66` | `_CHIP_STYLE`, `_EXTERNAL_STYLE` | token-based / QSS object names |
| `gui/widgets/help_panel.py:29` | `#eef2fb` bg, `#4a6fd0` blue border | `accent_soft` bg, `accent` border (now green) |
| `gui/experiment_panel.py:109` | `color:#666` | `ink_muted` |
| `gui/experiment_panel.py:114` | `color:#b00020; italic` | `error` token (`role="error"`) |
| `gui/widgets/param_form.py:99` | bold group header | QSS `QLabel[role="group-header"]` |
| `gui/widgets/param_form.py:165` | `color:#b00020; bold` | `error` token |
| `gui/widgets/log_console.py:58` | `#b00020` on error | `error` token |
| `gui/stage_view.py:205` | banner style string | `role`-based banner QSS (error/warning/success/info) |

## Files

- **New:** `gui/theme.py`, `tests/test_theme.py`
- **Edit:** `gui/app.py`, `gui/main_window.py`, `gui/widgets/mpl_canvas.py`,
  `gui/widgets/pv_canvas.py`, `gui/overview_page.py`, `gui/widgets/help_panel.py`,
  `gui/experiment_panel.py`, `gui/widgets/param_form.py`,
  `gui/widgets/log_console.py`, `gui/stage_view.py`
- **Docs:** `docs/Usage.md`, `docs/Codebase.md`

## Testing

`tests/test_theme.py` (offscreen, following `tests/gui_smoke.py` /
`test_gui_viewers.py` patterns):

- `build_qss(LIGHT)` / `build_qss(DARK)` return non-empty `str`; light QSS
  contains `#009682`.
- `LIGHT` and `DARK` expose every token field and differ on `surface`/`ink`/`accent`.
- `ThemeController.set_mode("dark")` flips `current`/`mode` and emits
  `themeChanged` with the new palette (capture via signal spy).
- No unresolved `{...}` placeholders remain in the built QSS.
- Existing GUI smoke tests (`gui_smoke.py`, `test_gui_viewers.py`) stay green —
  proves no behavior/structure regression.

## Docs updates (repo contract)

- `docs/Usage.md` → "The main window": document the ☀/☾ light/dark toggle and that
  the choice persists.
- `docs/Codebase.md` → add `gui/theme.py` (`Palette`, `build_qss`, `apply_theme`,
  `ThemeController`) and the new `apply_theme` methods on `MplCanvas`/`PvCanvas`.

## Out of scope (YAGNI)

- No layout / information-architecture / workflow / stage-order changes.
- **No change to publication export.** Every `savefig(..., facecolor="white")`
  path (stages, `export_dialog`, `plotting`, `render`) stays white regardless of
  app theme; embedded-canvas theming is display-only. This is a hard invariant.
- No new dependencies.
- No "system/auto" theme detection — explicit light/dark only.

## Risks / watch items

- **Fusion restyles every widget** — sweep the whole app in both modes to catch
  anything unreadable.
- **matplotlib `NavigationToolbar2QT` icons** are fixed dark pixmaps; on a dark
  toolbar they may low-contrast. Mitigation: keep the toolbar strip light-ish, or
  accept default; verify and note.
- **`QListWidgetItem` foreground** is not QSS-reachable — must be refreshed
  explicitly on theme change (handled in `main_window`).
- **`QSettings`** location now depends on app/org name; harmless (no existing
  QSettings usage) but is new persisted state.
