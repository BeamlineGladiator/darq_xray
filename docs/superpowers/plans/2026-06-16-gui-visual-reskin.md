# GUI Visual Reskin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reskin the PySide6 GUI with the "Lab" direction (soft/rounded/roomy) and KIT-Grün accent, in light + dark with a persisted toggle — no layout, workflow, or behavior changes.

**Architecture:** A new Qt-only `gui/theme.py` is the single source of truth: light/dark `Palette` dataclasses, a `build_qss(palette)` global stylesheet, `apply_theme(app, mode)`, and a process-singleton `ThemeController` that emits `themeChanged`. Standard widgets restyle for free when the app stylesheet/palette are re-set (semantic colors via QSS dynamic `role` properties); the two embedded canvases (matplotlib, pyvista) can't be reached by QSS so they subscribe to `themeChanged`. Persistence (QSettings) lives in `app.py`/`main_window.py`, keeping `ThemeController` free of I/O and unit-testable.

**Tech Stack:** Python 3.10+, PySide6 (Fusion style + QPalette + QSS), matplotlib (embedded canvas display only), pyvista/pyvistaqt (3-D background), pytest (offscreen).

> **Refinement vs. spec:** the spec sketched `ThemeController.set_mode` doing the `QSettings` write. For testability we keep `ThemeController` I/O-free; the QSettings read happens in `app.py` at startup and the write in `main_window`'s toggle handler. Behavior is identical.

---

## File structure

- **Create** `gui/theme.py` — `Palette`, `LIGHT`/`DARK`, `PALETTES`, `build_qss`, `apply_theme`, `_qpalette`, `ThemeController`. One responsibility: all theming.
- **Create** `tests/test_theme.py` — pure unit tests (palettes, QSS, controller, apply_theme).
- **Modify** `gui/app.py` — set org/app name, read saved mode, apply theme before show.
- **Modify** `gui/main_window.py` — ☀/☾ toggle button, persistence, muted-row refresh.
- **Modify** `gui/widgets/mpl_canvas.py` — `apply_theme(palette)` + subscribe.
- **Modify** `gui/widgets/pv_canvas.py` — `apply_theme(palette)` + subscribe.
- **Modify** `gui/overview_page.py` — chips/external → `role` properties.
- **Modify** `gui/widgets/help_panel.py` — frame via QSS, cal-warning red follows theme.
- **Modify** `gui/experiment_panel.py` — summary/notes → `role` properties.
- **Modify** `gui/widgets/param_form.py` — group header + calibration label → `role`.
- **Modify** `gui/widgets/log_console.py` — error status → `role` + re-polish.
- **Modify** `gui/stage_view.py` — banner → `role` + re-polish; Run button `role="primary"`.
- **Modify** `tests/gui_smoke.py` — add theme regression assertions (step [20]).
- **Modify** `docs/Usage.md`, `docs/Codebase.md` — document the toggle + new module.

---

## Task 1: `gui/theme.py` core (TDD)

**Files:**
- Create: `gui/theme.py`
- Test: `tests/test_theme.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_theme.py`:

```python
"""Tests for gui.theme: palettes, QSS generation, ThemeController, apply_theme."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

from gui import theme  # noqa: E402

_FIELDS = (
    "surface", "panel", "ink", "ink_muted", "border", "accent", "accent_strong",
    "accent_soft", "accent_on", "error", "warning", "success", "rail_bg",
    "mpl_facecolor", "pv_background",
)


def test_palettes_have_all_fields_and_differ():
    for p in (theme.LIGHT, theme.DARK):
        for f in _FIELDS:
            val = getattr(p, f)
            assert isinstance(val, str) and val.startswith("#"), (f, val)
    assert theme.LIGHT.surface != theme.DARK.surface
    assert theme.LIGHT.ink != theme.DARK.ink
    assert theme.LIGHT.accent != theme.DARK.accent


def test_light_accent_is_exact_kit_green():
    assert theme.LIGHT.accent == "#009682"


def test_build_qss_is_nonempty_and_uses_tokens():
    qss = theme.build_qss(theme.LIGHT)
    assert isinstance(qss, str) and len(qss) > 200
    assert "#009682" in qss                 # KIT green present in light QSS
    assert theme.LIGHT.error in qss
    assert theme.LIGHT.border in qss
    assert theme.DARK.accent in theme.build_qss(theme.DARK)


def test_controller_emits_and_updates_without_app():
    tc = theme.ThemeController()  # fresh, not the singleton
    seen = []
    tc.themeChanged.connect(lambda pal: seen.append(pal))
    assert tc.mode == "light"
    tc.set_mode("dark")
    assert tc.mode == "dark"
    assert tc.palette is theme.DARK
    assert seen and seen[-1] is theme.DARK


def test_apply_theme_sets_palette_and_stylesheet():
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    pal = theme.apply_theme(app, "dark")
    assert pal is theme.DARK
    assert app.styleSheet()                          # non-empty global QSS
    assert theme.DARK.accent in app.styleSheet()
    # Once a stylesheet is set, Qt wraps the base style in a QStyleSheetStyle
    # proxy whose objectName() is '' — so verify the QPalette took effect, not
    # the style name.
    assert app.palette().color(QPalette.ColorRole.Window).name() == theme.DARK.surface
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3 -m pytest tests/test_theme.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gui.theme'`.

- [ ] **Step 3: Write `gui/theme.py`**

```python
"""Application theming: light/dark palettes, the global QSS, and a controller.

Qt-only — **never imported by ``dfxm``** (the core stays Qt-free). This is the
single source of truth for every colour the GUI draws. Standard widgets are
themed by a global stylesheet built from a :class:`Palette` (semantic colours
addressed via dynamic ``role`` properties); the two embedded canvases
(matplotlib, pyvista) cannot be reached by QSS, so they subscribe to
:attr:`ThemeController.themeChanged` and restyle themselves.

Direction "Lab": soft, rounded, roomy. Accent is KIT-Grün — exact ``#009682``
in light mode, nudged slightly brighter in dark mode for legibility on dark
panels.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Palette:
    """Every colour the GUI uses, for one mode (hex strings)."""

    surface: str
    panel: str
    ink: str
    ink_muted: str
    border: str
    accent: str
    accent_strong: str
    accent_soft: str
    accent_on: str
    error: str
    warning: str
    success: str
    rail_bg: str
    mpl_facecolor: str
    pv_background: str


LIGHT = Palette(
    surface="#eef1f0",
    panel="#ffffff",
    ink="#1f2a27",
    ink_muted="#62706b",
    border="#dde3e1",
    accent="#009682",
    accent_strong="#00786a",
    accent_soft="#d8efe9",
    accent_on="#ffffff",
    error="#b00020",
    warning="#b06a00",
    success="#00786a",
    rail_bg="#f5f7f6",
    mpl_facecolor="#ffffff",
    pv_background="#eef1f0",
)

DARK = Palette(
    surface="#15191b",
    panel="#1d2326",
    ink="#e7ecea",
    ink_muted="#93a09b",
    border="#2c3439",
    accent="#12a890",
    accent_strong="#46c9b6",
    accent_soft="#163a34",
    accent_on="#ffffff",
    error="#ef5a6a",
    warning="#e0a23a",
    success="#46c9b6",
    rail_bg="#181d20",
    mpl_facecolor="#1d2326",
    pv_background="#15191b",
)

PALETTES: dict[str, Palette] = {"light": LIGHT, "dark": DARK}

# Banner fills are mode-independent (saturated red/green with white text read on
# both light and dark surfaces); kept separate from the text ``error``/``success``
# tokens, which must contrast the *surface*.
_BANNER_ERROR = "#c62828"
_BANNER_SUCCESS = "#2e7d32"


def build_qss(p: Palette) -> str:
    """Return the global Qt Style Sheet for palette *p*."""
    return f"""
    QGroupBox {{
        background: {p.panel};
        border: 1px solid {p.border};
        border-radius: 9px;
        margin-top: 10px;
        padding: 10px;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 4px;
        color: {p.ink_muted};
        font-weight: 600;
    }}
    QLineEdit, QAbstractSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
        background: {p.panel};
        border: 1px solid {p.border};
        border-radius: 7px;
        padding: 4px 8px;
        selection-background-color: {p.accent};
        selection-color: {p.accent_on};
    }}
    QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus {{
        border: 1px solid {p.accent};
    }}
    QPushButton {{
        background: {p.panel};
        color: {p.ink};
        border: 1px solid {p.border};
        border-radius: 8px;
        padding: 5px 14px;
    }}
    QPushButton:hover {{ border-color: {p.accent}; }}
    QPushButton:disabled {{ color: {p.ink_muted}; }}
    QPushButton[role="primary"] {{
        background: {p.accent};
        color: {p.accent_on};
        border: 1px solid {p.accent};
        font-weight: 600;
    }}
    QPushButton[role="primary"]:hover {{
        background: {p.accent_strong};
        border-color: {p.accent_strong};
    }}
    QPushButton[role="primary"]:disabled {{
        background: {p.border};
        border-color: {p.border};
        color: {p.ink_muted};
    }}
    QTabBar::tab {{
        background: transparent;
        color: {p.ink_muted};
        padding: 6px 12px;
        border: none;
    }}
    QTabBar::tab:selected {{
        color: {p.ink};
        border-bottom: 2px solid {p.accent};
    }}
    QTabWidget::pane {{ border: 1px solid {p.border}; border-radius: 8px; }}
    QListWidget {{
        background: {p.rail_bg};
        border: 1px solid {p.border};
        border-radius: 8px;
        padding: 4px;
    }}
    QListWidget::item {{ padding: 5px 8px; border-radius: 6px; }}
    QListWidget::item:selected {{
        background: {p.accent_soft};
        color: {p.accent_strong};
    }}
    QProgressBar {{
        background: {p.panel};
        border: 1px solid {p.border};
        border-radius: 7px;
        text-align: center;
    }}
    QProgressBar::chunk {{ background: {p.accent}; border-radius: 6px; }}
    QLabel[role="muted"] {{ color: {p.ink_muted}; }}
    QLabel[role="error"] {{ color: {p.error}; }}
    QLabel[role="calib"] {{ color: {p.error}; font-weight: bold; }}
    QLabel[role="notes"] {{ color: {p.error}; font-style: italic; }}
    QLabel[role="group-header"] {{ font-weight: bold; color: {p.ink}; }}
    HelpPanel {{ background: {p.accent_soft}; border-left: 3px solid {p.accent}; }}
    QLabel[role="banner-error"] {{
        background: {_BANNER_ERROR}; color: #ffffff;
        border-radius: 7px; padding: 6px 10px;
    }}
    QLabel[role="banner-success"] {{
        background: {_BANNER_SUCCESS}; color: #ffffff;
        border-radius: 7px; padding: 6px 10px;
    }}
    QPushButton[role="chip"] {{
        background: {p.accent_soft};
        color: {p.accent_strong};
        border: 1px solid {p.border};
        border-radius: 11px;
        padding: 3px 12px;
    }}
    QPushButton[role="chip"]:hover {{ border-color: {p.accent}; }}
    QLabel[role="external"] {{
        color: {p.ink_muted};
        border: 1px dashed {p.border};
        border-radius: 11px;
        padding: 3px 10px;
    }}
    """


def _qpalette(p: Palette) -> QPalette:
    """Build a QPalette so Fusion-drawn bits and native dialogs follow the theme."""
    qp = QPalette()
    qp.setColor(QPalette.ColorRole.Window, QColor(p.surface))
    qp.setColor(QPalette.ColorRole.WindowText, QColor(p.ink))
    qp.setColor(QPalette.ColorRole.Base, QColor(p.panel))
    qp.setColor(QPalette.ColorRole.AlternateBase, QColor(p.surface))
    qp.setColor(QPalette.ColorRole.Text, QColor(p.ink))
    qp.setColor(QPalette.ColorRole.Button, QColor(p.panel))
    qp.setColor(QPalette.ColorRole.ButtonText, QColor(p.ink))
    qp.setColor(QPalette.ColorRole.Highlight, QColor(p.accent))
    qp.setColor(QPalette.ColorRole.HighlightedText, QColor(p.accent_on))
    qp.setColor(QPalette.ColorRole.ToolTipBase, QColor(p.panel))
    qp.setColor(QPalette.ColorRole.ToolTipText, QColor(p.ink))
    qp.setColor(QPalette.ColorRole.PlaceholderText, QColor(p.ink_muted))
    disabled = QColor(p.ink_muted)
    for role in (
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
        QPalette.ColorRole.WindowText,
    ):
        qp.setColor(QPalette.ColorGroup.Disabled, role, disabled)
    return qp


def apply_theme(app: QApplication, mode: str) -> Palette:
    """Apply *mode* ('light'|'dark') to *app*; return the active :class:`Palette`."""
    p = PALETTES.get(mode, LIGHT)
    app.setStyle("Fusion")          # set first: it resets the palette
    app.setPalette(_qpalette(p))
    app.setStyleSheet(build_qss(p))
    return p


class ThemeController(QObject):
    """Process-wide theme state. Applies a mode and notifies embedded canvases."""

    themeChanged = Signal(object)  # emits the active Palette

    _instance: "ThemeController | None" = None

    def __init__(self) -> None:
        super().__init__()
        self._mode = "light"
        self._palette = LIGHT

    @classmethod
    def instance(cls) -> "ThemeController":
        if cls._instance is None:
            cls._instance = ThemeController()
        return cls._instance

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def palette(self) -> Palette:
        return self._palette

    def set_mode(self, mode: str) -> None:
        """Apply *mode* to the running QApplication (if any) and emit themeChanged."""
        app = QApplication.instance()
        if app is not None:
            self._palette = apply_theme(app, mode)
        else:
            self._palette = PALETTES.get(mode, LIGHT)
        self._mode = mode if mode in PALETTES else "light"
        self.themeChanged.emit(self._palette)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python3 -m pytest tests/test_theme.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add gui/theme.py tests/test_theme.py
git commit -m "$(printf 'feat(gui): theme module — light/dark palettes, QSS, ThemeController\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 2: Wire theming into `gui/app.py`

**Files:**
- Modify: `gui/app.py`

- [ ] **Step 1: Edit `main()` to apply the saved theme before showing the window**

Replace the body of `main()` (currently lines ~13–24) so it reads:

```python
def main(argv: list[str] | None = None) -> int:
    # Bind matplotlib's Qt backend to PySide6 (must precede backend import).
    os.environ.setdefault("QT_API", "pyside6")

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from .main_window import MainWindow
    from .theme import ThemeController

    QApplication.setOrganizationName("dfxm")
    QApplication.setApplicationName("pipeline")

    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)

    mode = QSettings().value("theme", "light")
    if mode not in ("light", "dark"):
        mode = "light"
    ThemeController.instance().set_mode(mode)  # applies Fusion + palette + QSS

    window = MainWindow()
    window.show()
    return app.exec()
```

- [ ] **Step 2: Smoke-import check (no display needed)**

Run: `QT_QPA_PLATFORM=offscreen python3 -c "import gui.app; print('ok')"`
Expected: prints `ok` (no import errors).

- [ ] **Step 3: Commit**

```bash
git add gui/app.py
git commit -m "$(printf 'feat(gui): apply saved light/dark theme at startup\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 3: Toggle button + persistence in `gui/main_window.py`

**Files:**
- Modify: `gui/main_window.py`

- [ ] **Step 1: Import the theme controller**

Add to the existing imports near the top (after `from .stage_view import StageView`):

```python
from .theme import ThemeController
```

- [ ] **Step 2: Use the palette colour for the muted concat row**

Replace line 82:

```python
        muted = QBrush(QColor("#888888"))
```

with:

```python
        muted = QBrush(QColor(ThemeController.instance().palette.ink_muted))
```

- [ ] **Step 3: Add the ☀/☾ toggle button next to "Publication style…"**

Replace the block that builds `self._pub_style_btn` and the left column (lines ~108–116) with:

```python
        # "Publication style…" button — lives in the left column below the rail.
        self._pub_style_btn = QPushButton("Publication style…")
        self._pub_style_btn.clicked.connect(self._on_pub_style)

        # Light/dark theme toggle.
        self._theme_btn = QPushButton()
        self._theme_btn.setCheckable(True)
        self._theme_btn.setChecked(ThemeController.instance().mode == "dark")
        self._theme_btn.setToolTip("Switch between light and dark appearance")
        self._theme_btn.clicked.connect(self._on_theme_toggle)
        self._sync_theme_btn()
        ThemeController.instance().themeChanged.connect(self._on_theme_changed)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self._experiment_panel)
        left_layout.addWidget(self._nav, 1)
        left_layout.addWidget(self._pub_style_btn)
        left_layout.addWidget(self._theme_btn)
```

- [ ] **Step 4: Add the toggle handlers**

Add these methods to `MainWindow` (e.g. just after `global_plot_style` / `_on_pub_style`, before `# -- navigation`):

```python
    # -- theme --------------------------------------------------------------
    def _sync_theme_btn(self) -> None:
        dark = ThemeController.instance().mode == "dark"
        self._theme_btn.setText("☾ Dark" if dark else "☀ Light")

    def _on_theme_toggle(self, checked: bool) -> None:
        from PySide6.QtCore import QSettings

        mode = "dark" if checked else "light"
        ThemeController.instance().set_mode(mode)
        QSettings().setValue("theme", mode)

    def _on_theme_changed(self, palette) -> None:
        self._sync_theme_btn()
        # QListWidgetItem foreground is not reachable by QSS — refresh it here.
        item = self._status_items.get("concat")
        if item is not None:
            item.setForeground(QBrush(QColor(palette.ink_muted)))
```

- [ ] **Step 5: Import-check the window builds offscreen**

Run:
```bash
QT_QPA_PLATFORM=offscreen python3 -c "
from PySide6.QtWidgets import QApplication
from gui.theme import ThemeController
from gui.main_window import MainWindow
app = QApplication([])
ThemeController.instance().set_mode('light')
w = MainWindow()
assert w._theme_btn.text() == '☀ Light'
w._theme_btn.setChecked(True); w._on_theme_toggle(True)
assert ThemeController.instance().mode == 'dark'
assert w._theme_btn.text() == '☾ Dark'
print('ok')
"
```
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add gui/main_window.py
git commit -m "$(printf 'feat(gui): light/dark toggle button + QSettings persistence\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 4: Theme the embedded matplotlib canvas (`gui/widgets/mpl_canvas.py`)

**Files:**
- Modify: `gui/widgets/mpl_canvas.py`

- [ ] **Step 1: Import the controller**

Add after the existing imports:

```python
from ..theme import ThemeController
```

- [ ] **Step 2: Apply current theme on construction and subscribe to changes**

At the end of `__init__` (after `self.canvas.mpl_connect(...)`), add:

```python
        self.apply_theme(ThemeController.instance().palette)
        ThemeController.instance().themeChanged.connect(self.apply_theme)
```

- [ ] **Step 3: Add `apply_theme` and re-apply after clears**

Add this method (after `__init__`, before `_on_click`):

```python
    def apply_theme(self, palette) -> None:
        """Restyle the *display* canvas (figure/axes chrome) — exports unaffected."""
        fc = palette.mpl_facecolor
        ink = palette.ink
        self.figure.set_facecolor(fc)
        self.ax.set_facecolor(fc)
        self.ax.tick_params(colors=ink)
        for spine in self.ax.spines.values():
            spine.set_color(palette.border)
        self.ax.xaxis.label.set_color(ink)
        self.ax.yaxis.label.set_color(ink)
        self.ax.title.set_color(ink)
        self.canvas.draw_idle()
```

Then make `clear` and `show_image` re-apply the theme (since `ax.clear()` resets
the axes facecolor/colors). Replace the existing `clear` and `show_image`:

```python
    def clear(self) -> None:
        self.ax.clear()
        self.apply_theme(ThemeController.instance().palette)

    def show_image(self, data, **imshow_kw):
        """Replace the axes content with ``imshow(data)`` and redraw."""
        self.ax.clear()
        im = self.ax.imshow(data, **imshow_kw)
        self.apply_theme(ThemeController.instance().palette)
        return im
```

(`apply_theme` ends with `draw_idle()`, so these no longer need a separate draw.)

- [ ] **Step 4: Verify offscreen**

Run:
```bash
QT_QPA_PLATFORM=offscreen python3 -c "
from PySide6.QtWidgets import QApplication
from matplotlib.colors import to_hex
from gui.theme import ThemeController, LIGHT, DARK
from gui.widgets.mpl_canvas import MplCanvas
app = QApplication([])
ThemeController.instance().set_mode('light')
mc = MplCanvas()
assert to_hex(mc.figure.get_facecolor()) == LIGHT.mpl_facecolor
ThemeController.instance().set_mode('dark')
assert to_hex(mc.figure.get_facecolor()) == DARK.mpl_facecolor
mc.show_image([[0,1],[1,0]])
assert to_hex(mc.figure.get_facecolor()) == DARK.mpl_facecolor  # survives clear
print('ok')
"
```
Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add gui/widgets/mpl_canvas.py
git commit -m "$(printf 'feat(gui): theme the embedded matplotlib display canvas (light/dark)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 5: Theme the embedded pyvista canvas (`gui/widgets/pv_canvas.py`)

**Files:**
- Modify: `gui/widgets/pv_canvas.py`

- [ ] **Step 1: Import the controller and subscribe**

Add after the existing imports:

```python
from ..theme import ThemeController
```

At the end of `__init__` (after `self._available = False`), add:

```python
        ThemeController.instance().themeChanged.connect(self.apply_theme)
```

- [ ] **Step 2: Apply background on lazy creation**

In `ensure`, right after `self._plotter = QtInteractor(self)`, add:

```python
            self._plotter.set_background(ThemeController.instance().palette.pv_background)
```

- [ ] **Step 3: Add `apply_theme` (guarded, no pyvista import)**

Add this method (after `ensure`, before `clear`):

```python
    def apply_theme(self, palette) -> None:
        """Recolour the 3-D background; no-op until the plotter exists."""
        if self._plotter is not None:
            self._plotter.set_background(palette.pv_background)
```

- [ ] **Step 4: Verify offscreen (placeholder path stays safe)**

Run:
```bash
QT_QPA_PLATFORM=offscreen python3 -c "
from PySide6.QtWidgets import QApplication
from gui.theme import ThemeController
from gui.widgets.pv_canvas import PvCanvas
app = QApplication([])
ThemeController.instance().set_mode('light')
pv = PvCanvas()              # plotter not created yet
ThemeController.instance().set_mode('dark')   # apply_theme must be a safe no-op
print('ok')
"
```
Expected: prints `ok` (no exception even though 3-D was never initialised).

- [ ] **Step 5: Commit**

```bash
git add gui/widgets/pv_canvas.py
git commit -m "$(printf 'feat(gui): theme the embedded pyvista 3-D background (light/dark)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 6: Consolidate scattered inline styles into theme tokens

All semantic colours move to QSS `role` properties (or, for runtime toggles, a
property + re-polish). One source of truth; everything follows the theme.

**Files:**
- Modify: `gui/overview_page.py`, `gui/widgets/help_panel.py`,
  `gui/experiment_panel.py`, `gui/widgets/param_form.py`,
  `gui/widgets/log_console.py`, `gui/stage_view.py`

- [ ] **Step 1: `overview_page.py` — chips/external via `role`**

Delete the `_CHIP_STYLE` and `_EXTERNAL_STYLE` constants (lines 23–27).
Replace line 59 `btn.setStyleSheet(_CHIP_STYLE)` with:

```python
            btn.setProperty("role", "chip")
```

Replace line 66 `ext.setStyleSheet(_EXTERNAL_STYLE)` with:

```python
                ext.setProperty("role", "external")
```

- [ ] **Step 2: `help_panel.py` — frame via QSS, cal-warning red follows theme**

Replace the whole file body below the module docstring with:

```python
from __future__ import annotations

import html

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from dfxm.config.models import Param

from ..theme import ThemeController


class HelpPanel(QFrame):
    """Styled read-only box explaining the focused parameter."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Background/border come from the global QSS (HelpPanel selector).
        self._label = QLabel("")
        self._label.setWordWrap(True)
        self._idle_html = ""
        self._current: Param | None = None
        self._error_color = ThemeController.instance().palette.error
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.addWidget(self._label)
        ThemeController.instance().themeChanged.connect(self._on_theme_changed)

    def _on_theme_changed(self, palette) -> None:
        self._error_color = palette.error
        self._render()

    def _cal_warning(self) -> str:
        return (
            f'<span style="color:{self._error_color};">⚠ calibration — physically '
            "meaningful; confirm against the beamline calibration for your "
            "experiment.</span>"
        )

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
        p = self._current
        head = f"<b>{html.escape(p.label)}</b>"
        if p.unit:
            head += f" ({html.escape(p.unit)})"
        parts = [head]
        if p.calibration:
            parts.append(self._cal_warning())
        if p.help:
            parts.append(html.escape(p.help))
        self._label.setText("<br>".join(parts))
```

- [ ] **Step 3: `experiment_panel.py` — summary/notes via `role`**

Replace line 109 `self._summary.setStyleSheet("color: #666;")` with:

```python
        self._summary.setProperty("role", "muted")
```

Replace line 114 `self._notes.setStyleSheet("color: #b00020; font-style: italic;")` with:

```python
        self._notes.setProperty("role", "notes")
```

- [ ] **Step 4: `param_form.py` — group header + calibration label via `role`**

Replace line 99 `header.setStyleSheet("font-weight: bold; margin-top: 6px;")` with:

```python
                    header.setProperty("role", "group-header")
```

Replace line 165 `lbl.setStyleSheet("color: #b00020; font-weight: bold;")` with:

```python
            lbl.setProperty("role", "calib")
```

- [ ] **Step 5: `log_console.py` — error status via `role` + re-polish**

Replace `set_status` (lines 56–58) with:

```python
    def set_status(self, text: str, *, error: bool = False) -> None:
        self._status.setText(text)
        self._status.setProperty("role", "error" if error else "")
        self._status.style().unpolish(self._status)
        self._status.style().polish(self._status)
```

- [ ] **Step 6: `stage_view.py` — banner via `role` + re-polish; Run button primary**

After line 85 `self._run_btn = QPushButton("Run")`, add:

```python
        self._run_btn.setProperty("role", "primary")
```

Replace `_show_banner` (lines 197–207) with:

```python
    def _show_banner(self, html_text: str, *, error: bool) -> None:
        self._banner.setProperty("role", "banner-error" if error else "banner-success")
        self._banner.style().unpolish(self._banner)
        self._banner.style().polish(self._banner)
        self._banner.setText(html_text)
        self._banner.setVisible(True)
```

- [ ] **Step 7: Verify nothing broke — pure tests + ruff**

Run: `python3 -m pytest tests/test_theme.py -q && ruff check gui tests`
Expected: tests PASS; ruff reports no errors (an unused-import error for a now-deleted constant means a leftover reference — fix it).

- [ ] **Step 8: Commit**

```bash
git add gui/overview_page.py gui/widgets/help_panel.py gui/experiment_panel.py \
        gui/widgets/param_form.py gui/widgets/log_console.py gui/stage_view.py
git commit -m "$(printf 'refactor(gui): consolidate inline styles into theme role tokens\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 7: Extend the GUI smoke test with theme assertions

**Files:**
- Modify: `tests/gui_smoke.py`

- [ ] **Step 1: Add a theme regression block before the final success print**

Insert this immediately before the line `print("\nGUI SMOKE PASSED")` (line ~541):

```python
    # [20] Theme: light by default, toggling restyles the app + embedded canvases.
    from matplotlib.colors import to_hex

    from gui import theme as _theme
    from gui.widgets.mpl_canvas import MplCanvas as _MplCanvas

    tc = _theme.ThemeController.instance()
    tc.set_mode("light")
    assert "#009682" in app.styleSheet()                       # KIT green in light QSS
    mc = _MplCanvas()
    assert to_hex(mc.figure.get_facecolor()) == _theme.LIGHT.mpl_facecolor
    # The left-column toggle flips to dark and restyles everything.
    win._theme_btn.setChecked(True)
    win._on_theme_toggle(True)
    app.processEvents()
    assert tc.mode == "dark"
    assert _theme.DARK.accent in app.styleSheet()
    assert to_hex(mc.figure.get_facecolor()) == _theme.DARK.mpl_facecolor  # canvas followed
    assert win._theme_btn.text() == "☾ Dark"
    # The muted concat rail row recoloured to the dark muted ink.
    assert win._status_items["concat"].foreground().color().name() == _theme.DARK.ink_muted
    win._theme_btn.setChecked(False)
    win._on_theme_toggle(False)
    app.processEvents()
    assert tc.mode == "light"
    mc.deleteLater()
    print("[20] theme toggle restyles app QSS + matplotlib canvas + rail; persistence path OK")
```

- [ ] **Step 2: Run the full smoke test**

Run: `QT_QPA_PLATFORM=offscreen python3 tests/gui_smoke.py`
Expected: ends with `[20] theme toggle ...` then `GUI SMOKE PASSED` (exit 0).

> Note: `QColor.name()` returns lowercase hex (e.g. `#93a09b`); the `DARK.ink_muted`
> token is already lowercase, so the comparison matches.

- [ ] **Step 3: Commit**

```bash
git add tests/gui_smoke.py
git commit -m "$(printf 'test(gui): smoke-test the light/dark toggle end-to-end\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 8: Documentation (repo contract)

**Files:**
- Modify: `docs/Usage.md`, `docs/Codebase.md`

- [ ] **Step 1: `docs/Usage.md` — document the toggle**

In the "The main window" section (around line 75), add a short paragraph:

```markdown
**Appearance.** A light/dark toggle (☀ Light / ☾ Dark) sits in the bottom of the
left column, beside *Publication style…*. The choice is remembered between
sessions. Switching theme only affects the on-screen app and the embedded
plot/3-D viewers — exported figures are always written on a white background.
```

- [ ] **Step 2: `docs/Codebase.md` — document the new module**

In the `gui/` section, add an entry for `gui/theme.py`:

```markdown
- **`gui/theme.py`** — application theming (Qt-only; never imported by `dfxm`).
  - `Palette` (frozen dataclass) with `LIGHT`/`DARK` instances and `PALETTES`.
  - `build_qss(palette) -> str` — the global Qt Style Sheet; semantic colours are
    addressed via dynamic `role` properties (`error`, `calib`, `notes`, `muted`,
    `group-header`, `banner-error`, `banner-success`, `chip`, `external`,
    and `QPushButton[role="primary"]`).
  - `apply_theme(app, mode) -> Palette` — sets Fusion + `QPalette` + stylesheet.
  - `ThemeController(QObject)` — process singleton; `set_mode(mode)` applies the
    theme and emits `themeChanged(Palette)`. Standard widgets restyle via the
    rebuilt stylesheet; `MplCanvas`/`PvCanvas` subscribe to `themeChanged`.
```

Also note the new `apply_theme(palette)` methods on `MplCanvas` and `PvCanvas`,
and that `app.py` reads/writes the `theme` key via `QSettings("dfxm","pipeline")`.

- [ ] **Step 3: Commit**

```bash
git add docs/Usage.md docs/Codebase.md
git commit -m "$(printf 'docs: light/dark theme toggle + gui/theme.py reference\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Task 9: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Full test suite**

Run: `python3 -m pytest -q`
Expected: all pass (the prior baseline plus the new `tests/test_theme.py`); no new skips beyond the documented vs-legacy self-skips.

- [ ] **Step 2: Lint + format**

Run: `ruff check . && ruff format --check .`
Expected: clean.

- [ ] **Step 3: GUI smoke**

Run: `QT_QPA_PLATFORM=offscreen python3 tests/gui_smoke.py`
Expected: `GUI SMOKE PASSED`.

- [ ] **Step 4: Manual eyeball (needs a display)**

Run: `python3 -m gui.app`
Check: app opens in light mode; the ☀/☾ button toggles dark; the **Run** button
is KIT-green; focused fields show a green focus ring; the active tab has a green
underline; the active stage in the rail is green-tinted; error/success banners are
legible in both modes; relaunching preserves the last theme.

- [ ] **Step 5: (Optional) merge — see finishing-a-development-branch**

The branch is `gui-visual-reskin`. When satisfied, integrate per the
`superpowers:finishing-a-development-branch` skill.

---

## Self-review notes (author)

- **Spec coverage:** theme module (T1), app wiring (T2), toggle+persistence (T3),
  mpl canvas (T4), pv canvas (T5), full consolidation list — every file:line in
  the spec table (T6), tests (T1 + T7), docs (T8). Export-stays-white invariant is
  preserved by construction (only `MplCanvas`/`PvCanvas` *display* is themed; every
  `savefig(..., facecolor="white")` path is untouched) and eyeballed in T9.4.
- **Persistence location** intentionally moved from `ThemeController` to
  `app.py`/`main_window` (documented above) to keep the controller unit-testable.
- **Type consistency:** `Palette` field names are used identically across
  `build_qss`, `_qpalette`, `apply_theme`, the canvases, and tests; `set_mode`,
  `palette`, `mode`, and `themeChanged(object→Palette)` match every call site.
```
