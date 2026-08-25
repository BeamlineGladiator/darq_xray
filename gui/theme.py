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
_BANNER_INFO = "#37474f"

# The banner geometry, shared by all three roles and mode-independent — only the
# fill above distinguishes them. Public because `StageView` must apply it to the
# banner *widget* as well: a role is a dynamic property, and `QStyleSheetStyle`
# caches the geometry it resolves for a widget without one, which
# `unpolish()`/`polish()` does not invalidate. Colours survive that (they are
# resolved at paint time) but the padding never reached the size hint, so a
# wrapped message was clipped to one unpadded line. Keep the two in step: this
# constant is the single source, and `tests/test_gui_stage_advice.py`
# ::test_the_banner_grows_to_fit_a_wrapped_message pins the result.
BANNER_GEOMETRY_QSS = "border-radius: 7px; padding: 6px 10px;"


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
    QLineEdit:focus, QAbstractSpinBox:focus, QComboBox:focus, QPlainTextEdit:focus, QTextEdit:focus {{
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
    QLabel[role="warning"] {{ color: {p.warning}; }}
    QLabel[role="calib"] {{ color: {p.error}; font-weight: bold; }}
    QLabel[role="notes"] {{ color: {p.error}; font-style: italic; }}
    QLabel[role="hint"] {{ color: {p.ink_muted}; font-style: italic; }}
    QLabel[role="group-header"] {{ font-weight: bold; color: {p.ink}; }}
    HelpPanel {{ background: {p.accent_soft}; border-left: 3px solid {p.accent}; }}
    QLabel[role="banner-error"] {{
        background: {_BANNER_ERROR}; color: #ffffff;
        {BANNER_GEOMETRY_QSS}
    }}
    QLabel[role="banner-success"] {{
        background: {_BANNER_SUCCESS}; color: #ffffff;
        {BANNER_GEOMETRY_QSS}
    }}
    QLabel[role="banner-info"] {{
        background: {_BANNER_INFO}; color: #ffffff;
        {BANNER_GEOMETRY_QSS}
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
    app.setStyle("Fusion")  # set first: it resets the palette
    app.setPalette(_qpalette(p))
    app.setStyleSheet(build_qss(p))
    return p


class ThemeController(QObject):
    """Process-wide theme state. Applies a mode and notifies embedded canvases."""

    # Palette isn't a registered Qt metatype, so the signal carries it as object.
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
