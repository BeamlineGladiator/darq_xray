"""Tests for gui.theme: palettes, QSS generation, ThemeController, apply_theme."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("QT_API", "pyside6")

from gui import theme  # noqa: E402

_FIELDS = (
    "surface",
    "panel",
    "ink",
    "ink_muted",
    "border",
    "accent",
    "accent_strong",
    "accent_soft",
    "accent_on",
    "error",
    "warning",
    "success",
    "rail_bg",
    "mpl_facecolor",
    "pv_background",
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
    assert "#009682" in qss  # KIT green present in light QSS
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


def test_instance_returns_singleton():
    assert theme.ThemeController.instance() is theme.ThemeController.instance()


def test_apply_theme_sets_palette_and_stylesheet():
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    pal = theme.apply_theme(app, "dark")
    assert pal is theme.DARK
    assert app.styleSheet()  # non-empty global QSS
    assert theme.DARK.accent in app.styleSheet()
    # The QPalette was applied (note: once a stylesheet is set, Qt wraps the base
    # style in a QStyleSheetStyle proxy whose objectName() is '', so we verify the
    # palette took effect rather than the style name).
    assert app.palette().color(QPalette.ColorRole.Window).name() == theme.DARK.surface
