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


def test_applied_theme_puts_all_three_process_wide_things_back():
    """The helper must restore style, palette AND stylesheet, not just one.

    `apply_theme` changes all three and pytest shares one QApplication, so a
    partial restore leaves later Qt modules under a mixture no production path
    produces — and the resulting failure lands on whichever test happens to be
    collected next.
    """
    from PySide6.QtWidgets import QApplication

    from tests.qt_helpers import applied_theme

    app = QApplication.instance() or QApplication([])
    before = (app.style().objectName(), app.palette(), app.styleSheet())
    # Whichever theme this session already sits under, apply the other one —
    # otherwise the precondition below is vacuous. Some earlier Qt test in a
    # full run leaves the app themed, so "dark" is not a safe assumption.
    other = "light" if before[2] == theme.build_qss(theme.DARK) else "dark"
    with applied_theme(app, other):
        assert app.styleSheet() != before[2]
    assert app.style().objectName() == before[0]
    assert app.palette() == before[1]
    assert app.styleSheet() == before[2]


def test_apply_theme_sets_palette_and_stylesheet():
    from PySide6.QtGui import QPalette
    from PySide6.QtWidgets import QApplication

    from tests.qt_helpers import applied_theme

    app = QApplication.instance() or QApplication([])
    # Scoped: this restored nothing at all, so it left every Qt module collected
    # after it running under the dark palette.
    with applied_theme(app, "dark") as pal:
        assert pal is theme.DARK
        assert app.styleSheet()  # non-empty global QSS
        assert theme.DARK.accent in app.styleSheet()
        # The QPalette was applied (note: once a stylesheet is set, Qt wraps the
        # base style in a QStyleSheetStyle proxy whose objectName() is '', so we
        # verify the palette took effect rather than the style name).
        assert app.palette().color(QPalette.ColorRole.Window).name() == theme.DARK.surface
