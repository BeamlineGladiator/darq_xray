"""See-also pointers rendered inline and in the help panel (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import Param, ParamType, SeeAlso  # noqa: E402
from gui.widgets.help_panel import HelpPanel, param_help_html  # noqa: E402
from gui.widgets.param_form import ParamForm  # noqa: E402

_PARAMS = (
    Param("colormap", ParamType.ENUM, "Colormap", default="fast", choices=("fast", "gray")),
    Param("plain", ParamType.STR, "Plain"),
)

_STAGE_PTR = SeeAlso("", "Colormaps are set in Publication style… (left panel).")
_PARAM_PTR = SeeAlso("param:colormap", "Publication style wins for standard quantities.")


def test_a_form_without_pointers_renders_no_pointer_rows():
    form = ParamForm(_PARAMS)
    assert form._see_also_labels == {}


def test_a_stage_pointer_renders_one_always_visible_row():
    form = ParamForm(_PARAMS, see_also=(_STAGE_PTR,))
    label = form._see_also_labels[""]
    assert label.text() == _STAGE_PTR.text
    # The point of a pointer is that it needs no expanding or focusing.
    assert label.isVisibleTo(form) is True


def test_a_param_pointer_renders_under_that_param_only():
    form = ParamForm(_PARAMS, see_also=(_PARAM_PTR,))
    assert set(form._see_also_labels) == {"colormap"}
    assert form._see_also_labels["colormap"].text() == _PARAM_PTR.text
    assert form._see_also_labels["colormap"].isVisibleTo(form) is True


def test_pointer_rows_are_styled_as_hints_not_warnings():
    # Advisory notes use role="warning" and are hidden by default; pointers are
    # a different thing and must not borrow that styling.
    form = ParamForm(_PARAMS, see_also=(_STAGE_PTR, _PARAM_PTR))
    for label in form._see_also_labels.values():
        assert label.property("role") == "hint"


def test_a_pointer_does_not_wrap_the_editor_widget():
    # gui_smoke and the wheel-guard tests reach into _editors[name] directly.
    form = ParamForm(_PARAMS, see_also=(_PARAM_PTR,))
    assert isinstance(form._editors["colormap"], QComboBox)


def test_a_pointer_for_an_unknown_param_renders_nothing_rather_than_crashing():
    form = ParamForm(_PARAMS, see_also=(SeeAlso("param:ghost", "text"),))
    assert form._see_also_labels == {}


def test_param_help_html_appends_the_pointer():
    html = param_help_html(_PARAMS[0], see_also="Set in Publication style…")
    assert "See also:" in html
    assert "Set in Publication style…" in html


def test_param_help_html_without_a_pointer_has_no_see_also_line():
    assert "See also:" not in param_help_html(_PARAMS[0])


def test_the_help_panel_idle_text_carries_a_stage_pointer():
    panel = HelpPanel()
    panel.set_idle("Strain", "Compute strain maps.", see_also=_STAGE_PTR.text)
    assert "Compute strain maps." in panel._label.text()  # precondition
    assert _STAGE_PTR.text in panel._label.text()


def test_the_help_panel_shows_a_param_pointer_when_that_param_is_focused():
    panel = HelpPanel()
    panel.set_see_also({"colormap": _PARAM_PTR.text})
    panel.show_param(_PARAMS[0])
    assert _PARAM_PTR.text in panel._label.text()
    panel.show_param(_PARAMS[1])
    assert _PARAM_PTR.text not in panel._label.text()
