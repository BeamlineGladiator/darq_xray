"""See-also pointers rendered inline and in the help panel (offscreen Qt)."""

import dataclasses
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QComboBox, QFormLayout  # noqa: E402

_app = QApplication.instance() or QApplication([])

from darq_xray.config.models import Experiment, Param, ParamType, SeeAlso  # noqa: E402
from darq_xray.gui.bindings import STAGE_SPECS  # noqa: E402
from darq_xray.gui.stage_view import StageView  # noqa: E402
from darq_xray.gui.widgets.help_panel import HelpPanel, param_help_html  # noqa: E402
from darq_xray.gui.widgets.param_form import ParamForm  # noqa: E402

_PARAMS = (
    Param("colormap", ParamType.ENUM, "Colormap", default="fast", choices=("fast", "gray")),
    Param("plain", ParamType.STR, "Plain"),
)

_STAGE_PTR = SeeAlso("", "Colormaps are set in Publication style… (left panel).")
_PARAM_PTR = SeeAlso("param:colormap", "Publication style wins for standard quantities.")
_MARKUP_PTR = "a < b & c"
_MARKUP_ESCAPED = "a &lt; b &amp; c"


def _form_layout_holding(form: ParamForm, widget) -> QFormLayout:
    """The `QFormLayout` inside *form* whose rows include *widget*."""
    outer = form.layout()
    for i in range(outer.count()):
        sub = outer.itemAt(i).layout()
        if isinstance(sub, QFormLayout) and sub.getWidgetPosition(widget)[0] >= 0:
            return sub
    raise AssertionError(f"{widget} is in no QFormLayout of this form")


def test_a_form_without_pointers_renders_no_pointer_rows():
    form = ParamForm(_PARAMS)
    assert form._see_also_labels == {}


def test_a_stage_pointer_renders_one_always_visible_row():
    form = ParamForm(_PARAMS, see_also=(_STAGE_PTR,))
    label = form._see_also_labels[""]
    assert label.text() == _STAGE_PTR.text
    # The point of a pointer is that it needs no expanding or focusing.
    assert label.isVisibleTo(form) is True
    # …and it sits above every row, not wherever the layout happened to grow.
    assert form.layout().count() > 1  # precondition: there are rows below it
    assert form.layout().indexOf(label) == 0


def test_a_param_pointer_renders_under_that_param_only():
    form = ParamForm(_PARAMS, see_also=(_PARAM_PTR,))
    label = form._see_also_labels["colormap"]
    assert set(form._see_also_labels) == {"colormap"}
    assert label.text() == _PARAM_PTR.text
    assert label.isVisibleTo(form) is True
    # "under" is the whole point: the row directly below that param's editor.
    rows = _form_layout_holding(form, label)
    editor_row, _ = rows.getWidgetPosition(form._editors["colormap"])
    assert editor_row >= 0  # precondition: same layout as the editor
    assert rows.getWidgetPosition(label)[0] == editor_row + 1


def test_pointer_rows_are_styled_as_hints_not_warnings():
    # Advisory notes use role="warning" and are hidden by default; pointers are
    # a different thing and must not borrow that styling.
    form = ParamForm(_PARAMS, see_also=(_STAGE_PTR, _PARAM_PTR))
    # Also the only place both anchor kinds are asserted to coexist on one form.
    assert len(form._see_also_labels) == 2
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


def test_param_help_html_escapes_the_pointer_text():
    # The pointer is authored text dropped into rich text: markup must not run.
    html = param_help_html(_PARAMS[0], see_also=_MARKUP_PTR)
    assert _MARKUP_ESCAPED in html


def test_the_help_panel_idle_text_carries_a_stage_pointer():
    panel = HelpPanel()
    panel.set_idle("Strain", "Compute strain maps.", see_also=_STAGE_PTR.text)
    assert "Compute strain maps." in panel._label.text()  # precondition
    assert _STAGE_PTR.text in panel._label.text()


def test_the_help_panel_escapes_a_stage_pointer():
    panel = HelpPanel()
    panel.set_idle("Strain", "Compute strain maps.", see_also=_MARKUP_PTR)
    assert _MARKUP_ESCAPED in panel._label.text()


def test_the_help_panel_shows_a_param_pointer_when_that_param_is_focused():
    panel = HelpPanel()
    panel.set_see_also({"colormap": _PARAM_PTR.text})
    panel.show_param(_PARAMS[0])
    assert _PARAM_PTR.text in panel._label.text()
    panel.show_param(_PARAMS[1])
    assert _PARAM_PTR.text not in panel._label.text()


def test_the_stage_view_wires_a_spec_s_pointers_into_the_form_and_the_help_panel():
    """The user-visible path: spec → form rows → idle text → focused-param text."""
    base = STAGE_SPECS["strain"]
    target = base.params[0].name
    spec = dataclasses.replace(
        base,
        see_also=(
            SeeAlso("", _STAGE_PTR.text),
            SeeAlso(f"param:{target}", _PARAM_PTR.text),
        ),
    )
    assert spec.see_also_problems() == []  # precondition: the anchor names a real param
    view = StageView("strain", spec, Experiment())
    assert set(view._form._see_also_labels) == {"", target}
    assert _STAGE_PTR.text in view._help._label.text()
    view._help.show_param(spec.params[0])
    assert _PARAM_PTR.text in view._help._label.text()
