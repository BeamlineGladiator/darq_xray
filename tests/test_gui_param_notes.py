"""Per-field advisory notes rendered under a form widget (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import Param, ParamType  # noqa: E402
from gui.widgets.param_form import ParamForm  # noqa: E402

_PARAMS = (
    Param(
        "mode",
        ParamType.ENUM,
        "Mode",
        default="volume",
        choices=("volume", "surface"),
        advice_key="3d_texture",
    ),
    Param("plain", ParamType.STR, "Plain"),
)


def test_a_note_row_exists_only_for_params_that_declare_a_key():
    form = ParamForm(_PARAMS)
    assert "mode" in form._notes
    assert "plain" not in form._notes


def test_setting_a_note_shows_it_and_clearing_hides_it():
    form = ParamForm(_PARAMS)
    assert form._notes["mode"].isVisibleTo(form) is False  # precondition
    form.set_field_note("mode", "this GL stack caps 3-D textures at 2048 px")
    assert "2048" in form._notes["mode"].text()
    assert form._notes["mode"].isVisibleTo(form) is True
    form.set_field_note("mode", "")
    assert form._notes["mode"].text() == ""
    assert form._notes["mode"].isVisibleTo(form) is False


def test_apply_hints_routes_by_advice_key_and_clears_the_rest():
    form = ParamForm(_PARAMS)
    form.apply_hints({"3d_texture": "downsample 2x"})
    assert "downsample 2x" in form._notes["mode"].text()
    form.apply_hints({})
    assert form._notes["mode"].text() == ""


def test_setting_a_note_on_a_keyless_param_is_a_no_op():
    form = ParamForm(_PARAMS)
    form.set_field_note("plain", "ignored")  # must not raise


def test_the_editor_dict_still_holds_the_real_widget():
    """Other code and tests reach into _editors expecting the editor itself
    (tests/gui_smoke.py:255, tests/test_gui_wheel_guard.py) — a note row must
    not wrap it."""
    from PySide6.QtWidgets import QComboBox

    form = ParamForm(_PARAMS)
    assert isinstance(form._editors["mode"], QComboBox)
