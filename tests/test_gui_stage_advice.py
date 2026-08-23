"""The live cost line under a stage form (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.advisory import Advisory  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import StageView  # noqa: E402
from tests.machine_fixtures import workstation_sw_gl  # noqa: E402


def _advisory(
    headline="needs ~1.0 GB, 4.0 GB safely available — runs in memory", details=("a reason",)
):
    return Advisory(workstation_sw_gl(), None, None, headline, details)


def test_advice_line_starts_hidden():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    assert view._advice_label.isVisibleTo(view) is False


def test_advice_line_shows_headline_and_tooltips_details():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    view._show_advisory(_advisory())
    assert "runs in memory" in view._advice_label.text()
    assert "a reason" in view._advice_label.toolTip()
    assert view._advice_label.isVisibleTo(view) is True


def test_an_empty_headline_hides_the_line_again():
    """A stage with no estimator, or a cleared form, must not leave stale text."""
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    view._show_advisory(_advisory())
    assert view._advice_label.text()  # precondition: it really was shown
    assert view._advice_label.isVisibleTo(view) is True  # precondition: and visible
    view._show_advisory(_advisory(headline="", details=()))
    assert view._advice_label.text() == ""
    assert view._advice_label.isVisibleTo(view) is False


def test_form_changes_ask_the_advisor_for_a_refresh():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    asked = []
    view._advisor.request = lambda: asked.append(1)
    view._form.changed.emit()
    assert asked == [1]
