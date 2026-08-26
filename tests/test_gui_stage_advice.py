"""The live cost line under a stage form (offscreen Qt)."""

import html
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.advisory import Advisory  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import StageView  # noqa: E402
from tests.machine_fixtures import workstation_sw_gl  # noqa: E402
from tests.qt_helpers import applied_theme  # noqa: E402

# The real headline from the first STO2 `visualize` run (in the wording that
# replaced "safely available") — long enough to wrap in the banner at the width
# this view gives it, which is the precondition the banner test asserts.
_WRAPPING_HEADLINE = "needs ~10.5 GiB RAM, 326.3 GiB budget — expected to run in memory"


def _advisory(
    headline="needs ~1.0 GiB RAM, 4.0 GiB budget — expected to run in memory",
    details=("a reason",),
):
    return Advisory(workstation_sw_gl(), None, None, headline, details)


def test_advice_line_starts_hidden():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    assert view._advice_label.isVisibleTo(view) is False


def test_advice_line_shows_headline_and_tooltips_details():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    view._show_advisory(_advisory())
    assert "expected to run in memory" in view._advice_label.text()
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


from PySide6.QtWidgets import QMessageBox  # noqa: E402

from dfxm.common.advice import RunPlan  # noqa: E402
from dfxm.config.models import CostEstimate  # noqa: E402

GB = 1024**3


def _blocked_advisory():
    plan = RunPlan(
        "chunked",
        4 * GB,
        8,
        1,
        "/scratch",
        ("a reason",),
        "needs 100.0 GiB of scratch disk but only 40.0 GiB is free",
    )
    return Advisory(
        workstation_sw_gl(),
        CostEstimate(200 * GB, 100 * GB, (76, 1200, 1800), True, scratch_bytes=100 * GB),
        plan,
        "needs ~200.0 GiB RAM, 4.0 GiB budget — expected to stream",
        ("a reason",),
        plan.blocked,
    )


def _view_with_advisory(advisory, monkeypatch):
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    monkeypatch.setattr(view._advisor, "compute_blocking", lambda: advisory)
    started = []
    monkeypatch.setattr(view, "_start_runner", lambda params: started.append(params))
    return view, started


def test_run_shows_the_cost_in_an_info_banner(monkeypatch):
    view, started = _view_with_advisory(_advisory(), monkeypatch)
    view._on_run()
    assert started, "the run must still start"
    assert view._banner.isVisibleTo(view)
    assert view._banner.property("role") == "banner-info"
    assert "expected to run in memory" in view._banner.text()


def test_a_blocked_run_asks_and_starts_when_accepted(monkeypatch):
    adv = _blocked_advisory()
    assert adv.blocked  # precondition: this fixture really is blocked
    view, started = _view_with_advisory(adv, monkeypatch)
    monkeypatch.setattr(view, "_confirm_blocked", lambda reason: True)
    view._on_run()
    assert started


def test_a_blocked_run_starts_nothing_when_declined(monkeypatch):
    adv = _blocked_advisory()
    assert adv.blocked  # precondition
    view, started = _view_with_advisory(adv, monkeypatch)
    monkeypatch.setattr(view, "_confirm_blocked", lambda reason: False)
    view._on_run()
    assert started == []


def test_the_confirmation_defaults_to_cancel(monkeypatch):
    """A stray Enter must not launch a run the machine cannot finish."""
    seen = {}

    def fake_exec(self):
        seen["default"] = self.defaultButton()
        return QMessageBox.StandardButton.Cancel

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    assert view._confirm_blocked("needs more disk") is False
    assert seen["default"].text().endswith("Cancel")


def test_hints_reach_the_form():
    view = StageView("visualize", STAGE_SPECS["visualize"], Experiment())
    view._show_advisory(
        Advisory(
            workstation_sw_gl(),
            None,
            None,
            "a headline",
            (),
            hints={"3d_texture": "downsample 2x or volume mode renders blank"},
        )
    )
    assert "renders blank" in view._form._notes["render_mode"].text()
    view._show_advisory(Advisory(workstation_sw_gl(), None, None, "a headline", ()))
    assert view._form._notes["render_mode"].text() == ""


def test_the_banner_grows_to_fit_a_wrapped_message():
    """A pre-flight message that wraps must not be clipped to one line's height.

    `role` is a dynamic property, and `QStyleSheetStyle` caches the geometry it
    resolved for the banner back when it had no role — a cache that
    `style().unpolish()`/`polish()`, the documented way to re-evaluate such a
    selector, does not invalidate. Colours came through anyway (they resolve at
    paint time), so the banner painted its themed fill at a padding-free
    one-line height: on the first real STO2 `visualize` run the two-line cost
    message sat cut off in its box, and switching theme — which re-applies the
    sheet globally — was what made it right.

    The oracle is a label built fresh under the same sheet, which resolves the
    themed padding correctly by construction; asserting against it rather than
    a pixel count keeps this honest across fonts and DPI.
    """
    app = QApplication.instance()
    with applied_theme(app, "light"):
        view = StageView("visualize", STAGE_SPECS["visualize"], Experiment())
        view.resize(1000, 800)
        view.show()
        app.processEvents()
        view._show_banner(html.escape(_WRAPPING_HEADLINE), role="banner-info")
        app.processEvents()
        width = view._banner.width()

        parent_layout = view._banner.parentWidget().layout()
        styled = QLabel(html.escape(_WRAPPING_HEADLINE))
        styled.setWordWrap(True)
        styled.setTextFormat(view._banner.textFormat())
        styled.setProperty("role", "banner-info")
        parent_layout.addWidget(styled)
        plain = QLabel(html.escape(_WRAPPING_HEADLINE))
        plain.setWordWrap(True)
        plain.setTextFormat(view._banner.textFormat())
        parent_layout.addWidget(plain)
        app.processEvents()

        # precondition: at this width the role really does add padding, so a
        # banner that ignored it would measure short and this test could fail.
        assert styled.heightForWidth(width) > plain.heightForWidth(width)
        assert view._banner.heightForWidth(width) == styled.heightForWidth(width)
