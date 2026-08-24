"""Every action button explains itself (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.config.models import Experiment  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import EXPORT_TIP_DISABLED, EXPORT_TIP_ENABLED, StageView  # noqa: E402


def _view(stage):
    # StageView(stage_name, spec, experiment) — the third argument is required;
    # see tests/test_gui_stage_eta.py for the same construction.
    return StageView(stage, STAGE_SPECS[stage], Experiment())


def test_every_stage_action_button_has_a_tooltip():
    for stage in ("profiles", "slices", "strain"):
        view = _view(stage)
        buttons = [
            view._pick_btn,
            view._jobs_marks_btn,
            view._replot_btn,
            view._pin_btn,
            view._mark_btn,
            *view._roi_buttons.values(),
        ]
        present = [b for b in buttons if b is not None]
        assert present, f"{stage} builds no action buttons"  # precondition
        for btn in present:
            assert btn.toolTip().strip(), f"{stage}: {btn.text()!r} has no tooltip"


def test_each_action_tooltip_says_what_that_button_does():
    """A tooltip that merely echoes the label would pass the sweep above."""
    view = _view("profiles")
    assert "job list" in view._pick_btn.toolTip()
    assert "starred" in view._jobs_marks_btn.toolTip()
    assert "without re-running" in view._replot_btn.toolTip()

    slices = _view("slices")
    assert "later runs" in slices._pin_btn.toolTip()
    assert "/marks" in slices._mark_btn.toolTip()
    assert all("pixel bounds" in b.toolTip() for b in slices._roi_buttons.values())


def test_the_export_buttons_explain_why_they_are_disabled():
    view = _view("strain")
    assert view._export_btn.isEnabled() is False  # precondition
    assert view._export_btn.toolTip() == EXPORT_TIP_DISABLED
    assert view._export_all_btn.toolTip() == EXPORT_TIP_DISABLED
    assert "run" in EXPORT_TIP_DISABLED.lower()


def test_the_export_tooltip_changes_once_a_run_has_produced_figures():
    view = _view("strain")
    assert view._export_btn.toolTip() == EXPORT_TIP_DISABLED  # precondition
    view._enable_exports()
    assert view._export_btn.isEnabled() is True
    assert view._export_all_btn.isEnabled() is True
    assert view._export_btn.toolTip() == EXPORT_TIP_ENABLED
    assert view._export_all_btn.toolTip() == EXPORT_TIP_ENABLED
    assert EXPORT_TIP_ENABLED != EXPORT_TIP_DISABLED


def test_starting_a_new_run_puts_the_exports_back_on_the_disabled_wording(monkeypatch):
    """A new run disables the exports again; the wording must follow, not lie.

    Driven through the real ``_on_run`` so the reset site is covered, not just
    the helper. StageRunner is faked the way tests/test_gui_stage_eta.py does:
    the reset happens before the runner is constructed, so it need only exist.
    """

    class _FakeRunner:
        def __init__(self, *_a, **_k):
            pass

        def start(self):
            pass

        def is_alive(self):
            return False

        def poll(self):
            return []

        def cancel(self):
            pass

    monkeypatch.setattr("gui.stage_view.StageRunner", _FakeRunner)
    view = _view("strain")
    view._enable_exports()
    assert view._export_btn.toolTip() == EXPORT_TIP_ENABLED  # precondition
    view._on_run()
    view._timer.stop()  # never let the fake runner's poll() tick after the test
    assert view._export_btn.isEnabled() is False  # precondition: the run started
    assert view._export_btn.toolTip() == EXPORT_TIP_DISABLED
    assert view._export_all_btn.toolTip() == EXPORT_TIP_DISABLED


def test_the_3d_tab_carries_a_tooltip_on_volume_stages():
    view = _view("visualize")
    idx = [view._tabs.tabText(i) for i in range(view._tabs.count())].index("3D")
    tip = view._tabs.tabToolTip(idx)
    assert tip.strip()
    # It must name the button the user has to press, not just say "3-D view".
    assert "Open 3D viewer" in tip


def test_the_left_panel_buttons_explain_themselves():
    from gui.main_window import MainWindow

    win = MainWindow()
    assert win._pub_style_btn.toolTip().strip()
    assert win._figure_builder_btn.toolTip().strip()
    assert win._system_check_btn.toolTip().strip()
    assert "colormap" in win._pub_style_btn.toolTip().lower()
    assert "multi-panel" in win._figure_builder_btn.toolTip().lower()
