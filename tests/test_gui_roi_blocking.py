"""An impossible ROI turns the banner red, marks the field and disables Run."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.common.roi import RoiProblem  # noqa: E402


def _view(stage: str = "rocking"):
    from gui.main_window import MainWindow

    _ = QApplication.instance() or QApplication([])
    win = MainWindow()
    return win, win._views[stage]


def _advisory(problems=(), headline=""):
    """A stand-in Advisory carrying only what StageView reads off one."""

    class _Adv:
        roi_problems = tuple(problems)
        hints: dict = {}
        details: tuple = ()
        blocked = None

        @property
        def roi_blockers(self):
            return tuple(p for p in self.roi_problems if p.blocking)

    _Adv.headline = headline
    return _Adv()


def test_a_blocking_problem_disables_run_and_shows_a_red_banner():
    win, view = _view()
    try:
        assert view._run_btn.isEnabled()  # precondition
        view._show_advisory(_advisory([RoiProblem("roi_y", "crops to nothing", True)]))
        assert not view._run_btn.isEnabled()
        assert view._banner.isVisibleTo(view)
        assert view._banner.property("role") == "banner-error"
        assert "ROI Y" in view._banner.text()  # the label, not the param name
        assert "crops to nothing" in view._banner.text()
    finally:
        win.close()


def test_the_field_carries_its_own_note():
    win, view = _view()
    try:
        view._show_advisory(_advisory([RoiProblem("roi_y", "crops to nothing", True)]))
        note = view._form._errors["roi_y"]
        assert note.isVisibleTo(view._form) and "crops to nothing" in note.text()
        assert not view._form._errors["roi_x"].isVisibleTo(view._form)
    finally:
        win.close()


def test_fixing_the_roi_clears_everything():
    win, view = _view()
    try:
        view._show_advisory(_advisory([RoiProblem("roi_y", "crops to nothing", True)]))
        view._show_advisory(_advisory())
        assert view._run_btn.isEnabled()
        assert not view._form._errors["roi_y"].isVisibleTo(view._form)
        assert not view._banner.isVisibleTo(view)
    finally:
        win.close()


def test_an_advisory_problem_marks_the_field_but_keeps_run_live():
    """An end past the extent still yields real pixels — say so, don't block."""
    win, view = _view()
    try:
        view._show_advisory(_advisory([RoiProblem("roi_x", "the run crops at 2048", False)]))
        assert view._run_btn.isEnabled()
        assert view._form._errors["roi_x"].isVisibleTo(view._form)
    finally:
        win.close()


def test_a_blocked_roi_survives_a_run_that_finishes():
    """`_set_running(False)` re-enables Run at the end of every run. It must not
    hand the button back while the ROI that would waste the next one stands."""
    win, view = _view()
    try:
        view._show_advisory(_advisory([RoiProblem("roi_y", "crops to nothing", True)]))
        view._set_running(True)
        view._set_running(False)
        assert not view._run_btn.isEnabled()
    finally:
        win.close()


def test_clicking_run_with_an_impossible_roi_never_starts_a_child(monkeypatch):
    """The hard gate: even if the debounced advisory has not landed, the click
    itself must re-check and refuse."""
    win, view = _view()
    try:
        started = []
        monkeypatch.setattr(view, "_start_runner", lambda params: started.append(params))
        view._form.set_values({"roi_x": "105,1937", "roi_y": "1330,630"})
        view._on_run()
        assert started == []
        assert view._banner.property("role") == "banner-error"
    finally:
        win.close()


def test_the_stage_form_still_has_no_pick_roi_button():
    """rocking declares roi_axis for validation only — it has no map to draw a
    picker on, and `roi_group` is what grows the button."""
    win, view = _view()
    try:
        assert not view._roi_buttons
    finally:
        win.close()


def test_slices_advanced_roi_field_is_revealed_when_run_is_refused(monkeypatch):
    """slices keeps its ROI inside the collapsed Advanced expander, so the red
    note under the field is out of sight — the banner names it, and the Run
    click reveals it. Its params are `align_roi_*`, which also proves the check
    is schema-driven rather than keyed on the name `roi_x`."""
    win, view = _view("slices")
    try:
        monkeypatch.setattr(view, "_start_runner", lambda params: None)
        assert not view._form._adv_toggle.isChecked()  # precondition
        view._form.set_values({"align_roi_y": "1330,630"})
        view._on_run()
        assert view._form._adv_toggle.isChecked()
        assert "Map ROI Y" in view._banner.text() or "ROI Y" in view._banner.text()
    finally:
        win.close()


def test_every_stage_with_an_roi_field_can_render_a_problem():
    """Each ROI-taking stage has an error row for each of its ROI params — a
    stage whose param the form never built would silently swallow the note."""
    from gui.main_window import MainWindow

    _ = QApplication.instance() or QApplication([])
    win = MainWindow()
    try:
        seen = {}
        for name, view in win._views.items():
            axes = [p.name for p in view._spec.params if p.roi_axis]
            if not axes:
                continue
            seen[name] = axes
            for param in axes:
                assert param in view._form._errors, f"{name}.{param} has no error row"
        assert set(seen) == {"strain", "rocking", "visualize", "paraview", "slices"}
    finally:
        win.close()


def test_an_extent_only_blocker_stops_the_run_too(monkeypatch):
    """The click's own gate is data-free, so a start past the REAL data is
    caught one step later, by the synchronous advisory. Without a second return
    there, Run greyed out and the child started anyway."""
    win, view = _view()
    try:
        started = []
        monkeypatch.setattr(view, "_start_runner", lambda params: started.append(params))
        monkeypatch.setattr(view, "_validate_inputs", lambda params: None)
        # A ROI the DATA-FREE gate is happy with, so this test can only be
        # stopped by the extent-aware verdict below — without this the click's
        # first gate would refuse for its own reasons and the test would pass
        # whether or not the second gate exists.
        view._form.set_values({"roi_x": "105,1937", "roi_y": "630,1330"})
        assert view._form.values()["roi_y"] == "630,1330"
        blocker = RoiProblem("roi_x", "start 5000 is past this data's 2048 px extent", True)
        monkeypatch.setattr(
            view._advisor, "compute_blocking", lambda: _advisory([blocker], headline="")
        )
        view._on_run()
        assert started == []
        assert not view._run_btn.isEnabled()
    finally:
        win.close()
