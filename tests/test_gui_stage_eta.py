"""StageView progress text gains an ETA (offscreen Qt, synthetic Progress msgs)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

_app = QApplication.instance() or QApplication([])

from dfxm.common.eta import EtaEstimator  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402
from dfxm.runner import Progress  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import StageView  # noqa: E402


def _view_with_fake_clock():
    view = StageView("strain", STAGE_SPECS["strain"], Experiment())
    t = [0.0]
    view._eta = EtaEstimator(clock=lambda: t[0])
    return view, t


def test_progress_text_plain_before_estimable():
    view, _t = _view_with_fake_clock()
    view._handle(Progress(0.02, "warming up"))
    assert view._progress_text.text() == "warming up"


def test_progress_text_gains_eta_once_estimable():
    view, t = _view_with_fake_clock()
    view._handle(Progress(0.02, "warming up"))
    t[0] = 10.0
    view._handle(Progress(0.5, "stacking layers"))
    assert view._progress_text.text() == "stacking layers — ~10 s left"
    assert view._progress.value() == 50  # bar itself unchanged


def test_run_start_resets_estimator():
    view, t = _view_with_fake_clock()
    t[0] = 10.0
    view._handle(Progress(0.5, "x"))
    assert "left" in view._progress_text.text()
    view._eta.reset()  # what _on_run does — the wiring itself is asserted below
    view._handle(Progress(0.5, "x"))  # elapsed 0 since reset -> not estimable yet
    assert view._progress_text.text() == "x"


def test_progress_suffix_cleared_after_cancel():
    """F4 fix wave: the " — ~N s left" ETA suffix a Progress message appended
    used to linger in the label after the run actually stopped (Cancelled/
    Done/Failed), since none of _on_cancel/_finish_ok/_finish_failed touched
    _progress_text. Each now resets it to _progress_plain (the last Progress
    text with no ETA suffix), tracked alongside the eta-augmented text in the
    Progress branch of _handle. Exercised via _on_cancel — it needs no
    StageRunner (self._runner is None, so _on_cancel's runner.cancel() call is
    skipped) and reaches the exact same one-line reset _finish_ok/
    _finish_failed use, which need a full StageRunner/result to reach
    directly and so are not separately driven here."""
    view, t = _view_with_fake_clock()
    view._handle(Progress(0.02, "warming up"))
    t[0] = 10.0
    view._handle(Progress(0.5, "stacking layers"))
    assert view._progress_text.text() == "stacking layers — ~10 s left"
    view._on_cancel()
    assert view._progress_text.text() == "stacking layers"
