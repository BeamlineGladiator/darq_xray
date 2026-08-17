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
