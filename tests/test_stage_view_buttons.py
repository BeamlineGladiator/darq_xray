"""Offscreen test: Replot… button is present on the right stages."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from darq_xray.config.models import Experiment  # noqa: E402
from darq_xray.gui.bindings import STAGE_SPECS  # noqa: E402
from darq_xray.gui.stage_view import StageView  # noqa: E402

_app = QApplication.instance() or QApplication([])

_REPLOT_STAGES = {"slices", "strain", "mosaicity", "rocking", "profiles"}
_PIN_STAGES = {"slices"}
_JOBS_MARKS_STAGES = {"profiles"}


@pytest.mark.parametrize("stage", sorted(STAGE_SPECS))
def test_replot_button_only_on_map_stages(stage):
    view = StageView(stage, STAGE_SPECS[stage], Experiment())
    has_button = view._replot_btn is not None
    assert has_button == (stage in _REPLOT_STAGES)


@pytest.mark.parametrize("stage", sorted(STAGE_SPECS))
def test_pin_planes_button_only_on_slices(stage):
    view = StageView(stage, STAGE_SPECS[stage], Experiment())
    has_button = view._pin_btn is not None
    assert has_button == (stage in _PIN_STAGES)


@pytest.mark.parametrize("stage", sorted(STAGE_SPECS))
def test_jobs_from_marks_button_only_on_profiles(stage):
    view = StageView(stage, STAGE_SPECS[stage], Experiment())
    has_button = view._jobs_marks_btn is not None
    assert has_button == (stage in _JOBS_MARKS_STAGES)
