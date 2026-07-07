"""Offscreen test: Replot… button is present on the right stages."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.config.models import Experiment  # noqa: E402
from gui.bindings import STAGE_SPECS  # noqa: E402
from gui.stage_view import StageView  # noqa: E402

_app = QApplication.instance() or QApplication([])

_REPLOT_STAGES = {"slices", "strain", "mosaicity", "rocking"}


@pytest.mark.parametrize("stage", sorted(STAGE_SPECS))
def test_replot_button_only_on_map_stages(stage):
    view = StageView(stage, STAGE_SPECS[stage], Experiment())
    has_button = view._replot_btn is not None
    assert has_button == (stage in _REPLOT_STAGES)
