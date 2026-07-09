"""Offscreen construction + selection test for the ROI picker dialog."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication  # noqa: E402


def _previews():
    arr = np.arange(200 * 100, dtype=float).reshape(200, 100)  # (H=200, W=100)
    return [("layer 0", lambda a=arr: (a, 0.152, 0.385))]  # sx=0.152 (X), sy=0.385 (Y)


def test_dialog_selection_returns_indices_and_readout():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())
    # simulate a drag by calling the selector callback with data coords
    dlg._on_rect_change(12.0, 89.0, 40.0, 161.0)
    assert dlg._use.isEnabled()
    assert "µm" in dlg._readout.text() and "px" in dlg._readout.text()
    dlg._accept()
    assert dlg.result == (40, 161, 12, 89)
    dlg.deleteLater()


def test_dialog_no_selection_result_none():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())
    assert dlg.result is None
    assert not dlg._use.isEnabled()  # disabled until a non-degenerate rect exists
    dlg.deleteLater()
