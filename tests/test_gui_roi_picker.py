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


def _two_previews():
    a = np.arange(200 * 100, dtype=float).reshape(200, 100)
    b = np.ones((200, 100))
    return [
        ("map A", lambda a=a: (a, 0.152, 0.385)),
        ("map B", lambda b=b: (b, 0.152, 0.385)),
    ]


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


def test_initial_rectangle_drawn_on_open():
    """initial= pre-populates the selector so the rectangle is actually visible."""
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    # initial=(r0, r1, c0, c1); extents order is (xmin, xmax, ymin, ymax)=(c0, c1, r0, r1)
    dlg = ROIPickerDialog(_previews(), initial=(40, 160, 12, 88))
    assert dlg._use.isEnabled()
    assert tuple(round(v) for v in dlg._selector.extents) == (12, 88, 40, 160)
    dlg.deleteLater()


# -- "Keep size" lock ---------------------------------------------------------
def test_keep_size_lock_moves_without_resizing():
    """With the lock on, a drag keeps the locked px size, centred on the new rect."""
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)  # 40 cols × 60 rows
    dlg._lock.setChecked(True)
    dlg._on_rect_change(30.0, 96.0, 100.0, 190.0)  # attempted move + resize
    assert "lock" in dlg._readout.text().lower()
    # centre (63, 145) kept, size coerced back to 40×60, snapped to whole px
    assert tuple(round(v) for v in dlg._selector.extents) == (43, 83, 115, 175)
    dlg._accept()
    assert dlg.result == (115, 175, 43, 83)
    dlg.deleteLater()


def test_keep_size_lock_clamps_inside_image():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())  # image is 200 rows × 100 cols
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)  # lock 40×60
    dlg._lock.setChecked(True)
    dlg._on_rect_change(80.0, 98.0, 150.0, 198.0)  # centre near the top-right corner
    dlg._accept()
    assert dlg.result == (140, 200, 60, 100)  # shifted back fully inside
    dlg.deleteLater()


def test_keep_size_lock_first_drag_establishes_size():
    """Lock checked before any rectangle: the first drag defines the locked size."""
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())
    dlg._lock.setChecked(True)
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)  # establishes 40×60
    dlg._on_rect_change(0.0, 10.0, 0.0, 10.0)  # too small — coerced to 40×60 at the corner
    dlg._accept()
    assert dlg.result == (0, 60, 0, 40)
    dlg.deleteLater()


def test_keep_size_unlock_frees_resizing():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)
    dlg._lock.setChecked(True)
    dlg._lock.setChecked(False)
    dlg._on_rect_change(0.0, 10.0, 0.0, 10.0)
    dlg._accept()
    assert dlg.result == (0, 10, 0, 10)
    dlg.deleteLater()


def test_per_preview_picks_collect_one_roi_per_moved_map():
    """per_preview=True: every preview the user draws/moves on gets its OWN pick."""
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_two_previews(), per_preview=True)
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)  # user draws on map A
    dlg._combo.setCurrentIndex(1)  # swap to map B (rect carries over as a hint)
    assert dlg.picked == {0: (20, 80, 10, 50)}  # carry-over alone is NOT a pick
    dlg._on_rect_change(30.0, 70.0, 100.0, 160.0)  # user moves it on map B
    dlg._accept()
    assert dlg.picked == {0: (20, 80, 10, 50), 1: (100, 160, 30, 70)}
    assert dlg.result == (100, 160, 30, 70)
    dlg.deleteLater()


def test_per_preview_carried_rect_not_applied_until_moved():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_two_previews(), per_preview=True)
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)
    dlg._combo.setCurrentIndex(1)  # only LOOK at map B
    assert "not applied" in dlg._readout.text()
    dlg._accept()
    assert dlg.picked == {0: (20, 80, 10, 50)}  # map B untouched → no pick for it
    dlg.deleteLater()


def test_per_preview_returning_restores_that_maps_own_pick():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_two_previews(), per_preview=True)
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)  # pick on A
    dlg._combo.setCurrentIndex(1)
    dlg._on_rect_change(30.0, 70.0, 100.0, 160.0)  # different pick on B
    dlg._combo.setCurrentIndex(0)  # back to A → A's own rectangle restored
    assert tuple(round(v) for v in dlg._selector.extents) == (10, 50, 20, 80)
    dlg.deleteLater()


def test_per_preview_reset_drops_only_current_maps_pick():
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_two_previews(), per_preview=True)
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)
    dlg._combo.setCurrentIndex(1)
    dlg._on_rect_change(30.0, 70.0, 100.0, 160.0)
    dlg._on_reset()  # forgets map B's pick only
    assert dlg.picked == {0: (20, 80, 10, 50)}
    assert dlg._use.isEnabled()  # map A's pick still accepts
    dlg.deleteLater()


def test_reset_clears_locked_size():
    """Reset forgets the locked size; the next drag re-establishes it."""
    from gui.widgets.roi_picker import ROIPickerDialog

    _ = QApplication.instance() or QApplication([])
    dlg = ROIPickerDialog(_previews())
    dlg._on_rect_change(10.0, 50.0, 20.0, 80.0)  # 40×60
    dlg._lock.setChecked(True)
    dlg._on_reset()
    dlg._on_rect_change(0.0, 10.0, 0.0, 10.0)  # fresh drag → new locked size 10×10
    dlg._on_rect_change(20.0, 26.0, 30.0, 40.0)  # still 10×10, recentred
    dlg._accept()
    assert dlg.result == (30, 40, 18, 28)
    dlg.deleteLater()
