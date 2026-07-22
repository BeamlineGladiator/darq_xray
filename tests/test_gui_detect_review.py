"""DetectReviewDialog: pre-check rules, row states, ROI gating, applied_values."""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")
from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from dfxm.config.detect import Detection  # noqa: E402
from dfxm.config.models import Experiment  # noqa: E402

DEFAULTS = Experiment().to_dict()


def _dlg(detections, current=None):
    from gui.widgets.detect_review import DetectReviewDialog

    _ = QApplication.instance() or QApplication([])
    cur = dict(DEFAULTS)
    cur.update(current or {})
    return DetectReviewDialog(detections, current=cur, defaults=DEFAULTS)


def _check_item(dlg, row):
    return dlg._table.item(row, 4)


def test_blank_current_prechecked_and_applied():
    dlg = _dlg([Detection("folder_pattern", "s__*", "3 folders")])
    assert _check_item(dlg, 0).checkState() == Qt.CheckState.Checked
    assert dlg.applied_values() == {"folder_pattern": "s__*"}


def test_default_valued_current_prechecked():
    # pixel_size_x_um default is 1.0 -> still counts as "not user-set"
    dlg = _dlg([Detection("pixel_size_x_um", 0.151733, "M=20.4")])
    assert _check_item(dlg, 0).checkState() == Qt.CheckState.Checked
    assert dlg.applied_values() == {"pixel_size_x_um": 0.151733}


def test_user_set_current_unchecked_and_marked():
    dlg = _dlg([Detection("ccmth_ref_deg", 7.1442, "median")], current={"ccmth_ref_deg": 7.144})
    item = _check_item(dlg, 0)
    assert item.checkState() == Qt.CheckState.Unchecked
    assert "differs" in dlg._table.item(0, 3).text()
    assert dlg.applied_values() == {}  # nothing checked -> nothing applied


def test_error_row_disabled():
    dlg = _dlg([Detection("mosa_pattern", error="no folder family containing 'mosa'")])
    item = _check_item(dlg, 0)
    assert item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert "mosa" in dlg._table.item(0, 3).text()
    assert dlg.applied_values() == {}


def test_info_row_disabled():
    dlg = _dlg([Detection("darfix_roi", None, "✓ size matches maps.h5 (1832×1266)")])
    item = _check_item(dlg, 0)
    assert item is None or not (item.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert dlg.applied_values() == {}


def test_partial_roi_gated_until_origin_typed():
    dlg = _dlg([Detection("darfix_roi", "?,?,1832,1266", "map size — replace ?,?")])
    check = _check_item(dlg, 0)
    assert not (check.flags() & Qt.ItemFlag.ItemIsEnabled)  # gated
    dlg._table.item(0, 2).setText("105,230,1832,1266")  # user types the origin
    check = _check_item(dlg, 0)
    assert check.flags() & Qt.ItemFlag.ItemIsEnabled
    assert check.checkState() == Qt.CheckState.Checked  # auto-checks once valid
    assert dlg.applied_values() == {"darfix_roi": "105,230,1832,1266"}


def test_partial_roi_invalid_edit_regates():
    dlg = _dlg([Detection("darfix_roi", "?,?,1832,1266", "map size")])
    dlg._table.item(0, 2).setText("105,230,1832,1266")
    dlg._table.item(0, 2).setText("banana")  # edited back to nonsense
    check = _check_item(dlg, 0)
    assert not (check.flags() & Qt.ItemFlag.ItemIsEnabled)
    assert dlg.applied_values() == {}


def test_unchecking_excludes_from_applied():
    dlg = _dlg(
        [
            Detection("folder_pattern", "s__*", ""),
            Detection("entry_suffix", ".1", ""),
        ]
    )
    _check_item(dlg, 0).setCheckState(Qt.CheckState.Unchecked)
    assert dlg.applied_values() == {"entry_suffix": ".1"}


def test_field_column_shows_schema_label():
    dlg = _dlg([Detection("pixel_size_x_um", 0.15, "")])
    assert dlg._table.item(0, 0).text() == "Pixel size X"
