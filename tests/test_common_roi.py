"""dfxm.common.roi — darfix-window / map-frame conversions and validation."""

from __future__ import annotations

import pytest

from dfxm.common import roi as R

STO2_DARFIX = "105,230,1832,1266"


def test_parse_pair():
    assert R.parse_pair("400,1100") == (400, 1100)
    assert R.parse_pair(" 400 , 1100 ") == (400, 1100)
    assert R.parse_pair("") is None
    assert R.parse_pair(None) is None
    with pytest.raises(ValueError):
        R.parse_pair("400")
    with pytest.raises(ValueError):
        R.parse_pair("a,b")


def test_parse_darfix_roi():
    win = R.parse_darfix_roi(STO2_DARFIX)
    assert (win.origin_x, win.origin_y, win.width, win.height) == (105, 230, 1832, 1266)
    assert (win.x0, win.x1, win.y0, win.y1) == (105, 1937, 230, 1496)
    assert R.parse_darfix_roi("") is None
    with pytest.raises(ValueError):
        R.parse_darfix_roi("105,230,1832")


def test_map_detector_round_trip():
    assert R.map_to_detector((400, 1100), 230) == (630, 1330)
    assert R.detector_to_map((630, 1330), 230) == (400, 1100)
    pair = (12, 345)
    assert R.detector_to_map(R.map_to_detector(pair, 105), 105) == pair


def test_analysis_detector_window_sto2_golden():
    """The 2026-07-18 incident's hand-conversion, as a derivation."""
    det_x, det_y = R.analysis_detector_window(STO2_DARFIX, "0,1832", "400,1100")
    assert det_x == (105, 1937)
    assert det_y == (630, 1330)


def test_analysis_detector_window_blank_axis_falls_back_to_full_window():
    det_x, det_y = R.analysis_detector_window(STO2_DARFIX, "", "400,1100")
    assert det_x == (105, 1937)  # full darfix width
    assert det_y == (630, 1330)


def test_analysis_detector_window_no_darfix_derives_nothing():
    assert R.analysis_detector_window("", "0,1832", "400,1100") == (None, None)


def test_format_pair():
    assert R.format_pair((630, 1330)) == "630,1330"


def test_validate_rois_ok_and_blank():
    assert R.validate_rois(STO2_DARFIX, "0,1832", "400,1100") == []
    assert R.validate_rois("", "", "") == []
    assert R.validate_rois("", "0,100", "") == []  # analysis without darfix is allowed


def test_validate_rois_problems():
    assert R.validate_rois("105,230", "", "")  # malformed darfix
    assert R.validate_rois("105,230,0,1266", "", "")  # non-positive size
    assert R.validate_rois(STO2_DARFIX, "banana", "")  # malformed pair
    assert R.validate_rois(STO2_DARFIX, "", "1100,400")  # end <= start
    assert R.validate_rois(STO2_DARFIX, "", "-5,100")  # negative start
    msgs = R.validate_rois(STO2_DARFIX, "", "400,1300")  # 1300 > height 1266
    assert msgs and "1266" in msgs[0]
