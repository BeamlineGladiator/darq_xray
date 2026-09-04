"""darq_xray.common.roi — darfix-window / map-frame conversions and validation."""

from __future__ import annotations

import pytest

from darq_xray.common import roi as R
from darq_xray.config.models import Param, ParamType, StageSpec

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


# --- validate_roi_params: per-stage ROI fields ------------------------------
#
# `validate_rois` above answers the EXPERIMENT editor's question (a darfix
# window plus one analysis window). These answer the STAGE form's: every param
# a spec declares as a range, checked against the data extent when one is
# known. Blocking vs advisory is the distinction the GUI acts on — see
# `RoiProblem.blocking`.

_PAIR_SPEC = StageSpec(
    name="demo",
    label="Demo",
    description="",
    params=(
        Param("roi_x", ParamType.STR, "ROI X", "", roi_axis="x", roi_frame="detector"),
        Param("roi_y", ParamType.STR, "ROI Y", "", roi_axis="y", roi_frame="detector"),
        Param("not_an_roi", ParamType.STR, "Other", ""),
    ),
)

_BOTH_SPEC = StageSpec(
    name="demo4",
    label="Demo 4",
    description="",
    params=(Param("roi", ParamType.STR, "ROI", "", roi_axis="both", roi_frame="map"),),
)


def test_validate_roi_params_passes_a_good_roi():
    p = {"roi_x": "105,1937", "roi_y": "630,1330", "not_an_roi": "1330,630"}
    assert R.validate_roi_params(_PAIR_SPEC, p) == ()
    # The extent is generous enough for both axes, so it changes nothing.
    assert R.validate_roi_params(_PAIR_SPEC, p, extent=(2048, 2048)) == ()


def test_validate_roi_params_ignores_blank_axes():
    assert R.validate_roi_params(_PAIR_SPEC, {"roi_x": "", "roi_y": None}) == ()


def test_validate_roi_params_blocks_an_inverted_pair():
    problems = R.validate_roi_params(_PAIR_SPEC, {"roi_x": "105,1937", "roi_y": "1330,630"})
    assert [p.param for p in problems] == ["roi_y"]
    assert problems[0].blocking
    assert "1330,630" in problems[0].message


def test_validate_roi_params_blocks_a_negative_start():
    problems = R.validate_roi_params(_PAIR_SPEC, {"roi_y": "-5,100"})
    assert len(problems) == 1 and problems[0].blocking
    assert problems[0].param == "roi_y"


def test_validate_roi_params_blocks_malformed_text():
    problems = R.validate_roi_params(_PAIR_SPEC, {"roi_x": "banana", "roi_y": "105"})
    assert {p.param for p in problems} == {"roi_x", "roi_y"}
    assert all(p.blocking for p in problems)


def test_validate_roi_params_blocks_a_start_past_the_extent():
    # Nothing survives the crop: numpy clamps `3000:4000` on a 2048-px axis to
    # an empty slice, which is the same blank product an inverted pair gives.
    problems = R.validate_roi_params(_PAIR_SPEC, {"roi_x": "3000,4000"}, extent=(2048, 2048))
    assert len(problems) == 1 and problems[0].blocking
    assert "2048" in problems[0].message


def test_validate_roi_params_only_warns_about_a_partial_overrun():
    # `105:4000` on a 2048-px axis still yields 1943 real columns — the run
    # succeeds on less than was asked for, so say so without blocking it.
    problems = R.validate_roi_params(_PAIR_SPEC, {"roi_x": "105,4000"}, extent=(2048, 2048))
    assert len(problems) == 1 and not problems[0].blocking
    assert "2048" in problems[0].message


def test_validate_roi_params_checks_each_axis_against_its_own_extent():
    # extent is (height, width): roi_y is bounded by the height, roi_x by the
    # width. A square extent could not tell a swapped pair apart.
    problems = R.validate_roi_params(
        _PAIR_SPEC, {"roi_x": "0,900", "roi_y": "0,900"}, extent=(700, 2891)
    )
    assert [p.param for p in problems] == ["roi_y"]


def test_validate_roi_params_reads_the_four_int_both_axis():
    assert R.validate_roi_params(_BOTH_SPEC, {"roi": "630,1330,105,1937"}) == ()
    rows_bad = R.validate_roi_params(_BOTH_SPEC, {"roi": "1330,630,105,1937"})
    assert len(rows_bad) == 1 and rows_bad[0].blocking and rows_bad[0].param == "roi"
    assert "rows" in rows_bad[0].message
    cols_bad = R.validate_roi_params(_BOTH_SPEC, {"roi": "630,1330,1937,105"})
    assert len(cols_bad) == 1 and "columns" in cols_bad[0].message
    assert R.validate_roi_params(_BOTH_SPEC, {"roi": "630,1330"})[0].blocking  # wrong arity


def test_validate_roi_params_never_raises_on_junk():
    # It runs on every keystroke of a form being filled in; a half-typed value
    # is the ordinary state, not an error.
    for junk in ("105,", ",", "1,2,3", [], {}, 7):
        R.validate_roi_params(_PAIR_SPEC, {"roi_x": junk})


def test_strict_end_turns_an_overrun_into_a_blocker():
    """`strain.apply_roi` raises on `r1 > rows` instead of clamping, so the
    same input that is merely advisory elsewhere must block there — and the
    message must not promise a crop the stage will refuse."""
    p = {"roi_x": "105,4000"}
    lax = R.validate_roi_params(_PAIR_SPEC, p, extent=(2048, 2048))
    assert len(lax) == 1 and not lax[0].blocking and "crops at 2048" in lax[0].message
    strict = R.validate_roi_params(_PAIR_SPEC, p, extent=(2048, 2048), strict_end=True)
    assert len(strict) == 1 and strict[0].blocking
    assert "crops at" not in strict[0].message and "refuses" in strict[0].message


def test_strict_end_changes_nothing_about_the_empty_cases():
    for strict in (False, True):
        problems = R.validate_roi_params(
            _PAIR_SPEC, {"roi_x": "1330,630"}, extent=(2048, 2048), strict_end=strict
        )
        assert len(problems) == 1 and problems[0].blocking
        assert (
            R.validate_roi_params(
                _PAIR_SPEC, {"roi_x": "105,1937"}, extent=(2048, 2048), strict_end=strict
            )
            == ()
        )
