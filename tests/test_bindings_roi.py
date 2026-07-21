"""experiment_overrides derives each stage's ROI fields in its native frame."""

from __future__ import annotations

from dfxm.config.models import Experiment
from gui.bindings import experiment_overrides

STO2_ROIS = dict(darfix_roi="105,230,1832,1266", analysis_roi_x="0,1832", analysis_roi_y="400,1100")

ROI_KEYS = ("roi_x", "roi_y", "align_roi_x", "align_roi_y", "roi")


def test_rocking_gets_absolute_detector_window():
    ov = experiment_overrides("rocking", Experiment(**STO2_ROIS))
    assert ov["roi_x"] == "105,1937"
    assert ov["roi_y"] == "630,1330"  # the incident's hand-conversion, automated


def test_map_stages_get_map_frame_values():
    exp = Experiment(**STO2_ROIS)
    for stage in ("visualize", "paraview"):
        ov = experiment_overrides(stage, exp)
        assert ov["roi_x"] == "0,1832" and ov["roi_y"] == "400,1100"
    ov = experiment_overrides("slices", exp)
    assert ov["align_roi_x"] == "0,1832" and ov["align_roi_y"] == "400,1100"
    assert experiment_overrides("strain", exp)["roi"] == "400,1100,0,1832"


def test_blank_rois_prefill_nothing():
    exp = Experiment()
    for stage in ("rocking", "visualize", "paraview", "slices", "strain"):
        ov = experiment_overrides(stage, exp)
        assert not any(k in ov for k in ROI_KEYS), (stage, ov)


def test_partial_analysis_falls_back_to_full_window():
    exp = Experiment(darfix_roi="105,230,1832,1266", analysis_roi_y="400,1100")
    ov = experiment_overrides("rocking", exp)
    assert ov["roi_x"] == "105,1937"  # blank X -> full darfix width
    assert ov["roi_y"] == "630,1330"
    assert "align_roi_x" not in experiment_overrides("slices", exp)
    assert "roi" not in experiment_overrides("strain", exp)  # strain needs both axes


def test_analysis_without_darfix_fills_map_stages_only():
    exp = Experiment(analysis_roi_y="400,1100")
    assert "roi_y" not in experiment_overrides("rocking", exp)  # underivable
    assert experiment_overrides("slices", exp)["align_roi_y"] == "400,1100"


def test_malformed_preset_prefills_nothing():
    exp = Experiment(darfix_roi="banana", analysis_roi_y="400,1100")
    ov = experiment_overrides("rocking", exp)
    assert not any(k in ov for k in ROI_KEYS)


def test_existing_overrides_untouched():
    ov = experiment_overrides("rocking", Experiment(**STO2_ROIS, raw_root="/r"))
    assert ov["raw_root"] == "/r"  # ROI merge does not clobber the base dict
